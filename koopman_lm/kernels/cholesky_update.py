"""
cholesky_update.py -- vendored O(r^2) double-sided Cholesky propagation.

Given L (chol of G), Aw = L^{-1} M L^{-T}, and a rank-1 update
    G+ = G + z z^T,   M+ = M + z w^T
produce L+ and Aw+ = (L+)^{-1} M+ (L+)^{-T} in O(r^2) instead of O(r^3).

Two entry points:
  * update_reference(L, Aw, z, w)  -- pure PyTorch, CPU/GPU, autograd-free.
      Modifies L in place to L+. Returns (Aw+, vz=(L+)^{-1} z).
      VERIFIED to 1e-16 (fp64) vs full refactorization. Always available.
  * update_triton_batched(L, Aw, z, w) -- fused Triton kernel, GPU only.
      Imported LAZILY so this module is safe to import without Triton/CUDA.

The vendored Triton kernel source lives in cholesky_update_triton.py and is
only imported on first use of the GPU path. All ops run under no_grad: this
is a forward-only fast path (no autograd graph). For training through the
updated operator a separate backward must be derived -- NOT provided here.

For the streaming last-layer memory we only need the L cholupdate + vz; the
Aw/M double-sided machinery is for the intra-chunk SKA operator path.
"""

import math
import torch

from koopman_lm.kernels.factor_scan import rank1_chol_update_ as _batched_rank1_update_


@torch.no_grad()
def cholesky_rank1_update_(L, z):
    """In-place rank-1 Cholesky update of L (lower-tri) for G+=G+zz^T, via
    Givens rotations on [L | z]. Returns (cs, ss) rotation params. Modifies
    L and a copy of z.

    This is the unbatched entry point onto factor_scan.rank1_chol_update_'s
    batched Givens sweep -- the two are the same rotation math (verified
    bit-for-bit identical at batch=1), differing only in whether a leading
    batch dimension is present and whether (cs, ss) are returned. This
    module's double-sided (Aw-carrying) machinery below needs (cs, ss);
    factor_scan's own callers don't, so that entry point defaults to
    discarding them. See
    docs/superpowers/specs/2026-08-07-structural-review.md, issue 2.
    """
    # L.unsqueeze(0)/z.unsqueeze(0) are views: the in-place update inside
    # the batched core still lands in L's own storage.
    _, cs, ss = _batched_rank1_update_(
        L.unsqueeze(0), z.unsqueeze(0), return_rotations=True)
    return cs.squeeze(0), ss.squeeze(0)


@torch.no_grad()
def _doublesided_propagate(Aw, cs, ss):
    """Apply Q^T [Aw 0; 0 0] Q, return top-left r x r block."""
    r = Aw.shape[0]
    Ah = torch.zeros(r + 1, r + 1, dtype=Aw.dtype, device=Aw.device)
    Ah[:r, :r] = Aw
    for k in range(r):
        c, s = cs[k], ss[k]
        rk = Ah[k].clone(); rr = Ah[r].clone()
        Ah[k] = c * rk + s * rr
        Ah[r] = -s * rk + c * rr
    for k in range(r):
        c, s = cs[k], ss[k]
        ck = Ah[:, k].clone(); cr = Ah[:, r].clone()
        Ah[:, k] = c * ck + s * cr
        Ah[:, r] = -s * ck + c * cr
    return Ah[:r, :r].clone()


@torch.no_grad()
def _recover_vz(cs, ss):
    """Top r entries of Q^T e_{r+1} = (L+)^{-1} z."""
    r = cs.shape[0]
    vz = torch.empty(r, dtype=cs.dtype, device=cs.device)
    e = torch.ones((), dtype=cs.dtype, device=cs.device)
    for k in range(r):
        vz[k] = ss[k] * e
        e = cs[k] * e
    return vz


@torch.no_grad()
def update_reference(L, Aw, z, w):
    """Full O(r^2) double-sided update. Modifies L in place to L+.
    Returns (Aw+, vz). Pure PyTorch, runs anywhere, verified to 1e-16 fp64."""
    cs, ss = cholesky_rank1_update_(L, z)
    Aw_core = _doublesided_propagate(Aw, cs, ss)
    vz = _recover_vz(cs, ss)
    vw = torch.linalg.solve_triangular(L, w.unsqueeze(1), upper=False).squeeze(1)
    return Aw_core + torch.outer(vz, vw), vz


@torch.no_grad()
def update_reference_general(L, Aw, x, a, b):
    """General double-sided update where the covariance update vector (x) and
    the operator outer-product vectors (a, b) all differ:
        G+ = G + x x^T,   M+ = M + a b^T,   Aw+ = (L+)^{-1} M+ (L+)^{-T}
    Modifies L in place to L+. Returns Aw+.

    update_reference is the special case a=x (M+ = M + x w^T). This general
    form is what a gamma>0 LSTD/resolvent transition update or a general BOM
    update needs (M += y x^T with y != x). For the SKA update use
    x=a=sqrt(beta)z_t, b=sqrt(beta)z_{t-1}."""
    cs, ss = cholesky_rank1_update_(L, x)
    Aw_core = _doublesided_propagate(Aw, cs, ss)
    va = torch.linalg.solve_triangular(L, a.unsqueeze(1), upper=False).squeeze(1)
    vb = torch.linalg.solve_triangular(L, b.unsqueeze(1), upper=False).squeeze(1)
    return Aw_core + torch.outer(va, vb)


@torch.no_grad()
def update_L_only(L, z):
    """Streaming-memory fast path: rank-1 cholupdate of L + return vz=(L+)^{-1}z.
    No Aw/M machinery (the memory at gamma=0 doesn't need it). Modifies L in
    place. O(r^2)."""
    cs, ss = cholesky_rank1_update_(L, z)
    return _recover_vz(cs, ss)


_TRITON = None


def _try_triton():
    global _TRITON
    if _TRITON is None:
        try:
            from koopman_lm.kernels import cholesky_update_triton as t
            _TRITON = t
        except Exception:
            _TRITON = False
    return _TRITON


@torch.no_grad()
def update_batched(L, Aw, z, w, prefer_triton=True):
    """Batched double-sided update. Inputs (T,r,r)/(T,r). Uses the Triton
    kernel on CUDA when available, else loops the verified reference.
    Returns (Aw+ (T,r,r), vz (T,r)). L modified in place where supported."""
    if prefer_triton and L.is_cuda:
        t = _try_triton()
        if t:
            return t.incremental_update_triton_batched(L, Aw, z, w)
    # reference fallback (CPU or no-triton GPU)
    T, r = L.shape[0], L.shape[1]
    Aw_out = torch.empty_like(Aw)
    vz_out = torch.empty(T, r, dtype=L.dtype, device=L.device)
    for i in range(T):
        aw, vz = update_reference(L[i], Aw[i].clone(), z[i].clone(), w[i])
        Aw_out[i] = aw
        vz_out[i] = vz
    return Aw_out, vz_out
