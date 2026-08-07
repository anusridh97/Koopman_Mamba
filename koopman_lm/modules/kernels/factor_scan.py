"""
factor_scan.py : all-prefix Cholesky factors via a PSD square-root scan.

Implements the resolution from the "transport scan fails / factor scan
succeeds" note:

  * The scan monoid is the PSD Gram-sum monoid
        S_I = sum_{i in I} w_i w_i^T,    represented by canonical thin
    square roots W_I with W_I W_I^T = S_I.  Associativity is immediate
    because canon(S_I + S_J + S_K) is tree-shape independent.
  * UPSWEEP:  thin PSD square-root merge.  Segment factors are generally
    rank-deficient, so the merge is canon([W_I, W_J]), a TSQR-style thin
    QR on the stacked factor (NOT an SPD Cholesky update).  Cost per merge
    at width q is O(r q^2); the level sums are geometric, W_up = O(T r^2).
  * DOWNSWEEP: the prefix accumulator includes sqrt(ridge) * I, so every
    downsweep prefix is a genuine SPD Cholesky factor and the combiner is
    EXACTLY the rank-k positive Cholesky update primitive (Givens sweeps,
    O(r^2 k) per merge; the vendored cholesky_update.py / Triton kernel is
    the fused single-node version of the same operation).
    W_down = O(T r^2 (1 + log_+(r/S))) under the plain Blelloch tree.

Output: L_t = chol(ridge*I + sum_{i<t} w_i w_i^T) for EVERY t (exclusive
prefix), i.e. exactly the per-token G_t factor that chunk_stats_exact_torch
(path B) needs, without the Omega(r^3)-per-prefix factorization, and
without ever forming a transport/WY summary.

Gradients ("It Cancels", Appendix C.7): the scan is a NUMERICS-ONLY way to
obtain stable factors.  SKACoreGivenL below reuses the existing whitened
adjoint; L is never differentiated; dG/dM/dCv/dq flow through the (linear)
prefix sums exactly as in ska_core_torch.SKACoreFn.

Gating note: positive scalar decay stays inside the monoid
((a2,B2)o(a1,B1) = (a2 a1, a2 B1 + B2): the merged factor is
canon([sqrt(a2) W_1, W_2])).  Subtractive erasure (downdates) and full
matrix gates are NOT covered; they leave the PSD-positive-update setting.

Integration points
1. Training "path C" (exact per-token stats, fast factors):
     Gf, Mf, Cf, qf, shp = exact_stats(...)          # unchanged, keeps graph
     Lf = all_prefix_chol(w.reshape(N_bh, T, r), ridge).reshape(N, r, r)
     Y  = ska_core_given_L(Gf, Mf, Cf, qf, Lf, K)    # no per-token cholesky
   where w = sqrt(beta) * z_n  (so w w^T == beta * z z^T == the G update).

2. Decode (recurrent.py SPEED NOTE): carry st.L; READ with the carried L,
   WRITE with rank1_chol_update_(st.L, sqrt(beta)*z) for O(r^2)/token, no
   re-Cholesky.  See __main__ test 4 and the patch in the accompanying note.
"""

import math
import torch

try:
    from koopman_lm.modules.kernels.ska_operator import (
        _spec_w, _tri_solve_lower, _tri_solve_lowerT, _whiten_M)
except Exception:  # standalone use / tests
    def _tri_solve_lower(L, A):
        return torch.linalg.solve_triangular(L, A, upper=False)

    def _tri_solve_lowerT(L, A):
        return torch.linalg.solve_triangular(L.transpose(-1, -2), A, upper=True)

    def _whiten_M(L, M):
        LiM = _tri_solve_lower(L, M)
        return _tri_solve_lower(L, LiM.transpose(-1, -2)).transpose(-1, -2)

    def _spec_w(W, iters=20):
        r = W.shape[-1]
        v = torch.ones(*W.shape[:-1], 1, device=W.device, dtype=W.dtype) / (r ** 0.5)
        with torch.no_grad():
            for _ in range(iters):
                u = W @ v
                u = u / (u.norm(dim=-2, keepdim=True) + 1e-8)
                v = W.transpose(-1, -2) @ u
                v = v / (v.norm(dim=-2, keepdim=True) + 1e-8)
            sigma = (W @ v).norm(dim=-2, keepdim=True).squeeze(-1)
            alpha = 1.0 / torch.clamp(sigma, min=1.0)
        return alpha


# canonical thin square root (upsweep merge primitive)

def _canon(W):
    """Canonical thin factor of W W^T.

    W: (..., r, k)  ->  (..., r, min(k, r)) lower-trapezoidal, nonneg diag.

    Via thin QR of W^T:  W^T = Q R  =>  W W^T = R^T R.  The deterministic
    sign fix pins the gauge; the represented Gram is exact regardless (the
    stable object is W W^T, not the gauge of W; see the note's stability
    caveat for rank-deficient upsweep factors).
    """
    Q, R = torch.linalg.qr(W.transpose(-1, -2).contiguous(), mode='reduced')
    d = torch.diagonal(R, dim1=-2, dim2=-1).sign()
    d = torch.where(d == 0, torch.ones_like(d), d)
    return (R * d.unsqueeze(-1)).transpose(-1, -2).contiguous()


# rank-1 / rank-k SPD Cholesky update (downsweep merge primitive)

@torch.no_grad()
def rank1_chol_update_(L, x):
    """Batched in-place rank-1 Cholesky update: L L^T += x x^T.

    L: (B, r, r) lower-triangular (SPD factor), modified in place.
    x: (B, r), consumed via an internal copy.
    Givens sweep, vectorized over batch and rows; r sequential panel steps
    (the sequential panel dependency the note flags; the vendored Triton
    kernel in cholesky_update_triton.py is the fused version of this).
    """
    B, r, _ = L.shape
    x = x.clone()
    for k in range(r):
        a = L[:, k, k]
        b = x[:, k]
        rho = torch.sqrt(a * a + b * b)
        zero = rho == 0
        c = torch.where(zero, torch.ones_like(rho), a / rho)
        s = torch.where(zero, torch.zeros_like(rho), b / rho)
        col = L[:, :, k].clone()
        L[:, :, k] = c.unsqueeze(1) * col + s.unsqueeze(1) * x
        x = -s.unsqueeze(1) * col + c.unsqueeze(1) * x
    return L


@torch.no_grad()
def rankk_chol_update_(L, W):
    """Batched in-place rank-k update: L L^T += W W^T.  L:(B,r,r), W:(B,r,k).
    k sequential rank-1 sweeps (O(r^2 k) flops).  Zero columns are no-ops."""
    for j in range(W.shape[-1]):
        rank1_chol_update_(L, W[:, :, j])
    return L


# the factor scan

@torch.no_grad()
def all_prefix_chol(w, ridge, downsweep='givens'):
    """All-prefix Cholesky factors via the Blelloch factor scan.

    Args:
      w:      (N, T, r) per-step update vectors.  For SKA: w_t = sqrt(beta_t)
              * z_t (per-token L2-normalized key), so that
              w_t w_t^T == beta_t z_t z_t^T == the G_t increment.
      ridge:  ridge eps; the prefix accumulator is seeded with sqrt(ridge)*I
              so every downsweep prefix is a genuine SPD Cholesky factor
              (this is what licenses the SPD rank-k update on the downsweep).
      downsweep: 'givens' (rank-k Cholesky update, O(r^2 k) per merge, the
              note's primitive) or 'qr' (canon([L, W]) refactor merge; more
              flops, fewer/bigger batched kernels, often faster in eager
              PyTorch on GPU).

    Returns:
      L: (N, T, r, r) with L[:, t] = chol(ridge*I + sum_{i<t} w_i w_i^T),
         the EXCLUSIVE prefix, matching exact_stats / chunk_stats
         boundary conventions.  (Inclusive prefix at t == exclusive at t+1.)

    Work: upsweep O(T r^2); downsweep O(T r^2 (1 + log_+ r)) for leaves of
    width 1 (set leaf granularity = chunks of size S to trade the log).
    Depth: O(log T) batched levels.
    """
    N, T, r = w.shape
    dev, dt = w.device, w.dtype

    Tp = 1 if T <= 1 else 1 << (T - 1).bit_length()
    if Tp != T:
        w = torch.nn.functional.pad(w, (0, 0, 0, Tp - T))

    # upsweep: thin square-root merges, left factors saved
    cur = w.unsqueeze(-1)                      # (N, Tp, r, 1) leaf factors
    lefts = []
    while cur.shape[1] > 1:
        Lh, Rh = cur[:, 0::2], cur[:, 1::2]
        lefts.append(Lh)                       # needed by the downsweep
        cat = torch.cat([Lh, Rh], dim=-1)      # exact: Gram is the sum
        cur = _canon(cat) if cat.shape[-1] > r else cat

    # downsweep: SPD rank-k updates from the prefix factor
    eye = torch.eye(r, device=dev, dtype=dt)
    pref = (math.sqrt(ridge) * eye).expand(N, 1, r, r).contiguous()
    for Lh in reversed(lefts):
        n2 = Lh.shape[1]
        if downsweep == 'qr':
            right = _canon(torch.cat([pref, Lh], dim=-1))
        else:
            right = pref.reshape(N * n2, r, r).clone()
            rankk_chol_update_(right, Lh.reshape(N * n2, r, -1))
            right = right.reshape(N, n2, r, r)
        nxt = torch.empty(N, 2 * n2, r, r, device=dev, dtype=dt)
        nxt[:, 0::2] = pref                    # left child: parent's prefix
        nxt[:, 1::2] = right                   # right child: prefix + left seg
        pref = nxt
    return pref[:, :T]


# SKA whitened core with a PRECOMPUTED L  (gradients identical to ska_core)

class SKACoreGivenL(torch.autograd.Function):
    """y = Cv L^{-T} (alpha W)^K L^{-1} q with L supplied (e.g. by the factor
    scan) instead of torch.linalg.cholesky(G).

    L is numerics-only and NEVER differentiated; the backward is the same
    whitened adjoint as ska_core_torch.SKACoreFn, returning dG expressed
    through L (exact because L L^T = G).  G is an input only so its grad
    slot exists; the forward never reads it.
    """

    @staticmethod
    def forward(ctx, G, M, Cv, q, L, K):
        W = _whiten_M(L, M)
        alpha = _spec_w(W)
        a = alpha.unsqueeze(-1)
        U = [_tri_solve_lower(L, q)]
        for _ in range(K):
            U.append(a * (W @ U[-1]))
        XK = _tri_solve_lowerT(L, U[K])
        y = Cv @ XK
        ctx.K = K
        ctx.save_for_backward(L, W, Cv, alpha, *U)
        return y

    @staticmethod
    def backward(ctx, dY):
        K = ctx.K
        L, W, Cv, alpha, *U = ctx.saved_tensors
        a = alpha.unsqueeze(-1)
        XK = _tri_solve_lowerT(L, U[K])
        dCv = dY @ XK.transpose(-1, -2)
        P = _tri_solve_lower(L, Cv.transpose(-1, -2) @ dY)
        dMw = torch.zeros_like(W)
        dGw = torch.zeros_like(W)
        for i in range(K, 0, -1):
            dMw = dMw + a * (P @ U[i - 1].transpose(-1, -2))
            dGw = dGw - (P @ U[i].transpose(-1, -2))
            P = a * (W.transpose(-1, -2) @ P)
        dGw = dGw - (P @ U[0].transpose(-1, -2))
        dq = _tri_solve_lowerT(L, P)

        def unwhiten(Aw):
            t = _tri_solve_lowerT(L, Aw)
            return _tri_solve_lowerT(L, t.transpose(-1, -2)).transpose(-1, -2)

        dM = unwhiten(dMw)
        dG = unwhiten(dGw)
        dG = 0.5 * (dG + dG.transpose(-1, -2))
        return dG, dM, dCv, dq, None, None


def ska_core_given_L(G, M, Cv, q, L, K):
    return SKACoreGivenL.apply(G, M, Cv, q, L, K)


# verification

if __name__ == "__main__":
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)
    N, T, r, P, K, ridge = 3, 96, 16, 8, 2, 1e-3

    # beta-gated keys, as in SKA: w = sqrt(beta) * z_n
    z = torch.randn(N, T, r)
    z = z / (z.norm(dim=-1, keepdim=True) + 1e-12)
    beta = torch.rand(N, T)
    w = beta.sqrt().unsqueeze(-1) * z

    # 1. exclusive all-prefix factors vs per-prefix refactorization
    for mode in ('givens', 'qr'):
        Ls = all_prefix_chol(w, ridge, downsweep=mode)
        err = 0.0
        eye = torch.eye(r)
        for t in range(T):
            G_t = ridge * eye + (w[:, :t].transpose(-1, -2) @ w[:, :t]
                                 if t else torch.zeros(N, r, r))
            L_ref = torch.linalg.cholesky(G_t)
            err = max(err, (Ls[:, t] - L_ref).abs().max().item())
        print(f"  scan({mode:6s}) vs fresh cholesky over all {T} prefixes: "
              f"{err:.2e}  (expect ~1e-14)")

    # 2. represented Gram (gauge-free check)
    Gs = Ls @ Ls.transpose(-1, -2)
    Wc = torch.cumsum(torch.einsum('ntr,nts->ntrs', w, w), dim=1)
    G_ref = ridge * torch.eye(r) + torch.cat(
        [torch.zeros(N, 1, r, r), Wc[:, :-1]], dim=1)
    print(f"  represented Gram L L^T vs prefix sums: "
          f"{(Gs - G_ref).abs().max().item():.2e}")

    # 3. ska_core_given_L fwd+bwd vs cholesky-based reference
    A = torch.randn(N, r, r)
    G = A @ A.transpose(-1, -2) + r * torch.eye(r)
    M = 0.3 * torch.randn(N, r, r)
    Cv = torch.randn(N, P, r)
    q = torch.randn(N, r, 1)
    Lg = torch.linalg.cholesky(G)

    def ref(G, M, Cv, q):
        L = torch.linalg.cholesky(G)
        W = _whiten_M(L, M)
        a = _spec_w(W).unsqueeze(-1)
        U = _tri_solve_lower(L, q)
        for _ in range(K):
            U = a * (W @ U)
        return Cv @ _tri_solve_lowerT(L, U)

    ins_c = [t.clone().requires_grad_(True) for t in (G, M, Cv, q)]
    ska_core_given_L(*ins_c, Lg, K).sum().backward()
    ins_r = [t.clone().requires_grad_(True) for t in (G, M, Cv, q)]
    ref(*ins_r).sum().backward()
    for nm, gc, gr in zip("GMCq", [t.grad for t in ins_c],
                          [t.grad for t in ins_r]):
        rel = (gc - gr).norm() / (gr.norm() + 1e-12)
        print(f"  given-L core d/{nm}: {rel:.2e}")

    # 4. decode: carried L + rank-1 cholupdate vs refactor
    Lc = math.sqrt(ridge) * torch.eye(r).expand(N, r, r).contiguous()
    err = 0.0
    for t in range(T):
        err = max(err, (Lc - Ls[:, t]).abs().max().item())   # read-before-write
        rank1_chol_update_(Lc, w[:, t])                       # write after read
    print(f"  streaming rank-1 cholupdate vs scan prefixes: {err:.2e}")
    print("  all checks passed" if err < 1e-10 else "  CHECK FAILED")
