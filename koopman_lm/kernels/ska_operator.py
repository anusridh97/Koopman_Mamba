"""
ska_core_torch.py -- PyTorch port of echo_jax.py's whitened SKA operator core.

Mirrors, in the SAME gauge and math as the JAX reference:
  forward:  y = C_v @ L^{-T} (alpha W)^K L^{-1} q,   W = L^{-1} M L^{-T}
            alpha = 1/max(sigma_max(W), 1)   (detached spectral norm, power iter)
  backward: hand-derived whitened adjoint (reverse Krylov recurrence);
            the Cholesky factor L is NEVER differentiated ("It Cancels").

This is the correctness path. It is validated here against autograd-THROUGH
the same forward (the baseline repo's implicit method) to ~1e-6 relative grad,
exactly like echo_jax.py::test_backward.

Operates on a single (batched-by-vmap-equivalent) instance set:
  G,M : (..., r, r)   Cv : (..., P, r)   q : (..., r, p)   -> y : (..., P, p)
We implement the per-instance math and rely on torch batched matmul/cholesky
over leading dims (the BCH flattening done by the caller).
"""

import torch

from koopman_lm.kernels.lin_alg import (
    spec_w, tri_solve_lower, tri_solve_lowerT, whiten_M, inv_sqrt_ns)


class SKACoreFn(torch.autograd.Function):
    """Whitened SKA core with hand-derived O(K r^2) backward (Cholesky not diff'd).

    Mirrors echo_jax.py _ska_fwd / _ska_bwd exactly.
    """

    @staticmethod
    def forward(ctx, G, M, Cv, q, K):
        # All in fp32 for the linear algebra (caller upcasts).
        L = torch.linalg.cholesky(G)                  # G = L L^T
        W = whiten_M(L, M)                            # W = L^{-1} M L^{-T}
        alpha = spec_w(W)                             # (...,1) detached
        a = alpha.unsqueeze(-1)                        # (...,1,1) for broadcasting

        U = [tri_solve_lower(L, q)]                   # U_0 = L^{-1} q
        for _ in range(K):
            U.append(a * (W @ U[-1]))                  # (alpha W) applied
        XK = tri_solve_lowerT(L, U[K])                # un-whiten once
        y = Cv @ XK

        ctx.K = K
        ctx.save_for_backward(L, W, Cv, alpha, *U)
        return y

    @staticmethod
    def backward(ctx, dY):
        K = ctx.K
        L, W, Cv, alpha, *U = ctx.saved_tensors
        a = alpha.unsqueeze(-1)

        # dCv = dY @ XK^T,  XK = L^{-T} U[K]
        XK = tri_solve_lowerT(L, U[K])
        dCv = dY @ XK.transpose(-1, -2)

        # P_K = L^{-1} (Cv^T dY)
        P = tri_solve_lower(L, Cv.transpose(-1, -2) @ dY)

        dMw = torch.zeros_like(W)
        dGw = torch.zeros_like(W)
        for i in range(K, 0, -1):
            dMw = dMw + a * (P @ U[i - 1].transpose(-1, -2))
            dGw = dGw - (P @ U[i].transpose(-1, -2))
            P = a * (W.transpose(-1, -2) @ P)
        dGw = dGw - (P @ U[0].transpose(-1, -2))       # i = 0 term
        dq = tri_solve_lowerT(L, P)

        # un-whiten the whitened-space grads: dA = L^{-T} dAw L^{-1}
        def unwhiten(Aw):
            t = tri_solve_lowerT(L, Aw)               # L^{-T} Aw
            return tri_solve_lowerT(L, t.transpose(-1, -2)).transpose(-1, -2)

        dM = unwhiten(dMw)
        dG = unwhiten(dGw)
        dG = 0.5 * (dG + dG.transpose(-1, -2))
        return dG, dM, dCv, dq, None


def ska_core(G, M, Cv, q, K):
    return SKACoreFn.apply(G, M, Cv, q, K)


def _ska_whitened_forward(L, M, Cv, q, K, need_trace=False):
    """The whitened SKA forward core, given a precomputed Cholesky factor L:

        y = Cv @ L^{-T} (alpha W)^K L^{-1} q,   W = L^{-1} M L^{-T}

    This is the single shared implementation behind three call sites that
    were previously bit-identical copies of this loop: _ref_core (below,
    which computes L = chol(G) itself), factor_scan.SKACoreGivenL.forward
    (which takes L as an argument and needs the intermediate U's for its
    custom backward), and models.recurrent's decode-time whitened apply
    (which takes L, runs under no_grad, and additionally scales by
    gamma_value ** K). See
    docs/superpowers/specs/2026-08-07-structural-review.md, issue 3.

    If need_trace, also returns (W, alpha, U) with U = [U_0, ..., U_K], the
    state SKACoreGivenL.forward needs to save for backward.
    """
    W = whiten_M(L, M)
    alpha = spec_w(W)
    a = alpha.unsqueeze(-1)
    if need_trace:
        U = [tri_solve_lower(L, q)]
        for _ in range(K):
            U.append(a * (W @ U[-1]))
        XK = tri_solve_lowerT(L, U[K])
        y = Cv @ XK
        return y, W, alpha, U
    U = tri_solve_lower(L, q)
    for _ in range(K):
        U = a * (W @ U)
    XK = tri_solve_lowerT(L, U)
    return Cv @ XK


def ska_decode_whitened(L, M, Cv, q, K, gamma_value):
    """Decode-time whitened apply: the shared forward core plus the
    gamma_value ** K scaling the O(1)-state recurrent decode path needs.
    No backward (decode runs under no_grad).
    L:(N,r,r) M:(N,r,r) Cv:(N,P,r) q:(N,r,1) -> (N,P,1)."""
    y = _ska_whitened_forward(L, M, Cv, q, K)
    if isinstance(gamma_value, float):
        if gamma_value != 1.0:
            y = y * (gamma_value ** K)
    else:
        y = y * (gamma_value ** K)
    return y


# --- reference: autodiff THROUGH the same forward (gauge-identical) ---
def _ref_core(G, M, Cv, q, K):
    L = torch.linalg.cholesky(G)
    return _ska_whitened_forward(L, M, Cv, q, K)


# ---------------------------------------------------------------------------
# Newton-Schulz (all-matmul) core -- Cholesky-free alternative.
#
# The Cholesky core above uses torch.linalg.cholesky + triangular solves (a
# sequential dependency that compiles poorly). This computes the SAME operator
#     y = Cv (G^{-1} M)^K G^{-1} q   (with the spectral-norm clamp)
# using a symmetric Newton-Schulz inverse square root (pure matmuls), the
# "all-MXU" path echo_jax.py::test_ska_newton_schulz checks. It is what the
# fast-inference track wants to profile against Cholesky on B200.
#
# Gauge invariance: for ANY factor F with F F^T = G^{-1},
#     F (alpha F^T M F)^K F^T = (alpha-scaled) (G^{-1} M)^K G^{-1},
# so the Cholesky factor (F=L^{-T}) and the symmetric root (F=G^{-1/2}) give the
# SAME y -- EXACTLY when the operator is contractive (alpha == 1, no clamp), and
# approximately otherwise (the clamp uses sigma_max(W), which is gauge-dependent
# for non-symmetric M). See tests/test_newton_schulz.py.
# ---------------------------------------------------------------------------


def ska_core_ns(G, M, Cv, q, K, ns_iters=25):
    """All-matmul (Newton-Schulz) analogue of :func:`ska_core` -- no Cholesky,
    no triangular solves. Uses a symmetric NS inverse-sqrt for the whitening.
    Forward-only (no custom backward); for training use ``ska_core``. See module
    docstring for the gauge-equivalence conditions vs the Cholesky core."""
    Gi2 = inv_sqrt_ns(G, ns_iters)                # symmetric G^{-1/2}
    W = Gi2 @ M @ Gi2                              # whitened operator (sym gauge)
    alpha = spec_w(W).unsqueeze(-1)               # (...,1,1)
    U = Gi2 @ q
    for _ in range(K):
        U = alpha * (W @ U)
    XK = Gi2 @ U
    return Cv @ XK


if __name__ == "__main__":
    torch.manual_seed(0)
    r, P, p, K = 16, 8, 4, 2
    A = torch.randn(r, r, dtype=torch.float64)
    G = A @ A.T + r * torch.eye(r, dtype=torch.float64)
    M = 0.3 * torch.randn(r, r, dtype=torch.float64)
    Cv = torch.randn(P, r, dtype=torch.float64)
    q = torch.randn(r, p, dtype=torch.float64)

    # custom backward
    Gc, Mc, Cc, qc = [t.clone().requires_grad_(True) for t in (G, M, Cv, q)]
    ska_core(Gc, Mc, Cc, qc, K).sum().backward()

    # reference (autodiff through forward)
    Gr, Mr, Cr, qr = [t.clone().requires_grad_(True) for t in (G, M, Cv, q)]
    _ref_core(Gr, Mr, Cr, qr, K).sum().backward()

    for nm, gc, gr in zip("GMCq", (Gc.grad, Mc.grad, Cc.grad, qc.grad),
                                   (Gr.grad, Mr.grad, Cr.grad, qr.grad)):
        rel = (gc - gr).norm() / (gr.norm() + 1e-12)
        print(f"  d/{nm}: {rel:.2e}")
