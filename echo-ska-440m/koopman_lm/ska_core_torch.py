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


def _spec_w(W, iters=20):
    """sigma_max(W) via detached power iteration; returns alpha=1/max(sigma,1).
    iters=20 matches the JAX core (converges on ill-conditioned chunks).
    Detached: straight-through, no grad through the scale (as in JAX _specW).
    """
    r = W.shape[-1]
    v = torch.ones(*W.shape[:-1], 1, device=W.device, dtype=W.dtype) / (r ** 0.5)
    with torch.no_grad():
        for _ in range(iters):
            u = W @ v
            u = u / (u.norm(dim=-2, keepdim=True) + 1e-8)
            v = W.transpose(-1, -2) @ u
            v = v / (v.norm(dim=-2, keepdim=True) + 1e-8)
        sigma = (W @ v).norm(dim=-2, keepdim=True)            # (...,1,1)? -> (...,1)
        sigma = sigma.squeeze(-1)                              # (...,1) -> match
        alpha = 1.0 / torch.clamp(sigma, min=1.0)
    return alpha  # (..., 1)


def _tri_solve_lower(L, A):     # L^{-1} A
    return torch.linalg.solve_triangular(L, A, upper=False)


def _tri_solve_lowerT(L, A):    # L^{-T} A
    return torch.linalg.solve_triangular(L.transpose(-1, -2), A, upper=True)


def _whiten_M(L, M):            # W = L^{-1} M L^{-T}
    # L^{-1} M, then ( L^{-1} (that)^T )^T = L^{-1} M L^{-T}
    LiM = _tri_solve_lower(L, M)
    return _tri_solve_lower(L, LiM.transpose(-1, -2)).transpose(-1, -2)


class SKACoreFn(torch.autograd.Function):
    """Whitened SKA core with hand-derived O(K r^2) backward (Cholesky not diff'd).

    Mirrors echo_jax.py _ska_fwd / _ska_bwd exactly.
    """

    @staticmethod
    def forward(ctx, G, M, Cv, q, K):
        # All in fp32 for the linear algebra (caller upcasts).
        L = torch.linalg.cholesky(G)                  # G = L L^T
        W = _whiten_M(L, M)                            # W = L^{-1} M L^{-T}
        alpha = _spec_w(W)                             # (...,1) detached
        a = alpha.unsqueeze(-1)                        # (...,1,1) for broadcasting

        U = [_tri_solve_lower(L, q)]                   # U_0 = L^{-1} q
        for _ in range(K):
            U.append(a * (W @ U[-1]))                  # (alpha W) applied
        XK = _tri_solve_lowerT(L, U[K])                # un-whiten once
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
        XK = _tri_solve_lowerT(L, U[K])
        dCv = dY @ XK.transpose(-1, -2)

        # P_K = L^{-1} (Cv^T dY)
        P = _tri_solve_lower(L, Cv.transpose(-1, -2) @ dY)

        dMw = torch.zeros_like(W)
        dGw = torch.zeros_like(W)
        for i in range(K, 0, -1):
            dMw = dMw + a * (P @ U[i - 1].transpose(-1, -2))
            dGw = dGw - (P @ U[i].transpose(-1, -2))
            P = a * (W.transpose(-1, -2) @ P)
        dGw = dGw - (P @ U[0].transpose(-1, -2))       # i = 0 term
        dq = _tri_solve_lowerT(L, P)

        # un-whiten the whitened-space grads: dA = L^{-T} dAw L^{-1}
        def unwhiten(Aw):
            t = _tri_solve_lowerT(L, Aw)               # L^{-T} Aw
            return _tri_solve_lowerT(L, t.transpose(-1, -2)).transpose(-1, -2)

        dM = unwhiten(dMw)
        dG = unwhiten(dGw)
        dG = 0.5 * (dG + dG.transpose(-1, -2))
        return dG, dM, dCv, dq, None


def ska_core(G, M, Cv, q, K):
    return SKACoreFn.apply(G, M, Cv, q, K)


# --- reference: autodiff THROUGH the same forward (gauge-identical) ---
def _ref_core(G, M, Cv, q, K):
    L = torch.linalg.cholesky(G)
    W = _whiten_M(L, M)
    alpha = _spec_w(W)
    a = alpha.unsqueeze(-1)
    U = _tri_solve_lower(L, q)
    for _ in range(K):
        U = a * (W @ U)
    XK = _tri_solve_lowerT(L, U)
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
