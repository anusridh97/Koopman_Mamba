"""
lin_alg.py -- shared linear-algebra primitives for the whitened SKA operator.

Small, generic numerical building blocks. They compose into the operator in
ska_operator.py, but are ALSO reused directly by the recurrent decode path and
factor_scan -- so they live on their own rather than inside the operator file:

  * _tri_solve_lower / _tri_solve_lowerT -- triangular solves L^{-1}A, L^{-T}A
  * _whiten_M                            -- double-sided whitening L^{-1} M L^{-T}
  * _spec_w                              -- detached spectral-norm power iteration
                                            (returns alpha = 1/max(sigma_max, 1))
  * _inv_sqrt_ns                         -- symmetric inverse sqrt G^{-1/2} via
                                            coupled Newton-Schulz (matmul-only)

All match echo_jax.py's gauge and math exactly; see ska_operator.py for how they
assemble into the forward/backward.
"""

import torch


def _spec_w(W, iters=20, return_sigma=False):
    """sigma_max(W) via detached power iteration; returns alpha=1/max(sigma,1).
    iters=20 matches the JAX core (converges on ill-conditioned chunks).
    Detached: straight-through, no grad through the scale (as in JAX _specW).

    return_sigma=True additionally returns sigma_max(W) itself (the pre-clamp
    spectral norm), for diagnostics -- the default (alpha only) is unchanged so
    the forward-path callers are untouched.
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
    if return_sigma:
        return alpha, sigma  # both (..., 1)
    return alpha  # (..., 1)


def _tri_solve_lower(L, A):     # L^{-1} A
    return torch.linalg.solve_triangular(L, A, upper=False)


def _tri_solve_lowerT(L, A):    # L^{-T} A
    return torch.linalg.solve_triangular(L.transpose(-1, -2), A, upper=True)


def _whiten_M(L, M):            # W = L^{-1} M L^{-T}
    # L^{-1} M, then ( L^{-1} (that)^T )^T = L^{-1} M L^{-T}
    LiM = _tri_solve_lower(L, M)
    return _tri_solve_lower(L, LiM.transpose(-1, -2)).transpose(-1, -2)


def _inv_sqrt_ns(G, iters=25):
    """Symmetric inverse square root G^{-1/2} via coupled Newton-Schulz (matmul
    only). G: (...,r,r) SPD. Scaled by ||G||_F (>= spectral norm for SPD) so the
    eigenvalues land in (0,1] and the coupled iteration converges; convergence
    is fast for well-conditioned G and slows with the condition number."""
    r = G.shape[-1]
    eye = torch.eye(r, device=G.device, dtype=G.dtype).expand_as(G)
    norm = torch.linalg.matrix_norm(G, ord='fro', keepdim=True)
    Y = G / norm
    Z = eye.clone()
    for _ in range(iters):
        T = 1.5 * eye - 0.5 * (Z @ Y)
        Y = Y @ T
        Z = T @ Z
    # Z -> (G/norm)^{-1/2}  ==>  G^{-1/2} = Z / sqrt(norm)
    return Z / torch.sqrt(norm)
