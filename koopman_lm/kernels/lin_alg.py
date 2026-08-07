"""
lin_alg.py -- shared linear-algebra primitives for the whitened SKA operator.

Small, generic numerical building blocks. They compose into the operator in
ska_operator.py, but are ALSO reused directly by the recurrent decode path and
factor_scan -- so they live on their own rather than inside the operator file:

  * tri_solve_lower / tri_solve_lowerT -- triangular solves L^{-1}A, L^{-T}A
  * whiten_M                           -- double-sided whitening L^{-1} M L^{-T}
  * spec_w                             -- detached spectral-norm power iteration
                                          (returns alpha = 1/max(sigma_max, 1))
  * inv_sqrt_ns                        -- symmetric inverse sqrt G^{-1/2} via
                                          coupled Newton-Schulz (matmul-only)

All match echo_jax.py's gauge and math exactly; see ska_operator.py for how they
assemble into the forward/backward.

Consumers: ska_operator.py, factor_scan.py, incremental_transport.py, and
models/recurrent.py. These names are deliberately PUBLIC (no leading
underscore): a module whose purpose is to be imported by four others should
not mark its exports internal.
"""

import torch

__all__ = [
    "spec_w",
    "tri_solve_lower",
    "tri_solve_lowerT",
    "whiten_M",
    "inv_sqrt_ns",
]


def spec_w(W, iters=20):
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


def tri_solve_lower(L, A):     # L^{-1} A
    return torch.linalg.solve_triangular(L, A, upper=False)


def tri_solve_lowerT(L, A):    # L^{-T} A
    return torch.linalg.solve_triangular(L.transpose(-1, -2), A, upper=True)


def whiten_M(L, M):            # W = L^{-1} M L^{-T}
    # L^{-1} M, then ( L^{-1} (that)^T )^T = L^{-1} M L^{-T}
    LiM = tri_solve_lower(L, M)
    return tri_solve_lower(L, LiM.transpose(-1, -2)).transpose(-1, -2)


def inv_sqrt_ns(G, iters=25):
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
