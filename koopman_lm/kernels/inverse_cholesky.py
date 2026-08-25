"""
inverse_cholesky.py -- EXACT per-token SKA via the inverse-Cholesky
representation (small-rank path).

Replaces the chunk-64 statistics + cross-chunk boundary machinery. The chunked
path gives token t only the statistics of COMPLETED chunks < c, so it never
sees the most recent (up to 63) tokens, and the cross-chunk boundary term only
patches the single chunk-edge lag -- short-range recall stays ~100% stale.
Here every token reads the exclusive prefix over ALL earlier tokens, including
t-1, so training matches the per-token decode recurrence exactly.

This is the training-time analogue of the fused CUDA step kernel
(csrc/small_rank_ext.cu), which maintains P = L^{-1} and the whitened operator
incrementally with O(r^2)/token Givens rotations at decode. In training all
tokens are available at once, so instead of a sequential recurrence we:

  1. build exact per-token exclusive-prefix stats G_t, M_t, Cv_t by cumulative
     sums (differentiable, cheap) -- reusing chunk_stats_exact.exact_stats;
  2. factor ALL prefixes in one batched Cholesky over (B*T*H, r, r) and invert
     the factor once: P_t = L_t^{-1} (numerics-only, no_grad, never
     differentiated -- "It Cancels");
  3. run the whitened core entirely with batched MATMULS against P
     (W = P M P^T, u = P q, y = Cv P^T W^K u): no triangular solves in the hot
     path and NO spectral power iteration.

Dropping the power iteration is safe by construction: with the v1.1 symmetric
sqrt(beta) keys (x = sqrt(beta) z in both slots), M_t = X_1^T X_0 where the
rows of X_1 and X_0 are subsets of the rows whose Gram (plus ridge) is G_t, so
||P M P^T||_2 <= 1 by Cauchy-Schwarz. The 20-iteration power iteration over
B*T*H matrices was a dominant cost of the previous exact path; here alpha == 1
identically.

Cost/memory: per-token stats are (B,T,H,r,r) (and Cv (B,T,H,P,r)), so this
path is intended for SMALLER rank (r <= 32 recommended, hard cap enforced by
the caller). At r=16-32 the batched r x r matmuls are cheap and the whole path
is a handful of large batched GEMMs -- no sequential scan, no QR factor scan,
no per-chunk bookkeeping.

Gradients: SKACoreInvChol reuses the verified whitened adjoint of
core.SKACoreFn / factor_scan.SKACoreGivenL with every triangular solve
replaced by a matmul with P (exact because P = L^{-1} and L L^T = G). L/P are
numerics-only; dG/dM/dCv/dq flow through the (linear) prefix sums.
"""

import torch

from koopman_lm.kernels.chunk_stats_exact import exact_stats


@torch.no_grad()
def prefix_inverse_factors(Gf):
    """Batched inverse Cholesky factors P = L^{-1} of SPD Gf (N, r, r).

    Numerics-only (no_grad): the factor is never differentiated. Uses the same
    jitter fallback as the chunked path for prefixes that are numerically
    semidefinite in fp32.
    """
    N, r, _ = Gf.shape
    eye = torch.eye(r, dtype=Gf.dtype, device=Gf.device)
    L, info = torch.linalg.cholesky_ex(Gf)
    if bool((info > 0).any()):
        Lj, _ = torch.linalg.cholesky_ex(Gf + 1e-4 * eye)
        L = torch.where((info > 0).reshape(-1, 1, 1), Lj, L)
    P = torch.linalg.solve_triangular(L, eye.expand(N, r, r), upper=False)
    return P


class SKACoreInvChol(torch.autograd.Function):
    """y = Cv P^T W^K P q with W = P M P^T, P = L^{-1} supplied, alpha == 1.

    Same math and gauge as factor_scan.SKACoreGivenL with the spectral clamp
    removed and every L-solve replaced by a matmul with the precomputed P. P is
    NEVER differentiated; G is an input only so its grad slot exists (the
    forward never reads it).

    The clamp is removable because BOTH SLOTS OF M COME FROM THE SAME KEY
    STREAM whose Gram (plus ridge) is G, which bounds ||P M P^T||_2 by 1 --
    NOT because of the square root specifically. This said "contractive by the
    sqrt-beta convention" until 2026-08-24, which is the same claim narrowed to
    one member of the family that satisfies it: beta == 1
    (`ska_beta_policy='one'`) and beta in both slots (`'linear'`) are equally
    contractive and equally legal on this route. What the two-slot requirement
    rules out is the retired ASYMMETRIC form, which exceeds the bound by 22.8x
    on a sharp gate and would apply an expansive operator here without failing.
    Pinned by code-tests/test_ska_contractivity_contract.py.
    """

    @staticmethod
    def forward(ctx, G, M, Cv, q, P, K):
        Pt = P.transpose(-1, -2)
        W = P @ M @ Pt                     # whitened operator, ||W|| <= 1
        U = [P @ q]                        # U_0 = L^{-1} q
        for _ in range(K):
            U.append(W @ U[-1])
        XK = Pt @ U[K]                     # un-whiten once
        y = Cv @ XK
        ctx.K = K
        ctx.save_for_backward(P, W, Cv, *U)
        return y

    @staticmethod
    def backward(ctx, dY):
        K = ctx.K
        P, W, Cv, *U = ctx.saved_tensors
        Pt = P.transpose(-1, -2)

        XK = Pt @ U[K]
        dCv = dY @ XK.transpose(-1, -2)

        # R_K = L^{-1} (Cv^T dY)
        R = P @ (Cv.transpose(-1, -2) @ dY)
        dMw = torch.zeros_like(W)
        dGw = torch.zeros_like(W)
        for i in range(K, 0, -1):
            dMw = dMw + R @ U[i - 1].transpose(-1, -2)
            dGw = dGw - R @ U[i].transpose(-1, -2)
            R = W.transpose(-1, -2) @ R
        dGw = dGw - R @ U[0].transpose(-1, -2)     # i = 0 term
        dq = Pt @ R

        # un-whiten: dA = L^{-T} dAw L^{-1} = P^T dAw P
        dM = Pt @ dMw @ P
        dG = Pt @ dGw @ P
        dG = 0.5 * (dG + dG.transpose(-1, -2))
        return dG, dM, dCv, dq, None, None


def ska_core_inv_chol(G, M, Cv, q, P, K):
    return SKACoreInvChol.apply(G, M, Cv, q, P, K)


def ska_exact_inverse_cholesky(x_n, zq_n, v_w, ridge, K):
    """Full small-rank path: exact per-token stats -> batched P -> matmul core.

    Inputs follow the v1.1 symmetric convention (see symmetric_key_value):
      x_n  (B,T,H,r)  symmetric key sqrt(beta) * z_n, fed to BOTH key slots
      zq_n (B,T,H,r)  normalized query (NOT beta-weighted)
      v_w  (B,T,H,P)  sqrt(beta) * v
    Returns y (B,T,H,P): exact per-token-causal output (token t reads the
    exclusive prefix over ALL earlier tokens, including t-1).
    """
    Gf, Mf, Cf, qf, (B, T, H, P) = exact_stats(x_n, x_n, zq_n, v_w, ridge)
    Pf = prefix_inverse_factors(Gf)
    Y = ska_core_inv_chol(Gf, Mf, Cf, qf, Pf, K)          # (B*T*H, P, 1)
    return Y.reshape(B, T, H, P)
