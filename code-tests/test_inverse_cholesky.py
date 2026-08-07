"""Small-rank exact per-token SKA path (inverse-Cholesky representation).

Validates inverse_cholesky.py -- the replacement for chunk-64 statistics +
cross-chunk boundary terms:

  1. forward == a per-token streaming reference (read prefix stats, then
     write token t), i.e. training == decode semantics by construction;
  2. the whitened operator is contractive under the v1.1 symmetric sqrt-beta
     keys, so dropping the spectral power iteration (alpha == 1) is exact;
  3. custom whitened adjoint (P never differentiated) == autograd through a
     Cholesky-solve reference of the same function;
  4. strict causality, AND visibility of token t-1 (the staleness the chunked
     path suffers within a chunk);
  5. SKAModule integration: inverse_cholesky=True runs fwd+bwd and agrees
     with the (slow) exact_intrachunk factor-scan path.

All pure-torch linear algebra -> runs on CPU.
"""
import math

import pytest
import torch

from koopman_lm.modules.kernels.inverse_cholesky import (
    prefix_inverse_factors, ska_core_inv_chol, ska_exact_inverse_cholesky)
from koopman_lm.modules.kernels.chunk_stats import symmetric_key_value
from koopman_lm.modules.kernels.chunk_stats_exact import exact_stats

pytestmark = pytest.mark.correctness

RIDGE = 1e-3
JITTER = 1e-4          # exact_stats adds this on top of the ridge
K = 2


def _inputs(B=2, T=48, H=2, r=16, P=8, seed=0, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(B, T, H, r, generator=g, dtype=dtype)
    z = z / (z.norm(dim=-1, keepdim=True) + 1e-12)
    zq = torch.randn(B, T, H, r, generator=g, dtype=dtype)
    zq = zq / (zq.norm(dim=-1, keepdim=True) + 1e-12)
    beta = torch.rand(B, T, H, generator=g, dtype=dtype)
    v = torch.randn(B, T, H, P, generator=g, dtype=dtype)
    x, vbar = symmetric_key_value(z, beta, v)
    return x, zq, vbar


def _reference_streaming(x, zq, vbar, ridge_total, K):
    """Per-token decode reference: READ from prefix stats via fresh Cholesky
    solves (no clamp -- contractive), then WRITE token t. O(T r^3), fp64."""
    B, T, H, r = x.shape
    P = vbar.shape[-1]
    dt = x.dtype
    eye = torch.eye(r, dtype=dt)
    G = ridge_total * eye.expand(B, H, r, r).clone()
    M = torch.zeros(B, H, r, r, dtype=dt)
    Cv = torch.zeros(B, H, P, r, dtype=dt)
    x_last = None
    y = torch.zeros(B, T, H, P, dtype=dt)
    for t in range(T):
        L = torch.linalg.cholesky(G)
        q = zq[:, t].unsqueeze(-1)                       # (B,H,r,1)
        u = torch.linalg.solve_triangular(L, q, upper=False)
        W = torch.linalg.solve_triangular(L, M, upper=False)
        W = torch.linalg.solve_triangular(
            L, W.transpose(-1, -2), upper=False).transpose(-1, -2)
        for _ in range(K):
            u = W @ u
        u = torch.linalg.solve_triangular(
            L.transpose(-1, -2), u, upper=True)
        y[:, t] = (Cv @ u).squeeze(-1)
        xt = x[:, t]
        G = G + torch.einsum('bhr,bhs->bhrs', xt, xt)
        if x_last is not None:
            M = M + torch.einsum('bhr,bhs->bhrs', xt, x_last)
        Cv = Cv + torch.einsum('bhp,bhr->bhpr', vbar[:, t], xt)
        x_last = xt
    return y


def test_forward_matches_streaming_decode_reference():
    x, zq, vbar = _inputs()
    y = ska_exact_inverse_cholesky(x, zq, vbar, RIDGE, K)
    y_ref = _reference_streaming(x, zq, vbar, RIDGE + JITTER, K)
    rel = (y - y_ref).norm() / (y_ref.norm() + 1e-30)
    assert rel < 1e-10, f"train path vs streaming decode reference: rel={rel:.3e}"


def test_whitened_operator_is_contractive():
    """||P M P^T||_2 <= 1 for every token under symmetric sqrt-beta keys --
    the guarantee that licenses alpha == 1 (no power iteration)."""
    x, zq, vbar = _inputs(T=64, seed=1)
    Gf, Mf, _, _, _ = exact_stats(x, x, zq, vbar, RIDGE)
    P = prefix_inverse_factors(Gf)
    W = P @ Mf @ P.transpose(-1, -2)
    sigma = torch.linalg.matrix_norm(W, ord=2)
    assert float(sigma.max()) <= 1.0 + 1e-9, \
        f"sigma_max(W) = {float(sigma.max()):.6f} > 1"


def test_gradients_match_autograd_reference():
    """Custom whitened adjoint (P constant) vs autograd through a
    Cholesky-solve forward of the same function y = Cv (G^-1 M)^K G^-1 q."""
    g = torch.Generator().manual_seed(2)
    N, r, P_dim = 6, 12, 5
    A = torch.randn(N, r, r, generator=g, dtype=torch.float64)
    G = A @ A.transpose(-1, -2) + r * torch.eye(r, dtype=torch.float64)
    M = 0.3 * torch.randn(N, r, r, generator=g, dtype=torch.float64)
    Cv = torch.randn(N, P_dim, r, generator=g, dtype=torch.float64)
    q = torch.randn(N, r, 1, generator=g, dtype=torch.float64)

    def ref(G, M, Cv, q):
        L = torch.linalg.cholesky(G)
        u = torch.linalg.solve_triangular(L, q, upper=False)
        W = torch.linalg.solve_triangular(L, M, upper=False)
        W = torch.linalg.solve_triangular(
            L, W.transpose(-1, -2), upper=False).transpose(-1, -2)
        for _ in range(K):
            u = W @ u
        u = torch.linalg.solve_triangular(L.transpose(-1, -2), u, upper=True)
        return Cv @ u

    ins_c = [t.clone().requires_grad_(True) for t in (G, M, Cv, q)]
    Pinv = prefix_inverse_factors(G)
    ska_core_inv_chol(*ins_c, Pinv, K).sum().backward()
    ins_r = [t.clone().requires_grad_(True) for t in (G, M, Cv, q)]
    ref(*ins_r).sum().backward()
    for nm, gc, gr in zip("GMCq", [t.grad for t in ins_c],
                          [t.grad for t in ins_r]):
        rel = (gc - gr).norm() / (gr.norm() + 1e-30)
        assert rel < 1e-9, f"d/{nm}: rel={rel:.3e}"


def test_causality_and_recent_token_visibility():
    x, zq, vbar = _inputs(T=40, seed=3)
    y1 = ska_exact_inverse_cholesky(x, zq, vbar, RIDGE, K)

    # perturb token t0 (mid-chunk position -- NOT a chunk-64 boundary)
    t0 = 21
    x2, vbar2 = x.clone(), vbar.clone()
    x2[:, t0] = torch.randn_like(x2[:, t0])
    vbar2[:, t0] = torch.randn_like(vbar2[:, t0])
    y2 = ska_exact_inverse_cholesky(x2, zq, vbar2, RIDGE, K)

    # strict causality: outputs at t <= t0 unchanged (stats are exclusive)
    leak = (y1[:, :t0 + 1] - y2[:, :t0 + 1]).abs().max()
    assert float(leak) < 1e-12, f"future->past leak {float(leak):.3e}"
    # the fix vs chunked stats: the VERY NEXT token already sees the change
    seen = (y1[:, t0 + 1] - y2[:, t0 + 1]).abs().max()
    assert float(seen) > 1e-8, \
        "token t+1 blind to a write at t -- staleness not fixed"


def test_ska_module_integration_and_exact_path_agreement():
    from koopman_lm.modules.token_mixer.ska import SKAModule
    torch.manual_seed(4)
    kw = dict(d_model=64, n_heads=2, rank=16, head_dim=8, power_K=2,
              chunk_size=16, backend='pytorch', eta_learnable=False,
              eta_value=1.0, gamma_learnable=False, gamma_value=1.0,
              layerscale=True)
    m_new = SKAModule(inverse_cholesky=True, **kw)
    m_old = SKAModule(exact_intrachunk=True, **kw)
    m_old.load_state_dict(m_new.state_dict())

    h = torch.randn(2, 33, 64)
    out = m_new(h)
    assert out.shape == (2, 33, 64)
    out.square().mean().backward()
    for n, p in m_new.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), f"non-finite grad in {n}"

    # both are exact per-token paths; the factor-scan path's spectral clamp is
    # inactive (contractive operator), so outputs agree to fp32 numerics.
    with torch.no_grad():
        y_new = m_new(h)
        y_old = m_old(h)
    rel = (y_new - y_old).norm() / (y_old.norm() + 1e-30)
    assert rel < 1e-4, f"inverse_cholesky vs exact_intrachunk: rel={rel:.3e}"
