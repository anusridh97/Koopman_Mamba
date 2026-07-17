"""Causal norm-clip (memo §6) property tests -- exercises the shipped helper.

Requires torch (CI). Validates the claims the norm-clip is for: it bounds
leverage, does NOT inflate low-norm tokens to unit norm (the L2 pathology
Appendix E Remark 5 warns about), and leaves contractivity intact.
"""
import pytest
import torch

from koopman_lm.globals.modules.ska.chunk_stats import causal_normalize

pytestmark = pytest.mark.correctness


def test_norm_clip_vs_l2_properties():
    torch.manual_seed(0)
    r = 24; c = r ** 0.5
    U = torch.cat([0.1 * torch.randn(5, r),      # low-norm (distractor-like)
                   3.0 * torch.randn(5, r),
                   8.0 * torch.randn(5, r)])      # high-norm (fact-like)
    l2 = causal_normalize(U, None)                # legacy per-token L2
    cl = causal_normalize(U, c)                   # norm-clip
    raw = U.norm(dim=-1); low = raw < c
    # L2 forces EVERY token to unit norm (inflates low-norm distractors)
    assert torch.allclose(l2.norm(dim=-1), torch.ones(15), atol=1e-5)
    # clip bounds leverage and keeps sub-threshold tokens exactly
    assert (cl.norm(dim=-1) <= c + 1e-4).all()
    assert torch.allclose(cl[low], U[low], atol=1e-6)
    assert torch.allclose(cl[~low].norm(dim=-1),
                          torch.full(((~low).sum(),), c), atol=1e-3)


def test_norm_clip_preserves_contractivity():
    """A = L^-1 M L^-T stays contractive with clipped keys -- the bound needs
    only G = eps I + Σ x xᵀ (any x), not unit-norm keys."""
    torch.manual_seed(1)
    r, T, eps, c = 24, 1500, 1e-3, 24 ** 0.5
    Z = torch.randn(T, r); beta = torch.rand(T) * 0.95 + 0.05
    x = beta.sqrt().unsqueeze(-1) * causal_normalize(Z, c)
    G = eps * torch.eye(r) + x.T @ x
    M = x[1:].T @ x[:-1]
    Li = torch.linalg.inv(torch.linalg.cholesky(G))
    assert torch.linalg.matrix_norm(Li @ M @ Li.T, ord=2) <= 1 + 1e-5
