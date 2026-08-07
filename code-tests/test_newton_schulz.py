"""Newton-Schulz (all-matmul) vs Cholesky core equivalence (Phase 0, test 3).

The matmul-only NS core (ska_core_ns) must compute the same SKA operator as the
Cholesky core (ska_core). Equivalence is EXACT (up to NS inverse-sqrt accuracy)
when the operator is contractive (spectral clamp alpha == 1), which we enforce
by scaling M small. fp32/fp64 run on CPU here; bf16 (the plan's <= 1e-3 target)
is marked gpu.
"""
import pytest
import torch

from koopman_lm.modules.kernels.ska_operator import (
    ska_core,
    ska_core_ns,
    _inv_sqrt_ns,
    _whiten_M,
    _spec_w,
)

pytestmark = pytest.mark.correctness


def _contractive_inputs(N, r, P, p, dtype, seed=0):
    """SPD G, small M so sigma_max(W) < 1 in both gauges (alpha == 1)."""
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(N, r, r, dtype=dtype, generator=g)
    G = A @ A.transpose(-1, -2) + r * torch.eye(r, dtype=dtype)
    M = 0.01 * torch.randn(N, r, r, dtype=dtype, generator=g)   # tiny -> contractive
    Cv = torch.randn(N, P, r, dtype=dtype, generator=g)
    q = torch.randn(N, r, p, dtype=dtype, generator=g)
    return G, M, Cv, q


def test_inv_sqrt_ns_accuracy_fp64():
    g = torch.Generator().manual_seed(1)
    A = torch.randn(4, 16, 16, dtype=torch.float64, generator=g)
    G = A @ A.transpose(-1, -2) + 16 * torch.eye(16, dtype=torch.float64)
    Gi2 = _inv_sqrt_ns(G, iters=30)
    eye = torch.eye(16, dtype=torch.float64)
    # G^{-1/2} G G^{-1/2} == I  and symmetric
    recon = Gi2 @ G @ Gi2
    assert (recon - eye).abs().max() < 1e-7
    assert (Gi2 - Gi2.transpose(-1, -2)).abs().max() < 1e-9


@pytest.mark.parametrize("dtype,tol", [(torch.float64, 1e-7), (torch.float32, 2e-3)])
def test_ns_equals_cholesky_core(dtype, tol):
    N, r, P, p, K = 4, 16, 8, 4, 2
    G, M, Cv, q = _contractive_inputs(N, r, P, p, dtype)
    # confirm we are in the alpha == 1 (contractive) regime in both gauges
    L = torch.linalg.cholesky(G)
    assert torch.allclose(_spec_w(_whiten_M(L, M)), torch.ones(N, 1, dtype=dtype))
    y_chol = ska_core(G, M, Cv, q, K)
    y_ns = ska_core_ns(G, M, Cv, q, K, ns_iters=30 if dtype == torch.float64 else 20)
    rel = (y_ns - y_chol).norm() / (y_chol.norm() + 1e-12)
    assert rel < tol, f"NS vs Cholesky rel err {rel:.2e} (dtype={dtype})"


@pytest.mark.gpu
def test_ns_equals_cholesky_bf16():
    N, r, P, p, K = 4, 16, 8, 4, 2
    G, M, Cv, q = _contractive_inputs(N, r, P, p, torch.float32)
    G, M, Cv, q = (t.cuda() for t in (G, M, Cv, q))
    y_chol = ska_core(G.float(), M.float(), Cv.float(), q.float(), K)
    y_ns = ska_core_ns(G.bfloat16().float(), M.bfloat16().float(),
                       Cv.bfloat16().float(), q.bfloat16().float(), K, ns_iters=20)
    rel = (y_ns - y_chol).norm() / (y_chol.norm() + 1e-12)
    assert rel < 1e-3, f"NS vs Cholesky bf16 rel err {rel:.2e}"

