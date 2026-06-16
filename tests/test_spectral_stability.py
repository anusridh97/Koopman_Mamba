"""Spectral-norm stability over 500 random inputs (scaling plan Phase 0, test 4).

After the detached spectral clamp alpha = 1/max(sigma_max(W), 1), the scaled
operator alpha*W must stay bounded: sigma_max(alpha*W) <= 1 (so the spectral
radius rho(alpha*W) <= 1 too). This verifies the power-iteration estimate in
_spec_w is accurate enough that the TRUE largest singular value of the clamped
operator does not blow past 1 across 500 random instances. fp32 on CPU here;
bf16 marked gpu.
"""
import pytest
import torch

from koopman_lm.ska_core_torch import _whiten_M, _spec_w

pytestmark = pytest.mark.correctness

N_RANDOM = 500
SLACK = 0.05      # tolerance for the 20-iter power-method estimate vs true svd


def _random_instances(n, r, dtype, device="cpu", seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    A = torch.randn(n, r, r, dtype=dtype, device=device, generator=g)
    G = A @ A.transpose(-1, -2) + r * torch.eye(r, dtype=dtype, device=device)
    # M with a spread of scales so some instances WOULD exceed 1 pre-clamp
    scales = torch.logspace(-1, 1, n, dtype=dtype, device=device).reshape(n, 1, 1)
    M = scales * torch.randn(n, r, r, dtype=dtype, device=device, generator=g)
    return G, M


def _check_bounded(G, M, slack):
    L = torch.linalg.cholesky(G)
    W = _whiten_M(L, M)
    alpha = _spec_w(W).unsqueeze(-1)               # (n,1,1)
    scaled = alpha * W
    # true largest singular value of the clamped operator
    sigma_true = torch.linalg.matrix_norm(scaled, ord=2)        # (n,)
    # spectral radius (max |eigenvalue|) <= sigma always; check both finite
    eig = torch.linalg.eigvals(scaled)
    rho = eig.abs().amax(dim=-1)
    assert torch.isfinite(sigma_true).all()
    assert torch.isfinite(rho).all()
    assert sigma_true.max().item() <= 1.0 + slack, \
        f"max sigma_max(alpha*W) = {sigma_true.max().item():.4f} > 1+{slack}"
    assert rho.max().item() <= 1.0 + slack


def test_spectral_radius_bounded_fp32_500():
    G, M = _random_instances(N_RANDOM, r=24, dtype=torch.float32)
    _check_bounded(G, M, SLACK)


def test_spectral_radius_bounded_fp64():
    # fp64 numerical path; the clamp bound is governed by the 20-iter power
    # method (iteration-bound, not dtype-bound), so the same slack applies.
    G, M = _random_instances(N_RANDOM, r=24, dtype=torch.float64, seed=1)
    _check_bounded(G, M, SLACK)


@pytest.mark.gpu
def test_spectral_radius_bounded_bf16():
    G, M = _random_instances(N_RANDOM, r=24, dtype=torch.float32, device="cuda", seed=2)
    # exercise the bf16 whitening path; clamp must still bound the operator
    _check_bounded(G.bfloat16().float(), M.bfloat16().float(), 0.1)
