import pytest
import torch

from koopman_lm.kernels.cuda_prefix_scan import fused_ska_prefix_scan
from koopman_lm.kernels.prefix_scan import dense_exact_oracle

pytestmark = [pytest.mark.gpu, pytest.mark.correctness]


def _cuda_ready() -> bool:
    return torch.cuda.is_available()


@pytest.mark.skipif(not _cuda_ready(), reason="CUDA is required")
@pytest.mark.parametrize("length", [1, 7, 8, 31, 32, 33, 65])
def test_fused_forward_matches_dense(length):
    torch.manual_seed(1000 + length)
    device = torch.device("cuda")
    x = (0.08 * torch.randn(1, length, 1, 24, device=device, dtype=torch.float32)).requires_grad_()
    q = torch.randn_like(x, requires_grad=True)
    v = (0.08 * torch.randn(1, length, 1, 64, device=device, dtype=torch.float32)).requires_grad_()
    y = fused_ska_prefix_scan(x, q, v, ridge=1e-2)
    ref = dense_exact_oracle(x, q, v, ridge=1e-2, power_k=1)
    torch.testing.assert_close(y, ref, atol=2e-4, rtol=2e-4)


@pytest.mark.skipif(not _cuda_ready(), reason="CUDA is required")
def test_fused_backward_matches_dense():
    torch.manual_seed(2026)
    device = torch.device("cuda")
    length = 37
    x = (0.05 * torch.randn(1, length, 1, 24, device=device)).requires_grad_()
    q = torch.randn_like(x, requires_grad=True)
    v = (0.05 * torch.randn(1, length, 1, 64, device=device)).requires_grad_()
    weight = torch.randn(1, length, 1, 64, device=device)

    y = fused_ska_prefix_scan(x, q, v, ridge=1e-2)
    grads = torch.autograd.grad((y * weight).sum(), (x, q, v), retain_graph=False)

    xr = x.detach().clone().requires_grad_()
    qr = q.detach().clone().requires_grad_()
    vr = v.detach().clone().requires_grad_()
    ref = dense_exact_oracle(xr, qr, vr, ridge=1e-2, power_k=1)
    ref_grads = torch.autograd.grad((ref * weight).sum(), (xr, qr, vr))
    for got, expected in zip(grads, ref_grads):
        torch.testing.assert_close(got, expected, atol=8e-4, rtol=8e-4)


@pytest.mark.skipif(not _cuda_ready(), reason="CUDA is required")
def test_fused_native_layout_multibatch_multihead_matches_dense():
    """Catches errors in the direct [B,T,H,W] CUDA address mapping."""
    torch.manual_seed(9102)
    device = torch.device("cuda")
    batch, length, heads = 2, 35, 3
    x = (0.05 * torch.randn(batch, length, heads, 24, device=device)).requires_grad_()
    q = torch.randn_like(x, requires_grad=True)
    v = (0.05 * torch.randn(batch, length, heads, 64, device=device)).requires_grad_()
    weight = torch.randn_like(v)

    y = fused_ska_prefix_scan(x, q, v, ridge=1e-2)
    got = torch.autograd.grad((y * weight).sum(), (x, q, v))

    xr = x.detach().clone().requires_grad_()
    qr = q.detach().clone().requires_grad_()
    vr = v.detach().clone().requires_grad_()
    ref = dense_exact_oracle(xr, qr, vr, ridge=1e-2, power_k=1)
    expected = torch.autograd.grad((ref * weight).sum(), (xr, qr, vr))

    torch.testing.assert_close(y, ref, atol=2e-4, rtol=2e-4)
    for actual, target in zip(got, expected):
        torch.testing.assert_close(actual, target, atol=8e-4, rtol=8e-4)
