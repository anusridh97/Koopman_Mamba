"""Koopman MLP rotation: exact norm-preserving (angle-only) vs unit-disk clamp.

The 2x2 block [[g, o], [-o, g]] = radius * R(theta) has BOTH singular values
equal to radius = sqrt(g^2 + o^2). So:

  * norm_preserving=True  -> ANGLE-ONLY param: g=cos(theta), o=sin(theta) -> the
    block is exactly R(theta): sigma_min = sigma_max = 1, and it is smooth
    EVERYWHERE (no radius to divide by, so no 1/radius gradient singularity at
    the origin). Paper S3.3 "Gradient preservation".
  * norm_preserving=False -> learned (g, o) clamped to the unit disk -> |lambda|
    <= 1 but contractive once radius < 1 (legacy behavior).

Locks in the audit fix "MLP disk-clamp is not norm-preserving" AND guards the
reviewer-flagged 1/radius origin singularity: the angle-only construction has
no such point, so gradients stay finite for any theta.
"""
import pytest
import torch

from koopman_lm.modules.channel_mixer.koopman import (
    SpectralKoopmanMLP, SpectralKoopmanMLPGated, _disk_clamp, _rotation_coeffs)

pytestmark = pytest.mark.correctness


def _block_singular_values(gamma, omega):
    """Singular values of every [[g,o],[-o,g]] block, shape (P, 2)."""
    row0 = torch.stack([gamma, omega], dim=-1)
    row1 = torch.stack([-omega, gamma], dim=-1)
    blocks = torch.stack([row0, row1], dim=-2)          # (P, 2, 2)
    return torch.linalg.svdvals(blocks)                 # (P, 2)


@pytest.mark.parametrize("theta", [0.0, 0.1, 1.0, -2.5, 100.0])
def test_angle_only_is_orthogonal(theta):
    m = SpectralKoopmanMLP(64, norm_preserving=True)
    with torch.no_grad():
        m.theta.fill_(float(theta))
    gamma, omega = _rotation_coeffs(m)
    sv = _block_singular_values(gamma, omega)
    assert torch.allclose(sv, torch.ones_like(sv), atol=1e-6), sv


def test_disk_clamp_is_contractive_when_radius_below_one():
    # radius = sqrt(0.6^2 + 0.2^2) = 0.632 < 1 -> disk clamp leaves it unchanged
    g = torch.full((8,), 0.6)
    o = torch.full((8,), 0.2)
    gp, op = _disk_clamp(g, o)
    sv = _block_singular_values(gp, op)
    assert torch.all(sv < 0.99)                          # strictly contractive


def test_norm_preserving_has_no_origin_singularity():
    """Reviewer catch: a `scale = 1/radius` construction blows up as (g,o)->0.
    The angle-only param has no radius, so forward+backward is finite even for
    extreme theta and near-zero SiLU activations."""
    torch.manual_seed(0)
    m = SpectralKoopmanMLP(64, norm_preserving=True)
    with torch.no_grad():
        m.theta.copy_(torch.linspace(-50.0, 50.0, m.theta.numel()))
    x = torch.randn(4, 7, 64, requires_grad=True)
    y = m(x)
    y.pow(2).mean().backward()
    for name, p in m.named_parameters():
        assert torch.isfinite(p.grad).all(), name
    assert torch.isfinite(x.grad).all()


@pytest.mark.parametrize("Klass", [SpectralKoopmanMLP, SpectralKoopmanMLPGated])
def test_rotation_preserves_lifted_norm(Klass):
    """In norm_preserving mode each 2x2 block is orthogonal, so the rotation
    preserves the L2 norm of the SiLU-lifted vector g -- for BOTH variants."""
    torch.manual_seed(0)
    d = 64
    mlp = Klass(d, norm_preserving=True).eval()
    with torch.no_grad():
        mlp.theta.copy_(torch.empty_like(mlp.theta).uniform_(-3.0, 3.0))
    x = torch.randn(2, 5, d)
    with torch.no_grad():
        h = mlp.norm(x)
        g = torch.nn.functional.silu(mlp.lift(h))
        gamma, omega = _rotation_coeffs(mlp)
        gp = g.view(*g.shape[:-1], mlp.d_k // 2, 2)
        z1 = gamma * gp[..., 0] + omega * gp[..., 1]
        z2 = -omega * gp[..., 0] + gamma * gp[..., 1]
        z = torch.stack([z1, z2], dim=-1).reshape_as(g)
    assert torch.allclose(z.norm(dim=-1), g.norm(dim=-1), atol=1e-5)
    assert torch.isfinite(mlp(x)).all()
