"""Koopman MLP eigenvalue-scaling: unit-circle (norm-preserving) vs unit-disk.

The 2x2 block [[g, o], [-o, g]] = radius * R(theta) has BOTH singular values
equal to radius = sqrt(g^2 + o^2). So:

  * norm_preserving=True  -> radius forced to 1 -> sigma_min = sigma_max = 1
    (exact norm-preserving rotation; paper S3.3 "Gradient preservation").
  * norm_preserving=False -> radius clamped to <=1 -> contractive once radius<1
    (unit-disk / legacy behavior).

This test locks in the fix for the audit item "MLP disk-clamp is not
norm-preserving": with the circle projection the whole SiLU-lifted vector's L2
norm is preserved through the rotation, and each block is exactly orthogonal.
"""
import pytest
import torch

from koopman_lm.globals.modules.koopman_mlp import (
    SpectralKoopmanMLP, SpectralKoopmanMLPGated, _radius_scale)

pytestmark = pytest.mark.correctness


def _block_singular_values(gamma, omega):
    """Singular values of every [[g,o],[-o,g]] block, shape (P, 2)."""
    row0 = torch.stack([gamma, omega], dim=-1)
    row1 = torch.stack([-omega, gamma], dim=-1)
    blocks = torch.stack([row0, row1], dim=-2)          # (P, 2, 2)
    return torch.linalg.svdvals(blocks)                 # (P, 2)


@pytest.mark.parametrize("gamma0,omega0", [(1.0, 0.1), (0.6, 0.2), (0.3, 0.05), (2.0, 1.0)])
def test_circle_projection_is_orthogonal(gamma0, omega0):
    g = torch.full((8,), float(gamma0))
    o = torch.full((8,), float(omega0))
    gp, op = _radius_scale(g, o, norm_preserving=True)
    sv = _block_singular_values(gp, op)
    # every singular value is exactly 1 -> exact norm-preserving rotation
    assert torch.allclose(sv, torch.ones_like(sv), atol=1e-6), sv


def test_disk_clamp_is_contractive_when_radius_below_one():
    # radius = sqrt(0.6^2 + 0.2^2) = 0.632 < 1 -> disk clamp leaves it unchanged
    g = torch.full((8,), 0.6)
    o = torch.full((8,), 0.2)
    gp, op = _radius_scale(g, o, norm_preserving=False)
    sv = _block_singular_values(gp, op)
    assert torch.all(sv < 0.99)                          # strictly contractive
    # ...whereas the circle projection makes the SAME pair norm-preserving
    gpc, opc = _radius_scale(g, o, norm_preserving=True)
    svc = _block_singular_values(gpc, opc)
    assert torch.allclose(svc, torch.ones_like(svc), atol=1e-6)


@pytest.mark.parametrize("Klass", [SpectralKoopmanMLP, SpectralKoopmanMLPGated])
def test_rotation_preserves_lifted_norm(Klass):
    """In norm_preserving mode the rotation preserves the L2 norm of the
    SiLU-lifted vector g (each 2x2 block is orthogonal), for BOTH variants."""
    torch.manual_seed(0)
    d = 64
    mlp = Klass(d, norm_preserving=True).eval()
    # push the learned pair off the unit circle so the projection has to work
    with torch.no_grad():
        mlp.gamma.copy_(torch.empty_like(mlp.gamma).uniform_(0.2, 2.0))
        mlp.omega.copy_(torch.empty_like(mlp.omega).uniform_(-1.0, 1.0))
    x = torch.randn(2, 5, d)
    with torch.no_grad():
        h = mlp.norm(x)
        g = torch.nn.functional.silu(mlp.lift(h))
        gamma, omega = _radius_scale(mlp.gamma, mlp.omega, True)
        gp = g.view(*g.shape[:-1], mlp.d_k // 2, 2)
        z1 = gamma * gp[..., 0] + omega * gp[..., 1]
        z2 = -omega * gp[..., 0] + gamma * gp[..., 1]
        z = torch.stack([z1, z2], dim=-1).reshape_as(g)
    # orthogonal per-pair rotation -> ||z|| == ||g|| exactly
    assert torch.allclose(z.norm(dim=-1), g.norm(dim=-1), atol=1e-5)
    # sanity: forward runs and is finite
    assert torch.isfinite(mlp(x)).all()
