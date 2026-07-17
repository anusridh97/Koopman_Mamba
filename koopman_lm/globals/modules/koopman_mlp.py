import torch
import torch.nn as nn
import torch.nn.functional as F


def _disk_clamp(gamma, omega):
    """Legacy unit-DISK clamp of a (gamma, omega) eigenvalue pair.

    The 2x2 block [[g, o], [-o, g]] = radius * R(theta), radius = sqrt(g^2+o^2),
    so both singular values equal radius. Scaling by min(radius,1)/radius caps
    radius at 1 (|lambda| <= 1, non-expansive) but leaves radius < 1 untouched,
    so it is NOT norm-preserving once radius < 1. Kept for the legacy
    (norm_preserving=False) path and existing checkpoints.
    """
    radius = torch.sqrt(gamma * gamma + omega * omega).clamp(min=1e-8)
    scale = torch.clamp(radius, max=1.0) / radius
    return gamma * scale, omega * scale


def _rotation_coeffs(m):
    """(gamma, omega) coefficients of the block [[g, o], [-o, g]] for module m.

    norm_preserving=True  -> ANGLE-ONLY parameterization: gamma = cos(theta),
        omega = sin(theta). The block is then exactly R(theta): sigma_min =
        sigma_max = 1 for every pair, and it is smooth EVERYWHERE -- there is no
        radius to divide by, so no 1/radius gradient singularity at the origin
        (the failure mode a `scale = 1/radius` construction has when the learned
        pair drifts toward (0,0)). This is the exact norm-preserving rotation of
        the paper's S3.3 "Gradient preservation". Retrain-only, so dropping the
        (gamma, omega) params here costs no checkpoint migration.
    norm_preserving=False -> legacy learned (gamma, omega), unit-disk clamped.
    """
    if m.norm_preserving:
        return torch.cos(m.theta), torch.sin(m.theta)
    gamma, omega = m.gamma, m.omega
    if m.spectral_norm_gamma:
        gamma, omega = _disk_clamp(gamma, omega)
    return gamma, omega


def _init_rotation_params(m):
    """Register the rotation parameters for module m per its parameterization."""
    if m.norm_preserving:
        # theta ~ N(0, 0.1): cos(theta) ~ 1, sin(theta) ~ small -> matches the
        # legacy (gamma=1, omega~N(0,0.1)) rotation direction, radius == 1 exactly.
        m.theta = nn.Parameter(torch.empty(m.d_k // 2).normal_(0, 0.1))
    else:
        m.gamma = nn.Parameter(torch.ones(m.d_k // 2))
        m.omega = nn.Parameter(torch.empty(m.d_k // 2).normal_(0, 0.1))


class SpectralKoopmanMLP(nn.Module):
    def __init__(self, d, expand=2.667, spectral_norm_gamma=True,
                 norm_preserving=False):
        super().__init__()
        self.d_k = ((int(d * expand) + 63) // 64) * 64
        self.spectral_norm_gamma = spectral_norm_gamma
        self.norm_preserving = norm_preserving

        self.norm = nn.LayerNorm(d)
        self.lift = nn.Linear(d, self.d_k, bias=False)
        _init_rotation_params(self)
        self.readout = nn.Linear(self.d_k, d, bias=False)

        nn.init.xavier_uniform_(self.lift.weight)
        nn.init.xavier_uniform_(self.readout.weight)

    def forward(self, x):
        h = self.norm(x)
        g_x = F.silu(self.lift(h))

        g_pair = g_x.view(*g_x.shape[:-1], self.d_k // 2, 2)
        g1 = g_pair[..., 0]
        g2 = g_pair[..., 1]

        gamma, omega = _rotation_coeffs(self)

        z1 = gamma * g1 + omega * g2
        z2 = -omega * g1 + gamma * g2

        z = torch.stack([z1, z2], dim=-1).reshape_as(g_x)
        return x + self.readout(z)


class SpectralKoopmanMLPGated(nn.Module):
    def __init__(self, d, expand=2.667, spectral_norm_gamma=True,
                 norm_preserving=False):
        super().__init__()
        self.d_k = ((int(d * expand) + 63) // 64) * 64
        self.spectral_norm_gamma = spectral_norm_gamma
        self.norm_preserving = norm_preserving

        self.norm = nn.LayerNorm(d)
        self.lift = nn.Linear(d, self.d_k, bias=False)
        self.gate = nn.Linear(d, self.d_k, bias=False)
        _init_rotation_params(self)
        self.readout = nn.Linear(self.d_k, d, bias=False)

        nn.init.xavier_uniform_(self.lift.weight)
        nn.init.xavier_uniform_(self.gate.weight)
        nn.init.xavier_uniform_(self.readout.weight)

    def forward(self, x):
        h = self.norm(x)
        main = F.silu(self.lift(h))
        g = torch.sigmoid(self.gate(h))

        pair = main.view(*main.shape[:-1], self.d_k // 2, 2)
        p1, p2 = pair[..., 0], pair[..., 1]

        gamma, omega = _rotation_coeffs(self)

        z1 = gamma * p1 + omega * p2
        z2 = -omega * p1 + gamma * p2
        z = torch.stack([z1, z2], dim=-1).reshape_as(main)

        return x + self.readout(z * g)
