import torch
import torch.nn as nn
import torch.nn.functional as F


def _radius_scale(gamma, omega, norm_preserving):
    """Scale factor applied to a (gamma, omega) eigenvalue pair.

    The 2x2 block [[g, o], [-o, g]] equals radius * R(theta) with
    radius = sqrt(g^2 + o^2), so its two singular values BOTH equal radius.

      norm_preserving=True  -> project onto the unit CIRCLE (scale = 1/radius):
          radius becomes exactly 1, so sigma_min = sigma_max = 1 -- the exact
          norm-preserving orthogonal rotation the paper (S3.3 "Gradient
          preservation") claims. Smooth in (gamma, omega) for radius > 0.
      norm_preserving=False -> clamp to the unit DISK (scale = min(radius,1)/radius):
          |lambda| <= 1 (non-expansive) but NOT norm-preserving once radius < 1.
          Legacy behavior; kept for checkpoint/behavior compatibility.
    """
    radius = torch.sqrt(gamma * gamma + omega * omega).clamp(min=1e-8)
    if norm_preserving:
        scale = 1.0 / radius
    else:
        scale = torch.clamp(radius, max=1.0) / radius
    return gamma * scale, omega * scale


class SpectralKoopmanMLP(nn.Module):
    def __init__(self, d, expand=2.667, spectral_norm_gamma=True,
                 norm_preserving=False):
        super().__init__()
        self.d_k = ((int(d * expand) + 63) // 64) * 64
        self.spectral_norm_gamma = spectral_norm_gamma
        self.norm_preserving = norm_preserving

        self.norm = nn.LayerNorm(d)
        self.lift = nn.Linear(d, self.d_k, bias=False)
        self.gamma = nn.Parameter(torch.ones(self.d_k // 2))
        self.omega = nn.Parameter(torch.empty(self.d_k // 2).normal_(0, 0.1))
        self.readout = nn.Linear(self.d_k, d, bias=False)

        nn.init.xavier_uniform_(self.lift.weight)
        nn.init.xavier_uniform_(self.readout.weight)

    def forward(self, x):
        h = self.norm(x)
        g_x = F.silu(self.lift(h))

        g_pair = g_x.view(*g_x.shape[:-1], self.d_k // 2, 2)
        g1 = g_pair[..., 0]
        g2 = g_pair[..., 1]

        gamma = self.gamma
        omega = self.omega
        if self.spectral_norm_gamma:
            gamma, omega = _radius_scale(gamma, omega, self.norm_preserving)

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
        self.gamma = nn.Parameter(torch.ones(self.d_k // 2))
        self.omega = nn.Parameter(torch.empty(self.d_k // 2).normal_(0, 0.1))
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

        gamma, omega = self.gamma, self.omega
        if self.spectral_norm_gamma:
            gamma, omega = _radius_scale(gamma, omega, self.norm_preserving)

        z1 = gamma * p1 + omega * p2
        z2 = -omega * p1 + gamma * p2
        z = torch.stack([z1, z2], dim=-1).reshape_as(main)

        return x + self.readout(z * g)
