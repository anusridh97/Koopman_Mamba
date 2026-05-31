"""
koopman_mlp.py -- Spectral Koopman MLP (SwiGLU replacement).

Ported into echo-ska-440m to fill a missing-module gap: model.py imports
SpectralKoopmanMLP / SpectralKoopmanMLPGated from here, but the file was absent
from this checkout, making koopman_lm.model un-importable (so neither training
nor the diagnostics could load). Implementation matches koopman-lm-fast and the
constructor signature model.py uses: (d, expand, spectral_norm_gamma).

The layer applies one step of a spectrally-constrained block-diagonal dynamical
system: lift -> SiLU -> learnable complex eigenvalue rotation per 2D pair with
modulus clamped to the unit disk (|lambda| <= 1) -> readout. Two matrices
instead of SwiGLU's three (a 1/3 parameter reduction). The gated variant adds a
sigmoid gate to reach SwiGLU parameter parity while retaining the spectral core.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralKoopmanMLP(nn.Module):
    def __init__(self, d, expand=2.667, spectral_norm_gamma=True):
        super().__init__()
        self.d_k = ((int(d * expand) + 63) // 64) * 64
        self.spectral_norm_gamma = spectral_norm_gamma

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
            radius = torch.sqrt(gamma * gamma + omega * omega).clamp(min=1e-8)
            scale = torch.clamp(radius, max=1.0) / radius
            gamma = gamma * scale
            omega = omega * scale

        z1 = gamma * g1 + omega * g2
        z2 = -omega * g1 + gamma * g2

        z = torch.stack([z1, z2], dim=-1).reshape_as(g_x)
        return x + self.readout(z)


class SpectralKoopmanMLPGated(nn.Module):
    def __init__(self, d, expand=2.667, spectral_norm_gamma=True):
        super().__init__()
        self.d_k = ((int(d * expand) + 63) // 64) * 64
        self.spectral_norm_gamma = spectral_norm_gamma

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
            radius = torch.sqrt(gamma * gamma + omega * omega).clamp(min=1e-8)
            scale = torch.clamp(radius, max=1.0) / radius
            gamma = gamma * scale
            omega = omega * scale

        z1 = gamma * p1 + omega * p2
        z2 = -omega * p1 + gamma * p2
        z = torch.stack([z1, z2], dim=-1).reshape_as(main)

        return x + self.readout(z * g)
