"""
recurrent_state.py -- plain state containers for RecurrentKoopmanLM's SKA
decode paths.

Extracted out of models/recurrent.py (structural-review issue 3): these are
__slots__-based data containers with no nn.Module ties, and
RecurrentKoopmanLM is their sole consumer. Kept in models/ (not kernels/)
because, unlike the numerics they carry, they are model-state bookkeeping
specific to one nn.Module -- see
docs/superpowers/specs/2026-08-07-structural-review.md, issue 3.
"""

import math
import torch


class SKAState:
    """Fixed-size recurrent state for one SKA layer (sqrt-beta symmetric)."""
    __slots__ = ['G', 'M', 'C_v', 'x_last', 'L']

    def __init__(self, B, H, r, P, device, dtype=torch.float32, ridge_eps=1e-3):
        eye = torch.eye(r, device=device, dtype=dtype)
        self.G = ridge_eps * eye.reshape(1, 1, r, r).expand(B, H, r, r).clone()
        self.L = math.sqrt(ridge_eps) * eye.reshape(1, 1, r, r) \
                     .expand(B, H, r, r).clone()      # carried Cholesky of G
        self.M = torch.zeros(B, H, r, r, device=device, dtype=dtype)
        self.C_v = torch.zeros(B, H, P, r, device=device, dtype=dtype)
        # (B,H,r) previous SYMMETRIC key x = sqrt(beta)*z (for the boundary M
        # cross-term sqrt(beta_t beta_{t-1})). NOT the raw key -- carrying raw z
        # here is the train/decode divergence bug.
        self.x_last = None


class PrefixSKAState:
    """Compact exact state used by the prefix-scan recurrence.

    Invariants per batch/head are

        L L^T = G,
        A = L^{-1} M L^{-T},
        R = C L^{-T},
        h_prev = L^{-1} x_last.

    The state contains two r-by-r matrices instead of the legacy three raw
    matrices plus a factor, and every decode write is quadratic.
    """
    __slots__ = ['L', 'A', 'R', 'h_prev', 'has_prev']

    def __init__(self, B, H, r, P, device, dtype=torch.float32, ridge_eps=1e-3):
        eye = torch.eye(r, device=device, dtype=dtype)
        self.L = (math.sqrt(ridge_eps) * eye).reshape(1, 1, r, r) \
            .expand(B, H, r, r).clone()
        self.A = torch.zeros(B, H, r, r, device=device, dtype=dtype)
        self.R = torch.zeros(B, H, P, r, device=device, dtype=dtype)
        self.h_prev = torch.zeros(B, H, r, device=device, dtype=dtype)
        self.has_prev = torch.zeros(B, H, device=device, dtype=torch.bool)
