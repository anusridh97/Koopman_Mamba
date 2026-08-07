"""Conventional gated feed-forward blocks used by quality baselines."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from koopman_lm.modules.norm import make_norm


class SwiGLUMLP(nn.Module):
    """Pre-norm SwiGLU residual block.

    This is deliberately small and ordinary: it is the control needed to tell
    whether a sequence-memory idea helps without simultaneously changing the
    feed-forward function class.  ``d_ff`` is aligned to 64 for tensor cores.
    """

    def __init__(
        self,
        d: int,
        expand: float = 2.667,
        *,
        norm_type: str = "layernorm",
        norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.d_ff = ((int(d * expand) + 63) // 64) * 64
        self.norm = make_norm(d, norm_type, norm_eps)
        self.w1 = nn.Linear(d, self.d_ff, bias=False)
        self.w2 = nn.Linear(d, self.d_ff, bias=False)
        self.w3 = nn.Linear(self.d_ff, d, bias=False)

    @property
    def residual_output_weight(self) -> torch.nn.Parameter:
        return self.w3.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        return x + self.w3(F.silu(self.w1(h)) * self.w2(h))


__all__ = ["SwiGLUMLP"]
