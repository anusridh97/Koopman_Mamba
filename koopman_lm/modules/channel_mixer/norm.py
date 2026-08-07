"""Shared normalization factory.

Keeping every residual branch on the same normalization policy is important for
clean architecture ablations.  The original repository hard-coded LayerNorm in
Mamba, SKA, the MLP, and the final norm independently, which made it difficult
to test a Llama/Mamba-style RMSNorm model without changing four implementations.
"""
from __future__ import annotations

import torch.nn as nn


def make_norm(d_model: int, norm_type: str = "layernorm", eps: float = 1e-5) -> nn.Module:
    kind = str(norm_type).lower().replace("_", "")
    if kind in {"layernorm", "ln"}:
        return nn.LayerNorm(d_model, eps=eps)
    if kind in {"rmsnorm", "rms"}:
        if not hasattr(nn, "RMSNorm"):
            raise RuntimeError("RMSNorm requires a PyTorch build with torch.nn.RMSNorm")
        return nn.RMSNorm(d_model, eps=eps)
    raise ValueError(f"unknown norm_type={norm_type!r}; expected layernorm or rmsnorm")


__all__ = ["make_norm"]
