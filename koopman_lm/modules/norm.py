"""Normalization factory.

One policy for every residual branch, so a LayerNorm-vs-RMSNorm ablation is one
config flag (`cfg.norm_type`) rather than four edits. Seven call sites depend on
that: seq/{mamba,ska_block,attention}.py, mlp/{swiglu,koopman}.py, and
models/{koopman_lm,baselines}.py for the final norm.

Both kinds are `torch.nn` builtins, so this file exists only to map the config
string onto a class. There is deliberately no `norm/` package and no custom
implementation here -- there would be nothing to put in them. If a norm ever
needs its own math (gated RMSNorm, QK-norm), it gets a module of its own and
this factory learns one more name.

Note `Mamba2Block` also has an *internal* `RMSNormGated(d_inner)` that
`mamba_ssm` owns; it is unrelated to this and is not switched by `norm_type`.
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
