"""SwiGLU feed-forward MLP -- the standard (non-Koopman) channel mixer.

Used by the Table 2 ablation baselines (mamba_only, mamba_attn,
mamba_ska_swiglu, transformer) as the MLP-slot occupant, so the Koopman MLP's
contribution can be isolated by swapping only this block.
"""
import torch.nn as nn
import torch.nn.functional as F


class SwiGLUMLP(nn.Module):
    def __init__(self, d, expand=2.667):
        super().__init__()
        d_ff = ((int(d * expand) + 63) // 64) * 64  # tensor-core aligned
        self.norm = nn.LayerNorm(d)
        self.w1 = nn.Linear(d, d_ff, bias=False)
        self.w2 = nn.Linear(d, d_ff, bias=False)
        self.w3 = nn.Linear(d_ff, d, bias=False)

    def forward(self, x):
        h = self.norm(x)
        return x + self.w3(F.silu(self.w1(h)) * self.w2(h))


__all__ = ["SwiGLUMLP"]
