import torch
import torch.nn as nn
import torch.nn.functional as F
from koopman_lm.config import KoopmanLMConfig


def _apply_rope(x):
    """Rotary position embeddings applied to x of shape (B, H, T, D)."""
    B, H, T, D = x.shape
    half = D // 2
    device = x.device
    theta = 1.0 / (10000 ** (torch.arange(0, half, device=device).float() / half))
    pos   = torch.arange(T, device=device).float()
    ang   = pos.unsqueeze(1) * theta.unsqueeze(0)        # (T, D/2)
    cos   = ang.cos()[None, None]                         # (1, 1, T, D/2)
    sin   = ang.sin()[None, None]
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin,
                      x1 * sin + x2 * cos], dim=-1)


class CausalAttentionBlock(nn.Module):
    """Causal multi-head attention with optional RoPE.

    rope=True matches the SSM+Attn baseline in the paper (Section 4.1).
    rope=False is used when position information comes from context alone.
    """
    def __init__(self, cfg: KoopmanLMConfig, rope: bool = True):
        super().__init__()
        self.n_heads  = cfg.ska_n_heads
        self.head_dim = cfg.head_dim
        self.rope     = rope
        self.norm = nn.LayerNorm(cfg.d_model)
        self.qkv  = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x):
        B, T, d = x.shape
        H, D = self.n_heads, self.head_dim
        h   = self.norm(x)
        qkv = self.qkv(h).reshape(B, T, 3, H, D)
        q   = qkv[:, :, 0].transpose(1, 2)   # (B, H, T, D)
        k   = qkv[:, :, 1].transpose(1, 2)
        v   = qkv[:, :, 2].transpose(1, 2)
        if self.rope:
            q = _apply_rope(q)
            k = _apply_rope(k)
        out = F.scaled_dot_product_attention(q, k, v,
                                             attn_mask=None,
                                             dropout_p=0.0,
                                             is_causal=True)
        out = out.transpose(1, 2).reshape(B, T, d)
        return x + self.proj(out)


__all__ = ["CausalAttentionBlock"]
