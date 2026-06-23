import torch.nn as nn
import torch.nn.functional as F
from koopman_lm.globals.config import KoopmanLMConfig


class CausalAttentionBlock(nn.Module):
    """Causal multi-head attention using F.scaled_dot_product_attention.
    Auto-dispatches to Flash Attention 2 on H100/B200."""
    def __init__(self, cfg: KoopmanLMConfig):
        super().__init__()
        self.n_heads = cfg.ska_n_heads
        self.head_dim = cfg.head_dim
        self.norm = nn.LayerNorm(cfg.d_model)
        self.qkv  = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x):
        B, T, d = x.shape
        H, D = self.n_heads, self.head_dim
        h   = self.norm(x)
        qkv = self.qkv(h).reshape(B, T, 3, H, D)
        q   = qkv[:, :, 0].transpose(1, 2)
        k   = qkv[:, :, 1].transpose(1, 2)
        v   = qkv[:, :, 2].transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v,
                                             attn_mask=None,
                                             dropout_p=0.0,
                                             is_causal=True)
        out = out.transpose(1, 2).reshape(B, T, d)
        return x + self.proj(out)


__all__ = ["CausalAttentionBlock"]
