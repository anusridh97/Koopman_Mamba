"""
baselines.py -- Baseline model variants for ablation comparison.

Variant 1: Mamba-only + SwiGLU MLP (no global retrieval)
Variant 2: Mamba + causal attention (Flash Attention 2) + SwiGLU MLP
Variant 3: Mamba + SKA + SwiGLU MLP (isolates SKA contribution)
Variant 4: Mamba + SKA + Koopman MLP (full method, in model.py)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from koopman_lm.config import KoopmanLMConfig
from koopman_lm.ska import SKAModule


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


class CausalAttentionBlock(nn.Module):
    """
    Causal multi-head attention using F.scaled_dot_product_attention.
    Auto-dispatches to Flash Attention 2 on H100/B200.
    """
    def __init__(self, cfg: KoopmanLMConfig):
        super().__init__()
        self.n_heads = cfg.ska_n_heads
        self.head_dim = cfg.head_dim
        self.norm = nn.LayerNorm(cfg.d_model)
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x):
        B, T, d = x.shape
        H, D = self.n_heads, self.head_dim

        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, T, 3, H, D)
        q = qkv[:, :, 0].transpose(1, 2)  # [B, H, T, D]
        k = qkv[:, :, 1].transpose(1, 2)
        v = qkv[:, :, 2].transpose(1, 2)

        # Dispatches to Flash Attention 2 / memory-efficient attention
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=True,
        )

        out = out.transpose(1, 2).reshape(B, T, d)
        return x + self.proj(out)


class SKABlock(nn.Module):
    def __init__(self, cfg: KoopmanLMConfig):
        super().__init__()
        self.norm = nn.LayerNorm(cfg.d_model)
        self.ska = SKAModule(
            d_model=cfg.d_model,
            n_heads=cfg.ska_n_heads,
            rank=cfg.ska_rank,
            head_dim=cfg.head_dim,
            ridge_eps=cfg.ska_ridge,
            scale=cfg.ska_scale,
            power_K=cfg.ska_power_K,
            chunk_size=cfg.ska_chunk_size,
            backend=cfg.ska_backend,
        )

    def forward(self, x):
        return x + self.ska(self.norm(x))


class Mamba2Block(nn.Module):
    def __init__(self, cfg: KoopmanLMConfig):
        super().__init__()
        from mamba_ssm import Mamba2
        self.norm = nn.LayerNorm(cfg.d_model)
        self.mamba = Mamba2(
            d_model=cfg.d_model,
            d_state=cfg.d_state,
            d_conv=cfg.d_conv,
            expand=cfg.mamba_expand,
        )

    def forward(self, x):
        return x + self.mamba(self.norm(x))


def _build_model(cfg, seq_layer_fn, mlp_fn):
    """Generic model builder shared by all variants."""

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.cfg = cfg
            self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
            ska_set = set(cfg.ska_layer_indices)
            self.seq_layers = nn.ModuleList()
            self.mlp_layers = nn.ModuleList()
            for i in range(cfg.n_layers):
                self.seq_layers.append(seq_layer_fn(cfg, i, i in ska_set))
                self.mlp_layers.append(mlp_fn(cfg))
            self.norm_f = nn.LayerNorm(cfg.d_model)
            self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
            if cfg.tie_embeddings:
                self.lm_head.weight = self.embed.weight

        def forward(self, input_ids, labels=None):
            h = self.embed(input_ids)
            for seq, mlp in zip(self.seq_layers, self.mlp_layers):
                h = seq(h)
                h = mlp(h)
            h = self.norm_f(h)
            logits = self.lm_head(h)
            loss = None
            if labels is not None:
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100,
                )
            return {"loss": loss, "logits": logits}

        def param_summary(self):
            total = sum(p.numel() for p in self.parameters())
            print(f"Total parameters: {total:,}")
            return total

    return _Model()


def build_mamba_only(cfg: KoopmanLMConfig):
    """Variant 1: all Mamba-2, no global retrieval. SwiGLU MLPs."""
    def seq_fn(c, i, is_ska):
        return Mamba2Block(c)
    def mlp_fn(c):
        return SwiGLUMLP(c.d_model, c.mlp_expand)
    return _build_model(cfg, seq_fn, mlp_fn)


def build_mamba_attention(cfg: KoopmanLMConfig):
    """Variant 2: Mamba-2 + Flash Attention + SwiGLU MLP."""
    def seq_fn(c, i, is_ska):
        return CausalAttentionBlock(c) if is_ska else Mamba2Block(c)
    def mlp_fn(c):
        return SwiGLUMLP(c.d_model, c.mlp_expand)
    return _build_model(cfg, seq_fn, mlp_fn)


def build_mamba_ska_swiglu(cfg: KoopmanLMConfig):
    """Variant 3: Mamba-2 + SKA + SwiGLU MLP."""
    def seq_fn(c, i, is_ska):
        return SKABlock(c) if is_ska else Mamba2Block(c)
    def mlp_fn(c):
        return SwiGLUMLP(c.d_model, c.mlp_expand)
    return _build_model(cfg, seq_fn, mlp_fn)
