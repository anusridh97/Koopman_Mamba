"""
config_50m_mamba_attn.py

50M-parameter Mamba-2 + standard causal attention baseline.

Architecture (matches Variant 2 from baselines.py, scaled down):
  - 12 layers total, 25% attention (3 attn + 9 Mamba-2)
  - SwiGLU MLP after every layer
  - Attention layers evenly spaced at indices [2, 5, 8]
  - Tied embeddings

Layout (0-indexed):
  M M [Attn] M M [Attn] M M [Attn] M M M
  Each layer followed by a SwiGLU MLP block.

Param budget (~49.3M):
  Embedding:  14.3M  (29%)
  Mamba-2:    12.9M  (26%)
  Attention:   2.4M  ( 5%)
  SwiGLU MLP: 19.6M  (40%)
"""

from koopman_lm.config import KoopmanLMConfig


def config_50m_mamba_attn():
    """50M Mamba-2 + 25% causal attention, SwiGLU MLPs."""
    return KoopmanLMConfig(
        d_model=448,
        n_layers=12,
        vocab_size=32000,

        # Mamba-2
        d_state=128,
        d_conv=4,
        mamba_expand=2,

        # Attention config (reuses ska_n_heads / head_dim for CausalAttentionBlock)
        ska_n_heads=7,          # 7 heads × 64 head_dim = 448
        ska_rank=48,            # unused by attention, kept for config compat
        ska_layer_indices=[2, 5, 8],   # 3/12 = 25% attention

        # SwiGLU MLP
        mlp_expand=2.667,
        mlp_gated=False,        # ignored -- baselines.py uses SwiGLU directly

        # Training
        max_seq_len=2048,
        tie_embeddings=True,
    )
