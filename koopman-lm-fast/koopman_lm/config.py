from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class KoopmanLMConfig:
    # 180M model, Nemotron-H layout ratios
    d_model: int = 768          # multiple of 64 (tensor core aligned)
    n_layers: int = 24
    vocab_size: int = 32000

    # Mamba-2 config
    d_state: int = 128          # multiple of 8
    d_conv: int = 4
    mamba_expand: int = 2

    # SKA config
    ska_n_heads: int = 12
    ska_rank: int = 48          # multiple of 16
    ska_ridge: float = 1e-3
    ska_scale: float = 1.5
    ska_power_K: int = 2
    ska_chunk_size: int = 64    # multiple of 8
    ska_backend: str = 'auto'

    # SKA adaptive chunking
    ska_chunk_strategy: str = 'standard'
    ska_overlap_fraction: float = 0.5
    ska_decay_alpha: float = 0.95

    # Koopman MLP config
    mlp_expand: float = 2.667
    mlp_spectral_norm: bool = True
    mlp_gated: bool = False

    # Layer layout
    ska_layer_indices: Optional[List[int]] = None

    # Training
    max_seq_len: int = 8192
    tie_embeddings: bool = True

    def __post_init__(self):
        if self.ska_layer_indices is None:
            self.ska_layer_indices = [4, 8, 12, 16, 20, 23]

        # Validate tensor-core alignment
        assert self.d_model % 64 == 0, \
            f"d_model={self.d_model} must be a multiple of 64 for tensor cores"
        assert self.d_state % 8 == 0, \
            f"d_state={self.d_state} must be a multiple of 8"
        assert self.ska_chunk_size % 8 == 0, \
            f"ska_chunk_size={self.ska_chunk_size} must be a multiple of 8"
        assert self.d_model % self.ska_n_heads == 0, \
            f"d_model={self.d_model} must be divisible by ska_n_heads={self.ska_n_heads}"

    @property
    def head_dim(self):
        return self.d_model // self.ska_n_heads

    def param_count_estimate(self):
        d = self.d_model
        V = self.vocab_size
        n = self.n_layers
        n_ska = len(self.ska_layer_indices)
        n_mamba = n - n_ska

        embed = V * d * (1 if self.tie_embeddings else 2)

        d_inner = d * self.mamba_expand
        per_mamba = (
            d * d_inner * 2 +
            d_inner * self.d_state * 2 +
            d_inner * self.d_conv +
            d_inner +
            d_inner * d
        )
        mamba_total = per_mamba * n_mamba

        per_ska = (
            d * self.ska_n_heads * self.ska_rank * 2 +
            d * self.ska_n_heads * self.head_dim +
            self.ska_n_heads * self.head_dim * d +
            2
        )
        ska_total = per_ska * n_ska

        d_k = ((int(d * self.mlp_expand) + 63) // 64) * 64
        n_mlp_proj = 3 if self.mlp_gated else 2
        per_mlp = d * d_k * n_mlp_proj + d_k
        mlp_total = per_mlp * n

        norms = n * d * 2 + d

        total = embed + mamba_total + ska_total + mlp_total + norms
        return total


# ============================================================================
# Layer index helpers
# ============================================================================

def _evenly_spaced_indices(n_layers, n_special):
    """
    Place n_special layers evenly across n_layers.
    Returns sorted list of indices.

    For 24 layers, 6 special -> [3, 7, 11, 15, 19, 23]  (every 4th, roughly)
    """
    if n_special == 0:
        return []
    if n_special >= n_layers:
        return list(range(n_layers))
    # Space them evenly, biased toward later layers (deeper = more retrieval)
    step = n_layers / n_special
    indices = [int(round((i + 1) * step)) - 1 for i in range(n_special)]
    # Clamp and deduplicate
    indices = sorted(set(min(idx, n_layers - 1) for idx in indices))
    return indices


# ============================================================================
# 180M configs — ~25% of sequence layers are SKA/attention (6 of 24)
# ============================================================================

def config_180m():
    """Koopman LM: 18 Mamba-2 + 6 SKA, 24 Koopman MLP."""
    return KoopmanLMConfig(
        d_model=768,
        n_layers=24,
        vocab_size=32000,
        d_state=128,
        ska_n_heads=12,
        ska_rank=48,
        ska_layer_indices=_evenly_spaced_indices(24, 6),
        mlp_gated=False,
    )


def config_180m_gated():
    c = config_180m()
    c.mlp_gated = True
    return c


def config_370m():
    """Koopman LM 370M: 21 Mamba-2 + 7 SKA, 28 Koopman MLP."""
    return KoopmanLMConfig(
        d_model=1024,
        n_layers=28,
        vocab_size=32000,
        d_state=128,
        ska_n_heads=16,
        ska_rank=64,
        ska_layer_indices=_evenly_spaced_indices(28, 7),
        mlp_gated=False,
    )
