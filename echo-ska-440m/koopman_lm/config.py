from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class KoopmanLMConfig:
    # 180M model, Nemotron-H layout ratios
    d_model: int = 768          # multiple of 64 (tensor core aligned)
    n_layers: int = 24
    vocab_size: int = 32000     # Llama-2 / Mistral tokenizers are both 32000

    # Mamba-2 config
    d_state: int = 128          # multiple of 8
    d_conv: int = 4
    mamba_expand: int = 2

    # SKA config
    ska_n_heads: int = 12
    ska_rank: int = 48          # multiple of 16
    ska_ridge: float = 1e-3
    ska_scale: float = 1.5      # legacy eta-init; IGNORED when ska_eta_learnable=False
    ska_power_K: int = 2
    ska_chunk_size: int = 64    # multiple of 8
    ska_backend: str = 'auto'

    # --- SKA scale-parameter policy ---
    # DEFAULTS reproduce the PUBLISHED baseline (180m/370m) exactly:
    #   eta learnable (init=ska_scale), gamma learnable + clamped [1.0,1.5],
    #   out_proj zero-initialized, no LayerScale.
    # config_440m() OVERRIDES these to the new policy (eta=gamma=1 fixed,
    # LayerScale residual). This keeps published checkpoints bit-reproducible
    # and isolates the rewrite to 440M.
    ska_eta_learnable: bool = True
    ska_eta_value: float = 1.5          # used only when eta_learnable=False
    ska_gamma_learnable: bool = True
    ska_gamma_clamp: tuple = (1.0, 1.5) # baseline clamp; None => no clamp
    ska_gamma_value: float = 1.0        # used only when gamma_learnable=False

    # Residual injection. Baseline = exact-zero out_proj (layerscale off,
    # out_proj_std ignored -> zeros). 440M sets layerscale=True.
    ska_layerscale: bool = False
    ska_layerscale_init: float = 1e-4
    ska_out_proj_std: float = 0.02      # used only when layerscale=True

    # --- SKA short-range conv path (new) ---
    # A depthwise causal conv runs in PARALLEL with SKA inside the SKA block and
    # is summed into the residual. Rationale: chunked SKA stats discard the
    # within-chunk lag-1..lag-(S-1) cross-covariance terms, so the chunked
    # operator is poor at SHORT-RANGE recall (measured: ~100% rel error vs a
    # per-token reference at every chunk size > 1). That poisons SKA's early
    # gradient and the model learns to route around it (lazy-branch death). The
    # conv gives short-range its own chunking-immune path so SKA only has to
    # handle the long-range regime where chunking is exact. Mirrors the
    # conv-before-mixer pattern Mamba-2 / GLA already use. Default ON for 440M.
    ska_short_conv: bool = False
    ska_short_conv_kernel: int = 4      # causal depthwise kernel width (Mamba uses 4)
    ska_short_conv_gate_init: float = 1e-2  # per-channel gate so the conv path
    # starts SMALL (not full-strength). The delta-kernel init means an ungated
    # conv would inject ~LayerNorm(x) at full residual strength, confounding the
    # ablation (you'd be testing "extra residual path helps", not "local
    # evidence restored"). Gate init 1e-2 (> layerscale 1e-4) because the local
    # path is meant to learn EARLY.

    # --- Exact intra-chunk causal stats (the REAL fix; conv is a bypass) ---
    # When True, SKA uses true per-token exclusive-prefix stats (across + WITHIN
    # chunk) instead of exclusive-chunk-prefix. Eliminates the within-chunk
    # staleness entirely (verified: 0.3% vs per-token ref, vs ~100% chunked),
    # reusing the verified ska_core (no new backward). COST: T core-solves
    # instead of T/S -> slower training. This is "path B" -- correct but
    # expensive; the optimized rank-1-scan/Woodbury kernel ("path C") is the
    # speed version and is NOT yet built (needs its own backward derivation).
    # Default OFF; flip on for the ablation vs the conv path.
    ska_exact_intrachunk: bool = False

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
        assert self.ska_rank % 16 == 0, \
            f"ska_rank={self.ska_rank} must be a multiple of 16"
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
            d * self.ska_n_heads * self.ska_rank * 2 +       # key+query proj
            d * self.ska_n_heads * self.head_dim +            # value proj
            self.ska_n_heads * self.head_dim * d +            # out proj
            d * self.ska_n_heads + self.ska_n_heads +         # beta_proj (W + bias)
            (self.d_model if self.ska_layerscale else 0) +    # LayerScale diag
            (d * (self.ska_short_conv_kernel + 1) + d        # depthwise conv (w+bias) + gate
             if self.ska_short_conv else 0) +
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


def config_440m():
    """Koopman LM 440M: 26 Mamba-2 + 8 SKA, 34 Koopman MLP.

    Scaled from 370M by depth (L 28->34) + SKA count (7->8) + rank (64->96).
    Depth adds the inter-layer re-encoding capacity that multi-hop actually
    uses; the 8 SKA layers bound the re-encoded content-hop reach; r=96 gives
    the Koopman operator more spectral modes for signal/junk separation.
    ~444M params (tied embeddings). eta=gamma=1 fixed; LayerScale residual.
    """
    return KoopmanLMConfig(
        d_model=1024,
        n_layers=34,
        vocab_size=32000,
        d_state=128,
        ska_n_heads=16,
        ska_rank=96,
        ska_chunk_size=96,
        ska_layer_indices=_evenly_spaced_indices(34, 8),
        mlp_gated=False,
        # --- NEW policy (440M only; baselines keep dataclass defaults) ---
        ska_eta_learnable=False, ska_eta_value=1.0,
        ska_gamma_learnable=False, ska_gamma_value=1.0, ska_gamma_clamp=None,
        ska_layerscale=True, ska_layerscale_init=1e-4, ska_out_proj_std=0.02,
        ska_short_conv=True, ska_short_conv_kernel=4,
    )


def config_50m():
    """Koopman LM 50M: 12 Mamba-2 + 4 SKA, 16 Koopman MLP (for niah_quick etc.)."""
    return KoopmanLMConfig(
        d_model=448,
        n_layers=16,
        vocab_size=32000,
        d_state=128,
        ska_n_heads=7,
        ska_rank=48,
        ska_layer_indices=_evenly_spaced_indices(16, 4),
        mlp_gated=False,
    )
