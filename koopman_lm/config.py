from dataclasses import dataclass, replace, asdict
from typing import Optional, Tuple
import hashlib
import json
from pathlib import Path

import yaml


@dataclass(frozen=True)
class KoopmanLMConfig:
    """Immutable model config.

    Frozen so a config can't be mutated after creation (configs are content-
    addressed via ``config_hash`` and stored in every checkpoint). To derive a
    variant, use ``dataclasses.replace(cfg, field=value)`` — NOT attribute
    assignment, which raises ``FrozenInstanceError``. Runtime adjustments
    (e.g. ``vocab_size`` from the tokenizer, ``max_seq_len`` from CLI args) must
    therefore go through ``replace`` at the call site.
    """
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

    # --- SKA scale-parameter policy ---
    ska_eta_learnable: bool = True
    ska_eta_value: float = 1.5
    # Smooth sigmoid-squash bounds (echo_jax.py parity), distinct from the
    # hard torch.clamp ska_gamma_clamp below: squash keeps a nonzero gradient
    # at the boundary, clamp kills it. None preserves prior (clamp/fixed)
    # behavior; set both to reproduce the original paper-faithful regime
    # (eta in [1.4, 1.7] init 1.5, gamma in [0.5, 1.5] init 0.7).
    ska_eta_bounds: Optional[tuple] = None
    ska_gamma_learnable: bool = True
    ska_gamma_clamp: Optional[tuple] = (1.0, 1.5)
    ska_gamma_bounds: Optional[tuple] = None
    ska_gamma_value: float = 1.0

    # Residual injection
    ska_layerscale: bool = False
    ska_layerscale_init: float = 1e-4
    ska_out_proj_std: float = 0.02

    # --- SKA short-range conv path ---
    ska_short_conv: bool = False
    ska_short_conv_kernel: int = 4
    ska_short_conv_gate_init: float = 1e-2

    # --- Exact intra-chunk causal stats ---
    ska_exact_intrachunk: bool = False

    # Koopman MLP config
    mlp_expand: float = 2.667
    mlp_spectral_norm: bool = True
    mlp_gated: bool = False

    # Layer layout
    ska_layer_indices: Optional[Tuple[int, ...]] = None

    # Training
    max_seq_len: int = 8192
    tie_embeddings: bool = True

    def __post_init__(self):
        if self.ska_layer_indices is None:
            object.__setattr__(self, "ska_layer_indices", (4, 8, 12, 16, 20, 23))
        else:
            object.__setattr__(self, "ska_layer_indices", tuple(self.ska_layer_indices))
        if self.ska_gamma_clamp is not None:
            object.__setattr__(self, "ska_gamma_clamp", tuple(self.ska_gamma_clamp))
        if self.ska_eta_bounds is not None:
            object.__setattr__(self, "ska_eta_bounds", tuple(self.ska_eta_bounds))
        if self.ska_gamma_bounds is not None:
            object.__setattr__(self, "ska_gamma_bounds", tuple(self.ska_gamma_bounds))

        # % 16 (not 64) allows the paper's d=96 while still ensuring reasonable alignment.
        # Production runs (d≥448) are always multiples of 64 via their YAML configs.
        assert self.d_model % 16 == 0, \
            f"d_model={self.d_model} must be a multiple of 16"
        assert self.d_state % 8 == 0, \
            f"d_state={self.d_state} must be a multiple of 8"
        assert self.ska_chunk_size % 8 == 0, \
            f"ska_chunk_size={self.ska_chunk_size} must be a multiple of 8"
        # % 8 (not 16) allows the paper's rank=24 (24 % 8 == 0).
        assert self.ska_rank % 8 == 0, \
            f"ska_rank={self.ska_rank} must be a multiple of 8"
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
            d * self.ska_n_heads + self.ska_n_heads +
            (self.d_model if self.ska_layerscale else 0) +
            (d * (self.ska_short_conv_kernel + 1) + d
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
# Config hashing
# ============================================================================

def config_hash(cfg: KoopmanLMConfig) -> str:
    """Stable SHA-256 hex digest of a config's contents."""
    payload = json.dumps(asdict(cfg), sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ============================================================================
# Layer index helpers
# ============================================================================

def _evenly_spaced_indices(n_layers, n_special):
    if n_special == 0:
        return []
    if n_special >= n_layers:
        return list(range(n_layers))
    step = n_layers / n_special
    indices = [int(round((i + 1) * step)) - 1 for i in range(n_special)]
    indices = sorted(set(min(idx, n_layers - 1) for idx in indices))
    return indices


# ============================================================================
# YAML / JSON loader
# ============================================================================

# koopman_lm/config.py -> repo root is two hops up (koopman_lm/ -> root).
_CONFIGS_ROOT = Path(__file__).parent.parent / "configs"

CONFIG_REGISTRY = {
    # Paper sub-million experiments (Table 2 / Table 3)
    "1m":         "1m.yaml",
    # Production scales
    "50m":        "50m.yaml",
    "180m":       "180m.yaml",
    "180m_gated": "180m_gated.yaml",
    "370m":       "370m.yaml",
    "440m":       "440m.yaml",
    "880m":       "880m.yaml",
    "1p5b":       "1p5b.yaml",
    "3b":         "3b.yaml",
}


def load_config(path) -> KoopmanLMConfig:
    """Load a KoopmanLMConfig from a YAML or JSON file.

    YAML lists are normalised to tuples by __post_init__, so ska_layer_indices
    and ska_gamma_clamp round-trip cleanly.
    """
    path = Path(path)
    if path.suffix in (".yaml", ".yml"):
        data = yaml.safe_load(path.read_text())
    elif path.suffix == ".json":
        data = json.loads(path.read_text())
    else:
        raise ValueError(f"Unsupported config format: {path.suffix}")
    # Strip YAML anchor helper keys not present in the dataclass
    data.pop("production", None)
    return KoopmanLMConfig(**data)


def build_config(model_size: str) -> KoopmanLMConfig:
    """Resolve a model size name or file path to a KoopmanLMConfig.

    Accepts:
      "440m"                    → looks up CONFIG_REGISTRY → loads YAML
      "path/to/custom.yaml"     → loads directly (one-off sweep configs)
    """
    p = Path(model_size)
    if p.suffix in (".yaml", ".yml", ".json"):
        return load_config(p)
    if model_size not in CONFIG_REGISTRY:
        raise ValueError(
            f"Unknown model_size: {model_size!r}. "
            f"Known: {sorted(CONFIG_REGISTRY)}"
        )
    return load_config(_CONFIGS_ROOT / CONFIG_REGISTRY[model_size])


# Backward-compat shim: callers that do `factory()` still work.
# Values are zero-arg callables (lambdas) returning KoopmanLMConfig, not strings.
CONFIG_FACTORIES = {name: (lambda n=name: build_config(n)) for name in CONFIG_REGISTRY}
