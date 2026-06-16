"""Echo / SKA language model — unified package (50M–3B scaling effort).

Importing this package is CPU-safe: the CUDA-only Mamba2 backbone is imported
lazily inside model/recurrent/baselines, so `import koopman_lm` works without
mamba_ssm installed (you just can't instantiate the model until you add the
[cuda] extra).
"""
from koopman_lm.config import (
    KoopmanLMConfig,
    config_50m,
    config_180m,
    config_180m_gated,
    config_370m,
    config_440m,
    config_880m,
    config_1p5b,
    config_3b,
    config_hash,
    build_config,
    CONFIG_FACTORIES,
)
from koopman_lm.model import KoopmanLM
from koopman_lm.ska import SKAModule
from koopman_lm.koopman_mlp import SpectralKoopmanMLP, SpectralKoopmanMLPGated
from koopman_lm.recurrent import RecurrentKoopmanLM
from koopman_lm.adaptive_chunking import (
    compute_chunk_stats_overlap,
    compute_chunk_stats_decay,
)

__all__ = [
    "KoopmanLMConfig",
    "config_50m",
    "config_180m",
    "config_180m_gated",
    "config_370m",
    "config_440m",
    "config_880m",
    "config_1p5b",
    "config_3b",
    "config_hash",
    "build_config",
    "CONFIG_FACTORIES",
    "KoopmanLM",
    "SKAModule",
    "SpectralKoopmanMLP",
    "SpectralKoopmanMLPGated",
    "RecurrentKoopmanLM",
    "compute_chunk_stats_overlap",
    "compute_chunk_stats_decay",
]
