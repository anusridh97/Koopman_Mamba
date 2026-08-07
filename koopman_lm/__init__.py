"""Echo / SKA language model — unified package (50M–3B scaling effort).

Importing this package is CPU-safe: the CUDA-only Mamba2 backbone is imported
lazily inside models/recurrent, so `import koopman_lm` works without
mamba_ssm installed (you just can't instantiate the model until you add the
[cuda] extra).
"""
from koopman_lm.config import (
    KoopmanLMConfig,
    config_hash,
    load_config,
    build_config,
    CONFIG_REGISTRY,
    CONFIG_FACTORIES,
)
from koopman_lm.models.koopman_lm import KoopmanLM
from koopman_lm.modules.seq.ska import SKAModule
from koopman_lm.modules.mlp.koopman import SpectralKoopmanMLP, SpectralKoopmanMLPGated
from koopman_lm.models.recurrent import RecurrentKoopmanLM
from koopman_lm.kernels.adaptive_chunking import (
    compute_chunk_stats_overlap,
    compute_chunk_stats_decay,
)

__all__ = [
    "KoopmanLMConfig",
    "config_hash",
    "load_config",
    "build_config",
    "CONFIG_REGISTRY",
    "CONFIG_FACTORIES",
    "KoopmanLM",
    "SKAModule",
    "SpectralKoopmanMLP",
    "SpectralKoopmanMLPGated",
    "RecurrentKoopmanLM",
    "compute_chunk_stats_overlap",
    "compute_chunk_stats_decay",
]
