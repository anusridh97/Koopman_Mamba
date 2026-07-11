"""Echo / SKA language model — unified package (50M–3B scaling effort).

Importing this package is CPU-safe: the CUDA-only Mamba2 backbone is imported
lazily inside models/recurrent, so `import koopman_lm` works without
mamba_ssm installed (you just can't instantiate the model until you add the
[cuda] extra).
"""
from koopman_lm.globals.config import (
    KoopmanLMConfig,
    config_hash,
    load_config,
    build_config,
    CONFIG_REGISTRY,
    CONFIG_FACTORIES,
)
from koopman_lm.models.koopman_lm import KoopmanLM
from koopman_lm.globals.modules.token_mixer import SKAModule
from koopman_lm.globals.modules.channel_mixer import SpectralKoopmanMLP, SpectralKoopmanMLPGated
from koopman_lm.models.recurrent import RecurrentKoopmanLM

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
]
