# Redirect stub — the authoritative config system is koopman_lm.globals.config.
# This file exists only to prevent ImportError in any legacy code that still
# does `from koopman_lm.config import ...`. New code should import from
# koopman_lm.globals.config directly.
from koopman_lm.globals.config import (  # noqa: F401
    KoopmanLMConfig,
    config_hash,
    load_config,
    build_config,
    CONFIG_REGISTRY,
    CONFIG_FACTORIES,
    _evenly_spaced_indices,
)
