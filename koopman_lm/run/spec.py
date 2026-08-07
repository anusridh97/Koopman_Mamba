"""RunSpec: the unit of configuration and of run identity (§3.1, §3.3).

Four frozen dataclasses: model (KoopmanLMConfig, unchanged), data (polymorphic
DataSpec), optim (OptimSpec), runtime (RuntimeSpec). data.kind is the only
place the pretraining and synthetic-experiment paths differ (§6).
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Union

from koopman_lm.config import KoopmanLMConfig

_SYNTHETIC_GENERATORS = {"mqar", "toolcall", "sysprompt", "niah"}


@dataclass(frozen=True)
class ShardDataSpec:
    """A pretraining shard on disk. Verified against its meta.json at launch
    (§3.5) by koopman_lm.run.data_verify.verify_shard."""
    kind: str = "shard"
    shard_dir: str = ""
    tokenizer: str = ""
    mix: Dict[str, float] = field(default_factory=dict)
    n_tokens: int = 0

    def __post_init__(self):
        if self.kind != "shard":
            raise ValueError(f"ShardDataSpec.kind must be 'shard', got {self.kind!r}")
        if not self.shard_dir:
            raise ValueError("ShardDataSpec.shard_dir is required")
        if not self.tokenizer:
            raise ValueError("ShardDataSpec.tokenizer is required")
        if self.n_tokens <= 0:
            raise ValueError("ShardDataSpec.n_tokens must be positive")


@dataclass(frozen=True)
class SyntheticDataSpec:
    """A curricula.py generator (§6 synthetic-experiment path). Building a
    RunSpec around one of these is legal today; actually launching it through
    koopman_lm.training.train is not yet wired -- that needs TrainTask/
    SyntheticTask (§6.2), which is a separate, later plan."""
    kind: str = "synthetic"
    generator: str = ""
    params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.kind != "synthetic":
            raise ValueError(f"SyntheticDataSpec.kind must be 'synthetic', got {self.kind!r}")
        if self.generator not in _SYNTHETIC_GENERATORS:
            raise ValueError(
                f"generator={self.generator!r}; expected one of "
                f"{sorted(_SYNTHETIC_GENERATORS)}")


DataSpec = Union[ShardDataSpec, SyntheticDataSpec]


def data_spec_from_dict(d: Dict[str, Any]) -> DataSpec:
    d = dict(d)
    kind = d.get("kind")
    if kind == "shard":
        return ShardDataSpec(**d)
    if kind == "synthetic":
        return SyntheticDataSpec(**d)
    raise ValueError(f"data.kind must be 'shard' or 'synthetic', got {kind!r}")
