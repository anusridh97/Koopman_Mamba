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


@dataclass(frozen=True)
class OptimSpec:
    lr: float
    warmup_steps: int
    max_steps: int
    schedule: str = "cosine"
    effective_batch: int = 512
    per_device_batch_size: int = 8
    weight_decay: float = 0.1
    grad_clip: float = 1.0

    def __post_init__(self):
        if not isinstance(self.lr, (int, float)) or isinstance(self.lr, bool):
            raise TypeError(
                f"OptimSpec.lr must be numeric, got {type(self.lr).__name__} "
                f"({self.lr!r}) -- check your YAML: PyYAML parses '4e-4' as a "
                f"string (no decimal point in the mantissa); use '4.0e-4' or "
                f"'0.0004'.")
        if self.schedule != "cosine":
            raise ValueError(f"schedule={self.schedule!r}; only 'cosine' is implemented")
        if self.warmup_steps < 0 or self.max_steps <= 0:
            raise ValueError("warmup_steps must be >= 0 and max_steps > 0")
        if self.effective_batch % self.per_device_batch_size != 0:
            raise ValueError(
                f"effective_batch={self.effective_batch} must be a multiple of "
                f"per_device_batch_size={self.per_device_batch_size}")


_BATCH_REQUIRED_ACCOUNT = "marlowe-m000151-pm06"
_BATCH_REQUIRED_QOS = "medium"
_KNOWN_PARTITIONS = {"batch", "hero", "preempt"}


@dataclass(frozen=True)
class RuntimeSpec:
    """Execution details (§3.3): NOT hashed into run_id/group_id. A different
    partition/account/worker count/ddp setting is the same experiment run
    differently."""
    seed: int = 42
    precision: str = "bf16"
    ddp: bool = False
    gpus: int = 1
    nodes: int = 1
    partition: str = "batch"
    account: str = "marlowe-m000151-pm06"
    qos: str = "medium"
    workers: int = 4
    time_limit: str = "2-00:00:00"
    # H100 80GB, compute capability 9.0 (sm_90). Marlowe has NO B200/sm_100
    # nodes -- scripts/train_50m.sh's TORCH_CUDA_ARCH_LIST=10.0 is a bug.
    gpu_arch: str = "9.0"

    def __post_init__(self):
        if self.precision not in {"bf16", "fp32"}:
            raise ValueError(f"precision={self.precision!r}; expected 'bf16' or 'fp32'")
        if self.partition not in _KNOWN_PARTITIONS:
            raise ValueError(
                f"partition={self.partition!r}; expected one of {sorted(_KNOWN_PARTITIONS)}")
        if self.gpus < 1 or self.nodes < 1:
            raise ValueError("gpus and nodes must be >= 1")
        if self.partition == "batch":
            # Tribal knowledge, now enforced: accountless jobs are rejected,
            # and 'batch' requires QoS medium, which lives on
            # marlowe-m000151-pm06 -- not the default marlowe-m000151
            # (QoS normal).
            if self.account != _BATCH_REQUIRED_ACCOUNT:
                raise ValueError(
                    f"partition='batch' requires account={_BATCH_REQUIRED_ACCOUNT!r} "
                    f"(qos={_BATCH_REQUIRED_QOS!r}); got account={self.account!r}. "
                    f"The default account 'marlowe-m000151' (qos=normal) cannot "
                    f"submit to 'batch'.")
            if self.qos != _BATCH_REQUIRED_QOS:
                raise ValueError(
                    f"partition='batch' with account={_BATCH_REQUIRED_ACCOUNT!r} "
                    f"requires qos={_BATCH_REQUIRED_QOS!r}, got qos={self.qos!r}")


@dataclass(frozen=True)
class RunSpec:
    """The unit of configuration and of identity (§3.1)."""
    name: str
    model: KoopmanLMConfig
    data: DataSpec
    optim: OptimSpec
    runtime: RuntimeSpec

    def __post_init__(self):
        if not self.name:
            raise ValueError("RunSpec.name is required")
        if not isinstance(self.model, KoopmanLMConfig):
            raise TypeError(f"RunSpec.model must be a KoopmanLMConfig, got {type(self.model)}")
        if not isinstance(self.data, (ShardDataSpec, SyntheticDataSpec)):
            raise TypeError(
                f"RunSpec.data must be a ShardDataSpec or SyntheticDataSpec, "
                f"got {type(self.data)}")
        if not isinstance(self.optim, OptimSpec):
            raise TypeError(f"RunSpec.optim must be an OptimSpec, got {type(self.optim)}")
        if not isinstance(self.runtime, RuntimeSpec):
            raise TypeError(f"RunSpec.runtime must be a RuntimeSpec, got {type(self.runtime)}")
