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

from koopman_lm.config import KoopmanLMConfig, build_config

_SYNTHETIC_GENERATORS = {"mqar", "toolcall", "sysprompt", "niah"}


@dataclass(frozen=True)
class ShardDataSpec:
    """A pretraining shard on disk. Verified against its meta.json at launch
    (§3.5) by experimentation.run.data_verify.verify_shard."""
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
    experimentation.training.train is not yet wired -- that needs TrainTask/
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


def resolve_model_config(value) -> KoopmanLMConfig:
    """Build a spec's `model:` section, however it was spelled.

    Accepts an inline dict (the materialized form every spec.yaml carries), a
    CONFIG_REGISTRY name like "50m", or a path to a YAML/JSON config.

    Lives here, beside data_spec_from_dict, because both answer the same
    question -- how does one RunSpec section become its dataclass -- and
    splitting the two across spec.py and resolve.py made the module boundary
    look arbitrary. spec.py owns "what a spec is and how to build one from a
    dict"; resolve.py owns inheritance, provenance, and writing it back out.
    """
    if isinstance(value, dict):
        return KoopmanLMConfig(**value)
    return build_config(value)   # registry name ("50m") or a path to a YAML/JSON


@dataclass(frozen=True)
class OptimSpec:
    lr: float
    warmup_steps: int
    max_steps: int
    schedule: str = "cosine"
    effective_batch: int = 512
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
        # The effective_batch / per_device_batch_size divisibility check now spans
        # two specs, so it lives on RunSpec -- the only place that sees both.


_BATCH_REQUIRED_ACCOUNT = "marlowe-m000151-pm06"
_BATCH_REQUIRED_QOS = "medium"
_KNOWN_PARTITIONS = {"batch", "hero", "preempt"}


@dataclass(frozen=True)
class RuntimeSpec:
    """Execution details (§3.3): NOT hashed into run_id/group_id. A different
    partition/account/worker count/ddp setting is the same experiment run
    differently."""
    seed: int = 42
    # Moved here from OptimSpec on 2026-08-19. A microbatch is how a run is FITTED
    # INTO MEMORY, not what it computes: halving it while doubling gradient
    # accumulation preserves effective_batch exactly and changes no result. While
    # it sat in OptimSpec -- hashed whole into run_id -- descending a rung of the
    # OOM ladder renamed the experiment. See
    # specs/2026-08-19-microbatch-identity-mapping.md.
    per_device_batch_size: int = 8
    # None means "follow model.compute_precision". DEPRECATED (design 6) -- see
    # __post_init__. The sentinel default is load-bearing rather than tidy: with a
    # concrete default, overriding model.compute_precision in a sweep would
    # disagree with an untouched runtime.precision and every such cell would fail
    # to construct -- which would contradict design 2's own reason for making
    # these flat fields ("sweepable for free"). None cannot disagree. It is the
    # same pattern mlp_precision uses in the same design.
    precision: Optional[str] = None
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
        # DEPRECATED, not deleted (precision design 6): model.compute_precision
        # supersedes this. It stays because RuntimeSpec(**d) raises on unknown
        # keys, and every materialized spec.yaml on scratch sets it -- which
        # load_materialized_spec reads when a run RESUMES. Deleting it would break
        # resume for runs already on disk. The domain matches
        # precision.COMPUTE_PRECISIONS so this field can still mirror the new one;
        # a disagreement between them is caught in RunSpec.__post_init__, which is
        # the only place that can see both.
        from koopman_lm.precision import COMPUTE_PRECISIONS

        if self.precision is not None and self.precision not in COMPUTE_PRECISIONS:
            raise ValueError(
                f"precision={self.precision!r}; expected None (follow "
                f"model.compute_precision) or one of {sorted(COMPUTE_PRECISIONS)}")
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
        # Cross-spec, so it can only live here: effective_batch is the scientific
        # quantity (OptimSpec) and the microbatch is a memory detail
        # (RuntimeSpec), but the split has to divide into whole steps or the run
        # trains on a different batch than it declares.
        if self.optim.effective_batch % self.runtime.per_device_batch_size != 0:
            raise ValueError(
                f"optim.effective_batch={self.optim.effective_batch} must be a "
                f"multiple of runtime.per_device_batch_size="
                f"{self.runtime.per_device_batch_size}")
        # Two fields describe the compute precision for one release (design 6).
        # Refuse a disagreement rather than preferring one: "which of these wins"
        # is not a question a reader should have to answer from source, and the
        # whole point of the policy is that a run's precision is legible from its
        # config. Only RunSpec can check this -- RuntimeSpec cannot see the model.
        if (self.runtime.precision is not None
                and self.runtime.precision != self.model.compute_precision):
            raise ValueError(
                f"runtime.precision={self.runtime.precision!r} disagrees with "
                f"model.compute_precision={self.model.compute_precision!r}. "
                f"runtime.precision is deprecated and will be removed once no "
                f"spec.yaml on scratch carries it; until then the two must "
                f"match. Set them the same, or drop runtime.precision from an "
                f"authored spec and let it default.")


def _json_stable(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, default=str)


def _scientific_payload(spec: RunSpec, include_seed: bool) -> Dict[str, Any]:
    payload = {
        "model": dataclasses.asdict(spec.model),
        "data": dataclasses.asdict(spec.data),
        "optim": dataclasses.asdict(spec.optim),
    }
    if include_seed:
        payload["seed"] = spec.runtime.seed
    return payload


def group_id(spec: RunSpec) -> str:
    """sha256(model + data + optim)[:8] -- the experiment (§3.3)."""
    blob = _json_stable(_scientific_payload(spec, include_seed=False))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:8]


def run_id(spec: RunSpec) -> str:
    """sha256(model + data + optim + seed)[:8] -- the datapoint (§3.3)."""
    blob = _json_stable(_scientific_payload(spec, include_seed=True))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:8]


def run_dir_name(spec: RunSpec) -> str:
    return f"{spec.name}.{group_id(spec)}"


def attempt_dir_name(spec: RunSpec) -> str:
    return f"seed{spec.runtime.seed}.{run_id(spec)}"


def run_dir_path(run_root, spec: RunSpec) -> Path:
    """$RUN_ROOT/<name>.<group_id>/seed<seed>.<run_id>/ (§3.3)."""
    return Path(run_root) / run_dir_name(spec) / attempt_dir_name(spec)
