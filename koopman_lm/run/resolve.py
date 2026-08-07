"""Two-stage run spec lifecycle (§3.2): author `configs/runs/<name>.yaml`
using `extends:` plus a short override block; at launch, flatten inheritance,
stamp provenance, and write a fully-materialized spec.yaml (no extends:, no
defaults, every value spelled out -- including the model config inlined) into
the run directory. Every downstream consumer reads only this file.
"""
from __future__ import annotations

import dataclasses
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict

import torch
import yaml

from koopman_lm.config import KoopmanLMConfig, build_config
from koopman_lm.run.artifacts import atomic_write_text
from koopman_lm.run.spec import (
    OptimSpec, RuntimeSpec, RunSpec, data_spec_from_dict, group_id, run_id,
)


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge `override` onto `base`: dicts merge key-wise,
    everything else (including lists) is replaced wholesale."""
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_raw_chain(path: Path) -> Dict[str, Any]:
    """Load `path`, following its `extends:` chain (base -> ... -> leaf), and
    return one flat dict with `extends` removed."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text()) or {}
    extends = raw.pop("extends", None)
    if extends is None:
        return raw
    base_path = (path.parent / extends).resolve()
    base = _load_raw_chain(base_path)
    return _deep_merge(base, raw)


def _resolve_model(value) -> KoopmanLMConfig:
    if isinstance(value, dict):
        return KoopmanLMConfig(**value)
    return build_config(value)   # registry name ("50m") or a path to a YAML/JSON


def resolve_run_spec(path) -> RunSpec:
    """Flatten a configs/runs/<name>.yaml (possibly with extends:) into a
    RunSpec. Callers should immediately materialize() the result -- nothing
    downstream should re-read the authoring YAML."""
    raw = _load_raw_chain(Path(path))
    model = _resolve_model(raw["model"])
    data = data_spec_from_dict(raw["data"])
    optim = OptimSpec(**raw["optim"])
    runtime = RuntimeSpec(**raw.get("runtime", {}))
    return RunSpec(name=raw["name"], model=model, data=data, optim=optim, runtime=runtime)


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True)
        return out.stdout.strip()
    except Exception:
        return "unknown"


def provenance() -> Dict[str, Any]:
    """Stamp git commit + torch/CUDA/python versions at materialization time
    (§3.2)."""
    return {
        "git_commit": git_commit(),
        # str(...): torch.__version__ is a TorchVersion (str subclass), and
        # PyYAML's SafeRepresenter keys off the exact type, not isinstance --
        # it can't represent the subclass without this cast (verified: without
        # it, yaml.safe_dump raises RepresenterError("cannot represent an
        # object", '2.13.0+cpu')).
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "python_version": sys.version.split()[0],
        "materialized_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def to_flat_dict(spec: RunSpec) -> Dict[str, Any]:
    """The self-contained representation: no extends, no defaults left
    implicit, every value spelled out -- including the model config inlined
    (never a registry name), so a later drift in configs/*.yaml cannot
    retroactively change a finished run's meaning."""
    return {
        "name": spec.name,
        "run_id": run_id(spec),
        "group_id": group_id(spec),
        "model": dataclasses.asdict(spec.model),
        "data": dataclasses.asdict(spec.data),
        "optim": dataclasses.asdict(spec.optim),
        "runtime": dataclasses.asdict(spec.runtime),
    }


def materialize(spec: RunSpec, run_dir) -> Path:
    """Write the fully-flattened spec + provenance to run_dir/spec.yaml,
    atomically. The only file downstream consumers (eval, resume, analysis)
    read."""
    run_dir = Path(run_dir)
    payload = to_flat_dict(spec)
    payload["provenance"] = provenance()
    out_path = run_dir / "spec.yaml"
    atomic_write_text(out_path, yaml.safe_dump(payload, sort_keys=False))
    return out_path


def load_materialized_spec(spec_yaml_path) -> RunSpec:
    """Read a materialized spec.yaml back into a RunSpec (used by eval/resume;
    never re-reads a base config)."""
    raw = yaml.safe_load(Path(spec_yaml_path).read_text())
    model = KoopmanLMConfig(**raw["model"])
    data = data_spec_from_dict(raw["data"])
    optim = OptimSpec(**raw["optim"])
    runtime = RuntimeSpec(**raw["runtime"])
    return RunSpec(name=raw["name"], model=model, data=data, optim=optim, runtime=runtime)
