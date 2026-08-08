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
from typing import Any, Dict, List, Optional

import torch
import yaml

from koopman_lm.config import KoopmanLMConfig, build_config
from experimentation.run.artifacts import atomic_write_text
from experimentation.run.spec import (
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


class DirtyTreeError(RuntimeError):
    """Raised when the working tree has uncommitted changes and
    --allow-dirty was not passed. Launching from an uncommitted tree makes
    code_id (the git commit, §4.2) meaningless -- the point of this guard is
    that launching uncommitted becomes a deliberate act, not an accident."""


def git_dirty_paths() -> List[str]:
    """`git status --porcelain` lines for the working tree, or [] if clean
    (or git is unavailable). --porcelain respects .gitignore, so run outputs
    and scratch never trip this."""
    try:
        out = subprocess.run(["git", "status", "--porcelain"],
                              capture_output=True, text=True, check=True)
    except Exception:
        return []
    return [line for line in out.stdout.splitlines() if line.strip()]


def check_git_clean(*, allow_dirty: bool) -> bool:
    """Refuse to launch from a dirty tree unless `allow_dirty` is set.

    Returns whether the tree was dirty, so the caller can stamp
    `dirty: true` into the materialized spec.yaml. Raises DirtyTreeError
    (listing the offending paths) when dirty and not allowed -- this must
    run before anything is materialized.
    """
    paths = git_dirty_paths()
    if not paths:
        return False
    if not allow_dirty:
        listing = "\n".join(f"  {p}" for p in paths)
        raise DirtyTreeError(
            f"refusing to launch from a dirty working tree "
            f"({len(paths)} uncommitted path(s)):\n{listing}\n"
            f"Commit or stash these changes, or pass --allow-dirty to "
            f"launch anyway (recorded as dirty: true in spec.yaml).")
    listing = "\n".join(f"  {p}" for p in paths)
    print(f"[experimentation.run] WARNING: launching with --allow-dirty -- "
          f"{len(paths)} uncommitted path(s), code_id will not reflect "
          f"them:\n{listing}")
    return True


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


def materialize(spec: RunSpec, run_dir, *, dirty: bool = False,
                 extra: Optional[Dict[str, Any]] = None) -> Path:
    """Write the fully-flattened spec + provenance to run_dir/spec.yaml,
    atomically. The only file downstream consumers (eval, resume, analysis)
    read.

    `code_id` (the git commit that produced these bytes) is stamped
    top-level, orthogonal to `run_id`: a code-only change must not perturb
    run_id (that would make every unrelated commit spawn a new run identity),
    but it must still be recoverable so two runs that collide on run_id
    (same declared science) but ran different code can be told apart (§3.7's
    create_run_dir collision check reads this back).

    `dirty=True` records that this launch proceeded with uncommitted local
    changes (--allow-dirty, the escape hatch from check_git_clean's default
    refusal) -- omitted entirely when the tree was clean.

    `extra`, if given, is merged into the top-level payload after
    provenance/code_id/dirty -- e.g. experimentation.sweep stamps `sweep_id` and
    `sweep_name` here so `python -m experimentation.results` can group cells by
    the sweep that produced them. This is metadata only: it plays no part in
    run_id/group_id, which are computed purely from `spec` (§3.3) before this
    function is ever called.
    """
    run_dir = Path(run_dir)
    payload = to_flat_dict(spec)
    payload["provenance"] = provenance()
    payload["code_id"] = payload["provenance"]["git_commit"]
    if dirty:
        payload["dirty"] = True
    if extra:
        payload.update(extra)
    out_path = run_dir / "spec.yaml"
    atomic_write_text(out_path, yaml.safe_dump(payload, sort_keys=False))
    return out_path


def _check_model_key_set(model_dict: Dict[str, Any]) -> None:
    """A spec.yaml written before a field was added to KoopmanLMConfig would
    otherwise silently receive the new field's default on load, making an old
    run look like it declared a value it never had. Compare the materialized
    spec's model key set against dataclasses.fields(KoopmanLMConfig) and raise
    with both lists on any mismatch.

    Scoped to *materialized* specs only -- authoring specs with `extends:`
    legitimately carry partial key sets (a leaf overrides only what it
    changes), so this must not run in resolve_run_spec's _resolve_model path.
    """
    expected = {f.name for f in dataclasses.fields(KoopmanLMConfig)}
    actual = set(model_dict)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing or unknown:
        raise ValueError(
            "spec.yaml does not match this code's config schema.\n"
            f"  missing: {missing}\n"
            f"  unknown: {unknown}")


def load_materialized_spec(spec_yaml_path) -> RunSpec:
    """Read a materialized spec.yaml back into a RunSpec (used by eval/resume;
    never re-reads a base config)."""
    raw = yaml.safe_load(Path(spec_yaml_path).read_text())
    _check_model_key_set(raw["model"])
    model = KoopmanLMConfig(**raw["model"])
    data = data_spec_from_dict(raw["data"])
    optim = OptimSpec(**raw["optim"])
    runtime = RuntimeSpec(**raw["runtime"])
    return RunSpec(name=raw["name"], model=model, data=data, optim=optim, runtime=runtime)


def load_raw_spec(path) -> Dict[str, Any]:
    """Public entry point to the `extends:` chain loader (`_load_raw_chain`),
    for callers that need the flattened raw dict *before* RunSpec
    construction -- namely experimentation.sweep, which must apply per-cell
    "<section>.<field>" overrides onto a base run spec's raw sections before
    building each cell's RunSpec. resolve_run_spec itself only ever returns
    the fully-built RunSpec, which is too late for that."""
    return _load_raw_chain(Path(path))


def resolve_model_config(value) -> KoopmanLMConfig:
    """Public wrapper around the same model-value resolution
    resolve_run_spec uses internally (a registry name, a path to a YAML/
    JSON config, or an inline dict) -- exposed so callers building RunSpecs
    outside the single-file `extends:` flow (experimentation.sweep) can resolve a
    base spec's `model:` field the same way, regardless of how that base
    spelled it."""
    return _resolve_model(value)
