"""Two-stage run spec lifecycle (§3.2): author `configs/runs/<name>.yaml`
using `extends:` plus a short override block; at launch, flatten inheritance,
stamp provenance, and write a fully-materialized spec.yaml (no extends:, no
defaults, every value spelled out -- including the model config inlined) into
the run directory. Every downstream consumer reads only this file.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from koopman_lm.config import KoopmanLMConfig
from experimentation.atomic_io import atomic_write_text
from experimentation.run.provenance import provenance
from experimentation.run.spec import (
    OptimSpec, RuntimeSpec, RunSpec, data_spec_from_dict, group_id,
    resolve_model_config, run_id,
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


_MICROBATCH = "per_device_batch_size"


def _migrate_microbatch(raw: Dict[str, Any], source: str) -> Dict[str, Any]:
    """Move a pre-2026-08-19 `optim.per_device_batch_size` into `runtime`.

    The field moved because it is a memory-fitting detail rather than a
    scientific input (see run/spec.py). Every authored spec in the repo and every
    already-materialized spec.yaml still puts it under `optim:`, and a run
    directory's spec.yaml is the only record of what that run did -- so a bare
    move would make finished runs unreadable.

    Migrated with a DeprecationWarning rather than silently: a silent move leaves
    the file looking correct while meaning something new, and the warning is what
    prompts the file to be updated. Follows the same "deprecated, not deleted"
    precedent the precision design sets for runtime.precision.
    """
    optim = raw.get("optim")
    if not isinstance(optim, dict) or _MICROBATCH not in optim:
        return raw
    runtime = raw.setdefault("runtime", {})
    if not isinstance(runtime, dict):
        raise ValueError(f"{source}: 'runtime' must be a mapping")
    legacy = optim.pop(_MICROBATCH)
    if _MICROBATCH in runtime:
        if runtime[_MICROBATCH] != legacy:
            raise ValueError(
                f"{source} declares {_MICROBATCH} in both 'optim' ({legacy}) and "
                f"'runtime' ({runtime[_MICROBATCH]}). Guessing which one wins "
                f"would be worse than refusing; delete the 'optim' entry.")
        return raw
    import warnings

    warnings.warn(
        f"{source}: optim.{_MICROBATCH} moved to runtime.{_MICROBATCH} on "
        f"2026-08-19 (a microbatch is a memory detail, not a scientific input, "
        f"so it must not be hashed into run_id). Migrated automatically; update "
        f"the file to silence this.",
        DeprecationWarning, stacklevel=3)
    runtime[_MICROBATCH] = legacy
    return raw


def resolve_run_spec(path) -> RunSpec:
    """Flatten a configs/runs/<name>.yaml (possibly with extends:) into a
    RunSpec. Callers should immediately materialize() the result -- nothing
    downstream should re-read the authoring YAML."""
    raw = _migrate_microbatch(_load_raw_chain(Path(path)), str(path))
    model = resolve_model_config(raw["model"])
    data = data_spec_from_dict(raw["data"])
    optim = OptimSpec(**raw["optim"])
    runtime = RuntimeSpec(**raw.get("runtime", {}))
    return RunSpec(name=raw["name"], model=model, data=data, optim=optim,
                   runtime=runtime, schedules=raw.get("schedules") or {})


def to_flat_dict(spec: RunSpec) -> Dict[str, Any]:
    """The self-contained representation: no extends, no defaults left
    implicit, every value spelled out -- including the model config inlined
    (never a registry name), so a later drift in configs/*.yaml cannot
    retroactively change a finished run's meaning."""
    payload = {
        "name": spec.name,
        "run_id": run_id(spec),
        "group_id": group_id(spec),
        "model": dataclasses.asdict(spec.model),
        "data": dataclasses.asdict(spec.data),
        "optim": dataclasses.asdict(spec.optim),
        "runtime": dataclasses.asdict(spec.runtime),
    }
    # §6.1: schedules is a top-level sibling of model/data/optim/runtime, in
    # the same materialized file but its own section. Written only when
    # non-empty, mirroring _scientific_payload's omit-when-absent rule so the
    # file matches what was hashed.
    if spec.schedules:
        payload["schedules"] = spec.schedules
    return payload


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
    changes), so this must not run in resolve_run_spec's resolve_model_config path.
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
    raw = _migrate_microbatch(
        yaml.safe_load(Path(spec_yaml_path).read_text()), str(spec_yaml_path))
    _check_model_key_set(raw["model"])
    model = KoopmanLMConfig(**raw["model"])
    data = data_spec_from_dict(raw["data"])
    optim = OptimSpec(**raw["optim"])
    runtime = RuntimeSpec(**raw["runtime"])
    return RunSpec(name=raw["name"], model=model, data=data, optim=optim,
                   runtime=runtime, schedules=raw.get("schedules") or {})


def load_raw_spec(path) -> Dict[str, Any]:
    """Public entry point to the `extends:` chain loader (`_load_raw_chain`),
    for callers that need the flattened raw dict *before* RunSpec
    construction -- namely experimentation.sweep, which must apply per-cell
    "<section>.<field>" overrides onto a base run spec's raw sections before
    building each cell's RunSpec. resolve_run_spec itself only ever returns
    the fully-built RunSpec, which is too late for that.

    Migrates a pre-2026-08-19 `optim.per_device_batch_size` like the other two
    loaders do. Missing it here was a real gap: this is the entry point
    `sweep/spec.py::_base_sections` uses, so a sweep over a legacy base spec would
    have failed at OptimSpec construction while a lone run of the same spec
    migrated cleanly."""
    return _migrate_microbatch(_load_raw_chain(Path(path)), str(path))
