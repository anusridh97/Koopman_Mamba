"""SweepSpec: configs/sweeps/<name>.yaml (§4.2 of
docs/superpowers/specs/2026-08-07-run-system-design.md).

"The sweep grid is declared exactly once" is the requirement this module
exists to satisfy -- the anti-pattern it replaces is
scripts/slurm_array.sh hardcoding MODEL_TYPES/KV_PAIRS/GAPS in bash beneath a
comment reading "must match PAPER_MODEL_TYPES, ... in mqar_finetune.py". Two
copies of a grid kept in sync by a comment is a drift bug with a countdown on
it; here there is exactly one data structure (this module's `SweepSpec` +
`expand_cells`), and experimentation/run/slurm.py's array job only ever reads the
list `expand_cells` produced -- bash never sees axis values.

Two ways to declare the grid, and exactly one may be used per sweep:

  axes:   Dict[str, List[Any]] -- a cartesian product. The right shape for a
          rectangular hyperparameter grid (the common case, and what the
          design doc's own example uses).
  cells:  List[Dict[str, Any]] -- an explicit list of override dicts, for a
          sweep that is *not* a rectangle: a hand-curated Pareto-style list
          of points, or a paired/diagonal design where two axes must move
          together rather than cover every combination. `axes:` cannot
          express that without either enumerating the product and excluding
          most of it (fragile -- the excluded set grows faster than the
          kept one) or lying about independence. `cells:` is the honest
          escape hatch, and it is still one declared list, not two.

Declaring both -- or neither -- is a schema error: exactly one place must
declare the grid, so `axes:`/`cells:` are mutually exclusive by construction,
not by convention.

Every axis/cell key is "<section>.<field>", section one of {model, data,
optim, runtime} -- the four RunSpec sub-specs (§3.1). `exclude:` entries are
predicates over those same keys: a cell is dropped if it matches *every*
key/value pair in *any* one exclude entry (AND within an entry, OR across
entries) -- the same semantics Hydra/sweep tools use for exclusion lists.
"""
from __future__ import annotations

import dataclasses
import hashlib
import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from koopman_lm.config import KoopmanLMConfig
from experimentation.run.resolve import load_raw_spec, resolve_model_config
from experimentation.run.spec import OptimSpec, RunSpec, RuntimeSpec, data_spec_from_dict

_AXIS_SECTIONS = ("model", "data", "optim", "runtime")


@dataclass(frozen=True)
class SweepSpec:
    """A parsed configs/sweeps/<name>.yaml. Pure data -- no filesystem
    access happens until expand_cells() resolves `base`."""
    name: str
    base: str
    axes: Dict[str, List[Any]] = field(default_factory=dict)
    cells: Optional[List[Dict[str, Any]]] = None
    exclude: List[Dict[str, Any]] = field(default_factory=list)
    max_concurrent: Optional[int] = None

    def __post_init__(self):
        if not self.name:
            raise ValueError("SweepSpec.name is required")
        if not self.base:
            raise ValueError("SweepSpec.base is required")
        if self.axes and self.cells is not None:
            raise ValueError(
                "a sweep may declare 'axes:' or 'cells:', not both -- exactly "
                "one place must declare the grid (see experimentation.sweep.spec "
                "module docstring for when to use which).")
        if not self.axes and self.cells is None:
            raise ValueError(
                "a sweep must declare 'axes:' (a cartesian product) or "
                "'cells:' (an explicit list) -- nothing else can generate "
                "the grid.")
        if self.max_concurrent is not None and self.max_concurrent < 1:
            raise ValueError("max_concurrent (the Slurm array's %K cap) must be >= 1")


def load_sweep_spec(path) -> SweepSpec:
    """Read configs/sweeps/<name>.yaml into a SweepSpec. `base` is left as
    the literal string from the YAML (resolved later, relative to the
    process's cwd -- the same convention `python -m experimentation.run
    <spec.yaml>`'s own positional argument uses)."""
    raw = yaml.safe_load(Path(path).read_text()) or {}
    return SweepSpec(
        name=raw.get("name", ""),
        base=raw.get("base", ""),
        axes=raw.get("axes") or {},
        cells=raw.get("cells"),
        exclude=raw.get("exclude") or [],
        max_concurrent=raw.get("max_concurrent"),
    )


def sweep_id(sweep: SweepSpec) -> str:
    """A content hash of the sweep's own declaration (name, base, axes/
    cells, exclude) -- stamped into every cell's materialized spec.yaml
    (sweep_id + sweep_name) so `python -m experimentation.results` can group by
    it. Deliberately NOT part of run_id/group_id (§3.3): sweep membership is
    metadata about how a run was launched, not a scientific input, so
    changing an unrelated sweep-spec field (e.g. adding an axis value that
    doesn't affect this cell) must not perturb this cell's identity -- only
    its provenance."""
    payload = {
        "name": sweep.name, "base": sweep.base,
        "axes": sweep.axes, "cells": sweep.cells, "exclude": sweep.exclude,
    }
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:8]


def _parse_axis_key(key: str) -> Tuple[str, str]:
    if "." not in key:
        raise ValueError(
            f"sweep axis/cell key {key!r} must be '<section>.<field>' "
            f"(section one of {_AXIS_SECTIONS})")
    section, field_name = key.split(".", 1)
    if section not in _AXIS_SECTIONS:
        raise ValueError(
            f"sweep axis/cell key {key!r}: unknown section {section!r}, "
            f"expected one of {_AXIS_SECTIONS}")
    return section, field_name


def cartesian_cells(axes: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """The cartesian product over `axes`, as a list of one
    {'<section>.<field>': value, ...} override dict per cell."""
    keys = list(axes.keys())
    return [dict(zip(keys, combo))
            for combo in itertools.product(*(axes[k] for k in keys))]


def _excluded(cell: Dict[str, Any], exclude: List[Dict[str, Any]]) -> bool:
    return any(all(cell.get(k) == v for k, v in predicate.items())
               for predicate in exclude)


def raw_cells(sweep: SweepSpec) -> List[Dict[str, Any]]:
    """The list of override dicts this sweep declares, after exclude
    filtering -- the one and only materialization of the grid, before any
    RunSpec is built."""
    cells = list(sweep.cells) if sweep.cells is not None else cartesian_cells(sweep.axes)
    return [c for c in cells if not _excluded(c, sweep.exclude)]


def _base_sections(base_path) -> Dict[str, Dict[str, Any]]:
    """Load the base run spec's `extends:` chain and split it into per-
    section plain dicts ready for per-cell field overrides. `model` is
    always expanded to a full dict via resolve_model_config -- even when the
    base spec names a bare registry model (e.g. `model: 50m`) rather than an
    inline dict -- so `model.<field>` overrides apply uniformly regardless
    of how the base spelled its model."""
    raw = load_raw_spec(base_path)
    model_dict = dataclasses.asdict(resolve_model_config(raw["model"]))
    return {
        "model": model_dict,
        "data": dict(raw["data"]),
        "optim": dict(raw.get("optim", {})),
        "runtime": dict(raw.get("runtime", {})),
    }


def build_cell_run_spec(name: str, sections: Dict[str, Dict[str, Any]],
                         overrides: Dict[str, Any]) -> RunSpec:
    """Apply one cell's {'<section>.<field>': value} overrides onto `sections`
    (a base spec's per-section dicts, as returned by _base_sections) and
    build the resulting RunSpec. `name` becomes RunSpec.name for every cell
    -- it plays no part in run_id/group_id (experimentation.run.spec hashes only
    model+data+optim[+seed]), only in the run_dir path, so every cell of a
    sweep sorts together under `$RUN_ROOT/<sweep-name>.<group_id>/...`."""
    cell_sections = {section: dict(fields) for section, fields in sections.items()}
    for key, value in overrides.items():
        section, field_name = _parse_axis_key(key)
        cell_sections[section][field_name] = value
    return RunSpec(
        name=name,
        model=KoopmanLMConfig(**cell_sections["model"]),
        data=data_spec_from_dict(cell_sections["data"]),
        optim=OptimSpec(**cell_sections["optim"]),
        runtime=RuntimeSpec(**cell_sections["runtime"]),
    )


@dataclass(frozen=True)
class SweepCell:
    """One materialized point of a sweep: the RunSpec plus the raw override
    dict that produced it (surfaced for --dry_run's printout and for
    debugging which axis values a given run_id came from)."""
    spec: RunSpec
    overrides: Dict[str, Any]


def expand_cells(sweep: SweepSpec) -> List[SweepCell]:
    """The one and only place a sweep's grid becomes concrete RunSpecs. Each
    cell gets its run_id/group_id/run_dir_path from the exact same
    experimentation.run.spec machinery a lone `python -m experimentation.run` launch
    uses -- sweep membership changes nothing about how identity is computed
    (§3.3). In particular, sweeping runtime.seed yields distinct run_ids
    that all share one group_id, exactly as a hand-launched replicate set
    would."""
    sections = _base_sections(sweep.base)
    return [SweepCell(spec=build_cell_run_spec(sweep.name, sections, overrides),
                       overrides=overrides)
            for overrides in raw_cells(sweep)]
