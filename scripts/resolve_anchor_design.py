#!/usr/bin/env python
"""Print a study's anchor designs as RESOLVED parameters. Read-only, no optuna.

    python scripts/resolve_anchor_design.py configs/search/<study>.yaml
    python scripts/resolve_anchor_design.py configs/search/<study>.yaml --format=md

Why this exists as a script and not only as a test. A design file is written in
RELATIVE terms (`ridge_factor: 0.3`) and resolved against both the base config
and the study's declared space, and resolution both scales AND snaps: a
categorical goes to its nearest declared choice, a float clamps into the
declared bounds, and a fixed axis collapses onto its singleton. Every one of
those is silent. `example_anchors.yaml`'s own header records the case that made
this worth a script -- `ridge_factor: 4.0` asking for 0.04 and resolving to 0.03,
`lr_factor: 0.5` asking for 2.0e-4 and resolving to 2.6e-4 -- where a design
became a smaller move than its author intended and nothing said so.

So: one command that shows what each named design will actually run at, plus the
`run_id` it will land under, so an anchor can be traced to a run directory
without opening a journal. `test_proxy_anchor_design.py` writes the same table as
a committed artifact; this is the interactive form.

Deliberately imports no optuna: resolution happens entirely below the optuna
line (`space.py`, `anchors.py`, `geometry.py`), so this works on a laptop.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experimentation.run.spec import group_id, resolve_model_config, run_id
from experimentation.sweep.search.anchors import load_designs, resolve_design
from experimentation.sweep.search.space import (
    dropped_base_values, params_to_overrides, restrict_space, search_space)
from experimentation.sweep.search.studyspec import load_study_spec

#: The order columns appear in. Fixed rather than sorted so the table reads as
#: architecture-then-optimizer, which is how the study is reasoned about.
PARAM_ORDER = (
    "ska_rank", "n_ska_layers", "placement", "ska_power_K", "ska_ridge",
    "ska_layerscale_init", "norm_clip_multiplier", "gamma_value",
    "learning_rate", "weight_decay", "warmup_ratio", "grad_clip",
)


def resolve_all(study_path):
    """(study_spec, base_sections, base_model, space, [(design, params, ...)])."""
    from experimentation.sweep.spec import _base_sections

    study_spec = load_study_spec(study_path)
    if not study_spec.design_file:
        raise SystemExit(f"{study_path} declares no design_file")

    base_sections = _base_sections(str(REPO / study_spec.base)
                                   if not Path(study_spec.base).is_absolute()
                                   and not Path(study_spec.base).exists()
                                   else study_spec.base)
    base_model = resolve_model_config(base_sections["model"])
    space = restrict_space(search_space(base_model, base_name=study_spec.base),
                          base_model, axes=study_spec.search_axes,
                          fixed=study_spec.fixed_params)
    base_lr = base_sections["optim"].get("lr", 4e-4)

    design_path = study_spec.design_file
    if not Path(design_path).exists():
        design_path = REPO / design_path
    designs = load_designs(design_path, minimum=1)

    rows = []
    for design in designs:
        params = resolve_design(design, base_model, space, base_lr=base_lr)
        overrides = params_to_overrides(
            params, base_model, max_steps=study_spec.max_steps,
            seq_len=study_spec.seq_len, backend_policy=study_spec.backend_policy)
        rows.append((design, params, overrides))
    return study_spec, base_sections, base_model, space, base_lr, rows


def _identity(study_spec, base_sections, overrides, seed=None):
    """The run_id an anchor will land under, or None if it cannot be computed.

    Wrapped because building a full RunSpec validates the whole thing -- which is
    a feature (a design that cannot become a launchable spec should say so here,
    not on a GPU) but must not stop the table from printing.

    `seed` mirrors what `driver.run_trial` does with a design's designated
    `runtime.seed`, and it MUST be applied here: two members of a reference group
    resolve to byte-identical params, and their whole purpose is to be distinct
    runs. Omitting it would print one run_id for five designs and the table would
    look like a collision instead of a replicate set.
    """
    from experimentation.sweep.spec import build_cell_run_spec

    extra = {"optim.max_steps": int(study_spec.max_steps)}
    if seed is not None:
        extra["runtime.seed"] = int(seed)
    try:
        spec = build_cell_run_spec(study_spec.name, base_sections,
                                   {**overrides, **extra},
                                   schedules=base_sections.get("schedules"))
    except Exception as exc:                              # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}
    return {"run_id": run_id(spec), "group_id": group_id(spec),
            "seed": int(spec.runtime.seed),
            "param_count": int(spec.model.param_count_estimate()),
            "ska_layer_indices": list(spec.model.ska_layer_indices),
            "ska_norm_clip_c": spec.model.ska_norm_clip_c}


def _designated_seed(design):
    """A design's explicit `runtime.seed`, or None to inherit the base spec's."""
    seed = getattr(design, "seed", "baseline")
    return None if seed == "baseline" else int(seed)


def _fmt(value):
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def render_markdown(study_spec, base_sections, base_model, space, base_lr, rows):
    out = []
    out.append(f"# Resolved anchor design: {study_spec.name}")
    out.append("")
    out.append(f"- base spec: `{study_spec.base}`")
    out.append(f"- design file: `{study_spec.design_file}`")
    out.append(f"- anchors: **{len(rows)}**")
    out.append(f"- study max_steps: {study_spec.max_steps}")
    out.append(f"- base parameter count: "
               f"{int(base_model.param_count_estimate()):,}")
    out.append("")
    out.append("Generated by `scripts/resolve_anchor_design.py`. These are the "
               "values each named design ACTUALLY runs at, after scaling by its "
               "factors and snapping/clamping onto the study's declared space.")
    out.append("")

    fixed = sorted(study_spec.fixed_params)
    if fixed:
        out.append(f"Fixed axes (singleton categoricals): "
                   + ", ".join(f"`{k}={study_spec.fixed_params[k]}`"
                               for k in fixed))
        out.append("")

    lost = dropped_base_values(space, base_model, base_lr=base_lr,
                               base_optim=base_sections.get("optim"))
    if lost:
        out.append("**Axes on which this space cannot express the base config:**")
        for axis, why in sorted(lost.items()):
            out.append(f"- `{axis}`: {why}")
    else:
        out.append("Every axis can still express the base config's own value, "
                   "so the reference point is inside the space.")
    out.append("")

    groups = {}
    for design, _params, _overrides in rows:
        group = getattr(design, "reference_group", None)
        if group is not None:
            groups.setdefault(str(group), []).append(design.name)
    if groups:
        out.append("## Replicate sets (the study's noise floor)")
        out.append("")
        out.append("Members of a `reference_group` are identical in every "
                   "scientific factor and differ only in `runtime.seed`, so the "
                   "SPREAD of their held-out losses is the smallest effect this "
                   "study can resolve. `anchors._check_reference_groups` enforces "
                   "the \"identical in everything else\" half at load time; "
                   "`anchors.check_replicates_resolve` enforces distinct resolved "
                   "seeds at `--dry_run`.")
        out.append("")
        for group, names in sorted(groups.items()):
            out.append(f"- `{group}`: {len(names)} evaluation(s) -- "
                       + ", ".join(f"`{n}`" for n in sorted(names)))
        out.append("")

    header = ["name", "group", "seed"] + list(PARAM_ORDER) + [
        "indices", "clip_c", "params", "run_id"]
    out.append("| " + " | ".join(header) + " |")
    out.append("|" + "|".join(["---"] * len(header)) + "|")
    for design, params, overrides in rows:
        identity = _identity(study_spec, base_sections, overrides,
                             seed=_designated_seed(design))
        cells = [f"`{design.name}`",
                 f"`{design.reference_group}`"
                 if getattr(design, "reference_group", None) else "-",
                 str(identity.get("seed", "-"))]
        cells += [_fmt(params[k]) for k in PARAM_ORDER]
        cells += [
            str(identity.get("ska_layer_indices", "-")).replace(" ", ""),
            _fmt(identity.get("ska_norm_clip_c", "-")),
            f"{identity.get('param_count', 0):,}" if "param_count" in identity
            else identity.get("error", "-"),
            f"`{identity.get('run_id', '-')}`",
        ]
        out.append("| " + " | ".join(cells) + " |")
    out.append("")

    # Snapping, made explicit rather than left to be noticed.
    notes = []
    for design, params, _ in rows:
        if design.warmup_ratio != params["warmup_ratio"]:
            notes.append(f"`{design.name}`: warmup_ratio "
                         f"{design.warmup_ratio} -> {params['warmup_ratio']}")
    if notes:
        out.append("## Values that SNAPPED during resolution")
        out.append("")
        out.append("A design asked for one value and the study's space resolved "
                   "it to another. Not an error -- the space is what bounds a "
                   "trial -- but a design cannot reach past it.")
        out.append("")
        for note in sorted(set(notes)):
            out.append(f"- {note}")
        out.append("")
    return "\n".join(out) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("study", help="path to configs/search/<study>.yaml")
    parser.add_argument("--format", choices=("md", "json"), default="md")
    args = parser.parse_args(argv)

    study_spec, base_sections, base_model, space, base_lr, rows = resolve_all(
        args.study)

    if args.format == "json":
        payload = {
            "study": study_spec.name,
            "base": study_spec.base,
            "base_param_count": int(base_model.param_count_estimate()),
            "anchors": [
                {"name": d.name,
                 "reference_group": getattr(d, "reference_group", None),
                 "params": p,
                 **_identity(study_spec, base_sections, o,
                             seed=_designated_seed(d))}
                for d, p, o in rows
            ],
        }
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    else:
        print(render_markdown(study_spec, base_sections, base_model, space,
                              base_lr, rows), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
