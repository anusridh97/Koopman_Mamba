#!/usr/bin/env python3
"""Turn a curated design file into a launchable `cells:` sweep.

    python scripts/gen_anchor_sweep.py \\
        --designs configs/search/curated_15.yaml \\
        --base    configs/runs/50m-fineweb-3b.yaml \\
        --name    ska-anchors-15 \\
        --out     configs/sweeps/ska-anchors-15.yaml \\
        --minimum 15 --max-concurrent 8

    python -m experimentation.sweep configs/sweeps/ska-anchors-15.yaml \\
        --launcher slurm --dry_run

This is the whole payoff of writing the search space as data: a hand-reasoned
design set becomes an ordinary static sweep, so it launches through the
unmodified run system -- content-hashed run_ids, the dirty-tree gate,
verify_shard, one Slurm array -- with no sampler, and with optuna installed
nowhere.

Two things it writes.

`<out>` is the sweep spec, using `cells:` rather than `axes:` because a curated
set is a list of chosen points and not a rectangle. `sweep/spec.py` makes those
mutually exclusive by construction for exactly this reason.

`<out>.anchors.json` maps each design's name to the `run_id` its cell will get.
That file exists because `build_cell_run_spec` (spec.py:184) sets `RunSpec.name`
to the *sweep's* name for every cell -- a cell's name plays no part in identity,
so a design's name has nowhere to live in a `cells:` entry and would be lost.
"Which anchor produced this run?" is the first question a curated study has to
answer, so the mapping is computed by running the generated sweep through
`expand_cells` -- the same code path the launcher uses -- rather than by
re-deriving identities here, where a plausible-but-different answer would be
worse than none.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Sequence

# Running a script puts scripts/ on sys.path, not the repo root, and
# experimentation/ is deliberately not pip-installed (pyproject packages only
# koopman_lm*). scripts/gen_identity_baseline.py solves this by keeping its
# package imports lazy inside main(); this file imports at module level, so the
# bootstrap has to come first. Idempotent when the root is already importable.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import yaml

from experimentation.atomic_io import atomic_write_json, atomic_write_text
from experimentation.run.resolve import load_raw_spec
from experimentation.run.spec import group_id, resolve_model_config, run_id
from experimentation.sweep.search.anchors import designs_to_cells, load_designs
from experimentation.sweep.search.space import (
    BACKEND_POLICIES, default_base_lr, search_space)
from experimentation.sweep.spec import SweepSpec, expand_cells


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="scripts/gen_anchor_sweep.py",
                                description=__doc__.splitlines()[0])
    p.add_argument("--designs", required=True, type=Path,
                   help="a designs.yaml (see experimentation/sweep/search/anchors.py)")
    p.add_argument("--base", required=True, type=Path,
                   help="the base run spec every cell overrides")
    p.add_argument("--name", required=True, help="sweep name")
    p.add_argument("--out", required=True, type=Path,
                   help="where to write the sweep spec")
    p.add_argument("--minimum", type=int, default=1,
                   help="refuse to generate from fewer designs than this")
    p.add_argument("--max-concurrent", type=int, default=None,
                   help="the Slurm array's %%K cap")
    p.add_argument("--backend-policy", choices=BACKEND_POLICIES, default="exact_auto")
    p.add_argument("--seq-len", type=int, default=None,
                   help="override model.max_seq_len for every cell")
    return p.parse_args(argv)


def build_sweep_dict(args: argparse.Namespace) -> tuple[Dict[str, Any], list]:
    """The sweep spec, plus the expanded cells the launcher will build.

    Validation happens here, before anything is written, so a rejected
    generation leaves no half-made sweep behind.
    """
    designs = load_designs(args.designs, minimum=args.minimum)

    raw_base = load_raw_spec(args.base)
    base_model = resolve_model_config(raw_base["model"])
    max_steps = int(raw_base["optim"]["max_steps"])
    # A registry name ("50m") is a better learning-rate hint than a run name;
    # fall back to the spec's own name when the model is spelled inline.
    name_hint = raw_base["model"] if isinstance(raw_base["model"], str) else raw_base.get("name", "")

    space = search_space(base_model, base_name=str(name_hint))
    base_lr = default_base_lr(base_model, str(name_hint))
    cells = designs_to_cells(designs, base_model, space, base_lr=base_lr,
                             max_steps=max_steps,
                             backend_policy=args.backend_policy,
                             seq_len=args.seq_len)

    sweep: Dict[str, Any] = {"name": args.name, "base": str(args.base)}
    if args.max_concurrent is not None:
        sweep["max_concurrent"] = args.max_concurrent
    sweep["cells"] = cells

    # Identities from the same path the launcher takes.
    expanded = expand_cells(SweepSpec(name=args.name, base=str(args.base), cells=cells,
                                      max_concurrent=args.max_concurrent))
    return sweep, list(zip(designs, expanded))


def _header(args: argparse.Namespace, count: int) -> str:
    """A contiguous comment block: no blank lines, so it stays one block."""
    return "\n".join([
        f"# GENERATED by scripts/gen_anchor_sweep.py -- do not hand-edit.",
        f"# Regenerate instead; hand edits are lost on the next run.",
        f"#   designs: {args.designs}",
        f"#   base:    {args.base}",
        f"#   policy:  {args.backend_policy}   cells: {count}",
        f"# `cells:` not `axes:` -- a curated design set is a list of chosen",
        f"# points, not a rectangle. Design names live in the companion",
        f"# .anchors.json, because a cell's name plays no part in run identity.",
    ])


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    sweep, paired = build_sweep_dict(args)

    body = yaml.safe_dump(sweep, sort_keys=False, default_flow_style=False)
    atomic_write_text(args.out, _header(args, len(sweep["cells"])) + "\n\n" + body)

    mapping = {
        "sweep_name": args.name,
        "designs": str(args.designs),
        "base": str(args.base),
        "backend_policy": args.backend_policy,
        "anchors": [
            {"design": design.name,
             "run_id": run_id(cell.spec),
             "group_id": group_id(cell.spec),
             "overrides": cell.overrides}
            for design, cell in paired
        ],
    }
    atomic_write_json(args.out.with_suffix(".anchors.json"), mapping)

    print(f"[gen_anchor_sweep] wrote {args.out} ({len(sweep['cells'])} cells)")
    print(f"[gen_anchor_sweep] wrote {args.out.with_suffix('.anchors.json')}")
    for anchor in mapping["anchors"]:
        print(f"  {anchor['run_id']}  group={anchor['group_id']}  {anchor['design']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
