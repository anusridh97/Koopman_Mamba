#!/usr/bin/env python
"""Prepare the exponent arm's run root, and refuse to prepare a bad one.

Run on the LOGIN NODE before submitting the array. Two jobs:

  1. Run every pre-flight gate over the whole arm and exit non-zero on any
     failure. Doing it here rather than inside the array means one gate failure
     costs seconds instead of N queued tasks that each discover it separately --
     which on a GPU-saturated cluster is the difference between a typo and a day.
  2. Write the per-cell flat model YAMLs and the manifest that each array task
     will read to find out which cell it is.

Also prints the `--array` width, so the sbatch cannot be submitted with a width
that disagrees with the cell list. `task_for_index` raises on an out-of-range
index precisely so that disagreement fails loudly, but printing the right number
here means it does not have to.

Usage:
    python scripts/prepare_beta_exponent_arm.py --run_root <dir> [--dry_run]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from experimentation.experiments.beta_exponent_cells import (  # noqa: E402
    CELLS, EVAL_EVERY, GAP, KV, SEEDS, STEPS, TASK_VOCAB, manifest,
    preflight_failures, resolve_cell, route_of, task_count, write_model_yaml)
from koopman_lm.config import config_hash  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--run_root", type=Path, required=True)
    p.add_argument("--dry_run", action="store_true",
                   help="run the gates and print the plan, write nothing")
    p.add_argument("--proxy", type=Path,
                   default=REPO / "configs/runs/proxy-256x17.yaml")
    args = p.parse_args(argv)

    from experimentation.run.resolve import resolve_run_spec
    base = resolve_run_spec(args.proxy).model

    print(f"proxy:    {args.proxy}")
    print(f"run_root: {args.run_root}")
    print(f"budget:   {STEPS} steps, eval every {EVAL_EVERY}")
    print(f"cell:     kv={KV} gap={GAP} task_vocab={TASK_VOCAB}")
    print(f"seeds:    {list(SEEDS)}")
    print()

    failures = preflight_failures(CELLS, base)
    for cell in CELLS:
        cfg = resolve_cell(cell, base)
        print(f"  {cell.name:<24} policy={cell.policy:<22} "
              f"ridge={cell.ridge:<7} route={route_of(cfg):<17} "
              f"gamma={cfg.ska_gamma_value} K={cfg.ska_power_K} "
              f"hash={config_hash(cfg)[:8]}")
    if failures:
        print()
        for f in failures:
            print(f"GATE FAILURE: {f}", file=sys.stderr)
        print(f"\n{len(failures)} gate failure(s). NOT preparing.", file=sys.stderr)
        return 1
    print(f"\nall gates passed: {len(CELLS)} cells, {task_count()} tasks")

    if args.dry_run:
        print("\n--dry_run: nothing written")
        print(f"array width would be: 0-{task_count() - 1}")
        return 0

    args.run_root.mkdir(parents=True, exist_ok=True)
    for cell in CELLS:
        path = write_model_yaml(cell, resolve_cell(cell, base), args.run_root)
        # Round-trip HERE, so a malformed extraction fails on the login node
        # rather than after a GPU is claimed.
        from koopman_lm.config import load_config
        if load_config(path) != resolve_cell(cell, base):
            print(f"FATAL: {path} does not round-trip", file=sys.stderr)
            return 1
    tasks = manifest()
    (args.run_root / "manifest.json").write_text(json.dumps(
        {"steps": STEPS, "eval_every": EVAL_EVERY, "kv": KV, "gap": GAP,
         "task_vocab": TASK_VOCAB, "seeds": list(SEEDS),
         "tasks": [{"index": t.index, "cell": t.cell.name,
                    "policy": t.cell.policy, "ridge": t.cell.ridge,
                    "seed": t.seed, "run_name": t.run_name} for t in tasks]},
        indent=2))
    print(f"wrote {len(CELLS)} model YAML(s) and manifest.json")
    print(f"\nARRAY_WIDTH=0-{len(tasks) - 1}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
