#!/usr/bin/env python3
"""Emit the run-identity baseline that code-tests/test_identity_baseline.py pins.

Run identity (config_hash / group_id / run_id / sweep_id) is what names every
run directory, groups replicates, and stamps every checkpoint. A refactor that
changes any of these silently renames completed science: re-deriving a finished
run's id from its own spec would produce a different directory, and its
checkpoint's stored cfg_hash would stop matching a freshly-loaded config.

This script exists so the expected values can be generated BEFORE a refactor and
committed first. A pin written afterwards records whatever the refactored code
happens to produce, which proves nothing.

    python scripts/gen_identity_baseline.py > code-tests/identity_baseline.json

Regenerate ONLY when an intentional change to run identity has been decided and
recorded -- never to make a red test go green.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

RUN_SPECS = [
    "configs/runs/50m-fineweb-3b.yaml",
    "configs/runs/50m-first-real.yaml",
    "configs/runs/50m-mqar-smoke.yaml",
]
SWEEP_SPECS = [
    "configs/sweeps/ska-rank-lr.yaml",
]


def build_baseline() -> dict:
    from koopman_lm.config import CONFIG_REGISTRY, build_config, config_hash
    from koopman_lm.run.resolve import resolve_run_spec
    from koopman_lm.run.spec import group_id, run_id
    from koopman_lm.sweep.spec import expand_cells, load_sweep_spec, sweep_id

    out: dict = {"config_hash": {}, "run_spec": {}, "sweep": {}}

    for name in sorted(CONFIG_REGISTRY):
        out["config_hash"][name] = config_hash(build_config(name))

    for rel in RUN_SPECS:
        spec = resolve_run_spec(REPO_ROOT / rel)
        out["run_spec"][rel] = {
            "config_hash": config_hash(spec.model),
            "group_id": group_id(spec),
            "run_id": run_id(spec),
        }

    for rel in SWEEP_SPECS:
        sweep = load_sweep_spec(REPO_ROOT / rel)
        cells = expand_cells(sweep)
        out["sweep"][rel] = {
            "sweep_id": sweep_id(sweep),
            "n_cells": len(cells),
            # str() the override values so 2.0e-4 round-trips through JSON
            # identically regardless of float repr.
            "cells": [
                {
                    "overrides": {k: str(v) for k, v in sorted(c.overrides.items())},
                    "group_id": group_id(c.spec),
                    "run_id": run_id(c.spec),
                }
                for c in cells
            ],
        }

    return out


def main() -> int:
    sys.path.insert(0, str(REPO_ROOT))
    json.dump(build_baseline(), sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
