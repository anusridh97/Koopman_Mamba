"""python -m experimentation.sweep <sweep.yaml> [--launcher local|slurm]
[--dry_run] [--skip_done] [--run_root PATH] [--force] [--allow-dirty]

Expands configs/sweeps/<name>.yaml (§4.2) into one RunSpec per cell via
experimentation.sweep.spec.expand_cells -- the single place the grid is declared
-- then drives each surviving cell through the same primitives
`python -m experimentation.run` uses per-run: experimentation.run.resolve/artifacts/
launch. This is a NEW entry point, deliberately not a modification of
experimentation/run/__main__.py (a separate agent owns that file's SIGUSR1 work).

Orchestration order, per invocation:
    expand the sweep into cells (experimentation.sweep.spec.expand_cells)
    print the full cell list (run_id/group_id/seed/status) -- always, so
      --dry_run is a complete, GPU-free view of what the sweep would do
    drop cells already done (--skip_done: run_dir/final/ exists)
    refuse to launch from a dirty working tree (unless --allow-dirty)
    for each surviving cell: verify data, create_run_dir, materialize
      spec.yaml (stamped with this sweep's sweep_id/sweep_name), append an
      attempt record -- byte-for-byte what experimentation.run.__main__ does for
      a single run
    hand off every surviving cell to a Launcher:
      --launcher local  one LocalLauncher.submit() per cell
      --launcher slurm  one SlurmLauncher.submit_array() for the whole
                         sweep (experimentation/run/slurm.py's array-job support,
                         §4.3) -- a single #SBATCH --array=0-N[%K] job, not N
                         separate submissions
"""
from __future__ import annotations

import argparse
import os
import socket
from pathlib import Path
from typing import List, Tuple

from experimentation.run.artifacts import append_attempt, create_run_dir, make_attempt_record
from experimentation.run.data_verify import verify_shard
from experimentation.run.launch import LocalLauncher
from experimentation.run.resolve import check_git_clean, git_commit, materialize
from experimentation.run.slurm import SlurmLauncher
from experimentation.run.spec import (
    RunSpec, ShardDataSpec, group_id as compute_group_id, run_dir_path,
    run_id as compute_run_id,
)
from experimentation.sweep.spec import SweepCell, SweepSpec, expand_cells, load_sweep_spec
from experimentation.sweep.spec import sweep_id as compute_sweep_id

LAUNCHERS = ("local", "slurm")


def parse_args(argv=None):
    p = argparse.ArgumentParser(prog="python -m experimentation.sweep")
    p.add_argument("sweep", help="path to configs/sweeps/<name>.yaml")
    p.add_argument("--launcher", choices=LAUNCHERS, default="local")
    p.add_argument("--run_root", default=os.environ.get("RUN_ROOT", "./runs"))
    p.add_argument("--dry_run", action="store_true",
                    help="materialize + build commands/sbatch without ever "
                         "running training or calling sbatch -- the "
                         "GPU-free testing surface")
    p.add_argument("--skip_done", action="store_true",
                    help="skip cells whose run dir already has final/, so a "
                         "partially-completed sweep resumes")
    p.add_argument("--force", action="store_true",
                    help="proceed even if a target run dir has final/ "
                         "(recorded in attempts.jsonl); see --skip_done for "
                         "the usual resume path")
    p.add_argument("--allow-dirty", dest="allow_dirty", action="store_true",
                    help="launch despite uncommitted changes (loud escape "
                         "hatch; recorded as dirty: true in each cell's "
                         "spec.yaml). Default is refusal.")
    return p.parse_args(argv)


def is_cell_done(run_dir) -> bool:
    return (Path(run_dir) / "final").is_dir()


def _print_plan(sweep: SweepSpec, cells: List[SweepCell], run_root) -> None:
    sid = compute_sweep_id(sweep)
    print(f"[experimentation.sweep] {sweep.name} sweep_id={sid}  {len(cells)} cell(s)")
    for cell in cells:
        run_dir = run_dir_path(run_root, cell.spec)
        status = "DONE" if is_cell_done(run_dir) else "pending"
        overrides = ", ".join(f"{k}={v}" for k, v in sorted(cell.overrides.items()))
        print(f"  group_id={compute_group_id(cell.spec)} "
              f"run_id={compute_run_id(cell.spec)} seed={cell.spec.runtime.seed} "
              f"[{status}] {overrides}  -> {run_dir}")


def _materialize_cell(cell: SweepCell, run_root, sweep: SweepSpec, *,
                       dirty: bool, force: bool, dry_run: bool) -> Path:
    spec = cell.spec
    if isinstance(spec.data, ShardDataSpec):
        verify_shard(spec.data, dry_run=dry_run)
    run_dir = run_dir_path(run_root, spec)
    create_run_dir(run_dir, force=force, code_id=git_commit())
    materialize(spec, run_dir, dirty=dirty,
                extra={"sweep_id": compute_sweep_id(sweep), "sweep_name": sweep.name})
    append_attempt(run_dir, make_attempt_record(
        host=socket.gethostname(), job_id=os.environ.get("SLURM_JOB_ID"),
        git_commit=git_commit(), forced=force))
    return run_dir


def main(argv=None):
    args = parse_args(argv)
    sweep = load_sweep_spec(args.sweep)
    cells = expand_cells(sweep)
    if not cells:
        raise ValueError(
            f"sweep {sweep.name!r} expanded to zero cells -- check "
            f"axes/cells and exclude in {args.sweep}")

    _print_plan(sweep, cells, args.run_root)

    pending = [c for c in cells if not (
        args.skip_done and is_cell_done(run_dir_path(args.run_root, c.spec)))]
    if args.skip_done and len(pending) < len(cells):
        print(f"[experimentation.sweep] --skip_done: "
              f"{len(cells) - len(pending)} already-done cell(s) skipped, "
              f"{len(pending)} remaining")
    if not pending:
        print("[experimentation.sweep] nothing to do (every cell is done)")
        return []

    dirty = check_git_clean(allow_dirty=args.allow_dirty)

    materialized: List[Tuple[RunSpec, Path]] = [
        (cell.spec, _materialize_cell(cell, args.run_root, sweep,
                                       dirty=dirty, force=args.force,
                                       dry_run=args.dry_run))
        for cell in pending
    ]

    if args.launcher == "local":
        return [LocalLauncher().submit(spec, run_dir, dry_run=args.dry_run)
                for spec, run_dir in materialized]

    sweep_dir = Path(args.run_root) / "_sweeps" / f"{sweep.name}.{compute_sweep_id(sweep)}"
    return SlurmLauncher().submit_array(
        sweep.name, materialized, sweep_dir,
        concurrency=sweep.max_concurrent, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
