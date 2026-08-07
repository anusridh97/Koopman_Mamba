"""python -m koopman_lm.run <spec.yaml> [--launcher local|slurm] [--resume]
[--force] [--run_root PATH] [--dry_run] [--allow-dirty] (§3.4 orchestration
order):

    refuse to launch from a dirty working tree (unless --allow-dirty)
    resolve spec (extends -> flat) + validate
    verify data shard matches spec (tokenizer, mix, n_tokens)
    materialize spec.yaml + attempt record into the run dir
    hand off to a Launcher
"""
from __future__ import annotations

import argparse
import os
import socket
from pathlib import Path

from koopman_lm.run.artifacts import append_attempt, create_run_dir, make_attempt_record
from koopman_lm.run.data_verify import verify_shard
from koopman_lm.run.launch import LocalLauncher
from koopman_lm.run.resolve import check_git_clean, git_commit, materialize, resolve_run_spec
from koopman_lm.run.slurm import SlurmLauncher
from koopman_lm.run.spec import ShardDataSpec, run_dir_path, run_id as compute_run_id

LAUNCHERS = {"local": LocalLauncher, "slurm": SlurmLauncher}


def parse_args(argv=None):
    p = argparse.ArgumentParser(prog="python -m koopman_lm.run")
    p.add_argument("spec", help="path to configs/runs/<name>.yaml")
    p.add_argument("--launcher", choices=sorted(LAUNCHERS), default="local")
    p.add_argument("--run_root", default=os.environ.get("RUN_ROOT", "./runs"))
    p.add_argument("--resume", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--dry_run", action="store_true",
                    help="build launch.sbatch / the local command without submitting")
    p.add_argument("--allow-dirty", dest="allow_dirty", action="store_true",
                    help="launch despite uncommitted changes (loud escape hatch; "
                         "recorded as dirty: true in spec.yaml). Default is refusal.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    dirty = check_git_clean(allow_dirty=args.allow_dirty)
    spec = resolve_run_spec(args.spec)

    if isinstance(spec.data, ShardDataSpec):
        verify_shard(spec.data)

    run_dir = run_dir_path(args.run_root, spec)
    create_run_dir(run_dir, resume=args.resume, force=args.force, code_id=git_commit())

    materialize(spec, run_dir, dirty=dirty)
    append_attempt(run_dir, make_attempt_record(
        host=socket.gethostname(),
        job_id=os.environ.get("SLURM_JOB_ID"),
        git_commit=git_commit(),
        forced=args.force,
    ))

    launcher = LAUNCHERS[args.launcher]()
    result = launcher.submit(spec, run_dir, dry_run=args.dry_run, resume=args.resume)
    print(f"[koopman_lm.run] run_id={compute_run_id(spec)} run_dir={run_dir}")
    return result


if __name__ == "__main__":
    main()
