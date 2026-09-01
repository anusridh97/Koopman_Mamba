"""Run-directory write policy (§3.7).

Earned bytes (checkpoints, result JSONs -- hours to days of GPU time) are
protected by a refuse-to-clobber guard on run-directory creation. Derived
bytes (spec.yaml, sbatch files, attempts.jsonl entries, aggregation tables)
are cheap and reproducible, and use the ordinary atomic writes in
experimentation/atomic_io.py so a preemption mid-write cannot corrupt them.

Renamed from run/artifacts.py: "artifacts" described neither half of what that
file held. This half is run-directory lifecycle -- create, guard, append -- which
is what the module docstring already called it.
"""
from __future__ import annotations

import contextlib
import json
import os
import shutil
import socket
import time
from pathlib import Path
from typing import Any, Dict, Optional

from experimentation.atomic_io import atomic_write_text


class RunDirConflictError(RuntimeError):
    """Raised when a run directory already holds a completed run (`final/`).

    ``reusable`` is true only when the existing run records the exact current
    commit.  An adaptive search can then score the already-earned result instead
    of launching the same content-derived run twice.  A code-id mismatch (or a
    legacy run with no code id) remains a hard conflict: identical scientific
    inputs do not make results from different code interchangeable.
    """

    def __init__(self, message: str, *, reusable: bool = False):
        super().__init__(message)
        self.reusable = bool(reusable)


class RunDirClaimedError(RuntimeError):
    """Raised when another launcher already holds this run directory's claim."""


def _existing_code_id(run_dir: Path) -> Optional[str]:
    """Best-effort read of a prior materialized spec.yaml's code_id (§3.7 /
    §4.2 provenance). Returns None if there's no spec.yaml or it can't be
    parsed -- callers fall back to the generic collision message."""
    spec_path = Path(run_dir) / "spec.yaml"
    if not spec_path.is_file():
        return None
    try:
        import yaml
        raw = yaml.safe_load(spec_path.read_text()) or {}
    except Exception:
        return None
    return raw.get("code_id")


def create_run_dir(run_dir, *, resume: bool = False, force: bool = False,
                    code_id: Optional[str] = None) -> Path:
    """Create (or reuse) a run directory, refusing to clobber a finished run.

    Raises RunDirConflictError if `run_dir/final/` exists and neither
    `resume` nor `force` is set. `resume` continues from resume.pt (§5,
    not implemented by this plan); `force` proceeds anyway and is recorded
    in attempts.jsonl by the caller.

    `code_id` (the current git commit) is compared against the existing
    run's materialized spec.yaml, if present. run_id collisions across
    different code (a code-only change alters no scientific input, so it
    collides on run_id by design -- see design doc §3.3/§3.7) get an
    explicit message instead of being folded into the generic clobber error.
    """
    run_dir = Path(run_dir)
    final_dir = run_dir / "final"
    if final_dir.is_dir() and not (resume or force):
        existing = _existing_code_id(run_dir)
        if code_id is not None and existing is not None and existing != code_id:
            raise RunDirConflictError(
                f"{run_dir} already contains a completed run (final/ exists) "
                f"with the same run_id but a different code_id: existing "
                f"spec.yaml records code_id={existing!r}, this launch is "
                f"code_id={code_id!r}. This is a code change colliding on "
                f"run_id (run_id hashes only model+data+optim+seed, not "
                f"code) -- these are likely different experiments, not a "
                f"relaunch of the same one. Pass --resume to continue from "
                f"resume.pt or --force to overwrite (recorded in "
                f"attempts.jsonl).")
        raise RunDirConflictError(
            f"{run_dir} already contains a completed run (final/ exists). "
            f"Pass --resume to continue from resume.pt or --force to "
            f"overwrite (recorded in attempts.jsonl).",
            # ``unknown`` is not provenance.  Treating two unknowns as the same
            # code would silently bless a result whose implementation cannot be
            # recovered.
            reusable=(code_id not in (None, "unknown") and existing == code_id))
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def append_attempt(run_dir, attempt: Dict[str, Any]) -> None:
    """Append one JSON line to `run_dir/attempts.jsonl`. Never rewrites prior
    lines -- the audit trail of every execution against this run_id."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "attempts.jsonl"
    line = json.dumps(attempt, default=str)
    with open(path, "a") as f:
        f.write(line + "\n")


def make_attempt_record(*, host: str, job_id: Optional[str], git_commit: str,
                         forced: bool = False,
                         extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """One attempt record: timestamp, host, job id, git commit, whether it
    was forced. Answers *which execution produced these bytes* (§3.3).

    `code_id` duplicates `git_commit` under the name used elsewhere (spec.yaml,
    create_run_dir's collision check) for the commit that actually executed --
    so every attempts.jsonl entry is self-describing without cross-referencing
    field names.
    """
    record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": host,
        "job_id": job_id,
        "git_commit": git_commit,
        "code_id": git_commit,
        "forced": forced,
    }
    if extra:
        record.update(extra)
    return record


CLAIM_SENTINEL = ".running"


@contextlib.contextmanager
def claim_run_dir(run_dir, *, force: bool = False):
    """Mutual exclusion over one run directory's *materialization*.

    `create_run_dir`'s guard keys on `final/` -- on runs that have already
    finished -- and its `mkdir(exist_ok=True)` is a check-then-act. Neither
    stops two launchers that resolve the same spec at the same moment from
    both proceeding into one directory and interleaving their `spec.yaml`,
    `model_config.json` and `attempts.jsonl` writes. A parallel searcher makes
    that collision ordinary rather than exotic, since it materializes many
    cells in a burst.

    Scope is deliberately the materialization critical section, not the run
    lifetime. Holding a claim for the duration of training would mean a
    launcher process outliving its own Slurm submission (it does not -- the job
    starts later, elsewhere), and would make every sequential relaunch of a
    failed run require `--force`. The window this closes is the launcher's own
    few hundred milliseconds of writing, which is exactly where the race is.

    `os.mkdir` is the primitive because it is atomic even on NFS, where
    `open(..., 'x')`'s O_EXCL is not reliably so. A claim record naming the
    holder goes inside, because a sentinel left behind by a launcher killed
    mid-materialize is only actionable if it says who left it.

    `force=True` steals an existing claim, matching what `--force` already
    means elsewhere: proceed anyway, and let the caller record that it did.
    """
    run_dir = Path(run_dir)
    sentinel = run_dir / CLAIM_SENTINEL
    run_dir.mkdir(parents=True, exist_ok=True)
    try:
        os.mkdir(sentinel)
    except FileExistsError:
        if not force:
            raise RunDirClaimedError(
                f"{run_dir} is already claimed by another launcher "
                f"({_describe_claim(sentinel)}). Two launchers materializing "
                f"one spec would interleave their writes. Wait for it to "
                f"finish, or pass --force to steal the claim if you believe "
                f"the holder is dead.")
    atomic_write_text(sentinel / "claim.json", json.dumps({
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }, indent=2, sort_keys=True) + "\n")
    try:
        yield run_dir
    finally:
        shutil.rmtree(sentinel, ignore_errors=True)


def _describe_claim(sentinel: Path) -> str:
    """Best-effort rendering of an existing claim, for the error message."""
    try:
        claim = json.loads((sentinel / "claim.json").read_text())
    except Exception:
        return "no readable claim record"
    return (f"host={claim.get('host')} pid={claim.get('pid')} "
            f"job_id={claim.get('job_id')} at {claim.get('timestamp')}")
