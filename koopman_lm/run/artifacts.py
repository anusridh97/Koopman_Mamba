"""Artifact write policy (§3.7).

Earned bytes (checkpoints, result JSONs -- hours to days of GPU time) are
protected by a refuse-to-clobber guard on run-directory creation. Derived
bytes (spec.yaml, sbatch files, attempts.jsonl entries, aggregation tables)
are cheap and reproducible, and use ordinary atomic writes (temp file +
os.replace) so a preemption mid-write cannot corrupt them.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional


class RunDirConflictError(RuntimeError):
    """Raised when a run directory already holds a completed run (`final/`)."""


def atomic_write_text(path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / (path.name + f".tmp{os.getpid()}")
    tmp.write_text(text)
    os.replace(tmp, path)


def atomic_write_json(path, data: Dict[str, Any], indent: int = 2) -> None:
    atomic_write_text(path, json.dumps(data, indent=indent, default=str))


def atomic_torch_save(path, obj: Any) -> None:
    """Binary counterpart of atomic_write_text: temp file + os.replace, so a
    kill mid-write (a preemption, e.g.) cannot leave a truncated resume.pt.
    torch is imported lazily so this module keeps working without it."""
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / (path.name + f".tmp{os.getpid()}")
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)
    except BaseException:
        if tmp.exists():
            tmp.unlink()
        raise


def create_run_dir(run_dir, *, resume: bool = False, force: bool = False) -> Path:
    """Create (or reuse) a run directory, refusing to clobber a finished run.

    Raises RunDirConflictError if `run_dir/final/` exists and neither
    `resume` nor `force` is set. `resume` continues from resume.pt (§5,
    not implemented by this plan); `force` proceeds anyway and is recorded
    in attempts.jsonl by the caller.
    """
    run_dir = Path(run_dir)
    final_dir = run_dir / "final"
    if final_dir.is_dir() and not (resume or force):
        raise RunDirConflictError(
            f"{run_dir} already contains a completed run (final/ exists). "
            f"Pass --resume to continue from resume.pt or --force to "
            f"overwrite (recorded in attempts.jsonl).")
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
    was forced. Answers *which execution produced these bytes* (§3.3)."""
    record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": host,
        "job_id": job_id,
        "git_commit": git_commit,
        "forced": forced,
    }
    if extra:
        record.update(extra)
    return record
