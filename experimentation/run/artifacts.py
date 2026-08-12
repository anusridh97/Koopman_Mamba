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
