"""Code provenance for a run: is the tree clean, and what commit is this?

Split out of resolve.py, which was three unrelated things -- the spec-file
lifecycle (extends: chains, materialize, load), this, and flat-dict conversion.
What is here answers "which bytes produced this run?"; resolve.py answers "what
does this spec say?".

The `code_id` stamped into every run directory IS `git_commit()` below
(run/write_policy.py passes it to create_run_dir, and materialize copies it into
spec.yaml's provenance block). That is why `check_git_clean` refuses a dirty tree
by default: a commit hash that does not describe the running code makes every
downstream "what did this run do?" question unanswerable.
"""
from __future__ import annotations

import subprocess
import sys
import time
from typing import Any, Dict, List

import torch

__all__ = ["DirtyTreeError", "git_dirty_paths", "check_git_clean", "git_commit",
           "provenance"]


class DirtyTreeError(RuntimeError):
    """Raised when the working tree has uncommitted changes and
    --allow-dirty was not passed. Launching from an uncommitted tree makes
    code_id (the git commit, §4.2) meaningless -- the point of this guard is
    that launching uncommitted becomes a deliberate act, not an accident."""


def git_dirty_paths() -> List[str]:
    """`git status --porcelain` lines for the working tree, or [] if clean
    (or git is unavailable). --porcelain respects .gitignore, so run outputs
    and scratch never trip this."""
    try:
        out = subprocess.run(["git", "status", "--porcelain"],
                              capture_output=True, text=True, check=True)
    except Exception:
        return []
    return [line for line in out.stdout.splitlines() if line.strip()]


def check_git_clean(*, allow_dirty: bool) -> bool:
    """Refuse to launch from a dirty tree unless `allow_dirty` is set.

    Returns whether the tree was dirty, so the caller can stamp
    `dirty: true` into the materialized spec.yaml. Raises DirtyTreeError
    (listing the offending paths) when dirty and not allowed -- this must
    run before anything is materialized.
    """
    paths = git_dirty_paths()
    if not paths:
        return False
    if not allow_dirty:
        listing = "\n".join(f"  {p}" for p in paths)
        raise DirtyTreeError(
            f"refusing to launch from a dirty working tree "
            f"({len(paths)} uncommitted path(s)):\n{listing}\n"
            f"Commit or stash these changes, or pass --allow-dirty to "
            f"launch anyway (recorded as dirty: true in spec.yaml).")
    listing = "\n".join(f"  {p}" for p in paths)
    print(f"[experimentation.run] WARNING: launching with --allow-dirty -- "
          f"{len(paths)} uncommitted path(s), code_id will not reflect "
          f"them:\n{listing}")
    return True


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True)
        return out.stdout.strip()
    except Exception:
        return "unknown"


def provenance() -> Dict[str, Any]:
    """Stamp git commit + torch/CUDA/python versions at materialization time
    (§3.2)."""
    return {
        "git_commit": git_commit(),
        # str(...): torch.__version__ is a TorchVersion (str subclass), and
        # PyYAML's SafeRepresenter keys off the exact type, not isinstance --
        # it can't represent the subclass without this cast (verified: without
        # it, yaml.safe_dump raises RepresenterError("cannot represent an
        # object", '2.13.0+cpu')).
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "python_version": sys.version.split()[0],
        "materialized_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
