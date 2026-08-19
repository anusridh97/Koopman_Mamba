"""Materialize one run directory: the sequence every launch path shares.

Extracted from `sweep/__main__.py::_materialize_cell`. Two entry points need
this exact sequence -- the static sweep (`python -m experimentation.sweep`) and
the adaptive searcher (`python -m experimentation.sweep.search`) -- and a
private function inside a `__main__` module cannot be imported by the second
without importing a module whose job is to be executed. Extracting it is what
makes one implementation serve both, rather than two that drift.

The generalisation is the only behavioural change: it takes a `RunSpec` plus an
opaque `extra` mapping instead of a `SweepCell` plus a `SweepSpec`. A static
sweep stamps `sweep_id`/`sweep_name`; a search stamps its own study
identifiers. This module does not need to know either vocabulary, and adding a
third caller should not require editing it.

The order of the steps is load-bearing and is byte-for-byte what
`run/__main__.py` does for a single run (§3.4):

    verify data     -- before anything exists on disk, so an unlaunchable spec
                       leaves no directory, no spec.yaml, and no fabricated
                       attempts.jsonl entry (the PR #23 gate)
    claim           -- mutual exclusion over the writes below
    create_run_dir  -- refuse to clobber a completed run
    materialize     -- the flattened spec.yaml, the only file downstream reads
    append_attempt  -- the audit trail of this execution
"""
from __future__ import annotations

import os
import socket
from pathlib import Path
from typing import Any, Mapping, Optional

from experimentation.run.data_verify import verify_data
from experimentation.run.provenance import git_commit
from experimentation.run.resolve import materialize
from experimentation.run.spec import RunSpec, run_dir_path
from experimentation.run.write_policy import (
    append_attempt, claim_run_dir, create_run_dir, make_attempt_record)

__all__ = ["materialize_cell"]


def materialize_cell(spec: RunSpec, run_root, *,
                     extra: Optional[Mapping[str, Any]] = None,
                     dirty: bool = False,
                     force: bool = False,
                     dry_run: bool = False) -> Path:
    """Prepare one run directory for `spec` and return its path.

    Does not launch anything: hand the returned path plus `spec` to a Launcher.
    Keeping preparation and hand-off separate is what lets the sweep path
    collect every materialized cell first and then submit them as a single
    Slurm array, and what lets the searcher decide per trial.

    `extra` is merged into the materialized spec.yaml as top-level metadata. It
    plays no part in run_id/group_id, which are computed from `spec` alone
    (§3.3) -- how a run was launched is not a scientific input.
    """
    # Single gate for every data kind, BEFORE any directory exists.
    verify_data(spec.data, dry_run=dry_run)
    run_dir = run_dir_path(run_root, spec)
    # A sweep or a search materializes many cells in a burst, so two overlapping
    # ones sharing a base spec collide here far more readily than two
    # hand-launched runs would.
    with claim_run_dir(run_dir, force=force):
        create_run_dir(run_dir, force=force, code_id=git_commit())
        materialize(spec, run_dir, dirty=dirty, extra=dict(extra) if extra else None)
        append_attempt(run_dir, make_attempt_record(
            host=socket.gethostname(), job_id=os.environ.get("SLURM_JOB_ID"),
            git_commit=git_commit(), forced=force))
    return run_dir
