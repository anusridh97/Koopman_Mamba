"""Reading a run back: its objective, and why it died.

The searcher's only channel from a finished run is that run's directory. This is
deliberately the whole interface -- nothing here holds a subprocess handle or a
pipe, so the same code works whether the run finished a second ago on this machine
or four hours ago on a compute node.

`read_quick_eval_objective` is the default `read_objective` for the driver: it
finds the run's newest quick_eval result and scalarises it. It returns None rather
than a number when there is nothing to read, because the driver turns None into a
FAIL and any stand-in value would be a fiction that steers every later proposal.

`looks_like_oom` exists so the OOM ladder only descends when descending can help.
Retrying a shape error at a smaller microbatch burns a queue slot to fail the
same way. The markers are broad on purpose -- an allocation failure surfaces
differently from PyTorch, from cuBLAS and from the caching allocator, and all
three mean the same thing to a searcher.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from experimentation.sweep.search.driver import objective_from_metrics

__all__ = ["OOM_MARKERS", "looks_like_oom", "read_quick_eval_metrics",
           "read_quick_eval_objective"]

# Lowercase substrings. Broad by design: the same condition is reported by the
# allocator, by cuBLAS, and by torch's own error type.
OOM_MARKERS = (
    "cuda out of memory",
    "outofmemoryerror",
    "cublas_status_alloc_failed",
    "failed to allocate",
    "out of memory",
)

# Where a run's stdout lands: the Slurm launcher routes it to
# run_dir/slurm-%j.out (see run/launchers.py's sbatch template); a local run's
# log, when captured, sits beside it.
_LOG_GLOBS = ("slurm-*.out", "*.log")

_CHECKPOINT_PREFERENCE = ("final",)


def looks_like_oom(run_dir) -> bool:
    """Did this run die for want of memory?

    Reads the tail rather than the whole file: a diverged 15,000-step run's log
    can be large, and an allocation failure is always near the end.
    """
    run_dir = Path(run_dir)
    for pattern in _LOG_GLOBS:
        for log in sorted(run_dir.glob(pattern)):
            try:
                text = log.read_text(errors="replace")[-200_000:].lower()
            except OSError:
                continue
            if any(marker in text for marker in OOM_MARKERS):
                return True
    return False


def read_quick_eval_metrics(run_dir) -> Optional[Dict[str, Any]]:
    """The newest quick_eval payload under `run_dir/eval/`, or None.

    "final" wins when present, since a run that saved intermediate checkpoints
    and then finished should be scored on the finished model; otherwise the
    most recently written result is used.
    """
    eval_dir = Path(run_dir) / "eval"
    if not eval_dir.is_dir():
        return None

    candidates = sorted(eval_dir.glob("*/quick_eval.json"))
    if not candidates:
        return None

    chosen = None
    for preferred in _CHECKPOINT_PREFERENCE:
        for candidate in candidates:
            if candidate.parent.name == preferred:
                chosen = candidate
                break
        if chosen is not None:
            break
    if chosen is None:
        chosen = max(candidates, key=lambda p: p.stat().st_mtime)

    try:
        envelope = json.loads(chosen.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    # The result envelope nests the numbers under "metrics"; tolerate a bare
    # payload so a hand-written file is still readable.
    return envelope.get("metrics", envelope)


def read_quick_eval_objective(run_dir, **weights) -> Optional[float]:
    """The scalar for this run, or None if it produced no readable result."""
    metrics = read_quick_eval_metrics(run_dir)
    if metrics is None or "full" not in metrics:
        return None
    return objective_from_metrics(metrics, **weights)
