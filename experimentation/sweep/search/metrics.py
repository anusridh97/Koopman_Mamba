"""Reading a run back: its objective, and why it died.

The searcher's only channel from a finished run is that run's directory. This is
deliberately the whole interface -- nothing here holds a subprocess handle or a
pipe, so the same code works whether the run finished a second ago on this machine
or four hours ago on a compute node.

`read_quick_eval_objective` scores a FINISHED run: it finds the newest quick_eval
result and scalarises it. It stays a pure `(run_dir) -> float|None`; wrap it in
`fixed_reader` to satisfy the driver's factory-shaped `objective_reader_for`. It returns None rather
than a number when there is nothing to read, because the driver turns None into a
FAIL and any stand-in value would be a fiction that steers every later proposal.

`looks_like_oom` exists so the OOM ladder only descends when descending can help.
Retrying a shape error at a smaller microbatch burns a queue slot to fail the
same way. The markers are broad on purpose -- an allocation failure surfaces
differently from PyTorch, from cuBLAS and from the caching allocator, and all
three mean the same thing to a searcher.

**Pruning, asynchronously.** The original harness pruned by holding a
`subprocess.PIPE` open and regex-matching the child's stdout as it arrived. That
works, and it is also the one thing that forced the whole design into a single
long-lived process and off Slurm. The observation that unlocks the async version
is that `run/launchers.py`'s sbatch template *already* routes training stdout to
`run_dir/slurm-%j.out` -- so the same regex works against a durable file,
readable at any time by a process that has never met the training job.

`wait_for_objective` is deliberately shaped to be the reader the driver's
`objective_reader_for` returns, which is why pruning needed no driver change
beyond honouring `TrialPruned`. It needs the trial -- pruning IS `trial.report`
plus `trial.should_prune` -- which is exactly why that parameter is a factory and
not a plain reader. It polls the log, reports each newly-seen step, asks the pruner,
cancels and raises when told to, and otherwise returns the objective once the
eval result lands.

`TRAIN_RE` is coupled to a `print` statement (`training/train.py:415-416`). That
coupling is real and worth naming: reformat that f-string and pruning silently
stops working, because a non-matching line is indistinguishable from no progress.
The alternative -- a structured metric stream from the trainer -- is the cleaner
fix and is recorded in the backlog as A2(c); it modifies a file the run-system
design calls frozen, so it is a decision rather than a detail.
"""
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

__all__ = ["OOM_MARKERS", "TRAIN_RE", "Progress", "looks_like_oom",
           "objective_from_metrics",
           "parse_progress", "read_progress", "ema_losses",
           "fixed_reader",
           "read_quick_eval_metrics", "read_quick_eval_objective",
           "wait_for_objective"]

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


def objective_from_metrics(metrics: Mapping[str, Any], *,
                           parameter_penalty: float = 0.0,
                           param_count: Optional[int] = None,
                           baseline_param_count: Optional[int] = None,
                           throughput_penalty: float = 0.0,
                           target_tokens_per_sec: float = 0.0,
                           ska_delta_reward: float = 0.0,
                           ska_delta_cap: float = 0.10) -> float:
    """A quick_eval payload -> the scalar optuna minimises.

    Pure, and every weight defaults to zero so the objective starts as the
    measured loss and nothing else. Each term is one-sided: a config smaller than
    the baseline earns no bonus, a config faster than the target earns no bonus,
    and an ablation delta that went the wrong way earns no bonus. One-sidedness
    matters because these are constraints being expressed as penalties, not
    quantities being jointly optimised.
    """
    full = metrics.get("full", {})
    score = float(full.get("loss", 0.0))

    if parameter_penalty > 0 and param_count and baseline_param_count:
        excess_millions = max(0.0, (param_count - baseline_param_count) / 1e6)
        score += parameter_penalty * excess_millions

    if throughput_penalty > 0 and target_tokens_per_sec > 0:
        measured = float(full.get("tokens_per_sec") or 0.0)
        if measured > 0:
            score += throughput_penalty * max(0.0, target_tokens_per_sec / measured - 1.0)

    ablation = metrics.get("ska_ablation", {})
    if ska_delta_reward > 0 and ablation.get("supported"):
        delta = float(ablation.get("loss_delta") or 0.0)
        # Capped, and floored at zero: a negative delta means zeroing SKA made
        # the model better, which must not become a reward by sign error.
        score -= ska_delta_reward * max(0.0, min(delta, ska_delta_cap))

    return score

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


def fixed_reader(reader):
    """Adapt a `(run_dir) -> float|None` reader to the driver's factory shape.

    `run_trial` takes one scoring parameter, `objective_reader_for`, of shape
    `(study, trial) -> (run_dir) -> float|None`. Pruning needs the trial, so the
    factory is the general shape; a reader that does not care simply ignores both
    arguments. This is that adapter, so `read_quick_eval_objective` and friends
    stay exactly the pure "score a finished run" function they are documented to
    be, with no second signature to maintain.

    Deliberately NOT the reverse: widening the pure reader to
    `(run_dir, study, trial)` would force `read_quick_eval_objective` to accept
    two arguments it ignores and drag optuna into this module's scope, where the
    import is lazy on purpose (see `wait_for_objective`).
    """
    return lambda _study, _trial: reader


def read_quick_eval_objective(run_dir, **weights) -> Optional[float]:
    """The scalar for this run, or None if it produced no readable result."""
    metrics = read_quick_eval_metrics(run_dir)
    if metrics is None or "full" not in metrics:
        return None
    return objective_from_metrics(metrics, **weights)


# training/train.py:415-416 prints, with a right-aligned 6-wide step:
#   step     10/600 | loss 7.1234 | ppl 1234.5 | lr 4.00e-04 | 12.3K tok/s
# The numeric class excludes "nan" on purpose: a diverged run prints nan, and a
# nan is not a datapoint to hand a pruner.
TRAIN_RE = re.compile(
    r"step\s+(?P<step>\d+)\s*/\s*(?P<total>\d+)\s*\|\s*"
    r"loss\s+(?P<loss>[0-9.eE+\-]+)\s*\|\s*"
    r"ppl\s+(?P<ppl>[0-9.eE+\-]+)\s*\|\s*"
    r"lr\s+(?P<lr>[0-9.eE+\-]+)\s*\|\s*"
    r"(?P<ktps>[0-9.eE+\-]+)K\s+tok/s")

# The original harness's smoothing. Raw step loss is noisy enough that one bad
# step could prune a good config.
DEFAULT_EMA_ALPHA = 0.45
DEFAULT_POLL_SECONDS = 30.0


@dataclass(frozen=True)
class Progress:
    """One logged training step."""
    step: int
    loss: float
    ppl: float
    lr: float
    tokens_per_sec: float


def parse_progress(text: str) -> List[Progress]:
    """Every training-progress line in `text`, in the order it appears."""
    found: List[Progress] = []
    for match in TRAIN_RE.finditer(text):
        try:
            found.append(Progress(
                step=int(match.group("step")),
                loss=float(match.group("loss")),
                ppl=float(match.group("ppl")),
                lr=float(match.group("lr")),
                tokens_per_sec=float(match.group("ktps")) * 1000.0,
            ))
        except ValueError:                 # pragma: no cover -- regex guards this
            continue
    return found


def read_progress(run_dir) -> List[Progress]:
    """Progress from every log in `run_dir`, ordered by step.

    Ordered rather than concatenated because a requeued job (`--requeue` is in the
    sbatch template) writes a second slurm-<jobid>.out, and interleaved steps
    would show the pruner a jagged curve that never happened.
    """
    run_dir = Path(run_dir)
    found: List[Progress] = []
    for pattern in _LOG_GLOBS:
        for log in sorted(run_dir.glob(pattern)):
            try:
                found.extend(parse_progress(log.read_text(errors="replace")))
            except OSError:
                continue
    return sorted(found, key=lambda p: p.step)


def ema_losses(values, alpha: float = DEFAULT_EMA_ALPHA) -> List[float]:
    """Exponential moving average, seeded with the first value."""
    smoothed: List[float] = []
    current: Optional[float] = None
    for value in values:
        current = float(value) if current is None else alpha * float(value) + (1 - alpha) * current
        smoothed.append(current)
    return smoothed


def _slurm_job_ids(run_dir: Path) -> List[str]:
    ids = []
    for log in sorted(run_dir.glob("slurm-*.out")):
        stem = log.stem[len("slurm-"):]
        job = stem.split("_")[0]
        if job.isdigit():
            ids.append(job)
    return ids


def _local_pid(run_dir) -> Optional[int]:
    """The PID of a non-blocking local run, if one was launched here.

    LocalLauncher writes this when submitted with wait=False. Reading it back
    from the run directory keeps cancellation uniform: a canceller needs only the
    run_dir, never the process object, exactly as the Slurm path needs only the
    run_dir to find its job ids.
    """
    path = Path(run_dir) / ".local_pid"
    try:
        return int(path.read_text().strip())
    except (FileNotFoundError, ValueError):
        return None


def _default_cancel(run_dir) -> None:
    """Stop whatever is running in this run directory, however it was launched.

    Pruning that does not actually stop the job saves nothing -- the point is the
    GPU-hours, not the bookkeeping. Which means this has to cover BOTH launchers:
    a Slurm trial is one job per trial, while a locally-launched trial is a child
    process, and the latter is the mode a held GPU allocation uses to avoid paying
    a queue wait per trial. Handling only Slurm made pruning silently ineffective
    there -- no error, the trial simply ran to max_steps.
    """
    run_dir = Path(run_dir)

    pid = _local_pid(run_dir)
    if pid is not None:
        # Signal the GROUP, not the process. LocalLauncher starts a new session
        # for this reason: a torchrun launch has one worker per GPU, and killing
        # only the parent leaves them holding the hardware.
        #
        # Both signals are sent unconditionally, with no liveness check between
        # them. Checking is what an earlier version did, and it does not work:
        # the training process is a CHILD of whoever is pruning, so between dying
        # and being waited on it is a zombie -- and os.kill(pid, 0) succeeds on a
        # zombie. So the check could never observe death and always escalated
        # anyway. SIGKILL to an already-dead group is a harmless
        # ProcessLookupError, which is cheaper than getting the check right.
        try:
            pgid = os.getpgid(pid)
        except (ProcessLookupError, PermissionError):
            pgid = None
        if pgid is not None:
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(pgid, sig)
                except (ProcessLookupError, PermissionError):
                    break
                time.sleep(0.5)

    for job_id in _slurm_job_ids(run_dir):
        try:
            subprocess.run(["scancel", job_id], check=False,
                           capture_output=True, text=True)
        except FileNotFoundError:
            # No scancel here (a local run, or a login node without Slurm).
            return


def wait_for_objective(study: optuna.study.Study, trial, run_dir, *,
                       poll_seconds: float = DEFAULT_POLL_SECONDS,
                       timeout_seconds: Optional[float] = None,
                       prune_anchors: bool = False,
                       ema_alpha: float = DEFAULT_EMA_ALPHA,
                       sleep: Callable[[float], Any] = time.sleep,
                       clock: Callable[[], float] = time.monotonic,
                       cancel: Optional[Callable[[Any], Any]] = None,
                       **weights) -> Optional[float]:
    """Poll a run to completion, reporting progress and pruning if hopeless.

    Returns the objective, or None on timeout -- the driver turns None into a
    FAIL, and a stand-in number would steer every later proposal off a fiction.
    Raises `optuna.TrialPruned` when the pruner says stop, after cancelling the
    job.

    Anchors are exempt by default: they *are* the reference set a median pruner
    compares against, so pruning them removes the thing later trials are judged
    by.
    """
    # optuna and is_anchor are imported here rather than at module scope: the
    # rest of this module -- log parsing, OOM detection, objective reading -- is
    # pure and must stay importable with optuna absent. Hard-importing it at the
    # top silently pulled the whole search package's optional dependency into the
    # OOM ladder, which broke the CPU suite.
    import optuna

    from experimentation.sweep.search.study import is_anchor

    run_dir = Path(run_dir)
    cancel = cancel if cancel is not None else _default_cancel
    prunable = prune_anchors or not is_anchor(trial)
    reported: set[int] = set()
    started = clock()

    while True:
        progress = read_progress(run_dir)
        if progress:
            smoothed = ema_losses([p.loss for p in progress], alpha=ema_alpha)
            for point, value in zip(progress, smoothed):
                # A poll loop re-reads the whole log every pass, and optuna
                # raises on a duplicate report step.
                if point.step not in reported:
                    trial.report(value, point.step)
                    reported.add(point.step)

        if prunable and reported and trial.should_prune():
            cancel(run_dir)
            raise optuna.TrialPruned(
                f"pruned at step {max(reported)} (run_dir={run_dir})")

        objective = read_quick_eval_objective(run_dir, **weights)
        if objective is not None:
            return objective

        if timeout_seconds is not None and clock() - started >= timeout_seconds:
            return None
        sleep(poll_seconds)
