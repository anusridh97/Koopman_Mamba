"""A study, written down: `configs/search/<name>.yaml` -> `StudySpec`.

Mirrors `sweep/spec.py`'s `SweepSpec`, for the same reason and with the same
discipline: a frozen dataclass, `__post_init__` validation, and a content hash
used for **provenance, not identity**.

**Why this file has to exist, given the optuna journal already records so much.**

Measured against optuna 4.9.0's `JournalStorage`: the journal records
`study_name`, `directions`, per-trial params **with their distributions**,
`values`, `intermediate_value` and `step` for every pruning report, `state`,
`user_attr`, timestamps and `worker_id`. So the search space and every trial's
full trajectory are already durable, and this spec deliberately does NOT
re-declare the space -- `space.py` owns that, once.

What the journal does not record is everything optuna never sees. From optuna's
side a trial is a black box from params to a float; it does not know a model is
being trained. So `max_steps`, which base spec, which shard, the launcher --
constants of *our* objective function -- are invisible to it by construction, not
by omission.

And one thing it does use but does not persist: **the sampler and pruner**. Those
are reconstructed by whoever opens the study, and reopening with a different
pruner is accepted silently. That is a live footgun, and pinning them in a
committed file is the fix.

**The sharper argument is multi-worker agreement.** Parallelism here is N
processes sharing one journal file (`study.py` rejects `optimize(n_jobs=N)`'s
thread pool for exactly this reason). If `max_steps` lives in a command-line
argument, worker 3 can silently disagree with worker 1 -- and optuna will happily
mix trials trained for 300 steps with trials trained for 3000 into one study,
because the journal records no step budget to notice the disagreement with. A
committed spec is what makes N workers *provably* the same study rather than
hopefully.

**Above the optuna line.** This module imports no optuna, so a study can be
authored, validated and diffed on a machine that has never installed it -- the
same property that lets `anchors.py` ship a curated design set as an ordinary
`cells:` sweep.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

__all__ = ["StudySpec", "load_study_spec", "study_id"]

# Only "minimize" is meaningful for a loss objective, and multi-objective is a
# distinct decision rather than a knob: optuna 4.9 raises NotImplementedError
# from Trial.report under multiple directions, so a Pareto study cannot prune at
# all. Since pruning is what makes a study affordable, the recommended route is
# to search single-objective and compute the front post hoc from trials.csv,
# which already records every metric per trial.
_DIRECTIONS = ("minimize", "maximize")

_LAUNCHERS = ("slurm", "local")


@dataclass(frozen=True)
class StudySpec:
    """A parsed configs/search/<name>.yaml. Pure data; touches no filesystem."""

    name: str
    #: Path to the base run spec every trial is a perturbation of. Left as the
    #: literal string from the YAML and resolved later against the process cwd,
    #: matching `python -m experimentation.run <spec.yaml>`'s own convention.
    base: str
    #: The study's TARGET size, not "this many more" -- resuming a 15-trial study
    #: that finished 10 runs 5. Matches drive()'s own semantics.
    n_trials: int
    #: Steps per trial. The single most important field to pin: it is invisible
    #: to the journal, and two workers disagreeing about it produce trials that
    #: are not comparable while looking like one study.
    max_steps: int
    run_root: str = "runs"
    launcher: str = "slurm"
    direction: str = "minimize"
    #: Where the shared journal lives. In the spec rather than on the command
    #: line so two workers cannot open two different journals and believe they
    #: are collaborating.
    storage: Optional[str] = None
    seed: int = 2026
    #: A hint that a fleet exists, not a fleet. It only flips the sampler's
    #: constant_liar on; launching N workers is the operator's job.
    n_jobs: int = 1
    prune_after_step: int = 200
    logging_steps: int = 10
    #: Optional anchor designs to enqueue before adaptive sampling starts.
    design_file: Optional[str] = None
    #: How long to wait for a trial's objective before giving up on it.
    #:
    #: Load-bearing once the driver stops blocking on submit. With a blocking
    #: submit, a crashed trial surfaced as a non-zero exit from subprocess.run;
    #: without one, it surfaces only as "the objective never appeared". No
    #: timeout means a single dead trial hangs the whole study indefinitely.
    #:
    #: 2 hours by default -- generous next to the minutes a small trial takes, and
    #: still far short of a night.
    trial_timeout_seconds: float = 7200.0
    #: Held-out shard for a trial's own end-of-run scoring. Defaults to the base
    #: spec's training shard, which is fine for a proxy objective but means the
    #: score is not held out -- set it for anything whose ranking you trust.
    eval_data_dir: Optional[str] = None
    backend_policy: str = "exact_invchol"
    seq_len: Optional[int] = None
    batch_ladder: bool = False
    #: Objective weights, passed through to metrics.objective_from_metrics.
    #: Every one defaults to zero there, so an empty dict means "minimise the
    #: measured loss and nothing else".
    objective: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.name:
            raise ValueError("StudySpec.name is required")
        if not self.base:
            raise ValueError(
                "StudySpec.base is required -- a study is a perturbation of some "
                "base run spec, and without one there is nothing to perturb")
        if self.n_trials < 1:
            raise ValueError(f"n_trials must be >= 1, got {self.n_trials}")
        if self.max_steps < 1:
            raise ValueError(f"max_steps must be >= 1, got {self.max_steps}")
        if self.direction not in _DIRECTIONS:
            raise ValueError(
                f"direction={self.direction!r}; expected one of {list(_DIRECTIONS)}. "
                "Multiple objectives are not a list here on purpose: optuna 4.9 "
                "cannot prune a multi-objective study at all, so a Pareto front "
                "is computed post hoc from trials.csv instead.")
        if self.launcher not in _LAUNCHERS:
            raise ValueError(
                f"launcher={self.launcher!r}; expected one of {list(_LAUNCHERS)}")
        if self.n_jobs < 1:
            raise ValueError(f"n_jobs must be >= 1, got {self.n_jobs}")
        if self.prune_after_step < 0:
            raise ValueError("prune_after_step must be >= 0")
        if self.logging_steps < 1:
            raise ValueError(
                "logging_steps must be >= 1 -- it becomes the pruner's "
                "interval_steps, which optuna requires to be positive")
        if self.trial_timeout_seconds <= 0:
            raise ValueError(
                "trial_timeout_seconds must be positive -- without a timeout one "
                "dead trial hangs the study forever, and the driver no longer "
                "blocks on submit, so a crash has no other way to surface")
        if self.prune_after_step >= self.max_steps:
            raise ValueError(
                f"prune_after_step={self.prune_after_step} >= max_steps="
                f"{self.max_steps}: no trial could ever be pruned, so every "
                "trial would run to completion while appearing to be pruned-"
                "enabled. Lower it, or say so by setting it to 0.")


def load_study_spec(path) -> StudySpec:
    """Read configs/search/<name>.yaml into a StudySpec.

    Unknown keys are a hard error rather than a shrug: a study is expensive, and
    a misspelled `max_step` silently taking the default would produce a whole
    study at the wrong budget.
    """
    raw = yaml.safe_load(Path(path).read_text()) or {}
    known = {f for f in StudySpec.__dataclass_fields__}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(
            f"{path}: unknown key(s) {unknown}. Known keys: {sorted(known)}")
    missing = [k for k in ("name", "base", "n_trials", "max_steps") if k not in raw]
    if missing:
        raise ValueError(f"{path}: missing required key(s) {missing}")
    return StudySpec(**raw)


def study_id(study_spec: StudySpec) -> str:
    """A content hash of the study's own declaration.

    Stamped into each trial's materialized spec.yaml alongside `study_name`, so
    results can be grouped by which study produced them. Deliberately NOT part
    of run_id or group_id, for the reason `sweep/spec.py::sweep_id` gives: study
    membership is metadata about how a run was launched, not a scientific input.
    Two studies that happen to propose the same config must produce the same
    run_id, or the content-addressed run directory stops being content-addressed.
    """
    blob = json.dumps(asdict(study_spec), sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:8]
