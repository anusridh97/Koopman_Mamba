"""ask -> materialize -> launch -> read -> tell. The loop, without a closure.

The original harness wrapped training in an `objective(trial)` that optuna
called, which forced one long-lived process holding a child process, a GPU thread
pool and a stdout pipe -- and therefore forced abandoning Slurm. optuna's
ask/tell interface removes that constraint: the gap between `ask()` and `tell()`
can be a scheduler queue, so this drives the *existing* run system instead of
replacing it.

    trial   = study.ask(distributions)          # params, or a queued anchor
    spec    = build_cell_run_spec(...)          # sweep/spec.py, unmodified
    run_dir = materialize_cell(spec, ...)       # sweep/launch.py, unmodified
    launcher.submit(spec, run_dir)              # run/launchers.py, unmodified
    study.tell(trial, objective)

Everything the run system guarantees comes along for free: a content-hashed
run_id and group_id, the refusal to launch from a dirty tree, verify_shard
against the data's own meta.json, atomic materialization, the claim against
concurrent writers, an attempts.jsonl audit trail, and -- through
`results.py` -- a queryable table at the end.

**The launcher and the objective reader are injected.** Not for testability
alone: `LocalLauncher`/`SlurmLauncher` are already passed as objects here, and
the reader genuinely has to differ by launcher, because a local run can be read
the moment `submit()` returns while a queued one cannot be read for hours. That
same seam is what lets the whole loop be exercised on a CPU box.

**Scalarisation is one pure function.** `objective_from_metrics` turns a
quick_eval payload into the single number optuna minimises, and every penalty
weight defaults to zero -- so the objective is the measured held-out loss until
someone deliberately trades it against parameter count, throughput, or how much
the SKA branch earns. The parameter penalty is worth turning on: the default
space spans a ~5.9% parameter range, and without it a config can win on capacity
rather than on architecture, which is not a claim a paper can make.

**A failed trial is told, not raised.** One OOM must not end a study, and a
missing metric must never be replaced with a plausible number -- a fabricated
objective would steer every later proposal.

**Pruning arrives through `objective_reader_for`.** A reader that polls a running job
raises `optuna.TrialPruned` when the pruner says stop; this records the trial as
PRUNED and moves on. Nothing else here changes, which is the payoff of having made
the reader injectable in the first place -- see
`experimentation.sweep.search.metrics.wait_for_objective`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

import optuna

from koopman_lm.config import KoopmanLMConfig
from experimentation.run.train_argv import batch_plans
from experimentation.sweep.launch import materialize_cell
from experimentation.sweep.search.metrics import (
    looks_like_oom, objective_from_metrics)
from experimentation.sweep.search.space import params_to_overrides
from experimentation.sweep.search.study import ANCHOR_ATTR, to_distributions
from experimentation.sweep.spec import build_cell_run_spec
from experimentation.run.spec import group_id, run_id

__all__ = ["TRIAL_ATTRS", "TrialOutcome", "objective_from_metrics", "run_trial",
           "drive"]

#: The user-attr keys a trial records at MATERIALIZATION time -- before it is
#: launched, so they survive a failure and a pruning.
#:
#: All of these end up in `trials.csv` as `attr_<key>` columns (report.trial_row
#: flattens `user_attrs` wholesale), which is the only channel a post-hoc
#: analysis has: `trial.params` records what the sampler CHOSE, and nothing else
#: records what that choice resolved to. Without `param_count` there is no
#: loss-versus-size Pareto front; without `worker_id` and `sampler_seed` a
#: concurrency artefact cannot be told from a real effect; without
#: `per_device_batch_size` a trial that descended the OOM ladder is
#: indistinguishable from one that did not.
TRIAL_ATTRS = ("param_count", "baseline_param_count", "worker_id",
               "sampler", "sampler_seed", "per_device_batch_size",
               "anchor_name")


@dataclass(frozen=True)
class TrialOutcome:
    """What happened to one trial, in terms a human can act on."""
    trial_number: int
    state: str                      # "complete" | "pruned" | "failed"
    run_id: str
    group_id: str
    run_dir: Path
    objective: Optional[float]
    anchor: Optional[str]
    # Which rung of the OOM ladder actually ran. Still recorded, though the
    # reason has changed: per_device_batch_size moved to RuntimeSpec on
    # 2026-08-19, so it is no longer hashed and every rung shares one run_id. The
    # number is now provenance -- which microbatch the result was produced at --
    # rather than the only way to find the directory.
    per_device_batch_size: Optional[int] = None


def run_trial(study: optuna.study.Study, trial, *,
              base_sections: Mapping[str, Mapping[str, Any]],
              base_model: KoopmanLMConfig,
              space: Mapping[str, Mapping[str, Any]],
              max_steps: int,
              run_root,
              study_name: str,
              launcher,
              objective_reader_for: Callable[
                  [Any, Any], Callable[[Path], Optional[float]]],
              base_lr: float = 4e-4,
              backend_policy: str = "exact_invchol",
              seq_len: Optional[int] = None,
              dry_run: bool = False,
              force: bool = False,
              dirty: bool = False,
              batch_ladder: bool = False,
              worker_id: Optional[int] = None,
              sampler_name: Optional[str] = None,
              sampler_seed: Optional[int] = None) -> TrialOutcome:
    """Turn one asked-for trial into a launched run, and report the result back.

    `base_lr` is accepted and unused here -- the sampler has already produced a
    concrete learning rate by the time a trial exists. It stays in the signature
    so `drive` can pass one context dict to both this and `enqueue_anchors`.
    """
    anchor = trial.user_attrs.get(ANCHOR_ATTR)
    overrides = params_to_overrides(trial.params, base_model, max_steps=max_steps,
                                   seq_len=seq_len, backend_policy=backend_policy)

    # The trial's BUDGET, applied here rather than in params_to_overrides.
    #
    # space.py deliberately does not own run length -- its docstring says so, and
    # it is right: the space samples architecture and optimizer hyperparameters,
    # not how long to train. It takes max_steps only because warmup_ratio is
    # meaningless without a run length to be a ratio OF.
    #
    # But nothing then applied it, so a trial inherited the BASE spec's max_steps
    # and the study's declared budget silently scaled warmup alone. Measured on a
    # real study (job 439754): a spec asking for 200-step trials trained 400,
    # because configs/runs/4m-golden.yaml says 400. Against
    # configs/runs/50m-fineweb-3b.yaml, whose max_steps is 15000, a 12-trial study
    # declaring 600 steps would have run 180,000 steps instead of 7,200 -- a 25x
    # overspend, with nothing in the output saying so.
    #
    # It also broke the schedule twice over: warmup was computed against 200 while
    # the run was 400, so the LR ramp was half the intended fraction, and
    # StudySpec's prune_after_step < max_steps validation was checking against a
    # number the run did not use.
    #
    # So the driver applies it: the driver is what knows the study's budget, while
    # the space stays pure.
    overrides["optim.max_steps"] = int(max_steps)

    # Reuse sweep's stamp keys so results.py's existing sweep_name column stays
    # meaningful, and add the study-specific pair beside them.
    stamp = {
        "sweep_name": study_name,
        "study_name": study_name,
        "trial_number": trial.number,
    }
    if anchor:
        stamp["anchor_name"] = anchor

    rungs = _ladder(base_sections, enabled=batch_ladder)
    identity: Dict[str, Any] = {}
    last_failure = "no attempt was made"

    for position, pdbs in enumerate(rungs):
        attempt_overrides = dict(overrides)
        if pdbs is not None:
            attempt_overrides["runtime.per_device_batch_size"] = pdbs
        spec = build_cell_run_spec(study_name, base_sections, attempt_overrides,
                                  schedules=base_sections.get("schedules"))
        # Since the microbatch moved to RuntimeSpec every rung resolves to the SAME
        # run_dir, which is the point: a retry is the same experiment fitted into
        # memory differently, so it reuses the directory and appends another
        # attempts.jsonl record rather than forking a second identity. force=True
        # is what lets the second rung write into a directory the first already
        # created.
        run_dir = materialize_cell(spec, run_root, extra=stamp, dirty=dirty,
                                   force=force or position > 0, dry_run=dry_run)
        identity = dict(trial_number=trial.number, run_id=run_id(spec),
                        group_id=group_id(spec), run_dir=run_dir, anchor=anchor,
                        per_device_batch_size=spec.runtime.per_device_batch_size)
        # Recorded HERE -- after the spec exists, before the launch can fail --
        # so a FAILED or PRUNED trial still carries its resolved parameter count
        # and its provenance. Recording it after the objective arrives would lose
        # exactly the trials an underperforming study most needs to explain.
        #
        # Re-set on every rung on purpose: `per_device_batch_size` is the one
        # value that changes when the OOM ladder descends, and the attr has to
        # say which microbatch actually ran rather than which one was tried first.
        _record_trial_attrs(
            trial, spec, base_model,
            worker_id=worker_id, sampler_name=sampler_name,
            sampler_seed=sampler_seed, anchor=anchor)
        try:
            # wait=False so the objective reader can WATCH this run rather than
            # only inspect its corpse. With a blocking submit the reader starts
            # after training ended, so pruning had nothing to prune.
            launcher.submit(spec, run_dir, dry_run=dry_run, wait=False)
            break
        except Exception as exc:                   # noqa: BLE001 -- see docstring
            last_failure = f"{type(exc).__name__}: {exc}"
            has_next_rung = position + 1 < len(rungs)
            if has_next_rung and looks_like_oom(run_dir):
                # Descending helps only for memory. Retrying a shape error at a
                # smaller microbatch burns another queue slot to fail identically.
                continue
            trial.set_user_attr("failure", last_failure)
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
            return TrialOutcome(state="failed", objective=None, **identity)
    else:
        trial.set_user_attr("failure", f"every microbatch rung failed: {last_failure}")
        study.tell(trial, state=optuna.trial.TrialState.FAIL)
        return TrialOutcome(state="failed", objective=None, **identity)

    run_dir = identity["run_dir"]
    # ONE scoring parameter, of the factory shape (study, trial) -> (run_dir) ->
    # float|None. A pure "score a finished run" reader is expressed by ignoring
    # both arguments -- `metrics.fixed_reader` is that adapter.
    #
    # Why the factory is the general shape and not the pure reader: pruning IS
    # trial.report() followed by trial.should_prune(), so a reader that polls a
    # running job must see the trial it is scoring. `drive` builds one **kwargs
    # dict before any trial exists, so a fixed reader could never prune -- which
    # is exactly the bug this seam was added to close, where pruning was
    # unreachable through drive() despite the docstring claiming otherwise.
    #
    # The rejected alternatives, unchanged: widening the pure reader to
    # (run_dir, study, trial) forces read_quick_eval_objective to accept two
    # arguments it ignores and drags optuna into metrics.py's module scope, where
    # the import is lazy on purpose; and sniffing with inspect.signature fails
    # silently the moment a signature drifts, which is the failure mode this file
    # is full of guards against.
    #
    # Previously this was TWO optional parameters resolved by a ternary, with a
    # hand-written TypeError for the neither-given case and a silent tie-break
    # for both-given. Four states where there is one. A required keyword argument
    # gets Python's own error for free.
    reader = objective_reader_for(study, trial)
    try:
        objective = reader(run_dir)
    except optuna.TrialPruned as pruned:
        # metrics.wait_for_objective raises this after cancelling the job. It
        # arrives through the objective_reader_for seam rather than a driver
        # flag, which is why pruning needed no structural change here.
        trial.set_user_attr("pruned", str(pruned))
        study.tell(trial, state=optuna.trial.TrialState.PRUNED)
        return TrialOutcome(state="pruned", objective=None, **identity)
    if objective is None:
        # Training can exit cleanly and still leave no metric -- a missing
        # checkpoint, a preemption between the save and the eval. Telling optuna
        # a stand-in number would steer every later proposal off a fiction.
        trial.set_user_attr("failure", "no objective could be read")
        study.tell(trial, state=optuna.trial.TrialState.FAIL)
        return TrialOutcome(state="failed", objective=None, **identity)

    trial.set_user_attr("run_id", identity["run_id"])
    trial.set_user_attr("run_dir", str(run_dir))
    study.tell(trial, float(objective))
    return TrialOutcome(state="complete", objective=float(objective), **identity)


def _record_trial_attrs(trial, spec, base_model: KoopmanLMConfig, *,
                        worker_id: Optional[int],
                        sampler_name: Optional[str],
                        sampler_seed: Optional[int],
                        anchor: Optional[str]) -> None:
    """Stamp the resolved facts about this trial onto the trial itself.

    Not onto the run directory. Both would be defensible and the trial is the
    right one: `report.trial_row` flattens `user_attrs` into `trials.csv`, so a
    post-hoc analysis gets these for free and for EVERY trial, including the
    failed and pruned ones whose run directories may not have a readable spec.

    `param_count` is the estimate, not a measurement -- nothing here builds a
    model. `config.param_count_estimate()` is arithmetic over the config,
    reconciled against a real GPU instantiation (build 415208: 50,034,044
    measured against 50,765,216 for the pre-fix formula, which is why the formula
    was rewritten), so it is exact for the layout it models rather than
    approximate.
    """
    attrs = {
        "param_count": int(spec.model.param_count_estimate()),
        "baseline_param_count": int(base_model.param_count_estimate()),
        "per_device_batch_size": int(spec.runtime.per_device_batch_size),
    }
    if worker_id is not None:
        attrs["worker_id"] = int(worker_id)
    if sampler_name is not None:
        attrs["sampler"] = str(sampler_name)
    if sampler_seed is not None:
        attrs["sampler_seed"] = int(sampler_seed)
    # ANCHOR_ATTR is already set by enqueue_anchors; re-stating it would be a
    # write with no new information. It is listed in TRIAL_ATTRS because a
    # consumer reading that tuple needs to know the column exists.
    for key, value in attrs.items():
        trial.set_user_attr(key, value)


def _ladder(base_sections: Mapping[str, Mapping[str, Any]], *,
            enabled: bool) -> list:
    """The microbatch rungs to attempt, or a single no-override attempt.

    Opt-in: a lone hand-launched run should fail loudly on an OOM rather than
    quietly consume four more queue slots discovering the same thing.
    """
    if not enabled:
        return [None]
    optim = base_sections.get("optim", {})
    runtime = base_sections.get("runtime", {})
    effective = int(optim.get("effective_batch", 0) or 0)
    # runtime since 2026-08-19 -- which is what makes the ladder
    # identity-preserving instead of identity-forking.
    initial = int(runtime.get("per_device_batch_size", 8) or 0)
    if effective < 1 or initial < 1:
        return [None]
    return [pdbs for pdbs, _ in batch_plans(effective, initial)]


def _finished(study: optuna.study.Study) -> int:
    terminal = {optuna.trial.TrialState.COMPLETE,
                optuna.trial.TrialState.PRUNED,
                optuna.trial.TrialState.FAIL}
    return sum(1 for t in study.trials if t.state in terminal)


def drive(study: optuna.study.Study, *, n_trials: int, **kwargs) -> List[TrialOutcome]:
    """Run trials until the study holds `n_trials` finished ones.

    `n_trials` is the study's target size, not "this many more". Resuming a
    15-trial study that already finished 10 runs 5 -- which is the behaviour a
    human expects after a launcher crash, and the opposite of what "run 15" would
    do.
    """
    distributions = to_distributions(kwargs["space"])
    outcomes: List[TrialOutcome] = []
    while _finished(study) < n_trials:
        trial = study.ask(distributions)
        outcomes.append(run_trial(study, trial, **kwargs))
    return outcomes
