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

import dataclasses
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

import optuna
import yaml

from koopman_lm.config import KoopmanLMConfig
from experimentation.run.train_argv import batch_plans
from experimentation.run.write_policy import RunDirConflictError
from experimentation.sweep.launch import materialize_cell
from experimentation.sweep.search.metrics import (
    looks_like_oom, objective_from_metrics)
from experimentation.sweep.search.space import params_to_overrides
from experimentation.sweep.search.study import (
    ANCHOR_ATTR, REFERENCE_GROUP_ATTR, SEED_ATTR, to_distributions)
from experimentation.sweep.spec import build_cell_run_spec
from experimentation.run.spec import group_id, run_dir_path, run_id

__all__ = ["FATAL_EXCEPTIONS", "TRIAL_ATTRS", "FatalTrialError", "TrialOutcome",
           "objective_from_metrics", "run_trial", "drive"]


class FatalTrialError(Exception):
    """Raised when a fault will recur identically for EVERY remaining trial.

    The distinction `drive`'s handler cannot make on its own. A claimed run
    directory is this trial's problem; a data shard that does not exist is every
    trial's problem, and continuing spends 256 GPU-trials proving one fact.

    No `except` clause can tell those apart from an exception type or a message
    string, so the caller that knows says so by raising this. Everything not
    raised as this -- and not in `FATAL_EXCEPTIONS` -- is treated as the current
    trial's problem, which is the safe default: a study that fails every trial
    for one reason is diagnosable in one glance from `_print_failures`, while a
    study that stopped at trial 3 of 256 has wasted the allocation.
    """


#: Exceptions that are never a single trial's problem, so a per-trial handler must
#: not convert them into a FAILED trial and carry on.
#:
#: `KeyboardInterrupt` and `SystemExit` do not derive from `Exception`, so a bare
#: `except Exception` already misses them -- they are listed for the reader, and
#: because the handler catches `BaseException` in order to record a reason before
#: re-raising. `MemoryError` is the one that genuinely needs the allowlist: it
#: IS an `Exception`, and a supervisor process out of host memory will not
#: recover by starting another trial.
FATAL_EXCEPTIONS = (KeyboardInterrupt, SystemExit, MemoryError)

#: How many consecutive non-advancing attempts mean the journal, not the trial,
#: is broken. Small: the condition it detects is binary (states are being
#: recorded or they are not), so a large number only delays the diagnosis. Not 1,
#: because `study.ask` on a WAITING enqueued trial legitimately returns without
#: the count moving if that trial is then told by another worker.
_MAX_STALLED_ATTEMPTS = 3

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
               "anchor_name",
               # Which trial this was for THIS worker process, 0-based.
               #
               # It exists for one measured reason. Job 445689, four trials on the
               # proxy base at 600 steps: 17,818 / 111,416 / 104,760 / 112,685
               # tok/s. The 6x outlier was trial 0. Those four configs differ only
               # in `ska_layerscale_init` -- one scalar multiply on a gate -- so
               # nothing architectural can cause a 6x throughput gap. It paid
               # first-trial CUDA context creation and kernel autotuning, and
               # `tokens_per_sec` comes from the eval pass, which is early enough
               # in the process's life to still be inside that.
               #
               # At `concurrent_trials: 8` that is EIGHT poisoned points, not one:
               # every worker process pays its own warmup. So the number is
               # RECORDED here and `analysis.throughput_pareto` excludes ordinal 0
               # while reporting how many it excluded. Discarding it in the driver
               # would delete a measurement; excluding it silently in the analysis
               # would produce a front nobody could reconcile with the trial count.
               "worker_trial_ordinal",
               # The seed the trial ACTUALLY trained at, read off the resolved
               # spec. Recorded for every trial, seeded or not: reconstructing a
               # trial from trials.csv needs the seed that ran, and `run_id`
               # hashes it, so a reader without this column cannot check the
               # identity it is given. Distinct from `designated_seed`, which is
               # the REQUEST and is absent unless a design made one.
               "model_seed",
               # Which replicate set this trial belongs to, when it belongs to
               # one. This is the column the noise floor is computed over.
               "reference_group",
               # The ABSOLUTE norm clip that ran, not the multiplier the sampler
               # chose. Derivable from `norm_clip_multiplier` and `ska_rank`, and
               # derivable is not recorded -- see `_record_trial_attrs`.
               "ska_norm_clip_c",
               # The resolved SKA layer indices. `trial.params` records
               # (n_ska_layers, placement); nothing else records what that pair
               # resolved to, and the placement axis exists precisely because
               # the resolution is not the identity.
               "ska_layer_indices",
               # Stamped after the run finishes, not at materialization: they
               # come from quick_eval.json. Absent when the run produced none,
               # and `analysis` excludes rather than zero-fills such a trial.
               "ska_delta", "tokens_per_sec", "peak_memory_gib", "n_eval_tokens",
               # Operational facts for collision reuse and local-process
               # teardown. Neither changes scientific identity.
               # Set only when a trial COMPLETED with an objective but its
               # trainer exited nonzero. That combination no longer fails the
               # trial (the result was measured and written), so without a
               # column it would be invisible -- and a run of them is the first
               # sign that packing several trials onto one GPU is going wrong.
               "reused_run", "process_exit_code", "process_failure")


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


def _completed_run_metadata(run_dir: Path, spec, conflict: RunDirConflictError,
                            *, dirty: bool) -> Mapping[str, Any]:
    """Validate that a completed content collision is safe to score again.

    ATI's branch caught every ``RunDirConflictError`` and reused the directory.
    That also catches the deliberately explicit *different code_id* conflict,
    which would silently mix implementations in one Optuna study. Reuse here is
    narrower: same recorded commit, clean code on both sides, and equivalent
    scientific sections. Runtime details other than the seed may differ because
    they are deliberately unhashed (for example an OOM ladder's microbatch).
    """
    if dirty or not conflict.reusable:
        raise conflict
    spec_path = Path(run_dir) / "spec.yaml"
    try:
        raw = yaml.safe_load(spec_path.read_text()) or {}
    except Exception as exc:                              # noqa: BLE001
        raise RunDirConflictError(
            f"{conflict} Automatic reuse refused because {spec_path} is not "
            f"readable: {type(exc).__name__}: {exc}") from conflict
    if raw.get("dirty"):
        raise RunDirConflictError(
            f"{conflict} Automatic reuse refused because the completed run was "
            f"materialized from a dirty tree.") from conflict
    if raw.get("run_id") != run_id(spec) or raw.get("group_id") != group_id(spec):
        raise RunDirConflictError(
            f"{conflict} Automatic reuse refused because the directory's "
            f"recorded identity does not match the proposed RunSpec.") from conflict

    expected = {
        "model": dataclasses.asdict(spec.model),
        "data": dataclasses.asdict(spec.data),
        "optim": dataclasses.asdict(spec.optim),
        "seed": spec.runtime.seed,
        "schedules": spec.schedules or {},
    }
    actual = {
        "model": raw.get("model"),
        "data": raw.get("data"),
        "optim": raw.get("optim"),
        "seed": (raw.get("runtime") or {}).get("seed"),
        "schedules": raw.get("schedules") or {},
    }
    # JSON normalization makes YAML lists and dataclass tuples compare as the
    # same sequence while retaining every value and mapping key.
    def normalize(value):
        return json.dumps(value, sort_keys=True, default=str)

    if normalize(actual) != normalize(expected):
        raise RunDirConflictError(
            f"{conflict} Automatic reuse refused because the completed run's "
            f"scientific sections differ despite the short hash collision.") from conflict
    return raw


def _finish_launch(launcher, handle, run_dir) -> tuple[Optional[int], Optional[str]]:
    """Ask a launcher to release local resources; return provenance + failure."""
    waiter = getattr(launcher, "wait_for_exit", None)
    if handle is None or not callable(waiter):
        return None, None
    try:
        returncode = waiter(handle, run_dir)
    except Exception as exc:                              # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"
    if returncode is None:
        return None, None
    returncode = int(returncode)
    if returncode != 0:
        return returncode, f"local training process exited with code {returncode}"
    return returncode, None


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
              sampler_seed: Optional[int] = None,
              worker_trial_ordinal: Optional[int] = None) -> TrialOutcome:
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

    # The TRAINING seed, when -- and only when -- a design designated one.
    #
    # This is the whole mechanism behind the study's noise floor: a handful of
    # designated repeats of one reference configuration, differing in nothing but
    # `runtime.seed`, whose spread in held-out loss is the smallest difference
    # this study can resolve. Every other trial falls through this branch and
    # keeps the base spec's own seed, which is what keeps ORDINARY trials
    # comparable to each other -- a seed that moved per trial would fold the
    # noise floor into every measurement instead of isolating it.
    #
    # `runtime.seed` is inside `run/spec.py::_scientific_payload(include_seed=
    # True)` and outside `group_id`, so a repeat gets its own `run_id` and its own
    # directory while sharing a `group_id` with its siblings. That is the run
    # system's existing spelling for "one experiment, N datapoints" and is why
    # this needs no new identity machinery.
    #
    # NOT the sampler seed. `study.sampler_seed_for` is per-worker and seeds the
    # PROPOSAL stream; nothing here touches it, so which worker pulled a trial
    # still cannot change that trial's result.
    designated_seed = trial.user_attrs.get(SEED_ATTR)
    if designated_seed is not None:
        overrides["runtime.seed"] = int(designated_seed)

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
    launch_handle = None

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
        reused_completed_run = False
        reused_metadata: Mapping[str, Any] = {}
        try:
            run_dir = materialize_cell(spec, run_root, extra=stamp, dirty=dirty,
                                       force=force or position > 0,
                                       dry_run=dry_run)
        except RunDirConflictError as conflict:
            run_dir = run_dir_path(run_root, spec)
            reused_metadata = _completed_run_metadata(
                run_dir, spec, conflict, dirty=dirty)
            reused_completed_run = True
        actual_pdbs = ((reused_metadata.get("runtime") or {}).get(
            "per_device_batch_size", spec.runtime.per_device_batch_size)
            if reused_completed_run else spec.runtime.per_device_batch_size)
        identity = dict(trial_number=trial.number, run_id=run_id(spec),
                        group_id=group_id(spec), run_dir=run_dir, anchor=anchor,
                        per_device_batch_size=int(actual_pdbs))
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
            sampler_seed=sampler_seed, anchor=anchor,
            reference_group=trial.user_attrs.get(REFERENCE_GROUP_ATTR),
            worker_trial_ordinal=worker_trial_ordinal)
        if reused_completed_run:
            trial.set_user_attr("reused_run", True)
            # _record_trial_attrs saw the newly proposed runtime. If the earned
            # result fit via a lower OOM-ladder rung, provenance must name the
            # microbatch that actually produced it.
            trial.set_user_attr("per_device_batch_size", int(actual_pdbs))
        try:
            # wait=False so the objective reader can WATCH this run rather than
            # only inspect its corpse. With a blocking submit the reader starts
            # after training ended, so pruning had nothing to prune.
            if not reused_completed_run:
                launch_handle = launcher.submit(
                    spec, run_dir, dry_run=dry_run, wait=False)
            break
        except Exception as exc:                   # noqa: BLE001 -- see docstring
            last_failure = f"{type(exc).__name__}: {exc}"
            has_next_rung = position + 1 < len(rungs)
            if has_next_rung and looks_like_oom(run_dir):
                # Descending helps only for memory. Retrying a shape error at a
                # smaller microbatch burns another queue slot to fail identically.
                continue
            # An EXHAUSTED LADDER is said so, because the analysis has to be able
            # to tell four failure modes apart from `trials.csv` alone: a config
            # too big for the smallest microbatch, a config that OOMs once and
            # fits on the next rung, a shape error (which never descends -- see
            # above), and a launcher fault. All four used to arrive here with the
            # same shape of string, and `per_device_batch_size` alone cannot
            # distinguish them: rung 0 IS the only rung when the ladder is off.
            #
            # It is spelled on the LAST rung rather than in the `for ... else`
            # below because this branch is how the loop actually ends: the final
            # rung has no next rung, so it returns from inside the loop and the
            # `else` clause is only reachable if `_ladder` ever returns nothing.
            exhausted = len(rungs) > 1 and looks_like_oom(run_dir)
            reason = (f"every microbatch rung failed ({len(rungs)} rung(s), down "
                      f"to per_device_batch_size="
                      f"{spec.runtime.per_device_batch_size}): {last_failure}"
                      if exhausted else last_failure)
            trial.set_user_attr("failure", reason)
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
            return TrialOutcome(state="failed", objective=None, **identity)
    else:                                              # pragma: no cover
        # Only reachable if `_ladder` returns an empty list, which it cannot --
        # it returns `[None]` in every degenerate case. Kept as the honest answer
        # to "what if it did" rather than deleted, since deleting it would make
        # `identity` unbound below.
        trial.set_user_attr("failure",
                            f"no microbatch rung was attempted: {last_failure}")
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
    pruned = None
    try:
        objective = reader(run_dir)
    except optuna.TrialPruned as exc:
        pruned = exc
        objective = None
    finally:
        exit_code, process_failure = _finish_launch(
            launcher, launch_handle, run_dir)
        if exit_code is not None:
            trial.set_user_attr("process_exit_code", exit_code)

    if pruned is not None:
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
        #
        # Checked BEFORE `process_failure`, and the two are reported together.
        # `metrics.wait_for_objective` now cancels the run on timeout, so a
        # timed-out trial dies by SIGKILL and arrives here with a nonzero exit
        # code as well. Reporting only the exit code would relabel every timeout
        # as "exited with code -9" and erase the string that distinguishes a
        # missing objective from a crashed process -- which is how the 3M
        # study's 102 OOM failures were classified as a single cause at all.
        reason = "no objective could be read"
        if process_failure is not None:
            reason = f"{reason}; {process_failure}"
        trial.set_user_attr("failure", reason)
        study.tell(trial, state=optuna.trial.TrialState.FAIL)
        return TrialOutcome(state="failed", objective=None, **identity)
    if process_failure is not None:
        # An objective EXISTS and the process has been reaped. The GPU-safety
        # goal is already met by `_finish_launch` waiting -- the context is gone
        # before this function returns, whatever the exit code was. Discarding a
        # result that was measured and written would be a second, unrelated
        # policy, and it contradicts this module's own rule that a lost score
        # must not lose the training run. So this is recorded as provenance and
        # the trial completes.
        trial.set_user_attr("process_failure", process_failure)

    trial.set_user_attr("run_id", identity["run_id"])
    trial.set_user_attr("run_dir", str(run_dir))
    # The SKA ablation delta, stamped from the eval payload now that one exists.
    #
    # Found in review as an inert field: `analysis._top_by_ska_delta` reads
    # `ska_delta` off the trial and NOTHING wrote it, so `top_by_ska_delta.csv`
    # would have been header-only for every real study -- and its test could not
    # catch that, because `_write_csv` always writes a header and the assertion
    # was `st_size > 0`.
    #
    # It has to happen HERE rather than in `_record_trial_attrs`: the number
    # comes from `run_dir/eval/*/quick_eval.json`, which does not exist until the
    # run has finished, and materialization is long past by then.
    #
    # Why it is worth recording at all: it is the ranking that asks whether SKA
    # earned its place. A config with a good loss whose SKA branch contributes
    # nothing is a good Mamba model, not evidence for SKA -- and that is
    # invisible in a loss ranking.
    _stamp_measured_metrics(trial, run_dir)
    study.tell(trial, float(objective))
    return TrialOutcome(state="complete", objective=float(objective), **identity)


#: `quick_eval.json`'s `metrics.full` key -> the trial attr it becomes.
#:
#: All three were already MEASURED for every trial and none of them reached the
#: trial, so none reached `trials.csv`, so the loss/throughput Pareto front the
#: study's cost argument rests on could not be computed at all. The measurement
#: existed; the wire did not. Promoting them is the whole fix.
#:
#: `n_tokens` is renamed to `n_eval_tokens` on the way across, because `n_tokens`
#: is already the DATA section's field name (the shard's total token count, which
#: `verify_shard` checks) and one column meaning two things a few characters apart
#: is how a reader comes to divide by the wrong number.
_FULL_METRIC_ATTRS = (
    ("tokens_per_sec", "tokens_per_sec"),
    ("peak_memory_gib", "peak_memory_gib"),
    ("n_tokens", "n_eval_tokens"),
)


def _stamp_measured_metrics(trial, run_dir) -> None:
    """Copy the measured, non-objective numbers from the eval payload onto the
    trial: the SKA ablation delta, throughput, peak memory, eval token count.

    Best-effort and silent on absence, for one reason stated once: **absent is
    not zero.** The ablation is an optional part of quick_eval, and a trial
    recorded at 0 tokens/sec would sit at the wrong end of every cost ranking
    while looking exactly like a measurement. `analysis` excludes a trial with no
    value rather than zero-filling it, and that only works if this writes nothing
    when there is nothing to write.

    Deliberately does not fail a trial that trained fine. These are provenance
    for post-hoc rankings, not the objective -- raising here would turn a missing
    optional metric into a lost result. That is not hypothetical: `ska_delta` was
    found in review as an INERT field, written by nothing, so
    `top_by_ska_delta.csv` was header-only for every real study.

    Why here and not in `_record_trial_attrs`: every number below comes from
    `run_dir/eval/*/quick_eval.json`, which does not exist until the run has
    finished, and materialization is long past by then.
    """
    from experimentation.sweep.search.metrics import read_quick_eval_metrics

    try:
        metrics = read_quick_eval_metrics(run_dir) or {}
    except Exception:                                  # noqa: BLE001
        # A malformed eval file must not cost a completed trial its objective.
        return

    try:
        ablation = metrics.get("ska_ablation") or {}
        delta = ablation.get("loss_delta") if ablation.get("supported") else None
        if delta is not None:
            trial.set_user_attr("ska_delta", float(delta))

        full = metrics.get("full") or {}
        for key, attr in _FULL_METRIC_ATTRS:
            value = full.get(key)
            if value is None:
                continue
            trial.set_user_attr(attr, int(value) if attr == "n_eval_tokens"
                                else float(value))
    except (TypeError, ValueError, AttributeError):
        # A payload with a string where a number belongs. Same rule: the trial
        # trained and scored, so it keeps its objective.
        return


def _record_trial_attrs(trial, spec, base_model: KoopmanLMConfig, *,
                        worker_id: Optional[int],
                        sampler_name: Optional[str],
                        sampler_seed: Optional[int],
                        anchor: Optional[str],
                        reference_group: Optional[str] = None,
                        worker_trial_ordinal: Optional[int] = None) -> None:
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
        # Off the RESOLVED spec, not off the trial's request. `run_id` hashes
        # `runtime.seed`, so a reader handed a run_id and no seed column cannot
        # check the identity -- and a reference repeat is a trial whose only
        # distinguishing input IS this number.
        "model_seed": int(spec.runtime.seed),
        # What (n_ska_layers, placement) actually resolved to. The placement axis
        # exists because that resolution is not the identity, so recording only
        # the request records only half the experiment: `geometry.make_layer_
        # indices` CLAMPS a count past the usable window rather than raising, and
        # a clamped trial is indistinguishable from an unclamped one without this.
        "ska_layer_indices": [int(i) for i in spec.model.ska_layer_indices],
        # The ABSOLUTE norm clip. `params` records the MULTIPLIER and the model
        # runs at `multiplier * sqrt(rank)`, so the value that ran is derivable
        # from two recorded columns -- and derivable is not recorded. A reader has
        # to know the formula, and `params_to_overrides` ROUNDS the product to 8
        # decimal places, so a hand-recomputed value can differ in the last bits:
        # enough to make a run_id reconstruction fail for a reason that is not the
        # bug it looks like.
        "ska_norm_clip_c": (None if spec.model.ska_norm_clip_c is None
                            else float(spec.model.ska_norm_clip_c)),
    }
    if reference_group is not None:
        attrs[REFERENCE_GROUP_ATTR] = str(reference_group)
    if worker_trial_ordinal is not None:
        attrs["worker_trial_ordinal"] = int(worker_trial_ordinal)
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


def _fail_trial(study, trial, exc) -> TrialOutcome:
    """Record why a trial died and tell optuna, so it does not sit RUNNING.

    Both a summary and the traceback. The summary is what `_print_failures`
    groups -- N failures with ONE reason is a harness bug, N with N reasons is a
    rough study, and a count cannot tell them apart. The traceback is what makes
    the cause findable: this driver's own history includes a `TypeError` swallowed
    by the OOM ladder's broad `except` and surfacing as an unrelated assertion in
    22 tests.
    """
    import traceback as _traceback

    reason = f"{type(exc).__name__}: {exc}"
    try:
        trial.set_user_attr("failure", reason)
        trial.set_user_attr("traceback", _traceback.format_exc())
        study.tell(trial, state=optuna.trial.TrialState.FAIL)
    except Exception:                                  # noqa: BLE001
        # The storage itself is failing. Nothing useful left to do here, and
        # raising would mask the original exception.
        pass
    return TrialOutcome(trial_number=getattr(trial, "number", -1),
                        state="failed", run_id="", group_id="",
                        run_dir=Path("."), objective=None, anchor=None)


def drive(study: optuna.study.Study, *, n_trials: int, **kwargs) -> List[TrialOutcome]:
    """Run trials until the study holds `n_trials` finished ones.

    `n_trials` is the study's target size, not "this many more". Resuming a
    15-trial study that already finished 10 runs 5 -- which is the behaviour a
    human expects after a launcher crash, and the opposite of what "run 15" would
    do.

    **A raising trial is failed, not fatal.** `run_trial` guards the LAUNCH, but
    `params_to_overrides`, `build_cell_run_spec` and `materialize_cell` (which
    calls `verify_shard` and `claim_run_dir`) all run outside that try. So one
    claimed directory or one transient `/scratch` error used to raise straight out
    of here: the worker exited non-zero, its trial sat RUNNING in the journal
    forever, and the remaining hours of an 8-worker study ran 7-wide with nothing
    saying why. On a multi-hour study that is the likeliest way the whole thing
    quietly degrades.

    The needle this threads: catching everything turns a genuinely broken study
    into 256 identical failures, and catching nothing leaves the above. So a
    trial-specific fault is recorded and the loop continues, while
    `FATAL_EXCEPTIONS` and `FatalTrialError` still stop the worker -- see those
    for where the line is and why it cannot be drawn by inspecting a message.
    """
    distributions = to_distributions(kwargs["space"])
    outcomes: List[TrialOutcome] = []
    # A trial that neither COMPLETEs, PRUNEs nor FAILs does not advance
    # `_finished`, so the loop would ask for another one forever. That is not
    # hypothetical: `_fail_trial` swallows a storage error when telling FAIL (it
    # has to -- raising there would mask the original exception), so a journal
    # that has gone read-only turns this loop into a spin that burns the
    # allocation at 100% CPU and looks like progress. Found because a mutation
    # that removed the `tell` hung the test run instead of failing it.
    #
    # The guard is "did the last attempt move the counter", not a trial cap: the
    # counter is the thing the loop condition reads, so watching it is exact,
    # whereas a cap has to guess a margin.
    stalled = 0
    # How many trials THIS CALL has attempted. Stamped onto each trial as
    # `worker_trial_ordinal` so the analysis can tell a warm-up throughput
    # measurement from an architectural one -- job 445689 measured trial 0 at
    # 17,818 tok/s against ~110,000 for its siblings, from CUDA context creation
    # and kernel autotuning rather than from anything in the config. See
    # `TRIAL_ATTRS`.
    #
    # Per CALL, not per study, and that is the point: warmup is a property of the
    # PROCESS. A resumed worker starts at 0 again because the new process pays the
    # warmup again, and a study-wide counter would mark the resumed trial as warm
    # when it is the coldest one in the journal.
    ordinal = 0
    while _finished(study) < n_trials:
        before = _finished(study)
        trial = study.ask(distributions)
        try:
            outcomes.append(run_trial(study, trial,
                                      worker_trial_ordinal=ordinal, **kwargs))
        except FATAL_EXCEPTIONS:
            # Record, then re-raise: a worker that exits leaving a trial RUNNING
            # is the failure this whole handler exists to prevent, and that is
            # true whether it exits by choice or not.
            _fail_trial(study, trial, sys.exc_info()[1])
            raise
        except FatalTrialError as exc:
            _fail_trial(study, trial, exc)
            raise
        except Exception as exc:                       # noqa: BLE001
            # This trial's problem, as far as anything here can tell. Recorded,
            # told, and the worker moves to the next one.
            outcomes.append(_fail_trial(study, trial, exc))
        # Incremented for every ATTEMPT, including the failed ones, because the
        # warmup it tracks is paid by the process on its first attempt whether or
        # not that attempt produced a result.
        ordinal += 1
        if _finished(study) > before:
            stalled = 0
            continue
        stalled += 1
        if stalled >= _MAX_STALLED_ATTEMPTS:
            raise FatalTrialError(
                f"{stalled} consecutive trial(s) finished without advancing the "
                f"study's completed count ({before} of {n_trials}). The journal "
                f"is not recording terminal states -- most likely it has become "
                f"unwritable -- so this loop would spin forever asking for "
                f"trials that never finish. Stopping instead of burning the "
                f"allocation.")
    return outcomes
