"""One bad trial must not kill a worker for the remaining hours.

`run_trial` guards the LAUNCH -- `launcher.submit` sits in a try/except with the
OOM ladder, and the reader's try catches `TrialPruned`. Everything before the
launch does not: `params_to_overrides`, `build_cell_run_spec`, and
`materialize_cell` (which calls `verify_shard` and `claim_run_dir`) all run
outside it.

So on an 8-worker, multi-hour study, one claimed run directory or one transient
`/scratch` hiccup raised out of `drive`, exited that worker non-zero, and left
its trial **RUNNING in the journal forever**. The remaining budget was then run
by 7 workers, and `len(study.trials)` could exceed the documented overshoot
bound while `_finished` could not -- so the sbatch readback printed a trial count
above the expected range with nothing explaining it.

The fix has to thread a needle. Catching everything and continuing turns a
genuinely broken study into 256 identical failures and burns the whole budget
discovering one fact. Catching nothing leaves the failure above. So:

  * a TRANSIENT or TRIAL-SPECIFIC fault -- a claimed directory, an OOM the ladder
    could not escape, one unreadable spec -- is recorded on the trial, told to
    optuna as FAIL, and the worker moves on;
  * a FATAL fault -- one that will recur identically for every trial -- re-raises
    and stops the worker.

The line is drawn by asking "would the next trial hit this too?". Because that is
a judgement no `except` clause can make from a message string, it is encoded two
ways rather than guessed: an explicit `FATAL_EXCEPTIONS` allowlist for things
that are never a single trial's problem (KeyboardInterrupt, SystemExit,
MemoryError), and an explicit `FatalTrialError` that the caller raises when it
knows the fault is structural. Everything else is treated as this trial's
problem, which is the safe default -- a study that fails every trial for one
reason is diagnosable in one glance, while a study that stopped at trial 3 of 256
has wasted the allocation.

Nothing is swallowed silently: every caught fault lands in
`trial.user_attrs["failure"]` (grouped by `_print_failures`) and its traceback in
`trial.user_attrs["traceback"]`.
"""
from __future__ import annotations

import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

optuna = pytest.importorskip("optuna")

pytestmark = pytest.mark.correctness

from experimentation.sweep.search.driver import (                  # noqa: E402
    FATAL_EXCEPTIONS, FatalTrialError, drive)


class _Trial:
    def __init__(self, number):
        self.number = number
        self.params = {}
        self.user_attrs = {}
        self.state = optuna.trial.TrialState.RUNNING

    def set_user_attr(self, key, value):
        self.user_attrs[key] = value


class _Study:
    def __init__(self):
        self.trials = []
        self._next = 0

    def ask(self, distributions):
        trial = _Trial(self._next)
        self._next += 1
        self.trials.append(trial)
        return trial

    def tell(self, trial, value=None, state=None):
        trial.state = state or optuna.trial.TrialState.COMPLETE


def _ok(trial):
    """What a successful `run_trial` returns. A bare `object()` would not have
    `.state`, and the outcomes list mixes these with `_fail_trial`'s."""
    from experimentation.sweep.search.driver import TrialOutcome

    return TrialOutcome(trial_number=trial.number, state="complete",
                        run_id="deadbeef", group_id="cafebabe",
                        run_dir=pathlib.Path("."), objective=1.0, anchor=None)


def _drive_with(runner, n_trials=6):
    from experimentation.sweep.search import driver as driver_mod

    study = _Study()
    original = driver_mod.run_trial
    driver_mod.run_trial = runner
    try:
        outcomes = drive(study, n_trials=n_trials, space={})
    finally:
        driver_mod.run_trial = original
    return study, outcomes


def _states(study):
    tally = {}
    for trial in study.trials:
        tally[trial.state.name] = tally.get(trial.state.name, 0) + 1
    return tally


# ---------------------------------------------- a transient fault is survived ----

def test_one_raising_trial_does_not_stop_the_worker():
    calls = {"n": 0}

    def runner(study, trial, **kwargs):
        calls["n"] += 1
        if trial.number == 2:
            raise RuntimeError("run_dir already claimed by another writer")
        study.tell(trial, 1.0)
        return _ok(trial)

    study, _ = _drive_with(runner, n_trials=6)
    # 5 COMPLETE + 1 FAIL = 6 finished. A FAIL counts toward `_finished` -- that
    # is the existing contract and the reason a fully-broken study terminates
    # instead of looping forever, so the budget buys 6 ATTEMPTS, not 6 successes.
    assert _states(study) == {"COMPLETE": 5, "FAIL": 1}, _states(study)
    assert calls["n"] == 6
    assert study.trials[2].state == optuna.trial.TrialState.FAIL, (
        "the worker did not continue past the bad trial")
    assert study.trials[5].state == optuna.trial.TrialState.COMPLETE


def test_the_failed_trial_is_told_rather_than_left_running():
    """A trial left RUNNING is never cleaned up: it sits in the journal forever,
    and `len(study.trials)` then exceeds the overshoot bound `_finished`
    respects, with nothing in the output explaining the discrepancy."""
    def runner(study, trial, **kwargs):
        if trial.number == 0:
            raise OSError("transient /scratch error")
        study.tell(trial, 1.0)
        return _ok(trial)

    study, _ = _drive_with(runner, n_trials=3)
    assert not any(t.state == optuna.trial.TrialState.RUNNING
                   for t in study.trials), "a trial was abandoned RUNNING"


def test_the_reason_is_recorded_on_the_trial_not_swallowed():
    """`_print_failures` groups these. 48 trials failing for ONE reason is a bug
    in the harness; 48 for 48 reasons is a rough study, and a count cannot tell
    them apart."""
    def runner(study, trial, **kwargs):
        if trial.number == 0:
            raise RuntimeError("run_dir already claimed")
        study.tell(trial, 1.0)
        return _ok(trial)

    study, _ = _drive_with(runner, n_trials=2)
    failed = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
    assert len(failed) == 1
    reason = failed[0].user_attrs["failure"]
    assert "RuntimeError" in reason and "already claimed" in reason


def test_the_traceback_is_recorded_so_the_cause_is_recoverable():
    """A type and a message are not always enough to find WHERE. This driver's own
    history includes a TypeError swallowed by the OOM ladder's broad except and
    surfacing as an unrelated assertion in 22 tests."""
    def runner(study, trial, **kwargs):
        raise RuntimeError("boom")

    study, _ = _drive_with(runner, n_trials=1)
    trial = study.trials[0]
    assert "traceback" in trial.user_attrs
    assert "boom" in trial.user_attrs["traceback"]


def test_a_failed_trial_appears_in_the_returned_outcomes():
    def runner(study, trial, **kwargs):
        if trial.number == 0:
            raise RuntimeError("nope")
        study.tell(trial, 1.0)
        return _ok(trial)

    _, outcomes = _drive_with(runner, n_trials=2)
    assert [o.state for o in outcomes].count("failed") == 1


def test_every_trial_failing_still_terminates():
    """The loop must not spin forever when nothing can succeed -- a FAIL counts
    toward `_finished`, so the budget is spent and the worker exits."""
    def runner(study, trial, **kwargs):
        raise RuntimeError("always")

    study, outcomes = _drive_with(runner, n_trials=5)
    assert len(outcomes) == 5
    assert _states(study) == {"FAIL": 5}


def test_a_systemic_fault_is_visible_as_one_grouped_reason():
    """Not swallowed: the whole point of recording the reason is that N failures
    with ONE reason is diagnosable at a glance."""
    def runner(study, trial, **kwargs):
        raise RuntimeError("CUDA out of memory")

    study, _ = _drive_with(runner, n_trials=4)
    reasons = {t.user_attrs["failure"] for t in study.trials}
    assert len(reasons) == 1, reasons


# ------------------------------------------------- a fatal fault still stops ----

def test_the_fatal_allowlist_is_not_empty():
    """Guards the guard: an empty allowlist makes every test below vacuous and
    turns a broken study into 256 identical failures."""
    assert FATAL_EXCEPTIONS
    assert KeyboardInterrupt in FATAL_EXCEPTIONS


@pytest.mark.parametrize("exc", [KeyboardInterrupt, SystemExit, MemoryError])
def test_an_interrupt_or_exit_is_never_converted_into_a_failed_trial(exc):
    """Ctrl-C must stop the worker, not fail one trial and carry on for hours.
    `SystemExit` and `KeyboardInterrupt` do not derive from `Exception`, so a bare
    `except Exception` would already miss them -- `MemoryError` does, and is the
    one that needs the allowlist."""
    def runner(study, trial, **kwargs):
        raise exc("stop")

    with pytest.raises(exc):
        _drive_with(runner, n_trials=3)


def test_a_fault_that_will_recur_for_every_trial_stops_the_worker():
    """The budget-burning case. If the base spec cannot be resolved or the shard
    is missing, trial 2 fails identically to trial 1, so continuing spends the
    whole study proving one fact."""
    def runner(study, trial, **kwargs):
        raise FatalTrialError("data shard /scratch/nope does not exist")

    with pytest.raises(FatalTrialError):
        _drive_with(runner, n_trials=8)


def test_a_fatal_fault_still_records_why_before_re_raising():
    """Stopping is right; stopping with no record is not -- the journal should say
    what happened to the trial that was in flight, or the operator sees a worker
    that exited non-zero and a trial stuck RUNNING."""
    from experimentation.sweep.search import driver as driver_mod

    def runner(study, trial, **kwargs):
        raise FatalTrialError("shard missing")

    study = _Study()
    original = driver_mod.run_trial
    driver_mod.run_trial = runner
    try:
        with pytest.raises(FatalTrialError):
            drive(study, n_trials=4, space={})
    finally:
        driver_mod.run_trial = original
    assert study.trials[0].state == optuna.trial.TrialState.FAIL
    assert "shard missing" in study.trials[0].user_attrs["failure"]
    assert len(study.trials) == 1, "it kept going after a fatal fault"


def test_a_fatal_error_is_not_in_the_allowlist_but_still_propagates():
    """Two mechanisms, deliberately. The allowlist covers things that are never
    one trial's problem; `FatalTrialError` is how a CALLER says "this will recur",
    which no exception type could express."""
    assert FatalTrialError not in FATAL_EXCEPTIONS


# ------------------------------------------------------- the happy path holds ----

def test_a_clean_study_is_unaffected():
    """The wrapper must not change the normal case."""
    def runner(study, trial, **kwargs):
        study.tell(trial, 1.0)
        return _ok(trial)

    study, outcomes = _drive_with(runner, n_trials=5)
    assert len(outcomes) == 5
    assert _states(study) == {"COMPLETE": 5}
    assert not any("failure" in t.user_attrs for t in study.trials)


def test_resume_still_counts_what_already_finished():
    """`n_trials` is the study's TARGET size. The wrapper must not break that."""
    from experimentation.sweep.search import driver as driver_mod

    study = _Study()
    for _ in range(3):
        study.tell(study.ask({}), 1.0)

    def runner(study_, trial, **kwargs):
        study_.tell(trial, 1.0)
        return _ok(trial)

    original = driver_mod.run_trial
    driver_mod.run_trial = runner
    try:
        outcomes = drive(study, n_trials=5, space={})
    finally:
        driver_mod.run_trial = original
    assert len(outcomes) == 2


# ---------------------------------------- a non-advancing loop must not spin ----

def test_a_journal_that_records_nothing_stops_instead_of_spinning():
    """The bug the mutation run exposed by HANGING rather than failing.

    `_fail_trial` deliberately swallows a storage error when telling FAIL --
    raising there would mask the original exception. So if the journal has gone
    unwritable, no trial ever reaches a terminal state, `_finished` never
    advances, and this loop asks for trials forever: 100% CPU, no progress, and
    it looks like work. On an 8-GPU allocation that is the whole booking wasted.
    """
    class _DeafStudy(_Study):
        def tell(self, trial, value=None, state=None):
            pass                       # the journal accepts nothing

    from experimentation.sweep.search import driver as driver_mod

    def runner(study, trial, **kwargs):
        return _ok(trial)

    study = _DeafStudy()
    original = driver_mod.run_trial
    driver_mod.run_trial = runner
    try:
        with pytest.raises(FatalTrialError, match="spin forever"):
            drive(study, n_trials=100, space={})
    finally:
        driver_mod.run_trial = original
    # It stopped early rather than asking for all 100.
    assert len(study.trials) < 10, len(study.trials)


def test_the_stall_guard_tolerates_a_single_non_advancing_attempt():
    """Not a hair trigger. `study.ask` on a WAITING enqueued trial can legitimately
    return without the count moving if another worker tells it, so tripping at the
    first stall would abort a healthy fleet."""
    from experimentation.sweep.search import driver as driver_mod
    from experimentation.sweep.search.driver import _MAX_STALLED_ATTEMPTS

    assert _MAX_STALLED_ATTEMPTS > 1

    state = {"skips": 0}

    def runner(study, trial, **kwargs):
        # Fail to advance exactly once, then behave.
        if state["skips"] < 1:
            state["skips"] += 1
            return _ok(trial)          # returns WITHOUT telling
        study.tell(trial, 1.0)
        return _ok(trial)

    study = _Study()
    original = driver_mod.run_trial
    driver_mod.run_trial = runner
    try:
        outcomes = drive(study, n_trials=3, space={})
    finally:
        driver_mod.run_trial = original
    assert len(outcomes) == 4, "one stalled attempt should not abort the study"
