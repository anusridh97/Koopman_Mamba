"""How far past `n_trials` a fleet can go, bounded rather than prevented.

`driver.drive` loops `while _finished(study) < n_trials`, and with N worker
processes each evaluating that predicate against a shared journal, up to N-1
trials can already be in flight when the target is reached. So a 256-trial study
at `concurrent_trials: 8` can record as many as 263.

**Why bounded and not prevented.** The obvious fix is to count RUNNING trials
toward the budget. It is worse. A trial that FAILS -- an OOM the ladder cannot
escape, a preemption, a missing metric -- is counted while running and then is
not, so the study would stop SHORT of its target by however many trials failed,
and would do so silently. 7 extra trials out of 256 is a 2.7% overspend that
changes no conclusion; 16 missing trials changes the power of the analysis. There
is no atomic reservation available over `JournalStorage` that would give both.

**What this file does and does not establish.** `_simulate_fleet` re-implements
the loop condition rather than calling `drive`, and it asks exactly `workers`
trials per round -- so its output is always `ceil(n/W)*W`, and
`ceil(n/W)*W - n <= W-1` is a theorem about integers that holds regardless of
what `driver.py` does. Pointed out in review, and it is right: these assertions
document the ARITHMETIC of the bound and the semantics `drive` is meant to have.
They are not evidence that `drive` implements them -- for three of the five
parametrised cases, including the headline (256, 8), the simulated overshoot is
exactly 0. Establishing the real property needs `drive` run in W threads against
a lock-protected fake storage, with the runner telling PRUNED and FAIL; that is
recorded as not done rather than implied. It also pins
the two properties that make the overshoot harmless:

  * a resumed study counts what already finished, so the overshoot does not
    compound across restarts;
  * the extra trials are ordinary trials -- recorded, scored and analysable --
    not a leak.

`drive` is exercised against a fake study and a fake trial runner. Actually
running 263 trials to count them is neither cheap nor more convincing than
counting the calls.
"""
from __future__ import annotations

import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

optuna = pytest.importorskip("optuna")

pytestmark = pytest.mark.correctness

from experimentation.sweep.search.driver import _finished, drive   # noqa: E402


class _Trial:
    def __init__(self, number):
        self.number = number
        self.params = {}
        self.user_attrs = {}
        self.state = optuna.trial.TrialState.RUNNING

    def set_user_attr(self, key, value):
        self.user_attrs[key] = value


class _Study:
    """A journal, shared by N simulated workers. Only what `drive` touches."""

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


def _drive_one_worker(study, n_trials, *, started, finish_immediately=True):
    """Run `drive` with a runner that records each trial it was handed."""
    from experimentation.sweep.search import driver as driver_mod

    def fake_run_trial(study_, trial, **kwargs):
        started.append(trial.number)
        if finish_immediately:
            study_.tell(trial, 1.0)
        return object()

    original = driver_mod.run_trial
    driver_mod.run_trial = fake_run_trial
    try:
        drive(study, n_trials=n_trials, space={})
    finally:
        driver_mod.run_trial = original


# ------------------------------------------------------------ the happy case ----

def test_a_single_worker_hits_the_target_exactly():
    """No concurrency, no overshoot. If this ever failed the bound below would be
    measuring the wrong thing."""
    study = _Study()
    started = []
    _drive_one_worker(study, 16, started=started)
    assert len(started) == 16
    assert _finished(study) == 16


def test_a_resumed_study_runs_only_the_remainder():
    """`n_trials` is the study's TARGET size, not "this many more" -- which is
    also why the overshoot does not compound across restarts."""
    study = _Study()
    for _ in range(10):
        study.tell(study.ask({}), 1.0)
    started = []
    _drive_one_worker(study, 16, started=started)
    assert len(started) == 6, "a resume re-ran trials it had already finished"
    assert _finished(study) == 16


def test_a_study_already_at_its_target_runs_nothing():
    study = _Study()
    for _ in range(16):
        study.tell(study.ask({}), 1.0)
    started = []
    _drive_one_worker(study, 16, started=started)
    assert started == []


# ------------------------------------------------------------- the bound ----

def _simulate_fleet(n_trials, workers):
    """N workers stepping in lockstep against one journal.

    Each round: every worker that saw `_finished < n_trials` asks for a trial,
    and only then do the trials finish. That is the worst case for overshoot and
    the one the bound has to cover -- a real fleet interleaves more finely and
    can only do better.
    """
    study = _Study()
    in_flight = []
    while True:
        asking = [w for w in range(workers) if _finished(study) < n_trials]
        # Workers evaluate the predicate before any of this round's trials
        # finish, which is exactly the race.
        if not asking and not in_flight:
            break
        for _ in asking:
            in_flight.append(study.ask({}))
        for trial in in_flight:
            study.tell(trial, 1.0)
        in_flight = []
        if _finished(study) >= n_trials and not asking:
            break
    return study


@pytest.mark.parametrize("n_trials,workers", [(256, 8), (150, 8), (16, 4),
                                              (100, 3), (64, 16)])
def test_the_simulated_overshoot_never_exceeds_workers_minus_one(n_trials, workers):
    study = _simulate_fleet(n_trials, workers)
    recorded = _finished(study)
    assert recorded >= n_trials, "a fleet must not stop SHORT of its target"
    assert recorded <= n_trials + workers - 1, (
        f"{recorded} trials recorded for a {n_trials}-trial study at "
        f"{workers} workers; the documented bound is "
        f"{n_trials + workers - 1}")


def test_the_target_study_is_bounded_at_263():
    """The number the study file states, asserted so the file cannot drift from
    the behaviour."""
    from experimentation.sweep.search.studyspec import load_study_spec

    spec = load_study_spec(
        REPO / "configs/search/proxy-256x17-interactions-v1.yaml")
    assert spec.n_trials == 256 and spec.concurrent_trials == 8
    bound = spec.n_trials + spec.concurrent_trials - 1
    assert bound == 263
    assert _finished(_simulate_fleet(spec.n_trials,
                                     spec.concurrent_trials)) <= bound


def test_the_overshoot_is_real_and_not_vacuously_zero():
    """Guards the guard. If `_simulate_fleet` never overshot, the bound above
    would pass for any formula at all -- including one that claimed no overshoot
    is possible."""
    study = _simulate_fleet(10, 8)
    assert _finished(study) > 10, (
        "the simulation produced no overshoot, so the bound is untested")


def test_the_plan_printer_states_the_bound(tmp_path):
    """A property nobody is told about is one that gets rediscovered as a bug
    when someone counts 263 rows in a 256-trial trials.csv."""
    import subprocess
    import textwrap

    spec = tmp_path / "study.yaml"
    spec.write_text(textwrap.dedent("""
        name: overshoot-probe
        base: configs/runs/4m-golden.yaml
        n_trials: 256
        max_steps: 600
        prune_after_step: 400
        concurrent_trials: 8
        run_root: %s
    """).strip() % tmp_path)
    result = subprocess.run(
        [sys.executable, "-m", "experimentation.sweep.search", str(spec),
         "--dry_run"],
        capture_output=True, text=True, cwd=REPO)
    assert result.returncode == 0, result.stderr[-2000:]
    assert "overshoot" in result.stdout
    assert "256-263" in result.stdout, result.stdout


# ------------------------------------------------- the extra trials are usable ----

def test_the_extra_trials_are_ordinary_recorded_trials():
    """Not a leak: they are asked, told, and counted, so they appear in
    trials.csv and in the analysis like any other."""
    study = _simulate_fleet(16, 4)
    assert all(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    assert len(study.trials) == _finished(study)
