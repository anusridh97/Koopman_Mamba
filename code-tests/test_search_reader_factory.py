"""Pruning has to be reachable from the entry point people actually call.

`read_objective` reads a FINISHED run: `(run_dir) -> float | None`, pure, with no
optuna in sight. That is the right shape for scoring after the fact, and
`metrics.read_quick_eval_objective` is exactly it.

Pruning cannot use that shape. Pruning IS `trial.report(value, step)` followed by
`trial.should_prune()`, so `metrics.wait_for_objective` needs
`(study, trial, run_dir)`. And `drive()` builds one `**kwargs` dict *before any
trial exists*, so it could only ever hand `run_trial` a fixed reader -- one that
had no way to reference the trial it was scoring.

The consequence was that pruning was **unreachable through `drive()`** while
`driver.py`'s own module docstring said it "arrives through the read_objective
seam". True if you call `run_trial` yourself; false via the entry point anyone
would use, which is the one the CLI will call.

A factory `(study, trial) -> reader` closes it without widening the pure reader's
contract. Rejected alternatives, both recorded at the call site: making
`read_objective` take `(run_dir, study, trial)` forces the pure function to accept
two arguments it ignores and drags optuna into `metrics.py`'s module scope where
it is lazy on purpose; and `inspect.signature` sniffing fails silently the moment a
signature drifts.
"""

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

optuna = pytest.importorskip("optuna")

from test_search_driver import _FakeLauncher, _context  # noqa: E402

from experimentation.sweep.search.driver import drive, run_trial  # noqa: E402


@pytest.fixture(autouse=True)
def _quiet_optuna():
    optuna.logging.set_verbosity(optuna.logging.WARNING)


def _study():
    return optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))


def _ask(study, context):
    from experimentation.sweep.search.study import to_distributions
    return study.ask(to_distributions(context["space"]))


# --------------------------------------------------- the factory is honoured ----

def test_the_factory_receives_the_study_and_the_trial(tmp_path):
    """The whole point: a reader that can see the trial it is scoring."""
    context = _context(tmp_path)
    study = _study()
    trial = _ask(study, context)
    seen = {}

    def factory(s, t):
        seen["study"] = s
        seen["trial_number"] = t.number
        return lambda run_dir: 3.5

    outcome = run_trial(study, trial, launcher=_FakeLauncher(),
                        read_objective_factory=factory, **context)

    assert seen["study"] is study
    assert seen["trial_number"] == trial.number
    assert outcome.objective == 3.5


def test_the_factory_is_built_per_trial_not_once(tmp_path):
    """A reader captured once could not distinguish trial 0 from trial 3, which
    is exactly why a fixed read_objective cannot prune."""
    context = _context(tmp_path)
    study = _study()
    numbers = []

    def factory(s, t):
        numbers.append(t.number)
        return lambda run_dir: 1.0 + t.number

    drive(study, n_trials=3, launcher=_FakeLauncher(),
          read_objective_factory=factory, **context)

    assert numbers == [0, 1, 2], f"factory saw {numbers}"


def test_a_factory_can_prune_through_drive(tmp_path):
    """The defect, end to end. A factory-built reader raises TrialPruned using
    the trial it was handed; before this seam existed there was no way for a
    reader reaching drive() to do that at all."""
    context = _context(tmp_path)
    study = _study()

    def factory(s, t):
        def reader(run_dir):
            t.set_user_attr("saw_trial", t.number)
            raise optuna.TrialPruned("hopeless")
        return reader

    outcomes = drive(study, n_trials=2, launcher=_FakeLauncher(),
                     read_objective_factory=factory, **context)

    assert [o.state for o in outcomes] == ["pruned", "pruned"]
    pruned = [t for t in study.trials
              if t.state == optuna.trial.TrialState.PRUNED]
    assert len(pruned) == 2
    assert [t.user_attrs.get("saw_trial") for t in pruned] == [0, 1]


# ------------------------------------------------ the old contract survives ----

def test_a_plain_reader_still_works(tmp_path):
    """Backward compatibility is not incidental -- read_quick_eval_objective is
    this shape, and it must stay usable without a wrapper."""
    context = _context(tmp_path)
    study = _study()
    trial = _ask(study, context)
    outcome = run_trial(study, trial, launcher=_FakeLauncher(),
                        read_objective=lambda run_dir: 2.25, **context)
    assert outcome.objective == 2.25


def test_the_factory_wins_when_both_are_given(tmp_path):
    context = _context(tmp_path)
    study = _study()
    trial = _ask(study, context)
    outcome = run_trial(study, trial, launcher=_FakeLauncher(),
                        read_objective=lambda run_dir: 9.9,
                        read_objective_factory=lambda s, t: (lambda rd: 1.1),
                        **context)
    assert outcome.objective == 1.1


def test_neither_is_a_loud_error(tmp_path):
    """Silently scoring nothing would FAIL the trial after paying for its
    training, which reads as a training bug rather than a wiring bug."""
    context = _context(tmp_path)
    study = _study()
    trial = _ask(study, context)
    with pytest.raises(TypeError, match="read_objective"):
        run_trial(study, trial, launcher=_FakeLauncher(), **context)
