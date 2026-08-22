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
contract. Rejected alternatives, both recorded at the call site: making the pure
reader take `(run_dir, study, trial)` forces it to accept two arguments it ignores
and drags optuna into `metrics.py`'s module scope where it is lazy on purpose; and
`inspect.signature` sniffing fails silently the moment a signature drifts.

**One parameter, not two.** `run_trial` briefly took both a plain `read_objective`
and a `read_objective_factory`, resolved by a ternary with a hand-written
TypeError for neither-given and a silent tie-break for both-given -- four states
where there is one. The factory is strictly more general, so the plain shape is
expressible in it via `metrics.fixed_reader` and the pure reader keeps its exact
documented contract. What looked like backward compatibility turned out to be
compatibility with this file: `read_objective=` had no caller outside the test
suite, since `__main__.py` has always passed a factory.
"""

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

optuna = pytest.importorskip("optuna")

from test_search_driver import _FakeLauncher, _context  # noqa: E402

from experimentation.sweep.search.driver import drive, run_trial  # noqa: E402
from experimentation.sweep.search.metrics import fixed_reader  # noqa: E402


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
                        objective_reader_for=factory, **context)

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
          objective_reader_for=factory, **context)

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
                     objective_reader_for=factory, **context)

    assert [o.state for o in outcomes] == ["pruned", "pruned"]
    pruned = [t for t in study.trials
              if t.state == optuna.trial.TrialState.PRUNED]
    assert len(pruned) == 2
    assert [t.user_attrs.get("saw_trial") for t in pruned] == [0, 1]


# ------------------------------------------------ the old contract survives ----

def test_a_plain_reader_works_through_the_adapter(tmp_path):
    """`read_quick_eval_objective` and friends stay the pure
    `(run_dir) -> float|None` they are documented to be; `fixed_reader` is what
    lets one satisfy the driver's single factory-shaped parameter.

    This replaces test_a_plain_reader_still_works, whose docstring claimed
    "backward compatibility is not incidental". It was compatibility with itself:
    `read_objective=` had no caller outside this suite -- __main__.py has always
    passed the factory -- so the second parameter existed only to keep these
    tests compiling.
    """
    context = _context(tmp_path)
    study = _study()
    trial = _ask(study, context)
    outcome = run_trial(study, trial, launcher=_FakeLauncher(),
                        objective_reader_for=fixed_reader(lambda run_dir: 2.25),
                        **context)
    assert outcome.objective == 2.25


def test_the_adapter_ignores_the_study_and_trial(tmp_path):
    """What makes the pure shape expressible in the general one, asserted
    directly rather than only through the driver."""
    reader = fixed_reader(lambda run_dir: 7.5)
    assert reader("any study", "any trial")("any run dir") == 7.5
    assert reader(None, None) is reader(object(), object()), \
        "fixed_reader must hand back the SAME reader, not rebuild one per trial"


def test_a_missing_reader_is_pythons_own_error(tmp_path):
    """Was test_neither_is_a_loud_error, which pinned a hand-written TypeError
    for a state that no longer exists. `objective_reader_for` is a required
    keyword argument now, so the loudness comes from Python and cannot drift out
    of step with the signature."""
    context = _context(tmp_path)
    study = _study()
    trial = _ask(study, context)
    with pytest.raises(TypeError, match="objective_reader_for"):
        run_trial(study, trial, launcher=_FakeLauncher(), **context)
