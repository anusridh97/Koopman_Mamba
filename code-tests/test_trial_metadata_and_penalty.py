"""What a trial RECORDS, and the parameter penalty that recorded nothing.

Two subjects, joined because the second depends on the first.

**Trial metadata.** `trial.params` records what the sampler CHOSE. Nothing
recorded what that choice RESOLVED to -- so a finished study could not answer
"how many parameters did trial 91 have", which is the question a loss/size
Pareto front is made of, nor "which worker ran it", which is how a concurrency
artefact is told from a real effect. `driver._record_trial_attrs` stamps those at
MATERIALIZATION time, before the launch can fail, so a FAILED or PRUNED trial
still carries them -- and `report.trial_row` flattens `user_attrs` wholesale, so
they reach `trials.csv` with no further wiring.

**The parameter penalty, which was inert.** `objective_from_metrics` has always
accepted `parameter_penalty` next to `param_count` and `baseline_param_count`,
and guarded on `if parameter_penalty > 0 and param_count and
baseline_param_count`. Nothing ever supplied the counts. So the guard was false
for every trial ever run: a documented, unit-tested, silently dead option -- the
exact "valid, validated, and read by nothing" shape `HANDOFF-2026-08-21.md` §6
names as this codebase's most common failure.

The fix has two halves and both are tested here: the counts now reach
`objective_from_metrics` through `metrics.weights_for_trial`, and a declared
penalty with missing counts RAISES instead of silently reverting to pure loss.
A wrong-but-complete answer is worse than a crash at trial 0.
"""
from __future__ import annotations

import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

pytestmark = pytest.mark.correctness

from experimentation.sweep.search.metrics import (           # noqa: E402
    OBJECTIVE_WEIGHTS, objective_from_metrics, weights_for_trial)


class _Trial:
    """The parts of an optuna Trial these functions touch."""

    def __init__(self, number=0, attrs=None):
        self.number = number
        self.user_attrs = dict(attrs or {})

    def set_user_attr(self, key, value):
        self.user_attrs[key] = value


# ------------------------------------------------- the weight list stays in step ----

def test_studyspec_and_metrics_agree_about_the_weight_names():
    """`studyspec._OBJECTIVE_WEIGHTS` is a hand-copied duplicate of
    `metrics.OBJECTIVE_WEIGHTS`, because studyspec must stay importable with
    optuna absent. A duplicate that can drift silently is worse than no
    validation, so this is the guard that makes the duplication safe."""
    import inspect

    from experimentation.sweep.search.studyspec import (
        _OBJECTIVE_DERIVED, _OBJECTIVE_WEIGHTS)

    signature = inspect.signature(objective_from_metrics)
    accepted = {name for name, p in signature.parameters.items()
                if p.kind == p.KEYWORD_ONLY}
    assert set(OBJECTIVE_WEIGHTS) | _OBJECTIVE_DERIVED == accepted, (
        "objective_from_metrics accepts a keyword nobody declared, or declares "
        "one it does not accept")
    assert _OBJECTIVE_WEIGHTS == set(OBJECTIVE_WEIGHTS)


def test_a_misspelled_objective_weight_is_refused_at_parse_time():
    """It used to raise TypeError deep inside the objective reader, which the
    driver's own `except Exception` turned into a FAILED trial with no visible
    cause."""
    from experimentation.sweep.search.studyspec import StudySpec

    with pytest.raises(ValueError, match="unknown key"):
        StudySpec(name="x", base="b.yaml", n_trials=10, max_steps=1000,
                  objective={"parameter_penalies": 0.5})


def test_a_parameter_count_cannot_be_declared_as_a_weight():
    """A constant count would apply the same number to every trial and turn the
    penalty into a fixed offset -- i.e. no penalty at all, while looking like
    one."""
    from experimentation.sweep.search.studyspec import StudySpec

    with pytest.raises(ValueError, match="DERIVED PER TRIAL"):
        StudySpec(name="x", base="b.yaml", n_trials=10, max_steps=1000,
                  objective={"parameter_penalty": 0.5, "param_count": 26_000_000})


# ------------------------------------------------------- weights_for_trial ----

def test_with_no_penalty_the_weights_pass_through_untouched():
    """Every committed study has `objective: {}`. This must be exactly what they
    got before any of this existed."""
    assert weights_for_trial({}, _Trial()) == {}
    assert weights_for_trial({"ska_delta_reward": 0.2}, _Trial()) == {
        "ska_delta_reward": 0.2}


def test_a_zero_penalty_does_not_require_the_counts():
    """0.0 means "off", and off must not be a startup failure on a trial that
    happens to lack the attrs."""
    assert weights_for_trial({"parameter_penalty": 0.0}, _Trial()) == {
        "parameter_penalty": 0.0}


def test_a_positive_penalty_pulls_both_counts_off_the_trial():
    trial = _Trial(attrs={"param_count": 26_000_000,
                          "baseline_param_count": 25_352_736})
    resolved = weights_for_trial({"parameter_penalty": 0.5}, trial)
    assert resolved == {"parameter_penalty": 0.5,
                        "param_count": 26_000_000,
                        "baseline_param_count": 25_352_736}


def test_a_positive_penalty_with_no_counts_raises_rather_than_reverting():
    """THE guard. Silently dropping the penalty produces a complete, plausible,
    wrong answer -- which is the failure this option already had once."""
    with pytest.raises(ValueError, match="REFUSING"):
        weights_for_trial({"parameter_penalty": 0.5}, _Trial(number=91))


def test_the_refusal_names_the_trial_and_the_missing_keys():
    with pytest.raises(ValueError) as exc:
        weights_for_trial({"parameter_penalty": 0.5},
                          _Trial(number=91, attrs={"param_count": 26_000_000}))
    message = str(exc.value)
    assert "91" in message and "baseline_param_count" in message


def test_the_caller_is_not_mutated():
    """`study_spec.objective` is read once per trial; mutating it would leave
    trial 0's counts on trial 1."""
    weights = {"parameter_penalty": 0.5}
    weights_for_trial(weights, _Trial(attrs={"param_count": 26_000_000,
                                             "baseline_param_count": 25_000_000}))
    assert weights == {"parameter_penalty": 0.5}


# --------------------------------------------- the penalty changes the objective ----

_METRICS = {"full": {"loss": 3.0}}


def test_the_objective_is_the_bare_loss_with_no_penalty():
    assert objective_from_metrics(_METRICS) == 3.0


def test_turning_the_penalty_on_changes_the_objective():
    """The observation that was impossible before: a positive penalty with real
    counts moves the number."""
    plain = objective_from_metrics(_METRICS)
    penalised = objective_from_metrics(
        _METRICS, **weights_for_trial(
            {"parameter_penalty": 0.5},
            _Trial(attrs={"param_count": 26_352_736,
                          "baseline_param_count": 25_352_736})))
    assert penalised != plain
    # 1.0M parameters over baseline x 0.5 per million.
    assert penalised == pytest.approx(3.5)


def test_the_penalty_is_one_sided():
    """A config SMALLER than the baseline earns no bonus. These are constraints
    expressed as penalties, not quantities being jointly optimised."""
    smaller = objective_from_metrics(
        _METRICS, **weights_for_trial(
            {"parameter_penalty": 0.5},
            _Trial(attrs={"param_count": 24_000_000,
                          "baseline_param_count": 25_352_736})))
    assert smaller == 3.0


def test_the_penalty_scales_with_the_excess():
    def score(count):
        return objective_from_metrics(
            _METRICS, **weights_for_trial(
                {"parameter_penalty": 1.0},
                _Trial(attrs={"param_count": count,
                              "baseline_param_count": 25_000_000})))

    assert score(26_000_000) == pytest.approx(4.0)
    assert score(27_000_000) == pytest.approx(5.0)


def test_the_real_anchor_spread_would_move_a_penalised_objective():
    """Not a synthetic number: `layers-8` resolves to 26,079,792 parameters
    against the base's 25,352,736, so the penalty has something real to act on.
    A space whose configs all had the same size would make the option pointless
    even once wired."""
    penalised = objective_from_metrics(
        _METRICS, **weights_for_trial(
            {"parameter_penalty": 1.0},
            _Trial(attrs={"param_count": 26_079_792,
                          "baseline_param_count": 25_352_736})))
    assert penalised == pytest.approx(3.0 + 0.727056)


# -------------------------------------------------- what the driver records ----

def test_the_recorded_attrs_are_the_ones_the_module_declares():
    """Guards the guard: `TRIAL_ATTRS` is what a consumer reads to know which
    columns exist, so it must not drift from what is written."""
    # `driver` imports optuna at module scope -- it is one of the two
    # modules that deliberately do. The penalty tests above must keep
    # running with optuna ABSENT (metrics.py is optuna-free on purpose,
    # and the CPU suite is what guards that), so the skip is per-test
    # rather than module-level.
    pytest.importorskip("optuna")
    from experimentation.sweep.search.driver import TRIAL_ATTRS

    assert set(TRIAL_ATTRS) == {
        "param_count", "baseline_param_count", "worker_id", "sampler",
        "sampler_seed", "per_device_batch_size", "anchor_name",
        # Stamped after the run, from quick_eval.json, not at materialization.
        "ska_delta"}


def _spec_and_base():
    from experimentation.run.resolve import resolve_run_spec

    spec = resolve_run_spec(REPO / "configs/runs/proxy-256x17.yaml")
    return spec, spec.model


def test_record_trial_attrs_writes_every_declared_key():
    # `driver` imports optuna at module scope -- it is one of the two
    # modules that deliberately do. The penalty tests above must keep
    # running with optuna ABSENT (metrics.py is optuna-free on purpose,
    # and the CPU suite is what guards that), so the skip is per-test
    # rather than module-level.
    pytest.importorskip("optuna")
    from experimentation.sweep.search.driver import _record_trial_attrs

    spec, base_model = _spec_and_base()
    trial = _Trial()
    _record_trial_attrs(trial, spec, base_model, worker_id=3,
                        sampler_name="tpe_multivariate", sampler_seed=2029,
                        anchor="rank-8")
    assert trial.user_attrs["param_count"] == 25_352_736
    assert trial.user_attrs["baseline_param_count"] == 25_352_736
    assert trial.user_attrs["worker_id"] == 3
    assert trial.user_attrs["sampler"] == "tpe_multivariate"
    assert trial.user_attrs["sampler_seed"] == 2029
    assert trial.user_attrs["per_device_batch_size"] == 8


def test_the_recorded_count_is_the_TRIALS_count_not_the_bases():
    """The two differ for every trial that moved rank or layer count, which is
    most of them -- and if they were always equal the Pareto front would be a
    ranking with an extra column."""
    import dataclasses

    # `driver` imports optuna at module scope -- it is one of the two
    # modules that deliberately do. The penalty tests above must keep
    # running with optuna ABSENT (metrics.py is optuna-free on purpose,
    # and the CPU suite is what guards that), so the skip is per-test
    # rather than module-level.
    pytest.importorskip("optuna")
    from experimentation.sweep.search.driver import _record_trial_attrs

    spec, base_model = _spec_and_base()
    wider = dataclasses.replace(
        spec, model=dataclasses.replace(
            base_model, ska_layer_indices=[3, 5, 6, 8, 10, 12, 13, 15]))
    trial = _Trial()
    _record_trial_attrs(trial, wider, base_model, worker_id=0,
                        sampler_name="tpe", sampler_seed=1, anchor=None)
    assert trial.user_attrs["param_count"] == 26_079_792
    assert trial.user_attrs["baseline_param_count"] == 25_352_736
    assert trial.user_attrs["param_count"] > trial.user_attrs["baseline_param_count"]


def test_absent_provenance_is_omitted_rather_than_recorded_as_none():
    """A single-process study has no worker id. `attr_worker_id` as an empty CSV
    cell reads as "worker unknown"; as the string "None" it reads as a worker
    called None."""
    # `driver` imports optuna at module scope -- it is one of the two
    # modules that deliberately do. The penalty tests above must keep
    # running with optuna ABSENT (metrics.py is optuna-free on purpose,
    # and the CPU suite is what guards that), so the skip is per-test
    # rather than module-level.
    pytest.importorskip("optuna")
    from experimentation.sweep.search.driver import _record_trial_attrs

    spec, base_model = _spec_and_base()
    trial = _Trial()
    _record_trial_attrs(trial, spec, base_model, worker_id=None,
                        sampler_name=None, sampler_seed=None, anchor=None)
    assert "worker_id" not in trial.user_attrs
    assert "sampler" not in trial.user_attrs
    assert "sampler_seed" not in trial.user_attrs
    # But the counts, which are never unknown, are always there.
    assert trial.user_attrs["param_count"] == 25_352_736


def test_the_attrs_reach_trials_csv():
    """The only channel a post-hoc analysis has. `report.trial_row` flattens
    user_attrs into `attr_<key>` columns, so this is what makes the metadata
    useful rather than merely stored."""
    pytest.importorskip("optuna")
    from experimentation.sweep.search.report import trial_row

    class _Frozen:
        number = 7
        params = {"ska_rank": 24}
        user_attrs = {"param_count": 26_079_792, "worker_id": 3,
                      "sampler": "tpe_multivariate", "sampler_seed": 2029,
                      "per_device_batch_size": 4, "anchor_name": "layers-8"}
        value = 3.21
        datetime_start = None
        datetime_complete = None

        class state:
            name = "COMPLETE"

    row = trial_row(_Frozen())
    assert row["attr_param_count"] == 26_079_792
    assert row["attr_worker_id"] == 3
    assert row["attr_sampler"] == "tpe_multivariate"
    assert row["attr_sampler_seed"] == 2029
    assert row["attr_per_device_batch_size"] == 4
    assert row["attr_anchor_name"] == "layers-8"
    assert row["param_ska_rank"] == 24


# ------------------------------------- ska_delta is WRITTEN, not just read ----

def test_ska_delta_is_stamped_from_the_eval_payload(tmp_path):
    """Found in review as an inert field: `analysis._top_by_ska_delta` reads
    `ska_delta` and NOTHING wrote it, so `top_by_ska_delta.csv` would have been
    header-only for every real study."""
    pytest.importorskip("optuna")
    import json

    from experimentation.sweep.search.driver import _stamp_ska_delta

    run_dir = tmp_path / "run"
    (run_dir / "eval" / "final").mkdir(parents=True)
    (run_dir / "eval" / "final" / "quick_eval.json").write_text(json.dumps(
        {"metrics": {"full": {"loss": 3.2},
                     "ska_ablation": {"supported": True, "loss_delta": 0.0123}}}))
    trial = _Trial()
    _stamp_ska_delta(trial, run_dir)
    assert trial.user_attrs["ska_delta"] == 0.0123


def test_an_unsupported_ablation_stamps_nothing(tmp_path):
    """Absent is not zero. `analysis` EXCLUDES a trial with no delta rather than
    ranking it as zero, so writing 0.0 here would fabricate a whole ranking."""
    pytest.importorskip("optuna")
    import json

    from experimentation.sweep.search.driver import _stamp_ska_delta

    run_dir = tmp_path / "run"
    (run_dir / "eval" / "final").mkdir(parents=True)
    (run_dir / "eval" / "final" / "quick_eval.json").write_text(json.dumps(
        {"metrics": {"full": {"loss": 3.2},
                     "ska_ablation": {"supported": False}}}))
    trial = _Trial()
    _stamp_ska_delta(trial, run_dir)
    assert "ska_delta" not in trial.user_attrs


def test_a_missing_or_malformed_eval_does_not_cost_the_trial(tmp_path):
    """This is provenance for a post-hoc ranking, not the objective. Raising
    would turn a missing optional metric into a lost result."""
    pytest.importorskip("optuna")
    from experimentation.sweep.search.driver import _stamp_ska_delta

    trial = _Trial()
    _stamp_ska_delta(trial, tmp_path / "nothing-here")     # no eval dir at all
    assert "ska_delta" not in trial.user_attrs

    run_dir = tmp_path / "bad"
    (run_dir / "eval" / "final").mkdir(parents=True)
    (run_dir / "eval" / "final" / "quick_eval.json").write_text("{not json")
    _stamp_ska_delta(trial, run_dir)
    assert "ska_delta" not in trial.user_attrs


def test_ska_delta_is_declared_in_the_attr_roster():
    pytest.importorskip("optuna")
    from experimentation.sweep.search.driver import TRIAL_ATTRS

    assert "ska_delta" in TRIAL_ATTRS
