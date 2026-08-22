"""experimentation/sweep/search/driver.py -- ask, materialize, launch, tell.

The structural heart of the integration. The original harness wrapped training in
an `objective(trial)` closure, which forced one long-lived process holding a
child, a GPU thread pool and a stdout pipe. optuna's ask/tell interface removes
that constraint: the gap between `ask()` and `tell()` can be a Slurm queue, so
the driver reuses `sweep/launch.py::materialize_cell` and the existing launchers
and inherits content-hashed run_ids, the dirty-tree gate, verify_shard and the
array job instead of reimplementing them.

The launcher and the objective reader are injected. That is not test scaffolding
for its own sake -- `LocalLauncher`/`SlurmLauncher` are already passed around as
objects in this codebase, and the reader has to be swappable because a local run
can be read immediately while a queued one cannot. It also means the whole loop
is exercised on a CPU box with no GPU and no optuna-driven training.
"""
import json

import pytest
import yaml

optuna = pytest.importorskip("optuna", reason="optuna is an optional dependency")

pytestmark = pytest.mark.correctness


@pytest.fixture(autouse=True)
def _quiet_optuna():
    previous = optuna.logging.get_verbosity()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    yield
    optuna.logging.set_verbosity(previous)


def _base_sections(tmp_path):
    """The four raw sections a cell's RunSpec is built from."""
    import dataclasses

    from koopman_lm.config import build_config

    shard = tmp_path / "shard"
    shard.mkdir(exist_ok=True)
    (shard / "meta.json").write_text(json.dumps({
        "n_tokens": 1000, "tokenizer": "NousResearch/Llama-2-7b-hf",
        "mix": {"fineweb": 1.0},
    }))
    return {
        "model": dataclasses.asdict(build_config("50m")),
        "data": {"kind": "shard", "shard_dir": str(shard),
                 "tokenizer": "NousResearch/Llama-2-7b-hf",
                 "mix": {"fineweb": 1.0}, "n_tokens": 1000},
        "optim": {"lr": 4.0e-4, "warmup_steps": 10, "max_steps": 100,
                  "effective_batch": 16},
        "runtime": {"per_device_batch_size": 16, "seed": 42},
    }


class _FakeLauncher:
    """Records what it was asked to run instead of running it."""

    def __init__(self, fail_on=()):
        self.submitted = []
        self.fail_on = set(fail_on)

    def submit(self, spec, run_dir, dry_run=False, resume=False, wait=False):
        self.submitted.append((spec, run_dir))
        if len(self.submitted) in self.fail_on:
            raise RuntimeError("simulated launch failure")
        return ["python", "-m", "experimentation.training.train"]


def _context(tmp_path, **overrides):
    from koopman_lm.config import build_config
    from experimentation.sweep.search.space import search_space

    cfg = build_config("50m")
    context = {
        "base_sections": _base_sections(tmp_path),
        "base_model": cfg,
        "space": search_space(cfg, base_name="50m"),
        "max_steps": 100,
        "run_root": tmp_path / "runs",
        "study_name": "ska-depth",
        "base_lr": 4e-4,
    }
    context.update(overrides)
    return context


# ------------------------------------------------------ scalarisation ----

def test_the_objective_is_validation_loss_by_default():
    """Every penalty weight defaults to zero, so the objective is the measured
    loss and nothing else until someone deliberately trades it away."""
    from experimentation.sweep.search.driver import objective_from_metrics

    metrics = {"full": {"loss": 2.5, "tokens_per_sec": 1000.0},
               "ska_ablation": {"supported": True, "loss_delta": 0.4}}
    assert objective_from_metrics(metrics) == pytest.approx(2.5)


def test_a_parameter_penalty_charges_per_million_over_the_baseline():
    """The default space spans a 5.9% parameter range, so without this a bigger
    config can win on capacity rather than on architecture."""
    from experimentation.sweep.search.driver import objective_from_metrics

    metrics = {"full": {"loss": 2.0}, "ska_ablation": {"supported": False}}
    score = objective_from_metrics(metrics, parameter_penalty=0.01,
                                   param_count=52_000_000,
                                   baseline_param_count=50_000_000)
    assert score == pytest.approx(2.0 + 0.01 * 2.0)


def test_a_smaller_than_baseline_config_is_not_rewarded_by_the_penalty():
    from experimentation.sweep.search.driver import objective_from_metrics

    metrics = {"full": {"loss": 2.0}, "ska_ablation": {"supported": False}}
    score = objective_from_metrics(metrics, parameter_penalty=0.01,
                                   param_count=48_000_000,
                                   baseline_param_count=50_000_000)
    assert score == pytest.approx(2.0), "the penalty is one-sided, not a bonus"


def test_the_ska_delta_reward_is_capped():
    """Rewarding the ablation delta uncapped would let one enormous delta buy an
    arbitrarily bad loss."""
    from experimentation.sweep.search.driver import objective_from_metrics

    metrics = {"full": {"loss": 2.0},
               "ska_ablation": {"supported": True, "loss_delta": 5.0}}
    score = objective_from_metrics(metrics, ska_delta_reward=1.0, ska_delta_cap=0.1)
    assert score == pytest.approx(2.0 - 0.1)


def test_an_unsupported_ablation_earns_no_reward():
    from experimentation.sweep.search.driver import objective_from_metrics

    metrics = {"full": {"loss": 2.0}, "ska_ablation": {"supported": False}}
    assert objective_from_metrics(metrics, ska_delta_reward=1.0) == pytest.approx(2.0)


def test_a_negative_ska_delta_earns_no_reward():
    """A negative delta means zeroing SKA made the model BETTER. That must not
    become a bonus by sign error."""
    from experimentation.sweep.search.driver import objective_from_metrics

    metrics = {"full": {"loss": 2.0},
               "ska_ablation": {"supported": True, "loss_delta": -0.3}}
    assert objective_from_metrics(metrics, ska_delta_reward=1.0) == pytest.approx(2.0)


def test_a_throughput_penalty_charges_for_missing_the_target():
    from experimentation.sweep.search.driver import objective_from_metrics

    metrics = {"full": {"loss": 2.0, "tokens_per_sec": 500.0},
               "ska_ablation": {"supported": False}}
    score = objective_from_metrics(metrics, throughput_penalty=0.1,
                                   target_tokens_per_sec=1000.0)
    assert score == pytest.approx(2.0 + 0.1 * 1.0)


# ------------------------------------------------------------- a trial ----

def test_run_trial_materializes_a_real_run_directory(tmp_path):
    from experimentation.sweep.search.driver import run_trial
    from experimentation.sweep.search.study import create_study, to_distributions

    context = _context(tmp_path)
    study = create_study(study_name="ska-depth", study_dir=tmp_path, seed=1)
    trial = study.ask(to_distributions(context["space"]))
    launcher = _FakeLauncher()

    outcome = run_trial(study, trial, launcher=launcher,
                        read_objective=lambda run_dir: 2.5, **context)

    assert outcome.state == "complete"
    assert outcome.objective == pytest.approx(2.5)
    assert (outcome.run_dir / "spec.yaml").is_file()
    assert (outcome.run_dir / "attempts.jsonl").is_file()
    assert len(launcher.submitted) == 1


def test_the_run_directory_records_which_study_and_trial_produced_it(tmp_path):
    """A search's runs must be attributable. sweep stamps sweep_id/sweep_name
    through materialize_cell's extra=; a study stamps its own identifiers through
    the same door."""
    from experimentation.sweep.search.driver import run_trial
    from experimentation.sweep.search.study import create_study, to_distributions

    context = _context(tmp_path)
    study = create_study(study_name="ska-depth", study_dir=tmp_path, seed=1)
    trial = study.ask(to_distributions(context["space"]))
    outcome = run_trial(study, trial, launcher=_FakeLauncher(),
                        read_objective=lambda run_dir: 1.0, **context)

    spec = yaml.safe_load((outcome.run_dir / "spec.yaml").read_text())
    assert spec["study_name"] == "ska-depth"
    assert spec["trial_number"] == trial.number
    assert spec["sweep_name"] == "ska-depth", (
        "reusing sweep_name keeps results.py's existing column meaningful "
        "without a schema change")


def test_the_trial_result_reaches_the_study(tmp_path):
    from experimentation.sweep.search.driver import run_trial
    from experimentation.sweep.search.study import create_study, to_distributions

    context = _context(tmp_path)
    study = create_study(study_name="ska-depth", study_dir=tmp_path, seed=1)
    trial = study.ask(to_distributions(context["space"]))
    run_trial(study, trial, launcher=_FakeLauncher(),
              read_objective=lambda run_dir: 1.75, **context)

    assert study.best_value == pytest.approx(1.75)
    assert study.trials[0].state.name == "COMPLETE"


def test_run_trial_records_the_run_id_and_the_anchor_name(tmp_path):
    from experimentation.sweep.search.anchors import Design
    from experimentation.sweep.search.driver import run_trial
    from experimentation.sweep.search.study import (
        create_study, enqueue_anchors, to_distributions)

    context = _context(tmp_path)
    study = create_study(study_name="ska-depth", study_dir=tmp_path, seed=1)
    enqueue_anchors(study, [Design(name="baseline")], context["base_model"],
                    context["space"], base_lr=4e-4)
    trial = study.ask(to_distributions(context["space"]))
    outcome = run_trial(study, trial, launcher=_FakeLauncher(),
                        read_objective=lambda run_dir: 1.0, **context)

    assert outcome.anchor == "baseline"
    assert len(outcome.run_id) == 8
    assert outcome.run_id in str(outcome.run_dir)


def test_a_failed_launch_marks_the_trial_failed_without_raising(tmp_path):
    """One bad config must not end the study. A search that dies on its first
    OOM has wasted every trial before it."""
    from experimentation.sweep.search.driver import run_trial
    from experimentation.sweep.search.study import create_study, to_distributions

    context = _context(tmp_path)
    study = create_study(study_name="ska-depth", study_dir=tmp_path, seed=1)
    trial = study.ask(to_distributions(context["space"]))
    outcome = run_trial(study, trial, launcher=_FakeLauncher(fail_on=(1,)),
                        read_objective=lambda run_dir: 1.0, **context)

    assert outcome.state == "failed"
    assert outcome.objective is None
    assert study.trials[0].state.name == "FAIL"


def test_an_unreadable_objective_marks_the_trial_failed(tmp_path):
    """Training can exit 0 and still leave no metric -- a missing checkpoint, a
    preemption between the save and the eval. Telling optuna a fabricated number
    would poison every later proposal."""
    from experimentation.sweep.search.driver import run_trial
    from experimentation.sweep.search.study import create_study, to_distributions

    context = _context(tmp_path)
    study = create_study(study_name="ska-depth", study_dir=tmp_path, seed=1)
    trial = study.ask(to_distributions(context["space"]))
    outcome = run_trial(study, trial, launcher=_FakeLauncher(),
                        read_objective=lambda run_dir: None, **context)

    assert outcome.state == "failed"
    assert study.trials[0].state.name == "FAIL"


# -------------------------------------------------------------- the loop ----

def test_drive_runs_the_requested_number_of_trials(tmp_path):
    from experimentation.sweep.search.driver import drive
    from experimentation.sweep.search.study import create_study

    context = _context(tmp_path)
    study = create_study(study_name="ska-depth", study_dir=tmp_path, seed=1)
    launcher = _FakeLauncher()
    outcomes = drive(study, n_trials=3, launcher=launcher,
                     read_objective=lambda run_dir: 2.0, **context)

    assert len(outcomes) == 3
    assert len(launcher.submitted) == 3
    assert len({o.run_id for o in outcomes}) == 3, "distinct configs, distinct ids"


def test_drive_counts_trials_already_completed_when_resuming(tmp_path):
    """n_trials is the study's target size, not "run this many more". Resuming a
    15-trial study that finished 10 should run 5."""
    from experimentation.sweep.search.driver import drive
    from experimentation.sweep.search.study import create_study

    context = _context(tmp_path)
    study = create_study(study_name="ska-depth", study_dir=tmp_path, seed=1)
    drive(study, n_trials=2, launcher=_FakeLauncher(),
          read_objective=lambda run_dir: 2.0, **context)

    reopened = create_study(study_name="ska-depth", study_dir=tmp_path, seed=1)
    launcher = _FakeLauncher()
    outcomes = drive(reopened, n_trials=3, launcher=launcher,
                     read_objective=lambda run_dir: 2.0, **context)
    assert len(launcher.submitted) == 1
    assert len(outcomes) == 1


def test_drive_runs_the_anchors_first(tmp_path):
    from experimentation.sweep.search.anchors import Design
    from experimentation.sweep.search.driver import drive
    from experimentation.sweep.search.study import create_study, enqueue_anchors

    context = _context(tmp_path)
    study = create_study(study_name="ska-depth", study_dir=tmp_path, seed=1)
    enqueue_anchors(study, [Design(name="first"), Design(name="second", rank=8)],
                    context["base_model"], context["space"], base_lr=4e-4)
    outcomes = drive(study, n_trials=3, launcher=_FakeLauncher(),
                     read_objective=lambda run_dir: 2.0, **context)

    assert [o.anchor for o in outcomes[:2]] == ["first", "second"]
    assert outcomes[2].anchor is None, "the third is sampled, not curated"


def test_drive_keeps_going_after_a_failure(tmp_path):
    from experimentation.sweep.search.driver import drive
    from experimentation.sweep.search.study import create_study

    context = _context(tmp_path)
    study = create_study(study_name="ska-depth", study_dir=tmp_path, seed=1)
    outcomes = drive(study, n_trials=3, launcher=_FakeLauncher(fail_on=(2,)),
                     read_objective=lambda run_dir: 2.0, **context)

    assert [o.state for o in outcomes] == ["complete", "failed", "complete"]
    assert study.best_value == pytest.approx(2.0)
