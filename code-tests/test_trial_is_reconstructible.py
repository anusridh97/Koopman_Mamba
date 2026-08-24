"""Everything the study's questions need, durable per trial -- and provably so.

The study asks which SKA capacity, placement, recurrence, stabilization and
learning-rate choices matter, and which interact. Answering any of those post hoc
requires that a trial be RECONSTRUCTIBLE from what was stored: `trial.params`
records what the sampler CHOSE, and nothing else records what that choice
resolved to.

Two concrete gaps this file closes, both found by inspection rather than by a
failing test, which is why they had survived.

**Throughput and peak memory were measured and thrown away.** `quick_eval.json`
already carries `metrics.full.tokens_per_sec`, `metrics.full.peak_memory_gib` and
`metrics.full.n_tokens` for every trial. `driver`'s attr block recorded neither,
so neither reached `user_attrs`, therefore neither reached `trials.csv`, therefore
the loss/throughput Pareto front the study's cost argument rests on could not be
computed at all. The measurement existed; the wire did not.

**The resolved layer indices were unrecoverable.** `n_ska_layers` and `placement`
are recorded as REQUESTS. `geometry.make_layer_indices` CLAMPS a count past the
usable window rather than raising -- that clamp is the exact mechanism that made
two of nine axes dead on `4m-golden` -- so without the resolved indices a clamped
trial is indistinguishable from an unclamped one in the output.

The last test is the general guard, and it is the one worth keeping honest: take
a trial's stored metadata, rebuild its `RunSpec` from scratch, and require the
content-addressed `run_id` to match the one recorded. `run_id` is
`sha256(model + data + optim + seed)`, so an equality there is a statement about
every scientific field at once -- including `runtime.seed`, which is why the
`model_seed` column has to exist for this test to be possible.
"""
import json

import pytest
import yaml

pytestmark = pytest.mark.correctness

optuna = pytest.importorskip("optuna", reason="optuna is an optional dependency")


@pytest.fixture(autouse=True)
def _quiet_optuna():
    previous = optuna.logging.get_verbosity()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    yield
    optuna.logging.set_verbosity(previous)


def _base_sections(tmp_path):
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
    def __init__(self):
        self.submitted = []

    def submit(self, spec, run_dir, dry_run=False, resume=False, wait=False):
        self.submitted.append((spec, run_dir))
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
        "study_name": "provenance",
        "base_lr": 4e-4,
    }
    context.update(overrides)
    return context


_PAYLOAD = {
    "metrics": {
        "full": {"loss": 3.25, "ppl": 25.8, "n_tokens": 262144,
                 "n_batches": 32, "tokens_per_sec": 41234.5,
                 "peak_memory_gib": 17.75},
        "ska_ablation": {"supported": True, "loss_delta": 1.166e-4},
    }
}


def _write_eval(run_dir, payload=None):
    out = run_dir / "eval" / "final"
    out.mkdir(parents=True, exist_ok=True)
    (out / "quick_eval.json").write_text(json.dumps(payload or _PAYLOAD))


def _one_trial(tmp_path, payload=None):
    """Run one trial whose eval payload lands before the objective is read."""
    from experimentation.sweep.search.driver import run_trial
    from experimentation.sweep.search.study import create_study, to_distributions

    context = _context(tmp_path)
    study = create_study(study_name="provenance", study_dir=tmp_path, seed=1)
    trial = study.ask(to_distributions(context["space"]))

    def reader_for(_study, _trial):
        def read(run_dir):
            _write_eval(run_dir, payload)
            from experimentation.sweep.search.metrics import (
                read_quick_eval_objective)
            return read_quick_eval_objective(run_dir)
        return read

    outcome = run_trial(study, trial, launcher=_FakeLauncher(),
                        objective_reader_for=reader_for,
                        worker_id=3, sampler_name="tpe_multivariate",
                        sampler_seed=2029, **context)
    return study, study.trials[0], outcome


# ------------------------------------------------- the promoted measurements ----

def test_throughput_reaches_the_trial(tmp_path):
    """Without it the loss/throughput Pareto front cannot be computed, and the
    cost half of the study's argument has no data."""
    _study, trial, _outcome = _one_trial(tmp_path)
    assert trial.user_attrs["tokens_per_sec"] == pytest.approx(41234.5)


def test_peak_memory_reaches_the_trial(tmp_path):
    _study, trial, _outcome = _one_trial(tmp_path)
    assert trial.user_attrs["peak_memory_gib"] == pytest.approx(17.75)


def test_the_eval_token_count_reaches_the_trial(tmp_path):
    """A loss over 2,000 tokens and a loss over 262,144 tokens are not the same
    measurement, and the objective is a bare number that cannot say which."""
    _study, trial, _outcome = _one_trial(tmp_path)
    assert trial.user_attrs["n_eval_tokens"] == 262144


def test_the_ska_delta_still_reaches_the_trial(tmp_path):
    """Regression guard: the new stamps share a code path with this one."""
    _study, trial, _outcome = _one_trial(tmp_path)
    assert trial.user_attrs["ska_delta"] == pytest.approx(1.166e-4)


def test_a_missing_metric_is_absent_rather_than_zero_filled(tmp_path):
    """Absent is not zero. A trial recorded at 0 tokens/sec would sit at the
    wrong end of every cost ranking and look like a measurement."""
    payload = {"metrics": {"full": {"loss": 3.0}}}
    _study, trial, _outcome = _one_trial(tmp_path, payload)
    assert "tokens_per_sec" not in trial.user_attrs
    assert "peak_memory_gib" not in trial.user_attrs
    assert "n_eval_tokens" not in trial.user_attrs
    assert trial.value == pytest.approx(3.0)


def test_a_malformed_eval_does_not_cost_the_trial_its_objective(tmp_path):
    """The stamps are provenance for a post-hoc ranking, not the objective.
    Raising here would turn a missing optional metric into a lost result."""
    from experimentation.sweep.search.driver import run_trial
    from experimentation.sweep.search.study import create_study, to_distributions

    context = _context(tmp_path)
    study = create_study(study_name="provenance", study_dir=tmp_path, seed=1)
    trial = study.ask(to_distributions(context["space"]))

    def reader_for(_study, _trial):
        def read(run_dir):
            out = run_dir / "eval" / "final"
            out.mkdir(parents=True, exist_ok=True)
            (out / "quick_eval.json").write_text("{not json")
            return 2.5
        return read

    outcome = run_trial(study, trial, launcher=_FakeLauncher(),
                        objective_reader_for=reader_for, **context)
    assert outcome.state == "complete"
    assert study.trials[0].value == pytest.approx(2.5)


def test_every_promoted_key_is_declared_in_the_attr_roster():
    """`TRIAL_ATTRS` is the roster a consumer reads to know which columns exist.
    A stamped-but-undeclared attr is invisible to anything that trusts it."""
    from experimentation.sweep.search.driver import TRIAL_ATTRS

    for key in ("tokens_per_sec", "peak_memory_gib", "n_eval_tokens",
                "model_seed", "reference_group", "ska_layer_indices"):
        assert key in TRIAL_ATTRS, key


# ------------------------------------------------- the resolved architecture ----

def test_the_resolved_layer_indices_reach_the_trial(tmp_path):
    """`trial.params` records the REQUEST (n_ska_layers, placement).
    `make_layer_indices` clamps a count past the usable window rather than
    raising -- the exact mechanism that made two axes dead on 4m-golden -- so
    without this a clamped trial is indistinguishable from an unclamped one."""
    _study, trial, outcome = _one_trial(tmp_path)
    recorded = trial.user_attrs["ska_layer_indices"]
    spec_indices = yaml.safe_load(
        (outcome.run_dir / "spec.yaml").read_text())["model"]["ska_layer_indices"]
    assert list(recorded) == list(spec_indices)
    assert len(recorded) == trial.params["n_ska_layers"]


def test_the_indices_survive_a_failed_launch(tmp_path):
    """Recorded at MATERIALIZATION, so the trials a study most needs to explain
    still carry their resolved architecture."""
    from experimentation.sweep.search.driver import run_trial
    from experimentation.sweep.search.metrics import fixed_reader
    from experimentation.sweep.search.study import create_study, to_distributions

    class _Broken:
        def submit(self, *a, **k):
            raise RuntimeError("simulated launch failure")

    context = _context(tmp_path)
    study = create_study(study_name="provenance", study_dir=tmp_path, seed=1)
    trial = study.ask(to_distributions(context["space"]))
    run_trial(study, trial, launcher=_Broken(),
              objective_reader_for=fixed_reader(lambda d: 1.0), **context)

    assert study.trials[0].state.name == "FAIL"
    assert study.trials[0].user_attrs["ska_layer_indices"]
    assert study.trials[0].user_attrs["model_seed"] == 42


# ---------------------------------------------------------- trials.csv ----

_REQUIRED_COLUMNS = (
    # identity and outcome
    "number", "state", "objective",
    # every searched axis
    "param_ska_rank", "param_n_ska_layers", "param_placement",
    "param_ska_ridge", "param_ska_layerscale_init",
    "param_norm_clip_multiplier", "param_gamma_value", "param_ska_power_K",
    "param_learning_rate",
    # the fixed axes, which are singletons and still recorded
    "param_weight_decay", "param_warmup_ratio", "param_grad_clip",
    # resolved facts and measurements
    "attr_param_count", "attr_baseline_param_count", "attr_ska_layer_indices",
    "attr_per_device_batch_size", "attr_model_seed",
    "attr_tokens_per_sec", "attr_peak_memory_gib", "attr_n_eval_tokens",
    "attr_ska_delta",
    # provenance
    "attr_run_id", "attr_worker_id", "attr_sampler", "attr_sampler_seed",
)


def test_trials_csv_carries_every_column_the_questions_need(tmp_path):
    import csv

    from experimentation.sweep.search.report import write_trials_csv

    study, _trial, _outcome = _one_trial(tmp_path)
    path = write_trials_csv(study, tmp_path)
    columns = set(next(csv.reader(path.read_text().splitlines())))
    missing = sorted(c for c in _REQUIRED_COLUMNS if c not in columns)
    assert not missing, f"trials.csv is missing {missing}"


def test_a_pruned_trial_appears_in_trials_csv_with_its_reason(tmp_path):
    """Pruned, failed, OOM and completed have to be TELLABLE APART in the output,
    and nothing previously put a PRUNED trial through the CSV writer at all."""
    import csv

    from experimentation.sweep.search.report import write_trials_csv
    from experimentation.sweep.search.study import create_study, to_distributions

    context = _context(tmp_path)
    study = create_study(study_name="provenance", study_dir=tmp_path, seed=1)
    trial = study.ask(to_distributions(context["space"]))
    trial.set_user_attr("pruned", "pruned at step 450")
    study.tell(trial, state=optuna.trial.TrialState.PRUNED)

    rows = list(csv.DictReader(
        write_trials_csv(study, tmp_path).read_text().splitlines()))
    assert rows[0]["state"] == "PRUNED"
    assert "step 450" in rows[0]["attr_pruned"]


def test_an_exhausted_oom_ladder_is_labelled_as_such(tmp_path):
    """An OOM that consumed every rung must be distinguishable in `trials.csv`
    from a shape error that failed once. The distinguishing signal is the
    `failure` string plus the batch size that was reached."""
    from experimentation.sweep.search.driver import run_trial
    from experimentation.sweep.search.metrics import fixed_reader
    from experimentation.sweep.search.study import create_study, to_distributions

    class _AlwaysOOM:
        def submit(self, spec, run_dir, dry_run=False, resume=False, wait=False):
            (run_dir / "run.log").write_text("CUDA out of memory. Tried to "
                                             "allocate 2.00 GiB")
            raise RuntimeError("CUDA out of memory")

    context = _context(tmp_path)
    study = create_study(study_name="provenance", study_dir=tmp_path, seed=1)
    trial = study.ask(to_distributions(context["space"]))
    run_trial(study, trial, launcher=_AlwaysOOM(), batch_ladder=True,
              objective_reader_for=fixed_reader(lambda d: 1.0), **context)

    attrs = study.trials[0].user_attrs
    assert study.trials[0].state.name == "FAIL"
    assert "every microbatch rung failed" in attrs["failure"]
    assert attrs["per_device_batch_size"] >= 1


# -------------------------------------------- the general reconstruction ----

def test_a_trial_is_reconstructed_exactly_from_its_stored_metadata(tmp_path):
    """The general guard, and the point of every column above.

    Rebuild the RunSpec from `trial.params` plus the stored attrs and require the
    content hash to match. `run_id` is `sha256(model + data + optim + seed)`, so
    one equality here is a statement about every scientific field at once -- and
    it can only be made because `model_seed` is stored, since the seed is inside
    that hash.
    """
    from experimentation.run.spec import group_id, run_id
    from experimentation.sweep.search.space import params_to_overrides
    from experimentation.sweep.spec import build_cell_run_spec

    study, trial, outcome = _one_trial(tmp_path)
    attrs = trial.user_attrs

    overrides = params_to_overrides(trial.params, _context(tmp_path)["base_model"],
                                    max_steps=100, seq_len=None,
                                    backend_policy="exact_invchol")
    overrides["optim.max_steps"] = 100
    overrides["runtime.seed"] = int(attrs["model_seed"])
    sections = _base_sections(tmp_path)
    rebuilt = build_cell_run_spec("provenance", sections, overrides,
                                  schedules=sections.get("schedules"))

    assert run_id(rebuilt) == attrs["run_id"] == outcome.run_id
    assert group_id(rebuilt) == outcome.group_id
    assert list(rebuilt.model.ska_layer_indices) == list(
        attrs["ska_layer_indices"])
    assert int(rebuilt.model.param_count_estimate()) == attrs["param_count"]


def test_a_reference_repeat_reconstructs_to_its_own_run_id(tmp_path):
    """The reconstruction has to survive the one trial kind whose identity comes
    from a field that is not a search parameter."""
    from experimentation.run.spec import run_id
    from experimentation.sweep.search.anchors import Design
    from experimentation.sweep.search.driver import run_trial
    from experimentation.sweep.search.metrics import fixed_reader
    from experimentation.sweep.search.space import params_to_overrides
    from experimentation.sweep.search.study import (
        create_study, enqueue_anchors, to_distributions)
    from experimentation.sweep.spec import build_cell_run_spec

    context = _context(tmp_path)
    study = create_study(study_name="provenance", study_dir=tmp_path, seed=1)
    enqueue_anchors(study, [Design(name="ref-a", reference_group="ref"),
                            Design(name="ref-b", seed=101,
                                   reference_group="ref")],
                    context["base_model"], context["space"], base_lr=4e-4)
    for _ in range(2):
        trial = study.ask(to_distributions(context["space"]))
        run_trial(study, trial, launcher=_FakeLauncher(),
                  objective_reader_for=fixed_reader(lambda d: 1.0), **context)

    for trial in study.trials:
        attrs = trial.user_attrs
        overrides = params_to_overrides(trial.params, context["base_model"],
                                        max_steps=100, seq_len=None,
                                        backend_policy="exact_invchol")
        overrides["optim.max_steps"] = 100
        overrides["runtime.seed"] = int(attrs["model_seed"])
        sections = _base_sections(tmp_path)
        rebuilt = build_cell_run_spec("provenance", sections, overrides,
                                      schedules=sections.get("schedules"))
        assert run_id(rebuilt) == attrs["run_id"], attrs["anchor_name"]

    assert {t.user_attrs["model_seed"] for t in study.trials} == {42, 101}
    assert len({t.user_attrs["run_id"] for t in study.trials}) == 2
