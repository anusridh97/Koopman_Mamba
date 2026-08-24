"""Every new StudySpec field is APPLIED, not merely accepted and validated.

This file exists because of a measured hole, not a hypothetical one. With the
single line in `__main__.main` that calls `restrict_space` deleted -- so
`search_axes` and `fixed_params` were parsed, type-checked, domain-validated,
printed in the plan, and then had no effect on any trial -- **163 tests stayed
green**, including every test of `restrict_space` itself and every test of the
committed study config.

That is `HANDOFF-2026-08-21.md` §6's first named failure shape, verbatim: "Valid,
validated, and inert... Unit tests of a mechanism cannot see that nothing invokes
it." `max_steps` shipped that way once already (declared on StudySpec, applied
nowhere, so trials silently ran the base spec's length -- job 439754, a 2x
overspend with nothing in the output saying so), and `test_search_trial_budget.py`
is the test written afterwards. This is the same test for the fields added since.

The discipline, so a later field cannot slip through: for each one, assert on the
value that reaches the CONSUMER -- the `space` dict handed to `drive`, the sampler
optuna actually constructs -- rather than on the StudySpec, and name in the
docstring which line's deletion the assertion catches.
"""
from __future__ import annotations

import pathlib
import sys
import textwrap

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

optuna = pytest.importorskip("optuna")

pytestmark = pytest.mark.correctness


def _study_file(tmp_path, **extra):
    """A study on the real proxy base, restricted in a way nothing else is.

    The bounds below are deliberately DIFFERENT from both `space.py`'s defaults
    and the committed study's, so an assertion cannot pass by coincidence.
    """
    body = {
        "name": "apply-probe",
        "base": "configs/runs/proxy-256x17.yaml",
        "n_trials": 1,
        "max_steps": 300,
        "prune_after_step": 200,
        "launcher": "local",
        "run_root": str(tmp_path / "runs"),
        "search_axes": {
            "ska_rank": {"kind": "categorical", "choices": [16, 32]},
            "ska_ridge": {"kind": "float", "low": 0.004, "high": 0.006,
                          "log": True},
        },
        "fixed_params": {"weight_decay": 0.1, "grad_clip": 1.0,
                         "warmup_ratio": 0.04, "gamma_value": 1.0},
    }
    body.update(extra)
    import yaml
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "study.yaml"
    path.write_text(yaml.safe_dump(body, sort_keys=False))
    return path


def _run_main(study_path, *flags):
    """Call `main` with the driver and the git gate stubbed, and capture what
    reached `drive`.

    Patched at the DRIVER module rather than at `__main__`, because `__main__`
    imports `drive` inside the function body (deliberately -- everything above
    that import works with optuna absent).
    """
    from experimentation.sweep.search import __main__ as cli
    from experimentation.sweep.search import driver as driver_mod

    captured = {}

    def fake_drive(study, **kwargs):
        captured["study"] = study
        captured.update(kwargs)
        return []

    original_drive = driver_mod.drive
    original_gate = cli.check_git_clean
    driver_mod.drive = fake_drive
    cli.check_git_clean = lambda allow_dirty=False: False
    try:
        code = cli.main([str(study_path), "--force-no-eval", *flags])
    finally:
        driver_mod.drive = original_drive
        cli.check_git_clean = original_gate
    return code, captured


# ------------------------------------------ search_axes reaches the driver ----

def test_a_replaced_axis_reaches_the_space_the_driver_samples_from(tmp_path):
    """Catches deletion of: `space = restrict_space(...)` in `__main__.main`.

    Measured: with that one line reverted to a bare `search_space(...)` call,
    163 tests -- including every test of restrict_space -- stayed green.
    """
    code, captured = _run_main(_study_file(tmp_path))
    assert code == 0
    space = captured["space"]
    assert space["ska_rank"]["choices"] == [16, 32], (
        "the study's ska_rank restriction did not reach the driver; the trial "
        "would have sampled space.py's default choices instead")
    assert space["ska_ridge"]["low"] == 0.004
    assert space["ska_ridge"]["high"] == 0.006


def test_the_unrestricted_default_is_genuinely_different(tmp_path):
    """Guards the guard. If `search_space()` happened to return [16, 32] for
    ska_rank anyway, the assertion above would pass with the apply-line gone."""
    from experimentation.run.spec import resolve_model_config
    from experimentation.sweep.search.space import search_space
    from experimentation.sweep.spec import _base_sections

    base = resolve_model_config(
        _base_sections(str(REPO / "configs/runs/proxy-256x17.yaml"))["model"])
    default = search_space(base, base_name="configs/runs/proxy-256x17.yaml")
    assert default["ska_rank"]["choices"] != [16, 32]
    assert default["ska_ridge"]["low"] != 0.004
    assert "weight_decay" in default and len(
        default["weight_decay"]["choices"]) > 1


# ----------------------------------------- fixed_params reach the driver ----

def test_a_fixed_param_reaches_the_driver_as_a_singleton(tmp_path):
    """Catches deletion of the same line, via the other field. A fixed axis that
    did not reach the driver would be SAMPLED across its default choices, so
    'weight_decay is fixed at 0.1' would be false for most trials."""
    _, captured = _run_main(_study_file(tmp_path))
    space = captured["space"]
    for axis, value in (("weight_decay", 0.1), ("grad_clip", 1.0),
                        ("warmup_ratio", 0.04), ("gamma_value", 1.0)):
        assert space[axis] == {"kind": "categorical", "choices": [value]}, (
            f"{axis} reached the driver as {space[axis]}, not as a singleton")


def test_a_fixed_axis_survives_translation_into_an_optuna_distribution(tmp_path):
    """The singleton has to be something optuna can sample, or the study dies at
    trial 0 -- which would be loud, but only on a GPU."""
    from experimentation.sweep.search.study import to_distributions

    _, captured = _run_main(_study_file(tmp_path))
    distributions = to_distributions(captured["space"])
    assert distributions["weight_decay"].choices == (0.1,)
    assert distributions["weight_decay"].single() is True


def test_a_fixed_axis_reaches_the_resolved_run_spec(tmp_path):
    """One step further than the space: the value has to arrive in the
    RunSpec overrides, since a parameter that reaches the sampler and not the
    config is still inert."""
    from experimentation.run.spec import resolve_model_config
    from experimentation.sweep.search.space import params_to_overrides
    from experimentation.sweep.spec import _base_sections

    _, captured = _run_main(_study_file(tmp_path))
    space = captured["space"]
    point = {name: (decl["choices"][0] if decl["kind"] == "categorical"
                    else decl["low"]) for name, decl in space.items()}
    base = resolve_model_config(
        _base_sections(str(REPO / "configs/runs/proxy-256x17.yaml"))["model"])
    overrides = params_to_overrides(point, base, max_steps=300)
    assert overrides["optim.weight_decay"] == 0.1
    assert overrides["optim.grad_clip"] == 1.0
    assert overrides["optim.warmup_steps"] == 12          # 0.04 x 300
    assert overrides["model.ska_gamma_value"] == 1.0


# ------------------------------------------- the sampler fields reach optuna ----

def test_the_sampler_field_reaches_the_study_the_driver_is_given(tmp_path):
    """Catches deletion of `sampler=study_spec.sampler` from `study_kwargs`.

    Asserted on the Study object handed to `drive`, not on a `make_sampler` call:
    `make_sampler` has its own unit tests and they all passed while `__main__`
    could still have failed to pass the field.
    """
    _, captured = _run_main(_study_file(tmp_path, sampler="tpe_multivariate"))
    assert captured["study"].sampler._multivariate is True


def test_the_sampler_startup_field_reaches_optuna(tmp_path):
    """Catches deletion of `sampler_startup_trials=...` from `study_kwargs`."""
    _, captured = _run_main(
        _study_file(tmp_path, n_trials=50, sampler_startup_trials=17))
    assert captured["study"].sampler._n_startup_trials == 17


def test_the_prune_startup_field_reaches_optuna(tmp_path):
    """Catches deletion of `prune_startup_trials=...` from `study_kwargs`.

    This one was ALREADY a hole before this work: `create_study` accepted the
    argument and `__main__` never passed it, so the pruner always used the
    size-derived default however a study was written.
    """
    _, captured = _run_main(
        _study_file(tmp_path, n_trials=50, prune_startup_trials=11))
    assert captured["study"].pruner._n_startup_trials == 11


def test_the_derived_default_is_genuinely_different(tmp_path):
    """Guards the guard above: `min(6, max(3, 50 // 3))` is 6, not 11, so the
    assertion cannot pass with the field unpassed."""
    from experimentation.sweep.search.studyspec import (
        derived_prune_startup_trials)

    assert derived_prune_startup_trials(50) == 6


def test_a_random_sampler_study_gets_a_random_sampler(tmp_path):
    _, captured = _run_main(_study_file(tmp_path, sampler="random"))
    assert isinstance(captured["study"].sampler, optuna.samplers.RandomSampler)


# ------------------------------------- the per-worker sampler seed is applied ----

def test_the_sampler_seed_reaches_the_driver_as_provenance(tmp_path):
    """Catches deletion of `sampler_seed=sampler_seed` from the `drive` call.
    Without it the attr is never stamped and `trials.csv` has no
    `attr_sampler_seed` column -- so a concurrency artefact cannot be told from
    a real effect after the fact."""
    _, captured = _run_main(_study_file(tmp_path, seed=2026))
    assert captured["sampler_seed"] == 2026
    assert captured["sampler_name"] is not None


def test_a_worker_gets_its_own_sampler_seed_recorded(tmp_path, monkeypatch):
    """Catches deletion of `sampler_seed=sampler_seed` / `worker_id=worker_index`
    from the `drive` call -- i.e. the RECORDED provenance."""
    from experimentation.sweep.search.__main__ import WORKER_ENV

    monkeypatch.setenv(WORKER_ENV, "5")
    _, captured = _run_main(_study_file(tmp_path, seed=2026))
    assert captured["sampler_seed"] == 2031, "seed + worker index"
    assert captured["worker_id"] == 5


def _first_draws(tmp_path, worker, monkeypatch, n=5):
    """What the sampler optuna ACTUALLY built draws, for a given worker.

    Behavioural rather than introspective: `TPESampler` exposes no seed
    attribute (`_rng` is a `LazyRandomState` with no `_seed`), so the only honest
    way to ask "was this sampler seeded per-worker" is to make it sample. With no
    completed trials TPE is inside its startup window and delegates to its own
    seeded random sampler, so the first draws ARE the seeded stream.
    """
    from experimentation.sweep.search.__main__ import WORKER_ENV

    monkeypatch.setenv(WORKER_ENV, str(worker))
    _, captured = _run_main(_study_file(tmp_path, seed=2026))
    study = captured["study"]
    distributions = {"probe": optuna.distributions.FloatDistribution(0.0, 1.0)}
    return [study.ask(distributions).params["probe"] for _ in range(n)]


def test_the_sampler_optuna_built_is_actually_seeded_per_worker(tmp_path,
                                                               monkeypatch):
    """Catches deletion of `worker_id=worker_index` from the `create_study`
    kwargs -- the one apply-line the recorded-provenance test above does NOT
    cover, and the more dangerous of the two.

    Deleting it leaves `attr_sampler_seed` recording a per-worker seed while
    every worker's sampler is actually seeded with the study seed. The metadata
    would assert eight distinct streams and there would be one, so the fleet
    would draw eight copies of a single sequence -- exactly the gap per-worker
    seeding exists to close -- and `trials.csv` would say otherwise.
    """
    worker_0 = _first_draws(tmp_path / "w0", 0, monkeypatch)
    worker_5 = _first_draws(tmp_path / "w5", 5, monkeypatch)
    assert worker_0 != worker_5, (
        "two workers' samplers drew the SAME stream; the per-worker seed did "
        "not reach the sampler optuna constructed, so constant_liar is left "
        "repelling proposals that were identical by construction")


def test_the_same_worker_reproduces_its_own_stream(tmp_path, monkeypatch):
    """Guards the guard: if the draws were nondeterministic the test above would
    pass for the wrong reason, and a study would not be reproducible at all."""
    first = _first_draws(tmp_path / "a", 3, monkeypatch)
    second = _first_draws(tmp_path / "b", 3, monkeypatch)
    assert first == second


def test_the_model_seed_does_NOT_vary_with_the_worker(tmp_path, monkeypatch):
    """The constraint on the whole per-worker-seed idea. If the MODEL seed moved
    with the worker, a trial's result would depend on which worker happened to
    pull it, two identical proposals would get different run_ids, and the run
    directory would stop being content-addressed."""
    from experimentation.sweep.search.__main__ import WORKER_ENV

    seeds = {}
    for worker in ("0", "5"):
        monkeypatch.setenv(WORKER_ENV, worker)
        _, captured = _run_main(_study_file(tmp_path, seed=2026))
        seeds[worker] = captured["base_sections"]["runtime"]["seed"]
    assert seeds["0"] == seeds["5"] == 42, (
        f"the model/data seed moved with the worker: {seeds}")


# ----------------------------------------------- the budget, still applied ----

def test_max_steps_still_reaches_the_driver(tmp_path):
    """The original instance of this failure shape, kept in the same file as its
    descendants: `max_steps` was declared, validated, and applied nowhere, so
    trials ran the BASE spec's length (job 439754, a 2x overspend)."""
    _, captured = _run_main(_study_file(tmp_path, max_steps=300))
    assert captured["max_steps"] == 300


def test_every_declared_field_that_shapes_a_trial_is_asserted_somewhere():
    """The roster, discovered rather than hardcoded.

    A new StudySpec field is the thing this file exists to catch, so the field
    list is read off the dataclass and each name must appear in this module's
    source -- either asserted above or listed below as deliberately not
    trial-shaping. A field that is neither fails here, which is the only way a
    later addition cannot slip through silently.
    """
    from experimentation.sweep.search.studyspec import StudySpec

    #: Fields that do not shape a trial's RunSpec or the sampler, with the reason.
    NOT_TRIAL_SHAPING = {
        "name": "the study's label; reaches the run stamp, not the config",
        "base": "which spec is perturbed -- covered by every test above using it",
        "direction": "optuna study direction; only 'minimize' is meaningful",
        "storage": "where the journal lives",
        "design_file": "covered by test_proxy_anchor_design.py",
        "n_trials": "the loop bound -- covered by test_search_fleet_overshoot.py",
        "concurrent_trials": "the fleet -- covered by test_search_fanout.py",
        "launcher": "which launcher object is built",
        "run_root": "where run directories go",
        "logging_steps": "the pruner's interval -- covered by the startup tests",
        "prune_after_step": "the pruner's warmup -- covered by the startup tests",
        "trial_timeout_seconds": "the objective reader's timeout",
        "eval_data_dir": "set on the LAUNCHER, not the spec",
        "backend_policy": "covered by test_backend_geometry.py",
        "seq_len": "covered by test_search_space.py",
        "batch_ladder": "covered by test_search_driver.py",
        "objective": "covered by test_trial_metadata_and_penalty.py",
        "seed": "covered by the sampler-seed tests above",
    }
    source = pathlib.Path(__file__).read_text()
    unexplained = [
        name for name in StudySpec.__dataclass_fields__
        if name not in NOT_TRIAL_SHAPING and f'"{name}"' not in source
        and f"{name}=" not in source and f"'{name}'" not in source]
    assert not unexplained, (
        f"StudySpec field(s) {unexplained} are neither asserted to be applied "
        f"nor listed as not trial-shaping. Add an assertion that reads the "
        f"value off the CONSUMER, or add the field to NOT_TRIAL_SHAPING with "
        f"the reason and a pointer to whatever does cover it.")
