"""Repeated reference anchors: the study's own empirical noise floor.

**Why this file exists, stated before any test.** The proxy study ranks configs on
held-out loss after 600 steps, and job 445689 measured how big those differences
actually are: four trials on this base at 600 steps gave losses spanning 0.069,
with the best two 0.004 apart. If the study's own trial-to-trial reproducibility
noise is of that order, then no ranking over these axes means anything and no
sampler can repair it -- the study would be measuring its own seed.

(An earlier version of this docstring cited a 1.17e-4 SKA ablation delta from a
4m-geometry smoke study as the effect size to worry about. Job 445689 corrected
that: at the same `ska_layerscale_init` the proxy gives 1.17e-2, a hundred times
larger, and every delta is positive. The 4m number was a scale artefact. The
question this file exists for did not change -- only its magnitude did.)

So the study has to measure that noise, and the only honest way to measure it is
to run the SAME configuration several times differing ONLY in the training seed.
That is what a `reference_group` is: a set of designs whose every scientific
factor is identical and whose `seed` differs.

Three properties make it a controlled measurement rather than a gesture, and each
one is a test below:

  1. **Only the designated repeats vary the seed.** Every other trial -- anchor
     or sampled -- keeps the base spec's own `runtime.seed`, so ordinary trials
     stay comparable to each other. A study that varied the seed per trial would
     have folded the noise floor INTO every measurement instead of isolating it.

  2. **The sampler seed is untouched.** `sampler_seed_for` is per-worker and
     already handled; the model/data seed is a different number with a different
     job, and conflating them is how a search comes to depend on which worker
     happened to pull a trial.

  3. **A repeat is a distinct trial with a distinct run_id, and says so.**
     `runtime.seed` is inside `_scientific_payload(include_seed=True)`, so a
     seeded repeat hashes to its own `run_id` and its own directory, while
     `group_id` (which excludes the seed) stays shared -- which is exactly how the
     run system already spells "one experiment, N datapoints".

Property 3 has a sharp edge that a naive implementation gets wrong, and
`test_identical_params_are_not_collapsed_by_enqueue` is the guard: optuna's
`enqueue_trial(..., skip_if_exists=True)` deduplicates on PARAMS, and a repeat's
params are identical by construction (the seed is not a search axis). Measured
against optuna 4.9.0: three enqueues of `{'a': 1}` produce ONE waiting trial. So
anchor idempotency has to key on the anchor NAME, not on the params, or the noise
floor silently collapses to a single observation and the study reports a spread
of zero.
"""
import json

import pytest

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
        "study_name": "noise-floor",
        "base_lr": 4e-4,
    }
    context.update(overrides)
    return context


# ------------------------------------------------------- the Design field ----

def test_a_design_can_declare_a_training_seed():
    """Without this field nothing anywhere varies `runtime.seed` per trial, so
    the study has no way to observe its own reproducibility."""
    from experimentation.sweep.search.anchors import Design

    design = Design(name="reference-s101", seed=101)
    assert design.seed == 101


def test_a_design_inherits_the_base_specs_seed_by_default():
    """Every design that existed before this field must keep meaning exactly what
    it meant: one seed, the base spec's, for every trial."""
    from experimentation.sweep.search.anchors import Design

    assert Design(name="anything").seed == "baseline"


def test_a_design_can_name_the_replicate_set_it_belongs_to():
    from experimentation.sweep.search.anchors import Design

    design = Design(name="reference-s101", seed=101, reference_group="reference")
    assert design.reference_group == "reference"


def test_seed_and_reference_group_are_loadable_from_a_design_file(tmp_path):
    from experimentation.sweep.search.anchors import load_designs

    path = tmp_path / "designs.yaml"
    path.write_text(
        "designs:\n"
        "  - name: ref-a\n"
        "    reference_group: ref\n"
        "  - name: ref-b\n"
        "    seed: 101\n"
        "    reference_group: ref\n")
    designs = load_designs(path)
    assert [(d.name, d.seed, d.reference_group) for d in designs] == [
        ("ref-a", "baseline", "ref"), ("ref-b", 101, "ref")]


# ------------------------------------------- a group is a CONTROLLED set ----

def test_a_replicate_group_whose_members_differ_in_a_factor_is_refused(tmp_path):
    """The whole value of a noise floor is that the members differ in NOTHING
    except the seed. A group whose members differ in rank measures the rank
    effect and reports it as noise, which would make every effect below it look
    unresolvable -- the most damaging possible failure of this mechanism."""
    from experimentation.sweep.search.anchors import load_designs

    path = tmp_path / "designs.yaml"
    path.write_text(
        "designs:\n"
        "  - name: ref-a\n"
        "    reference_group: ref\n"
        "  - name: ref-b\n"
        "    seed: 101\n"
        "    reference_group: ref\n"
        "    rank: 8\n")
    with pytest.raises(ValueError, match="reference_group"):
        load_designs(path)


def test_the_refusal_names_the_field_that_differs(tmp_path):
    from experimentation.sweep.search.anchors import load_designs

    path = tmp_path / "designs.yaml"
    path.write_text(
        "designs:\n"
        "  - name: ref-a\n"
        "    reference_group: ref\n"
        "  - name: ref-b\n"
        "    seed: 101\n"
        "    reference_group: ref\n"
        "    ridge_factor: 3.0\n")
    with pytest.raises(ValueError, match="ridge_factor"):
        load_designs(path)


def test_two_members_of_a_group_may_not_share_a_seed(tmp_path):
    """Two identical designs at one seed are one datapoint recorded twice, and
    they would report a spread of zero -- i.e. claim perfect reproducibility."""
    from experimentation.sweep.search.anchors import load_designs

    path = tmp_path / "designs.yaml"
    path.write_text(
        "designs:\n"
        "  - name: ref-a\n"
        "    seed: 101\n"
        "    reference_group: ref\n"
        "  - name: ref-b\n"
        "    seed: 101\n"
        "    reference_group: ref\n")
    with pytest.raises(ValueError, match="seed"):
        load_designs(path)


def test_a_group_of_one_is_refused(tmp_path):
    """A single-member group cannot produce a spread, so it is a declaration
    that looks like a measurement."""
    from experimentation.sweep.search.anchors import load_designs

    path = tmp_path / "designs.yaml"
    path.write_text(
        "designs:\n"
        "  - name: ref-a\n"
        "    reference_group: ref\n")
    with pytest.raises(ValueError, match="at least two"):
        load_designs(path)


def test_a_seed_outside_any_group_is_still_allowed(tmp_path):
    """A one-off seed change is a legitimate thing to want; only a GROUP carries
    the controlled-replicate promise."""
    from experimentation.sweep.search.anchors import load_designs

    path = tmp_path / "designs.yaml"
    path.write_text("designs:\n  - name: solo\n    seed: 7\n")
    assert load_designs(path)[0].seed == 7


# ------------------------------------------------ resolution ignores it ----

def test_the_seed_is_not_a_search_parameter():
    """`resolve_design` produces the params dict optuna's distributions have to
    accept. A seed key there would be rejected at enqueue time, so the seed
    travels as trial metadata instead."""
    from koopman_lm.config import build_config
    from experimentation.sweep.search.anchors import Design, resolve_design
    from experimentation.sweep.search.space import search_space

    cfg = build_config("50m")
    space = search_space(cfg, base_name="50m")
    params = resolve_design(Design(name="r", seed=101, reference_group="ref"),
                            cfg, space, base_lr=4e-4)
    assert "seed" not in params
    assert "reference_group" not in params


def test_two_members_of_a_group_resolve_to_identical_params():
    """Which is exactly why enqueue cannot deduplicate on params."""
    from koopman_lm.config import build_config
    from experimentation.sweep.search.anchors import Design, resolve_design
    from experimentation.sweep.search.space import search_space

    cfg = build_config("50m")
    space = search_space(cfg, base_name="50m")
    a = resolve_design(Design(name="a", reference_group="ref"), cfg, space,
                       base_lr=4e-4)
    b = resolve_design(Design(name="b", seed=101, reference_group="ref"), cfg,
                       space, base_lr=4e-4)
    assert a == b


# --------------------------------------------------------------- enqueue ----

def test_identical_params_are_not_collapsed_by_enqueue(tmp_path):
    """The load-bearing test of this whole file.

    Measured against optuna 4.9.0: `enqueue_trial(params, skip_if_exists=True)`
    three times with identical params leaves ONE waiting trial. A repeat's params
    ARE identical, so anchor idempotency must key on the anchor NAME. Without
    this the noise floor is a single observation and the study reports a spread
    of zero -- a claim of perfect reproducibility, from one run.
    """
    from experimentation.sweep.search.anchors import Design
    from experimentation.sweep.search.study import create_study, enqueue_anchors

    context = _context(tmp_path)
    study = create_study(study_name="noise-floor", study_dir=tmp_path, seed=1)
    designs = [Design(name="ref-a", reference_group="ref"),
               Design(name="ref-b", seed=101, reference_group="ref"),
               Design(name="ref-c", seed=102, reference_group="ref")]
    added = enqueue_anchors(study, designs, context["base_model"],
                            context["space"], base_lr=4e-4)

    assert added == 3
    assert len(study.trials) == 3
    assert sorted(t.user_attrs["anchor_name"] for t in study.trials) == [
        "ref-a", "ref-b", "ref-c"]


def test_the_seed_reaches_the_trial_as_metadata(tmp_path):
    from experimentation.sweep.search.anchors import Design
    from experimentation.sweep.search.study import (
        REFERENCE_GROUP_ATTR, SEED_ATTR, create_study, enqueue_anchors)

    context = _context(tmp_path)
    study = create_study(study_name="noise-floor", study_dir=tmp_path, seed=1)
    enqueue_anchors(study, [Design(name="ref-b", seed=101,
                                   reference_group="ref")],
                    context["base_model"], context["space"], base_lr=4e-4)

    attrs = study.trials[0].user_attrs
    assert attrs[SEED_ATTR] == 101
    assert attrs[REFERENCE_GROUP_ATTR] == "ref"


def test_an_unseeded_design_carries_no_seed_attr(tmp_path):
    """Absent, not the base value: the driver must be able to tell "this trial
    designates a seed" from "this trial inherits whatever the base says", and a
    stamped copy of the base's seed makes those two states identical."""
    from experimentation.sweep.search.anchors import Design
    from experimentation.sweep.search.study import (
        SEED_ATTR, create_study, enqueue_anchors)

    context = _context(tmp_path)
    study = create_study(study_name="noise-floor", study_dir=tmp_path, seed=1)
    enqueue_anchors(study, [Design(name="plain")], context["base_model"],
                    context["space"], base_lr=4e-4)

    assert SEED_ATTR not in study.trials[0].user_attrs


def test_enqueueing_the_same_designs_twice_adds_nothing(tmp_path):
    """Resume safety. Re-running the supervisor against a live journal must not
    re-run 24 anchors, and keying idempotency on the name rather than the params
    must not have cost that."""
    from experimentation.sweep.search.anchors import Design
    from experimentation.sweep.search.study import create_study, enqueue_anchors

    context = _context(tmp_path)
    study = create_study(study_name="noise-floor", study_dir=tmp_path, seed=1)
    designs = [Design(name="ref-a", reference_group="ref"),
               Design(name="ref-b", seed=101, reference_group="ref")]
    enqueue_anchors(study, designs, context["base_model"], context["space"],
                    base_lr=4e-4)
    added = enqueue_anchors(study, designs, context["base_model"],
                            context["space"], base_lr=4e-4)

    assert added == 0
    assert len(study.trials) == 2


# ---------------------------------------------------------------- driver ----

def test_a_seeded_anchor_trains_at_its_own_seed(tmp_path):
    from experimentation.sweep.search.anchors import Design
    from experimentation.sweep.search.driver import run_trial
    from experimentation.sweep.search.metrics import fixed_reader
    from experimentation.sweep.search.study import (
        create_study, enqueue_anchors, to_distributions)

    context = _context(tmp_path)
    study = create_study(study_name="noise-floor", study_dir=tmp_path, seed=1)
    enqueue_anchors(study, [Design(name="ref-b", seed=101,
                                   reference_group="ref")],
                    context["base_model"], context["space"], base_lr=4e-4)
    trial = study.ask(to_distributions(context["space"]))
    launcher = _FakeLauncher()
    run_trial(study, trial, launcher=launcher,
              objective_reader_for=fixed_reader(lambda d: 1.0), **context)

    spec, _run_dir = launcher.submitted[0]
    assert spec.runtime.seed == 101


def test_an_ordinary_trial_keeps_the_base_specs_seed(tmp_path):
    """The isolation property. If a sampled trial's seed moved too, the noise
    floor would be inside every measurement instead of beside them."""
    from experimentation.sweep.search.driver import run_trial
    from experimentation.sweep.search.metrics import fixed_reader
    from experimentation.sweep.search.study import create_study, to_distributions

    context = _context(tmp_path)
    study = create_study(study_name="noise-floor", study_dir=tmp_path, seed=1)
    trial = study.ask(to_distributions(context["space"]))
    launcher = _FakeLauncher()
    run_trial(study, trial, launcher=launcher,
              objective_reader_for=fixed_reader(lambda d: 1.0), **context)

    spec, _run_dir = launcher.submitted[0]
    assert spec.runtime.seed == 42


def test_repeats_get_distinct_run_ids_and_a_shared_group_id(tmp_path):
    """`runtime.seed` is inside `_scientific_payload(include_seed=True)` and
    outside `group_id`, so the run system already spells this correctly: one
    experiment, N datapoints. Asserted here because the noise floor depends on
    the repeats being separate runs rather than one directory overwritten."""
    from experimentation.sweep.search.anchors import Design
    from experimentation.sweep.search.driver import run_trial
    from experimentation.sweep.search.metrics import fixed_reader
    from experimentation.sweep.search.study import (
        create_study, enqueue_anchors, to_distributions)

    context = _context(tmp_path)
    study = create_study(study_name="noise-floor", study_dir=tmp_path, seed=1)
    enqueue_anchors(study, [Design(name="ref-a", reference_group="ref"),
                            Design(name="ref-b", seed=101,
                                   reference_group="ref"),
                            Design(name="ref-c", seed=102,
                                   reference_group="ref")],
                    context["base_model"], context["space"], base_lr=4e-4)
    outcomes = []
    for _ in range(3):
        trial = study.ask(to_distributions(context["space"]))
        outcomes.append(run_trial(
            study, trial, launcher=_FakeLauncher(),
            objective_reader_for=fixed_reader(lambda d: 1.0), **context))

    assert len({o.run_id for o in outcomes}) == 3
    assert len({o.group_id for o in outcomes}) == 1
    assert len({str(o.run_dir) for o in outcomes}) == 3


def test_the_trial_records_the_seed_it_actually_trained_at(tmp_path):
    """`model_seed` is the RESOLVED value off the spec, recorded for every trial
    -- seeded or not. Reconstructing a trial from `trials.csv` needs the seed
    that ran, and "no attr" would leave a reader guessing at the base spec."""
    from experimentation.sweep.search.driver import run_trial
    from experimentation.sweep.search.metrics import fixed_reader
    from experimentation.sweep.search.study import create_study, to_distributions

    context = _context(tmp_path)
    study = create_study(study_name="noise-floor", study_dir=tmp_path, seed=1)
    trial = study.ask(to_distributions(context["space"]))
    run_trial(study, trial, launcher=_FakeLauncher(),
              objective_reader_for=fixed_reader(lambda d: 1.0), **context)

    assert study.trials[0].user_attrs["model_seed"] == 42


def test_a_repeat_is_identifiable_as_one_in_the_trial_record(tmp_path):
    from experimentation.sweep.search.anchors import Design
    from experimentation.sweep.search.driver import run_trial
    from experimentation.sweep.search.metrics import fixed_reader
    from experimentation.sweep.search.study import (
        REFERENCE_GROUP_ATTR, create_study, enqueue_anchors, to_distributions)

    context = _context(tmp_path)
    study = create_study(study_name="noise-floor", study_dir=tmp_path, seed=1)
    enqueue_anchors(study, [Design(name="ref-b", seed=101,
                                   reference_group="ref")],
                    context["base_model"], context["space"], base_lr=4e-4)
    trial = study.ask(to_distributions(context["space"]))
    run_trial(study, trial, launcher=_FakeLauncher(),
              objective_reader_for=fixed_reader(lambda d: 1.0), **context)

    assert study.trials[0].user_attrs[REFERENCE_GROUP_ATTR] == "ref"


def test_the_reference_group_reaches_trials_csv(tmp_path):
    from experimentation.sweep.search.anchors import Design
    from experimentation.sweep.search.driver import run_trial
    from experimentation.sweep.search.metrics import fixed_reader
    from experimentation.sweep.search.report import trial_row
    from experimentation.sweep.search.study import (
        create_study, enqueue_anchors, to_distributions)

    context = _context(tmp_path)
    study = create_study(study_name="noise-floor", study_dir=tmp_path, seed=1)
    enqueue_anchors(study, [Design(name="ref-b", seed=101,
                                   reference_group="ref")],
                    context["base_model"], context["space"], base_lr=4e-4)
    trial = study.ask(to_distributions(context["space"]))
    run_trial(study, trial, launcher=_FakeLauncher(),
              objective_reader_for=fixed_reader(lambda d: 1.0), **context)

    row = trial_row(study.trials[0])
    assert row["attr_reference_group"] == "ref"
    assert row["attr_model_seed"] == 101


def test_the_sampler_seed_is_untouched_by_a_reference_repeat():
    """Two different numbers with two different jobs. `sampler_seed_for` is
    per-worker provenance for the PROPOSAL stream; `model_seed` is what seeds
    initialisation and batch order. Conflating them would make a trial's result
    depend on which worker pulled it."""
    from experimentation.sweep.search.study import sampler_seed_for

    assert sampler_seed_for(2026, 3) == 2029
    assert sampler_seed_for(2026, None) == 2026
