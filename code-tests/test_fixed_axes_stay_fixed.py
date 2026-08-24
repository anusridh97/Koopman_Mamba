"""Fixed axes stay fixed across many SAMPLED trials, not just enqueued anchors.

Every other test of `fixed_params` checks a static object: that the space holds a
singleton, that the singleton translates to a `CategoricalDistribution` with one
choice, that `params_to_overrides` reads the pinned value, that the 24 anchors
resolve onto it. All necessary; none of them exercises the sampler.

That is the gap this file closes, and it is not hypothetical. The whole point of
`fixed_params` becoming a validated SINGLETON rather than a deletion is that the
sampler still sees the axis -- so the sampler is what has to be observed. A
singleton `CategoricalDistribution` is an unusual thing to hand a multivariate
TPE: `TPESampler.sample_relative` under `group=True` explicitly SKIPS
distributions where `distribution.single()` is true, and if such an axis were
dropped from the returned params, `params_to_overrides` would raise a KeyError on
trial 0 -- or, worse, if a default were ever introduced, would silently train at
a weight decay nobody chose.

So: build the REAL study's space, ask a real optuna study for a few hundred
trials, and assert the three pinned axes take exactly one value each. Run for
both `tpe` and `tpe_multivariate`, and across the sampler's startup boundary --
TPE draws from its internal RandomSampler below `n_startup_trials` and from the
fitted model above it, which are two different code paths and only one of them is
exercised by a short study.
"""
from __future__ import annotations

import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

optuna = pytest.importorskip("optuna")

pytestmark = pytest.mark.correctness

STUDY = REPO / "configs/search/proxy-256x17-interactions-v1.yaml"

#: The study's own pinned values. Read from the committed file rather than
#: written here, so this cannot agree with a study that changed underneath it.
EXPECTED_FIXED = {"weight_decay": 0.1, "warmup_ratio": 0.04, "grad_clip": 1.0}


@pytest.fixture(scope="module")
def context():
    """(study_spec, base_model, space, distributions) from the committed files."""
    from experimentation.run.spec import resolve_model_config
    from experimentation.sweep.search.space import restrict_space, search_space
    from experimentation.sweep.search.study import to_distributions
    from experimentation.sweep.search.studyspec import load_study_spec
    from experimentation.sweep.spec import _base_sections

    study_spec = load_study_spec(STUDY)
    base_sections = _base_sections(str(REPO / study_spec.base))
    base_model = resolve_model_config(base_sections["model"])
    space = restrict_space(
        search_space(base_model, base_name=study_spec.base), base_model,
        axes=study_spec.search_axes, fixed=study_spec.fixed_params)
    return study_spec, base_model, space, to_distributions(space)


def test_the_study_really_pins_the_three_axes_this_file_checks(context):
    """Guards every assertion below: if the study stopped fixing these, the
    parametrised tests would pass by checking nothing."""
    study_spec = context[0]
    assert dict(study_spec.fixed_params) == EXPECTED_FIXED


def _sample(distributions, *, sampler, n, seed=2026, startup=None):
    from experimentation.sweep.search.study import make_sampler

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    # `startup` is dropped for `random`, which REFUSES it -- RandomSampler has no
    # startup window, so passing one would be silently ignored and `make_sampler`
    # raises rather than letting a study look configured when it is not. Caught by
    # that guard while writing this file, which is the guard working.
    if sampler == "random":
        startup = None
    study = optuna.create_study(
        sampler=make_sampler(seed=seed, sampler=sampler,
                             startup_trials=startup))
    drawn = []
    for index in range(n):
        trial = study.ask(distributions)
        drawn.append(dict(trial.params))
        # A real objective, so TPE has something to model once it leaves its
        # startup window -- a constant would make the fitted path degenerate and
        # this test would only ever exercise the random one.
        study.tell(trial, 1.0 + 0.001 * (index % 17))
    return drawn


@pytest.mark.parametrize("sampler", ["tpe", "tpe_multivariate", "random"])
def test_the_pinned_axes_take_one_value_across_two_hundred_sampled_trials(
        context, sampler):
    drawn = _sample(context[3], sampler=sampler, n=200, startup=20)
    assert len(drawn) == 200
    for axis, value in EXPECTED_FIXED.items():
        observed = {point[axis] for point in drawn}
        assert observed == {value}, (
            f"{sampler} sampled {axis} at {sorted(observed)}; the study pins it "
            f"to {value}")


@pytest.mark.parametrize("sampler", ["tpe", "tpe_multivariate"])
def test_a_pinned_axis_is_present_in_every_sampled_point(context, sampler):
    """Presence, separately from value. `sample_relative` DROPS `single()`
    distributions under `group=True`, and a dropped axis would reach
    `params_to_overrides` as a KeyError on trial 0 -- on a GPU, after a run
    directory had been claimed."""
    drawn = _sample(context[3], sampler=sampler, n=120, startup=20)
    for index, point in enumerate(drawn):
        for axis in EXPECTED_FIXED:
            assert axis in point, f"trial {index} has no {axis}"


@pytest.mark.parametrize("sampler", ["tpe", "tpe_multivariate"])
def test_the_pinned_axes_hold_on_BOTH_sides_of_the_startup_boundary(context,
                                                                   sampler):
    """TPE draws from its internal RandomSampler below `n_startup_trials` and
    from the fitted model above it. Two code paths; a short study exercises only
    the first, so a study with `sampler_startup_trials: 64` could behave
    differently at trial 65 than at trial 5."""
    startup = 15
    drawn = _sample(context[3], sampler=sampler, n=90, startup=startup)
    below, above = drawn[:startup], drawn[startup + 5:]
    assert below and above, "the split produced an empty half"
    for label, half in (("startup", below), ("modelled", above)):
        for axis, value in EXPECTED_FIXED.items():
            observed = {point[axis] for point in half}
            assert observed == {value}, (
                f"{sampler}, {label} phase: {axis} took {sorted(observed)}")


def test_the_non_fixed_axes_DO_vary_over_the_same_draws(context):
    """The essential counter-assertion. If the sampler were somehow returning a
    constant point every time, every test above would pass while measuring
    nothing at all."""
    drawn = _sample(context[3], sampler="tpe_multivariate", n=200, startup=20)
    varying = {axis for axis in drawn[0]
               if len({point[axis] for point in drawn}) > 1}
    expected = set(context[2]) - set(EXPECTED_FIXED)
    assert varying == expected, (
        f"axes that varied: {sorted(varying)}; expected the nine searched axes "
        f"{sorted(expected)}. An axis in `expected` and not in `varying` never "
        f"moved across 200 draws.")


def test_every_sampled_point_resolves_to_the_pinned_values_in_the_run_spec(
        context):
    """One step past the sampler: the pinned value has to arrive in the RunSpec
    overrides. An axis that reaches the sampler and not the config is inert."""
    from experimentation.sweep.search.space import params_to_overrides

    study_spec, base_model, _, distributions = context
    drawn = _sample(distributions, sampler=study_spec.sampler, n=60, startup=10)
    for index, point in enumerate(drawn):
        overrides = params_to_overrides(point, base_model,
                                        max_steps=study_spec.max_steps)
        assert overrides["optim.weight_decay"] == 0.1, index
        assert overrides["optim.grad_clip"] == 1.0, index
        # 0.04 x 600 steps, rounded -- the ratio only means anything against a
        # run length, which is why the driver applies max_steps.
        assert overrides["optim.warmup_steps"] == 24, index


def test_a_sampled_trial_and_an_enqueued_anchor_agree_about_the_fixed_axes(
        context, tmp_path):
    """The two ways a trial gets its params must not disagree. An anchor snaps
    onto the singleton and a sampled point draws it; if those produced different
    values, half the study would train at a weight decay nobody chose."""
    from experimentation.sweep.search.anchors import load_designs
    from experimentation.sweep.search.study import create_study, enqueue_anchors
    from experimentation.sweep.spec import _base_sections

    study_spec, base_model, space, distributions = context
    base_sections = _base_sections(str(REPO / study_spec.base))
    designs = load_designs(REPO / study_spec.design_file, minimum=24)

    study = create_study(
        study_name=study_spec.name, study_dir=tmp_path, seed=study_spec.seed,
        sampler=study_spec.sampler,
        sampler_startup_trials=study_spec.sampler_startup_trials,
        prune_after_step=study_spec.prune_after_step,
        prune_startup_trials=study_spec.prune_startup_trials,
        n_trials=study_spec.n_trials, logging_steps=study_spec.logging_steps)
    enqueue_anchors(study, designs, base_model, space,
                    base_lr=base_sections["optim"]["lr"])

    anchored, sampled = [], []
    for index in range(30):
        trial = study.ask(distributions)
        target = anchored if trial.user_attrs.get("anchor_name") else sampled
        target.append(dict(trial.params))
        study.tell(trial, 1.0 + index / 100.0)

    assert anchored, "no anchor was pulled"
    assert sampled, "no sampled trial was pulled -- raise the loop count"
    for axis, value in EXPECTED_FIXED.items():
        assert {p[axis] for p in anchored} == {value}, axis
        assert {p[axis] for p in sampled} == {value}, axis
