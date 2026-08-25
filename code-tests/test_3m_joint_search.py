"""`scripts/search_3m_joint.py`: the arithmetic and the wiring, not the loop.

The script exists because `space.py` has no macro-architecture axes and adding
them there would widen the space every other study inherits by omission. That
choice puts four axes outside the module the rest of the suite tests, so the two
things nothing else can catch are checked here:

**The parameter band.** The whole study is "at a fixed ~3M budget, which
architecture wins". If a sampled combination leaves 2.8M-3.25M, the answer is
partly "the one that was allowed to be bigger", and there is no way to tell after
the fact. So the band is asserted over the EXHAUSTIVE size-bearing product, the
same way `--dry_run` checks it before a launch.

**The wiring.** This repo's recurring failure is a field that is accepted,
validated, hashed -- and read by nothing. A macro axis that did not reach the
RunSpec would produce a study whose `trials.csv` reports depths that never ran.
So the tests below resolve a sampled point all the way to a `RunSpec` and assert
the model it describes, including the two DERIVED things: SKA layer placement
must come from the sampled depth (not the base spec's 13) and the backend head
geometry from the sampled head count.

Not tested here: the trial loop, the fanout and the report writing. Those are
`driver.py`, `__main__.py` and `report.py` unmodified, and they have their own
tests -- duplicating them would assert that an import statement works.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from experimentation.run.resolve import resolve_model_config
from experimentation.sweep.search.geometry import make_layer_indices
from experimentation.sweep.search.space import (
    params_to_overrides, restrict_space, search_space)
from experimentation.sweep.search.studyspec import load_study_spec
from experimentation.sweep.spec import _base_sections, build_cell_run_spec

STUDY = REPO / "configs" / "search" / "3m-joint-v1.yaml"


def _script():
    spec = importlib.util.spec_from_file_location(
        "search_3m_joint", REPO / "scripts" / "search_3m_joint.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def study():
    """The committed study, its base, and the space the script would build.

    The space is restricted against the SHALLOWEST depth exactly as the script
    does it, because that is the depth an `n_ska_layers` choice has to survive.
    """
    from dataclasses import replace

    script = _script()
    spec = load_study_spec(STUDY)
    sections = _base_sections(spec.base)
    base_model = resolve_model_config(sections["model"])
    shallowest = replace(
        base_model, n_layers=script.MIN_DEPTH,
        ska_layer_indices=tuple(make_layer_indices(
            script.MIN_DEPTH, len(base_model.ska_layer_indices), "even",
            list(base_model.ska_layer_indices))))
    space = restrict_space(search_space(shallowest, base_name=spec.base),
                           shallowest, axes=spec.search_axes,
                           fixed=spec.fixed_params)
    return script, spec, sections, base_model, space


def _sample(script, space, **overrides):
    """One complete sampled point: every macro axis and every declared axis."""
    params = {name: decl["choices"][0] if decl["kind"] == "categorical"
              else decl["low"]
              for name, decl in {**script.MACRO_AXES, **space}.items()}
    params.update(overrides)
    return params


def _resolve(script, spec, sections, base_model, params):
    """A sampled point -> the RunSpec a trial would actually train.

    Deliberately the script's `apply_macro` followed by the UNMODIFIED
    `params_to_overrides` and `build_cell_run_spec`, in the same order
    `driver.run_trial` calls them. A helper that reimplemented the composition
    could pass while the script's own path was broken.
    """
    trial_sections, trial_model = script.apply_macro(sections, base_model, params)
    overrides = params_to_overrides(params, trial_model,
                                    max_steps=spec.max_steps,
                                    seq_len=spec.seq_len,
                                    backend_policy=spec.backend_policy)
    overrides["optim.max_steps"] = int(spec.max_steps)
    return build_cell_run_spec(spec.name, trial_sections, overrides,
                              schedules=trial_sections.get("schedules"))


# ------------------------------------------------------------- the param band

def test_every_size_bearing_combination_is_inside_the_band(study):
    """The study's central claim, checked exhaustively rather than at corners.

    Break `DEPTHS` by one layer -- say (3, 'full') to 12 -- and this fails with
    the offending cell named. That is the point: a band violation found on trial
    1400 has already spent the allocation.
    """
    script, _spec, _sections, base_model, space = study
    violations, checked, lo, hi = script.check_band(base_model, space)
    assert violations == [], f"{len(violations)} of {checked}: {violations[:5]}"
    assert checked == (
        len(script.MACRO_AXES["mamba_expand"]["choices"])
        * len(script.MACRO_AXES["depth_tier"]["choices"])
        * len(script.MACRO_AXES["d_state"]["choices"])
        * len(script.MACRO_AXES["ska_n_heads"]["choices"])
        * len(space["ska_rank"]["choices"])
        * len(space["n_ska_layers"]["choices"])
        * len(space["beta_policy"]["choices"])), (
        "check_band skipped part of its own product")
    assert script.PARAM_BAND[0] <= lo <= hi <= script.PARAM_BAND[1]


def test_the_expansion_axis_is_iso_parameter_within_a_tier(study):
    """Under 1% spread across expansions at fixed tier.

    This is what makes `mamba_expand` a claim about architecture rather than
    about capacity. If it drifts past 1%, the axis still runs -- it just stops
    being interpretable, and nothing else in the suite would notice.
    """
    script, _spec, _sections, base_model, space = study
    cells = script.tier_table(base_model, space)
    for tier in script.MACRO_AXES["depth_tier"]["choices"]:
        means = [mean for (_expand, t), (_depth, mean) in cells.items()
                 if t == tier]
        spread = (max(means) - min(means)) / min(means)
        assert spread < 0.01, f"tier {tier!r} spans {100 * spread:.2f}%"


def test_the_tier_axis_is_a_real_capacity_step(study):
    """And the OTHER half: the tiers must differ, or the axis measures nothing.

    `depth_tier` is in the space to give the study its own loss-per-parameter
    slope, which is what calibrates the 1-2.5% capacity side effects of rank,
    head count and SKA layer count. Two tiers at the same size would leave every
    one of those effects uninterpretable.
    """
    script, _spec, _sections, base_model, space = study
    cells = script.tier_table(base_model, space)
    for expand in script.MACRO_AXES["mamba_expand"]["choices"]:
        lean = cells[(expand, "lean")][1]
        full = cells[(expand, "full")][1]
        assert 0.01 < (full - lean) / lean < 0.05, (
            f"expand {expand}: lean {lean:,.0f} -> full {full:,.0f}")


# ----------------------------------------------------------------- the wiring

def test_macro_axes_reach_the_resolved_run_spec(study):
    """Every macro axis lands on the model a trial would build.

    The failure this exists for: a sampled value that is recorded in the journal
    and applied to nothing, so `trials.csv` describes models that never ran.
    """
    script, spec, sections, base_model, space = study
    params = _sample(script, space, mamba_expand=3, depth_tier="full",
                     d_state=32, ska_n_heads=8, ska_rank=16, n_ska_layers=4)
    model = _resolve(script, spec, sections, base_model, params).model

    assert model.mamba_expand == 3
    assert model.n_layers == script.DEPTHS[(3, "full")]
    assert model.d_state == 32
    assert model.ska_n_heads == 8
    assert model.ska_rank == 16
    # And the base spec's own values are genuinely gone, not coincidentally
    # equal: this assertion is what makes the four above mean something.
    assert base_model.n_layers != model.n_layers
    assert base_model.d_state != model.d_state


def test_ska_placement_follows_the_sampled_depth_not_the_base_depth(study):
    """`params_to_overrides` derives indices from the base model it is GIVEN.

    That is the single seam the whole script rests on -- it is why no change to
    `space.py` was needed. If it regressed, an 11-layer trial would inherit the
    13-layer base's `[2, 5, 8, 11]`: index 11 is the last layer of an 11-layer
    backbone, which `geometry` reserves, so the study would be silently placing
    SKA where it declares it does not.
    """
    script, spec, sections, base_model, space = study
    params = _sample(script, space, mamba_expand=3, depth_tier="full",
                     n_ska_layers=4, placement="even")
    model = _resolve(script, spec, sections, base_model, params).model

    depth = script.DEPTHS[(3, "full")]
    assert model.n_layers == depth
    assert list(model.ska_layer_indices) == make_layer_indices(
        depth, 4, "even", list(base_model.ska_layer_indices))
    assert len(model.ska_layer_indices) == 4, "the layer count clamped"
    assert max(model.ska_layer_indices) < depth - 1, (
        "SKA landed on the final layer, which geometry.py reserves")
    assert list(base_model.ska_layer_indices) != list(model.ska_layer_indices)


def test_every_macro_level_moves_the_resolved_spec(study):
    """No dead levels: each value of each macro axis gives a distinct model.

    The general form of the check `test_search_axis_restriction.py` runs on the
    declared axes. A level that resolves to the same config as its neighbour is a
    trial budget spent on a difference that does not exist, and a parameter
    importance computed over it.
    """
    script, spec, sections, base_model, space = study
    for axis, decl in script.MACRO_AXES.items():
        seen = {}
        for level in decl["choices"]:
            params = _sample(script, space, **{axis: level})
            # The dataclass repr, not a hand-picked field list: a level that
            # moved only a field the list forgot would pass a narrower check.
            key = repr(_resolve(script, spec, sections, base_model,
                                params).model)
            assert key not in seen, (
                f"{axis}={level!r} resolves identically to "
                f"{axis}={seen[key]!r}")
            seen[key] = level


def test_the_backend_head_geometry_follows_the_sampled_head_count(study):
    """`_backend_overrides` is passed head_dim from the SAMPLED head count.

    It matters even though this study pins `exact_invchol` for every trial:
    `fused_only` requires value width exactly 64, and a future variant of this
    study that asked for it while computing head_dim off the base config would
    silently accept an unsupported geometry.
    """
    script, spec, sections, base_model, space = study
    for heads in script.MACRO_AXES["ska_n_heads"]["choices"]:
        params = _sample(script, space, ska_n_heads=heads)
        _sections, trial_model = script.apply_macro(sections, base_model, params)
        assert trial_model.head_dim == trial_model.d_model // heads


# -------------------------------------------------------- the committed study

def test_the_committed_study_declares_no_macro_axis(study):
    """`restrict_space` would reject them, and the script refuses the overlap.

    Both halves matter. A macro axis in `search_axes` is an unknown axis to
    `space.py` and fails loudly. A macro axis that space.py has GROWN, on the
    other hand, would be sampled twice -- once by each declaration -- and the
    script's collision check is what turns that into an error instead of a study
    whose journal disagrees with itself.
    """
    script, spec, _sections, _base_model, space = study
    assert not set(script.MACRO_AXES) & set(spec.search_axes)
    assert not set(script.MACRO_AXES) & set(spec.fixed_params)
    assert not set(script.MACRO_AXES) & set(space), (
        "space.py has grown a macro axis; delete it from the script rather "
        "than sampling it twice")


def test_the_committed_study_excludes_baseline_placement(study):
    """With depth searched there is no base backbone for `baseline` to reproduce.

    `make_layer_indices` does not raise on it -- it interpolates across the base
    indices clipped to the sampled depth, which is a near-duplicate of `even`.
    So this is a dead level rather than a crash, and the script refuses it.
    """
    _script_mod, spec, _sections, _base_model, space = study
    assert "baseline" not in space["placement"]["choices"]
    assert "baseline" not in spec.search_axes["placement"]["choices"]


def test_the_committed_study_declares_an_empty_objective(study):
    """A parameter penalty cannot mean what it says here.

    `driver._record_trial_attrs` stamps `baseline_param_count` from the base
    model it is given, and the script gives it the PER-TRIAL macro base -- so
    baseline and actual are the same number and the penalty is identically zero.
    Silently zero is worse than refused, so `main` refuses it.
    """
    _script_mod, spec, _sections, _base_model, _space = study
    assert spec.objective == {}


# --------------------------------------------------- the sampler's real draws

def test_real_tpe_draws_all_resolve_to_launchable_run_specs(study, tmp_path):
    """The same integration the exhaustive band check cannot do: real draws.

    `check_band` crosses the size-bearing axes and holds the rest at one
    representative, which is what makes it exhaustive and cheap. This does the
    complement -- optuna's own sampler, over all 17 dimensions, through
    `apply_macro` -> `params_to_overrides` -> `build_cell_run_spec`, the exact
    composition `driver.run_trial` performs -- and asserts each result is a spec
    that could actually be launched.

    Cheap enough to be worth 120 draws: nothing here touches a GPU, a filesystem
    or a data shard. It is the test that would have caught a distribution the
    sampler can emit but the resolver cannot consume.
    """
    optuna = pytest.importorskip("optuna",
                                 reason="optuna is an optional dependency")
    from experimentation.run.spec import run_id
    from experimentation.sweep.search.study import create_study, to_distributions

    script, spec, sections, base_model, space = study
    distributions = to_distributions({**script.MACRO_AXES, **space})
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    opt_study = create_study(study_name="probe", study_dir=tmp_path, seed=7,
                            sampler=spec.sampler,
                            sampler_startup_trials=spec.sampler_startup_trials,
                            n_trials=120)

    run_ids, counts = set(), []
    for i in range(120):
        trial = opt_study.ask(distributions)
        resolved = _resolve(script, spec, sections, base_model, trial.params)
        counts.append(resolved.model.param_count_estimate())

        assert script.PARAM_BAND[0] <= counts[-1] <= script.PARAM_BAND[1], (
            f"{counts[-1]:,} parameters from {trial.params}")
        # The layer count the sampler asked for is the layer count that ran.
        assert (len(resolved.model.ska_layer_indices)
                == trial.params["n_ska_layers"])
        assert max(resolved.model.ska_layer_indices) < resolved.model.n_layers - 1
        # One backend for the whole study. `exact_auto` would put three of four
        # ranks on a 167x-738x slower path, correlated with a searched axis.
        assert resolved.model.ska_inverse_cholesky
        assert not resolved.model.ska_prefix_scan
        # The study's budget, not the base spec's.
        assert resolved.optim.max_steps == spec.max_steps
        assert resolved.model.max_seq_len == spec.seq_len
        run_ids.add(run_id(resolved))
        # Descending losses so the multivariate sampler actually leaves its
        # startup window and proposes from a fitted model rather than uniformly.
        opt_study.tell(trial, 5.0 - i * 0.001)

    # Distinct identities: a study whose draws collided would silently train one
    # config N times and record N trials.
    assert len(run_ids) > 100, f"only {len(run_ids)} distinct run_ids in 120 draws"


def test_the_token_budget_is_seventy_per_parameter(study):
    """The requested budget, as arithmetic rather than as a comment.

    Held CONSTANT across the band on purpose: the space spans ~12% in parameter
    count, and scaling tokens with size would confound architecture with data.
    """
    _script_mod, spec, sections, base_model, _space = study
    tokens = (int(sections["optim"]["effective_batch"])
              * int(spec.seq_len) * int(spec.max_steps))
    assert tokens == 209_977_344
    assert 68 <= tokens / base_model.param_count_estimate() <= 72
