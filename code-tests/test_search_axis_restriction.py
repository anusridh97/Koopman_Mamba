"""Per-study restriction of the declared space: narrow an axis, or pin it.

`space.search_space()` is what the REPO declares. A study may replace an axis's
declaration or fix a parameter to one value, and nothing else -- there is no
generic override dict, because `params_to_overrides` requires twelve named
parameters and `KoopmanLMConfig` asserts on four of them, so an unvalidated
passthrough converts a YAML typo into a failed trial on a GPU after a run
directory has been claimed.

Three groups of assertions, in order of how much it would cost to have them
wrong:

**Absence is exactly the old behaviour.** Every committed study says nothing
about axes, so `restrict_space(space, base)` with no arguments has to be the
identity or 4m-adaptive's space silently changed.

**A restriction that cannot work is refused here, not on a GPU.** An unknown
axis, a rank that is not a multiple of 8, a layer count past the usable window,
a placement geometry.py does not declare, a float range on a counting axis, a
fixed value outside its own declaration.

**No dead axes.** The one that motivated the whole mechanism: `make_layer_indices`
CLAMPS a too-large count instead of raising, so `n_ska_layers: [2, 4, 8]` on a
four-layer backbone resolves 4 and 8 to the SAME indices -- two choices, one
config, and a parameter importance computed over a difference that does not
exist. `test_every_non_fixed_axis_can_move_the_resolved_spec` is the general
form of that check and is run against the real interaction study's space.
"""
from __future__ import annotations

import copy
import math
import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from koopman_lm.config import KoopmanLMConfig                       # noqa: E402
from experimentation.sweep.search import space as space_mod         # noqa: E402
from experimentation.sweep.search.geometry import PLACEMENTS        # noqa: E402
from experimentation.sweep.search.space import (                    # noqa: E402
    REQUIRED_PARAMS, dropped_base_values, params_to_overrides, restrict_space,
    search_space)

pytestmark = pytest.mark.correctness


def _base_model(**kw) -> KoopmanLMConfig:
    """The proxy geometry, which is what the target study actually uses."""
    fields = dict(
        d_model=256, n_layers=17, vocab_size=32000, d_state=48, d_conv=4,
        mamba_expand=2, mamba_headdim=32, ska_n_heads=4,
        ska_rank=24, ska_ridge=0.01, ska_power_K=1,
        ska_layer_indices=[3, 7, 11, 15], ska_mode="parallel",
        ska_prefix_scan=False, ska_inverse_cholesky=True,
        ska_exact_intrachunk=False,
        ska_norm_clip=True, ska_norm_clip_c=4.0,
        ska_layerscale=True, ska_layerscale_init=0.01,
        max_seq_len=1024, mlp_type="swiglu", mlp_expand=2.667,
    )
    fields.update(kw)
    return KoopmanLMConfig(**fields)


# ------------------------------------------------------- absence is identity ----

def test_no_restriction_is_the_identity():
    """Every committed study says nothing about axes. If this is not the
    identity, 4m-adaptive's space changed without anyone editing it."""
    base = _base_model()
    declared = search_space(base)
    assert restrict_space(declared, base) == dict(declared)


def test_none_arguments_behave_like_empty_ones():
    base = _base_model()
    declared = search_space(base)
    assert restrict_space(declared, base, axes=None, fixed=None) == dict(declared)


def test_the_input_space_is_not_mutated():
    """`restrict_space` is called on the space the plan printer also holds."""
    base = _base_model()
    declared = search_space(base)
    before = copy.deepcopy(declared)
    restrict_space(declared, base,
                   axes={"ska_rank": {"kind": "categorical", "choices": [8, 16]}},
                   fixed={"grad_clip": 1.0})
    assert declared == before


# ------------------------------------------------------------ every axis is guarded ----

def test_every_declared_axis_has_a_domain_validator():
    """Guards the guard. An axis with no entry in `_AXIS_DOMAINS` would KeyError
    the moment a study tried to restrict it -- or, worse, if the lookup were
    written with `.get`, would accept anything."""
    declared = search_space(_base_model())
    assert set(declared) == set(space_mod._AXIS_DOMAINS), (
        "search_space() and _AXIS_DOMAINS disagree about which axes exist")


def test_required_params_is_exactly_what_the_space_declares():
    assert set(REQUIRED_PARAMS) == set(search_space(_base_model()))


# ------------------------------------------------------------------ refusals ----

def test_an_unknown_axis_is_refused_and_the_message_lists_the_real_ones():
    base = _base_model()
    with pytest.raises(ValueError) as exc:
        restrict_space(search_space(base), base,
                       axes={"ska_ridge_": {"kind": "float", "low": 1.0,
                                            "high": 2.0}})
    assert "ska_ridge_" in str(exc.value) and "ska_ridge" in str(exc.value)


def test_a_field_path_is_not_mistaken_for_an_axis_name():
    """`model.ska_rank` is a RunSpec override; `ska_rank` is an axis. Accepting
    the former would create an axis nothing samples."""
    base = _base_model()
    with pytest.raises(ValueError, match="unknown search axis"):
        restrict_space(search_space(base), base,
                       fixed={"model.ska_rank": 24})


def test_a_rank_that_is_not_a_multiple_of_eight_is_refused():
    """KoopmanLMConfig asserts this. Without the check the trial materializes a
    run directory and dies constructing the model."""
    base = _base_model()
    with pytest.raises(ValueError, match="multiple of 8"):
        restrict_space(search_space(base), base,
                       axes={"ska_rank": {"kind": "categorical",
                                          "choices": [8, 20, 32]}})


def test_a_layer_count_past_the_usable_window_is_refused_as_a_dead_axis():
    """The failure this whole mechanism exists for. `make_layer_indices` clamps,
    so 16 and 15 on a 17-layer backbone are the same config."""
    base = _base_model()
    with pytest.raises(ValueError, match="DEAD AXIS"):
        restrict_space(search_space(base), base,
                       axes={"n_ska_layers": {"kind": "categorical",
                                              "choices": [2, 4, 16]}})


def test_an_undeclared_placement_is_refused():
    base = _base_model()
    with pytest.raises(ValueError, match="placement"):
        restrict_space(search_space(base), base,
                       axes={"placement": {"kind": "categorical",
                                           "choices": ["baseline", "early"]}})


def test_a_non_integral_value_on_a_counting_axis_is_refused():
    """`int(1.5)` is 1, silently. A journal recording 1 for a study that asked
    for 1.5 is worse than a crash."""
    base = _base_model()
    with pytest.raises(ValueError, match="whole number"):
        restrict_space(search_space(base), base, fixed={"ska_power_K": 1.5})


def test_a_float_range_on_a_counting_axis_is_refused():
    """A FloatDistribution over ska_rank would put 19.4 in the journal, which no
    anchor could ever match and KoopmanLMConfig would reject."""
    base = _base_model()
    with pytest.raises(ValueError, match="counts or names"):
        restrict_space(search_space(base), base,
                       axes={"ska_rank": {"kind": "float", "low": 8.0,
                                          "high": 32.0}})


# ---------------------------------- both layers agree about what is legal ----
#
# `StudySpec` validates a declaration's SHAPE and `restrict_space` validates its
# DOMAIN, and the split is right -- only the second has a base model. But two
# layers with overlapping rules is how the weaker one quietly becomes the real
# contract, so where they overlap they must AGREE. Both of the cases below were
# found by an adversarial probe against `restrict_space` called directly: each
# was refused by StudySpec and accepted here, which meant any caller not coming
# through a study file got the weaker rule.


def test_a_boolean_is_refused_rather_than_silently_becoming_an_int():
    """The one Python hides. `bool` IS an `int`, so `float(True)` is 1.0 and
    `int(1.0)` is 1 -- `ska_power_K: true` was accepted as K=1. YAML makes it
    reachable rather than theoretical: `true`, `yes` and `on` all parse to True.
    """
    base = _base_model()
    for axis in ("ska_power_K", "grad_clip", "ska_rank", "placement"):
        with pytest.raises(ValueError, match="boolean"):
            restrict_space(search_space(base), base, fixed={axis: True})


def test_a_boolean_choice_in_a_declaration_is_refused_too():
    base = _base_model()
    with pytest.raises(ValueError, match="boolean"):
        restrict_space(search_space(base), base,
                       axes={"ska_power_K": {"kind": "categorical",
                                             "choices": [1, True]}})


def test_a_degenerate_float_range_is_refused_here_as_well_as_in_studyspec():
    """low == high would become a FloatDistribution optuna considers `single()`
    -- legal, and indistinguishable in the journal from an axis the sampler
    simply never varied. The study spec rejects it; so must this."""
    base = _base_model()
    with pytest.raises(ValueError, match="strictly less than"):
        restrict_space(search_space(base), base,
                       axes={"ska_ridge": {"kind": "float", "low": 0.01,
                                           "high": 0.01}})


def test_an_inverted_float_range_is_refused():
    base = _base_model()
    with pytest.raises(ValueError, match="strictly less than"):
        restrict_space(search_space(base), base,
                       axes={"ska_ridge": {"kind": "float", "low": 0.03,
                                           "high": 0.003}})


def test_a_numeric_string_is_still_accepted_and_coerced():
    """Deliberate leniency, not an oversight, and worth pinning so it is not
    "tightened" away: PyYAML parses `3e-3` as a STRING (no decimal point in the
    mantissa) -- `OptimSpec.lr`'s own error message exists because of that. A
    numeric string still goes through the same domain check, so it can only land
    on a value the axis already allows."""
    base = _base_model()
    restricted = restrict_space(search_space(base), base,
                                fixed={"ska_rank": "24", "ska_ridge": "0.01"})
    assert restricted["ska_rank"]["choices"] == [24]
    assert restricted["ska_ridge"]["choices"] == [0.01]
    assert isinstance(restricted["ska_rank"]["choices"][0], int)


def test_a_non_numeric_string_on_a_numeric_axis_is_refused():
    base = _base_model()
    with pytest.raises((ValueError, TypeError)):
        restrict_space(search_space(base), base, fixed={"ska_rank": "twenty"})


def test_a_nan_is_refused():
    """`nan > 0` is False, so the positivity check catches it -- asserted because
    that is a property of the comparison rather than an explicit guard, and a
    future refactor could lose it."""
    base = _base_model()
    with pytest.raises(ValueError):
        restrict_space(search_space(base), base,
                       axes={"gamma_value": {"kind": "categorical",
                                             "choices": [float("nan")]}})


def test_an_absurdly_large_int_is_refused_by_containment():
    base = _base_model()
    with pytest.raises(ValueError, match="outside this axis"):
        restrict_space(search_space(base), base, fixed={"ska_rank": 2 ** 40})


def test_a_non_positive_ridge_is_refused():
    base = _base_model()
    with pytest.raises(ValueError, match="ska_ridge"):
        restrict_space(search_space(base), base,
                       axes={"ska_ridge": {"kind": "float", "low": 0.0,
                                           "high": 0.03}})


def test_a_warmup_ratio_above_one_is_refused():
    base = _base_model()
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        restrict_space(search_space(base), base, fixed={"warmup_ratio": 1.4})


def test_duplicate_choices_are_refused_after_coercion():
    """1 and 1.0 are two categories to optuna and one value to the model."""
    base = _base_model()
    with pytest.raises(ValueError, match="duplicate"):
        restrict_space(search_space(base), base,
                       axes={"ska_power_K": {"kind": "categorical",
                                             "choices": [1, 1.0, 2]}})


# ---------------------------------------------------------------- fixing ----

def test_a_fixed_value_becomes_a_singleton_categorical():
    """Not a deletion. `params_to_overrides` requires the parameter, the journal
    records it, and `resolve_design` snaps anchors onto it."""
    base = _base_model()
    restricted = restrict_space(search_space(base), base,
                                fixed={"weight_decay": 0.1})
    assert restricted["weight_decay"] == {"kind": "categorical",
                                          "choices": [0.1]}


def test_fixing_leaves_every_required_parameter_in_the_space():
    base = _base_model()
    restricted = restrict_space(
        search_space(base), base,
        fixed={"weight_decay": 0.1, "warmup_ratio": 0.04, "grad_clip": 1.0})
    assert set(REQUIRED_PARAMS) <= set(restricted)


def test_a_fixed_value_outside_the_base_declaration_is_refused():
    """The base ridge axis is [3e-3, 3e-2]. Fixing 0.5 inside it would be a
    silent widening -- restriction must not be a way to escape the space."""
    base = _base_model()
    with pytest.raises(ValueError, match="outside this axis"):
        restrict_space(search_space(base), base, fixed={"ska_ridge": 0.5})


def test_a_fixed_value_outside_a_replaced_declaration_is_also_refused():
    base = _base_model()
    with pytest.raises(ValueError, match="outside this axis"):
        restrict_space(
            search_space(base), base,
            axes={"ska_rank": {"kind": "categorical", "choices": [8, 16]}},
            fixed={"ska_power_K": 3})


def test_a_value_inside_a_widened_declaration_is_accepted():
    """Widening is legal and must be stated in search_axes, where a reader sees
    it -- which is the whole difference between this and an escape hatch."""
    base = _base_model()
    restricted = restrict_space(
        search_space(base), base,
        axes={"ska_power_K": {"kind": "categorical", "choices": [1, 2, 3]}})
    assert restricted["ska_power_K"]["choices"] == [1, 2, 3]


def test_an_axis_cannot_be_both_restricted_and_fixed():
    """Checked in StudySpec, because it needs no base model."""
    from experimentation.sweep.search.studyspec import StudySpec

    with pytest.raises(ValueError, match="BOTH"):
        StudySpec(name="x", base="b.yaml", n_trials=10, max_steps=1000,
                  search_axes={"grad_clip": {"kind": "categorical",
                                             "choices": [1.0]}},
                  fixed_params={"grad_clip": 1.0})


# ------------------------------------------------------- baseline containment ----

def test_dropping_the_bases_own_value_is_reported_rather_than_hidden():
    """`search_space()` folds the base config's own values into every axis so a
    study can reproduce the config it is trying to beat. A restriction may drop
    one; doing it silently is what this reports."""
    base = _base_model(ska_rank=24)
    restricted = restrict_space(search_space(base), base,
                                axes={"ska_rank": {"kind": "categorical",
                                                   "choices": [8, 16]}})
    lost = dropped_base_values(restricted, base)
    assert "ska_rank" in lost and "24" in lost["ska_rank"]


def test_a_restriction_that_keeps_the_base_reports_nothing():
    base = _base_model()
    restricted = restrict_space(search_space(base), base,
                                axes={"ska_rank": {"kind": "categorical",
                                                   "choices": [8, 24]}})
    assert dropped_base_values(restricted, base) == {}


def test_the_target_studys_restriction_keeps_every_base_value_reachable():
    """The interaction study's axes are all supersets of the proxy base's own
    values, so its reference point is inside its own space. If this ever fails,
    `reference-k1` stopped being reproducible by the sampler."""
    base = _base_model()
    restricted = restrict_space(search_space(base), base, axes=_TARGET_AXES,
                                fixed=_TARGET_FIXED)
    assert dropped_base_values(
        restricted, base, base_lr=4.0e-4,
        base_optim={"weight_decay": 0.1, "grad_clip": 1.0}) == {}


# --------------------------------------------- the target study's own space ----

_TARGET_AXES = {
    "ska_rank": {"kind": "categorical", "choices": [8, 16, 24, 32]},
    "n_ska_layers": {"kind": "categorical", "choices": [2, 3, 4, 6, 8]},
    "placement": {"kind": "categorical",
                  "choices": ["baseline", "even", "midlate", "late"]},
    "ska_ridge": {"kind": "float", "low": 0.003, "high": 0.03, "log": True},
    # Widened upward 2026-08-24 on the evidence of job 445689: the best loss in
    # a four-trial probe was at 0.1, OUTSIDE the old [0.002, 0.03], and the gate
    # at 0.01 grew 2.36x during training while 0.1/0.5/1.0 stayed put. Pinned
    # against the committed study file by
    # `test_method_guarantees.py::test_the_hardcoded_target_axes_still_match_the_committed_study`.
    "ska_layerscale_init": {"kind": "float", "low": 0.005, "high": 0.3,
                            "log": True},
    "norm_clip_multiplier": {"kind": "categorical",
                             "choices": [0.75, 4.0 / math.sqrt(24), 1.0, 1.25]},
    "gamma_value": {"kind": "categorical", "choices": [0.90, 1.0, 1.05]},
    "ska_power_K": {"kind": "categorical", "choices": [1, 2]},
    "learning_rate": {"kind": "float", "low": 0.00032, "high": 0.00048,
                      "log": True},
}
_TARGET_FIXED = {"weight_decay": 0.1, "warmup_ratio": 0.04, "grad_clip": 1.0,
                 # Pinned by the committed study on 2026-08-24 when
                 # `beta_policy` became a repo-declared axis, so that a new
                 # repo axis could not retroactively widen a study designed
                 # around nine factors. See the study file's own note.
                 "beta_policy": "learned"}


def _resolved(params, base, **kw):
    return params_to_overrides(params, base, max_steps=600, **kw)


def _reference_point(restricted):
    """One concrete point: the first/lowest value of every axis."""
    point = {}
    for name, decl in restricted.items():
        point[name] = (decl["choices"][0] if decl["kind"] == "categorical"
                       else decl["low"])
    return point


def test_every_non_fixed_axis_can_move_the_resolved_spec():
    """The general form of the dead-axis check, run against the real study.

    For every axis with more than one declared value, two points differing ONLY
    in that axis must produce different RunSpec overrides. The current
    four-layer `4m-adaptive` study fails this on `placement` -- with two SKA
    layers in a four-layer backbone every placement collapses to the same pair
    of indices -- which is exactly why the interaction study uses a 17-layer
    proxy.
    """
    base = _base_model()
    restricted = restrict_space(search_space(base), base, axes=_TARGET_AXES,
                                fixed=_TARGET_FIXED)
    reference = _reference_point(restricted)
    baseline_overrides = _resolved(reference, base)

    moved, dead = [], []
    for name, decl in sorted(restricted.items()):
        alternatives = (decl["choices"] if decl["kind"] == "categorical"
                        else [decl["low"], decl["high"]])
        others = [v for v in alternatives if v != reference[name]]
        if not others:
            continue                       # a fixed axis; covered separately
        changed = False
        for value in others:
            probe = {**reference, name: value}
            if _resolved(probe, base) != baseline_overrides:
                changed = True
                break
        (moved if changed else dead).append(name)

    assert not dead, (
        f"axes with more than one declared value that cannot change the "
        f"resolved RunSpec: {dead}. A sampler spending trials on them is "
        f"measuring nothing, and a parameter importance over them is noise.")
    # Guard the guard: a probe loop that found nothing to vary would pass
    # vacuously.
    assert len(moved) == 9, f"expected 9 live axes, exercised {moved}"


def test_the_four_layer_study_is_the_dead_axis_this_test_would_catch():
    """Documents WHY the interaction study is not run on `4m-adaptive`'s base.

    Not a hypothetical: configs/runs/4m-golden.yaml is 4 layers with SKA at
    [2, 3], so the usable window is {1, 2} and every placement resolves to the
    same indices at every legal count.
    """
    from experimentation.sweep.search.geometry import make_layer_indices

    resolved = {p: tuple(make_layer_indices(4, 2, p, [2, 3]))
                for p in PLACEMENTS}
    assert len(set(resolved.values())) == 1, (
        f"4m-golden's placement axis is no longer dead: {resolved}")


def test_a_seventeen_layer_backbone_makes_placement_live():
    from experimentation.sweep.search.geometry import make_layer_indices

    for count in (2, 3, 4, 6, 8):
        resolved = {p: tuple(make_layer_indices(17, count, p, [3, 7, 11, 15]))
                    for p in PLACEMENTS}
        assert len(set(resolved.values())) == len(PLACEMENTS), (
            f"at n_ska_layers={count} two placements collapse: {resolved}")


def test_fixed_axes_reach_the_resolved_spec_at_their_pinned_value():
    """Fixing must not mean 'dropped and defaulted somewhere else'."""
    base = _base_model()
    restricted = restrict_space(search_space(base), base, axes=_TARGET_AXES,
                                fixed=_TARGET_FIXED)
    overrides = _resolved(_reference_point(restricted), base)
    assert overrides["optim.weight_decay"] == 0.1
    assert overrides["optim.grad_clip"] == 1.0
    # warmup_ratio 0.04 x 600 steps
    assert overrides["optim.warmup_steps"] == 24
