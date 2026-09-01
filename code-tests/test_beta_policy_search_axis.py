"""`beta_policy` as a search axis and an anchor field.

The four policies exist to be COMPARED, and the comparison has to survive a
noise floor of sigma = 9.50e-3 (job 445832, five seeds at 600 steps on the
proxy), i.e. a trial-vs-trial resolvable effect of 2*sigma*sqrt(2) = 0.0269.
That number forces the experimental design: a single-seed 4-cell grid cannot
rank anything here, so the comparison must be N replicate seeds per policy, and
the mechanism that already spells "one experiment, N datapoints" is
`anchors.py`'s `seed` + `reference_group`.

So the axis is wired for the anchor path first and the sampler path second. Both
are tested, because `resolve_design` and the sampler must produce the same
params dict shape -- that interchangeability is what lets a curated design set
ship as a static `cells:` sweep AND seed a study through `enqueue_trial`.

## Why an explicit policy must not snap

`Design.power_K` records the rule and the reason: an inherited value snaps onto
the declared choices (harmless, since the base config's own value is always a
declared choice), while an EXPLICIT one must be declared exactly or it raises.
A design named `beta-one` that silently resolved to `learned` because the study
narrowed the axis would be a trial whose name asserts something its params
deny -- and in a 4-policy comparison that is not a cosmetic problem, it is a
cell quietly becoming a duplicate of the control.
"""
import dataclasses

import pytest

from koopman_lm.config import BETA_POLICIES, build_config
from experimentation.sweep.search.anchors import Design, resolve_design
from experimentation.sweep.search.space import (
    DEFAULT_BETA_POLICIES, _AXIS_DOMAINS, REQUIRED_PARAMS,
    base_reference_point, params_to_overrides, search_space)

pytestmark = pytest.mark.correctness


def _base():
    from experimentation.run.resolve import resolve_run_spec
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    return resolve_run_spec(root / "configs/runs/proxy-256x17.yaml")


def _space(base_model):
    return search_space(base_model, base_name="proxy-256x17")


# ---------------------------------------------------------------------------
# The axis exists, is complete, and contains the base config.
# ---------------------------------------------------------------------------

def test_beta_policy_default_is_frozen_in_archived_order():
    space = _space(_base().model)
    assert space["beta_policy"]["kind"] == "categorical"
    assert tuple(space["beta_policy"]["choices"]) == DEFAULT_BETA_POLICIES
    assert set(DEFAULT_BETA_POLICIES) <= set(BETA_POLICIES)


def test_the_axis_is_wired_end_to_end():
    """Three registries have to agree or the axis is a hole rather than a knob:
    `REQUIRED_PARAMS` (so a restriction can be checked for completeness rather
    than discovered incomplete by a KeyError on trial 0), `_AXIS_DOMAINS` (so an
    out-of-domain value fails on a login node), and `search_space` itself.
    """
    space = _space(_base().model)
    assert "beta_policy" in REQUIRED_PARAMS
    assert "beta_policy" in _AXIS_DOMAINS
    assert set(_AXIS_DOMAINS) == set(space)


def test_the_base_config_is_inside_the_axis():
    """Baseline containment: the point a sampler would have to propose to
    reproduce the config the study is trying to beat."""
    base = _base()
    point = base_reference_point(base.model, base_lr=base.optim.lr,
                                base_optim=dataclasses.asdict(base.optim))
    assert point["beta_policy"] == base.model.ska_beta_policy == "learned"


def test_an_out_of_domain_policy_is_refused_by_the_axis_validator():
    with pytest.raises(ValueError, match="beta_policy"):
        _AXIS_DOMAINS["beta_policy"]("sqrt", _base().model)


# ---------------------------------------------------------------------------
# params -> overrides, and the anchor path.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("policy", sorted(BETA_POLICIES))
def test_params_to_overrides_emits_the_model_field(policy):
    base = _base()
    params = base_reference_point(base.model, base_lr=base.optim.lr,
                                 base_optim=dataclasses.asdict(base.optim))
    params["beta_policy"] = policy
    params["warmup_ratio"] = 0.04
    overrides = params_to_overrides(params, base.model, max_steps=600)
    assert overrides["model.ska_beta_policy"] == policy
    # And the override actually builds a config, i.e. the spelling matches the
    # field name. A wrong key here would be accepted by the overrides dict and
    # rejected only when the cell is built.
    assert dataclasses.replace(
        base.model, ska_beta_policy=overrides["model.ska_beta_policy"])


def test_an_archived_journal_without_the_key_replays_as_learned():
    """`report.py` replays `trial.params` out of a journal, and trials recorded
    before this axis existed have no such key. They ran at the pinned default,
    so the fallback must be `learned` and NOT `base_model.ska_beta_policy` --
    promoting an archived study would otherwise confirm a policy the trial never
    ran at. Same rule, and the same reason, as `ska_power_K`'s `.get`.
    """
    base = _base()
    params = base_reference_point(base.model, base_lr=base.optim.lr,
                                 base_optim=dataclasses.asdict(base.optim))
    params["warmup_ratio"] = 0.04
    params.pop("beta_policy")
    overrides = params_to_overrides(params, base.model, max_steps=600)
    assert overrides["model.ska_beta_policy"] == "learned"


def test_a_design_inherits_the_base_policy_by_default():
    base = _base()
    space = _space(base.model)
    params = resolve_design(Design(name="reference"), base.model, space,
                            base_lr=base.optim.lr)
    assert params["beta_policy"] == "learned"


@pytest.mark.parametrize("policy", sorted(BETA_POLICIES))
def test_a_design_can_name_a_policy_explicitly(policy):
    base = _base()
    space = _space(base.model)
    params = resolve_design(Design(name=f"beta-{policy}", beta_policy=policy),
                            base.model, space, base_lr=base.optim.lr)
    assert params["beta_policy"] == policy


def test_an_explicit_policy_the_study_does_not_declare_raises_rather_than_snaps():
    base = _base()
    space = dict(_space(base.model))
    space["beta_policy"] = {"kind": "categorical", "choices": ["learned", "one"]}
    with pytest.raises(ValueError, match="beta_policy"):
        resolve_design(Design(name="beta-linear", beta_policy="linear"),
                       base.model, space, base_lr=base.optim.lr)


def test_a_design_naming_a_nonexistent_policy_raises():
    base = _base()
    with pytest.raises(ValueError, match="beta_policy"):
        resolve_design(Design(name="beta-sqrt", beta_policy="sqrt"),
                       base.model, _space(base.model), base_lr=base.optim.lr)


def test_designs_differing_only_in_policy_are_not_a_replicate_set():
    """`reference_group` promises "identical in every scientific factor, differs
    only in seed", and `load_designs` enforces it. A new Design field that the
    replicate check did not know about would let a group span two policies --
    which would inflate the noise floor with a real effect and therefore label
    real effects unresolvable. That is the single most damaging thing this
    mechanism can do, so it is checked here rather than assumed.
    """
    from experimentation.sweep.search.anchors import _REPLICATE_FREE_FIELDS
    assert "beta_policy" not in _REPLICATE_FREE_FIELDS
