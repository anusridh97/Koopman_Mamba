"""§7 per-group optimizer settings: ParamGroupSpec and param_groups overrides.

Pure-Python + a tiny nn.Module, no mamba_ssm -- runs on CPU/CI. See
docs/superpowers/specs/2026-08-08-extension-mechanisms-design.md §7.
"""
import pytest
import torch
import torch.nn as nn

pytestmark = pytest.mark.correctness


class _Tiny(nn.Module):
    """Parameter names deliberately mirror the real model's: an `embed.weight`
    2-D embedding, an `.ska.` submodule, a bias, and a norm scale -- enough to
    exercise every branch of the decay policy and of pattern matching."""

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(16, 8)
        self.ska = nn.Linear(8, 8, bias=True)
        self.mlp = nn.Linear(8, 8, bias=True)
        self.norm = nn.LayerNorm(8)


def _by_id(group):
    return {id(p) for p in group["params"]}


def _named(model):
    return {id(p): n for n, p in model.named_parameters()}


def _group_of(groups, model, name):
    """The single group containing the parameter called `name`."""
    target = dict(model.named_parameters())[name]
    hits = [g for g in groups if any(p is target for p in g["params"])]
    assert len(hits) == 1, f"{name} appeared in {len(hits)} groups, expected 1"
    return hits[0]


# --------------------------------------------------------------------------
# Backwards compatibility: no groups must be bit-identical to today's policy.
# --------------------------------------------------------------------------

def test_no_groups_keeps_the_existing_two_bucket_policy():
    from experimentation.training.optim import param_groups

    m = _Tiny()
    groups = param_groups(m, weight_decay=0.1)
    assert len(groups) == 2
    decay, no_decay = groups
    assert decay["weight_decay"] == 0.1
    assert no_decay["weight_decay"] == 0.0
    names = _named(m)
    assert {names[i] for i in _by_id(decay)} == {"ska.weight", "mlp.weight"}
    assert {names[i] for i in _by_id(no_decay)} == {
        "embed.weight", "ska.bias", "mlp.bias", "norm.weight", "norm.bias"}


def test_empty_groups_list_is_identical_to_no_groups():
    from experimentation.training.optim import param_groups

    m = _Tiny()
    baseline = param_groups(m, weight_decay=0.1)
    with_empty = param_groups(m, weight_decay=0.1, groups=[])
    assert [_by_id(g) for g in with_empty] == [_by_id(g) for g in baseline]
    assert [g["weight_decay"] for g in with_empty] == \
           [g["weight_decay"] for g in baseline]


# --------------------------------------------------------------------------
# lr_mult
# --------------------------------------------------------------------------

def test_lr_mult_sets_an_explicit_per_group_lr():
    from experimentation.training.optim import ParamGroupSpec, param_groups

    m = _Tiny()
    groups = param_groups(m, weight_decay=0.1, lr=4e-4,
                          groups=[ParamGroupSpec(match="ska.*", lr_mult=10.0)])
    assert _group_of(groups, m, "ska.weight")["lr"] == pytest.approx(4e-3)
    # Unmatched parameters keep the base lr.
    assert _group_of(groups, m, "mlp.weight")["lr"] == pytest.approx(4e-4)


def test_lr_mult_without_a_base_lr_is_an_error():
    """lr_mult is a multiplier; there is nothing to multiply without the base
    lr, and silently ignoring it would run an experiment nobody asked for."""
    from experimentation.training.optim import ParamGroupSpec, param_groups

    m = _Tiny()
    with pytest.raises(ValueError, match="lr"):
        param_groups(m, weight_decay=0.1,
                     groups=[ParamGroupSpec(match="ska.*", lr_mult=10.0)])


def test_a_matched_group_still_splits_decay_from_no_decay():
    """Matching must not defeat the decay policy: ska.weight decays, ska.bias
    does not, even though one pattern covers both."""
    from experimentation.training.optim import ParamGroupSpec, param_groups

    m = _Tiny()
    groups = param_groups(m, weight_decay=0.1, lr=1e-3,
                          groups=[ParamGroupSpec(match="ska.*", lr_mult=10.0)])
    assert _group_of(groups, m, "ska.weight")["weight_decay"] == 0.1
    assert _group_of(groups, m, "ska.bias")["weight_decay"] == 0.0
    assert _group_of(groups, m, "ska.bias")["lr"] == pytest.approx(1e-2)


# --------------------------------------------------------------------------
# weight_decay override
# --------------------------------------------------------------------------

def test_weight_decay_override_beats_the_default_policy():
    from experimentation.training.optim import ParamGroupSpec, param_groups

    m = _Tiny()
    groups = param_groups(
        m, weight_decay=0.1, lr=1e-3,
        groups=[ParamGroupSpec(match="mlp.weight", weight_decay=0.0)])
    assert _group_of(groups, m, "mlp.weight")["weight_decay"] == 0.0
    assert _group_of(groups, m, "ska.weight")["weight_decay"] == 0.1


def test_weight_decay_override_can_force_decay_onto_a_no_decay_parameter():
    from experimentation.training.optim import ParamGroupSpec, param_groups

    m = _Tiny()
    groups = param_groups(
        m, weight_decay=0.1, lr=1e-3,
        groups=[ParamGroupSpec(match="embed.weight", weight_decay=0.05)])
    assert _group_of(groups, m, "embed.weight")["weight_decay"] == 0.05


# --------------------------------------------------------------------------
# Ordering and validation
# --------------------------------------------------------------------------

def test_first_match_wins():
    from experimentation.training.optim import ParamGroupSpec, param_groups

    m = _Tiny()
    groups = param_groups(m, weight_decay=0.1, lr=1e-3, groups=[
        ParamGroupSpec(match="ska.weight", lr_mult=2.0),
        ParamGroupSpec(match="ska.*", lr_mult=100.0),
    ])
    assert _group_of(groups, m, "ska.weight")["lr"] == pytest.approx(2e-3)
    assert _group_of(groups, m, "ska.bias")["lr"] == pytest.approx(0.1)


def test_a_pattern_matching_nothing_raises():
    """A silently-dead pattern is how you believe you ran an experiment you
    did not (design §7)."""
    from experimentation.training.optim import ParamGroupSpec, param_groups

    m = _Tiny()
    with pytest.raises(ValueError, match="matched no parameters"):
        param_groups(m, weight_decay=0.1, lr=1e-3,
                     groups=[ParamGroupSpec(match="nonexistent.*")])


def test_pattern_matching_nothing_names_the_offending_pattern():
    from experimentation.training.optim import ParamGroupSpec, param_groups

    m = _Tiny()
    with pytest.raises(ValueError, match=r"decoder\.\*"):
        param_groups(m, weight_decay=0.1, lr=1e-3, groups=[
            ParamGroupSpec(match="ska.*"),
            ParamGroupSpec(match="decoder.*"),
        ])


def test_group_spec_rejects_an_empty_match():
    from experimentation.training.optim import ParamGroupSpec

    with pytest.raises(ValueError):
        ParamGroupSpec(match="")


def test_group_spec_rejects_a_non_positive_lr_mult():
    from experimentation.training.optim import ParamGroupSpec

    with pytest.raises(ValueError):
        ParamGroupSpec(match="ska.*", lr_mult=0.0)
    with pytest.raises(ValueError):
        ParamGroupSpec(match="ska.*", lr_mult=-1.0)


def test_group_spec_rejects_a_negative_weight_decay():
    from experimentation.training.optim import ParamGroupSpec

    with pytest.raises(ValueError):
        ParamGroupSpec(match="ska.*", weight_decay=-0.1)


def test_parse_group_specs_accepts_yaml_dicts():
    from experimentation.training.optim import ParamGroupSpec, parse_group_specs

    specs = parse_group_specs([
        {"match": "*.ska.*", "lr_mult": 10.0},
        {"match": "embed.weight", "lr_mult": 0.1, "weight_decay": 0.0},
    ])
    assert specs == (
        ParamGroupSpec(match="*.ska.*", lr_mult=10.0),
        ParamGroupSpec(match="embed.weight", lr_mult=0.1, weight_decay=0.0),
    )


def test_parse_group_specs_rejects_an_unknown_key():
    from experimentation.training.optim import parse_group_specs

    with pytest.raises((ValueError, TypeError)):
        parse_group_specs([{"match": "ska.*", "lr_multiplier": 10.0}])


# --------------------------------------------------------------------------
# Composition with the real LR schedule -- the reason it is lr_mult and not lr.
# --------------------------------------------------------------------------

def test_lr_mult_composes_with_a_lambdalr_warmup_cosine():
    """get_cosine_schedule_with_warmup is a LambdaLR: it snapshots each group's
    lr as initial_lr and multiplies all of them by one shared factor. So a
    per-group multiplier must survive the schedule as a constant ratio."""
    from transformers import get_cosine_schedule_with_warmup
    from experimentation.training.optim import ParamGroupSpec, param_groups

    m = _Tiny()
    base_lr = 4e-4
    opt = torch.optim.AdamW(
        param_groups(m, weight_decay=0.1, lr=base_lr,
                     groups=[ParamGroupSpec(match="ska.*", lr_mult=10.0)]),
        lr=base_lr)
    sched = get_cosine_schedule_with_warmup(
        opt, num_warmup_steps=10, num_training_steps=100)

    ska_group = _group_of(opt.param_groups, m, "ska.weight")
    mlp_group = _group_of(opt.param_groups, m, "mlp.weight")
    for p in m.parameters():
        p.grad = torch.zeros_like(p)
    for _ in range(50):
        opt.step()
        sched.step()
        if mlp_group["lr"] > 0:
            assert ska_group["lr"] / mlp_group["lr"] == pytest.approx(10.0)


# --------------------------------------------------------------------------
# Frozen parameters (§6.5 needs the optimizer built over all of them).
# --------------------------------------------------------------------------

def test_frozen_parameters_are_excluded_by_default():
    from experimentation.training.optim import param_groups

    m = _Tiny()
    m.ska.weight.requires_grad_(False)
    groups = param_groups(m, weight_decay=0.1)
    covered = set().union(*(_by_id(g) for g in groups))
    assert id(m.ska.weight) not in covered


def test_include_frozen_puts_frozen_parameters_in_the_optimizer():
    """§6.5: a parameter frozen at step 0 must still enter the optimizer, or a
    freeze schedule could never unfreeze it."""
    from experimentation.training.optim import param_groups

    m = _Tiny()
    m.ska.weight.requires_grad_(False)
    groups = param_groups(m, weight_decay=0.1, include_frozen=True)
    covered = set().union(*(_by_id(g) for g in groups))
    assert id(m.ska.weight) in covered


def test_shared_parameters_are_still_deduplicated():
    from experimentation.training.optim import param_groups

    m = _Tiny()
    m.mlp.weight = m.ska.weight          # weight tying
    groups = param_groups(m, weight_decay=0.1, include_frozen=True)
    all_ids = [id(p) for g in groups for p in g["params"]]
    assert len(all_ids) == len(set(all_ids))
