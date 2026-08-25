"""`--ska_lr_mult` on the MQAR trainer: the EXISTING mechanism, wired through.

Phase 4 of the exponent programme crosses LayerScale init against an SKA learning
-rate factor. `optim.groups` already implements per-group `lr_mult` -- 19 tests in
`code-tests/test_optim_groups.py` pin it -- and `experiments/mqar_finetune.py`
calls `param_groups(model, weight_decay=0.1)` with no `groups` and no `lr`, so
the mechanism was unreachable from that trainer. This adds the flag that reaches
it. It does NOT add a second multiplier, and it does not mutate the optimizer
after construction.

## The three things that have to be true

  * **Absent or 1.0 must be bit-identical to today.** `param_groups`' own
    contract is that "an empty or absent `groups` leaves the output
    bit-identical to the historical two-bucket policy". Every archived MQAR
    result was produced without this flag, so the default cannot change any of
    them.
  * **The multiplier must land on the SKA parameters and nothing else.** A
    pattern that also caught the Mamba branch would make a "SKA LR sweep" a
    global LR sweep, which is a different experiment with the same name.
  * **It must compose with the cosine schedule rather than fight it.**
    `ParamGroupSpec`'s docstring is explicit: LambdaLR snapshots each group's lr
    as `initial_lr` and scales every group by the same factor per step, so a
    MULTIPLIER composes and an absolute per-group lr does not. That is why
    `param_groups` needs `lr=` passed alongside a non-unit `lr_mult`, and
    forgetting it is a silent no-op rather than an error.

Tested at the `param_groups` level on a CPU-constructible SKA stack: the property
is about which parameters get which lr, and a Mamba backbone would make it a GPU
test for no gain.
"""
import dataclasses

import pytest
import torch
import torch.nn as nn

from experimentation.experiments.mqar_finetune import ska_param_groups
from experimentation.training.optim import param_groups
from koopman_lm.config import KoopmanLMConfig
from koopman_lm.modules.seq.ska_block import SKABlock

pytestmark = pytest.mark.correctness


class _Stack(nn.Module):
    """SKA blocks under a `seq_layers` ModuleList, as the real models spell it,
    plus non-SKA parameters the multiplier must NOT touch."""

    def __init__(self, d_model=32, n=2):
        super().__init__()
        cfg = dataclasses.replace(
            KoopmanLMConfig(), d_model=d_model, ska_n_heads=4, ska_rank=16,
            ska_inverse_cholesky=True, ska_layerscale=True, ska_short_conv=False)
        self.embed = nn.Embedding(64, d_model)
        self.seq_layers = nn.ModuleList([SKABlock(cfg) for _ in range(n)])
        self.mlp_layers = nn.ModuleList([nn.Linear(d_model, d_model)
                                         for _ in range(n)])
        self.lm_head = nn.Linear(d_model, 64, bias=False)


def _named(model):
    return dict(model.named_parameters())


def _lr_of(groups, name, model):
    """The lr assigned to parameter `name`, or None if no group carries one."""
    target = _named(model)[name]
    for g in groups:
        if any(p is target for p in g["params"]):
            return g.get("lr")
    raise AssertionError(f"{name} landed in no parameter group at all")


# ---------------------------------------------------------------------------
# The default must not move.
# ---------------------------------------------------------------------------

def test_a_unit_multiplier_reproduces_the_historical_groups_exactly():
    """Every archived MQAR result was produced without this flag."""
    m = _Stack()
    base = param_groups(m, weight_decay=0.1)
    got = ska_param_groups(m, weight_decay=0.1, lr=1e-3, ska_lr_mult=1.0)
    assert len(got) == len(base)
    for a, b in zip(got, base):
        assert a["weight_decay"] == b["weight_decay"]
        assert [id(p) for p in a["params"]] == [id(p) for p in b["params"]]


def test_no_group_carries_an_explicit_lr_at_a_unit_multiplier():
    """`param_groups` only stamps an explicit `lr` when some spec has a non-unit
    multiplier. A unit multiplier that stamped one anyway would give LambdaLR a
    different `initial_lr` bookkeeping than the archived runs had."""
    m = _Stack()
    for g in ska_param_groups(m, weight_decay=0.1, lr=1e-3, ska_lr_mult=1.0):
        assert "lr" not in g or g["lr"] is None


def test_every_parameter_is_still_covered_exactly_once():
    """A pattern that dropped or duplicated a parameter would silently stop
    training it, or train it twice per step."""
    m = _Stack()
    groups = ska_param_groups(m, weight_decay=0.1, lr=1e-3, ska_lr_mult=3.0)
    seen = [id(p) for g in groups for p in g["params"]]
    assert len(seen) == len(set(seen)), "a parameter is in two groups"
    assert set(seen) == {id(p) for p in m.parameters()}


# ---------------------------------------------------------------------------
# The multiplier lands where it claims to.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mult", [3.0, 10.0])
def test_the_ska_parameters_get_the_multiplied_learning_rate(mult):
    m = _Stack()
    groups = ska_param_groups(m, weight_decay=0.1, lr=1e-3, ska_lr_mult=mult)
    ska_names = [n for n in _named(m) if ".ska." in n]
    assert ska_names, "the fixture has no SKA parameters; the test is vacuous"
    for name in ska_names:
        assert _lr_of(groups, name, m) == pytest.approx(1e-3 * mult), name


@pytest.mark.parametrize("mult", [3.0, 10.0])
def test_nothing_outside_ska_is_multiplied(mult):
    """A pattern that also caught the backbone would make an "SKA LR sweep" a
    global LR sweep with the same name."""
    m = _Stack()
    groups = ska_param_groups(m, weight_decay=0.1, lr=1e-3, ska_lr_mult=mult)
    others = [n for n in _named(m) if ".ska." not in n]
    assert others, "the fixture has no non-SKA parameters; the test is vacuous"
    for name in others:
        lr = _lr_of(groups, name, m)
        assert lr == pytest.approx(1e-3), f"{name} got lr {lr}, expected the base"


def test_the_layerscale_gate_counts_as_an_ska_parameter():
    """LayerScale is the other axis of the Phase 4 wave, and it lives inside the
    SKA module. If the multiplier missed it, a LayerScale x SKA-LR grid would be
    measuring two things that do not interact through the parameter it names."""
    m = _Stack()
    groups = ska_param_groups(m, weight_decay=0.1, lr=1e-3, ska_lr_mult=3.0)
    gates = [n for n in _named(m) if "layerscale_gate" in n]
    assert gates, "the fixture built no LayerScale gate"
    for name in gates:
        assert _lr_of(groups, name, m) == pytest.approx(3e-3), name


def test_the_write_gate_projection_is_multiplied():
    """The beta gate is what the whole programme is about; a multiplier that
    skipped it would be an SKA LR sweep that does not touch the write gate."""
    m = _Stack()
    groups = ska_param_groups(m, weight_decay=0.1, lr=1e-3, ska_lr_mult=10.0)
    betas = [n for n in _named(m) if "beta_proj" in n]
    assert betas, "the fixture built no beta_proj"
    for name in betas:
        assert _lr_of(groups, name, m) == pytest.approx(1e-2), name


# ---------------------------------------------------------------------------
# Refusals.
# ---------------------------------------------------------------------------

def test_a_non_positive_multiplier_is_refused():
    """0 would silently freeze SKA and report itself as an LR setting; a negative
    multiplier is ascent. Both must fail at the call, not train."""
    m = _Stack()
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError):
            ska_param_groups(m, weight_decay=0.1, lr=1e-3, ska_lr_mult=bad)


def test_a_non_unit_multiplier_without_a_base_lr_is_refused():
    """`param_groups` needs `lr` to have something to multiply. Omitting it is a
    silent no-op there, which would make a whole Phase 4 wave a replicate of the
    reference cell -- so it is refused here instead."""
    m = _Stack()
    with pytest.raises(ValueError):
        ska_param_groups(m, weight_decay=0.1, lr=None, ska_lr_mult=3.0)


def test_the_flag_exists_on_the_cli_and_defaults_to_one():
    """A flag the launcher passes must exist, and its default must be the value
    that reproduces every archived run."""
    from experimentation.experiments.mqar_finetune import parse_args
    args = parse_args(["--output_dir", "/tmp/x"])
    assert args.ska_lr_mult == 1.0
    args = parse_args(["--output_dir", "/tmp/x", "--ska_lr_mult", "3"])
    assert args.ska_lr_mult == 3.0
