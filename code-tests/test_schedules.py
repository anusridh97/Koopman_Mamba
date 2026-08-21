"""§6 schedules: pure functions of global step, the SCHEDULABLE whitelist,
the setattr applier, freeze, and the seq_len curriculum.

Runs on CPU (SKAModule instantiates without mamba_ssm). See
docs/superpowers/specs/2026-08-08-extension-mechanisms-design.md §6.
"""
import pytest
import torch
import torch.nn as nn

pytestmark = pytest.mark.correctness


def _ska(**kw):
    from koopman_lm.modules.seq.ska import SKAModule
    kw.setdefault("d_model", 32)
    kw.setdefault("n_heads", 2)
    kw.setdefault("rank", 16)
    return SKAModule(**kw)


class _Stack(nn.Module):
    """Two SKA modules plus an unrelated linear, so 'apply to every matching
    module' is distinguishable from 'apply to the first one'."""

    def __init__(self, **kw):
        super().__init__()
        self.seq_layers = nn.ModuleList([_ska(**kw), _ska(**kw)])
        self.embed = nn.Embedding(16, 32)
        self.head = nn.Linear(32, 16, bias=False)


# --------------------------------------------------------------------------
# Kinds: each is a pure function of global step.
# --------------------------------------------------------------------------

def test_constant_ignores_the_step():
    from experimentation.training.schedules import make_schedule

    s = make_schedule({"kind": "constant", "value": 1e-3})
    assert s(0) == 1e-3
    assert s(10_000) == 1e-3


def test_linear_interpolates_between_the_endpoints():
    from experimentation.training.schedules import make_schedule

    s = make_schedule({"kind": "linear", "from": 1e-2, "to": 1e-3,
                       "over": [0, 1000]})
    assert s(0) == pytest.approx(1e-2)
    assert s(1000) == pytest.approx(1e-3)
    assert s(500) == pytest.approx((1e-2 + 1e-3) / 2)


def test_linear_clamps_outside_its_window():
    """A schedule is defined for every step, not just inside `over` -- the
    training loop calls it at step 0 and at max_steps alike."""
    from experimentation.training.schedules import make_schedule

    s = make_schedule({"kind": "linear", "from": 1e-2, "to": 1e-3,
                       "over": [100, 200]})
    assert s(0) == pytest.approx(1e-2)
    assert s(99) == pytest.approx(1e-2)
    assert s(10_000) == pytest.approx(1e-3)


def test_cosine_matches_its_endpoints_and_midpoint():
    from experimentation.training.schedules import make_schedule

    s = make_schedule({"kind": "cosine", "from": 1.0, "to": 0.0,
                       "over": [0, 100]})
    assert s(0) == pytest.approx(1.0)
    assert s(100) == pytest.approx(0.0, abs=1e-12)
    assert s(50) == pytest.approx(0.5)


def test_cosine_is_monotonic_across_its_window():
    from experimentation.training.schedules import make_schedule

    s = make_schedule({"kind": "cosine", "from": 1.0, "to": 0.0,
                       "over": [0, 100]})
    vals = [s(i) for i in range(101)]
    assert all(b <= a + 1e-12 for a, b in zip(vals, vals[1:]))


def test_piecewise_is_a_staircase_that_holds_each_value():
    """The design's own example is a seq_len curriculum (512 -> 1024 -> 2048),
    which must yield exactly those values, never an interpolated 768."""
    from experimentation.training.schedules import make_schedule

    s = make_schedule({"kind": "piecewise", "at": [0, 4000, 8000],
                       "values": [512, 1024, 2048]})
    assert s(0) == 512
    assert s(3999) == 512
    assert s(4000) == 1024
    assert s(7999) == 1024
    assert s(8000) == 2048
    assert s(99_999) == 2048


def test_step_decays_geometrically_every_n_steps():
    from experimentation.training.schedules import make_schedule

    s = make_schedule({"kind": "step", "from": 1.0, "gamma": 0.5, "every": 100})
    assert s(0) == pytest.approx(1.0)
    assert s(99) == pytest.approx(1.0)
    assert s(100) == pytest.approx(0.5)
    assert s(250) == pytest.approx(0.25)


def test_a_schedule_is_a_pure_function_of_step():
    """§6.2: no internal state, so replaying after a resume is free and exact.
    Calling out of order must not change any value."""
    from experimentation.training.schedules import make_schedule

    s = make_schedule({"kind": "linear", "from": 1.0, "to": 0.0,
                       "over": [0, 100]})
    forward = [s(i) for i in range(101)]
    backward = [s(i) for i in range(100, -1, -1)][::-1]
    repeat = [s(i) for i in range(101)]
    assert forward == backward == repeat


# --------------------------------------------------------------------------
# Spec validation
# --------------------------------------------------------------------------

def test_unknown_kind_raises_and_lists_the_known_ones():
    from experimentation.training.schedules import make_schedule

    with pytest.raises(ValueError, match="exponential"):
        make_schedule({"kind": "exponential", "from": 1.0, "to": 0.0})


def test_missing_required_argument_raises():
    from experimentation.training.schedules import make_schedule

    with pytest.raises((ValueError, KeyError), match="to"):
        make_schedule({"kind": "linear", "from": 1e-2, "over": [0, 100]})


def test_unknown_argument_raises():
    from experimentation.training.schedules import make_schedule

    with pytest.raises(ValueError, match="until"):
        make_schedule({"kind": "linear", "from": 1e-2, "to": 1e-3,
                       "over": [0, 100], "until": 50})


def test_over_must_be_a_forward_interval():
    from experimentation.training.schedules import make_schedule

    with pytest.raises(ValueError):
        make_schedule({"kind": "linear", "from": 1.0, "to": 0.0, "over": [100, 100]})
    with pytest.raises(ValueError):
        make_schedule({"kind": "linear", "from": 1.0, "to": 0.0, "over": [200, 100]})


def test_piecewise_rejects_mismatched_lengths():
    from experimentation.training.schedules import make_schedule

    with pytest.raises(ValueError):
        make_schedule({"kind": "piecewise", "at": [0, 100], "values": [1, 2, 3]})


def test_piecewise_requires_increasing_breakpoints_starting_at_zero():
    from experimentation.training.schedules import make_schedule

    with pytest.raises(ValueError):
        make_schedule({"kind": "piecewise", "at": [100, 0], "values": [1, 2]})
    with pytest.raises(ValueError, match="0"):
        make_schedule({"kind": "piecewise", "at": [10, 20], "values": [1, 2]})


def test_a_custom_kind_can_be_registered():
    """Anything exotic goes in a registry keyed by name, exactly like probes
    (§6.3)."""
    from experimentation.training.schedules import make_schedule, register_schedule

    @register_schedule("test-sqrt-decay")
    def _factory(spec):
        scale = spec["scale"]
        return lambda step: scale / (1.0 + step) ** 0.5

    s = make_schedule({"kind": "test-sqrt-decay", "scale": 2.0})
    assert s(0) == pytest.approx(2.0)
    assert s(3) == pytest.approx(1.0)


# --------------------------------------------------------------------------
# The whitelist (§6.4)
# --------------------------------------------------------------------------

def test_validate_accepts_the_whitelisted_targets():
    from experimentation.training.schedules import validate_schedules

    validate_schedules({
        "model.ska_ridge": {"kind": "linear", "from": 1e-2, "to": 1e-3,
                            "over": [0, 500]},
        "model.ska_power_K": {"kind": "piecewise", "at": [0, 100],
                              "values": [2, 3]},
        "data.seq_len": {"kind": "piecewise", "at": [0, 100],
                         "values": [512, 1024]},
        "freeze": {"match": "seq_layers.*.ska.*", "frozen_until": 200},
    })


def test_validate_rejects_an_unwhitelisted_target():
    from experimentation.training.schedules import validate_schedules

    with pytest.raises(ValueError, match="model.d_model"):
        validate_schedules({"model.d_model": {"kind": "constant", "value": 8}})


def test_validate_rejects_optim_lr_with_a_pointed_message():
    """§6.6: the existing LambdaLR owns the learning rate."""
    from experimentation.training.schedules import validate_schedules

    with pytest.raises(ValueError, match="LambdaLR|OptimSpec.schedule"):
        validate_schedules({"optim.lr": {"kind": "linear", "from": 1e-3,
                                         "to": 1e-4, "over": [0, 100]}})


def test_validate_rejects_scheduling_a_dimension():
    """§6.6: rank/width/d_state are fixed at construction."""
    from experimentation.training.schedules import validate_schedules

    with pytest.raises(ValueError):
        validate_schedules({"model.ska_rank": {"kind": "piecewise",
                                               "at": [0, 100],
                                               "values": [16, 32]}})


def test_validate_rejects_a_freeze_without_frozen_until():
    from experimentation.training.schedules import validate_schedules

    with pytest.raises((ValueError, KeyError)):
        validate_schedules({"freeze": {"match": "seq_layers.*"}})


# --------------------------------------------------------------------------
# ScheduleApplier: the setattr loop
# --------------------------------------------------------------------------

def test_applier_sets_the_attribute_on_every_matching_module():
    from experimentation.training.schedules import ScheduleApplier

    m = _Stack(ridge_eps=1e-2)
    applier = ScheduleApplier(m, {"model.ska_ridge": {
        "kind": "linear", "from": 1e-2, "to": 1e-3, "over": [0, 100]}})
    applier.apply(100)
    assert [l.ridge_eps for l in m.seq_layers] == \
           pytest.approx([1e-3, 1e-3])


def test_applier_returns_every_value_for_logging():
    """§6.4: without logging you cannot distinguish 'the schedule ran' from
    'the schedule was silently a no-op'."""
    from experimentation.training.schedules import ScheduleApplier

    m = _Stack()
    applier = ScheduleApplier(m, {
        "model.ska_ridge": {"kind": "linear", "from": 1e-2, "to": 1e-3,
                            "over": [0, 100]},
        "model.ska_power_K": {"kind": "constant", "value": 3},
    })
    vals = applier.apply(0)
    assert vals["model.ska_ridge"] == pytest.approx(1e-2)
    assert vals["model.ska_power_K"] == 3


def test_applier_keeps_power_k_an_integer():
    """power_K indexes range(); a float would raise deep in the forward."""
    from experimentation.training.schedules import ScheduleApplier

    m = _Stack()
    applier = ScheduleApplier(m, {"model.ska_power_K": {
        "kind": "linear", "from": 2.0, "to": 4.0, "over": [0, 100]}})
    applier.apply(50)
    for layer in m.seq_layers:
        assert isinstance(layer.power_K, int)
        assert layer.power_K == 3


def test_applier_rejects_an_unknown_target_at_construction():
    from experimentation.training.schedules import ScheduleApplier

    m = _Stack()
    with pytest.raises(ValueError, match="model.nonexistent"):
        ScheduleApplier(m, {"model.nonexistent": {"kind": "constant", "value": 1}})


def test_applier_raises_when_a_target_matches_no_module():
    """§6.4: an empty match must raise, not warn."""
    from experimentation.training.schedules import ScheduleApplier

    plain = nn.Linear(4, 4)
    with pytest.raises(ValueError, match="matched no modules"):
        ScheduleApplier(plain, {"model.ska_ridge": {"kind": "constant",
                                                    "value": 1e-3}})


def test_applier_with_no_schedules_is_inert():
    from experimentation.training.schedules import ScheduleApplier

    m = _Stack(ridge_eps=1e-2)
    applier = ScheduleApplier(m, {})
    assert applier.apply(0) == {}
    assert m.seq_layers[0].ridge_eps == 1e-2


def test_applied_value_survives_a_resume_replay():
    """§6.2: replaying a schedule at the resumed step must reproduce exactly
    the value an uninterrupted run held there."""
    from experimentation.training.schedules import ScheduleApplier

    sched = {"model.ska_ridge": {"kind": "cosine", "from": 1e-2, "to": 1e-4,
                                 "over": [0, 1000]}}
    uninterrupted = ScheduleApplier(_Stack(), sched)
    for step in range(501):
        last = uninterrupted.apply(step)
    resumed = ScheduleApplier(_Stack(), sched)
    assert resumed.apply(500) == last


# --------------------------------------------------------------------------
# freeze (§6.5)
# --------------------------------------------------------------------------

def test_freeze_clears_requires_grad_before_the_threshold():
    from experimentation.training.schedules import ScheduleApplier

    m = _Stack()
    applier = ScheduleApplier(m, {"freeze": {"match": "seq_layers.*",
                                             "frozen_until": 200}})
    applier.apply(0)
    assert not m.seq_layers[0].query_proj.weight.requires_grad
    assert m.head.weight.requires_grad, "an unmatched parameter must be untouched"


def test_freeze_restores_requires_grad_at_the_threshold():
    from experimentation.training.schedules import ScheduleApplier

    m = _Stack()
    applier = ScheduleApplier(m, {"freeze": {"match": "seq_layers.*",
                                             "frozen_until": 200}})
    applier.apply(0)
    applier.apply(200)
    assert m.seq_layers[0].query_proj.weight.requires_grad


def test_freeze_reports_its_state_for_logging():
    from experimentation.training.schedules import ScheduleApplier

    m = _Stack()
    applier = ScheduleApplier(m, {"freeze": {"match": "seq_layers.*",
                                             "frozen_until": 200}})
    assert applier.apply(0)["freeze"] == 1.0
    assert applier.apply(200)["freeze"] == 0.0


def test_freeze_raises_when_its_pattern_matches_no_parameter():
    from experimentation.training.schedules import ScheduleApplier

    m = _Stack()
    with pytest.raises(ValueError, match="matched no parameters"):
        ScheduleApplier(m, {"freeze": {"match": "decoder.*", "frozen_until": 10}})


def test_a_frozen_parameter_receives_no_update_then_starts_moving():
    """The composed behaviour that matters: frozen means no update, and
    unfreezing at the threshold actually resumes training that parameter."""
    from experimentation.training.optim import param_groups
    from experimentation.training.schedules import ScheduleApplier

    m = _Stack()
    applier = ScheduleApplier(m, {"freeze": {"match": "seq_layers.0.*",
                                             "frozen_until": 3}})
    opt = torch.optim.AdamW(
        param_groups(m, weight_decay=0.0, include_frozen=True), lr=0.1)
    watched = m.seq_layers[0].query_proj.weight
    before = watched.detach().clone()

    for step in range(3):
        applier.apply(step)
        opt.zero_grad(set_to_none=True)
        sum(p.sum() for p in m.parameters() if p.requires_grad).backward()
        opt.step()
    assert torch.equal(watched, before), "frozen parameter moved"

    for step in range(3, 6):
        applier.apply(step)
        opt.zero_grad(set_to_none=True)
        sum(p.sum() for p in m.parameters() if p.requires_grad).backward()
        opt.step()
    assert not torch.equal(watched, before), "unfrozen parameter never moved"


# --------------------------------------------------------------------------
# data.seq_len (§6.5)
# --------------------------------------------------------------------------

def test_seq_len_at_returns_an_integer_for_the_epoch_loop():
    from experimentation.training.schedules import ScheduleApplier

    m = _Stack()
    applier = ScheduleApplier(m, {"data.seq_len": {
        "kind": "piecewise", "at": [0, 100], "values": [512, 1024]}})
    assert applier.seq_len_at(0) == 512
    assert isinstance(applier.seq_len_at(0), int)
    assert applier.seq_len_at(150) == 1024


def test_seq_len_at_is_none_without_a_seq_len_schedule():
    from experimentation.training.schedules import ScheduleApplier

    assert ScheduleApplier(_Stack(), {}).seq_len_at(0) is None


def test_seq_len_is_not_pushed_onto_modules_by_apply():
    """data.seq_len is consumed by the epoch loop, not settable on a module."""
    from experimentation.training.schedules import ScheduleApplier

    m = _Stack()
    applier = ScheduleApplier(m, {"data.seq_len": {
        "kind": "piecewise", "at": [0, 100], "values": [512, 1024]}})
    applier.apply(0)
    assert not hasattr(m.seq_layers[0], "seq_len")


def test_apply_does_not_report_seq_len_as_a_per_step_value():
    """apply() reports what it just made true. seq_len is NOT per-step: the
    dataset is only rebuilt at an epoch boundary, so a step past a breakpoint
    is still feeding the OLD length. Reporting the schedule's raw value here
    would log a change that has not taken effect -- observed in a real 8-step
    run, which logged `seq_len 128` for four steps while the loader was still
    yielding 64-token samples (one epoch was 1562 steps, so no rebuild ever
    happened). The effective length is the epoch loop's to report."""
    from experimentation.training.schedules import ScheduleApplier

    applier = ScheduleApplier(_Stack(), {"data.seq_len": {
        "kind": "piecewise", "at": [0, 4], "values": [64, 128]}})
    assert "data.seq_len" not in applier.apply(5)
    # Still queryable for the epoch loop, which is the only legitimate consumer.
    assert applier.seq_len_at(5) == 128
