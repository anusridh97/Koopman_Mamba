"""The trainer's half of §6/§7: reading the two optional sections off the
materialized spec.yaml, and the seq_len curriculum's epoch seam.

The §5.1 lesson applies here -- "two correct units, wrong composition" -- so
these test the composed path (spec.yaml -> trainer decision), not just the
units. See docs/superpowers/specs/2026-08-08-extension-mechanisms-design.md
§6.5, §7.
"""
import pytest
import torch.nn as nn
import yaml

pytestmark = pytest.mark.correctness


def _write_spec(tmp_path, *, optim_extra=None, schedules=None):
    """A materialized spec.yaml, produced the way the run layer produces it."""
    from koopman_lm.config import build_config
    from experimentation.run.resolve import materialize
    from experimentation.run.spec import OptimSpec, RunSpec, RuntimeSpec, ShardDataSpec

    optim = dict(lr=4.0e-4, warmup_steps=10, max_steps=100)
    optim.update(optim_extra or {})
    spec = RunSpec(
        name="t", model=build_config("1m"),
        data=ShardDataSpec(shard_dir="/d", tokenizer="tok",
                           mix={"fineweb": 1.0}, n_tokens=10),
        optim=OptimSpec(**optim), runtime=RuntimeSpec(),
        schedules=schedules or {},
    )
    return materialize(spec, tmp_path)


# --------------------------------------------------------------------------
# read_spec_extensions: the composed path off a real materialized spec.yaml
# --------------------------------------------------------------------------

def test_read_spec_extensions_returns_nothing_for_a_plain_spec(tmp_path):
    from experimentation.training.train import read_spec_extensions

    groups, schedules = read_spec_extensions(_write_spec(tmp_path))
    assert groups == ()
    assert schedules == {}


def test_read_spec_extensions_reads_optim_groups(tmp_path):
    from experimentation.training.optim import ParamGroupSpec
    from experimentation.training.train import read_spec_extensions

    path = _write_spec(tmp_path, optim_extra={
        "groups": [{"match": "*.ska.*", "lr_mult": 10.0}]})
    groups, _ = read_spec_extensions(path)
    assert groups == (ParamGroupSpec(match="*.ska.*", lr_mult=10.0),)


def test_read_spec_extensions_reads_schedules(tmp_path):
    from experimentation.training.train import read_spec_extensions

    path = _write_spec(tmp_path, schedules={
        "model.ska_ridge": {"kind": "linear", "from": 1.0e-2, "to": 1.0e-3,
                            "over": [0, 50]}})
    _, schedules = read_spec_extensions(path)
    assert schedules["model.ska_ridge"]["kind"] == "linear"


def test_read_spec_extensions_of_none_is_empty():
    from experimentation.training.train import read_spec_extensions

    assert read_spec_extensions(None) == ((), {})


def test_read_spec_extensions_rejects_a_missing_file(tmp_path):
    from experimentation.training.train import read_spec_extensions

    with pytest.raises(SystemExit):
        read_spec_extensions(tmp_path / "nope.yaml")


def test_read_spec_extensions_validates_rather_than_deferring(tmp_path):
    """A bad schedule must fail when the trainer starts, not silently at the
    first step -- and certainly not as a no-op."""
    from experimentation.training.train import read_spec_extensions

    path = tmp_path / "spec.yaml"
    path.write_text(yaml.safe_dump({
        "schedules": {"model.nonexistent": {"kind": "constant", "value": 1}}}))
    with pytest.raises(ValueError, match="model.nonexistent"):
        read_spec_extensions(path)


# --------------------------------------------------------------------------
# The seq_len curriculum's epoch seam (§6.5)
# --------------------------------------------------------------------------

def test_epoch_seq_len_falls_back_to_the_cli_value_without_a_schedule():
    from experimentation.training.schedules import ScheduleApplier
    from experimentation.training.train import epoch_seq_len

    applier = ScheduleApplier(nn.Linear(4, 4), {})
    assert epoch_seq_len(applier, epoch_start_step=0, default=2048) == 2048


def test_epoch_seq_len_uses_the_schedule_value_at_the_epoch_start():
    from experimentation.training.schedules import ScheduleApplier
    from experimentation.training.train import epoch_seq_len

    applier = ScheduleApplier(nn.Linear(4, 4), {"data.seq_len": {
        "kind": "piecewise", "at": [0, 100], "values": [512, 1024]}})
    assert epoch_seq_len(applier, epoch_start_step=0, default=2048) == 512
    assert epoch_seq_len(applier, epoch_start_step=100, default=2048) == 1024


def test_epoch_seq_len_is_keyed_on_the_epoch_start_not_the_current_step():
    """The bug this prevents: an uninterrupted run holds one seq_len for a
    whole epoch, so a run resumed mid-epoch must NOT pick up a value from a
    breakpoint the epoch already crossed -- that would rebuild the dataset at a
    different length than its uninterrupted twin and desync the index
    arithmetic. epoch_start_step is checkpointed for exactly this reason."""
    from experimentation.training.schedules import ScheduleApplier
    from experimentation.training.train import epoch_seq_len

    applier = ScheduleApplier(nn.Linear(4, 4), {"data.seq_len": {
        "kind": "piecewise", "at": [0, 50], "values": [512, 1024]}})
    # An epoch that began at step 0 keeps 512 even when the loop is at step 80.
    assert epoch_seq_len(applier, epoch_start_step=0, default=2048) == 512


def test_epoch_seq_len_refuses_to_exceed_the_models_max_seq_len():
    """The model is built once, for model.max_seq_len; a curriculum that walks
    past it would fail deep in the forward on a compute node."""
    from experimentation.training.schedules import ScheduleApplier
    from experimentation.training.train import epoch_seq_len

    applier = ScheduleApplier(nn.Linear(4, 4), {"data.seq_len": {
        "kind": "piecewise", "at": [0, 100], "values": [512, 4096]}})
    with pytest.raises(ValueError, match="max_seq_len|2048"):
        epoch_seq_len(applier, epoch_start_step=100, default=2048)


# --------------------------------------------------------------------------
# The optimizer must cover frozen parameters when a freeze schedule exists
# --------------------------------------------------------------------------

def test_a_freeze_schedule_reports_that_it_freezes_parameters():
    from experimentation.training.schedules import ScheduleApplier

    m = nn.Sequential(nn.Linear(4, 4))
    assert ScheduleApplier(m, {}).freezes_parameters is False
    assert ScheduleApplier(m, {"freeze": {"match": "0.*",
                                          "frozen_until": 10}}).freezes_parameters


# --------------------------------------------------------------------------
# build_train_argv must hand the spec to the trainer (§5.1 ordering: every
# path that builds a model has to see these sections)
# --------------------------------------------------------------------------

def test_build_train_argv_passes_the_materialized_spec_path(tmp_path):
    from experimentation.run.train_argv import build_train_argv
    from experimentation.run.resolve import load_materialized_spec

    path = _write_spec(tmp_path, optim_extra={
        "groups": [{"match": "*.ska.*", "lr_mult": 10.0}]})
    spec = load_materialized_spec(path)
    argv = build_train_argv(spec, tmp_path)
    assert "--spec" in argv
    assert argv[argv.index("--spec") + 1] == str(tmp_path / "spec.yaml")


def test_train_cli_accepts_the_spec_flag():
    from experimentation.training.train import parse_args

    args = parse_args(["--spec", "/tmp/spec.yaml"])
    assert args.spec == "/tmp/spec.yaml"


def test_train_cli_defaults_spec_to_none():
    from experimentation.training.train import parse_args

    assert parse_args([]).spec is None


# --------------------------------------------------------------------------
# Against a REAL KoopmanLM. The unit tests above use a hand-rolled stack, which
# cannot catch the failure that actually matters: a whitelist entry or a
# documented match pattern that does not correspond to anything in the shipped
# architecture. Both would be silent -- a beautiful annealing curve on a value
# the model never reads, or a "matched no parameters" that nobody hits until a
# real run.
#
# An all-SKA config is used because ska_mode='replace' still puts a Mamba2Block
# at every non-SKA index, and mamba_ssm is unavailable on CPU dev boxes.
# --------------------------------------------------------------------------

def _all_ska_model():
    import dataclasses
    from koopman_lm.config import build_config
    from koopman_lm.models.koopman_lm import KoopmanLM

    cfg = build_config("1m")
    cfg = dataclasses.replace(cfg, ska_layer_indices=tuple(range(cfg.n_layers)))
    return KoopmanLM(cfg), cfg


def test_ska_ridge_schedule_reaches_every_ska_module_of_a_real_model():
    from koopman_lm.modules.seq.ska import SKAModule
    from experimentation.training.schedules import ScheduleApplier

    model, cfg = _all_ska_model()
    sites = [m for m in model.modules() if isinstance(m, SKAModule)]
    assert len(sites) == 4
    # Endpoints deliberately chosen to differ from cfg.ska_ridge. With the
    # config default as the target, a SCHEDULABLE entry naming a nonexistent
    # attribute would setattr a fresh field and this test would still pass --
    # verified by mutating "ridge_eps" and watching it stay green.
    assert cfg.ska_ridge not in (5.0e-2, 7.0e-3)
    applier = ScheduleApplier(model, {"model.ska_ridge": {
        "kind": "linear", "from": 5.0e-2, "to": 7.0e-3, "over": [0, 100]}})
    applier.apply(0)
    assert all(m.ridge_eps == pytest.approx(5.0e-2) for m in sites)
    applier.apply(100)
    assert all(m.ridge_eps == pytest.approx(7.0e-3) for m in sites)


def test_the_designs_own_group_patterns_match_a_real_model():
    """§7's worked example is `*.ska.*` and `embed.weight`. If either matched
    nothing, param_groups would raise at optimizer construction."""
    from experimentation.training.optim import ParamGroupSpec, param_groups

    model, _ = _all_ska_model()
    groups = param_groups(model, weight_decay=0.1, lr=4.0e-4, groups=[
        ParamGroupSpec(match="*.ska.*", lr_mult=10.0),
        ParamGroupSpec(match="embed.weight", lr_mult=0.1, weight_decay=0.0),
    ])
    named = dict(model.named_parameters())
    embed_group = [g for g in groups
                   if any(p is named["embed.weight"] for p in g["params"])]
    assert len(embed_group) == 1
    assert embed_group[0]["lr"] == pytest.approx(4.0e-5)
    assert embed_group[0]["weight_decay"] == 0.0
    ska_w = named["seq_layers.0.ska.key_proj.weight"]
    ska_group = [g for g in groups if any(p is ska_w for p in g["params"])]
    assert ska_group[0]["lr"] == pytest.approx(4.0e-3)


def test_every_parameter_of_a_real_model_lands_in_exactly_one_group():
    """A parameter silently dropped from every group is never trained; one in
    two groups is updated twice per step."""
    from experimentation.training.optim import ParamGroupSpec, param_groups

    model, _ = _all_ska_model()
    groups = param_groups(model, weight_decay=0.1, lr=4.0e-4, groups=[
        ParamGroupSpec(match="*.ska.*", lr_mult=10.0),
        ParamGroupSpec(match="embed.weight", lr_mult=0.1),
    ])
    placed = [id(p) for g in groups for p in g["params"]]
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    assert len(placed) == len(set(placed))
    assert set(placed) == expected


def test_a_freeze_pattern_over_a_real_model_freezes_only_the_ska_stack():
    from experimentation.training.schedules import ScheduleApplier

    model, _ = _all_ska_model()
    applier = ScheduleApplier(model, {"freeze": {
        "match": "seq_layers.*.ska.*", "frozen_until": 200}})
    applier.apply(0)
    named = dict(model.named_parameters())
    assert not named["seq_layers.0.ska.key_proj.weight"].requires_grad
    assert named["embed.weight"].requires_grad
    applier.apply(200)
    assert named["seq_layers.0.ska.key_proj.weight"].requires_grad
