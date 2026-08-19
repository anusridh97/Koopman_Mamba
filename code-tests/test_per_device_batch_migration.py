"""per_device_batch_size moves from OptimSpec to RuntimeSpec.

The reason, found while building the OOM ladder: OptimSpec is hashed whole into
run_id, so descending a rung -- halving the microbatch and doubling gradient
accumulation, which preserves effective_batch exactly -- changed a run's identity
even though its science did not. Before this change:

    effective_batch=64, pdbs=16  ->  run_id 58674511
    effective_batch=64, pdbs=8   ->  run_id 02705fcd

RuntimeSpec is documented as "NOT hashed into run_id/group_id -- a different
partition/account/worker count/ddp setting is the same experiment run
differently", which is exactly what a microbatch is. effective_batch stays in
OptimSpec, because that one genuinely changes the result.

Two consequences this pins.

The divisibility constraint now spans two specs, so it moves to
RunSpec.__post_init__ -- the only place that sees both. It must still fire, or a
spec that cannot be split into whole steps launches and silently trains on a
different batch than it declares.

Old materialized spec.yaml files carry `optim.per_device_batch_size`, and a bare
move would make every existing run directory unloadable. So the loaders migrate
the key forward with a warning, following the same "deprecated, not deleted"
precedent the precision design sets for runtime.precision.
"""
import textwrap

import pytest
import yaml

pytestmark = pytest.mark.correctness


def _spec(pdbs=16, effective=64):
    from koopman_lm.config import build_config
    from experimentation.run.spec import (OptimSpec, RunSpec, RuntimeSpec,
                                          ShardDataSpec)
    return RunSpec(
        name="x", model=build_config("50m"),
        data=ShardDataSpec(kind="shard", shard_dir="/tmp/s", tokenizer="t",
                           mix={"f": 1.0}, n_tokens=10),
        optim=OptimSpec(lr=4e-4, warmup_steps=10, max_steps=100,
                        effective_batch=effective),
        runtime=RuntimeSpec(seed=42, per_device_batch_size=pdbs),
    )


# ------------------------------------------------------------- placement ----

def test_the_microbatch_lives_on_runtime():
    from experimentation.run.spec import RuntimeSpec

    assert RuntimeSpec().per_device_batch_size == 8
    assert RuntimeSpec(per_device_batch_size=16).per_device_batch_size == 16


def test_optim_no_longer_declares_the_microbatch():
    import dataclasses

    from experimentation.run.spec import OptimSpec

    names = {f.name for f in dataclasses.fields(OptimSpec)}
    assert "per_device_batch_size" not in names
    assert "effective_batch" in names, (
        "effective batch stays: it genuinely changes the result")


# ---------------------------------------------------------- the payoff ----

def test_the_microbatch_no_longer_affects_run_identity():
    """The whole reason for the move. An OOM ladder descends rungs without
    renaming the experiment."""
    from experimentation.run.spec import group_id, run_id

    for a, b in ((16, 8), (8, 4), (4, 1)):
        assert run_id(_spec(a)) == run_id(_spec(b))
        assert group_id(_spec(a)) == group_id(_spec(b))


def test_the_effective_batch_still_affects_run_identity():
    """The other half: if this collapsed too, the move would have thrown away a
    real scientific input."""
    from experimentation.run.spec import run_id

    assert run_id(_spec(effective=64)) != run_id(_spec(effective=96))


def test_the_run_directory_no_longer_moves_with_the_microbatch():
    from experimentation.run.spec import run_dir_path

    assert run_dir_path("/runs", _spec(16)) == run_dir_path("/runs", _spec(2))


# -------------------------------------------------------- the constraint ----

def test_an_indivisible_split_is_still_rejected():
    """Now a cross-spec check, so it lives on RunSpec. Losing it would let a run
    train on a different batch than it declares."""
    with pytest.raises(ValueError, match="multiple of"):
        _spec(pdbs=7, effective=64)


def test_a_divisible_split_is_accepted():
    assert _spec(pdbs=16, effective=64).runtime.per_device_batch_size == 16


def test_the_constraint_message_names_both_fields():
    """It spans two specs now, so the message has to say which two."""
    with pytest.raises(ValueError) as excinfo:
        _spec(pdbs=7, effective=64)
    message = str(excinfo.value)
    assert "effective_batch" in message
    assert "per_device_batch_size" in message


# ----------------------------------------------------------- migration ----

def _legacy_spec_yaml(tmp_path, section="optim"):
    """A spec written before the move: microbatch under optim:."""
    shard = tmp_path / "shard"
    shard.mkdir(exist_ok=True)
    path = tmp_path / "legacy.yaml"
    path.write_text(textwrap.dedent(f"""
        name: legacy-run
        model: 50m
        data:
          kind: shard
          shard_dir: {shard}
          tokenizer: NousResearch/Llama-2-7b-hf
          mix: {{fineweb: 1.0}}
          n_tokens: 1000
        optim:
          lr: 0.0004
          warmup_steps: 10
          max_steps: 100
          effective_batch: 64
          per_device_batch_size: 16
        runtime:
          seed: 42
    """))
    return path


def test_a_pre_move_authoring_spec_still_resolves(tmp_path):
    """Every configs/runs/*.yaml in the wild, and anything a teammate has locally,
    puts the microbatch under optim:."""
    from experimentation.run.resolve import resolve_run_spec

    spec = resolve_run_spec(_legacy_spec_yaml(tmp_path))
    assert spec.runtime.per_device_batch_size == 16


def test_the_migration_warns_rather_than_migrating_silently(tmp_path):
    """A silent move would leave the file looking correct while meaning something
    new. The warning is what prompts the file to be updated."""
    from experimentation.run.resolve import resolve_run_spec

    with pytest.warns(DeprecationWarning, match="per_device_batch_size"):
        resolve_run_spec(_legacy_spec_yaml(tmp_path))


def test_a_pre_move_materialized_spec_still_loads(tmp_path):
    """Existing run directories must stay readable -- their spec.yaml is the only
    record of what they ran."""
    from experimentation.run.resolve import load_materialized_spec, to_flat_dict
    from koopman_lm.config import build_config

    flat = to_flat_dict(_spec(16))
    # Rewrite it into the pre-move shape.
    flat["optim"]["per_device_batch_size"] = flat["runtime"].pop(
        "per_device_batch_size")
    path = tmp_path / "spec.yaml"
    path.write_text(yaml.safe_dump(flat, sort_keys=False))

    with pytest.warns(DeprecationWarning):
        spec = load_materialized_spec(path)
    assert spec.runtime.per_device_batch_size == 16


def test_a_post_move_spec_does_not_warn(tmp_path):
    import warnings

    from experimentation.run.resolve import load_materialized_spec, to_flat_dict

    path = tmp_path / "spec.yaml"
    path.write_text(yaml.safe_dump(to_flat_dict(_spec(16)), sort_keys=False))
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        assert load_materialized_spec(path).runtime.per_device_batch_size == 16


def test_a_conflicting_duplicate_is_an_error(tmp_path):
    """If both sections name it, guessing which one wins would be worse than
    refusing."""
    from experimentation.run.resolve import resolve_run_spec

    path = _legacy_spec_yaml(tmp_path)
    raw = yaml.safe_load(path.read_text())
    raw["runtime"]["per_device_batch_size"] = 4        # disagrees with optim's 16
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="both"):
        resolve_run_spec(path)


# ------------------------------------------------------------- argv ----

def test_the_training_command_reads_the_microbatch_from_runtime():
    from experimentation.run.train_argv import build_train_argv

    argv = build_train_argv(_spec(pdbs=16, effective=64), "/tmp/run")
    index = argv.index("--per_device_train_batch_size")
    assert argv[index + 1] == "16"
    accum = argv.index("--gradient_accumulation_steps")
    assert argv[accum + 1] == "4", "16 x 4 = the declared effective batch of 64"


# --------------------------------------------------------- the ladder ----

def test_the_oom_ladder_now_keeps_one_identity_across_rungs():
    """The finding that motivated all of this, stated as its resolution."""
    from experimentation.run.spec import run_id
    from experimentation.run.train_argv import batch_plans

    ids = {run_id(_spec(pdbs=pdbs, effective=64))
           for pdbs, _ in batch_plans(64, 16)}
    assert len(ids) == 1, (
        "every rung of the ladder must now be the same run identity")
