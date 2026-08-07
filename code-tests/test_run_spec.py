"""RunSpec sub-specs: DataSpec (polymorphic), OptimSpec, RuntimeSpec, RunSpec.

Pure-Python, no torch model instantiation -- runs on CPU/CI. See
docs/superpowers/specs/2026-08-07-run-system-design.md §3.1, §3.3.
"""
import dataclasses

import pytest

pytestmark = pytest.mark.correctness


def test_shard_data_spec_round_trips_fields():
    from koopman_lm.run.spec import ShardDataSpec

    d = ShardDataSpec(
        shard_dir="/scratch/data/fineweb_50m_train",
        tokenizer="NousResearch/Llama-2-7b-hf",
        mix={"fineweb": 1.0},
        n_tokens=3_000_000_000,
    )
    assert d.kind == "shard"
    assert d.shard_dir == "/scratch/data/fineweb_50m_train"
    assert d.n_tokens == 3_000_000_000
    with pytest.raises(dataclasses.FrozenInstanceError):
        d.n_tokens = 1


def test_shard_data_spec_rejects_missing_fields():
    from koopman_lm.run.spec import ShardDataSpec

    with pytest.raises(ValueError):
        ShardDataSpec(shard_dir="", tokenizer="t", mix={"fineweb": 1.0}, n_tokens=10)
    with pytest.raises(ValueError):
        ShardDataSpec(shard_dir="d", tokenizer="", mix={"fineweb": 1.0}, n_tokens=10)
    with pytest.raises(ValueError):
        ShardDataSpec(shard_dir="d", tokenizer="t", mix={"fineweb": 1.0}, n_tokens=0)


def test_synthetic_data_spec_validates_generator_name():
    from koopman_lm.run.spec import SyntheticDataSpec

    d = SyntheticDataSpec(generator="mqar", params={"num_kv_pairs": 8})
    assert d.kind == "synthetic"
    with pytest.raises(ValueError):
        SyntheticDataSpec(generator="not_a_real_generator", params={})


def test_data_spec_from_dict_dispatches_on_kind():
    from koopman_lm.run.spec import (
        ShardDataSpec, SyntheticDataSpec, data_spec_from_dict,
    )

    shard = data_spec_from_dict({
        "kind": "shard", "shard_dir": "d", "tokenizer": "t",
        "mix": {"fineweb": 1.0}, "n_tokens": 100,
    })
    assert isinstance(shard, ShardDataSpec)

    synth = data_spec_from_dict({
        "kind": "synthetic", "generator": "niah", "params": {},
    })
    assert isinstance(synth, SyntheticDataSpec)

    with pytest.raises(ValueError):
        data_spec_from_dict({"kind": "bogus"})


def test_optim_spec_validates_batch_divisibility():
    from koopman_lm.run.spec import OptimSpec

    ok = OptimSpec(lr=4e-4, warmup_steps=300, max_steps=15000,
                    effective_batch=96, per_device_batch_size=16)
    assert ok.schedule == "cosine"
    with pytest.raises(ValueError):
        OptimSpec(lr=4e-4, warmup_steps=300, max_steps=15000,
                   effective_batch=100, per_device_batch_size=16)


def test_optim_spec_rejects_non_numeric_lr():
    from koopman_lm.run.spec import OptimSpec

    # The exact PyYAML footgun: yaml.safe_load("lr: 4e-4") parses to the
    # STRING "4e-4" (no decimal point in the mantissa), not a float.
    with pytest.raises(TypeError):
        OptimSpec(lr="4e-4", warmup_steps=300, max_steps=15000)


def test_runtime_spec_batch_defaults_are_valid():
    from koopman_lm.run.spec import RuntimeSpec

    rt = RuntimeSpec()
    assert rt.partition == "batch"
    assert rt.account == "marlowe-m000151-pm06"
    assert rt.qos == "medium"
    assert rt.gpu_arch == "9.0"   # H100 (sm_90), not B200/sm_100


def test_runtime_spec_rejects_batch_with_default_account():
    from koopman_lm.run.spec import RuntimeSpec

    with pytest.raises(ValueError):
        RuntimeSpec(partition="batch", account="marlowe-m000151")


def test_runtime_spec_rejects_batch_with_wrong_qos():
    from koopman_lm.run.spec import RuntimeSpec

    with pytest.raises(ValueError):
        RuntimeSpec(partition="batch", qos="normal")


def test_runtime_spec_rejects_unknown_partition():
    from koopman_lm.run.spec import RuntimeSpec

    with pytest.raises(ValueError):
        RuntimeSpec(partition="not_a_real_partition")


def test_runtime_spec_hero_and_preempt_allowed():
    from koopman_lm.run.spec import RuntimeSpec

    # hero/preempt have no stated account/qos constraint -- only 'batch' does.
    RuntimeSpec(partition="hero")
    RuntimeSpec(partition="preempt")


def test_run_spec_type_checks_its_fields():
    from koopman_lm.config import build_config
    from koopman_lm.run.spec import (
        OptimSpec, RuntimeSpec, RunSpec, SyntheticDataSpec,
    )

    cfg = build_config("50m")
    optim = OptimSpec(lr=4e-4, warmup_steps=300, max_steps=15000)
    runtime = RuntimeSpec()
    data = SyntheticDataSpec(generator="mqar", params={})

    spec = RunSpec(name="50m-mqar-smoke", model=cfg, data=data, optim=optim, runtime=runtime)
    assert spec.name == "50m-mqar-smoke"

    with pytest.raises(ValueError):
        RunSpec(name="", model=cfg, data=data, optim=optim, runtime=runtime)
    with pytest.raises(TypeError):
        RunSpec(name="x", model="not-a-config", data=data, optim=optim, runtime=runtime)
    with pytest.raises(TypeError):
        RunSpec(name="x", model=cfg, data="not-a-data-spec", optim=optim, runtime=runtime)


def _make_spec(seed=42, lr=4e-4, partition="batch", account="marlowe-m000151-pm06",
                qos="medium", workers=4, ddp=False):
    from koopman_lm.config import build_config
    from koopman_lm.run.spec import OptimSpec, RuntimeSpec, RunSpec, SyntheticDataSpec

    return RunSpec(
        name="50m-mqar-smoke",
        model=build_config("50m"),
        data=SyntheticDataSpec(generator="mqar", params={"num_kv_pairs": 8}),
        optim=OptimSpec(lr=lr, warmup_steps=300, max_steps=15000),
        runtime=RuntimeSpec(seed=seed, partition=partition, account=account,
                              qos=qos, workers=workers, ddp=ddp),
    )


def test_group_id_and_run_id_are_stable_hex8():
    from koopman_lm.run.spec import group_id, run_id

    spec = _make_spec()
    g1, g2 = group_id(spec), group_id(_make_spec())
    r1, r2 = run_id(spec), run_id(_make_spec())
    assert g1 == g2 and len(g1) == 8 and all(c in "0123456789abcdef" for c in g1)
    assert r1 == r2 and len(r1) == 8


def test_run_id_changes_with_seed_but_group_id_does_not():
    from koopman_lm.run.spec import group_id, run_id

    a, b = _make_spec(seed=42), _make_spec(seed=1337)
    assert group_id(a) == group_id(b)
    assert run_id(a) != run_id(b)


def test_group_id_changes_with_scientific_inputs():
    from koopman_lm.run.spec import group_id

    a, b = _make_spec(lr=4e-4), _make_spec(lr=3e-4)
    assert group_id(a) != group_id(b)


def test_group_id_and_run_id_ignore_execution_details():
    from koopman_lm.run.spec import group_id, run_id

    a = _make_spec(partition="batch", account="marlowe-m000151-pm06", qos="medium", workers=4)
    b = _make_spec(partition="hero", account="marlowe-m000151-pm06", qos="medium", workers=16)
    assert group_id(a) == group_id(b)
    assert run_id(a) == run_id(b)


def test_run_dir_path_layout():
    from koopman_lm.run.spec import attempt_dir_name, group_id, run_dir_name, run_dir_path, run_id

    spec = _make_spec()
    path = run_dir_path("/scratch/runs", spec)
    assert path.parts[-2] == run_dir_name(spec) == f"50m-mqar-smoke.{group_id(spec)}"
    assert path.parts[-1] == attempt_dir_name(spec) == f"seed42.{run_id(spec)}"
