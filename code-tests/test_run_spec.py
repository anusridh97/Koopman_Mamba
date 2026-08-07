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
