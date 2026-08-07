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
