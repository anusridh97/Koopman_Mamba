"""Data verification at launch (§3.5): read the shard's meta.json (the format
koopman_lm.training.data.pretokenize.py writes) and assert tokenizer, mix,
and n_tokens match the spec -- failing before any GPU time is spent.
"""
import json

import pytest

pytestmark = pytest.mark.correctness


def _write_meta(shard_dir, **overrides):
    shard_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "n_tokens": 3_000_000_000,
        "tokenizer": "NousResearch/Llama-2-7b-hf",
        "mix": {"fineweb": 1.0, "pg19": 0.0, "scrolls": 0.0},
    }
    meta.update(overrides)
    (shard_dir / "meta.json").write_text(json.dumps(meta))


def test_verify_shard_passes_when_meta_matches(tmp_path):
    from koopman_lm.run.data_verify import verify_shard
    from koopman_lm.run.spec import ShardDataSpec

    shard_dir = tmp_path / "fineweb_50m_train"
    _write_meta(shard_dir)
    data = ShardDataSpec(
        shard_dir=str(shard_dir), tokenizer="NousResearch/Llama-2-7b-hf",
        mix={"fineweb": 1.0, "pg19": 0.0, "scrolls": 0.0}, n_tokens=3_000_000_000,
    )
    verify_shard(data)   # no raise


def test_verify_shard_raises_on_tokenizer_mismatch(tmp_path):
    from koopman_lm.run.data_verify import DataVerificationError, verify_shard
    from koopman_lm.run.spec import ShardDataSpec

    shard_dir = tmp_path / "shard"
    _write_meta(shard_dir, tokenizer="mistralai/Mistral-7B-v0.1")
    data = ShardDataSpec(
        shard_dir=str(shard_dir), tokenizer="NousResearch/Llama-2-7b-hf",
        mix={"fineweb": 1.0, "pg19": 0.0, "scrolls": 0.0}, n_tokens=3_000_000_000,
    )
    with pytest.raises(DataVerificationError, match="tokenizer"):
        verify_shard(data)


def test_verify_shard_raises_on_mix_mismatch(tmp_path):
    from koopman_lm.run.data_verify import DataVerificationError, verify_shard
    from koopman_lm.run.spec import ShardDataSpec

    shard_dir = tmp_path / "shard"
    _write_meta(shard_dir, mix={"fineweb": 0.5, "pg19": 0.5, "scrolls": 0.0})
    data = ShardDataSpec(
        shard_dir=str(shard_dir), tokenizer="NousResearch/Llama-2-7b-hf",
        mix={"fineweb": 1.0, "pg19": 0.0, "scrolls": 0.0}, n_tokens=3_000_000_000,
    )
    with pytest.raises(DataVerificationError, match="mix"):
        verify_shard(data)


def test_verify_shard_raises_on_n_tokens_mismatch(tmp_path):
    from koopman_lm.run.data_verify import DataVerificationError, verify_shard
    from koopman_lm.run.spec import ShardDataSpec

    shard_dir = tmp_path / "shard"
    _write_meta(shard_dir, n_tokens=1_000)
    data = ShardDataSpec(
        shard_dir=str(shard_dir), tokenizer="NousResearch/Llama-2-7b-hf",
        mix={"fineweb": 1.0, "pg19": 0.0, "scrolls": 0.0}, n_tokens=3_000_000_000,
    )
    with pytest.raises(DataVerificationError, match="n_tokens"):
        verify_shard(data)


def test_verify_shard_raises_when_meta_missing(tmp_path):
    from koopman_lm.run.data_verify import DataVerificationError, verify_shard
    from koopman_lm.run.spec import ShardDataSpec

    data = ShardDataSpec(
        shard_dir=str(tmp_path / "does_not_exist"), tokenizer="t",
        mix={"fineweb": 1.0}, n_tokens=10,
    )
    with pytest.raises(DataVerificationError, match="meta.json"):
        verify_shard(data)
