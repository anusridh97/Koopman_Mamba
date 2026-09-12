"""Llama-3.1's 128,256 vocab overflows uint16; the shard dtype must follow it.

The failure this guards is silent, not loud: writing uint32 ids and reading them
back as uint16 fuses every PAIR of tokens into one wrong id, doubling the
apparent token count and training on garbage without raising anything.
"""
import json

import numpy as np
import pytest

from experimentation.training.data.pretokenize import token_dtype_for


def test_dtype_follows_the_vocab():
    assert token_dtype_for(32000) == np.dtype(np.uint16)      # Llama-2
    assert token_dtype_for(65535) == np.dtype(np.uint16)      # exactly the max
    assert token_dtype_for(65536) == np.dtype(np.uint32)      # one past it
    assert token_dtype_for(128256) == np.dtype(np.uint32)     # Llama-3.1
    with pytest.raises(ValueError):
        token_dtype_for(2 ** 33)


def test_llama31_ids_survive_a_round_trip_through_the_shard(tmp_path):
    """The concrete bug: id 128255 written as uint32, read back as uint16."""
    from experimentation.training.data.dataset import MemmapPackedDataset

    vocab, n = 128256, 4096
    dt = token_dtype_for(vocab)
    ids = np.arange(n, dtype=dt) % vocab
    ids[:4] = [128255, 65536, 70000, 1]        # ids uint16 cannot represent
    (tmp_path / "train.bin").write_bytes(ids.tobytes())
    (tmp_path / "weights.bin").write_bytes(np.ones(n, dtype=np.uint8).tobytes())
    (tmp_path / "meta.json").write_text(json.dumps(
        {"n_tokens": n, "vocab_size": vocab, "tokenizer": "meta-llama/Llama-3.1-8B",
         "dtype": dt.name, "weight_dtype": "uint8", "mix": {"fineweb": 1.0}}))

    ds = MemmapPackedDataset(str(tmp_path), max_seq_len=64, seed=0)
    assert ds.data.dtype == np.uint32, "reader ignored meta's dtype"
    assert ds.data[0] == 128255 and ds.data[2] == 70000
    assert len(ds.data) == n, "length doubled -> ids were read at the wrong width"
    assert int(ds.data.max()) < vocab


def test_a_uint16_shard_still_reads_without_a_dtype_key(tmp_path):
    """Backward compatibility: every existing 32k shard predates the key."""
    from experimentation.training.data.dataset import MemmapPackedDataset

    n = 2048
    ids = (np.arange(n, dtype=np.uint16) % 32000)
    (tmp_path / "train.bin").write_bytes(ids.tobytes())
    (tmp_path / "meta.json").write_text(json.dumps(
        {"n_tokens": n, "vocab_size": 32000,
         "tokenizer": "NousResearch/Llama-2-7b-hf", "mix": {"fineweb": 1.0}}))
    ds = MemmapPackedDataset(str(tmp_path), max_seq_len=64, seed=0)
    assert ds.data.dtype == np.uint16 and len(ds.data) == n
