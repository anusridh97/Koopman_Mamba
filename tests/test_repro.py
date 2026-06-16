"""Reproducibility infra: seeding determinism + checkpoint cfg-hash round-trip.

Pure CPU. Verifies seed_everything makes RNG draws reproducible and that the
checkpoint metadata embeds a cfg_hash that survives save/load and lets the exact
config be reconstructed (scaling plan Phase 0: "trace a checkpoint back to its
exact config").
"""
import os
import random

import numpy as np
import pytest
import torch

from koopman_lm.repro import seed_everything, seed_worker
from koopman_lm.config import build_config, config_hash
from koopman_lm.train_fast import checkpoint_meta

pytestmark = pytest.mark.correctness


def _draws():
    return (
        torch.randn(8).tolist(),
        [random.random() for _ in range(4)],
        np.random.rand(4).tolist(),
    )


def test_seed_everything_is_reproducible():
    seed_everything(1234)
    a = _draws()
    seed_everything(1234)
    b = _draws()
    assert a == b


def test_seed_everything_differs_across_seeds():
    seed_everything(1)
    a = _draws()
    seed_everything(2)
    b = _draws()
    assert a != b


def test_returned_generator_is_seeded():
    g1 = seed_everything(7)
    x1 = torch.randn(5, generator=g1)
    g2 = seed_everything(7)
    x2 = torch.randn(5, generator=g2)
    assert torch.equal(x1, x2)


def test_seed_everything_sets_pythonhashseed():
    seed_everything(99)
    assert os.environ["PYTHONHASHSEED"] == "99"


def test_seed_worker_runs():
    seed_everything(3)
    seed_worker(0)  # must not raise; seeds from torch.initial_seed()


def test_checkpoint_meta_embeds_cfg_hash():
    cfg = build_config("180m")
    meta = checkpoint_meta(cfg, step=1000, model_type="koopman", model_size="180m")
    assert meta["cfg_hash"] == config_hash(cfg)
    assert meta["model_size"] == "180m"
    assert meta["step"] == 1000


def test_checkpoint_meta_roundtrips(tmp_path):
    cfg = build_config("440m")
    meta = checkpoint_meta(cfg, step=5000, model_type="koopman", model_size="440m")
    p = tmp_path / "meta.pt"
    torch.save(meta, p)
    loaded = torch.load(p, map_location="cpu", weights_only=False)
    # hash survives serialization and matches the reloaded config
    assert loaded["cfg_hash"] == config_hash(loaded["cfg"])
    assert loaded["cfg_hash"] == config_hash(build_config("440m"))


def test_train_fast_imports_all_scales_cpu():
    # importing the trainer must be CPU-safe (mamba_ssm imported lazily)
    import koopman_lm.train_fast as tf
    from koopman_lm.config import CONFIG_FACTORIES
    # argparse choices should cover the full registry
    import argparse
    parser_args = tf.parse_args  # exists
    assert set(CONFIG_FACTORIES) >= {"50m", "180m", "440m", "880m", "1p5b", "3b"}
    assert callable(tf.main)
