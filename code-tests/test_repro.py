"""Reproducibility infra: seeding determinism.

Pure CPU. Verifies seed_everything makes RNG draws reproducible across
torch/random/numpy and that seed_worker runs without error.
"""
import os
import random

import numpy as np
import pytest
import torch

from koopman_lm.training.repro import seed_everything, seed_worker

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

