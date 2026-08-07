"""Exact resume (§5): RNG capture/restore, deterministic epoch-permutation
replay (dataloader position), and resume-state save/load/apply. All pure CPU
-- no mamba_ssm, no GPU. See docs/superpowers/specs/2026-08-07-run-system-design.md §5.
"""
import random

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.correctness


def _draw_everything():
    return (
        random.random(),
        float(np.random.rand()),
        torch.randn(3).tolist(),
    )


def test_rng_roundtrip_reproduces_subsequent_draws():
    from koopman_lm.training.resume import capture_rng_state, restore_rng_state

    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    # burn some draws so state is "mid-stream", not fresh-seeded
    _draw_everything()
    state = capture_rng_state()
    expected = _draw_everything()

    # perturb all three streams
    random.random(); np.random.rand(); torch.randn(1)

    restore_rng_state(state)
    actual = _draw_everything()
    assert actual == expected


def test_capture_rng_state_has_expected_keys():
    from koopman_lm.training.resume import capture_rng_state

    state = capture_rng_state()
    assert set(state) == {"python", "numpy", "torch", "torch_cuda"}
    if not torch.cuda.is_available():
        assert state["torch_cuda"] is None
