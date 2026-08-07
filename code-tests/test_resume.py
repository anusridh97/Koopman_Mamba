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


def test_epoch_permutation_matches_a_live_random_sampler(tmp_path):
    """epoch_permutation must reproduce exactly what train.py's DataLoader
    (shuffle=True, generator=<seed>) actually draws for a given epoch -- this
    is the mechanism §5.2 relies on to reconstruct dataloader position from
    (seed, epoch) alone."""
    from torch.utils.data import RandomSampler

    from koopman_lm.training.data.pretokenize import write_synthetic_corpus
    from koopman_lm.training.data.dataset import MemmapPackedDataset
    from koopman_lm.training.resume import epoch_permutation

    data_dir = write_synthetic_corpus(str(tmp_path / "data"), n_tokens=20_000,
                                       vocab_size=64, seed=0)
    ds = MemmapPackedDataset(data_dir, max_seq_len=32, seed=0)

    seed = 4242
    g = torch.Generator()
    g.manual_seed(seed)
    sampler = RandomSampler(ds, generator=g)
    live_epoch_0 = list(iter(sampler))
    live_epoch_1 = list(iter(sampler))
    live_epoch_2 = list(iter(sampler))

    assert epoch_permutation(len(ds), seed, 0) == live_epoch_0
    assert epoch_permutation(len(ds), seed, 1) == live_epoch_1
    assert epoch_permutation(len(ds), seed, 2) == live_epoch_2


def test_resume_indices_is_the_tail_of_the_epoch_permutation():
    from koopman_lm.training.resume import epoch_permutation, resume_indices

    full = epoch_permutation(100, seed=7, epoch=3)
    tail = resume_indices(100, seed=7, epoch=3, samples_consumed=40)
    assert tail == full[40:]
    assert len(tail) == 60


import torch.nn as nn


def _tiny_optimizer_and_scheduler():
    model = nn.Linear(4, 4)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: 1.0 - step / 100)
    return model, opt, sched


def test_resume_state_round_trips_optimizer_scheduler_and_position(tmp_path):
    from koopman_lm.training.resume import save_resume_state, load_resume_state, apply_resume_state

    model, opt, sched = _tiny_optimizer_and_scheduler()
    x = torch.randn(2, 4)
    for _ in range(5):
        loss = model(x).sum()
        loss.backward()
        opt.step(); sched.step(); opt.zero_grad()

    path = tmp_path / "resume.pt"
    save_resume_state(path, step=5, epoch=0, samples_consumed=17,
                       optimizer=opt, scheduler=sched)
    # re-read the just-written file for the "expected" snapshot -- opt.state_dict()
    # returns live references to the optimizer's own tensors, which the next
    # opt.step() calls mutate in place, so comparing against it later would
    # silently compare against post-mutation values instead of the saved ones.
    saved_snapshot = load_resume_state(path)
    saved_opt_sd = saved_snapshot["optimizer"]
    saved_sched_sd = saved_snapshot["scheduler"]

    # advance further -- this must NOT be what gets restored
    for _ in range(3):
        loss = model(x).sum()
        loss.backward()
        opt.step(); sched.step(); opt.zero_grad()
    assert sched.state_dict()["_step_count"] != saved_sched_sd["_step_count"]

    # fresh optimizer/scheduler, as a resumed process would build
    _, fresh_opt, fresh_sched = _tiny_optimizer_and_scheduler()
    state = load_resume_state(path)
    step, epoch, samples_consumed = apply_resume_state(state, optimizer=fresh_opt, scheduler=fresh_sched)

    assert (step, epoch, samples_consumed) == (5, 0, 17)
    assert fresh_sched.state_dict()["_step_count"] == saved_sched_sd["_step_count"]
    # AdamW moment tensors match exactly (not the fresh, zero-initialized ones)
    for group_a, group_b in zip(fresh_opt.state_dict()["state"].values(), saved_opt_sd["state"].values()):
        assert torch.equal(group_a["exp_avg"], group_b["exp_avg"])


def test_apply_resume_state_restores_rng_by_default(tmp_path):
    from koopman_lm.training.resume import save_resume_state, load_resume_state, apply_resume_state

    _, opt, sched = _tiny_optimizer_and_scheduler()
    random.seed(9); np.random.seed(9); torch.manual_seed(9)
    random.random(); np.random.rand(); torch.randn(1)  # burn in

    path = tmp_path / "resume.pt"
    save_resume_state(path, step=0, epoch=0, samples_consumed=0, optimizer=opt, scheduler=sched)
    expected = (random.random(), float(np.random.rand()), torch.randn(1).item())

    random.random(); np.random.rand(); torch.randn(1)  # perturb
    state = load_resume_state(path)
    apply_resume_state(state, optimizer=opt, scheduler=sched)
    actual = (random.random(), float(np.random.rand()), torch.randn(1).item())
    assert actual == expected
