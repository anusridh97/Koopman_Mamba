"""§5.5, the acceptance criterion for exact resume: training N steps, killing,
resuming, and training to 2N must produce the same weights as training 2N
uninterrupted, under determinism.

mamba_ssm isn't installed and there's no GPU, so KoopmanLM can't be
instantiated here. The resume logic under test (RNG capture/restore, the
epoch-permutation replay, and resume-state save/load/apply) is model-agnostic
by construction -- that's what makes this loop, built around a tiny stand-in
nn.Module, an honest test of the same code path train.py's Task 7 loop uses,
not a separate reimplementation.

Tolerance: exact equality (torch.equal, atol=0). With torch.set_num_threads(1)
and torch.use_deterministic_algorithms(True), every op in this loop (matmul,
addmm, embedding lookup, dropout mask generation from a restored RNG stream,
AdamW's elementwise updates) is a deterministic function of its inputs and the
RNG stream state -- there is no GPU, no multi-threaded reduction, and no
cuDNN algorithm-selection nondeterminism in play, so bitwise equality is the
correct bar, not an approximation of it.
"""
import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from experimentation.training.data.dataset import MemmapPackedDataset
from experimentation.training.data.pretokenize import write_synthetic_corpus
from experimentation.training.repro import seed_everything
from experimentation.training.resume import (
    apply_resume_state, epoch_permutation, load_resume_state,
    resume_indices, save_resume_state,
)

pytestmark = pytest.mark.correctness

VOCAB = 64
SEQ_LEN = 16
HIDDEN = 32


class _StandInLM(nn.Module):
    """Not KoopmanLM (needs mamba_ssm/GPU) -- a couple of Linear layers plus
    dropout. Dropout makes RNG restoration load-bearing: without it, resumed
    weights would match using only optimizer-state restore, which would not
    actually exercise §5.2's RNG requirement."""

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, HIDDEN)
        self.drop = nn.Dropout(p=0.3)
        self.lin1 = nn.Linear(HIDDEN, HIDDEN)
        self.lin2 = nn.Linear(HIDDEN, VOCAB)

    def forward(self, input_ids):
        x = self.embed(input_ids)
        x = self.drop(x)
        x = F.relu(self.lin1(x))
        return self.lin2(x)


def _make_dataset(tmp_path):
    data_dir = write_synthetic_corpus(str(tmp_path / "data"), n_tokens=8_000,
                                       vocab_size=VOCAB, seed=0)
    return MemmapPackedDataset(data_dir, max_seq_len=SEQ_LEN, seed=0)


def _lr_lambda(step, warmup, max_steps):
    if step < warmup:
        return step / max(warmup, 1)
    t = (step - warmup) / max(max_steps - warmup, 1)
    return 0.5 * (1 + math.cos(math.pi * t))


def _run_loop(ds, seed, batch_size, start_step, end_step, warmup, max_steps,
              model, optimizer, scheduler, start_epoch=0, start_samples_consumed=0):
    """Mirrors train.py's Task-7 epoch loop: explicit per-epoch index order via
    epoch_permutation/resume_indices, so a mid-epoch resume is index
    arithmetic, never a second data read of already-consumed samples."""
    step = start_step
    epoch = start_epoch
    model.train()
    while step < end_step:
        samples_consumed = 0
        if epoch == start_epoch and start_samples_consumed > 0:
            indices = resume_indices(len(ds), seed, epoch, start_samples_consumed)
            samples_consumed = start_samples_consumed
        else:
            indices = epoch_permutation(len(ds), seed, epoch)
        # generator= mirrors train.py: DataLoader.__iter__ otherwise draws from
        # the GLOBAL torch RNG to seed _base_seed, so a mid-epoch resume pays a
        # draw the uninterrupted run never pays there and dropout masks desync.
        loader = DataLoader(ds, batch_size=batch_size, sampler=indices,
                            drop_last=True, generator=torch.Generator())
        for batch in loader:
            if step >= end_step:
                break
            ids = batch["input_ids"]
            labels = batch["labels"]
            logits = model(ids)
            loss = F.cross_entropy(logits.reshape(-1, VOCAB), labels.reshape(-1))
            loss.backward()
            optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
            step += 1
            samples_consumed += ids.size(0)
            if step >= end_step:
                return step, epoch, samples_consumed
        epoch += 1
    return step, epoch, 0


def _build_model_optim_sched(seed, warmup, max_steps):
    seed_everything(seed)
    model = _StandInLM()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, betas=(0.9, 0.95))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: _lr_lambda(step, warmup, max_steps))
    return model, optimizer, scheduler


def test_interrupted_and_resumed_run_matches_uninterrupted_run(tmp_path):
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

    ds = _make_dataset(tmp_path)
    seed = 1234
    batch_size = 4
    N, warmup, max_steps = 20, 5, 40

    # --- Run A: uninterrupted, straight to 2N ---
    model_a, opt_a, sched_a = _build_model_optim_sched(seed, warmup, max_steps)
    _run_loop(ds, seed, batch_size, 0, 2 * N, warmup, max_steps, model_a, opt_a, sched_a)

    # --- Run B: N steps, "kill", resume, train to 2N ---
    model_b, opt_b, sched_b = _build_model_optim_sched(seed, warmup, max_steps)
    step, epoch, samples_consumed = _run_loop(
        ds, seed, batch_size, 0, N, warmup, max_steps, model_b, opt_b, sched_b)

    ckpt_path = tmp_path / "model_at_kill.pt"
    resume_path = tmp_path / "resume.pt"
    torch.save(model_b.state_dict(), ckpt_path)
    save_resume_state(resume_path, step=step, epoch=epoch,
                       samples_consumed=samples_consumed,
                       optimizer=opt_b, scheduler=sched_b)

    # simulate the kill: throw away every in-memory object and rebuild fresh,
    # exactly as a new `python -m experimentation.training.train --resume` process would
    del model_b, opt_b, sched_b

    model_c = _StandInLM()  # fresh init -- immediately overwritten by the checkpoint
    model_c.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
    optimizer_c = torch.optim.AdamW(model_c.parameters(), lr=1e-2, betas=(0.9, 0.95))
    scheduler_c = torch.optim.lr_scheduler.LambdaLR(
        optimizer_c, lambda step: _lr_lambda(step, warmup, max_steps))
    state = load_resume_state(resume_path)
    # apply_resume_state restores RNG too -- must run AFTER model_c's fresh
    # (RNG-consuming) init and BEFORE the resumed loop's first forward pass,
    # or the very state being restored gets perturbed by initialization draws
    # that happen after it, or the resumed dropout masks diverge from Run A's.
    r_step, r_epoch, r_samples_consumed = apply_resume_state(
        state, optimizer=optimizer_c, scheduler=scheduler_c)
    assert (r_step, r_epoch) == (step, epoch)

    _run_loop(ds, seed, batch_size, r_step, 2 * N, warmup, max_steps, model_c,
              optimizer_c, scheduler_c, start_epoch=r_epoch,
              start_samples_consumed=r_samples_consumed)

    sd_a = model_a.state_dict()
    sd_c = model_c.state_dict()
    assert set(sd_a) == set(sd_c)
    for k in sd_a:
        assert torch.equal(sd_a[k], sd_c[k]), f"{k} diverged after resume"


# ------------------------------------------------- the logging window ----
#
# Found by the resume golden (job 440211), which is the first thing to compare a
# resumed run against an uninterrupted one through train.py's OWN loop. Training
# was bit-exact -- every step from 280 to 400 matched at 0.000000 -- but the
# single log line at step 270 differed by 0.0011.
#
# Cause: `running_loss`/`loss_count` are a WINDOW, reset after each log. A run
# that stopped at 266 and resumed reported the average over 267..270 while the
# uninterrupted run reported 261..270. Same training, different denominator.
#
# Worth fixing rather than excusing, because the alternative is a standing
# "ignore the first post-resume log line" exemption in the comparator -- at
# exactly the step where a real resume bug would first appear.


def test_resume_state_round_trips_a_nested_log_window(tmp_path):
    """train.py stashes the window through `extra`, so the contract it depends on
    is that `extra` survives with nested dicts and float/int types intact."""
    import torch

    from experimentation.training.resume import (load_resume_state,
                                                 save_resume_state)

    model = nn.Linear(3, 3)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.0)

    path = tmp_path / "resume.pt"
    save_resume_state(path, step=266, epoch=0, samples_consumed=2128,
                      optimizer=opt, scheduler=sched,
                      extra={"log_window": {"running_loss": 39.7248,
                                            "loss_count": 6}})
    state = load_resume_state(path)
    assert state["log_window"] == {"running_loss": 39.7248, "loss_count": 6}
    assert state["step"] == 266


def test_a_resume_state_without_a_log_window_still_loads(tmp_path):
    """Every resume.pt written before this existed has no log_window, and such a
    run must still resume -- reporting one short window, which is the behaviour
    it had anyway."""
    import torch

    from experimentation.training.resume import (load_resume_state,
                                                 save_resume_state)

    model = nn.Linear(3, 3)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.0)

    path = tmp_path / "resume.pt"
    save_resume_state(path, step=10, epoch=0, samples_consumed=80,
                      optimizer=opt, scheduler=sched)
    state = load_resume_state(path)
    assert "log_window" not in state
    # The exact expression train.py uses, so this fails if that idiom changes.
    win = state.get("log_window") or {}
    assert float(win.get("running_loss", 0.0)) == 0.0
    assert int(win.get("loss_count", 0)) == 0


def test_train_py_both_saves_and_restores_the_window():
    """A tripwire, not a proof -- the proof is the resume golden on a GPU. But
    deleting either half of this would silently reintroduce a 0.0011 step and the
    CPU suite would stay green, so the two halves are asserted to both exist."""
    import pathlib as _pl

    src = (_pl.Path(__file__).resolve().parents[1]
           / "experimentation/training/train.py").read_text()

    assert 'extra={"log_window"' in src, \
        "train.py no longer PERSISTS the logging window across a save"
    assert 'resume_state.get("log_window")' in src, \
        "train.py no longer RESTORES the logging window on resume"
    # Saving without restoring, or vice versa, is the failure mode that reads as
    # working -- both must reference the same key.
    assert src.count("log_window") >= 3

