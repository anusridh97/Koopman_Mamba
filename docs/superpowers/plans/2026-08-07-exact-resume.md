# Exact Resume Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `koopman_lm/training/train.py` able to resume exactly — optimizer, LR
schedule, RNG streams, and dataloader position, not just weights — implementing
§5 (Exact resume) of `docs/superpowers/specs/2026-08-07-run-system-design.md`.
The acceptance criterion is §5.5: training *N* steps, killing, resuming, and
training to *2N* must produce the same weights as training *2N* uninterrupted,
under `--deterministic`. That is implemented here as a real passing test, not a
claim.

**Architecture:** A new, model-agnostic module `koopman_lm/training/resume.py`
holds every piece of resume logic that does not require a real `KoopmanLM`
(RNG capture/restore, deterministic epoch-permutation replay, resume-state
save/load/apply). `koopman_lm/run/artifacts.py` gains one atomic binary-write
primitive (`atomic_torch_save`) that `resume.py` reuses — no second atomic-write
implementation. `koopman_lm/training/train.py` is wired to call `resume.py`'s
functions: a `--resume` flag, a SIGUSR1 handler, and a restructured epoch loop
that builds each epoch's index order explicitly (so a forward skip is index
arithmetic, never a data read). `koopman_lm/run/slurm.py`'s sbatch template
gains `--signal=B:USR1@300` and `--requeue`. Nothing here touches
`mqar_finetune.py`, `table2.py`, or `experiments/` — those are consolidated in
a later plan (§6).

**Tech Stack:** Python 3.10 stdlib (`signal`, `random`, `dataclasses`,
`pathlib`), `torch` 2.13.0+cpu (`torch.optim`, RNG state, `torch.randperm`),
`numpy`, `pytest`. No new third-party dependencies.

## Global Constraints

- Venv: `/scratch/m000151/jkli/venvs/koopman-cpu` (`$V` below) — Python 3.10.12,
  torch 2.13.0+cpu, numpy 2.2.6. `tomllib` does not exist here.
- Full suite: `PYTHONPATH=. $V/bin/pytest code-tests -q -p no:cacheprovider`.
  Baseline is **4 failed, 296 passed, 28 skipped** — `test_gated_variant_differs_only_in_mlp_gated`,
  `test_param_count_estimate_reflects_v2_additions`,
  `test_baseline_embedding_std_and_tied_weight_identity[build_transformer]`,
  `test_transformer_initial_logit_scale_and_loss_near_uniform`. These are
  PARKED and out of scope. Every task below must leave exactly those 4 failing
  and add zero new failures/errors.
- No GPU, no `mamba_ssm` — `KoopmanLM` cannot be instantiated. All new tests
  use a tiny stand-in `nn.Module` (Linear/Embedding/Dropout, no Mamba), plus
  the real `MemmapPackedDataset` (pure numpy/torch, no CUDA dependency) so the
  dataloader-position logic is tested against the actual class it protects.
- `yaml.safe_dump` cannot serialize `torch.__version__` (`TorchVersion` is a
  `str` subclass) — always `str(torch.__version__)` before it goes into any
  saved dict that might later be YAML-dumped.
- Do NOT modify `mqar_finetune.py`, `table2.py`, anything under `experiments/`,
  or the shell scripts in `scripts/`.
- Do NOT touch `.worktrees/gpu-env` or `.worktrees/trainer-unify`.

---

## File Structure

| Path | Responsibility | Task |
|---|---|---|
| `koopman_lm/run/artifacts.py` | + `atomic_torch_save()` | 1 |
| `koopman_lm/training/resume.py` | RNG capture/restore, epoch-permutation replay, resume-state save/load/apply | 2, 3, 4 |
| `koopman_lm/run/slurm.py` | sbatch template gains `--signal=B:USR1@300`, `--requeue` | 5 |
| `koopman_lm/training/train.py` | SIGUSR1 handler, `--resume` CLI flag, restructured epoch loop, paired archival+resume writes | 6, 7 |
| `code-tests/test_run_artifacts.py` | Task 1 test | 1 |
| `code-tests/test_resume.py` | Task 2, 3, 4 tests | 2, 3, 4 |
| `code-tests/test_run_slurm.py` | Task 5 test | 5 |
| `code-tests/test_train_preemption_handler.py` | Task 6 test | 6 |
| `code-tests/test_resume_equivalence.py` | Task 8, the §5.5 acceptance test | 8 |

---

### Task 1: `atomic_torch_save` — the binary counterpart of `atomic_write_text`

**Files:**
- Modify: `koopman_lm/run/artifacts.py`
- Modify: `code-tests/test_run_artifacts.py`

**Interfaces:**
- Consumes: nothing new (stdlib `os`, `pathlib.Path`; `torch.save`, imported lazily inside the function so `koopman_lm.run.artifacts` keeps working in contexts without torch).
- Produces: `atomic_torch_save(path, obj: Any) -> None` — `torch.save(obj, tmp)` then `os.replace(tmp, path)`, mirroring `atomic_write_text`'s temp-file-plus-rename pattern exactly (same `.tmp{pid}` suffix convention).

- [ ] **Step 1: Write the failing test**

```python
# appended to code-tests/test_run_artifacts.py

def test_atomic_torch_save_round_trips_and_leaves_no_tmp(tmp_path):
    import torch
    from koopman_lm.run.artifacts import atomic_torch_save

    target = tmp_path / "resume.pt"
    atomic_torch_save(target, {"step": 7, "tensor": torch.arange(4)})
    loaded = torch.load(target, map_location="cpu", weights_only=False)
    assert loaded["step"] == 7
    assert torch.equal(loaded["tensor"], torch.arange(4))
    leftovers = list(tmp_path.rglob("*.tmp*"))
    assert leftovers == []


def test_atomic_torch_save_failure_does_not_corrupt_existing_file(tmp_path, monkeypatch):
    import torch
    from koopman_lm.run.artifacts import atomic_torch_save

    target = tmp_path / "resume.pt"
    atomic_torch_save(target, {"step": 1})

    def _boom(*a, **k):
        raise RuntimeError("simulated kill mid-write")

    monkeypatch.setattr("torch.save", _boom)
    with pytest.raises(RuntimeError):
        atomic_torch_save(target, {"step": 2})
    # old contents survive -- a kill mid-write must not corrupt the rolling file
    loaded = torch.load(target, map_location="cpu", weights_only=False)
    assert loaded["step"] == 1
    assert list(tmp_path.rglob("*.tmp*")) == []
```

Run: `PYTHONPATH=. $V/bin/pytest code-tests/test_run_artifacts.py -q` — both new
tests fail with `ImportError`/`AttributeError` (`atomic_torch_save` doesn't
exist yet).

- [ ] **Step 2: Implement**

```python
# koopman_lm/run/artifacts.py, after atomic_write_json

def atomic_torch_save(path, obj: Any) -> None:
    """Binary counterpart of atomic_write_text: temp file + os.replace, so a
    kill mid-write (a preemption, e.g.) cannot leave a truncated resume.pt."""
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / (path.name + f".tmp{os.getpid()}")
    torch.save(obj, tmp)
    os.replace(tmp, path)
```

The monkeypatched-failure test passes because `torch.save` raises before
`os.replace` runs, so `tmp` (if partially written) is simply never promoted —
but note `tmp` itself could be left on disk in that failure path, which the
first test's "no leftovers" assertion does NOT cover for the *failure* case
(only the success case). Add a `finally`-free but explicit cleanup: catch, remove
`tmp` if it exists, re-raise. Update the implementation:

```python
def atomic_torch_save(path, obj: Any) -> None:
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / (path.name + f".tmp{os.getpid()}")
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)
    except BaseException:
        if tmp.exists():
            tmp.unlink()
        raise
```

Run: `PYTHONPATH=. $V/bin/pytest code-tests/test_run_artifacts.py -q` — all
pass (7 total in that file now).

- [ ] **Step 3: Full suite + commit**

Run: `PYTHONPATH=. $V/bin/pytest code-tests -q -p no:cacheprovider` — expect
4 failed (the parked 4), 298 passed, 28 skipped.

Commit: `feat(training): add atomic_torch_save for binary checkpoint state`

---

### Task 2: RNG capture/restore

**Files:**
- Create: `koopman_lm/training/resume.py`
- Create: `code-tests/test_resume.py`

**Interfaces:**
- Consumes: `random`, `numpy`, `torch` (stdlib/already-vendored).
- Produces: `capture_rng_state() -> dict` with keys `python`, `numpy`, `torch`,
  `torch_cuda` (the last `None` on a CPU box); `restore_rng_state(state: dict) -> None`.

- [ ] **Step 1: Write the failing test**

```python
# code-tests/test_resume.py
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
```

Run: `PYTHONPATH=. $V/bin/pytest code-tests/test_resume.py -q` — fails,
`koopman_lm.training.resume` doesn't exist.

- [ ] **Step 2: Implement**

```python
# koopman_lm/training/resume.py
"""Exact resume (§5 of the run-system design): model-agnostic primitives for
optimizer/scheduler/RNG/dataloader-position persistence. Deliberately
independent of KoopmanLM -- these are the pieces testable on a CPU box
without mamba_ssm, and are exactly what train.py's SIGUSR1 handler and
--resume flag call into.
"""
from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

import numpy as np
import torch


def capture_rng_state() -> Dict[str, Any]:
    """Snapshot python/numpy/torch (CPU+CUDA) RNG state (§5.2)."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: Dict[str, Any]) -> None:
    """Inverse of capture_rng_state. Restores whichever streams were captured;
    torch_cuda is a no-op if the snapshot has none or no CUDA device is present."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch"])
    if state.get("torch_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])
```

Run: `PYTHONPATH=. $V/bin/pytest code-tests/test_resume.py -q` — both pass.

- [ ] **Step 3: Full suite + commit**

Run full suite (same command as Task 1 Step 3) — expect 4 failed, 300 passed,
28 skipped.

Commit: `feat(training): add RNG capture/restore for exact resume`

---

### Task 3: Deterministic epoch-permutation replay (dataloader position)

**Files:**
- Modify: `koopman_lm/training/resume.py`
- Modify: `code-tests/test_resume.py`

**Interfaces:**
- Consumes: `koopman_lm.training.data.dataset.MemmapPackedDataset` (existing,
  unmodified), `koopman_lm.training.data.pretokenize.write_synthetic_corpus`
  (existing, used by `test_smoke_e2e.py` already) as the real fixture dataset.
- Produces: `epoch_permutation(dataset_len: int, seed: int, epoch: int) -> List[int]`
  — replays `torch.randperm` `epoch + 1` times on a freshly-seeded
  `torch.Generator`, returning only the last permutation (this is exactly what
  `RandomSampler(dataset, generator=g)` does across `epoch + 1` successive
  `iter()` calls on the *same* generator object — see docstring for the
  equivalence argument). `resume_indices(dataset_len: int, seed: int, epoch: int, samples_consumed: int) -> List[int]` — `epoch_permutation(...)[samples_consumed:]`.

- [ ] **Step 1: Write the failing test**

```python
# appended to code-tests/test_resume.py

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
```

Run: fails, `epoch_permutation`/`resume_indices` don't exist.

- [ ] **Step 2: Implement**

```python
# appended to koopman_lm/training/resume.py

def epoch_permutation(dataset_len: int, seed: int, epoch: int) -> List[int]:
    """The sample order train.py's DataLoader(shuffle=True, generator=<seed>)
    draws for `epoch`, reconstructed from (seed, epoch) alone -- index
    arithmetic, no data read (§5.2).

    torch.utils.data.RandomSampler(dataset, generator=g), when iterated once
    per epoch on the SAME generator object, calls
    `torch.randperm(len(dataset), generator=g)` exactly once per iter() call;
    the generator's internal state (not the epoch number) is what determines
    the next permutation. So replaying `torch.randperm` `epoch + 1` times on a
    generator freshly seeded with `seed` reproduces the exact sequence of
    permutations a live RandomSampler would have produced, and the last call
    is epoch `epoch`'s draw.
    """
    g = torch.Generator()
    g.manual_seed(seed)
    perm: Optional[torch.Tensor] = None
    for _ in range(epoch + 1):
        perm = torch.randperm(dataset_len, generator=g)
    return perm.tolist()


def resume_indices(dataset_len: int, seed: int, epoch: int, samples_consumed: int) -> List[int]:
    """The indices still owed for `epoch`, after `samples_consumed` have
    already been drawn -- the forward skip of §5.2, pure index arithmetic."""
    return epoch_permutation(dataset_len, seed, epoch)[samples_consumed:]
```

Run: `PYTHONPATH=. $V/bin/pytest code-tests/test_resume.py -q` — 4 pass.

- [ ] **Step 3: Full suite + commit**

Expect 4 failed, 302 passed, 28 skipped.

Commit: `feat(training): add deterministic epoch-permutation replay for dataloader resume`

---

### Task 4: Resume-state save/load/apply

**Files:**
- Modify: `koopman_lm/training/resume.py`
- Modify: `code-tests/test_resume.py`

**Interfaces:**
- Consumes: `koopman_lm.run.artifacts.atomic_torch_save` (Task 1); any
  `torch.optim.Optimizer` and any LR-scheduler object exposing
  `state_dict()`/`load_state_dict()` (duck-typed, matches
  `torch.optim.lr_scheduler._LRScheduler` and `LambdaLR`/`get_cosine_schedule_with_warmup`'s
  return type alike).
- Produces:
  - `save_resume_state(path, *, step: int, epoch: int, samples_consumed: int, optimizer, scheduler, extra: Optional[dict] = None) -> None`
  - `load_resume_state(path) -> dict`
  - `apply_resume_state(state: dict, *, optimizer, scheduler, restore_rng: bool = True) -> tuple[int, int, int]` (returns `(step, epoch, samples_consumed)`)

- [ ] **Step 1: Write the failing test**

```python
# appended to code-tests/test_resume.py

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
    saved_opt_sd = opt.state_dict()
    saved_sched_sd = sched.state_dict()

    # advance further -- this must NOT be what gets restored
    for _ in range(3):
        loss = model(x).sum()
        loss.backward()
        opt.step(); sched.step(); opt.zero_grad()
    assert opt.state_dict() != saved_opt_sd or sched.state_dict()["_step_count"] != saved_sched_sd["_step_count"]

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
```

Run: fails (`save_resume_state` etc. don't exist).

- [ ] **Step 2: Implement**

```python
# appended to koopman_lm/training/resume.py

from koopman_lm.run.artifacts import atomic_torch_save


def save_resume_state(path, *, step: int, epoch: int, samples_consumed: int,
                       optimizer, scheduler, extra: Optional[Dict[str, Any]] = None) -> None:
    """Write the rolling resume.pt (§5.3): optimizer + scheduler + RNG + epoch
    + samples-consumed, atomically (temp file + os.replace) so a kill
    mid-write cannot corrupt it. Deliberately excludes model weights -- those
    stay in the periodic, weights-only step_<N>/ archival checkpoints;
    train.py's caller is responsible for writing a step_<N>/ at the same
    `step` whenever it calls this."""
    state = {
        "step": step,
        "epoch": epoch,
        "samples_consumed": samples_consumed,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "rng": capture_rng_state(),
        "torch_version": str(torch.__version__),
    }
    if extra:
        state.update(extra)
    atomic_torch_save(path, state)


def load_resume_state(path) -> Dict[str, Any]:
    """resume.pt holds RNG state (numpy/python tuples, not just tensors), so
    weights_only=False -- same reasoning as mqar_finetune.py's meta.pt load."""
    return torch.load(path, map_location="cpu", weights_only=False)


def apply_resume_state(state: Dict[str, Any], *, optimizer, scheduler,
                        restore_rng: bool = True):
    """Restore optimizer/scheduler (and, by default, RNG) from a loaded
    resume-state dict. Returns (step, epoch, samples_consumed)."""
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    if restore_rng:
        restore_rng_state(state["rng"])
    return state["step"], state["epoch"], state["samples_consumed"]
```

Add `import random` / `import numpy as np` are already present from Task 2.

Run: `PYTHONPATH=. $V/bin/pytest code-tests/test_resume.py -q` — 6 pass.

- [ ] **Step 3: Full suite + commit**

Expect 4 failed, 304 passed, 28 skipped.

Commit: `feat(training): add resume-state save/load/apply for optimizer+scheduler+RNG+position`

---

### Task 5: `SlurmLauncher` emits `--signal=B:USR1@300` and `--requeue`

**Files:**
- Modify: `koopman_lm/run/slurm.py`
- Modify: `code-tests/test_run_slurm.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: no signature change — `_SBATCH_TEMPLATE`'s rendered text gains two
  directives; `render_sbatch()`'s return type is unchanged (`str`).

- [ ] **Step 1: Write the failing test**

```python
# appended to code-tests/test_run_slurm.py

def test_render_sbatch_requests_preemption_signal_and_requeue(tmp_path):
    """§5.4: Slurm must warn the job 300s before killing it (SIGUSR1) so
    train.py's handler can write resume.pt and exit cleanly, and --requeue so
    a preempted/timed-out job is resubmitted rather than lost."""
    from koopman_lm.run.slurm import SlurmLauncher

    spec = _shard_spec()
    text = SlurmLauncher().render_sbatch(spec, tmp_path)
    assert "#SBATCH --signal=B:USR1@300" in text
    assert "#SBATCH --requeue" in text
```

Run: fails, directives absent.

- [ ] **Step 2: Implement**

```python
# koopman_lm/run/slurm.py -- edit _SBATCH_TEMPLATE
_SBATCH_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --account={account}
#SBATCH --partition={partition}
#SBATCH --qos={qos}
#SBATCH --nodes={nodes}
#SBATCH --gpus-per-node={gpus}
#SBATCH --time={time_limit}
#SBATCH --signal=B:USR1@300
#SBATCH --requeue
#SBATCH --output={run_dir}/slurm-%j.out
...
"""
```

(`B:` sends the signal to the batch shell rather than only step 0's rank-0
task, matching how `torchrun`/single-process launches under this template are
actually invoked — the batch script itself is what needs to forward or react
to it.)

Run: `PYTHONPATH=. $V/bin/pytest code-tests/test_run_slurm.py -q` — all pass
(the pre-existing exact-string sbatch assertions don't pin the full template,
only substrings, so this is additive).

- [ ] **Step 3: Full suite + commit**

Expect 4 failed, 305 passed, 28 skipped.

Commit: `feat(run): request SIGUSR1 preemption warning + --requeue in generated sbatch`

---

### Task 6: SIGUSR1 handler in `train.py` (isolated, testable)

**Files:**
- Modify: `koopman_lm/training/train.py`
- Create: `code-tests/test_train_preemption_handler.py`

**Interfaces:**
- Consumes: `signal` (stdlib).
- Produces: `class PreemptionFlag` with `.set()`/`.is_set()` (a plain
  bool-holder, not `threading.Event`, since nothing here is multi-threaded —
  it just needs to survive being flipped inside a signal handler and read from
  the main loop); `install_sigusr1_handler(flag: PreemptionFlag) -> None`.

- [ ] **Step 1: Write the failing test**

```python
# code-tests/test_train_preemption_handler.py
"""§5.4: the trainer installs a SIGUSR1 handler that flags for a clean exit
(train.py's loop then writes resume.pt and returns) rather than dying wherever
Slurm's warning signal happens to land mid-step."""
import os
import signal
import time

import pytest

pytestmark = pytest.mark.correctness


def test_sigusr1_sets_the_preemption_flag():
    from koopman_lm.training.train import PreemptionFlag, install_sigusr1_handler

    flag = PreemptionFlag()
    assert not flag.is_set()
    install_sigusr1_handler(flag)
    try:
        os.kill(os.getpid(), signal.SIGUSR1)
        # signal delivery to the main thread happens between bytecode
        # instructions; give it a moment
        for _ in range(100):
            if flag.is_set():
                break
            time.sleep(0.01)
        assert flag.is_set()
    finally:
        signal.signal(signal.SIGUSR1, signal.SIG_DFL)
```

Run: fails, `PreemptionFlag`/`install_sigusr1_handler` don't exist.

- [ ] **Step 2: Implement**

Add near the top of `koopman_lm/training/train.py` (after the existing
imports, before `enable_gradient_checkpointing`):

```python
import signal


class PreemptionFlag:
    """Set from inside a signal handler, read from the training loop.
    Plain attribute (not threading.Event) -- train.py's loop is
    single-threaded; this only needs to survive a signal handler write."""

    def __init__(self):
        self._flag = False

    def set(self):
        self._flag = True

    def is_set(self) -> bool:
        return self._flag


def install_sigusr1_handler(flag: PreemptionFlag) -> None:
    """§5.4: SlurmLauncher's --signal=B:USR1@300 fires 300s before a
    preemption/timeout kill. The handler only flips a flag -- it does no I/O
    itself -- so the actual resume.pt write happens on the main thread at the
    next safe point in the training loop, never inside signal-handler
    context."""
    def _handler(signum, frame):
        flag.set()
    signal.signal(signal.SIGUSR1, _handler)
```

Run: `PYTHONPATH=. $V/bin/pytest code-tests/test_train_preemption_handler.py -q` — passes.

- [ ] **Step 3: Full suite + commit**

Expect 4 failed, 306 passed, 28 skipped.

Commit: `feat(training): add a SIGUSR1 preemption flag/handler to train.py`

---

### Task 7: Wire `--resume`, paired archival+resume writes, and the index-based epoch loop into `train.py`

**Files:**
- Modify: `koopman_lm/training/train.py`

**Interfaces:**
- Consumes: `koopman_lm.training.resume.{save_resume_state, load_resume_state, apply_resume_state, epoch_permutation, resume_indices}` (Tasks 2–4); `PreemptionFlag`/`install_sigusr1_handler` (Task 6); `koopman_lm.run.artifacts.atomic_torch_save` (indirectly, via `resume.py`).
- Produces (all inside `train.py`, no new public module):
  - `parse_args()` gains `--resume` (`action="store_true", default=False`).
  - `train(args)`'s loop is restructured: each epoch's sample order is built
    explicitly via `epoch_permutation`/`resume_indices` instead of relying on
    `DataLoader(shuffle=True)`'s implicit `RandomSampler`; `samples_consumed`
    is tracked and reset each epoch; on `--resume`, `resume.pt` + the
    matching `step_<N>/model.pt` are loaded before the loop starts; on
    `step % save_steps == 0` and on `PreemptionFlag.is_set()`, `_save_checkpoint`
    (weights) and `resume.py`'s `save_resume_state` (optimizer/scheduler/RNG/
    position) are called together, tied to the same `step`, so `resume.pt`
    always names a `step_<N>/` that actually exists.

This task is not itself independently unit-testable end to end (it needs a
real `KoopmanLM`/GPU to run `train()`), but every primitive it calls already
has a passing unit test from Tasks 1–6, and Task 8's equivalence test exercises
the same call sequence against a stand-in model. Verification here is: (a) the
module still imports cleanly (the existing `test_init_from.py`,
`test_run_slurm.py`, `test_run_launch.py`, `test_run_main.py`,
`test_smoke_e2e.py` all import from or shell out to `koopman_lm.training.train`
and must keep passing), and (b) a light CLI-parsing test.

- [ ] **Step 1: Write the failing test for `--resume` parsing**

```python
# appended to code-tests/test_train_preemption_handler.py

def test_parse_args_accepts_resume_flag(monkeypatch):
    from koopman_lm.training.train import parse_args

    monkeypatch.setattr("sys.argv", ["train.py", "--data_dir", "/tmp/x"])
    args = parse_args()
    assert args.resume is False

    monkeypatch.setattr("sys.argv", ["train.py", "--data_dir", "/tmp/x", "--resume"])
    args = parse_args()
    assert args.resume is True
```

Run: fails (`--resume` not a recognized flag -> `SystemExit`/argparse error is
what "fails" looks like here, since `--resume` isn't defined yet, argparse
will raise `error: unrecognized arguments` — actually the *first* call with no
`--resume` already succeeds trivially with today's parser since it just won't
have a `.resume` attribute at all, so this assertion fails with `AttributeError`).

- [ ] **Step 2: Implement**

Add to `parse_args()` (near `--deterministic`):

```python
    p.add_argument("--resume", action="store_true", default=False,
                   help="resume from <output_dir>/resume.pt + its matching "
                        "step_<N>/model.pt (optimizer, scheduler, RNG, and "
                        "dataloader position restored exactly; §5)")
```

Run: `PYTHONPATH=. $V/bin/pytest code-tests/test_train_preemption_handler.py -q` — passes.

- [ ] **Step 3: Restructure `train()`'s loop and checkpoint writes**

Replace the body of `train(args)` from the `model.train()` line to the end of
the function. Full new body:

```python
    from koopman_lm.training.resume import (
        apply_resume_state, epoch_permutation, load_resume_state,
        resume_indices, save_resume_state,
    )

    preempt_flag = PreemptionFlag()
    install_sigusr1_handler(preempt_flag)

    start_step, start_epoch, start_samples_consumed = 0, 0, 0
    if args.resume:
        resume_path = os.path.join(args.output_dir, "resume.pt")
        if not os.path.exists(resume_path):
            raise SystemExit(f"--resume given but no resume.pt at {resume_path}")
        state = load_resume_state(resume_path)
        ckpt_dir = os.path.join(args.output_dir, f"step_{state['step']}")
        model_path = os.path.join(ckpt_dir, "model.pt")
        if not os.path.exists(model_path):
            raise SystemExit(
                f"resume.pt names step {state['step']} but {model_path} "
                f"is missing -- archival checkpoint and resume.pt must be "
                f"written together")
        raw_model.load_state_dict(
            torch.load(model_path, map_location="cpu", weights_only=True))
        start_step, start_epoch, start_samples_consumed = apply_resume_state(
            state, optimizer=optimizer, scheduler=scheduler)
        if is_main:
            print(f"  Resumed from {ckpt_dir} at step {start_step}, "
                  f"epoch {start_epoch}, samples_consumed {start_samples_consumed}")

    def _save_all(step, epoch, samples_consumed, dirname=None):
        _save_checkpoint(raw_model, cfg, tokenizer, step, args, dirname=dirname)
        resume_path = os.path.join(args.output_dir, "resume.pt")
        save_resume_state(resume_path, step=step, epoch=epoch,
                           samples_consumed=samples_consumed,
                           optimizer=optimizer, scheduler=scheduler)

    model.train()
    step = start_step; micro_step = 0
    running_loss = torch.tensor(0.0, device=device); loss_count = 0
    t_start = time.time(); tokens_seen = 0
    if is_main:
        eff = args.per_device_train_batch_size * args.gradient_accumulation_steps * world_size
        print(f"\nTraining: {args.max_steps} steps, eff_batch={eff}, "
              f"tok/step={eff*args.max_seq_len:,}")
    optimizer.zero_grad(set_to_none=True)

    epoch = start_epoch
    while step < args.max_steps:
        if hasattr(train_ds, 'set_epoch'): train_ds.set_epoch(epoch)
        if is_ddp:
            sampler.set_epoch(epoch)
            indices = list(sampler)
        else:
            indices = epoch_permutation(len(train_ds), args.seed + local_rank, epoch)
        samples_consumed = 0
        if epoch == start_epoch and start_samples_consumed > 0:
            indices = indices[start_samples_consumed:]
            samples_consumed = start_samples_consumed
        epoch_loader = DataLoader(
            train_ds, batch_size=args.per_device_train_batch_size,
            sampler=indices, num_workers=args.num_workers,
            pin_memory=True, drop_last=True, worker_init_fn=seed_worker)
        for batch in epoch_loader:
            if step >= args.max_steps: break
            ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            lw = batch.get("loss_weights")
            if lw is not None: lw = lw.to(device, non_blocking=True)
            with autocast_ctx:
                outputs = model(input_ids=ids, labels=labels, loss_weights=lw)
                raw_loss = outputs["loss"]
                scaled_loss = raw_loss / args.gradient_accumulation_steps
            scaled_loss.backward()
            running_loss += raw_loss.detach(); loss_count += 1
            tokens_seen += ids.numel(); micro_step += 1
            samples_consumed += ids.size(0)
            if micro_step % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step(); scheduler.step()
                optimizer.zero_grad(set_to_none=True); step += 1
                if is_main and step % args.logging_steps == 0:
                    avg = running_loss.item() / max(loss_count, 1)
                    lr = optimizer.param_groups[0]["lr"]; el = time.time() - t_start
                    tps = tokens_seen / el; ppl = math.exp(min(avg, 20))
                    print(f"step {step:>6d}/{args.max_steps} | loss {avg:.4f} | "
                          f"ppl {ppl:.1f} | lr {lr:.2e} | {tps/1e3:.1f}K tok/s")
                    if args.wandb_project:
                        import wandb
                        wandb.log({"loss": avg, "ppl": ppl, "lr": lr,
                                   "tokens_per_sec": tps}, step=step)
                    running_loss = torch.tensor(0.0, device=device); loss_count = 0
                if is_main and step > 0 and step % args.save_steps == 0:
                    _save_all(step, epoch, samples_consumed)
                if is_main and preempt_flag.is_set():
                    _save_all(step, epoch, samples_consumed)
                    print(f"  SIGUSR1 received -- wrote resume.pt at step {step}, exiting cleanly")
                    if is_ddp:
                        torch.distributed.destroy_process_group()
                    return
        epoch += 1

    if is_main:
        _save_checkpoint(raw_model, cfg, tokenizer, step, args, dirname="final")
        print(f"\nDone in {(time.time()-t_start)/3600:.1f}h, {tokens_seen/1e9:.2f}B tokens")
    if is_ddp:
        torch.distributed.destroy_process_group()
```

Notes on this rewrite relative to the original:
- `sampler` (the `DistributedSampler`, DDP-only) and `train_loader` (the
  single persistent `DataLoader`) are no longer built before the loop in the
  same way `train_loader` was — `sampler` (DDP) is still built once, same as
  today; the per-epoch `DataLoader` is now built fresh each epoch since its
  `sampler=` argument (a plain index list) changes every epoch. This trades
  persistent-worker reuse for a testable, skippable index order, exactly per
  §5.2's requirement.
- `_save_checkpoint`'s existing weights-only write is unchanged and still
  reused (not duplicated) — `_save_all` just also calls
  `resume.save_resume_state` right after it, so `step_<N>/model.pt` and
  `resume.pt`'s named step always agree.

Also remove the now-dead pre-loop construction of `train_loader` (it's
replaced by the per-epoch `epoch_loader` inside the while-loop) and keep the
`sampler = DistributedSampler(...)` construction (used by the DDP branch
above) but drop `shuffle=(sampler is None)` / `generator=data_gen` from the
deleted `DataLoader(...)` call since that whole call is gone.

- [ ] **Step 4: Run the full suite**

`PYTHONPATH=. $V/bin/pytest code-tests -q -p no:cacheprovider` — must still
collect cleanly (this is the load-bearing check for this task: a syntax error
or bad import in `train.py` breaks collection for every test that imports it —
`test_init_from.py`, `test_run_slurm.py`, `test_run_launch.py`,
`test_run_main.py`, `test_smoke_e2e.py`). Expect 4 failed, 308 passed, 28
skipped (the two new parse_args assertions from Step 1 plus the Task 6 test
already counted).

- [ ] **Step 5: Commit**

Commit: `feat(training): wire --resume, paired archival+resume writes, and index-based epoch loop into train.py`

---

### Task 8: The §5.5 acceptance test — interrupted-and-resumed equals uninterrupted

**Files:**
- Create: `code-tests/test_resume_equivalence.py`

**Interfaces:**
- Consumes: `koopman_lm.training.resume.{epoch_permutation, resume_indices, save_resume_state, load_resume_state, apply_resume_state, capture_rng_state, restore_rng_state}` (Tasks 2–4, the real functions `train.py` calls — not reimplemented); `koopman_lm.training.data.pretokenize.write_synthetic_corpus` and `koopman_lm.training.data.dataset.MemmapPackedDataset` (real, unmodified, CPU-only); `koopman_lm.training.repro.seed_everything` (real, unmodified).
- Produces: no new production code — this is the test that proves Tasks 1–7
  actually satisfy §5.5. It runs its own small loop (a stand-in `nn.Module`
  in place of `KoopmanLM`, since `mamba_ssm` isn't installed) that mirrors
  `train.py`'s Task-7 loop structure exactly: per-epoch `epoch_permutation`/
  `resume_indices`, AdamW + `LambdaLR` cosine, dropout (to make RNG restore
  load-bearing, not just optimizer-state restore), and paired
  weights+resume-state writes on "kill".

- [ ] **Step 1: Write the test (it must fail first if any Task 1–7 piece is
  wrong; write it now to lock in the acceptance criterion, then run it)**

```python
# code-tests/test_resume_equivalence.py
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
import random

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from koopman_lm.training.data.dataset import MemmapPackedDataset
from koopman_lm.training.data.pretokenize import write_synthetic_corpus
from koopman_lm.training.repro import seed_everything
from koopman_lm.training.resume import (
    apply_resume_state, epoch_permutation, load_resume_state,
    resume_indices, save_resume_state,
)

pytestmark = pytest.mark.correctness

VOCAB = 64
SEQ_LEN = 16
HIDDEN = 32


class _StandInLM(nn.Module):
    """Not KoopmanLM (needs mamba_ssm/GPU) -- a couple of Linear layers plus
    dropout, exactly the stand-in the task calls for. Dropout makes RNG
    restoration load-bearing: without it, resumed weights would match using
    only optimizer-state restore, which would not actually test §5.2's RNG
    requirement."""

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
    import math
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
        indices = epoch_permutation(len(ds), seed, epoch)
        samples_consumed = 0
        if epoch == start_epoch and start_samples_consumed > 0:
            indices = resume_indices(len(ds), seed, epoch, start_samples_consumed)
            samples_consumed = start_samples_consumed
        loader = DataLoader(ds, batch_size=batch_size, sampler=indices, drop_last=True)
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
    # exactly as a new `python -m koopman_lm.training.train --resume` process would
    del model_b, opt_b, sched_b

    model_c = _StandInLM()  # fresh init -- immediately overwritten by the checkpoint
    model_c.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
    optimizer_c = torch.optim.AdamW(model_c.parameters(), lr=1e-2, betas=(0.9, 0.95))
    scheduler_c = torch.optim.lr_scheduler.LambdaLR(
        optimizer_c, lambda step: _lr_lambda(step, warmup, max_steps))
    state = load_resume_state(resume_path)
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
```

Run: `PYTHONPATH=. $V/bin/pytest code-tests/test_resume_equivalence.py -q -v`.

- [ ] **Step 2: If it fails, diagnose before touching Tasks 1–7**

Two likely failure modes, and what each implies:
1. **Dropout masks diverge post-resume** → RNG wasn't actually restored before
   the dropout call in the resumed loop's first forward pass. Check that
   `apply_resume_state` is called (and `restore_rng_state` inside it) *before*
   `_run_loop` resumes, not after any stray `torch.randn`/`model(...)` call
   consumes the torch RNG stream first (e.g. building `model_c` fresh with
   `_StandInLM()` consumes RNG for random init *before* `load_state_dict`
   overwrites it with the checkpoint — that consumption happens on the torch
   global RNG stream and must occur before `apply_resume_state`'s restore, not
   after, or it'll perturb the very state being restored). Reorder so
   `model_c = _StandInLM()` + `load_state_dict` complete, then
   `apply_resume_state` (which restores RNG) runs last, immediately before
   `_run_loop` resumes.
2. **AdamW moment tensors mismatch** → `save_resume_state`/`apply_resume_state`
   round-trip is fine per Task 4's test, so this would point at `_run_loop`
   itself computing a different sequence of steps (off-by-one in
   `start_step`/`end_step`, or `samples_consumed` accounting drifting from
   what `resume_indices` expects). Print `(step, epoch, samples_consumed)` at
   the kill point and after resume to isolate it.

Iterate on the test/harness (not on Tasks 1–7's already-tested primitives)
until `torch.equal` holds for every key.

- [ ] **Step 3: Full suite + commit**

Run: `PYTHONPATH=. $V/bin/pytest code-tests -q -p no:cacheprovider` — expect 4
failed (the parked 4, unchanged), 309 passed, 28 skipped, 0 errors.

Commit: `test(training): add the §5.5 exact-resume equivalence test`

---

## Final verification

- [ ] Run `PYTHONPATH=. $V/bin/pytest code-tests -q -p no:cacheprovider` one
      more time and confirm the failing set is exactly the 4 parked tests
      named in Global Constraints — a fifth failure means stop and diagnose
      before declaring done.
- [ ] Update `.superpowers/sdd/2026-08-07-exact-resume-report.md` with: plan
      path, per-task commit SHAs, the final exact suite line, confirmation the
      failing set is unchanged, and what the §5.5 test does plus the tolerance
      it achieved (exact equality, justified above).
