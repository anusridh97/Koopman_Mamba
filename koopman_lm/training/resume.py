"""Exact resume (§5 of the run-system design): model-agnostic primitives for
optimizer/scheduler/RNG/dataloader-position persistence. Deliberately
independent of KoopmanLM -- these are the pieces testable on a CPU box
without mamba_ssm, and are exactly what train.py's SIGUSR1 handler and
--resume flag call into.
"""
from __future__ import annotations

import random
from typing import Any, Dict, List, Optional


def capture_rng_state() -> Dict[str, Any]:
    """Snapshot python/numpy/torch (CPU+CUDA) RNG state (§5.2)."""
    import numpy as np
    import torch

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: Dict[str, Any]) -> None:
    """Inverse of capture_rng_state. Restores whichever streams were captured;
    torch_cuda is a no-op if the snapshot has none or no CUDA device is present."""
    import numpy as np
    import torch

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch"])
    if state.get("torch_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def epoch_permutation(dataset_len: int, seed: int, epoch: int) -> List[int]:
    """The sample order train.py's DataLoader(shuffle=True, generator=<seed>)
    draws for `epoch`, reconstructed from (seed, epoch) alone -- index
    arithmetic, no data read (§5.2).

    torch.utils.data.RandomSampler(dataset, generator=g), when iterated once
    per epoch on the SAME generator object, draws from `g` in a fixed pattern
    per __iter__() call: `torch.randperm(n, generator=g)` for the main
    permutation, THEN a second `torch.randperm(n, generator=g)` for a
    "remainder" pass that is sliced to `num_samples % n` elements -- zero
    elements (and so silently discarded) whenever num_samples == n, which is
    always true for the plain `shuffle=True` case train.py uses (no custom
    `num_samples`). That second draw still *consumes* generator state even
    though its output is thrown away, so replaying epochs must reproduce both
    calls per epoch, not just the one whose output is kept -- otherwise every
    epoch after the first desyncs from the live sampler.
    """
    import torch

    g = torch.Generator()
    g.manual_seed(seed)
    perm: Optional["torch.Tensor"] = None
    for _ in range(epoch + 1):
        perm = torch.randperm(dataset_len, generator=g)
        torch.randperm(dataset_len, generator=g)  # RandomSampler's remainder draw
    return perm.tolist()


def resume_indices(dataset_len: int, seed: int, epoch: int, samples_consumed: int) -> List[int]:
    """The indices still owed for `epoch`, after `samples_consumed` have
    already been drawn -- the forward skip of §5.2, pure index arithmetic."""
    return epoch_permutation(dataset_len, seed, epoch)[samples_consumed:]
