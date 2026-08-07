"""Exact resume (§5 of the run-system design): model-agnostic primitives for
optimizer/scheduler/RNG/dataloader-position persistence. Deliberately
independent of KoopmanLM -- these are the pieces testable on a CPU box
without mamba_ssm, and are exactly what train.py's SIGUSR1 handler and
--resume flag call into.
"""
from __future__ import annotations

import random
from typing import Any, Dict


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
