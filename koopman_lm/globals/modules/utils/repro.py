"""Reproducibility helpers: deterministic seeding + a determinism toggle.

The scaling plan (Phase 0) requires that "two runs with the same config hash
produce identical loss curves for at least the first 1K steps". These helpers
provide the seeding and determinism flags to make that achievable.

NOTE on bitwise determinism: ``enable_determinism()`` makes torch ops
deterministic where possible, but the ``mamba_ssm`` CUDA kernels are not
guaranteed bit-reproducible across runs. The practical target is *matching loss
curves over the first ~1K steps*, not bitwise-identical tensors. Determinism
mode also costs throughput (cudnn.benchmark off), so it is opt-in via
``--deterministic`` for repro runs and off for production throughput.
"""
import os
import random

import numpy as np
import torch

_UINT32 = 2 ** 32


def seed_everything(seed: int) -> torch.Generator:
    """Seed python ``random``, numpy, and torch (CPU+CUDA); set PYTHONHASHSEED.

    Returns a CPU ``torch.Generator`` seeded from ``seed`` for use as a
    DataLoader ``generator=`` (so shuffling is reproducible).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed % _UINT32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def enable_determinism(warn_only: bool = True) -> None:
    """Flip torch into deterministic-algorithm mode.

    Sets CUBLAS_WORKSPACE_CONFIG (required for deterministic cuBLAS GEMMs),
    disables cudnn.benchmark, enables cudnn.deterministic, and calls
    ``torch.use_deterministic_algorithms``. ``warn_only=True`` keeps ops that
    have no deterministic implementation from hard-erroring (they warn instead).
    """
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    try:
        torch.use_deterministic_algorithms(True, warn_only=warn_only)
    except TypeError:  # older torch without warn_only kwarg
        torch.use_deterministic_algorithms(True)


def seed_worker(worker_id: int) -> None:
    """DataLoader ``worker_init_fn``: seed each worker's RNGs deterministically.

    torch sets a distinct base seed per worker; derive numpy/random seeds from it
    so augmentation/shuffle inside workers is reproducible.
    """
    worker_seed = torch.initial_seed() % _UINT32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
