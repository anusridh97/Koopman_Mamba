"""Shared pytest fixtures + marker-based auto-skip.

Markers are declared in pyproject.toml ([tool.pytest.ini_options].markers):
  correctness  numerical gate (runs in CI before any training commit)
  gpu          requires CUDA (mamba_ssm backbone / bf16 kernels)
  slow         long-running (e.g. end-to-end smoke test)
  jax          requires JAX (reference-oracle parity)

`gpu` tests auto-skip when no CUDA device is present, so the same suite runs on
a CPU dev box (collecting the pure-torch SKA/Koopman math) and on a GPU box
(adding the full-model + bf16 paths). `jax` tests skip when jax isn't installed.
"""
import importlib.util

import pytest
import torch


def _has_jax():
    return importlib.util.find_spec("jax") is not None


def pytest_collection_modifyitems(config, items):
    cuda = torch.cuda.is_available()
    jax = _has_jax()
    skip_gpu = pytest.mark.skip(reason="no CUDA device available")
    skip_jax = pytest.mark.skip(reason="jax not installed")
    for item in items:
        if "gpu" in item.keywords and not cuda:
            item.add_marker(skip_gpu)
        if "jax" in item.keywords and not jax:
            item.add_marker(skip_jax)
