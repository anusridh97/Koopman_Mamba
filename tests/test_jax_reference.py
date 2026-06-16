"""JAX reference-oracle parity (scaling plan Phase 0).

Collects the original reference checks from reference/echo_jax.py
(test_backward, test_mamba_recurrence, test_ska_causal, test_ska_decode,
test_ska_newton_schulz) into the suite. These validate the JAX reference the
torch port mirrors. Marked `jax` -> auto-skipped when JAX isn't installed (the
torch-side equivalents in test_*_parity / test_newton_schulz / test_cholesky_*
cover the same properties on the actually-scaled code path).
"""
import importlib.util
import os

import pytest

pytestmark = [pytest.mark.correctness, pytest.mark.jax]

_REF = os.path.join(os.path.dirname(__file__), "..", "reference", "echo_jax.py")

_REFERENCE_TESTS = [
    "test_backward",
    "test_mamba_recurrence",
    "test_ska_causal",
    "test_ska_decode",
    "test_ska_newton_schulz",
]


@pytest.fixture(scope="module")
def echo_jax():
    pytest.importorskip("jax")
    spec = importlib.util.spec_from_file_location("echo_jax_ref", _REF)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("fn", _REFERENCE_TESTS)
def test_reference_runs(echo_jax, fn):
    getattr(echo_jax, fn)()      # reference checks print rel errors; must not raise
