"""Threading ska_precision into the SKA core (precision design step 3).

Step 3 is a no-op on behaviour: the core already cast to fp32 with a hardcoded
`.float()`, and `ska_precision` defaults to "fp32", so the default path must come
out bit-identical. "Must" is doing real work there -- this is the live production
forward path, both shipped configs run it, and a CPU test suite cannot instantiate
a full KoopmanLM to catch a regression downstream.

So the gate is a byte hash of a real forward pass, captured from the code BEFORE
the change and committed as `golden_ska_forward.json`. Ordering is the point: a
golden recorded after a refactor records whatever the refactor produced, which
proves nothing. Both live backends are covered -- the exact prefix scan (what
both production configs use) and the chunked path.

What the change buys: the cast is now stated by config instead of implied by a
`.float()` call, and fp64 becomes reachable on the PyTorch path, where the exact
prefix scan already accepts float64.
"""
import hashlib
import json
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.correctness

_GOLDEN = json.loads(
    (Path(__file__).resolve().parent / "golden_ska_forward.json").read_text())

_CASES = {
    "prefix_scan": dict(d_model=32, n_heads=2, rank=16, chunk_size=8,
                        backend="pytorch", prefix_scan=True,
                        prefix_scan_block_size=8),
    "chunked": dict(d_model=32, n_heads=2, rank=16, chunk_size=8,
                    backend="pytorch", prefix_scan=False),
}


def _forward(label, **extra):
    """The same construction and input the golden was captured with."""
    from koopman_lm.modules.seq.ska import SKAModule

    kwargs = dict(_CASES[label])
    kwargs.update(extra)
    torch.manual_seed(1234)
    module = SKAModule(**kwargs).eval()
    torch.manual_seed(99)
    x = torch.randn(2, 16, 32)
    with torch.no_grad():
        y = module(x)
    return y[0] if isinstance(y, tuple) else y


def _hash(tensor):
    return hashlib.sha256(
        tensor.detach().contiguous().numpy().tobytes()).hexdigest()


@pytest.mark.parametrize("label", sorted(_CASES))
def test_the_default_precision_is_bit_identical_to_the_hardcoded_float_cast(label):
    """The step-3 gate. If this fails, the wiring changed the production path."""
    out = _forward(label)
    assert list(out.shape) == _GOLDEN[label]["shape"]
    assert str(out.dtype) == _GOLDEN[label]["dtype"]
    assert _hash(out) == _GOLDEN[label]["sha256"], (
        f"{label}: the default SKA path is no longer bit-identical to the "
        f"hardcoded .float() it replaced")


@pytest.mark.parametrize("label", sorted(_CASES))
def test_naming_fp32_explicitly_matches_the_default(label):
    """ska_precision='fp32' and the default must be the same thing, or the
    default is not what the config says it is."""
    assert _hash(_forward(label, precision="fp32")) == _GOLDEN[label]["sha256"]


def test_the_module_records_its_core_precision():
    from koopman_lm.modules.seq.ska import SKAModule

    assert SKAModule(d_model=32, n_heads=2, rank=16).precision == "fp32"
    assert SKAModule(d_model=32, n_heads=2, rank=16,
                     precision="fp64").precision == "fp64"


@pytest.mark.parametrize("label", sorted(_CASES))
def test_an_fp64_core_changes_the_result(label):
    """Otherwise the field is decoration. fp64 must actually raise the precision
    of the whitened core, not merely be accepted."""
    assert _hash(_forward(label, precision="fp64")) != _GOLDEN[label]["sha256"]


@pytest.mark.parametrize("label", sorted(_CASES))
def test_an_fp64_core_agrees_with_fp32_to_single_precision(label):
    """The two must be the same computation at different widths. Disagreement
    beyond float32's resolution would mean the fp64 path is not the same
    algorithm."""
    fp32 = _forward(label, precision="fp32")
    fp64 = _forward(label, precision="fp64")
    assert torch.allclose(fp32, fp64.float(), atol=1e-5, rtol=1e-4)


def test_an_invalid_precision_is_rejected_at_construction():
    """Failing in the constructor beats failing mid-forward on a GPU node."""
    from koopman_lm.modules.seq.ska import SKAModule

    with pytest.raises(ValueError):
        SKAModule(d_model=32, n_heads=2, rank=16, precision="bf16")


def test_the_block_threads_the_config_field_into_the_module():
    """SKAModule takes scalars and never imports KoopmanLMConfig -- deliberately,
    it is config-free numerics. ska_block.py is what bridges them, and this
    asserts the new field crossed that bridge like the other 28."""
    import inspect

    from koopman_lm.modules.seq import ska_block

    source = inspect.getsource(ska_block)
    assert "ska_precision" in source, (
        "ska_block.py must pass cfg.ska_precision into SKAModule; otherwise the "
        "config field exists and controls nothing")
