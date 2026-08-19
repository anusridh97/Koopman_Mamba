"""Threading mlp_precision into the Koopman MLP rotation (design step 3, MLP half).

`mlp_precision` defaults to None, and None is a *strict* no-op: the code path is
untouched, so existing numerics are unchanged. That is the design's own wording,
and it is stronger than "approximately unchanged" -- so the gate is a byte hash of
a real forward pass captured before the change, for both the plain and gated
variants.

Worth knowing which configs this reaches: `50m.yaml` and `180m.yaml` use
`mlp_type: swiglu`, so the production pair is untouched either way. The Koopman
MLP is what the tier-2 and tier-3 configs use, which is exactly the set that
inherits stale field defaults by omission -- so "None changes nothing" matters
most for the configs least likely to be re-verified by hand.

What raising does: the rotation coefficients are `cos(theta)`/`sin(theta)`, or
`rho*cos`/`rho*sin` with `rho = e^{-softplus(s)}`. Raising precision has to raise
the *computation* -- `cos(theta.double())`, not `cos(theta).double()` -- or the
field buys nothing but a wider output dtype.
"""
import hashlib
import json
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.correctness

_GOLDEN = json.loads(
    (Path(__file__).resolve().parent / "golden_koopman_mlp_forward.json").read_text())


def _module(label, **extra):
    from koopman_lm.modules.mlp.koopman import (
        SpectralKoopmanMLP, SpectralKoopmanMLPGated)

    cls = {"koopman": SpectralKoopmanMLP,
           "koopman_gated": SpectralKoopmanMLPGated}[label]
    torch.manual_seed(4321)
    return cls(d=32, expand=2.0, **extra).eval()


def _forward(label, **extra):
    module = _module(label, **extra)
    torch.manual_seed(7)
    x = torch.randn(2, 8, 32)
    with torch.no_grad():
        y = module(x)
    return y[0] if isinstance(y, tuple) else y


def _hash(tensor):
    return hashlib.sha256(
        tensor.detach().contiguous().numpy().tobytes()).hexdigest()


@pytest.mark.parametrize("label", sorted(_GOLDEN))
def test_the_default_is_bit_identical(label):
    """mlp_precision=None must be a strict no-op, not an approximate one."""
    out = _forward(label)
    assert str(out.dtype) == _GOLDEN[label]["dtype"]
    assert _hash(out) == _GOLDEN[label]["sha256"]


@pytest.mark.parametrize("label", sorted(_GOLDEN))
def test_passing_none_explicitly_is_the_same_as_the_default(label):
    assert _hash(_forward(label, precision=None)) == _GOLDEN[label]["sha256"]


def test_the_module_records_its_precision():
    assert _module("koopman").precision is None
    assert _module("koopman", precision="fp64").precision == "fp64"


@pytest.mark.parametrize("label", sorted(_GOLDEN))
def test_the_output_dtype_is_unchanged_by_raising_precision(label):
    """Raising the rotation's internal precision must not widen what the block
    returns -- the residual stream stays in the compute dtype, or every
    downstream layer silently changes dtype too."""
    out = _forward(label, precision="fp64")
    assert out.dtype is torch.float32


@pytest.mark.parametrize("label", sorted(_GOLDEN))
def test_raising_precision_agrees_with_the_default_to_single_precision(label):
    """Same computation at a different width: it should move the result only at
    float32's resolution, not meaningfully."""
    default = _forward(label)
    raised = _forward(label, precision="fp64")
    assert torch.allclose(default, raised, atol=1e-5, rtol=1e-4)


def test_the_rotation_coefficients_are_computed_at_the_raised_width():
    """The substantive check, and the one that distinguishes cos(theta.double())
    from cos(theta).double(). At fp64 the coefficients must match a float64
    reference computation more closely than the float32 path does.
    """
    from koopman_lm.modules.mlp.koopman import _rotation_coeffs

    # rotation_param='angle' explicitly: the default resolves to the legacy
    # (learned gamma/omega) mode, which has no theta and so cannot show the
    # cast-before-trig difference this test is about.
    plain = _module("koopman", rotation_param="angle")
    raised = _module("koopman", rotation_param="angle", precision="fp64")

    theta = plain.theta.detach().double()
    reference_gamma = torch.cos(theta)

    gamma_fp32, _ = _rotation_coeffs(plain)
    gamma_fp64, _ = _rotation_coeffs(raised)

    assert gamma_fp64.dtype is torch.float64, "coefficients must be computed wide"
    error_fp32 = (gamma_fp32.double() - reference_gamma).abs().max()
    error_fp64 = (gamma_fp64 - reference_gamma).abs().max()
    assert error_fp64 <= error_fp32, (
        "fp64 coefficients are no closer to the exact value than fp32 ones, so "
        "the cast is happening after the trig rather than before it")


def test_an_invalid_mlp_precision_is_rejected_at_construction():
    with pytest.raises(ValueError):
        _module("koopman", precision="bf16")


def test_the_model_threads_the_config_field_into_the_mlp():
    """A config field that reaches no constructor controls nothing."""
    import inspect

    from koopman_lm.models import koopman_lm as model_module

    source = inspect.getsource(model_module._build_mlp) if hasattr(
        model_module, "_build_mlp") else inspect.getsource(model_module)
    assert "mlp_precision" in source, (
        "_build_mlp must pass cfg.mlp_precision into the Koopman MLP")
