"""koopman_lm/precision.py -- one place that knows what a precision name means.

Step 1 of the precision-policy design's implementation order. It landed inert --
nothing imported it -- and step 2 (the three config fields) made config.py its
first and so far only consumer; test_only_the_expected_modules_consume_precision
at the bottom is how that stays a checked claim.

Two domain decisions are load-bearing enough to pin here rather than leave to a
docstring.

`compute_precision` excludes fp64, and `ska_precision`/`mlp_precision` exclude
bf16 and fp16. Together those make the design's "component fields may only ever
RAISE precision" invariant unsatisfiable to violate -- no ordering check is
needed, because the global floor can never exceed fp32 and fp32 is the minimum
either component field allows. Widening either domain silently reintroduces the
need for that check, so the domains are asserted.

The component domains exclude bf16 for a concrete reason: the SKA core takes a
Cholesky of a Gram matrix, and bf16's 8 mantissa bits make that unreliable --
part of why ska_ridge exists at all. A field whose purpose is protecting a
fragile component must not be able to break it.
"""
import pytest
import torch

pytestmark = pytest.mark.correctness


# ------------------------------------------------------------ mapping ----

@pytest.mark.parametrize("name,expected", [
    ("fp32", torch.float32),
    ("bf16", torch.bfloat16),
    ("fp16", torch.float16),
    ("fp64", torch.float64),
])
def test_every_precision_name_maps_to_a_torch_dtype(name, expected):
    from koopman_lm.precision import dtype_of

    assert dtype_of(name) is expected


def test_an_unknown_precision_name_is_rejected_by_name(tmp_path):
    """The message has to name the offender: this is reached from a config file,
    where a typo like "bfloat16" is the likely input."""
    from koopman_lm.precision import dtype_of

    with pytest.raises(ValueError, match="bfloat16"):
        dtype_of("bfloat16")


def test_dtype_of_does_not_silently_accept_a_torch_dtype():
    """Passing torch.bfloat16 where a name is expected should fail loudly rather
    than work by accident, or the config surface becomes two things."""
    from koopman_lm.precision import dtype_of

    with pytest.raises((ValueError, TypeError)):
        dtype_of(torch.bfloat16)


# ------------------------------------------------------------- domains ----

def test_the_compute_domain_excludes_fp64():
    """This is what makes "components may only raise" unsatisfiable to violate:
    the global floor can never exceed fp32. Adding fp64 here requires adding an
    explicit ordering check."""
    from koopman_lm.precision import COMPUTE_PRECISIONS

    assert "fp64" not in COMPUTE_PRECISIONS
    assert set(COMPUTE_PRECISIONS) == {"fp32", "bf16", "fp16"}


def test_the_component_domain_excludes_the_low_precisions():
    """ska_precision exists to protect a Cholesky of a Gram matrix; a field that
    could set bf16 would be able to break the thing it exists to protect."""
    from koopman_lm.precision import COMPONENT_PRECISIONS

    assert "bf16" not in COMPONENT_PRECISIONS
    assert "fp16" not in COMPONENT_PRECISIONS
    assert set(COMPONENT_PRECISIONS) == {"fp32", "fp64"}


def test_the_component_minimum_is_at_least_the_compute_maximum():
    """The invariant stated as an assertion over the domains themselves, so it
    survives someone editing one tuple without reading the other."""
    from koopman_lm.precision import COMPONENT_PRECISIONS, COMPUTE_PRECISIONS, rank_of

    assert min(rank_of(p) for p in COMPONENT_PRECISIONS) >= max(
        rank_of(p) for p in COMPUTE_PRECISIONS)


def test_precision_ranks_order_the_names_by_width():
    from koopman_lm.precision import rank_of

    assert rank_of("fp16") == rank_of("bf16") < rank_of("fp32") < rank_of("fp64")


# ------------------------------------------------------- grad scaler ----

def test_a_grad_scaler_is_needed_only_for_fp16():
    """Derived, not configured, so "fp16 without a scaler" is not a reachable
    state. bf16 has fp32's exponent range and needs no scaling."""
    from koopman_lm.precision import needs_grad_scaler

    assert needs_grad_scaler("fp16") is True
    assert needs_grad_scaler("bf16") is False
    assert needs_grad_scaler("fp32") is False


# ---------------------------------------------------------- autocast ----

def test_autocast_casts_matmuls_to_the_requested_dtype():
    """Behavioural rather than flag-based: torch's autocast introspection helpers
    have changed name across versions, but what a matmul produces has not."""
    from koopman_lm.precision import autocast

    a = torch.ones(4, 4)
    b = torch.ones(4, 4)
    with autocast("cpu", "bf16"):
        assert (a @ b).dtype is torch.bfloat16


def test_autocast_at_fp32_is_a_no_op():
    """Autocasting *to* fp32 is meaningless, and enabling it anyway would add
    overhead and change nothing -- so fp32 must disable rather than configure."""
    from koopman_lm.precision import autocast

    a = torch.ones(4, 4)
    with autocast("cpu", "fp32"):
        assert (a @ a).dtype is torch.float32


def test_autocast_restores_the_previous_state_on_exit():
    from koopman_lm.precision import autocast

    a = torch.ones(4, 4)
    with autocast("cpu", "bf16"):
        pass
    assert (a @ a).dtype is torch.float32


def test_autocast_can_be_disabled_explicitly():
    """The call sites that currently pass enabled=args.bf16 need this to keep
    their CLI behaviour while reading the dtype from config."""
    from koopman_lm.precision import autocast

    a = torch.ones(4, 4)
    with autocast("cpu", "bf16", enabled=False):
        assert (a @ a).dtype is torch.float32


def test_the_autocast_context_can_be_entered_more_than_once():
    """Every trainer builds this ONCE before the loop and enters it per step:

        autocast_ctx = amp_for(cfg, 'cuda', ...)      # once
        for step in ...:
            with autocast_ctx:                        # every step

    torch.amp.autocast supports that; a @contextlib.contextmanager generator does
    not -- _GeneratorContextManager.__enter__ deletes self.args, so the second
    entry raises `AttributeError: args`. The first version of this helper was a
    generator, and every test entered it exactly once, so 843 CPU tests passed
    over a bug that killed real training at step 2 (job 435898). Enter it three
    times.
    """
    from koopman_lm.precision import autocast

    context = autocast("cpu", "bf16")
    a = torch.ones(4, 4)
    for _ in range(3):
        with context:
            assert (a @ a).dtype is torch.bfloat16


def test_the_disabled_context_can_also_be_entered_more_than_once():
    """The fp32/disabled branch has to be reusable too, or `compute_precision:
    fp32` breaks on step 2 instead of bf16 doing so."""
    from koopman_lm.precision import autocast

    for precision, enabled in (("fp32", True), ("bf16", False)):
        context = autocast("cpu", precision, enabled=enabled)
        a = torch.ones(4, 4)
        for _ in range(3):
            with context:
                assert (a @ a).dtype is torch.float32


def test_autocast_rejects_an_unknown_precision():
    from koopman_lm.precision import autocast

    with pytest.raises(ValueError):
        with autocast("cpu", "float8"):
            pass


# ------------------------------------------------------------ layering ----

def test_precision_does_not_import_experimentation():
    """The dependency edge runs experimentation -> koopman_lm only, and this
    module has to be usable from both sides."""
    import ast
    from pathlib import Path

    source = (Path(__file__).resolve().parent.parent
              / "koopman_lm" / "precision.py").read_text()
    tree = ast.parse(source)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert not any(name.startswith("experimentation") for name in imported)


# Modules allowed to import precision.py at each stage of the design's
# implementation order. config.py arrived with step 2 (the three fields). The
# model and training call sites are steps 3 onward and are NOT wired yet, so
# this list is how "not yet wired" stays a checked claim rather than a memory.
_EXPECTED_CONSUMERS = {
    "koopman_lm/config.py",           # step 2: the three fields + validation
    "koopman_lm/modules/seq/ska.py",  # step 3: the whitened core's dtype
    "koopman_lm/modules/mlp/koopman.py",  # step 3: the rotation coefficients
    "experimentation/run/spec.py",         # step 4: the runtime/model cross-check
    "experimentation/training/amp.py",      # step 5: the trainers' shared amp helper
    "experimentation/evaluation/lm_harness_eval.py",  # step 6: the serving cast
}


def test_only_the_expected_modules_consume_precision():
    """The design lands in numbered steps, and each is meant to be separately
    reviewable. This pins which step we are on: widening the set is the signal
    that the model/training wiring (steps 3 onward) has landed, and that the
    commit doing it owes a behavioural argument rather than "no-op".
    """
    import subprocess
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        ["grep", "-rl", "--include=*.py", "koopman_lm.precision",
         str(root / "koopman_lm"), str(root / "experimentation")],
        capture_output=True, text=True)
    consumers = {
        str(Path(line).resolve().relative_to(root))
        for line in result.stdout.split() if line
    }
    assert consumers == _EXPECTED_CONSUMERS, (
        f"unexpected consumers: {sorted(consumers - _EXPECTED_CONSUMERS)}; "
        f"missing: {sorted(_EXPECTED_CONSUMERS - consumers)}")
