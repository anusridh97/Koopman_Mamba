"""One place that turns a config into (autocast context, GradScaler).

Precision design step 5. Four trainers each hardcoded their own answer:

    training/train.py:304          bfloat16, enabled=args.bf16
    retrieval/adapt.py:179         bfloat16, enabled=args.bf16
    experiments/mqar_finetune.py   bfloat16
    experiments/table2.py:233      float16 + GradScaler, both hardcoded

Four copies of a policy is four places for it to drift, and table2.py had already
drifted -- it trains in fp16 while every other trainer uses bf16, a fact visible
nowhere except that line.

**The GradScaler is derived, never configured.** It exists if and only if the
compute precision is fp16. "fp16 without a scaler" silently produces NaNs once
gradients underflow, and "bf16 with a scaler" is pointless overhead -- neither
should be a reachable state, so neither is expressible.

**`enabled` stays a separate knob** so `--bf16/--no_bf16` keeps working. The dtype
comes from the config; the flag only turns autocast off. That is what makes this
step a no-op at the defaults: cfg says bf16, the flag says on, and the result is
the bfloat16 autocast every trainer already had.
"""
import pytest
import torch

pytestmark = pytest.mark.correctness


def _cfg(compute="bf16"):
    import dataclasses

    from koopman_lm.config import KoopmanLMConfig, build_config
    return KoopmanLMConfig(
        **dict(dataclasses.asdict(build_config("50m")), compute_precision=compute))


# ------------------------------------------------------------- context ----

def test_bf16_yields_a_bfloat16_autocast_and_no_scaler():
    """The default, and what all four trainers do today."""
    from experimentation.training.amp import amp_for

    context, scaler = amp_for(_cfg("bf16"), device_type="cpu")
    assert scaler is None, "bf16 has fp32's exponent range and needs no scaling"
    a = torch.ones(4, 4)
    with context:
        assert (a @ a).dtype is torch.bfloat16


def test_fp32_disables_autocast_entirely():
    """Autocasting *to* fp32 is meaningless; "fp32" must mean "no autocast"."""
    from experimentation.training.amp import amp_for

    context, scaler = amp_for(_cfg("fp32"), device_type="cpu")
    assert scaler is None
    a = torch.ones(4, 4)
    with context:
        assert (a @ a).dtype is torch.float32


def test_the_disable_flag_is_honoured_so_no_bf16_keeps_working():
    """train.py's --no_bf16 must still switch autocast off without the config
    having to lie about the precision."""
    from experimentation.training.amp import amp_for

    context, _ = amp_for(_cfg("bf16"), device_type="cpu", enabled=False)
    a = torch.ones(4, 4)
    with context:
        assert (a @ a).dtype is torch.float32


def test_the_context_survives_being_entered_once_per_step():
    """The shape every trainer uses: build once, enter per step. This is the test
    that was missing when a generator-based autocast shipped and killed real
    training at step 2 on an H100 (job 435898) with all 843 CPU tests green --
    every one of them entered the context exactly once."""
    from experimentation.training.amp import amp_for

    for compute in ("bf16", "fp32", "fp16"):
        context, _ = amp_for(_cfg(compute), device_type="cpu")
        for _ in range(4):
            with context:
                pass


# -------------------------------------------------------- grad scaler ----

def test_fp16_derives_an_enabled_grad_scaler():
    """The pairing that must not be separable: fp16 without a scaler silently
    NaNs once gradients underflow."""
    from experimentation.training.amp import amp_for

    _, scaler = amp_for(_cfg("fp16"), device_type="cpu")
    assert scaler is not None
    assert scaler.is_enabled() is True


def test_no_precision_other_than_fp16_gets_a_scaler():
    from experimentation.training.amp import amp_for

    for compute in ("fp32", "bf16"):
        _, scaler = amp_for(_cfg(compute), device_type="cpu")
        assert scaler is None, compute


def test_disabling_autocast_also_disables_the_scaler():
    """A scaler that scales gradients for an autocast that is not running would
    inflate them for no reason."""
    from experimentation.training.amp import amp_for

    _, scaler = amp_for(_cfg("fp16"), device_type="cpu", enabled=False)
    assert scaler is None or scaler.is_enabled() is False


def test_the_scaler_is_not_a_parameter():
    """Derived, not configured: there is no argument that produces fp16 without a
    scaler, which is the whole point."""
    import inspect

    from experimentation.training import amp

    names = set(inspect.signature(amp.amp_for).parameters)
    assert "scaler" not in names and "use_scaler" not in names
    assert names <= {"cfg", "device_type", "enabled"}


# ------------------------------------------------------- the call sites ----

_TRAINERS = [
    "experimentation/training/train.py",
    "experimentation/retrieval/adapt.py",
    "experimentation/experiments/mqar_finetune.py",
    "experimentation/experiments/table2.py",
]


@pytest.mark.parametrize("relpath", _TRAINERS)
def test_no_trainer_hardcodes_its_autocast_dtype(relpath):
    """The point of the step. A hardcoded dtype is a policy decision invisible
    from the config, which is how table2.py came to train in fp16 while the other
    three used bf16 with nothing recording the difference.

    Checked by AST rather than by string search: these files now *document* the
    dtype they used to hardcode, and a text search cannot tell an explanation from
    an instruction. An earlier version of this test failed on its own prose."""
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    tree = ast.parse((root / relpath).read_text())
    offenders = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        # Only autocast calls. A dtype= on torch.tensor(...) is a legitimate
        # tensor construction, and an earlier version of this flagged one.
        if "autocast" not in ast.unparse(node.func):
            continue
        for kw in node.keywords:
            if kw.arg == "dtype":
                offenders.append(f"{ast.unparse(kw.value)} at line {node.lineno}")
    assert not offenders, f"{relpath} hardcodes an autocast dtype: {offenders}"


@pytest.mark.parametrize("relpath", _TRAINERS)
def test_every_trainer_derives_its_amp_from_the_shared_helper(relpath):
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    source = (root / relpath).read_text()
    assert "amp_for" in source, f"{relpath} must use experimentation.training.amp"


def test_table2_still_declares_fp16_so_its_numbers_do_not_move():
    """table2.py trained in fp16 before this step. Reading the registry default
    would silently switch it to bf16 and move Table 2's numbers -- so it declares
    fp16 on the config it builds instead. The design's wording is that its fp16
    becomes "visible in the config rather than buried in the trainer", which is
    preservation plus legibility, not a change."""
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    source = (root / "experimentation/experiments/table2.py").read_text()
    assert 'compute_precision' in source and 'fp16' in source, (
        "table2.py must set compute_precision='fp16' explicitly")
