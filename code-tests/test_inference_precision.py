"""The inference side (precision design step 6), including what it leaves alone.

Two genuine hardcodes get fixed, and two deliberate non-changes get pinned so a
later reader does not "finish the job" by changing them.

**Fixed.** `evaluate_retrieval.py:541` autocasts bf16 for its fine-tune loop --
that is a compute-precision decision and belongs with the other four training
sites. `lm_harness_eval.py:69` casts *weights* to bf16, which design 5 says is a
different thing entirely: serving precision is a per-invocation caller choice, not
a config field, because during training weight storage must be fp32 (bf16 master
weights are unsafe), so a config field for it would be meaningful only at
inference. Renaming it `weight_dtype` is the point -- "the checkpoint declares
compute, the caller chooses storage, and they never contend for the same decision".

**Left alone, deliberately.**

`recurrent.py` allocates recurrent state with `dtype=x.dtype`, matching the
activations it is handed. The design says to leave it, and it is right: that is
correct under any policy.

`evaluate.py` evaluates in fp32 today -- it installs no autocast at all. Reading
`compute_precision` there would move every reported eval number, including the
held-out ppl 214.6 on record for 50m-first-real. There is also an argument it
would be wrong on the merits: `compute_precision` records how a model was
*trained*, and applying it at eval time would make two models trained at
different precisions non-comparable on the same benchmark, which is backwards.
Evaluating everything in fp32 measures the model rather than the arithmetic.
Flagged as a decision rather than taken.
"""
import ast
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.correctness

_ROOT = Path(__file__).resolve().parent.parent


def _autocast_dtypes(relpath):
    """dtype= keywords on autocast calls only."""
    tree = ast.parse((_ROOT / relpath).read_text())
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and "autocast" in ast.unparse(node.func):
            for kw in node.keywords:
                if kw.arg == "dtype":
                    found.append(ast.unparse(kw.value))
    return found


# --------------------------------------------------- serving dtype helper ----

def test_a_serving_dtype_can_be_named_or_passed_as_a_torch_dtype():
    """Serving precision arrives from a CLI or a caller, so both spellings turn
    up. dtype_of deliberately refuses a torch.dtype to keep the *config* surface
    one thing; this is the inference-side entry point where both are legitimate."""
    from koopman_lm.precision import as_dtype

    assert as_dtype("bf16") is torch.bfloat16
    assert as_dtype(torch.bfloat16) is torch.bfloat16
    assert as_dtype("fp32") is torch.float32
    assert as_dtype(torch.float16) is torch.float16


def test_fp16_is_admissible_for_serving():
    """Design 5: inference has no gradients, so fp16 carries none of the
    loss-scaling requirement that makes fp16 *training* need a GradScaler."""
    from koopman_lm.precision import as_dtype

    assert as_dtype("fp16") is torch.float16


def test_an_unknown_serving_dtype_is_rejected():
    from koopman_lm.precision import as_dtype

    with pytest.raises(ValueError):
        as_dtype("float8")


# ------------------------------------------------------------- retrieval ----

def test_the_retrieval_finetune_no_longer_hardcodes_its_autocast_dtype():
    assert _autocast_dtypes("experimentation/evaluation/evaluate_retrieval.py") == []


def test_the_retrieval_finetune_uses_the_shared_amp_helper():
    source = (_ROOT / "experimentation/evaluation/evaluate_retrieval.py").read_text()
    assert "amp_for" in source


# ------------------------------------------------------------ lm-harness ----

_HARNESS = "experimentation/evaluation/lm_harness_eval.py"


def _harness_init():
    tree = ast.parse((_ROOT / _HARNESS).read_text())
    return next(node for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef) and node.name == "__init__")


def test_the_harness_takes_no_precision_parameter():
    """The checkpoint declares its precision, so there is nothing for a caller to
    choose. This used to be a `weight_dtype='bf16'` parameter, which was wrong
    twice over: it was the only value the code could ever have (no caller passed
    it), and truncating weights is not what compute_precision='bf16' means --
    that means fp32 weights with a bf16 autocast, which keeps layer_norm and
    softmax in fp32. A parameter whose only correct value is derivable from the
    file is not configuration."""
    names = {arg.arg for arg in _harness_init().args.args
             + _harness_init().args.kwonlyargs}
    offenders = {n for n in names if "dtype" in n or "precision" in n}
    assert not offenders, f"precision must come from the checkpoint, not: {offenders}"


def test_the_harness_does_not_truncate_the_weights():
    """`.to(device)` and never `.to(device, dtype=...)`. Casting parameters has no
    op list -- it would take layer_norm and softmax to bf16 as well, which
    autocast deliberately does not, and would feed the SKA core already-truncated
    inputs despite ska_precision='fp32'."""
    casts = [node for node in ast.walk(_harness_init())
             if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Attribute) and node.func.attr == "to"]
    assert casts, "expected a .to(...) call moving the model to its device"
    for call in casts:
        assert not any(kw.arg == "dtype" for kw in call.keywords), \
            f"weights must stay fp32: {ast.unparse(call)}"


def test_the_harness_autocasts_at_the_checkpoints_declared_precision():
    calls = [node for node in ast.walk(_harness_init())
             if isinstance(node, ast.Call)
             and getattr(node.func, "id", None) == "autocast"]
    assert len(calls) == 1, "expected exactly one autocast built in __init__"
    args = [ast.unparse(a) for a in calls[0].args]
    assert any("compute_precision" in a for a in args), args


def test_the_harness_reuses_one_autocast_across_every_model_call():
    """Three call sites -- _model_call, and the prefill and decode loop in
    _model_generate -- share one context object, so it is entered many times per
    evaluation. That is only safe because 3be37b7 made precision.autocast return
    torch.autocast/nullcontext instead of a single-use @contextmanager
    generator, whose second __enter__ raises AttributeError: args."""
    tree = ast.parse((_ROOT / _HARNESS).read_text())
    entries = [item for node in ast.walk(tree) if isinstance(node, ast.With)
               for item in node.items
               if ast.unparse(item.context_expr) == "self._autocast"]
    assert len(entries) >= 3, f"only {len(entries)} site(s) enter self._autocast"


def test_the_harness_autocasts_its_generation_prefill():
    """The prefill used to sit outside the no_grad block entirely, so it had no
    precision control at all -- the first generated token came from different
    arithmetic than every token after it."""
    tree = ast.parse((_ROOT / _HARNESS).read_text())
    gen = next(node for node in ast.walk(tree)
               if isinstance(node, ast.FunctionDef) and node.name == "_model_generate")
    prefills = [node for node in ast.walk(gen) if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute) and node.func.attr == "prefill"]
    assert prefills, "expected a recurrent prefill in _model_generate"
    guarded = [node for node in ast.walk(gen) if isinstance(node, ast.With)
               and any(ast.unparse(i.context_expr) == "self._autocast"
                       for i in node.items)
               and any(call is p for p in prefills
                       for call in ast.walk(node))]
    assert guarded, "the prefill must run under the same autocast as the decode loop"


# ------------------------------------------------- the deliberate non-changes ----

def test_evaluate_installs_no_autocast():
    """Pinned as a decision, not an oversight. evaluate.py measures in fp32;
    reading compute_precision here would move the held-out ppl 214.6 on record for
    50m-first-real, and would arguably be wrong anyway -- compute_precision records
    how a model was TRAINED, and applying it at eval time makes two models trained
    at different precisions non-comparable on the same benchmark."""
    assert _autocast_dtypes("experimentation/evaluation/evaluate.py") == []
    source = (_ROOT / "experimentation/evaluation/evaluate.py").read_text()
    assert "amp_for" not in source, (
        "evaluate.py evaluating at the training precision is a decision for a "
        "human, not a consistency fix -- see this test's docstring")


def test_recurrent_state_still_matches_the_dtype_it_is_handed():
    """The design says to leave this alone and it is right: allocating recurrent
    state to match the incoming activations is correct under any policy, and
    forcing a configured dtype here would break mixed-precision decode."""
    source = (_ROOT / "koopman_lm/models/recurrent.py").read_text()
    assert "dtype=x.dtype" in source
    assert "compute_precision" not in source
