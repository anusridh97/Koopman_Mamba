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

def test_the_harness_names_its_weight_cast_weight_dtype():
    """It casts weights, not compute. Calling that parameter `dtype` invited
    exactly the confusion design 5 exists to dissolve."""
    source = (_ROOT / "experimentation/evaluation/lm_harness_eval.py").read_text()
    tree = ast.parse(source)
    init = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    names = {arg.arg for arg in init.args.args + init.args.kwonlyargs}
    assert "weight_dtype" in names, "the serving cast must be named weight_dtype"


def test_the_harness_default_still_serves_in_bf16():
    """Preserved on purpose: changing the default would silently change every
    lm-eval-harness number this repo has reported."""
    source = (_ROOT / "experimentation/evaluation/lm_harness_eval.py").read_text()
    tree = ast.parse(source)
    init = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    args = init.args.args + init.args.kwonlyargs
    defaults = ([None] * (len(init.args.args) - len(init.args.defaults))
                + list(init.args.defaults) + list(init.args.kw_defaults))
    by_name = {arg.arg: default for arg, default in zip(args, defaults)}
    rendered = ast.unparse(by_name["weight_dtype"])
    assert "bf16" in rendered or "bfloat16" in rendered, rendered


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
