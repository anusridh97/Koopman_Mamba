# Evaluate at training precision

**Status:** `lm_harness_eval.py` **done**. `quick_eval` is safe to do immediately.
`evaluate.py` moves published numbers and must re-measure in the same commit.

> `lm_harness_eval.py` was done ahead of the others not because its numbers matter
> less, but because it turned out to be **broken outright**: commit e2dc98a renamed
> its `dtype` parameter and left `self._dtype = dtype` behind, so `__init__` raised
> `NameError` on every call. Nothing could catch it — `lm_eval` is not installed,
> so no test can import the module. Fixing the crash meant touching the parameter
> anyway, and reinstating a default that was demonstrably wrong would have been
> perverse. Its lm-eval numbers therefore need re-measuring, but there were no
> valid numbers being produced in the interim.

**Origin:** a manual review of `jack/search-and-provenance`. The reviewer's
observation was simply that training and eval precision should agree by default.
Chasing it found that no eval path in the repo installs an autocast at all, and
that one of them truncates weights instead — which is a different thing.

## 1. Current state

| site | what it measures | precision | status |
|---|---|---|:--|
| `evaluation/quick_eval.py` | the **search objective** | no autocast → fp32 | **wrong**, fix now |
| `evaluation/lm_harness_eval.py` | benchmark suites | fp32 weights + autocast at `cfg.compute_precision` | **done** |
| `evaluation/evaluate.py` | held-out ppl, NIAH, MQAR | no autocast → fp32 | defensible but unchosen |

Before this spec, none of the three read `cfg.compute_precision`, which every
checkpoint records. `lm_harness_eval.py` now does.

## 2. Casting weights is not autocasting

This is the part that is easy to get wrong, and the previous comment in
`lm_harness_eval.py` did get it wrong.

`compute_precision: bf16` means **fp32 weights with a bf16 autocast**. Autocast
carries an op list: matmul and conv run in bf16, while `layer_norm`, `softmax`
and reductions deliberately stay in fp32, because normalization and
exponentiation in bf16 lose accuracy that matters.

`model.to(torch.bfloat16)` has no op list. It truncates every parameter, so
normalization runs in bf16 as well. It also partly defeats `ska_precision: fp32`:
the whitened core still casts up to fp32, but from inputs that were already
truncated, so the width the config asks for is only nominally present.

So `lm_harness_eval.py` today reproduces neither training (which keeps norms in
fp32) nor a clean fp32 measurement. It is a third regime that nobody selected and
no config field describes.

## 3. Two kinds of eval, two different answers

Conflating these is what made this hard to reason about.

**Selection eval** — `quick_eval`, read by the Optuna driver to score a trial.
Its entire job is to predict how a config will behave in a real training run, so
it must match training precision. Measuring in fp32 is train/selection skew: a
config whose bf16 numerics are marginal scores clean and then degrades in the
real run. For a spectral model with eigenvalue and rotation arithmetic that is
exactly the failure mode most worth catching, so the objective is currently blind
to the risk it most needs to see. Note `compute_precision` is **not** in the
search space, so there is no cross-trial comparability argument against this —
every trial trains at the same precision.

**Reporting eval** — `evaluate.py`, `lm_harness_eval.py`, compared against
published baselines and against this repo's own history. Training precision is
still the right default (a reported number should describe how the model runs),
but two things follow that do not apply to selection eval: an fp32 measurement
must remain available on request, and changing the default invalidates every
number already on record.

## 4. The change

**`quick_eval.py`** — thread the trial's precision in and wrap the loop:

```python
def evaluate_loss(model, device, loader, *, max_batches=None, precision='fp32'):
    ...
    with torch.no_grad(), autocast(str(device).split(':')[0], precision):
```

`precision` flows from `run_quick_eval(..., precision=cfg.compute_precision)`.
The `koopman_lm.precision.autocast` helper already returns a reusable context
manager for exactly this shape, and returns `nullcontext()` for fp32, so the
default argument keeps current behaviour until a caller opts in.

**`lm_harness_eval.py`** — done. The `weight_dtype` parameter is **gone**, not
defaulted: weights stay fp32 and `self._autocast = autocast(device,
cfg.compute_precision)` is built once and entered at each of the three model-call
sites (`_model_call`, and the prefill and decode loop in `_model_generate`).

No knob was reintroduced. A parameter whose only correct value is derivable from
the checkpoint is not configuration, it is an invitation to disagree with the
file. If a genuine serving-precision experiment is ever wanted, it should arrive
as an explicit argument *then*, with a warning when it contradicts
`cfg.compute_precision` — not as a default sitting there in the meantime.

Note the prefill was previously **outside** the `no_grad` block and so outside any
precision control; it now shares the decode loop's autocast, because otherwise the
first generated token comes from different arithmetic than the rest.

This is also the first real consumer of the reusability fix in `3be37b7`: one
autocast object is entered three-plus times per generation call.

**`evaluate.py`** — autocast at `cfg.compute_precision` by default, with
`--eval-precision fp32` to force the arithmetic-independent measurement. Record
the precision used in the results envelope, which nothing currently does.

## 5. Why the reporting paths are not in this PR

Not because the current behaviour is defensible — it isn't — but because the fix
is incomplete without re-measurement. The ppl 214.6 on record for
`50m-first-real` and every lm-eval-harness number this repo has reported were
produced under the current regime, and none of them record which precision that
was. Flipping the default silently makes old and new numbers non-comparable with
no way to tell them apart after the fact.

So the reporting change must land together with: re-measured numbers for anything
on record, and a `precision` field in the results envelope so this cannot recur.
That is a bounded task but a distinct one.

`quick_eval` has no history and no published numbers, and it is the search
objective, so it should just be fixed.

## 6. Testing

1. `evaluate_loss` with `precision='bf16'` on CUDA produces a different loss than
   `precision='fp32'` on the same weights and batch — proving the autocast is
   actually installed rather than silently no-op.
2. `precision='fp32'` is bit-identical to no autocast at all (`nullcontext`), so
   the default argument is a strict no-op.
3. The autocast context is **entered per batch across many batches** — the reuse
   bug from `3be37b7` was a single-use `@contextmanager` that 843 tests passed
   over because each entered it exactly once.
4. `KoopmanEvalWrapper` with no `weight_dtype` leaves parameters in fp32
   (`next(model.parameters()).dtype is torch.float32`).
5. An explicit `weight_dtype` disagreeing with `cfg.compute_precision` warns, and
   names both.
6. GPU: a checkpoint evaluated at its training precision and at fp32 gives two
   numbers; both are recorded in the results envelope, and the envelope says
   which is which.
