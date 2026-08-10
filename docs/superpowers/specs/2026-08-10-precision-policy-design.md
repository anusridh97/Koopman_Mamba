# Precision policy: per-component dtypes declared on the model config

Written 2026-08-10. Every claim about current behavior below was verified by
reading or running the code, with file:line given. Where something is
unverified, it says so.

---

## 1. The problem

Precision in this repo is decided in ten places, by three different mechanisms,
with no single declaration and nothing recorded.

**Inference does it three incompatible ways.** These are not merely different
defaults — weight-casting and autocast have different numerics and different
memory behavior, and one of them destroys the fp32 master weights:

| Path | What it does | Effect |
|---|---|---|
| `evaluation/evaluate.py` | sets nothing | runs fp32 |
| `evaluation/lm_harness_eval.py:69,137` | `.to(dtype=torch.bfloat16)` | casts the **weights** |
| `evaluation/evaluate_retrieval.py:541` | `autocast(dtype=bfloat16)` | weights stay fp32 |
| `models/recurrent.py:221,245` | `dtype=x.dtype` | inherits whatever arrives |

**The SKA fp32 contract is invisible.** `modules/seq/ska.py:535-542` disables
autocast and upcasts to fp32 with no stated reason:

```python
ctx = torch.amp.autocast('cuda', enabled=False) if hidden_states.is_cuda else nullcontext()
with ctx:
    z_f = z.float(); zq_f = zq.float(); v_f = v.float(); beta_f = beta.float()
```

The reason is real — the whitened core Choleskys a Gram matrix
(`kernels/ska_operator.py:35`) — but a reader has to infer it, and nothing
enforces it.

**fp16 is used but inexpressible.** `RuntimeSpec.precision` allows only
`{bf16, fp32}`, while `experiments/table2.py:229-230` trains in fp16 with a
`GradScaler`. A trainer in this repo uses a precision the spec cannot describe.
This is the numerics difference recorded in
`docs/superpowers/HANDOFF-2026-08-08.md` §7.

**The Koopman MLP has no guard for comparable math.** 9 of 11 configs use the
Koopman MLP with `mlp_spectral_norm=True`, and `modules/mlp/koopman.py:120-142`
runs, per forward, either `_disk_clamp(gamma, omega)` (a divide-by-radius
normalization; 8 configs) or `rho = exp(-softplus(s))` (`180m_v2`). Both execute
in bf16 today, while SKA forces fp32 for mathematically similar work.

**Nothing is recorded.** No checkpoint, `spec.yaml`, or result envelope states
what precision produced it.

---

## 2. The design

Three flat fields on `KoopmanLMConfig`, governed by one rule:

> **`compute_precision` sets the floor for the whole network.
> `ska_precision` and `mlp_precision` raise it for components whose math needs
> more.**

```python
compute_precision: str = "bf16"        # fp32 | bf16 | fp16 -- global autocast dtype
ska_precision: str = "fp32"            # fp32 | fp64 -- the whitened core
mlp_precision: Optional[str] = None    # None -> follow compute_precision; fp32 | fp64
```

`compute_precision` names the dtype **ops run in**. Weights are always fp32;
storage is a separate concern (§5).

`ska_precision` and `mlp_precision` may only ever *raise* precision. That is why
`bf16` and `fp16` are not in their domain: these fields exist to protect
numerically fragile components, so they cannot be used to break them.

The "only raise" invariant needs no ordering check, because the domains make it
unsatisfiable to violate: `compute_precision` deliberately **excludes `fp64`**,
so the global floor can never exceed `fp32`, which is the minimum either
component field allows. If `fp64` is ever added to `compute_precision`, an
explicit ordering validation becomes necessary.

### Why flat, not a nested `PrecisionPolicy`

The sweep machinery decides this. `sweep/spec.py:_parse_axis_key` splits an axis
key on the **first** `.` only, and `build_cell_run_spec` then does
`KoopmanLMConfig(**cell_sections["model"])`:

```
model.ska_precision      -> section=model, field=ska_precision      works
model.precision.default  -> section=model, field="precision.default"
                            -> TypeError: unexpected keyword argument
```

Flat fields are sweepable for free, with correct grouping, because `model` is
hashed into `group_id`. A nested dataclass would not be reachable by overrides.
Flat also matches `KoopmanLMConfig`'s existing shape (56 flat fields, no
nesting) and its component-first naming (`ska_rank`, `mlp_expand`).

### Why on the model config and not `RuntimeSpec`

Three reasons:

1. **Inference needs it and has no `RuntimeSpec`.** That absence is the direct
   cause of the four hardcoded paths in §1.
2. **It changes results, so it is a scientific input.** It belongs on the side
   of the line that is hashed.
3. **It travels for free.** `training/train.py:519` already writes
   `checkpoint_meta(cfg, ...)` into `meta.pt`, and `evaluation/harness.py:39-40`
   already reads it — so putting precision on the config delivers it to every
   inference path with no new plumbing (§4.3).

---

## 3. Validation

Both checks run in `KoopmanLMConfig.__post_init__`, so they fail at config-load
time, before any GPU is allocated.

**`ska_precision` must be `fp32` or `fp64`.** `bf16`/`fp16` are rejected with the
reason: the core does `torch.linalg.cholesky` on a Gram matrix
(`kernels/ska_operator.py:35`), and bf16's 8 mantissa bits make that unreliable
— which is part of why `ska_ridge` exists.

**`ska_precision == "fp64"` is incompatible with `ska_backend == "cuda_prefix"`.**
The fused kernel is fp32-only, enforced twice:

```
kernels/csrc/prefix_scan_ext.cu:57   #define CHECK_F32(x) TORCH_CHECK(... == at::kFloat, ...)
kernels/cuda_prefix_scan.py:84       x.dtype == q.dtype == vbar.dtype == torch.float32
```

The error message cites both, so the constraint is discoverable at load time
rather than as a CUDA failure mid-run.

**fp64 is not hypothetical.** The exact prefix-scan path already accepts it —
`kernels/prefix_scan.py:1178`: `if x.dtype not in (torch.float32, torch.float64)`
— and fp64 is already used for numerical validation across
`kernels/ska_operator.py:178-182`, `kernels/chunk_stats_exact.py:62`,
`kernels/factor_scan.py:245`, and `code-tests/test_spectral_stability.py`. So
`ska_precision`'s domain is genuinely two-valued on the PyTorch path, and
one-valued only under `cuda_prefix`.

**`mlp_precision`** is `None` or one of `{fp32, fp64}`. `None` is the default and
a strict no-op: the code path is untouched, so existing numerics are unchanged.

---

## 4. Consumption

### 4.1 One shared helper

`koopman_lm/precision.py` — name→`torch.dtype` mapping plus an autocast context
helper. Placed at the package root beside `config.py` and `pooling.py`: the
category of thing every subpackage may use and none owns. It must live under
`koopman_lm/` because the dependency edge runs
`experimentation -> koopman_lm` only, enforced by
`code-tests/test_package_boundary.py`.

### 4.2 Model side — two call sites

**`modules/seq/ska.py:535-542`** — the autocast-disable and `.float()` become
`.to(dtype_of(ska_precision))`. Identical behavior at the default; now stated
and enforced instead of conventional.

**`modules/mlp/koopman.py:_rotation_coeffs`** — wrapped only when
`mlp_precision` is not `None`.

`SKAModule.__init__` takes 28 scalars and never imports `KoopmanLMConfig`
(deliberately — it is config-free numerics). `ska_precision` becomes arg 29,
threaded by `SKABlock`, exactly as the other 28 already are. `mlp_precision`
threads through `KoopmanLM._build_mlp` the same way.

### 4.3 Training side — five call sites, one shape

```
training/train.py:313          autocast(dtype=bfloat16, enabled=args.bf16) -> dtype from cfg
retrieval/adapt.py:178         same
experiments/mqar_finetune.py:271  same
experiments/table2.py:229-230  fp16 + GradScaler -> the now-expressible fp16 case
training/train.py:549-550      --bf16/--no_bf16 kept for CLI compat
run/launch.py:86               derives the flag from cfg.compute_precision
```

`GradScaler` is constructed **iff** `compute_precision == "fp16"`. It is derived,
not configured, so fp16-without-a-scaler is not a reachable state.

### 4.4 Inference side — the four paths collapse

`compute_precision` arrives with the checkpoint (§2), so each path stops
hardcoding:

```
evaluate.py                 nothing            -> read cfg from meta.pt
lm_harness_eval.py:69,137   weight-cast bf16   -> weight_dtype arg; compute from meta
evaluate_retrieval.py:541   hardcoded bf16     -> from meta
recurrent.py:221,245        dtype=x.dtype      -> unchanged; state matching its input
```

`recurrent.py` is deliberately left alone: it allocates recurrent state to match
the dtype of the activations it is handed, which is correct regardless of policy.

---

## 5. `weight_dtype` stays outside the config

Serving precision (`fp32 | bf16 | fp16`) is a **per-invocation** argument on the
eval entry points, recorded in `run/eval_result.py`'s envelope — not a config
field. `fp16` is admitted here because inference has no gradients, so it carries
none of the loss-scaling requirement that makes fp16 training need a
`GradScaler`; it can still overflow on large activations, which is the caller's
trade to make and is why it is recorded in the envelope.

During training, weight storage must be fp32; bf16 master weights are unsafe. So
a `weight_dtype` field on the model config would be meaningful only at inference,
i.e. a training config carrying an inference-only value.

Keeping the two separate also dissolves override precedence: the checkpoint
declares **compute**, the caller chooses **storage**, and they never contend for
the same decision.

Consequence, stated so it is not a surprise: serving precision is **not**
sweepable. If measuring served-quality-vs-memory becomes a research question,
that is a later change.

---

## 6. `runtime.precision`: deprecated, not deleted

`RuntimeSpec(**d)` raises `TypeError` on unknown keys. All three
`configs/runs/*.yaml` set `runtime.precision`, and so does every materialized
`spec.yaml` on scratch — which `run/resolve.py:load_materialized_spec` reads when
a run **resumes**. Deleting the field would break resume for existing runs.

Note the asymmetry: the model section has a migration story
(`run/resolve.py:_check_model_key_set` raises listing exactly which keys drifted);
the runtime section has none.

So: the field stays and is validated. `run/launch.py` stops reading it and
derives the flag from `model.compute_precision`.

The disagreement check must live in **`RunSpec.__post_init__`** (`spec.py:177`),
not `RuntimeSpec.__post_init__`: `RuntimeSpec` is a standalone frozen dataclass
with no access to `spec.model`, whereas `RunSpec` holds both sections. It raises
if `runtime.precision` is set to something that disagrees with
`model.compute_precision`, rather than silently preferring one.

The field is removed in a separate later cleanup, once nothing on scratch
carries it.

---

## 7. Error handling

- **Config load** — the two validations in §3. Both fail before GPU allocation.
- **Old checkpoints** — `meta.pt` written before this change has no
  `compute_precision`, so `KoopmanLMConfig(**old)` fills the default. The default
  is `bf16` **specifically because** that is what existing runs trained with
  (`train.py:549`'s `--bf16` defaults to `True`). Choosing `fp32` as the default
  would silently misdescribe every existing checkpoint.
- **Hardware mismatch** — bf16 requested on a card without it **raises**; it does
  not silently fall back. Silent fallback is the failure mode that already cost
  this repo twice: the `except Exception` that disabled the Triton kernel
  (`kernels/cholesky_update.py:128`) and `pretrain.sh`'s truncating
  `GA_USE=$(( GA / NPROC ))`.

---

## 8. Testing

All CPU-runnable, so all of it lands in the `correctness` gate suite.

1. `ska_precision="bf16"` raises; `ska_precision="fp64"` with
   `ska_backend="cuda_prefix"` raises citing `CHECK_F32`.
2. **`mlp_precision=None` is bit-identical to today.** Pinned, so the no-op
   default cannot silently become a behavior change.
3. `SKAModule(ska_precision="fp64")` on the PyTorch path: the core's tensors are
   actually fp64 (not merely accepted and downcast).
4. **All four inference paths resolve the same dtype from the same `meta.pt`.**
   This pins the §1 inconsistency shut so it cannot reappear.
5. `compute_precision="fp16"` ⇒ the trainer builds a `GradScaler`; `bf16`/`fp32`
   ⇒ it does not.
6. Identity baseline regenerated, with the old→new mapping recorded in the commit
   message, per `code-tests/test_identity_baseline.py`'s stated doctrine.

---

## 9. The identity break

Three new fields on `KoopmanLMConfig` change `config_hash` for all 11 registry
configs. Therefore:

- every `group_id` and `run_id` moves;
- the one existing real checkpoint's stamped `cfg_hash` (`train.py:522`) stops
  matching a freshly-loaded config.

This is strictly larger than a `RuntimeSpec`-only change, because `config_hash`
itself moves. `test_identity_baseline.py` exists to force this to be deliberate;
its instruction is followed literally — regenerate via
`scripts/gen_identity_baseline.py`, record the old→new mapping, never to make a
red test green.

The affected completed run is `50m-first-real.81033b58/seed42.5467934f` (job
415896, 100M tokens, held-out ppl 214.6). Recommended handling: accept the
lineage break and note it, rather than patch anything, given that run is a smoke
test. `results.py` will still find it — it globs `spec.yaml` and reads contents —
but will group it separately from post-change runs.

---

## 10. Decisions made against

**A nested `PrecisionPolicy` dataclass.** Not sweepable; see §2.

**`SKA_CORE_DTYPE` as a module constant instead of a field.** Considered, and
rejected once `prefix_scan.py:1178` showed fp64 is already accepted on the
PyTorch path. A constant would misrepresent a two-valued domain as one-valued,
and would not be enforced.

**Dropping `mlp_precision` as YAGNI.** Rejected: the justification was that
`PairMixer._cayley` is unreachable, but that was the wrong code. `_disk_clamp`
and `exp(-softplus(s))` run every forward in 9 of 11 configs.

**One field whose meaning differs by context** (autocast when training, weight
cast when serving). Rejected: the same declared value meaning two things is
precisely the ambiguity that produced §1.

**Keeping `runtime.precision` as a hardware capability ceiling** with
`effective = min(requested, capability)`. Rejected: it reintroduces "the same
experiment ran at two precisions," which is the problem being solved.

---

## 11. Non-goals

**fp8 is explicitly out of scope.** `torch.float8_e4m3fn` does not compose with
`torch.autocast` the way bf16 does. It requires per-tensor scaling factors
maintained across steps (amax history, delayed-scaling recipes) or NVIDIA
Transformer Engine's `te.Linear` replacements — a stateful subsystem, not a
config value. The H100s (sm_90) would support it; it belongs in a later spec.
`compute_precision` must reject `"fp8"` with a message saying so, so nobody
assumes the string will work.

**Quantized serving** (int8/int4, GPTQ/AWQ-style) is out of scope for the same
reason: it is a separate pipeline, not a dtype.

**`modules/seq/attention.py`'s RoPE fp32 tables** (lines 26-33) are already
dtype-correct by construction — computed in fp32, cast back to `x.dtype`, with a
docstring explaining why. Unchanged by this design.

**The `mlp_pair_mixer='learned'` Cayley solve** (`koopman.py:202`,
`torch.linalg.solve`) is left relying on PyTorch autocast's own fp32 op policy.
Unreachable in all 11 configs (`mlp_pair_mixer: None` everywhere), so it is
recorded here as a known gap rather than covered.

---

## 12. Suggested implementation order

This touches ~16 files, so the order matters: each step below is independently
verifiable, and every step through 4 is a **no-op on behavior** (the defaults
reproduce today exactly), which keeps the diff reviewable.

1. **`koopman_lm/precision.py`** — the name→dtype map and autocast helper, plus
   its unit tests. Nothing consumes it yet.
2. **The three fields + validation** on `KoopmanLMConfig`, with the §3 tests.
   This is where `config_hash` moves; regenerate the identity baseline here and
   record the mapping, so exactly one commit owns the identity break.
3. **Model side** — `ska.py`, `koopman.py`, threaded via `ska_block.py` and
   `_build_mlp`. Gate: test 2 of §8 (`mlp_precision=None` bit-identical) and the
   existing 480-test suite.
4. **`RunSpec` cross-check + `launch.py` derivation + `runtime.precision`
   deprecation** (§6). Gate: existing `spec.yaml` files still load.
5. **Training side** — the four autocast sites and the derived `GradScaler`.
   First behavior change: `table2.py`'s fp16 becomes declared rather than
   hardcoded.
6. **Inference side** — the four paths, plus `weight_dtype` and the envelope
   field. Gate: test 4 of §8 (all four resolve identically).

Steps 5 and 6 are the only ones that can change a number, and they change it
only for configs that opt into a non-default value — except `table2.py`, whose
fp16 becomes visible in the config rather than buried in the trainer.
