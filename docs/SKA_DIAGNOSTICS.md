# SKA Health Diagnostics (Phase 1)

Instrumentation for the `koopman_lm` package that answers one question during
training: **are the Structured Kernel Attention (SKA) layers actually
contributing to the model's output, or is the model learning to route around
them?**

It emits per-layer / per-head SKA health metrics to the console and to Weights &
Biases, cheaply enough to run during training, and ships a CPU test suite.

Ported from the `origin/phase1` old-layout branch (`echo-ska-440m/`) onto the
refactored package; see "Semantic adaptations" below for what changed in the
port.

---

## Files

| File | Status | Purpose |
|---|---|---|
| `koopman_lm/modules/token_mixer/ska.py` | edited | `SKAModule.collect_diagnostics()` — applied + **raw** spectral radius, α clamp factor |
| `koopman_lm/training/diagnostics.py` | **new** | `SKAHealthMonitor`, `GradFlowMonitor`, `profile_overhead()` |
| `koopman_lm/training/train.py` | edited | wires the monitors into the training loop + CLI flags (opt-in) |
| `koopman_lm/evaluation/harness.py` | edited | `eval_load_bearing()` — four-mode SKA/Mamba decomposition |
| `koopman_lm/evaluation/calibrate.py` | **new** | Phase-1 threshold calibration on reference models |
| `code-tests/test_diagnostics.py` | **new** | CPU-only test suite (no GPU / mamba_ssm / wandb needed) |

---

## The five health metrics

All are computed from the **prefix-mode, per-chunk Koopman operators that the
training forward actually applies** (see "Faithfulness" below), not a proxy.

| Metric | What it measures | Healthy range |
|---|---|---|
| **Spectral radius** of `A_eff` (per layer/head) | dominant eigenvalue magnitude of the APPLIED transition operator `A_eff = γ·α·(L⁻¹ M L⁻ᵀ)`. `α` clamps `σ_max(A_eff) ≤ 1`, so this is **always ≤ 1**. Identifies whether key→value bindings are persistent. | `[0.3, 0.95]`; `~0` = operator unlearned |
| **Raw spectral radius** = `radius / α` (pre-clamp, per layer/head) | radius of `γ·W` **before** the α safety clamp — where instability actually shows. `frac_unstable` is measured on THIS, not the clamped radius. | `≤ 1`; `> 1` = unstable (α clamp is load-bearing) |
| **α (clamp factor)** | `1/max(σ_max(W), 1) ∈ (0, 1]`. `alpha_min` / `frac_clamped` report how hard / how often the spectral-norm clamp engages. | near 1 = clamp rarely fires |
| **λmin(G̃)** | smallest eigenvalue of the ridge-regularized Gram matrix; Cholesky conditioning. | comfortably above the ridge floor `ε`; pinned at `ε` ⇒ rank-deficient keys |
| **Gap** `‖A_eff^K − A_eff‖ / ‖A_eff‖` | how much the power filter (squaring, K=2) reshapes the operator. | `< 0.5`; above ⇒ the filter dominates rather than confirms |
| **Write-gate magnitude** | the LayerScale residual gate (`layerscale_gate`); how hard SKA is injected into the residual. Plot on a **log scale**; watch for monotonic growth. | grows off its `1e-4` init; flat ⇒ SKA effectively dead |
| **Residual-norm contribution** `‖Δ_SKA‖ / ‖Δ_Mamba‖` | SKA's contribution to the residual stream relative to the Mamba layers. | within ~1 order of magnitude; orders smaller ⇒ not doing real work |

---

## 1. `SKAModule.collect_diagnostics()`  (`koopman_lm/modules/token_mixer/ska.py`)

The core measurement. A detached, fp32, no-grad method on the SKA module.

```python
metrics = ska_module.collect_diagnostics(hidden_states, max_batch=2)
```

**Input**
- `hidden_states` — `(B, T, d_model)` float tensor: the layer-normed input that
  feeds the SKA module (i.e. `block.norm(x)`).
- `max_batch` — caps the batch used for the per-chunk eigen/operator work so the
  cost is bounded regardless of training batch size (default 2).

**Output** — a `dict` of GPU tensors (the caller does the CPU sync):

| key | shape | meaning |
|---|---|---|
| `spectral_radius` | `(B, nc, H)` | `max eig(A_eff)` per (batch, chunk, head); applied (clamped) operator, ≤ 1 |
| `raw_spectral_radius` | `(B, nc, H)` | pre-clamp radius `= spectral_radius / α`; `> 1` ⇒ unstable |
| `alpha` | `(B, nc, H)` | spectral-norm clamp factor `∈ (0, 1]` |
| `lambda_min` | `(B, nc, H)` | smallest eig of `G̃` per instance |
| `gap` | `(B, nc, H)` | `‖A_eff^K − A_eff‖ / ‖A_eff‖` per instance |
| `n_chunks` | `int` | number of chunks (`nc`); chunk 0 has no history |
| `gate_mag` | scalar | mean `|layerscale_gate|` (falls back to the resolved `|eta|` if no LayerScale) |
| `beta_mean` | scalar | mean `sigmoid(beta)` causal write-gate |
| `outproj_norm` | scalar | `‖out_proj.weight‖_F` |
| `eta` | scalar | the resolved η this forward applies (see regimes below) |
| `gamma` | scalar | the resolved γ folded into `A_eff` |
| `ridge_eps` | scalar | the ridge floor, for the `λmin` ratio |

The three operator metrics come back as the **full `(B, nc, H)` distributions**
so downstream code can aggregate per head, per chunk, or over the whole pool
however it wants.

**Faithfulness.** It calls the *same* `chunk_stats` function the training
forward uses (`koopman_lm/modules/kernels/chunk_stats.py`: β-gated,
strictly-causal, exclusive-prefix sufficient statistics) and forms each chunk's
operator with the same `_whiten_M` / `_spec_w` helpers `ska_core` uses
(`koopman_lm/modules/kernels/lin_alg.py`):
`A_eff = γ · α · (L⁻¹ M L⁻ᵀ)`, with `α = 1/max(σ_max, 1)`. So every operator
measured is one a real query sees, at the model's chunk level.

**γ is part of the operator now.** The forward computes
`Y = ska_core(...) · γ^K`, i.e. each of the K filter steps effectively applies
`γ·α·W`. In the phase1 old layout γ was a fixed 1.0 buffer so the diagnostics
ignored it; the refactored `SKAModule` supports learnable and sigmoid-squashed
γ regimes (see below), so the diagnosed radius/gap now include the resolved γ.
With the fixed default `γ = 1.0` this is numerically identical to phase1.

**η/γ regimes.** The refactored module resolves its scales through
`_resolve_eta()` / `_resolve_gamma()`, each with three regimes:
- *fixed*: registered buffer, constant (the 440M default; γ=1.0);
- *learnable*: plain `nn.Parameter` (γ optionally hard-clamped via
  `gamma_clamp`);
- *squash* (echo_jax.py parity): raw parameter smoothly bounded to
  `eta_bounds` / `gamma_bounds` via sigmoid (in this regime there is no `.eta`
  attribute, only `eta_raw`).

The diagnostics always report the **resolved** values as the `eta` / `gamma`
scalars, and use the resolved η for the `gate_mag` fallback when LayerScale is
disabled.

**`exact_intrachunk` note.** When the forward runs with per-token exact stats
(`ska_exact_intrachunk=True`), `collect_diagnostics` still reports the
CHUNK-level operators (same β-gated statistics family) as the bounded-cost
health view.

---

## 2. `SKAHealthMonitor`  (`koopman_lm/training/diagnostics.py`)

Wraps `collect_diagnostics` for use during training via forward hooks. Cheap
when inactive (one boolean check per block).

```python
from koopman_lm.training.diagnostics import SKAHealthMonitor

monitor = SKAHealthMonitor(raw_model)        # attach hooks to model.seq_layers
...
with monitor.capture():                      # activate for ONE forward
    raw_model(input_ids=ids)
metrics = monitor.collect()                  # wandb-ready dict (one CPU sync)
wandb.log(metrics, step=step)
```

**Constructor**
```python
SKAHealthMonitor(model, ska_cls=None, mamba_cls=None,
                 prefix="ska", exclude_first_chunk=True, healthy_band=(0.3, 0.95))
```
- `model` — the **raw** model (before DDP wrap) exposing `.seq_layers`.
- `ska_cls` / `mamba_cls` — block classes used to classify layers (default:
  `koopman_lm.models.koopman_lm.SKABlock` / `Mamba2Block`, imported lazily so
  baseline models can pass their own classes). Any non-SKA seq block is
  bucketed as the Mamba baseline for the residual ratio.
- `exclude_first_chunk` — drop the history-less chunk 0 from summaries (default on).
- `healthy_band` — `(lo, hi)` for the `frac_healthy` fraction.

**`capture()`** — context manager that sets `.active = True` for the enclosed
forward; the hooks then record each block's residual delta and, for SKA blocks,
call `collect_diagnostics` on the layer's normed input (`block.norm(x)`).
Inactive otherwise. Note the residual delta of an `SKABlock` includes the
optional parallel short-conv path when `ska_short_conv` is enabled — it is the
block's true residual contribution.

**`collect(wrap_histograms=True)`** — reduces the buffered records into a flat,
wandb-ready dict and clears the buffer. One GPU→CPU sync for all scalars.
wandb is imported **lazily here only**; histograms are wrapped in
`wandb.Histogram` when wandb is importable, else returned as plain lists. There
is no hard wandb dependency at import time.

### Output schema (wandb keys)

Per SKA layer `L{idx}`:

**Scalars**
```
ska/L{idx}/spectral_radius_mean | _max | _min   # applied (clamped) radius, <=1
ska/L{idx}/raw_spectral_radius_mean | _max       # pre-clamp radius (instability lives here)
ska/L{idx}/frac_healthy           # fraction of (chunk,head) applied-radius in [0.3, 0.95]
ska/L{idx}/frac_unstable          # fraction with RAW radius > 1.0
ska/L{idx}/alpha_min              # smallest clamp factor (strongest clamp)
ska/L{idx}/frac_clamped           # fraction of ops that hit the spectral-norm clamp
ska/L{idx}/lambda_min_min
ska/L{idx}/lambda_min_over_ridge  # ~1 ⇒ Cholesky is all regularization
ska/L{idx}/gap_mean | _max
ska/L{idx}/gate_mag
ska/L{idx}/beta_mean
ska/L{idx}/eta                    # resolved eta (new vs phase1)
ska/L{idx}/gamma                  # resolved gamma (new vs phase1)
ska/L{idx}/outproj_norm
ska/L{idx}/residual_delta
```
**Histograms** (full pool + per-head + per-chunk views; chunk 0 excluded)
```
ska/L{idx}/spectral_radius        | _by_head | _by_chunk
ska/L{idx}/raw_spectral_radius    | _by_head | _by_chunk
ska/L{idx}/lambda_min             | _by_head | _by_chunk
ska/L{idx}/gap                    | _by_head | _by_chunk
```
**Global** (only when the model has both SKA and non-SKA seq blocks)
```
ska/residual_ratio                # mean‖Δ_SKA‖ / mean‖Δ_Mamba‖  ( <1 = SKA suppressed )
ska/residual_delta_ska_mean
ska/residual_delta_mamba_mean
```

### `GradFlowMonitor`

Probe for gradient flow into SKA vs non-SKA branches. It records references to
the SKA projection weights (`key_proj`/`query_proj`/`value_proj`/`out_proj`) and
every `nn.Linear` weight of non-SKA seq blocks, then **reads their `.grad`
directly** via `snapshot()` — NOT backward hooks. Call `snapshot()` after the
last microbatch `backward()` and **before** `clip_grad_norm_`/`zero_grad`, so it
measures the ACCUMULATED gradient of the real optimizer step. Under DDP those
grads are already all-reduced when `backward()` returns, so a rank-0 snapshot is
well-defined (no separate probe backward, no reducer desync). `capture()` is
kept as single-backward sugar (snapshots on exit) for tests/profiling.

Metrics emitted:
```
ska/grad_norm_ratio               # mean||grad_SKA|| / mean||grad_Mamba||  (<0.1 = alarm)
ska/grad_norm_ska_mean
ska/grad_norm_mamba_mean
ska/L{idx}/grad_norm
ska/L{idx}/key_projection_grad_rank       # SVs of the key_proj gradient above 1% of sigma_max
ska/L{idx}/key_projection_grad_rank_frac  # rank / total SVs; toward 0 = gradient rank collapse
```
NOTE: `key_projection_grad_rank` is the numerical rank of `grad(key_proj.weight)`
— a gradient-signal-collapse proxy, **not** the SKA input→output Jacobian
`J = η·Bv·L·Aw^K·L⁻¹` (renamed from the old misleading `jacobian_rank`; the true
Jacobian rank/conditioning is a Phase-2 follow-up).

### `profile_overhead(model, batch_fn, monitor, ...)`

Times a plain training step vs a diagnostic step and reports the amortized
overhead at a few cadences, so the "<3% of step time" target can be verified on
real hardware rather than assumed.

```python
stats = profile_overhead(model, batch_fn, monitor, n_iter=10, device="cuda")
# -> {plain_step_s, diag_step_s, diag_extra_s, overhead_every_100, overhead_every_500}
```
`batch_fn()` returns the kwargs dict passed to `model(**kwargs)`.

---

## 3. Training-loop wiring  (`koopman_lm/training/train.py`)

The monitors are attached to the raw model (koopman model type, main rank
only) and run on a cadence. **Opt-in and off by default.** New CLI flags:

```
--diag_enable                 # emit SKA health metrics (default OFF)
--diag_every N                # cadence in steps (default 500; use 100 for 50M)
--diag_grad                   # also track gradient-norm flow (default OFF)
```

On `step % diag_every == 0`:
- `SKAHealthMonitor`: a separate `no_grad` diagnostic forward of the current
  batch under `monitor.capture()`, then `collect()`, a one-line console
  summary, and the metrics merged into `wandb.log`. A standalone diagnostic
  forward (rather than hooking the training forward) keeps it clean of
  gradient-accumulation bookkeeping and `torch.compile` graph breaks; at this
  cadence the extra forward amortizes well under budget.
- `GradFlowMonitor` (only with `--diag_grad`): `snapshot()` reads the
  ACCUMULATED `.grad` of the real training step at the accumulation boundary,
  **before** `clip_grad_norm_` / `zero_grad` — no extra forward+backward. Works
  under DDP (grads are already all-reduced when `backward()` returns; the rank-0
  snapshot sees the reduced gradient), so it is no longer skipped under DDP.

Console lines:
```
[ska-health] step 4000: radius~0.62 gate~3.4e-03 resid_ratio~7.1e-02 lmin/ridge~18.4
[ska-grad] step 4000: grad_norm_ratio~2.3e-01
```

---

## 4. Load-bearing eval — four-mode SKA-zeroing

The four-mode PPL decomposition (full / ska_zeroed / mamba_zeroed / both_zeroed)
lives in `eval_load_bearing(model, ppl_fn)` in
`koopman_lm/evaluation/harness.py` — **one shared zeroing path**: each mode
toggles the block-level `_ablate` passthrough flags through
`KoopmanLM.ablate(zero_ska=..., zero_mamba=...)` (`koopman_lm/models/koopman_lm.py`),
with no duplicated forward-hook helpers anywhere (the tests call the production
`eval_load_bearing` / `ablate` too).

It returns `ppl_{full,ska_zeroed,mamba_zeroed,both_zeroed}` plus
`ska_delta = ppl_ska_zeroed − ppl_full` and `mamba_delta`; a large positive
`ska_delta` means SKA is load-bearing, near zero means the model routes around
it. Exposed as the harness `load_bearing` task (`--tasks load_bearing`); the
legacy `ska_zeroed_delta` key is derived from the same four-mode result.

---

## 5. Test suite — `code-tests/test_diagnostics.py`

CPU-only, no GPU / `mamba_ssm` / wandb required. 26 tests (25 CPU + 1
GPU-gated), selected by the correctness gate:

```bash
python -m pytest code-tests/test_diagnostics.py -m "correctness and not gpu" -q
```

Coverage:
1. **invariants** — return keys (incl. `eta`/`gamma`/`raw_spectral_radius`/`alpha`);
   metrics are full `(B, nc, H)` tensors; finiteness; applied `radius ≤ 1` (α caps
   σmax, γ=1 default), `α ∈ (0,1]`, `applied_radius ≤ raw_radius`, `λmin ≥ ridge`
   (Lemma A.4), `gap ≥ 0`; at init `β ≈ 0.5` and `gate ≈ 1e-4`.
2. **max_batch capping** — the diagnosed batch is capped at `max_batch`.
3. **gate fallbacks** — with LayerScale off, `gate_mag` falls back to the
   resolved `|eta|`, including in the squash regime (no `.eta` attribute).
4. **gamma scaling** — halving a fixed γ exactly halves the diagnosed radius
   (γ is folded into `A_eff`).
5. **persistence tracks radius** — a near-constant key sequence (persistent,
   lag-1 operator ≈ I) yields a higher spectral radius than random keys.
6. **rank-deficiency pins λmin** — exactly rank-1 keys pin `λmin` at the ridge
   floor (`ridge + 1e-4` jitter).
7. **monitor schema / inactive no-op** — full wandb dict schema on a stand-in
   stack (fake Mamba + real `SKABlock`), `full pool == B·(nc−1)·H`, finite
   positive `residual_ratio`; a forward without `capture()` buffers nothing.
8. **real-model attach** — the monitor attaches to an all-SKA `KoopmanLM`
   (constructible on CPU) with default class resolution; no `residual_ratio`
   without a non-SKA baseline bucket.
9. **profiler** — `profile_overhead` runs and returns finite numbers.
10. **four-mode zeroing via `_ablate`** — isolation (zeroing SKA leaves Mamba
    computing and vice versa), flag restoration, four finite/distinct losses,
    and a known-delta synthetic check (SKA gate=1, Mamba~0 ⇒ zeroing SKA
    perturbs the output more).
11. **load-bearing production path** — a real tiny checkpoint is saved, reloaded
    (logit parity), and run through the production `eval_load_bearing` /
    `KoopmanLM.ablate` (not a re-implemented flag); a GPU-gated test covers the
    mixed SKA+Mamba four-mode decomposition.
12. **gradient flow** — schema after a real backward; `snapshot()` reads the
    ACCUMULATED grad (two backwards ⇒ 2× the norm); no-snapshot no-op;
    frozen-SKA (ratio absent); key-projection grad-rank unit + integration.

Because `Mamba2Block` requires `mamba_ssm` (GPU box only), the stacked-model
tests use a fake-Mamba stand-in and the real-model test uses an all-SKA layer
layout.

---

## Semantic adaptations vs the phase1 branch

This section records how the diagnostics differ from the original `origin/phase1`
branch. The `1c4a058` port brought them onto the consolidated `koopman_lm`
package; the `phase1-finalize` pass then fixed the readiness gaps below.

| phase1 (old layout) | now (`phase1-finalize` on `code-refactor`) |
|---|---|
| `koopman_lm/ska.py`, `koopman_lm/diagnostics.py`, `koopman_lm/train_fast.py`, `koopman_lm/model.py` | `koopman_lm/modules/token_mixer/ska.py`, `koopman_lm/training/diagnostics.py`, `koopman_lm/training/train.py`, `koopman_lm/models/koopman_lm.py` |
| `_whiten_M`, `_spec_w` from `koopman_lm.ska_core_torch` | from `koopman_lm.modules.kernels.lin_alg` |
| `A_eff = α·W` (γ fixed at 1.0) | `A_eff = γ·α·W` with γ resolved via `_resolve_gamma()` (fixed / clamped-learnable / squash) |
| spectral radius read off the α-clamped `A_eff` only ⇒ **`>1` alarm could never fire** | also reports **raw** radius (`= radius/α`) + `α`; `frac_unstable` now measured on the raw radius, so instability actually fires |
| `jacobian_rank` (misnamed: it is `rank(grad(key_proj.weight))`) | renamed `key_projection_grad_rank` + honest doc; true SKA Jacobian deferred to Phase 2 |
| `GradFlowMonitor` ran a **separate fwd+bwd** on one microbatch, **skipped under DDP** | `snapshot()` reads the **accumulated** `.grad` before clip; DDP-correct (grads already all-reduced), no extra backward |
| four-mode load-bearing lived only as a two-mode `ska_delta` in the harness / re-implemented `_ablate` in tests | full four-mode `eval_load_bearing` in the harness; tests use the production path + a real checkpoint |
| `gate_mag` fallback reads `self.eta` directly | fallback via `_resolve_eta()` (the squash regime has `eta_raw`, no `.eta`) |
| diagnostics on by default (`--diag_enable` default True) | **off by default** (`--diag_enable` default False) |
| `wandb_smoke.py` dashboard harness | not ported (superseded by `code-tests/test_smoke_e2e.py`) |
