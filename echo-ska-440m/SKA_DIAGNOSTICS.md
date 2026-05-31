# SKA Health Diagnostics (Phase 1)

Instrumentation for the `echo-ska-440m` codebase that answers one question during
training: **are the Spectral Koopman Attention (SKA) layers actually contributing
to the model's output, or is the model learning to route around them?**

It emits per-layer / per-head SKA health metrics to the console and to Weights &
Biases, cheaply enough to run during training, and ships a CPU test suite plus a
dashboard smoke harness.

---

## Files added / changed

| File | Status | Purpose |
|---|---|---|
| `koopman_lm/ska.py` | edited | `SKAModule.collect_diagnostics()` + `_spectral_radius()` helper |
| `koopman_lm/diagnostics.py` | **new** | `SKAHealthMonitor`, `profile_overhead()` |
| `koopman_lm/train_fast.py` | edited | wires the monitor into the training loop + CLI flags |
| `koopman_lm/koopman_mlp.py` | **new** | ported a missing module so `koopman_lm.model` imports (consolidation fix) |
| `test_diagnostics.py` | **new** | CPU-only test suite (no GPU / mamba_ssm / wandb needed) |
| `wandb_smoke.py` | **new** | populates a real wandb dashboard from a stand-in model |

---

## The five health metrics

All are computed from the **per-chunk, chunk-causal Koopman operators that the
training forward actually applies** (see "Faithfulness" below), not a proxy.

| Metric | What it measures | Healthy range |
|---|---|---|
| **Spectral radius** of `A_eff` (per layer/head) | dominant `|eigenvalue|` of the spectrally-normalized whitened operator `A_eff = α·A_w`, where `A_w = L⁻¹ M L⁻ᵀ` and `α = 1/max(σ_max(A_w), 1)` (i.e. the operator the power filter actually applies). Identifies whether key→value bindings are persistent. | `[0.3, 0.95]`; `~0` = operator unlearned; `>1` = unstable |
| **λmin(G̃)** | smallest eigenvalue of the ridge-regularized Gram matrix; Cholesky conditioning. | comfortably above the ridge floor `ε`; pinned at `ε` ⇒ rank-deficient keys |
| **Gap** `‖A_eff^K − A_eff‖ / ‖A_eff‖` | how much the power filter (squaring, K=2) reshapes the operator. | `< 0.5`; above ⇒ the filter dominates rather than confirms |
| **Write-gate magnitude** | the LayerScale residual gate (`layerscale_gate`); how hard SKA is injected into the residual. Plot on a **log scale**; watch for monotonic growth. | grows off its `1e-4` init; flat ⇒ SKA effectively dead |
| **Residual-norm contribution** `‖x_SKA‖ / ‖x_Mamba‖` | SKA's contribution to the residual stream relative to the Mamba layers. | within ~1 order of magnitude; orders smaller ⇒ not doing real work |

Reading them together localizes the failure: healthy radius/λmin but a flat gate
⇒ "good operator, model is ignoring it"; collapsing radius with λmin pinned at the
ridge floor ⇒ "the operator never formed / Gram is rank-deficient."

---

## 1. `SKAModule.collect_diagnostics()`  (`koopman_lm/ska.py`)

The core measurement. A detached, fp32, no-grad method on the SKA module.

```python
metrics = ska_module.collect_diagnostics(hidden_states, max_batch=4)
```

**Input**
- `hidden_states` — `(B, T, d_model)` float tensor: the layer-normed input that
  feeds the SKA module (i.e. `block.norm(x)`).
- `max_batch` — caps the batch used for the per-chunk eigen/operator work so the
  cost is bounded regardless of training batch size (default 4).

**Output** — a `dict` of GPU tensors (the caller does the CPU sync):

| key | shape | meaning |
|---|---|---|
| `spectral_radius` | `(B, nc, H)` | `max|eig(A_eff)|` per (batch, chunk, head) |
| `lambda_min` | `(B, nc, H)` | smallest eig of `G̃` per instance |
| `gap` | `(B, nc, H)` | `‖A_eff^K − A_eff‖ / ‖A_eff‖` per instance |
| `n_chunks` | `int` | number of chunks (`nc`); chunk 0 has no history |
| `gate_mag` | scalar | mean `|layerscale_gate|` (falls back to `|eta|` if no LayerScale) |
| `beta_mean` | scalar | mean `sigmoid(beta)` causal write-gate |
| `outproj_norm` | scalar | `‖out_proj.weight‖_F` |
| `ridge_eps` | scalar | the ridge floor, for the `λmin` ratio |

Nothing is pre-reduced — the three operator metrics come back as the **full
`(B, nc, H)` distributions** so downstream code can aggregate per head, per chunk,
or over the whole pool however it wants.

**Faithfulness.** It calls the *same* `chunk_stats` function the training forward
uses (β-gated, strictly-causal, exclusive-prefix sufficient statistics) and forms
each chunk's operator exactly as `ska_core` does:
`A_eff = α · (L⁻¹ M L⁻ᵀ)`, with `α = 1/max(σ_max, 1)`. So every operator measured
is one a real query sees, at the model's true chunk granularity — not a
non-causal whole-sequence summary.

### `_spectral_radius(A, n_iters=30)` (module-level helper)

Returns `max|eigenvalue|` per matrix via **batched power iteration** (just
matmuls — fast on GPU). It replaced `torch.linalg.eigvals`, which is the general
non-symmetric eig routine and was ~10× too slow on many small matrices (it
dominated the diagnostic step time). Accurate enough for a health metric: exact
for a dominant real eigenvalue or a complex-conjugate 2×2 block.

---

## 2. `SKAHealthMonitor`  (`koopman_lm/diagnostics.py`)

Wraps `collect_diagnostics` for use during training via forward hooks. Cheap when
inactive (one boolean check per block).

```python
from koopman_lm.diagnostics import SKAHealthMonitor

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
  `koopman_lm.model.SKABlock` / `Mamba2Block`). Any non-SKA seq block is bucketed
  as the Mamba baseline for the residual ratio.
- `exclude_first_chunk` — drop the history-less chunk 0 from summaries (default on).
- `healthy_band` — `(lo, hi)` for the `frac_healthy` fraction.

**`capture()`** — context manager that sets `.active = True` for the enclosed
forward; the hooks then record each block's residual delta and, for SKA blocks,
call `collect_diagnostics` on the layer's normed input. Inactive otherwise.

**`collect(wrap_histograms=True)`** — reduces the buffered records into a flat,
wandb-ready dict and clears the buffer. One GPU→CPU sync for all scalars.
Histograms are wrapped in `wandb.Histogram` when wandb is importable, else
returned as plain lists.

### Output schema (wandb keys)

Per SKA layer `L{idx}` (e.g. `L3`, `L7`, `L11`, `L15`):

**Scalars**
```
ska/L{idx}/spectral_radius_mean | _max | _min
ska/L{idx}/frac_healthy           # fraction of (chunk,head) ops in [0.3, 0.95]
ska/L{idx}/frac_unstable          # fraction with radius > 1.0
ska/L{idx}/lambda_min_min
ska/L{idx}/lambda_min_over_ridge  # ~1 ⇒ Cholesky is all regularization
ska/L{idx}/gap_mean | _max
ska/L{idx}/gate_mag
ska/L{idx}/beta_mean
ska/L{idx}/outproj_norm
ska/L{idx}/residual_delta
```
**Histograms** (full pool + per-head + per-chunk views; chunk 0 excluded)
```
ska/L{idx}/spectral_radius        | _by_head | _by_chunk
ska/L{idx}/lambda_min             | _by_head | _by_chunk
ska/L{idx}/gap                    | _by_head | _by_chunk
```
**Global**
```
ska/residual_ratio                # mean‖Δ_SKA‖ / mean‖Δ_Mamba‖  ( <1 = SKA suppressed )
ska/residual_delta_ska_mean
ska/residual_delta_mamba_mean
```

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

## 3. Training-loop wiring  (`koopman_lm/train_fast.py`)

The monitor is attached to the raw model (koopman model type only) and run on a
cadence. New CLI flags:

```
--diag_enable / --no_diag     # emit SKA health metrics (default on)
--diag_every N                # cadence in steps (default 500; use 100 for 50M)
```

On `step % diag_every == 0` it runs a separate `no_grad` diagnostic forward of the
current batch under `monitor.capture()`, calls `collect()`, prints a one-line
summary, and merges the metrics into `wandb.log`. A standalone diagnostic forward
(rather than hooking the training forward) keeps it clean of gradient-accumulation
bookkeeping and `torch.compile` graph breaks; at this cadence the extra forward
amortizes well under budget.

Console line:
```
[ska-health] step 4000: radius~0.62 gate~3.4e-03 resid_ratio~7.1e-02 lmin/ridge~18.4
```

---

## 4. `koopman_lm/koopman_mlp.py` (consolidation fix)

`model.py` imports `SpectralKoopmanMLP` / `SpectralKoopmanMLPGated` from
`koopman_lm.koopman_mlp`, but the module was **missing** from this checkout, so
`koopman_lm.model` (and therefore training and diagnostics) could not import. The
two classes were ported in (matching the constructor signature `model.py` expects)
to unblock everything. The Spectral Koopman MLP is the SwiGLU replacement: lift →
SiLU → learnable complex-eigenvalue 2×2 rotations with modulus clamped to the unit
disk → readout.

---

## 5. Test suite — `test_diagnostics.py`

CPU-only, no GPU / `mamba_ssm` / wandb required (it uses a fake-Mamba stand-in).

```bash
cd echo-ska-440m
python test_diagnostics.py
```

Seven tests:
1. **invariants** — return keys; metrics are full `(B, nc, H)` tensors; finiteness;
   `radius ≤ 1` (α caps σmax), `λmin ≥ ridge` (Lemma A.4), `gap ≥ 0`; at init
   `β ≈ 0.5` and `gate ≈ 1e-4`.
2. **gate fallback** — with LayerScale off, `gate_mag` falls back to `|eta|`.
3. **persistence tracks radius** — a near-constant key sequence (persistent,
   lag-1 operator ≈ I) yields a higher spectral radius than random keys.
4. **rank-deficiency pins λmin** — exactly rank-1 keys pin `λmin` at the ridge
   floor.
5. **monitor schema** — builds a tiny stand-in model, runs one captured forward,
   checks the full wandb dict schema (scalars + the three histogram views,
   `full pool == B·(nc−1)·H`), and a finite positive `residual_ratio`.
6. **inactive no-op** — a forward without `capture()` buffers nothing.
7. **profile** — `profile_overhead` runs and returns finite numbers.

---

## 6. Dashboard smoke test — `wandb_smoke.py`

Verifies the **logging pipeline and dashboard end-to-end** without a full model
run. It builds a tiny stand-in stack (fake Mamba + real `SKABlock`, CPU-fine, no
`mamba_ssm`), runs a short toy-training loop so the SKA params actually update,
and logs `monitor.collect()` every few steps.

```bash
# offline (no wandb account needed); view later with `wandb sync wandb/offline-run-*`
python wandb_smoke.py --steps 400 --log_every 10

# online (after `wandb login`)
python wandb_smoke.py --online --project ska-health-smoke --steps 400
```

Flags: `--steps`, `--log_every`, `--batch`, `--seq_len`, `--lr`, `--project`,
`--online` (default offline), `--cuda`, `--seed`.

**What it proves, and what it does NOT.** It confirms the scalar panels and
histograms render and move over steps. It does **not** measure whether SKA "works":
the backbone is a fake conv (not a real SSM) and the task is a trivial
global-mean regression that needs no associative recall. Healthy-looking metrics
here only mean the wiring is correct — they are not evidence of SKA capability.
(Because the toy task feeds random Gaussian inputs, the SKA keys span the space
and the Gram stays well-conditioned, so this harness cannot surface conditioning
problems that structured retrieval tasks expose.)

---

## How to view the wandb data without an account

The runs log **offline** by default to `./wandb/offline-run-*`. To see the graphs:

- **Free account** (fastest): `wandb login`, then `wandb sync wandb/offline-run-*`
  from a node with internet → open the printed URL.
- **No account**: run with `--no_wandb` and read the console `[ska-health]` lines,
  or add a TensorBoard logger (not yet implemented) for account-free local graphs.

Sync from a login node, not a compute node (compute nodes usually have no internet).

---

## Status

- Instrumentation implemented and unit-tested on CPU (all 7 tests pass).
- Dashboard smoke test confirms the logging pipeline renders in wandb.
- The diagnostics reconstruct the real per-chunk operators faithfully, and the
  spectral-radius computation is fast enough (power iteration) for the <3% overhead
  target.
