# Phase 1 Reconciliation — `phase1` vs the consolidated tree

Short version: **`origin/phase1` is superseded and can be retired.** The Phase-1
SKA diagnostics now live, finalized, on the consolidated `koopman_lm` package.

## Branch lineage

```
main
 └─ jackshi (2a8ba61 "phase0 complete")   Phase-0 consolidation (one koopman_lm/
                                          package, frozen config + build_config,
                                          eval_harness, repro, pytest+CI). CPU-tested.
     ├─ ati-phase0-refactor (+10)         + YAML configs + Echo-50M Table-4 results
     │                                    + 180M paper config ([8,16]) + slurm.
     └─ code-refactor (+22)               + merged ati + ROLE-BASED reorg
                                          (modules/{token_mixer,channel_mixer,kernels})
                                          + PORTED the Phase-1 diagnostics (1c4a058)
                                          + unified SKABlock.  <-- authoritative base
```

`origin/phase1` (Task 1/2/3 diagnostics) was written on the **pre-consolidation**
`echo-ska-440m/` tree. jackshi's author explicitly flagged: *"phase 1 seems to be
complete, but does not build on phase 0. we should address this."* Commit
`1c4a058` (Jul 7) on `code-refactor` addressed it by porting the diagnostics into
the consolidated package — a **near-superset** of `phase1` (Tasks 1+2+3, +eta/gamma
metrics, more tests, corrected docs, `GradFlowMonitor` attached in `train.py`).

`code-refactor` and `phase1` are divergent (merge-base `5523a39`); nothing on
`phase1` is missing from `code-refactor` except that the runnable four-mode eval
had been reduced to a two-mode `ska_delta` — restored here.

## What `phase1-finalize` changed (on top of `code-refactor`)

The port carried three latent gaps forward; this branch fixes them, in-place:

1. **Spectral-radius instability signal now fires.** The radius was measured only
   on the α-clamped `A_eff` (σ_max ≤ 1 by construction), so the plan's ">1 =
   unstable" alarm was structurally dead. `collect_diagnostics` now also returns
   the **raw** radius (`= radius/α`) and the clamp factor `α`; `frac_unstable` is
   measured on the raw radius, and `alpha_min`/`frac_clamped` expose the clamp.
2. **`jacobian_rank` → `key_projection_grad_rank`.** It was the numerical rank of
   `grad(key_proj.weight)`, not the SKA Jacobian. Renamed + documented honestly;
   the true `J = η·Bv·L·Aw^K·L⁻¹` rank/conditioning is a Phase-2 follow-up.
3. **Gradient capture is now the real accumulated gradient.** `GradFlowMonitor`
   reads `param.grad` directly at the accumulation boundary, **before**
   `clip_grad_norm_` (via `snapshot()`), instead of a separate single-microbatch
   fwd+bwd. It is DDP-correct (grads already all-reduced) and no longer skipped
   under DDP.
4. **Four-mode load-bearing restored + shared path.** `eval_load_bearing` in the
   harness runs full / ska_zeroed / mamba_zeroed / both_zeroed through the single
   `KoopmanLM.ablate` path; the tests now exercise that production path on a real
   (saved+reloaded) checkpoint rather than a re-implemented `_ablate` flag.

Plus: calibration tooling (`calibrate.py`, `--init_only`), SCG smoke/calibration
slurm scripts, a checkpoint load-compat probe, and updated `SKA_DIAGNOSTICS.md`.

## Verified vs pending

- **Verified locally (CPU):** `pytest code-tests/ -m "correctness and not gpu"` →
  all green (incl. the new raw-radius, rename, accumulated-grad, and real-checkpoint
  four-mode tests). Run in a CPU-only torch venv (no `mamba_ssm` needed).
- **Pending on SCG (scripts prepared, one-command submit):** the green GPU
  end-to-end smoke (`scripts/slurm_smoke_scg.sh`) and threshold calibration
  (`scripts/inspect_checkpoint.py` → `scripts/slurm_calibrate_scg.sh`). These need
  the H200 node; they could not run in the dev environment.

## Recommendation

Merge `phase1-finalize` into `code-refactor` (or promote `code-refactor` to the
mainline), run the two SCG jobs, then **delete `origin/phase1`** — it no longer
carries anything unique.
