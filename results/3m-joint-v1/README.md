# 3M joint architecture + optimizer search — results

Study `3m-joint-v1`, commit `84d2ac9`, Marlowe, 8×H100.
Journal: `study-3m-joint-v1/_studies/3m-joint-v1.9229da2d.c1d3441f/optuna_journal.log`

## Headline

| claim | status |
|---|---|
| The search improved held-out NTP loss 4.3437 → **4.15179** (ppl 76.99 → **63.55**, −17.5%) over the reference config | **supported** — 135× the seed σ measured at that config |
| SKA is load-bearing *inside* a trained SKA model | **supported** — ablation Δ = +0.0845 (FineWeb-Edu val), +0.2174 (WikiText-103); 0/1041 trials have Δ ≤ 0; Pearson r(loss, Δ) = −0.734 |
| SKA is ~10× more seed-stable than Mamba-2-only | **supported** — σ 0.00142 vs 0.01411 / 0.01417 |
| SKA beats an equal-parameter Mamba-2-only model | **NOT supported** — arms tie, no pair resolved at n=3 |
| The 17.5% gain is attributable to SKA | **NOT supported** — the baseline was never tuned; the gain is tuning, measured within one architecture family |

## Search

2,007 terminal trials (target 2,000; overshoot is by design — 8 workers evaluate the
stop condition independently). 1,041 COMPLETE, 810 PRUNED, 156 FAIL.
17 dimensions = 13 shared axes + 4 macro (`mamba_expand`, `depth_tier`, `d_state`,
`ska_n_heads`). Multivariate TPE, 256 random startup draws, MedianPruner from step 400.
Every trial: 1,068 steps × 96 × 2,048 = 209,977,344 tokens (~70 tok/param), fixed
across architectures so parameter count is never confounded with data.

**Best — trial 1794, loss 4.15179, 3,080,356 params, ska_delta 0.2098**

    mamba_expand 3   depth_tier full (n_layers 11)   d_state 32
    ska_rank 32      ska_n_heads 4      n_ska_layers 6     placement late
    beta_policy linear   ska_power_K 1   gamma_value 1.0
    ska_ridge 0.02285    ska_layerscale_init 0.2452   norm_clip_multiplier 0.75
    learning_rate 5.406e-3   warmup_ratio 0.04   weight_decay 0.15   grad_clip 1.0

Loss distribution over completed trials: best 4.1518 · p10 4.1887 · median 4.2321 · worst 4.8086.

## Baselines (iso-parameter, all untuned at lr 4e-3, seeds 42/43/44)

| arm | params | mean | sd |
|---|---|---|---|
| Mamba-2 only, d64 e2 s24 depth14 | 2,974,640 | 4.3613 | 0.01411 |
| Mamba-2 only, d64 e3 s32 depth12 | 3,010,800 | 4.3407 | 0.01417 |
| Mamba-2 + SKA, d64 e2 s24 depth13 | 2,991,928 | 4.3442 | 0.00142 |

Welch: SKA − mamba_best = +0.0035 (t = −0.43); SKA − mamba_ref = +0.0171 (t = +2.09,
df 2, crit 4.30). **None resolved.** Cost: SKA 15.6 min/run vs Mamba-2 ~5.0 — 3.1×.

## Noise floor (new — the study shipped without one)

σ at the reference config = **0.00142** (n=3) → trial-vs-trial resolvable 2σ√2 = 0.0040.
Mamba-2-only arms are ~10× noisier (σ ≈ 0.0141). Measured at ONE configuration; a
badly-conditioned corner plausibly spreads more.

## Reference-run diagnostics (`figures/fig4`)

Write gate 0.01 → 0.164 (16×), ‖Δ SKA‖/‖Δ Mamba‖ 0.113 → 0.266, grad-norm ratio
0.052 → 0.154, spectral radius 0.967 → 0.889 (healthy band). **λmin/ridge decayed
3.79 → 1.02** — pinned at the floor, the rank-deficient-keys signature. `ska_rank`
won at the ceiling of its searched range (32), so that axis may be bound by its limit.

## Failures

156 of 2,007 (7.8%): 102 OOM, 54 Lustre `EDQUOT`. Both are infrastructure, not config.
- **OOM**: next trial starts a median **0.5 s** after the previous ends on the same
  worker, before a 40+ GiB CUDA context is torn down. `wait_for_objective` returns as
  soon as `quick_eval.json` appears, which `train.py` writes *before* exiting.
  `batch_ladder` never fires — it only catches a *synchronous* `submit()` raise.
- **EDQUOT**: bursty per-MDT quota grants near the group's 512K inode ceiling.
  Mitigated by pruning redundant checkpoints (reclaimed 15,616 inodes / 51.9 GiB).

Recovered by re-running one trial per dedicated GPU: **126 of 144 attempted**
(OOM 90/94, quota 36/50). Re-runs reuse the original `run_id` and append to
`attempts.jsonl`. Best recovered 4.1675 — **no re-run beats the study best**, so the
ranking is unchanged. The journal still records them FAIL (optuna cannot un-fail a
trial), so treat `recovered_failures.csv` as a supplement, not part of the search.

## Not done

- No seed replicates of the winning config (best is a single seed).
- Winner never evaluated on WikiText-103; only the reference was (269.93 ppl).
- No **tuned** Mamba-2-only search — the control needed before any SKA-vs-baseline claim.
- MQAR and RULER are at floor on a 3M NTP checkpoint and carry no signal here.

## Files

    data/trials.csv                  2,015 rows: params, state, loss, ska_delta, param_count
    data/recovered_failures.csv      144 re-run attempts
    data/baselines.csv               3 arms × 3 seeds
    data/ska_diagnostics.csv         reference-run health trajectory
    data/reference_harness_*.json    WikiText-103 ppl, MQAR grid, RULER, ska_delta
    figures/fig1_search_progress.png
    figures/fig2_arch_vs_baseline.png
    figures/fig3_ska_delta_vs_loss.png
    figures/fig4_ska_diagnostics.png
    make_figures.py
