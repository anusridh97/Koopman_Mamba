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

## Which swept axis contributed what (`figures/fig5–7`)

**Learning rate dominates.** PedANOVA importance 0.686 global / 0.589 local — larger
than every architecture axis combined. Inside the 256-trial random startup, LR alone
accounts for **70.9%** of the objective variance.

**The marginals in `main_effects.csv` are confounded, and that is not a footnote.**
TPE spent 715 of 1,041 completed trials on `ska_rank=32` and 868 on the top LR bin,
so a raw level mean mixes "this level is good" with "TPE co-selected it alongside a
good LR". Splitting the trials makes the size of that confound visible:

| axis | var share, random startup (n=186) | var share, good-LR band (n=868) |
|---|---|---|
| mamba_expand | 2.7% | **40.2%** |
| ska_rank | 2.3% | **35.4%** |
| beta_policy | 2.7% | **24.7%** |
| ska_power_K | 0.8% | **20.8%** |
| warmup_ratio | 2.1% | 15.1% |
| n_ska_layers | 1.4% | 10.9% |
| depth_tier | 0.2% | 8.0% |
| d_state | 2.6% | 5.9% |

**Read the left column as "cannot resolve", NOT as "no effect".** The random phase is
underpowered: per-level SE is ~0.019 at n≈46, so a two-level gap needs >0.054 to
register, and the observed spreads there are 0.011–0.056 — mostly inside the bar.

**This is the joint search paying for itself.** The study asked "which architecture
wins, and does the answer depend on the optimizer?" The answer to the second half is
yes: at uncontrolled LR no architecture axis is rankable, and once LR is in its good
band `mamba_expand` and `ska_rank` become the two largest terms in the model. An
architecture study with LR pinned at the wrong value would have concluded nothing;
one with LR pinned at the right value would have gotten the ranking without knowing
it depended on that choice.

Caveat: the good-LR band is itself TPE-selected, so residual confounding remains —
it is a controlled comparison, not a randomised one. The clean fix is a small
confirmation grid at fixed LR, which is cheap and not yet run.

Ordering inside the good-LR band (`fig7`, every gap ≥ 4× the 0.0040 resolvable floor):
`mamba_expand` 3 > 2 > 1 · `ska_rank` 32 > 24 > 16 > 8 · `beta_policy` linear >
learned > head_scalar > one · `n_ska_layers` 6 > 4 ≈ 3 > 2.


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
    figures/fig5_axis_importance.png          PedANOVA importance, local vs global
    figures/fig6_conditional_variance.png     variance share: random startup vs good-LR band
    figures/fig7_level_means_controlled.png   level means inside the good-LR band
    data/importances.csv  main_effects.csv  conditional_effects.csv
    data/sampler_correlations.csv  macro_pairwise.json  interactions.md
    make_figures.py
