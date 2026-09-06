# 50M joint architecture + optimizer search

Third rung of the scaling ladder (3M → 10M → 50M → 180M). One protocol across
every rung: total parameters held inside a band, ~65 tokens per **total**
parameter, FineWeb-Edu, and a measured seed-noise floor that every claim is
compared against.

| | |
|---|---|
| study | `50m-joint-v1` |
| trials | **63 — 30 completed, 33 pruned, 0 failed** |
| of which anchors | 24, all completed — 5 reference replicates + **19 one-factor contrasts**, pruning-exempt |
| horizon | 17,100 steps × 96 × 2048 = **3,361,996,800 tokens** (65.0 tok/total-param) |
| parameter band | 45.5 M – 54.5 M |
| best | **trial 50, loss 2.75870** |
| untuned reference | 2.78960, **σ = 0.00582** (n = 5 seeds) |
| tuning gain | **0.0309** below the reference |
| hardware | 2 × 8×H100, one trial per GPU, ~44 h wall-clock |

## The headline results

**1. SKA is load-bearing, now at a third scale.** `ska_delta` — the loss when
the SKA path is ablated at eval minus the loss as trained — over all 30
completed trials: min **0.0119**, median **0.1467**, max **0.2259**, and
**0 of 30 ≤ 0**. Ablating SKA has never once helped, at any scale:

| rung | completed trials with `ska_delta` | `ska_delta` ≤ 0 | r(loss, Δ) |
|---|---|---|---|
| 3M | 1,041 | 0 | **−0.734** |
| 10M | 141 | 0 | — |
| 50M | 30 | 0 | **−0.156** |

**The correlation weakens sharply with scale, and that should be reported.** At
3M, trials that leaned harder on SKA were markedly better (r = −0.734). At 50M
the sign is the same but the relationship is weak (r = −0.156, n = 30). So the
defensible claim is the *sign* one — ablating SKA never helps, 0 of 1,212 trials
across three scales — and **not** "more SKA reliance means lower loss," which
holds at 3M and does not survive to 50M. Part of that is range restriction: at
50M the search concentrated `ska_delta` between 0.12 and 0.22, so there is
little spread left to correlate against.

**2. Once the learning rate is right, architecture is flat at 50M.** Against the
trial-vs-trial floor (2σ√2 = 0.0165), only four axes resolve, and the
optimizer dominates:

| axis | variance share | spread | resolved |
|---|---|---|---|
| `learning_rate` | **0.751** | 0.0645 | yes |
| `ska_rank` | 0.187 | 0.0210 | yes |
| `ska_ridge` | 0.096 | 0.0270 | yes |
| `mamba_expand` | 0.096 | 0.0201 | yes |
| `d_state` | 0.123 | 0.0153 | no |
| `warmup_ratio` | 0.066 | 0.0110 | no |
| `weight_decay` | 0.021 | 0.0078 | no |
| `n_ska_layers` | 0.014 | 0.0124 | no |
| `ska_n_heads` | 0.008 | 0.0080 | no |
| `beta_policy` | 0.001 | 0.0024 | no |
| `ska_layerscale_init` | **0.0000** | 0.0005 | no |
| `depth_tier` | **0.0000** | 0.0001 | no |

`depth_tier` is the interesting one: it was the **one** architecture axis that
resolved at 10M (full beat lean by 0.0247 against a 0.0127 floor) and is
completely inert here. That is a scale-dependent effect, not noise.

**3. The learning-rate optimum is still censored — third rung in a row.** The
anchor ladder is monotone decreasing all the way into the top of the searched
range, so every "optimal LR" this project has reported is a **lower bound**, not
a location:

| rung | d_model | range ceiling | measured optimum | |
|---|---|---|---|---|
| 3M | 64 | 5.5e-3 | 5.406e-3 | 98% of ceiling |
| 10M | 128 | 7.5e-3 | 7.52e-3 | **at** the ceiling |
| 50M | 384 | 9.0e-3 | 8.844e-3 | 98% of ceiling |

This also **refutes the 1/width rule** the 50M range was centred on: width
scaling from 10M predicted 0.0027 and the answer was ≥ 0.00884, off by 3.3×.
The 180M rung is the first with two anchors above 9e-3, so it is the first that
can bracket the optimum rather than run into its own ceiling.

## Best configuration (trial 50)

```
loss 2.75870   52,574,968 params   n_layers 15   ska_delta 0.1801   84,740 tok/s

mamba_expand 3    depth_tier full   d_state 32    ska_n_heads 4
ska_rank 48       n_ska_layers 4    placement late    ska_power_K 1
beta_policy learned    ska_ridge 0.007555   ska_layerscale_init 0.010941
learning_rate 0.004830   weight_decay 0.1   warmup_ratio 0.02   grad_clip 1.0
```

Note that trial 50's LR (0.00483) is *interior* to the range while the anchor
ladder peaks at the ceiling. Those are not in conflict — the anchor ladder is
the only clean one-factor LR read; trial 50 differs on five other axes at once,
so it is not evidence about LR.

## Reading order, and one trap

Read **`figures/fig3_anchor_contrasts.png`** first. It is the controlled
experiment: one factor moved from one reference point, with the measured noise
floor drawn behind it.

Then `fig2_axis_contribution.png` — but note that it plots **three** estimators
because they disagree, and the disagreement matters. PED-ANOVA with a local
reference scores `ska_layerscale_init` at 0.34 (0.44 against the declared prior)
while its partial-dependence spread is 0.0005 and **all three of its controlled
anchors are flat**. Local PED-ANOVA compares top-quantile trials against the
study's own remaining trials, so an adaptive sampler's path inflates it. The
repo's own `summary.json` carries this warning; `analysis.py` notes that on the
test fixture a null axis outranked a planted main effect under that setting.
**Cite the anchors, not the importance scores.**

## Files

`data/` — the study's own analysis output, unmodified except `trials.csv`, which
is reshaped to the 3M schema (`objective`→`loss`, `param_`/`attr_` prefixes
stripped) so one loader reads both rungs.

| file | what it holds |
|---|---|
| `trials.csv` | 63 trials, one row each, all swept axes + `ska_delta` + throughput |
| `anchor_contrasts.csv` | the 19 one-factor contrasts, with σ, threshold, resolved |
| `noise_floor.csv` | the 5-seed reference group — the scale for everything else |
| `main_effects.csv` | per-axis variance share and level spread |
| `conditional_effects.csv` | the 14 pre-specified interaction probes |
| `importances.csv` | PED-ANOVA, local and global (read the trap above) |
| `shortlist.csv` | six promotion candidates, never a single winner |
| `summary.json` | everything above plus `cannot_establish` |
| `promotions.yaml` | a materialized 3-seed replicate sweep of the top trials |
| `interactions.md`, `macro_pairwise.json` | full interaction tables |
| `pareto.csv`, `throughput_pareto.csv` | loss vs params, loss vs tok/s |
| `objective_vs_params.csv`, `rank_curve.csv` | the scatter and the best-so-far curve |

Regenerate the figures with `python make_figures.py` (needs matplotlib; honours
`RESULTS_DIR` to point at a scratch copy).

## What this cannot establish

`summary.json` carries the full list. The two that matter most here:

- **That anything transfers to 180M.** This rung constrains the 180M search
  space; it does not predict it. `depth_tier` going from the only resolved
  architecture axis at 10M to completely inert at 50M is the standing
  counterexample.
- **A location for the LR optimum.** It is censored at the ceiling. 8.844e-3 is
  a lower bound.

One further limitation specific to this rung: **rank 48 is the largest rank the
parameter band admits at d_model 384** (56 exceeds 54.5 M over the macro space),
so `ska_rank` winning at 48 is a granularity limit, not a demonstrated optimum.
