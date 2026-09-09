# Echo architecture+optimizer sweeps — findings

Four completed searches on one protocol: parameter count held in a band,
~65 tokens/parameter, FineWeb-Edu (held-out val disjoint by document range),
seq 2048, effective batch 96, BF16, and a **measured seed-noise floor at every
rung** that every claim below is tested against.

## Results

| rung | N (total) | tokens | best loss | **best ppl** | vs untuned | σ (seeds) | trials |
|---|---|---|---|---|---|---|---|
| 3M | 2,991,928 | 210 M | 4.1518 | **63.55** | −17.5% ppl | 0.00142 | 1,041 |
| 10M | 10,217,328 | 668 M | 3.4387 | **31.15** | −11.1% | 0.00448 | 177 |
| 50M | 51,698,320 | 3.36 B | 2.7587 | **15.78** | −3.0% | 0.00582 | 30 |
| 180M | 182,665,776 | 10.0 B | 2.4022 | **11.05** | −0.5% | 0.00144 | 14* |

\* 180M ran **anchors only** — 15 controlled one-factor contrasts, no
architecture search. Its "best" is the reference geometry at a better LR.

**The ladder is a clean power law**: `L ∝ N^−0.144`, R² = 0.996 over 61× in N.
(This is the slope *along* our D ≈ 65N ray — it is not Chinchilla's α and mixes
the N and D contributions. See "open items".)

## Best configuration per rung

| axis | 3M | 10M | 50M | 180M* |
|---|---|---|---|---|
| `mamba_expand` | 3 | 3 | 3 | 2 |
| `depth_tier` | full | full | full | full |
| `d_state` | 32 | 24 | 32 | 24 |
| `ska_n_heads` | 4 | 2 | 4 | 4 |
| `ska_rank` | 32 | 48 | 48 | 24 |
| `n_ska_layers` | 6 | 6 | 4 | 4 |
| `beta_policy` | linear | learned | learned | learned |
| `learning_rate` | 5.41e-3 | 7.52e-3 | 4.83e-3 | **2.56e-3** |
| `weight_decay` | 0.15 | 0.1 | 0.1 | 0.1 |
| `warmup_ratio` | 0.04 | 0.06 | 0.02 | 0.02 |

## Relative importance of each axis (share of loss variance)

| axis | 3M | 10M | 50M | 180M |
|---|---|---|---|---|
| **`learning_rate`** | **0.752** | **0.589**\* | **0.751**\* | **0.584**\* |
| `ska_rank` | 0.340 | 0.292\* | 0.187\* | 0.075\* |
| `beta_policy` | 0.210 | 0.114\* | 0.001 | — |
| `warmup_ratio` | 0.154 | 0.257\* | 0.065 | 0.001 |
| `weight_decay` | 0.091 | 0.167\* | 0.021 | 0.029\* |
| `mamba_expand` | 0.287 | 0.052\* | 0.096\* | — |
| `n_ska_layers` | 0.058 | 0.236\* | 0.014 | 0.018\* |
| `d_state` | 0.047 | 0.047\* | 0.123 | — |
| `ska_n_heads` | 0.042 | 0.168\* | 0.008 | — |
| `ska_layerscale_init` | 0.025 | 0.126\* | 0.000 | — |
| `depth_tier` | 0.103 | 0.022\* | 0.000 | — |

\* = resolved against that rung's own noise floor. The 3M study shipped without
a reference group, so nothing there is marked resolved (its σ was measured
afterwards). "—" = not varied at that rung.

## Three findings

**1. SKA is load-bearing at every scale.** `ska_delta` — loss with the SKA path
ablated at eval minus loss as trained — is positive in **1,262 of 1,262
completed trials** across all four rungs. Not one configuration at any scale
would have been better off without SKA. Median 0.144 / 0.197 / 0.147 / 0.062.

The contribution does fade with scale, and this survives the obvious confound
(ablation delta tracks `ska_rank`, and 180M's anchors are mostly rank 24).
Matched at equal rank: 50M → 180M is 0.1396 → 0.0619 at rank 24, and
0.2114 → 0.1523 at rank 48.

**2. The learning rate dominates at every scale; architecture stops mattering.**
LR is the top axis at all four rungs (0.58–0.75 variance share), always 2–8×
the next axis. Meanwhile `ska_rank` — the only architecture axis that resolved
at more than one rung — decays monotonically 0.340 → 0.292 → 0.187 → 0.075, and
at 180M is **flat**: rank 8, 24 and 48 are all within the 0.00316 noise
threshold. `depth_tier` did the same faster (the one resolved architecture axis
at 10M, exactly 0.000 at 50M).

*Practical consequence for scaling up:* a fixed sensible geometry plus a tuned
LR captures nearly all the available gain. Architecture need not be re-searched
at each new size.

**3. Optimal LR follows no rule we could find, and must be re-tuned per scale.**
3M / 10M / 50M each peaked at ~98% of their searched ceiling, so those three are
**lower bounds, not locations**. 180M is the first rung where the optimum is
interior and bracketed on both sides — and it came in at 2.56e-3, **2.9× BELOW**
the 10M value. Two rules were tried and both failed: 1/width predicted 0.0027 at
50M against ≥0.0088 measured, and the assumption that the optimum keeps rising
was refuted at 180M. At 180M the highest LR tested (1.79e-2) diverged to NaN.

Mis-tuning is expensive: a 3× LR error costs 0.03–0.06 loss, which is larger
than the *entire* tuning gain available at 50M or 180M.

## Open items

- **Chinchilla α and β are not yet identifiable.** All four rungs sit on the
  D ≈ 65N ray by design, so in (log N, log D) they are 94% collinear. Five free
  parameters against four points: profiling shows **198 distinct (α, β) pairs
  fit these points equally well**, α anywhere in 0.05–1.00. A 19-cell × 3-LR
  off-ray grid (672 GPU-h) is 39/57 cells trained; it is stalled because the
  `batch` Slurm partition has been administratively **down** for 13+ hours.
- **No tuned Mamba-2 comparison exists.** At 3M the SKA and Mamba-2-only arms
  **tie** when both are untuned. `ska_delta` shows SKA is load-bearing *within*
  this architecture — a different and weaker claim than "SKA beats Mamba-2". A
  matched tuned Mamba-2 ladder is ~250–400 GPU-h and is the cheapest way to
  close that gap.
- **180M has no architecture search**, only 15 anchors. `rank-48` being flat
  there is evidence about rank at 180M; it is not evidence that the reference
  geometry is optimal.

Full data, figures and per-rung detail: `results/ladder-summary/`,
`results/3m-joint-v1/`, `results/50m-joint-v1/`.
