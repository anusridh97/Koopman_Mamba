# The ladder, summarised: 3M → 10M → 50M → 180M

Four completed architecture+optimizer searches on one protocol — total
parameters in a band, ~65 tokens per total parameter, FineWeb-Edu, and a
measured seed-noise floor at every rung that every claim is compared against.

| rung | N | D | trials | untuned ref | σ | best found | gain |
|---|---|---|---|---|---|---|---|
| 3M | 2,991,928 | 209,977,344 | 1,041 | 4.34420 | 0.00142 | **4.15179** | 0.1924 (136σ) |
| 10M | 10,217,328 | 668,467,200 | 177 | 3.55655 | 0.00448 | **3.43874** | 0.1178 (26σ) |
| 50M | 51,698,320 | 3,361,996,800 | 30 | 2.78960 | 0.00582 | **2.75870** | 0.0309 (5.3σ) |
| 180M | 182,665,776 | 9,999,876,096 | 14† | 2.40712 | 0.00144 | **2.40218** | 0.0049 (3.4σ) |

† 180M ran **anchors only** — 15 controlled one-factor contrasts, no
architecture search. Its "best" is the `lr-0.55×` anchor, i.e. the reference
geometry at a better learning rate. That distinction matters everywhere below.

---

## The direct answer to "do the rungs tell the same story?"

**Half yes, and the half that fails is the important one.**

### What transfers

**The learning rate dominates at every single scale.** Variance share 0.752 /
0.589 / 0.751 / 0.584 across the four rungs — always the largest axis by a
factor of 2–8 over anything else, and resolved against the noise floor at every
rung that had one. This is the most robust finding in the whole ladder.

**The loss is a clean power law along the ray.** Over 61× in N:

```
untuned reference   L ~ N^-0.1444    R2 = 0.99613
best found          L ~ N^-0.1334    R2 = 0.99579
```

That is a genuinely good fit and it is what makes extrapolation credible at all.
**It is not Chinchilla's α** — it is the slope *along* D = 65N, mixing the N and
D contributions, and it cannot be decomposed from these four points. See the
caveat at the bottom.

**SKA is load-bearing at every scale.** `ska_delta` (loss with the SKA path
ablated at eval, minus loss as trained) is positive in **0 of 1,262 completed
trials** across all four rungs. Not one configuration, at any scale, would have
been better off without SKA.

### What does not transfer

**The optimal learning rate follows no rule we have found** (`fig3`):

| rung | best LR | |
|---|---|---|
| 3M | ≥ 0.00541 | at range ceiling |
| 10M | ≥ 0.00752 | at range ceiling |
| 50M | ≥ 0.00483 | at range ceiling |
| 180M | **0.00256** | **interior, bracketed both sides** |

The first three are **lower bounds, not locations** — each peaked at ~98% of its
searched ceiling. 180M is the first rung where the optimum is bracketed, and it
is **2.9× BELOW** the 10M value. Two rules were tried and both failed: 1/width
predicted 0.0027 at 50M against ≥0.0088 measured, and the assumption that the
optimum keeps rising was refuted at 180M, where it fell.

**No architecture value is shared by all four best configs** (`data/best_configs.csv`):

| axis | 3M | 10M | 50M | 180M† |
|---|---|---|---|---|
| `mamba_expand` | 3 | 3 | 3 | 2 |
| `depth_tier` | full | full | full | full |
| `d_state` | 32 | 24 | 32 | 24 |
| `ska_n_heads` | 4 | 2 | 4 | 4 |
| `ska_rank` | 32 | 48 | 48 | 24 |
| `n_ska_layers` | 6 | 6 | 4 | 4 |
| `beta_policy` | linear | learned | learned | learned |
| `learning_rate` | 0.00541 | 0.00752 | 0.00483 | 0.00256 |

Only `depth_tier: full` and `ska_power_K: 1` are unanimous, and `power_K` was
pinned by hand from 10M onward so it is not evidence.

**But read that table with `fig2` beside it, because it is less damning than it
looks.** Most of those disagreements are on axes that never resolved at any
scale. An unresolved axis's "winner" is noise, so `d_state` reading 32/24/32 is
not four rungs contradicting each other — it is four rungs agreeing that
`d_state` does not matter.

### The real pattern, which is neither of the above

**Architecture importance decays with scale, and the optimizer's does not**
(`fig2`). `ska_rank` is the only architecture axis that resolved at more than one
rung, and its share halves at every step:

```
ska_rank   0.340 -> 0.292 -> 0.187 -> 0.075
```

At 180M it is **flat** against `rank 24` (Δ = −0.00102 against a 0.00316
threshold) after resolving *better* at both 10M and 50M. `depth_tier` did the
same thing faster: the one resolved architecture axis at 10M, and exactly 0.000
at 50M.

`ska_delta` shows the same fade, and it survives the obvious confound. Ablation
delta tracks rank strongly (50M: rank 8 → 0.0119, 24 → 0.1396, 32 → 0.1571,
48 → 0.2114), and 180M's anchors are mostly rank 24 — so a naive median
comparison would be unfair. Matched at the same rank it still declines:

| matched rank | 50M | 180M | |
|---|---|---|---|
| 24 | 0.1396 | 0.0619 | 2.3× lower |
| 48 | 0.2114 | 0.1523 | 1.4× lower |

## What this means for scaling

**The good news, and it is real:** you do not need to re-search architecture at
each new scale. The searchable architecture axes stop paying by 50M–180M, so a
fixed sensible geometry plus a tuned learning rate captures nearly all of the
available gain. That is what makes the next scale cheap.

**The bad news, and it is the thing to plan around:** you *must* re-tune the
learning rate at every scale, you cannot extrapolate it, and getting it wrong is
expensive — 0.03–0.06 loss for a 3× error, which is larger than the entire
tuning gain available at 50M or 180M.

**The caution:** the tuning gain collapses from 0.192 to 0.0049, but so did the
search budget (1,041 → 14 trials), so those two are confounded. The clean
comparison is the 50M/180M pair: 30 searched trials bought 0.031 at 50M, while
at 180M the best of 14 *controlled* anchors beat the reference by only 0.005.
That is consistent with diminishing returns to search at scale, but 180M never
had an architecture search, so it is not proof of one.

## Files

`data/rungs.csv` — one row per rung: N, D, reference loss, σ, best loss, best
LR (with a censoring flag), `ska_delta` median, completed-trial count, and
whether architecture was searched.

`data/axis_importance.csv` — 17 axes × 4 rungs: variance share, spread,
resolved, varied. The source for `fig2`.

`data/best_configs.csv` — the best configuration per rung, side by side.

Figures regenerate with `python make_figures.py` (needs matplotlib; honours
`RESULTS_DIR`). Per-rung detail lives in `results/3m-joint-v1/` and
`results/50m-joint-v1/`.

## What this cannot establish

- **Chinchilla's α or β.** All four rungs sit on the ray D ≈ 65N by design, so
  in (log N, log D) they are **94% collinear** — 0.248 of off-ray leverage
  against a 4.11 range in log N. Five free parameters against four points is
  under-determined, and profiling confirms **198 distinct (α, β) pairs fit these
  four points equally well**, α anywhere in 0.05–1.00. The −0.1444 ray exponent
  above is not α and must not be reported as one. Separating them needs runs off
  the ray, which is what `scripts/make_scaling_grid.py` builds.
- **That SKA beats Mamba-2.** The 3M baselines showed the two arms TIE when both
  are untuned (`results/3m-joint-v1/data/baselines.csv`), and no tuned
  Mamba-2 ladder has been run. `ska_delta` establishes that SKA is load-bearing
  *within* this architecture, which is a different claim.
- **That the 180M numbers reflect a searched optimum.** They are 15 controlled
  contrasts around one geometry. `rank-48` being flat there is evidence about
  rank at 180M; it is not evidence that the reference geometry is best.
