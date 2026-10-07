# Interaction analysis: 50m-joint-v1

## How to read this, and what it is not

The sampler was ADAPTIVE. Later trials were proposed inside whatever region the sampler believed was good, so cell counts are uneven by construction and the sampled columns are correlated with each other for reasons that have nothing to do with the model. Read cell means WITH their n. These 14 pairs were PRESPECIFIED before any trial ran; nine axes admit 36 pairs, and choosing which to report after looking is how one of 36 comes out looking significant.

14 of 36 possible pairs are reported. All 14 were named before any trial of this study ran -- 7 when this module was written and 7 when the study was commissioned -- so none was chosen after looking at the data. That is the protection prespecification buys and it is the only one: with 14 tables, an uncorrected 'largest observed interaction' is still the largest of 14 draws. Every magnitude below is therefore reported against the measured noise floor rather than against zero.

Five different kinds of number appear below and they are NOT interchangeable:

0. **The noise floor** -- the spread of one configuration across training seeds. Every magnitude below is stated against it, and one smaller than it is labelled `unresolvable` rather than ranked.
1. **Controlled anchor contrasts** -- one factor moved from one reference point. The only causal statements here.
2. **Partial dependence and pairwise response structure** -- descriptive, over trials the sampler chose. Read every cell mean with its n.
3. **PED-ANOVA importance** -- a divergence between two distributions the sampler produced. Informative, not proof; see its own section.
4. **Sampler-induced correlation** -- a diagnostic of where the sampler went. NOT evidence about the model.

## Trial states

- completed: **30**
- pruned: **33**
- failed: **0**
- total recorded: **63**
- of which anchors: **24**

## 0. The noise floor -- read this before any number below

**sigma = 0.0058183** held-out loss, from 4 degree(s) of freedom.

This is what ONE configuration does when only `runtime.seed` changes. It is measured from designated reference replicates and from nothing else -- not from the spread of the study, which is a spread ACROSS configurations and is the quantity being measured.

**Minimum resolvable effect = 0.0164566.**

An effect is called RESOLVED when its magnitude exceeds 2 x sigma x sqrt(2), where sigma is the within-group standard deviation of the reference replicates. The sqrt(2) is there because almost every effect below is a DIFFERENCE of two single-trial measurements, and the difference of two independent draws of sd sigma has sd sigma*sqrt(2) -- comparing a difference against a bare sigma would over-resolve by 41%. This is a CONVENTION and not a p-value: n=5 replicates estimate sigma to about +/-35%, the 14 reported tables are 14 chances for noise to clear any fixed bar, and nothing here corrects for either. Read 'resolved' as 'larger than this study can explain by seed alone', never as 'significant'.

sigma = 0.0058183 is what ONE configuration's held-out loss does when only the training seed changes, pooled over 4 degree(s) of freedom. Two single trials therefore have to differ by more than 0.0164566 before the difference is anything but seed. Compare that number to the effect sizes this study is searching for BEFORE reading any ranking below: if the design effects are smaller, the ranking is a ranking of seeds and no sampler can repair it.

| group | n | seeds | mean | sd | range | trials |
|:--|---:|:--|---:|---:|---:|:--|
| `reference` | 5 | [42, 43, 44, 45, 46] | 2.7896 | 0.005818 | 0.01417 | [0, 1, 2, 3, 4] |

## 0b. Controlled anchor contrasts

The only part of this report that is a CONTROLLED EXPERIMENT rather than an observational summary. Each anchor is a one-factor move from a single reference point -- verified after resolution, so no second axis moved on the way through -- and each delta is against the reference replicate MEAN rather than against one of its seeds.

Read this table before the importance table and before the pairwise tables. It is the only place a difference is attributable to a named factor without an argument about the sampler.

| anchor | trial | objective | delta vs reference | sigmas | verdict |
|:--|---:|---:|---:|---:|:--|
| `lr-0.3x` | 5 | 2.84939 | +0.05979 | 9.4 | RESOLVED |
| `lr-3.3x` | 9 | 2.76329 | -0.0263 | 4.1 | RESOLVED |
| `lr-0.55x` | 6 | 2.81114 | +0.02154 | 3.4 | RESOLVED |
| `lr-2.2x` | 8 | 2.77148 | -0.01812 | 2.8 | RESOLVED |
| `rank-48` | 12 | 2.77417 | -0.01543 | 2.4 | RESOLVED |
| `lr-1.5x` | 7 | 2.7761 | -0.0135 | 2.1 | RESOLVED |
| `rank-32` | 11 | 2.77754 | -0.01206 | 1.9 | unresolvable |
| `wd-0.05` | 20 | 2.80132 | +0.01172 | 1.8 | unresolvable |
| `warmup-0.10` | 23 | 2.77893 | -0.01067 | 1.7 | unresolvable |
| `wd-0.15` | 21 | 2.78291 | -0.00669 | 1.0 | unresolvable |
| `rank-8` | 10 | 2.78316 | -0.006437 | 1.0 | unresolvable |
| `warmup-0.06` | 22 | 2.78356 | -0.006036 | 0.9 | unresolvable |
| `layerscale-20x` | 17 | 2.78427 | -0.005333 | 0.8 | unresolvable |
| `ska-layers-6` | 14 | 2.7849 | -0.004701 | 0.7 | unresolvable |
| `layerscale-5x` | 16 | 2.78516 | -0.004435 | 0.7 | unresolvable |
| `layerscale-0.5x` | 15 | 2.78599 | -0.00361 | 0.6 | unresolvable |
| `beta-head-scalar` | 18 | 2.78621 | -0.003386 | 0.5 | unresolvable |
| `beta-linear` | 19 | 2.79173 | +0.002128 | 0.3 | unresolvable |
| `ska-layers-3` | 13 | 2.79165 | +0.002053 | 0.3 | unresolvable |

`unresolvable` does NOT mean the factor does not matter. It means this study, at 600 steps, cannot tell that move apart from a change of seed -- and more trials would not change that, because the limit is the measurement and not the sample size.

## 0c. Partial dependence / main effects

Each row is a PARTIAL DEPENDENCE table: the mean objective of every completed trial at each level of one axis, marginalising over the others by averaging rather than by holding them fixed. `spread` is max-min of those level means and `variance_share` is the count-weighted between-level variance divided by the total variance of the objective.

This is DESCRIPTIVE, not causal, and the reason is the sampler. It chose which trials exist, so the other axes are not balanced across the levels of this one -- a level the sampler visited mostly alongside a good ridge looks good. It also concentrated its later proposals, so an axis it pinned near one value shows a small spread whether or not it matters; check that axis's own spread in trials.csv before concluding it is unimportant. The one-factor ANCHOR CONTRASTS in section 0b are the controlled version of this question and should be read first; this table is what extends it to the sampled body, at the cost of the control.

**The verdict column here is the weakest of the three that carry one.** A level mean averages over many trials, so its own seed-noise standard error is `sigma/sqrt(n)` rather than `sigma`, and the spread of two such means is correspondingly tighter than the trial-vs-trial threshold it is being compared against. So an axis marked `unresolvable` here may still have a resolvable effect -- read the controlled anchor contrasts in section 0b, which compare like with like. A `RESOLVED` here is safe in the other direction, and `DID NOT VARY` means the axis took one value and was never measured at all.

| axis | variance share | spread of level means | verdict | levels (mean, n) |
|:--|---:|---:|:--|:--|
| `learning_rate` | 0.7506 | 0.06451 | RESOLVED | [0.000804, 0.001788]: 2.8303 (n=2); [0.001788, 0.003977]: 2.7852 (n=23); [0.003977, 0.008844]: 2.7658 (n=5) |
| `ska_rank` | 0.1871 | 0.02097 | RESOLVED | 8: 2.7832 (n=1); 24: 2.7893 (n=21); 32: 2.7684 (n=2); 48: 2.7756 (n=6) |
| `d_state` | 0.1226 | 0.01528 | unresolvable | 24: 2.7875 (n=25); 32: 2.7722 (n=5) |
| `ska_ridge` | 0.0964 | 0.02698 | RESOLVED | [0.007555, 0.01406]: 2.7862 (n=28); [0.01406, 0.02056]: 2.7592 (n=1); [0.02056, 0.02707]: 2.7773 (n=1) |
| `mamba_expand` | 0.0955 | 0.02014 | RESOLVED | 2: 2.7863 (n=28); 3: 2.7662 (n=2) |
| `warmup_ratio` | 0.0655 | 0.01097 | unresolvable | 0.02: 2.7873 (n=23); 0.04: 2.7773 (n=1); 0.06: 2.7781 (n=4); 0.1: 2.7763 (n=2) |
| `weight_decay` | 0.0212 | 0.007798 | unresolvable | 0.05: 2.7781 (n=3); 0.1: 2.7859 (n=26); 0.15: 2.7829 (n=1) |
| `n_ska_layers` | 0.0139 | 0.01237 | unresolvable | 3: 2.7917 (n=1); 4: 2.7852 (n=27); 6: 2.7793 (n=2) |
| `ska_n_heads` | 0.0078 | 0.007986 | unresolvable | 2: 2.7773 (n=1); 4: 2.7852 (n=29) |
| `beta_policy` | 0.0010 | 0.002442 | unresolvable | head_scalar: 2.7862 (n=1); learned: 2.7851 (n=25); linear: 2.7838 (n=4) |
| `ska_layerscale_init` | 0.0000 | 0.0005105 | unresolvable | [0.005, 0.01893]: 2.785 (n=26); [0.01893, 0.07164]: 2.7852 (n=1); [0.07164, 0.2712]: 2.7847 (n=3) |
| `depth_tier` | 0.0000 | 0.0001435 | unresolvable | full: 2.785 (n=28); lean: 2.7848 (n=2) |
| `gamma_value` | 0.0000 | - | DID NOT VARY | 1.0: 2.785 (n=30) |
| `grad_clip` | 0.0000 | - | DID NOT VARY | 1.0: 2.785 (n=30) |
| `norm_clip_multiplier` | 0.0000 | - | DID NOT VARY | 1.0: 2.785 (n=30) |
| `placement` | 0.0000 | - | DID NOT VARY | late: 2.785 (n=30) |
| `ska_power_K` | 0.0000 | - | DID NOT VARY | 1: 2.785 (n=30) |

## 0d. Conditional effects -- the interaction magnitudes

How much the effect of `left` CHANGES across the levels of `right`. That is what an interaction is, so these are the one-number-per-pair summaries of the question the study was launched to answer. The full cell tables are in section 2.

Both are differences OF differences, so they accumulate noise from four cell means while being tested against a two-single-trials threshold -- which makes this the most OPTIMISTIC comparison in this report. A pair that fails it is certainly unresolvable; a pair that passes it narrowly needs its cell counts read before it is believed.

**Two magnitudes, because an interaction can change a SIZE or a DIRECTION and only one of those invalidates a single recommended value.** `size` is how much the unsigned effect of `left` varies across the levels of `right`. `signed` is the same statistic over the SIGNED contrast (last minus first level of `left`), and it is the one that catches a crossover: if `left` is better at one level of `right` and worse by the same amount at another, the unsigned ranges are identical and `size` is exactly 0.0 while the interaction is maximal. The verdict reads whichever is larger, and `sign change` marks the rows where the contrasts do not all point the same way -- those are the ones where there is no single best value of `left` to recommend.

| left | right | size | signed | sign change | sigmas | verdict | conditional effects (level: effect, signed contrast, n) |
|:--|:--|---:|---:|:--|---:|:--|:--|
| `learning_rate` | `ska_rank` | 0.04163 | 0.04163 | no | 5.1 | RESOLVED | 8: - / - (n=1); 24: 0.05998 / -0.05998 (n=21); 32: 0.01834 / -0.01834 (n=2); 48: 0.02026 / -0.02026 (n=6) |
| `ska_rank` | `n_ska_layers` | 0.009843 | 0.004039 | no | 1.2 | unresolvable | 3: - / - (n=1); 4: 0.02108 / -0.007197 (n=27); 6: 0.01124 / -0.01124 (n=2) |
| `ska_rank` | `ska_ridge` | - | - | no | - | NO FLOOR | [0.007555, 0.01406]: 0.01409 / -0.007916 (n=28); [0.01406, 0.02056]: - / - (n=1); [0.02056, 0.02707]: - / - (n=1) |
| `ska_rank` | `norm_clip_multiplier` | - | - | no | - | NO FLOOR | 1.0: 0.02097 / -0.007581 (n=30) |
| `ska_rank` | `ska_power_K` | - | - | no | - | NO FLOOR | 1: 0.02097 / -0.007581 (n=30) |
| `n_ska_layers` | `ska_layerscale_init` | - | - | no | - | NO FLOOR | [0.005, 0.01893]: 0.01237 / -0.01237 (n=26); [0.01893, 0.07164]: - / - (n=1); [0.07164, 0.2712]: - / - (n=3) |
| `n_ska_layers` | `placement` | - | - | no | - | NO FLOOR | late: 0.01237 / -0.01237 (n=30) |
| `ska_ridge` | `ska_power_K` | - | - | no | - | NO FLOOR | 1: 0.02698 / -0.008916 (n=30) |
| `ska_layerscale_init` | `learning_rate` | - | - | no | - | NO FLOOR | [0.000804, 0.001788]: - / - (n=2); [0.001788, 0.003977]: 0.0006623 / -0.0006623 (n=23); [0.003977, 0.008844]: - / - (n=5) |
| `ska_power_K` | `gamma_value` | - | - | no | - | NO FLOOR | 1.0: - / - (n=30) |
| `ska_ridge` | `ska_layerscale_init` | - | - | no | - | NO FLOOR | [0.005, 0.01893]: 0.02721 / -0.009149 (n=26); [0.01893, 0.07164]: - / - (n=1); [0.07164, 0.2712]: - / - (n=3) |
| `gamma_value` | `norm_clip_multiplier` | - | - | no | - | NO FLOOR | 1.0: - / - (n=30) |
| `learning_rate` | `n_ska_layers` | - | - | no | - | NO FLOOR | 3: - / - (n=1); 4: 0.06451 / -0.06451 (n=27); 6: - / - (n=2) |
| `learning_rate` | `ska_power_K` | - | - | no | - | NO FLOOR | 1: 0.06451 / -0.06451 (n=30) |

## 1. PED-ANOVA importance (a divergence, not a decomposition)

Evaluator: `PedAnovaImportanceEvaluator(evaluate_on_local=True)`.

WHAT THIS MEASURES. PED-ANOVA scores each axis by the DIVERGENCE between its distribution among the study's top-quantile trials and its distribution in a reference set. Two reference sets are reported and they answer different questions:
  * `local` (optuna's default, `evaluate_on_local=True`) compares against the study's OWN remaining trials. Because the sampler was adaptive, that empirical distribution is largely a description of the search path -- so a high local score can mean 'this axis separates good from bad' or 'the sampler moved this axis a lot late on', and the number cannot tell them apart. On this repo's own analysis test fixture a NULL axis outranked a planted main effect under this setting.
  * `global` (`evaluate_on_local=False`) compares against the DECLARED prior instead, so it is not a function of where the sampler went -- at the cost of being a comparison against a region the study may barely have sampled.

Neither is proof about the model. Treat a large gap between the two columns as a warning that the axis's score is search-path dependent. For claims about the MODEL, read the controlled anchor contrasts first (one factor moved from one reference point, with the noise floor beside it) and the partial-dependence table second. An axis with a low score here may be unimportant, or may simply have been pinned near one value by the sampler after the startup window; its spread in trials.csv is what distinguishes those.

| axis | local | global |
|:--|---:|---:|
| `learning_rate` | 0.3521 | 0.1925 |
| `ska_layerscale_init` | 0.3414 | 0.4376 |
| `ska_ridge` | 0.2943 | 0.1025 |
| `d_state` | 0.0030 | 0.0283 |
| `ska_rank` | 0.0025 | 0.0166 |
| `weight_decay` | 0.0017 | 0.0210 |
| `mamba_expand` | 0.0017 | 0.0210 |
| `warmup_ratio` | 0.0012 | 0.0283 |
| `n_ska_layers` | 0.0007 | 0.0403 |
| `ska_n_heads` | 0.0006 | 0.0403 |
| `beta_policy` | 0.0005 | 0.0403 |
| `depth_tier` | 0.0003 | 0.0315 |
| `ska_power_K` | 0.0000 | 0.0000 |
| `placement` | 0.0000 | 0.0000 |
| `norm_clip_multiplier` | 0.0000 | 0.0000 |
| `grad_clip` | 0.0000 | 0.0000 |
| `gamma_value` | 0.0000 | 0.0000 |

## 2. Pairwise response structure (prespecified)

### ska_rank x ska_ridge

| ska_rank \ ska_ridge | [0.007555, 0.01406] | [0.01406, 0.02056] | [0.02056, 0.02707] |
|:--|---:|---:|---:|
| 8 | 2.7832 (n=1) | - | - |
| 24 | 2.7893 (n=21) | - | - |
| 32 | 2.7775 (n=1) | 2.7592 (n=1) | - |
| 48 | 2.7752 (n=5) | - | 2.7773 (n=1) |

Binned: ska_ridge. Cell values are mean objective with the trial count.

### ska_rank x norm_clip_multiplier

**DEGENERATE: norm_clip_multiplier took fewer than two distinct values across the completed trials, so this table cannot show structure. A fixed axis produces this; so does an axis the sampler never varied.**

### ska_rank x ska_power_K

**DEGENERATE: ska_power_K took fewer than two distinct values across the completed trials, so this table cannot show structure. A fixed axis produces this; so does an axis the sampler never varied.**

### n_ska_layers x ska_layerscale_init

| n_ska_layers \ ska_layerscale_init | [0.005, 0.01893] | [0.01893, 0.07164] | [0.07164, 0.2712] |
|:--|---:|---:|---:|
| 3 | 2.7917 (n=1) | - | - |
| 4 | 2.7852 (n=23) | 2.7852 (n=1) | 2.7847 (n=3) |
| 6 | 2.7793 (n=2) | - | - |

Binned: ska_layerscale_init. Cell values are mean objective with the trial count.

### n_ska_layers x placement

**DEGENERATE: placement took fewer than two distinct values across the completed trials, so this table cannot show structure. A fixed axis produces this; so does an axis the sampler never varied.**

### ska_ridge x ska_power_K

**DEGENERATE: ska_power_K took fewer than two distinct values across the completed trials, so this table cannot show structure. A fixed axis produces this; so does an axis the sampler never varied.**

### ska_layerscale_init x learning_rate

| ska_layerscale_init \ learning_rate | [0.000804, 0.001788] | [0.001788, 0.003977] | [0.003977, 0.008844] |
|:--|---:|---:|---:|
| [0.005, 0.01893] | 2.8303 (n=2) | 2.7853 (n=19) | 2.7658 (n=5) |
| [0.01893, 0.07164] | - | 2.7852 (n=1) | - |
| [0.07164, 0.2712] | - | 2.7847 (n=3) | - |

Binned: ska_layerscale_init, learning_rate. Cell values are mean objective with the trial count.

### ska_rank x n_ska_layers

| ska_rank \ n_ska_layers | 3 | 4 | 6 |
|:--|---:|---:|---:|
| 8 | - | 2.7832 (n=1) | - |
| 24 | 2.7917 (n=1) | 2.7894 (n=19) | 2.7849 (n=1) |
| 32 | - | 2.7684 (n=2) | - |
| 48 | - | 2.7760 (n=5) | 2.7737 (n=1) |

### ska_power_K x gamma_value

**DEGENERATE: ska_power_K, gamma_value took fewer than two distinct values across the completed trials, so this table cannot show structure. A fixed axis produces this; so does an axis the sampler never varied.**

### ska_ridge x ska_layerscale_init

| ska_ridge \ ska_layerscale_init | [0.005, 0.01893] | [0.01893, 0.07164] | [0.07164, 0.2712] |
|:--|---:|---:|---:|
| [0.007555, 0.01406] | 2.7864 (n=24) | 2.7852 (n=1) | 2.7847 (n=3) |
| [0.01406, 0.02056] | 2.7592 (n=1) | - | - |
| [0.02056, 0.02707] | 2.7773 (n=1) | - | - |

Binned: ska_ridge, ska_layerscale_init. Cell values are mean objective with the trial count.

### gamma_value x norm_clip_multiplier

**DEGENERATE: gamma_value, norm_clip_multiplier took fewer than two distinct values across the completed trials, so this table cannot show structure. A fixed axis produces this; so does an axis the sampler never varied.**

### learning_rate x ska_rank

| learning_rate \ ska_rank | 8 | 24 | 32 | 48 |
|:--|---:|---:|---:|---:|
| [0.000804, 0.001788] | - | 2.8303 (n=2) | - | - |
| [0.001788, 0.003977] | 2.7832 (n=1) | 2.7878 (n=16) | 2.7775 (n=1) | 2.7790 (n=5) |
| [0.003977, 0.008844] | - | 2.7703 (n=3) | 2.7592 (n=1) | 2.7587 (n=1) |

Binned: learning_rate. Cell values are mean objective with the trial count.

### learning_rate x n_ska_layers

| learning_rate \ n_ska_layers | 3 | 4 | 6 |
|:--|---:|---:|---:|
| [0.000804, 0.001788] | - | 2.8303 (n=2) | - |
| [0.001788, 0.003977] | 2.7917 (n=1) | 2.7855 (n=20) | 2.7793 (n=2) |
| [0.003977, 0.008844] | - | 2.7658 (n=5) | - |

Binned: learning_rate. Cell values are mean objective with the trial count.

### learning_rate x ska_power_K

**DEGENERATE: ska_power_K took fewer than two distinct values across the completed trials, so this table cannot show structure. A fixed axis produces this; so does an axis the sampler never varied.**

## 3. Sampler-induced correlation (DIAGNOSTIC, NOT EVIDENCE)

SAMPLER DIAGNOSTIC, NOT EVIDENCE. These are correlations between sampled columns. The sampler was adaptive, so it concentrated its later proposals in a region it believed was good, and any correlation here is primarily a description of that search path. It is not a finding about the model and must not be reported as one. Its legitimate use is to qualify the importance table: an axis the sampler pinned near one value will show low importance whether or not it matters.

Constant columns (no correlation is defined): `gamma_value`, `grad_clip`, `norm_clip_multiplier`, `ska_power_K`

Excluded (categorical labels have no meaningful r): `beta_policy`, `depth_tier`, `placement`

| a | b | r | n |
|:--|:--|---:|---:|
| `ska_n_heads` | `ska_ridge` | -0.944 | 30 |
| `d_state` | `ska_rank` | +0.698 | 30 |
| `d_state` | `mamba_expand` | +0.598 | 30 |
| `d_state` | `warmup_ratio` | +0.560 | 30 |
| `mamba_expand` | `ska_rank` | +0.501 | 30 |
| `d_state` | `ska_layerscale_init` | +0.487 | 30 |
| `mamba_expand` | `n_ska_layers` | +0.447 | 30 |
| `ska_rank` | `warmup_ratio` | +0.427 | 30 |
| `d_state` | `weight_decay` | -0.415 | 30 |
| `ska_layerscale_init` | `ska_rank` | +0.395 | 30 |
| `warmup_ratio` | `weight_decay` | -0.394 | 30 |
| `ska_rank` | `ska_ridge` | +0.353 | 30 |
| `ska_n_heads` | `ska_rank` | -0.348 | 30 |
| `n_ska_layers` | `warmup_ratio` | +0.340 | 30 |
| `mamba_expand` | `warmup_ratio` | +0.334 | 30 |

## 4. Loss / parameter-count Pareto front

Computed POST HOC, which is why the study is single-objective: optuna cannot prune a multi-objective study at all, and pruning is what makes the study affordable. This is also where parameter count is meant to be traded against loss -- a scalar `parameter_penalty` bakes one exchange rate into every subsequent proposal, unrecoverably.

| params | objective | trial | anchor |
|---:|---:|---:|:--|
| 49,992,032 | 2.78485 | 25 |  |
| 51,501,712 | 2.78316 | 10 | rank-8 |
| 51,695,240 | 2.77726 | 59 |  |
| 51,698,320 | 2.76329 | 9 | lr-3.3x |
| 51,908,656 | 2.75920 | 45 |  |
| 52,574,968 | 2.75870 | 50 |  |

## 4b. Loss / throughput Pareto front

The other half of the cost question, and until `tokens_per_sec` was promoted onto the trial it could not be computed at all -- the number was measured in every `quick_eval.json` and reached neither `user_attrs` nor `trials.csv`.

Read WITH `per_device_batch_size`: a trial that descended the OOM ladder was measured at a smaller microbatch than its neighbours, so its throughput is not comparable to theirs. That caveat is why the loss/throughput exchange rate stays the reader's rather than becoming a `throughput_penalty` baked into every proposal.

A worker process's FIRST trial (`attr_worker_trial_ordinal == 0`) is excluded from the throughput front. Measured, job 445689: four trials on the proxy base at 600 steps reported 17,818 / 111,416 / 104,760 / 112,685 tok/s, and the 6x outlier was trial 0. Those four configs differ only in `ska_layerscale_init` -- one scalar multiply on a gate -- so nothing architectural can produce a 6x throughput gap. It is first-trial CUDA context creation and kernel autotuning, and `tokens_per_sec` is measured during the eval pass, early enough in the process's life to still be inside that. At `concurrent_trials: 8` this is EIGHT poisoned points, not one, because every worker process pays its own warmup. The trials are NOT dropped from anything else -- their loss is a real measurement and only their throughput is suspect -- and `throughput_pareto(exclude_warmup=False)` returns them for anyone who wants to check the claim or who is on hardware where it does not hold. A journal written before this column existed has no ordinals and nothing is excluded, rather than everything.

Of 30 completed trial(s): 30 carry a throughput measurement, 0 do not, and 15 were excluded as a worker's first trial (trials [0, 1, 2, 3, 4, 5, 6, 7, 18, 19, 20, 21, 22, 23, 25]).

| tok/s | objective | trial | pdbs | peak GiB | params | anchor |
|---:|---:|---:|---:|---:|---:|:--|
| 138,159 | 2.77148 | 8 | 12 | 2.85 | 51,698,320 | lr-2.2x |
| 106,078 | 2.76329 | 9 | 12 | 2.85 | 51,698,320 | lr-3.3x |
| 101,849 | 2.75920 | 45 | 12 | 2.85 | 51,908,656 |  |
| 84,740 | 2.75870 | 50 | 12 | 3.56 | 52,574,968 |  |

## 5. Shortlist for confirmation (NOT a winner)

A SHORTLIST FOR CONFIRMATION, not a ranking and not a result. Each row answers a different question, and rows whose `resolved_vs_reference` is false lead the reference by less than this study can measure -- they are on the list because their SLOT matters, not because they won. Read `cannot_establish` before using any of them.

### `best_loss`

Lowest held-out loss. The obvious candidate and the most likely to be luck: it is the minimum of ~230 draws, so it carries the largest selection bias of any row here. Confirm it, do not believe it.

- trial **50**
- objective **2.7587**, lead over reference +0.0309 -> **RESOLVED BETTER**
- run_id `8e035888`, seed 42, params 52,574,968, 8.474e+04 tok/s, 3.56 GiB peak
- SKA indices [5, 8, 10, 12], rank 48, K 1, ridge 0.007555, lr 0.00483

### `best_cost_adjusted`

Best loss among trials on the loss/throughput Pareto front, i.e. the best config that is not also the slowest. Kept separate from best_loss because the objective is pure loss on purpose -- a `throughput_penalty` would bake one exchange rate into every later proposal, unrecoverably.

- trial **50**
- objective **2.7587**, lead over reference +0.0309 -> **RESOLVED BETTER**
- run_id `8e035888`, seed 42, params 52,574,968, 8.474e+04 tok/s, 3.56 GiB peak
- SKA indices [5, 8, 10, 12], rank 48, K 1, ridge 0.007555, lr 0.00483

### `best_k1`

Best config at ska_power_K=1. K is the axis most likely to change the SIGN of another axis's effect, so a confirmation study needs the best candidate at EACH K rather than whichever K happened to win here.

- trial **50**
- objective **2.7587**, lead over reference +0.0309 -> **RESOLVED BETTER**
- run_id `8e035888`, seed 42, params 52,574,968, 8.474e+04 tok/s, 3.56 GiB peak
- SKA indices [5, 8, 10, 12], rank 48, K 1, ridge 0.007555, lr 0.00483

### `best_k2`

Best config at ska_power_K=2, for the same reason.

**Unfilled**: no completed trial ran at ska_power_K=2

### `best_low_capacity`

Best config in the bottom half of the parameter range. If it is within the noise floor of the best overall, the capacity axis did not earn its parameters and the confirmation study should be cheaper.

- trial **9** (anchor `lr-3.3x`)
- objective **2.76329**, lead over reference +0.0263 -> **RESOLVED BETTER**
- run_id `9c6d2caf`, seed 42, params 51,698,320, 1.061e+05 tok/s, 2.85 GiB peak
- SKA indices [6, 10, 13, 15], rank 24, K 1, ridge 0.01, lr 0.008844

### `interaction_probe`

A config chosen to TEST the largest resolved interaction rather than to win: it sits at the corner that interaction predicts is good, which is the only way a confirmation run can falsify it. If the interaction is an artefact of adaptive sampling, this is the row that says so.

- trial **9** (anchor `lr-3.3x`)
- objective **2.76329**, lead over reference +0.0263 -> **RESOLVED BETTER**
- run_id `9c6d2caf`, seed 42, params 51,698,320, 1.061e+05 tok/s, 2.85 GiB peak
- SKA indices [6, 10, 13, 15], rank 24, K 1, ridge 0.01, lr 0.008844
- tests learning_rate x ska_rank (magnitude 0.041634, 5.1 sigma of a single-trial difference) at its best-performing cell [0.003977, 0.008844] x 24

## 6. What this study CANNOT establish

Written down rather than left to judgement, because the failure mode is a reader taking the shortlist's first row as a result.

- **Which configuration is best.** 600 steps at 25.35M parameters is a SCREEN. Rankings at 600 steps and at convergence differ systematically, not randomly: a config that warms up fast is rewarded here whether or not it ends better, which is why `prune_after_step` is 450 of 600 rather than something cheaper. The output is a shortlist for confirmation and there is no defensible single winner in it.

- **That anything here transfers to 50m.** The proxy keeps 50m's depth (17) and SKA placement ([3,7,11,15]) and narrows d_model from 384 to 256, so every effect measured here is measured at 2/3 the width. Effects that interact with width -- which includes the norm-clip multiplier and the layerscale, both of which act on per-head quantities -- can change sign between the two. `d_state` is 48 here against 50m's 64, which is a SECOND unresolved difference and is flagged as an open question rather than quietly treated as immaterial.

- **How much SKA earns its place, beyond that it does.** This one is PARTLY settled and the correction is worth recording. An earlier 4m-geometry smoke study measured an ablation delta of 1.17e-4 and raised the worry that this study might be measuring SKA's hyperparameters without establishing that the branch does anything. Job 445689 settles the sign: on THIS base at 600 steps the delta is 1.17e-2 to 7.48e-2 depending on the layerscale, a hundred times larger, and positive throughout -- removing SKA makes the model worse. The 4m number was a scale artefact. What remains unestablished is the MAGNITUDE at any scale that matters: 25.35M parameters and 600 steps is not 50m and it is not convergence, and an ablation delta measured on a model that has barely left its warmup transient is not the delta at convergence. Compare the ska_delta column to the noise floor directly, and do not read its size as a claim about a trained model.

- **Any effect smaller than the measured noise floor.** Not 'probably not real' -- unmeasurable by this study, at any trial count, because it is inside what one configuration does to itself when the seed changes. More trials narrow the sampler's search, not this.

- **Non-determinism, or seed sensitivity at other configurations.** The base spec sets `deterministic: true` and the replicates are all at ONE configuration. So the floor is a deterministic-mode floor at the reference point; a non-deterministic run spreads more, and a badly-conditioned corner of the space (rank 8 with a light ridge, say) plausibly spreads more still. Applying one floor to the whole space is an assumption, and it is the optimistic one.

- **Statistical significance of anything.** Fourteen prespecified pairwise tables, nine axes, one seed per sampled trial, and cell counts that are uneven BY CONSTRUCTION because the sampler was adaptive. 'Resolved' here means 'larger than seed noise', with no multiplicity correction and no model of the sampling process. It is a filter for what to confirm, not a test.

- **Interactions among the fixed axes, or between them and anything.** `weight_decay`, `warmup_ratio` and `grad_clip` are singletons in this study, so their columns are constant and every table over them is degenerate by design, not by accident.

