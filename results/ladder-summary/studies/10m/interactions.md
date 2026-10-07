# Interaction analysis: 10m-joint-v1

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

- completed: **177**
- pruned: **94**
- failed: **0**
- total recorded: **271**
- of which anchors: **24**

## 0. The noise floor -- read this before any number below

**sigma = 0.00447617** held-out loss, from 4 degree(s) of freedom.

This is what ONE configuration does when only `runtime.seed` changes. It is measured from designated reference replicates and from nothing else -- not from the spread of the study, which is a spread ACROSS configurations and is the quantity being measured.

**Minimum resolvable effect = 0.0126605.**

An effect is called RESOLVED when its magnitude exceeds 2 x sigma x sqrt(2), where sigma is the within-group standard deviation of the reference replicates. The sqrt(2) is there because almost every effect below is a DIFFERENCE of two single-trial measurements, and the difference of two independent draws of sd sigma has sd sigma*sqrt(2) -- comparing a difference against a bare sigma would over-resolve by 41%. This is a CONVENTION and not a p-value: n=5 replicates estimate sigma to about +/-35%, the 14 reported tables are 14 chances for noise to clear any fixed bar, and nothing here corrects for either. Read 'resolved' as 'larger than this study can explain by seed alone', never as 'significant'.

sigma = 0.00447617 is what ONE configuration's held-out loss does when only the training seed changes, pooled over 4 degree(s) of freedom. Two single trials therefore have to differ by more than 0.0126605 before the difference is anything but seed. Compare that number to the effect sizes this study is searching for BEFORE reading any ranking below: if the design effects are smaller, the ranking is a ranking of seeds and no sampler can repair it.

| group | n | seeds | mean | sd | range | trials |
|:--|---:|:--|---:|---:|---:|:--|
| `reference` | 5 | [42, 43, 44, 45, 46] | 3.55655 | 0.004476 | 0.009715 | [0, 1, 2, 3, 4] |

## 0b. Controlled anchor contrasts

The only part of this report that is a CONTROLLED EXPERIMENT rather than an observational summary. Each anchor is a one-factor move from a single reference point -- verified after resolution, so no second axis moved on the way through -- and each delta is against the reference replicate MEAN rather than against one of its seeds.

Read this table before the importance table and before the pairwise tables. It is the only place a difference is attributable to a named factor without an argument about the sampler.

| anchor | trial | objective | delta vs reference | sigmas | verdict |
|:--|---:|---:|---:|---:|:--|
| `lr-0.4x` | 5 | 3.71831 | +0.1618 | 33.0 | RESOLVED |
| `rank-8` | 9 | 3.61609 | +0.05954 | 12.1 | RESOLVED |
| `lr-2.5x` | 8 | 3.51098 | -0.04556 | 9.3 | RESOLVED |
| `lr-0.7x` | 6 | 3.59985 | +0.04331 | 8.8 | RESOLVED |
| `lr-1.5x` | 7 | 3.5261 | -0.03045 | 6.2 | RESOLVED |
| `ska-layers-2` | 12 | 3.58499 | +0.02845 | 5.8 | RESOLVED |
| `rank-48` | 11 | 3.52924 | -0.02731 | 5.6 | RESOLVED |
| `wd-0.0` | 20 | 3.58022 | +0.02368 | 4.8 | RESOLVED |
| `warmup-0.01` | 22 | 3.57601 | +0.01946 | 4.0 | RESOLVED |
| `beta-one` | 19 | 3.57453 | +0.01798 | 3.7 | RESOLVED |
| `rank-32` | 10 | 3.54415 | -0.01239 | 2.5 | RESOLVED |
| `warmup-0.06` | 23 | 3.54602 | -0.01053 | 2.1 | RESOLVED |
| `beta-head-scalar` | 17 | 3.56279 | +0.006245 | 1.3 | unresolvable |
| `ska-layers-6` | 13 | 3.55121 | -0.005334 | 1.1 | unresolvable |
| `layerscale-5x` | 15 | 3.55171 | -0.004833 | 1.0 | unresolvable |
| `layerscale-20x` | 16 | 3.55851 | +0.001965 | 0.4 | unresolvable |
| `beta-linear` | 18 | 3.55543 | -0.001115 | 0.2 | unresolvable |
| `wd-0.15` | 21 | 3.55741 | +0.0008604 | 0.2 | unresolvable |
| `layerscale-0.5x` | 14 | 3.5569 | +0.0003503 | 0.1 | unresolvable |

`unresolvable` does NOT mean the factor does not matter. It means this study, at 600 steps, cannot tell that move apart from a change of seed -- and more trials would not change that, because the limit is the measurement and not the sample size.

## 0c. Partial dependence / main effects

Each row is a PARTIAL DEPENDENCE table: the mean objective of every completed trial at each level of one axis, marginalising over the others by averaging rather than by holding them fixed. `spread` is max-min of those level means and `variance_share` is the count-weighted between-level variance divided by the total variance of the objective.

This is DESCRIPTIVE, not causal, and the reason is the sampler. It chose which trials exist, so the other axes are not balanced across the levels of this one -- a level the sampler visited mostly alongside a good ridge looks good. It also concentrated its later proposals, so an axis it pinned near one value shows a small spread whether or not it matters; check that axis's own spread in trials.csv before concluding it is unimportant. The one-factor ANCHOR CONTRASTS in section 0b are the controlled version of this question and should be read first; this table is what extends it to the sampled body, at the cost of the control.

**The verdict column here is the weakest of the three that carry one.** A level mean averages over many trials, so its own seed-noise standard error is `sigma/sqrt(n)` rather than `sigma`, and the spread of two such means is correspondingly tighter than the trial-vs-trial threshold it is being compared against. So an axis marked `unresolvable` here may still have a resolvable effect -- read the controlled anchor contrasts in section 0b, which compare like with like. A `RESOLVED` here is safe in the other direction, and `DID NOT VARY` means the axis took one value and was never measured at all.

| axis | variance share | spread of level means | verdict | levels (mean, n) |
|:--|---:|---:|:--|:--|
| `learning_rate` | 0.5888 | 0.101 | RESOLVED | [0.0012, 0.00346]: 3.5756 (n=29); [0.00346, 0.005719]: 3.498 (n=60); [0.005719, 0.007979]: 3.4746 (n=88) |
| `ska_rank` | 0.2921 | 0.08105 | RESOLVED | 8: 3.5204 (n=11); 16: 3.5615 (n=4); 24: 3.5402 (n=33); 32: 3.4989 (n=31); 48: 3.4804 (n=98) |
| `warmup_ratio` | 0.2575 | 0.08219 | RESOLVED | 0.01: 3.5649 (n=3); 0.02: 3.5312 (n=53); 0.04: 3.4853 (n=35); 0.06: 3.4827 (n=86) |
| `n_ska_layers` | 0.2360 | 0.0536 | RESOLVED | 2: 3.515 (n=20); 3: 3.503 (n=36); 4: 3.5286 (n=46); 6: 3.475 (n=75) |
| `ska_n_heads` | 0.1681 | 0.0451 | RESOLVED | 1: 3.4888 (n=81); 2: 3.4859 (n=35); 4: 3.531 (n=46); 8: 3.4878 (n=15) |
| `weight_decay` | 0.1671 | 0.05165 | RESOLVED | 0.0: 3.5297 (n=11); 0.05: 3.5209 (n=18); 0.1: 3.5112 (n=72); 0.15: 3.478 (n=76) |
| `ska_ridge` | 0.1592 | 0.04099 | RESOLVED | [0.003359, 0.01205]: 3.5248 (n=60); [0.01205, 0.02074]: 3.4875 (n=69); [0.02074, 0.02943]: 3.4838 (n=48) |
| `ska_layerscale_init` | 0.1265 | 0.03676 | RESOLVED | [0.005, 0.01916]: 3.5199 (n=67); [0.01916, 0.07344]: 3.4899 (n=54); [0.07344, 0.2815]: 3.4831 (n=56) |
| `beta_policy` | 0.1144 | 0.03654 | RESOLVED | head_scalar: 3.48 (n=36); learned: 3.5166 (n=74); linear: 3.4875 (n=55); one: 3.5022 (n=12) |
| `mamba_expand` | 0.0522 | 0.0455 | RESOLVED | 1: 3.5326 (n=4); 2: 3.5061 (n=102); 3: 3.4871 (n=71) |
| `d_state` | 0.0470 | 0.02737 | RESOLVED | 8: 3.5137 (n=18); 16: 3.4887 (n=18); 24: 3.5055 (n=90); 32: 3.4864 (n=51) |
| `depth_tier` | 0.0216 | 0.0206 | RESOLVED | full: 3.4966 (n=155); lean: 3.5172 (n=22) |
| `gamma_value` | 0.0000 | - | DID NOT VARY | 1.0: 3.4991 (n=177) |
| `grad_clip` | 0.0000 | - | DID NOT VARY | 1.0: 3.4991 (n=177) |
| `norm_clip_multiplier` | 0.0000 | - | DID NOT VARY | 1.0: 3.4991 (n=177) |
| `placement` | 0.0000 | - | DID NOT VARY | late: 3.4991 (n=177) |
| `ska_power_K` | 0.0000 | - | DID NOT VARY | 1: 3.4991 (n=177) |

## 0d. Conditional effects -- the interaction magnitudes

How much the effect of `left` CHANGES across the levels of `right`. That is what an interaction is, so these are the one-number-per-pair summaries of the question the study was launched to answer. The full cell tables are in section 2.

Both are differences OF differences, so they accumulate noise from four cell means while being tested against a two-single-trials threshold -- which makes this the most OPTIMISTIC comparison in this report. A pair that fails it is certainly unresolvable; a pair that passes it narrowly needs its cell counts read before it is believed.

**Two magnitudes, because an interaction can change a SIZE or a DIRECTION and only one of those invalidates a single recommended value.** `size` is how much the unsigned effect of `left` varies across the levels of `right`. `signed` is the same statistic over the SIGNED contrast (last minus first level of `left`), and it is the one that catches a crossover: if `left` is better at one level of `right` and worse by the same amount at another, the unsigned ranges are identical and `size` is exactly 0.0 while the interaction is maximal. The verdict reads whichever is larger, and `sign change` marks the rows where the contrasts do not all point the same way -- those are the ones where there is no single best value of `left` to recommend.

| left | right | size | signed | sign change | sigmas | verdict | conditional effects (level: effect, signed contrast, n) |
|:--|:--|---:|---:|:--|---:|:--|:--|
| `ska_rank` | `n_ska_layers` | 0.1452 | 0.1509 | **YES** | 23.8 | RESOLVED | 2: 0.188 / -0.05772 (n=20); 3: 0.1528 / -0.1447 (n=36); 4: 0.04283 / -0.009043 (n=46); 6: 0.05839 / +0.006249 (n=75) |
| `learning_rate` | `ska_rank` | 0.1353 | 0.1353 | no | 21.4 | RESOLVED | 8: 0.1549 / -0.1549 (n=11); 16: 0.1932 / -0.1932 (n=4); 24: 0.09322 / -0.09322 (n=33); 32: 0.08812 / -0.08812 (n=31); 48: 0.05792 / -0.05792 (n=98) |
| `ska_rank` | `ska_ridge` | 0.04017 | 0.1066 | **YES** | 16.8 | RESOLVED | [0.003359, 0.01205]: 0.1066 / -0.1066 (n=60); [0.01205, 0.02074]: 0.1015 / +2.375e-06 (n=69); [0.02074, 0.02943]: 0.06642 / -0.00958 (n=48) |
| `ska_layerscale_init` | `learning_rate` | 0.03435 | 0.0677 | **YES** | 10.7 | RESOLVED | [0.0012, 0.00346]: 0.04996 / +0.04996 (n=29); [0.00346, 0.005719]: 0.01561 / -0.01561 (n=60); [0.005719, 0.007979]: 0.02212 / -0.01775 (n=88) |
| `learning_rate` | `n_ska_layers` | 0.06709 | 0.06709 | no | 10.6 | RESOLVED | 2: 0.1362 / -0.1362 (n=20); 3: 0.1183 / -0.1183 (n=36); 4: 0.09431 / -0.09431 (n=46); 6: 0.06913 / -0.06913 (n=75) |
| `ska_ridge` | `ska_layerscale_init` | 0.05809 | 0.05809 | no | 9.2 | RESOLVED | [0.005, 0.01916]: 0.06603 / -0.06603 (n=67); [0.01916, 0.07344]: 0.02112 / -0.01684 (n=54); [0.07344, 0.2815]: 0.007934 / -0.007934 (n=56) |
| `n_ska_layers` | `ska_layerscale_init` | 0.02429 | 0.05604 | no | 8.9 | RESOLVED | [0.005, 0.01916]: 0.03877 / -0.02499 (n=67); [0.01916, 0.07344]: 0.06306 / -0.004808 (n=54); [0.07344, 0.2815]: 0.06085 / -0.06085 (n=56) |
| `ska_rank` | `norm_clip_multiplier` | - | - | no | - | NO FLOOR | 1.0: 0.08105 / -0.03999 (n=177) |
| `ska_rank` | `ska_power_K` | - | - | no | - | NO FLOOR | 1: 0.08105 / -0.03999 (n=177) |
| `n_ska_layers` | `placement` | - | - | no | - | NO FLOOR | late: 0.0536 / -0.04002 (n=177) |
| `ska_ridge` | `ska_power_K` | - | - | no | - | NO FLOOR | 1: 0.04099 / -0.04099 (n=177) |
| `ska_power_K` | `gamma_value` | - | - | no | - | NO FLOOR | 1.0: - / - (n=177) |
| `gamma_value` | `norm_clip_multiplier` | - | - | no | - | NO FLOOR | 1.0: - / - (n=177) |
| `learning_rate` | `ska_power_K` | - | - | no | - | NO FLOOR | 1: 0.101 / -0.101 (n=177) |

## 1. PED-ANOVA importance (a divergence, not a decomposition)

Evaluator: `PedAnovaImportanceEvaluator(evaluate_on_local=True)`.

WHAT THIS MEASURES. PED-ANOVA scores each axis by the DIVERGENCE between its distribution among the study's top-quantile trials and its distribution in a reference set. Two reference sets are reported and they answer different questions:
  * `local` (optuna's default, `evaluate_on_local=True`) compares against the study's OWN remaining trials. Because the sampler was adaptive, that empirical distribution is largely a description of the search path -- so a high local score can mean 'this axis separates good from bad' or 'the sampler moved this axis a lot late on', and the number cannot tell them apart. On this repo's own analysis test fixture a NULL axis outranked a planted main effect under this setting.
  * `global` (`evaluate_on_local=False`) compares against the DECLARED prior instead, so it is not a function of where the sampler went -- at the cost of being a comparison against a region the study may barely have sampled.

Neither is proof about the model. Treat a large gap between the two columns as a warning that the axis's score is search-path dependent. For claims about the MODEL, read the controlled anchor contrasts first (one factor moved from one reference point, with the noise floor beside it) and the partial-dependence table second. An axis with a low score here may be unimportant, or may simply have been pinned near one value by the sampler after the startup window; its spread in trials.csv is what distinguishes those.

| axis | local | global |
|:--|---:|---:|
| `learning_rate` | 0.5557 | 0.6270 |
| `ska_n_heads` | 0.1049 | 0.0189 |
| `ska_layerscale_init` | 0.0982 | 0.0237 |
| `n_ska_layers` | 0.0645 | 0.0508 |
| `mamba_expand` | 0.0363 | 0.0184 |
| `warmup_ratio` | 0.0342 | 0.0358 |
| `ska_rank` | 0.0310 | 0.0651 |
| `d_state` | 0.0231 | 0.0222 |
| `weight_decay` | 0.0196 | 0.0283 |
| `beta_policy` | 0.0177 | 0.0222 |
| `ska_ridge` | 0.0144 | 0.0600 |
| `depth_tier` | 0.0004 | 0.0276 |
| `ska_power_K` | 0.0000 | 0.0000 |
| `placement` | 0.0000 | 0.0000 |
| `norm_clip_multiplier` | 0.0000 | 0.0000 |
| `grad_clip` | 0.0000 | 0.0000 |
| `gamma_value` | 0.0000 | 0.0000 |

## 2. Pairwise response structure (prespecified)

### ska_rank x ska_ridge

| ska_rank \ ska_ridge | [0.003359, 0.01205] | [0.01205, 0.02074] | [0.02074, 0.02943] |
|:--|---:|---:|---:|
| 8 | 3.5873 (n=4) | 3.4811 (n=6) | 3.4886 (n=1) |
| 16 | - | 3.5826 (n=2) | 3.5404 (n=2) |
| 24 | 3.5649 (n=23) | 3.4872 (n=7) | 3.4739 (n=3) |
| 32 | 3.5366 (n=5) | 3.4986 (n=11) | 3.4865 (n=15) |
| 48 | 3.4808 (n=28) | 3.4811 (n=43) | 3.4790 (n=27) |

Binned: ska_ridge. Cell values are mean objective with the trial count.

### ska_rank x norm_clip_multiplier

**DEGENERATE: norm_clip_multiplier took fewer than two distinct values across the completed trials, so this table cannot show structure. A fixed axis produces this; so does an axis the sampler never varied.**

### ska_rank x ska_power_K

**DEGENERATE: ska_power_K took fewer than two distinct values across the completed trials, so this table cannot show structure. A fixed axis produces this; so does an axis the sampler never varied.**

### n_ska_layers x ska_layerscale_init

| n_ska_layers \ ska_layerscale_init | [0.005, 0.01916] | [0.01916, 0.07344] | [0.07344, 0.2815] |
|:--|---:|---:|---:|
| 2 | 3.5241 (n=9) | 3.4762 (n=5) | 3.5335 (n=6) |
| 3 | 3.4961 (n=16) | 3.5345 (n=10) | 3.4826 (n=10) |
| 4 | 3.5349 (n=34) | 3.5206 (n=7) | 3.4970 (n=5) |
| 6 | 3.4992 (n=8) | 3.4714 (n=32) | 3.4727 (n=35) |

Binned: ska_layerscale_init. Cell values are mean objective with the trial count.

### n_ska_layers x placement

**DEGENERATE: placement took fewer than two distinct values across the completed trials, so this table cannot show structure. A fixed axis produces this; so does an axis the sampler never varied.**

### ska_ridge x ska_power_K

**DEGENERATE: ska_power_K took fewer than two distinct values across the completed trials, so this table cannot show structure. A fixed axis produces this; so does an axis the sampler never varied.**

### ska_layerscale_init x learning_rate

| ska_layerscale_init \ learning_rate | [0.0012, 0.00346] | [0.00346, 0.005719] | [0.005719, 0.007979] |
|:--|---:|---:|---:|
| [0.005, 0.01916] | 3.5689 (n=21) | 3.5040 (n=25) | 3.4898 (n=21) |
| [0.01916, 0.07344] | 3.5847 (n=6) | 3.5009 (n=15) | 3.4677 (n=33) |
| [0.07344, 0.2815] | 3.6189 (n=2) | 3.4884 (n=20) | 3.4720 (n=34) |

Binned: ska_layerscale_init, learning_rate. Cell values are mean objective with the trial count.

### ska_rank x n_ska_layers

| ska_rank \ n_ska_layers | 2 | 3 | 4 | 6 |
|:--|---:|---:|---:|---:|
| 8 | 3.5489 (n=1) | 3.6389 (n=2) | 3.5092 (n=4) | 3.4653 (n=4) |
| 16 | 3.6792 (n=1) | - | 3.5222 (n=3) | - |
| 24 | 3.5850 (n=1) | 3.5241 (n=5) | 3.5429 (n=25) | 3.5237 (n=2) |
| 32 | 3.5593 (n=2) | 3.4861 (n=15) | 3.5273 (n=5) | 3.4910 (n=9) |
| 48 | 3.4912 (n=15) | 3.4942 (n=14) | 3.5001 (n=9) | 3.4716 (n=60) |

### ska_power_K x gamma_value

**DEGENERATE: ska_power_K, gamma_value took fewer than two distinct values across the completed trials, so this table cannot show structure. A fixed axis produces this; so does an axis the sampler never varied.**

### ska_ridge x ska_layerscale_init

| ska_ridge \ ska_layerscale_init | [0.005, 0.01916] | [0.01916, 0.07344] | [0.07344, 0.2815] |
|:--|---:|---:|---:|
| [0.003359, 0.01205] | 3.5553 (n=29) | 3.5027 (n=19) | 3.4862 (n=12) |
| [0.01205, 0.02074] | 3.4955 (n=22) | 3.4815 (n=23) | 3.4857 (n=24) |
| [0.02074, 0.02943] | 3.4893 (n=16) | 3.4858 (n=12) | 3.4782 (n=20) |

Binned: ska_ridge, ska_layerscale_init. Cell values are mean objective with the trial count.

### gamma_value x norm_clip_multiplier

**DEGENERATE: gamma_value, norm_clip_multiplier took fewer than two distinct values across the completed trials, so this table cannot show structure. A fixed axis produces this; so does an axis the sampler never varied.**

### learning_rate x ska_rank

| learning_rate \ ska_rank | 8 | 16 | 24 | 32 | 48 |
|:--|---:|---:|---:|---:|---:|
| [0.0012, 0.00346] | 3.6332 (n=2) | 3.6792 (n=1) | 3.5743 (n=20) | 3.5756 (n=2) | 3.5276 (n=4) |
| [0.00346, 0.005719] | 3.5296 (n=3) | 3.5404 (n=2) | 3.4934 (n=7) | 3.5022 (n=12) | 3.4926 (n=36) |
| [0.005719, 0.007979] | 3.4782 (n=6) | 3.4860 (n=1) | 3.4811 (n=6) | 3.4875 (n=17) | 3.4697 (n=58) |

Binned: learning_rate. Cell values are mean objective with the trial count.

### learning_rate x n_ska_layers

| learning_rate \ n_ska_layers | 2 | 3 | 4 | 6 |
|:--|---:|---:|---:|---:|
| [0.0012, 0.00346] | 3.6321 (n=2) | 3.5970 (n=3) | 3.5709 (n=22) | 3.5387 (n=2) |
| [0.00346, 0.005719] | 3.5096 (n=8) | 3.5023 (n=22) | 3.5028 (n=12) | 3.4845 (n=18) |
| [0.005719, 0.007979] | 3.4959 (n=10) | 3.4787 (n=11) | 3.4766 (n=12) | 3.4695 (n=55) |

Binned: learning_rate. Cell values are mean objective with the trial count.

### learning_rate x ska_power_K

**DEGENERATE: ska_power_K took fewer than two distinct values across the completed trials, so this table cannot show structure. A fixed axis produces this; so does an axis the sampler never varied.**

## 3. Sampler-induced correlation (DIAGNOSTIC, NOT EVIDENCE)

SAMPLER DIAGNOSTIC, NOT EVIDENCE. These are correlations between sampled columns. The sampler was adaptive, so it concentrated its later proposals in a region it believed was good, and any correlation here is primarily a description of that search path. It is not a finding about the model and must not be reported as one. Its legitimate use is to qualify the importance table: an axis the sampler pinned near one value will show low importance whether or not it matters.

Constant columns (no correlation is defined): `gamma_value`, `grad_clip`, `norm_clip_multiplier`, `ska_power_K`

Excluded (categorical labels have no meaningful r): `beta_policy`, `depth_tier`, `placement`

| a | b | r | n |
|:--|:--|---:|---:|
| `learning_rate` | `warmup_ratio` | +0.407 | 177 |
| `n_ska_layers` | `ska_layerscale_init` | +0.323 | 177 |
| `ska_layerscale_init` | `warmup_ratio` | +0.312 | 177 |
| `learning_rate` | `ska_ridge` | +0.286 | 177 |
| `learning_rate` | `ska_n_heads` | -0.275 | 177 |
| `learning_rate` | `ska_rank` | +0.273 | 177 |
| `n_ska_layers` | `warmup_ratio` | +0.262 | 177 |
| `learning_rate` | `n_ska_layers` | +0.262 | 177 |
| `ska_layerscale_init` | `ska_rank` | +0.258 | 177 |
| `ska_n_heads` | `warmup_ratio` | -0.244 | 177 |
| `learning_rate` | `ska_layerscale_init` | +0.243 | 177 |
| `d_state` | `ska_rank` | -0.232 | 177 |
| `n_ska_layers` | `ska_rank` | +0.229 | 177 |
| `ska_ridge` | `warmup_ratio` | +0.211 | 177 |
| `d_state` | `ska_n_heads` | -0.204 | 177 |

## 4. Loss / parameter-count Pareto front

Computed POST HOC, which is why the study is single-objective: optuna cannot prune a multi-objective study at all, and pruning is what makes the study affordable. This is also where parameter count is meant to be traded against loss -- a scalar `parameter_penalty` bakes one exchange rate into every subsequent proposal, unrecoverably.

| params | objective | trial | anchor |
|---:|---:|---:|:--|
| 9,696,840 | 3.62746 | 31 |  |
| 9,765,472 | 3.59043 | 28 |  |
| 9,883,270 | 3.51685 | 89 |  |
| 9,894,006 | 3.48252 | 175 |  |
| 9,969,276 | 3.47800 | 235 |  |
| 10,048,294 | 3.46548 | 224 |  |
| 10,075,746 | 3.45661 | 198 |  |
| 10,101,934 | 3.45342 | 256 |  |
| 10,109,852 | 3.45206 | 165 |  |
| 10,126,236 | 3.45027 | 179 |  |
| 10,201,774 | 3.44994 | 188 |  |
| 10,235,844 | 3.44525 | 259 |  |
| 10,276,276 | 3.43874 | 163 |  |

## 4b. Loss / throughput Pareto front

The other half of the cost question, and until `tokens_per_sec` was promoted onto the trial it could not be computed at all -- the number was measured in every `quick_eval.json` and reached neither `user_attrs` nor `trials.csv`.

Read WITH `per_device_batch_size`: a trial that descended the OOM ladder was measured at a smaller microbatch than its neighbours, so its throughput is not comparable to theirs. That caveat is why the loss/throughput exchange rate stays the reader's rather than becoming a `throughput_penalty` baked into every proposal.

A worker process's FIRST trial (`attr_worker_trial_ordinal == 0`) is excluded from the throughput front. Measured, job 445689: four trials on the proxy base at 600 steps reported 17,818 / 111,416 / 104,760 / 112,685 tok/s, and the 6x outlier was trial 0. Those four configs differ only in `ska_layerscale_init` -- one scalar multiply on a gate -- so nothing architectural can produce a 6x throughput gap. It is first-trial CUDA context creation and kernel autotuning, and `tokens_per_sec` is measured during the eval pass, early enough in the process's life to still be inside that. At `concurrent_trials: 8` this is EIGHT poisoned points, not one, because every worker process pays its own warmup. The trials are NOT dropped from anything else -- their loss is a real measurement and only their throughput is suspect -- and `throughput_pareto(exclude_warmup=False)` returns them for anyone who wants to check the claim or who is on hardware where it does not hold. A journal written before this column existed has no ordinals and nothing is excluded, rather than everything.

Of 177 completed trial(s): 177 carry a throughput measurement, 0 do not, and 16 were excluded as a worker's first trial (trials [0, 1, 2, 3, 4, 5, 6, 7, 109, 110, 111, 112, 113, 114, 115, 116]).

| tok/s | objective | trial | pdbs | peak GiB | params | anchor |
|---:|---:|---:|---:|---:|---:|:--|
| 208,719 | 3.62746 | 31 | 12 | 2.37 | 9,696,840 |  |
| 201,693 | 3.47986 | 189 | 12 | 2.38 | 10,109,852 |  |
| 187,545 | 3.47566 | 121 | 12 | 2.38 | 10,093,595 |  |
| 180,807 | 3.47503 | 156 | 12 | 2.38 | 10,093,211 |  |
| 174,441 | 3.45772 | 206 | 12 | 2.38 | 10,126,236 |  |
| 168,974 | 3.45342 | 256 | 12 | 2.38 | 10,101,934 |  |
| 147,696 | 3.45206 | 165 | 12 | 2.38 | 10,109,852 |  |
| 144,097 | 3.45027 | 179 | 12 | 2.38 | 10,126,236 |  |
| 128,403 | 3.44364 | 215 | 12 | 2.38 | 10,276,276 |  |
| 113,769 | 3.43874 | 163 | 12 | 2.38 | 10,276,276 |  |

## 5. Shortlist for confirmation (NOT a winner)

A SHORTLIST FOR CONFIRMATION, not a ranking and not a result. Each row answers a different question, and rows whose `resolved_vs_reference` is false lead the reference by less than this study can measure -- they are on the list because their SLOT matters, not because they won. Read `cannot_establish` before using any of them.

### `best_loss`

Lowest held-out loss. The obvious candidate and the most likely to be luck: it is the minimum of ~230 draws, so it carries the largest selection bias of any row here. Confirm it, do not believe it.

- trial **163**
- objective **3.43874**, lead over reference +0.1178 -> **RESOLVED BETTER**
- run_id `8f27c0e6`, seed 42, params 10,276,276, 1.138e+05 tok/s, 2.38 GiB peak
- SKA indices [5, 8, 11, 13, 15, 16], rank 48, K 1, ridge 0.01164, lr 0.007522

### `best_cost_adjusted`

Best loss among trials on the loss/throughput Pareto front, i.e. the best config that is not also the slowest. Kept separate from best_loss because the objective is pure loss on purpose -- a `throughput_penalty` would bake one exchange rate into every later proposal, unrecoverably.

- trial **163**
- objective **3.43874**, lead over reference +0.1178 -> **RESOLVED BETTER**
- run_id `8f27c0e6`, seed 42, params 10,276,276, 1.138e+05 tok/s, 2.38 GiB peak
- SKA indices [5, 8, 11, 13, 15, 16], rank 48, K 1, ridge 0.01164, lr 0.007522

### `best_k1`

Best config at ska_power_K=1. K is the axis most likely to change the SIGN of another axis's effect, so a confirmation study needs the best candidate at EACH K rather than whichever K happened to win here.

- trial **163**
- objective **3.43874**, lead over reference +0.1178 -> **RESOLVED BETTER**
- run_id `8f27c0e6`, seed 42, params 10,276,276, 1.138e+05 tok/s, 2.38 GiB peak
- SKA indices [5, 8, 11, 13, 15, 16], rank 48, K 1, ridge 0.01164, lr 0.007522

### `best_k2`

Best config at ska_power_K=2, for the same reason.

**Unfilled**: no completed trial ran at ska_power_K=2

### `best_low_capacity`

Best config in the bottom half of the parameter range. If it is within the noise floor of the best overall, the capacity axis did not earn its parameters and the confirmation study should be cheaper.

- trial **259**
- objective **3.44525**, lead over reference +0.1113 -> **RESOLVED BETTER**
- run_id `5bd75a2b`, seed 42, params 10,235,844, 1.147e+05 tok/s, 2.38 GiB peak
- SKA indices [5, 8, 11, 13, 15, 16], rank 48, K 1, ridge 0.02093, lr 0.007611

### `interaction_probe`

A config chosen to TEST the largest resolved interaction rather than to win: it sits at the corner that interaction predicts is good, which is the only way a confirmation run can falsify it. If the interaction is an artefact of adaptive sampling, this is the row that says so.

- trial **182**
- objective **3.45557**, lead over reference +0.101 -> **RESOLVED BETTER**
- run_id `c0b54c29`, seed 42, params 10,179,998, 1.547e+05 tok/s, 2.38 GiB peak
- SKA indices [5, 8, 11, 13, 15, 16], rank 8, K 1, ridge 0.01192, lr 0.007566
- tests ska_rank x n_ska_layers (magnitude 0.1452, 23.8 sigma of a single-trial difference) at its best-performing cell 8 x 6

## 6. What this study CANNOT establish

Written down rather than left to judgement, because the failure mode is a reader taking the shortlist's first row as a result.

- **Which configuration is best.** 600 steps at 25.35M parameters is a SCREEN. Rankings at 600 steps and at convergence differ systematically, not randomly: a config that warms up fast is rewarded here whether or not it ends better, which is why `prune_after_step` is 450 of 600 rather than something cheaper. The output is a shortlist for confirmation and there is no defensible single winner in it.

- **That anything here transfers to 50m.** The proxy keeps 50m's depth (17) and SKA placement ([3,7,11,15]) and narrows d_model from 384 to 256, so every effect measured here is measured at 2/3 the width. Effects that interact with width -- which includes the norm-clip multiplier and the layerscale, both of which act on per-head quantities -- can change sign between the two. `d_state` is 48 here against 50m's 64, which is a SECOND unresolved difference and is flagged as an open question rather than quietly treated as immaterial.

- **How much SKA earns its place, beyond that it does.** This one is PARTLY settled and the correction is worth recording. An earlier 4m-geometry smoke study measured an ablation delta of 1.17e-4 and raised the worry that this study might be measuring SKA's hyperparameters without establishing that the branch does anything. Job 445689 settles the sign: on THIS base at 600 steps the delta is 1.17e-2 to 7.48e-2 depending on the layerscale, a hundred times larger, and positive throughout -- removing SKA makes the model worse. The 4m number was a scale artefact. What remains unestablished is the MAGNITUDE at any scale that matters: 25.35M parameters and 600 steps is not 50m and it is not convergence, and an ablation delta measured on a model that has barely left its warmup transient is not the delta at convergence. Compare the ska_delta column to the noise floor directly, and do not read its size as a claim about a trained model.

- **Any effect smaller than the measured noise floor.** Not 'probably not real' -- unmeasurable by this study, at any trial count, because it is inside what one configuration does to itself when the seed changes. More trials narrow the sampler's search, not this.

- **Non-determinism, or seed sensitivity at other configurations.** The base spec sets `deterministic: true` and the replicates are all at ONE configuration. So the floor is a deterministic-mode floor at the reference point; a non-deterministic run spreads more, and a badly-conditioned corner of the space (rank 8 with a light ridge, say) plausibly spreads more still. Applying one floor to the whole space is an assumption, and it is the optimistic one.

- **Statistical significance of anything.** Fourteen prespecified pairwise tables, nine axes, one seed per sampled trial, and cell counts that are uneven BY CONSTRUCTION because the sampler was adaptive. 'Resolved' here means 'larger than seed noise', with no multiplicity correction and no model of the sampling process. It is a filter for what to confirm, not a test.

- **Interactions among the fixed axes, or between them and anything.** `weight_decay`, `warmup_ratio` and `grad_clip` are singletons in this study, so their columns are constant and every table over them is degenerate by design, not by accident.

