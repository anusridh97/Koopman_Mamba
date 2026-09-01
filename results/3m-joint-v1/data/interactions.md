# Interaction analysis: 3m-joint-v1

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

- completed: **1041**
- pruned: **810**
- failed: **156**
- total recorded: **2015**
- of which anchors: **0**

### Failure reasons

One reason dominating is a bug in the harness; many distinct reasons is a rough study. A count alone cannot tell them apart.

- 102x `no objective could be read`
- 1x `OSError: [Errno 122] Disk quota exceeded: '/scratch/m000151-pm06/cqiu/study-3m-joint-v1/3m-joint-v1.d99c26a1/seed42.2ff44b5a/.running'`
- 1x `OSError: [Errno 122] Disk quota exceeded: '/scratch/m000151-pm06/cqiu/study-3m-joint-v1/3m-joint-v1.5c586950'`
- 1x `OSError: [Errno 122] Disk quota exceeded: '/scratch/m000151-pm06/cqiu/study-3m-joint-v1/3m-joint-v1.7f6e58fb/seed42.180411df'`
- 1x `OSError: [Errno 122] Disk quota exceeded: '/scratch/m000151-pm06/cqiu/study-3m-joint-v1/3m-joint-v1.7d564579/seed42.70ff6a95'`
- 1x `OSError: [Errno 122] Disk quota exceeded: '/scratch/m000151-pm06/cqiu/study-3m-joint-v1/3m-joint-v1.7847fcef'`
- 1x `OSError: [Errno 122] Disk quota exceeded: '/scratch/m000151-pm06/cqiu/study-3m-joint-v1/3m-joint-v1.8a051429'`
- 1x `OSError: [Errno 122] Disk quota exceeded: '/scratch/m000151-pm06/cqiu/study-3m-joint-v1/3m-joint-v1.62795fbe/seed42.772eb8f2/.running'`
- 1x `OSError: [Errno 122] Disk quota exceeded: '/scratch/m000151-pm06/cqiu/study-3m-joint-v1/3m-joint-v1.14af183d/seed42.bd843fdc'`
- 1x `OSError: [Errno 122] Disk quota exceeded: '/scratch/m000151-pm06/cqiu/study-3m-joint-v1/3m-joint-v1.84ad2843/seed42.797c9551'`
- 1x `OSError: [Errno 122] Disk quota exceeded: '/scratch/m000151-pm06/cqiu/study-3m-joint-v1/3m-joint-v1.4cb436c5/seed42.f0d02a17'`
- 1x `OSError: [Errno 122] Disk quota exceeded: '/scratch/m000151-pm06/cqiu/study-3m-joint-v1/3m-joint-v1.2ba5bb2b'`
- 1x `OSError: [Errno 122] Disk quota exceeded: '/scratch/m000151-pm06/cqiu/study-3m-joint-v1/3m-joint-v1.137370fe/seed42.58806062/.running'`
- 1x `OSError: [Errno 122] Disk quota exceeded: '/scratch/m000151-pm06/cqiu/study-3m-joint-v1/3m-joint-v1.95d28e9c/seed42.df98aeca/attempts.jsonl'`
- 1x `OSError: [Errno 122] Disk quota exceeded: '/scratch/m000151-pm06/cqiu/study-3m-joint-v1/3m-joint-v1.107338f1/seed42.abe84950'`
- 1x `OSError: [Errno 122] Disk quota exceeded: '/scratch/m000151-pm06/cqiu/study-3m-joint-v1/3m-joint-v1.cca2feb7/seed42.cea47038'`
- 1x `OSError: [Errno 122] Disk quota exceeded: '/scratch/m000151-pm06/cqiu/study-3m-joint-v1/3m-joint-v1.34faf85e/seed42.b5488f5d/.running'`
- 1x `OSError: [Errno 122] Disk quota exceeded: '/scratch/m000151-pm06/cqiu/study-3m-joint-v1/3m-joint-v1.ef219d67/seed42.da5ceef2/.running'`
- 1x `OSError: [Errno 122] Disk quota exceeded: '/scratch/m000151-pm06/cqiu/study-3m-joint-v1/3m-joint-v1.30bad4be/seed42.b014a8ad/.running'`
- 1x `OSError: [Errno 122] Disk quota exceeded: '/scratch/m000151-pm06/cqiu/study-3m-joint-v1/3m-joint-v1.8950010d'`
- 1x `OSError: [Errno 122] Disk quota exceeded: '/scratch/m000151-pm06/cqiu/study-3m-joint-v1/3m-joint-v1.f33ce379/seed42.c9f37dfa/.running'`
- 1x `OSError: [Errno 122] Disk quota exceeded: '/scratch/m000151-pm06/cqiu/study-3m-joint-v1/3m-joint-v1.4bfdd174'`
- 1x `OSError: [Errno 122] Disk quota exceeded: '/scratch/m000151-pm06/cqiu/study-3m-joint-v1/3m-joint-v1.f864c272/seed42.1f666137/.running'`
- 1x `OSError: [Errno 122] Disk quota exceeded: '/scratch/m000151-pm06/cqiu/study-3m-joint-v1/3m-joint-v1.253fcdeb/seed42.054d3de2/.running'`
- 1x `OSError: [Errno 122] Disk quota exceeded: '/scratch/m000151-pm06/cqiu/study-3m-joint-v1/3m-joint-v1.0b8e1dbb'`
- 1x `OSError: [Errno 122] Disk quota exceeded: '/scratch/m000151-pm06/cqiu/study-3m-joint-v1/3m-joint-v1.dd5757c9/seed42.e14de54a'`
- 1x `OSError: [Errno 122] Disk quota exceeded: '/scratch/m000151-pm06/cqiu/study-3m-joint-v1/3m-joint-v1.1825a876/seed42.f4ea0e77'`
- 1x `OSError: [Errno 122] Disk quota exceeded: '/scratch/m000151-pm06/cqiu/study-3m-joint-v1/3m-joint-v1.6d115445/seed42.f798d115/.running'`
- 1x `OSError: [Errno 122] Disk quota exceeded: '/scratch/m000151-pm06/cqiu/study-3m-joint-v1/3m-joint-v1.d19f31c6/seed42.b4e83fe1'`
- 1x `OSError: [Errno 122] Disk quota exceeded: '/scratch/m000151-pm06/cqiu/study-3m-joint-v1/3m-joint-v1.83bd0343/seed42.805778c1'`
- 1x `OSError: [Errno 122] Disk quota exceeded: '/scratch/m000151-pm06/cqiu/study-3m-joint-v1/3m-joint-v1.765bb0a2'`
- 1x `OSError: [Errno 122] Disk quota exceeded: '/scratch/m000151-pm06/cqiu/study-3m-joint-v1/3m-joint-v1.8cb320e1/seed42.111e07b9/.running'`
- 1x `OSError: [Errno 122] Disk quota exceeded: '/scratch/m000151-pm06/cqiu/study-3m-joint-v1/3m-joint-v1.c2cef26b/seed42.001b007c'`
- 1x `OSError: [Errno 122] Disk quota exceeded: '/scratch/m000151-pm06/cqiu/study-3m-joint-v1/3m-joint-v1.0dc465d3/seed42.9582da6c/.running'`
- 1x `OSError: [Errno 122] Disk quota exceeded: '/scratch/m000151-pm06/cqiu/study-3m-joint-v1/3m-joint-v1.6419293c/seed42.371df1ba'`
- 1x `OSError: [Errno 122] Disk quota exceeded: '/scratch/m000151-pm06/cqiu/study-3m-joint-v1/3m-joint-v1.f71f7a1c/seed42.620c8975'`
- 1x `OSError: [Errno 122] Disk quota exceeded: '/scratch/m000151-pm06/cqiu/study-3m-joint-v1/3m-joint-v1.212c64af/seed42.3addcf82'`
- 1x `OSError: [Errno 122] Disk quota exceeded: '/scratch/m000151-pm06/cqiu/study-3m-joint-v1/3m-joint-v1.562a2f20'`
- 1x `OSError: [Errno 122] Disk quota exceeded: '/scratch/m000151-pm06/cqiu/study-3m-joint-v1/3m-joint-v1.bb76a68e'`
- 1x `OSError: [Errno 122] Disk quota exceeded: '/scratch/m000151-pm06/cqiu/study-3m-joint-v1/3m-joint-v1.2ae78081/seed42.3d147435/.running'`
- 1x `OSError: [Errno 122] Disk quota exceeded: '/scratch/m000151-pm06/cqiu/study-3m-joint-v1/3m-joint-v1.a600b4db/seed42.f19b98b5'`
- 1x `OSError: [Errno 122] Disk quota exceeded: '/scratch/m000151-pm06/cqiu/study-3m-joint-v1/3m-joint-v1.44877537/seed42.a74f3c5e/train.log'`
- 1x `OSError: [Errno 122] Disk quota exceeded: '/scratch/m000151-pm06/cqiu/study-3m-joint-v1/3m-joint-v1.754aeb7c/seed42.e1b4593c'`
- 1x `OSError: [Errno 122] Disk quota exceeded: '/scratch/m000151-pm06/cqiu/study-3m-joint-v1/3m-joint-v1.5ac8fbaf/seed42.231aaa0f/.running'`
- 1x `OSError: [Errno 122] Disk quota exceeded: '/scratch/m000151-pm06/cqiu/study-3m-joint-v1/3m-joint-v1.c57be6bd/seed42.923dfc2a'`
- 1x `OSError: [Errno 122] Disk quota exceeded: '/scratch/m000151-pm06/cqiu/study-3m-joint-v1/3m-joint-v1.4630c6f4'`
- 1x `OSError: [Errno 122] Disk quota exceeded: '/scratch/m000151-pm06/cqiu/study-3m-joint-v1/3m-joint-v1.e9ae3954/seed42.82dc779b'`
- 1x `OSError: [Errno 122] Disk quota exceeded: '/scratch/m000151-pm06/cqiu/study-3m-joint-v1/3m-joint-v1.f718d3e2/seed42.cdc51c5b/.running'`
- 1x `OSError: [Errno 122] Disk quota exceeded: '/scratch/m000151-pm06/cqiu/study-3m-joint-v1/3m-joint-v1.4650026b/seed42.a9ad883b/.running'`
- 1x `OSError: [Errno 122] Disk quota exceeded: '/scratch/m000151-pm06/cqiu/study-3m-joint-v1/3m-joint-v1.01f0ee02/seed42.afbc16c0'`
- 1x `OSError: [Errno 122] Disk quota exceeded: '/scratch/m000151-pm06/cqiu/study-3m-joint-v1/3m-joint-v1.bf997d98'`
- 1x `OSError: [Errno 122] Disk quota exceeded: '/scratch/m000151-pm06/cqiu/study-3m-joint-v1/3m-joint-v1.c3c67c2e/seed42.b11e6f1a'`
- 1x `OSError: [Errno 122] Disk quota exceeded: '/scratch/m000151-pm06/cqiu/study-3m-joint-v1/3m-joint-v1.46ad2836/seed42.5dea449c/train.log'`
- 1x `OSError: [Errno 122] Disk quota exceeded: '/scratch/m000151-pm06/cqiu/study-3m-joint-v1/3m-joint-v1.a3751126'`
- 1x `OSError: [Errno 122] Disk quota exceeded: '/scratch/m000151-pm06/cqiu/study-3m-joint-v1/3m-joint-v1.d353d8a1'`

## 0. The noise floor -- read this before any number below

**UNAVAILABLE.** no COMPLETED trial carries a `reference_group` attr, so this study measured no replicates and has no noise floor. Every effect below is therefore reported WITHOUT a scale: a difference of 1e-4 and a difference of 1e-1 are presented identically and neither can be called resolvable. Add a `reference_group` of 3-5 designs differing only in `seed` to the study's design file.

Every magnitude in this report is therefore presented WITHOUT A SCALE. A difference of 1e-4 and a difference of 1e-1 look identical here -- and job 445689 measured the losses this study must separate as spanning 0.069 with the best two 0.004 apart, so that distinction is the whole question. Treat every ranking below as unvalidated ordering.

## 0b. Controlled anchor contrasts

The only part of this report that is a CONTROLLED EXPERIMENT rather than an observational summary. Each anchor is a one-factor move from a single reference point -- verified after resolution, so no second axis moved on the way through -- and each delta is against the reference replicate MEAN rather than against one of its seeds.

Read this table before the importance table and before the pairwise tables. It is the only place a difference is attributable to a named factor without an argument about the sampler.

No completed anchor trials outside the replicate set.

## 0c. Partial dependence / main effects

Each row is a PARTIAL DEPENDENCE table: the mean objective of every completed trial at each level of one axis, marginalising over the others by averaging rather than by holding them fixed. `spread` is max-min of those level means and `variance_share` is the count-weighted between-level variance divided by the total variance of the objective.

This is DESCRIPTIVE, not causal, and the reason is the sampler. It chose which trials exist, so the other axes are not balanced across the levels of this one -- a level the sampler visited mostly alongside a good ridge looks good. It also concentrated its later proposals, so an axis it pinned near one value shows a small spread whether or not it matters; check that axis's own spread in trials.csv before concluding it is unimportant. The one-factor ANCHOR CONTRASTS in section 0b are the controlled version of this question and should be read first; this table is what extends it to the sampled body, at the cost of the control.

**The verdict column here is the weakest of the three that carry one.** A level mean averages over many trials, so its own seed-noise standard error is `sigma/sqrt(n)` rather than `sigma`, and the spread of two such means is correspondingly tighter than the trial-vs-trial threshold it is being compared against. So an axis marked `unresolvable` here may still have a resolvable effect -- read the controlled anchor contrasts in section 0b, which compare like with like. A `RESOLVED` here is safe in the other direction, and `DID NOT VARY` means the axis took one value and was never measured at all.

| axis | variance share | spread of level means | verdict | levels (mean, n) |
|:--|---:|---:|:--|:--|
| `learning_rate` | 0.7516 | 0.3594 | NO FLOOR | [0.001501, 0.002833]: 4.5938 (n=72); [0.002833, 0.004166]: 4.3564 (n=101); [0.004166, 0.005499]: 4.2344 (n=868) |
| `ska_rank` | 0.3396 | 0.1802 | NO FLOOR | 8: 4.4113 (n=85); 16: 4.3852 (n=87); 24: 4.3152 (n=154); 32: 4.2311 (n=715) |
| `mamba_expand` | 0.2866 | 0.2373 | NO FLOOR | 1: 4.4815 (n=57); 2: 4.3223 (n=185); 3: 4.2442 (n=799) |
| `ska_power_K` | 0.2437 | 0.1676 | NO FLOOR | 1: 4.2513 (n=918); 2: 4.4189 (n=123) |
| `beta_policy` | 0.2105 | 0.1273 | NO FLOOR | head_scalar: 4.3329 (n=102); learned: 4.3303 (n=131); linear: 4.2364 (n=698); one: 4.3637 (n=110) |
| `warmup_ratio` | 0.1544 | 0.1177 | NO FLOOR | 0.01: 4.3617 (n=88); 0.02: 4.3482 (n=90); 0.04: 4.3027 (n=145); 0.06: 4.244 (n=718) |
| `depth_tier` | 0.1032 | 0.0846 | NO FLOOR | full: 4.2522 (n=809); lean: 4.3368 (n=232) |
| `weight_decay` | 0.0907 | 0.09268 | NO FLOOR | 0.0: 4.3397 (n=97); 0.05: 4.3244 (n=108); 0.1: 4.2804 (n=231); 0.15: 4.247 (n=605) |
| `n_ska_layers` | 0.0581 | 0.08535 | NO FLOOR | 2: 4.3339 (n=117); 3: 4.2897 (n=195); 4: 4.2621 (n=404); 6: 4.2486 (n=325) |
| `grad_clip` | 0.0540 | 0.05605 | NO FLOOR | 0.5: 4.3108 (n=303); 1.0: 4.2548 (n=738) |
| `d_state` | 0.0473 | 0.07471 | NO FLOOR | 8: 4.3274 (n=108); 16: 4.295 (n=174); 24: 4.2609 (n=457); 32: 4.2527 (n=302) |
| `ska_n_heads` | 0.0420 | 0.06055 | NO FLOOR | 1: 4.2921 (n=190); 2: 4.2561 (n=398); 4: 4.2571 (n=313); 8: 4.3167 (n=140) |
| `placement` | 0.0391 | 0.06097 | NO FLOOR | even: 4.3169 (n=184); late: 4.2559 (n=392); midlate: 4.2658 (n=465) |
| `ska_layerscale_init` | 0.0249 | 0.03829 | NO FLOOR | [0.005044, 0.01968]: 4.2915 (n=252); [0.01968, 0.07679]: 4.2838 (n=294); [0.07679, 0.2996]: 4.2532 (n=495) |
| `gamma_value` | 0.0155 | 0.0335 | NO FLOOR | 0.9: 4.2594 (n=522); 1.0: 4.2929 (n=254); 1.05: 4.2733 (n=265) |
| `norm_clip_multiplier` | 0.0036 | 0.01892 | NO FLOOR | 0.75: 4.2642 (n=235); 0.8164965809277261: 4.2745 (n=267); 1.0: 4.2832 (n=175); 1.25: 4.2672 (n=364) |
| `ska_ridge` | 0.0009 | 0.006537 | NO FLOOR | [0.003004, 0.012]: 4.274 (n=574); [0.012, 0.021]: 4.2675 (n=261); [0.021, 0.02999]: 4.2676 (n=206) |

## 0d. Conditional effects -- the interaction magnitudes

How much the effect of `left` CHANGES across the levels of `right`. That is what an interaction is, so these are the one-number-per-pair summaries of the question the study was launched to answer. The full cell tables are in section 2.

Both are differences OF differences, so they accumulate noise from four cell means while being tested against a two-single-trials threshold -- which makes this the most OPTIMISTIC comparison in this report. A pair that fails it is certainly unresolvable; a pair that passes it narrowly needs its cell counts read before it is believed.

**Two magnitudes, because an interaction can change a SIZE or a DIRECTION and only one of those invalidates a single recommended value.** `size` is how much the unsigned effect of `left` varies across the levels of `right`. `signed` is the same statistic over the SIGNED contrast (last minus first level of `left`), and it is the one that catches a crossover: if `left` is better at one level of `right` and worse by the same amount at another, the unsigned ranges are identical and `size` is exactly 0.0 while the interaction is maximal. The verdict reads whichever is larger, and `sign change` marks the rows where the contrasts do not all point the same way -- those are the ones where there is no single best value of `left` to recommend.

| left | right | size | signed | sign change | sigmas | verdict | conditional effects (level: effect, signed contrast, n) |
|:--|:--|---:|---:|:--|---:|:--|:--|
| `ska_rank` | `n_ska_layers` | 0.1848 | 0.1912 | no | - | NO FLOOR | 2: 0.1865 / -0.1764 (n=117); 3: 0.1847 / -0.1847 (n=195); 4: 0.1139 / -0.1075 (n=404); 6: 0.2987 / -0.2987 (n=325) |
| `learning_rate` | `ska_rank` | 0.1407 | 0.1407 | no | - | NO FLOOR | 8: 0.2698 / -0.2698 (n=85); 16: 0.3052 / -0.3052 (n=87); 24: 0.3159 / -0.3159 (n=154); 32: 0.4104 / -0.4104 (n=715) |
| `ska_rank` | `ska_ridge` | 0.08647 | 0.1209 | no | - | NO FLOOR | [0.003004, 0.012]: 0.179 / -0.179 (n=574); [0.012, 0.021]: 0.2344 / -0.2344 (n=261); [0.021, 0.02999]: 0.1479 / -0.1135 (n=206) |
| `gamma_value` | `norm_clip_multiplier` | 0.03467 | 0.1016 | **YES** | - | NO FLOOR | 0.75: 0.0675 / +0.06743 (n=235); 0.8164965809277261: 0.04247 / +0.04247 (n=267); 1.0: 0.03283 / +0.03283 (n=175); 1.25: 0.03945 / -0.03416 (n=364) |
| `ska_rank` | `ska_power_K` | 0.03885 | 0.08068 | no | - | NO FLOOR | 1: 0.1597 / -0.1597 (n=918); 2: 0.1208 / -0.07898 (n=123) |
| `ska_rank` | `norm_clip_multiplier` | 0.07988 | 0.07988 | no | - | NO FLOOR | 0.75: 0.228 / -0.228 (n=235); 0.8164965809277261: 0.2227 / -0.2227 (n=267); 1.0: 0.1772 / -0.167 (n=175); 1.25: 0.1481 / -0.1481 (n=364) |
| `n_ska_layers` | `ska_layerscale_init` | 0.03797 | 0.07522 | no | - | NO FLOOR | [0.005044, 0.01968]: 0.1046 / -0.03236 (n=252); [0.01968, 0.07679]: 0.06857 / -0.03132 (n=294); [0.07679, 0.2996]: 0.1065 / -0.1065 (n=495) |
| `learning_rate` | `n_ska_layers` | 0.06579 | 0.06579 | no | - | NO FLOOR | 2: 0.3582 / -0.3582 (n=117); 3: 0.3186 / -0.3186 (n=195); 4: 0.3535 / -0.3535 (n=404); 6: 0.3844 / -0.3844 (n=325) |
| `learning_rate` | `ska_power_K` | 0.06132 | 0.06132 | no | - | NO FLOOR | 1: 0.3532 / -0.3532 (n=918); 2: 0.2919 / -0.2919 (n=123) |
| `n_ska_layers` | `placement` | 0.01958 | 0.04523 | no | - | NO FLOOR | even: 0.1081 / -0.04826 (n=184); late: 0.09349 / -0.09349 (n=392); midlate: 0.08851 / -0.08656 (n=465) |
| `ska_power_K` | `gamma_value` | 0.03885 | 0.03885 | no | - | NO FLOOR | 0.9: 0.1434 / +0.1434 (n=522); 1.0: 0.1823 / +0.1823 (n=254); 1.05: 0.1823 / +0.1823 (n=265) |
| `ska_ridge` | `ska_layerscale_init` | 0.007614 | 0.03396 | **YES** | - | NO FLOOR | [0.005044, 0.01968]: 0.02071 / -0.02071 (n=252); [0.01968, 0.07679]: 0.01494 / +0.01325 (n=294); [0.07679, 0.2996]: 0.0131 / -0.0131 (n=495) |
| `ska_layerscale_init` | `learning_rate` | 0.02259 | 0.03393 | **YES** | - | NO FLOOR | [0.001501, 0.002833]: 0.01499 / +0.002251 (n=72); [0.002833, 0.004166]: 0.00909 / -0.00909 (n=101); [0.004166, 0.005499]: 0.03168 / -0.03168 (n=868) |
| `ska_ridge` | `ska_power_K` | 0.01882 | 0.02206 | no | - | NO FLOOR | 1: 0.007563 / -0.004323 (n=918); 2: 0.02639 / -0.02639 (n=123) |

## 1. PED-ANOVA importance (a divergence, not a decomposition)

Evaluator: `PedAnovaImportanceEvaluator(evaluate_on_local=True)`.

WHAT THIS MEASURES. PED-ANOVA scores each axis by the DIVERGENCE between its distribution among the study's top-quantile trials and its distribution in a reference set. Two reference sets are reported and they answer different questions:
  * `local` (optuna's default, `evaluate_on_local=True`) compares against the study's OWN remaining trials. Because the sampler was adaptive, that empirical distribution is largely a description of the search path -- so a high local score can mean 'this axis separates good from bad' or 'the sampler moved this axis a lot late on', and the number cannot tell them apart. On this repo's own analysis test fixture a NULL axis outranked a planted main effect under this setting.
  * `global` (`evaluate_on_local=False`) compares against the DECLARED prior instead, so it is not a function of where the sampler went -- at the cost of being a comparison against a region the study may barely have sampled.

Neither is proof about the model. Treat a large gap between the two columns as a warning that the axis's score is search-path dependent. For claims about the MODEL, read the controlled anchor contrasts first (one factor moved from one reference point, with the noise floor beside it) and the partial-dependence table second. An axis with a low score here may be unimportant, or may simply have been pinned near one value by the sampler after the startup window; its spread in trials.csv is what distinguishes those.

| axis | local | global |
|:--|---:|---:|
| `learning_rate` | 0.5888 | 0.6861 |
| `ska_layerscale_init` | 0.1550 | 0.0537 |
| `n_ska_layers` | 0.0529 | 0.0139 |
| `ska_ridge` | 0.0398 | 0.0122 |
| `depth_tier` | 0.0320 | 0.0220 |
| `placement` | 0.0234 | 0.0080 |
| `warmup_ratio` | 0.0216 | 0.0441 |
| `d_state` | 0.0179 | 0.0084 |
| `ska_n_heads` | 0.0134 | 0.0094 |
| `ska_rank` | 0.0128 | 0.0369 |
| `weight_decay` | 0.0126 | 0.0241 |
| `norm_clip_multiplier` | 0.0083 | 0.0011 |
| `mamba_expand` | 0.0067 | 0.0220 |
| `beta_policy` | 0.0060 | 0.0292 |
| `gamma_value` | 0.0046 | 0.0046 |
| `grad_clip` | 0.0030 | 0.0073 |
| `ska_power_K` | 0.0010 | 0.0172 |

## 2. Pairwise response structure (prespecified)

### ska_rank x ska_ridge

| ska_rank \ ska_ridge | [0.003004, 0.012] | [0.012, 0.021] | [0.021, 0.02999] |
|:--|---:|---:|---:|
| 8 | 4.4064 (n=59) | 4.4648 (n=16) | 4.3547 (n=10) |
| 16 | 4.3907 (n=47) | 4.3693 (n=21) | 4.3891 (n=19) |
| 24 | 4.3369 (n=78) | 4.2871 (n=53) | 4.3064 (n=23) |
| 32 | 4.2274 (n=390) | 4.2304 (n=171) | 4.2412 (n=154) |

Binned: ska_ridge. Cell values are mean objective with the trial count.

### ska_rank x norm_clip_multiplier

| ska_rank \ norm_clip_multiplier | 0.75 | 0.8164965809277261 | 1.0 | 1.25 |
|:--|---:|---:|---:|---:|
| 8 | 4.4567 (n=14) | 4.4519 (n=17) | 4.4012 (n=16) | 4.3807 (n=38) |
| 16 | 4.3641 (n=19) | 4.4266 (n=17) | 4.4113 (n=20) | 4.3584 (n=31) |
| 24 | 4.3069 (n=33) | 4.3326 (n=48) | 4.3415 (n=22) | 4.2927 (n=51) |
| 32 | 4.2288 (n=169) | 4.2292 (n=185) | 4.2342 (n=117) | 4.2326 (n=244) |

### ska_rank x ska_power_K

| ska_rank \ ska_power_K | 1 | 2 |
|:--|---:|---:|
| 8 | 4.3833 (n=49) | 4.4494 (n=36) |
| 16 | 4.3514 (n=66) | 4.4912 (n=21) |
| 24 | 4.2970 (n=124) | 4.3901 (n=30) |
| 32 | 4.2237 (n=679) | 4.3704 (n=36) |

### n_ska_layers x ska_layerscale_init

| n_ska_layers \ ska_layerscale_init | [0.005044, 0.01968] | [0.01968, 0.07679] | [0.07679, 0.2996] |
|:--|---:|---:|---:|
| 2 | 4.3622 (n=26) | 4.3246 (n=40) | 4.3268 (n=51) |
| 3 | 4.3524 (n=40) | 4.3116 (n=41) | 4.2597 (n=114) |
| 4 | 4.2576 (n=158) | 4.2561 (n=129) | 4.2747 (n=117) |
| 6 | 4.3298 (n=28) | 4.2933 (n=84) | 4.2202 (n=213) |

Binned: ska_layerscale_init. Cell values are mean objective with the trial count.

### n_ska_layers x placement

| n_ska_layers \ placement | even | late | midlate |
|:--|---:|---:|---:|
| 2 | 4.3615 (n=22) | 4.3127 (n=47) | 4.3421 (n=48) |
| 3 | 4.3869 (n=27) | 4.2896 (n=61) | 4.2651 (n=107) |
| 4 | 4.2788 (n=69) | 4.2681 (n=117) | 4.2536 (n=218) |
| 6 | 4.3133 (n=66) | 4.2192 (n=167) | 4.2555 (n=92) |

### ska_ridge x ska_power_K

| ska_ridge \ ska_power_K | 1 | 2 |
|:--|---:|---:|
| [0.003004, 0.012] | 4.2513 (n=499) | 4.4252 (n=75) |
| [0.012, 0.021] | 4.2545 (n=241) | 4.4235 (n=20) |
| [0.021, 0.02999] | 4.2470 (n=178) | 4.3988 (n=28) |

Binned: ska_ridge. Cell values are mean objective with the trial count.

### ska_layerscale_init x learning_rate

| ska_layerscale_init \ learning_rate | [0.001501, 0.002833] | [0.002833, 0.004166] | [0.004166, 0.005499] |
|:--|---:|---:|---:|
| [0.005044, 0.01968] | 4.5975 (n=17) | 4.3611 (n=34) | 4.2538 (n=201) |
| [0.01968, 0.07679] | 4.5847 (n=26) | 4.3563 (n=33) | 4.2403 (n=235) |
| [0.07679, 0.2996] | 4.5997 (n=29) | 4.3520 (n=34) | 4.2221 (n=432) |

Binned: ska_layerscale_init, learning_rate. Cell values are mean objective with the trial count.

### ska_rank x n_ska_layers

| ska_rank \ n_ska_layers | 2 | 3 | 4 | 6 |
|:--|---:|---:|---:|---:|
| 8 | 4.4406 (n=13) | 4.4266 (n=22) | 4.3445 (n=34) | 4.5083 (n=16) |
| 16 | 4.4507 (n=18) | 4.4176 (n=20) | 4.3510 (n=30) | 4.3429 (n=19) |
| 24 | 4.3570 (n=27) | 4.3016 (n=29) | 4.3043 (n=45) | 4.3105 (n=53) |
| 32 | 4.2642 (n=59) | 4.2419 (n=124) | 4.2371 (n=295) | 4.2096 (n=237) |

### ska_power_K x gamma_value

| ska_power_K \ gamma_value | 0.9 | 1.0 | 1.05 |
|:--|---:|---:|---:|
| 1 | 4.2459 (n=473) | 4.2685 (n=220) | 4.2458 (n=225) |
| 2 | 4.3894 (n=49) | 4.4508 (n=34) | 4.4280 (n=40) |

### ska_ridge x ska_layerscale_init

| ska_ridge \ ska_layerscale_init | [0.005044, 0.01968] | [0.01968, 0.07679] | [0.07679, 0.2996] |
|:--|---:|---:|---:|
| [0.003004, 0.012] | 4.2978 (n=149) | 4.2777 (n=166) | 4.2580 (n=259) |
| [0.012, 0.021] | 4.2874 (n=52) | 4.2926 (n=62) | 4.2498 (n=147) |
| [0.021, 0.02999] | 4.2771 (n=51) | 4.2909 (n=66) | 4.2449 (n=89) |

Binned: ska_ridge, ska_layerscale_init. Cell values are mean objective with the trial count.

### gamma_value x norm_clip_multiplier

| gamma_value \ norm_clip_multiplier | 0.75 | 0.8164965809277261 | 1.0 | 1.25 |
|:--|---:|---:|---:|---:|
| 0.9 | 4.2401 (n=151) | 4.2542 (n=134) | 4.2688 (n=91) | 4.2782 (n=146) |
| 1.0 | 4.3076 (n=44) | 4.2943 (n=90) | 4.2942 (n=33) | 4.2835 (n=87) |
| 1.05 | 4.3076 (n=40) | 4.2967 (n=43) | 4.3016 (n=51) | 4.2441 (n=131) |

### learning_rate x ska_rank

| learning_rate \ ska_rank | 8 | 16 | 24 | 32 |
|:--|---:|---:|---:|---:|
| [0.001501, 0.002833] | 4.5915 (n=22) | 4.5928 (n=25) | 4.5776 (n=16) | 4.6307 (n=9) |
| [0.002833, 0.004166] | 4.3980 (n=22) | 4.3736 (n=10) | 4.3798 (n=27) | 4.3156 (n=42) |
| [0.004166, 0.005499] | 4.3217 (n=41) | 4.2876 (n=52) | 4.2616 (n=111) | 4.2203 (n=664) |

Binned: learning_rate. Cell values are mean objective with the trial count.

### learning_rate x n_ska_layers

| learning_rate \ n_ska_layers | 2 | 3 | 4 | 6 |
|:--|---:|---:|---:|---:|
| [0.001501, 0.002833] | 4.6244 (n=18) | 4.5594 (n=19) | 4.5930 (n=15) | 4.5994 (n=20) |
| [0.002833, 0.004166] | 4.3800 (n=13) | 4.3744 (n=26) | 4.3348 (n=40) | 4.3607 (n=22) |
| [0.004166, 0.005499] | 4.2662 (n=86) | 4.2408 (n=150) | 4.2395 (n=349) | 4.2151 (n=283) |

Binned: learning_rate. Cell values are mean objective with the trial count.

### learning_rate x ska_power_K

| learning_rate \ ska_power_K | 1 | 2 |
|:--|---:|---:|
| [0.001501, 0.002833] | 4.5819 (n=37) | 4.6063 (n=35) |
| [0.002833, 0.004166] | 4.3370 (n=71) | 4.4026 (n=30) |
| [0.004166, 0.005499] | 4.2287 (n=810) | 4.3143 (n=58) |

Binned: learning_rate. Cell values are mean objective with the trial count.

## 3. Sampler-induced correlation (DIAGNOSTIC, NOT EVIDENCE)

SAMPLER DIAGNOSTIC, NOT EVIDENCE. These are correlations between sampled columns. The sampler was adaptive, so it concentrated its later proposals in a region it believed was good, and any correlation here is primarily a description of that search path. It is not a finding about the model and must not be reported as one. Its legitimate use is to qualify the importance table: an axis the sampler pinned near one value will show low importance whether or not it matters.

Excluded (categorical labels have no meaningful r): `beta_policy`, `depth_tier`, `placement`

| a | b | r | n |
|:--|:--|---:|---:|
| `learning_rate` | `ska_rank` | +0.439 | 1041 |
| `learning_rate` | `ska_power_K` | -0.375 | 1041 |
| `ska_power_K` | `ska_rank` | -0.348 | 1041 |
| `learning_rate` | `mamba_expand` | +0.303 | 1041 |
| `mamba_expand` | `ska_power_K` | -0.301 | 1041 |
| `learning_rate` | `warmup_ratio` | +0.289 | 1041 |
| `mamba_expand` | `ska_rank` | +0.288 | 1041 |
| `learning_rate` | `weight_decay` | +0.230 | 1041 |
| `grad_clip` | `mamba_expand` | +0.230 | 1041 |
| `ska_rank` | `warmup_ratio` | +0.226 | 1041 |
| `grad_clip` | `ska_rank` | +0.219 | 1041 |
| `ska_power_K` | `warmup_ratio` | -0.203 | 1041 |
| `grad_clip` | `learning_rate` | +0.194 | 1041 |
| `gamma_value` | `norm_clip_multiplier` | +0.189 | 1041 |
| `ska_rank` | `weight_decay` | +0.186 | 1041 |

## 4. Loss / parameter-count Pareto front

Computed POST HOC, which is why the study is single-objective: optuna cannot prune a multi-objective study at all, and pruning is what makes the study affordable. This is also where parameter count is meant to be traded against loss -- a scalar `parameter_penalty` bakes one exchange rate into every subsequent proposal, unrecoverably.

| params | objective | trial | anchor |
|---:|---:|---:|:--|
| 2,834,600 | 4.80202 | 19 |  |
| 2,838,220 | 4.41599 | 151 |  |
| 2,838,562 | 4.37760 | 24 |  |
| 2,840,138 | 4.30627 | 282 |  |
| 2,854,667 | 4.22235 | 833 |  |
| 2,872,588 | 4.20084 | 1166 |  |
| 2,889,228 | 4.19853 | 519 |  |
| 2,903,150 | 4.19455 | 1633 |  |
| 2,925,230 | 4.19223 | 1196 |  |
| 2,950,196 | 4.17684 | 1412 |  |
| 2,956,208 | 4.16774 | 849 |  |
| 3,018,280 | 4.15367 | 1645 |  |
| 3,030,424 | 4.15355 | 1028 |  |
| 3,080,356 | 4.15179 | 1794 |  |

## 4b. Loss / throughput Pareto front

The other half of the cost question, and until `tokens_per_sec` was promoted onto the trial it could not be computed at all -- the number was measured in every `quick_eval.json` and reached neither `user_attrs` nor `trials.csv`.

Read WITH `per_device_batch_size`: a trial that descended the OOM ladder was measured at a smaller microbatch than its neighbours, so its throughput is not comparable to theirs. That caveat is why the loss/throughput exchange rate stays the reader's rather than becoming a `throughput_penalty` baked into every proposal.

A worker process's FIRST trial (`attr_worker_trial_ordinal == 0`) is excluded from the throughput front. Measured, job 445689: four trials on the proxy base at 600 steps reported 17,818 / 111,416 / 104,760 / 112,685 tok/s, and the 6x outlier was trial 0. Those four configs differ only in `ska_layerscale_init` -- one scalar multiply on a gate -- so nothing architectural can produce a 6x throughput gap. It is first-trial CUDA context creation and kernel autotuning, and `tokens_per_sec` is measured during the eval pass, early enough in the process's life to still be inside that. At `concurrent_trials: 8` this is EIGHT poisoned points, not one, because every worker process pays its own warmup. The trials are NOT dropped from anything else -- their loss is a real measurement and only their throughput is suspect -- and `throughput_pareto(exclude_warmup=False)` returns them for anyone who wants to check the claim or who is on hardware where it does not hold. A journal written before this column existed has no ordinals and nothing is excluded, rather than everything.

Of 1041 completed trial(s): 1041 carry a throughput measurement, 0 do not, and 9 were excluded as a worker's first trial (trials [0, 1, 2, 3, 4, 5, 6, 7, 1593]).

| tok/s | objective | trial | pdbs | peak GiB | params | anchor |
|---:|---:|---:|---:|---:|---:|:--|
| 621,200 | 4.41599 | 151 | 24 | 4.49 | 2,838,220 |  |
| 605,747 | 4.28450 | 298 | 24 | 4.49 | 2,951,308 |  |
| 550,593 | 4.21302 | 713 | 24 | 4.49 | 2,931,246 |  |
| 509,241 | 4.20945 | 422 | 24 | 4.49 | 2,955,534 |  |
| 465,395 | 4.20084 | 1166 | 24 | 4.49 | 2,872,588 |  |
| 456,101 | 4.19151 | 1795 | 24 | 4.49 | 2,963,856 |  |
| 419,203 | 4.18425 | 898 | 24 | 4.49 | 2,968,354 |  |
| 417,547 | 4.18055 | 715 | 24 | 4.49 | 2,956,208 |  |
| 415,139 | 4.17549 | 717 | 24 | 4.49 | 2,956,208 |  |
| 393,012 | 4.16774 | 849 | 24 | 4.49 | 2,956,208 |  |
| 307,521 | 4.16526 | 1052 | 24 | 4.49 | 3,030,424 |  |
| 289,632 | 4.15355 | 1028 | 24 | 4.49 | 3,030,424 |  |
| 216,138 | 4.15179 | 1794 | 24 | 4.49 | 3,080,356 |  |

## 5. Shortlist for confirmation (NOT a winner)

A SHORTLIST FOR CONFIRMATION, not a ranking and not a result. Each row answers a different question, and rows whose `resolved_vs_reference` is false lead the reference by less than this study can measure -- they are on the list because their SLOT matters, not because they won. Read `cannot_establish` before using any of them.

### `best_loss`

Lowest held-out loss. The obvious candidate and the most likely to be luck: it is the minimum of ~230 draws, so it carries the largest selection bias of any row here. Confirm it, do not believe it.

- trial **1794**
- objective **4.15179**, lead over reference - -> **NO FLOOR**
- run_id `ef35825a`, seed 42, params 3,080,356, 2.161e+05 tok/s, 4.49 GiB peak
- SKA indices [3, 4, 6, 7, 8, 9], rank 32, K 1, ridge 0.02285, lr 0.005406

### `best_cost_adjusted`

Best loss among trials on the loss/throughput Pareto front, i.e. the best config that is not also the slowest. Kept separate from best_loss because the objective is pure loss on purpose -- a `throughput_penalty` would bake one exchange rate into every later proposal, unrecoverably.

- trial **1794**
- objective **4.15179**, lead over reference - -> **NO FLOOR**
- run_id `ef35825a`, seed 42, params 3,080,356, 2.161e+05 tok/s, 4.49 GiB peak
- SKA indices [3, 4, 6, 7, 8, 9], rank 32, K 1, ridge 0.02285, lr 0.005406

### `best_k1`

Best config at ska_power_K=1. K is the axis most likely to change the SIGN of another axis's effect, so a confirmation study needs the best candidate at EACH K rather than whichever K happened to win here.

- trial **1794**
- objective **4.15179**, lead over reference - -> **NO FLOOR**
- run_id `ef35825a`, seed 42, params 3,080,356, 2.161e+05 tok/s, 4.49 GiB peak
- SKA indices [3, 4, 6, 7, 8, 9], rank 32, K 1, ridge 0.02285, lr 0.005406

### `best_k2`

Best config at ska_power_K=2, for the same reason.

- trial **542**
- objective **4.22386**, lead over reference - -> **NO FLOOR**
- run_id `ba55abb1`, seed 42, params 3,030,428, 2.842e+05 tok/s, 4.5 GiB peak
- SKA indices [3, 5, 7, 9], rank 32, K 2, ridge 0.009952, lr 0.005379

### `best_low_capacity`

Best config in the bottom half of the parameter range. If it is within the noise floor of the best overall, the capacity axis did not earn its parameters and the confirmation study should be cheaper.

- trial **849**
- objective **4.16774**, lead over reference - -> **NO FLOOR**
- run_id `18fdce60`, seed 42, params 2,956,208, 3.93e+05 tok/s, 4.49 GiB peak
- SKA indices [3, 5, 7, 9], rank 32, K 1, ridge 0.01187, lr 0.005479

### `interaction_probe`

A config chosen to TEST the largest resolved interaction rather than to win: it sits at the corner that interaction predicts is good, which is the only way a confirmation run can falsify it. If the interaction is an artefact of adaptive sampling, this is the row that says so.

**Unfilled**: no prespecified interaction is resolvable against the noise floor, so there is nothing to probe. That is itself the study's answer to its own question: at 600 steps on this proxy, no pair of these axes interacts by more than one configuration varies against its own seed.

## 6. What this study CANNOT establish

Written down rather than left to judgement, because the failure mode is a reader taking the shortlist's first row as a result.

- **Which configuration is best.** 600 steps at 25.35M parameters is a SCREEN. Rankings at 600 steps and at convergence differ systematically, not randomly: a config that warms up fast is rewarded here whether or not it ends better, which is why `prune_after_step` is 450 of 600 rather than something cheaper. The output is a shortlist for confirmation and there is no defensible single winner in it.

- **That anything here transfers to 50m.** The proxy keeps 50m's depth (17) and SKA placement ([3,7,11,15]) and narrows d_model from 384 to 256, so every effect measured here is measured at 2/3 the width. Effects that interact with width -- which includes the norm-clip multiplier and the layerscale, both of which act on per-head quantities -- can change sign between the two. `d_state` is 48 here against 50m's 64, which is a SECOND unresolved difference and is flagged as an open question rather than quietly treated as immaterial.

- **How much SKA earns its place, beyond that it does.** This one is PARTLY settled and the correction is worth recording. An earlier 4m-geometry smoke study measured an ablation delta of 1.17e-4 and raised the worry that this study might be measuring SKA's hyperparameters without establishing that the branch does anything. Job 445689 settles the sign: on THIS base at 600 steps the delta is 1.17e-2 to 7.48e-2 depending on the layerscale, a hundred times larger, and positive throughout -- removing SKA makes the model worse. The 4m number was a scale artefact. What remains unestablished is the MAGNITUDE at any scale that matters: 25.35M parameters and 600 steps is not 50m and it is not convergence, and an ablation delta measured on a model that has barely left its warmup transient is not the delta at convergence. Compare the ska_delta column to the noise floor directly, and do not read its size as a claim about a trained model.

- **Any effect smaller than the measured noise floor.** Not 'probably not real' -- unmeasurable by this study, at any trial count, because it is inside what one configuration does to itself when the seed changes. More trials narrow the sampler's search, not this.

- **Non-determinism, or seed sensitivity at other configurations.** The base spec sets `deterministic: true` and the replicates are all at ONE configuration. So the floor is a deterministic-mode floor at the reference point; a non-deterministic run spreads more, and a badly-conditioned corner of the space (rank 8 with a light ridge, say) plausibly spreads more still. Applying one floor to the whole space is an assumption, and it is the optimistic one.

- **Statistical significance of anything.** Fourteen prespecified pairwise tables, nine axes, one seed per sampled trial, and cell counts that are uneven BY CONSTRUCTION because the sampler was adaptive. 'Resolved' here means 'larger than seed noise', with no multiplicity correction and no model of the sampling process. It is a filter for what to confirm, not a test.

- **Interactions among the fixed axes, or between them and anything.** `weight_decay`, `warmup_ratio` and `grad_clip` are singletons in this study, so their columns are constant and every table over them is degenerate by design, not by accident.

