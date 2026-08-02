# Phase 2a experiment contract

Status: infrastructure-ready draft. Scientific runs are blocked until every
preflight item marked `BLOCKED` in `INTEGRATION_MATRIX.md` is resolved.

This contract translates Phase 2a of `reference/Scaling_Echo_3B.pdf` into one
reproducible study. It deliberately keeps paper Echo and the proposed updated
architecture separate.

## The 1M decision

`configs/1m.yaml` is the four-layer, vocabulary-128 synthetic model used for the
paper's Table 2. It cannot support a 32K-token WikiText perplexity objective, and
15%, 20%, 25%, and 33% SKA mostly collapse to the same layer count.

Phase 2a therefore uses `configs/phase2a_1m_core.json`:

- 16 sequence layers, `d_model=64`, four SKA heads, `d_state=16`
- tied 32,000-token embedding and sequence length 2,048
- 2, 3, 4, or 5 SKA layers for the four requested fractions
- approximately 0.85M-1.10M non-embedding parameters over the sweep
- approximately 2.90M-3.15M total parameters after the 2.048M tied embedding

The study label is **1M-core**, not 1M-total. Every run must log both counts and
use core parameter count and measured FLOPs as covariates. This definition must
be approved by the team before the first scientific run. Changing it creates a
new study version; it must never silently alter an existing study.
The portable materializer's count excludes not-yet-integrated extension
parameters such as beta/QKNorm. The finalized adapter must count the built
model, enforce the same band, and fail before training if the core or total
estimate drifts by more than 5%. It also records analytic and profiler-measured
training FLOPs per optimizer step; their relative drift may not exceed 15%.
Every healthy screen/final result carries these values plus cumulative actual
GPU-seconds/GPU-hours so later batch planning uses observed rather than
requested compute.

## Architecture modes

The modes are not interchangeable hyperparameters.

1. `updated_sweep` is the Scaling Echo candidate: beta write gate, LayerScale,
   short convolution, and optionally QKNorm. All normal search trials use it.
2. `paper_control` freezes the paper's SKA semantics: sequence-max
   normalization, no beta parameter, two-sided whitening, exclusive
   cross-chunk boundary, FP32 statistics and linear algebra, `K=2`, and ridge
   `1e-3`. At 1M core it uses the explicitly declared 25% middle-clustered
   layout; it is a matched-scale mechanism control, not a literal Table 4
   model-size/topology reproduction.
3. `mamba_only` is the canonical all-Mamba-2 + SwiGLU baseline.
4. `transformer` is the canonical all-causal-attention + SwiGLU baseline.

The two standard baselines share the Phase 2 config, tokenizer, packed data,
optimizer protocol, token budget, and evaluation harness. They contain no SKA
or Koopman MLP modules. Their parameter counts are reported separately; the
team must still approve whether final comparison matching uses total
parameters, non-embedding parameters, or measured training FLOPs.

Beta initialization applies only to `updated_sweep` and means
`beta_bias = logit(p)`, so `sigmoid(beta_bias)` equals 0.1, 0.3, or 0.5.
Diagnostics and optimizer grouping must tolerate the complete absence of beta
parameters in paper mode.
The scaling document searches LayerScale/short-conv initializations but does
not include an off value, so those components are always enabled in candidate
trials and disabled only in the paper control.

Candidate placements never use layer 0 or layer 15: first and last are Mamba-2.
Fixed-control layouts are explicit in the search spec and are not inferred from
the candidate placement policy.

Placement names have exact algorithms:

- `uniform`: quantiles evenly spanning all interior layers.
- `late_biased`: quantiles evenly spanning the final 60% of interior layers.
- `middle_clustered`: quantiles evenly spanning 33%-75% model depth.

Indices are zero-based, sorted, unique, materialized in every manifest, and
covered by tests. Fraction-to-count uses round-half-up:
`floor(n_layers * fraction + 0.5)`.

## Initial search space

The authoritative machine-readable definition is
`configs/phase2a_search.json`.

| Axis | Initial values |
|---|---|
| SKA rank | 32, 48, 64, 96, 128 |
| SKA fraction | 15%, 20%, 25%, 33% |
| placement | uniform, late-biased, middle-clustered |
| chunk size | 32, 64, 96, 128 |
| ridge epsilon | 1e-4, 1e-3, 1e-2 |
| beta init probability | 0.1, 0.3, 0.5 |
| LayerScale init | 1e-5, 1e-4, 1e-3 |
| short-conv width | 4, 8, 16 |
| short-conv gate init | 1e-3, 1e-2, 1e-1 |
| QKNorm | off, on |
| SKA projection LR ratio | 2.5, 5, 10 |
| gamma/eta LR ratio | 5, 10, 25, 50 |
| Koopman eigenvalue LR ratio | 5, 10, 20 |
| norm LR ratio | 0.25, 0.5, 1 |
| MLP LR ratio | 0.75, 1.5, 3 |
| Mamba/backbone LR ratio | 0.5, 1, 2 |
| objective | next-token only, Birdie-style mixture |
| Birdie retrieval fraction | 0.1, 0.2, 0.3, 0.4, 0.5 |
| Birdie copy share | 0.25, 0.5, 0.75 |

Only gamma/eta's grid is literal from the scaling document. The other
learning-rate grids are provisional log-scale grids centered on its reference
ratios. The fixed base LR is provisionally `1e-5`; this is an explicit approval
item, not a hidden default. DeltaNet is not a parameter group unless a future
backbone arm actually contains DeltaNet.

For a Birdie trial with retrieval fraction `r` and copy share `c`:

```
w_ntp    = 1 - r
w_copy   = r * c
w_infill = r * (1 - c)
```

Weights must sum to one. Data generation, loss normalization, and token budgets
must make objective arms comparable.

## Training and pruning

- Optuna grouped multivariate TPE, sampler seed 42, target 2,000 fresh sampled
  configurations. Checkpoint-recovery rows replay an existing immutable
  manifest and do not consume that quota. Grouped decomposition keeps
  conditional Birdie subspaces from being silently reduced to independent
  sampling.
- Effective batch 96 sequences, length 2,048, BF16.
- One GPU, microbatch 24, four accumulation steps; AdamW
  `(beta1=0.9, beta2=0.95, epsilon=1e-8)`, fused implementation, weight decay
  0.1, cosine-to-zero schedule, 2% warmup, and gradient clipping at 1.0.
- Step means an optimizer update, never a microbatch.
- Evaluate the frozen 20-cell MQAR grid at optimizer step 500.
- Once 20 comparable completed trials exist, prune when the step-500 MQAR
  macro-average is below their median.
- Compare pruning medians only within the same fidelity and architecture mode.
- Nonfinite loss, numerical corruption, missing metrics, or failed Cholesky are
  failed trials, not merely low-performing pruned trials.
- Survivors provisionally train to step 6,000, matching the paper's
  sub-million schedule. This budget requires team/compute approval.
- Controls and promoted trials run seeds 42, 43, and 44.

Because final selection is multiobjective while the pruner needs one scalar,
MQAR is the pruning scalar and the final pair remains:

1. maximize MQAR macro accuracy;
2. minimize frozen WikiText-103 perplexity.

No scalar blend is allowed for final ranking.
The Optuna study itself is single-objective `maximize MQAR` so native
`MedianPruner` semantics remain valid. Final WikiText PPL is stored as a trial
attribute/artifact and Pareto selection is computed post hoc.
This means TPE does **not** actively explore the perplexity objective. The
post-hoc MQAR/PPL Pareto front may therefore be under-sampled, especially at
the low-perplexity end. Treat that limitation as part of the study conclusion;
do not describe the resulting front as a multiobjective-optimized frontier.

## Evaluation and diagnostics

The MQAR grid is the Phase 0/Scaling Plan grid:

- sequence lengths 256, 512, 1,024, 2,048;
- 4, 8, 16, 32, 64 key-value pairs.

Save all 20 cells, macro average, hard-cell average, worst hard cell, seed,
generator version, and sample count. The hard subset is exactly
`{length_1024,length_2048} x {pairs_32,pairs_64}`; ties for worst cell use the
lexicographically first cell ID. This is not the paper Table 2 protocol, whose
cells and independent-training setup differ.

The existing repo-local MQAR generator is provisional and is not yet an exact
Zoology implementation. The exact-fit 256-by-64 cell is now retained, but
scientific launch still requires a pinned generator plus oracle fixtures.

WikiText evaluation freezes dataset revision, tokenizer revision, tokenized
checksum, packing behavior, and maximum token count. The placeholder revisions
in the search spec intentionally fail preflight.

Each trial must emit the diagnostics listed in the search spec. Phase 1
diagnostics are evidence, not standalone objectives. The hard numerical gates
are normalized-pre-γ radius at most 1.0001, state-rebuild parity at most 1e-4,
chunk-boundary parallel/decode parity at most 1e-4, and no nonfinite required
metric. Within-chunk drift at offsets 1, floor(chunk-size/2), and chunk-size-1
is informational while `exact_intrachunk=false`. The applied-post-γ radius is
logged separately and may exceed one.

The requested ridge is also observable evidence, not merely a config field.
Each SKA head reports the actual effective ridge extrema. With zero Cholesky
escalations both extrema must equal the requested `ska_ridge`; any increase
must have an explicitly positive escalation count. The integration gate must
instrument the final operator to rule out the current hidden fixed increment
before scientific launch.

Because rank and SKA fraction change actual capacity, every importance report
contains both ordinary conditional fANOVA and a capacity-adjusted view that
residualizes each target on actual non-embedding parameter count and measured
training FLOPs. The parameter/FLOP ranges stay visible in Pareto and W&B
tables; the nominal “1M-core” label is not a substitute for these covariates.

Diagnostics are not reduced to one model-wide scalar. Each realized SKA layer
contains one record per head for raw/normalized/applied radius, clamp factor
and fraction, spectral gap, lambda-min, ridge conditioning, and ridge
escalations, plus layer-level residual, gradient, effective-Jacobian, beta, and
LayerScale metrics. W&B-friendly aggregates are retained alongside the nested
records.
Final survivors also report all four load-bearing perplexities: full,
SKA-zeroed, Mamba-zeroed, and both-zeroed. The step-500 screen requires the
health/gradient subset but not the four expensive ablations.
Per-head records include means and extrema over sampled batch/chunks; hard
gates use normalized-pre-γ maxima and lambda-min minima, not averages.
Model-wide radii are maxima of the per-head maxima, model-wide
`lambda_min_gram` is the minimum of per-head minima, and the named mean
aggregates are equal-weight means of their per-head or per-layer records. Ridge
condition is the maximum per-head maximum; ridge escalation is the sum of
per-head counts. All domains and reductions are enforced by
`validate_step_metrics`; nulls are accepted only for explicitly mode-approved
N/A metrics. The effective Jacobian rank must be a true input-output
diagnostic. Phase 1's key-projection gradient-rank proxy cannot be relabeled as
the Jacobian rank.

## Identity and artifact contract

`trial_hash` is SHA-256 over all outcome-affecting inputs:

```
model config + realized layer indices
+ optimizer groups and effective LRs
+ objective weights and generators
+ tokenizer/data/evaluator manifests
+ seed and training budget
+ code commit and capability version
```

The model-only config hash is retained but cannot identify a trial.
`promotion_config_hash` covers the same outcome-affecting identity with only
the run seed removed. It is the sole key used to group seeds 42/43/44;
displayed parameter strings or the model-only hash are not sufficient.

Every trial directory contains:

- immutable `trial_manifest.json` and `model_config.json`;
- Git commit and dirty flag;
- requested and realized SKA fraction/indices;
- model/core parameter counts and estimated/measured compute;
- scale-band and estimator-drift outcomes plus cumulative GPU time;
- optimizer group names, parameter names/counts, LR, weight decay, exhaustive
  coverage/exclusivity evidence, and recomputable SHA-256 digests;
- tokenizer, data, and evaluator revisions/checksums, including actual
  train/WikiText/MQAR byte sizes and streamed file SHA-256 values;
- frozen sampler revision, shuffle seed, shard-order manifest/checksum,
  epoch/repeat behavior, deterministic worker partitioning, and resumable
  global sequence offset;
- frozen MQAR vocabulary identity and token-map revision/checksum;
- an approved runtime lock whose Python, OS kernel, package, PyTorch, CUDA,
  cuDNN, driver, and GPU properties match the live worker;
- step-500 and final metrics JSON;
- checkpoint/resume provenance binding the final stage to a controller-hashed,
  immutable step-500 file/directory bundle (path, byte/file counts, canonical
  digest), optimizer/scheduler/RNG restoration, and global data sequence
  offset;
- Phase 1 diagnostics, throughput, peak memory, and wall time;
- W&B run ID, Optuna trial ID, exit status, and structured failure reason;
- checkpoint only for controls, survivors, or promoted configurations.

Study-level outputs are the offline W&B payload, Pareto table, per-objective
fANOVA results, control false-pruning-risk report, promotion manifest, seeded
stability report, confirmed shortlist, and falsified-hypotheses scaffold.
Optuna storage snapshot receipts, W&B publication-verification receipts,
reviewed-hypothesis receipts, and an automatic next-batch planner are deferred
until after the pilot. They are operational reporting hardening rather than
inputs to the initial 1M sweep. Concurrent target-guard rows remain visible in
Optuna but never count toward the approved fresh quota.
Checkpoints, datasets, token caches, databases, and Slurm logs remain untracked.

## Promotion

1. Keep stable, nondominated Pareto trials.
2. Re-run leading candidates at three seeds.
3. Require confidence intervals to remain competitive.
4. Promote roughly 8-12 diverse configurations, plus all controls, to 5M/20M.
5. Run fANOVA separately for MQAR and perplexity and within conditional
   subspaces where appropriate.
6. If a top-three 1M configuration falls below the median at 20M, fix that axis
   instead of continuing to sweep it at larger scales.

Phase 2a output is not a single winner. It is axis importance, a Pareto front,
top configurations, falsified hypotheses, and a defensible promotion list.

The analysis artifact retains a global fANOVA view and adds explicit
`birdie_only`, `next_token_only`, and `updated_sweep_only` views. This is
necessary because Optuna's global parameter intersection omits conditional
parameters such as `birdie_retrieval_fraction` and `birdie_copy_share`.
Perplexity fANOVA uses only completed trials with a finite positive PPL;
missing PPL is reported as a view count and does not remove the same trial from
MQAR analysis.

Promotion selection is deterministic: exclude any explicitly failed
health/stability record, take the stable Pareto set, retain the highest-MQAR
and lowest-PPL anchors, then greedily add candidates with the greatest
categorical architecture distance. Preparation-time analysis may retain a
record whose stability evidence has not yet been merged, but it labels that
evidence `not_recorded`. The final promotion artifact must be regenerated with
the default strict stability gate, capped near ten, after the three-seed
checks. `--allow-missing-stability-evidence` is exploratory only.

The initial shortlist is a candidate list, not a claim of seed stability.
`koopman-phase2-promotions` materializes its missing seeds under the same
seedless configuration identity. Final analysis emits a separate
`confirmed_promotion_shortlist.json`; a configuration is eligible only when
all three required seeds complete with explicit passing health evidence.

The falsified-hypotheses output is deliberately a review scaffold. fANOVA
importance alone never marks a hypothesis falsified; the reviewer must attach
seeded matched comparisons, uncertainty, numerical-health evidence, and
larger-scale evidence where applicable. The W&B-ready payload is offline JSON
(`network_write_performed=false`) and requires a separate authorized upload.
