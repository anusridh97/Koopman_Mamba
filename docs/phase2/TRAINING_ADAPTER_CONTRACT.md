# Final training adapter contract

The only architecture-dependent Phase 2 component should be the executable
`koopman-phase2-trial-worker`. The Optuna controller invokes it as:

```text
koopman-phase2-trial-worker
  --manifest PATH
  --run-dir PATH
  --stage screen|final
  --metrics-out PATH
```

It must not resample or override the manifest.
It must also assert the pinned optimizer betas/epsilon, fused mode, scheduler,
minimum LR, microbatch/accumulation/world size, clipping order, checkpointing
mode, and loss normalization instead of inheriting trainer defaults.

## Model mapping

`model_config.json` contains fields understood by
`koopman_lm.globals.config.KoopmanLMConfig`. The pinned architecture base also
adds `ska_inverse_cholesky`; it must remain off unless a future, explicitly
versioned Phase 2 axis selects the small-rank per-token path.
`trial_manifest.json:model_extensions` contains fields that
must be mapped after the final architecture lands:

| Extension | Required canonical behavior |
|---|---|
| `stats_mode=candidate_beta_gated` | candidate causal statistics plus beta write gate |
| `beta_init_probability` | initialize with `bias=logit(p)` |
| `qk_norm` | apply the team-approved QKNorm/BCNorm definition |
| `stats_mode=paper_sequence_max` | shared sequence-max key/query scale; freeze prefill scale during decode |
| `write_gate=none` | beta parameter is absent, not initialized near zero |
| `whitening=two_sided` | `L^-1 M L^-T` |
| `cross_chunk_boundary=exclusive` | no current-token leakage into the boundary term |
| `statistics_dtype=float32` | covariance accumulation and linear algebra remain FP32 |
| `chunk_semantics=strict_causal_chunked` | requested chunk size reaches the active chunked path; do not select a backend that ignores it |

Paper mode must ignore candidate LayerScale/short-conv/QKNorm behavior.
Mamba-only mode must call `build_mamba_only` and Transformer mode must call
`build_transformer`. Both use SwiGLU channel mixers and contain no SKA or
Koopman MLP modules; they are not Echo models with branches zeroed at runtime.

After building the finalized model, count actual total and non-embedding
trainable parameters, including beta/QKNorm and other extension parameters.
Tied embeddings count once. Emit
`model_accounting.parameter_counts_actual`; the central validator recomputes
`total = embedding + non_embedding_core`, checks the actual total against the
manifest's 3M-total band, requires the embedding count to match the model-config
estimate, and compares actual core/total counts with the provisional manifest
estimator. The relative drift ceiling is 5%. An out-of-band model or larger
estimator drift fails before promotion.

Report both an architecture-aware analytic FLOP estimate and a profiler-derived
measurement over one full optimizer update (all accumulation microbatches).
Both use the scope
`training_forward_backward_per_optimizer_step`; optimizer bookkeeping kernels
are excluded. Include the estimator/profiler method and number of measured
steps. The central validator recomputes
`abs(estimated - measured) / measured` and enforces the manifest's 15% drift
ceiling.

Each `screen` or `final` adapter payload reports actual GPU-seconds and
GPU-hours for that adapter invocation only. Do not include earlier stages or
preemption attempts. GPU time is wall time multiplied by active world size;
the validator checks `gpu_hours = gpu_seconds / 3600` and that reported world
size matches the manifest. The controller persists every validated successful
stage in the claim, persists each recoverable systemic attempt separately, and
sums those records exactly once into the terminal trial summary. During the
blocking adapter process, the claim heartbeat maintains a durable wall-time ×
world-size estimate; hard preemption is charged through the last heartbeat and
therefore has at most one heartbeat interval per GPU of unobserved tail time.
The adapter's validated measurement replaces that estimate on a normal return.
A structured trial-local failure envelope must likewise report the failing
invocation's GPU seconds/hours. Parameter, FLOP, and GPU-time accounting are
result evidence, not W&B-only metadata.

Before launch, a capability test must trace or instrument all four chunk sizes
and prove the selected backend consumes them. In the current globals code,
`ska_exact_intrachunk` and `ska_inverse_cholesky` bypass the standard chunked
path, so Phase 2a materializes both as false. Silently enabling either path
would invalidate the chunk-size axis.

## Optimizer mapping

The manifest provides effective LRs and conceptual ownership. The adapter must
assign actual parameters by module/type semantics and assert:

- every `requires_grad` parameter appears exactly once;
- no parameter appears in two groups;
- inactive/absent component groups are omitted;
- norm parameters use zero weight decay;
- fixed eta/gamma are not placed in an optimizer group;
- beta, LayerScale, and short-conv gate ownership is logged explicitly.

The searched gamma/eta multiplier owns only gamma and eta. Beta's learned
linear projection belongs to SKA projections, LayerScale belongs to the
zero-weight-decay norm/scale group, and the short-conv scalar gate uses the
fixed base LR. Do not silently make the gamma/eta axis change all four
mechanisms.

Write a machine-readable group audit containing every trainable parameter's
fully qualified name and `numel`, tensor/scalar counts, LR, weight decay, and
the conceptual manifest group it realizes. Parameters are sorted by name and
actual groups by `group_name`. Per-group parameter digests are SHA-256 over
canonical JSON `[{"name": ..., "numel": ...}, ...]`; the coverage digest is
SHA-256 over the canonical, name-sorted assignment rows containing name,
numel, actual group, manifest group, LR, and weight decay. The central
validator recomputes every count and digest, checks LRs against the manifest,
rejects unknown conceptual groups or invalid weight decay, and requires empty
unassigned/duplicate lists. Assigned scalar coverage must equal the actual
total trainable-parameter count.

## Objective mapping

For `next_token_only`, use ordinary next-token cross entropy.

For `birdie_mix`, use the exact normalized weights from the manifest:

```text
w_ntp + w_selective_copy + w_infilling = 1
```

The adapter must use pinned selective-copy and infilling generators, equal token
budgets, documented loss normalization, and deterministic sample identities.
Evaluation data must never enter the training objective.

## Stage semantics

`screen`:

- starts from initialization or resumes an incomplete screen checkpoint;
- stops exactly after optimizer update 500;
- saves model, optimizer, scheduler, gradient-scaler if used, RNG states,
  sampler/epoch, and exact global sequence offset atomically;
- evaluates the frozen 20-cell MQAR grid; step-500 WikiText is optional;
- records a normalized run-relative checkpoint bundle path, total byte/file
  counts, the canonical whole-bundle SHA-256, checkpoint-format revision,
  optimizer step, and global sequence index in `checkpoint_provenance`;
- leaves that bundle immutable; the controller independently hashes every
  regular file and rejects symlinks, path escapes, empty files, or a mismatch;
- writes `metrics/step_500.json`.

`final`:

- requires the complete screen checkpoint;
- resumes without resetting optimizer, scheduler, RNG, or data stream;
- records the source checkpoint bundle path/byte count/file count/SHA-256,
  source step/format, restored global sequence index, and explicit
  optimizer/scheduler/RNG/data-stream restoration booleans in
  `resume_provenance`; the controller rehashes the unchanged source bundle
  after the final-stage process exits;
- stops at `manifest.fidelity.max_steps`;
- evaluates identical metrics and writes `metrics/final.json`.

Step is always an optimizer update. Gradient accumulation microsteps do not
advance it.

## Result mapping

Output is checked by `validate_step_metrics`. A standalone JSON Schema is
intentionally deferred until the architecture and metric payload are final.

Every healthy screen/final payload contains `model_accounting`,
`optimizer_group_audit`, `gpu_seconds_actual`, and the canonical
`gpu_hours_actual` for that stage invocation. It also contains a nonempty
`wandb_run_id`, the full 20-cell MQAR grid, the four-cell hard average, worst
hard cell/value, and the exact frozen MQAR seed, generator revision, and
samples-per-cell. The controller copies the
accounting/audit blocks and writes cumulative, exactly-once GPU seconds/hours
to `trial_summary.json`. Every summary, including `FAIL`, carries cumulative
GPU seconds/hours (zero is allowed only when no GPU work began) so recovery and
batch planning can enforce the approved compute ceiling.

All required diagnostic keys must exist. Truly absent quantities are JSON
`null` and named in `not_applicable_diagnostics`; only mode-approved absences
are accepted. The Mamba-only, Transformer, and paper controls therefore do not need fake beta
statistics.

`diagnostics_by_layer` must exactly match the manifest's realized zero-based SKA
indices. Each layer contains `per_head` records for radius/clamp/gap/lambda-min
and a `layer` record for residual contribution, gradient ratio, and effective
Jacobian rank. Final output also includes full/SKA-zero/Mamba-zero/both-zero
WikiText perplexities with an arithmetically consistent SKA-zero delta.
Per-head records retain mean and max/min extrema over sampled batch/chunks.
Spectral gates apply to `spectral_radius_normalized_pre_gamma_max`, after the
clamp but before learned γ, and conditioning gates apply to
`lambda_min_gram_min` relative to the effective ridge used for that head. The
separately logged
`spectral_radius_applied_post_gamma_*` may legitimately exceed one when γ
restores scale. `layerscale_magnitude_mean` is the mean absolute LayerScale
coefficient, not a signed mean that can cancel to zero. Do not substitute the existing
key-projection-gradient rank proxy for a true effective Jacobian rank.

Ridge reporting is literal. `ridge_requested` must equal
`model_config.ska_ridge`; each head reports the minimum and maximum
regularization coefficient that actually reached the operator. Effective ridge
cannot be below the request. If it differs from the request, the corresponding
`ridge_escalation_count` must be positive; if the count is zero, min and max
must equal the request. Model-wide effective-ridge min/max are the extrema over
heads and escalation counts sum. A fixed hidden ridge increment is therefore a
contract failure, not an implementation detail. Correctness evidence must
instrument the operator and prove the literal requested value is the normal
path before scientific launch.

Decode parity is three separate measurements:

- `state_rebuild_decode_prefill_max_abs_error` rebuilds recurrent state from
  the same prefix and compares the first continuation token; it is hard-gated
  at `1e-4`.
- `chunk_boundary_parallel_decode_max_abs_error` uses prompt lengths that are
  integer multiples of the active chunk size and compares the first
  continuation token from parallel prefill versus recurrent decode; it is
  hard-gated at `1e-4`.
- `within_chunk_parallel_decode_drift_max_abs_error` repeats the comparison at
  within-chunk offsets `1`, `floor(chunk_size / 2)`, and `chunk_size - 1`.
  It is required and nonnegative but informational for the current
  `exact_intrachunk=false` sweep.

If a future study enables `exact_intrachunk=true`, it must version the contract
and promote the within-chunk drift metric to a hard gate.

The model-wide aggregate reductions are deterministic:

- raw, normalized-pre-γ, and applied-post-γ radii are the maximum of all
  per-head maxima;
- `clamp_factor_mean`, `clamp_fraction`, and `spectral_gap_mean` are equal-weight arithmetic
  means of the corresponding per-head means;
- `spectral_gap_max` is the maximum per-head maximum;
- `lambda_min_gram` is the minimum per-head minimum;
- `ridge_condition_number` is the maximum per-head maximum and
  `ridge_escalation_count` is the sum of per-head counts;
- effective ridge min/max are the minimum/maximum of their per-head extrema;
- SKA branch norm ratio, SKA-to-Mamba gradient ratio, effective Jacobian rank,
  beta mean/saturation, and LayerScale magnitude are equal-weight means of
  their per-layer records.

Every head summarizes the same sampled batch/chunk set, so equal head weighting
is part of the contract. Aggregate equality uses `rel_tol=1e-6` and
`abs_tol=1e-8`. Radii, gaps, lambda minima, norm/gradient ratios, parity error,
LayerScale magnitude, and effective rank are nonnegative; mean extrema cannot
exceed maxima. Clamp factors lie in `(0, 1]`, probability/fraction metrics in
`[0, 1]`, condition-number means cannot exceed their maxima, condition numbers
are at least one, and ridge escalation/memory counters are
nonnegative/positive integers as applicable. Step time and every perplexity are
strictly positive. Effective Jacobian rank cannot exceed `d_model`.

Paper controls report beta and LayerScale aggregates as null/N/A. Mamba-only and Transformer
controls report all SKA-specific aggregates as null/N/A and emit an empty
`diagnostics_by_layer` object. Metrics outside the mode-approved N/A set may
never be hidden with null. Paper-control per-layer beta and LayerScale records
are also null; candidate-mode records must be finite.

Writes must be atomic and strict JSON: no NaN or Infinity. Numerical corruption
is a failed trial. A healthy below-median trial is pruned. These statuses must
never be collapsed.

The adapter distinguishes a sampled configuration that failed from a broken
worker by writing a structured trial-local failure to the requested metrics
path:

```json
{
  "schema_version": 1,
  "trial_hash": "<64-hex manifest trial hash>",
  "status": "failed",
  "gpu_seconds_actual": 123.4,
  "gpu_hours_actual": 0.03427777777777778,
  "failure": {
    "scope": "trial",
    "code": "nonfinite_loss",
    "detail": "loss became non-finite at optimizer step 31"
  }
}
```

Known per-configuration outcomes such as non-finite loss, OOM after the
manifest's fixed batch settings, or exhausted Cholesky/ridge recovery may use
`scope=trial`. Both `code` and `detail` are required. Every failure envelope
also reports the GPU time consumed by that adapter invocation, with
`gpu_hours_actual == gpu_seconds_actual / 3600`; this is required even when the
failure happens before the first optimizer step. Only this exact envelope is
caught by Optuna and recorded as `FAIL`; the worker then continues to its next
trial. Missing or malformed output, an unstructured nonzero exit, and
`scope=systemic` stop the worker immediately. Environment, dataset,
checkpoint-format, command-line, and controller/adapter contract errors are
systemic and must never be mislabeled as bad hyperparameters.

## W&B mapping

Each run logs:

- `trial_hash`, `model_hash`, spec hash, Git commit, dirty flag;
- Optuna trial/study IDs;
- requested and realized layer layouts;
- core and total parameter counts, estimated/measured FLOPs, and actual GPU hours;
- optimizer group audit;
- objective weights;
- data/tokenizer/evaluator fingerprints;
- Phase 1 diagnostics and all evaluation metrics.

The W&B group is the study name; the run name includes Optuna trial number and
the first 12 trial-hash characters. Return the same nonempty `wandb_run_id` in
both screen and final metrics JSON. The adapter must initialize/resume that
identity deterministically rather than creating a second run for the final
stage.

The trainer must consume `protocol.data_stream` and the identical
`data.training.stream` contract verbatim: sampler revision, shuffle seed,
frozen shard order, epoch/repeat policy, and world-size partitioning. Resume
must restore the exact global sequence index before constructing the next
batch; reconstructing a fresh shuffled iterator is a contract failure.
