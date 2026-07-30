# Phase 2a launch runbook

This runbook is intentionally fail-closed. Passing a dry run means the study
contract is coherent; it does not mean the current architecture is scientifically
launchable.

Use one environment vocabulary throughout:

```bash
export PHASE2_STUDY_NAME=echo-phase2a-1m-v1
export PHASE2_STORAGE_URL='postgresql+psycopg://...'
export PHASE2_OUTPUT_ROOT=/absolute/path/phase2a-scientific-runs
export PHASE2_CALIBRATION_ROOT=/absolute/path/phase2a-nonscientific-calibration
```

The calibration root must not be inside the scientific output root.

## 1. Freeze the integration commit

- Complete the integration order in `INTEGRATION_MATRIX.md`.
- Record the clean Git commit in the capability manifest.
- Record the finalized spec hash, canonical data-manifest SHA-256, and absolute
  evidence paths plus their byte-level SHA-256 values. Preflight verifies the
  current HEAD, clean worktree, report context/status/check list, approval
  metadata, the environment-lock file itself, and the active runtime against
  the contents of that lock.
- Resolve the architecture, 1M-core, base-LR, tokenizer, dataset, QKNorm, and
  training-protocol decisions needed for the non-scientific calibration.
- Leave only the four values calibration is intended to determine unresolved:
  total GPU-hours, maximum full-trial GPU-hours, required GPU type, and storage
  backend. The final survivor, storage, and compute approvals happen after the
  calibration measurements in section 5.

## 2. Freeze data once

- Pretokenize FineWeb-Edu in a standalone Slurm dependency job.
- Materialize the WikiText validation set and all MQAR samples once.
- Write immutable manifests with revisions, sample IDs, tokenizer revision,
  packing configuration, byte sizes, and SHA-256 checksums. Also freeze sampler
  revision, shuffle seed, shard-order manifest/checksum, epoch/repeat policy,
  and deterministic worker partitioning. Exact resume restores the global
  sequence offset into this stream.
- Materialize the tokenizer fingerprint, shard-order manifest, MQAR oracle,
  MQAR sample IDs, MQAR vocabulary/token map, and Birdie fixtures/sample
  identities as real read-only auxiliary files. Record each absolute path,
  positive byte size, and SHA-256 under `auxiliary_artifacts`; preflight
  rehashes them and cross-checks each digest against its protocol field.
  Auxiliary files must be nonempty UTF-8 JSON objects/arrays; bulk and
  auxiliary files must be regular, non-symlink files with all write-mode bits
  removed. No two artifacts may share a path, inode, or content SHA-256.
- Store MQAR as one frozen read-only artifact. Record its absolute path, byte
  size, SHA-256, vocabulary size/identity, and token-map revision/SHA-256.
  A generator revision or sample-ID digest alone is not an artifact identity.
- Make worker jobs read-only consumers of those artifacts.
- Use absolute existing paths for the train/validation/MQAR files; record
  positive integer byte sizes and token/sample counts. Every preflight,
  including each worker launch, streams all bulk and auxiliary files and compares their
  actual byte sizes and SHA-256 values. It also cross-checks tokenizer, dataset
  revisions, packing, the exact 20-cell MQAR grid, Birdie generators, loss
  normalization, and equal-token accounting against the search spec.
- Compute the canonical JSON identity recorded as
  `approved_data_manifest_sha256`:

```bash
koopman-phase2-preflight \
  --spec configs/phase2a_search.json \
  --data-manifest path/to/final_data_manifest.json \
  --print-data-hash
```

The canonical data-manifest hash is distinct from each artifact checksum inside
the manifest and from the raw-byte checksum of an evidence report.

Each evidence report uses schema version 1, `status=pass`, the exact evidence
type, integration commit, spec hash, and all required named checks enforced by
preflight. Data-validation and pilot reports also carry the approved canonical
data-manifest hash. The capability manifest stores each report's absolute path
and raw-file SHA-256; existence alone is never enough. Standalone JSON Schemas
are deferred until the architecture and payloads are final.

## 3. Freeze and verify the worker runtime

Generate the environment lock inside the same Slurm allocation shape used by
workers, after loading the final modules and environment, with exactly one GPU
visible. Do not create it on a login node:

```bash
koopman-phase2-runtime --write-lock /absolute/path/phase2a_runtime.lock.json
sha256sum /absolute/path/phase2a_runtime.lock.json
```

Put that absolute path and raw-file SHA-256 in
`evidence.environment_lock`. The lock contains its own canonical fingerprint
hash and records Python executable/version, OS kernel/libc/architecture,
the complete installed-distribution set plus critical package versions,
loaded compiler/CUDA modules, PyTorch build/ABI, CUDA/cuDNN/device properties,
CUDA toolkit, and driver versions. Verify a prospective worker environment:

```bash
koopman-phase2-runtime --verify-lock /absolute/path/phase2a_runtime.lock.json
```

Preflight performs the same strict comparison. Missing Torch/CUDA or required
training/kernel packages is a hard failure even if the bad environment was the
one captured in the lock. It imports the Mamba, causal-conv, Triton, Optuna,
scikit-learn, W&B, and PostgreSQL modules/extensions and executes a tiny CUDA
matrix multiplication; installed-version strings alone do not pass.
Hostnames, Slurm IDs, and CUDA device ordinals are intentionally excluded;
software, kernel, and GPU properties are not.

## 4. Run calibration preflight

```bash
python -m koopman_lm.experiments.phase2.preflight \
  --spec configs/phase2a_search.json \
  --capabilities path/to/calibration_capabilities.json \
  --data-manifest path/to/final_data_manifest.json \
  --stage calibration
```

Calibration is non-scientific. It permits unresolved compute ceilings, GPU
type, storage backend, the three approvals those measurements inform, and the
storage/pilot evidence reports. Architecture capabilities, final data,
correctness/resume/data evidence, approval metadata, clean commit, environment
lock, and exactly one visible GPU remain mandatory.

## 5. Dry-run and measure the calibration arms

```bash
python -m koopman_lm.experiments.phase2.dry_run \
  --spec configs/phase2a_search.json \
  --trials 32 \
  --output /tmp/echo-phase2a-dry-run
```

Inspect realized indices, parameter bands, conditionals, hashes, and control
coverage. No training is performed.

Materialize exactly three screen-only timing arms: the updated default, the
rank-128/ridge-1e-4/chunk-128/QKNorm stability extreme, and Birdie mix.

```bash
koopman-phase2-calibration \
  --spec configs/phase2a_search.json \
  --capabilities path/to/calibration_capabilities.json \
  --data-manifest path/to/final_data_manifest.json \
  --output "$PHASE2_CALIBRATION_ROOT"

koopman-phase2-run-manifest \
  --spec configs/phase2a_search.json \
  --capabilities path/to/calibration_capabilities.json \
  --manifest /path/to/one/calibration/trial_manifest.json \
  --stage screen \
  --preflight-stage calibration \
  -- koopman-phase2-trial-worker
```

`PHASE2_CALIBRATION_ROOT` must be outside `PHASE2_OUTPUT_ROOT`. Run these as at
most three one-GPU Slurm jobs with an explicit scheduler
walltime; they intentionally do not consume the not-yet-approved scientific
claim ledger. Use their measured GPU time, peak memory, and observed GPU name
to choose conservative per-trial and total ceilings (including the 20% retry
reserve), required GPU type, and concurrency, and to validate the timing
feasibility of the already fixed 6,000-step survivor budget.

Now update the versioned spec, choose/validate the distributed storage backend,
replace every remaining placeholder, record all approvals, and set
`status=scientific_ready`. Because this changes the spec hash, reissue every
context-bound evidence report and capability manifest against the final hash.
Calibration summaries are never scientific observations and are not included
in Phase 2a analysis.

For any materialized fixed plan, `scripts/slurm_phase2a_fixed_manifests_scg.sh`
reads its `execution_plan` by array index. Set `PHASE2_FIXED_ROOT`,
`PHASE2_PLAN`, `PHASE2_CAPABILITIES`, and `PHASE2_PREFLIGHT_STAGE`, then submit
the exact range (for calibration, `sbatch --array=0-2 --time=02:00:00 ...`).
Set `PHASE2_PLAN` to an absolute path inside `PHASE2_FIXED_ROOT`, or to a path
relative to that root; paths escaping the fixed root are rejected.
Study-stage controls/promotions additionally require `PHASE2_OUTPUT_ROOT` for
the shared budget ledger.

## 6. Pipeline-validation pilot

Before Optuna, run a fixed pilot containing:

- paper control, updated default, and Mamba-only;
- minimum and maximum rank;
- minimum and maximum chunk;
- rank 128 plus ridge 1e-4;
- QKNorm on and off;
- next-token and Birdie arms;
- every distinct placement realization.

First run `preflight --stage pilot` against the final spec. This stage permits
only `evidence.pilot_report` to be unresolved. Then run every fixed arm through
step 500 and finish the required subset through step 6,000. The fixed matrix
validates real metric schemas, accounting, exact resume, W&B cross-links, and
diagnostics. Separately, the required storage-concurrency report must exercise
the actual Optuna `report`/median-prune path, concurrent claims, hard-preemption
accounting, and restart idempotence; the fixed runner does not claim to test
Optuna pruning.

After preflight is green, rematerialize the pilot with finalized identities:

```bash
koopman-phase2-pilot \
  --spec configs/phase2a_search.json \
  --capabilities path/to/pilot_capabilities.json \
  --data-manifest path/to/final_data_manifest.json \
  --output /labs/mpsnyder/cody1212/phase2a-pilot

koopman-phase2-run-manifest \
  --spec configs/phase2a_search.json \
  --capabilities path/to/pilot_capabilities.json \
  --manifest /path/to/one/trial_manifest.json \
  --stage both \
  --preflight-stage pilot \
  -- koopman-phase2-trial-worker
```

The generated `pilot_summary.json:execution_plan` marks every arm as `screen`
or `final`. Run `--stage screen` for screen-only entries and `--stage both` for
final entries. Controls and the riskiest rank/ridge, QKNorm, and Birdie arms
must finish the final stage; all layout arms must finish at least the screen.

After all planned stages finish:

```bash
koopman-phase2-audit-pilot \
  --spec configs/phase2a_search.json \
  --capabilities path/to/pilot_capabilities.json \
  --data-manifest path/to/final_data_manifest.json \
  --pilot-root /labs/mpsnyder/cody1212/phase2a-pilot \
  --output /absolute/path/pilot_report.json
```

Add that report's absolute path and raw SHA-256 to the capability manifest,
then run the full study-stage preflight:

```bash
koopman-phase2-preflight \
  --spec configs/phase2a_search.json \
  --capabilities path/to/final_capabilities.json \
  --data-manifest path/to/final_data_manifest.json \
  --stage study
```

Only `stage=study` is accepted by the Optuna controller. TPE trials use
`koopman-phase2-study`; fixed pilot/control arms use `run-manifest`.

Materialize the three fixed controls at seeds 42, 43, and 44 only after the
study-stage preflight passes:

```bash
koopman-phase2-controls \
  --spec configs/phase2a_search.json \
  --capabilities path/to/final_capabilities.json \
  --data-manifest path/to/final_data_manifest.json \
  --output "$PHASE2_OUTPUT_ROOT/control-runs"
```

Execute every `control_summary.json:execution_plan` entry through
`koopman-phase2-run-manifest --preflight-stage study --stage both
--budget-root "$PHASE2_OUTPUT_ROOT" --storage "$PHASE2_STORAGE"`. The fixed
runner refuses to create a ledger: it requires the immutable ledger contract
already created by the Optuna study at that exact root and checks its
credential-free storage-instance hash, so a typo cannot split controls from
the sampled-run GPU-hour ceiling.

## 7. Scientific Phase 2a

- Use `postgresql+psycopg://` or SCG-validated JournalStorage, not shared
  SQLite. Runtime URL type must match the backend recorded in the spec.
- Use one trial per GPU; reserve DeepSpeed for the later larger-scale phases.
- Start with at least 20 unpruned startup trials.
- Launch the three fixed controls at all promotion seeds.
- Preserve failed, pruned, and completed statuses distinctly.
- Monitor failure rate, extreme-rank stability, diagnostic overhead, and false
  pruning of full-budget controls.
- Keep the Slurm array task count at or below
  `study.compute_budget.maximum_concurrent_trials`. The claim ledger enforces
  that ceiling atomically across Optuna workers, fixed controls, and promotion
  arrays—not only within one Slurm array. The controller refuses new sampled
  configurations beyond the approved 2,000-fresh-trial target; checkpoint
  recovery rows are tagged separately and do not consume that quota. Every
  running claim also reserves
  `study.compute_budget.maximum_full_trial_gpu_hours` atomically against the
  approved total ceiling.

Each array worker runs a small number of Optuna trials:

```bash
koopman-phase2-study \
  --spec configs/phase2a_search.json \
  --capabilities path/to/final_capabilities.json \
  --data-manifest path/to/final_data_manifest.json \
  --storage "$PHASE2_STORAGE_URL" \
  --output-root "$PHASE2_OUTPUT_ROOT" \
  --trials 1 \
  -- koopman-phase2-trial-worker
```

For the full study, submit bounded Slurm-array batches no larger than
`study.compute_budget.maximum_concurrent_trials`. Before each batch, inspect
the Optuna dashboard and the claim files under
`$PHASE2_OUTPUT_ROOT/.phase2_claims/`; stop when the approved fresh-trial target
or GPU-hour ceiling is reached. The controller repeats the reservation check
under a shared atomic lock, so concurrent workers cannot overspend the
approved ceiling.

The finalized `koopman-phase2-trial-worker` adapter must accept:

```
--manifest PATH --run-dir PATH --stage {screen,final} --metrics-out PATH
```

`screen` trains/resumes exactly to optimizer step 500 and writes the fixed MQAR
result (step-500 WikiText is optional). `final` resumes the same
optimizer/scheduler/RNG/data state to step 6,000 and must write WikiText PPL.
The controller validates all 20 MQAR cells and diagnostics,
reports step 500 to MedianPruner, preserves `FAIL` versus `PRUNED`, and records
final PPL as an Optuna user attribute. It rejects shared SQLite and stops a
worker on systemic adapter errors instead of hiding them. It binds the shared
study to one spec/code/data/capability identity, gives each Slurm worker a
distinct sampler seed, and atomically claims each full trial hash so duplicate
suggestions do not train twice.

Each trial-hash claim is a JSON lease with `running`, `recovery_queued`, or
`terminal` state, an `optuna_study` or `fixed_manifest` origin, owner host/PID,
timestamps, an opaque lease ID, and an absolute run directory. Optuna startup
recovery ignores fixed-manifest claims; rerunning the fixed manifest reclaims
its own stale lease. The prepared heartbeat is 60 seconds and
`storage.claim_stale_after_seconds` is 600 seconds; the controller requires the
threshold to exceed three heartbeat intervals. A live lease deduplicates the
suggestion, while a preempted worker becomes recoverable after roughly ten
minutes instead of waiting longer than the 24-hour Slurm walltime. A stale
lease is queued once and reclaimed under an `O_EXCL` lock with a one-time
token, preserving its original run directory and checkpoint.
`COMPLETE`, `PRUNED`, and trial-local `FAIL` claims become terminal and are
never reclaimed. Systemic attempts preserve their measured GPU time and
checkpoint for the startup recovery queue. While an adapter subprocess is
running, each heartbeat also durably accrues wall time × world size. A hard
preemption is therefore charged through the last heartbeat, with a documented
worst-case gap of one heartbeat interval per active GPU, instead of disappearing
from the ledger. If actual cumulative time exceeds the conservative per-trial
reservation, the terminal claim records the overrun and the status planner
holds further launches. An overrun is an immutable violation of that study
version: do not delete or edit the claim. Approve larger ceilings under a new
versioned `study_name`, rematerialize against its new spec hash, and use a new
output root.
Before the scientific launch, exercise simultaneous claim/reclaim operations
on the actual SCG filesystem; local atomicity alone does not approve shared
JournalStorage or filesystem locking.

## 8. Analysis and promotion

Generate:

- MQAR versus WikiText Pareto front;
- fANOVA separately for MQAR and perplexity, retaining both the global view and
  the Birdie-only/updated-sweep conditional views;
- capacity-adjusted fANOVA residualized on actual non-embedding parameters and
  measured training FLOPs, so rank/fraction effects are not silently treated
  as equal-capacity comparisons;
- an initial health-gated shortlist, then a separate three-seed confirmed
  shortlist keyed by the seedless `promotion_config_hash`;
- a reviewed falsified-hypotheses report;
- a retrospective control false-pruning report using the final candidate
  step-500 median (clearly labeled post-hoc, not the historical online median);
- the offline W&B table/panel payload.

The Optuna controller samples for MQAR only. WikiText PPL is post-hoc, so the
observed Pareto front may under-sample low-PPL configurations. Record that
limitation in the W&B report and in any promotion decision.

```bash
koopman-phase2-analyze \
  --results-root "$PHASE2_OUTPUT_ROOT" \
  --storage "$PHASE2_STORAGE_URL" \
  --study-name "$PHASE2_STUDY_NAME" \
  --shortlist-cap 10 \
  --require-stability-evidence \
  --output "$PHASE2_OUTPUT_ROOT/analysis/phase2a_analysis.json" \
  --artifact-dir "$PHASE2_OUTPUT_ROOT/analysis/artifacts"
```

Keep controls in `"$PHASE2_OUTPUT_ROOT/control-runs"` and promotions in
`"$PHASE2_OUTPUT_ROOT/promotion-runs"` so one recursive analysis contains every
scientific result. Directory-based scientific analysis requires a matching terminal claim for
every summary and verifies its outcome and cumulative GPU-hours. If controls or
promotions live outside `"$PHASE2_OUTPUT_ROOT"` while sharing its ledger, pass
`--claims-root "$PHASE2_OUTPUT_ROOT"` when analyzing that result tree. This prevents a
summary written immediately before a worker kill from being mistaken for a
committed terminal result.

The first analysis pass selects up to ten candidates but does not call a
single-seed result “stable.” Materialize the missing seeds from its shortlist:

```bash
koopman-phase2-promotions \
  --spec configs/phase2a_search.json \
  --shortlist "$PHASE2_OUTPUT_ROOT/analysis/artifacts/promotion_shortlist.json" \
  --capabilities path/to/final_capabilities.json \
  --data-manifest path/to/final_data_manifest.json \
  --output "$PHASE2_OUTPUT_ROOT/promotion-runs"
```

The tool verifies that every shortlisted seedless configuration hash still
matches the current spec/code/data/capability identity, reuses the already
completed source seed logically, and materializes only missing seeds. Execute
every new plan entry with
`run-manifest --preflight-stage study --stage both
--budget-root "$PHASE2_OUTPUT_ROOT" --storage "$PHASE2_STORAGE"`,
then rerun analysis over the common results root. Only configurations with
healthy results for all seeds 42, 43, and 44 enter
`confirmed_promotion_shortlist.json`; its report includes per-metric mean,
standard deviation, range, and descriptive 95% interval.

Promote approximately 8-12 diverse candidates plus controls to Phase 2b at 5M
and 20M. Do not infer a single winner from a scalarized score.

`wandb_report_payload.json` is an offline publication payload, not a published
W&B report. Publication, reviewed-hypothesis sign-off, automatic batch
planning, and storage/publication receipts are deferred until after the pilot.
They do not affect trial sampling, pruning, required metrics, Pareto analysis,
fANOVA, or promotion selection. Revisit them once the canonical architecture
and the team's operational reporting process are fixed.

## Compute accounting

The study spec fixes 96 sequences x 2,048 tokens per optimizer update:
196,608 tokens per step. Step 500 is 98,304,000 tokens; step 6,000 is
1,179,648,000 tokens for a full survivor.

Before freezing the final spec, benchmark the three calibration arms and
calculate:

```
GPU-hours =
  startup/full trial hours
  + expected pruned trial hours
  + controls and seed replications
  + evaluation overhead
  + 20% failure/retry reserve
```

Record the approved GPU-hour ceiling in a versioned study manifest. The
requested 2,000 trials are a target from "thousands," not permission to exceed
the approved compute budget.
