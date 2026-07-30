# Phase 2a preparation

Start here:

1. `EXPERIMENT_CONTRACT.md` defines the scientific study and every initial
   1M-core sweep axis.
2. `INTEGRATION_MATRIX.md` lists what the finalized codebase must provide and
   what still blocks real training.
3. `LAUNCH_RUNBOOK.md` gives the data, preflight, pilot, launch, and promotion
   sequence.
4. `TRAINING_ADAPTER_CONTRACT.md` is the narrow API the finalized trainer must
   implement.

Machine-readable inputs:

- `configs/phase2a_1m_core.json`
- `configs/phase2a_search.json`
- `configs/phase2a_capabilities.template.json`
- `configs/phase2a_data_manifest.template.json`
- `configs/phase2a_hypothesis_decisions.template.json`
- `configs/phase2a_wandb_verification.template.json`

Machine-readable output contracts live in `docs/phase2/schemas/`, including
`analysis_report.schema.json` for the aggregate post-run artifact and
`wandb_report_verification.schema.json` for the required publication audit.

Portable tools:

```bash
# Search-contract and trial-materialization smoke test
koopman-phase2-dry-run \
  --spec configs/phase2a_search.json \
  --trials 32 \
  --output /tmp/echo-phase2a-dry-run

# After calibration-stage preflight, materialize 3 screen-only cost arms
koopman-phase2-calibration \
  --spec configs/phase2a_search.json \
  --capabilities /path/to/calibration_capabilities.json \
  --data-manifest /path/to/final_data_manifest.json \
  --output "$PHASE2_CALIBRATION_ROOT"

# Materialize the fixed pipeline-validation pilot
koopman-phase2-pilot \
  --spec configs/phase2a_search.json \
  --capabilities /path/to/pilot_capabilities.json \
  --data-manifest /path/to/final_data_manifest.json \
  --output /tmp/echo-phase2a-pilot

# After executing the pilot's generated execution_plan
koopman-phase2-audit-pilot \
  --spec configs/phase2a_search.json \
  --capabilities /path/to/pilot_capabilities.json \
  --data-manifest /path/to/final_data_manifest.json \
  --pilot-root /path/to/echo-phase2a-pilot \
  --output /path/to/pilot_report.json

# After the pilot report is approved, materialize 3 controls x 3 seeds
koopman-phase2-controls \
  --spec configs/phase2a_search.json \
  --capabilities /path/to/final_capabilities.json \
  --data-manifest /path/to/final_data_manifest.json \
  --output "$PHASE2_OUTPUT_ROOT/control-runs"

# Materialize seeds 43/44 (or any missing required seed) for shortlisted configs
koopman-phase2-promotions \
  --spec configs/phase2a_search.json \
  --shortlist "$PHASE2_OUTPUT_ROOT/analysis/artifacts/promotion_shortlist.json" \
  --capabilities /path/to/final_capabilities.json \
  --data-manifest /path/to/final_data_manifest.json \
  --output "$PHASE2_OUTPUT_ROOT/promotion-runs"

# This intentionally fails on the preparation branch
koopman-phase2-preflight \
  --spec configs/phase2a_search.json \
  --capabilities configs/phase2a_capabilities.template.json \
  --data-manifest configs/phase2a_data_manifest.template.json \
  --stage pilot

# On a one-GPU compute allocation, capture/verify the exact worker stack
koopman-phase2-runtime --write-lock /absolute/path/phase2a_runtime.lock.json
koopman-phase2-runtime --verify-lock /absolute/path/phase2a_runtime.lock.json

# Build an entirely offline Phase 2a analysis artifact
koopman-phase2-analyze \
  --results-root "$PHASE2_OUTPUT_ROOT" \
  --storage "$PHASE2_STORAGE_URL" \
  --study-name "$PHASE2_STUDY_NAME" \
  --shortlist-cap 10 \
  --output "$PHASE2_OUTPUT_ROOT/analysis/phase2a_analysis.json" \
  --artifact-dir "$PHASE2_OUTPUT_ROOT/analysis/artifacts"

# After the storage snapshot, scientist review, and a verified W&B publication
koopman-phase2-finalize-report \
  --analysis "$PHASE2_OUTPUT_ROOT/analysis/phase2a_analysis.json" \
  --decisions /path/to/reviewed_hypothesis_decisions.json \
  --wandb-payload "$PHASE2_OUTPUT_ROOT/analysis/artifacts/wandb_report_payload.json" \
  --storage-snapshot-receipt "$PHASE2_OUTPUT_ROOT/analysis/storage_snapshot_receipt.json" \
  --wandb-verification /path/to/final_wandb_report_verification.json \
  --reviewer TEAM_IDENTITY \
  --reviewed-at-utc 2026-01-01T00:00:00Z \
  --publisher TEAM_IDENTITY \
  --published-at-utc 2026-01-01T00:05:00Z \
  --output-dir "$PHASE2_OUTPUT_ROOT/analysis/reporting-receipts-v1"

# Inspect target/concurrency/GPU-hour state before each scientific array batch
koopman-phase2-status \
  --spec configs/phase2a_search.json \
  --storage "$PHASE2_STORAGE_URL" \
  --study-name "$PHASE2_STUDY_NAME" \
  --capabilities /path/to/final_capabilities.json \
  --data-manifest /path/to/final_data_manifest.json \
  --output-root "$PHASE2_OUTPUT_ROOT"
```

The dry-run, calibration, and pilot materializers do not launch training. That
is deliberate:
paper mode, the finalized candidate architecture, optimizer groups, Birdie objectives,
exact resume, and the evaluator callback still have unresolved integration
work. Once those capabilities pass preflight, `koopman-phase2-study` provides
the step-500/prune/resume controller and calls the finalized
`koopman-phase2-trial-worker` adapter. Fixed controls and promotion seeds use
`koopman-phase2-run-manifest --budget-root <the study output root> --storage
<the same Optuna URL>`. They must match the sampled study's immutable ledger
contract before claiming GPU time. Preflight prevents a plausible-looking but
scientifically invalid sweep. It hashes the actual frozen train,
WikiText-validation, MQAR, tokenizer, shard-order, oracle, token-map, and
Birdie fixture files at static preflight and again at worker
launch, then compares the active Python/PyTorch/CUDA/package/kernel stack to
the approved runtime lock.

Keep `PHASE2_CALIBRATION_ROOT` outside `PHASE2_OUTPUT_ROOT`; calibration
timings are non-scientific and recursive scientific analysis rejects their
presence. All fixed study-stage runs must pass
`--budget-root "$PHASE2_OUTPUT_ROOT" --storage "$PHASE2_STORAGE"` so controls,
promotions, and sampled trials share one claim and GPU-hour ledger.

The analysis command does not contact W&B. It emits a W&B-ready payload with
table data and panel specifications, alongside the Pareto front, conditional
fANOVA views, capacity/FLOP-adjusted fANOVA views, promotion shortlist, and
falsified-hypotheses review scaffold.
When both results and an Optuna study are supplied, it joins trial summaries to
their Optuna parameters strictly by trial hash; trial-number-only fallback is
rejected. Health evidence from the trial summary remains authoritative.
Publishing remains an explicitly authorized manual action. The finalization
tool records the reviewed hypothesis decisions and the published W&B URL in
hash-bound receipts; those receipts are required before Phase 2a is considered
reported.
Promotion requires explicit passing stability evidence by default. The
`--allow-missing-stability-evidence` escape hatch is only for exploratory
preparation reports and labels missing evidence `not_recorded`.
All scientific analysis inputs must carry the same manifest-derived study
identity; trial-number-only joins and cross-study result mixing are rejected.
