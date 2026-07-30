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

The executable validators in `koopman_lm/experiments/phase2/results.py`,
`preflight.py`, and `analysis.py` are the current output contracts. Standalone
JSON Schemas are deferred until the architecture and metric payload are final.

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
workflow should be chosen after the pilot, when the team has settled the
canonical architecture and reporting process.
Promotion requires explicit passing stability evidence by default. The
`--allow-missing-stability-evidence` escape hatch is only for exploratory
preparation reports and labels missing evidence `not_recorded`.
All scientific analysis inputs must carry the same manifest-derived study
identity; trial-number-only joins and cross-study result mixing are rejected.

Deferred until after the pilot: an automatic next-batch planner, Optuna storage
snapshot receipts, reviewed-hypothesis receipts, and W&B publication
verification receipts. These are end-of-study operational hardening, not part
of the initial 1M sweep or its required Pareto/fANOVA analysis, and can be
reintroduced once the canonical architecture and team workflow are fixed.
