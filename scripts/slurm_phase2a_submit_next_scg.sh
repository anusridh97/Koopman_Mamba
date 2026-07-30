#!/bin/bash
# Submit exactly one budget- and concurrency-bounded Phase 2a array batch.
# Re-run after each batch; the status planner returns HOLD once the target,
# concurrency ceiling, or reserved GPU-hour ceiling prevents another launch.

set -eo pipefail
set +u

REPO_ROOT="${PHASE2_REPO_ROOT:-/labs/mpsnyder/cody1212/Koopman_Mamba}"
VENV="${PHASE2_VENV:-/labs/mpsnyder/cody1212/Koopman_Mamba/koopman-lm-fast/.venv}"
SPEC="${PHASE2_SPEC:-$REPO_ROOT/configs/phase2a_search.json}"

: "${PHASE2_STORAGE_URL:?Set PHASE2_STORAGE_URL to PostgreSQL or validated JournalStorage}"
: "${PHASE2_STUDY_NAME:?Set PHASE2_STUDY_NAME to the exact frozen spec study_name}"
: "${PHASE2_OUTPUT_ROOT:?Set PHASE2_OUTPUT_ROOT outside the Git checkout}"
: "${PHASE2_CAPABILITIES:?Set PHASE2_CAPABILITIES to the finalized capability manifest}"
: "${PHASE2_DATA_MANIFEST:?Set PHASE2_DATA_MANIFEST to the frozen data manifest}"

cd "$REPO_ROOT"

ARRAY_EXPRESSION="$(
  "$VENV/bin/python" -m koopman_lm.experiments.phase2.study_status \
    --spec "$SPEC" \
    --storage "$PHASE2_STORAGE_URL" \
    --study-name "$PHASE2_STUDY_NAME" \
    --capabilities "$PHASE2_CAPABILITIES" \
    --data-manifest "$PHASE2_DATA_MANIFEST" \
    --output-root "$PHASE2_OUTPUT_ROOT" \
    --print-next-array
)"

JOB_ID="$(
  sbatch --parsable \
    --array="$ARRAY_EXPRESSION" \
    scripts/slurm_phase2a_1m_workers_scg.sh
)"

echo "Submitted Phase 2a batch job ${JOB_ID} with array ${ARRAY_EXPRESSION}"
