#!/bin/bash
# Execute one entry from a calibration/pilot/control/promotion execution plan.
# Override --array to the exact zero-based plan range when calling sbatch:
#   calibration 0-2; current pilot 0-20; controls 0-8.

#SBATCH --job-name=echo-p2a-fixed
#SBATCH --account=mpsnyder
#SBATCH --partition=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-20
#SBATCH --output=/labs/mpsnyder/cody1212/koopman_runs/logs/phase2a_fixed_%A_%a.out
#SBATCH --error=/labs/mpsnyder/cody1212/koopman_runs/logs/phase2a_fixed_%A_%a.err

set -eo pipefail
set +u

module load cuda/12.3.2_545.23.08_cudNN_9.0.0.312
module unload gcc/13.3.0 2>/dev/null || true
module unload gcc/11.2.0 2>/dev/null || true
module load gcc/9.2.0-centos_7

export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$(dirname "$(dirname "$(command -v gcc)")")/lib64:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export CC="$(command -v gcc)"
export CXX="$(command -v g++)"
export CUDAHOSTCXX="$CXX"
export HF_HOME=/labs/mpsnyder/cody1212/.hf_cache
export UV_CACHE_DIR=/labs/mpsnyder/cody1212/.uv_cache
export TRITON_CACHE_DIR="/labs/mpsnyder/cody1212/tmp/triton_cache/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$TRITON_CACHE_DIR"

REPO_ROOT="${PHASE2_REPO_ROOT:-/labs/mpsnyder/cody1212/Koopman_Mamba}"
VENV="${PHASE2_VENV:-/labs/mpsnyder/cody1212/Koopman_Mamba/koopman-lm-fast/.venv}"
SPEC="${PHASE2_SPEC:-$REPO_ROOT/configs/phase2a_search.json}"
ADAPTER="${PHASE2_ADAPTER:-koopman-phase2-trial-worker}"

: "${PHASE2_CAPABILITIES:?Set PHASE2_CAPABILITIES to the matching capability manifest}"
: "${PHASE2_FIXED_ROOT:?Set PHASE2_FIXED_ROOT to the materialized plan root}"
: "${PHASE2_PLAN:?Set PHASE2_PLAN to calibration_summary.json, pilot_summary.json, control_summary.json, or promotion_run_summary.json}"
: "${PHASE2_PREFLIGHT_STAGE:?Set PHASE2_PREFLIGHT_STAGE to calibration, pilot, or study}"

cd "$REPO_ROOT"
source "$VENV/bin/activate"

mapfile -t PLAN_FIELDS < <(
  python - "$PHASE2_PLAN" "$PHASE2_FIXED_ROOT" "$SLURM_ARRAY_TASK_ID" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[2]).expanduser().resolve()
plan_path = Path(sys.argv[1]).expanduser()
if not plan_path.is_absolute():
    plan_path = root / plan_path
plan_path = plan_path.resolve()
plan_path.relative_to(root)
index = int(sys.argv[3])
payload = json.loads(plan_path.read_text())
entries = payload.get("execution_plan")
if not isinstance(entries, list) or not 0 <= index < len(entries):
    raise SystemExit(
        f"Array index {index} is outside execution_plan[0:{len(entries) if isinstance(entries, list) else 0}]"
    )
entry = entries[index]
run_dir = (root / entry["run_directory"]).resolve()
run_dir.relative_to(root)
stage = "both" if entry.get("required_stage") == "final" else "screen"
print(run_dir / "trial_manifest.json")
print(stage)
PY
)

MANIFEST="${PLAN_FIELDS[0]}"
STAGE="${PLAN_FIELDS[1]}"
COMMAND=(
  python -m koopman_lm.experiments.phase2.run_manifest
  --spec "$SPEC"
  --capabilities "$PHASE2_CAPABILITIES"
  --manifest "$MANIFEST"
  --stage "$STAGE"
  --preflight-stage "$PHASE2_PREFLIGHT_STAGE"
)

if [[ "$PHASE2_PREFLIGHT_STAGE" == "study" ]]; then
  : "${PHASE2_OUTPUT_ROOT:?Set PHASE2_OUTPUT_ROOT to the shared scientific budget root}"
  : "${PHASE2_STORAGE:?Set PHASE2_STORAGE to the exact Optuna storage URL}"
  COMMAND+=(
    --budget-root "$PHASE2_OUTPUT_ROOT"
    --storage "$PHASE2_STORAGE"
  )
fi

"${COMMAND[@]}" -- "$ADAPTER"
