#!/bin/bash
# Phase 2a: one Optuna trial per GPU worker. Scientific execution is blocked by
# the fail-closed preflight until the finalized capability/data manifests pass.
#
# Before sbatch:
#   mkdir -p /labs/mpsnyder/cody1212/koopman_runs/logs
#   export PHASE2_CAPABILITIES=/absolute/path/final_capabilities.json
#   export PHASE2_DATA_MANIFEST=/absolute/path/final_data_manifest.json
#   export PHASE2_STORAGE_URL='postgresql+psycopg://...'
#   export PHASE2_STUDY_NAME=echo-phase2a-3m-v1
#   export PHASE2_OUTPUT_ROOT=/labs/mpsnyder/cody1212/phase2a-runs
#   sbatch --array=0-15 scripts/slurm_phase2a_3m_workers_scg.sh
#
# Do not use this study-worker script for the fixed pilot; pilot manifests run
# through koopman-phase2-run-manifest in a separate allocation/array. Before
# each scientific batch, inspect Optuna and the shared claim ledger; keep the
# array no larger than maximum_concurrent_trials.

#SBATCH --job-name=echo-p2a-3m
#SBATCH --account=mpsnyder
#SBATCH --partition=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-15
#SBATCH --output=/labs/mpsnyder/cody1212/koopman_runs/logs/phase2a_3m_%A_%a.out
#SBATCH --error=/labs/mpsnyder/cody1212/koopman_runs/logs/phase2a_3m_%A_%a.err

set -eo pipefail
# SCG module scripts and /etc/bashrc reference unset variables.
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
TRIALS_PER_WORKER="${PHASE2_TRIALS_PER_WORKER:-1}"

: "${PHASE2_CAPABILITIES:?Set PHASE2_CAPABILITIES to the finalized capability manifest}"
: "${PHASE2_DATA_MANIFEST:?Set PHASE2_DATA_MANIFEST to the frozen data manifest}"
: "${PHASE2_STORAGE_URL:?Set PHASE2_STORAGE_URL to PostgreSQL or validated JournalStorage}"
: "${PHASE2_OUTPUT_ROOT:?Set PHASE2_OUTPUT_ROOT outside the Git checkout}"

cd "$REPO_ROOT"
source "$VENV/bin/activate"

python -m koopman_lm.experiments.phase2.run_study \
  --spec "$SPEC" \
  --capabilities "$PHASE2_CAPABILITIES" \
  --data-manifest "$PHASE2_DATA_MANIFEST" \
  --storage "$PHASE2_STORAGE_URL" \
  --output-root "$PHASE2_OUTPUT_ROOT" \
  --trials "$TRIALS_PER_WORKER" \
  --worker-index "$SLURM_ARRAY_TASK_ID" \
  -- "$ADAPTER"
