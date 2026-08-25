#!/bin/bash
#SBATCH --job-name=koopman-beta-contrast
#SBATCH --account=marlowe-m000151-pm06
#SBATCH --partition=batch
#SBATCH --qos=medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/m000151-pm06/jkli/logs/koopman-tests/%x-%j.out
#SBATCH --error=/scratch/m000151-pm06/jkli/logs/koopman-tests/%x-%j.err
#
# Run scripts/report_beta_contrast_on_mqar.py against an ALREADY-FINISHED MQAR
# run root. Needed as its own job because the diagnostic builds the model, and
# `Mamba2Block` imports `mamba_ssm`, which only exists in the CUDA venv -- so it
# cannot run on a login node no matter how little compute it needs.
#
# `run_beta_policy_mqar.sbatch` now runs this step inline at the end of a fresh
# sweep. This script exists for run roots produced before that (job 446102) and
# for re-running the diagnostic with a different batch size or cell without
# retraining anything.
#
# Usage:
#   MQAR_ROOT=/scratch/.../beta-mqar-446102 KV=8 GAP=128 \
#     sbatch scripts/report_beta_contrast.sbatch

set -uo pipefail

export REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
if [ ! -d "$REPO_ROOT/experimentation" ]; then
    echo "FATAL: REPO_ROOT=$REPO_ROOT is not a Koopman_Mamba checkout."
    exit 1
fi
: "${MQAR_ROOT:?set MQAR_ROOT to a beta-mqar-<jobid> directory}"
export KV="${KV:-8}"
export GAP="${GAP:-128}"
export TASK_VOCAB="${TASK_VOCAB:-128}"
export SCRATCH=/scratch/m000151-pm06/jkli
export VENV=$SCRATCH/venvs/koopman-cuda

echo "=== job ${SLURM_JOB_ID:-<interactive>} on $(hostname) ==="
date -Is
cd "$REPO_ROOT" || exit 1
echo "commit: $(git rev-parse HEAD)"
echo "root:   $MQAR_ROOT  (kv=$KV gap=$GAP)"

module purge
module load cuda12.9/toolkit/12.9.1
module load gcc/13.1.0
# shellcheck disable=SC1091
source "$VENV/bin/activate"
export PYTHONPATH=$REPO_ROOT
export HF_HOME=$SCRATCH/hf
export TRITON_CACHE_DIR=$SCRATCH/triton-cache

python "$REPO_ROOT/scripts/report_beta_contrast_on_mqar.py" \
    --kv "$KV" --gap "$GAP" --task_vocab "$TASK_VOCAB" \
    --json_out "$MQAR_ROOT/beta-contrast.json" \
    "$MQAR_ROOT"/mqar-*-seed*
echo "STATUS contrast=$?"
date -Is
echo "=== END ==="
