#!/bin/bash
#SBATCH --job-name=koopman-mqar-ablate
#SBATCH --account=marlowe-m000151-pm06
#SBATCH --partition=batch
#SBATCH --qos=medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=40G
#SBATCH --time=00:25:00
#SBATCH --output=/scratch/m000151-pm06/jkli/logs/koopman-tests/%x-%j.out
#SBATCH --error=/scratch/m000151-pm06/jkli/logs/koopman-tests/%x-%j.err
#
# SKA-on vs SKA-zeroed MQAR retrieval on checkpoints THAT ALREADY EXIST.
#
# Job 446145 trained four beta policies x three seeds to 24000 steps at
# kv=8/gap=128 on the proxy geometry, and its checkpoints are still on scratch.
# What it never measured is the quantity that decides whether any of those runs
# learned MQAR *in SKA*: the ablation delta.
#
# That matters independently of the exponent arm. 446145's accuracy table says
# `linear` 3/3, `learned` 2/3, `one` 1/3, `head_scalar` 1/3 -- and `one` HAS NO
# GATE AT ALL yet solves the cell. So a policy reaching 1.0000 is not evidence
# that its write gate did anything. Held-out LM loss cannot settle it either: it
# is proven non-responsive to SKA damage here (44% of SKA's contribution
# destroyed moved mean loss by -1.6e-4, far under the 7.54e-3 floor). The
# ablation delta is the instrument that is not blind to it.
#
# One GPU, ~12 checkpoints, two forward passes each. Requested at 1 GPU / 14 CPUs
# (the node's 112/8 ratio) precisely so it backfills instead of waiting behind
# the whole-node exponent job.
#
# RUN_ROOTS may be overridden to point at any set of `beta-mqar-*` /
# `beta-exponent-*` roots.

set -uo pipefail

export REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
if [ ! -d "$REPO_ROOT/experimentation" ]; then
    echo "FATAL: REPO_ROOT=$REPO_ROOT is not a Koopman_Mamba checkout."; exit 1
fi
export SCRATCH=/scratch/m000151-pm06/jkli
export VENV=$SCRATCH/venvs/koopman-cuda
export KV="${KV:-8}"
export GAP="${GAP:-128}"
export TASK_VOCAB="${TASK_VOCAB:-128}"
RUN_ROOTS="${RUN_ROOTS:-$SCRATCH/beta-mqar-446145 $SCRATCH/beta-mqar-446106}"

echo "=== job ${SLURM_JOB_ID:-<interactive>} on $(hostname) ==="
date -Is
cd "$REPO_ROOT" || exit 1
echo "commit: $(git rev-parse HEAD)"
echo "roots:  $RUN_ROOTS"

module purge
module load cuda12.9/toolkit/12.9.1
module load gcc/13.1.0
# shellcheck disable=SC1091
source "$VENV/bin/activate"
export PYTHONPATH=$REPO_ROOT
export HF_HOME=$SCRATCH/hf
export TRITON_CACHE_DIR=$SCRATCH/triton-cache

python -c "
import torch
assert torch.cuda.is_available(), 'no CUDA device visible to torch'
print('torch', torch.__version__, 'devices', torch.cuda.device_count())
" || { echo "FATAL: torch cannot see a GPU. An ablation delta measured on CPU-only would not exist at all."; exit 1; }

rc=0
for root in $RUN_ROOTS; do
    if [ ! -d "$root" ]; then
        echo "SKIP $root: not a directory"; continue
    fi
    echo "############################################################"
    echo "### $root"
    echo "############################################################"
    python "$REPO_ROOT/scripts/report_beta_exponent_arm.py" "$root" \
        --kv "$KV" --gap "$GAP" --task_vocab "$TASK_VOCAB" \
        --json_out "$root/ska-ablation-on-mqar.json" || rc=1
done

date -Is
echo "=== END (rc=$rc) ==="
exit $rc
