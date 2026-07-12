#!/bin/bash
# Phase-1 threshold calibration on an SCG H200 node.
#
# Runs the health / grad-flow / four-mode load-bearing diagnostics on a set of
# REFERENCE models and writes RAW continuous values to JSON, so the Phase-2
# pruning thresholds ([0.3,0.95] band, 0.1 grad ratio, 0.5 ska_delta) are set
# from data. BLOCKING PREREQ: run scripts/inspect_checkpoint.py first to confirm
# the completed checkpoints strict-load (or to size a load shim).
#
# Submit from the SCG repo root:
#   sbatch scripts/slurm_calibrate_scg.sh

#SBATCH --job-name=phase1_calib
#SBATCH --account=mpsnyder
#SBATCH --partition=gpu_short
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --chdir=/oak/stanford/scg/lab_mpsnyder/cody1212/Koopman_Mamba
#SBATCH --output=/labs/mpsnyder/cody1212/koopman_runs/logs/phase1_calib_%j.out
#SBATCH --error=/labs/mpsnyder/cody1212/koopman_runs/logs/phase1_calib_%j.err

set -e

module load cuda/12.3.2_545.23.08_cudNN_9.0.0.312
module unload gcc/13.3.0 2>/dev/null || true
module unload gcc/11.2.0 2>/dev/null || true
module load gcc/9.2.0-centos_7
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$(dirname $(dirname $(which gcc)))/lib64:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export CC=$(which gcc); export CXX=$(which g++); export CUDAHOSTCXX=$(which g++)
export HF_HOME=/labs/mpsnyder/cody1212/.hf_cache
export OMP_NUM_THREADS=8

# venv is independent of the repo layout; override with KOOPMAN_VENV if moved.
VENV="${KOOPMAN_VENV:-/labs/mpsnyder/cody1212/Koopman_Mamba/koopman-lm-fast/.venv}"
set +u
source "$VENV/bin/activate"
set -u

CKPT_50M="/labs/mpsnyder/cody1212/runs/echo-50m-fineweb-3B/final/model.pt"
OUT="/labs/mpsnyder/cody1212/results/phase1_calibration"
TOK="NousResearch/Llama-2-7b-hf"
mkdir -p "$OUT" /labs/mpsnyder/cody1212/koopman_runs/logs

# BLOCKING check first: does the completed checkpoint strict-load here?
echo "== load-compat check (50M) =="
python scripts/inspect_checkpoint.py --checkpoint "$CKPT_50M" --model_size 50m \
    || { echo "!! 50M checkpoint does not strict-load -- write a shim before calibrating"; exit 1; }

# Reference set: trained 50M, an untrained init of the SAME arch, and a
# from-scratch 50M scale. (Add the 180M checkpoint / a mamba_only baseline the
# same way once available.)
echo "== calibrate: trained 50M =="
python -m koopman_lm.evaluation.calibrate --checkpoint "$CKPT_50M" \
    --tokenizer "$TOK" --data wikitext --out "$OUT/calib_50m_trained.json"

echo "== calibrate: untrained init (same arch as the 50M checkpoint) =="
python -m koopman_lm.evaluation.calibrate --checkpoint "$CKPT_50M" --init_only \
    --tokenizer "$TOK" --data wikitext --out "$OUT/calib_50m_untrained.json"

echo "CALIBRATION DONE. Raw values under $OUT/"
