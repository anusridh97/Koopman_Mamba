#!/bin/bash
# =============================================================================
# Clean one-command launcher for Echo 50M / 180M / 440M (uv-native, SLURM).
#
# First-time setup (once, uv):
#   bash scripts/setup_env.sh                         # or: uv sync --extra cuda
#
# Submit a run (size is the one argument):
#   sbatch scripts/slurm_pretrain.sh 50m              # 3B tokens
#   sbatch --time=96:00:00  scripts/slurm_pretrain.sh 180m            # 10B tokens
#   sbatch --time=168:00:00 --gres=gpu:h100:4 scripts/slurm_pretrain.sh 440m  # 20B, 4-GPU DDP
#
# Runs locally too (no SLURM):  bash scripts/slurm_pretrain.sh 50m
#
# Wraps scripts/pretrain.sh (the single source of truth for per-size budgets and
# the tokenize -> train -> eval pipeline). Everything is env-overridable
# (TOKENS/STEPS/PDBS/GA/LR/DATA_ROOT/RUN_ROOT/NPROC/EXTRA_TRAIN_ARGS -- see
# TRAINING.md). SBATCH defaults suit one 80GB GPU; raise --time / --gres for the
# bigger runs (180M ≈ days, 440M ≈ a week single-GPU -> use multi-GPU DDP).
# =============================================================================
#SBATCH --job-name=echo_pretrain
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --output=logs/echo_%x_%j.out
#SBATCH --error=logs/echo_%x_%j.err
set -eo pipefail

SIZE="${1:?usage: sbatch scripts/slurm_pretrain.sh <50m|180m|440m>}"
cd "${SLURM_SUBMIT_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || echo .)}"
mkdir -p logs

# ---- cluster setup (edit for your site; matches scripts/setup_env.sh) ----
module load python/3.11 cuda/12.4 gcc/12 2>/dev/null || true
export HF_HOME="${HF_HOME:-${SCRATCH:-$HOME}/.hf_cache}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

# data / run roots default to $SCRATCH; DDP auto-scales to the GPU allocation
export DATA_ROOT="${DATA_ROOT:-${SCRATCH:-.}/data}"
export RUN_ROOT="${RUN_ROOT:-${SCRATCH:-.}/runs}"
export NPROC="${NPROC:-$(nvidia-smi -L 2>/dev/null | grep -c GPU)}"
[ "${NPROC:-0}" -ge 1 ] 2>/dev/null || NPROC=1
export NPROC

echo "launch: size=$SIZE  gpus(NPROC)=$NPROC  data=$DATA_ROOT  runs=$RUN_ROOT"

# uv-native: `uv run` auto-syncs the project env from pyproject.toml's [tool.uv]
# (torch from the CUDA index, mamba-ssm/causal-conv1d built no-isolation) and runs
# within it -- no manual venv activation. Child python/torchrun inherit the env.
if command -v uv >/dev/null 2>&1; then
  uv run --extra cuda bash scripts/pretrain.sh "$SIZE"
else
  # fallback: a pre-activated venv (e.g. from setup_env.sh)
  bash scripts/pretrain.sh "$SIZE"
fi
