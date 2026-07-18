#!/bin/bash
# One-time uv setup for training + eval. uv manages a .venv from pyproject.toml.
#
#   bash scripts/setup_env.sh
#
# Then launch:  sbatch scripts/slurm_pretrain.sh 50m
#          or:  uv run --extra cuda bash scripts/pretrain.sh 50m
set -e

module load python/3.11 cuda/12.4 gcc/12 2>/dev/null || true

# install uv if missing (astral installer -> ~/.local/bin)
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi

# Sync project + CUDA kernels in ONE command. pyproject.toml's [tool.uv] pins
# torch to the cu124 wheel index and builds mamba-ssm / causal-conv1d with no
# build isolation (against the installed torch), so this replaces the old
# torch-first-then-kernels sequence.
export UV_CACHE_DIR="${UV_CACHE_DIR:-${SLURM_TMPDIR:-/tmp}/uv-cache}"
MAX_JOBS="${MAX_JOBS:-4}" uv sync --extra cuda

# Zero-shot lm-eval-harness (only needed before koopman_lm.evaluation.lm_harness_eval):
#   uv sync --extra cuda --extra lmharness
#
# Manual fallback if `uv sync` ever struggles with the kernel build order:
#   uv venv .venv --python 3.11 && source .venv/bin/activate
#   uv pip install torch --index-url https://download.pytorch.org/whl/cu124
#   MAX_JOBS=4 uv pip install mamba-ssm causal-conv1d triton --no-build-isolation
#   uv pip install -e .

echo "Done. Launch: sbatch scripts/slurm_pretrain.sh 50m"
