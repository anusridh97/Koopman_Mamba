#!/bin/bash
# Run once on the cluster to set up the environment with UV.
# UV is used instead of pip for faster installs on shared filesystems.
#
# Usage: bash scripts/setup_env.sh

set -e

module load python/3.11 cuda/12.4 gcc/12

# Install UV (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env

# Create environment
uv venv .venv --python 3.11
source .venv/bin/activate

# Install torch first — mamba_ssm build requires it
uv pip install torch --index-url https://download.pytorch.org/whl/cu124

# Install mamba_ssm (must be compiled against the torch above)
export PIP_CACHE_DIR=${SLURM_TMPDIR:-/tmp}/pip-cache
MAX_JOBS=4 uv pip install mamba-ssm causal-conv1d \
    --no-build-isolation --no-cache-dir

# Install the project
uv pip install -e ".[dev]"

echo "Done. Activate with: source .venv/bin/activate"
