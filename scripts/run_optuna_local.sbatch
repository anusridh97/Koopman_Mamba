#!/bin/bash
# Run one complete Optuna study sequentially inside this single GPU allocation.
# Submit, for example:
#   sbatch scripts/run_optuna_local.sbatch configs/search/10m-adaptive.yaml
# Override the wall-time per scale when needed:
#   sbatch --time=24:00:00 scripts/run_optuna_local.sbatch configs/search/50m-nonembed-adaptive.yaml
#SBATCH --job-name=optuna-local
#SBATCH --account=cis261066-ai
#SBATCH --partition=ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=optuna-local-%j.out
#SBATCH --error=optuna-local-%j.err

set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "usage: sbatch $0 configs/search/<study>.yaml" >&2
    exit 2
fi

module purge
module load modtree/gpu
module load cuda/13.1.0
unset PYTHONPATH PYTHONHOME

source "$SCRATCH/research/Koopman_Mamba/.venv/bin/activate"
cd "$SCRATCH/research/Koopman_Mamba"

python -m experimentation.sweep.search "$1" --launcher local --allow-dirty
