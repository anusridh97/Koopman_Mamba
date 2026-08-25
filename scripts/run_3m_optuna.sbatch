#!/bin/bash
# Starts the Optuna driver for the complete 3M architecture/hyperparameter sweep.
# The driver submits and monitors the individual GPU trials.
# Submit with: sbatch scripts/run_3m_optuna.sbatch
# No job array is used here.
#SBATCH --job-name=3m-optuna
#SBATCH --account=cis261066-ai
#SBATCH --partition=ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=2-00:00:00
#SBATCH --output=/anvil/scratch/x-ajohansen/3m-optuna-runs/driver-%j.out
#SBATCH --error=/anvil/scratch/x-ajohansen/3m-optuna-runs/driver-%j.err

set -euo pipefail

module purge
module load modtree/gpu
module load cuda/13.1.0
unset PYTHONPATH PYTHONHOME

source "$SCRATCH/research/Koopman_Mamba/.venv/bin/activate"
cd "$SCRATCH/research/Koopman_Mamba"

mkdir -p /anvil/scratch/x-ajohansen/3m-optuna-runs

python -m experimentation.sweep.search \
  configs/search/3m-adaptive.yaml \
  --launcher slurm \
  --allow-dirty
