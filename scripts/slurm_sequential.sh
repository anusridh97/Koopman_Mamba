#!/bin/bash
# SLURM single job — runs all 112 cells sequentially on one H100.
# Use slurm_array.sh instead when possible (84x faster).
#
# Submit: sbatch scripts/slurm_sequential.sh

#SBATCH --job-name=mqar_sweep_seq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=32G
#SBATCH --time=5-00:00:00
#SBATCH --output=logs/mqar_seq_%j.out
#SBATCH --error=logs/mqar_seq_%j.err

set -e
mkdir -p logs

module load python/3.11 cuda/12.4 gcc/12
source "$SCRATCH/Koopman_Mamba/.venv/bin/activate"

python -m koopman_lm.experiments.mqar_finetune --sweep \
    --output_root   "$SCRATCH/mqar-sweep" \
    --model_types   koopman mamba_attn mamba_only transformer \
    --batch_size    64 \
    --eval_batch    64 \
    --max_steps     10000 \
    --save_steps    2000
