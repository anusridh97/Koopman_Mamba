#!/bin/bash
# Table 2: Length generalization on NIAH -- Echo paper Section 4.1.
#
# Trains three ~1M models on MQAR (KV=4) at seq_len=64 -- a stand-in for the
# paper's unspecified "mixed system-prompt and tool-trace" curriculum, see
# koopman_lm/experiments/table2.py's docstring -- then evaluates zero-shot on
# NIAH (MQAR, KV=1, held out from training) at 64-4096 tokens.
# 3 jobs, 2 concurrent (2 GPU limit). ~15 min per job.
#
# Submit: sbatch scripts/slurm_table2.sh

#SBATCH --job-name=table2
#SBATCH --array=0-2%2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/table2_%A_%a.out
#SBATCH --error=logs/table2_%A_%a.err

set -e
mkdir -p logs

module load python/3.11 cuda/12.4 gcc/12
source "$SCRATCH/Koopman_Mamba/.venv/bin/activate"

MODEL_TYPES=(mamba_only mamba_attn mamba_ska_swiglu)
MODEL_TYPE=${MODEL_TYPES[$SLURM_ARRAY_TASK_ID]}

echo "Table 2 job ${SLURM_ARRAY_TASK_ID}: model=${MODEL_TYPE}"

python -m koopman_lm.experiments.table2 \
    --model_type  "$MODEL_TYPE" \
    --model_size  1m \
    --batch_size  16 \
    --epoch_size  200000 \
    --lr          3e-4 \
    --max_steps   6000 \
    --eval_every  1000 \
    --eval_batch  256 \
    --output_dir  "$SCRATCH/table2/$MODEL_TYPE"
