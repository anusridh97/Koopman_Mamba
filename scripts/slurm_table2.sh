#!/bin/bash
# Table 2: Length generalization on NIAH -- Echo paper Section 4.1.
#
# Trains three ~1M models on an alternating tool-calling / system-prompt
# curriculum (koopman_lm.experiments.curricula.make_toolcall /
# make_sysprompt) at seq_len=64 -- a stand-in for the paper's unspecified
# "mixed system-prompt and tool-trace examples," modeled on Appendix G.4's
# task descriptions -- then evaluates zero-shot on NIAH (a structurally
# distinct random-needle-position generator, curricula.make_niah, never
# produced by the training curricula) at 64-4096 tokens. See
# koopman_lm/experiments/table2.py's docstring for details.
#
# max_steps=12000, not the paper's stated 6000: empirically, 6000 steps
# plateaus well short of convergence for this curriculum (~58% at the
# training length); a sharp phase transition to >95% happens between
# 6000-10000 steps. Pass --max_steps 6000 to reproduce the paper's literal
# step budget instead (and see the shortfall for yourself).
#
# Uses the project's real model stack (koopman_lm.models.baselines), which
# requires mamba_ssm -- make sure it's installed in the job's venv.
#
# Submit: sbatch scripts/slurm_table2.sh

#SBATCH --job-name=table2
#SBATCH --array=0-2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/table2_%A_%a.out
#SBATCH --error=logs/table2_%A_%a.err

set -euo pipefail

MODELS=(mamba_only mamba_attn mamba_ska_swiglu)
MODEL_TYPE=${MODELS[$SLURM_ARRAY_TASK_ID]}

module load cuda || true
source .venv/bin/activate
mkdir -p logs

echo "Table 2 job ${SLURM_ARRAY_TASK_ID}: model=${MODEL_TYPE}"

python -m koopman_lm.experiments.table2 \
    --model_type "$MODEL_TYPE" \
    --model_size 1m \
    --batch_size 16 \
    --lr 3e-4 \
    --max_steps 12000 \
    --eval_every 1000 \
    --eval_batch 256 \
    --output_dir "${SCRATCH:-.}/table2/$MODEL_TYPE"
