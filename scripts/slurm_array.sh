#!/bin/bash
# SLURM job array - 84 cells, at most 2 running at a time (2 GPU limit).
# 3 model types x 4 KV counts x 7 gap lengths = 84 total.
# Expected total wall time: ~7 days (84 cells x ~4h / 2 GPUs).
#
# Submit: sbatch scripts/slurm_array.sh
# Monitor: squeue -u $USER

#SBATCH --job-name=mqar_sweep
#SBATCH --array=0-83%2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/mqar_%A_%a.out
#SBATCH --error=logs/mqar_%A_%a.err

set -e
mkdir -p logs

module load python/3.11 cuda/12.4 gcc/12
source "$SCRATCH/Koopman_Mamba/.venv/bin/activate"

# Grid — must match PAPER_MODEL_TYPES, PAPER_KV_PAIRS, PAPER_GAPS in mqar_finetune.py
MODEL_TYPES=(mamba_ska_swiglu mamba_attn mamba_only)
KV_PAIRS=(4 8 16 32)
GAPS=(64 128 256 512 1024 2048 4096)

N_KV=${#KV_PAIRS[@]}    # 4
N_GAPS=${#GAPS[@]}      # 7
N_CELLS=$((N_KV * N_GAPS))   # 28 cells per model type

MODEL_IDX=$(( SLURM_ARRAY_TASK_ID / N_CELLS ))
CELL_IDX=$(( SLURM_ARRAY_TASK_ID % N_CELLS ))
KV_IDX=$(( CELL_IDX / N_GAPS ))
GAP_IDX=$(( CELL_IDX % N_GAPS ))

MODEL_TYPE=${MODEL_TYPES[$MODEL_IDX]}
KV=${KV_PAIRS[$KV_IDX]}
GAP=${GAPS[$GAP_IDX]}

OUTDIR=$SCRATCH/mqar-sweep/${MODEL_TYPE}/kv${KV}_gap${GAP}

echo "Job ${SLURM_ARRAY_TASK_ID}: model=${MODEL_TYPE} kv=${KV} gap=${GAP}"

# Skip if already completed
if [ -f "$OUTDIR/final/model.pt" ]; then
    echo "Already done, skipping."
    exit 0
fi

python -m koopman_lm.experiments.mqar_finetune \
    --model_type    "$MODEL_TYPE" \
    --model_size    50m \
    --num_kv_pairs  "$KV" \
    --distractor_gap "$GAP" \
    --task_vocab_size 128 \
    --batch_size    64 \
    --eval_batch    64 \
    --max_steps     10000 \
    --save_steps    2000 \
    --output_dir    "$OUTDIR"
