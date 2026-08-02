#!/bin/bash
# Four independent one-GPU infrastructure trials:
#   {mamba_only, transformer} x {seed 42, seed 43}
#
# This validates concurrent Slurm trials, the shared LM trainer, weighted-loss
# data, checkpoint metadata, and baseline isolation before the Echo sweep.
# It is deliberately NOT a scientific Phase 2 comparison: the final parameter
# matching and full training budget still require team approval.
#
# Before submission:
#   mkdir -p logs
#   export PHASE2_DATA_DIR=/absolute/path/to/pretokenized/data
#   export PHASE2_BASELINE_ROOT=/absolute/path/to/output
#   sbatch scripts/slurm_phase2a_baseline_smoke.sh

#SBATCH --job-name=p2a-baseline-smoke
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/p2a_baseline_%A_%a.out
#SBATCH --error=logs/p2a_baseline_%A_%a.err

set -eo pipefail
set +u

MODELS=(mamba_only transformer mamba_only transformer)
SEEDS=(42 42 43 43)
INDEX="${SLURM_ARRAY_TASK_ID:-0}"
MODEL_TYPE="${MODELS[$INDEX]}"
SEED="${SEEDS[$INDEX]}"

: "${PHASE2_DATA_DIR:?Set PHASE2_DATA_DIR to one frozen pretokenized shard}"
: "${PHASE2_BASELINE_ROOT:?Set PHASE2_BASELINE_ROOT outside the Git checkout}"

REPO_ROOT="${PHASE2_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
PYTHON_BIN="${PHASE2_PYTHON:-python}"
MODEL_CONFIG="${PHASE2_BASELINE_CONFIG:-$REPO_ROOT/configs/phase2a_3m_total.json}"
TOKENIZER="${PHASE2_TOKENIZER:-NousResearch/Llama-2-7b-hf}"
STEPS="${PHASE2_SMOKE_STEPS:-500}"
SEQ_LEN="${PHASE2_SEQ_LEN:-2048}"
MICROBATCH="${PHASE2_MICROBATCH:-8}"
GRAD_ACCUM="${PHASE2_GRAD_ACCUM:-12}"
OUTPUT_DIR="$PHASE2_BASELINE_ROOT/$MODEL_TYPE/seed-$SEED"

cd "$REPO_ROOT"
if [ -n "${PHASE2_VENV:-}" ]; then
  source "$PHASE2_VENV/bin/activate"
  PYTHON_BIN=python
fi

mkdir -p "$OUTPUT_DIR"

COMMAND=(
  "$PYTHON_BIN" -m koopman_lm.training.train
  --model_type "$MODEL_TYPE"
  --model_size "$MODEL_CONFIG"
  --data_dir "$PHASE2_DATA_DIR"
  --tokenizer "$TOKENIZER"
  --max_seq_len "$SEQ_LEN"
  --per_device_train_batch_size "$MICROBATCH"
  --gradient_accumulation_steps "$GRAD_ACCUM"
  --max_steps "$STEPS"
  --learning_rate "${PHASE2_BASELINE_LR:-0.0006}"
  --warmup_steps "${PHASE2_WARMUP_STEPS:-20}"
  --weight_decay "${PHASE2_WEIGHT_DECAY:-0.1}"
  --max_grad_norm "${PHASE2_GRAD_CLIP:-1.0}"
  --bf16
  --no_compile
  --no_gradient_checkpointing
  --num_workers "${PHASE2_NUM_WORKERS:-4}"
  --logging_steps "${PHASE2_LOGGING_STEPS:-10}"
  --save_steps "${PHASE2_SAVE_STEPS:-500}"
  --output_dir "$OUTPUT_DIR"
  --phase_tag phase2-infra-baseline
  --seed "$SEED"
)

if [ -n "${WANDB_PROJECT:-}" ]; then
  COMMAND+=(--wandb_project "$WANDB_PROJECT" --wandb_group phase2-infra-baselines)
fi

echo "model=$MODEL_TYPE seed=$SEED config=$MODEL_CONFIG output=$OUTPUT_DIR"
printf 'command='
printf '%q ' "${COMMAND[@]}"
printf '\n'
if [ "${PHASE2_DRY_RUN:-0}" = "1" ]; then
  exit 0
fi
"${COMMAND[@]}"
