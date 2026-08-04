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
#   export PHASE2_DATA_DIR=/absolute/path/to/pretokenized/data
#   export PHASE2_BASELINE_ROOT=/absolute/path/to/output   # outside the checkout
#   export PHASE2_VENV=/absolute/path/to/venv              # optional
#   sbatch -A <account> -p <partition> \
#          -o <logdir>/p2a_baseline_%A_%a.out \
#          -e <logdir>/p2a_baseline_%A_%a.err \
#          scripts/slurm_phase2a_baseline_smoke.sh
#
# Account, partition, and log paths are deliberately NOT baked in: they are
# site-private and differ per cluster. Supply them to sbatch. Logs must land
# outside the Git checkout.

#SBATCH --job-name=p2a-baseline-smoke
#SBATCH --array=0-3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00

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

DRY_RUN="${PHASE2_DRY_RUN:-0}"

cd "$REPO_ROOT"

# Site modules are opt-in so the script stays cluster-neutral.
if [ "$DRY_RUN" != "1" ] && [ -n "${PHASE2_MODULES:-}" ]; then
  if command -v module >/dev/null 2>&1; then
    module purge
    # shellcheck disable=SC2086
    module load $PHASE2_MODULES
  else
    echo "WARN: PHASE2_MODULES set but 'module' is unavailable" >&2
  fi
fi

if [ -n "${PHASE2_VENV:-}" ]; then
  # shellcheck disable=SC1091
  source "$PHASE2_VENV/bin/activate"
  PYTHON_BIN=python
fi

# Keep HF/Triton/Torch caches off the small home filesystem. Triton and the
# Torch extension dir are made task-local so concurrent array tasks on one
# node cannot race on the same compile cache.
if [ -n "${PHASE2_CACHE_ROOT:-}" ]; then
  export HF_HOME="${HF_HOME:-$PHASE2_CACHE_ROOT/hf}"
  export TRITON_CACHE_DIR="$PHASE2_CACHE_ROOT/triton/task-${SLURM_ARRAY_JOB_ID:-local}-$INDEX"
  export TORCH_EXTENSIONS_DIR="$PHASE2_CACHE_ROOT/torch_ext/task-${SLURM_ARRAY_JOB_ID:-local}-$INDEX"
  export TMPDIR="${TMPDIR:-$PHASE2_CACHE_ROOT/tmp}"
  [ "$DRY_RUN" = "1" ] || mkdir -p "$HF_HOME" "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR" "$TMPDIR"
fi

# ---- fail early on missing prerequisites -------------------------------
fail=0
check() {  # check <label> <path>
  if [ -e "$2" ]; then
    echo "  OK      $1: $2"
  else
    echo "  MISSING $1: $2"
    fail=1
  fi
}
echo "--- prerequisite checks ---"
check "model config" "$MODEL_CONFIG"
check "data dir"     "$PHASE2_DATA_DIR"
check "train.bin"    "$PHASE2_DATA_DIR/train.bin"
check "meta.json"    "$PHASE2_DATA_DIR/meta.json"
if [ -n "${PHASE2_VENV:-}" ]; then
  check "venv python" "$PHASE2_VENV/bin/python"
fi

# The output root must not live inside the Git checkout.
case "$(readlink -m "$PHASE2_BASELINE_ROOT")/" in
  "$(readlink -m "$REPO_ROOT")"/*)
    echo "  ERROR   PHASE2_BASELINE_ROOT is inside the Git checkout: $PHASE2_BASELINE_ROOT"
    fail=1 ;;
  *) echo "  OK      output root is outside the checkout" ;;
esac

if [ "$fail" -ne 0 ]; then
  if [ "$DRY_RUN" = "1" ]; then
    echo "  (dry run: continuing despite missing prerequisites)"
  else
    echo "ERROR: prerequisite check failed; refusing to start." >&2
    exit 2
  fi
fi

# ---- provenance banner (no credentials) --------------------------------
GIT_COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
GIT_DIRTY="$(git -C "$REPO_ROOT" status --porcelain 2>/dev/null | head -c1)"
[ -n "$GIT_DIRTY" ] && GIT_DIRTY="dirty" || GIT_DIRTY="clean"
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)"
EFF_BATCH=$((MICROBATCH * GRAD_ACCUM))
LR="${PHASE2_BASELINE_LR:-0.0006}"

cat <<BANNER
=================== phase2a baseline smoke ===================
  job            : ${SLURM_ARRAY_JOB_ID:-none}_${SLURM_ARRAY_TASK_ID:-none} (task index $INDEX)
  hostname       : $(hostname)
  gpu            : ${GPU_NAME:-<none visible>}
  git commit     : $GIT_COMMIT ($GIT_DIRTY)
  model_type     : $MODEL_TYPE
  seed           : $SEED
  model config   : $MODEL_CONFIG
  tokenizer      : $TOKENIZER
  seq len        : $SEQ_LEN
  microbatch     : $MICROBATCH
  grad accum     : $GRAD_ACCUM
  EFFECTIVE BATCH: $EFF_BATCH   (tokens/step = $((EFF_BATCH * SEQ_LEN)))
  learning rate  : $LR   [INFRA SMOKE ONLY - not the frozen scientific LR]
  max steps      : $STEPS
  data dir       : $PHASE2_DATA_DIR
  output dir     : $OUTPUT_DIR
==============================================================
BANNER

[ "$DRY_RUN" = "1" ] || mkdir -p "$OUTPUT_DIR"

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
  --learning_rate "$LR"
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
if [ "$DRY_RUN" = "1" ]; then
  exit 0
fi

# Sample peak GPU memory out-of-band so the shared trainer stays untouched.
MEM_LOG="$OUTPUT_DIR/gpu_mem_samples.txt"
( while true; do
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
      | head -1 >> "$MEM_LOG"
    sleep 5
  done ) &
MEM_SAMPLER_PID=$!
trap 'kill "$MEM_SAMPLER_PID" 2>/dev/null' EXIT

START_TS=$(date +%s)
set +e
"${COMMAND[@]}"
RC=$?
set -e
END_TS=$(date +%s)

kill "$MEM_SAMPLER_PID" 2>/dev/null
trap - EXIT

PEAK_MIB=$(sort -n "$MEM_LOG" 2>/dev/null | tail -1)
echo "=================== smoke run summary ==================="
echo "  exit code        : $RC"
echo "  elapsed seconds  : $((END_TS - START_TS))"
echo "  peak GPU mem MiB : ${PEAK_MIB:-unknown}"
echo "  output dir       : $OUTPUT_DIR"
echo "========================================================="
exit $RC
