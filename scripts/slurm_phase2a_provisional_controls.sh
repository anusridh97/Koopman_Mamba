#!/bin/bash
# Six PROVISIONAL fixed-baseline control runs:
#   {mamba_only, transformer} x {seed 42, 43, 44}, 6,000 optimizer steps each.
#
# ============================ PROVISIONAL ==================================
# These are NOT the scientific Phase 2 control path. The finalized
# exact-resume adapter (koopman-phase2-trial-worker) does not exist yet, so
# these runs do NOT provide: the four-flag adapter CLI, manifest consumption,
# pinned-protocol assertions, model/FLOP/GPU-second accounting, the
# optimizer-group audit (the trainer has a single AdamW group), exact resume of
# optimizer/scheduler/RNG/data-offset, metrics/step_500.json, the structured
# failure envelope, or the step-500 MQAR callback.
#
# Data order is seed-reproducible from scratch (the DataLoader generator is
# seeded), but there is no deterministic global sequence index, so a preempted
# run cannot be resumed exactly -- it can only be restarted from step 0.
# Run on a non-preemptible partition.
# ===========================================================================
#
# FROZEN protocol (team decision, do not drift):
#   learning rate      4e-3
#   warmup             120 steps (2% of 6,000)
#   schedule           cosine decay to zero
#   effective batch    96 sequences = 24 microbatch x 4 accumulation
#   sequence length    2,048
#   precision          bf16, no gradient checkpointing, no torch.compile
#   optimizer          AdamW, betas (0.9, 0.95), weight decay 0.1, clip 1.0
#   seeds              42, 43, 44
#   scoring            WikiText-103 VALIDATION (test is reserved for final
#                      reporting and must not be used for selection)
#
# Before submission:
#   export PHASE2_CONTROL_ROOT=/absolute/path/outside/the/checkout
#   export PHASE2_DATA_DIR=/absolute/path/to/frozen/fineweb-edu-train-1p3b
#   export PHASE2_TOKENIZER=/absolute/path/to/frozen/tokenizer-<revision>
#   export PHASE2_VENV=/absolute/path/to/venv
#   sbatch -A <account> -p <non-preemptible partition> \
#          -o <logdir>/p2a_ctrl_%A_%a.out -e <logdir>/p2a_ctrl_%A_%a.err \
#          scripts/slurm_phase2a_provisional_controls.sh

#SBATCH --job-name=p2a-prov-controls
#SBATCH --array=0-5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00

set -eo pipefail
set +u

MODELS=(mamba_only transformer mamba_only transformer mamba_only transformer)
SEEDS=(42 42 43 43 44 44)
INDEX="${SLURM_ARRAY_TASK_ID:-0}"
MODEL_TYPE="${MODELS[$INDEX]}"
SEED="${SEEDS[$INDEX]}"

: "${PHASE2_CONTROL_ROOT:?Set PHASE2_CONTROL_ROOT outside the Git checkout}"
: "${PHASE2_DATA_DIR:?Set PHASE2_DATA_DIR to the frozen 1.3B FineWeb-Edu shard}"
: "${PHASE2_TOKENIZER:?Set PHASE2_TOKENIZER to the frozen tokenizer directory}"

REPO_ROOT="${PHASE2_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
PYTHON_BIN="${PHASE2_PYTHON:-python}"
MODEL_CONFIG="${PHASE2_CONTROL_CONFIG:-$REPO_ROOT/configs/phase2a_3m_total.json}"

# ---- frozen protocol constants -----------------------------------------
STEPS="${PHASE2_CONTROL_STEPS:-6000}"
LR="${PHASE2_CONTROL_LR:-0.004}"
WARMUP="${PHASE2_CONTROL_WARMUP:-120}"
MICROBATCH="${PHASE2_MICROBATCH:-24}"
GRAD_ACCUM="${PHASE2_GRAD_ACCUM:-4}"
SEQ_LEN="${PHASE2_SEQ_LEN:-2048}"
WEIGHT_DECAY="${PHASE2_WEIGHT_DECAY:-0.1}"
GRAD_CLIP="${PHASE2_GRAD_CLIP:-1.0}"
SAVE_STEPS="${PHASE2_SAVE_STEPS:-500}"
LOGGING_STEPS="${PHASE2_LOGGING_STEPS:-50}"
# Identity of the approved training artifact. Overridable, but a mismatch is a
# hard stop: these runs are only comparable against this exact shard.
EXPECT_TRAIN_SHA="${PHASE2_TRAIN_SHA256:-f9f512e06ae34c99d4ee46a3e4340a4eb4b45f1aef74ebd5506be550c40d5e29}"

OUTPUT_DIR="$PHASE2_CONTROL_ROOT/$MODEL_TYPE/seed-$SEED"
DRY_RUN="${PHASE2_DRY_RUN:-0}"
EFF_BATCH=$((MICROBATCH * GRAD_ACCUM))

cd "$REPO_ROOT"

if [ "$DRY_RUN" != "1" ] && [ -n "${PHASE2_MODULES:-}" ]; then
  if command -v module >/dev/null 2>&1; then module purge; module load $PHASE2_MODULES; fi
fi
if [ -n "${PHASE2_VENV:-}" ]; then
  # shellcheck disable=SC1091
  source "$PHASE2_VENV/bin/activate"; PYTHON_BIN=python
fi
if [ -n "${PHASE2_CACHE_ROOT:-}" ]; then
  export HF_HOME="${HF_HOME:-$PHASE2_CACHE_ROOT/hf}"
  export TRITON_CACHE_DIR="$PHASE2_CACHE_ROOT/triton/ctrl-${SLURM_ARRAY_JOB_ID:-local}-$INDEX"
  export TORCH_EXTENSIONS_DIR="$PHASE2_CACHE_ROOT/torch_ext/ctrl-${SLURM_ARRAY_JOB_ID:-local}-$INDEX"
  export TMPDIR="${TMPDIR:-$PHASE2_CACHE_ROOT/tmp}"
  [ "$DRY_RUN" = "1" ] || mkdir -p "$HF_HOME" "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR" "$TMPDIR"
fi

# ---- prerequisites ------------------------------------------------------
fail=0
check() { if [ -e "$2" ]; then echo "  OK      $1"; else echo "  MISSING $1: $2"; fail=1; fi; }
echo "--- prerequisite checks ---"
check "model config"   "$MODEL_CONFIG"
check "train.bin"      "$PHASE2_DATA_DIR/train.bin"
check "meta.json"      "$PHASE2_DATA_DIR/meta.json"
check "tokenizer.json" "$PHASE2_TOKENIZER/tokenizer.json"
check "tokenizer.model" "$PHASE2_TOKENIZER/tokenizer.model"
case "$(readlink -m "$PHASE2_CONTROL_ROOT")/" in
  "$(readlink -m "$REPO_ROOT")"/*) echo "  ERROR   control root inside the checkout"; fail=1 ;;
  *) echo "  OK      control root is outside the checkout" ;;
esac
if [ "$EFF_BATCH" -ne 96 ]; then
  echo "  ERROR   effective batch is $EFF_BATCH, must be 96"; fail=1
else
  echo "  OK      effective batch 96 ($MICROBATCH x $GRAD_ACCUM)"
fi
if [ "$fail" -ne 0 ]; then
  if [ "$DRY_RUN" = "1" ]; then echo "  (dry run: continuing)"; else
    echo "ERROR: prerequisite check failed" >&2; exit 2; fi
fi

# Verify the frozen artifact byte-for-byte (contract: every worker launch
# streams the bulk file and compares its SHA-256).
if [ "$DRY_RUN" != "1" ] && [ "${PHASE2_SKIP_SHA:-0}" != "1" ]; then
  echo "--- verifying frozen training artifact ---"
  ACTUAL_SHA=$(sha256sum "$PHASE2_DATA_DIR/train.bin" | awk '{print $1}')
  if [ "$ACTUAL_SHA" != "$EXPECT_TRAIN_SHA" ]; then
    echo "ERROR: train.bin SHA-256 mismatch." >&2
    echo "  expected $EXPECT_TRAIN_SHA" >&2
    echo "  actual   $ACTUAL_SHA" >&2
    exit 3
  fi
  echo "  OK      train.bin sha256 matches the approved artifact"
fi

GIT_COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
GIT_DIRTY="$(git -C "$REPO_ROOT" status --porcelain 2>/dev/null | head -c1)"
[ -n "$GIT_DIRTY" ] && GIT_DIRTY="dirty" || GIT_DIRTY="clean"
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)"

cat <<BANNER
============ phase2a PROVISIONAL fixed-baseline control ============
  *** PROVISIONAL: not the scientific Phase 2 control path.        ***
  *** No exact-resume adapter, no optimizer-group audit, no MQAR.  ***
  job              : ${SLURM_ARRAY_JOB_ID:-none}_${SLURM_ARRAY_TASK_ID:-none} (index $INDEX)
  hostname / gpu   : $(hostname) / ${GPU_NAME:-<none>}
  git commit       : $GIT_COMMIT ($GIT_DIRTY)
  model_type       : $MODEL_TYPE
  seed             : $SEED
  model config     : $MODEL_CONFIG
  tokenizer        : $PHASE2_TOKENIZER
  data dir         : $PHASE2_DATA_DIR
  seq len          : $SEQ_LEN
  microbatch x acc : $MICROBATCH x $GRAD_ACCUM
  EFFECTIVE BATCH  : $EFF_BATCH  (tokens/step = $((EFF_BATCH * SEQ_LEN)))
  max steps        : $STEPS  (tokens = $((STEPS * EFF_BATCH * SEQ_LEN)))
  learning rate    : $LR   [FROZEN]
  warmup / sched   : $WARMUP steps / cosine decay to zero   [FROZEN]
  weight decay     : $WEIGHT_DECAY   grad clip: $GRAD_CLIP
  checkpoints every: $SAVE_STEPS  (plus a final checkpoint)
  scoring          : WikiText-103 VALIDATION (test reserved for final report)
  output dir       : $OUTPUT_DIR
====================================================================
BANNER

if [ "$DRY_RUN" = "1" ]; then
  printf 'command='
  printf '%q ' "$PYTHON_BIN" -m koopman_lm.training.train \
    --model_type "$MODEL_TYPE" --model_size "$MODEL_CONFIG" \
    --data_dir "$PHASE2_DATA_DIR" --tokenizer "$PHASE2_TOKENIZER" \
    --max_seq_len "$SEQ_LEN" --per_device_train_batch_size "$MICROBATCH" \
    --gradient_accumulation_steps "$GRAD_ACCUM" --max_steps "$STEPS" \
    --learning_rate "$LR" --warmup_steps "$WARMUP" \
    --weight_decay "$WEIGHT_DECAY" --max_grad_norm "$GRAD_CLIP" \
    --bf16 --no_compile --no_gradient_checkpointing \
    --num_workers "${PHASE2_NUM_WORKERS:-4}" --logging_steps "$LOGGING_STEPS" \
    --save_steps "$SAVE_STEPS" --output_dir "$OUTPUT_DIR" \
    --phase_tag phase2-provisional-control --seed "$SEED"
  printf '\n'
  exit 0
fi

mkdir -p "$OUTPUT_DIR"

# Per-run provenance record. Not a substitute for the finalized manifest, but
# it pins what this run actually used.
cat > "$OUTPUT_DIR/run_provenance.json" <<PROV
{
  "status": "PROVISIONAL_NOT_SCIENTIFIC_CONTROL_PATH",
  "study": "echo-phase2a-3m-v1",
  "model_type": "$MODEL_TYPE",
  "seed": $SEED,
  "git_commit": "$GIT_COMMIT",
  "git_state": "$GIT_DIRTY",
  "slurm_job": "${SLURM_ARRAY_JOB_ID:-none}_${SLURM_ARRAY_TASK_ID:-none}",
  "hostname": "$(hostname)",
  "gpu": "${GPU_NAME:-unknown}",
  "model_config": "$MODEL_CONFIG",
  "tokenizer_path": "$PHASE2_TOKENIZER",
  "train_data_dir": "$PHASE2_DATA_DIR",
  "train_bin_sha256": "$EXPECT_TRAIN_SHA",
  "sequence_length": $SEQ_LEN,
  "microbatch": $MICROBATCH,
  "gradient_accumulation": $GRAD_ACCUM,
  "effective_batch": $EFF_BATCH,
  "tokens_per_step": $((EFF_BATCH * SEQ_LEN)),
  "max_steps": $STEPS,
  "total_tokens": $((STEPS * EFF_BATCH * SEQ_LEN)),
  "learning_rate": $LR,
  "warmup_steps": $WARMUP,
  "schedule": "cosine_decay_to_zero",
  "weight_decay": $WEIGHT_DECAY,
  "max_grad_norm": $GRAD_CLIP,
  "precision": "bf16",
  "gradient_checkpointing": false,
  "torch_compile": false,
  "scoring_split": "wikitext103-validation",
  "reserved_for_final_report": "wikitext103-test"
}
PROV

MEM_LOG="$OUTPUT_DIR/gpu_mem_samples.txt"
( while true; do
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 >> "$MEM_LOG"
    sleep 15
  done ) &
MEM_PID=$!
trap 'kill "$MEM_PID" 2>/dev/null' EXIT

START_TS=$(date +%s)
set +e
"$PYTHON_BIN" -m koopman_lm.training.train \
  --model_type "$MODEL_TYPE" --model_size "$MODEL_CONFIG" \
  --data_dir "$PHASE2_DATA_DIR" --tokenizer "$PHASE2_TOKENIZER" \
  --max_seq_len "$SEQ_LEN" --per_device_train_batch_size "$MICROBATCH" \
  --gradient_accumulation_steps "$GRAD_ACCUM" --max_steps "$STEPS" \
  --learning_rate "$LR" --warmup_steps "$WARMUP" \
  --weight_decay "$WEIGHT_DECAY" --max_grad_norm "$GRAD_CLIP" \
  --bf16 --no_compile --no_gradient_checkpointing \
  --num_workers "${PHASE2_NUM_WORKERS:-4}" --logging_steps "$LOGGING_STEPS" \
  --save_steps "$SAVE_STEPS" --output_dir "$OUTPUT_DIR" \
  --phase_tag phase2-provisional-control --seed "$SEED"
RC=$?
set -e
END_TS=$(date +%s)
kill "$MEM_PID" 2>/dev/null; trap - EXIT

PEAK_MIB=$(sort -n "$MEM_LOG" 2>/dev/null | tail -1)
ELAPSED=$((END_TS - START_TS))
echo "================ provisional control summary ================"
echo "  model/seed       : $MODEL_TYPE / $SEED"
echo "  exit code        : $RC"
echo "  elapsed seconds  : $ELAPSED  ($((ELAPSED / 60)) min)"
echo "  peak GPU mem MiB : ${PEAK_MIB:-unknown}"
echo "  output dir       : $OUTPUT_DIR"
echo "============================================================="
exit $RC
