#!/bin/bash
set -euo pipefail

# ============================================================================
# launch_fast.sh -- Train + evaluate all three models.
#
# Default: 3 models on 3 separate GPUs (one each):
#   GPU 0: mamba_only   (pure Mamba-2, no global retrieval)
#   GPU 1: mamba_attn   (Mamba-2 + Flash Attention + SwiGLU)
#   GPU 2: koopman      (Mamba-2 + SKA + Koopman MLP)
#
# Or train a single model:
#   MODEL=koopman    bash launch_fast.sh
#   MODEL=mamba_attn bash launch_fast.sh
#   MODEL=mamba_only bash launch_fast.sh
#
# Override GPU assignment:
#   GPU_MAMBA_ONLY=0  GPU_MAMBA_ATTN=1  GPU_KOOPMAN=2  bash launch_fast.sh
#
# All models use identical data, hyperparameters, seeds, and evaluation.
# ============================================================================

export OMP_NUM_THREADS=16
export TOKENIZERS_PARALLELISM=false

# ---- Shared configuration (identical for all models) ----
MODEL_SIZE="${MODEL_SIZE:-180m}"
DATASET="${DATASET:-HuggingFaceFW/fineweb-edu}"
SUBSET="${SUBSET:-sample-10BT}"
TOKENIZER="${TOKENIZER:-mistralai/Mistral-7B-v0.1}"
SEQ_LEN="${SEQ_LEN:-2048}"
BATCH="${BATCH:-64}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
LR="${LR:-6e-4}"
STEPS="${STEPS:-100000}"
DATA_DIR="${DATA_DIR:-./tokenized_data}"
WORKERS="${WORKERS:-4}"
SEED="${SEED:-42}"

# Which model(s) to train
MODEL="${MODEL:-all}"

# GPU assignment (override with env vars)
GPU_MAMBA_ONLY="${GPU_MAMBA_ONLY:-0}"
GPU_MAMBA_ATTN="${GPU_MAMBA_ATTN:-1}"
GPU_KOOPMAN="${GPU_KOOPMAN:-2}"

echo "============================================"
echo "Koopman LM Training Setup"
echo "============================================"
echo "  Model size:      ${MODEL_SIZE}"
echo "  Seq length:      ${SEQ_LEN}"
echo "  Micro batch:     ${BATCH}"
echo "  Grad accum:      ${GRAD_ACCUM}"
echo "  Effective batch: $((BATCH * GRAD_ACCUM))"
echo "  Tokens/step:     $((BATCH * GRAD_ACCUM * SEQ_LEN))"
echo "  Total steps:     ${STEPS}"
echo "  Seed:            ${SEED}"
echo "  Training:        ${MODEL}"
echo ""

# ---- Step 1: Pre-tokenize (skip if exists) ----
if [ ! -f "${DATA_DIR}/train.bin" ]; then
    echo "============================================"
    echo "Step 1: Pre-tokenizing dataset"
    echo "============================================"
    python pretokenize.py \
        --dataset_name "$DATASET" \
        --dataset_subset "$SUBSET" \
        --tokenizer "$TOKENIZER" \
        --output_dir "$DATA_DIR"
    echo ""
else
    echo "Step 1: Pre-tokenized data exists at ${DATA_DIR}"
    echo ""
fi

# ---- Training function ----
run_train() {
    local MODEL_TYPE=$1
    local OUTPUT_DIR=$2
    local GPU=$3

    echo "============================================"
    echo "Training: ${MODEL_TYPE} (${MODEL_SIZE}) on GPU ${GPU}"
    echo "  Output: ${OUTPUT_DIR}"
    echo "============================================"

    CUDA_VISIBLE_DEVICES="${GPU}" python train_fast.py \
        --model_type "${MODEL_TYPE}" \
        --model_size "${MODEL_SIZE}" \
        --data_dir "${DATA_DIR}" \
        --tokenizer "${TOKENIZER}" \
        --max_seq_len "${SEQ_LEN}" \
        --per_device_train_batch_size "${BATCH}" \
        --gradient_accumulation_steps "${GRAD_ACCUM}" \
        --max_steps "${STEPS}" \
        --learning_rate "${LR}" \
        --output_dir "${OUTPUT_DIR}" \
        --num_workers "${WORKERS}" \
        --seed "${SEED}" \
        --bf16 \
        --compile \
        --gradient_checkpointing \
        --wandb_project koopman-lm \
        --save_steps 5000 \
        --logging_steps 10
}

# ---- Step 2: Train ----
MAMBA_ONLY_DIR="./mamba-only-${MODEL_SIZE}-fast"
MAMBA_ATTN_DIR="./mamba-attn-${MODEL_SIZE}-fast"
KOOPMAN_DIR="./koopman-${MODEL_SIZE}-fast"

PIDS=()

if [ "$MODEL" = "mamba_only" ]; then
    run_train "mamba_only" "$MAMBA_ONLY_DIR" "$GPU_MAMBA_ONLY"

elif [ "$MODEL" = "mamba_attn" ]; then
    run_train "mamba_attn" "$MAMBA_ATTN_DIR" "$GPU_MAMBA_ATTN"

elif [ "$MODEL" = "koopman" ]; then
    run_train "koopman" "$KOOPMAN_DIR" "$GPU_KOOPMAN"

elif [ "$MODEL" = "all" ]; then
    # Launch all 3 in parallel, each on its own GPU
    echo "============================================"
    echo "Launching all 3 models in parallel"
    echo "  GPU ${GPU_MAMBA_ONLY}: mamba_only"
    echo "  GPU ${GPU_MAMBA_ATTN}: mamba_attn"
    echo "  GPU ${GPU_KOOPMAN}: koopman"
    echo "============================================"
    echo ""

    run_train "mamba_only" "$MAMBA_ONLY_DIR" "$GPU_MAMBA_ONLY" \
        > "${MAMBA_ONLY_DIR}.log" 2>&1 &
    PIDS+=($!)
    echo "  Started mamba_only  (PID ${PIDS[-1]}, log: ${MAMBA_ONLY_DIR}.log)"

    run_train "mamba_attn" "$MAMBA_ATTN_DIR" "$GPU_MAMBA_ATTN" \
        > "${MAMBA_ATTN_DIR}.log" 2>&1 &
    PIDS+=($!)
    echo "  Started mamba_attn  (PID ${PIDS[-1]}, log: ${MAMBA_ATTN_DIR}.log)"

    run_train "koopman" "$KOOPMAN_DIR" "$GPU_KOOPMAN" \
        > "${KOOPMAN_DIR}.log" 2>&1 &
    PIDS+=($!)
    echo "  Started koopman     (PID ${PIDS[-1]}, log: ${KOOPMAN_DIR}.log)"

    echo ""
    echo "Waiting for all training jobs to finish..."
    echo "  Monitor: tail -f ${MAMBA_ONLY_DIR}.log ${MAMBA_ATTN_DIR}.log ${KOOPMAN_DIR}.log"
    echo ""

    FAIL=0
    for pid in "${PIDS[@]}"; do
        if ! wait "$pid"; then
            echo "  ERROR: PID $pid failed"
            FAIL=1
        fi
    done

    if [ "$FAIL" -ne 0 ]; then
        echo "One or more training jobs failed. Check logs."
        exit 1
    fi
    echo "All training jobs complete."
    echo ""

else
    echo "ERROR: Unknown MODEL=${MODEL}"
    echo "  Valid options: all, koopman, mamba_attn, mamba_only"
    exit 1
fi

# ---- Step 3: Evaluate & compare (only when all 3 checkpoints exist) ----
ALL_EXIST=true
for DIR in "$MAMBA_ONLY_DIR" "$MAMBA_ATTN_DIR" "$KOOPMAN_DIR"; do
    if [ ! -f "${DIR}/final/model.pt" ]; then
        ALL_EXIST=false
    fi
done

if [ "$ALL_EXIST" = true ]; then
    echo "============================================"
    echo "Step 3: Evaluating all 3 models"
    echo "============================================"

    # Pairwise comparisons: Koopman vs each baseline
    python evaluate.py \
        --checkpoint "${KOOPMAN_DIR}/final/model.pt" \
        --checkpoint2 "${MAMBA_ATTN_DIR}/final/model.pt" \
        --model_size "${MODEL_SIZE}" \
        --max_seq_len "${SEQ_LEN}" \
        --niah_context_lens 128 256 512 1024 2048 4096 \
        --output koopman_vs_mamba_attn.json

    python evaluate.py \
        --checkpoint "${KOOPMAN_DIR}/final/model.pt" \
        --checkpoint2 "${MAMBA_ONLY_DIR}/final/model.pt" \
        --model_size "${MODEL_SIZE}" \
        --max_seq_len "${SEQ_LEN}" \
        --niah_context_lens 128 256 512 1024 2048 4096 \
        --output koopman_vs_mamba_only.json

    python evaluate.py \
        --checkpoint "${MAMBA_ATTN_DIR}/final/model.pt" \
        --checkpoint2 "${MAMBA_ONLY_DIR}/final/model.pt" \
        --model_size "${MODEL_SIZE}" \
        --max_seq_len "${SEQ_LEN}" \
        --niah_context_lens 128 256 512 1024 2048 4096 \
        --output mamba_attn_vs_mamba_only.json

    echo ""
    echo "============================================"
    echo "All done! Results:"
    echo "  koopman_vs_mamba_attn.json"
    echo "  koopman_vs_mamba_only.json"
    echo "  mamba_attn_vs_mamba_only.json"
    echo "============================================"

elif [ "$MODEL" != "all" ]; then
    # Single model evaluation
    CKPT_DIR=""
    case "$MODEL" in
        koopman)    CKPT_DIR="$KOOPMAN_DIR" ;;
        mamba_attn) CKPT_DIR="$MAMBA_ATTN_DIR" ;;
        mamba_only) CKPT_DIR="$MAMBA_ONLY_DIR" ;;
    esac

    if [ -n "$CKPT_DIR" ] && [ -f "${CKPT_DIR}/final/model.pt" ]; then
        echo "============================================"
        echo "Step 3: Evaluating ${MODEL}"
        echo "============================================"
        python evaluate.py \
            --checkpoint "${CKPT_DIR}/final/model.pt" \
            --model_size "${MODEL_SIZE}" \
            --max_seq_len "${SEQ_LEN}" \
            --niah_context_lens 128 256 512 1024 2048 4096 \
            --output "${MODEL}_results.json"
    fi
fi
