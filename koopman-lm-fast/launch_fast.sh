#!/bin/bash
set -euo pipefail

# ============================================================================
# launch_fast.sh -- Train + evaluate both models on separate B200 instances.
#
# Run on B200 #1:  MODEL=koopman    bash launch_fast.sh
# Run on B200 #2:  MODEL=mamba_attn bash launch_fast.sh
#
# Or run both sequentially on one machine:
#   bash launch_fast.sh
#
# Both models train on the exact same data, same hyperparameters, same seeds.
# Evaluation uses the same tasks, seeds, and context lengths.
# ============================================================================

export OMP_NUM_THREADS=16
export TOKENIZERS_PARALLELISM=false

# ---- Shared configuration (identical for both models) ----
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

# Which model to train (default: both)
MODEL="${MODEL:-both}"

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

    echo "============================================"
    echo "Training: ${MODEL_TYPE} (${MODEL_SIZE})"
    echo "  Output: ${OUTPUT_DIR}"
    echo "============================================"

    python train_fast.py \
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

    echo "  Done: ${MODEL_TYPE}"
    echo ""
}

# ---- Step 2: Train ----
KOOPMAN_DIR="./koopman-${MODEL_SIZE}-fast"
MAMBA_ATTN_DIR="./mamba-attn-${MODEL_SIZE}-fast"

if [ "$MODEL" = "koopman" ] || [ "$MODEL" = "both" ]; then
    run_train "koopman" "$KOOPMAN_DIR"
fi

if [ "$MODEL" = "mamba_attn" ] || [ "$MODEL" = "both" ]; then
    run_train "mamba_attn" "$MAMBA_ATTN_DIR"
fi

# ---- Step 3: Evaluate & compare (only if both are done) ----
if [ -f "${KOOPMAN_DIR}/final/model.pt" ] && [ -f "${MAMBA_ATTN_DIR}/final/model.pt" ]; then
    echo "============================================"
    echo "Step 3: Evaluating & comparing both models"
    echo "============================================"

    python evaluate.py \
        --checkpoint "${KOOPMAN_DIR}/final/model.pt" \
        --checkpoint2 "${MAMBA_ATTN_DIR}/final/model.pt" \
        --model_size "${MODEL_SIZE}" \
        --max_seq_len "${SEQ_LEN}" \
        --niah_context_lens 128 256 512 1024 2048 4096 \
        --output comparison_results.json

    echo ""
    echo "============================================"
    echo "All done! Results in comparison_results.json"
    echo "============================================"
elif [ "$MODEL" != "both" ]; then
    # Single model evaluation
    CKPT_DIR=""
    if [ "$MODEL" = "koopman" ]; then
        CKPT_DIR="$KOOPMAN_DIR"
    else
        CKPT_DIR="$MAMBA_ATTN_DIR"
    fi

    if [ -f "${CKPT_DIR}/final/model.pt" ]; then
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
