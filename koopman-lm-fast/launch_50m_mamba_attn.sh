#!/bin/bash
# launch_50m_mamba_attn.sh
#
# Train 50M Mamba-2 + Attention baseline (Variant 2 @ 50M scale).
# Same hyperparameters as the main Koopman LM launch.sh.
#
# Architecture:
#   12 layers: 9 Mamba-2 + 3 causal attention (25%)
#   SwiGLU MLPs, tied embeddings, ~49.3M params
#
# Usage:
#   bash launch_50m_mamba_attn.sh                   # 1 GPU
#   NUM_GPUS=2 bash launch_50m_mamba_attn.sh        # 2 GPUs

set -euo pipefail

export NCCL_P2P_DISABLE=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=16

NUM_GPUS="${NUM_GPUS:-1}"
DATASET="${DATASET:-HuggingFaceFW/fineweb-edu}"
SUBSET="${SUBSET:-sample-10BT}"
SEQ_LEN="${SEQ_LEN:-2048}"
BATCH="${BATCH:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
LR="${LR:-6e-4}"
STEPS="${STEPS:-100000}"
OUTPUT="${OUTPUT:-./mamba-attn-50m-output}"

# Update ds_config.json batch sizes
TRAIN_BATCH=$((BATCH * GRAD_ACCUM * NUM_GPUS))
sed -i "s/\"train_batch_size\": [0-9]*/\"train_batch_size\": ${TRAIN_BATCH}/" ds_config.json
sed -i "s/\"train_micro_batch_size_per_gpu\": [0-9]*/\"train_micro_batch_size_per_gpu\": ${BATCH}/" ds_config.json
sed -i "s/\"gradient_accumulation_steps\": [0-9]*/\"gradient_accumulation_steps\": ${GRAD_ACCUM}/" ds_config.json

echo "=============================================="
echo "  50M Mamba-2 + Attention Baseline"
echo "  12 layers: 9 Mamba-2 + 3 Attention (25%)"
echo "  ~49.3M params"
echo "=============================================="
echo "  GPUs:       ${NUM_GPUS}"
echo "  Batch:      ${BATCH} × ${GRAD_ACCUM} × ${NUM_GPUS} = ${TRAIN_BATCH}"
echo "  Seq len:    ${SEQ_LEN}"
echo "  LR:         ${LR}"
echo "  Steps:      ${STEPS}"
echo "  Output:     ${OUTPUT}"
echo "=============================================="

deepspeed --num_gpus="$NUM_GPUS" train_50m_mamba_attn.py \
    --dataset_name "$DATASET" \
    --dataset_subset "$SUBSET" \
    --max_seq_len "$SEQ_LEN" \
    --per_device_train_batch_size "$BATCH" \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --learning_rate "$LR" \
    --max_steps "$STEPS" \
    --output_dir "$OUTPUT" \
    --wandb_run_name "mamba-attn-50m" \
    --deepspeed ds_config.json
