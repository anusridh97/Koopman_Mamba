#!/bin/bash
# SCG script used for the Echo-50M Table 4 reproduction attempt.
# Trains Echo-50M on 3B FineWeb-Edu tokens, then checkpoints under /labs.
#
# Submit from repo root:
#   sbatch scripts/slurm_pretrain_50m_scg_repro.sh

#SBATCH --job-name=echo50m_pretrain
#SBATCH --account=mpsnyder
#SBATCH --partition=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --chdir=/oak/stanford/scg/lab_mpsnyder/cody1212/Koopman_Mamba
#SBATCH --output=/labs/mpsnyder/cody1212/koopman_runs/logs/echo50m_pretrain_%j.out
#SBATCH --error=/labs/mpsnyder/cody1212/koopman_runs/logs/echo50m_pretrain_%j.err

set -e

source ~/.bashrc
module load cuda/12.3.2_545.23.08_cudNN_9.0.0.312
module unload gcc/13.3.0 2>/dev/null || true
module unload gcc/11.2.0 2>/dev/null || true
module load gcc/9.2.0-centos_7

export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$(dirname $(dirname $(which gcc)))/lib64:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export CC=$(which gcc)
export CXX=$(which g++)
export CUDAHOSTCXX=$(which g++)
export TRITON_CACHE_DIR=/labs/mpsnyder/cody1212/tmp/triton_cache
export HF_HOME=/labs/mpsnyder/cody1212/.hf_cache
export UV_CACHE_DIR=/labs/mpsnyder/cody1212/.uv_cache
export OMP_NUM_THREADS=8

source /labs/mpsnyder/cody1212/Koopman_Mamba/koopman-lm-fast/.venv/bin/activate

TOKENIZER="NousResearch/Llama-2-7b-hf"
TRAIN_DIR="/labs/mpsnyder/cody1212/data/fineweb_50m_train"
VAL_DIR="/labs/mpsnyder/cody1212/data/fineweb_50m_val"
RUN_DIR="/labs/mpsnyder/cody1212/runs/echo-50m-fineweb-3B"

mkdir -p /labs/mpsnyder/cody1212/koopman_runs/logs

if [ ! -f "$TRAIN_DIR/train.bin" ]; then
    echo "Tokenizing train shard (3B tokens, pure FineWeb-Edu)..."
    python -m koopman_lm.training.data.pretokenize \
        --output_dir "$TRAIN_DIR" \
        --tokenizer "$TOKENIZER" \
        --fineweb HuggingFaceFW/fineweb-edu --fineweb_subset sample-10BT \
        --mix 1.0 0.0 0.0 \
        --max_tokens 3000000000 --shard_size 50000000 --seed 42
else
    echo "Train shard already exists at $TRAIN_DIR, skipping tokenization."
fi

if [ ! -f "$VAL_DIR/train.bin" ]; then
    DOCS_CONSUMED=$(python -c "import json; print(json.load(open('$TRAIN_DIR/meta.json'))['fineweb_docs_consumed'])")
    SKIP_DOCS=$((DOCS_CONSUMED + 10000))
    echo "Tokenizing held-out val shard (skip_docs=$SKIP_DOCS)..."
    python -m koopman_lm.training.data.pretokenize \
        --output_dir "$VAL_DIR" \
        --tokenizer "$TOKENIZER" \
        --fineweb HuggingFaceFW/fineweb-edu --fineweb_subset sample-10BT \
        --mix 1.0 0.0 0.0 \
        --max_tokens 20000000 --skip_docs "$SKIP_DOCS" --seed 1337
else
    echo "Val shard already exists at $VAL_DIR, skipping tokenization."
fi

echo "Training Echo-50M..."
python -m koopman_lm.training.train \
    --model_type koopman --model_size 50m \
    --data_dir "$TRAIN_DIR" \
    --tokenizer "$TOKENIZER" \
    --max_seq_len 2048 \
    --per_device_train_batch_size 16 --gradient_accumulation_steps 6 \
    --max_steps 15000 --learning_rate 6e-4 --warmup_steps 300 \
    --weight_decay 0.1 --max_grad_norm 1.0 \
    --bf16 --compile --no_gradient_checkpointing \
    --num_workers 4 --logging_steps 10 --save_steps 1000 \
    --output_dir "$RUN_DIR" --seed 42

echo "Done. Checkpoint: $RUN_DIR/final/model.pt"
