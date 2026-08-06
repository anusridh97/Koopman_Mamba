#!/bin/bash
# Echo-50M pretraining on FineWeb-Edu -- Table 4 reproduction (50M row only;
# the 180M-class baseline rows in Table 4 are cited from prior published work,
# not trained here). See the paper's Section 4.3/5.3 and the plan this script
# was generated from for the hyperparameter provenance (paper-derived vs.
# reasonable engineering defaults where the paper is silent).
#
# Runs, in order: pretokenize the pure-FineWeb-Edu train shard (3B tokens,
# skipped if train.bin already exists), pretokenize a disjoint held-out val
# shard (skips past however many source documents the train shard consumed,
# via pretokenize.py's --skip_docs), then trains the 50M Echo model.
#
# NOTE: tokenization is CPU-bound and can take a while for 3B tokens -- the
# GPU sits idle during that phase. If your cluster bills/allocates strictly by
# GPU-time, split this into a separate CPU-only preprocessing job and a
# GPU-only training job instead of running both under one #SBATCH allocation.
#
# Adjust --gres=gpu:h100:1 and the per-device batch size / grad-accum below to
# whatever GPU your allocation actually gets (their product must stay 96 to
# match the paper's stated effective batch size for this experiment family).
#
# Submit: sbatch scripts/slurm_pretrain_50m.sh

#SBATCH --job-name=echo50m_pretrain
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/echo50m_pretrain_%j.out
#SBATCH --error=logs/echo50m_pretrain_%j.err

set -e
mkdir -p logs

module load python/3.11 cuda/12.4 gcc/12
source "$SCRATCH/Koopman_Mamba/.venv/bin/activate"

TOKENIZER="NousResearch/Llama-2-7b-hf"   # ungated mirror of meta-llama/Llama-2-7b-hf
TRAIN_DIR="$SCRATCH/data/fineweb_50m_train"
VAL_DIR="$SCRATCH/data/fineweb_50m_val"
RUN_DIR="$SCRATCH/runs/echo-50m-fineweb-3B"

# 1. Train shard: 3B tokens, pure FineWeb-Edu (--mix 1.0 0.0 0.0)
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

# 2. Val shard: disjoint from train (skip past train's consumed docs + margin)
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

# 3. Train the 50M Echo model
# per_device_train_batch_size x gradient_accumulation_steps must equal 96
# (paper's stated effective batch size) times number of GPUs -- adjust both
# to fit your GPU's VRAM while keeping the product at 96 for a single GPU.
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

echo "Done. Evaluate with:"
echo "  python -m koopman_lm.evaluation.evaluate --checkpoint $RUN_DIR/final/model.pt --model_size 50m --mode ppl"
echo "  python -m koopman_lm.evaluation.evaluate --checkpoint $RUN_DIR/final/model.pt --model_size 50m --mode fineweb_ppl --held_out_data_dir $VAL_DIR"
echo "  python -m koopman_lm.evaluation.lm_harness_eval --model koopman --model_args checkpoint=$RUN_DIR/final/model.pt,model_size=50m --tasks hellaswag,piqa,arc_easy,arc_challenge,winogrande,lambada_openai --batch_size 16 --device cuda"
