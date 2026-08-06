#!/bin/bash
# Pre-flight validation for the Echo-50M FineWeb-Edu pretraining run
# (scripts/slurm_pretrain_50m.sh). Exercises every code path that run will use
# -- data pipeline, training loop, checkpointing, and all three eval modes --
# at tiny scale, so a broken tokenizer/HF-auth/CLI/OOM issue surfaces in
# minutes instead of after burning most of a multi-hour, multi-GPU-hour
# budget on the real 3B-token run.
#
# Fails fast: `set -e` stops the script (and the whole job) at the first
# failing stage. Check the SLURM .err log to see which stage failed.
#
# Submit: sbatch scripts/slurm_validate_50m.sh
# Run this BEFORE: sbatch scripts/slurm_pretrain_50m.sh

#SBATCH --job-name=echo50m_validate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/echo50m_validate_%j.out
#SBATCH --error=logs/echo50m_validate_%j.err

set -e
mkdir -p logs

module load python/3.11 cuda/12.4 gcc/12
source "$SCRATCH/Koopman_Mamba/.venv/bin/activate"

TOKENIZER="NousResearch/Llama-2-7b-hf"
DRYRUN_DIR="$SCRATCH/data/fineweb_50m_dryrun_train"
DRYRUN_VAL_DIR="$SCRATCH/data/fineweb_50m_dryrun_val"
DRYRUN_RUN="$SCRATCH/runs/echo-50m-dryrun"
RESULTS_DIR="$SCRATCH/results"
mkdir -p "$RESULTS_DIR"

# ----------------------------------------------------------------------------
echo "=== Stage 1/5: CPU correctness suite (fast, no GPU/network needed) ==="
pytest code-tests/ -m "correctness and not gpu" -q

# ----------------------------------------------------------------------------
echo "=== Stage 2/5: GPU smoke test (synthetic data: train -> checkpoint -> reload -> decode) ==="
pytest code-tests/test_smoke_e2e.py -m gpu -q -v

# ----------------------------------------------------------------------------
echo "=== Stage 3/5: Real-data mini pretokenize (5M train + 1M val tokens, pure FineWeb-Edu) ==="
python -m koopman_lm.training.data.pretokenize \
    --output_dir "$DRYRUN_DIR" \
    --tokenizer "$TOKENIZER" \
    --fineweb HuggingFaceFW/fineweb-edu --fineweb_subset sample-10BT \
    --mix 1.0 0.0 0.0 \
    --max_tokens 5000000 --shard_size 5000000 --seed 42

DOCS_CONSUMED=$(python -c "import json; print(json.load(open('$DRYRUN_DIR/meta.json'))['fineweb_docs_consumed'])")
SKIP_DOCS=$((DOCS_CONSUMED + 1000))
python -m koopman_lm.training.data.pretokenize \
    --output_dir "$DRYRUN_VAL_DIR" \
    --tokenizer "$TOKENIZER" \
    --fineweb HuggingFaceFW/fineweb-edu --fineweb_subset sample-10BT \
    --mix 1.0 0.0 0.0 \
    --max_tokens 1000000 --skip_docs "$SKIP_DOCS" --seed 1337

# ----------------------------------------------------------------------------
echo "=== Stage 4/5: Real-data mini train (100 steps, same config as the full run) ==="
python -m koopman_lm.training.train \
    --model_type koopman --model_size 50m \
    --data_dir "$DRYRUN_DIR" \
    --tokenizer "$TOKENIZER" \
    --max_seq_len 2048 \
    --per_device_train_batch_size 16 --gradient_accumulation_steps 6 \
    --max_steps 100 --learning_rate 6e-4 --warmup_steps 10 \
    --weight_decay 0.1 --max_grad_norm 1.0 \
    --bf16 --compile --no_gradient_checkpointing \
    --num_workers 2 --logging_steps 10 --save_steps 100 \
    --output_dir "$DRYRUN_RUN" --seed 42

# ----------------------------------------------------------------------------
echo "=== Stage 5/5: Evaluation smoke test (all three modes touched by the plan) ==="
echo "--- ppl (WikiText-103) ---"
python -m koopman_lm.evaluation.evaluate \
    --checkpoint "$DRYRUN_RUN/final/model.pt" \
    --model_size 50m --mode ppl --max_seq_len 2048 --batch_size 4 \
    --output "$RESULTS_DIR/dryrun_wikitext_ppl.json"

echo "--- fineweb_ppl (held-out FineWeb-Edu val shard) ---"
python -m koopman_lm.evaluation.evaluate \
    --checkpoint "$DRYRUN_RUN/final/model.pt" \
    --model_size 50m --mode fineweb_ppl --held_out_data_dir "$DRYRUN_VAL_DIR" \
    --max_seq_len 2048 --batch_size 4 \
    --output "$RESULTS_DIR/dryrun_fineweb_ppl.json"

echo "--- zero-shot lm-eval-harness (tiny --limit, only if [lmharness] extra is installed) ---"
if python -c "import lm_eval" 2>/dev/null; then
    python -m koopman_lm.evaluation.lm_harness_eval \
        --model koopman \
        --model_args checkpoint=$DRYRUN_RUN/final/model.pt,model_size=50m,max_length=2048 \
        --tasks hellaswag --limit 5 --batch_size 4 --device cuda \
        --output_path "$RESULTS_DIR/dryrun_zeroshot.json"
else
    echo "  lm_eval not installed (pip install -e \".[lmharness]\") -- skipping this check."
    echo "  Not required to proceed, but the real run's zero-shot eval step will need it."
fi

echo ""
echo "=== ALL VALIDATION STAGES PASSED ==="
echo "Safe to submit the full run: sbatch scripts/slurm_pretrain_50m.sh"
