#!/bin/bash
# Create FineWeb-Edu training shards for 100M and 180M model sweeps.
#
# The budgets use 70 tokens/non-embedding-parameter: 7B tokens for 100M and
# 12.6B tokens for 180M. The pretokenizer
# writes the exact (slightly overshot) count to meta.json; use that exact value
# in each run spec's data.n_tokens field.
#
# Submit from the repository root:
#   sbatch scripts/tokenize_100m_180m.sbatch
#
# Anvil's ai partition requires a GPU request even though tokenization is CPU-bound.
# Reruns skip any shard whose meta.json already exists.
#SBATCH --job-name=fineweb-6b-10b
#SBATCH --account=cis261066-ai
#SBATCH --partition=ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --output=tokenize_100m_180m-%j.out
#SBATCH --error=tokenize_100m_180m-%j.err

set -euo pipefail

module purge
module load modtree/gpu
module load cuda/13.1.0
unset PYTHONPATH PYTHONHOME

source "$SCRATCH/research/Koopman_Mamba/.venv/bin/activate"
cd "$SCRATCH/research/Koopman_Mamba"

TOKENIZER="NousResearch/Llama-2-7b-hf"
DATA_ROOT="$SCRATCH/data"
LOG_ROOT="$SCRATCH/tokenize-logs"
mkdir -p "$DATA_ROOT" "$LOG_ROOT"

tokenize_shard() {
    local label="$1"
    local token_count="$2"
    local output_dir="$DATA_ROOT/$label"

    if [ -f "$output_dir/meta.json" ]; then
        echo "$label already complete; skipping."
    else
        echo "Tokenizing $label with target ${token_count} tokens."
        python -m experimentation.training.data.pretokenize \
            --output_dir "$output_dir" \
            --tokenizer "$TOKENIZER" \
            --fineweb HuggingFaceFW/fineweb-edu \
            --fineweb_subset sample-10BT \
            --mix 1.0 0.0 0.0 \
            --max_tokens "$token_count" \
            --shard_size 50000000 \
            --seed 42 \
            2>&1 | tee "$LOG_ROOT/${label}-${SLURM_JOB_ID}.log"
    fi

    test -f "$output_dir/meta.json"
    python -c "import json; print('$label n_tokens =', json.load(open('$output_dir/meta.json'))['n_tokens'])"
}

tokenize_shard fineweb_7b_train 7000000000
tokenize_shard fineweb_12_6b_train 12600000000

echo "Done. Copy the exact printed n_tokens values into the 100M and 180M run specs."
