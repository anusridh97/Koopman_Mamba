#!/bin/bash
# Create the FineWeb-Edu shards for the 30M and 50M non-embedding runs.
# Submit from the repository root:
#   sbatch scripts/tokenize_30m_50m.sbatch
# A shard is complete only when its meta.json exists; reruns skip completed work.
# Anvil's ai partition requires a GPU request even though tokenization is CPU-bound.
#SBATCH --job-name=fineweb-2b-3b
#SBATCH --account=cis261066-ai
#SBATCH --partition=ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=tokenize_30m_50m-%j.out
#SBATCH --error=tokenize_30m_50m-%j.err

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

tokenize_shard fineweb_2b 2000000000
tokenize_shard fineweb_3b 3000000000

echo "Done. Update configs/runs/30m-golden.yaml and configs/runs/50m-nonembed-golden.yaml with the printed exact n_tokens values."
