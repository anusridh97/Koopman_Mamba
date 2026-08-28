#!/bin/bash
# Evaluate one checkpoint on the repository's RULER subset at 2K context.
#
# Usage:
#   sbatch scripts/eval_ruler_2k.sbatch /path/to/model.pt 3m_nonembed
#
# The two positional arguments are a checkpoint and its fallback model config.
# Checkpoint metadata takes precedence when present.
#SBATCH --job-name=ruler-2k
#SBATCH --account=cis261066-ai
#SBATCH --partition=ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=ruler-2k-%j.out
#SBATCH --error=ruler-2k-%j.err

set -euo pipefail

if [ "$#" -ne 2 ]; then
    echo "usage: sbatch $0 /path/to/model.pt <model_size>" >&2
    exit 2
fi

CHECKPOINT="$1"
MODEL_SIZE="$2"

module purge
module load modtree/gpu
module load cuda/13.1.0
unset PYTHONPATH PYTHONHOME

source "$SCRATCH/research/Koopman_Mamba/.venv/bin/activate"
cd "$SCRATCH/research/Koopman_Mamba"

python - "$CHECKPOINT" "$MODEL_SIZE" <<'PY'
import json
import sys
from pathlib import Path

import torch

from experimentation.evaluation.loader import load_model
from experimentation.evaluation.ruler import eval_ruler_subset

checkpoint, model_size = sys.argv[1:]
device = torch.device("cuda")
model, _, tokenizer, _ = load_model(
    checkpoint,
    model_size=model_size,
    tokenizer_name="NousResearch/Llama-2-7b-hf",
)
model = model.to(device).eval()
metrics = eval_ruler_subset(
    model,
    tokenizer,
    device,
    context_lens=(2048,),
    n_examples=50,
)
output = Path(checkpoint).parent / "ruler-2k.json"
output.write_text(json.dumps(metrics, indent=2))
print(json.dumps(metrics, indent=2))
print(f"Wrote {output}")
PY
