#!/bin/bash +u
# Run the Echo-180M Table 4 zero-shot evaluations on SCG.
#
# Submit from the repository root with:
#   sbatch scripts/slurm_eval_180m_zeroshot_scg.sh

#SBATCH --job-name=echo180m_zs
#SBATCH --account=mpsnyder
#SBATCH --partition=gpu_normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --chdir=/oak/stanford/scg/lab_mpsnyder/cody1212/Koopman_Mamba
#SBATCH --output=/labs/mpsnyder/cody1212/koopman_runs/logs/echo180m_zs_%j.out
#SBATCH --error=/labs/mpsnyder/cody1212/koopman_runs/logs/echo180m_zs_%j.err

set +u
set -eo pipefail

module load cuda/12.3.2_545.23.08_cudNN_9.0.0.312
module unload gcc/13.3.0 2>/dev/null || true
module unload gcc/11.2.0 2>/dev/null || true
module load gcc/9.2.0-centos_7

export CUDA_HOME
CUDA_HOME=$(dirname "$(dirname "$(command -v nvcc)")")
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$(dirname "$(dirname "$(command -v gcc)")")/lib64:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TRITON_CACHE_DIR=/labs/mpsnyder/cody1212/tmp/triton_cache
export HF_HOME=/labs/mpsnyder/cody1212/.hf_cache
export UV_CACHE_DIR=/labs/mpsnyder/cody1212/.uv_cache
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false

VENV=/labs/mpsnyder/cody1212/Koopman_Mamba/koopman-lm-fast/.venv
CHECKPOINT=/labs/mpsnyder/cody1212/runs/echo-180m-fineweb-10B/final/model.pt
RESULT_DIR=/labs/mpsnyder/cody1212/results/echo180m_table4

source "$VENV/bin/activate"
mkdir -p "$RESULT_DIR"
test -s "$CHECKPOINT"

echo "=== Environment preflight ==="
ldd --version | head -n 1
python - <<'PY'
from importlib.metadata import version

import torch
import mamba_ssm
import lm_eval

print("torch", torch.__version__)
print("torch CUDA", torch.version.cuda)
print("CUDA available", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable; evaluation must run on the allocated GPU node")
print("mamba_ssm import OK")
print("lm_eval import OK")
print("lm-eval version", version("lm-eval"))
PY

echo "=== Table 4 zero-shot evaluation ==="
python -m koopman_lm.evaluation.lm_harness_eval \
    --model koopman \
    --model_args "checkpoint=$CHECKPOINT,model_size=180m,max_length=2048" \
    --tasks hellaswag,piqa,arc_easy,arc_challenge,winogrande,lambada_openai \
    --num_fewshot 0 \
    --batch_size 8 \
    --device cuda \
    --output_path "$RESULT_DIR/zeroshot.json"

echo "Zero-shot evaluation complete: $RESULT_DIR/zeroshot.json"
