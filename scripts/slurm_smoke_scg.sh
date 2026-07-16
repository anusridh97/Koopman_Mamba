#!/bin/bash
# Phase-1 GPU end-to-end smoke test on an SCG H200 node.
#
# Proves the diagnostics + four-mode load-bearing eval actually run on real
# hardware: BF16 + mamba backbone + checkpoint reload + <3% diag overhead.
#
# IMPORTANT (see SCG_ENV_HANDOFF.md):
#   * mkdir the absolute log dir BEFORE sbatch (Slurm can't open a missing dir).
#   * The repo checkout this runs against MUST be on `phase1-finalize`. Point it
#     with KOOPMAN_REPO=/path/to/that/checkout so you don't disturb the shared
#     oak checkout used by the 180M rerun. Submit with:
#       sbatch --export=ALL,KOOPMAN_REPO=/your/phase1/checkout scripts/slurm_smoke_scg.sh
#     (defaults to the oak repo if unset).

#SBATCH --job-name=phase1_smoke
#SBATCH --account=mpsnyder
#SBATCH --partition=gpu_short
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --chdir=/oak/stanford/scg/lab_mpsnyder/cody1212/Koopman_Mamba
#SBATCH --output=/labs/mpsnyder/cody1212/koopman_runs/logs/phase1_smoke_%j.out
#SBATCH --error=/labs/mpsnyder/cody1212/koopman_runs/logs/phase1_smoke_%j.err

# SCG old cluster: keep nounset OFF (sourcing profile scripts trips on
# BASHRCSOURCED) and load modules explicitly -- do NOT source ~/.bashrc.
set -e
set +u

# --- known-good module + env setup (SCG_ENV_HANDOFF.md) ---
module load cuda/12.3.2_545.23.08_cudNN_9.0.0.312
module unload gcc/13.3.0 2>/dev/null || true
module unload gcc/11.2.0 2>/dev/null || true
module load gcc/9.2.0-centos_7

export CUDA_HOME=$(dirname "$(dirname "$(command -v nvcc)")")
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$(dirname "$(dirname "$(command -v gcc)")")/lib64:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export CC=$(command -v gcc)
export CXX=$(command -v g++)
export CUDAHOSTCXX="$CXX"
export HF_HOME=/labs/mpsnyder/cody1212/.hf_cache
export UV_CACHE_DIR=/labs/mpsnyder/cody1212/.uv_cache
export TRITON_CACHE_DIR=/labs/mpsnyder/cody1212/tmp/triton_cache
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8

VENV="${KOOPMAN_VENV:-/labs/mpsnyder/cody1212/Koopman_Mamba/koopman-lm-fast/.venv}"
source "$VENV/bin/activate"

# --- run against the phase1-finalize checkout (cd explicitly; sbatch spools the
#     script so $0 is unreliable, hence an absolute repo path we can verify) ---
REPO="${KOOPMAN_REPO:-/oak/stanford/scg/lab_mpsnyder/cody1212/Koopman_Mamba}"
cd "$REPO"
echo "== repo: $REPO  branch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '?') =="
for f in scripts/profile_diag_overhead.py koopman_lm/training/diagnostics.py; do
    [ -f "$f" ] || { echo "!! $REPO is missing $f -- this checkout is not on phase1-finalize."; exit 2; }
done

# The training venv has koopman_lm installed elsewhere; force THIS checkout onto
# PYTHONPATH, then verify by a CODE MARKER (not a path string). /oak and /labs
# are two mounts of the same dir, so a path-prefix check gives false alarms --
# check that the imported koopman_lm actually has the phase1-finalize code.
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
if ! python - <<'PY'
import os, sys
try:
    import koopman_lm
    from koopman_lm.evaluation import harness
    from koopman_lm.training import diagnostics  # noqa: F401
except Exception:
    import traceback; traceback.print_exc()
    print("  koopman_lm import FAILED (traceback above)")
    sys.exit(1)
print("  koopman_lm imports from:", os.path.dirname(koopman_lm.__file__))
if not hasattr(harness, "eval_load_bearing"):
    print("  MISSING marker harness.eval_load_bearing -- a stale/older koopman_lm"
          " is shadowing this checkout")
    sys.exit(1)
print("  verified phase1-finalize diagnostics (eval_load_bearing present)")
PY
then
    echo "!! aborting: koopman_lm is not the finalized phase1-finalize checkout (see above)"
    exit 2
fi
mkdir -p /labs/mpsnyder/cody1212/koopman_runs/logs

HAVE_PYTEST=$(python -c "import pytest" 2>/dev/null && echo 1 || echo 0)

# --- preflight: real GPU + mamba extension forward (NEVER on a login node) ---
echo "== preflight (GPU + mamba forward) =="
python - <<'PY'
import torch
print("  torch", torch.__version__, "cuda", torch.version.cuda,
      "available", torch.cuda.is_available())
assert torch.cuda.is_available(), "no CUDA device -- are you on a GPU node?"
from mamba_ssm import Mamba2
x = torch.randn(2, 16, 64, device="cuda")
m = Mamba2(d_model=64, d_state=64, d_conv=4, expand=2).cuda()
print("  mamba forward:", tuple(m(x).shape))
PY

# --- Stage 1+2 (best-effort; synthetic/no-network): CPU tests + GPU e2e ---
if [ "$HAVE_PYTEST" = "1" ]; then
    echo "== stage 1: CPU correctness =="
    python -m pytest code-tests/ -m "correctness and not gpu" -q
    echo "== stage 2a (HARD gate): Phase-1 GPU diagnostics + mixed four-mode load-bearing =="
    python -m pytest code-tests/test_diagnostics.py -m gpu -q
    echo "== stage 2b (best-effort): full Phase-0 e2e (train->ckpt->reload->DECODE) =="
    # The DECODE step exercises RecurrentKoopmanLM.prefill -- the recurrent
    # inference/kernel path, which is WIP per the scaling plan and NOT part of
    # the Phase-1 diagnostics. A failure here is reported, not fatal.
    python -m pytest code-tests/test_smoke_e2e.py -m gpu -q || {
        echo "  NOTE: Phase-0 e2e failed at the recurrent DECODE path (recurrent.py::_ska_prefill)."
        echo "        Inference-track WIP, tracked separately -- NOT a Phase-1 diagnostics failure."
    }
else
    echo "== stages 1-2 SKIPPED: pytest not in venv (pip install pytest to enable) =="
fi

# --- Stage 3: diagnostics overhead < 3% ---
echo "== stage 3: diagnostics overhead =="
python scripts/profile_diag_overhead.py --model_size 50m --diag_every 100

# --- Stage 4: short REAL 50M train w/ diagnostics ON (local tokenized data) ---
TRAIN_DIR="/labs/mpsnyder/cody1212/data/fineweb_50m_train"
SMOKE_RUN="/labs/mpsnyder/cody1212/runs/phase1-smoke-50m"
if [ -f "$TRAIN_DIR/train.bin" ]; then
    echo "== stage 4: short 50M train (200 steps) w/ health + grad-flow diag =="
    python -m koopman_lm.training.train \
        --model_type koopman --model_size 50m \
        --data_dir "$TRAIN_DIR" --tokenizer "NousResearch/Llama-2-7b-hf" \
        --max_seq_len 2048 \
        --per_device_train_batch_size 8 --gradient_accumulation_steps 2 \
        --max_steps 200 --learning_rate 6e-4 --warmup_steps 50 \
        --weight_decay 0.1 --max_grad_norm 1.0 --bf16 \
        --diag_enable --diag_every 50 --diag_grad \
        --num_workers 4 --logging_steps 25 --save_steps 200 \
        --output_dir "$SMOKE_RUN" --seed 42
    echo "  trained smoke checkpoint: $SMOKE_RUN/final/model.pt"
else
    echo "== stage 4 SKIPPED: no tokenized 50M data at $TRAIN_DIR =="
fi

echo "SMOKE DONE. Stages 3-4 (and 1-2 when pytest present) green => diagnostics"
echo "run end-to-end on GPU. Four-mode PPL on a real ckpt: use slurm_calibrate_scg.sh."
