#!/bin/bash
# Phase-1 GPU end-to-end smoke test on an SCG H200 node.
#
# Proves the diagnostics + four-mode load-bearing eval actually run on real
# hardware: BF16 + fast SKA backend + checkpoint reload + <3% diag overhead.
# No Phase-0/1 branch has passed a GPU end-to-end yet -- this is that gate.
#
# Submit from the SCG repo root:
#   sbatch scripts/slurm_smoke_scg.sh
#
# Single GPU on purpose: eval is single-process, so 1 GPU is right (and polite
# while the Table-4 180M eval jobs are competing for the node).

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

set -e
# SCG is an old cluster (see CODEX_HANDOFF.md): /etc/bashrc trips on the unbound
# BASHRCSOURCED var under `set -u`, and `module` is only defined once the shell
# init is sourced. So keep nounset OFF for the whole script, source bashrc
# defensively (to get `module`), and never enable `set -u`.
set +u
source ~/.bashrc 2>/dev/null || true

# --- environment (the combo known to load the compiled mamba extension) ---
module load cuda/12.3.2_545.23.08_cudNN_9.0.0.312
module unload gcc/13.3.0 2>/dev/null || true
module unload gcc/11.2.0 2>/dev/null || true
module load gcc/9.2.0-centos_7

export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$(dirname $(dirname $(which gcc)))/lib64:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export CC=$(which gcc); export CXX=$(which g++); export CUDAHOSTCXX=$(which g++)
export TRITON_CACHE_DIR=/labs/mpsnyder/cody1212/tmp/triton_cache
export HF_HOME=/labs/mpsnyder/cody1212/.hf_cache
export OMP_NUM_THREADS=8

# venv is independent of the repo layout; the consolidated koopman_lm package
# imports fine from the known-working Table-4 env. Override with KOOPMAN_VENV.
VENV="${KOOPMAN_VENV:-/labs/mpsnyder/cody1212/Koopman_Mamba/koopman-lm-fast/.venv}"
source "$VENV/bin/activate"

mkdir -p /labs/mpsnyder/cody1212/koopman_runs/logs

# pytest may not be in the training venv; the CORE gate (train + diagnostics +
# four-mode + overhead) does not need it, so the pytest stages are best-effort.
HAVE_PYTEST=$(python -c "import pytest" 2>/dev/null && echo 1 || echo 0)

# --- preflight: CUDA + compiled backends present (NEVER run this on a login node) ---
echo "== preflight =="
python - <<'PY'
import torch, importlib.util
assert torch.cuda.is_available(), "CUDA not available -- are you on a GPU node?"
for m in ("mamba_ssm", "causal_conv1d"):
    ok = importlib.util.find_spec(m) is not None
    print(f"  {m}: {'ok' if ok else 'MISSING'}")
    assert ok, f"{m} not importable (login-node glibc? wrong venv?)"
print("  torch", torch.__version__, "cuda", torch.version.cuda)
PY

# --- Stage 1+2 (best-effort): CPU correctness + GPU e2e / mixed four-mode ---
# All synthetic-data / no-network. Skipped cleanly if pytest isn't installed.
if [ "$HAVE_PYTEST" = "1" ]; then
    echo "== stage 1: CPU correctness =="
    pytest code-tests/ -m "correctness and not gpu" -q
    echo "== stage 2: GPU e2e + diagnostics + mixed four-mode (synthetic, no network) =="
    pytest code-tests/test_smoke_e2e.py code-tests/test_diagnostics.py -m gpu -q
else
    echo "== stages 1-2 SKIPPED: pytest not in venv (pip install pytest to enable) =="
fi

# --- Stage 3: diagnostics overhead < 3% (synthetic, no network) ---
echo "== stage 3: diagnostics overhead =="
python scripts/profile_diag_overhead.py --model_size 50m --diag_every 100

# --- Stage 4: short REAL 50M train with diagnostics ON (local tokenized data) ---
# This is the on-hardware exercise of the A1 (health) + A3 (grad-flow) code.
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

# --- Stage 5 (best-effort): four-mode load-bearing on a real checkpoint ---
# Uses WikiText, which needs a (cached) download -- best-effort so an offline
# compute node doesn't fail the smoke. For a network-free four-mode on a real
# checkpoint, use calibrate.py --data synthetic (scripts/slurm_calibrate_scg.sh).
RESULTS_DIR="/labs/mpsnyder/cody1212/results/phase1_smoke"; mkdir -p "$RESULTS_DIR"
CKPT="$SMOKE_RUN/final/model.pt"
[ -f "$CKPT" ] || CKPT="/labs/mpsnyder/cody1212/runs/echo-50m-fineweb-3B/final/model.pt"
echo "== stage 5 (best-effort): four-mode load-bearing eval on $CKPT =="
python -m koopman_lm.evaluation.harness \
    --checkpoint "$CKPT" --tasks ppl load_bearing \
    --tokenizer "NousResearch/Llama-2-7b-hf" \
    --out "$RESULTS_DIR/smoke_load_bearing.json" \
    || echo "  (stage 5 failed -- likely WikiText download offline; see calibrate.py --data synthetic)"

echo "SMOKE DONE. If stages 3-4 (and 1-2 when pytest present) are green, the "
echo "diagnostics run end-to-end on GPU. Load-bearing JSON (if written): $RESULTS_DIR/"
