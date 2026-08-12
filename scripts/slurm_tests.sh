#!/bin/bash
#SBATCH --job-name=koopman-tests
#SBATCH --account=marlowe-m000151-pm06
#SBATCH --partition=batch
#SBATCH --qos=medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/m000151/%u/logs/koopman-tests/%x-%j.out
#SBATCH --error=/scratch/m000151/%u/logs/koopman-tests/%x-%j.err
#
# Run the FULL test suite -- including the `gpu`-marked tests that the GitHub
# Actions CPU job cannot touch -- on one GPU node.
#
#   mkdir -p /scratch/m000151/$USER/logs/koopman-tests   # Slurm will NOT mkdir for you
#   sbatch scripts/slurm_tests.sh
#
# Marlowe requires an explicit --account, and the `batch` partition only accepts
# QoS `medium`, which lives on the project sub-account (not the default one).
# Override for a different project with:
#   sbatch --account=<your-account> scripts/slurm_tests.sh
# Dry-run the directives without queueing anything:  sbatch --test-only ...
#
# Environment (all overridable via `sbatch --export=...` or the shell):
#   KOOPMAN_VENV   virtualenv to activate. Must already have a CUDA-enabled torch
#                  and, for the mamba_ssm-backed tests, `pip install -e '.[cuda,dev]'`
#                  (see scripts/setup_env.sh). Tests needing mamba_ssm skip
#                  themselves if it is absent, so a torch-only venv still works.
#   PYTEST_ARGS    extra pytest args, e.g. PYTEST_ARGS='-x -k cholesky'.
#   PYTEST_MARKS   marker expression. Default empty = run everything. Use
#                  PYTEST_MARKS=gpu for the GPU subset only.
#
# Partitions on this cluster: hero (30d), batch (2d), preempt (12h). `batch` is
# the right home for a ~1h job: preempt would risk a mid-run kill and hero is
# for long training runs. Nodes have 8 GPUs; the suite is single-device, so we
# ask for exactly one and leave the other seven for training jobs.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

: "${KOOPMAN_VENV:=/scratch/m000151/${USER}/venvs/koopman-cuda}"
: "${PYTEST_ARGS:=}"
: "${PYTEST_MARKS:=}"

echo "=== job ${SLURM_JOB_ID:-<interactive>} on $(hostname) ==="
echo "repo:  $REPO_ROOT"
echo "venv:  $KOOPMAN_VENV"
echo "start: $(date -Is)"

# ---- modules -------------------------------------------------------------
# lmod is not always initialised in a non-interactive Slurm shell.
if ! command -v module >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    [ -f /etc/profile.d/modules.sh ] && source /etc/profile.d/modules.sh
fi
if command -v module >/dev/null 2>&1; then
    module purge || true
    # nvcc is needed only if the fused prefix-scan extension gets JIT-built.
    module load cuda12.9/toolkit/12.9.1 || echo "WARN: cuda toolkit module unavailable"
fi

# ---- python env ----------------------------------------------------------
if [ ! -x "$KOOPMAN_VENV/bin/python" ]; then
    echo "ERROR: no python at $KOOPMAN_VENV/bin/python" >&2
    echo "       create it first (see scripts/setup_env.sh) or set KOOPMAN_VENV." >&2
    exit 1
fi
# shellcheck disable=SC1091
source "$KOOPMAN_VENV/bin/activate"

# Keep HF/W&B/Triton scratch off the small home quota.
export HF_HOME="${HF_HOME:-/scratch/m000151/${USER}/hf}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/scratch/m000151/${USER}/triton-cache}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/scratch/m000151/${USER}/inductor-cache}"
mkdir -p "$HF_HOME" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

nvidia-smi || true
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

# ---- static import gate (cheap; same check the CI cpu job runs) ----------
python scripts/check_imports.py || echo "WARN: check_imports.py reported unresolved imports (continuing)"

# ---- the suite -----------------------------------------------------------
# No `-m "not gpu"`: this job exists precisely to exercise the gpu markers.
# conftest.py auto-skips gpu tests when CUDA is missing, so a bad allocation
# degrades to a green-but-skipped run rather than a false pass -- the
# nvidia-smi/torch banner above is how you tell the two apart.
MARK_ARGS=()
if [ -n "$PYTEST_MARKS" ]; then
    MARK_ARGS=(-m "$PYTEST_MARKS")
fi

set +e
python -m pytest code-tests/ \
    "${MARK_ARGS[@]}" \
    -v -ra --durations=25 --continue-on-collection-errors \
    ${PYTEST_ARGS}
STATUS=$?
set -e

echo "end:   $(date -Is)"
echo "pytest exit status: $STATUS"
exit "$STATUS"
