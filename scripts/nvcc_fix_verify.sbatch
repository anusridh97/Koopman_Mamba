#!/bin/bash
#SBATCH --job-name=koopman-nvcc-fix
#SBATCH --account=marlowe-m000151-pm06
#SBATCH --partition=batch
#SBATCH --qos=medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/m000151-pm06/jkli/logs/nvcc-fix/%x-%j.out
#SBATCH --error=/scratch/m000151-pm06/jkli/logs/nvcc-fix/%x-%j.err
#
# Proves, on real H100 hardware, that the one-token nvcc flag bug in
# koopman_lm/kernels/cuda_prefix_scan.py (`"--lineinfo"` -> nvcc fatal:
# Unknown option '--lineinfo', confirmed on CUDA 12.9 / job 415208) is fixed,
# and that the fused exact SKA prefix-scan CUDA extension -- which
# configs/50m.yaml and configs/180m.yaml select via ska_backend: cuda_prefix,
# and which has never built before this fix -- now compiles and passes its
# correctness suite, including a real forward+backward pass through
# KoopmanLM with ska_backend left AT ITS CONFIG VALUE of 'cuda_prefix' (no
# override to 'pytorch' -- that is the whole point of this job).
#
# Uses the ALREADY-BUILT CUDA venv (do not rebuild -- that took ~25 min):
#   /scratch/m000151-pm06/jkli/venvs/koopman-cuda
#   torch 2.13.0+cu129, cuda 12.9, is_available True, mamba_ssm 2.3.2.post1,
#   causal_conv1d ok.
#
# Every step's exit status is captured and echoed, even on failure (no
# `set -e`), and step 5's rule is honored: if step 1 (build) fails, we skip
# steps 2 and 4 (which depend on a working extension) but still run step 3
# (CPU-coverable contract/parity tests).

set -uo pipefail

# NOT hardcoded, so the same script verifies whichever checkout invoked it --
# the main repo, a worktree, or a subagent's. Hardcoding meant a gate run from a
# different tree either died on `cd` or, worse, silently built and PASSED the old
# tree while reporting on changes it never saw.
#
# SLURM_SUBMIT_DIR, not BASH_SOURCE: sbatch COPIES the script to a spool dir and
# runs it there, so ${BASH_SOURCE[0]} is /cm/local/apps/slurm/var/spool/jobNNN/
# slurm_script and deriving from it yields /cm/local/apps/slurm/var/spool.
# Measured in job 440764, not assumed. BASH_SOURCE is the fallback for running
# this file directly; REPO_ROOT=... overrides both.
export REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
# Whatever it resolved to must actually BE the repo. Without this a submit from
# the wrong cwd silently verifies nothing at all.
if [ ! -d "$REPO_ROOT/experimentation" ] || [ ! -d "$REPO_ROOT/code-tests" ]; then
    echo "FATAL: REPO_ROOT=$REPO_ROOT is not a Koopman_Mamba checkout."
    echo "       sbatch from the repo root, or pass REPO_ROOT=/path/to/checkout."
    exit 1
fi
export SCRATCH=/scratch/m000151-pm06/jkli
export VENV=$SCRATCH/venvs/koopman-cuda
LOGDIR=$SCRATCH/logs/nvcc-fix

echo "=== job ${SLURM_JOB_ID:-<interactive>} on $(hostname) ==="
date -Is
mkdir -p "$LOGDIR"
cd "$REPO_ROOT" || { echo "FATAL: no such repo root $REPO_ROOT"; exit 1; }
echo "repo: $REPO_ROOT"
git rev-parse HEAD
git status --short
echo "--- the fix (koopman_lm/kernels/cuda_prefix_scan.py cuda_flags) ---"
grep -n "lineinfo\|use_fast_math\|extra-device-vectorization\|Xptxas" koopman_lm/kernels/cuda_prefix_scan.py

module purge
module load cuda12.9/toolkit/12.9.1
module load gcc/13.1.0
echo "nvcc: $(command -v nvcc)"
nvcc --version
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader

# shellcheck disable=SC1091
source "$VENV/bin/activate"
export PYTHONPATH=$REPO_ROOT
python -c "
import torch
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'is_available', torch.cuda.is_available())
assert torch.cuda.is_available(), 'expected a visible CUDA device on this GPU node'
print('device', torch.cuda.get_device_name(0), 'capability', torch.cuda.get_device_capability(0))
"
VENV_SANITY_STATUS=$?
echo "STATUS venv_sanity=$VENV_SANITY_STATUS"
if [ "$VENV_SANITY_STATUS" -ne 0 ]; then
    echo "FATAL: the pre-built koopman-cuda venv does not see CUDA on this node. Aborting."
    echo "SUMMARY venv_sanity=FAIL build=SKIP fused_tests=SKIP contract_tests=SKIP model_fwd_bwd=SKIP"
    exit 1
fi

export HF_HOME=$SCRATCH/hf
export TRITON_CACHE_DIR=$SCRATCH/triton-cache
export TORCHINDUCTOR_CACHE_DIR=$SCRATCH/inductor-cache
mkdir -p "$HF_HOME" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
export TORCH_CUDA_ARCH_LIST=9.0
export SKA_REQUIRE_B200=0
export MAX_JOBS=8
export SKA_PREFIX_SCAN_BUILD_VERBOSE=1

##############################################################################
# STEP 1: build the fused SKA prefix-scan CUDA extension for sm_90 (H100)
##############################################################################
echo "############################################################"
echo "### STEP 1: build_prefix_scan_cuda.py (TORCH_CUDA_ARCH_LIST=9.0)"
echo "############################################################"
python scripts/build_prefix_scan_cuda.py 2>&1 | tee "$LOGDIR/build-${SLURM_JOB_ID:-manual}.log"
BUILD_STATUS=${PIPESTATUS[0]}
echo "STATUS step1_build_fused_prefix_scan_sm90=$BUILD_STATUS"

##############################################################################
# STEP 2: fused-kernel correctness suite -- NEVER EXECUTED BEFORE THIS JOB
##############################################################################
if [ "$BUILD_STATUS" -eq 0 ]; then
    echo "############################################################"
    echo "### STEP 2: test_fused_prefix_scan_cuda.py (9 tests, first-ever run)"
    echo "############################################################"
    PYTHONPATH=. python -m pytest code-tests/test_fused_prefix_scan_cuda.py -v 2>&1 \
        | tee "$LOGDIR/fused-cuda-tests-${SLURM_JOB_ID:-manual}.log"
    FUSED_TESTS_STATUS=${PIPESTATUS[0]}
else
    echo "############################################################"
    echo "### STEP 2: SKIPPED -- step 1 (build) failed, nothing to test"
    echo "############################################################"
    FUSED_TESTS_STATUS=-1
fi
echo "STATUS step2_fused_cuda_tests=$FUSED_TESTS_STATUS"

##############################################################################
# STEP 3: surrounding contract + parity tests (run regardless of step 1)
##############################################################################
echo "############################################################"
echo "### STEP 3: test_fused_prefix_scan_contract.py + test_prefix_scan.py"
echo "############################################################"
PYTHONPATH=. python -m pytest code-tests/test_fused_prefix_scan_contract.py code-tests/test_prefix_scan.py -q 2>&1 \
    | tee "$LOGDIR/contract-parity-tests-${SLURM_JOB_ID:-manual}.log"
CONTRACT_TESTS_STATUS=${PIPESTATUS[0]}
echo "STATUS step3_contract_and_parity_tests=$CONTRACT_TESTS_STATUS"

##############################################################################
# STEP 4: THE REAL PROOF -- KoopmanLM(50m) fwd+bwd with ska_backend=cuda_prefix
# (left AT ITS CONFIG VALUE -- NOT overridden to 'pytorch')
##############################################################################
if [ "$BUILD_STATUS" -eq 0 ]; then
    echo "############################################################"
    echo "### STEP 4: KoopmanLM(50m) forward+backward, ska_backend=cuda_prefix"
    echo "############################################################"
    python - <<'PY'
import torch
from koopman_lm.config import build_config
from koopman_lm.models.koopman_lm import KoopmanLM

cfg = build_config('50m')
assert cfg.ska_backend == 'cuda_prefix', f"expected config default 'cuda_prefix', got {cfg.ska_backend!r}"
print('ska_backend (left as configured):', cfg.ska_backend)

model = KoopmanLM(cfg).to('cuda')
n_params = sum(p.numel() for p in model.parameters())
print('PARAM_COUNT', n_params)

torch.manual_seed(0)
ids = torch.randint(0, cfg.vocab_size, (2, 64), device='cuda')
labels = ids.clone()
out = model(input_ids=ids, labels=labels)
loss = out['loss']
print('forward loss', loss.item())
loss.backward()
grad_sq = sum(float(p.grad.detach().float().pow(2).sum()) for p in model.parameters() if p.grad is not None)
grad_norm = grad_sq ** 0.5
print('backward grad_norm', grad_norm)
assert torch.isfinite(loss), 'loss is not finite'
assert grad_norm > 0, 'no gradient reached any parameter -- backward did not really run'
print('STEP4 OK: KoopmanLM(50m) forward+backward executed on', torch.cuda.get_device_name(0), 'with ska_backend=cuda_prefix (the fused CUDA kernel, unmocked)')
PY
    MODEL_FWD_BWD_STATUS=$?
else
    echo "############################################################"
    echo "### STEP 4: SKIPPED -- step 1 (build) failed"
    echo "############################################################"
    MODEL_FWD_BWD_STATUS=-1
fi
echo "STATUS step4_koopmanlm_cuda_prefix_fwd_bwd=$MODEL_FWD_BWD_STATUS"

##############################################################################
# FINAL SUMMARY
##############################################################################
echo "############################################################"
echo "### FINAL SUMMARY"
echo "############################################################"
echo "step1_build_fused_prefix_scan_sm90        = $BUILD_STATUS (0 = the fused kernel BUILT for the first time ever)"
echo "step2_fused_cuda_tests (9, first-ever run) = $FUSED_TESTS_STATUS"
echo "step3_contract_and_parity_tests            = $CONTRACT_TESTS_STATUS"
echo "step4_koopmanlm_cuda_prefix_fwd_bwd        = $MODEL_FWD_BWD_STATUS (0 = real proof: KoopmanLM(50m) trained a step with ska_backend=cuda_prefix)"
date -Is
echo "=== END ==="
