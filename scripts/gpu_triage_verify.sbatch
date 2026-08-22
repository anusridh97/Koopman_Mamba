#!/bin/bash
#SBATCH --job-name=koopman-gpu-triage
#SBATCH --account=marlowe-m000151-pm06
#SBATCH --partition=batch
#SBATCH --qos=medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/m000151-pm06/jkli/logs/koopman-tests/%x-%j.out
#SBATCH --error=/scratch/m000151-pm06/jkli/logs/koopman-tests/%x-%j.err
#
# Verifies, on real H100 hardware, the fixes made in docs/superpowers/specs/
# 2026-08-07-gpu-triage.md for three of the five independent (non-fused-
# kernel) GPU failures from job 415208:
#   - code-tests/test_prefix_scan.py::test_prefix_scan_model_recurrent_decode_matches_full_prefix
#     (real bug: SKAModule was feeding its legacy triton/pytorch post-Cholesky
#     backend choice into ska_prefix_scan's unrelated backend parameter)
#   - code-tests/test_newton_schulz.py::test_ns_equals_cholesky_bf16
#     (tolerance fix: 1e-3 -> 1e-2, derived from bf16's ~3.9e-3 unit roundoff)
#   - code-tests/test_table2_repro_contract.py::test_table2_forward_and_shifted_loss_smoke
#     (test bug: model/inputs were never moved to .cuda())
# and restores code-tests/test_jax_reference.py (archive/reference/echo_jax.py
# was missing from the tree; also runs on CPU/login node -- included here too
# for one full from-Slurm confirmation).
#
# This worktree (jack/gpu-triage) branches from jack/nvcc-fix, so it also
# carries the nvcc `-lineinfo` fix (job 415255's fix) -- the remaining two
# independent failures (test_full_model_decode_prefill_parity,
# test_e2e_train_checkpoint_reload_decode) and the 9 fused-kernel tests are
# collateral from that build bug, not touched here, so this job runs them too
# as a bonus signal but does not treat their outcome as this job's verdict.
#
# Uses the ALREADY-BUILT CUDA venv (do not rebuild -- ~25 min):
#   /scratch/m000151-pm06/jkli/venvs/koopman-cuda
#   torch 2.13.0+cu129, is_available True, mamba_ssm 2.3.2.post1.
# flax was pip-installed into this venv during triage (pure-python, no CUDA
# compile) so archive/reference/echo_jax.py's jax/flax reference actually runs.

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
LOGDIR=$SCRATCH/logs/koopman-tests

echo "=== job ${SLURM_JOB_ID:-<interactive>} on $(hostname) ==="
date -Is
mkdir -p "$LOGDIR"
cd "$REPO_ROOT" || { echo "FATAL: no such repo root $REPO_ROOT"; exit 1; }
echo "repo: $REPO_ROOT"
git rev-parse HEAD
git status --short

module purge
module load cuda12.9/toolkit/12.9.1
module load gcc/13.1.0
echo "nvcc: $(command -v nvcc)"
nvcc --version || true
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader || true

# shellcheck disable=SC1091
source "$VENV/bin/activate"
export PYTHONPATH=$REPO_ROOT
python -c "
import torch
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'is_available', torch.cuda.is_available())
assert torch.cuda.is_available(), 'expected a visible CUDA device on this GPU node'
print('device', torch.cuda.get_device_name(0))
"
VENV_SANITY_STATUS=$?
echo "STATUS venv_sanity=$VENV_SANITY_STATUS"
if [ "$VENV_SANITY_STATUS" -ne 0 ]; then
    echo "FATAL: koopman-cuda venv does not see CUDA on this node. Aborting."
    exit 1
fi

export HF_HOME=$SCRATCH/hf
export TRITON_CACHE_DIR=$SCRATCH/triton-cache
export TORCHINDUCTOR_CACHE_DIR=$SCRATCH/inductor-cache
mkdir -p "$HF_HOME" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
export TORCH_CUDA_ARCH_LIST=9.0
export MAX_JOBS=8

##############################################################################
# STEP 1: the three fixed tests + the restored jax reference suite
##############################################################################
echo "############################################################"
echo "### STEP 1: fixed tests (this triage's actual verification target)"
echo "############################################################"
python -m pytest \
    code-tests/test_jax_reference.py \
    code-tests/test_newton_schulz.py::test_ns_equals_cholesky_bf16 \
    "code-tests/test_prefix_scan.py::test_prefix_scan_model_recurrent_decode_matches_full_prefix" \
    code-tests/test_table2_repro_contract.py::test_table2_forward_and_shifted_loss_smoke \
    -v 2>&1 | tee "$LOGDIR/gpu-triage-fixed-${SLURM_JOB_ID:-manual}.log"
FIXED_STATUS=${PIPESTATUS[0]}
echo "STATUS step1_fixed_tests=$FIXED_STATUS"

##############################################################################
# STEP 2: full suite, informational only -- also exercises the nvcc-fix-
# dependent tests (fused kernel + decode_prefill_parity + smoke_e2e) that are
# NOT this job's responsibility to fix, but which this worktree's branch
# ancestry (jack/nvcc-fix) should also clear.
##############################################################################
echo "############################################################"
echo "### STEP 2: full code-tests/ suite (informational)"
echo "############################################################"
python -m pytest code-tests/ -v -ra --durations=25 --continue-on-collection-errors \
    2>&1 | tee "$LOGDIR/gpu-triage-full-${SLURM_JOB_ID:-manual}.log"
FULL_STATUS=${PIPESTATUS[0]}
echo "STATUS step2_full_suite=$FULL_STATUS"

echo "############################################################"
echo "### FINAL SUMMARY"
echo "############################################################"
echo "step1_fixed_tests (this job's actual verdict) = $FIXED_STATUS"
echo "step2_full_suite (informational)              = $FULL_STATUS"
date -Is
echo "=== END ==="
