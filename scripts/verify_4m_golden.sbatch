#!/bin/bash
#SBATCH --job-name=koopman-4m-golden-verify
#SBATCH --account=marlowe-m000151-pm06
#SBATCH --partition=batch
#SBATCH --qos=medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/m000151-pm06/jkli/logs/koopman-tests/%x-%j.out
#SBATCH --error=/scratch/m000151-pm06/jkli/logs/koopman-tests/%x-%j.err
#
# The gate `verify_goldens_after_refactor.sbatch` does not cover: the SHARD
# trainer, against code-tests/golden_4m_curve.json.
#
# ## Why this exists
#
# `verify_goldens_after_refactor.sbatch` re-runs table2 and mqar. Job 446736 ran
# it after the beta key/value exponent work and both reported MATCH at
# d 0.000000. But that work changed `SKAModule._weight_key_value`, which is on
# EVERY SKA path -- and `configs/runs/4m-golden.yaml` has ska_layer_indices
# (2, 3), so the shard trainer exercises it too and had no gate.
#
# The distinction matters because the two scripts cover different loops:
# mqar_finetune and table2 are synthetic-curriculum trainers, while
# `experimentation.run` -> `training.train` reads a packed memmap shard. A change
# that is bit-identical on one is not thereby bit-identical on the other -- the
# batch source, the accumulation split and the eval cadence all differ.
#
# ## Why it COMPARES rather than captures
#
# `capture_golden_curve.sbatch` runs two replicates A/B and reports their mutual
# delta -- a reproducibility check, not a check against the committed
# instrument. It also prints "copy it to code-tests/golden_4m_curve.json to make
# it the committed baseline", which is the operation that must never be
# automated: `identity_baseline.json`'s docstring states the rule for this class
# of artefact -- regenerate "ONLY when an intentional change to run identity has
# been decided ... never to make this go green".
#
# So this script runs ONE replicate and diffs it against the committed curve
# with `compare_golden_curve.py`, exactly as verify_goldens_after_refactor does
# for its two. It writes nothing into code-tests/.
#
# ## Reading the result
#
# The committed curve records noise_floor.max_abs_delta = 0.0 over 40 shared
# steps, so this comparison has no slack to hide in: the progress line prints
# loss at %.4f, `compare_golden_curve.py`'s MIN_TOLERANCE is that same 1e-4, and
# a floor of 0.000000 means the two runs agreed at the printed resolution. Any
# non-zero delta is a real behaviour change, not drift.
set -uo pipefail

# SLURM_SUBMIT_DIR, not BASH_SOURCE: sbatch copies the script to a spool dir, so
# ${BASH_SOURCE[0]} resolves under /cm/local/apps/slurm/var/spool/. Always
# sbatch from the repo root; the check below is what makes that failure loud.
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
if [ ! -d "$REPO_ROOT/experimentation" ] || [ ! -d "$REPO_ROOT/code-tests" ]; then
    echo "FATAL: $REPO_ROOT is not the repo root (no experimentation/ + code-tests/)."
    echo "       sbatch this from the repo root."
    exit 1
fi
export SCRATCH=/scratch/m000151-pm06/jkli
export VENV=$SCRATCH/venvs/koopman-cuda
export OUT=$SCRATCH/4m-golden-verify-${SLURM_JOB_ID:-manual}

echo "=== job ${SLURM_JOB_ID:-<interactive>} on $(hostname) ==="
date -Is
cd "$REPO_ROOT" || { echo "FATAL: no repo root"; exit 1; }
echo "commit: $(git rev-parse HEAD)"
DIRTY=$(git status --porcelain | wc -l)
echo "tree:   $DIRTY modified path(s)  (0 wanted)"
if [ "$DIRTY" -ne 0 ]; then
    # ABORT, not warn. `experimentation.run` refuses a dirty tree itself
    # (provenance.check_git_clean), so continuing only moves the failure 40
    # seconds later into the trainer and then hands the comparator an empty log
    # -- which is what job 447297 did, over one untracked scratch file. Failing
    # here costs nothing and names the cause.
    #
    # Not --allow-dirty: a golden comparison against a tree that does not match
    # the committed code is not a comparison. The point is to check THIS commit.
    git status --porcelain | sed 's/^/    /'
    echo "FATAL: refusing to compare a golden from a dirty tree. Commit or move"
    echo "       the paths above, then resubmit."
    exit 1
fi

module purge
module load cuda12.9/toolkit/12.9.1
module load gcc/13.1.0
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader || true

# shellcheck disable=SC1091
source "$VENV/bin/activate"
export PYTHONPATH=$REPO_ROOT
export HF_HOME=$SCRATCH/hf
export TRITON_CACHE_DIR=$SCRATCH/triton-cache
export TOKENIZERS_PARALLELISM=false
mkdir -p "$OUT"

python -c "
import torch
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
assert torch.cuda.is_available(), 'no CUDA device visible to torch'
print('device', torch.cuda.get_device_name(0))
import mamba_ssm; print('mamba_ssm OK')
"
SANITY=$?
echo "STATUS venv=$SANITY"
if [ "$SANITY" -ne 0 ]; then
    # Job 439811 landed on a node where nvidia-smi saw an H100 and
    # torch.cuda.is_available() was False, and ran anyway -- producing a failure
    # indistinguishable from a real bug. Abort instead.
    echo "FATAL: torch cannot see a GPU on $(hostname). Not comparing."
    exit 1
fi

# The spec is read OUT of the golden rather than hardcoded, so this script
# cannot drift from the curve it checks. Same discipline as
# verify_goldens_after_refactor's "args from golden_*_curve.json".
SPEC=$(python -c "
import json, pathlib
print(json.loads(pathlib.Path('code-tests/golden_4m_curve.json').read_text())['spec'])
")
echo "### shard trainer: spec from golden_4m_curve.json -> $SPEC"

python -m experimentation.run "$SPEC" \
    --launcher local --run_root "$OUT" \
    2>&1 | tee "$OUT/train.log"
echo "STATUS train=${PIPESTATUS[0]}"

echo "############################################################"
echo "### diff against the committed curve"
echo "############################################################"
python scripts/compare_golden_curve.py "$OUT/train.log" \
    --golden code-tests/golden_4m_curve.json
echo "STATUS compare=$?"

echo "### verdict"
echo "workdir: $OUT"
echo "MATCH above means the shard loop still trains identically. A non-zero"
echo "delta is a behaviour change: report it, do NOT re-capture the golden."
