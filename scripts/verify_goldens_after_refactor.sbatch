#!/bin/bash
#SBATCH --job-name=koopman-golden-verify
#SBATCH --account=marlowe-m000151-pm06
#SBATCH --partition=batch
#SBATCH --qos=medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/m000151-pm06/jkli/logs/koopman-tests/%x-%j.out
#SBATCH --error=/scratch/m000151-pm06/jkli/logs/koopman-tests/%x-%j.err
#
# The gate for the TrainTask migration: re-run every trainer that has a golden
# curve and diff against the committed instrument.
#
# This is the ONE thing that can tell a loop refactor from a loop change. All
# three curves still descend if the LR schedule shifted, if gradient
# accumulation regrouped, if batches are drawn in a different order, or if a
# curriculum's parity inverted. "Still trains" is not evidence; a bit-exact
# curve is.
#
# Each capture script re-derives its args from the golden it is checked against,
# so the args here are not free parameters -- they are copied from
# code-tests/golden_*_curve.json["args"] and must stay in step with it. The
# comparator prints the golden's own commit so a mismatch is attributable.
#
# --deterministic where the golden had it. Measured on mqar_finetune (job
# 439605): without it, two runs at the same seed diverged 0.53 in loss by step
# 400; with it, 0.000000. Note the SHARD golden was captured WITHOUT it (see
# capture_golden_curve.sbatch's own header) and so carries a 2e-4 noise floor
# rather than zero -- its comparison is correspondingly blunter, and that is a
# property of the instrument, not of this run.

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
export OUT=$SCRATCH/golden-verify-${SLURM_JOB_ID:-manual}

echo "=== job ${SLURM_JOB_ID:-<interactive>} on $(hostname) ==="
date -Is
cd "$REPO_ROOT" || { echo "FATAL: no repo root"; exit 1; }
echo "commit: $(git rev-parse HEAD)"
echo "tree:   $(git status --porcelain | wc -l) modified path(s)  (0 wanted)"

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
    # torch.cuda.is_available() was False, and ran a whole study anyway --
    # producing failures indistinguishable from a real bug.
    echo "FATAL: torch cannot see a GPU on $(hostname). Not comparing."
    exit 1
fi

############################################################
### table2 -- the loop this refactor actually changed
############################################################
echo "### table2: args from golden_table2_curve.json"
python -m experimentation.experiments.table2 \
    --model_size 1m \
    --max_steps 400 \
    --log_every 10 \
    --eval_every 200 \
    --curriculum batch_mixed \
    --seed 42 \
    --deterministic \
    --output_dir "$OUT/table2/run" \
    2>&1 | tee "$OUT/table2.log"
echo "STATUS table2_train=${PIPESTATUS[0]}"

python scripts/compare_golden_curve.py "$OUT/table2.log" \
    --golden code-tests/golden_table2_curve.json
echo "STATUS table2_compare=$?"

############################################################
### mqar -- already migrated; re-checked because synthetic_loss moved
############################################################
# Not redundant. This refactor extracted SyntheticTask.step_loss's body into
# task.py::synthetic_loss, so mqar_finetune's loss now reaches it through one
# more function call. That should be exactly identity -- which is precisely the
# kind of "obviously fine" change worth spending 90 seconds proving.
echo "### mqar: args from golden_mqar_curve.json"
python - <<'PY' > "$OUT/mqar_args.txt"
import json, pathlib
a = json.loads(pathlib.Path("code-tests/golden_mqar_curve.json").read_text())["args"]
print(" ".join(f"--{k} {v}" for k, v in sorted(a.items())
                if not isinstance(v, bool)))
PY
echo "mqar args: $(cat "$OUT/mqar_args.txt")"
# shellcheck disable=SC2046
python -m experimentation.experiments.mqar_finetune \
    $(cat "$OUT/mqar_args.txt") --deterministic \
    --output_dir "$OUT/mqar/run" \
    2>&1 | tee "$OUT/mqar.log"
echo "STATUS mqar_train=${PIPESTATUS[0]}"

python scripts/compare_golden_curve.py "$OUT/mqar.log" \
    --golden code-tests/golden_mqar_curve.json
echo "STATUS mqar_compare=$?"

echo "############################################################"
echo "### verdict"
echo "############################################################"
grep -E "^STATUS" "$SCRATCH/logs/koopman-tests/koopman-golden-verify-${SLURM_JOB_ID}.out" 2>/dev/null || true
echo "workdir: $OUT"
date -Is
echo "=== END ==="
