#!/bin/bash
#SBATCH --job-name=koopman-chunked-route
#SBATCH --account=marlowe-m000151-pm06
#SBATCH --partition=batch
#SBATCH --qos=medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=112G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/m000151-pm06/jkli/logs/koopman-tests/%x-%j.out
#SBATCH --error=/scratch/m000151-pm06/jkli/logs/koopman-tests/%x-%j.err
#
# The two things nobody has measured about the chunked SKA route.
#
# PHASE 1 (minutes) -- what the exact routes COST, per config geometry.
#   space.py argues "exactness is free" from 0.92x at 4m / 1.41x at 50m. Both are
#   rank 24. Every config still on the chunked route is rank 24, 48, 64, 96 or
#   128, and the exact path's per-token stats are (B,T,H,r,r) -- QUADRATIC in rank,
#   LINEAR in T, against the chunked path's (B,T/S,H,r,r). So the ratio cannot be
#   assumed to travel. This measures it at each config's own rank / heads / value
#   width / chunk size / sequence length, and records the refusals: inverse_
#   cholesky asserts rank <= 64, which 440m/880m (96) and 1p5b/3b (128) fail.
#
# PHASE 2 (hours) -- does the chunked route's SKA earn anything?
#   The PAIRED ablation delta, three cells x three seeds at the proxy geometry.
#   Not a between-cell loss comparison: SKA's entire contribution at 1500 steps
#   is 0.0251 and the trial-vs-trial floor is 0.0213 (sigma 7.54e-3, job 445994),
#   so a chunked-vs-exact loss contrast has a ceiling 1.18x its own noise and
#   REVIEW.md:341-350 already recorded that non-result at 2.3e-4. The within-run
#   delta is t = 8.4 at n = 5, which is why it can answer at n = 3.
#
#   Cells: exact-invchol (control), chunked-cs64 (the proxy's own chunk size),
#   chunked-cs16 (1m's -- the Echo Table 2 config). Each finished checkpoint is
#   then evaluated under BOTH routes, since the route flag changes no parameter
#   shape; that swap is the cheapest paired contrast in the file.
#
# 14 CPUs per GPU is the balanced request on a 112-CPU / 8-GPU node under
# 1-unit-per-CPU billing.
#
# Set PHASE=1 or PHASE=2 to run only one. Default: both.

set -uo pipefail

export REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
if [ ! -d "$REPO_ROOT/experimentation" ] || [ ! -d "$REPO_ROOT/code-tests" ]; then
    echo "FATAL: REPO_ROOT=$REPO_ROOT is not a Koopman_Mamba checkout."
    exit 1
fi
export SCRATCH=/scratch/m000151-pm06/jkli
export VENV=$SCRATCH/venvs/koopman-cuda
export OPTUNA_SITE=/users/jkli/.venvs/koopman-optuna/site
OUT_ROOT=$SCRATCH/chunked-route-${SLURM_JOB_ID:-manual}
PHASE="${PHASE:-both}"

echo "=== job ${SLURM_JOB_ID:-<interactive>} on $(hostname) ==="
date -Is
cd "$REPO_ROOT" || { echo "FATAL: no repo root"; exit 1; }
echo "commit: $(git rev-parse HEAD)"
echo "tree:   $(git status --porcelain | wc -l) modified path(s)"
echo "out:    $OUT_ROOT"
mkdir -p "$OUT_ROOT"

module purge
module load cuda12.9/toolkit/12.9.1
module load gcc/13.1.0
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

# shellcheck disable=SC1091
source "$VENV/bin/activate"
export PYTHONPATH=$REPO_ROOT:$OPTUNA_SITE
export HF_HOME=$SCRATCH/hf
export TRITON_CACHE_DIR=$SCRATCH/triton-cache
export TOKENIZERS_PARALLELISM=false

python -c "
import torch
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
assert torch.cuda.is_available(), 'no CUDA device visible to torch'
print('device', torch.cuda.get_device_name(0))
"
if [ $? -ne 0 ]; then
    echo "STATUS venv=FAIL -- aborting rather than measuring on a CPU"
    exit 1
fi

RC=0

if [ "$PHASE" = "1" ] || [ "$PHASE" = "both" ]; then
    echo
    echo "############ PHASE 1: route cost per config geometry ############"
    # Batch 1, each config's own max_seq_len. Batch 1 because the question is
    # whether the exact path fits AT ALL; if it OOMs at batch 1 no accumulation
    # schedule rescues it.
    python scripts/measure_chunked_route_cost.py \
        --batch 1 --json "$OUT_ROOT/route_cost.json"
    # Captured BEFORE the echo. `echo "...$?"` then `[ $? -ne 0 ]` tests the
    # echo's status, which is always 0, so the guard would never fire.
    P1=$?
    echo "STATUS phase1=$P1"
    [ "$P1" -ne 0 ] && RC=1
fi

if [ "$PHASE" = "2" ] || [ "$PHASE" = "both" ]; then
    echo
    echo "############ PHASE 2: paired ablation delta per route ############"
    # CELLS/SEEDS/MAX_STEPS are overridable so the three cells can go out as
    # three short concurrent jobs instead of one long serial one. A 3.5h request
    # sat behind (Priority) for an estimated 4 hours; a ~1.25h request backfills.
    # The driver writes its JSON after every trial, so a preemption or a timeout
    # still leaves every completed trial readable.
    python scripts/measure_chunked_route_ablation.py \
        --run_root "$OUT_ROOT/ablation" \
        --eval_data_dir "$SCRATCH/fineweb_small_val" \
        --max_steps "${MAX_STEPS:-1500}" --seeds "${SEEDS:-42,43,44}" \
        --cells "${CELLS:-exact-invchol,chunked-cs64,chunked-cs16}" \
        --json "$OUT_ROOT/route_ablation.json"
    P2=$?
    echo "STATUS phase2=$P2"
    [ $P2 -ne 0 ] && RC=1
fi

echo
echo "=== artefacts ==="
ls -la "$OUT_ROOT"/*.json 2>/dev/null || echo "(none)"
date -Is
echo "STATUS overall=$RC"
exit $RC
