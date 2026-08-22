#!/bin/bash
#SBATCH --job-name=koopman-table2-golden
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
# The THIRD golden: table2's loop, joining the shard (4m) and synthetic (MQAR)
# baselines. All three exist so the §6.2 unification is gated on every path it
# touches rather than the two that were convenient.
#
# Why table2 needs its own rather than being covered by MQAR's. Three things are
# unique to this loop, and each is a way the unification could break silently:
#
#   * It has NO DATASET. make_train_batch(step, args) is keyed on the STEP
#     COUNTER and returns a whole batch, so wrapping it as a Dataset needs
#     batch_size=None -- and wrapping it naively yields [B, B, T] while silently
#     consuming B steps of curriculum per iteration. A curve is the only thing
#     that catches the second half of that.
#   * It is the ONLY trainer running fp16 with a GradScaler. Nothing else
#     exercises the loop's scaler path.
#   * Its curriculum is step-parity dependent (--curriculum mixed alternates on
#     step % 2), so a change to how steps are counted shows up as a changed
#     curve rather than an error.
#
# --deterministic is mandatory, not optional. Measured on mqar_finetune (job
# 439605): without it two runs at the same seed diverged 0.53 in loss by step 400;
# with it, 0.000000. A golden captured without it measures the GPU, not the loop.
#
# Sized to match the other two goldens so all three read the same way: 400 steps,
# log_every 10 for 40 points, at table2's own 1m scale.

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
export OUT=$SCRATCH/table2-golden-${SLURM_JOB_ID:-manual}

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
mkdir -p "$OUT/A" "$OUT/B"

python -c "
import torch
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
assert torch.cuda.is_available(), 'no CUDA device visible to torch'
print('device', torch.cuda.get_device_name(0))
"
SANITY=$?
echo "STATUS venv=$SANITY"
if [ "$SANITY" -ne 0 ]; then
    # Abort rather than continue. Job 439811 landed on a node where nvidia-smi
    # saw an H100 and torch.cuda.is_available() was False, and that script ran a
    # whole study anyway -- producing four failures indistinguishable from a real
    # bug. A capture job with no GPU produces a golden of nothing.
    echo "FATAL: torch cannot see a GPU on $(hostname). Not capturing."
    exit 1
fi

for REP in A B; do
    echo "############################################################"
    echo "### replicate $REP  (identical args, identical seed)"
    echo "############################################################"
    python -m experimentation.experiments.table2 \
        --model_size 1m \
        --max_steps 400 \
        --log_every 10 \
        --eval_every 200 \
        --curriculum batch_mixed \
        --seed 42 \
        --deterministic \
        --output_dir "$OUT/$REP/run" \
        2>&1 | tee "$OUT/$REP/train.log"
    echo "STATUS replicate_${REP}=${PIPESTATUS[0]}"
done

echo "############################################################"
echo "### divergence between two identical runs = the noise floor"
echo "############################################################"
python - <<'PY'
import json
import os
import pathlib

from experimentation.sweep.search.metrics import parse_progress

out = pathlib.Path(os.environ["OUT"])
curves = {}
for rep in ("A", "B"):
    pts = parse_progress((out / rep / "train.log").read_text())
    curves[rep] = {p.step: p.loss for p in pts}
    print(f"replicate {rep}: {len(pts)} logged points, "
          f"steps {min(curves[rep], default=0)}..{max(curves[rep], default=0)}")

if not curves["A"]:
    print("PARSED NOTHING -- table2's canonical log prefix has regressed")
    raise SystemExit(1)

shared = sorted(set(curves["A"]) & set(curves["B"]))
if not shared:
    print("NO SHARED STEPS -- a replicate died")
    raise SystemExit(1)

deltas = [(s, abs(curves["A"][s] - curves["B"][s])) for s in shared]
worst_step, worst = max(deltas, key=lambda t: t[1])
mean = sum(d for _, d in deltas) / len(deltas)
print(f"\nshared steps: {len(shared)}")
print(f"mean |A-B|  : {mean:.6f}")
print(f"max  |A-B|  : {worst:.6f}  (at step {worst_step})")
for s, d in deltas:
    print(f"  step {s:>5}  A {curves['A'][s]:.6f}  B {curves['B'][s]:.6f}  d {d:.6f}")

payload = {
    "trainer": "experimentation.experiments.table2",
    "path": "synthetic-curriculum",
    "args": {"model_size": "1m", "max_steps": 400, "curriculum": "batch_mixed",
             "seed": 42, "deterministic": True},
    "commit": os.popen("git rev-parse HEAD").read().strip(),
    "replicates": curves,
    "noise_floor": {"mean_abs_delta": mean, "max_abs_delta": worst,
                    "max_at_step": worst_step, "n_shared_steps": len(shared)},
    "interpretation": (
        "The third golden, for the only loop that has no dataset (its batches "
        "are keyed on the step counter), the only one running fp16 with a "
        "GradScaler, and the only one whose curriculum depends on step parity. "
        "Each is a way the TrainTask unification could change training while "
        "still producing a descending loss."),
}
dest = out / "table2_golden_curve.json"
dest.write_text(json.dumps(payload, indent=2, sort_keys=True))
print(f"\nwrote {dest}")
PY
echo "STATUS compare=$?"
echo "workdir: $OUT"
date -Is
echo "=== END ==="
