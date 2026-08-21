#!/bin/bash
#SBATCH --job-name=koopman-synth-resume
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
# Does table2's resume reproduce its uninterrupted run? Measured, not assumed.
#
# WHY THIS IS ASKED. train.py's resume was just verified bit-exact (job 440216)
# because `save_resume_state` persists optimizer + scheduler + epoch +
# samples_consumed + RNG, and `apply_resume_state` restores all of it.
# mqar_finetune.py and table2.py do NOT use that mechanism. They have their own
# `load_checkpoint`, which restores model + optimizer + scheduler + step and
# contains **no RNG handling whatsoever** -- `grep -c rng` is 0 in both files.
#
# If that matters, a resumed run's dropout masks diverge from the uninterrupted
# run's from the first resumed step, and every later step differs. Nothing in the
# repo tests it, so this measures it.
#
# It also decides how the loop unification should be described. If resume here is
# already exact, collapsing three loops into one is deduplication. If it is not,
# the unification FIXES A CORRECTNESS BUG in two trainers -- and the standard in
# this branch is to measure that rather than claim it.
#
# --deterministic throughout, so any divergence found is the resume and not the
# GPU: measured on mqar_finetune (job 439605), two runs at the same seed diverge
# 0.53 by step 400 without it and 0.000000 with it.

set -uo pipefail

export REPO_ROOT=/users/jkli/Koopman_Mamba/.claude/worktrees/jack+search-and-provenance
export SCRATCH=/scratch/m000151-pm06/jkli
export VENV=$SCRATCH/venvs/koopman-cuda
export OUT=$SCRATCH/synth-resume-${SLURM_JOB_ID:-manual}
# table2 checkpoints at `step % eval_every == 0`, so this is also the interrupt
# point. Matches golden_table2_curve.json's own args.
export EVAL_EVERY=200
export CUT=200

echo "=== job ${SLURM_JOB_ID:-<interactive>} on $(hostname) ==="
date -Is
cd "$REPO_ROOT" || { echo "FATAL: no repo root"; exit 1; }
echo "commit: $(git rev-parse HEAD)"
echo "tree:   $(git status --porcelain | wc -l) modified path(s)  (0 wanted)"

module purge
module load cuda12.9/toolkit/12.9.1
module load gcc/13.1.0
nvidia-smi --query-gpu=name --format=csv,noheader || true

# shellcheck disable=SC1091
source "$VENV/bin/activate"
export PYTHONPATH=$REPO_ROOT
export HF_HOME=$SCRATCH/hf
export TRITON_CACHE_DIR=$SCRATCH/triton-cache
export TOKENIZERS_PARALLELISM=false
export CUBLAS_WORKSPACE_CONFIG=:4096:8
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
[ "$SANITY" -ne 0 ] && { echo "FATAL: no GPU. Not measuring."; exit 1; }

TABLE2_ARGS="--model_size 1m --max_steps 400 --log_every 10 \
--eval_every $EVAL_EVERY --curriculum batch_mixed --seed 42 --deterministic"

############################################################
### leg 1 -- uninterrupted, the reference
############################################################
echo "### uninterrupted 400 steps"
# shellcheck disable=SC2086
python -m experimentation.experiments.table2 $TABLE2_ARGS \
    --output_dir "$OUT/U/run" 2>&1 | tee "$OUT/U.log"
echo "STATUS uninterrupted=${PIPESTATUS[0]}"

############################################################
### leg 2 -- run, kill after the step-CUT checkpoint, resume from it
############################################################
echo "### interrupted run, to be killed after step_$CUT"
# shellcheck disable=SC2086
python -m experimentation.experiments.table2 $TABLE2_ARGS \
    --output_dir "$OUT/R/run" > "$OUT/R_first.log" 2>&1 &
PID=$!
for _ in $(seq 1 180); do
    [ -f "$OUT/R/run/step_$CUT/model.pt" ] && break
    kill -0 "$PID" 2>/dev/null || { echo "trainer exited early"; break; }
    sleep 5
done
if [ ! -f "$OUT/R/run/step_$CUT/model.pt" ]; then
    echo "FATAL: no step_$CUT checkpoint; cannot measure resume."
    tail -20 "$OUT/R_first.log"
    echo "STATUS interrupted=1"
    exit 1
fi
pkill -f "experimentation.experiments.table2" 2>/dev/null
sleep 5
wait "$PID" 2>/dev/null
echo "killed after step_$CUT"
echo "STATUS interrupted=0"

echo "### resume from step_$CUT"
# shellcheck disable=SC2086
python -m experimentation.experiments.table2 $TABLE2_ARGS \
    --output_dir "$OUT/R/run" --resume_from "$OUT/R/run/step_$CUT" \
    2>&1 | tee "$OUT/R_second.log"
echo "STATUS resumed=${PIPESTATUS[0]}"

############################################################
### compare
############################################################
python - <<'PY'
import json
import os
import pathlib

from experimentation.sweep.search.metrics import parse_progress

out = pathlib.Path(os.environ["OUT"])
cut = int(os.environ["CUT"])


def curve(*names):
    merged = {}
    for n in names:
        p = out / n
        if p.exists():
            for pt in parse_progress(p.read_text()):
                merged[pt.step] = pt.loss
    return merged


u = curve("U.log")
r = curve("R_first.log", "R_second.log")
print(f"uninterrupted: {len(u)} points   resumed: {len(r)} points")
if not u or not r:
    print("a leg produced no parseable points")
    raise SystemExit(1)

shared = sorted(set(u) & set(r))
post = [(s, abs(u[s] - r[s])) for s in shared if s > cut]
pre = [(s, abs(u[s] - r[s])) for s in shared if s <= cut]
if not post:
    print("no post-resume steps to compare")
    raise SystemExit(1)

for label, rows in ((f"steps <= {cut} (before the stop)", pre),
                    (f"steps > {cut} (after resuming)", post)):
    if not rows:
        continue
    ws, w = max(rows, key=lambda t: t[1])
    print(f"  {label}: {len(rows)} steps, max |delta| {w:.6f} (step {ws})")

print("\nfirst 6 post-resume steps:")
for s, d in post[:6]:
    print(f"  step {s:>4}  uninterrupted {u[s]:.4f}  resumed {r[s]:.4f}  d {d:.6f}")

worst = max(d for _, d in post)
print("\nVERDICT")
# 1e-4 is the log's own resolution: it prints `loss %.4f`.
if worst <= 1e-4:
    print(f"  table2 resume is EXACT ({worst:.6f}) -- the missing RNG restore "
          f"does not matter here, so unification is deduplication, not a fix.")
else:
    print(f"  table2 resume DIVERGES by {worst:.6f}, which is "
          f"{worst / 1e-4:.0f}x the log's own resolution.")
    print("  Its load_checkpoint restores model+optimizer+scheduler+step and no")
    print("  RNG (grep -c rng == 0), so a resumed run's dropout stream differs")
    print("  from the uninterrupted one. train.py's save_resume_state persists")
    print("  RNG and was verified bit-exact in job 440216 -- so the loop")
    print("  unification FIXES this rather than merely deduplicating it.")

(out / "synthetic_resume.json").write_text(json.dumps({
    "trainer": "experimentation.experiments.table2",
    "cut": cut,
    "commit": os.popen("git rev-parse HEAD").read().strip(),
    "uninterrupted": {str(k): v for k, v in sorted(u.items())},
    "resumed": {str(k): v for k, v in sorted(r.items())},
    "max_abs_delta_post_resume": worst,
    "n_post_resume_steps": len(post),
}, indent=2, sort_keys=True))
print(f"\nwrote {out / 'synthetic_resume.json'}")
PY
echo "STATUS compare=$?"
echo "workdir: $OUT"
date -Is
echo "=== END ==="
