#!/bin/bash
#SBATCH --job-name=koopman-resume-golden
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
# The FOURTH instrument, and an upgrade to the first. Two jobs in one, because
# the second needs the first.
#
# ## Part 1: re-capture the shard golden DETERMINISTICALLY
#
# golden_4m_curve.json's "noise floor" of 2.0e-4 is not a precision limit -- it
# is two nondeterministic runs disagreeing. The synthetic goldens reach 0.000000
# because their capture scripts call a trainer CLI directly and could pass
# --deterministic; a spec-driven run had no way to ask until RuntimeSpec grew a
# `deterministic` field. configs/runs/4m-golden.yaml now sets it.
#
# So one of three instruments was ~2000x blunter than the others. That mattered
# concretely: the SKA-route comparison (chunked vs exact) produced deltas of
# 2.3e-4 against a 2.0e-4 floor, which is unresolvable. A sharp shard instrument
# is what makes the remaining loop-unification work checkable at all.
#
# ## Part 2: the resume golden
#
# code-tests/test_resume_equivalence.py proves that A loop resumes correctly --
# it uses the real resume helpers and the real dataset, but its OWN `_run_loop`,
# its own LR lambda and a toy model. It does not touch train.py's loop. Design
# doc §3 says resume is entangled with data iteration, and the loop unification
# is going to move exactly that. So: run, interrupt at a real checkpoint, resume
# to 400, and require the resumed second half to match the uninterrupted curve.
#
# Only meaningful because of Part 1. Against a 2e-4 floor a resume that
# re-shuffled its data could pass; against 0.000000 it cannot.
#
# NOTE this compares against replicate A captured IN THIS JOB, at this commit --
# not against the committed golden. A resume test asks "does stopping change
# anything", which is a question about one binary, so both halves must come from
# the same one.

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
export OUT=$SCRATCH/resume-golden-${SLURM_JOB_ID:-manual}
export SPEC=configs/runs/4m-golden.yaml

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
# use_deterministic_algorithms needs this for cuBLAS GEMMs, and torch raises a
# RuntimeError naming it if it is unset. Setting it here rather than relying on
# the trainer means a missing export fails loudly at step 0, not at step 1.
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
if [ "$SANITY" -ne 0 ]; then
    echo "FATAL: torch cannot see a GPU on $(hostname). Not capturing."
    exit 1
fi

echo "--- the spec asks for determinism, and the argv must carry it ---"
python -c "
import sys; sys.path.insert(0, '.')
from experimentation.run.resolve import resolve_run_spec
from experimentation.run.train_argv import build_train_argv
s = resolve_run_spec('$SPEC')
argv = build_train_argv(s, '/tmp/probe')
assert s.runtime.deterministic, 'spec does not request determinism'
assert '--deterministic' in argv, 'runtime.deterministic did not reach the trainer'
print('OK: --deterministic present in argv')
"
echo "STATUS argv_check=$?"

############################################################
### Part 1 -- two identical deterministic runs = the new floor
############################################################
for REP in A B; do
    echo "### replicate $REP (400 steps, deterministic, identical seed)"
    python -m experimentation.run "$SPEC" \
        --launcher local \
        --run_root "$OUT/$REP" \
        --force \
        2>&1 | tee "$OUT/$REP.log"
    echo "STATUS replicate_${REP}=${PIPESTATUS[0]}"
done

############################################################
### Part 2 -- interrupt at a real checkpoint, resume, finish at 400
############################################################
# A REAL interruption, not a shortened run. Capping max_steps for the first leg
# would be wrong twice over: max_steps lives in `optim`, which IS hashed into
# run_id, so a capped leg lands in a different run directory and the resume would
# never find its checkpoint -- and it would test "a short run followed by a long
# one" rather than "a run that was interrupted".
#
# So: start the full run, wait for a real checkpoint, kill it, resume.
# train_argv sets save_steps = max_steps // 3, which is 133 here, so checkpoints
# land at 133 / 266 / 399. We wait for one at or past MIN_CKPT and then DISCOVER
# which step it actually was rather than assuming -- the resume picks up from
# whichever checkpoint is latest, so the comparison cut has to be that same step
# or the "before"/"after" split is a fiction.
export MIN_CKPT=200

echo "### leg 1: full run, to be interrupted at the first checkpoint >= $MIN_CKPT"
python -m experimentation.run "$SPEC" \
    --launcher local \
    --run_root "$OUT/R" \
    --force \
    > "$OUT/R_first.log" 2>&1 &
LAUNCH_PID=$!

CKPT_STEP=""
for _ in $(seq 1 180); do
    # Highest step_<N> present, if any -- that is what a resume would load.
    CKPT_STEP=$(find "$OUT/R" -maxdepth 4 -type d -name 'step_*' 2>/dev/null \
                | sed 's:.*/step_::' | grep -E '^[0-9]+$' | sort -n | tail -1)
    if [ -n "$CKPT_STEP" ] && [ "$CKPT_STEP" -ge "$MIN_CKPT" ]; then break; fi
    # If the launcher died before checkpointing, stop waiting for it.
    kill -0 "$LAUNCH_PID" 2>/dev/null || { echo "launcher exited early"; break; }
    sleep 5
done
echo "latest checkpoint step: ${CKPT_STEP:-<none>}"
if [ -z "$CKPT_STEP" ] || [ "$CKPT_STEP" -lt "$MIN_CKPT" ]; then
    echo "FATAL: no checkpoint >= $MIN_CKPT appeared; cannot test resume."
    echo "STATUS resume_first=1"
    tail -20 "$OUT/R_first.log"
    exit 1
fi
# The comparison's cut is the checkpoint the resume will actually load from.
export RESUME_AFTER="$CKPT_STEP"

# Kill the trainer, not just the launcher: with wait=True the launcher is
# blocked in subprocess.run, so killing only the parent orphans the child and
# it would keep training into the run directory we are about to resume.
# Safe to be blunt here -- this job holds a whole node and runs nothing else.
pkill -f "experimentation.training.train" 2>/dev/null
sleep 5
kill "$LAUNCH_PID" 2>/dev/null
wait "$LAUNCH_PID" 2>/dev/null
echo "interrupted after step $RESUME_AFTER"
echo "STATUS resume_first=0"

echo "### leg 2: resume the SAME run directory to the spec's own 400 steps"
python -m experimentation.run "$SPEC" \
    --launcher local \
    --run_root "$OUT/R" \
    --resume \
    2>&1 | tee "$OUT/R_second.log"
echo "STATUS resume_second=${PIPESTATUS[0]}"

############################################################
### compare
############################################################
python - <<'PY'
import json
import os
import pathlib

from experimentation.sweep.search.metrics import parse_progress

out = pathlib.Path(os.environ["OUT"])


def curve(*names):
    """Merge the progress points from one or more logs, later logs winning."""
    merged = {}
    for n in names:
        path = out / n
        if not path.exists():
            print(f"  (missing {n})")
            continue
        for p in parse_progress(path.read_text()):
            merged[p.step] = p.loss
    return merged


a, b = curve("A.log"), curve("B.log")
print(f"replicate A: {len(a)} points   replicate B: {len(b)} points")
if not a or not b:
    print("A REPLICATE PRODUCED NO PARSEABLE POINTS -- cannot measure a floor")
    raise SystemExit(1)

shared = sorted(set(a) & set(b))
deltas = [(s, abs(a[s] - b[s])) for s in shared]
worst_step, worst = max(deltas, key=lambda t: t[1])
mean = sum(d for _, d in deltas) / len(deltas)
print("\n=== PART 1: the new deterministic floor for the SHARD path ===")
print(f"shared steps {len(shared)}   mean |A-B| {mean:.6f}   max |A-B| {worst:.6f} "
      f"(step {worst_step})")
print("  previous floor, captured WITHOUT determinism: 0.000200")
if worst == 0.0:
    print("  -> EXACT. The shard instrument is now as sharp as the synthetic two.")
else:
    print(f"  -> NOT exact. Determinism did not fully close it; floor is {worst:.6f}.")

# The resumed run: first log has 1..200, second has 201..400.
r = curve("R_first.log", "R_second.log")
print("\n=== PART 2: resume vs uninterrupted ===")
print(f"resumed run: {len(r)} points, steps {min(r, default=0)}..{max(r, default=0)}")
rshared = sorted(set(a) & set(r))
if not rshared:
    print("NO SHARED STEPS between the resumed run and replicate A")
    raise SystemExit(1)

# Steps after the interruption are the ones that can differ; report both so a
# failure says WHERE.
cut = int(os.environ["RESUME_AFTER"])   # the step the resume loaded from
post = [(s, abs(a[s] - r[s])) for s in rshared if s > cut]
pre = [(s, abs(a[s] - r[s])) for s in rshared if s <= cut]
for label, rows in ((f"steps 1-{cut} (before the stop)", pre),
                    (f"steps {cut+1}-400 (after resuming)", post)):
    if not rows:
        print(f"  {label}: no shared steps")
        continue
    ws, w = max(rows, key=lambda t: t[1])
    print(f"  {label}: {len(rows)} steps, max |delta| {w:.6f} (step {ws})")

worst_post = max((d for _, d in post), default=None)
if worst_post is None:
    print("  NO post-resume steps to compare -- did the resume run at all?")
    raise SystemExit(1)

payload = {
    "trainer": "experimentation.training.train",
    "spec": "configs/runs/4m-golden.yaml",
    "path": "shard-resume",
    "args": {"max_steps": 400, "interrupt_at": cut, "seed": 42,
             "deterministic": True, "log_every": 10},
    "commit": os.popen("git rev-parse HEAD").read().strip(),
    "capture_job": os.environ.get("SLURM_JOB_ID", "manual"),
    "replicates": {"A": {str(k): v for k, v in sorted(a.items())},
                   "B": {str(k): v for k, v in sorted(b.items())}},
    "resumed": {str(k): v for k, v in sorted(r.items())},
    "noise_floor": {"max_abs_delta": worst, "mean_abs_delta": mean,
                    "max_at_step": worst_step, "n_shared_steps": len(shared)},
    "resume_delta": {"max_abs_delta": worst_post,
                     "n_post_resume_steps": len(post)},
    "interpretation": (
        "The fourth instrument, and the first one to exercise resume through "
        "train.py's OWN loop. test_resume_equivalence.py uses the real resume "
        "helpers and the real dataset but its own loop body, its own LR lambda "
        "and a toy model, so it cannot see a resume bug that lives in train.py. "
        "Also re-captures the shard golden with runtime.deterministic set, which "
        "is what makes a resume comparison meaningful: against the previous "
        "2.0e-4 nondeterministic floor, a resume that re-shuffled its data could "
        "have passed."),
}
dest = out / "resume_golden_curve.json"
dest.write_text(json.dumps(payload, indent=2, sort_keys=True))
print(f"\nwrote {dest}")
print("\nVERDICT")
print(f"  deterministic shard floor : {worst:.6f}")
print(f"  resume delta (post-200)   : {worst_post:.6f}")
print("  resume is EXACT" if worst_post == 0.0 else
      f"  resume DIFFERS by {worst_post:.6f} -- investigate before unifying loops")
PY
echo "STATUS compare=$?"
echo "workdir: $OUT"
date -Is
echo "=== END ==="
