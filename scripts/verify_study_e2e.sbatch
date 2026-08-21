#!/bin/bash
#SBATCH --job-name=koopman-study-e2e
#SBATCH --account=marlowe-m000151-pm06
#SBATCH --partition=batch
#SBATCH --qos=medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/m000151-pm06/jkli/logs/koopman-tests/%x-%j.out
#SBATCH --error=/scratch/m000151-pm06/jkli/logs/koopman-tests/%x-%j.err
#
# The first study that can close its own loop.
#
# Every piece of the adaptive search has been unit-tested and none of it has ever
# driven real training and received a real score. Until the previous commit it
# COULD not: nothing in the launch path wrote quick_eval.json, so a study trained
# N models and recorded N FAILs. This is the run that proves otherwise.
#
# Runs the study with LocalLauncher inside this one allocation -- the
# held-allocation mode, so trials run back-to-back with no queue wait each. That
# is also the mode whose pruning was silently dead until LocalLauncher learned not
# to block, which makes it the mode worth exercising rather than the easy one.
#
# What counts as success, in order of how much it would hurt to have wrong:
#   1. at least one trial COMPLETE with a real objective (not FAIL)
#   2. quick_eval.json present in that trial's run dir, with a finite loss
#   3. the journal holds intermediate_values -- i.e. progress was parsed and
#      reported, which is the prerequisite for pruning existing at all
#   4. trials.csv and top_trials.md written
#   5. each trial's spec.yaml stamped with study_name / trial_number
#
# A FAIL on every trial is the specific outcome this job exists to rule out.

set -uo pipefail

export REPO_ROOT=/users/jkli/Koopman_Mamba/.claude/worktrees/jack+search-and-provenance
export SCRATCH=/scratch/m000151-pm06/jkli
export VENV=$SCRATCH/venvs/koopman-cuda
export OPTUNA_SITE=/users/jkli/.venvs/koopman-optuna/site
export RUN_ROOT=$SCRATCH/study-smoke-${SLURM_JOB_ID:-manual}
# Which study to run. Defaults to the wiring smoke test; set STUDY_CONFIG to
# configs/search/smoke-prune.yaml to ask instead whether pruning fires.
export STUDY_CONFIG="${STUDY_CONFIG:-configs/search/smoke-4m.yaml}"
export STUDY_NAME="$(basename "$STUDY_CONFIG" .yaml)"
echo "study config: $STUDY_CONFIG  (study_name $STUDY_NAME)"

echo "=== job ${SLURM_JOB_ID:-<interactive>} on $(hostname) ==="
date -Is
cd "$REPO_ROOT" || { echo "FATAL: no repo root"; exit 1; }
echo "commit: $(git rev-parse HEAD)"
echo "tree:   $(git status --porcelain | wc -l) modified path(s)"

module purge
module load cuda12.9/toolkit/12.9.1
module load gcc/13.1.0
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader || true

# shellcheck disable=SC1091
source "$VENV/bin/activate"
export PYTHONPATH=$REPO_ROOT:$OPTUNA_SITE
export HF_HOME=$SCRATCH/hf
export TRITON_CACHE_DIR=$SCRATCH/triton-cache
export TOKENIZERS_PARALLELISM=false

python -c "
import torch, optuna
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
print('optuna', optuna.__version__)
if torch.cuda.is_available():
    print('device', torch.cuda.get_device_name(0))
assert torch.cuda.is_available(), 'no CUDA device visible to torch'
"
SANITY=$?
echo "STATUS venv=$SANITY"
if [ "$SANITY" -ne 0 ]; then
    # ABORT, do not continue. Job 439811 landed on a node where nvidia-smi saw an
    # H100 but torch.cuda.is_available() was False; this check caught it and the
    # script ran the whole study anyway, which then reported "4 trial(s): 4
    # failed" -- a result indistinguishable from a real wiring failure. A
    # verification job that keeps going after its own precondition fails produces
    # evidence about nothing.
    echo "FATAL: torch cannot see a GPU on $(hostname). Not running the study --"
    echo "       a study with no GPU fails every trial and tells you nothing."
    exit 1
fi

##############################################################################
echo "############################################################"
echo "### STEP 1: dry run -- the plan, with nothing spent"
echo "############################################################"
python -m experimentation.sweep.search "$STUDY_CONFIG" \
    --dry_run --run_root "$RUN_ROOT"
echo "STATUS dry_run=$?"

##############################################################################
echo "############################################################"
echo "### STEP 2: the real study"
echo "############################################################"
python -m experimentation.sweep.search "$STUDY_CONFIG" \
    --run_root "$RUN_ROOT" 2>&1 | tee "$SCRATCH/study-e2e-${SLURM_JOB_ID:-manual}.log"
echo "STATUS study=${PIPESTATUS[0]}"

##############################################################################
echo "############################################################"
echo "### STEP 3: did the loop actually close?"
echo "############################################################"
python - <<'PY'
import json
import os
import pathlib

import optuna

root = pathlib.Path(os.environ["RUN_ROOT"])
studies = sorted((root / "_studies").glob("*"))
print("study dirs:", [p.name for p in studies])
if not studies:
    print("NO STUDY DIRECTORY -- the driver never got started")
    raise SystemExit(1)
sd = studies[-1]

# study.py:160 names it optuna_journal.log. An earlier version of this script
# said journal.log, so JournalFileBackend happily created an EMPTY file and
# load_study raised KeyError: Record does not exist -- a verification step that
# crashed instead of reporting.
journal = sd / "optuna_journal.log"
if not journal.exists():
    print(f"NO JOURNAL at {journal}; contents: {[p.name for p in sd.iterdir()]}")
    raise SystemExit(1)
storage = optuna.storages.JournalStorage(
    optuna.storages.journal.JournalFileBackend(str(journal)))
study = optuna.load_study(study_name=os.environ["STUDY_NAME"], storage=storage)

print(f"\ntrials: {len(study.trials)}")
by_state = {}
for t in study.trials:
    by_state[t.state.name] = by_state.get(t.state.name, 0) + 1
print("by state:", by_state)

for t in study.trials:
    anchor = t.user_attrs.get("anchor_name", "-")
    n_inter = len(t.intermediate_values)
    print(f"  trial {t.number}: {t.state.name:9} value={t.value} "
          f"anchor={anchor:22} reported_steps={n_inter}")

# 1. at least one real objective
completed = [t for t in study.trials
             if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
print(f"\n[1] COMPLETE with a real objective: {len(completed)}")

# 2. the objective file exists and is finite
found = sorted(root.glob("**/eval/*/quick_eval.json"))
print(f"[2] quick_eval.json files: {len(found)}")
for f in found[:4]:
    payload = json.loads(f.read_text())
    full = payload.get("metrics", {}).get("full", {})
    print(f"      {f.relative_to(root)}  loss={full.get('loss')} ppl={full.get('ppl')}")

# 3. progress was parsed and reported -- pruning's prerequisite
reported = [t for t in study.trials if t.intermediate_values]
print(f"[3] trials with reported intermediate steps: {len(reported)}")

# 4. reports written
for name in ("trials.csv", "top_trials.md", "best_trial.json"):
    p = sd / name
    print(f"[4] {name:16} {'yes' if p.exists() else 'MISSING'}")

# 5. provenance stamped into each trial's spec.yaml
import yaml
specs = sorted(root.glob("*/*/spec.yaml"))
print(f"[5] materialized spec.yaml files: {len(specs)}")
for s in specs[:3]:
    raw = yaml.safe_load(s.read_text())
    print(f"      {s.parent.name}: study={raw.get('study_name')} "
          f"trial={raw.get('trial_number')} anchor={raw.get('anchor_name')}")

pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
print(f"[6] PRUNED trials: {len(pruned)}")
for t in pruned:
    print(f"      trial {t.number}: pruned at step "
          f"{max(t.intermediate_values, default='?')} -- "
          f"{t.user_attrs.get('pruned', '')}")
if not pruned:
    print("      none. Legitimate if no sampled config fell below the median at a")
    print("      reported step, or if every trial was anchor-exempt.")

ok = bool(completed) and bool(found) and bool(reported)
print(f"\nVERDICT: {'PASS' if ok else 'FAIL'}")
if not ok:
    print("  the loop did not close. FAIL on every trial is exactly what this")
    print("  job exists to rule out -- check STEP 2's log for the reason.")
raise SystemExit(0 if ok else 1)
PY
echo "STATUS verify=$?"
echo "run_root: $RUN_ROOT"
date -Is
echo "=== END ==="
