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
export OPTUNA_SITE=/users/jkli/.venvs/koopman-optuna/site
export RUN_ROOT=$SCRATCH/study-smoke-${SLURM_JOB_ID:-manual}
# Which study to run. Defaults to the wiring smoke test; set STUDY_CONFIG to
# configs/search/smoke-prune.yaml to ask instead whether pruning fires.
export STUDY_CONFIG="${STUDY_CONFIG:-configs/search/smoke-4m.yaml}"
export STUDY_NAME="$(basename "$STUDY_CONFIG" .yaml)"
echo "study config: $STUDY_CONFIG  (study_name $STUDY_NAME)"

# ---------------------------------------------------------------------------
# A PER-JOB STUDY SPEC, because the run root can no longer come from a flag.
# ---------------------------------------------------------------------------
#
# This gate used to pass `--run_root "$RUN_ROOT"`. That flag is gone, and it is
# not coming back: a run_root that changes without changing `study_id` sends two
# workers to different journals while each believes it is collaborating, which is
# exactly the drift a committed spec exists to prevent.
#
# But the per-JOB root it was doing was load-bearing, and losing it would break
# this gate in a subtler way than an argparse error. `smoke-4m.yaml` has a FIXED
# run_root, and a study is RESUMABLE: `n_trials` is the target size, not "this
# many more". So a second submission against a fixed root would find its 4 trials
# already COMPLETE, run ZERO trials, and report PASS -- a gate that passes for the
# wrong reason, which is worse than one that fails.
#
# So the run root moves INTO a generated spec. The invocation stays spec-only, the
# generated file is a recorded artefact of exactly what ran, and because
# `study_id` is a content hash of the spec, a different run_root yields a
# different study_id and therefore a different journal -- which is precisely the
# isolation this gate wants.

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

# Generated HERE, after the venv is active -- not next to the comment above
# that explains it. The first version ran this block at line 88, before
# `source $VENV/bin/activate`, so `python` was not on PATH and job 445656
# died in 2 seconds with "python: command not found" before spending a
# second of GPU time. Cheap, but it is the third gate in this file that the
# CPU suite cannot see: generation was verified by extracting the block and
# running it with a python already on PATH, which is exactly the condition
# that does not hold under sbatch.
export GENERATED_STUDY="$RUN_ROOT/study.generated.yaml"
mkdir -p "$RUN_ROOT" || { echo "FATAL: cannot create $RUN_ROOT"; exit 1; }
python - <<'PY' || { echo "FATAL: could not generate the per-job study spec"; exit 1; }
import os
import pathlib
import sys

import yaml

sys.path.insert(0, os.environ["REPO_ROOT"])

source = pathlib.Path(os.environ["STUDY_CONFIG"])
body = yaml.safe_load(source.read_text()) or {}
body["run_root"] = os.environ["RUN_ROOT"]
out = pathlib.Path(os.environ["GENERATED_STUDY"])
out.write_text(
    "# GENERATED by scripts/verify_study_e2e.sbatch -- do not hand-edit.\n"
    f"# Source: {source}\n"
    "# Only `run_root` differs, so this job is isolated from every other\n"
    "# submission: study_id is a content hash of this file, so a different\n"
    "# run_root means a different journal. Committed configs are never mutated.\n"
    + yaml.safe_dump(body, sort_keys=False))

# Validate it HERE, not on the first trial: an unparseable generated spec should
# fail before the venv check and the dry run, with the file to look at named.
from experimentation.sweep.search.studyspec import load_study_spec, study_id

spec = load_study_spec(out)
print(f"generated {out}")
print(f"  run_root  {spec.run_root}")
print(f"  study_id  {study_id(spec)}  (differs per job, so the journal does too)")
PY

##############################################################################
echo "############################################################"
echo "### STEP 1: dry run -- the plan, with nothing spent"
echo "############################################################"
python -m experimentation.sweep.search "$GENERATED_STUDY" --dry_run
echo "STATUS dry_run=$?"

##############################################################################
echo "############################################################"
echo "### STEP 2: the real study"
echo "############################################################"
python -m experimentation.sweep.search "$GENERATED_STUDY" \
    2>&1 | tee "$SCRATCH/study-e2e-${SLURM_JOB_ID:-manual}.log"
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
