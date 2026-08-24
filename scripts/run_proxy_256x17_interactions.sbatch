#!/bin/bash
#SBATCH --job-name=koopman-proxy256x17-interactions
#SBATCH --account=marlowe-m000151-pm06
#SBATCH --partition=batch
#SBATCH --qos=medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/m000151-pm06/jkli/logs/koopman-tests/%x-%j.out
#SBATCH --error=/scratch/m000151-pm06/jkli/logs/koopman-tests/%x-%j.err
#
# `koopman-tests`, which EXISTS, and not a tidier `koopman-studies`, which did
# not. Slurm does not create the directory for --output/--error: slurmstepd fails
# to open stdout and the batch step dies BEFORE LINE 1, so all three gates below
# are bypassed and there is no log anywhere saying why. Every other committed
# sbatch script in this repo writes here. If you want a separate directory,
# `mkdir -p` it before the first submission -- the `mkdir -p` further down runs
# far too late to help.
#
# The eight-GPU SKA interaction study: 256 trials x 600 steps on the 256x17
# proxy, eight concurrent, one node, one journal.
#
# ############################################################################
# THIS SCRIPT HAS NOT BEEN RUN. It is committed for review, not launched.
# ############################################################################
#
# ## ONE invocation, not eight
#
# The obvious shape for this file is a loop that starts eight
# `python -m experimentation.sweep.search` processes with `CUDA_VISIBLE_DEVICES`
# set per worker. Do NOT write that: it is a second, untested implementation of
# something the CLI already does. `concurrent_trials: 8` in the study file makes
# the command spawn eight children itself, pin each to a visible GPU round-robin,
# derive the sampler's `constant_liar` and each child's sampler seed from the same
# number, wait for all of them, and write one report --
# `experimentation/sweep/search/__main__.py::_fanout`, covered by
# `code-tests/test_search_fanout.py`.
#
# A hand-rolled loop here would duplicate the pinning (tested there, not here),
# would have to re-derive the per-worker sampler seeds (and would get them
# wrong, because the CLI derives them from an env marker this script does not
# set), and would leave the supervisor's single report unwritten. So: validate,
# dry-run, then call the CLI ONCE.
#
# ## Resource sizing, and why each number
#
# `--gpus-per-node=8` and `--nodes=1`. One node is not a convenience: the
# journal is a FILE (`JournalStorage` over `JournalFileBackend`), and file
# locking across nodes on a shared filesystem is exactly the thing not to bet a
# 12-hour study on. All eight workers share one host and one page cache.
#
# `--cpus-per-task=64` is 8 per trainer. Each trial's spec asks for
# `runtime.workers: 4` dataloader workers, plus its main process, plus headroom
# for the tokenizer and the eval pass. At 32 total the dataloaders would contend
# and the study would be CPU-bound on a node with eight idle-ish H100s.
#
# `--mem=512G` is 64 per trainer. A 25M-parameter model is small; the memory goes
# to eight independent shard readers over a 100M-token memmap and eight copies of
# the tokenizer.
#
# `--time=12:00:00`. Sizing is honest about being an ESTIMATE, because nothing
# has measured this geometry. Reference points: the 4m geometry measured ~25 s
# fixed per-trial cost plus ~0.12 s/step (job 439883), and this model is ~5x its
# parameters at ~2x the sequence length, so a 600-step trial is plausibly 3-8
# minutes. 256 trials / 8 workers = 32 waves x (3-8 min) = 1.6-4.3 hours, less
# once pruning starts at step 450. 12 hours is ~3x the pessimistic estimate. If
# it runs out, the study RESUMES -- see the run root below -- so an underestimate
# costs a resubmission and not a study.
#
# ## The run root is stable, deliberately
#
# It comes from the study FILE, not from `$SLURM_JOB_ID`. `run_root` is
# `/scratch/m000151-pm06/jkli/study-proxy-256x17-v1`, and the journal path is
# derived as `<run_root>/_studies/<name>.<study_id>/optuna_journal.log` where
# `study_id` is a content hash of the study spec. So every worker in every
# submission computes the same path with no coordination, and a resubmission
# ATTACHES to the study already in progress: `create_study(load_if_exists=True)`
# continues it, `n_trials` is the study's TARGET size rather than "this many
# more", and `enqueue_trial(skip_if_exists=True)` does not re-run the 24 anchors.
#
# A job-id-derived root would silently start a fresh study on every requeue --
# and `#SBATCH --requeue` is in the run system's own template, so that is not
# hypothetical. This is why the root is not in this file at all: putting it here
# would let a flag and the spec disagree, which is the drift that got
# `--run_root` removed from the CLI.
#
# ## Throughput is NOT in the objective, and cannot be
#
# `objective: {}` in the study file: pure held-out loss. That is not only a
# scientific preference, it is forced by this script. Eight trainers on one node
# contend for host memory bandwidth, PCIe and the shard's page cache, so a
# trial's measured tokens/sec depends on what its seven neighbours happened to be
# doing. `metrics.objective_from_metrics` has a `throughput_penalty` and it must
# stay zero here: under concurrency it would score the scheduler.
#
# ## What this script refuses to do
#
# Spend eight GPU-hours to discover a typo. Three gates, in increasing cost:
#   1. torch sees CUDA and optuna imports -- job 439811 landed on a node where
#      nvidia-smi saw an H100 and `torch.cuda.is_available()` was False;
#   2. eight devices are actually visible, since the whole point of this file is
#      the fleet and one visible GPU would run 8 workers on 1 device;
#   3. `--dry_run`, which validates the study spec, resolves the base config,
#      restricts and validates every search axis, loads and resolves all 24
#      anchors, and prints the plan -- all without materializing anything. A bad
#      axis, a missing shard path or a stale backend policy dies here.
# Each aborts. A job that continues past its own failed precondition produces
# evidence about nothing.
#
# It also does not pass `--allow-dirty`. A real launch refuses a dirty tree:
# `run/provenance.check_git_clean` is what makes each trial's spec.yaml
# attributable to a commit, and 256 trials whose code state is "some
# uncommitted mixture" are 256 unreproducible results. Commit first.

set -uo pipefail

# SLURM_SUBMIT_DIR, not BASH_SOURCE: sbatch COPIES this script to a spool dir and
# runs it there, so ${BASH_SOURCE[0]} is /cm/local/apps/slurm/var/spool/jobNNN/
# slurm_script and deriving a repo root from it yields
# /cm/local/apps/slurm/var/spool. Measured in job 440764, not assumed.
# BASH_SOURCE is the fallback for running this file directly; REPO_ROOT=...
# overrides both. Always sbatch from the repo root.
export REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
if [ ! -d "$REPO_ROOT/experimentation" ] || [ ! -d "$REPO_ROOT/code-tests" ]; then
    echo "FATAL: REPO_ROOT=$REPO_ROOT is not a Koopman_Mamba checkout."
    echo "       sbatch from the repo root, or pass REPO_ROOT=/path/to/checkout."
    exit 1
fi

export SCRATCH=/scratch/m000151-pm06/jkli
export VENV=$SCRATCH/venvs/koopman-cuda
export OPTUNA_SITE=/users/jkli/.venvs/koopman-optuna/site
export STUDY_CONFIG="${STUDY_CONFIG:-configs/search/proxy-256x17-interactions-v1.yaml}"

# Per-worker logs. The CLI's children inherit this process's stdout, so without
# somewhere to route them all eight trainers interleave into one file and a
# tracebacks's first line lands under another worker's progress. The run system
# already gives every TRIAL its own run_dir/*.log; this is for the WORKER
# processes' own output.
export WORKER_LOG_DIR="$SCRATCH/logs/koopman-studies/workers-${SLURM_JOB_ID:-manual}"
mkdir -p "$WORKER_LOG_DIR"

echo "=== job ${SLURM_JOB_ID:-<interactive>} on $(hostname) ==="
date -Is
cd "$REPO_ROOT" || { echo "FATAL: no repo root"; exit 1; }
echo "repo:    $REPO_ROOT"
echo "commit:  $(git rev-parse HEAD)"
echo "tree:    $(git status --porcelain | wc -l) modified path(s)"
echo "study:   $STUDY_CONFIG"
echo "logs:    $WORKER_LOG_DIR"

module purge
module load cuda12.9/toolkit/12.9.1
module load gcc/13.1.0
nvidia-smi --query-gpu=index,name,driver_version,compute_cap,memory.total \
    --format=csv,noheader || true

# shellcheck disable=SC1091
source "$VENV/bin/activate"
# Optuna is SIDE-LOADED. It is not installed in the venv and not in
# pyproject.toml's resolved environment; several modules are written to import
# without it on purpose, which is what lets --dry_run work anywhere.
export PYTHONPATH=$REPO_ROOT:$OPTUNA_SITE
export HF_HOME=$SCRATCH/hf
export TRITON_CACHE_DIR=$SCRATCH/triton-cache
# Eight trainers x N tokenizer threads each oversubscribes the node for no gain.
export TOKENIZERS_PARALLELISM=false

##############################################################################
echo "############################################################"
echo "### GATE 1: can torch see CUDA, and can python see optuna?"
echo "############################################################"
python - <<'PY'
import sys
import torch
import optuna
print("torch  ", torch.__version__, "cuda", torch.cuda.is_available())
print("optuna ", optuna.__version__, "from", optuna.__file__)
if not torch.cuda.is_available():
    print("FATAL: torch cannot see a GPU. nvidia-smi may still show one --")
    print("       job 439811 landed on exactly that node. A study with no GPU")
    print("       fails every trial and tells you nothing.")
    sys.exit(1)
count = torch.cuda.device_count()
for index in range(count):
    print(f"  cuda:{index}  {torch.cuda.get_device_name(index)}")
print("devices", count)
if count < 8:
    print(f"FATAL: {count} device(s) visible, need 8. This study's whole shape")
    print( "       is eight concurrent trials pinned one per GPU; with fewer,")
    print( "       _fanout wraps round-robin and N workers time-slice one")
    print( "       device -- so it would run, slowly, and the 12h limit was")
    print( "       sized for 8. Check --gpus-per-node and CUDA_VISIBLE_DEVICES.")
    sys.exit(1)
PY
GATE1=$?
echo "STATUS gate1=$GATE1"
if [ "$GATE1" -ne 0 ]; then
    echo "FATAL: aborting before spending anything."
    exit 1
fi

##############################################################################
echo "############################################################"
echo "### GATE 2: the dry run -- the whole plan, nothing spent"
echo "############################################################"
# Validates the study spec, resolves the base config, restricts and validates
# every declared search axis against its real domain, loads and resolves all 24
# anchors, and prints the plan. A bad axis, an undeclared anchor power_K, a
# retired backend policy or a missing base spec all die HERE rather than on
# trial 0 with a run directory already claimed.
python -m experimentation.sweep.search "$STUDY_CONFIG" --dry_run
GATE2=$?
echo "STATUS dry_run=$GATE2"
if [ "$GATE2" -ne 0 ]; then
    echo "FATAL: the dry run failed, so the study is misconfigured. Not"
    echo "       launching -- this is the gate that exists to catch exactly"
    echo "       this for free."
    exit 1
fi

##############################################################################
echo "############################################################"
echo "### THE STUDY: one invocation; the CLI spawns and pins the fleet"
echo "############################################################"
# ONE call. `concurrent_trials: 8` in the study file is what makes this a fleet:
# __main__._fanout starts eight children, sets CUDA_VISIBLE_DEVICES per child
# round-robin over the visible devices, gives each its own sampler seed
# (seed + worker index -- the SAMPLER's stream only; the model/data seed is
# identical for every trial, so identical params still give identical run_ids),
# waits for all of them, reattaches to the journal and writes one report.
#
# Do not add a loop here. See this file's header.
#
# No --allow-dirty: a real launch refuses a dirty tree, because 256 trials whose
# code state is an uncommitted mixture are 256 unreproducible results.
#
# `stdbuf -oL` so the supervisor's progress lines reach the log as they happen
# rather than in 4 KB blocks -- this job runs for hours and an unflushed pipe
# makes it impossible to tell "working" from "hung".
stdbuf -oL -eL python -m experimentation.sweep.search "$STUDY_CONFIG" \
    2>&1 | tee "$WORKER_LOG_DIR/supervisor.log"
STUDY_STATUS=${PIPESTATUS[0]}
echo "STATUS study=$STUDY_STATUS"

# The supervisor already waited for every worker and returns non-zero if ANY of
# them did (`_fanout` -> `(0 if all(c == 0 for c in codes) else 1)`, and it
# prints each failing worker's index and exit code). So there is no `wait` to do
# here and no per-child status to collect -- doing either would be a second,
# disagreeing account of the same thing. What this DOES do is refuse to report
# success when the supervisor did not.
if [ "$STUDY_STATUS" -ne 0 ]; then
    echo "############################################################"
    echo "### A WORKER EXITED NON-ZERO"
    echo "############################################################"
    echo "The supervisor's own output above names which worker and with what"
    echo "code. Completed trials are NOT lost: the journal is durable, so"
    echo "resubmitting this same script attaches to the same study and runs"
    echo "only the remainder."
fi

##############################################################################
echo "############################################################"
echo "### WHAT LANDED"
echo "############################################################"
# Read back from the journal rather than trusting the supervisor's summary: the
# journal is what actually happened, and this also surfaces the target overshoot
# (up to concurrent_trials-1 extra trials can be in flight when the target is
# reached, so 256-263 recorded is expected and not a bug).
python - <<'PY'
import os
import pathlib
import sys

sys.path.insert(0, os.environ["REPO_ROOT"])
from experimentation.sweep.search.studyspec import load_study_spec, study_id

spec = load_study_spec(os.environ["STUDY_CONFIG"])
study_dir = (pathlib.Path(spec.run_root) / "_studies"
             / f"{spec.name}.{study_id(spec)}")
print("study_dir", study_dir)
journal = study_dir / "optuna_journal.log"
if not journal.is_file():
    print(f"NO JOURNAL at {journal} -- the driver never started.")
    raise SystemExit(1)

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
storage = optuna.storages.JournalStorage(
    optuna.storages.journal.JournalFileBackend(str(journal)))
study = optuna.load_study(study_name=spec.name, storage=storage)

states = {}
for trial in study.trials:
    states[trial.state.name] = states.get(trial.state.name, 0) + 1
print("trials", len(study.trials), states)

lo, hi = spec.n_trials, spec.n_trials + spec.concurrent_trials - 1
finished = sum(v for k, v in states.items()
               if k in ("COMPLETE", "PRUNED", "FAIL"))
print(f"target {spec.n_trials}, expected range {lo}-{hi}, finished {finished}")
if finished > hi:
    print(f"WARNING: {finished} finished exceeds the documented overshoot bound "
          f"{hi}. That bound is workers-1 and is tested "
          f"(test_search_fleet_overshoot.py); exceeding it means more processes "
          f"attached to this journal than this script started.")

workers = {t.user_attrs.get("worker_id") for t in study.trials}
seeds = {t.user_attrs.get("sampler_seed") for t in study.trials}
print("worker ids     ", sorted(w for w in workers if w is not None))
print("sampler seeds  ", sorted(s for s in seeds if s is not None))
if len([s for s in seeds if s is not None]) == 1 and spec.concurrent_trials > 1:
    print("WARNING: every trial records ONE sampler seed. Eight workers should")
    print("         record eight (seed..seed+7). One means the fleet shared a")
    print("         random stream, so constant_liar was repelling proposals")
    print("         that were identical by construction.")

for name in ("trials.csv", "top_trials.md", "best_trial.json"):
    print(f"{name:16} {'yes' if (study_dir / name).is_file() else 'MISSING'}")
PY
echo "STATUS inspect=$?"

##############################################################################
echo "############################################################"
echo "### THE ANALYSIS (read-only; safe even if a worker died)"
echo "############################################################"
python - <<'PY' > /tmp/koopman-study-dir.$$ 2>/dev/null || true
import os, pathlib, sys
sys.path.insert(0, os.environ["REPO_ROOT"])
from experimentation.sweep.search.studyspec import load_study_spec, study_id
spec = load_study_spec(os.environ["STUDY_CONFIG"])
print(pathlib.Path(spec.run_root) / "_studies" / f"{spec.name}.{study_id(spec)}")
PY
STUDY_DIR="$(cat /tmp/koopman-study-dir.$$ 2>/dev/null)"
rm -f "/tmp/koopman-study-dir.$$"
if [ -n "$STUDY_DIR" ] && [ -f "$STUDY_DIR/optuna_journal.log" ]; then
    python scripts/analyze_interactions.py "$STUDY_DIR"
    echo "STATUS analysis=$?"
else
    echo "SKIPPED: no journal to analyse."
fi

echo
echo "run_root:    $(python -c "import sys,os;sys.path.insert(0,os.environ['REPO_ROOT']);from experimentation.sweep.search.studyspec import load_study_spec;print(load_study_spec(os.environ['STUDY_CONFIG']).run_root)")"
echo "worker logs: $WORKER_LOG_DIR"
date -Is
echo "=== END (study exit $STUDY_STATUS) ==="
exit "$STUDY_STATUS"
