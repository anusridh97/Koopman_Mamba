#!/bin/bash
#SBATCH --job-name=koopman-beta-exp-arr
#SBATCH --account=marlowe-m000151-pm06
#SBATCH --partition=batch
#SBATCH --qos=medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/m000151-pm06/jkli/logs/koopman-tests/%x-%A_%a.out
#SBATCH --error=/scratch/m000151-pm06/jkli/logs/koopman-tests/%x-%A_%a.err
#
# The exponent arm as a slurm ARRAY of one-GPU tasks.
#
# ============================ WHY AN ARRAY =================================
#
# The cluster is GPU-saturated: every node reports 8/8 GRES allocated. A
# whole-node 8-GPU request for this arm queued ~12 hours out, while a 1-GPU
# request BACKFILLED in ten minutes (job 446640, observed the same morning). An
# array of 1-GPU tasks fits the gaps; one 8-GPU job waits for a node to drain.
#
# 14 CPUs per GPU is the node's own 112/8 ratio, so one task is a balanced slice
# and the billing (1 unit per CPU) matches what it uses.
#
# ========================== HOW TO RUN IT ==================================
#
#   1. Prepare -- on the LOGIN NODE, so a gate failure costs seconds rather than
#      N queued tasks each discovering it separately:
#
#        RUN_ROOT=/scratch/m000151-pm06/jkli/beta-exponent-array-01
#        python scripts/prepare_beta_exponent_arm.py --run_root $RUN_ROOT
#
#      It prints `ARRAY_WIDTH=0-N`. That number comes from the SAME cell list the
#      tasks read, so the width cannot disagree with the design.
#
#   2. Submit:
#
#        sbatch --array=0-41 --export=ALL,RUN_ROOT=$RUN_ROOT \
#            scripts/run_beta_exponent_array.sbatch
#
#   3. Report, once the array is done (needs a GPU for the ablation pass):
#
#        sbatch --dependency=afterany:<ARRAYJOBID> \
#            --export=ALL,RUN_ROOTS=$RUN_ROOT \
#            scripts/measure_ska_ablation_on_existing_mqar.sbatch
#
# ===================== WHY EACH TASK RE-CHECKS ITS OWN CELL =================
#
# The preparer already ran every gate over the whole arm. This task runs them
# again for ITS cell, because nothing guarantees it is running the arm that was
# prepared: a stale RUN_ROOT, an array re-submitted against an edited cell list,
# or an index off the end of the manifest each produce a silently WRONG run
# rather than an error. `preflight_failures` is the same function both times --
# one copy, so the login-node check cannot drift from the GPU one.
#
# Tasks are ordered SEED-MAJOR (see `beta_exponent_cells.manifest`), so a
# partially-completed array is N COMPLETE paired seeds across all cells rather
# than a ragged set. A ragged set is not a paired design.

set -uo pipefail

export REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
if [ ! -d "$REPO_ROOT/experimentation" ]; then
    echo "FATAL: REPO_ROOT=$REPO_ROOT is not a Koopman_Mamba checkout."; exit 1
fi
export SCRATCH=/scratch/m000151-pm06/jkli
export VENV=$SCRATCH/venvs/koopman-cuda

if [ -z "${RUN_ROOT:-}" ]; then
    echo "FATAL: RUN_ROOT is unset. Run prepare_beta_exponent_arm.py first and"
    echo "       pass --export=ALL,RUN_ROOT=<that dir>. Defaulting it here would"
    echo "       let a task write into a run root nobody prepared."
    exit 1
fi
IDX="${SLURM_ARRAY_TASK_ID:-${IDX:-}}"
if [ -z "$IDX" ]; then
    echo "FATAL: no SLURM_ARRAY_TASK_ID and no IDX. This script is an array task."
    exit 1
fi

echo "=== array ${SLURM_ARRAY_JOB_ID:-<none>} task $IDX on $(hostname) ==="
date -Is
cd "$REPO_ROOT" || exit 1
echo "commit:   $(git rev-parse HEAD)"
echo "dirty:    $(git status --porcelain | wc -l) modified path(s)"
echo "run_root: $RUN_ROOT"

module purge
module load cuda12.9/toolkit/12.9.1
module load gcc/13.1.0
# shellcheck disable=SC1091
source "$VENV/bin/activate"
export PYTHONPATH=$REPO_ROOT
export HF_HOME=$SCRATCH/hf
export TRITON_CACHE_DIR=$SCRATCH/triton-cache
export TOKENIZERS_PARALLELISM=false

##############################################################################
# Resolve THIS task's cell, re-run the gates for it, and emit the command line
# as shell variables. Exits non-zero on any gate failure -- a gate, not a note.
##############################################################################
PLAN="$RUN_ROOT/task-$IDX.env"
IDX="$IDX" python - > "$PLAN" <<'PY' || { echo "FATAL: task pre-flight FAILED"; cat "$PLAN"; exit 1; }
import os, pathlib, sys
sys.path.insert(0, os.environ["REPO_ROOT"])
from experimentation.experiments.beta_exponent_cells import (
    EVAL_EVERY, GAP, KV, STEPS, TASK_VOCAB, preflight_failures, resolve_cell,
    route_of, task_for_index)
from experimentation.run.resolve import resolve_run_spec
from koopman_lm.config import config_hash, load_config

repo = pathlib.Path(os.environ["REPO_ROOT"])
run_root = pathlib.Path(os.environ["RUN_ROOT"])
task = task_for_index(int(os.environ["IDX"]))          # raises if out of range
base = resolve_run_spec(repo / "configs/runs/proxy-256x17.yaml").model

fails = preflight_failures([task.cell], base)

# The YAML this task will actually train from must exist (the preparer wrote it)
# and must still round-trip to the cell's resolved config. Checking the FILE and
# not just the in-memory config is what makes the ridge control real: the file is
# what `build_config` reads.
model_yaml = run_root / f"model-{task.cell.name}.yaml"
cfg = resolve_cell(task.cell, base)
if not model_yaml.exists():
    fails.append(f"{model_yaml} is missing -- was prepare_beta_exponent_arm.py "
                 f"run against this RUN_ROOT?")
else:
    on_disk = load_config(model_yaml)
    if on_disk != cfg:
        fails.append(f"{model_yaml} does not match cell {task.cell.name}: "
                     f"policy {on_disk.ska_beta_policy} ridge {on_disk.ska_ridge} "
                     f"vs expected {task.cell.policy} / {task.cell.ridge}")
    elif float(on_disk.ska_ridge) != float(task.cell.ridge):
        fails.append(f"{model_yaml} ridge is {on_disk.ska_ridge}, cell declares "
                     f"{task.cell.ridge}")

if fails:
    for f in fails:
        print(f"GATE FAILURE: {f}", file=sys.stderr)
    raise SystemExit(1)

print(f"CELL_NAME={task.cell.name}")
print(f"POLICY={task.cell.policy}")
print(f"RIDGE={task.cell.ridge}")
print(f"SEED={task.seed}")
print(f"RUN_NAME={task.run_name}")
print(f"MODEL_YAML={model_yaml}")
print(f"STEPS={STEPS}")
print(f"EVAL_EVERY={EVAL_EVERY}")
print(f"SAVE_STEPS={max(STEPS // 4, 1)}")
print(f"KV={KV}")
print(f"GAP={GAP}")
print(f"TASK_VOCAB={TASK_VOCAB}")
print(f"# route={route_of(cfg)} gamma={cfg.ska_gamma_value} "
      f"K={cfg.ska_power_K} hash={config_hash(cfg)[:8]}", file=sys.stderr)
PY

# shellcheck disable=SC1090
source "$PLAN"
echo "cell:  $CELL_NAME  policy=$POLICY ridge=$RIDGE seed=$SEED"
echo "budget: $STEPS steps, eval every $EVAL_EVERY, save every $SAVE_STEPS"

python -c "
import torch
assert torch.cuda.is_available(), 'no CUDA device visible to torch'
print('torch', torch.__version__, 'devices', torch.cuda.device_count())
" || { echo "FATAL: torch cannot see a GPU."; exit 1; }

##############################################################################
OUT="$RUN_ROOT/$RUN_NAME"
LOG="$RUN_ROOT/$RUN_NAME.log"
echo "training -> $OUT"
# `--resume_from` is not passed: a preempted task is re-run by slurm's requeue
# from scratch rather than resumed, because `--deterministic` makes a fresh run
# reproducible while a resume across a preemption boundary is not bit-exact and
# would put two different training trajectories in one accuracy curve.
# `-u`: stdout is a PIPE here (into tee), so python block-buffers it. A task
# killed by the walltime or by preemption would then lose the tail of its
# accuracy curve even though its checkpoints survived -- and the curve is what
# the predeclared statistics are computed from. Unbuffered costs nothing at this
# log volume and makes the log a live record rather than a post-mortem one.
python -u -m experimentation.experiments.mqar_finetune \
    --model_type mamba_ska_swiglu \
    --model_size "$MODEL_YAML" \
    --ska_beta_policy "$POLICY" \
    --num_kv_pairs "$KV" \
    --distractor_gap "$GAP" \
    --task_vocab_size "$TASK_VOCAB" \
    --max_steps "$STEPS" \
    --eval_every "$EVAL_EVERY" \
    --save_steps "$SAVE_STEPS" \
    --seed "$SEED" \
    --deterministic \
    --output_dir "$OUT" 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}

echo
echo "task $IDX ($RUN_NAME) exit=$rc"
if [ "$rc" -ne 0 ]; then
    echo "CLASSIFY THIS before it appears in any table: complete / censored /"
    echo "preempted / OOM / numerical failure / cancelled. An infrastructure"
    echo "failure retried is not a scientific failure."
fi
date -Is
echo "=== END ==="
exit $rc
