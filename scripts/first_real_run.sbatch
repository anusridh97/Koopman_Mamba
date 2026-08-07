#!/bin/bash
#SBATCH --job-name=koopman-first-real-run
#SBATCH --account=marlowe-m000151-pm06
#SBATCH --partition=batch
#SBATCH --qos=medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:45:00
#SBATCH --output=/scratch/m000151-pm06/jkli/logs/first-run/%x-%j.out
#SBATCH --error=/scratch/m000151-pm06/jkli/logs/first-run/%x-%j.err
#
# The repo's first genuinely real training run through koopman_lm/run/ (the
# new run system: RunSpec -> resolve -> materialize -> Launcher), on real
# FineWeb-Edu data, seq_len 2048, with ska_backend left at its production
# value 'cuda_prefix' -- the fused SKA prefix-scan CUDA extension that first
# compiled and passed its 9-test correctness suite on H100 today (the
# `-lineinfo` nvcc fix + the ska_backend='auto' fix). Nothing has ever trained
# through this path before this job.
#
# Steps (all in this one job, per the design in
# docs/superpowers/specs/2026-08-07-run-system-design.md):
#   1. tokenize a real 100M-token FineWeb-Edu train shard + a disjoint val
#      shard (pretokenize.py --skip_docs past the train shard's docs)
#   2. render + commit configs/runs/50m-first-real.yaml with the ACTUAL
#      token count pretokenize.py wrote (verify_shard requires an exact
#      match against meta.json, so this can't be authored before tokenizing)
#   3. launch via `python -m koopman_lm.run ... --launcher local`
#   4. verify the run-directory artifacts the design promises
#   5. kill the run after its first checkpoint (SIGTERM, not SIGUSR1 -- see
#      note below) and relaunch with --resume, confirming from the log that
#      training continues from the saved step
#   6. evaluate the final checkpoint (fineweb_ppl) and hand-wrap the result
#      in the §4.2 envelope (evaluate.py itself isn't wired to
#      koopman_lm.run.eval_result yet -- see FINDINGS at the end of the log)
#
# Uses the ALREADY-BUILT CUDA venv (do not rebuild -- ~25 min):
#   /scratch/m000151-pm06/jkli/venvs/koopman-cuda
#
# NOTE on SIGUSR1: job 415208 (an earlier E2E smoke through this same run
# system) died with an uncaught SIGUSR1 partway through, and a different
# agent is fixing that handler concurrently with this job. This job therefore
# uses SIGTERM (train.py has no handler for it -> prompt, unambiguous exit)
# against the last *checkpointed* step, not the exact instant of a
# preemption signal.

set -uo pipefail

export REPO_ROOT=/users/jkli/Koopman_Mamba/.claude/worktrees/first-run
export SCRATCH=/scratch/m000151-pm06/jkli
export VENV=$SCRATCH/venvs/koopman-cuda
export RUN_ROOT=$SCRATCH/runs
export TRAIN_DIR=$SCRATCH/fineweb_small_train
export VAL_DIR=$SCRATCH/fineweb_small_val
LOGDIR=$SCRATCH/logs/first-run
TOKENIZER=NousResearch/Llama-2-7b-hf
CFG_NAME=50m-first-real
CFG_PATH=configs/runs/${CFG_NAME}.yaml

echo "=== job ${SLURM_JOB_ID:-<interactive>} on $(hostname) ==="
date -Is
mkdir -p "$LOGDIR"
cd "$REPO_ROOT" || { echo "FATAL: no such repo root $REPO_ROOT"; exit 1; }
echo "repo: $REPO_ROOT"
git rev-parse HEAD
git status --short
echo "(expect clean -- the run system refuses to launch from a dirty tree)"

module purge
module load cuda12.9/toolkit/12.9.1
module load gcc/13.1.0
echo "nvcc: $(command -v nvcc)"
nvcc --version || true
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader || true

# shellcheck disable=SC1091
source "$VENV/bin/activate"
export PYTHONPATH=$REPO_ROOT
python -c "
import torch
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'is_available', torch.cuda.is_available())
assert torch.cuda.is_available(), 'expected a visible CUDA device on this GPU node'
print('device', torch.cuda.get_device_name(0), 'capability', torch.cuda.get_device_capability(0))
"
VENV_SANITY_STATUS=$?
echo "STATUS venv_sanity=$VENV_SANITY_STATUS"
if [ "$VENV_SANITY_STATUS" -ne 0 ]; then
    echo "FATAL: the pre-built koopman-cuda venv does not see CUDA on this node. Aborting."
    exit 1
fi

export HF_HOME=$SCRATCH/hf
export TRITON_CACHE_DIR=$SCRATCH/triton-cache
export TORCHINDUCTOR_CACHE_DIR=$SCRATCH/inductor-cache
mkdir -p "$HF_HOME" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
export TORCH_CUDA_ARCH_LIST=9.0
export SKA_REQUIRE_B200=0
export MAX_JOBS=8

##############################################################################
# STEP 0: build + numerically validate the fused CUDA extension (should hit
# the extension cache from job 415255 and be fast; this just re-confirms it
# on THIS node before spending any training time).
##############################################################################
echo "############################################################"
echo "### STEP 0: build_prefix_scan_cuda.py"
echo "############################################################"
python scripts/build_prefix_scan_cuda.py 2>&1 | tee "$LOGDIR/build-${SLURM_JOB_ID:-manual}.log"
BUILD_STATUS=${PIPESTATUS[0]}
echo "STATUS step0_build_fused_prefix_scan=$BUILD_STATUS"
if [ "$BUILD_STATUS" -ne 0 ]; then
    echo "FATAL: fused CUDA extension failed to build/validate. Aborting."
    exit 1
fi

##############################################################################
# STEP 1: tokenize a real 100M-token FineWeb-Edu train shard, then a disjoint
# 4M-token val shard past everything the train shard consumed.
##############################################################################
echo "############################################################"
echo "### STEP 1: tokenize train shard (real FineWeb-Edu, 100M tokens)"
echo "############################################################"
if [ ! -f "$TRAIN_DIR/train.bin" ]; then
    python -m koopman_lm.training.data.pretokenize \
        --output_dir "$TRAIN_DIR" --tokenizer "$TOKENIZER" \
        --fineweb HuggingFaceFW/fineweb-edu --fineweb_subset sample-10BT \
        --mix 1.0 0.0 0.0 --max_tokens 100000000 --shard_size 50000000 --seed 42 \
        2>&1 | tee "$LOGDIR/tokenize-train-${SLURM_JOB_ID:-manual}.log"
    TOKENIZE_TRAIN_STATUS=${PIPESTATUS[0]}
else
    echo "train shard already exists at $TRAIN_DIR, skipping (idempotent re-run)"
    TOKENIZE_TRAIN_STATUS=0
fi
echo "STATUS step1_tokenize_train=$TOKENIZE_TRAIN_STATUS"
if [ "$TOKENIZE_TRAIN_STATUS" -ne 0 ] || [ ! -f "$TRAIN_DIR/meta.json" ]; then
    echo "FATAL: train shard tokenization failed. Aborting."
    exit 1
fi

DOCS_CONSUMED=$(python -c "import json;print(json.load(open('$TRAIN_DIR/meta.json'))['fineweb_docs_consumed'])")
SKIP_DOCS=$(( DOCS_CONSUMED + 10000 ))
echo "train shard consumed $DOCS_CONSUMED fineweb docs; val shard will skip_docs=$SKIP_DOCS"

echo "############################################################"
echo "### STEP 1b: tokenize disjoint val shard (4M tokens)"
echo "############################################################"
if [ ! -f "$VAL_DIR/train.bin" ]; then
    python -m koopman_lm.training.data.pretokenize \
        --output_dir "$VAL_DIR" --tokenizer "$TOKENIZER" \
        --fineweb HuggingFaceFW/fineweb-edu --fineweb_subset sample-10BT \
        --mix 1.0 0.0 0.0 --max_tokens 4000000 --shard_size 4000000 \
        --skip_docs "$SKIP_DOCS" --seed 1337 \
        2>&1 | tee "$LOGDIR/tokenize-val-${SLURM_JOB_ID:-manual}.log"
    TOKENIZE_VAL_STATUS=${PIPESTATUS[0]}
else
    echo "val shard already exists at $VAL_DIR, skipping (idempotent re-run)"
    TOKENIZE_VAL_STATUS=0
fi
echo "STATUS step1b_tokenize_val=$TOKENIZE_VAL_STATUS"
if [ "$TOKENIZE_VAL_STATUS" -ne 0 ] || [ ! -f "$VAL_DIR/meta.json" ]; then
    echo "FATAL: val shard tokenization failed. Aborting."
    exit 1
fi

##############################################################################
# STEP 2: render configs/runs/50m-first-real.yaml with the ACTUAL n_tokens
# pretokenize.py wrote (koopman_lm.run.data_verify.verify_shard requires an
# exact match against the shard's meta.json, so this cannot be hand-authored
# ahead of tokenizing -- pretokenize.py's per-document overshoot means the
# real count is not knowable until the shard exists), then commit it (the
# run system refuses to launch from a dirty tree).
##############################################################################
echo "############################################################"
echo "### STEP 2: render + validate + commit $CFG_PATH"
echo "############################################################"
python - "$TRAIN_DIR" "$CFG_PATH" "$CFG_NAME" "$TOKENIZER" <<'PY'
import json
import sys

import yaml

train_dir, cfg_path, cfg_name, tokenizer = sys.argv[1:5]

with open(train_dir + "/meta.json") as f:
    meta = json.load(f)
n_tokens = meta["n_tokens"]
mix = meta["mix"]

with open("configs/50m.yaml") as f:
    model = yaml.safe_load(f)
# The run system's model field is either a registry name (defaults only) or
# a fully inlined dict (resolve.py._resolve_model) -- there is no per-field
# override of a registry entry yet, so extending "50m" with a non-default
# max_seq_len means inlining 50m.yaml's fields plus this override, not a
# YAML `extends:`. ska_backend stays at 50m.yaml's own value (cuda_prefix) --
# untouched, which is the entire point of this run.
model["max_seq_len"] = 2048

spec = {
    "name": cfg_name,
    "model": model,
    "data": {
        "kind": "shard",
        "shard_dir": train_dir,
        "tokenizer": tokenizer,
        "mix": mix,
        "n_tokens": n_tokens,
    },
    "optim": {
        "lr": 0.0004,
        "warmup_steps": 30,
        "max_steps": 300,
        "schedule": "cosine",
        "effective_batch": 64,
        "per_device_batch_size": 16,
        "weight_decay": 0.1,
        "grad_clip": 1.0,
    },
    "runtime": {
        "seed": 42,
        "precision": "bf16",
        "ddp": False,
        "gpus": 1,
        "nodes": 1,
        "partition": "batch",
        "account": "marlowe-m000151-pm06",
        "qos": "medium",
        "workers": 4,
        "time_limit": "01:45:00",
        "gpu_arch": "9.0",
    },
}

header = (
    "# The repo's first genuinely real training run through koopman_lm/run/,\n"
    "# on a real 100M-token FineWeb-Edu shard, with ska_backend left at\n"
    "# configs/50m.yaml's own value (cuda_prefix -- the fused CUDA prefix-scan\n"
    "# kernel). model: is configs/50m.yaml inlined with max_seq_len overridden\n"
    "# to 2048 (there is no field-level override of a registry model name yet,\n"
    "# see koopman_lm/run/resolve.py:_resolve_model). data.n_tokens is the\n"
    f"# ACTUAL count {train_dir}/meta.json recorded (pretokenize.py's\n"
    "# per-document overshoot makes the exact figure only knowable after\n"
    "# tokenizing -- see scripts/first_real_run.sbatch step 2). Generated and\n"
    "# committed by that script; not hand-authored.\n"
)
with open(cfg_path, "w") as f:
    f.write(header)
    yaml.safe_dump(spec, f, sort_keys=False)

print(f"wrote {cfg_path} (n_tokens={n_tokens:,}, mix={mix})")

# Fail fast, before committing, if this doesn't actually resolve.
sys.path.insert(0, ".")
from koopman_lm.run.resolve import resolve_run_spec
from koopman_lm.run.spec import group_id, run_id

resolved = resolve_run_spec(cfg_path)
assert resolved.model.ska_backend == "cuda_prefix", resolved.model.ska_backend
assert resolved.model.ska_prefix_scan is True
assert resolved.model.max_seq_len == 2048
print(f"resolved OK: group_id={group_id(resolved)} run_id={run_id(resolved)}")
PY
RENDER_STATUS=$?
echo "STATUS step2_render_config=$RENDER_STATUS"
if [ "$RENDER_STATUS" -ne 0 ]; then
    echo "FATAL: config render/validation failed. Aborting."
    exit 1
fi

git add "$CFG_PATH"
if git diff --cached --quiet -- "$CFG_PATH"; then
    echo "$CFG_PATH already committed identically (idempotent re-run) -- nothing to commit"
    GIT_COMMIT_STATUS=0
else
    git commit -m "config(runs): add ${CFG_NAME} (job ${SLURM_JOB_ID:-manual})

Real 100M-token FineWeb-Edu shard, seq_len 2048, ska_backend=cuda_prefix
(the fused SKA prefix-scan CUDA extension). Generated by
scripts/first_real_run.sbatch step 2 with the shard's actual token count."
    GIT_COMMIT_STATUS=$?
fi
echo "STATUS step2_git_commit=$GIT_COMMIT_STATUS"
git status --short
git log --oneline -1
if [ "$GIT_COMMIT_STATUS" -ne 0 ]; then
    echo "FATAL: could not commit $CFG_PATH -- run system will refuse a dirty tree. Aborting."
    exit 1
fi
if [ -n "$(git status --porcelain)" ]; then
    echo "FATAL: working tree still dirty after commit -- run system will refuse to launch."
    git status --short
    exit 1
fi

RUN_DIR=$(python -c "
from koopman_lm.run.resolve import resolve_run_spec
from koopman_lm.run.spec import run_dir_path
print(run_dir_path('$RUN_ROOT', resolve_run_spec('$CFG_PATH')))
")
echo "RUN_DIR=$RUN_DIR"

##############################################################################
# STEP 3: launch through the run system (not the old shell scripts),
# in the background so this script can kill it partway through for the
# resume test.
##############################################################################
echo "############################################################"
echo "### STEP 3: python -m koopman_lm.run $CFG_PATH --launcher local"
echo "############################################################"
setsid python -m koopman_lm.run "$CFG_PATH" --launcher local --run_root "$RUN_ROOT" \
    > "$LOGDIR/run1-${SLURM_JOB_ID:-manual}.log" 2>&1 &
RUN1_PID=$!
echo "run1 pid (session/process-group leader)=$RUN1_PID"

##############################################################################
# STEP 4 (partial, pre-kill): wait for the first archival checkpoint +
# resume.pt to appear, then SIGTERM the whole process group. train.py has no
# SIGTERM handler (unlike SIGUSR1, whose handler is being fixed by another
# agent concurrently -- see header note), so this is a prompt, unambiguous
# kill against the last state _save_all wrote together (§5.3): step_<N>/ and
# resume.pt are written in that order, so resume.pt existing implies
# step_<N>/model.pt already does too.
##############################################################################
echo "############################################################"
echo "### STEP 4: wait for first checkpoint, then SIGTERM (mid-run kill)"
echo "############################################################"
WAITED=0
TIMEOUT=1800
while [ ! -f "$RUN_DIR/resume.pt" ]; do
    if ! kill -0 "$RUN1_PID" 2>/dev/null; then
        echo "run1 (pid $RUN1_PID) already exited before any checkpoint appeared -- see run1 log"
        break
    fi
    if [ "$WAITED" -ge "$TIMEOUT" ]; then
        echo "WARNING: no checkpoint after ${TIMEOUT}s; giving up waiting and killing run1 as-is"
        break
    fi
    sleep 2
    WAITED=$((WAITED + 2))
done
echo "waited ${WAITED}s for first checkpoint"
ls -la "$RUN_DIR" 2>&1 || true

if kill -0 "$RUN1_PID" 2>/dev/null; then
    echo "sending SIGTERM to process group -$RUN1_PID"
    kill -TERM "-$RUN1_PID" 2>&1 || kill -TERM "$RUN1_PID" 2>&1
    sleep 3
    if kill -0 "$RUN1_PID" 2>/dev/null; then
        echo "still alive after SIGTERM+3s, sending SIGKILL"
        kill -KILL "-$RUN1_PID" 2>&1 || kill -KILL "$RUN1_PID" 2>&1
    fi
fi
wait "$RUN1_PID" 2>&1
RUN1_EXIT=$?
echo "STATUS step4_run1_exit=$RUN1_EXIT (nonzero/killed-by-signal is EXPECTED here)"
tail -n 40 "$LOGDIR/run1-${SLURM_JOB_ID:-manual}.log"

##############################################################################
# STEP 5: relaunch with --resume (foreground; runs to completion) and
# confirm from the log it continues from the saved step.
##############################################################################
echo "############################################################"
echo "### STEP 5: python -m koopman_lm.run $CFG_PATH --launcher local --resume"
echo "############################################################"
python -m koopman_lm.run "$CFG_PATH" --launcher local --run_root "$RUN_ROOT" --resume \
    2>&1 | tee "$LOGDIR/run2-resume-${SLURM_JOB_ID:-manual}.log"
RUN2_STATUS=${PIPESTATUS[0]}
echo "STATUS step5_run2_resume=$RUN2_STATUS"

echo "--- resume confirmation (grep 'Resumed from' in run2 log) ---"
grep -n "Resumed from" "$LOGDIR/run2-resume-${SLURM_JOB_ID:-manual}.log" || echo "NOT FOUND -- resume did not announce itself"

##############################################################################
# STEP 6: artifact verification (design §3.6/§3.7)
##############################################################################
echo "############################################################"
echo "### STEP 6: artifact verification"
echo "############################################################"
echo "--- run dir tree ---"
find "$RUN_DIR" -maxdepth 2 | sort
echo "--- spec.yaml has no 'extends:' key ---"
grep -c "^extends:" "$RUN_DIR/spec.yaml" && echo "FAIL: extends: present" || echo "OK: no extends: key"
echo "--- attempts.jsonl ---"
cat "$RUN_DIR/attempts.jsonl"
python -c "
import json
lines = [json.loads(l) for l in open('$RUN_DIR/attempts.jsonl')]
assert len(lines) >= 2, f'expected >=2 attempts (initial + resume), got {len(lines)}'
for rec in lines:
    assert 'git_commit' in rec and rec['git_commit'], rec
    assert 'code_id' in rec and rec['code_id'], rec
print(f'OK: {len(lines)} attempts, each with git_commit/code_id')
"
echo "--- step_*/ checkpoints ---"
ls -d "$RUN_DIR"/step_* 2>&1 || echo "FAIL: no step_*/ checkpoints"
echo "--- resume.pt ---"
ls -la "$RUN_DIR/resume.pt" 2>&1 || echo "FAIL: no resume.pt"
echo "--- final/ ---"
ls -la "$RUN_DIR/final" 2>&1 || echo "FAIL: no final/ (training did not complete)"

##############################################################################
# STEP 7: loss curve (report even if it doesn't fall -- that's a finding)
##############################################################################
echo "############################################################"
echo "### STEP 7: loss curve"
echo "############################################################"
echo "--- run1 (pre-kill) ---"
grep "^step " "$LOGDIR/run1-${SLURM_JOB_ID:-manual}.log" || echo "(no step lines logged before kill)"
echo "--- run2 (post-resume) ---"
grep "^step " "$LOGDIR/run2-resume-${SLURM_JOB_ID:-manual}.log" || echo "(no step lines logged)"

##############################################################################
# STEP 8: evaluate the final checkpoint, hand-wrap in the §4.2 envelope.
#
# evaluate.py is NOT wired to koopman_lm.run.eval_result yet (it takes
# --output as a bare path and writes raw metrics, not the run_id/task/
# checkpoint/git_commit/created_at envelope, and does not know about
# spec.yaml/run_dir at all) -- that integration is future work, not part of
# this job's remit (see FINDINGS below). We call the run library's own
# write_result() directly so the artifact this design promises actually
# lands at <run_dir>/eval/<checkpoint>/<task>.json with the real envelope.
##############################################################################
echo "############################################################"
echo "### STEP 8: evaluate final checkpoint (fineweb_ppl)"
echo "############################################################"
FINAL_CKPT="$RUN_DIR/final/model.pt"
RAW_EVAL_OUT="$RUN_DIR/eval/.raw_fineweb_ppl.json"
mkdir -p "$RUN_DIR/eval"
if [ -f "$FINAL_CKPT" ]; then
    python -m koopman_lm.evaluation.evaluate \
        --checkpoint "$FINAL_CKPT" \
        --model_size "$RUN_DIR/model_config.json" \
        --tokenizer "$TOKENIZER" \
        --mode fineweb_ppl --held_out_data_dir "$VAL_DIR" \
        --max_seq_len 2048 --batch_size 8 \
        --output "$RAW_EVAL_OUT" \
        2>&1 | tee "$LOGDIR/eval-${SLURM_JOB_ID:-manual}.log"
    EVAL_STATUS=${PIPESTATUS[0]}
else
    echo "no final/model.pt -- training did not complete, skipping eval"
    EVAL_STATUS=-1
fi
echo "STATUS step8_evaluate=$EVAL_STATUS"

if [ "$EVAL_STATUS" -eq 0 ] && [ -f "$RAW_EVAL_OUT" ]; then
    python -c "
import json
import yaml
from koopman_lm.run.eval_result import write_result

raw = json.load(open('$RAW_EVAL_OUT'))
spec = yaml.safe_load(open('$RUN_DIR/spec.yaml'))
path = write_result(
    '$RUN_DIR', checkpoint='final', task='fineweb_ppl',
    metrics=raw['fineweb_ppl'], run_id=spec['run_id'], git_commit=spec['code_id'],
)
print('wrote envelope ->', path)
print(open(path).read())
"
    WRAP_STATUS=$?
else
    WRAP_STATUS=-1
fi
echo "STATUS step8_wrap_envelope=$WRAP_STATUS"
echo "--- eval/final/fineweb_ppl.json ---"
cat "$RUN_DIR/eval/final/fineweb_ppl.json" 2>&1 || echo "FAIL: envelope file missing"

##############################################################################
# FINAL SUMMARY
##############################################################################
echo "############################################################"
echo "### FINAL SUMMARY"
echo "############################################################"
echo "step0_build_fused_prefix_scan   = $BUILD_STATUS"
echo "step1_tokenize_train            = $TOKENIZE_TRAIN_STATUS"
echo "step1b_tokenize_val             = $TOKENIZE_VAL_STATUS"
echo "step2_render_config             = $RENDER_STATUS"
echo "step2_git_commit                = $GIT_COMMIT_STATUS"
echo "step4_run1_exit (killed)        = $RUN1_EXIT (nonzero expected)"
echo "step5_run2_resume               = $RUN2_STATUS"
echo "step8_evaluate                  = $EVAL_STATUS"
echo "step8_wrap_envelope             = $WRAP_STATUS"
echo "RUN_DIR = $RUN_DIR"
echo ""
echo "FINDINGS (bugs hit and fixed in koopman_lm/run/launch.py, a file this"
echo "agent owns per the task's file-ownership list -- see git log):"
echo "  1. build_train_argv omitted --save_steps, leaving train.py's CLI"
echo "     default (5000) in force; any RunSpec with max_steps < 5000 (e.g."
echo "     this one, 300) would never write a checkpoint or resume.pt at"
echo "     all, making --resume untestable. Fixed: save_steps = max(1,"
echo "     max_steps // 3)."
echo "  2. build_train_argv omitted --no_compile/--no_gradient_checkpointing,"
echo "     leaving train.py's CLI defaults (both True) in force. That"
echo "     combination crashes the fused-prefix-scan architecture on H100"
echo "     with cudaErrorStreamCaptureInvalidated (reproduced in the"
echo "     50m-e2e-gpu-smoke run, job 415208's run2-resume log, under"
echo "     ska_backend=pytorch too -- not CUDA-kernel-specific). Fixed by"
echo "     mirroring scripts/pretrain.sh's already-proven-safe"
echo "     --no_compile --no_gradient_checkpointing for every launch."
echo "  3. evaluate.py is not wired to koopman_lm.run.eval_result (§4.2's"
echo "     envelope + <run_dir>/eval/<checkpoint>/<task>.json layout) --"
echo "     this job hand-wraps its raw --output JSON via write_result()"
echo "     instead of modifying evaluate.py itself (out of this job's"
echo "     scope / file ownership)."
date -Is
echo "=== END ==="
