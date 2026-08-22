#!/bin/bash
#SBATCH --job-name=koopman-verify-sp
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
# GPU verification for branch jack/search-and-provenance (29 commits).
#
# Every commit on that branch was verified on a CPU login node, where 28-37
# tests skip because they are GPU-marked and a real KoopmanLM cannot be
# instantiated without mamba_ssm. So the branch touches the live forward path
# (ska.py's core dtype), the trainer (train.py's autocast + checkpoint meta),
# the config schema (three new fields) and the run system (microbatch moved to
# RuntimeSpec) with ZERO GPU execution behind it. This job is what closes that.
#
# Reuses the already-built CUDA venv rather than rebuilding (~25 min, and
# rebuilding is how build 414632 and job 424523 died):
#   /scratch/m000151-pm06/jkli/venvs/koopman-cuda
#   torch 2.13.0+cu129, mamba_ssm 2.3.2.post1
# Module loads and env vars are copied from scripts/gpu_triage_verify.sbatch,
# which is the last configuration known to have worked on this cluster.
#
# Four steps, in increasing order of what they'd catch:
#   1. the previously-skipped GPU tests -- does the model still build and run
#   2. the full suite -- the 843-passing CPU number, now with nothing skipped
#   3. a REAL short training run through python -m experimentation.run -- the
#      only step that exercises train.py's amp_for wiring, the fused prefix-scan
#      kernel under the new ska_precision plumbing, and a checkpoint carrying
#      code_id/dirty
#   4. quick_eval on that checkpoint -- the metric the searcher reads, which has
#      never run against a real model
#
# Step 3 is the one that matters most: it is the first time any of this has
# trained anything.

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
LOGDIR=$SCRATCH/logs/koopman-tests
export WORKDIR=$SCRATCH/verify-sp-${SLURM_JOB_ID:-manual}

echo "=== job ${SLURM_JOB_ID:-<interactive>} on $(hostname) ==="
date -Is
mkdir -p "$LOGDIR" "$WORKDIR"
cd "$REPO_ROOT" || { echo "FATAL: no such repo root $REPO_ROOT"; exit 1; }
echo "repo: $REPO_ROOT"
git rev-parse HEAD
git status --short

module purge
module load cuda12.9/toolkit/12.9.1
module load gcc/13.1.0
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv,noheader || true

# shellcheck disable=SC1091
source "$VENV/bin/activate"
# The repo first, then optuna: the search tests skip without it, and they are
# worth running here too even though they are CPU-verified.
export PYTHONPATH=$REPO_ROOT:$OPTUNA_SITE
python -c "
import torch
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())
assert torch.cuda.is_available(), 'expected a visible CUDA device on this GPU node'
print('device', torch.cuda.get_device_name(0))
import koopman_lm, experimentation
print('koopman_lm     ->', koopman_lm.__file__)
print('experimentation ->', experimentation.__file__)
"
SANITY=$?
echo "STATUS venv_sanity=$SANITY"
if [ "$SANITY" -ne 0 ]; then
    echo "FATAL: venv does not see CUDA, or resolved the wrong tree. Aborting."
    exit 1
fi

export HF_HOME=$SCRATCH/hf
export TRITON_CACHE_DIR=$SCRATCH/triton-cache
export TORCHINDUCTOR_CACHE_DIR=$SCRATCH/inductor-cache
mkdir -p "$HF_HOME" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"
export TORCH_CUDA_ARCH_LIST=9.0
export MAX_JOBS=8
export TOKENIZERS_PARALLELISM=false

##############################################################################
echo "############################################################"
echo "### STEP 1: the tests that SKIP on a CPU box"
echo "############################################################"
# -m gpu selects exactly what the login node could never run.
python -m pytest code-tests/ -m gpu -v -ra \
    2>&1 | tee "$LOGDIR/verify-sp-gpu-marked-${SLURM_JOB_ID:-manual}.log"
STEP1=${PIPESTATUS[0]}
echo "STATUS step1_gpu_marked=$STEP1"

##############################################################################
echo "############################################################"
echo "### STEP 2: the full suite, nothing skipped for want of a GPU"
echo "############################################################"
python -m pytest code-tests/ -ra --durations=25 --continue-on-collection-errors \
    2>&1 | tee "$LOGDIR/verify-sp-full-${SLURM_JOB_ID:-manual}.log"
STEP2=${PIPESTATUS[0]}
echo "STATUS step2_full_suite=$STEP2"

##############################################################################
echo "############################################################"
echo "### STEP 3: a real short training run through the run system"
echo "############################################################"
# configs/runs/50m-first-real.yaml is the spec whose shard actually exists
# (/scratch/m000151-pm06/jkli/fineweb_small_train, 100M tokens) and whose
# max_steps is already only 300. --launcher local runs it in-process here.
#
# This is the step that exercises, together and for the first time on hardware:
#   * ska.py's self._core_dtype replacing the hardcoded .float()
#   * train.py's amp_for(cfg, ...) replacing the hardcoded bfloat16 autocast
#   * the fused cuda_prefix kernel with ska_precision=fp32
#   * checkpoint_meta writing code_id + dirty
#   * runtime.per_device_batch_size driving --per_device_train_batch_size
sed -e 's/^  max_steps: 300$/  max_steps: 30/' \
    configs/runs/50m-first-real.yaml > "$WORKDIR/smoke-30step.yaml"
echo "--- smoke spec ---"
grep -E "max_steps|per_device_batch_size|precision|ska_backend" "$WORKDIR/smoke-30step.yaml"

python -m experimentation.run "$WORKDIR/smoke-30step.yaml" \
    --launcher local --run_root "$WORKDIR/runs" --allow-dirty \
    2>&1 | tee "$LOGDIR/verify-sp-train-${SLURM_JOB_ID:-manual}.log"
STEP3=${PIPESTATUS[0]}
echo "STATUS step3_real_training=$STEP3"

echo "--- what the run produced ---"
find "$WORKDIR/runs" -maxdepth 4 -name "spec.yaml" -o -maxdepth 4 -name "model.pt" \
    -o -maxdepth 4 -name "attempts.jsonl" | sort
echo "--- does the checkpoint carry code_id and dirty? ---"
python - <<'PY'
import glob
import os
import torch

metas = sorted(glob.glob(os.path.join(os.environ["WORKDIR"], "runs", "**", "meta.pt"),
                         recursive=True))
if not metas:
    print("NO meta.pt FOUND -- step 3 did not reach a checkpoint")
    raise SystemExit(1)
meta = torch.load(metas[-1], map_location="cpu", weights_only=False)
print("meta.pt keys:", sorted(meta))
for key in ("code_id", "dirty", "cfg_hash", "step"):
    print(f"  {key} = {meta.get(key)!r}")
cfg = meta.get("cfg")
for field in ("compute_precision", "ska_precision", "mlp_precision"):
    print(f"  cfg.{field} = {getattr(cfg, field, '<missing>')!r}")
assert meta.get("code_id"), "checkpoint carries no code_id"
assert "dirty" in meta, "checkpoint carries no dirty flag"
print("OK: checkpoint is self-attributing")
PY
STEP3B=$?
echo "STATUS step3b_checkpoint_provenance=$STEP3B"

##############################################################################
echo "############################################################"
echo "### STEP 4: quick_eval against the real checkpoint"
echo "############################################################"
# The metric the searcher reads. Its arithmetic is CPU-tested against a stub;
# this is the first time it meets a real model and a real shard.
python - <<'PY'
import glob
import os

import torch

from experimentation.evaluation.evaluate import load_model
from experimentation.evaluation.quick_eval import run_quick_eval, write_quick_eval

ckpts = sorted(glob.glob(os.path.join(os.environ["WORKDIR"], "runs", "**",
                                      "final", "model.pt"), recursive=True))
if not ckpts:
    print("NO final/model.pt -- skipping step 4")
    raise SystemExit(1)
checkpoint = ckpts[-1]
print("checkpoint:", checkpoint)

device = torch.device("cuda")
# 4-tuple, and it does not move the model itself (evaluate.py:104).
model, cfg, tokenizer, model_type = load_model(checkpoint)
model = model.to(device).eval()
print("model_type:", model_type, "cfg.compute_precision:",
      getattr(cfg, "compute_precision", "<missing>"))
metrics = run_quick_eval(
    model, device,
    data_dir="/scratch/m000151-pm06/jkli/fineweb_small_val",
    max_seq_len=1024, batch_size=4, max_batches=8, ska_ablation=True)
print("metrics:", metrics)
assert metrics["full"]["n_batches"] == 8, metrics["full"]
assert metrics["full"]["loss"] > 0
if metrics["ska_ablation"].get("supported"):
    print("ska loss_delta =", metrics["ska_ablation"]["loss_delta"])
written = write_quick_eval(checkpoint, metrics)
print("wrote:", written)
assert written is not None, "quick_eval result did not land in the run directory"
print("OK: quick_eval ran on a real model and landed in the run dir")
PY
STEP4=$?
echo "STATUS step4_quick_eval=$STEP4"

##############################################################################
echo "############################################################"
echo "### FINAL SUMMARY"
echo "############################################################"
echo "step1_gpu_marked_tests        = $STEP1"
echo "step2_full_suite              = $STEP2"
echo "step3_real_training           = $STEP3"
echo "step3b_checkpoint_provenance  = $STEP3B"
echo "step4_quick_eval              = $STEP4"
echo "workdir: $WORKDIR"
date -Is
echo "=== END ==="
