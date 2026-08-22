#!/bin/bash
#SBATCH --job-name=koopman-precision-spread
#SBATCH --account=marlowe-m000151-pm06
#SBATCH --partition=batch
#SBATCH --qos=medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/m000151-pm06/jkli/logs/koopman-tests/%x-%j.out
#SBATCH --error=/scratch/m000151-pm06/jkli/logs/koopman-tests/%x-%j.err
#
# How much does eval precision actually move the number on THIS architecture?
#
# The discussion this settles: serving precision diverging from training
# precision is legitimate when measured and recorded, and the risky direction is
# DOWNWARD (truncating parameters to a width training never saw). SKA's whitened
# core takes a Cholesky of a Gram matrix and the team made ska_precision a config
# field, which implies they believe its width matters -- so feeding that core
# already-truncated inputs is the divergence most likely to bite this model.
#
# Four regimes on ONE checkpoint, same batches, same order:
#
#   A  fp32 weights, no autocast          <- evaluate.py today; "what did it learn"
#   B  fp32 weights + bf16 autocast       <- what training did; lm_harness_eval now
#   C  bf16 weights, no autocast          <- lm_harness_eval BEFORE this branch
#   D  bf16 weights + bf16 autocast       <- the fully-truncated floor
#
# A vs B  = cost of reproducing training arithmetic instead of measuring in fp32.
# B vs C  = what the old default was silently costing (the downward divergence).
# A vs D  = the total spread.
#
# If the spread is ~1e-4 the whole design argument is academic. If it is ~0.1 in
# loss then ska_precision is load-bearing and C was quietly degrading every
# lm-eval number this repo reported.

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
export CKPT=$SCRATCH/verify-sp-436063/runs/50m-first-real.da3613d7/seed42.23f9f94d/final/model.pt
export VALDIR=$SCRATCH/fineweb_small_val

echo "=== job ${SLURM_JOB_ID:-<interactive>} on $(hostname) ==="
date -Is
cd "$REPO_ROOT" || { echo "FATAL: no repo root"; exit 1; }
git rev-parse HEAD

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

python - <<'PY'
import os

import torch

from torch.utils.data import DataLoader

from koopman_lm.precision import autocast
from experimentation.evaluation.evaluate import load_model
from experimentation.training.data.dataset import MemmapPackedDataset

ckpt = os.environ["CKPT"]
valdir = os.environ["VALDIR"]
device = torch.device("cuda")

model, cfg, tokenizer, model_type = load_model(ckpt)
print("cfg.compute_precision =", getattr(cfg, "compute_precision", "?"),
      " ska_precision =", getattr(cfg, "ska_precision", "?"),
      " mlp_precision =", getattr(cfg, "mlp_precision", "?"))


# Truncating to bf16 is DESTRUCTIVE -- casting back to fp32 yields bf16 values in
# an fp32 container, not the original weights. So keep a pristine fp32 copy on the
# CPU and restore from it before every regime, rather than relying on running the
# fp32 regimes first.
pristine = {k: v.detach().clone() for k, v in model.state_dict().items()}


def measure(weight_dtype, autocast_precision, label):
    """Token-weighted loss over a FIXED set of batches, so regimes are comparable."""
    # Widen the CONTAINER before restoring: load_state_dict copies into each
    # param's existing dtype, so loading fp32 into a bf16 model stays bf16.
    model.to(torch.float32)
    model.load_state_dict(pristine)
    m = model.to(device=device, dtype=weight_dtype).eval()
    torch.cuda.reset_peak_memory_stats()
    ctx = autocast("cuda", autocast_precision)
    total_loss, total_tok = 0.0, 0
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    with torch.no_grad():
        # Rebuilt per regime with seed=0 and shuffle=False, so all four read the
        # same batches in the same order. A delta across different token sets
        # would measure the data, not the precision.
        loader = DataLoader(MemmapPackedDataset(valdir, 1024, seed=0),
                            batch_size=4, shuffle=False)
        for i, batch in enumerate(loader):
            if i >= 16:
                break
            ids = batch["input_ids"].to(device)
            lab = batch["labels"].to(device)
            with ctx:
                loss = m(input_ids=ids, labels=lab)["loss"]
            n = lab.numel()
            total_loss += float(loss.item()) * n
            total_tok += n
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end)
    avg = total_loss / total_tok
    peak = torch.cuda.max_memory_allocated() / 2**30
    print(f"{label:46} loss {avg:.6f}   ppl {torch.tensor(avg).exp():.3f}   "
          f"{total_tok/(ms/1000):,.0f} tok/s   peak {peak:.2f} GiB")
    return avg


print()
a = measure(torch.float32, "fp32", "A  fp32 weights, no autocast")
b = measure(torch.float32, "bf16", "B  fp32 weights + bf16 autocast (as trained)")
c = measure(torch.bfloat16, "fp32", "C  bf16 weights, no autocast (OLD default)")
d = measure(torch.bfloat16, "bf16", "D  bf16 weights + bf16 autocast")

print()
print(f"A vs B  (training arithmetic vs fp32)   delta = {b - a:+.6f}")
print(f"B vs C  (what the old default cost)     delta = {c - b:+.6f}")
print(f"A vs D  (total spread)                  delta = {d - a:+.6f}")
print()
print("interpretation: |delta| ~1e-4 -> precision policy is academic here;")
print("                |delta| ~1e-1 -> ska_precision is load-bearing.")
PY
echo "STATUS exit=$?"
date -Is
echo "=== END ==="
