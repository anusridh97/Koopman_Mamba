#!/bin/bash
#SBATCH --job-name=koopman-mqar-golden
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
# The SYNTHETIC-path golden, companion to capture_golden_curve.sbatch.
#
# Why a second one. The 4m golden covers the shard path only. MQAR is the path
# whose SHIFT CONVENTION differs -- aligned data with the offset applied at loss
# time, versus pre-offset data scored positionally -- and that is precisely where
# the §6.2 TrainTask unification could go wrong while still producing a
# descending loss. A shard-path golden would not notice.
#
# Sizing, all far below the defaults (which are 10000 steps at model_size 50m):
#   model_size 1m       737K params, the Table 2 scale, seconds per step
#   max_steps 400       matches the 4m golden so the two are read the same way
#   num_kv_pairs 8      the default 32 needs a long sequence; 8 fits comfortably
#   distractor_gap 64   default 1024 would make seq_len enormous for no benefit
#   eval_every 200      exercises the in-loop eval hook twice, since eval_mqar
#                       carries the same convention and must survive the refactor
#   log_every 10        40 logged points, same density as the 4m golden
#
# --deterministic is REQUIRED here, not optional. Job 439605 ran this without it:
# two runs at the same seed diverged to |A-B| = 0.53 by step 400, starting from
# 0.0002 at step 10 and separating progressively. A noise floor that large makes
# the golden useless -- it cannot distinguish a refactor from a Tuesday. The
# earlier version of this file predicted exactly this case and named the remedy.
#
# As with the 4m job, this runs TWICE at the same seed and reports the divergence.
# A single curve has no threshold, so it cannot answer "did the refactor change
# this?" -- only "here is a number". The measured A-vs-B spread IS the threshold.

set -uo pipefail

export REPO_ROOT=/users/jkli/Koopman_Mamba/.claude/worktrees/jack+search-and-provenance
export SCRATCH=/scratch/m000151-pm06/jkli
export VENV=$SCRATCH/venvs/koopman-cuda
export OUT=$SCRATCH/mqar-golden-${SLURM_JOB_ID:-manual}

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

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

for REP in A B; do
    echo "############################################################"
    echo "### replicate $REP  (identical args, identical seed)"
    echo "############################################################"
    python -m experimentation.experiments.mqar_finetune \
        --model_type mamba_ska_swiglu \
        --model_size 1m \
        --num_kv_pairs 8 \
        --distractor_gap 64 \
        --batch_size 32 \
        --max_steps 400 \
        --warmup_steps 40 \
        --log_every 10 \
        --eval_every 200 \
        --save_steps 400 \
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

# Same parser the pruner uses. Until this commit's predecessor, TRAIN_RE could
# not read this trainer's output at all -- parse_progress returned [] -- which
# is how the whole log-format defect surfaced.
from experimentation.sweep.search.metrics import parse_progress

out = pathlib.Path(os.environ["OUT"])
curves = {}
for rep in ("A", "B"):
    pts = parse_progress((out / rep / "train.log").read_text())
    curves[rep] = {p.step: p.loss for p in pts}
    print(f"replicate {rep}: {len(pts)} logged points, "
          f"steps {min(curves[rep], default=0)}..{max(curves[rep], default=0)}")

if not curves["A"]:
    print("PARSED NOTHING -- the log format regression is back, or the run died")
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
    "trainer": "experimentation.experiments.mqar_finetune",
    "path": "synthetic",
    "args": {"model_type": "mamba_ska_swiglu", "model_size": "1m",
             "num_kv_pairs": 8, "distractor_gap": 64, "batch_size": 32,
             "max_steps": 400, "warmup_steps": 40, "seed": 42,
             "deterministic": True},
    "commit": os.popen("git rev-parse HEAD").read().strip(),
    "replicates": curves,
    "noise_floor": {"mean_abs_delta": mean, "max_abs_delta": worst,
                    "max_at_step": worst_step, "n_shared_steps": len(shared)},
    "interpretation": (
        "The synthetic path's counterpart to golden_4m_curve.json. This one "
        "covers the loss convention that differs -- aligned data, offset applied "
        "at loss time -- so it is the golden that would catch a TrainTask "
        "unification collapsing the two conventions into one."),
}
dest = out / "mqar_golden_curve.json"
dest.write_text(json.dumps(payload, indent=2, sort_keys=True))
print(f"\nwrote {dest}")
PY
echo "STATUS compare=$?"
echo "workdir: $OUT"
date -Is
echo "=== END ==="
