#!/bin/bash
#SBATCH --job-name=koopman-golden-curve
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
# Capture a golden loss curve for configs/runs/4m-golden.yaml, to be re-run and
# compared after the §6.2 TrainTask unification.
#
# WHY TWICE.
#
# The obvious version of this job runs once and commits the curve. That produces
# an artifact nobody can use, because a single curve gives you no threshold: when
# the post-refactor run differs at step 200 by 0.004, is that the refactor or is
# it Tuesday? GPU training is not bitwise reproducible even at a fixed seed --
# cuBLAS split-k reductions, atomics in the backward pass and cuDNN autotuning all
# introduce run-to-run variation on identical inputs.
#
# So this runs the SAME spec, at the SAME seed, TWICE, into two run roots, and
# reports the divergence between them. That number is the noise floor: the
# smallest curve difference that could still mean "nothing changed". Post-refactor
# comparison is only meaningful against it.
#
# If A-vs-B diverges a lot, that is itself the finding -- it would mean this setup
# cannot detect a refactor regression at all, and we would need a deterministic
# configuration (CUBLAS_WORKSPACE_CONFIG, use_deterministic_algorithms) before the
# golden is worth capturing.
#
# The run is launched through experimentation.run WITHOUT --allow-dirty on
# purpose, so the dirty-tree gate runs and the checkpoints' code_id honestly
# identifies the commit that produced them.

set -uo pipefail

export REPO_ROOT=/users/jkli/Koopman_Mamba/.claude/worktrees/jack+search-and-provenance
export SCRATCH=/scratch/m000151-pm06/jkli
export VENV=$SCRATCH/venvs/koopman-cuda
export OUT=$SCRATCH/golden-${SLURM_JOB_ID:-manual}

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
    echo "### replicate $REP  (identical spec, identical seed)"
    echo "############################################################"
    python -m experimentation.run configs/runs/4m-golden.yaml \
        --launcher local --run_root "$OUT/$REP" \
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
import re

# Reuse the search layer's own parser rather than writing a second regex: if the
# format drifts, the golden and the pruner drift together instead of silently
# disagreeing.
from experimentation.sweep.search.metrics import parse_progress

out = pathlib.Path(os.environ["OUT"])
curves = {}
for rep in ("A", "B"):
    text = (out / rep / "train.log").read_text()
    pts = parse_progress(text)
    curves[rep] = {p.step: p.loss for p in pts}
    print(f"replicate {rep}: {len(pts)} logged points, "
          f"steps {min(curves[rep], default=0)}..{max(curves[rep], default=0)}")

shared = sorted(set(curves["A"]) & set(curves["B"]))
if not shared:
    print("NO SHARED STEPS -- parsing failed or a replicate died")
    raise SystemExit(1)

deltas = [(s, abs(curves["A"][s] - curves["B"][s])) for s in shared]
worst_step, worst = max(deltas, key=lambda t: t[1])
mean = sum(d for _, d in deltas) / len(deltas)

print(f"\nshared steps: {len(shared)}")
print(f"mean |A-B|  : {mean:.6f}")
print(f"max  |A-B|  : {worst:.6f}  (at step {worst_step})")
print("\nper-step comparison:")
for s, d in deltas:
    print(f"  step {s:>5}  A {curves['A'][s]:.6f}  B {curves['B'][s]:.6f}  d {d:.6f}")

payload = {
    "spec": "configs/runs/4m-golden.yaml",
    "commit": os.popen("git rev-parse HEAD").read().strip(),
    "replicates": curves,
    "noise_floor": {"mean_abs_delta": mean, "max_abs_delta": worst,
                    "max_at_step": worst_step, "n_shared_steps": len(shared)},
    "interpretation": (
        "max_abs_delta is the largest curve difference observed between two "
        "identical runs. A post-refactor curve differing by less than this says "
        "nothing; differing by more is evidence."),
}
dest = out / "golden_curve.json"
dest.write_text(json.dumps(payload, indent=2, sort_keys=True))
print(f"\nwrote {dest}")
print("copy it to code-tests/golden_4m_curve.json to make it the committed baseline")
PY
echo "STATUS compare=$?"
echo "workdir: $OUT"
date -Is
echo "=== END ==="
