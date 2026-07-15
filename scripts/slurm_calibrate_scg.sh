#!/bin/bash
# Phase-1 threshold calibration on an SCG H200 node.
#
# Runs the health / grad-flow / four-mode load-bearing diagnostics on reference
# models and writes RAW continuous values to JSON so Phase-2 thresholds are set
# from data. Runs inspect_checkpoint.py FIRST to confirm the completed
# checkpoints strict-load into the consolidated model.
#
# IMPORTANT (see SCG_ENV_HANDOFF.md):
#   * mkdir the absolute log dir BEFORE sbatch.
#   * The checkout this runs against MUST be on `phase1-finalize`; point it via
#     KOOPMAN_REPO so you don't disturb the shared oak checkout (180M rerun):
#       sbatch --export=ALL,KOOPMAN_REPO=/your/phase1/checkout scripts/slurm_calibrate_scg.sh

#SBATCH --job-name=phase1_calib
#SBATCH --account=mpsnyder
#SBATCH --partition=gpu_short
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --chdir=/oak/stanford/scg/lab_mpsnyder/cody1212/Koopman_Mamba
#SBATCH --output=/labs/mpsnyder/cody1212/koopman_runs/logs/phase1_calib_%j.out
#SBATCH --error=/labs/mpsnyder/cody1212/koopman_runs/logs/phase1_calib_%j.err

set -e
set +u

module load cuda/12.3.2_545.23.08_cudNN_9.0.0.312
module unload gcc/13.3.0 2>/dev/null || true
module unload gcc/11.2.0 2>/dev/null || true
module load gcc/9.2.0-centos_7

export CUDA_HOME=$(dirname "$(dirname "$(command -v nvcc)")")
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$(dirname "$(dirname "$(command -v gcc)")")/lib64:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export CC=$(command -v gcc)
export CXX=$(command -v g++)
export CUDAHOSTCXX="$CXX"
export HF_HOME=/labs/mpsnyder/cody1212/.hf_cache
export UV_CACHE_DIR=/labs/mpsnyder/cody1212/.uv_cache
export TRITON_CACHE_DIR=/labs/mpsnyder/cody1212/tmp/triton_cache
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8

VENV="${KOOPMAN_VENV:-/labs/mpsnyder/cody1212/Koopman_Mamba/koopman-lm-fast/.venv}"
source "$VENV/bin/activate"

REPO="${KOOPMAN_REPO:-/oak/stanford/scg/lab_mpsnyder/cody1212/Koopman_Mamba}"
cd "$REPO"
echo "== repo: $REPO  branch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '?') =="
for f in scripts/inspect_checkpoint.py koopman_lm/evaluation/calibrate.py; do
    [ -f "$f" ] || { echo "!! $REPO is missing $f -- this checkout is not on phase1-finalize."; exit 2; }
done

# The training venv has koopman_lm installed elsewhere; force THIS checkout onto
# PYTHONPATH, then verify by a CODE MARKER (not a path string). /oak and /labs
# are two mounts of the same dir, so a path-prefix check gives false alarms --
# check that the imported koopman_lm actually has the phase1-finalize code.
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
if ! python - <<'PY'
import os, sys
try:
    import koopman_lm
    from koopman_lm.evaluation import harness
    from koopman_lm.training import diagnostics  # noqa: F401
except Exception:
    import traceback; traceback.print_exc()
    print("  koopman_lm import FAILED (traceback above)")
    sys.exit(1)
print("  koopman_lm imports from:", os.path.dirname(koopman_lm.__file__))
if not hasattr(harness, "eval_load_bearing"):
    print("  MISSING marker harness.eval_load_bearing -- a stale/older koopman_lm"
          " is shadowing this checkout")
    sys.exit(1)
print("  verified phase1-finalize diagnostics (eval_load_bearing present)")
PY
then
    echo "!! aborting: koopman_lm is not the finalized phase1-finalize checkout (see above)"
    exit 2
fi
mkdir -p /labs/mpsnyder/cody1212/koopman_runs/logs

CKPT_50M="/labs/mpsnyder/cody1212/runs/echo-50m-fineweb-3B/final/model.pt"
OUT="/labs/mpsnyder/cody1212/results/phase1_calibration"
TOK="NousResearch/Llama-2-7b-hf"
mkdir -p "$OUT"

# BLOCKING pre-check with an HONEST verdict. inspect_checkpoint.py exits:
#   0 = strict load OK, 3 = key mismatch (needs a shim), anything else = it
#   could not run (missing file / import / unpickle error) -- NOT a load verdict.
echo "== load-compat check (50M) =="
if python scripts/inspect_checkpoint.py --checkpoint "$CKPT_50M" --model_size 50m; then
    echo "  load-compat OK -- proceeding to calibration"
else
    rc=$?
    if [ "$rc" -eq 3 ]; then
        echo "!! 50M checkpoint KEY MISMATCH: strict load fails. See the key diff above;"
        echo "   a load shim (or reconstructing the training-time cfg) is needed first."
    else
        echo "!! inspect_checkpoint.py did NOT run to a verdict (exit $rc). This is NOT a"
        echo "   load-compat result -- check the file exists in \$REPO and the env is set up"
        echo "   (env/import errors show above)."
    fi
    exit "$rc"
fi

# --data synthetic (random token ids) is network-free and robust on an offline
# compute node. If WikiText is cached under $HF_HOME, re-run with --data wikitext
# for more realistic health/grad-flow inputs.
echo "== calibrate: trained 50M =="
python -m koopman_lm.evaluation.calibrate --checkpoint "$CKPT_50M" \
    --tokenizer "$TOK" --data synthetic --out "$OUT/calib_50m_trained.json"

echo "== calibrate: untrained init (same arch as the 50M checkpoint) =="
python -m koopman_lm.evaluation.calibrate --checkpoint "$CKPT_50M" --init_only \
    --tokenizer "$TOK" --data synthetic --out "$OUT/calib_50m_untrained.json"

echo "CALIBRATION DONE. Raw values under $OUT/"
