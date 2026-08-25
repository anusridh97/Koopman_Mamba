#!/bin/bash
#SBATCH --job-name=koopman-beta-mqar
#SBATCH --account=marlowe-m000151-pm06
#SBATCH --partition=batch
#SBATCH --qos=medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=56
#SBATCH --mem=160G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/m000151-pm06/jkli/logs/koopman-tests/%x-%j.out
#SBATCH --error=/scratch/m000151-pm06/jkli/logs/koopman-tests/%x-%j.err
#
# The retrieval arm of the write-gate comparison: four beta policies on MQAR,
# at the SAME geometry and the SAME backend as the LM study.
#
# ======================== WHY THIS EXPERIMENT EXISTS ========================
#
# The v1.1 sqrt-beta convention's own acceptance spec
# (ACCEPTANCE_v1_1_sqrt_beta.md, which lives only on
# origin/cody-phase2a-1m-prep and is not an ancestor of HEAD) states that a
# contractivity proof is "necessary but NOT sufficient" and that the binding
# criterion is a RETRAINED retrieval + LM eval. The failure mode it singled out:
#
#     alpha == 1 AND retrieval regressed
#
# code-tests/test_ska_contractivity_contract.py now proves the first half for
# every backend. So the second half is the only open question, and it cannot be
# answered from the math side at all. It was never answered from the empirical
# side either: the harness was never merged and no verdict file exists in any
# ref.
#
# LM loss on a 25M model over 98M tokens is a weak instrument for a mechanism
# that decides what enters an associative memory. MQAR is the objective that can
# see it, and it is the objective that spec named.
#
# ===================== WHY THE PROXY GEOMETRY, NOT 1m =====================
#
# `configs/1m.yaml` is the obvious choice -- it is the Echo paper's Sec 4.1
# config and the one with a committed MQAR golden -- and it is the wrong one
# here, for two independent reasons:
#
#   1. It sets NONE of ska_prefix_scan / ska_inverse_cholesky /
#      ska_exact_intrachunk, so it runs the CHUNKED approximation. SKAModule
#      warns about that at construction, and the number in the warning is the
#      problem: ~100% relative error against a per-token-causal reference ON
#      SHORT-RANGE RECALL. MQAR *is* short-range recall. The arm would be
#      measuring the approximation rather than the operator.
#   2. It pins the LEGACY eta/gamma policy (learnable eta 1.5, gamma clamped to
#      [1.0, 1.5], no layerscale, no norm clip), so its results would not be
#      comparable to the LM study's, which runs the modern fixed-scalar policy.
#
# So this uses `configs/runs/proxy-256x17.yaml`'s model section: same 256x17
# geometry, same `exact_invchol` route, same eta/gamma policy as the LM arm.
# Only the OBJECTIVE differs between the two arms, which is the entire point --
# a difference between them is attributable to the objective and not to the
# architecture.
#
# The model section is EXTRACTED at launch into $RUN_ROOT rather than committed
# as a second file. `build_config` takes a flat model YAML and the proxy spec is
# a run spec with model/data/optim sections, so something has to bridge them;
# generating it per job follows verify_study_e2e.sbatch's precedent and cannot
# drift from the proxy config the way a committed copy would.
#
# =========================== HOW TO READ THE OUTPUT ===========================
#
# MQAR accuracy has its own noise floor and this job does NOT measure it -- one
# seed per policy. So the pilot question is answered first and separately:
#
#   * If every policy lands at ~0.0 or ~1.0, the cell is at floor or ceiling and
#     discriminates nothing. Re-run with a different (kv, gap) rather than
#     reporting the numbers.
#   * Only if accuracies land in between does a policy ordering mean anything,
#     and even then it needs replicate seeds before it is a finding. SEEDS is
#     parameterised for exactly that second pass.
#
# Writing that down here because "four numbers, one seed each, in a range" is
# precisely the shape of result that gets over-read -- and the LM arm exists in
# its 4x5 form because the same mistake was already available there.

set -uo pipefail

export REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
if [ ! -d "$REPO_ROOT/experimentation" ] || [ ! -d "$REPO_ROOT/code-tests" ]; then
    echo "FATAL: REPO_ROOT=$REPO_ROOT is not a Koopman_Mamba checkout."
    exit 1
fi
export SCRATCH=/scratch/m000151-pm06/jkli
export VENV=$SCRATCH/venvs/koopman-cuda
export RUN_ROOT=$SCRATCH/beta-mqar-${SLURM_JOB_ID:-manual}

# The cell. Defaults chosen to be mid-difficulty at this geometry; override to
# move off a floor or a ceiling.
export KV="${KV:-16}"
export GAP="${GAP:-256}"
export STEPS="${STEPS:-3000}"
export SEEDS="${SEEDS:-42}"
export TASK_VOCAB="${TASK_VOCAB:-128}"

echo "=== job ${SLURM_JOB_ID:-<interactive>} on $(hostname) ==="
date -Is
cd "$REPO_ROOT" || { echo "FATAL: no repo root"; exit 1; }
echo "commit: $(git rev-parse HEAD)"
echo "tree:   $(git status --porcelain | wc -l) modified path(s)"
echo "cell:   kv=$KV gap=$GAP steps=$STEPS seeds=$SEEDS"

module purge
module load cuda12.9/toolkit/12.9.1
module load gcc/13.1.0
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true

# shellcheck disable=SC1091
source "$VENV/bin/activate"
export PYTHONPATH=$REPO_ROOT
export HF_HOME=$SCRATCH/hf
export TRITON_CACHE_DIR=$SCRATCH/triton-cache
export TOKENIZERS_PARALLELISM=false

python -c "
import torch
assert torch.cuda.is_available(), 'no CUDA device visible to torch'
print('torch', torch.__version__, 'devices', torch.cuda.device_count())
" || { echo "FATAL: torch cannot see a GPU. Not running -- a retrieval arm with no GPU tells you nothing."; exit 1; }

mkdir -p "$RUN_ROOT" || { echo "FATAL: cannot create $RUN_ROOT"; exit 1; }

# ---------------------------------------------------------------------------
# Extract the proxy's model section into a flat model config.
# ---------------------------------------------------------------------------
export PROXY_MODEL="$RUN_ROOT/proxy-model.yaml"
python - <<'PY' || { echo "FATAL: could not extract the proxy model config"; exit 1; }
import dataclasses, os, pathlib, sys
sys.path.insert(0, os.environ["REPO_ROOT"])
import yaml
from experimentation.run.resolve import resolve_run_spec
from koopman_lm.config import config_hash, load_config

spec = resolve_run_spec(pathlib.Path(os.environ["REPO_ROOT"])
                        / "configs/runs/proxy-256x17.yaml")
out = pathlib.Path(os.environ["PROXY_MODEL"])
out.write_text(
    "# GENERATED by scripts/run_beta_policy_mqar.sbatch -- do not hand-edit.\n"
    "# The model section of configs/runs/proxy-256x17.yaml, flattened so that\n"
    "# koopman_lm.config.build_config (which takes a FLAT model config) can\n"
    "# load it. Generated per job rather than committed so it cannot drift\n"
    "# from the proxy spec the LM arm uses.\n"
    + yaml.safe_dump(dataclasses.asdict(spec.model), sort_keys=True))

# Round-trip it, so a malformed extraction fails here and not on a GPU.
reloaded = load_config(out)
assert reloaded == spec.model, "extracted model config does not round-trip"
print(f"wrote {out}")
print(f"  config_hash {config_hash(reloaded)[:8]}")
print(f"  route       prefix_scan={reloaded.ska_prefix_scan} "
      f"inverse_cholesky={reloaded.ska_inverse_cholesky} "
      f"exact_intrachunk={reloaded.ska_exact_intrachunk}")
print(f"  beta_policy {reloaded.ska_beta_policy}")
assert reloaded.ska_inverse_cholesky, (
    "the proxy spec must select the EXACT inverse-Cholesky route; a chunked "
    "backend is ~100% wrong on short-range recall, which is what MQAR measures")
PY

##############################################################################
echo "############################################################"
echo "### the four policies, one per GPU"
echo "############################################################"

POLICIES=(learned one head_scalar linear)
declare -a PIDS=()
gpu=0
for seed in $SEEDS; do
  for policy in "${POLICIES[@]}"; do
    out="$RUN_ROOT/mqar-$policy-seed$seed"
    log="$RUN_ROOT/mqar-$policy-seed$seed.log"
    echo "launch policy=$policy seed=$seed -> gpu=$gpu"
    CUDA_VISIBLE_DEVICES=$gpu python -m experimentation.experiments.mqar_finetune \
        --model_type mamba_ska_swiglu \
        --model_size "$PROXY_MODEL" \
        --ska_beta_policy "$policy" \
        --num_kv_pairs "$KV" \
        --distractor_gap "$GAP" \
        --task_vocab_size "$TASK_VOCAB" \
        --max_steps "$STEPS" \
        --eval_every $(( STEPS / 6 > 0 ? STEPS / 6 : 1 )) \
        --save_steps "$STEPS" \
        --seed "$seed" \
        --deterministic \
        --output_dir "$out" > "$log" 2>&1 &
    PIDS+=($!)
    gpu=$(( (gpu + 1) % 4 ))
    # One wave per 4 launches, so no two runs share a GPU. A shared GPU would
    # not change any accuracy (deterministic), but it would double the wall
    # clock and make an OOM look like a policy effect.
    if [ "${#PIDS[@]}" -eq 4 ]; then
        for pid in "${PIDS[@]}"; do wait "$pid" || echo "WARN: pid $pid exited nonzero"; done
        PIDS=()
    fi
  done
done
for pid in "${PIDS[@]}"; do wait "$pid" || echo "WARN: pid $pid exited nonzero"; done

##############################################################################
echo "############################################################"
echo "### results"
echo "############################################################"
python - <<'PY'
import json, os, pathlib, re

root = pathlib.Path(os.environ["RUN_ROOT"])
rows = []
for log in sorted(root.glob("mqar-*.log")):
    name = log.stem
    accs = [float(m) for m in re.findall(r"in-task accuracy \([^)]*\): ([0-9.]+)",
                                        log.read_text())]
    results = sorted(root.glob(f"{name}/step_*/mqar_result.json"))
    final = None
    if results:
        payload = json.loads(results[-1].read_text())
        intask = payload.get("mqar_intask") or {}
        if isinstance(intask, dict) and intask:
            final = max(float(v) for v in intask.values())
    rows.append((name, accs[-1] if accs else None, final, len(accs)))

print(f"{'run':>34} {'last acc':>9} {'final json':>11} {'evals':>6}")
for name, last, final, n in rows:
    print(f"{name:>34} "
          f"{('%9.4f' % last) if last is not None else '        -'} "
          f"{('%11.4f' % final) if final is not None else '          -'} "
          f"{n:>6}")

vals = [r[1] for r in rows if r[1] is not None]
print()
if not vals:
    print("VERDICT: NO ACCURACY PARSED -- check the per-run logs; this says "
          "nothing about any policy.")
elif max(vals) < 0.05:
    print(f"VERDICT: FLOOR (max acc {max(vals):.4f}). The cell is too hard at "
          f"this geometry/budget; every policy fails and the comparison "
          f"discriminates nothing. Re-run with smaller KV or GAP, or more "
          f"STEPS. Do NOT read an ordering out of these numbers.")
elif min(vals) > 0.95:
    print(f"VERDICT: CEILING (min acc {min(vals):.4f}). Every policy solves the "
          f"cell, so it cannot separate them. Re-run with larger KV or GAP.")
else:
    print(f"VERDICT: IN RANGE (acc {min(vals):.4f} - {max(vals):.4f}). The cell "
          f"discriminates. This is ONE SEED per policy and MQAR's own noise "
          f"floor is unmeasured, so re-run with SEEDS='42 43 44' before "
          f"reporting any ordering as a finding.")
print(f"\nrun_root: {root}")
PY

##############################################################################
echo "############################################################"
echo "### did the gate try and fail, or never try?"
echo "############################################################"
# Run HERE rather than as a follow-up job, because it needs the CUDA venv
# (Mamba2Block imports mamba_ssm) and the checkpoints are already on this
# node's filesystem. An accuracy table cannot distinguish "the gate separated
# facts from filler and the parameterisation threw the separation away" from
# "the gate never moved off its 0.5 init", and those call for opposite
# responses.
python "$REPO_ROOT/scripts/report_beta_contrast_on_mqar.py" \
    --kv "$KV" --gap "$GAP" --task_vocab "$TASK_VOCAB" \
    --json_out "$RUN_ROOT/beta-contrast.json" \
    "$RUN_ROOT"/mqar-*-seed* 2>&1 || echo "WARN: contrast diagnostic failed"

date -Is
echo "=== END ==="
