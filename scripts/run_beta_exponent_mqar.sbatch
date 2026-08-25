#!/bin/bash
#SBATCH --job-name=koopman-beta-exponent
#SBATCH --account=marlowe-m000151-pm06
#SBATCH --partition=batch
#SBATCH --qos=medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=112
#SBATCH --mem=320G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/m000151-pm06/jkli/logs/koopman-tests/%x-%j.out
#SBATCH --error=/scratch/m000151-pm06/jkli/logs/koopman-tests/%x-%j.err
#
# The key/value exponent decomposition, on MQAR, with the ridge confound
# controlled.
#
# ===================== WHAT THIS ARM DECIDES, AND WHY HERE ====================
#
# `learned` is (sqrt-beta key, sqrt-beta value); `linear` is (beta, beta). Those
# are the DIAGONAL of a 2x2, so the earlier arm's result -- `linear` grokked 3/3
# where `learned` grokked 1/3, Fisher p = 0.045 raw and 0.182 after the x4
# best-of-four penalty -- cannot say whether the key exponent, the value
# exponent, or their interaction did it. This arm adds the off-diagonal.
#
# It runs on MQAR and not on held-out LM loss because LM loss is PROVEN blind to
# this mechanism in this repo: 18 trials across three SKA routes (jobs
# 446363-446420) cut the ablation delta from 0.024418 to 0.013693 -- destroying
# 44% of SKA's contribution -- and moved mean held-out loss from 4.39030 to
# 4.39014, i.e. DOWN, and two orders of magnitude below the 7.54e-3 floor.
#
# Everything below is fixed by `docs/beta-exponent-preregistration.md`, which was
# committed BEFORE this file and before any run of the two new cells existed.
#
# ============================ THE MANDATORY CONTROL ===========================
#
# `linear` puts beta^2 in G. At the shared beta = 0.5 init its own-weight is 0.25
# against `learned`'s 0.5, so against the proxy's fixed ska_ridge = 0.01 it is
# EXACTLY TWICE as ridge-regularised at step 0 (factor measured by
# code-tests/test_beta_exponent_decomposition.py::
# test_the_key_exponent_changes_the_effective_ridge_at_init).
#
# So a linear-vs-sqrt difference may be an effective-ridge effect with nothing to
# do with the exponent. `learned@ridge2x` and `linear@ridge0.5x` are therefore
# CELLS, not contingencies: if they converge with the exponent cells, the
# exponent story is dead and the finding is about ridge. That is reported as the
# finding.
#
# ============================== PRE-FLIGHT GATES ==============================
#
# Enforced below as `exit 1`, not as comments:
#
#   1. REFUSE THE CHUNKED ROUTE. `mqar_finetune`'s default --model_size 50m is
#      prefix_scan and table2's default 1m is CHUNKED. On a chunked route
#      `beta_proj.bias` gradient cosine is ~0.00, so a beta comparison there
#      measures a gate that receives no usable gradient. Asserted on the RESOLVED
#      ska_prefix_scan / ska_inverse_cholesky / ska_exact_intrachunk.
#   2. gamma == 1 and K == 1, so no cell silently changes amplification or
#      operator depth.
#   3. Every cell's ridge is what the cell claims, so the ridge control cannot
#      silently run at the base ridge.
#   4. Every declared policy is a real BETA_POLICIES member AND the two new cells
#      are present -- a stale checkout would otherwise run a 5-cell arm and
#      report it as 7.
#
# ================================== KNOBS ====================================
#
#   SMOKE=1    60 steps, one seed, all seven cells. The required GPU smoke test:
#              proves every cell constructs, trains, evaluates and checkpoints on
#              a real device, and measures throughput so the real budget is a
#              calculation rather than a guess.
#   STEPS      default 8000 (the pre-registered budget; the earlier arm's ~4000
#              censored most non-`linear` runs)
#   SEEDS      default "42 43 44", PAIRED across cells
#   KV / GAP   default 8 / 128 -- the cell job 446102 used, reused UNCHANGED so
#              it cannot have been chosen to favour a new policy

set -uo pipefail

export REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}"
if [ ! -d "$REPO_ROOT/experimentation" ] || [ ! -d "$REPO_ROOT/code-tests" ]; then
    echo "FATAL: REPO_ROOT=$REPO_ROOT is not a Koopman_Mamba checkout."
    exit 1
fi
export SCRATCH=/scratch/m000151-pm06/jkli
export VENV=$SCRATCH/venvs/koopman-cuda

export SMOKE="${SMOKE:-0}"
export KV="${KV:-8}"
export GAP="${GAP:-128}"
export TASK_VOCAB="${TASK_VOCAB:-128}"
if [ "$SMOKE" = "1" ]; then
    export STEPS="${STEPS:-60}"
    export SEEDS="${SEEDS:-42}"
    export EVAL_EVERY="${EVAL_EVERY:-20}"
    export RUN_ROOT="${RUN_ROOT:-$SCRATCH/beta-exponent-smoke-${SLURM_JOB_ID:-manual}}"
else
    export STEPS="${STEPS:-8000}"
    export SEEDS="${SEEDS:-42 43 44}"
    export EVAL_EVERY="${EVAL_EVERY:-250}"
    export RUN_ROOT="${RUN_ROOT:-$SCRATCH/beta-exponent-${SLURM_JOB_ID:-manual}}"
fi

# The seven cells, as "name:policy:ridge". `name` is what the report groups on,
# so the two ridge controls stay distinct rows from the policies they share a
# `policy` with -- collapsing them would average a control into its own
# treatment.
CELLS=(
  "learned:learned:0.01"
  "one:one:0.01"
  "linear:linear:0.01"
  "key_linear_value_sqrt:key_linear_value_sqrt:0.01"
  "key_sqrt_value_linear:key_sqrt_value_linear:0.01"
  "learned-ridge2x:learned:0.02"
  "linear-ridge0.5x:linear:0.005"
)

echo "=== job ${SLURM_JOB_ID:-<interactive>} on $(hostname) ==="
date -Is
cd "$REPO_ROOT" || { echo "FATAL: no repo root"; exit 1; }
echo "commit:  $(git rev-parse HEAD)"
echo "dirty:   $(git status --porcelain | wc -l) modified path(s)"
echo "smoke:   $SMOKE"
echo "cell:    kv=$KV gap=$GAP steps=$STEPS eval_every=$EVAL_EVERY"
echo "seeds:   $SEEDS"
echo "run_root:$RUN_ROOT"
printf 'cells:   %s\n' "${CELLS[@]}"

module purge
module load cuda12.9/toolkit/12.9.1
module load gcc/13.1.0
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
NGPU=$(nvidia-smi --list-gpus | wc -l)
echo "visible GPUs: $NGPU"
[ "$NGPU" -ge 1 ] || { echo "FATAL: no GPU visible"; exit 1; }

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
" || { echo "FATAL: torch cannot see a GPU."; exit 1; }

mkdir -p "$RUN_ROOT" || { echo "FATAL: cannot create $RUN_ROOT"; exit 1; }

##############################################################################
echo "############################################################"
echo "### PRE-FLIGHT GATES  (a failure here must stop the job)"
echo "############################################################"
# One python block, so a gate cannot be skipped by a `||` on the wrong line, and
# it EXITS NON-ZERO -- the whole point of a gate rather than a comment.
CELL_SPEC="$(printf '%s\n' "${CELLS[@]}")" python - <<'PY' || { echo "FATAL: pre-flight gate FAILED -- not launching."; exit 1; }
import dataclasses, os, pathlib, sys
sys.path.insert(0, os.environ["REPO_ROOT"])
import yaml
from experimentation.run.resolve import resolve_run_spec
from koopman_lm.config import BETA_POLICIES, config_hash, load_config

root = pathlib.Path(os.environ["REPO_ROOT"])
run_root = pathlib.Path(os.environ["RUN_ROOT"])
spec = resolve_run_spec(root / "configs/runs/proxy-256x17.yaml")
base = spec.model

failures = []

# ---- gate 4: the two new cells must EXIST in this checkout ------------------
for needed in ("key_linear_value_sqrt", "key_sqrt_value_linear"):
    if needed not in BETA_POLICIES:
        failures.append(
            f"{needed} is not in BETA_POLICIES -- this checkout predates the "
            f"exponent decomposition, so a 7-cell arm would silently run 5")

cells = [line.split(":") for line in
         os.environ["CELL_SPEC"].strip().splitlines() if line.strip()]

for name, policy, ridge_s in cells:
    ridge = float(ridge_s)
    if policy not in BETA_POLICIES:
        failures.append(f"cell {name}: policy {policy!r} is not a BETA_POLICIES "
                        f"member; declared set is {sorted(BETA_POLICIES)}")
        continue
    cfg = dataclasses.replace(base, ska_beta_policy=policy, ska_ridge=ridge)

    # ---- gate 1: REFUSE THE CHUNKED ROUTE ----------------------------------
    exact = (cfg.ska_prefix_scan or cfg.ska_inverse_cholesky
             or cfg.ska_exact_intrachunk)
    route = ("prefix_scan" if cfg.ska_prefix_scan else
             "inverse_cholesky" if cfg.ska_inverse_cholesky else
             "exact_intrachunk" if cfg.ska_exact_intrachunk else "CHUNKED")
    if not exact:
        failures.append(
            f"cell {name}: resolved route is CHUNKED. On a chunked route "
            f"beta_proj.bias gradient cosine is ~0.00, so this cell would "
            f"measure a gate receiving no usable gradient. REFUSED.")

    # ---- gate 2: gamma == 1, K == 1 ---------------------------------------
    if float(cfg.ska_gamma_value) != 1.0 or cfg.ska_gamma_learnable:
        failures.append(f"cell {name}: gamma is {cfg.ska_gamma_value} "
                        f"(learnable={cfg.ska_gamma_learnable}); the arm pins "
                        f"gamma = 1 so no cell can be rescued by amplification")
    if int(cfg.ska_power_K) != 1:
        failures.append(f"cell {name}: power_K is {cfg.ska_power_K}, not 1")

    # ---- gate 3: the ridge is what the cell claims ------------------------
    if float(cfg.ska_ridge) != ridge:
        failures.append(f"cell {name}: ridge resolved to {cfg.ska_ridge}, "
                        f"declared {ridge}")

    # Write the flat model YAML this cell will train from, and round-trip it so
    # a malformed extraction fails HERE and not on a GPU.
    out = run_root / f"model-{name}.yaml"
    out.write_text(
        f"# GENERATED by scripts/run_beta_exponent_mqar.sbatch for cell {name}.\n"
        "# The model section of configs/runs/proxy-256x17.yaml with this cell's\n"
        "# beta policy and ridge applied, flattened for build_config (which\n"
        "# takes a FLAT model config). Generated per job rather than committed\n"
        "# so it cannot drift from the proxy spec the LM arm uses.\n"
        + yaml.safe_dump(dataclasses.asdict(cfg), sort_keys=True))
    reloaded = load_config(out)
    if reloaded != cfg:
        failures.append(f"cell {name}: model YAML does not round-trip")
    print(f"  {name:<24} policy={policy:<22} ridge={ridge:<7} "
          f"route={route:<17} gamma={cfg.ska_gamma_value} K={cfg.ska_power_K} "
          f"hash={config_hash(cfg)[:8]}")

# Distinct config hashes, or two cells would claim one run directory.
hashes = {}
for name, policy, ridge_s in cells:
    if policy not in BETA_POLICIES:
        continue
    cfg = dataclasses.replace(base, ska_beta_policy=policy,
                              ska_ridge=float(ridge_s))
    h = config_hash(cfg)
    if h in hashes:
        failures.append(f"cells {name} and {hashes[h]} share config_hash {h[:8]}"
                        f" -- they are the same experiment")
    hashes[h] = name

if failures:
    print("\n".join(f"GATE FAILURE: {f}" for f in failures), file=sys.stderr)
    raise SystemExit(1)
print(f"\nall gates passed: {len(cells)} cells, {len(hashes)} distinct configs")
PY

##############################################################################
echo "############################################################"
echo "### training: one run per GPU, paired seeds across cells"
echo "############################################################"

declare -a PIDS=()
declare -a TAGS=()
gpu=0
launched=0

flush_wave () {
    local i=0
    for pid in "${PIDS[@]}"; do
        if wait "$pid"; then
            echo "  OK   ${TAGS[$i]}"
        else
            echo "  FAIL ${TAGS[$i]} (exit $?) -- see its log; classify as "\
                 "complete/censored/preempted/OOM/numerical before reporting"
        fi
        i=$((i + 1))
    done
    PIDS=(); TAGS=()
}

for seed in $SEEDS; do
  for spec in "${CELLS[@]}"; do
    name="${spec%%:*}"
    rest="${spec#*:}"
    policy="${rest%%:*}"
    ridge="${rest#*:}"
    out="$RUN_ROOT/mqar-$name-seed$seed"
    log="$RUN_ROOT/mqar-$name-seed$seed.log"
    model_yaml="$RUN_ROOT/model-$name.yaml"
    echo "launch cell=$name policy=$policy ridge=$ridge seed=$seed -> gpu=$gpu"
    CUDA_VISIBLE_DEVICES=$gpu python -m experimentation.experiments.mqar_finetune \
        --model_type mamba_ska_swiglu \
        --model_size "$model_yaml" \
        --ska_beta_policy "$policy" \
        --num_kv_pairs "$KV" \
        --distractor_gap "$GAP" \
        --task_vocab_size "$TASK_VOCAB" \
        --max_steps "$STEPS" \
        --eval_every "$EVAL_EVERY" \
        --save_steps "$STEPS" \
        --seed "$seed" \
        --deterministic \
        --output_dir "$out" > "$log" 2>&1 &
    PIDS+=($!)
    TAGS+=("cell=$name seed=$seed")
    launched=$((launched + 1))
    gpu=$(( (gpu + 1) % NGPU ))
    # One wave per NGPU launches, so no two runs share a GPU. Sharing would not
    # change any accuracy (--deterministic) but it would make an OOM look like a
    # policy effect, and would invalidate the throughput number SMOKE exists to
    # measure.
    if [ "${#PIDS[@]}" -eq "$NGPU" ]; then flush_wave; fi
  done
done
[ "${#PIDS[@]}" -gt 0 ] && flush_wave
echo "launched $launched run(s)"

##############################################################################
echo "############################################################"
echo "### results: predeclared statistics + the SKA ablation delta"
echo "############################################################"
# Run HERE rather than as a follow-up job: the ablation pass needs the CUDA venv
# (Mamba2Block imports mamba_ssm) and the checkpoints are already on this node's
# filesystem.
python "$REPO_ROOT/scripts/report_beta_exponent_arm.py" "$RUN_ROOT" \
    --kv "$KV" --gap "$GAP" --task_vocab "$TASK_VOCAB" \
    --json_out "$RUN_ROOT/exponent-arm.json" 2>&1 \
    || echo "WARN: report script failed -- the run logs are still under $RUN_ROOT"

##############################################################################
echo "############################################################"
echo "### did the gate try and fail, or never try?"
echo "############################################################"
# An accuracy table cannot distinguish "the gate separated facts from filler and
# the parameterisation threw the separation away" from "the gate never moved off
# its 0.5 init", and those call for opposite responses.
python "$REPO_ROOT/scripts/report_beta_contrast_on_mqar.py" \
    --kv "$KV" --gap "$GAP" --task_vocab "$TASK_VOCAB" \
    --json_out "$RUN_ROOT/beta-contrast.json" \
    "$RUN_ROOT"/mqar-*-seed* 2>&1 || echo "WARN: contrast diagnostic failed"

echo
echo "run_root: $RUN_ROOT"
date -Is
echo "=== END ==="
