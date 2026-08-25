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
#   STEPS      default 24000 -- AMENDMENT 1 of the pre-registration. Job 446106
#              (~4000 steps) is where the brief's "linear 3/3, learned 1/3,
#              head_scalar 0/3" comes from; job 446145 ran the same cells at
#              24000 and got linear 3/3, learned 2/3, head_scalar 1/3 (grok at
#              step 12000). The horizon was carrying most of the separation. 24000
#              matches 446145 exactly, so this arm REPLICATES it rather than
#              having to be compared across horizons.
#   SEEDS      default six, PAIRED across cells. Affordable, not a trade against
#              STEPS: 446145's own log measures 87.8K tok/s at seq_len 160 /
#              batch 64 = ~8.6 steps/s = ~47 min per 24000-step run, so one seed
#              is one ~50-minute wave of seven cells on eight GPUs.
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
    export STEPS="${STEPS:-24000}"
    export SEEDS="${SEEDS:-42 43 44 45 46 47}"
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

# `launch_cell` sets --save_steps to a QUARTER of the budget rather than to the
# budget. Two reasons, both about a job that does not finish: a censored run needs
# a checkpoint to be RESUMED from rather than restarted, and the SKA ablation
# delta -- the primary discriminator -- can only be measured on a checkpoint that
# exists. A run killed at 90% of the walltime with save_steps == STEPS
# contributes an accuracy curve and no mechanism measurement at all.
echo "save_steps: $(( STEPS / 4 )) (quarters of $STEPS)"

declare -a PIDS=()
declare -a TAGS=()
launched=0
failed=0

flush_wave () {
    local i=0
    for pid in "${PIDS[@]}"; do
        if wait "$pid"; then
            echo "  OK   ${TAGS[$i]}"
        else
            code=$?
            failed=$((failed + 1))
            echo "  FAIL ${TAGS[$i]} (exit $code) -- classify as complete /"
            echo "       censored / preempted / OOM / numerical / cancelled from"
            echo "       its own log before this appears in any table"
        fi
        i=$((i + 1))
    done
    PIDS=(); TAGS=()
}

launch_cell () {                     # $1=cell spec  $2=seed  $3=gpu  $4=steps
    local spec="$1" seed="$2" gpu="$3" steps="$4"
    local name="${spec%%:*}"
    local rest="${spec#*:}"
    local policy="${rest%%:*}"
    local ridge="${rest#*:}"
    local out="$RUN_ROOT/mqar-$name-seed$seed"
    local log="$RUN_ROOT/mqar-$name-seed$seed.log"
    local save=$(( steps / 4 > 0 ? steps / 4 : 1 ))
    local ev=$(( EVAL_EVERY < steps ? EVAL_EVERY : steps ))
    echo "  launch cell=$name policy=$policy ridge=$ridge seed=$seed gpu=$gpu"
    CUDA_VISIBLE_DEVICES=$gpu python -m experimentation.experiments.mqar_finetune \
        --model_type mamba_ska_swiglu \
        --model_size "$RUN_ROOT/model-$name.yaml" \
        --ska_beta_policy "$policy" \
        --num_kv_pairs "$KV" \
        --distractor_gap "$GAP" \
        --task_vocab_size "$TASK_VOCAB" \
        --max_steps "$steps" \
        --eval_every "$ev" \
        --save_steps "$save" \
        --seed "$seed" \
        --deterministic \
        --output_dir "$out" > "$log" 2>&1 &
    PIDS+=($!)
    TAGS+=("cell=$name seed=$seed")
    launched=$((launched + 1))
}

##############################################################################
# GPU SMOKE, IN THIS JOB. The cluster is GPU-saturated, so a separate smoke job
# costs a second multi-hour queue wait. Running 60 steps of every cell here
# proves each one constructs, trains, evaluates, checkpoints and ablates on a
# real device -- and measures throughput, so the budget below is a calculation
# rather than a hope. Costs ~2 minutes against an 8-hour job.
##############################################################################
if [ "${SKIP_SMOKE:-0}" != "1" ]; then
  echo "############################################################"
  echo "### in-job GPU smoke: 60 steps x every cell"
  echo "############################################################"
  SMOKE_ROOT="$RUN_ROOT/_smoke"
  mkdir -p "$SMOKE_ROOT"
  for f in "$RUN_ROOT"/model-*.yaml; do cp "$f" "$SMOKE_ROOT/"; done
  smoke_t0=$SECONDS
  ( RUN_ROOT="$SMOKE_ROOT"
    gpu=0
    for spec in "${CELLS[@]}"; do
        launch_cell "$spec" 42 "$gpu" 60
        gpu=$(( (gpu + 1) % NGPU ))
        if [ "${#PIDS[@]}" -eq "$NGPU" ]; then flush_wave; fi
    done
    [ "${#PIDS[@]}" -gt 0 ] && flush_wave
    exit 0
  )
  smoke_secs=$((SECONDS - smoke_t0))
  echo "smoke wall time: ${smoke_secs}s for 60 steps"
  # A gate, not a print: if any cell produced no eval, the arm would report that
  # cell as "no data" for every seed and the job would have burnt 8 hours to
  # discover it.
  python - <<'PY' || { echo "FATAL: GPU smoke gate FAILED -- not spending the budget."; exit 1; }
import os, pathlib, re, sys
root = pathlib.Path(os.environ["RUN_ROOT"]) / "_smoke"
acc = re.compile(r"\[step\s+(\d+)\]\s+in-task accuracy\s*\([^)]*\):\s*([0-9.]+)")
logs = sorted(root.glob("mqar-*.log"))
bad = []
if not logs:
    bad.append("no smoke logs at all")
for log in logs:
    text = log.read_text(errors="replace")
    evals = acc.findall(text)
    ckpts = list((root / log.stem).glob("step_*/model.pt"))
    if not evals:
        tail = "\n      ".join(text.strip().splitlines()[-6:])
        bad.append(f"{log.stem}: NO eval line. tail:\n      {tail}")
    elif not ckpts:
        bad.append(f"{log.stem}: {len(evals)} eval(s) but NO checkpoint written")
    else:
        print(f"  ok {log.stem:<40} {len(evals)} eval(s), "
              f"{len(ckpts)} ckpt, last acc {evals[-1][1]}")
if bad:
    print("\n".join(f"SMOKE FAILURE: {b}" for b in bad), file=sys.stderr)
    raise SystemExit(1)
print(f"smoke gate passed: {len(logs)} cells trained on GPU")
PY
  # Throughput -> is the real budget reachable? Reported either way; a job that
  # silently cannot finish its own design is worse than one that says so.
  python - <<PY
steps, secs, n = 60, max($smoke_secs, 1), ${#CELLS[@]}
ngpu, nseeds = $NGPU, len("$SEEDS".split())
waves_smoke = -(-n // ngpu)
per_step = secs / (steps * waves_smoke)
waves_real = -(-n // ngpu) * nseeds
est = per_step * $STEPS * waves_real
print(f"  ~{per_step:.3f}s/step/wave -> {waves_real} wave(s) x $STEPS steps "
      f"= ~{est/3600:.2f}h of training")
print(f"  (excludes the ablation + contrast passes at the end)")
PY
fi

##############################################################################
echo "############################################################"
echo "### screening: one wave per SEED, so a wave is a complete paired set"
echo "############################################################"
# Waves are per SEED rather than a flat fill of the GPUs. If the walltime runs
# out mid-job, that leaves N COMPLETE paired seeds across all seven cells rather
# than a ragged set with some cells at three seeds and others at one -- and a
# ragged set is not a paired design.
for seed in $SEEDS; do
  echo "--- seed $seed (${#CELLS[@]} cells, $NGPU GPUs) ---"
  gpu=0
  for spec in "${CELLS[@]}"; do
    launch_cell "$spec" "$seed" "$gpu" "$STEPS"
    gpu=$(( (gpu + 1) % NGPU ))
    # No two runs share a GPU. Sharing would not change any accuracy
    # (--deterministic) but it would make an OOM look like a policy effect.
    if [ "${#PIDS[@]}" -eq "$NGPU" ]; then flush_wave; fi
  done
  [ "${#PIDS[@]}" -gt 0 ] && flush_wave
  echo "--- seed $seed complete; interim report ---"
  # Reported after EVERY seed, not only at the end: if the walltime expires the
  # job has still emitted a readable, complete result for the seeds that
  # finished. `--no_ablation` here because the ablation pass is the expensive
  # part and the final report does it once over everything.
  python "$REPO_ROOT/scripts/report_beta_exponent_arm.py" "$RUN_ROOT" \
      --kv "$KV" --gap "$GAP" --task_vocab "$TASK_VOCAB" --no_ablation \
      2>&1 | tail -30 || echo "WARN: interim report failed"
done
echo "launched $launched run(s), $failed nonzero exit(s)"

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
