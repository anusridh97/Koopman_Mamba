#!/bin/bash
# =============================================================================
# §C sqrt-beta retrain ablation -- the BINDING gate for the provisional
# sqrt-beta commit (Gated-By: SKA-C-retrain-eval). See ACCEPTANCE_v1_1_sqrt_beta.md.
#
# Residual parity (the incremental-transport oracle) is NECESSARY BUT NOT
# SUFFICIENT: sqrt-beta changes WHICH statistics are accumulated, invisible to a
# self-consistency residual. This retrain A/B is the only thing that certifies it.
#
# Stage 1 (binding): 1M transfer config, baseline (asymmetric beta + spectral
#   norm) vs v1.1 (sqrt-beta), same seed/config, trained on the paper's mixed
#   sysprompt+toolcall curriculum (table2.py), evaluated zero-shot on NIAH, plus
#   an alpha probe on the v1.1 checkpoint -> compare_verdict.py writes the verdict.
# Stage 2 (conditional, only if stage-1 retrieval holds): 50M FineWeb A/B,
#   WikiText-103 ppl -> extends the verdict with the LM branch. See run_stage2.sh.
#
# The A/B is done via TWO GIT REFS (not a runtime flag) so the only diff between
# runs is the beta convention -- the pinned v1.1 convention is not altered.
#
# REVIEW before launch: venv/module/SBATCH config below and the compute budget.
# Default refs: BASELINE = HEAD^ (pre-sqrt-beta), V11 = HEAD (sqrt-beta).
# =============================================================================
set -eo pipefail

REPO=$(git rev-parse --show-toplevel)
BASELINE_REF=${BASELINE_REF:-$(git -C "$REPO" rev-parse HEAD^)}
V11_REF=${V11_REF:-$(git -C "$REPO" rev-parse HEAD)}
OUT=${OUT:-$REPO/results/sqrt_beta_ablation}
SEED=${SEED:-42}
STEPS=${STEPS:-6000}
# Denser retrieval supervision helps the (future) input-dependent gate train;
# for the fixed-blend/baseline A/B the paper default is fine. Bump to probe it.
TOOLCALL_QUERIES=${TOOLCALL_QUERIES:-4}

mkdir -p "$OUT"
echo "baseline=$BASELINE_REF  v1.1=$V11_REF  seed=$SEED  steps=$STEPS  out=$OUT"

# --- cluster config (edit) --------------------------------------------------
# source /path/to/venv/bin/activate
# module load cuda/12.x
# ---------------------------------------------------------------------------

train_and_eval () {   # $1=ref  $2=tag
  local ref=$1 tag=$2 wt="$OUT/wt_$tag"
  echo "=== [$tag] worktree @ $ref ==="
  rm -rf "$wt"
  git -C "$REPO" worktree add --force --detach "$wt" "$ref"
  ( cd "$wt" && python -m koopman_lm.experiments.table2 \
        --model_type mamba_ska_koopman --model_size 1m \
        --curriculum batch_mixed --toolcall_queries "$TOOLCALL_QUERIES" \
        --max_steps "$STEPS" --seed "$SEED" \
        --output_dir "$OUT/train_$tag" )
  cp "$OUT/train_$tag/final/table2_results.json" "$OUT/niah_$tag.json"
}

# ---- Stage 1: 1M two-ref A/B ----
train_and_eval "$BASELINE_REF" baseline
train_and_eval "$V11_REF" v11

# alpha probe on the v1.1 checkpoint (run from the v1.1 worktree so the code
# path matches the trained model exactly)
( cd "$OUT/wt_v11" && python "$REPO/scripts/sqrt_beta_ablation/alpha_probe.py" \
      --checkpoint "$OUT/train_v11/final/model.pt" --model_size 1m \
      --out "$OUT/alpha_v11.json" )

python "$REPO/scripts/sqrt_beta_ablation/compare_verdict.py" \
    --baseline_niah "$OUT/niah_baseline.json" \
    --v11_niah "$OUT/niah_v11.json" \
    --v11_alpha "$OUT/alpha_v11.json" \
    --out "$OUT/verdict_stage1.json"

git -C "$REPO" worktree remove --force "$OUT/wt_baseline" || true
git -C "$REPO" worktree remove --force "$OUT/wt_v11" || true

NODE=$(python -c "import json;print(json.load(open('$OUT/verdict_stage1.json'))['node'])")
echo "=== Stage 1 verdict: $NODE  ($OUT/verdict_stage1.json) ==="
echo "If node==LAND_PENDING_LM, run the 50M LM stage (run_stage2.sh) before an"
echo "unconditional land. Any 'do not land' node blocks the Gated-By gate."
