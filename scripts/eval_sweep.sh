#!/bin/bash
# =============================================================================
# eval_sweep.sh -- rank every checkpoint in a run by ZERO-SHOT BENCHMARKS, so
# you can pick the best checkpoint by capability rather than by perplexity.
#
#   scripts/eval_sweep.sh runs/echo-180m_v2                 # sweep all step_*/final
#   TASKS=mmlu,arc_easy,piqa scripts/eval_sweep.sh runs/echo-180m_v2-cpt
#
# Motivation (see the continued-pretraining plan): FineWeb-Edu runs often keep
# improving val PPL while zero-shot regresses, so the LOWEST-PPL checkpoint is
# frequently NOT the best one to continue from or ship. This evaluates each
# saved checkpoint on the standard suite and writes one JSON per checkpoint;
# read them to choose the checkpoint that maximizes benchmark accuracy.
#
# Needs the eval extra:  pip install -e '.[lmharness]'   (or uv sync --extra lmharness)
# =============================================================================
set -eo pipefail

RUN_DIR="${1:?usage: scripts/eval_sweep.sh <run_dir> [model_size]}"
SIZE="${2:-180m_v2}"

: "${TASKS:=mmlu,arc_easy,arc_challenge,piqa,hellaswag,winogrande}"
: "${TOKENIZER:=NousResearch/Llama-2-7b-hf}"
: "${MAX_LENGTH:=2048}"
: "${BATCH_SIZE:=8}"
: "${DEVICE:=cuda}"
: "${OUT_DIR:=$RUN_DIR/eval_sweep}"
mkdir -p "$OUT_DIR"

# Collect step_*/ and final/ checkpoints, ordered by step number.
mapfile -t CKPTS < <(ls -d "$RUN_DIR"/step_* "$RUN_DIR"/final 2>/dev/null \
  | sort -t_ -k2 -n)
if [ "${#CKPTS[@]}" -eq 0 ]; then
  echo "no step_*/final checkpoints under $RUN_DIR"; exit 1
fi

echo "=== eval sweep: ${#CKPTS[@]} checkpoints in $RUN_DIR ==="
echo "  tasks: $TASKS"
for CK in "${CKPTS[@]}"; do
  MODEL_PT="$CK/model.pt"
  [ -f "$MODEL_PT" ] || { echo "  skip $CK (no model.pt)"; continue; }
  NAME="$(basename "$CK")"
  OUT="$OUT_DIR/${NAME}.json"
  if [ -f "$OUT" ]; then echo "  $NAME: cached ($OUT)"; continue; fi
  echo "--- evaluating $NAME ---"
  python -m koopman_lm.evaluation.lm_harness_eval \
    --model koopman \
    --model_args "checkpoint=$MODEL_PT,model_size=$SIZE,tokenizer=$TOKENIZER,max_length=$MAX_LENGTH" \
    --tasks "$TASKS" --batch_size "$BATCH_SIZE" --device "$DEVICE" \
    --output_path "$OUT"
done

echo ""
echo "=== per-checkpoint results written under $OUT_DIR ==="
echo "Pick the checkpoint that MAXIMIZES benchmark accuracy (not the lowest PPL)."
echo "Summarize with, e.g.:"
echo "  python - <<'PY'"
echo "  import json, glob, os"
echo "  for f in sorted(glob.glob('$OUT_DIR/*.json')):"
echo "      d = json.load(open(f)); r = d.get('results', d)"
echo "      accs = {k: v.get('acc,none', v.get('acc')) for k, v in r.items() if isinstance(v, dict)}"
echo "      print(os.path.basename(f), accs)"
echo "  PY"
