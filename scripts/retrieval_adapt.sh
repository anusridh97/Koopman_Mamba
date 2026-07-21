#!/bin/bash
# =============================================================================
# retrieval_adapt.sh -- Phase 2: adapt a continued-pretrained Koopman LM into a
# dense dual-encoder retriever (short contexts, InfoNCE + LM anchor).
#
#   scripts/retrieval_adapt.sh                        # defaults below
#   INIT_FROM=runs/echo-180m_v2-cpt/final/model.pt scripts/retrieval_adapt.sh
#
# Compute-cheap by design: query=64, passage=256 tokens, no long concatenated
# contexts; each query/passage encoded independently. Backbone nudged at ~8e-6,
# fresh projection head at ~1e-4. Recall@{1,5,20} every EVAL_STEPS; checkpoints
# saved alongside so scripts/eval_sweep.sh can rank them on MMLU/ARC/PIQA/... too
# (the goal: improve retrieval WITHOUT sacrificing general LM benchmarks).
#
# Needs datasets + the model deps (same env as training).
# =============================================================================
set -eo pipefail

: "${SIZE:=180m_v2}"
: "${INIT_FROM:=runs/echo-${SIZE}-cpt/final/model.pt}"     # Phase-1 output
: "${LM_DATA_DIR:=${SCRATCH:-.}/data/mix_${SIZE}_cpt_train}"  # LM-anchor corpus
: "${RUN_ROOT:=${SCRATCH:-.}/runs}"
: "${OUTPUT_DIR:=$RUN_ROOT/echo-${SIZE}-ret}"

# retrieval mix (renormalized). HotpotQA 35%, MuSiQue 20%, self-sup Wikipedia
# 45% (Wikipedia subsumes NQ evidence, which is Wikipedia-derived).
: "${SOURCES:=hotpotqa=0.35 musique=0.20 wikipedia=0.45}"

# objective / lengths (plan)
: "${Q_LEN:=64}"; : "${P_LEN:=256}"; : "${N_HARD:=2}"
: "${TEMPERATURE:=0.05}"; : "${PROJ_DIM:=768}"; : "${POOL:=mean}"

# LM anchor: 4 retrieval : 1 LM batch (schedule mode). For the combined form
# (InfoNCE + 0.1*LM every step) set LM_MODE=combined LM_ANCHOR_WEIGHT=0.1.
: "${LM_MODE:=schedule}"; : "${LM_ANCHOR_WEIGHT:=1.0}"; : "${RETRIEVAL_PER_LM:=4}"

# optimization (plan)
: "${STEPS:=5000}"; : "${WARMUP:=250}"; : "${BATCH_SIZE:=128}"
: "${BACKBONE_LR:=8e-6}"; : "${PROJ_LR:=1e-4}"; : "${GRAD_CLIP:=1.0}"
: "${EVAL_STEPS:=500}"; : "${SEED:=42}"; : "${NUM_WORKERS:=2}"
: "${TOKENIZER:=NousResearch/Llama-2-7b-hf}"
: "${EXTRA_ARGS:=}"

echo "=== retrieval adaptation: $SIZE ==="
echo "  init_from=$INIT_FROM"
echo "  sources: $SOURCES   pool=$POOL proj_dim=$PROJ_DIM"
echo "  steps=$STEPS batch=$BATCH_SIZE q=$Q_LEN p=$P_LEN K=$N_HARD tau=$TEMPERATURE"
echo "  lr(backbone=$BACKBONE_LR proj=$PROJ_LR)  lm_mode=$LM_MODE w=$LM_ANCHOR_WEIGHT"
echo "  out=$OUTPUT_DIR"

[ -f "$INIT_FROM" ] || echo "WARNING: init checkpoint $INIT_FROM not found"

LM_ARGS=""
[ -d "$LM_DATA_DIR" ] && LM_ARGS="--lm_data_dir $LM_DATA_DIR" || \
  echo "NOTE: LM anchor corpus $LM_DATA_DIR not found; running retrieval-only"

python -m koopman_lm.retrieval.adapt \
  --init_from "$INIT_FROM" --model_size "$SIZE" --tokenizer "$TOKENIZER" \
  --output_dir "$OUTPUT_DIR" --sources $SOURCES \
  --q_len "$Q_LEN" --p_len "$P_LEN" --n_hard "$N_HARD" \
  --temperature "$TEMPERATURE" --proj_dim "$PROJ_DIM" --pool "$POOL" \
  --lm_mode "$LM_MODE" --lm_anchor_weight "$LM_ANCHOR_WEIGHT" \
  --retrieval_per_lm "$RETRIEVAL_PER_LM" $LM_ARGS \
  --max_steps "$STEPS" --warmup_steps "$WARMUP" --batch_size "$BATCH_SIZE" \
  --backbone_lr "$BACKBONE_LR" --proj_lr "$PROJ_LR" --max_grad_norm "$GRAD_CLIP" \
  --eval_steps "$EVAL_STEPS" --seed "$SEED" --num_workers "$NUM_WORKERS" $EXTRA_ARGS

echo ""
echo "=== done -> $OUTPUT_DIR ==="
echo "Rank the adapted checkpoints on zero-shot LM benchmarks too (should NOT regress):"
echo "  scripts/eval_sweep.sh $OUTPUT_DIR $SIZE"
