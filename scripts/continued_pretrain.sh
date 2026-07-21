#!/bin/bash
# =============================================================================
# Continued pretraining -- warm-start an existing Echo/Koopman checkpoint on a
# 4-bucket mix, for ~2-3B tokens.
#
#   scripts/continued_pretrain.sh
#
# Data mix (renormalized to sum 1 by pretokenize.py):
#   40%   FineWeb-Edu           (fineweb)
#   25%   code / math           (code=StarCoder 12.5% + math=OpenWebMath 12.5%)
#   20%   structured QA/reason   (cosmopedia)
#   15%   retrieval-oriented LM  (wikipedia 9% + hotpotqa 3% + musique 3%; the
#                                 same evidence corpora Phase-2 adapts on, here
#                                 as plain causal-LM text)
#
# Warm start = WEIGHTS ONLY (--init_from). The optimizer, LR schedule, and step
# counter are fresh, so this runs a NEW short-warmup cosine over the continued-
# pretraining budget at a reduced peak LR -- the standard recipe for a data-mix
# change (not a resume of the base run's schedule).
#
# Point INIT_FROM at the BEST base checkpoint by zero-shot benchmark, which is
# NOT necessarily the final one -- FineWeb-Edu often keeps improving val PPL
# while zero-shot regresses. Use scripts/eval_sweep.sh to rank base checkpoints
# first, then set INIT_FROM to the winner. Checkpoints here are saved every
# ~100M tokens (SAVE_STEPS) so you can re-run eval_sweep.sh on THIS run too and
# pick by benchmark, not perplexity.
#
# Every knob is overridable inline, e.g.
#   TOKENS=3000000000 LR=1.5e-4 scripts/continued_pretrain.sh
#   INIT_FROM=/path/to/model.pt SOURCES="fineweb=0.5 code=0.25 math=0.25" ...
# See TRAINING.md for the full knob reference and hardware notes.
# =============================================================================
set -eo pipefail

# ---- base checkpoint + target config ----
: "${SIZE:=180m_v2}"
: "${INIT_FROM:=runs/echo-180m_v2/final/model.pt}"   # base weights to warm-start

# ---- continued-pretraining budget + optimization ----
: "${TOKENS:=1500000000}"     # ~1.5B tokens (plan: 1-2B for continued pretraining)
: "${PDBS:=8}"                 # per-device batch
: "${GA:=12}"                  # grad-accum  (eff batch = PDBS*GA*world_size)
: "${SEQ_LEN:=2048}"
: "${LR:=2.5e-4}"             # ~1/2-1/3 of the base 6e-4 peak (data-mix change)
: "${WEIGHT_DECAY:=0.1}"
: "${GRAD_CLIP:=1.0}"
: "${SEED:=42}"

# ---- data mix + tokenizer ----
: "${SOURCES:=fineweb=0.40 code=0.125 math=0.125 cosmopedia=0.20 wikipedia=0.09 hotpotqa=0.03 musique=0.03}"
: "${RECALL_WEIGHT:=4}"
: "${TOKENIZER:=NousResearch/Llama-2-7b-hf}"   # must match the base checkpoint vocab
: "${SHARD_SIZE:=50000000}"

# ---- paths / runtime ----
: "${DATA_ROOT:=${SCRATCH:-.}/data}"
: "${RUN_ROOT:=${SCRATCH:-.}/runs}"
: "${NUM_WORKERS:=4}"
: "${NPROC:=1}"                                 # GPUs for DDP; >1 -> torchrun + --ddp
: "${EXTRA_TRAIN_ARGS:=}"                        # e.g. "--no_compile --wandb_project echo"

EFF_BATCH=$(( PDBS * GA ))                        # × world_size under DDP
# steps ≈ tokens / (eff_batch × seq_len); override with STEPS=... if desired.
: "${STEPS:=$(( TOKENS / (EFF_BATCH * SEQ_LEN) ))}"
# warmup ~1.5% of steps (plan: 1-2%); re-warms the fresh optimizer.
: "${WARMUP:=$(( STEPS * 15 / 1000 ))}"; [ "$WARMUP" -lt 50 ] && WARMUP=50
# checkpoint ~every 100M tokens so eval_sweep.sh can pick by benchmark, not PPL.
: "${SAVE_STEPS:=$(( 100000000 / (EFF_BATCH * SEQ_LEN) ))}"; [ "$SAVE_STEPS" -lt 1 ] && SAVE_STEPS=1

TRAIN_DIR="$DATA_ROOT/mix_${SIZE}_cpt_train"
RUN_DIR="$RUN_ROOT/echo-${SIZE}-cpt"

echo "=== continued pretrain: $SIZE ==="
echo "  init_from=$INIT_FROM"
echo "  sources: $SOURCES"
echo "  tokens=$TOKENS  steps=$STEPS  seq=$SEQ_LEN  eff_batch=$EFF_BATCH (pdbs=$PDBS × ga=$GA)"
echo "  lr=$LR  warmup=$WARMUP  wd=$WEIGHT_DECAY  seed=$SEED  save_steps=$SAVE_STEPS"
echo "  tokenizer=$TOKENIZER"
echo "  data=$TRAIN_DIR   run=$RUN_DIR"

if [ ! -f "$INIT_FROM" ]; then
  echo "WARNING: base checkpoint '$INIT_FROM' not found. Set INIT_FROM=... or"
  echo "         drop --init_from to train from scratch on the new mix."
fi

# ---- 1. tokenize the mixed corpus (skip if already present) ----
if [ ! -f "$TRAIN_DIR/train.bin" ]; then
  echo "--- tokenizing mixed corpus ($TOKENS tokens) ---"
  python -m koopman_lm.training.data.pretokenize \
    --output_dir "$TRAIN_DIR" --tokenizer "$TOKENIZER" \
    --sources $SOURCES --recall_weight "$RECALL_WEIGHT" \
    --max_tokens "$TOKENS" --shard_size "$SHARD_SIZE" --seed "$SEED"
else
  echo "--- mixed corpus exists, skipping tokenization ---"
fi

# ---- 2. train (single-GPU python, or torchrun DDP when NPROC>1) ----
if [ "$NPROC" -gt 1 ]; then
  GA_USE=$(( GA / NPROC )); [ "$GA_USE" -lt 1 ] && GA_USE=1
  LAUNCH="torchrun --standalone --nproc_per_node=$NPROC -m koopman_lm.training.train"
  DDP_ARGS="--ddp"
  echo "--- training (DDP, $NPROC GPUs; ga $GA -> $GA_USE, eff_batch held at $((PDBS*GA_USE*NPROC))) ---"
else
  GA_USE=$GA; LAUNCH="python -m koopman_lm.training.train"; DDP_ARGS=""
  echo "--- training (single GPU) ---"
fi

INIT_ARG=""
[ -f "$INIT_FROM" ] && INIT_ARG="--init_from $INIT_FROM"

$LAUNCH \
  --model_type koopman --model_size "$SIZE" $INIT_ARG \
  --data_dir "$TRAIN_DIR" --tokenizer "$TOKENIZER" --max_seq_len "$SEQ_LEN" \
  --per_device_train_batch_size "$PDBS" --gradient_accumulation_steps "$GA_USE" \
  --max_steps "$STEPS" --learning_rate "$LR" --warmup_steps "$WARMUP" \
  --weight_decay "$WEIGHT_DECAY" --max_grad_norm "$GRAD_CLIP" \
  --bf16 --compile --num_workers "$NUM_WORKERS" \
  --logging_steps 10 --save_steps "$SAVE_STEPS" \
  --output_dir "$RUN_DIR" --seed "$SEED" $DDP_ARGS $EXTRA_TRAIN_ARGS

# ---- 3. evaluate (commands to run; eval needs the [lmharness] extra) ----
CKPT="$RUN_DIR/final/model.pt"
echo ""
echo "=== done: $CKPT ==="
echo "Evaluate with:"
echo "  python -m koopman_lm.evaluation.evaluate --checkpoint $CKPT --model_size $SIZE --mode ppl --output $RUN_DIR/wikitext_ppl.json"
echo "  python -m koopman_lm.evaluation.lm_harness_eval --model koopman --model_args checkpoint=$CKPT,model_size=$SIZE,max_length=2048 --tasks hellaswag,piqa,arc_easy,arc_challenge,winogrande,lambada_openai --batch_size 8 --device cuda --output_path $RUN_DIR/zeroshot.json"
