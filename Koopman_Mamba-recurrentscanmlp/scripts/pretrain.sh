#!/bin/bash
# =============================================================================
# Echo / SKA pretraining -- ONE entrypoint for the 50M, 180M, and 440M models.
#
#   scripts/pretrain.sh 50m                  # legacy/reproduction architecture
#   scripts/pretrain.sh 50m_quality                 # dense rank-24 SKA
#   scripts/pretrain.sh 50m_quality_r32             # dense rank-32 arm
#   scripts/pretrain.sh 50m_quality_sparse_diag     # two-block attribution control
#   scripts/pretrain.sh 50m_quality_mamba_control   # parameter-matched Mamba-only
#   scripts/pretrain.sh 180m_quality
#   scripts/pretrain.sh 440m
#
# Every knob below is overridable by an environment variable of the same name,
# e.g.  STEPS=1000 PDBS=8 scripts/pretrain.sh 50m   (a quick smoke run).
# See TRAINING.md for the full knob reference, install steps, and hardware notes.
#
# Runs as plain bash (single GPU) OR inside an sbatch/torchrun wrapper. For
# multi-GPU DDP, launch with torchrun and add --ddp (see TRAINING.md §Multi-GPU).
#
# NOTE ON CONVENTION: this branch's SKA statistics use the v1.1 √β symmetrization
# (PROVISIONAL, gated by the §C retrain -- see ECHO_V1_1_PLAN.md). Training here
# trains √β models. To train the pre-√β baseline, check out the parent ref
# (scripts/sqrt_beta_ablation/run_ablation.sh does exactly this A/B).
# =============================================================================
set -eo pipefail

SIZE="${1:?usage: scripts/pretrain.sh <config-name>}"

# ---- per-size defaults (provenance in TRAINING.md; all env-overridable) ----
case "$SIZE" in
  50m)          : "${TOKENS:=3000000000}";  : "${STEPS:=15000}"; : "${PDBS:=16}"; : "${GA:=6}";  : "${LR:=6e-4}"; : "${WARMUP:=300}"  ;;
  50m_quality|50m_prefix_scan|50m_quality_r32|50m_quality_chunk16|50m_quality_sparse_diag|50m_quality_sparse_diag_chunk16|50m_quality_mamba_control)  : "${TOKENS:=3000000000}";  : "${STEPS:=15000}"; : "${PDBS:=16}"; : "${GA:=6}";  : "${LR:=4e-4}"; : "${WARMUP:=300}"  ;;
  180m)         : "${TOKENS:=10000000000}"; : "${STEPS:=51000}"; : "${PDBS:=8}";  : "${GA:=12}"; : "${LR:=6e-4}"; : "${WARMUP:=1000}" ;;
  180m_quality|180m_prefix_scan|180m_quality_r32|180m_quality_chunk16|180m_quality_sparse_diag|180m_quality_sparse_diag_chunk16|180m_quality_mamba_control) : "${TOKENS:=10000000000}"; : "${STEPS:=51000}"; : "${PDBS:=8}";  : "${GA:=12}"; : "${LR:=3e-4}"; : "${WARMUP:=1000}" ;;
  440m)         : "${TOKENS:=20000000000}"; : "${STEPS:=76000}"; : "${PDBS:=4}";  : "${GA:=32}"; : "${LR:=4e-4}"; : "${WARMUP:=1000}" ;;
  *) echo "unknown size '$SIZE'"; exit 1 ;;
esac

# ---- shared knobs (env-overridable) ----
: "${SEQ_LEN:=2048}"
if [[ "$SIZE" == *_mamba_control ]]; then
  : "${MODEL_TYPE:=mamba_only}"
else
  : "${MODEL_TYPE:=koopman}"
fi
: "${WEIGHT_DECAY:=0.1}"
: "${GRAD_CLIP:=1.0}"
: "${SEED:=42}"
: "${TOKENIZER:=NousResearch/Llama-2-7b-hf}"   # ungated Llama-2 mirror (32k vocab)
: "${DATA_ROOT:=${SCRATCH:-.}/data}"
: "${RUN_ROOT:=${SCRATCH:-.}/runs}"
: "${VAL_TOKENS:=20000000}"
: "${NUM_WORKERS:=4}"
: "${NPROC:=1}"                                  # GPUs for DDP; >1 -> torchrun + --ddp
: "${EXTRA_TRAIN_ARGS:=}"                        # e.g. "--no_compile --wandb_project echo"

# All quality variants at a given scale share the exact same tokenized shard.
# This avoids both duplicate preprocessing and accidental data-order confounds.
case "$SIZE" in
  50m_quality*|50m_prefix_scan)  : "${DATA_TAG:=50m_quality}" ;;
  180m_quality*|180m_prefix_scan) : "${DATA_TAG:=180m_quality}" ;;
  *)             : "${DATA_TAG:=$SIZE}" ;;
esac
TRAIN_DIR="$DATA_ROOT/fineweb_${DATA_TAG}_train"
VAL_DIR="$DATA_ROOT/fineweb_${DATA_TAG}_val"
if [ "$MODEL_TYPE" = "koopman" ]; then
  : "${RUN_NAME:=echo-${SIZE}}"
else
  : "${RUN_NAME:=${MODEL_TYPE}-${SIZE}}"
fi
RUN_DIR="$RUN_ROOT/$RUN_NAME"
EFF_BATCH=$(( PDBS * GA ))                        # × world_size under DDP

echo "=== pretrain: $MODEL_TYPE / $SIZE ==="
echo "  tokens=$TOKENS  steps=$STEPS  seq=$SEQ_LEN  eff_batch=$EFF_BATCH (pdbs=$PDBS × ga=$GA)"
echo "  lr=$LR  warmup=$WARMUP  wd=$WEIGHT_DECAY  seed=$SEED"
echo "  tokenizer=$TOKENIZER"
echo "  data=$TRAIN_DIR   run=$RUN_DIR"

# ---- 1. tokenize train shard (pure FineWeb-Edu), skip if present ----
if [ ! -f "$TRAIN_DIR/train.bin" ]; then
  echo "--- tokenizing train shard ($TOKENS tokens) ---"
  python -m koopman_lm.training.data.pretokenize \
    --output_dir "$TRAIN_DIR" --tokenizer "$TOKENIZER" \
    --fineweb HuggingFaceFW/fineweb-edu --fineweb_subset sample-10BT \
    --mix 1.0 0.0 0.0 --max_tokens "$TOKENS" --shard_size 50000000 --seed "$SEED"
else
  echo "--- train shard exists, skipping tokenization ---"
fi

# ---- 2. tokenize disjoint held-out val shard, skip if present ----
if [ ! -f "$VAL_DIR/train.bin" ]; then
  DOCS=$(python -c "import json;print(json.load(open('$TRAIN_DIR/meta.json'))['fineweb_docs_consumed'])")
  SKIP=$(( DOCS + 10000 ))
  echo "--- tokenizing val shard (skip_docs=$SKIP) ---"
  python -m koopman_lm.training.data.pretokenize \
    --output_dir "$VAL_DIR" --tokenizer "$TOKENIZER" \
    --fineweb HuggingFaceFW/fineweb-edu --fineweb_subset sample-10BT \
    --mix 1.0 0.0 0.0 --max_tokens "$VAL_TOKENS" --skip_docs "$SKIP" --seed 1337
else
  echo "--- val shard exists, skipping tokenization ---"
fi

# ---- 3. train (single-GPU python, or torchrun DDP when NPROC>1) ----
# DDP multiplies effective batch by NPROC, so divide grad-accum to hold it fixed.
if [ "$NPROC" -gt 1 ]; then
  GA_USE=$(( GA / NPROC )); [ "$GA_USE" -lt 1 ] && GA_USE=1
  LAUNCH="torchrun --standalone --nproc_per_node=$NPROC -m koopman_lm.training.train"
  DDP_ARGS="--ddp"
  echo "--- training (DDP, $NPROC GPUs; ga $GA -> $GA_USE, eff_batch held at $((PDBS*GA_USE*NPROC))) ---"
else
  GA_USE=$GA; LAUNCH="python -m koopman_lm.training.train"; DDP_ARGS=""
  echo "--- training (single GPU) ---"
fi
$LAUNCH \
  --model_type "$MODEL_TYPE" --model_size "$SIZE" \
  --data_dir "$TRAIN_DIR" --tokenizer "$TOKENIZER" --max_seq_len "$SEQ_LEN" \
  --per_device_train_batch_size "$PDBS" --gradient_accumulation_steps "$GA_USE" \
  --max_steps "$STEPS" --learning_rate "$LR" --warmup_steps "$WARMUP" \
  --weight_decay "$WEIGHT_DECAY" --max_grad_norm "$GRAD_CLIP" \
  --bf16 --compile --num_workers "$NUM_WORKERS" \
  --logging_steps 10 --save_steps 1000 \
  --output_dir "$RUN_DIR" --seed "$SEED" $DDP_ARGS $EXTRA_TRAIN_ARGS

# ---- 4. evaluate (commands to run; eval needs the [lmharness] extra) ----
CKPT="$RUN_DIR/final/model.pt"
echo ""
echo "=== done: $CKPT ==="
echo "Evaluate with:"
echo "  python -m koopman_lm.evaluation.evaluate --checkpoint $CKPT --model_size $SIZE --mode fineweb_ppl --held_out_data_dir $VAL_DIR --output $RUN_DIR/fineweb_ppl.json"
echo "  python -m koopman_lm.evaluation.evaluate --checkpoint $CKPT --model_size $SIZE --mode ppl --output $RUN_DIR/wikitext_ppl.json"
echo "  python -m koopman_lm.evaluation.lm_harness_eval --model $MODEL_TYPE --model_args checkpoint=$CKPT,model_size=$SIZE,max_length=2048 --tasks hellaswag,piqa,arc_easy,arc_challenge,winogrande,lambada_openai --batch_size 8 --device cuda --output_path $RUN_DIR/zeroshot.json"
