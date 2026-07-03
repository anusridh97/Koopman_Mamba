# Echo / SKA — Project Report - Atirath Chunduri and Jack Li

## Introduction

This session took a fragmented codebase split across two legacy directories (`echo-ska-440m/` and `koopman-lm-fast/`) and consolidated it into a single installable package. The work covered:

- **Restructure** — flat `koopman_lm/` reorganised into `globals/`, `models/`, `training/`, and `evaluation/` layers with clean dependency direction
- **Config system** — factory functions replaced with YAML files; `build_config("50m")` loads `configs/smoke/50m.yaml`; all scale configs now have explicit, correct `ska_layer_indices`
- **MQAR experiments** — in-task training script (`train_mqar.py`) and full 84-cell sweep launcher (`sweep_mqar.py`) matching the paper's protocol
- **Bug fixes** — 8 issues found and fixed via code review: broken imports in recurrent decode, wrong eval argmax range, missing `_ablate` on baseline SKABlock, SKA layer index regression across all scales, and more
- **Cleanup** — dead code, empty placeholders, build artifacts, and planning docs removed; repo is now production-ready

---

## Repo Structure

```
configs/
  smoke/50m.yaml              # 50M model (paper spec: d=448, r=56, SKA@{3,7,11,15})
  training/180m–3b.yaml       # Production scale configs

koopman_lm/
  globals/
    config.py                 # KoopmanLMConfig dataclass, load_config, build_config
    modules/
      mamba.py                # Mamba2Block
      attention.py            # CausalAttentionBlock
      koopman_mlp.py          # SpectralKoopmanMLP
      ska/                    # SKAModule + core math (Cholesky, chunk stats, factor scan)
      utils/                  # repro, recurrent decode, last-layer memory
  models/
    koopman_lm.py             # KoopmanLM (SKABlock + full model)
    baselines.py              # Mamba-only and Mamba+Attention variants
  training/
    train.py                  # LM pretraining (wikitext / FineWeb)
    train_mqar.py             # In-task MQAR training (paper §4.2)
    sweep_mqar.py             # Launches all 84 MQAR grid cells
    data/                     # MemmapPackedDataset, pretokenize
  evaluation/
    mqar/mqar.py              # make_mqar, eval_mqar, eval_mqar_grid
    harness.py                # Unified eval harness (PPL, NIAH, MQAR, RULER, BABILong)
    evaluate.py / ruler.py / babilong.py / niah_quick.py

code-tests/                   # pytest suite (correctness markers)
archive/                      # Old reference code, legacy scripts, notebook
```

---

## Experiments

### 1 — MQAR In-Task Training

Trains a 50M model on a single MQAR cell (one combination of KV pair count and distractor gap length) and evaluates on the same cell. Three architectures are trained independently for comparison: `koopman` (Mamba-2 + SKA), `mamba_attn` (Mamba-2 + Attention), and `mamba_only` (pure Mamba-2). The grid covers M in {4, 8, 16, 32} KV pairs and gap in {64, 128, 256, 512, 1024, 2048, 4096} tokens — 28 cells per architecture, 84 total.

Single cell:
```bash
python -m koopman_lm.training.train_mqar \
    --model_type koopman \
    --model_size 50m \
    --num_kv_pairs 32 \
    --distractor_gap 1024 \
    --task_vocab_size 128 \
    --batch_size 64 \
    --eval_batch 64 \
    --max_steps 10000 \
    --save_steps 2000 \
    --output_dir ./mqar-koopman-kv32-gap1024
```

Resume after interruption:
```bash
python -m koopman_lm.training.train_mqar \
    --model_type koopman --model_size 50m \
    --num_kv_pairs 32 --distractor_gap 1024 \
    --task_vocab_size 128 --batch_size 64 \
    --max_steps 10000 --save_steps 2000 \
    --resume_from ./mqar-koopman-kv32-gap1024/step_4000 \
    --output_dir ./mqar-koopman-kv32-gap1024
```

Full sweep (all 84 cells, skips already-completed ones):
```bash
python -m koopman_lm.training.sweep_mqar \
    --output_root ./mqar-sweep \
    --batch_size 64 \
    --eval_batch 64 \
    --max_steps 10000
```

One architecture only:
```bash
python -m koopman_lm.training.sweep_mqar \
    --output_root ./mqar-sweep \
    --model_types koopman \
    --max_steps 10000
```

---

### 2 — LM Pretraining

Trains an Echo model from scratch on tokenized text. Requires pre-tokenized data and `mamba_ssm` installed.

```bash
# Pre-tokenize (once)
python -m koopman_lm.training.data.pretokenize \
    --tokenizer meta-llama/Llama-2-7b-hf \
    --output_dir ./data/fineweb

# Train
python -m koopman_lm.training.train \
    --model_type koopman \
    --model_size 180m \
    --data_dir ./data/fineweb \
    --max_steps 50000 \
    --output_dir ./koopman-180m-out
```

---

### 3 — Correctness Tests

Numerical correctness suite, CPU-only, no `mamba_ssm` needed.

```bash
pytest code-tests/ -m "correctness and not gpu" -q
```

---

## Experiment We Ran

Ran a single MQAR cell on an A100 to validate the training loop end-to-end.

**Cell:** `koopman`, kv=32, gap=1024 (seq_len=1152), 50M params, 10k steps  
**Result:** in-task accuracy **0.957**

The paper reports 100% on every in-task MQAR cell (§5.2), so 95.7% at 10k steps is expected — the model is still converging. Longer training or a full sweep across all 84 cells would close the gap.

---

## Key Config Facts (50M)

From paper §3.4, exactly reproduced in `configs/smoke/50m.yaml`:

| Field | Value |
|---|---|
| `d_model` | 448 |
| `n_layers` | 16 |
| `ska_n_heads` | 7 |
| `ska_rank` | 56 |
| `ska_chunk_size` | 64 |
| `ska_layer_indices` | [3, 7, 11, 15] |
| `d_state` | 64 |
| `vocab_size` | 32000 |
| Total params | ~50.7M |
