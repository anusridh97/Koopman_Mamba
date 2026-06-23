# Koopman LM: Optimized Training Setup

180M hybrid language models trained from scratch on 2× B200 GPUs.

## Two models, same data, same benchmark

| Model | Sequence layers | MLP layers | Global retrieval |
|---|---|---|---|
| **Koopman LM** | 22× Mamba-2 + 2× SKA | 24× Koopman MLP | SKA (Koopman operator) |
| **Mamba+Attention** | 22× Mamba-2 + 2× Flash Attention | 24× SwiGLU MLP | Standard causal attention |

Both models are 180M parameters, 768 hidden, 24 layers, trained on FineWeb-Edu
with identical hyperparameters, seeds, and evaluation suite.

## Quick start

```bash
pip install -e .

# Pre-tokenize (one-time, ~30 min for 10B tokens)
python pretokenize.py --output_dir ./tokenized_data

# Train on B200 #1 (Koopman LM):
MODEL=koopman bash launch_fast.sh

# Train on B200 #2 (Mamba+Attention baseline):
MODEL=mamba_attn bash launch_fast.sh

# Compare results after both finish:
python evaluate.py \
    --checkpoint ./koopman-180m-fast/final/model.pt \
    --checkpoint2 ./mamba-attn-180m-fast/final/model.pt \
    --output comparison_results.json
```

## Performance optimizations over original train.py

1. **Pre-tokenized memmap data** — no on-the-fly tokenization
2. **Multi-worker DataLoader** — `num_workers=4`, `pin_memory`, `prefetch_factor=4`
3. **torch.compile** — kernel fusion for Koopman MLP and norms
4. **Fused AdamW** — single CUDA kernel for optimizer step
5. **Gradient checkpointing** — on Mamba layers, ~60% less activation memory
6. **GPU-side loss accumulation** — `.item()` only at logging steps
7. **No DeepSpeed** — plain single-GPU or DDP (180M doesn't need ZeRO)
8. **BF16 autocast** — via `torch.amp.autocast`
9. **SKA optimizations** — conditional Cholesky, 2 vs 6 power iterations
10. **Flash Attention 2** — for the attention baseline via `F.scaled_dot_product_attention`

## Directory structure

```
koopman-lm-fast/
├── setup.py
├── pretokenize.py          # Step 1: pre-tokenize dataset
├── train_fast.py           # Step 2: optimized training loop
├── evaluate.py             # Step 3: unified eval for both models
├── launch_fast.sh          # Orchestrates everything
├── koopman_lm/
│   ├── __init__.py
│   ├── config.py           # Model configs (180m, 370m)
│   ├── model.py            # KoopmanLM (Mamba + SKA + Koopman MLP)
│   ├── baselines.py        # Mamba+Attention+SwiGLU (with Flash Attention)
│   ├── ska.py              # SKA module (Triton + PyTorch backends)
│   ├── ska_fast.py         # SKA performance patches
│   ├── koopman_mlp.py      # Spectral Koopman MLP
│   ├── recurrent.py        # O(1) state recurrent wrapper
│   └── adaptive_chunking.py # Overlap/decay chunk strategies
└── evals/
    └── lm_harness_eval.py  # lm-evaluation-harness wrapper
```

## Evaluation suite

Both models are evaluated on the exact same benchmarks with the same seeds:

- **Held-out PPL** — WikiText-103 test split
- **NIAH** — Needle-In-A-Haystack at 128, 256, 512, 1024, 2048, 4096 tokens
- **COPY** — Exact sequence reproduction (Ren & Li, 2024)
- **MQAR** — Multi-Query Associative Recall (You et al., 2024)
- **Inverse Match** — Reversed sequence matching (Chen et al., 2025)
- **lm-eval harness** — HellaSwag, PIQA, ARC, WinoGrande, LAMBADA

## Config variants

- `config_180m()`: 768 hidden, 24 layers, 2 SKA, Koopman MLP
- `config_180m_gated()`: same + gated Koopman MLP
- `config_370m()`: 1024 hidden, 28 layers, 3 SKA
