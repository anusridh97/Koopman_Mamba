
"""
train_fast.py -- Optimized training for Koopman LM and baselines.

Supports three model variants:
  --model_type koopman     75% Mamba-2 + 25% SKA + Koopman MLP
  --model_type mamba_attn  75% Mamba-2 + 25% Flash Attention + SwiGLU MLP
  --model_type mamba_only  100% Mamba-2 + SwiGLU MLP (no global retrieval)

All three use identical data, hyperparameters, seeds, and evaluation.

Performance fixes over train.py:
  1. Pre-tokenized memmap data loading (no on-the-fly tokenization)
  2. Multi-worker DataLoader with pin_memory and prefetch
  3. Targeted torch.compile on SKA/attention modules
  4. Fused AdamW optimizer
  5. Gradient checkpointing on Mamba layers
  6. loss.item() only at logging boundaries (no per-step GPU sync)
  7. No DeepSpeed for small models — plain DDP or single-GPU
  8. BF16 autocast (no GradScaler needed for BF16)
  9. SKA optimizations — conditional Cholesky, 2 vs 6 power iterations
  10. Flash Attention 2 — for the attention baseline via F.scaled_dot_product_attention

Usage (single GPU, one model per GPU):
  CUDA_VISIBLE_DEVICES=0 python train_fast.py --model_type mamba_only  --output_dir ./mamba-only-180m-fast &
  CUDA_VISIBLE_DEVICES=1 python train_fast.py --model_type mamba_attn  --output_dir ./mamba-attn-180m-fast &
  CUDA_VISIBLE_DEVICES=2 python train_fast.py --model_type koopman     --output_dir ./koopman-180m-fast &

Or use the launch script:
  bash launch_fast.sh
"""

import os
import sys
import math
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from koopman_lm.config import config_180m, config_180m_gated, config_370m
from koopman_lm.model import KoopmanLM, Mamba2Block, SKABlock
from koopman_lm.baselines import (
    build_mamba_attention, build_mamba_only, build_mamba_ska_swiglu,
    CausalAttentionBlock,
)


# ============================================================================
# Fast memmap dataset (replaces PackedDataset)
# ============================================================================

class MemmapPackedDataset(Dataset):
    """
    Memory-mapped dataset of pre-tokenized data.
    Zero-copy, O(1) random access, multi-worker safe.

    Shuffling is handled by the DataLoader (shuffle=True or DistributedSampler).
    Per-epoch random offset varies which tokens are grouped together.
    n_samples is computed conservatively so ALL samples get the offset
    (no silent fallback for boundary indices).
    """

    def __init__(self, data_dir, max_seq_len, seed=42):
        meta_path = os.path.join(data_dir, "meta.json")
        bin_path = os.path.join(data_dir, "train.bin")

        with open(meta_path) as f:
            meta = json.load(f)

        self.n_tokens = meta["n_tokens"]
        self.max_seq_len = max_seq_len
        self.seed = seed
        self._epoch_offset = 0

        # Memory-map the token file
        self.data = np.memmap(bin_path, dtype=np.uint16, mode='r')

        # Conservative n_samples: subtract max possible offset (max_seq_len-1)
        # so that every sample can safely use any offset in [0, max_seq_len).
        usable = self.n_tokens - (max_seq_len - 1)
        self.n_samples = max(1, (usable - 1) // max_seq_len)

    def set_epoch(self, epoch):
        """Set per-epoch random offset to vary token groupings."""
        rng = np.random.RandomState(self.seed + epoch)
        self._epoch_offset = rng.randint(0, self.max_seq_len)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        start = idx * self.max_seq_len + self._epoch_offset
        end = start + self.max_seq_len + 1

        chunk = self.data[start:end].astype(np.int64)
        input_ids = torch.from_numpy(chunk[:-1])
        labels = torch.from_numpy(chunk[1:])

        return {"input_ids": input_ids, "labels": labels}


# ============================================================================
# Gradient checkpointing wrapper
# ============================================================================

def enable_gradient_checkpointing(model):
    """
    Wrap Mamba2Block forward calls with gradient checkpointing.
    Trades ~30% more compute for ~60% less activation memory.
    Works for all model types.
    """
    from koopman_lm.baselines import Mamba2Block as BaselineMamba2Block

    layers = model.seq_layers if hasattr(model, 'seq_layers') else []

    for layer in layers:
        is_mamba = isinstance(layer, (Mamba2Block, BaselineMamba2Block))
        if is_mamba:
            original_forward = layer.forward

            def make_ckpt_forward(orig_fn):
                def ckpt_forward(x):
                    return grad_checkpoint(orig_fn, x, use_reentrant=False)
                return ckpt_forward

            layer.forward = make_ckpt_forward(original_forward)


# ============================================================================
# Model builder
# ============================================================================

def build_model(args, tokenizer):
    """Build one of three model variants."""

    if args.model_size == "180m":
        cfg = config_180m()
    elif args.model_size == "180m_gated":
        cfg = config_180m_gated()
    elif args.model_size == "370m":
        cfg = config_370m()
    else:
        raise ValueError(f"Unknown model_size: {args.model_size}")

    cfg.vocab_size = len(tokenizer)
    cfg.max_seq_len = args.max_seq_len

    n_ska = len(cfg.ska_layer_indices)
    n_mamba = cfg.n_layers - n_ska

    if args.model_type == "koopman":
        print(f"Building Koopman LM ({args.model_size})...")
        print(f"  Layout: {n_mamba} Mamba-2 + {n_ska} SKA + Koopman MLP")
        model = KoopmanLM(cfg)

        from koopman_lm.ska_fast import patch_ska_module
        for layer in model.seq_layers:
            if isinstance(layer, SKABlock):
                patch_ska_module(layer.ska)
        print("  Applied SKA fast patches (fused proj, BF16 einsums, "
              "cached spectral norm)")

    elif args.model_type == "mamba_attn":
        print(f"Building Mamba+Attention baseline ({args.model_size})...")
        print(f"  Layout: {n_mamba} Mamba-2 + {n_ska} Flash Attention "
              f"+ SwiGLU MLP")
        model = build_mamba_attention(cfg)

    elif args.model_type == "mamba_only":
        print(f"Building Mamba-only baseline ({args.model_size})...")
        print(f"  Layout: {cfg.n_layers} Mamba-2 + SwiGLU MLP "
              f"(no global retrieval)")
        model = build_mamba_only(cfg)

    else:
        raise ValueError(
            f"Unknown model_type: {args.model_type}. "
            f"Choose from: koopman, mamba_attn, mamba_only"
        )

    if args.gradient_checkpointing:
        print("  Enabling gradient checkpointing on Mamba layers...")
        enable_gradient_checkpointing(model)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    if hasattr(model, 'param_summary'):
        model.param_summary()

    return model, cfg


# ============================================================================
# Training loop
# ============================================================================

def train(args):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_ddp = args.ddp and world_size > 1

    if is_ddp:
        torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    is_main = local_rank == 0
    torch.manual_seed(args.seed + local_rank)
    torch.backends.cudnn.benchmark = True

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model, cfg = build_model(args, tokenizer)
    model = model.to(device)

    # Save raw model reference BEFORE compile/DDP for clean checkpointing
    raw_model = model

    # Targeted torch.compile: SKA and attention modules with max-autotune.
    # Avoids CUDA graph conflicts with gradient-checkpointed Mamba layers.
    if args.compile:
        if is_main:
            print("  Applying targeted torch.compile on SKA/attention modules...")
        n_compiled = 0
        layers = raw_model.seq_layers if hasattr(raw_model, 'seq_layers') else []
        for i, layer in enumerate(layers):
            if isinstance(layer, SKABlock):
                try:
                    layer.ska = torch.compile(layer.ska, mode="max-autotune")
                    n_compiled += 1
                except Exception as e:
                    if is_main:
                        print(f"    SKA compile failed: {e}")
            elif isinstance(layer, CausalAttentionBlock):
                try:
                    layers[i] = torch.compile(layer, mode="max-autotune")
                    n_compiled += 1
                except Exception:
                    pass
        if is_main:
            print(f"    Compiled {n_compiled} modules")

    if is_ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank],
            find_unused_parameters=False,
        )

    # ---- Dataset ----
    if args.data_dir:
        train_ds = MemmapPackedDataset(
            args.data_dir, args.max_seq_len, seed=args.seed)
        if is_main:
            print(f"  Loaded pre-tokenized data: {len(train_ds):,} samples "
                  f"({train_ds.n_tokens / 1e9:.2f}B tokens)")
    else:
        from datasets import load_dataset as hf_load
        from train import PackedDataset
        print("  WARNING: Using streaming dataset (slow). "
              "Run pretokenize.py first!")
        train_ds = PackedDataset(
            args.dataset_name, args.dataset_subset,
            tokenizer, args.max_seq_len, seed=args.seed)

    sampler = None
    if is_ddp and isinstance(train_ds, Dataset):
        sampler = torch.utils.data.distributed.DistributedSampler(
            train_ds, num_replicas=world_size, rank=local_rank, shuffle=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.per_device_train_batch_size,
        shuffle=(sampler is None and isinstance(train_ds, Dataset)),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=2 if args.num_workers > 0 else None,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
        fused=True,
    )

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_steps,
    )

    autocast_ctx = torch.amp.autocast(
        'cuda', dtype=torch.bfloat16, enabled=args.bf16)

    if is_main and args.wandb_project:
        import wandb
        run_name = f"{args.model_type}-{args.model_size}"
        wandb.init(project=args.wandb_project, name=run_name,
                   config=vars(args))

    # ---- Training loop ----
    model.train()
    step = 0
    micro_step = 0
    running_loss = torch.tensor(0.0, device=device)
    loss_count = 0
    t_start = time.time()
    tokens_seen = 0

    if is_main:
        eff_batch = (args.per_device_train_batch_size *
                     args.gradient_accumulation_steps * world_size)
        tokens_per_step = eff_batch * args.max_seq_len
        print(f"\nStarting training: {args.max_steps} optimizer steps")
        print(f"  Effective batch size: {eff_batch}")
        print(f"  Tokens per step: {tokens_per_step:,}")
        print(f"  Model type: {args.model_type}")
        print(f"  BF16: {args.bf16}, Compile: {args.compile}")
        print(f"  Gradient checkpointing: {args.gradient_checkpointing}")
        print(f"  Num workers: {args.num_workers}")
        print()

    optimizer.zero_grad(set_to_none=True)

    epoch = 0
    while step < args.max_steps:
        if sampler is not None:
            sampler.set_epoch(epoch)
        if hasattr(train_ds, 'set_epoch'):
            train_ds.set_epoch(epoch)

        for batch in train_loader:
            if step >= args.max_steps:
                break

            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            with autocast_ctx:
                outputs = model(input_ids=input_ids, labels=labels)
                raw_loss = outputs["loss"]
                scaled_loss = raw_loss / args.gradient_accumulation_steps

            scaled_loss.backward()

            # Accumulate UNSCALED loss on GPU (no .item() sync!)
            running_loss += raw_loss.detach()
            loss_count += 1
            tokens_seen += input_ids.numel()
            micro_step += 1

            if micro_step % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1

                if is_main and step % args.logging_steps == 0:
                    avg_loss = running_loss.item() / max(loss_count, 1)
                    lr = optimizer.param_groups[0]["lr"]
                    elapsed = time.time() - t_start
                    tps = tokens_seen / elapsed
                    ppl = math.exp(min(avg_loss, 20))

                    print(f"step {step:>6d}/{args.max_steps} | "
                          f"loss {avg_loss:.4f} | ppl {ppl:.1f} | "
                          f"lr {lr:.2e} | "
                          f"{tps/1e3:.1f}K tok/s | "
                          f"{elapsed:.0f}s")

                    if args.wandb_project:
                        import wandb
                        wandb.log({
                            "loss": avg_loss, "ppl": ppl, "lr": lr,
                            "tokens_per_sec": tps,
                            "tokens_seen": tokens_seen,
                        }, step=step)

                    running_loss = torch.tensor(0.0, device=device)
                    loss_count = 0

                if is_main and step > 0 and step % args.save_steps == 0:
                    _save_checkpoint(raw_model, cfg, tokenizer, step, args)

        epoch += 1

    if is_main:
        _save_checkpoint(raw_model, cfg, tokenizer, step, args,
                         dirname="final")
        elapsed = time.time() - t_start
        print(f"\nTraining complete in {elapsed/3600:.1f}h")
        print(f"  Total tokens: {tokens_seen:,} ({tokens_seen/1e9:.2f}B)")
        print(f"  Avg throughput: {tokens_seen/elapsed/1e3:.1f}K tok/s")

    if is_ddp:
        torch.distributed.destroy_process_group()


def _save_checkpoint(model, cfg, tokenizer, step, args, dirname=None):
    if dirname is None:
        dirname = f"step_{step}"
    ckpt_dir = os.path.join(args.output_dir, dirname)
    os.makedirs(ckpt_dir, exist_ok=True)

    # state_dict hooks in ska_fast.py automatically convert fused_proj
    # back to key_proj/query_proj/value_proj for checkpoint compatibility
    state = model.state_dict()
    torch.save(state, os.path.join(ckpt_dir, "model.pt"))
    torch.save({"step": step, "cfg": cfg, "model_type": args.model_type},
               os.path.join(ckpt_dir, "meta.pt"))
    tokenizer.save_pretrained(ckpt_dir)
    print(f"  Saved checkpoint to {ckpt_dir}")


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--model_type", type=str, default="koopman",
                   choices=["koopman", "mamba_attn", "mamba_only"])
    p.add_argument("--model_size", type=str, default="180m",
                   choices=["180m", "180m_gated", "370m"])

    p.add_argument("--data_dir", type=str, default=None)
    p.add_argument("--dataset_name", type=str,
                   default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--dataset_subset", type=str, default="sample-10BT")
    p.add_argument("--tokenizer", type=str,
                   default="mistralai/Mistral-7B-v0.1")
    p.add_argument("--max_seq_len", type=int, default=2048)

    p.add_argument("--per_device_train_batch_size", type=int, default=64)
    p.add_argument("--gradient_accumulation_steps", type=int, default=2)
    p.add_argument("--max_steps", type=int, default=100000)
    p.add_argument("--learning_rate", type=float, default=6e-4)
    p.add_argument("--warmup_steps", type=int, default=2000)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)

    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--no_bf16", action="store_false", dest="bf16")
    p.add_argument("--compile", action="store_true", default=True)
    p.add_argument("--no_compile", action="store_false", dest="compile")
    p.add_argument("--gradient_checkpointing", action="store_true",
                   default=True)
    p.add_argument("--no_gradient_checkpointing", action="store_false",
                   dest="gradient_checkpointing")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--ddp", action="store_true", default=False)

    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=5000)
    p.add_argument("--output_dir", type=str, default="./koopman-180m-fast")
    p.add_argument("--wandb_project", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
