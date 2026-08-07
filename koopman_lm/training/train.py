"""
train_fast.py -- Optimized training for Koopman LM and baselines.

440M-rewrite changes (vs original):
  * Uses the dual-stream weighted dataset (dataset_weighted.MemmapPackedDataset):
    reads train.bin + weights.bin and yields per-token loss_weights.
  * Passes loss_weights into model.forward -> recall-weighted CE (up-weights
    SCROLLS answer spans). Baseline behavior is recovered when weights are all
    ones (no weights.bin) -- identical to the original mean CE.
  * Tokenizer default -> meta-llama/Llama-2-7b-hf (use NousResearch mirror if gated).
  * For model_type==koopman the SKA fast-patch is OPTIONAL now (the custom
    autograd core already avoids autograd-through-cholesky); --ska_fast to enable.

Everything else (bf16 autocast, targeted torch.compile, gradient checkpointing,
fused AdamW, DDP, checkpointing) is unchanged.
"""

import os
import sys
import math
import json
import time
import argparse
import dataclasses
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from koopman_lm.config import build_config, config_hash, CONFIG_FACTORIES
from koopman_lm.training.repro import seed_everything, enable_determinism, seed_worker
from koopman_lm.training.optim import param_groups as _param_groups
from koopman_lm.models.koopman_lm import KoopmanLM
from koopman_lm.modules.seq.mamba import Mamba2Block
from koopman_lm.modules.seq.ska_block import SKABlock, MambaSKAParallelBlock
from koopman_lm.modules.seq.attention import CausalAttentionBlock
from koopman_lm.models.baselines import (
    build_mamba_attention, build_mamba_only, build_mamba_ska_swiglu,
    build_mamba_ska_koopman,
)
from koopman_lm.training.data.dataset import MemmapPackedDataset


def enable_gradient_checkpointing(model):
    """Checkpoint sequence blocks without losing nested parallel branches."""
    layers = model.seq_layers if hasattr(model, 'seq_layers') else []
    for layer in layers:
        # Checkpoint the whole parallel adapter once rather than wrapping its
        # Mamba child and then recomputing part of the block twice.
        if isinstance(layer, (Mamba2Block, MambaSKAParallelBlock)):
            original_forward = layer.forward

            def make_ckpt_forward(orig_fn):
                def ckpt_forward(x):
                    return grad_checkpoint(orig_fn, x, use_reentrant=False)
                return ckpt_forward

            layer.forward = make_ckpt_forward(original_forward)


# _param_groups moved to koopman_lm.training.optim.param_groups (imported
# above) so table2.py / mqar_finetune.py can share the same decay policy
# without pulling in this module's transformers/DDP-heavy imports.


def _load_init_weights(model, path):
    """Weights-only warm start for continued pretraining.

    Loads a base checkpoint's ``state_dict`` into a freshly built model (before
    any ska_fast fusion / compile). This is a WARM START, not a resume: the
    optimizer, LR schedule, and step counter are all fresh, so training runs a
    new cosine schedule over the continued-pretraining token budget. Loaded with
    strict=False so a mismatched head (e.g. a different tokenizer vocab) surfaces
    as a clear error rather than a raw RuntimeError, and benign missing/unexpected
    keys are reported instead of aborting.
    """
    if not os.path.exists(path):
        raise SystemExit(f"--init_from checkpoint not found: {path}")
    state = torch.load(path, map_location="cpu", weights_only=True)
    # _save_checkpoint writes a raw state_dict; tolerate a {'state_dict': ...} wrap.
    if isinstance(state, dict) and "state_dict" in state and \
            not any(k.endswith(".weight") for k in state):
        state = state["state_dict"]
    model_sd = model.state_dict()
    mismatched = [(k, tuple(v.shape), tuple(model_sd[k].shape))
                  for k, v in state.items()
                  if k in model_sd and tuple(v.shape) != tuple(model_sd[k].shape)]
    if mismatched:
        lines = "\n".join(f"    {k}: ckpt{a} vs model{b}"
                          for k, a, b in mismatched[:10])
        raise SystemExit(
            "--init_from shape mismatch (wrong --model_size, or a tokenizer with "
            "a different vocab than the base checkpoint?):\n" + lines)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"  Warm-started (weights only) from {path}")
    if missing:
        print(f"    [init_from] {len(missing)} missing keys kept at init, "
              f"e.g. {list(missing)[:3]}")
    if unexpected:
        print(f"    [init_from] {len(unexpected)} unexpected keys ignored, "
              f"e.g. {list(unexpected)[:3]}")


def build_model(args, tokenizer):
    # frozen config: derive runtime fields via dataclasses.replace, not assignment
    cfg = build_config(args.model_size)
    cfg = dataclasses.replace(
        cfg, vocab_size=len(tokenizer), max_seq_len=args.max_seq_len)

    n_ska = len(cfg.ska_layer_indices)
    n_mamba = cfg.n_layers if cfg.ska_mode == 'parallel' else cfg.n_layers - n_ska

    if args.model_type == "koopman":
        relation = "parallel adapters" if cfg.ska_mode == 'parallel' else "replacement layers"
        print(f"Building Koopman LM ({args.model_size}): {n_mamba} Mamba-2 + "
              f"{n_ska} SKA {relation}")
        model = KoopmanLM(cfg)
    elif args.model_type == "mamba_attn":
        model = build_mamba_attention(cfg)
    elif args.model_type == "mamba_only":
        model = build_mamba_only(cfg)
    elif args.model_type == "mamba_ska_swiglu":
        model = build_mamba_ska_swiglu(cfg)
    elif args.model_type == "mamba_ska_koopman":
        model = build_mamba_ska_koopman(cfg)
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")

    # Warm start BEFORE ska_fast fusion / gradient checkpointing / compile, so
    # weights load into the plain module the checkpoint was saved from.
    if getattr(args, "init_from", None):
        _load_init_weights(model, args.init_from)

    if args.model_type in {"koopman", "mamba_ska_swiglu", "mamba_ska_koopman"}:
        if args.ska_fast:
            from koopman_lm.kernels.fast import patch_ska_module
            for layer in model.modules():
                if isinstance(layer, SKABlock):
                    patch_ska_module(layer.ska)
            print("  Applied SKA fast patches (fused proj / bf16 einsums)")
        else:
            print("  SKA uses the custom autograd core (no fast-patch needed)")

    if args.gradient_checkpointing:
        enable_gradient_checkpointing(model)

    total = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total:,} ({total/1e6:.1f}M)")
    if hasattr(model, 'param_summary'):
        model.param_summary()
    return model, cfg


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
    # deterministic seeding (per-rank offset so DDP replicas differ but are reproducible)
    data_gen = seed_everything(args.seed + local_rank)
    if args.deterministic:
        enable_determinism(warn_only=True)
        if is_main:
            print("  Determinism mode ON (cudnn.benchmark off; throughput will drop)")
    else:
        torch.backends.cudnn.benchmark = True

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model, cfg = build_model(args, tokenizer)
    model = model.to(device)
    raw_model = model

    if args.compile:
        layers = raw_model.seq_layers if hasattr(raw_model, 'seq_layers') else []
        n_compiled = 0
        for i, layer in enumerate(layers):
            ska_block = (layer.ska if isinstance(layer, MambaSKAParallelBlock)
                         else layer if isinstance(layer, SKABlock) else None)
            if ska_block is not None:
                # The exact CUDA prefix scan is already a fused custom autograd
                # operator. Compiling its Python/JIT wrapper adds graph breaks and
                # can trigger redundant tracing without optimizing the kernel.
                if (getattr(ska_block.ska, "prefix_scan", False)
                        and getattr(ska_block.ska, "backend", "") in {"cuda", "cuda_prefix"}):
                    if is_main:
                        print("    SKA cuda_prefix left eager around the fused custom op")
                    continue
                try:
                    ska_block.ska = torch.compile(ska_block.ska, mode="max-autotune")
                    n_compiled += 1
                except Exception as e:
                    if is_main: print(f"    SKA compile failed: {e}")
            elif isinstance(layer, CausalAttentionBlock):
                try:
                    layers[i] = torch.compile(layer, mode="max-autotune"); n_compiled += 1
                except Exception:
                    pass
        if is_main: print(f"    Compiled {n_compiled} modules")

    if is_ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], find_unused_parameters=False)

    if not args.data_dir:
        raise SystemExit("Run pretokenize.py first and pass --data_dir (dual-stream).")
    train_ds = MemmapPackedDataset(args.data_dir, args.max_seq_len, seed=args.seed)
    if is_main:
        has_w = train_ds.weights is not None
        print(f"  Data: {len(train_ds):,} samples, {train_ds.n_tokens/1e9:.2f}B tokens, "
              f"recall-weights={'on' if has_w else 'off (all ones)'}")

    sampler = None
    if is_ddp:
        sampler = torch.utils.data.distributed.DistributedSampler(
            train_ds, num_replicas=world_size, rank=local_rank, shuffle=True)
    train_loader = DataLoader(
        train_ds, batch_size=args.per_device_train_batch_size,
        shuffle=(sampler is None), sampler=sampler, num_workers=args.num_workers,
        pin_memory=True, prefetch_factor=2 if args.num_workers > 0 else None,
        drop_last=True, persistent_workers=args.num_workers > 0,
        generator=data_gen, worker_init_fn=seed_worker)

    optimizer = torch.optim.AdamW(
        _param_groups(raw_model, args.weight_decay),
        lr=args.learning_rate, betas=(0.9, 0.95),
        weight_decay=args.weight_decay, fused=True)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_steps)
    autocast_ctx = torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=args.bf16)

    if is_main and args.wandb_project:
        import wandb
        ch = config_hash(cfg)
        # group runs by experiment so seeds/replicas of one config cluster
        # together, e.g. "p2a-rank-440m-c64c424d" (see scaling plan Phase 0).
        group = args.wandb_group or f"{args.phase_tag}-{args.model_size}-{ch[:8]}"
        wandb.init(project=args.wandb_project,
                   name=f"{args.model_type}-{args.model_size}-seed{args.seed}",
                   group=group,
                   config={**vars(args), "cfg_hash": ch})

    model.train()
    step = 0; micro_step = 0
    running_loss = torch.tensor(0.0, device=device); loss_count = 0
    t_start = time.time(); tokens_seen = 0
    if is_main:
        eff = args.per_device_train_batch_size * args.gradient_accumulation_steps * world_size
        print(f"\nTraining: {args.max_steps} steps, eff_batch={eff}, "
              f"tok/step={eff*args.max_seq_len:,}")
    optimizer.zero_grad(set_to_none=True)

    epoch = 0
    while step < args.max_steps:
        if sampler is not None: sampler.set_epoch(epoch)
        if hasattr(train_ds, 'set_epoch'): train_ds.set_epoch(epoch)
        for batch in train_loader:
            if step >= args.max_steps: break
            ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            lw = batch.get("loss_weights")
            if lw is not None: lw = lw.to(device, non_blocking=True)
            with autocast_ctx:
                outputs = model(input_ids=ids, labels=labels, loss_weights=lw)
                raw_loss = outputs["loss"]
                scaled_loss = raw_loss / args.gradient_accumulation_steps
            scaled_loss.backward()
            running_loss += raw_loss.detach(); loss_count += 1
            tokens_seen += ids.numel(); micro_step += 1
            if micro_step % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step(); scheduler.step()
                optimizer.zero_grad(set_to_none=True); step += 1
                if is_main and step % args.logging_steps == 0:
                    avg = running_loss.item() / max(loss_count, 1)
                    lr = optimizer.param_groups[0]["lr"]; el = time.time() - t_start
                    tps = tokens_seen / el; ppl = math.exp(min(avg, 20))
                    print(f"step {step:>6d}/{args.max_steps} | loss {avg:.4f} | "
                          f"ppl {ppl:.1f} | lr {lr:.2e} | {tps/1e3:.1f}K tok/s")
                    if args.wandb_project:
                        import wandb
                        wandb.log({"loss": avg, "ppl": ppl, "lr": lr,
                                   "tokens_per_sec": tps}, step=step)
                    running_loss = torch.tensor(0.0, device=device); loss_count = 0
                if is_main and step > 0 and step % args.save_steps == 0:
                    _save_checkpoint(raw_model, cfg, tokenizer, step, args)
        epoch += 1

    if is_main:
        _save_checkpoint(raw_model, cfg, tokenizer, step, args, dirname="final")
        print(f"\nDone in {(time.time()-t_start)/3600:.1f}h, {tokens_seen/1e9:.2f}B tokens")
    if is_ddp:
        torch.distributed.destroy_process_group()


def checkpoint_meta(cfg, step, model_type, model_size):
    """Metadata dict saved alongside every checkpoint.

    Includes the content-addressed ``cfg_hash`` so a checkpoint can always be
    traced back to its exact config (scaling plan Phase 0, reproducibility infra).
    """
    return {
        "step": step,
        "cfg": cfg,
        "cfg_hash": config_hash(cfg),
        "model_type": model_type,
        "model_size": model_size,
        "torch_version": torch.__version__,
    }


def _save_checkpoint(model, cfg, tokenizer, step, args, dirname=None):
    dirname = dirname or f"step_{step}"
    ckpt_dir = os.path.join(args.output_dir, dirname)
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(ckpt_dir, "model.pt"))
    torch.save(checkpoint_meta(cfg, step, args.model_type, args.model_size),
               os.path.join(ckpt_dir, "meta.pt"))
    tokenizer.save_pretrained(ckpt_dir)
    print(f"  Saved checkpoint to {ckpt_dir}  (cfg_hash={config_hash(cfg)[:8]})")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_type", type=str, default="koopman",
                   choices=["koopman", "mamba_attn", "mamba_only",
                            "mamba_ska_swiglu", "mamba_ska_koopman"])
    p.add_argument("--model_size", type=str, default="50m_prefix_scan",
                   help="model size name (50m_prefix_scan or 180m_prefix_scan) "
                        "or path to a custom YAML config")
    p.add_argument("--init_from", type=str, default=None,
                   help="path to a base checkpoint model.pt for a WEIGHTS-ONLY "
                        "warm start (continued pretraining). Loads weights only; "
                        "optimizer/schedule/step are fresh (new cosine over "
                        "--max_steps). Must match --model_size architecture and "
                        "the tokenizer vocab of the base checkpoint.")
    p.add_argument("--data_dir", type=str, default=None)
    p.add_argument("--tokenizer", type=str, default="meta-llama/Llama-2-7b-hf")
    p.add_argument("--max_seq_len", type=int, default=2048)
    p.add_argument("--per_device_train_batch_size", type=int, default=8)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--max_steps", type=int, default=100000)
    p.add_argument("--learning_rate", type=float, default=6e-4)
    p.add_argument("--warmup_steps", type=int, default=2000)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--no_bf16", action="store_false", dest="bf16")
    p.add_argument("--compile", action="store_true", default=True)
    p.add_argument("--no_compile", action="store_false", dest="compile")
    p.add_argument("--gradient_checkpointing", action="store_true", default=True)
    p.add_argument("--no_gradient_checkpointing", action="store_false", dest="gradient_checkpointing")
    p.add_argument("--ska_fast", action="store_true", default=False,
                   help="apply ska_fast fused-proj patch (optional; core is custom autograd)")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--ddp", action="store_true", default=False)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=5000)
    p.add_argument("--output_dir", type=str, default="./koopman-440m-fast")
    p.add_argument("--wandb_project", type=str, default=None)
    p.add_argument("--wandb_group", type=str, default=None,
                   help="override the wandb group (default: <phase_tag>-<model_size>-<cfg_hash[:8]>)")
    p.add_argument("--phase_tag", type=str, default="run",
                   help="experiment tag used in the default wandb group name")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--deterministic", action="store_true", default=False,
                   help="enable torch deterministic algorithms + seeded dataloader "
                        "(reproducible loss curves; lower throughput)")
    return p.parse_args()


def main():
    train(parse_args())


if __name__ == "__main__":
    main()
