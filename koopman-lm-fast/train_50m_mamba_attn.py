"""
train_50m_mamba_attn.py -- Train 50M Mamba-2 + Attention baseline.

Same training setup as the main Koopman LM (train.py):
  - Dataset:     FineWeb-Edu (sample-10BT), streaming
  - Tokenizer:   Mistral-7B-v0.1 (32K vocab)
  - Optimizer:   AdamW, betas=(0.9, 0.95), weight_decay=0.1
  - Schedule:    Cosine with warmup
  - Precision:   bf16 via DeepSpeed ZeRO-1
  - Seq length:  2048
  - Batch:       8 × 8 grad_accum × N_GPUs
  - LR:          6e-4
  - Steps:       100K (≈26B tokens)

Architecture: Variant 2 from baselines.py at 50M scale
  - 12 layers: 9 Mamba-2 + 3 causal attention (25%)
  - SwiGLU MLPs everywhere
  - Attention at layers [2, 5, 8] (evenly spaced)

Usage (single GPU):
  python train_50m_mamba_attn.py \\
      --output_dir ./mamba-attn-50m-output

Usage (multi-GPU with DeepSpeed):
  deepspeed --num_gpus=2 train_50m_mamba_attn.py \\
      --output_dir ./mamba-attn-50m-output \\
      --deepspeed ds_config.json

Usage (via launch script):
  bash launch_50m_mamba_attn.sh
"""

import os
import math
import argparse
import torch
from torch.utils.data import DataLoader, IterableDataset
from datasets import load_dataset
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

# -- Import the 50M config --
import sys
sys.path.insert(0, os.path.dirname(__file__))
from config_50m_mamba_attn import config_50m_mamba_attn

# -- Import baseline builder (Mamba-2 + causal attention + SwiGLU) --
from koopman_lm.baselines import build_mamba_attention


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_name", type=str, default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--dataset_subset", type=str, default="sample-10BT")
    p.add_argument("--tokenizer", type=str, default="mistralai/Mistral-7B-v0.1")
    p.add_argument("--max_seq_len", type=int, default=2048)
    p.add_argument("--per_device_train_batch_size", type=int, default=8)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--max_steps", type=int, default=100000)
    p.add_argument("--learning_rate", type=float, default=6e-4)
    p.add_argument("--min_lr_ratio", type=float, default=0.1)
    p.add_argument("--warmup_steps", type=int, default=2000)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=1000)
    p.add_argument("--eval_steps", type=int, default=1000)
    p.add_argument("--output_dir", type=str, default="./mamba-attn-50m-output")
    p.add_argument("--deepspeed", type=str, default=None)
    p.add_argument("--wandb_project", type=str, default="koopman-lm")
    p.add_argument("--wandb_run_name", type=str, default="mamba-attn-50m")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--local_rank", type=int, default=-1)
    return p.parse_args()


class PackedDataset(IterableDataset):
    """Stream text, tokenize, pack into fixed-length chunks."""

    def __init__(self, dataset_name, subset, tokenizer, max_len, seed=42):
        self.dataset = load_dataset(dataset_name, subset, split="train", streaming=True)
        self.dataset = self.dataset.shuffle(seed=seed, buffer_size=10000)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __iter__(self):
        buffer = []
        for example in self.dataset:
            text = example.get("text", "")
            if not text:
                continue
            tokens = self.tokenizer(text, truncation=False, add_special_tokens=False)["input_ids"]
            buffer.extend(tokens)
            buffer.append(self.tokenizer.eos_token_id)
            while len(buffer) >= self.max_len + 1:
                chunk = buffer[:self.max_len + 1]
                buffer = buffer[self.max_len:]
                input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                labels = torch.tensor(chunk[1:], dtype=torch.long)
                yield {"input_ids": input_ids, "labels": labels}


def main():
    args = parse_args()

    torch.manual_seed(args.seed)

    # ---- Build 50M config ----
    cfg = config_50m_mamba_attn()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    cfg.vocab_size = len(tokenizer)
    cfg.max_seq_len = args.max_seq_len

    # ---- Build model (Variant 2: Mamba-2 + causal attention + SwiGLU) ----
    print(f"Building 50M Mamba-2 + Attention baseline...")
    print(f"  d_model={cfg.d_model}, n_layers={cfg.n_layers}")
    print(f"  Attention layers: {cfg.ska_layer_indices} "
          f"({len(cfg.ska_layer_indices)}/{cfg.n_layers} = "
          f"{100*len(cfg.ska_layer_indices)/cfg.n_layers:.0f}%)")
    print(f"  Mamba-2 layers: {cfg.n_layers - len(cfg.ska_layer_indices)}")

    model = build_mamba_attention(cfg)

    # -- Param summary --
    total = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total:,} ({total/1e6:.1f}M)")

    # ---- Data ----
    train_ds = PackedDataset(
        args.dataset_name, args.dataset_subset,
        tokenizer, args.max_seq_len, seed=args.seed,
    )
    train_loader = DataLoader(train_ds, batch_size=args.per_device_train_batch_size)

    # ---- Optimizer + Scheduler (identical to main train.py) ----
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_steps,
    )

    # ---- DeepSpeed or plain CUDA ----
    if args.deepspeed:
        import deepspeed
        model, optimizer, _, scheduler = deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            lr_scheduler=scheduler,
            config=args.deepspeed,
        )
        device = model.device
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

    # ---- W&B ----
    if args.wandb_project and (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0):
        import wandb
        run_config = vars(args)
        run_config.update({
            "model_type": "mamba_attention",
            "d_model": cfg.d_model,
            "n_layers": cfg.n_layers,
            "n_attn_layers": len(cfg.ska_layer_indices),
            "attn_layer_indices": cfg.ska_layer_indices,
            "total_params": total,
        })
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=run_config,
        )

    # ---- Training loop (identical to main train.py) ----
    model.train()
    step = 0
    micro_step = 0
    running_loss = 0.0
    loss_samples = 0
    last_logged_step = -1
    last_saved_step = -1

    print(f"\nStarting training: {args.max_steps} optimizer steps")
    print(f"  Effective batch size: "
          f"{args.per_device_train_batch_size * args.gradient_accumulation_steps * max(1, torch.cuda.device_count())}")
    print(f"  Tokens per step: "
          f"{args.per_device_train_batch_size * args.gradient_accumulation_steps * max(1, torch.cuda.device_count()) * args.max_seq_len}")

    for batch in train_loader:
        if step >= args.max_steps:
            break

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs["loss"]

        raw_loss = loss.item()
        running_loss += raw_loss
        loss_samples += 1

        if args.deepspeed:
            model.backward(loss)
            model.step()
            if model.is_gradient_accumulation_boundary():
                step += 1
        else:
            loss = loss / args.gradient_accumulation_steps
            loss.backward()
            micro_step += 1
            if micro_step % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1

        # ---- Logging ----
        if step > 0 and step % args.logging_steps == 0 and step != last_logged_step:
            last_logged_step = step
            avg_loss = running_loss / loss_samples
            lr = optimizer.param_groups[0]["lr"]
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            if rank == 0:
                ppl = math.exp(min(avg_loss, 20))
                print(f"step {step}/{args.max_steps} | loss {avg_loss:.4f} | lr {lr:.2e} | ppl {ppl:.1f}")
                if args.wandb_project:
                    import wandb
                    wandb.log({"loss": avg_loss, "lr": lr, "ppl": ppl}, step=step)
            running_loss = 0.0
            loss_samples = 0

        # ---- Checkpointing ----
        if step > 0 and step % args.save_steps == 0 and step != last_saved_step:
            last_saved_step = step
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
            if rank == 0:
                ckpt_dir = os.path.join(args.output_dir, f"step_{step}")
                os.makedirs(ckpt_dir, exist_ok=True)
                state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
                torch.save(state, os.path.join(ckpt_dir, "model.pt"))
                torch.save({"step": step, "cfg": cfg}, os.path.join(ckpt_dir, "meta.pt"))
                print(f"Saved checkpoint to {ckpt_dir}")

    # ---- Final save ----
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    if rank == 0:
        final_dir = os.path.join(args.output_dir, "final")
        os.makedirs(final_dir, exist_ok=True)
        state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
        torch.save(state, os.path.join(final_dir, "model.pt"))
        torch.save({"step": step, "cfg": cfg}, os.path.join(final_dir, "meta.pt"))
        tokenizer.save_pretrained(final_dir)
        print(f"Training complete. Final checkpoint in {final_dir}")


if __name__ == "__main__":
    main()
