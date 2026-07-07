"""
train_mqar.py — In-task MQAR training for KoopmanLM and baselines.

Reproduces the experimental setup from the Echo/SKA paper:
  - 50M-parameter variants of Mamba-2, Mamba-2+Attention, Mamba-2+SKA
  - KV pairs M ∈ {4, 8, 16, 32}
  - Distractor gap 64 → 4096 tokens
  - Each (model_type, M, gap) cell is trained and evaluated independently (in-task)

Seq length is derived automatically: seq_len = 4*M + distractor_gap
(2*M for the KV block + gap fillers + 2*M for the query block).

Vocab is kept small (default 512) so the task difficulty comes purely from
the retrieval distance, not from the output space size.

Checkpointing
-------------
Every --save_steps a checkpoint is written to:
    <output_dir>/step_<N>/{model.pt, optimizer.pt, meta.pt, mqar_result.json}

Resume with --resume_from <output_dir>/step_<N>.

Single-cell example
-------------------
    python -m koopman_lm.training.train_mqar \\
        --model_type koopman \\
        --model_size 50m \\
        --num_kv_pairs 32 \\
        --distractor_gap 1024 \\
        --output_dir ./mqar-koopman-m32-g1024

Sweep all cells (see koopman_lm/training/sweep_mqar.py).
"""

import os
import json
import math
import time
import argparse
import dataclasses
import shutil
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from koopman_lm.globals.config import build_config, config_hash
from koopman_lm.models.koopman_lm import KoopmanLM
from koopman_lm.models.baselines import build_mamba_only, build_mamba_attention
from koopman_lm.evaluation.mqar.mqar import make_mqar, eval_mqar

# Paper grid (Echo/SKA, Arora et al. 2024)
PAPER_KV_PAIRS     = (4, 8, 16, 32)
PAPER_GAPS         = (64, 128, 256, 512, 1024, 2048, 4096)
PAPER_MODEL_TYPES  = ("koopman", "mamba_attn", "mamba_only")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MQARDataset(Dataset):
    """On-the-fly MQAR sequences. Each index uses a unique seed."""

    def __init__(self, seq_len, num_kv_pairs, vocab_size,
                 epoch_size=10000, base_seed=0):
        self.seq_len      = seq_len
        self.num_kv_pairs = num_kv_pairs
        self.vocab_size   = vocab_size
        self.epoch_size   = epoch_size
        self.base_seed    = base_seed

    def __len__(self):
        return self.epoch_size

    def __getitem__(self, idx):
        inputs, labels = make_mqar(
            batch=1,
            seq_len=self.seq_len,
            num_kv_pairs=self.num_kv_pairs,
            vocab_size=self.vocab_size,
            seed=self.base_seed + idx,
        )
        return inputs[0], labels[0]


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_model(model_type, cfg):
    if model_type == "koopman":
        return KoopmanLM(cfg)
    elif model_type == "mamba_attn":
        return build_mamba_attention(cfg)
    elif model_type == "mamba_only":
        return build_mamba_only(cfg)
    else:
        raise ValueError(f"Unknown model_type: {model_type!r}. "
                         f"Choose from {PAPER_MODEL_TYPES}")


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _ckpt_dir(output_dir, step):
    return Path(output_dir) / f"step_{step}"


def save_checkpoint(output_dir, step, model, optimizer, scheduler, cfg,
                    model_size, model_type, cell_result=None):
    d = _ckpt_dir(output_dir, step)
    d.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), d / "model.pt")
    torch.save({"optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict()}, d / "optimizer.pt")

    meta = {
        "step":       step,
        "cfg":        cfg,
        "cfg_hash":   config_hash(cfg),
        "model_size": model_size,
        "model_type": model_type,
    }
    if cell_result is not None:
        meta["mqar_intask"] = cell_result
    torch.save(meta, d / "meta.pt")

    if cell_result is not None:
        with open(d / "mqar_result.json", "w") as f:
            json.dump({"step": step, "model_type": model_type,
                       "mqar_intask": cell_result}, f, indent=2)

    print(f"  Checkpoint → {d}  (hash={config_hash(cfg)[:8]})")
    return d


def load_checkpoint(resume_from, model, optimizer, scheduler):
    p = Path(resume_from)
    model.load_state_dict(torch.load(p / "model.pt", map_location="cpu",
                                     weights_only=True))
    opt = torch.load(p / "optimizer.pt", map_location="cpu", weights_only=True)
    optimizer.load_state_dict(opt["optimizer"])
    scheduler.load_state_dict(opt["scheduler"])
    # meta.pt contains KoopmanLMConfig (a custom dataclass), so weights_only=False
    meta = torch.load(p / "meta.pt", map_location="cpu", weights_only=False)
    step = meta["step"]
    print(f"  Resumed from {p} at step {step}")
    return step


# ---------------------------------------------------------------------------
# In-task evaluation
# ---------------------------------------------------------------------------

def run_eval(model, seq_len, num_kv_pairs, vocab_size, eval_batch, device, step):
    """Evaluate on the exact training cell (in-task)."""
    acc = eval_mqar(model, batch=eval_batch, seq_len=seq_len,
                    num_kv_pairs=num_kv_pairs, vocab_size=vocab_size,
                    device=device, seed=999)
    print(f"  [step {step}] in-task accuracy "
          f"(seq={seq_len}, kv={num_kv_pairs}): {acc:.4f}")
    return acc


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    # seq_len is derived from distractor gap: 2*M KV block + gap + 2*M query block
    seq_len = 4 * args.num_kv_pairs + args.distractor_gap

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Cell: model={args.model_type}  kv={args.num_kv_pairs}  "
          f"gap={args.distractor_gap}  seq_len={seq_len}")

    cfg = build_config(args.model_size)
    # Model vocab stays at the config default (e.g. 32000) so parameter count
    # matches the paper's "50M" label. Task vocab is kept small (default 128)
    # so MQAR sequences only use tokens 0..task_vocab-1 and retrieval difficulty
    # comes purely from distractor gap length, not output-space size.
    cfg = dataclasses.replace(cfg, max_seq_len=seq_len)
    ch = config_hash(cfg)
    print(f"Config hash: {ch[:8]}")

    model = build_model(args.model_type, cfg).to(device)
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total:,} ({total/1e6:.1f}M)")
    print(f"Task vocab: {args.task_vocab_size}  Model vocab: {cfg.vocab_size}")

    dataset = MQARDataset(
        seq_len=seq_len,
        num_kv_pairs=args.num_kv_pairs,
        vocab_size=args.task_vocab_size,
        epoch_size=args.epoch_size,
        base_seed=args.seed,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=True,
                        drop_last=True)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr,
        betas=(0.9, 0.95), weight_decay=0.1)

    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(args.warmup_steps, 1)
        t = (step - args.warmup_steps) / max(args.max_steps - args.warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * t))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    start_step = 0
    if args.resume_from:
        start_step = load_checkpoint(args.resume_from, model, optimizer, scheduler)

    use_wandb = bool(args.wandb_project)
    if use_wandb:
        import wandb
        group = args.wandb_group or f"mqar-{args.model_size}-{ch[:8]}"
        wandb.init(
            project=args.wandb_project,
            name=f"{args.model_type}-kv{args.num_kv_pairs}-gap{args.distractor_gap}",
            group=group,
            config={**vars(args), "seq_len": seq_len, "cfg_hash": ch,
                    "n_params": total},
            resume="allow",
        )

    autocast = torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                  enabled=(device.type == "cuda"))
    model.train()
    step     = start_step
    epoch    = 0
    t0       = time.time()
    last_acc = None

    while step < args.max_steps:
        epoch += 1
        for inputs, labels in loader:
            if step >= args.max_steps:
                break

            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast:
                out    = model(input_ids=inputs)
                logits = out["logits"]
                # Shift: logits[t] predicts labels[t+1].
                # Matches eval_mqar which checks logits[:,:-1] vs labels[:,1:].
                loss = F.cross_entropy(
                    logits[:, :-1].reshape(-1, logits.size(-1)),
                    labels[:, 1:].reshape(-1),
                    ignore_index=-100,
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            if step % args.log_every == 0:
                ppl     = math.exp(min(loss.item(), 20))
                lr      = optimizer.param_groups[0]["lr"]
                elapsed = time.time() - t0
                print(f"step {step:>6d}/{args.max_steps}  "
                      f"loss {loss.item():.4f}  ppl {ppl:.1f}  "
                      f"lr {lr:.2e}  {elapsed:.0f}s")
                if use_wandb:
                    wandb.log({"loss": loss.item(), "ppl": ppl,
                               "lr": lr, "step": step})

            if step % args.eval_every == 0:
                last_acc = run_eval(model, seq_len, args.num_kv_pairs,
                                    args.task_vocab_size, args.eval_batch,
                                    device, step)
                if use_wandb:
                    wandb.log({"mqar_intask_acc": last_acc, "step": step})

            if step % args.save_steps == 0:
                acc = last_acc if step % args.eval_every == 0 else None
                save_checkpoint(args.output_dir, step, model, optimizer,
                                scheduler, cfg, args.model_size,
                                args.model_type, acc)

    # Final eval + checkpoint
    print("\nFinal evaluation:")
    final_acc = run_eval(model, seq_len, args.num_kv_pairs,
                         args.task_vocab_size, args.eval_batch, device, step)
    save_checkpoint(args.output_dir, step, model, optimizer, scheduler,
                    cfg, args.model_size, args.model_type, final_acc)

    # Copy to final/ so eval harness can find it by convention
    final_dir = Path(args.output_dir) / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    last_ckpt = _ckpt_dir(args.output_dir, step)
    for fname in ("model.pt", "meta.pt"):
        shutil.copy2(last_ckpt / fname, final_dir / fname)

    if use_wandb:
        wandb.finish()

    print(f"\nDone. Final in-task accuracy: {final_acc:.4f}")
    print(f"Checkpoints in {args.output_dir}/")
    return final_acc


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="In-task MQAR training (Echo/SKA paper setup)")

    # Cell definition
    p.add_argument("--model_type", type=str, default="koopman",
                   choices=list(PAPER_MODEL_TYPES),
                   help="which model variant to train")
    p.add_argument("--num_kv_pairs", type=int, default=32,
                   choices=list(PAPER_KV_PAIRS),
                   help="M ∈ {4, 8, 16, 32}")
    p.add_argument("--distractor_gap", type=int, default=1024,
                   choices=list(PAPER_GAPS),
                   help="filler tokens between KV block and query block")

    # Model
    p.add_argument("--model_size", type=str, default="50m",
                   help="model size name or path to a YAML config")

    # Task vocab — kept small (paper uses V=128) so difficulty comes purely from
    # retrieval distance, not output-space size. Model vocab stays at the config
    # default (32000) so parameter count matches the paper's "50M" label.
    p.add_argument("--task_vocab_size", type=int, default=128,
                   help="MQAR synthetic vocab: keys in [0,V//2), values in [V//2,V)")

    # Training
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epoch_size", type=int, default=10000)
    p.add_argument("--max_steps", type=int, default=10000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)

    # Checkpointing
    p.add_argument("--output_dir", type=str, default="./mqar-out")
    p.add_argument("--save_steps", type=int, default=2000)
    p.add_argument("--resume_from", type=str, default=None)

    # Evaluation
    p.add_argument("--eval_every", type=int, default=2000)
    p.add_argument("--eval_batch", type=int, default=64)

    # Logging
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--wandb_project", type=str, default=None)
    p.add_argument("--wandb_group", type=str, default=None)

    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    train(args)


if __name__ == "__main__":
    main()
