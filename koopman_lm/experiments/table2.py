"""
table2.py -- Reproduce Echo paper Table 2 (Section 4.1/5.1): NIAH length
generalization at sub-million scale.

What the paper actually specifies (verified against the paper text, not
inferred):
    - Models: SSM (4-layer Mamba-2, 1.11M params), SSM+Attn (2 Mamba-2 + 2
      causal-attn w/ RoPE, 998K), SSM+SKA (2 Mamba-2 + 2 SKA rank=24, 982K).
      d=128, 4 heads, d_state=16, V=128, SwiGLU MLPs (Sec 4.1).
    - Optimization: AdamW, lr=3e-4, 6,000 steps, batch size 16 (Sec 4.1).
    - Training data: "a mixed dataset of system-prompt and tool-trace
      examples" (Sec 4.1) -- no token format, generator, or loss/masking
      detail is given anywhere in the paper (main text or appendices) for
      this. Appendix G.4's "Tool-Calling Retrieval" / "System Prompt
      Amnesia" tasks belong to a DIFFERENT experiment (a Mamba-3+CG+SKA
      ablation study, d=96, different step counts, includes a "Resource
      Economy" task never mentioned in Sec 4.1) and do not describe this one
      -- an earlier version of this file wrongly borrowed that appendix;
      that mistake has been reverted.
    - Table 2 itself: NIAH "(KV=1)", trained at sequence length 64 and
      evaluated at up to 64x longer (Sec 4.1: "length generalization at
      2-64x beyond training sequence length"). "Each task varies KV pairs
      and sequence length" (Sec 4.1) frames NIAH as belonging to the same
      family as Sec 4.2's MQAR benchmark (cited to Arora et al. [2]) rather
      than a separately-specified format.

Given that gap, this script trains on MQAR itself (koopman_lm.experiments.
curricula.make_mqar, the one generator format the paper actually cites) at
num_kv_pairs=4, seq_len=64 -- KV=1 is held out entirely from training so it
remains a genuine zero-shot generalization test, matching Table 2's own
"(KV=1)" framing and never-trained-on requirement. This is a stand-in for
the unspecified "system-prompt and tool-trace" curriculum, not a literal
transcription of it; there isn't enough in the paper to build one. Loss is
plain sparse cross-entropy at the answer positions (ignore_index=-100),
matching the paper's own MQAR/NIAH convention elsewhere (Sec 5.2) -- no
dense/ppl-weighted objective, since the paper reports no perplexity metric
for this experiment and gives no dense-labeling detail.

Usage:
    python -m koopman_lm.experiments.table2 --model_type mamba_ska_swiglu
    python -m koopman_lm.experiments.table2 --model_type mamba_attn
    python -m koopman_lm.experiments.table2 --model_type mamba_only
"""

import json
import math
import time
import argparse
import shutil
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from koopman_lm.globals.config import build_config, config_hash
from koopman_lm.models.baselines import (
    build_mamba_only, build_mamba_attention, build_mamba_ska_swiglu)
from koopman_lm.experiments.curricula import make_mqar, eval_mqar

TABLE2_SEQ_LENS = (64, 128, 256, 512, 1024, 2048, 4096)
TRAIN_SEQ_LEN    = 64
TRAIN_KV_PAIRS   = 4      # never 1 -- KV=1 (NIAH) stays a zero-shot held-out cell


class MQARDataset(Dataset):
    """On-the-fly MQAR sequences at the training cell (seq_len=64, KV=4)."""

    def __init__(self, seq_len, num_kv_pairs, vocab_size,
                 epoch_size=100000, base_seed=0):
        self.seq_len      = seq_len
        self.num_kv_pairs = num_kv_pairs
        self.vocab_size   = vocab_size
        self.epoch_size   = epoch_size
        self.base_seed     = base_seed

    def __len__(self):
        return self.epoch_size

    def __getitem__(self, idx):
        inputs, labels = make_mqar(
            batch=1, seq_len=self.seq_len, num_kv_pairs=self.num_kv_pairs,
            vocab_size=self.vocab_size, seed=self.base_seed + idx)
        return inputs[0], labels[0]


def build_model(model_type, cfg):
    if model_type == "mamba_only":
        return build_mamba_only(cfg)
    elif model_type == "mamba_attn":
        return build_mamba_attention(cfg)
    elif model_type == "mamba_ska_swiglu":
        return build_mamba_ska_swiglu(cfg)
    raise ValueError(f"Unknown model_type: {model_type!r}")


def save_checkpoint(output_dir, step, model, optimizer, scheduler,
                    cfg, model_type, results=None):
    d = Path(output_dir) / f"step_{step}"
    d.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), d / "model.pt")
    torch.save({"optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict()}, d / "optimizer.pt")
    meta = {"step": step, "cfg": cfg, "cfg_hash": config_hash(cfg),
            "model_type": model_type}
    if results:
        meta["niah_table2"] = results
    torch.save(meta, d / "meta.pt")
    if results:
        with open(d / "table2_results.json", "w") as f:
            json.dump({"step": step, "model_type": model_type,
                       "niah_table2": results}, f, indent=2)
    print(f"  Checkpoint -> {d}  (hash={config_hash(cfg)[:8]})")


def load_checkpoint(resume_from, model, optimizer, scheduler):
    p = Path(resume_from)
    model.load_state_dict(torch.load(p / "model.pt", map_location="cpu",
                                     weights_only=True))
    opt = torch.load(p / "optimizer.pt", map_location="cpu", weights_only=True)
    optimizer.load_state_dict(opt["optimizer"])
    scheduler.load_state_dict(opt["scheduler"])
    meta = torch.load(p / "meta.pt", map_location="cpu", weights_only=False)
    print(f"  Resumed from {p} at step {meta['step']}")
    return meta["step"]


@torch.no_grad()
def run_table2_eval(model, device, eval_batch, task_vocab_size, step):
    """Zero-shot NIAH (MQAR, KV=1) at all Table 2 sequence lengths -- never
    trained on."""
    model.eval()
    results = {}
    print(f"  [step {step}] NIAH length-generalization (MQAR, KV=1, zero-shot):")
    for T in TABLE2_SEQ_LENS:
        acc = eval_mqar(model, batch=eval_batch, seq_len=T, num_kv_pairs=1,
                        vocab_size=task_vocab_size, device=device, seed=9999)
        results[T] = acc
        tag = " (train seq_len)" if T == TRAIN_SEQ_LEN else ""
        print(f"    seq={T:<5d}{tag}: {acc:.4f}")
    model.train()
    return results


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model: {args.model_type}  Config: {args.model_size}")

    cfg = build_config(args.model_size)
    print(f"Config hash: {config_hash(cfg)[:8]}")

    model = build_model(args.model_type, cfg).to(device)
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total:,} ({total/1e6:.2f}M)")

    dataset = MQARDataset(seq_len=TRAIN_SEQ_LEN, num_kv_pairs=TRAIN_KV_PAIRS,
                          vocab_size=args.task_vocab_size,
                          epoch_size=args.epoch_size, base_seed=args.seed)
    loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                         num_workers=args.num_workers, pin_memory=True,
                         drop_last=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  betas=(0.9, 0.95), weight_decay=0.01)

    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(args.warmup_steps, 1)
        t = (step - args.warmup_steps) / max(args.max_steps - args.warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * t))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    start_step = 0
    if args.resume_from:
        start_step = load_checkpoint(args.resume_from, model, optimizer, scheduler)

    autocast = torch.amp.autocast("cuda", dtype=torch.float16,
                                  enabled=(device.type == "cuda"))
    scaler   = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    model.train()
    step = start_step
    t0   = time.time()

    while step < args.max_steps:
        for inputs, labels in loader:
            if step >= args.max_steps:
                break
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast:
                out    = model(input_ids=inputs)
                logits = out["logits"]
                loss   = F.cross_entropy(
                    logits[:, :-1].reshape(-1, logits.size(-1)),
                    labels[:, 1:].reshape(-1),
                    ignore_index=-100,
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            step += 1

            if step % args.log_every == 0:
                ppl     = math.exp(min(loss.item(), 20))
                lr      = optimizer.param_groups[0]["lr"]
                elapsed = time.time() - t0
                print(f"step {step:>5d}/{args.max_steps}  "
                      f"loss {loss.item():.4f}  ppl {ppl:.1f}  "
                      f"lr {lr:.2e}  {elapsed:.0f}s")

            if step % args.eval_every == 0 or step == args.max_steps:
                results = run_table2_eval(model, device, args.eval_batch,
                                          args.task_vocab_size, step)
                save_checkpoint(args.output_dir, step, model, optimizer,
                                scheduler, cfg, args.model_type, results)

    final_dir = Path(args.output_dir) / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    last = Path(args.output_dir) / f"step_{step}"
    for fname in ("model.pt", "meta.pt"):
        shutil.copy2(last / fname, final_dir / fname)

    print(f"\nDone. Checkpoints in {args.output_dir}/")


def parse_args():
    p = argparse.ArgumentParser(description="Table 2: NIAH (MQAR, KV=1) length generalization")
    p.add_argument("--model_type", type=str, default="mamba_ska_swiglu",
                   choices=["mamba_only", "mamba_attn", "mamba_ska_swiglu"])
    p.add_argument("--model_size",   type=str,   default="1m")
    p.add_argument("--task_vocab_size", type=int, default=128,
                   help="MQAR synthetic vocab: keys in [0,V//2), values in [V//2,V)")
    p.add_argument("--batch_size",   type=int,   default=16)
    p.add_argument("--lr",           type=float, default=3e-4)
    p.add_argument("--warmup_steps", type=int,   default=200)
    p.add_argument("--max_steps",    type=int,   default=6000)
    p.add_argument("--epoch_size",   type=int,   default=200000)
    p.add_argument("--num_workers",  type=int,   default=2)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--output_dir",   type=str,   default="./table2-out")
    p.add_argument("--resume_from",  type=str,   default=None)
    p.add_argument("--eval_every",   type=int,   default=1000)
    p.add_argument("--eval_batch",   type=int,   default=256)
    p.add_argument("--log_every",    type=int,   default=200)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    train(args)


if __name__ == "__main__":
    main()
