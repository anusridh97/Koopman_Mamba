#!/usr/bin/env python
"""
train.py -- top-level launcher for the Echo/SKA 440M run.

This is a THIN launcher. The real training loop is koopman_lm/train_fast.py
(model build, DDP, weighted-CE, checkpointing). train.py just:
  1. fixes the 440M launch defaults we settled on (config_440m: 8 SKA layers,
     rank 96, chunk 96, lag-biased short-conv ON, LayerScale, eta=gamma=1),
  2. sets sane 8xB200 run defaults (bf16, DDP, seq len, accumulation),
  3. delegates to train_fast.train().

The 440M ARCHITECTURE is defined in config_440m() and is NOT overridable here
(model_size is pinned to "440m"). Run hyperparameters (batch, steps, lr, seq
len, data dir) ARE overridable on the command line.

WHAT THIS DOES NOT DO / ASSUMES:
  - Data must already be pretokenized into <data_dir>/train.bin (+ weights.bin)
    via koopman_lm/pretokenize.py. This script does NOT tokenize.
  - mamba_ssm + causal-conv1d + a CUDA torch must be installed (see setup.py).
    Building config_440m() instantiates Mamba2 (model.py imports mamba_ssm).
  - NONE of the GPU path has been executed in the environment that produced
    this package -- run the smoke gates (CHANGES_440M.md s18) before a long run.

HARDWARE NOTE (2x H200, 141GB each):
  444M is NOT memory-bound here -- model+grad+AdamW is ~7GB of 141. So:
    * plain DDP-2 is correct (NO ZeRO/sharding/offload -- those solve a memory
      problem you don't have).
    * you have huge headroom: use a LARGE per-device batch and consider turning
      gradient checkpointing OFF (--no_gradient_checkpointing) -- it trades
      compute for memory you don't need to save, so disabling it speeds up each
      step.
    * the real constraint is throughput: 2 GPUs vs 8 is ~4x fewer, so plan a
      proportionally smaller token budget or longer wall-clock.

Single GPU (smoke):
    python train.py --data_dir ./data --max_steps 50 --no_compile

2x H200 (real run), via torchrun:
    torchrun --nproc_per_node=2 train.py --data_dir ./data --ddp \
        --per_device_train_batch_size 16 --gradient_accumulation_steps <A> \
        --no_gradient_checkpointing

Pick the largest micro-batch that fits (you have room for a big one), then set
A = the minimum to hit your target global tokens/step. Do NOT just maximize A.
"""
import argparse
import os
import sys


def build_argv():
    """Map this launcher's flags onto train_fast.py's CLI, pinning 440M and the
    agreed run defaults, then hand off to train_fast.train()."""
    p = argparse.ArgumentParser(
        description="Launch the Echo/SKA 440M training run (delegates to train_fast).")
    # run-level (overridable)
    p.add_argument("--data_dir", type=str, required=True,
                   help="dir containing pretokenized train.bin (+ weights.bin)")
    p.add_argument("--output_dir", type=str, default="./echo-ska-440m-run")
    p.add_argument("--max_seq_len", type=int, default=4096,
                   help="long-context recall lives in PG-19/SCROLLS fraction; "
                        "8192 is ~free on B200 memory-wise. Default 4096.")
    p.add_argument("--per_device_train_batch_size", type=int, default=16,
                   help="2x H200 has huge headroom for 444M; default 16. "
                        "Raise until you fill memory or hit diminishing throughput.")
    p.add_argument("--gradient_accumulation_steps", type=int, default=1,
                   help="set to the MINIMUM that hits your target tokens/step")
    p.add_argument("--max_steps", type=int, default=None,
                   help="explicit step count; if omitted, computed from "
                        "--target_tokens and the global batch.")
    p.add_argument("--target_tokens", type=float, default=20e9,
                   help="token budget; default 20e9 (20B, ~45x for 444M, matches "
                        "the 180M's ~56x regime). max_steps is derived from this "
                        "and the global tokens/step unless --max_steps is given.")
    p.add_argument("--learning_rate", type=float, default=6e-4)
    p.add_argument("--warmup_steps", type=int, default=2000)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--tokenizer", type=str, default="meta-llama/Llama-2-7b-hf",
                   help="ungated mirror: NousResearch/Llama-2-7b-hf")
    p.add_argument("--ddp", action="store_true", default=False,
                   help="enable for multi-GPU (torchrun sets WORLD_SIZE>1)")
    p.add_argument("--ska_fast", action="store_true", default=False,
                   help="fused-QKV SKA training variant (ska_fast.py)")
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=5000)
    p.add_argument("--wandb_project", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    # toggles (default ON for the real run; off for smoke)
    p.add_argument("--no_bf16", action="store_false", dest="bf16", default=True)
    p.add_argument("--no_compile", action="store_false", dest="compile", default=True,
                   help="disable torch.compile -- recommended for the FIRST smoke "
                        "run; compile can mishandle the custom SKACoreFn backward "
                        "until verified on your stack.")
    p.add_argument("--no_gradient_checkpointing", action="store_false",
                   dest="gradient_checkpointing", default=True)
    args = p.parse_args()

    # Derive max_steps from the token budget if not given explicitly. Global
    # tokens/step = micro_bsz * accum * world_size * seq_len. world_size comes
    # from torchrun's env (1 if single-process).
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if args.max_steps is None:
        gtok = (args.per_device_train_batch_size * args.gradient_accumulation_steps
                * world_size * args.max_seq_len)
        args.max_steps = int(args.target_tokens / gtok)
        if int(os.environ.get("RANK", 0)) == 0:
            print(f"[train.py] target {args.target_tokens/1e9:.1f}B tokens / "
                  f"{gtok/1e6:.2f}M per step (B={args.per_device_train_batch_size} "
                  f"x A={args.gradient_accumulation_steps} x ws={world_size} "
                  f"x T={args.max_seq_len}) -> max_steps={args.max_steps}")
            if world_size == 1:
                print("[train.py] WARNING: WORLD_SIZE=1, so max_steps assumes a "
                      "SINGLE GPU. Under torchrun --nproc_per_node=2 it will be "
                      "recomputed for 2 GPUs (half the steps). Launch via torchrun "
                      "for the real run.")

    # Build the namespace train_fast.train() expects, PINNING the 440M arch.
    from types import SimpleNamespace
    return SimpleNamespace(
        model_type="koopman",
        model_size="440m",          # pinned: this launcher is 440M-only
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        tokenizer=args.tokenizer,
        max_seq_len=args.max_seq_len,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        bf16=args.bf16,
        compile=args.compile,
        gradient_checkpointing=args.gradient_checkpointing,
        ska_fast=args.ska_fast,
        num_workers=args.num_workers,
        ddp=args.ddp,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        wandb_project=args.wandb_project,
        seed=args.seed,
    )


def main():
    args = build_argv()
    # Fail early with a clear message if data isn't pretokenized.
    train_bin = os.path.join(args.data_dir, "train.bin")
    if not os.path.exists(train_bin):
        sys.exit(f"ERROR: {train_bin} not found. Pretokenize first:\n"
                 f"  python -m koopman_lm.pretokenize --output_dir {args.data_dir} ...\n"
                 f"(see koopman_lm/pretokenize.py for the data-mix / weighting args)")
    rank = int(os.environ.get("RANK", 0))
    if rank == 0:
        ws = int(os.environ.get("WORLD_SIZE", 1))
        print(f"[train.py] Echo/SKA 440M | world_size={ws} | seq_len={args.max_seq_len} "
              f"| micro_bsz={args.per_device_train_batch_size} "
              f"| accum={args.gradient_accumulation_steps} | bf16={args.bf16} "
              f"| compile={args.compile}")
        if ws == 1 and args.ddp:
            print("[train.py] NOTE: --ddp set but WORLD_SIZE==1; launch via "
                  "torchrun --nproc_per_node=8 for multi-GPU.")
    from koopman_lm.train_fast import train
    train(args)


if __name__ == "__main__":
    main()
