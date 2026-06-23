"""
sweep_mqar.py — Launch all MQAR cells from the Echo/SKA paper grid.

Trains one model per (model_type, num_kv_pairs, distractor_gap) cell,
sequentially on one GPU. For parallel execution across GPUs or nodes,
use --dry_run to print commands and dispatch them yourself.

Grid:
    model_types    : koopman, mamba_attn, mamba_only
    num_kv_pairs   : 4, 8, 16, 32
    distractor_gaps: 64, 128, 256, 512, 1024, 2048, 4096

Total cells: 3 × 4 × 7 = 84

Usage
-----
    # Run all cells sequentially (single GPU):
    python -m koopman_lm.training.sweep_mqar --output_root ./mqar-sweep

    # Print commands only (for manual dispatch / slurm):
    python -m koopman_lm.training.sweep_mqar --dry_run

    # Run only koopman cells (ablation baseline comparison later):
    python -m koopman_lm.training.sweep_mqar --model_types koopman

    # Resume a partial sweep (already-completed cells are skipped):
    python -m koopman_lm.training.sweep_mqar --output_root ./mqar-sweep
"""

import argparse
import subprocess
import sys
from pathlib import Path

from koopman_lm.training.train_mqar import PAPER_KV_PAIRS, PAPER_GAPS, PAPER_MODEL_TYPES


def cell_output_dir(output_root, model_type, num_kv_pairs, distractor_gap):
    return Path(output_root) / model_type / f"kv{num_kv_pairs}_gap{distractor_gap}"


def cell_is_done(output_root, model_type, num_kv_pairs, distractor_gap):
    """Skip if final checkpoint already exists."""
    d = cell_output_dir(output_root, model_type, num_kv_pairs, distractor_gap)
    return (d / "final" / "model.pt").exists()


def build_command(args, model_type, num_kv_pairs, distractor_gap):
    out = cell_output_dir(args.output_root, model_type, num_kv_pairs, distractor_gap)
    cmd = [
        sys.executable, "-m", "koopman_lm.training.train_mqar",
        "--model_type",      model_type,
        "--model_size",      args.model_size,
        "--num_kv_pairs",    str(num_kv_pairs),
        "--distractor_gap",  str(distractor_gap),
        "--task_vocab_size",  str(args.task_vocab_size),
        "--batch_size",      str(args.batch_size),
        "--eval_batch",      str(args.eval_batch),
        "--max_steps",       str(args.max_steps),
        "--save_steps",      str(args.save_steps),
        "--eval_every",      str(args.eval_every),
        "--log_every",       str(args.log_every),
        "--warmup_steps",    str(args.warmup_steps),
        "--output_dir",      str(out),
        "--seed",            str(args.seed),
    ]
    if args.wandb_project:
        cmd += ["--wandb_project", args.wandb_project,
                "--wandb_group",   f"mqar-sweep-{args.model_size}"]
    return cmd


def main():
    p = argparse.ArgumentParser(description="Sweep all MQAR cells")

    p.add_argument("--output_root", type=str, default="./mqar-sweep")
    p.add_argument("--model_types", nargs="+", default=list(PAPER_MODEL_TYPES),
                   choices=list(PAPER_MODEL_TYPES))
    p.add_argument("--kv_pairs", nargs="+", type=int, default=list(PAPER_KV_PAIRS))
    p.add_argument("--gaps", nargs="+", type=int, default=list(PAPER_GAPS))

    p.add_argument("--model_size",   type=str, default="50m")
    p.add_argument("--task_vocab_size", type=int, default=128)
    p.add_argument("--batch_size",   type=int, default=64)
    p.add_argument("--eval_batch",   type=int, default=64)
    p.add_argument("--max_steps",    type=int, default=10000)
    p.add_argument("--save_steps",   type=int, default=2000)
    p.add_argument("--eval_every",   type=int, default=2000)
    p.add_argument("--log_every",    type=int, default=100)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--wandb_project", type=str, default=None)
    p.add_argument("--seed",          type=int, default=42)

    p.add_argument("--dry_run", action="store_true",
                   help="print commands without running them")
    p.add_argument("--skip_done", action="store_true", default=True,
                   help="skip cells whose final/ checkpoint already exists")

    args = p.parse_args()

    cells = [(mt, kv, gap)
             for mt  in args.model_types
             for kv  in args.kv_pairs
             for gap in args.gaps]

    total  = len(cells)
    done   = 0
    skip   = 0

    print(f"MQAR sweep: {total} cells  "
          f"({len(args.model_types)} types × "
          f"{len(args.kv_pairs)} kv × "
          f"{len(args.gaps)} gaps)")
    print(f"Output root: {args.output_root}\n")

    for i, (model_type, kv, gap) in enumerate(cells, 1):
        seq_len = 4 * kv + gap
        tag     = f"[{i:>3d}/{total}] {model_type:<12} kv={kv:<2} gap={gap:<5} seq={seq_len}"

        if args.skip_done and cell_is_done(args.output_root, model_type, kv, gap):
            print(f"{tag}  SKIP (done)")
            skip += 1
            continue

        cmd = build_command(args, model_type, kv, gap)

        if args.dry_run:
            print(f"{tag}")
            print("  " + " ".join(cmd))
            continue

        print(f"{tag}  RUNNING...")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"  FAILED (exit {result.returncode}) — continuing sweep")
        else:
            done += 1

    if not args.dry_run:
        print(f"\nSweep complete: {done} trained, {skip} skipped, "
              f"{total - done - skip} failed")


if __name__ == "__main__":
    main()
