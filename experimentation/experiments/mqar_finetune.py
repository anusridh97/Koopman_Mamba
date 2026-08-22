"""
mqar_finetune.py -- Section 5.2: MQAR fine-tuning (Echo/SKA paper, 50M scale).

"Having established SKA's advantage at sub-million scale, we tested whether
the memory cliff persists at a more realistic parameter budget by
fine-tuning 50M-parameter models on MQAR. We tested a grid of MQAR
configurations (KV pairs M in {4,8,16,32}, distractor gaps 64-4,096 tokens),
trained and evaluated independently" (Sec 5.2). Each (model_type, M, gap)
cell is a fully independent training run -- unlike table2.py, there is no
mixed curriculum here; this reproduces the paper's own in-task protocol,
sparse answer-position supervision included.

Seq length is derived automatically: seq_len = 4*M + distractor_gap
(2*M for the KV block + gap fillers + 2*M for the query block).

Checkpointing
-------------
Every --save_steps a checkpoint is written to:
    <output_dir>/step_<N>/{model.pt, optimizer.pt, meta.pt, mqar_result.json}

Resume with --resume_from <output_dir>/step_<N>.

Single-cell example
--------------------
    python -m experimentation.experiments.mqar_finetune \\
        --model_type mamba_ska_swiglu \\
        --model_size 50m \\
        --num_kv_pairs 32 \\
        --distractor_gap 1024 \\
        --output_dir ./mqar-ska-swiglu-m32-g1024

Every cell (Section 5.2 grid: 3 model types x 4 KV counts x 7 gaps):
    python -m experimentation.experiments.mqar_finetune --sweep \\
        --output_root ./mqar-sweep
"""

import os
import sys
import json
import math
import time
import argparse
import dataclasses
import shutil
import subprocess
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from koopman_lm.config import build_config, config_hash
from experimentation.run.provenance import git_commit, git_dirty_paths
from experimentation.training.amp import amp_for
from experimentation.training.repro import enable_determinism
from experimentation.training.loop import run_training_loop
from experimentation.training.task import SyntheticTask
from experimentation.training.optim import param_groups
from koopman_lm.models.baselines import (
    build_mamba_only, build_mamba_attention, build_mamba_ska_swiglu)
from experimentation.experiments.curricula import make_mqar, eval_mqar

# Paper grid (Echo/SKA, Arora et al. 2024)
PAPER_KV_PAIRS     = (4, 8, 16, 32)
PAPER_GAPS         = (64, 128, 256, 512, 1024, 2048, 4096)
PAPER_MODEL_TYPES  = ("mamba_ska_swiglu", "mamba_attn", "mamba_only")


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
    if model_type == "mamba_ska_swiglu":
        return build_mamba_ska_swiglu(cfg)
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

    # code_id/dirty: see training/train.py::checkpoint_meta (provenance 4.1).
    meta = {
        "step":       step,
        "cfg":        cfg,
        "cfg_hash":   config_hash(cfg),
        "model_size": model_size,
        "model_type": model_type,
        "code_id":    git_commit(),
        "dirty":      bool(git_dirty_paths()),
    }
    if cell_result is not None:
        meta["mqar_intask"] = cell_result
    torch.save(meta, d / "meta.pt")

    if cell_result is not None:
        with open(d / "mqar_result.json", "w") as f:
            json.dump({"step": step, "model_type": model_type,
                       "mqar_intask": cell_result}, f, indent=2)

    print(f"  Checkpoint -> {d}  (hash={config_hash(cfg)[:8]})")
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

def run_eval(model, seq_len, num_kv_pairs, vocab_size, eval_batch, device, step,
             eval_seq_lens=None):
    """Evaluate accuracy.

    If eval_seq_lens is given, sweeps all of those sequence lengths with the
    same num_kv_pairs (length-generalization mode). Otherwise evaluates only
    on the training cell (standard in-task mode, matching Sec 5.2).
    """
    model.eval()
    results = {}

    if eval_seq_lens:
        print(f"  [step {step}] length-generalization eval (kv={num_kv_pairs}):")
        for T in eval_seq_lens:
            if 4 * num_kv_pairs >= T:
                continue
            acc = eval_mqar(model, batch=eval_batch, seq_len=T,
                            num_kv_pairs=num_kv_pairs, vocab_size=vocab_size,
                            device=device, seed=999)
            results[T] = acc
            marker = " (train)" if T == seq_len else ""
            print(f"    seq={T:<5d}{marker}: {acc:.4f}")
    else:
        acc = eval_mqar(model, batch=eval_batch, seq_len=seq_len,
                        num_kv_pairs=num_kv_pairs, vocab_size=vocab_size,
                        device=device, seed=999)
        results[seq_len] = acc
        print(f"  [step {step}] in-task accuracy (seq={seq_len}, kv={num_kv_pairs}): {acc:.4f}")

    model.train()
    return results


# ---------------------------------------------------------------------------
# Training loop (single cell)
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

    # Route through the shared decay/no-decay policy (see
    # experimentation.training.optim.param_groups) instead of flat model.parameters().
    # A flat AdamW(weight_decay=0.1) decayed norms, biases, embeddings, and the
    # Mamba state parameters (A_log, D, dt_bias) too -- at this script's 50m scale
    # a prior audit found ~24% of parameters affected at 10x the intended decay
    # coefficient. Published MQAR fine-tune numbers were produced under the old,
    # unfiltered-decay optimizer and are superseded.
    optimizer = torch.optim.AdamW(
        param_groups(model, weight_decay=0.1), lr=args.lr,
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

    autocast, grad_scaler = amp_for(cfg, "cuda",
                                    enabled=(device.type == "cuda"))
    model.train()
    step     = start_step
    epoch    = 0
    t0       = time.time()
    # Counted so the progress line can report tok/s in the same shape train.py
    # uses. Not a vanity metric: sweep.search.metrics.TRAIN_RE requires that
    # field, and without it nothing downstream can read this trainer's log.
    tokens_seen = 0
    # Which loss convention this loop runs. SyntheticTask means "the data is
    # aligned, so the loss applies the offset" -- the opposite of ShardTask, and
    # the distinction the whole TrainTask design exists to keep explicit.
    task = SyntheticTask(model_type=args.model_type)
    last_acc = None

    # §6.2: this trainer no longer carries its own loop. Everything below --
    # accumulation, clipping, the optimizer/schedule order, checkpoint cadence,
    # the progress line -- is training/loop.py's, shared with train.py and
    # table2. What stays SyntheticTask's is the loss convention, the batch order
    # and the MQAR eval; see its docstrings.
    #
    # The shared loop reads a handful of knobs this CLI does not spell. They are
    # set explicitly rather than let the loop getattr-default them, so the values
    # this trainer actually runs at are visible in one place.
    args.logging_steps = args.log_every
    args.per_device_train_batch_size = args.batch_size   # SyntheticTask reads
                                                         # batch_size; the banner
                                                         # reads this one
    args.gradient_accumulation_steps = 1   # mqar has never accumulated
    args.max_grad_norm = 1.0               # was hardcoded in the loop it replaces
    args.max_seq_len = seq_len             # banner only
    args.diag_every = 10 ** 9              # no SKA diagnostics on this path
    args.data_dir = None                   # only read under a seq_len schedule

    def _eval_fn(_model, current_step):
        eval_lens = args.eval_seq_lens if args.eval_seq_lens else None
        return run_eval(model, seq_len, args.num_kv_pairs, args.task_vocab_size,
                        args.eval_batch, device, current_step,
                        eval_seq_lens=eval_lens)

    task = SyntheticTask(model_type=args.model_type, eval_fn=_eval_fn,
                         eval_every=args.eval_every)

    def _save_all(step, epoch, samples_consumed, log_window=None,
                  epoch_start_step=0, dirname=None):
        # Matches the original: an accuracy is attached only when this step is
        # also an eval step, otherwise the checkpoint records None rather than a
        # stale number from an earlier eval.
        acc = None
        if step % args.eval_every == 0 and task.last_eval:
            acc = task.last_eval.get(
                seq_len, next(iter(task.last_eval.values())))
        save_checkpoint(args.output_dir, step, model, optimizer, scheduler, cfg,
                        args.model_size, args.model_type, acc)

    result = run_training_loop(
        task=task, model=model, raw_model=model, optimizer=optimizer,
        scheduler=scheduler, args=args, device=device, train_ds=dataset,
        autocast_ctx=autocast, _save_all=_save_all, scaler=grad_scaler,
        start_step=start_step, t_start=t0)
    step, tokens_seen = result.step, result.tokens_seen
    last_acc = (task.last_eval or {}).get(seq_len) if task.last_eval else None

    # Final eval + checkpoint
    print("\nFinal evaluation:")
    eval_lens = args.eval_seq_lens if args.eval_seq_lens else None
    final_results = run_eval(model, seq_len, args.num_kv_pairs,
                             args.task_vocab_size, args.eval_batch, device, step,
                             eval_seq_lens=eval_lens)
    final_acc = final_results.get(seq_len, next(iter(final_results.values())))
    save_checkpoint(args.output_dir, step, model, optimizer, scheduler,
                    cfg, args.model_size, args.model_type, final_results)

    # Copy to final/ so downstream tooling can find it by convention
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
# Sweep: launch every cell in the Section 5.2 grid, one subprocess per cell.
# ---------------------------------------------------------------------------

def cell_output_dir(output_root, model_type, num_kv_pairs, distractor_gap):
    return Path(output_root) / model_type / f"kv{num_kv_pairs}_gap{distractor_gap}"


def cell_is_done(output_root, model_type, num_kv_pairs, distractor_gap):
    """Skip if final checkpoint already exists."""
    d = cell_output_dir(output_root, model_type, num_kv_pairs, distractor_gap)
    return (d / "final" / "model.pt").exists()


def build_cell_command(args, model_type, num_kv_pairs, distractor_gap):
    out = cell_output_dir(args.output_root, model_type, num_kv_pairs, distractor_gap)
    cmd = [
        sys.executable, "-m", "experimentation.experiments.mqar_finetune",
        "--model_type",      model_type,
        "--model_size",      args.model_size,
        "--num_kv_pairs",    str(num_kv_pairs),
        "--distractor_gap",  str(distractor_gap),
        "--task_vocab_size", str(args.task_vocab_size),
        "--batch_size",      str(args.batch_size),
        "--eval_batch",      str(args.eval_batch),
        "--max_steps",       str(args.max_steps),
        "--save_steps",      str(args.save_steps),
        "--eval_every",      str(args.eval_every),
        "--log_every",       str(args.log_every),
        *(["--deterministic"] if args.deterministic else []),
        "--warmup_steps",    str(args.warmup_steps),
        "--output_dir",      str(out),
        "--seed",            str(args.seed),
    ]
    if args.wandb_project:
        cmd += ["--wandb_project", args.wandb_project,
                "--wandb_group",   f"mqar-sweep-{args.model_size}"]
    return cmd


def run_sweep(args):
    """Section 5.2: 'We tested a grid of MQAR configurations ... trained and
    evaluated independently.' Launches one subprocess per (model_type, KV
    pairs, distractor gap) cell, sequentially on one GPU.
    """
    cells = [(mt, kv, gap)
             for mt  in args.model_types
             for kv  in args.kv_pairs
             for gap in args.gaps]

    total = len(cells)
    done  = 0
    skip  = 0

    print(f"MQAR sweep (Section 5.2): {total} cells  "
          f"({len(args.model_types)} types x "
          f"{len(args.kv_pairs)} kv x "
          f"{len(args.gaps)} gaps)")
    print(f"Output root: {args.output_root}\n")

    for i, (model_type, kv, gap) in enumerate(cells, 1):
        seq_len = 4 * kv + gap
        tag = f"[{i:>3d}/{total}] {model_type:<12} kv={kv:<2} gap={gap:<5} seq={seq_len}"

        if args.skip_done and cell_is_done(args.output_root, model_type, kv, gap):
            print(f"{tag}  SKIP (done)")
            skip += 1
            continue

        cmd = build_cell_command(args, model_type, kv, gap)

        if args.dry_run:
            print(f"{tag}")
            print("  " + " ".join(cmd))
            continue

        print(f"{tag}  RUNNING...")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"  FAILED (exit {result.returncode}) -- continuing sweep")
        else:
            done += 1

    if not args.dry_run:
        print(f"\nSweep complete: {done} trained, {skip} skipped, "
              f"{total - done - skip} failed")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Section 5.2: MQAR fine-tuning (single cell, or --sweep for the full grid)")

    p.add_argument("--sweep", action="store_true",
                   help="run every (model_type, kv, gap) cell in the grid, "
                        "one subprocess per cell, instead of a single cell")

    # Cell definition (single-cell mode)
    p.add_argument("--model_type", type=str, default="mamba_ska_swiglu",
                   choices=list(PAPER_MODEL_TYPES),
                   help="which model variant to train")
    p.add_argument("--num_kv_pairs", type=int, default=32,
                   help="KV pairs per sequence (paper MQAR grid: 4/8/16/32)")
    p.add_argument("--distractor_gap", type=int, default=1024,
                   help="filler tokens between KV block and query block")

    # Model
    p.add_argument("--model_size", type=str, default="50m",
                   help="model size name or path to a YAML config")

    # Task vocab -- kept small (paper uses V=128) so difficulty comes purely from
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

    # Checkpointing (single-cell mode)
    p.add_argument("--output_dir", type=str, default="./mqar-out")
    p.add_argument("--save_steps", type=int, default=2000)
    p.add_argument("--resume_from", type=str, default=None)

    # Evaluation
    p.add_argument("--eval_every", type=int, default=2000)
    p.add_argument("--eval_batch", type=int, default=64)
    p.add_argument("--eval_seq_lens", nargs="+", type=int, default=None,
                   help="length-generalization eval: test at these seq lengths "
                        "after training. e.g. 64 128 256 512 1024 2048 4096")

    # Logging
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--deterministic", action="store_true", default=False,
                   help="enable torch deterministic algorithms "
                        "(reproducible loss curves; lower throughput). "
                        "Without it two runs at the same seed diverge to "
                        "~0.5 in loss by step 400 -- see main()")
    p.add_argument("--wandb_project", type=str, default=None)
    p.add_argument("--wandb_group", type=str, default=None)

    # Sweep mode (--sweep)
    p.add_argument("--output_root", type=str, default="./mqar-sweep",
                   help="[sweep] root dir; one subdir per (model_type, kv, gap) cell")
    p.add_argument("--model_types", nargs="+", default=list(PAPER_MODEL_TYPES),
                   choices=list(PAPER_MODEL_TYPES), help="[sweep] model types to cover")
    p.add_argument("--kv_pairs", nargs="+", type=int, default=list(PAPER_KV_PAIRS),
                   help="[sweep] KV-pair values to cover")
    p.add_argument("--gaps", nargs="+", type=int, default=list(PAPER_GAPS),
                   help="[sweep] distractor-gap values to cover")
    p.add_argument("--dry_run", action="store_true",
                   help="[sweep] print commands without running them")
    p.add_argument("--skip_done", action="store_true", default=True,
                   help="[sweep] skip cells whose final/ checkpoint already exists")

    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    if args.deterministic:
        # MEASURED (job 439605): without this, two runs of this trainer at the
        # SAME seed diverge to |A-B| = 0.53 in loss by step 400 -- 2700x the
        # shard trainer's 0.0002. They start identical (0.0002 at step 10, which
        # is the log format's precision) and separate progressively: 0.003 by
        # step 50, 0.018 by 100, 0.53 by 400. So the cause is not seeding or data
        # order -- both are correct, base_seed is tied to args.seed at :234 --
        # but ordinary GPU nondeterminism compounding through the optimizer.
        #
        # It bites here and not on the shard path because MQAR supervises only a
        # handful of positions per sequence (8 answer tokens in ~96) at lr 1e-3,
        # so the objective is sparse and sharp and sits in a regime where tiny
        # perturbations amplify rather than wash out.
        #
        # train.py has had this flag since the exact-resume work; this trainer
        # never did, which meant its results were not reproducible run-to-run and
        # nothing said so.
        enable_determinism(warn_only=True)
        print("  Determinism mode ON (cudnn.benchmark off; throughput will drop)")
    if args.sweep:
        run_sweep(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
