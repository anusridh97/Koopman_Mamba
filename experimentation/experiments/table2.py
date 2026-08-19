"""Table 2 reproduction: NIAH length generalization at sub-million scale
(Echo paper Section 4.1), built on the project's real model stack.

What the paper specifies:
    - Models: SSM (4-layer Mamba-2, 1.11M params), SSM+Attn (2 Mamba-2 + 2
      causal-attn w/ RoPE, 998K), SSM+SKA (2 Mamba-2 + 2 SKA rank=24, 982K).
      d=128, 4 heads, d_state=16, V=128, SwiGLU MLPs (Sec 4.1). These map to
      `configs/1m.yaml` and koopman_lm.models.baselines.build_mamba_only /
      build_mamba_attention / build_mamba_ska_swiglu.
    - Optimization (Table 6): AdamW (beta1=0.9, beta2=0.95), weight decay 0.01,
      lr=3e-4, cosine decay with linear warmup, grad clip max-norm=1.0, batch
      size 16, steps=6000 (Sec 4.1) though empirically this curriculum needs
      more to fully converge -- see --max_steps, mixed FP16 (AMP) when a GPU
      is available.
    - Training data: "a mixed dataset of system-prompt and tool-trace
      examples" (Sec 4.1) -- no exact token format is given anywhere in the
      main text, but Appendix G.4 describes both task families concretely
      for a related ablation: "Tool-Calling Retrieval" (SET/GET with
      overwrite, answer = latest binding) and "System Prompt Amnesia /
      Specific Recall" (n_vars variables set once, then a gap salted with
      confusing distractors). This script mixes
      experimentation.experiments.curricula.make_toolcall and .make_sysprompt at
      the batch level by default (--curriculum batch_mixed): each batch is
      half one, half the other, concatenated, so every gradient step is
      already blended. Step-level alternation (whole batches flipping per
      step, --curriculum mixed) was tried first and measurably interfered
      with convergence -- both tasks' losses plateaued flat for thousands
      of steps despite each training fine in isolation (--curriculum
      toolcall / sysprompt).
    - Table 2 itself: NIAH, trained at sequence length 64 and evaluated at
      up to 64x longer (2-64x beyond training length). Unlike an earlier
      version of this script (which read NIAH "(KV=1)" as literally the
      MQAR generator special-cased to one pair, and held out only that
      parameter value), the eval here uses curricula.make_niah -- a
      structurally distinct generator (single needle at a RANDOM position,
      not fixed at the front) that make_toolcall/make_sysprompt never
      produce in any form, so the zero-shot claim holds at the level of
      "never saw this token format," not just "never saw this parameter."

Requires the project's real Mamba-2 implementation (`mamba_ssm`, CUDA/Linux
only) -- there is no CPU/self-contained fallback in this file. Run it where
that package is installed (e.g. a Colab GPU runtime), not on a CPU-only box.

A fourth --model_type, mamba_ska_koopman, is also available: same 2 Mamba-2 +
2 SKA layout as mamba_ska_swiglu, but with the paper's actual Spectral
Koopman MLP (Sec 3.3) instead of SwiGLU. This is NOT part of Table 2's
literal protocol (Sec 4.1 specifies SwiGLU for all three models) -- it's an
exploratory variant for comparing the Koopman MLP's effect on the same task,
holding the sequence-layer layout identical to mamba_ska_swiglu.

Usage:
    python -m experimentation.experiments.table2 --model_type mamba_ska_swiglu
    python -m experimentation.experiments.table2 --model_type mamba_attn
    python -m experimentation.experiments.table2 --model_type mamba_only
    python -m experimentation.experiments.table2 --model_type mamba_ska_koopman
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from experimentation.experiments.curricula import eval_niah, make_sysprompt, make_toolcall
from koopman_lm.config import build_config, config_hash
from experimentation.training.optim import param_groups
from experimentation.training.amp import amp_for
from experimentation.run.provenance import git_commit, git_dirty_paths
from koopman_lm.models.baselines import (
    build_mamba_attention,
    build_mamba_only,
    build_mamba_ska_koopman,
    build_mamba_ska_swiglu,
)

TABLE2_SEQ_LENS = (64, 128, 256, 512, 1024, 2048, 4096)
TRAIN_SEQ_LEN = 64

# The paper's three Table 2 variants (Sec 4.1 -- all use SwiGLU MLPs).
TABLE2_MODEL_TYPES = ("mamba_only", "mamba_attn", "mamba_ska_swiglu")

MODEL_BUILDERS = {
    "mamba_only": build_mamba_only,
    "mamba_attn": build_mamba_attention,
    "mamba_ska_swiglu": build_mamba_ska_swiglu,
    # Not part of Table 2's literal protocol (which specifies SwiGLU for all
    # three models) -- an exploratory variant swapping in the paper's actual
    # "Echo" MLP (Sec 3.3) to see whether it helps on the same NIAH task.
    "mamba_ska_koopman": build_mamba_ska_koopman,
}


def _make_toolcall_batch(n: int, step: int, args, seed_offset: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    return make_toolcall(
        n, TRAIN_SEQ_LEN, num_keys=args.toolcall_keys, num_queries=args.toolcall_queries,
        vocab_size=args.task_vocab_size, overwrite_prob=args.overwrite_prob,
        seed=args.seed + step + seed_offset,
    )


def _make_sysprompt_batch(n: int, step: int, args, seed_offset: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    return make_sysprompt(
        n, TRAIN_SEQ_LEN, num_vars=args.sysprompt_vars, vocab_size=args.task_vocab_size,
        num_decoys=args.sysprompt_decoys, seed=args.seed + step + seed_offset,
    )


def make_train_batch(step: int, args) -> tuple[torch.Tensor, torch.Tensor]:
    """Sec 4.1: "mixed dataset of systemprompt and tool-trace examples."

    --curriculum batch_mixed (default): every batch is half toolcall + half
    sysprompt examples, concatenated, so each gradient step already reflects
    both tasks. --curriculum mixed: step-alternates whole batches between the
    two instead -- this measurably interfered with convergence (both losses
    plateaued flat for thousands of steps despite each task training fine in
    isolation) and is kept only for comparison. --curriculum toolcall /
    sysprompt: train on only one, the isolation diagnostic that identified
    the interference in the first place."""
    if args.curriculum == "toolcall":
        return _make_toolcall_batch(args.batch_size, step, args)
    if args.curriculum == "sysprompt":
        return _make_sysprompt_batch(args.batch_size, step, args)
    if args.curriculum == "mixed":
        use_toolcall = step % 2 == 0
        return _make_toolcall_batch(args.batch_size, step, args) if use_toolcall \
            else _make_sysprompt_batch(args.batch_size, step, args)
    # batch_mixed (default)
    half = args.batch_size // 2
    tc_inputs, tc_labels = _make_toolcall_batch(half, step, args)
    sp_inputs, sp_labels = _make_sysprompt_batch(args.batch_size - half, step, args, seed_offset=1)
    return torch.cat([tc_inputs, sp_inputs], dim=0), torch.cat([tc_labels, sp_labels], dim=0)


def build_model(model_type: str, cfg, koopman_mlp_expand: float | None = None):
    if model_type not in MODEL_BUILDERS:
        raise ValueError(f"Unknown model_type: {model_type!r} (expected {sorted(MODEL_BUILDERS)})")
    if model_type == "mamba_ska_koopman":
        return build_mamba_ska_koopman(cfg, mlp_expand=koopman_mlp_expand)
    return MODEL_BUILDERS[model_type](cfg)


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def run_table2_eval(model, device, eval_batch, task_vocab_size, step) -> dict[int, float]:
    """Zero-shot NIAH at all Table 2 sequence lengths -- never trained on
    (make_niah is a different generator entirely from make_toolcall/make_sysprompt)."""
    model.eval()
    results = {}
    print(f"  [step {step}] NIAH length-generalization (zero-shot):")
    for seq_len in TABLE2_SEQ_LENS:
        acc = eval_niah(
            model, batch=eval_batch, seq_len=seq_len,
            vocab_size=task_vocab_size, device=device, seed=9999,
        )
        results[seq_len] = acc
        tag = " (train seq_len)" if seq_len == TRAIN_SEQ_LEN else ""
        print(f"    seq={seq_len:<5d}{tag}: {acc:.4f}")
    model.train()
    return results


def save_checkpoint(output_dir, step, model, optimizer, scheduler, cfg, model_type, results=None):
    d = Path(output_dir) / f"step_{step}"
    d.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), d / "model.pt")
    torch.save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()}, d / "optimizer.pt")
    # code_id/dirty: see training/train.py::checkpoint_meta (provenance 4.1).
    meta = {"step": step, "cfg": cfg, "cfg_hash": config_hash(cfg),
            "model_type": model_type,
            "code_id": git_commit(), "dirty": bool(git_dirty_paths())}
    if results:
        meta["niah_table2"] = results
    torch.save(meta, d / "meta.pt")
    if results:
        with open(d / "table2_results.json", "w", encoding="utf-8") as f:
            json.dump({"step": step, "model_type": model_type, "niah_table2": results}, f, indent=2)
    print(f"  Checkpoint -> {d}  (hash={config_hash(cfg)[:8]})")


def _table2_config(model_size: str):
    """The registry config, with compute_precision declared as fp16.

    This trainer has always run fp16 with a GradScaler while the other three run
    bf16, and that fact lived only in a hardcoded `dtype=torch.float16` line. It is
    now on the config, so it travels with the checkpoint (meta.pt carries cfg) and
    is legible from the run directory rather than from this file.

    Declared rather than inherited on purpose: every registry config defaults to
    bf16, so reading the default would silently switch Table 2 to bf16 and move its
    numbers. The design's wording for this step is that table2's fp16 "becomes
    visible in the config rather than buried in the trainer" -- preservation plus
    legibility, not a change.

    Note config_hash therefore differs from the same registry name built plainly.
    That is correct: this IS a different configuration, and it always was; the
    difference simply used to be invisible.
    """
    import dataclasses

    from koopman_lm.config import KoopmanLMConfig

    base = build_config(model_size)
    return KoopmanLMConfig(
        **dict(dataclasses.asdict(base), compute_precision='fp16'))


def load_checkpoint(resume_from, model, optimizer, scheduler):
    p = Path(resume_from)
    model.load_state_dict(torch.load(p / "model.pt", map_location="cpu", weights_only=True))
    opt = torch.load(p / "optimizer.pt", map_location="cpu", weights_only=True)
    optimizer.load_state_dict(opt["optimizer"])
    scheduler.load_state_dict(opt["scheduler"])
    meta = torch.load(p / "meta.pt", map_location="cpu", weights_only=False)
    print(f"  Resumed from {p} at step {meta['step']}")
    return meta["step"]


def train(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model: {args.model_type}  Config: {args.model_size}")

    cfg = _table2_config(args.model_size)
    print(f"Config hash: {config_hash(cfg)[:8]}")

    model = build_model(args.model_type, cfg, koopman_mlp_expand=args.koopman_mlp_expand).to(device)
    total = count_params(model)
    print(f"Parameters: {total:,} ({total / 1e6:.2f}M)")

    # Route through the shared decay/no-decay policy (see
    # experimentation.training.optim.param_groups) instead of flat model.parameters().
    # A flat AdamW(weight_decay=0.01) decayed norms, biases, embeddings, and the
    # Mamba state parameters (A_log, D, dt_bias) too -- this is the same bug
    # train.py's _param_groups was written to avoid. Published Table 2 numbers
    # were produced under the old, unfiltered-decay optimizer and are superseded.
    optimizer = torch.optim.AdamW(
        param_groups(model, weight_decay=0.01),
        lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)

    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(args.warmup_steps, 1)
        t = (step - args.warmup_steps) / max(args.max_steps - args.warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * t))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    start_step = 0
    if args.resume_from:
        start_step = load_checkpoint(args.resume_from, model, optimizer, scheduler)

    # This trainer has always run fp16 with a scaler, unlike the other three,
    # and that fact lived only in this line. It is now declared on the config
    # (see _table2_config), so the precision travels with the checkpoint and is
    # legible from the run directory. Behaviour is unchanged on purpose: reading
    # the registry default instead would silently switch Table 2 to bf16 and move
    # its numbers.
    autocast, scaler = amp_for(cfg, "cuda", enabled=(device.type == "cuda"))

    model.train()
    t0 = time.time()
    # Per-task running loss, tracked regardless of --curriculum so the log is
    # always informative (this visibility is what caught two real bugs this
    # session -- a naive single "loss" number would have hidden both).
    loss_sum = {"toolcall": 0.0, "sysprompt": 0.0}
    loss_count = {"toolcall": 0, "sysprompt": 0}
    half = args.batch_size // 2

    for step in range(start_step + 1, args.max_steps + 1):
        inputs, labels = make_train_batch(step, args)
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast:
            out = model(input_ids=inputs)
            logits = out["logits"]
            loss = F.cross_entropy(
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

        with torch.no_grad():
            if args.curriculum == "toolcall":
                loss_sum["toolcall"] += loss.item(); loss_count["toolcall"] += 1
            elif args.curriculum == "sysprompt":
                loss_sum["sysprompt"] += loss.item(); loss_count["sysprompt"] += 1
            elif args.curriculum == "mixed":
                task = "toolcall" if step % 2 == 0 else "sysprompt"
                loss_sum[task] += loss.item(); loss_count[task] += 1
            else:  # batch_mixed: slice the already-computed logits, no extra forward pass
                tc_loss = F.cross_entropy(
                    logits[:half, :-1].reshape(-1, logits.size(-1)), labels[:half, 1:].reshape(-1), ignore_index=-100)
                sp_loss = F.cross_entropy(
                    logits[half:, :-1].reshape(-1, logits.size(-1)), labels[half:, 1:].reshape(-1), ignore_index=-100)
                loss_sum["toolcall"] += tc_loss.item(); loss_count["toolcall"] += 1
                loss_sum["sysprompt"] += sp_loss.item(); loss_count["sysprompt"] += 1

        if step % args.log_every == 0:
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - t0
            parts = []
            for name in ("toolcall", "sysprompt"):
                if loss_count[name] > 0:
                    parts.append(f"{name}_loss {loss_sum[name] / loss_count[name]:.4f}")
            print(f"step {step:>5d}/{args.max_steps}  " + "  ".join(parts) +
                  f"  lr {lr:.2e}  {elapsed:.0f}s")
            loss_sum = {"toolcall": 0.0, "sysprompt": 0.0}
            loss_count = {"toolcall": 0, "sysprompt": 0}

        if step % args.eval_every == 0 or step == args.max_steps:
            results = run_table2_eval(model, device, args.eval_batch, args.task_vocab_size, step)
            save_checkpoint(args.output_dir, step, model, optimizer, scheduler, cfg, args.model_type, results)

    final_dir = Path(args.output_dir) / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    last = Path(args.output_dir) / f"step_{args.max_steps}"
    for fname in ("model.pt", "meta.pt"):
        shutil.copy2(last / fname, final_dir / fname)

    print(f"\nDone. Checkpoints in {args.output_dir}/")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Table 2: NIAH length generalization")
    p.add_argument("--model_type", type=str, default="mamba_ska_swiglu", choices=sorted(MODEL_BUILDERS))
    p.add_argument("--model_size", type=str, default="1m")
    p.add_argument("--koopman_mlp_expand", type=float, default=5.0,
                    help="mamba_ska_koopman only: hidden-dim expansion for the Spectral Koopman "
                         "MLP, overriding the shared cfg.mlp_expand (2.667). SpectralKoopmanMLP has "
                         "2 weight matrices vs SwiGLUMLP's 3, so at the shared default it lands at "
                         "~0.74M total vs build_mamba_ska_swiglu's 933,732 -- not a matched-budget "
                         "comparison. 5.0 (d_k=640) computes to ~1.00M, matching the paper's ~1M "
                         "sub-million-model scale; the shared backbone (embeddings/Mamba-2/SKA) is "
                         "unaffected by this flag either way.")
    p.add_argument("--task_vocab_size", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup_steps", type=int, default=200)
    p.add_argument("--max_steps", type=int, default=6000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", type=str, default="./table2-out")
    p.add_argument("--resume_from", type=str, default=None)
    p.add_argument("--eval_every", type=int, default=1000)
    p.add_argument("--eval_batch", type=int, default=256)
    p.add_argument("--log_every", type=int, default=200)
    # Training curriculum knobs (Sec 4.1's "systemprompt and tool-trace" stand-in).
    p.add_argument("--curriculum", type=str, default="batch_mixed",
                    choices=("batch_mixed", "mixed", "toolcall", "sysprompt"),
                    help="'batch_mixed' (default): each batch is half toolcall + half sysprompt, "
                         "concatenated -- one blended gradient step per update. 'mixed': step-alternates "
                         "whole batches instead (measurably interfered with convergence, kept for "
                         "comparison). 'toolcall'/'sysprompt': train on only one -- the isolation "
                         "diagnostic that identified the interference.")
    p.add_argument("--toolcall_keys", type=int, default=8)
    p.add_argument("--toolcall_queries", type=int, default=4)
    p.add_argument("--overwrite_prob", type=float, default=0.3)
    p.add_argument("--sysprompt_vars", type=int, default=4)
    p.add_argument("--sysprompt_decoys", type=int, default=0,
                    help="0 by default: SKA's Gram-matrix stats can't distinguish decoy "
                         "bindings from real ones, which measurably stalled training. "
                         "Set >0 for the paper-appendix-faithful 'confusing distractors' version.")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    torch.manual_seed(args.seed)
    train(args)


if __name__ == "__main__":
    main()
