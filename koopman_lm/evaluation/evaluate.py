
"""
evaluate.py -- Unified evaluation for all three model variants.

Loads model_type from checkpoint meta.pt so all models use the exact same
evaluation code, same seeds, same data, same metrics.

Model types (auto-detected from checkpoint):
  koopman     — Mamba-2 + SKA + Koopman MLP
  mamba_attn  — Mamba-2 + Flash Attention + SwiGLU MLP
  mamba_only  — Mamba-2 + SwiGLU MLP (no global retrieval)

Evaluation modes:
  1. Held-out perplexity (WikiText-103 test, same as Mamba evals)
  2. NIAH (Needle-In-A-Haystack) recall at multiple context lengths
  3. COPY task (Ren & Li, 2024)
  4. MQAR-Distributed (You et al., 2024)
  5. Inverse Sequence Matching (Chen et al., 2025)
  6. Memory profiling (O(1) state demonstration, Koopman only)
  7. lm-evaluation-harness benchmarks

Usage:
  # Evaluate a single checkpoint (auto-detects model type)
  python evaluate.py --checkpoint ./koopman-180m-fast/final/model.pt

  # Compare two models side-by-side
  python evaluate.py \\
      --checkpoint ./koopman-180m-fast/final/model.pt \\
      --checkpoint2 ./mamba-attn-180m-fast/final/model.pt \\
      --output comparison_results.json
"""

import argparse
import math
import json
import os
import random
import time
import torch
from torch.utils.data import DataLoader, IterableDataset
import dataclasses
from koopman_lm.evaluation.loader import load_model  # noqa: F401 (re-exported for harness.py)
from koopman_lm.evaluation.tasks.niah import (
    _FILLER_SENTENCES,  # noqa: F401 (kept importable for compatibility)
    _build_niah_single1,
    _build_niah_single2,
    _build_niah_single3,
    _score_niah_parallel,
    _score_niah_recurrent,
)


# ============================================================================
# Held-out perplexity (WikiText-103 test)
# ============================================================================

class WikiTextDataset(IterableDataset):
    def __init__(self, tokenizer=None, max_len=2048):
        from datasets import load_dataset
        self.dataset = load_dataset("wikitext", "wikitext-103-raw-v1",
                                    split="test")
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __iter__(self):
        buffer = []
        for example in self.dataset:
            text = example.get("text", "")
            if not text or text.isspace():
                continue
            tokens = self.tokenizer(text, truncation=False,
                                    add_special_tokens=False)["input_ids"]
            buffer.extend(tokens)
            buffer.append(self.tokenizer.eos_token_id)
            while len(buffer) >= self.max_len + 1:
                chunk = buffer[:self.max_len + 1]
                buffer = buffer[self.max_len:]
                input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                labels = torch.tensor(chunk[1:], dtype=torch.long)
                yield {"input_ids": input_ids, "labels": labels}


def eval_held_out_ppl(model, device, tokenizer, max_seq_len=2048,
                      batch_size=8):
    """Held-out perplexity on WikiText-103 test split."""
    print("\n" + "=" * 60)
    print("Held-out perplexity (WikiText-103)")
    print("=" * 60)

    dataset = WikiTextDataset(tokenizer=tokenizer, max_len=max_seq_len)
    loader = DataLoader(dataset, batch_size=batch_size)

    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs["loss"]
            n_tokens = labels.numel()
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens

    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(min(avg_loss, 20))

    print(f"  Tokens: {total_tokens:,}")
    print(f"  Loss:   {avg_loss:.4f}")
    print(f"  PPL:    {ppl:.2f}")

    return {"loss": avg_loss, "ppl": ppl, "n_tokens": total_tokens}


# ============================================================================
# Held-out perplexity (FineWeb-Edu val shard)
#
# Table 4's caption labels its perplexity column "FineWeb-Edu perplexity"
# while the paper's body text calls the same column "WikiText-103
# perplexity" -- an unresolved contradiction in the paper itself. Rather than
# guess which one Table 4 actually means, we compute both and report them
# separately, clearly labeled.
# ============================================================================

def eval_fineweb_ppl(model, device, held_out_data_dir, max_seq_len=2048,
                     batch_size=8):
    """Held-out perplexity on a disjoint FineWeb-Edu shard (see
    koopman_lm.training.data.pretokenize's --skip_docs for how to produce a
    val shard that doesn't overlap the training corpus)."""
    from koopman_lm.training.data.dataset import MemmapPackedDataset

    print("\n" + "=" * 60)
    print("Held-out perplexity (FineWeb-Edu)")
    print("=" * 60)

    dataset = MemmapPackedDataset(held_out_data_dir, max_seq_len, seed=0)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs["loss"]
            n_tokens = labels.numel()
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens

    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(min(avg_loss, 20))

    print(f"  Tokens: {total_tokens:,}")
    print(f"  Loss:   {avg_loss:.4f}")
    print(f"  PPL:    {ppl:.2f}")

    return {"loss": avg_loss, "ppl": ppl, "n_tokens": total_tokens}


# ============================================================================
# NIAH (Needle-In-A-Haystack)
# ============================================================================

def _validate_context_lens(context_lens):
    """Ensure all context lengths are multiples of 8 for hardware alignment."""
    validated = []
    for cl in context_lens:
        aligned = ((cl + 7) // 8) * 8
        if aligned != cl:
            print(f"  Warning: rounding NIAH context_len {cl} -> {aligned} "
                  f"(must be multiple of 8)")
        validated.append(aligned)
    return validated


def eval_niah(model, device, tokenizer, model_type="koopman",
              batch_size=4, n_examples=200, context_lens=None, seed=42):
    """Run NIAH benchmarks. Works for all model types."""
    if context_lens is None:
        context_lens = [128, 256, 512, 1024, 2048, 4096]

    context_lens = _validate_context_lens(context_lens)

    print("\n" + "=" * 60)
    print(f"NIAH (Needle-In-A-Haystack) — model_type={model_type}")
    print("=" * 60)

    # Use recurrent scoring for Koopman, parallel for everything else
    if model_type == "koopman":
        score_fn = lambda m, tok, dev, exs, bs: _score_niah_recurrent(
            m, tok, dev, exs)
    else:
        score_fn = _score_niah_parallel

    builders = {
        "NIAH-Single-1": _build_niah_single1,
        "NIAH-Single-2": _build_niah_single2,
        "NIAH-Single-3": _build_niah_single3,
    }

    all_results = {}
    for task_name, builder in builders.items():
        all_results[task_name] = {}
        for ctx_len in context_lens:
            examples = builder(tokenizer, ctx_len, n_examples=n_examples,
                               seed=seed)
            acc = score_fn(model, tokenizer, device, examples, batch_size)
            all_results[task_name][ctx_len] = acc * 100.0
            print(f"  {task_name} @ {ctx_len:>5d}: {acc * 100.0:6.1f}%")

    # Summary table
    print(f"\n  {'':20s}", end="")
    for cl in context_lens:
        print(f"  {cl:>5d}", end="")
    print()
    for task_name in builders:
        print(f"  {task_name:20s}", end="")
        for cl in context_lens:
            print(f"  {all_results[task_name][cl]:5.1f}", end="")
        print()

    return all_results


# ============================================================================
# Evaluation orchestrator
# ============================================================================

def evaluate_checkpoint(checkpoint, args, device):
    """Run all requested evaluations on a single checkpoint."""
    print(f"\n{'#' * 60}")
    print(f"# Evaluating: {checkpoint}")
    print(f"{'#' * 60}")

    model, cfg, tokenizer, model_type = load_model(
        checkpoint, args.model_size, args.tokenizer)
    cfg = dataclasses.replace(cfg, max_seq_len=args.max_seq_len)   # frozen: use replace
    model = model.to(device).eval()

    if hasattr(model, 'param_summary'):
        model.param_summary()

    all_results = {"model_type": model_type}

    if args.mode in ("all", "ppl"):
        all_results["held_out_ppl"] = eval_held_out_ppl(
            model, device, tokenizer,
            max_seq_len=args.max_seq_len,
            batch_size=args.batch_size)

    # Only run under "all" if a held-out shard was actually given -- "all"
    # shouldn't error out for callers who haven't set up a FineWeb val shard.
    if args.mode == "fineweb_ppl" or (args.mode == "all" and args.held_out_data_dir):
        all_results["fineweb_ppl"] = eval_fineweb_ppl(
            model, device, args.held_out_data_dir,
            max_seq_len=args.max_seq_len,
            batch_size=args.batch_size)

    if args.mode in ("all", "niah"):
        all_results["niah"] = eval_niah(
            model, device, tokenizer,
            model_type=model_type,
            batch_size=args.batch_size,
            n_examples=args.niah_n_examples,
            context_lens=args.niah_context_lens,
            seed=args.seed)

    del model
    torch.cuda.empty_cache()
    return all_results


def compare_results(results1, results2, output_path=None):
    """Print side-by-side comparison of two model evaluations."""
    mt1 = results1.get("model_type", "model_1")
    mt2 = results2.get("model_type", "model_2")

    print(f"\n{'=' * 60}")
    print(f"COMPARISON: {mt1} vs {mt2}")
    print(f"{'=' * 60}")

    if "held_out_ppl" in results1 and "held_out_ppl" in results2:
        p1 = results1["held_out_ppl"]["ppl"]
        p2 = results2["held_out_ppl"]["ppl"]
        print(f"\n  Held-out PPL (WikiText-103):")
        print(f"    {mt1:20s}: {p1:.2f}")
        print(f"    {mt2:20s}: {p2:.2f}")
        print(f"    {'delta':20s}: {p1 - p2:+.2f}")

    if "fineweb_ppl" in results1 and "fineweb_ppl" in results2:
        p1 = results1["fineweb_ppl"]["ppl"]
        p2 = results2["fineweb_ppl"]["ppl"]
        print(f"\n  Held-out PPL (FineWeb-Edu):")
        print(f"    {mt1:20s}: {p1:.2f}")
        print(f"    {mt2:20s}: {p2:.2f}")
        print(f"    {'delta':20s}: {p1 - p2:+.2f}")

    if "niah" in results1 and "niah" in results2:
        print(f"\n  NIAH accuracy:")
        for task in results1["niah"]:
            if task in results2["niah"]:
                print(f"\n    {task}:")
                for cl in sorted(results1["niah"][task]):
                    v1 = results1["niah"][task].get(cl, 0)
                    v2 = results2["niah"][task].get(cl, 0)
                    delta = v1 - v2
                    print(f"      {cl:>5d}: {v1:5.1f}% vs {v2:5.1f}% "
                          f"({delta:+5.1f})")

    if output_path:
        combined = {"model_1": results1, "model_2": results2}
        with open(output_path, 'w') as f:
            json.dump(combined, f, indent=2, default=str)
        print(f"\n  Results saved to {output_path}")


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--checkpoint2", type=str, default=None,
                   help="Second checkpoint for side-by-side comparison")
    p.add_argument("--model_size", type=str, default="180m")
    p.add_argument("--tokenizer", type=str,
                   default="mistralai/Mistral-7B-v0.1")
    p.add_argument("--mode", type=str, default="all",
                   choices=["all", "ppl", "fineweb_ppl", "niah"])
    p.add_argument("--held_out_data_dir", type=str, default=None,
                   help="disjoint FineWeb-Edu val shard dir (from "
                        "pretokenize.py --skip_docs), required for "
                        "--mode fineweb_ppl")
    p.add_argument("--max_seq_len", type=int, default=2048)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--niah_context_lens", nargs="+", type=int,
                   default=[128, 256, 512, 1024, 2048, 4096])
    p.add_argument("--niah_n_examples", type=int, default=200)
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    if args.mode == "fineweb_ppl" and not args.held_out_data_dir:
        raise SystemExit("--mode fineweb_ppl requires --held_out_data_dir")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    results1 = evaluate_checkpoint(args.checkpoint, args, device)

    if args.checkpoint2:
        results2 = evaluate_checkpoint(args.checkpoint2, args, device)
        compare_results(results1, results2, args.output)
    elif args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(results1, f, indent=2, default=str)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
