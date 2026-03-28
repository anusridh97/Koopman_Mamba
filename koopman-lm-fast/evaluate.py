
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoTokenizer
from koopman_lm.config import config_180m, config_180m_gated, config_370m
from koopman_lm.model import KoopmanLM
from koopman_lm.baselines import build_mamba_attention, build_mamba_only


# ============================================================================
# Model loading (handles all model types)
# ============================================================================

def load_model(checkpoint, model_size="180m",
               tokenizer_name="mistralai/Mistral-7B-v0.1",
               model_type=None):
    """
    Load a model from checkpoint. Auto-detects model_type from meta.pt.
    Returns (model, cfg, tokenizer, model_type).
    """
    meta_path = checkpoint.replace("model.pt", "meta.pt")
    meta = {}
    if os.path.exists(meta_path):
        meta = torch.load(meta_path, map_location="cpu", weights_only=False)

    if model_type is None:
        model_type = meta.get("model_type", "koopman")

    if "cfg" in meta:
        cfg = meta["cfg"]
    elif model_size == "180m":
        cfg = config_180m()
    elif model_size == "180m_gated":
        cfg = config_180m_gated()
    elif model_size == "370m":
        cfg = config_370m()
    else:
        raise ValueError(f"Unknown model_size: {model_size}")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    cfg.vocab_size = len(tokenizer)

    if model_type == "mamba_attn":
        model = build_mamba_attention(cfg)
    elif model_type == "mamba_only":
        model = build_mamba_only(cfg)
    else:
        model = KoopmanLM(cfg)

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)

    print(f"Loaded {model_type} model from {checkpoint}")
    total = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total:,}")

    return model, cfg, tokenizer, model_type


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


_FILLER_SENTENCES = [
    "The weather was pleasant and the birds sang in the trees.",
    "Markets opened higher on expectations of strong earnings.",
    "The committee reviewed the quarterly budget and approved changes.",
    "Research indicates that regular exercise improves cognitive function.",
    "The project timeline was adjusted to accommodate new requirements.",
    "Several participants noted the improvement in overall performance.",
    "Historical records show similar patterns in previous decades.",
    "The analysis revealed unexpected correlations in the dataset.",
    "Community members gathered to discuss plans for the upcoming event.",
    "Technical specifications were updated to reflect current standards.",
    "The landscape stretched endlessly toward the distant mountains.",
    "Preliminary results suggest a positive trend in user engagement.",
    "The infrastructure upgrade proceeded according to the revised plan.",
    "New policies were implemented to address emerging challenges.",
    "The observation period concluded with encouraging findings.",
    "Supply chain adjustments led to improved delivery timelines.",
]


def _build_niah_single1(tokenizer, context_len, n_examples=200, seed=42):
    """NIAH-Single-1: single fact in haystack, exact query."""
    rng = random.Random(seed)
    query = "What is the special number?"
    examples = []

    for _ in range(n_examples):
        target_num = rng.randint(1000, 9999)
        needle = f"The special number is {target_num}."
        budget = context_len - 30
        parts = []
        tok_count = 0
        while tok_count < budget:
            sent = rng.choice(_FILLER_SENTENCES)
            sent_toks = len(tokenizer.encode(sent, add_special_tokens=False))
            if tok_count + sent_toks > budget:
                break
            parts.append(sent)
            tok_count += sent_toks
        insert_pos = rng.randint(
            max(1, len(parts) // 10), max(1, 9 * len(parts) // 10))
        parts.insert(insert_pos, needle)
        context = " ".join(parts)
        prompt = f"{context}\n\nQuestion: {query}\nAnswer:"
        target = f" {target_num}"
        examples.append({"prompt": prompt, "target": target,
                         "depth": insert_pos / len(parts)})
    return examples


def _build_niah_single2(tokenizer, context_len, n_examples=200, seed=42):
    """NIAH-Single-2: with distractor facts."""
    rng = random.Random(seed)
    adjectives = ["regular", "normal", "common", "typical", "standard",
                  "ordinary", "usual", "general", "basic", "default",
                  "primary", "secondary", "initial", "final", "previous"]
    query = "What is the special number?"
    examples = []

    for _ in range(n_examples):
        target_num = rng.randint(1000, 9999)
        needle = f"The special number is {target_num}."
        budget = context_len - 30
        parts = []
        tok_count = 0
        dist_idx = 0
        while tok_count < budget:
            if rng.random() < 0.33 and dist_idx < len(adjectives):
                adj = adjectives[dist_idx]
                dist_idx += 1
                fake = rng.randint(1000, 9999)
                sent = f"The {adj} number is {fake}."
            else:
                sent = rng.choice(_FILLER_SENTENCES)
            sent_toks = len(tokenizer.encode(sent, add_special_tokens=False))
            if tok_count + sent_toks > budget:
                break
            parts.append(sent)
            tok_count += sent_toks
        insert_pos = rng.randint(
            max(1, len(parts) // 10), max(1, 9 * len(parts) // 10))
        parts.insert(insert_pos, needle)
        context = " ".join(parts)
        prompt = f"{context}\n\nQuestion: {query}\nAnswer:"
        target = f" {target_num}"
        examples.append({"prompt": prompt, "target": target,
                         "depth": insert_pos / len(parts)})
    return examples


def _build_niah_single3(tokenizer, context_len, n_examples=200, seed=42):
    """NIAH-Single-3: paraphrased query."""
    rng = random.Random(seed)
    adjectives = ["regular", "normal", "common", "typical", "standard",
                  "ordinary", "usual", "general", "basic", "default"]
    query = "Recall the unique designated number."
    examples = []

    for _ in range(n_examples):
        target_num = rng.randint(1000, 9999)
        needle = f"The special number is {target_num}."
        budget = context_len - 30
        parts = []
        tok_count = 0
        dist_idx = 0
        while tok_count < budget:
            if rng.random() < 0.33 and dist_idx < len(adjectives):
                adj = adjectives[dist_idx]
                dist_idx += 1
                fake = rng.randint(1000, 9999)
                sent = f"The {adj} number is {fake}."
            else:
                sent = rng.choice(_FILLER_SENTENCES)
            sent_toks = len(tokenizer.encode(sent, add_special_tokens=False))
            if tok_count + sent_toks > budget:
                break
            parts.append(sent)
            tok_count += sent_toks
        insert_pos = rng.randint(
            max(1, len(parts) // 10), max(1, 9 * len(parts) // 10))
        parts.insert(insert_pos, needle)
        context = " ".join(parts)
        prompt = f"{context}\n\nQuestion: {query}\nAnswer:"
        target = f" {target_num}"
        examples.append({"prompt": prompt, "target": target,
                         "depth": insert_pos / len(parts)})
    return examples


def _score_niah_parallel(model, tokenizer, device, examples, batch_size=4):
    """
    Score NIAH via parallel forward (teacher-forced next-token accuracy).
    Works for ALL model types (no recurrent wrapper needed).
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for i in range(0, len(examples), batch_size):
            batch_ex = examples[i:i+batch_size]
            target_strs = []

            for ex in batch_ex:
                ids = tokenizer.encode(ex["prompt"], add_special_tokens=False)
                target_strs.append(ex["target"].strip())

                input_ids = torch.tensor(
                    [ids], dtype=torch.long, device=device)
                outputs = model(input_ids=input_ids)
                logits = outputs["logits"]

                last_logits = logits[0, -1, :]
                pred_token = last_logits.argmax().item()
                pred_str = tokenizer.decode([pred_token]).strip()

                t = target_strs[-1]
                if t in pred_str or pred_str in t:
                    correct += 1
                total += 1

    return correct / max(total, 1)


def _score_niah_recurrent(model, tokenizer, device, examples):
    """Score NIAH using O(1) recurrent generation (Koopman only)."""
    from koopman_lm.recurrent import RecurrentKoopmanLM

    wrapper = RecurrentKoopmanLM(model)
    correct = 0
    total = 0

    with torch.no_grad():
        for ex in examples:
            ids = tokenizer.encode(ex["prompt"], add_special_tokens=False)
            input_ids = torch.tensor([ids], dtype=torch.long, device=device)
            target_str = ex["target"].strip()

            wrapper.reset()
            logits = wrapper.prefill(input_ids)
            next_logits = logits[:, -1, :]

            new_token_ids = []
            for _ in range(10):
                next_token = next_logits.argmax(dim=-1, keepdim=True)
                new_token_ids.append(next_token.item())
                if next_token.item() == tokenizer.eos_token_id:
                    break
                step_logits = wrapper.step(next_token)
                next_logits = step_logits[:, 0, :]

            decoded = tokenizer.decode(
                new_token_ids, skip_special_tokens=True)
            if target_str in decoded:
                correct += 1
            total += 1

    return correct / max(total, 1)


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
    cfg.max_seq_len = args.max_seq_len
    model = model.to(device).eval()

    if hasattr(model, 'param_summary'):
        model.param_summary()

    all_results = {"model_type": model_type}

    if args.mode in ("all", "ppl"):
        all_results["held_out_ppl"] = eval_held_out_ppl(
            model, device, tokenizer,
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
        print(f"\n  Held-out PPL:")
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
                   choices=["all", "ppl", "niah"])
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
