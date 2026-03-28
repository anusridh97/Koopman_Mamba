"""
evaluate_retrieval.py -- Retrieval evaluation with zero-shot and fine-tuning.

Evaluates all three model variants on retrieval tasks in two regimes:

  1. Zero-shot: models are evaluated directly (no task-specific training)
  2. Fine-tuned: models are fine-tuned on task-specific training data,
     then evaluated on held-out test data

Fine-tuning uses a PREFIX MASK: loss is computed ONLY on query/answer
tokens (the retrieval output), not on the context/haystack tokens.
This means the model learns to retrieve, not to memorize filler text.

The fine-tuning and evaluation data are generated from DIFFERENT seeds
to prevent data leakage.

Retrieval tasks:
  - NIAH-Single-1: single fact, exact query
  - NIAH-Single-2: single fact + distractors
  - NIAH-Single-3: paraphrased query + distractors
  - KV-Retrieval: key-value pair lookup (synthetic phonebook)
  - Multi-Hop: two-step retrieval requiring composition

Usage:
  # Zero-shot only (no fine-tuning)
  python evaluate_retrieval.py \
      --checkpoint ./koopman-180m-fast/final/model.pt \
      --mode zero_shot

  # Fine-tune then evaluate
  python evaluate_retrieval.py \
      --checkpoint ./koopman-180m-fast/final/model.pt \
      --mode finetune

  # Both zero-shot and fine-tuned (full comparison)
  python evaluate_retrieval.py \
      --checkpoint ./koopman-180m-fast/final/model.pt \
      --mode both

  # Compare all three models
  python evaluate_retrieval.py \
      --checkpoint  ./koopman-180m-fast/final/model.pt \
      --checkpoint2 ./mamba-attn-180m-fast/final/model.pt \
      --checkpoint3 ./mamba-only-180m-fast/final/model.pt \
      --mode both \
      --output retrieval_comparison.json
"""

import argparse
import copy
import json
import math
import os
import random
import time
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from koopman_lm.config import config_180m, config_180m_gated, config_370m
from koopman_lm.model import KoopmanLM
from koopman_lm.baselines import build_mamba_attention, build_mamba_only


# ============================================================================
# Model loading (same as evaluate.py, supports all 3 types)
# ============================================================================

def load_model(checkpoint, model_size="180m",
               tokenizer_name="mistralai/Mistral-7B-v0.1",
               model_type=None):
    """Load model from checkpoint, auto-detecting model_type from meta.pt."""
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
# Retrieval task data generators
# ============================================================================

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

_FIRST_NAMES = [
    "Alice", "Bob", "Carol", "David", "Eve", "Frank", "Grace", "Henry",
    "Iris", "Jack", "Karen", "Leo", "Mia", "Noah", "Olivia", "Paul",
    "Quinn", "Rose", "Sam", "Tina", "Uma", "Victor", "Wendy", "Xavier",
]

_LAST_NAMES = [
    "Smith", "Chen", "Johnson", "Patel", "Williams", "Kim", "Brown",
    "Garcia", "Miller", "Davis", "Wilson", "Anderson", "Thomas", "Taylor",
    "Moore", "Jackson", "Martin", "Lee", "Thompson", "White",
]


def _build_niah_examples(tokenizer, context_len, n_examples, seed,
                         use_distractors=False, paraphrase_query=False):
    """
    Unified NIAH builder. Returns list of dicts with:
      prompt, target, query_start_idx (token index where query begins)
    """
    rng = random.Random(seed)
    adjectives = ["regular", "normal", "common", "typical", "standard",
                  "ordinary", "usual", "general", "basic", "default"]

    if paraphrase_query:
        query = "Recall the unique designated number."
    else:
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
            if use_distractors and rng.random() < 0.33 and dist_idx < len(adjectives):
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

        # Build full text and find where query starts
        query_prefix = f"\n\nQuestion: {query}\nAnswer:"
        full_text = context + query_prefix
        target = f" {target_num}"

        # Tokenize to find query boundary
        context_ids = tokenizer.encode(context, add_special_tokens=False)
        query_start_idx = len(context_ids)

        examples.append({
            "full_text": full_text,
            "target": target,
            "query_start_idx": query_start_idx,
            "depth": insert_pos / len(parts),
        })
    return examples


def _build_kv_retrieval(tokenizer, context_len, n_examples, seed):
    """
    Key-value retrieval: synthetic phonebook lookup.
    Context contains N name-number pairs, query asks for one specific name.
    """
    rng = random.Random(seed)
    examples = []

    for _ in range(n_examples):
        # Determine how many KV pairs fit in the context budget
        budget = context_len - 40  # reserve for query + filler
        pairs = []
        tok_count = 0
        used_names = set()

        while tok_count < budget:
            first = rng.choice(_FIRST_NAMES)
            last = rng.choice(_LAST_NAMES)
            name = f"{first} {last}"
            if name in used_names:
                continue
            used_names.add(name)
            number = f"{rng.randint(100, 999)}-{rng.randint(1000, 9999)}"
            entry = f"{name}: {number}."
            entry_toks = len(tokenizer.encode(entry, add_special_tokens=False))
            if tok_count + entry_toks > budget:
                break
            pairs.append((name, number, entry))
            tok_count += entry_toks

        if len(pairs) < 2:
            continue

        # Pick a random target pair
        target_idx = rng.randint(0, len(pairs) - 1)
        target_name, target_number, _ = pairs[target_idx]

        # Shuffle entries and add some filler
        entries = [p[2] for p in pairs]
        rng.shuffle(entries)

        # Intersperse with a few filler sentences
        parts = []
        for i, entry in enumerate(entries):
            parts.append(entry)
            if rng.random() < 0.2:
                parts.append(rng.choice(_FILLER_SENTENCES))

        context = " ".join(parts)
        query_prefix = f"\n\nQuestion: What is {target_name}'s number?\nAnswer:"
        full_text = context + query_prefix
        target = f" {target_number}"

        context_ids = tokenizer.encode(context, add_special_tokens=False)
        query_start_idx = len(context_ids)

        examples.append({
            "full_text": full_text,
            "target": target,
            "query_start_idx": query_start_idx,
            "depth": target_idx / len(pairs),
        })
    return examples


def _build_multi_hop(tokenizer, context_len, n_examples, seed):
    """
    Multi-hop retrieval: two facts must be composed.
    Fact 1: "The capital of Zaronia is Velthaven."
    Fact 2: "The mayor of Velthaven is Dr. Whitfield."
    Query: "Who is the mayor of the capital of Zaronia?"
    """
    rng = random.Random(seed)

    country_names = [
        "Zaronia", "Belvane", "Cresthia", "Dunmere", "Erathis",
        "Falmoor", "Galdwyn", "Helvoria", "Irthos", "Jandell",
        "Kelmoor", "Lithvane", "Morenth", "Neldara", "Orvalis",
    ]
    city_names = [
        "Velthaven", "Stonecross", "Ashford", "Brighthollow", "Thornwick",
        "Ravenspire", "Oakmere", "Silverdale", "Windbreak", "Ironhurst",
        "Cloudpeak", "Riverstone", "Elmshade", "Frostmere", "Goldcrest",
    ]
    person_names = [
        "Dr. Whitfield", "Prof. Aldridge", "Gen. Thornton", "Lady Carver",
        "Sir Pembroke", "Dr. Nakamura", "Prof. Delgado", "Mayor Ingram",
        "Dir. Castellan", "Gov. Fairbanks", "Col. Vasquez", "Sen. Hartwell",
        "Judge Morin", "Amb. Okafor", "Cmdr. Reyes",
    ]

    examples = []
    for _ in range(n_examples):
        ci = rng.randint(0, len(country_names) - 1)
        ti = rng.randint(0, len(city_names) - 1)
        pi = rng.randint(0, len(person_names) - 1)

        country = country_names[ci]
        city = city_names[ti]
        person = person_names[pi]

        fact1 = f"The capital of {country} is {city}."
        fact2 = f"The mayor of {city} is {person}."

        budget = context_len - 50
        parts = []
        tok_count = 0
        while tok_count < budget:
            sent = rng.choice(_FILLER_SENTENCES)
            sent_toks = len(tokenizer.encode(sent, add_special_tokens=False))
            if tok_count + sent_toks > budget:
                break
            parts.append(sent)
            tok_count += sent_toks

        if len(parts) < 4:
            continue

        # Insert facts at different positions (separated)
        pos1 = rng.randint(1, len(parts) // 3)
        parts.insert(pos1, fact1)
        pos2 = rng.randint(2 * len(parts) // 3, len(parts) - 1)
        parts.insert(pos2, fact2)

        context = " ".join(parts)
        query_prefix = (f"\n\nQuestion: Who is the mayor of the capital "
                        f"of {country}?\nAnswer:")
        full_text = context + query_prefix
        target = f" {person}"

        context_ids = tokenizer.encode(context, add_special_tokens=False)
        query_start_idx = len(context_ids)

        examples.append({
            "full_text": full_text,
            "target": target,
            "query_start_idx": query_start_idx,
            "depth": 0.5,
        })
    return examples


# ============================================================================
# Task registry
# ============================================================================

RETRIEVAL_TASKS = {
    "NIAH-Single-1": lambda tok, cl, n, s: _build_niah_examples(
        tok, cl, n, s, use_distractors=False, paraphrase_query=False),
    "NIAH-Single-2": lambda tok, cl, n, s: _build_niah_examples(
        tok, cl, n, s, use_distractors=True, paraphrase_query=False),
    "NIAH-Single-3": lambda tok, cl, n, s: _build_niah_examples(
        tok, cl, n, s, use_distractors=True, paraphrase_query=True),
    "KV-Retrieval": _build_kv_retrieval,
    "Multi-Hop": _build_multi_hop,
}


# ============================================================================
# Prefix-masked fine-tuning dataset
# ============================================================================

class RetrievalFineTuneDataset(Dataset):
    """
    Dataset for retrieval fine-tuning with prefix masking.

    Each example has:
      - input_ids: full sequence (context + query + answer)
      - labels: -100 for context tokens (masked), real token IDs for
                query + answer tokens (where loss is computed)

    The prefix mask ensures the model only learns to do retrieval
    (respond to queries), not to memorize the filler/haystack text.
    """

    def __init__(self, examples, tokenizer, max_seq_len):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.samples = []

        for ex in examples:
            full_text = ex["full_text"] + ex["target"]
            all_ids = tokenizer.encode(full_text, add_special_tokens=False)

            if len(all_ids) > max_seq_len:
                all_ids = all_ids[:max_seq_len]

            query_start = ex["query_start_idx"]

            input_ids = all_ids[:-1]
            labels = all_ids[1:]

            # Prefix mask: set labels to -100 for all positions BEFORE
            # the query. Only query + answer tokens contribute to loss.
            # The -1 accounts for the shift between input_ids and labels.
            mask_end = max(0, query_start - 1)
            labels[:mask_end] = [-100] * mask_end

            self.samples.append({
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def _collate_retrieval(batch):
    """Pad batch to max length, with -100 for label padding."""
    max_len = max(b["input_ids"].size(0) for b in batch)
    input_ids = []
    labels = []
    for b in batch:
        pad_len = max_len - b["input_ids"].size(0)
        input_ids.append(F.pad(b["input_ids"], (0, pad_len), value=0))
        labels.append(F.pad(b["labels"], (0, pad_len), value=-100))
    return {
        "input_ids": torch.stack(input_ids),
        "labels": torch.stack(labels),
    }


# ============================================================================
# Zero-shot evaluation (parallel forward, greedy decode)
# ============================================================================

@torch.no_grad()
def eval_zero_shot(model, tokenizer, device, examples, batch_size=4):
    """
    Zero-shot retrieval accuracy via greedy next-token prediction.
    Returns accuracy (fraction of examples where first predicted token
    matches the target).
    """
    model.eval()
    correct = 0
    total = 0

    for ex in examples:
        full_ids = tokenizer.encode(ex["full_text"], add_special_tokens=False)
        input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
        outputs = model(input_ids=input_ids)
        logits = outputs["logits"]

        last_logits = logits[0, -1, :]
        pred_token = last_logits.argmax().item()
        pred_str = tokenizer.decode([pred_token]).strip()
        target_str = ex["target"].strip()

        if target_str in pred_str or pred_str in target_str:
            correct += 1
        total += 1

    return correct / max(total, 1)


# ============================================================================
# Fine-tuning loop (prefix-masked)
# ============================================================================

def finetune_model(model, train_dataset, device, args):
    """
    Fine-tune a model on retrieval examples with prefix-masked loss.

    Returns the fine-tuned model (a deep copy; original is not modified).
    """
    # Deep copy so we don't modify the pretrained checkpoint
    ft_model = copy.deepcopy(model)
    ft_model.to(device)
    ft_model.train()

    loader = DataLoader(
        train_dataset,
        batch_size=args.ft_batch_size,
        shuffle=True,
        collate_fn=_collate_retrieval,
        drop_last=True,
    )

    optimizer = torch.optim.AdamW(
        ft_model.parameters(),
        lr=args.ft_lr,
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=min(100, args.ft_steps // 10),
        num_training_steps=args.ft_steps,
    )

    autocast_ctx = torch.amp.autocast('cuda', dtype=torch.bfloat16,
                                       enabled=device.type == 'cuda')

    step = 0
    epoch = 0
    t_start = time.time()

    while step < args.ft_steps:
        for batch in loader:
            if step >= args.ft_steps:
                break

            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            with autocast_ctx:
                outputs = ft_model(input_ids=input_ids, labels=labels)
                loss = outputs["loss"]

            loss.backward()

            torch.nn.utils.clip_grad_norm_(ft_model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            if step % max(1, args.ft_steps // 10) == 0:
                elapsed = time.time() - t_start
                print(f"    ft step {step:>5d}/{args.ft_steps} | "
                      f"loss {loss.item():.4f} | {elapsed:.0f}s")

        epoch += 1

    ft_model.eval()
    return ft_model


# ============================================================================
# Full retrieval evaluation pipeline
# ============================================================================

def evaluate_retrieval(checkpoint, args, device):
    """
    Run retrieval evaluation on one checkpoint.

    Returns dict with zero-shot and/or fine-tuned results per task and
    context length.
    """
    print(f"\n{'#' * 60}")
    print(f"# Retrieval eval: {checkpoint}")
    print(f"{'#' * 60}")

    model, cfg, tokenizer, model_type = load_model(
        checkpoint, args.model_size, args.tokenizer)
    cfg.max_seq_len = args.max_seq_len
    model = model.to(device).eval()

    if hasattr(model, 'param_summary'):
        model.param_summary()

    results = {"model_type": model_type}

    # Select tasks
    tasks = args.tasks if args.tasks else list(RETRIEVAL_TASKS.keys())

    context_lens = args.context_lens

    # ---- Zero-shot evaluation ----
    if args.mode in ("zero_shot", "both"):
        print(f"\n{'=' * 60}")
        print(f"Zero-shot retrieval — {model_type}")
        print(f"{'=' * 60}")

        zs_results = {}
        for task_name in tasks:
            if task_name not in RETRIEVAL_TASKS:
                print(f"  Warning: unknown task '{task_name}', skipping")
                continue
            builder = RETRIEVAL_TASKS[task_name]
            zs_results[task_name] = {}

            for ctx_len in context_lens:
                # Use eval seed for zero-shot
                examples = builder(tokenizer, ctx_len, args.n_examples,
                                   seed=args.eval_seed)
                acc = eval_zero_shot(model, tokenizer, device, examples)
                zs_results[task_name][ctx_len] = acc * 100.0
                print(f"  {task_name} @ {ctx_len:>5d}: {acc * 100.0:6.1f}%")

        results["zero_shot"] = zs_results

        # Print summary table
        _print_summary_table("Zero-shot", tasks, context_lens, zs_results)

    # ---- Fine-tuned evaluation ----
    if args.mode in ("finetune", "both"):
        print(f"\n{'=' * 60}")
        print(f"Fine-tuned retrieval — {model_type}")
        print(f"  FT steps: {args.ft_steps}, FT lr: {args.ft_lr}, "
              f"FT batch: {args.ft_batch_size}")
        print(f"  Train seed: {args.train_seed} (different from eval)")
        print(f"  Eval seed:  {args.eval_seed}")
        print(f"{'=' * 60}")

        ft_results = {}
        for task_name in tasks:
            if task_name not in RETRIEVAL_TASKS:
                continue
            builder = RETRIEVAL_TASKS[task_name]
            ft_results[task_name] = {}

            for ctx_len in context_lens:
                print(f"\n  --- {task_name} @ {ctx_len} ---")

                # Generate TRAINING data with train_seed
                train_examples = builder(
                    tokenizer, ctx_len, args.ft_n_train,
                    seed=args.train_seed)

                # Generate EVAL data with eval_seed (different!)
                eval_examples = builder(
                    tokenizer, ctx_len, args.n_examples,
                    seed=args.eval_seed)

                # Build prefix-masked dataset
                train_ds = RetrievalFineTuneDataset(
                    train_examples, tokenizer, args.max_seq_len)

                # Verify prefix masking is working
                sample = train_ds[0]
                n_masked = (sample["labels"] == -100).sum().item()
                n_active = (sample["labels"] != -100).sum().item()
                print(f"    Train examples: {len(train_ds)}, "
                      f"masked tokens: {n_masked}, "
                      f"active (query) tokens: {n_active}")

                # Fine-tune (deep copy)
                print(f"    Fine-tuning...")
                ft_model = finetune_model(model, train_ds, device, args)

                # Evaluate fine-tuned model
                acc = eval_zero_shot(
                    ft_model, tokenizer, device, eval_examples)
                ft_results[task_name][ctx_len] = acc * 100.0
                print(f"    Result: {acc * 100.0:6.1f}%")

                # Free the fine-tuned copy
                del ft_model
                torch.cuda.empty_cache()

        results["fine_tuned"] = ft_results

        _print_summary_table("Fine-tuned", tasks, context_lens, ft_results)

    del model
    torch.cuda.empty_cache()
    return results


def _print_summary_table(regime_name, tasks, context_lens, results):
    """Print a formatted results table."""
    print(f"\n  {regime_name} summary:")
    print(f"  {'':20s}", end="")
    for cl in context_lens:
        print(f"  {cl:>5d}", end="")
    print()
    for task_name in tasks:
        if task_name not in results:
            continue
        print(f"  {task_name:20s}", end="")
        for cl in context_lens:
            v = results[task_name].get(cl, 0)
            print(f"  {v:5.1f}", end="")
        print()


# ============================================================================
# Multi-model comparison
# ============================================================================

def compare_retrieval_results(all_results, output_path=None):
    """Print side-by-side comparison of retrieval results across models."""
    model_names = [r.get("model_type", f"model_{i}")
                   for i, r in enumerate(all_results)]

    for regime in ("zero_shot", "fine_tuned"):
        regime_data = [r.get(regime) for r in all_results]
        if not any(regime_data):
            continue

        regime_label = regime.replace("_", "-").title()
        print(f"\n{'=' * 70}")
        print(f"COMPARISON: {regime_label}")
        print(f"{'=' * 70}")

        # Collect all tasks and context lens
        all_tasks = set()
        all_cls = set()
        for rd in regime_data:
            if rd:
                for task, cls in rd.items():
                    all_tasks.add(task)
                    all_cls.update(cls.keys())

        all_cls = sorted(all_cls)

        for task in sorted(all_tasks):
            print(f"\n  {task}:")
            # Header
            print(f"    {'ctx_len':>8s}", end="")
            for mn in model_names:
                print(f"  {mn:>12s}", end="")
            print(f"  {'best':>8s}")

            for cl in all_cls:
                print(f"    {cl:>8d}", end="")
                values = []
                for i, rd in enumerate(regime_data):
                    v = 0.0
                    if rd and task in rd:
                        v = rd[task].get(cl, 0.0)
                    values.append(v)
                    print(f"  {v:11.1f}%", end="")

                # Mark best
                best_idx = max(range(len(values)), key=lambda i: values[i])
                print(f"  {model_names[best_idx]:>8s}")

    if output_path:
        combined = {mn: r for mn, r in zip(model_names, all_results)}
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(combined, f, indent=2, default=str)
        print(f"\n  Results saved to {output_path}")


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Retrieval evaluation with zero-shot and fine-tuning")

    # Checkpoints (up to 3 for comparison)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--checkpoint2", type=str, default=None)
    p.add_argument("--checkpoint3", type=str, default=None)

    # Model config
    p.add_argument("--model_size", type=str, default="180m")
    p.add_argument("--tokenizer", type=str,
                   default="mistralai/Mistral-7B-v0.1")
    p.add_argument("--max_seq_len", type=int, default=2048)

    # Evaluation mode
    p.add_argument("--mode", type=str, default="both",
                   choices=["zero_shot", "finetune", "both"])

    # Tasks and context lengths
    p.add_argument("--tasks", nargs="+", type=str, default=None,
                   help="Tasks to evaluate (default: all)")
    p.add_argument("--context_lens", nargs="+", type=int,
                   default=[256, 512, 1024, 2048])
    p.add_argument("--n_examples", type=int, default=200,
                   help="Number of eval examples per task/context_len")

    # Fine-tuning hyperparameters
    p.add_argument("--ft_steps", type=int, default=500,
                   help="Fine-tuning optimizer steps")
    p.add_argument("--ft_lr", type=float, default=1e-4,
                   help="Fine-tuning learning rate")
    p.add_argument("--ft_batch_size", type=int, default=8,
                   help="Fine-tuning batch size")
    p.add_argument("--ft_n_train", type=int, default=1000,
                   help="Number of fine-tuning training examples")

    # Seeds (different for train vs eval to prevent leakage)
    p.add_argument("--train_seed", type=int, default=1337,
                   help="Seed for generating fine-tuning training data")
    p.add_argument("--eval_seed", type=int, default=42,
                   help="Seed for generating evaluation data")

    # Output
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--batch_size", type=int, default=4)

    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.eval_seed)
    random.seed(args.eval_seed)

    checkpoints = [args.checkpoint]
    if args.checkpoint2:
        checkpoints.append(args.checkpoint2)
    if args.checkpoint3:
        checkpoints.append(args.checkpoint3)

    all_results = []
    for ckpt in checkpoints:
        results = evaluate_retrieval(ckpt, args, device)
        all_results.append(results)

    if len(all_results) > 1:
        compare_retrieval_results(all_results, args.output)
    elif args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(all_results[0], f, indent=2, default=str)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
