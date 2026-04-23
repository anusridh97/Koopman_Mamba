"""
niah_quick.py -- Quick Needle-In-A-Haystack evaluation using parallel forward.

No recurrent wrapper needed, avoids stride alignment issues.
Uses the standard model forward pass to compute logits over the full
sequence, then checks if the model assigns highest probability to
the correct answer tokens.

Three difficulty levels matching Mamba-3 paper Table 1:
  Single-1: Single fact, single query. Easy baseline.
  Single-2: Single fact among similar-format distractors.
  Single-3: Single fact, paraphrased query.

Evaluated at multiple context lengths and needle depths.

Usage:
  python niah_quick.py --checkpoint ./koopman-50m-output/step_24000/model.pt --model_size 50m
"""

import argparse
import random
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from koopman_lm.config import config_180m, config_180m_gated, config_370m, config_50m
from koopman_lm.model import KoopmanLM


FILLER_SENTENCES = [
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
    "Documentation was updated to include the latest revisions.",
    "The review board examined compliance with established protocols.",
    "Annual reporting requirements were met ahead of schedule.",
    "Quality assurance testing confirmed the expected outcomes.",
]

DISTRACTOR_ADJECTIVES = [
    "regular", "normal", "common", "typical", "standard",
    "ordinary", "usual", "general", "basic", "default",
    "primary", "secondary", "initial", "final", "previous",
]

PARAPHRASED_QUERIES = [
    "Recall the unique designated number.",
    "What was the distinct identified number?",
    "State the particular assigned number.",
    "Provide the singular marked number.",
    "Report the exclusive labeled number.",
]


def build_single1(tokenizer, context_len, n_examples=100, seed=42):
    """Single fact, single query, filler haystack."""
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
            sent = rng.choice(FILLER_SENTENCES)
            stoks = len(tokenizer.encode(sent, add_special_tokens=False))
            if tok_count + stoks > budget:
                break
            parts.append(sent)
            tok_count += stoks

        pos = rng.randint(max(1, len(parts) // 10), max(1, 9 * len(parts) // 10))
        parts.insert(pos, needle)
        depth = pos / len(parts)

        prompt = " ".join(parts) + f"\n\nQuestion: {query}\nAnswer:"
        target = f" {target_num}"
        examples.append({"prompt": prompt, "target": target, "depth": depth})

    return examples


def build_single2(tokenizer, context_len, n_examples=100, seed=42):
    """Single fact among similar-format distractors."""
    rng = random.Random(seed)
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
            if rng.random() < 0.33 and dist_idx < len(DISTRACTOR_ADJECTIVES):
                adj = DISTRACTOR_ADJECTIVES[dist_idx]
                dist_idx += 1
                fake = rng.randint(1000, 9999)
                sent = f"The {adj} number is {fake}."
            else:
                sent = rng.choice(FILLER_SENTENCES)
            stoks = len(tokenizer.encode(sent, add_special_tokens=False))
            if tok_count + stoks > budget:
                break
            parts.append(sent)
            tok_count += stoks

        pos = rng.randint(max(1, len(parts) // 10), max(1, 9 * len(parts) // 10))
        parts.insert(pos, needle)
        depth = pos / len(parts)

        prompt = " ".join(parts) + f"\n\nQuestion: {query}\nAnswer:"
        target = f" {target_num}"
        examples.append({"prompt": prompt, "target": target, "depth": depth})

    return examples


def build_single3(tokenizer, context_len, n_examples=100, seed=42):
    """Single fact, paraphrased query, with distractors."""
    rng = random.Random(seed)
    examples = []

    for _ in range(n_examples):
        target_num = rng.randint(1000, 9999)
        needle = f"The special number is {target_num}."
        query = rng.choice(PARAPHRASED_QUERIES)

        budget = context_len - 30
        parts = []
        tok_count = 0
        dist_idx = 0
        while tok_count < budget:
            if rng.random() < 0.33 and dist_idx < len(DISTRACTOR_ADJECTIVES):
                adj = DISTRACTOR_ADJECTIVES[dist_idx]
                dist_idx += 1
                fake = rng.randint(1000, 9999)
                sent = f"The {adj} number is {fake}."
            else:
                sent = rng.choice(FILLER_SENTENCES)
            stoks = len(tokenizer.encode(sent, add_special_tokens=False))
            if tok_count + stoks > budget:
                break
            parts.append(sent)
            tok_count += stoks

        pos = rng.randint(max(1, len(parts) // 10), max(1, 9 * len(parts) // 10))
        parts.insert(pos, needle)
        depth = pos / len(parts)

        prompt = " ".join(parts) + f"\n\nQuestion: {query}\nAnswer:"
        target = f" {target_num}"
        examples.append({"prompt": prompt, "target": target, "depth": depth})

    return examples


def score_examples(model, tokenizer, device, examples, max_gen=10):
    """
    Score NIAH examples using parallel forward + greedy argmax.

    For each example, concatenate prompt + target, run a single forward
    pass, and check if the model's greedy predictions at the answer
    positions match the target tokens.
    """
    correct = 0
    total = 0

    model.eval()
    with torch.no_grad():
        for ex in examples:
            prompt_ids = tokenizer.encode(ex["prompt"], add_special_tokens=False)
            target_str = ex["target"].strip()

            input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

            if input_ids.shape[1] > 2048:
                input_ids = input_ids[:, :2048]

            out = model(input_ids)
            logits = out["logits"]

            # Greedy decode from the last position
            generated_ids = []
            for step in range(max_gen):
                next_logits = logits[:, -1, :]
                next_token = next_logits.argmax(dim=-1).item()
                generated_ids.append(next_token)

                if next_token == tokenizer.eos_token_id:
                    break

                # Extend input and re-run forward (parallel, no recurrence)
                next_tensor = torch.tensor([[next_token]], dtype=torch.long, device=device)
                input_ids = torch.cat([input_ids, next_tensor], dim=1)

                if input_ids.shape[1] > 2048:
                    input_ids = input_ids[:, -2048:]

                out = model(input_ids)
                logits = out["logits"]

            decoded = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

            if target_str in decoded:
                correct += 1
            total += 1

    return correct / max(total, 1)


def main():
    parser = argparse.ArgumentParser(description="Quick NIAH eval (parallel forward)")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model_size", type=str, default="50m",
                        choices=["180m", "180m_gated", "370m", "50m"])
    parser.add_argument("--tokenizer", type=str, default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--context_lens", nargs="+", type=int,
                        default=[128, 256, 512, 1024, 2048])
    parser.add_argument("--n_examples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    # Load config
    if args.model_size == "180m":
        cfg = config_180m()
    elif args.model_size == "180m_gated":
        cfg = config_180m_gated()
    elif args.model_size == "370m":
        cfg = config_370m()
    elif args.model_size == "50m":
        cfg = config_50m()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    cfg.vocab_size = len(tokenizer)

    # Load model
    model = KoopmanLM(cfg)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model = model.to(args.device).float()
    model.eval()
    model.param_summary()

    # Builders
    builders = {
        "NIAH-Single-1": build_single1,
        "NIAH-Single-2": build_single2,
        "NIAH-Single-3": build_single3,
    }

    print("\n" + "=" * 60)
    print("NIAH Evaluation (parallel forward, no recurrence)")
    print("=" * 60)

    all_results = {}
    for task_name, builder in builders.items():
        all_results[task_name] = {}
        for ctx_len in args.context_lens:
            examples = builder(tokenizer, ctx_len,
                               n_examples=args.n_examples, seed=args.seed)
            acc = score_examples(model, tokenizer, args.device, examples)
            all_results[task_name][ctx_len] = acc * 100.0
            print(f"  {task_name} @ {ctx_len:>5d}: {acc * 100.0:6.1f}%")

    # Summary table
    print(f"\n  {'':20s}", end="")
    for ctx_len in args.context_lens:
        print(f"  {ctx_len:>5d}", end="")
    print()
    for task_name in builders:
        print(f"  {task_name:20s}", end="")
        for ctx_len in args.context_lens:
            val = all_results[task_name].get(ctx_len, 0)
            print(f"  {val:5.1f}", end="")
        print()


if __name__ == "__main__":
    main()
