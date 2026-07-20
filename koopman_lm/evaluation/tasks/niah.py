"""Unified NIAH (Needle-In-A-Haystack) generator + scorers.

One core needle/filler generator, shared by two renderers:
  - _build_niah_prompts / _build_niah_single1/2/3   -- text-prompt output,
    used by evaluate.py (and eval_niah's parallel/recurrent scorers below).
  - _build_niah_examples                            -- token-id output
    (BPE-safe separate tokenization), used by evaluate_retrieval.py.

Both renderers start from the same filler/needle-placement logic and format
the same three (context, query_text, target_text) strings -- they just
differ in whether those strings get concatenated into one prompt string or
tokenized separately into segment IDs. This avoids any decode/re-encode
round-trip that could subtly perturb token counts.

NOTE: evaluate.py's own NIAH-Single-2 historically used a 15-entry distractor
adjectives list, while evaluate_retrieval.py's NIAH-Single-2 task used only
10 (the same 10-entry list as NIAH-Single-3). This was a genuine divergence
between the two files, not an accident of copy-paste -- `adjectives` is kept
as an explicit parameter so each caller's original behavior is preserved
exactly rather than silently reconciled.
"""
import random

import torch

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

_ADJECTIVES_10 = ["regular", "normal", "common", "typical", "standard",
                  "ordinary", "usual", "general", "basic", "default"]
_ADJECTIVES_15 = _ADJECTIVES_10 + ["primary", "secondary", "initial",
                                   "final", "previous"]


def _generate_niah_needle(rng, tokenizer, context_len, use_distractors, adjectives):
    """Build the filler+needle context. Returns (parts, target_num, insert_pos)
    with the needle already inserted into `parts`."""
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
    return parts, target_num, insert_pos


def _build_niah_prompts(tokenizer, context_len, n_examples=200, seed=42,
                        use_distractors=False, paraphrase_query=False,
                        adjectives=_ADJECTIVES_10):
    """Text-prompt NIAH builder (evaluate.py's output shape:
    {"prompt", "target", "depth"})."""
    rng = random.Random(seed)
    query = ("Recall the unique designated number." if paraphrase_query
             else "What is the special number?")
    examples = []
    for _ in range(n_examples):
        parts, target_num, insert_pos = _generate_niah_needle(
            rng, tokenizer, context_len, use_distractors, adjectives)
        context = " ".join(parts)
        prompt = f"{context}\n\nQuestion: {query}\nAnswer:"
        target = f" {target_num}"
        examples.append({"prompt": prompt, "target": target,
                         "depth": insert_pos / len(parts)})
    return examples


def _build_niah_examples(tokenizer, context_len, n_examples, seed,
                         use_distractors=False, paraphrase_query=False,
                         adjectives=_ADJECTIVES_10):
    """Token-id NIAH builder (evaluate_retrieval.py's output shape,
    BPE-safe separate tokenization: {"context_ids", "query_ids",
    "target_ids", "target_str", "depth"})."""
    rng = random.Random(seed)
    query = ("Recall the unique designated number." if paraphrase_query
             else "What is the special number?")
    examples = []
    for _ in range(n_examples):
        parts, target_num, insert_pos = _generate_niah_needle(
            rng, tokenizer, context_len, use_distractors, adjectives)
        context = " ".join(parts)
        query_text = f"\n\nQuestion: {query}\nAnswer:"
        target_text = f" {target_num}"

        context_ids = tokenizer.encode(context, add_special_tokens=False)
        query_ids = tokenizer.encode(query_text, add_special_tokens=False)
        target_ids = tokenizer.encode(target_text, add_special_tokens=False)

        examples.append({
            "context_ids": context_ids,
            "query_ids": query_ids,
            "target_ids": target_ids,
            "target_str": target_text.strip(),
            "depth": insert_pos / len(parts),
        })
    return examples


def _build_niah_single1(tokenizer, context_len, n_examples=200, seed=42):
    """NIAH-Single-1: single fact in haystack, exact query."""
    return _build_niah_prompts(tokenizer, context_len, n_examples, seed,
                               use_distractors=False, paraphrase_query=False)


def _build_niah_single2(tokenizer, context_len, n_examples=200, seed=42):
    """NIAH-Single-2: with distractor facts (evaluate.py's own 15-entry
    adjectives list -- see module docstring)."""
    return _build_niah_prompts(tokenizer, context_len, n_examples, seed,
                               use_distractors=True, paraphrase_query=False,
                               adjectives=_ADJECTIVES_15)


def _build_niah_single3(tokenizer, context_len, n_examples=200, seed=42):
    """NIAH-Single-3: paraphrased query, with distractors (10-entry list)."""
    return _build_niah_prompts(tokenizer, context_len, n_examples, seed,
                               use_distractors=True, paraphrase_query=True,
                               adjectives=_ADJECTIVES_10)


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
    from koopman_lm.models.recurrent import RecurrentKoopmanLM

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
