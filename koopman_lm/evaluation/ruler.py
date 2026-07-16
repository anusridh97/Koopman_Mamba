"""RULER subset (Hsieh et al. 2024): synthetic long-context probes.

The scaling plan asks for "at minimum the multi-key NIAH, variable tracking, and
common-word extraction tasks at 4K and 8K." These builders generate (prompt,
answer) string pairs padded toward a target token budget; the scorer tokenizes,
runs the model, greedily decodes a few tokens, and checks whether the answer
appears.

Builders are pure functions (CPU-testable). ``eval_ruler_subset`` needs a model
and tokenizer (run on GPU).
"""
import random

import torch

from koopman_lm.evaluation.generation import _greedy_generate  # noqa: F401 (re-exported for babilong.py)

_FILLER = (
    "The grass is green. The sky is blue. The sun is bright today. "
    "We went to the store to buy groceries and then walked home slowly. "
)


def _pad_to_words(text, target_words, rng):
    """Append filler sentences until text reaches ~target_words words."""
    words = text.split()
    filler_words = _FILLER.split()
    while len(words) < target_words:
        start = rng.randrange(0, max(1, len(filler_words) - 8))
        words += filler_words[start:start + 8]
    return " ".join(words)


def build_multikey_niah(target_words=600, n_keys=4, seed=0):
    """Several 'key K is value V' needles in filler; ask for one key's value."""
    rng = random.Random(seed)
    keys = [f"K{rng.randrange(10000, 99999)}" for _ in range(n_keys)]
    vals = [str(rng.randrange(100000, 999999)) for _ in range(n_keys)]
    needles = [f"The special magic number for {k} is {v}." for k, v in zip(keys, vals)]
    body = _pad_to_words(" ".join(needles), target_words, rng)
    ask = rng.randrange(n_keys)
    prompt = (f"{body}\nQuestion: What is the special magic number for "
              f"{keys[ask]}?\nAnswer: The special magic number is")
    return prompt, vals[ask]


def build_variable_tracking(target_words=600, chain_len=4, seed=0):
    """Assignment chain X1=NUM, X2=X1, ...; ask the final variable's value."""
    rng = random.Random(seed)
    value = str(rng.randrange(100000, 999999))
    stmts = [f"VAR X1 = {value}."]
    for i in range(2, chain_len + 1):
        stmts.append(f"VAR X{i} = X{i-1}.")
    body = _pad_to_words(" ".join(stmts), target_words, rng)
    prompt = (f"{body}\nQuestion: What is the value of X{chain_len}?\n"
              f"Answer: The value of X{chain_len} is")
    return prompt, value


def build_common_word_extraction(target_words=600, n_words=6, repeats=5, seed=0):
    """A word list where one word repeats most; ask for the most common word."""
    rng = random.Random(seed)
    vocab = [f"item{rng.randrange(1000,9999)}" for _ in range(n_words)]
    common = vocab[0]
    seq = [common] * repeats + vocab[1:] + [common]      # `common` appears most
    rng.shuffle(seq)
    body = _pad_to_words("Words: " + ", ".join(seq) + ".", target_words, rng)
    prompt = (f"{body}\nQuestion: Which word appears most frequently?\n"
              f"Answer: The most common word is")
    return prompt, common


_BUILDERS = {
    "multikey_niah": build_multikey_niah,
    "variable_tracking": build_variable_tracking,
    "common_word_extraction": build_common_word_extraction,
}

# rough words-per-token for the filler; used to hit a token budget from words
_WORDS_PER_TOK = 0.75


@torch.no_grad()
def eval_ruler_subset(model, tokenizer, device, context_lens=(4096, 8192),
                      n_examples=20, max_new_tokens=12, tasks=None, seed=0):
    """Run the RULER subset at each context length.

    Returns {task: {ctx_len: accuracy}}. Accuracy = fraction of examples whose
    greedily-decoded continuation contains the gold answer string.
    """
    import torch
    from koopman_lm.models.recurrent import RecurrentKoopmanLM

    tasks = tasks or list(_BUILDERS)
    gen = model
    if hasattr(model, "cfg"):                # wrap KoopmanLM for O(1) decode
        try:
            gen = RecurrentKoopmanLM(model)
        except Exception:
            gen = model
    results = {t: {} for t in tasks}
    for task in tasks:
        builder = _BUILDERS[task]
        for ctx in context_lens:
            target_words = int(ctx * _WORDS_PER_TOK)
            hits = 0
            for i in range(n_examples):
                prompt, answer = builder(target_words=target_words, seed=seed + i)
                text = _greedy_generate(gen, tokenizer, device, prompt,
                                        max_new_tokens)
                hits += int(answer in text)
            results[task][ctx] = hits / max(n_examples, 1)
    return results
