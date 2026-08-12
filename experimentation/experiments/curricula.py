"""
Synthetic curricula for the Echo paper's retrieval experiments. All
generators share one contract: return (input_ids, labels), each (batch,
seq_len); labels are -100 except at answer positions, so a shifted
cross-entropy / argmax (logits[:, :-1] vs labels[:, 1:]) scores only those
positions. That shift convention matters: labels sit ON the echoed answer
token, so the position right before it is what's actually supervised.

    - make_mqar / eval_mqar / eval_mqar_grid: Multi-Query Associative Recall
      (Arora et al. 2024 / Zoology). Used by mqar_finetune.py (Sec 5.2: 50M
      in-task MQAR fine-tuning) and available for ad-hoc MQAR evaluation.
      Keys/values live in disjoint vocab halves; KV pairs always sit at the
      very front of the sequence.

    - make_toolcall / make_sysprompt: table2.py's (Sec 4.1) training
      curriculum -- a stand-in for the paper's unspecified "mixed
      system-prompt and tool-trace examples." make_toolcall mirrors
      Appendix G.4's "Tool-Calling Retrieval" (SET/GET with overwrite, so
      the answer must reflect the LATEST binding -- recency, not just any
      occurrence). make_sysprompt mirrors Appendix G.4's "System Prompt
      Amnesia / Specific Recall" (a fixed small set of variables set once,
      then a gap salted with decoy bindings that are never queried).

    - make_niah / eval_niah: table2.py's held-out zero-shot eval. Unlike
      make_mqar, the needle is placed at a RANDOM position in the context
      (not always at the front), so this tests position-invariant retrieval
      rather than long-range retention from a known offset -- and, being a
      structurally distinct generator from make_toolcall/make_sysprompt
      (different token layout entirely, not just a parameter change), it is
      never seen in any form during table2.py's training.
"""

import torch


def make_mqar(batch, seq_len, num_kv_pairs, vocab_size,
              num_queries=None, seed=0):
    """Build a batch of MQAR sequences.

    Returns (input_ids, labels) each (batch, seq_len). labels are -100 except at
    the answer positions (the position where the queried key's value must appear),
    so cross-entropy / argmax is scored only there.

    Layout per sequence:
        [k1 v1 k2 v2 ... kP vP]  <filler...>  [kq1 vq1? kq2 vq2? ...]
    where the second occurrence of each key is a "query" and the position right
    after it is supervised with that key's value.
    """
    num_queries = num_queries or num_kv_pairs
    assert vocab_size >= 4, "need room for disjoint key/value halves"
    assert 2 * num_kv_pairs + 2 * num_queries <= seq_len, \
        f"seq_len={seq_len} too short for {num_kv_pairs} pairs + {num_queries} queries"
    half = vocab_size // 2
    assert num_kv_pairs <= half, "not enough distinct keys/values for num_kv_pairs"

    g = torch.Generator().manual_seed(seed)
    inputs = torch.zeros(batch, seq_len, dtype=torch.long)
    labels = torch.full((batch, seq_len), -100, dtype=torch.long)

    for b in range(batch):
        keys = torch.randperm(half, generator=g)[:num_kv_pairs]
        vals = half + torch.randperm(vocab_size - half, generator=g)[:num_kv_pairs]

        seq = []
        for k, v in zip(keys.tolist(), vals.tolist()):
            seq += [k, v]                              # k v k v ... at the front

        # queries: re-present a random subset of keys near the end
        qsel = torch.randperm(num_kv_pairs, generator=g)[:num_queries]
        query_block = []
        ans_offsets = []                               # answer positions within query block
        for j in qsel.tolist():
            ans_offsets.append(len(query_block) + 1)   # position right after the query key
            query_block += [keys[j].item(), vals[j].item()]

        # filler between pairs and queries: random key-space tokens (distractors)
        n_filler = seq_len - len(seq) - len(query_block)
        filler = torch.randint(0, half, (n_filler,), generator=g).tolist()

        full = seq + filler + query_block
        inputs[b] = torch.tensor(full, dtype=torch.long)
        base = len(seq) + n_filler
        for off, j in zip(ans_offsets, qsel.tolist()):
            labels[b, base + off] = vals[j].item()
    return inputs, labels


@torch.no_grad()
def eval_mqar(model, batch, seq_len, num_kv_pairs, vocab_size, device,
              num_queries=None, seed=0):
    """Teacher-forced MQAR accuracy: argmax at the supervised answer positions."""
    inputs, labels = make_mqar(batch, seq_len, num_kv_pairs, vocab_size,
                               num_queries=num_queries, seed=seed)
    inputs, labels = inputs.to(device), labels.to(device)
    out = model(input_ids=inputs)
    logits = out["logits"] if isinstance(out, dict) else out
    # Constrain argmax to [0, vocab_size) so random noise in unused embedding
    # rows (when model vocab > task vocab) doesn't suppress accuracy.
    pred = logits[:, :-1, :vocab_size].argmax(-1)
    tgt = labels[:, 1:]
    mask = tgt != -100
    if mask.sum() == 0:
        return float("nan")
    correct = (pred[mask] == tgt[mask]).float().mean().item()
    return correct


@torch.no_grad()
def eval_mqar_grid(model, vocab_size, device, batch=64,
                   seq_lens=(256, 512, 1024, 2048),
                   kv_pairs=(4, 8, 16, 32, 64), seed=0):
    """Sweep the MQAR grid. Returns {seq_len: {num_kv: accuracy}}; skips cells
    where the sequence is too short for the requested number of pairs."""
    grid = {}
    for T in seq_lens:
        grid[T] = {}
        for P in kv_pairs:
            if 4 * P >= T:                   # need 2P KV tokens + filler + 2P query tokens < T
                continue
            grid[T][P] = eval_mqar(model, batch, T, P, vocab_size, device, seed=seed)
    return grid


# -----------------------------------------------------------------------------
# Tool-calling retrieval (App. G.4 "Tool-Calling Retrieval"): SET/GET trace
# with overwrite semantics, so the correct answer is the LATEST binding.
# -----------------------------------------------------------------------------

TOOLCALL_SET_ID = 0
TOOLCALL_GET_ID = 1
TOOLCALL_NOOP_ID = 2
TOOLCALL_FIRST_CONTENT_ID = 3


def make_toolcall(batch, seq_len, num_keys=8, num_queries=4, vocab_size=128,
                   overwrite_prob=0.3, seed=0):
    """SET/GET tool-call trace. Layout per sequence:

        [op op op ... op]  [GET k1 v1 GET k2 v2 ...]
        op ::= SET key value | NOOP junk junk        (3 tokens each)

    Every key is SET at least once in the first num_keys ops (bootstrap), so
    every query has a defined answer; later ops randomly SET (overwriting an
    earlier binding -- recency) or NOOP (pure distractor). Labels sit on the
    echoed value token in the query block, matching make_mqar's shift
    convention.
    """
    assert num_queries <= num_keys, "num_queries must be <= num_keys"
    assert vocab_size > TOOLCALL_FIRST_CONTENT_ID + num_keys, "vocab too small for num_keys"
    query_tokens = 3 * num_queries
    n_ops = (seq_len - query_tokens) // 3
    assert n_ops >= num_keys, f"seq_len={seq_len} too short for {num_keys} keys + {num_queries} queries"
    context_len = 3 * n_ops
    n_content = vocab_size - TOOLCALL_FIRST_CONTENT_ID

    g = torch.Generator().manual_seed(seed)
    inputs = torch.zeros(batch, seq_len, dtype=torch.long)
    labels = torch.full((batch, seq_len), -100, dtype=torch.long)

    for b in range(batch):
        keys = (torch.randperm(n_content, generator=g)[:num_keys] + TOOLCALL_FIRST_CONTENT_ID).tolist()
        latest = {}
        ops = []
        for i in range(n_ops):
            if i < num_keys:
                k, is_set = keys[i], True                          # bootstrap: bind every key once
            else:
                k = keys[torch.randint(0, num_keys, (1,), generator=g).item()]
                is_set = torch.rand(1, generator=g).item() < overwrite_prob
            if is_set:
                v = torch.randint(TOOLCALL_FIRST_CONTENT_ID, vocab_size, (1,), generator=g).item()
                latest[k] = v
                ops += [TOOLCALL_SET_ID, k, v]
            else:
                junk = torch.randint(TOOLCALL_FIRST_CONTENT_ID, vocab_size, (2,), generator=g).tolist()
                ops += [TOOLCALL_NOOP_ID, junk[0], junk[1]]
        inputs[b, :context_len] = torch.tensor(ops, dtype=torch.long)

        qkeys = [keys[i] for i in torch.randperm(num_keys, generator=g)[:num_queries].tolist()]
        qpos = context_len
        for k in qkeys:
            v = latest[k]
            inputs[b, qpos] = TOOLCALL_GET_ID
            inputs[b, qpos + 1] = k
            inputs[b, qpos + 2] = v
            labels[b, qpos + 2] = v
            qpos += 3
    return inputs, labels


# -----------------------------------------------------------------------------
# System-prompt recall (App. G.4 "System Prompt Amnesia / Specific Recall"):
# fixed variables set once at the front, then a noise gap. num_decoys=0 by
# default -- SKA fits its Gram-matrix statistics over the whole context with
# no mechanism to mark "this pair matters, that one doesn't" (confirmed by
# reading koopman_lm/modules/seq/ska.py), so decoy bindings that look
# structurally identical to the real ones measurably stalled training versus
# plain noise. Pass num_decoys>0 for the paper-appendix-faithful "confusing
# distractors" version if training budget allows it.
# -----------------------------------------------------------------------------

SYSPROMPT_FIRST_CONTENT_ID = 1  # 0 reserved/unused


def make_sysprompt(batch, seq_len, num_vars=4, vocab_size=128, num_decoys=0, seed=0):
    """Layout per sequence:

        [v1 x1 v2 x2 ... vN xN]  <gap w/ decoy bindings + noise>  [vq xq ...]

    num_vars variables are bound once at the front; the gap is mostly random
    noise but num_decoys positions hold plausible-looking (never-queried)
    variable/value pairs drawn from a disjoint id pool, so the model can't
    just pattern-match "any binding-shaped pair" -- it must track the
    specific num_vars ids from the header. Labels sit on the echoed value
    token in the query block.
    """
    header_len = 2 * num_vars
    query_tokens = 2 * num_vars
    gap_len = seq_len - header_len - query_tokens
    assert gap_len >= 2 * num_decoys, f"seq_len={seq_len} too short for {num_vars} vars + {num_decoys} decoys"
    assert vocab_size > SYSPROMPT_FIRST_CONTENT_ID + num_vars + num_decoys, "vocab too small"
    n_content = vocab_size - SYSPROMPT_FIRST_CONTENT_ID

    g = torch.Generator().manual_seed(seed)
    inputs = torch.zeros(batch, seq_len, dtype=torch.long)
    labels = torch.full((batch, seq_len), -100, dtype=torch.long)

    for b in range(batch):
        ids = (torch.randperm(n_content, generator=g)[:num_vars + num_decoys] + SYSPROMPT_FIRST_CONTENT_ID)
        var_ids = ids[:num_vars].tolist()
        decoy_ids = ids[num_vars:].tolist()
        values = torch.randint(SYSPROMPT_FIRST_CONTENT_ID, vocab_size, (num_vars,), generator=g).tolist()
        var_value = dict(zip(var_ids, values))

        header = []
        for k, v in zip(var_ids, values):
            header += [k, v]
        inputs[b, :header_len] = torch.tensor(header, dtype=torch.long)

        gap = torch.randint(SYSPROMPT_FIRST_CONTENT_ID, vocab_size, (gap_len,), generator=g)
        decoy_slots = torch.randperm(gap_len // 2, generator=g)[:num_decoys] * 2
        for i, slot in enumerate(decoy_slots.tolist()):
            gap[slot] = decoy_ids[i]
            gap[slot + 1] = torch.randint(SYSPROMPT_FIRST_CONTENT_ID, vocab_size, (1,), generator=g).item()
        inputs[b, header_len:header_len + gap_len] = gap

        qsel = torch.randperm(num_vars, generator=g).tolist()
        qpos = header_len + gap_len
        for j in qsel:
            k = var_ids[j]
            v = var_value[k]
            inputs[b, qpos] = k
            inputs[b, qpos + 1] = v
            labels[b, qpos + 1] = v
            qpos += 2
    return inputs, labels


# -----------------------------------------------------------------------------
# NIAH (needle-in-a-haystack): a single fact at a RANDOM position, queried
# once at the end. table2.py's held-out eval -- structurally distinct from
# make_toolcall/make_sysprompt, never seen during training.
# -----------------------------------------------------------------------------

def make_niah(batch, seq_len, vocab_size=128, seed=0):
    """One (key, value) pair placed at a random offset in a noisy context,
    then re-presented as a query at the very end. Unlike make_mqar (which
    always places KV pairs at position 0), the random offset means the
    model must actually search the context, not just check a known
    location -- the point of a needle-in-a-haystack test.
    """
    assert seq_len >= 6, "seq_len too short for a needle + query"
    assert vocab_size > 2, "vocab too small"
    context_len = seq_len - 2   # last 2 tokens reserved for the query

    g = torch.Generator().manual_seed(seed)
    inputs = torch.randint(1, vocab_size, (batch, seq_len), dtype=torch.long, generator=g)
    labels = torch.full((batch, seq_len), -100, dtype=torch.long)

    for b in range(batch):
        key = torch.randint(1, vocab_size, (1,), generator=g).item()
        value = torch.randint(1, vocab_size, (1,), generator=g).item()
        pos = torch.randint(0, context_len - 1, (1,), generator=g).item()
        inputs[b, pos] = key
        inputs[b, pos + 1] = value
        inputs[b, -2] = key
        inputs[b, -1] = value
        labels[b, -1] = value
    return inputs, labels


@torch.no_grad()
def eval_niah(model, batch, seq_len, vocab_size, device, seed=0):
    """Teacher-forced NIAH accuracy: argmax at the single supervised answer position."""
    inputs, labels = make_niah(batch, seq_len, vocab_size=vocab_size, seed=seed)
    inputs, labels = inputs.to(device), labels.to(device)
    out = model(input_ids=inputs)
    logits = out["logits"] if isinstance(out, dict) else out
    pred = logits[:, :-1, :vocab_size].argmax(-1)
    tgt = labels[:, 1:]
    mask = tgt != -100
    if mask.sum() == 0:
        return float("nan")
    return (pred[mask] == tgt[mask]).float().mean().item()
