"""MQAR (Multi-Query Associative Recall), Arora et al. 2024 / Zoology setup.

Token-id-level synthetic: keys and values live in disjoint halves of the vocab.
A sequence lists key-value pairs, then re-presents a subset of keys as queries;
the model must predict each queried key's value at the following position.

The grid sweeps sequence length x number of KV pairs, the axes the scaling plan
calls out (256/512/1K/2K x 4/8/16/32/64), to compare against published Mamba-2
and Transformer numbers.

``make_mqar`` is a pure generator (CPU-testable, no model). ``eval_mqar`` /
``eval_mqar_grid`` run a model (teacher-forced, argmax at the answer positions).
"""
import torch


def make_mqar(batch, seq_len, num_kv_pairs, vocab_size,
              num_queries=None, seed=0):
    """Build a batch of MQAR sequences.

    Returns (input_ids, labels) each (batch, seq_len). labels are -100 except at
    the answer positions (the token where the queried key's value must appear),
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
