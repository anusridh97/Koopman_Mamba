"""
mqar_data.py -- Multi-Query Associative Recall with an in-sequence delimiter, so
the context/query boundary is observable to every architecture (fair to attention),
and with a mask taxonomy (see prefix_masks.py).

Sequence layout (per example):

    [ k1 v1 k2 v2 ... kP vP   <noise ...>   SEP   q1 q2 ... qP ]
      \______ key/value ______/\__ noise __/  ^    \__ queries __/
             (context)                      delim   (predict paired value)

Reserved token ids:  0 = PAD, 1 = SEP (a real, visible delimiter token).
Keys / values / noise are drawn from [2, vocab_size).

Targets: at each query position we predict the value paired with that query key.
`loss_mask` is 1 only at query positions. `prefix_mask` marks the context region
(everything up to and including SEP). `seg_ids` is 0 on context, 1 on queries.
"""

from typing import Optional, Dict, Tuple
import torch

PAD_ID = 0
SEP_ID = 1
FIRST_CONTENT_ID = 2  # keys/values/noise live in [2, vocab_size)


def make_mqar_batch(
    batch_size: int,
    seq_len: int,
    num_kv_pairs: int,
    vocab_size: int = 8192,
    device: str = "cpu",
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
    """
    Generate one MQAR batch.

    Returns:
        x:           (B, T) long   input token ids
        y:           (B, T) long   target ids (paired value at query positions, 0 else)
        loss_mask:   (B, T) float  1.0 at query positions, 0.0 elsewhere
        prefix_mask: (B, T) float  1.0 on context (KV + noise + SEP), 0.0 on queries
        seg_ids:     (B, T) long   0 = context, 1 = query
        meta:        dict          {query_start, sep_pos, num_kv_pairs}
    """
    P = num_kv_pairs
    n_query = P
    query_start = seq_len - n_query
    sep_pos = query_start - 1
    if sep_pos < 2 * P:
        raise ValueError(
            f"seq_len={seq_len} too short for num_kv_pairs={P}: need "
            f"2*P (kv) + 1 (sep) + P (queries) = {3 * P + 1} positions."
        )
    if vocab_size <= FIRST_CONTENT_ID + P:
        raise ValueError(f"vocab_size={vocab_size} too small for {P} unique keys.")

    B, T = batch_size, seq_len
    dev = torch.device(device)

    # Fill entire sequence with noise, then overwrite structured positions.
    x = torch.randint(FIRST_CONTENT_ID, vocab_size, (B, T), device=dev, generator=generator)
    y = torch.zeros(B, T, dtype=torch.long, device=dev)
    loss_mask = torch.zeros(B, T, dtype=torch.float32, device=dev)

    n_content = vocab_size - FIRST_CONTENT_ID
    for b in range(B):
        keys = torch.randperm(n_content, generator=generator, device=dev)[:P] + FIRST_CONTENT_ID
        vals = torch.randint(FIRST_CONTENT_ID, vocab_size, (P,), device=dev, generator=generator)

        # Key/value pairs at the start of the context.
        x[b, 0:2 * P:2] = keys
        x[b, 1:2 * P:2] = vals

        # Delimiter.
        x[b, sep_pos] = SEP_ID

        # Queries (shuffled order) after the delimiter; label = the paired value.
        perm = torch.randperm(P, generator=generator, device=dev)
        qpos = torch.arange(query_start, query_start + P, device=dev)
        x[b, qpos] = keys[perm]
        y[b, qpos] = vals[perm]
        loss_mask[b, qpos] = 1.0

    prefix_mask = torch.zeros(B, T, dtype=torch.float32, device=dev)
    prefix_mask[:, : sep_pos + 1] = 1.0

    seg_ids = torch.ones(B, T, dtype=torch.long, device=dev)  # 1 = query by default
    seg_ids[:, : sep_pos + 1] = 0                             # 0 = context

    meta = {"query_start": query_start, "sep_pos": sep_pos, "num_kv_pairs": P}
    return x, y, loss_mask, prefix_mask, seg_ids, meta


def _self_test():
    """Shape / correctness sanity checks (needs torch, no GPU required)."""
    x, y, lm, pm, seg, meta = make_mqar_batch(4, 64, 4, vocab_size=256)
    B, T = x.shape
    assert x.shape == y.shape == lm.shape == pm.shape == seg.shape == (B, T)
    # exactly num_kv query targets per row
    assert (lm.sum(dim=1) == 4).all(), lm.sum(dim=1)
    # prefix mask 0 exactly on the query region
    assert (pm[:, meta["query_start"]:] == 0).all()
    assert (pm[:, : meta["sep_pos"] + 1] == 1).all()
    # seg ids consistent with prefix mask
    assert torch.equal((seg == 0).float(), pm)
    # SEP token present exactly once per row, at sep_pos
    assert (x[:, meta["sep_pos"]] == SEP_ID).all()
    assert (x == SEP_ID).sum(dim=1).eq(1).all()
    # targets at query positions are valid content tokens
    assert (y[lm.bool()] >= FIRST_CONTENT_ID).all()
    print("mqar_data self-test passed:", {k: meta[k] for k in meta})


if __name__ == "__main__":
    _self_test()
