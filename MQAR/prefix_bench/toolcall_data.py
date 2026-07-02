"""
toolcall_data.py -- A large, long-context tool-calling associative-recall task.

Motivation
----------
MQAR has no temporal structure: consecutive tokens are independent, so the Koopman
transition operator A_w = M G^{-1} is estimated from noise and the spectral/power
filter is dead weight. Real tool-call traces are different: they follow a grammar
(opcodes), state persists, and bindings get *overwritten* -- so retrieval must
respect recency, not just content. This is exactly the regime where the Koopman
dynamics (persistent |lambda|~1 modes) can earn their keep, and it comes with a
genuine, data-given prefix boundary (the trace precedes the answer turn).

Sequence layout (per example):

    [ op op op ... op            SEP    GET k_a  GET k_b ...  ]
      \_______ trace ________/    ^      \____ query turn ____/
             (context)          delim     (predict latest value bound to key)

    op ::= SET key value      # bind key -> value (may overwrite an earlier bind)
         | NOOP junk junk     # distractor

The trace is padded with NOOP distractors to reach a target `context_len` that can
be made very long (thousands of tokens). Some keys are SET multiple times; the
correct answer for a GET is the value from the *latest* SET (recency). A pure
order-agnostic associative memory blends all values for a key and gets overwritten
keys wrong; recency-aware retrieval does not.

Token scheme
------------
  0 = PAD, 1 = SEP, 2 = SET, 3 = GET, 4 = NOOP   (reserved opcodes)
  keys   in [KEY_BASE, KEY_BASE + num_keys)
  values in [VAL_BASE, vocab_size)               (disjoint from keys)
  noise  drawn from the value range.

Interface matches mqar_data.make_mqar_batch:
  returns x, y, loss_mask, prefix_mask, seg_ids, meta
"""

from typing import Optional, Dict, Tuple, List
import torch

PAD_ID = 0
SEP_ID = 1
SET_ID = 2
GET_ID = 3
NOOP_ID = 4
N_RESERVED = 5
KEY_BASE = N_RESERVED  # keys occupy [KEY_BASE, KEY_BASE + num_keys)


def make_toolcall_batch(
    batch_size: int,
    context_len: int,
    num_keys: int,
    num_query: int,
    vocab_size: int = 8192,
    overwrite_prob: float = 0.4,
    device: str = "cpu",
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
    """
    Generate one long-context tool-call batch.

    Args:
        context_len:    number of tokens in the trace region (before SEP). Made a
                        multiple of 3 (op width) internally. This is the "long
                        context" knob -- scale it to 4k, 16k, 32k, ...
        num_keys:       distinct keys bound in the trace.
        num_query:      number of GET queries after SEP (<= num_keys).
        overwrite_prob: probability a key is SET more than once (tests recency).

    Returns (x, y, loss_mask, prefix_mask, seg_ids, meta) with the same shapes and
    semantics as mqar_data.make_mqar_batch.
    """
    if num_query > num_keys:
        raise ValueError(f"num_query={num_query} > num_keys={num_keys}")
    val_base = KEY_BASE + num_keys
    if vocab_size <= val_base + 1:
        raise ValueError(
            f"vocab_size={vocab_size} too small: need > {val_base + 1} "
            f"(reserved + {num_keys} keys + value range)."
        )

    op_w = 3
    n_ops = max(num_keys, context_len // op_w)          # at least one op per key
    ctx_tokens = n_ops * op_w
    query_tokens = 2 * num_query                        # [GET, key] per query
    T = ctx_tokens + 1 + query_tokens                   # +1 for SEP
    sep_pos = ctx_tokens
    query_start = sep_pos + 1

    B = batch_size
    dev = torch.device(device)

    x = torch.full((B, T), PAD_ID, dtype=torch.long, device=dev)
    y = torch.zeros(B, T, dtype=torch.long, device=dev)
    loss_mask = torch.zeros(B, T, dtype=torch.float32, device=dev)

    def randint(lo, hi, n=()):
        return torch.randint(lo, hi, n if isinstance(n, tuple) else (n,),
                             device=dev, generator=generator)

    for b in range(B):
        keys = torch.arange(KEY_BASE, KEY_BASE + num_keys, device=dev)
        final_val = randint(val_base, vocab_size, num_keys)   # latest value per key

        # Build the event list: each key SET at least once with its final value;
        # overwritten keys also get earlier decoy SETs that must precede the final.
        # order_key controls temporal order; the final SET is forced to be last
        # among a key's events.
        events: List[Tuple[int, int, float]] = []  # (key_id, value_id, order)
        for ki in range(num_keys):
            k = int(keys[ki].item())
            fv = int(final_val[ki].item())
            n_decoys = 0
            if float(torch.rand((), device=dev, generator=generator)) < overwrite_prob:
                n_decoys = 1 + int(randint(0, 2).item())  # 1 or 2 decoys
            base_order = float(torch.rand((), device=dev, generator=generator))
            for d in range(n_decoys):
                dv = int(randint(val_base, vocab_size, 1).item())
                # decoy order strictly before the final for this key
                events.append((k, dv, base_order + 0.001 * d))
            events.append((k, fv, base_order + 1.0))       # final SET is latest

        # Sort by temporal order, then realize as SET ops.
        events.sort(key=lambda e: e[2])
        ops: List[Tuple[int, int, int]] = [(SET_ID, k, v) for (k, v, _) in events]

        # Pad with NOOP distractors to fill the trace, then shuffle op order while
        # keeping each key's SETs in their relative (temporal) order.
        n_noop = n_ops - len(ops)
        if n_noop < 0:
            # too many events for the requested context; keep the latest ops
            ops = ops[-n_ops:]
            n_noop = 0
        for _ in range(n_noop):
            j1 = int(randint(val_base, vocab_size, 1).item())
            j2 = int(randint(val_base, vocab_size, 1).item())
            ops.append((NOOP_ID, j1, j2))

        # Shuffle by assigning random slots but preserving SET-per-key ordering:
        # give NOOPs free random slots; give SETs monotonically increasing slots in
        # their current (already temporally sorted) order so recency is preserved.
        n_total = len(ops)
        slots = torch.randperm(n_total, generator=generator, device=dev).tolist()
        set_positions = sorted(slots[:sum(1 for o in ops if o[0] == SET_ID)])
        noop_positions = slots[sum(1 for o in ops if o[0] == SET_ID):]
        placed: List[Optional[Tuple[int, int, int]]] = [None] * n_total
        si, ni = 0, 0
        for o in ops:
            if o[0] == SET_ID:
                placed[set_positions[si]] = o
                si += 1
            else:
                placed[noop_positions[ni]] = o
                ni += 1

        # Flatten ops into the trace region.
        flat = []
        for o in placed:
            flat.extend(o)
        trace = torch.tensor(flat[:ctx_tokens], dtype=torch.long, device=dev)
        x[b, :ctx_tokens] = trace

        # Delimiter.
        x[b, sep_pos] = SEP_ID

        # Query turn: GET key -> predict latest value at the key position.
        q_keys = torch.randperm(num_keys, generator=generator, device=dev)[:num_query]
        for j in range(num_query):
            base = query_start + 2 * j
            k = int(keys[q_keys[j]].item())
            x[b, base] = GET_ID
            x[b, base + 1] = k
            y[b, base + 1] = int(final_val[q_keys[j]].item())
            loss_mask[b, base + 1] = 1.0

    prefix_mask = torch.zeros(B, T, dtype=torch.float32, device=dev)
    prefix_mask[:, : sep_pos + 1] = 1.0

    seg_ids = torch.ones(B, T, dtype=torch.long, device=dev)
    seg_ids[:, : sep_pos + 1] = 0

    meta = {
        "query_start": query_start,
        "sep_pos": sep_pos,
        "context_len": ctx_tokens,
        "seq_len": T,
        "num_keys": num_keys,
        "num_query": num_query,
    }
    return x, y, loss_mask, prefix_mask, seg_ids, meta


def _self_test():
    x, y, lm, pm, seg, meta = make_toolcall_batch(
        batch_size=3, context_len=120, num_keys=8, num_query=5, vocab_size=512
    )
    B, T = x.shape
    assert x.shape == y.shape == lm.shape == pm.shape == seg.shape == (B, T)
    assert (lm.sum(dim=1) == 5).all(), lm.sum(dim=1)
    assert (pm[:, meta["query_start"]:] == 0).all()
    assert (x[:, meta["sep_pos"]] == SEP_ID).all()
    assert torch.equal((seg == 0).float(), pm)
    # answers are valid value tokens (>= KEY_BASE + num_keys)
    assert (y[lm.bool()] >= KEY_BASE + meta["num_keys"]).all()
    print("toolcall_data self-test passed:", meta)


if __name__ == "__main__":
    _self_test()
