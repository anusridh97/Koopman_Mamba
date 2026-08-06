"""
chat_data.py -- Method 1 (chat / stopped-prefix) data under Causal Structured
Prefixing.

Multi-turn synthetic chat: each round has a USER block that states key->value facts
and an ASSISTANT block that must recall the latest value for queried keys. Facts
accumulate across rounds and may be overwritten (recency). Loss is only on the
assistant answer tokens -- the standard chat-SFT setting.

Every block is a release group (a "turn"): the USER block of round r is group 2r and
the ASSISTANT block is group 2r+1. Under CSP, an assistant token in group g may use
all strictly-earlier groups (all prior user facts, including the current round's user
block) plus causal context within its own block. So:

  * SKA (ska_mode="release") accumulates G, M, C_v over earlier groups only -- the
    prefix -- exactly the paper's prefix-mode accumulation, but multi-span.
  * Attention (attn_mode="segment") gets segment_causal_bias(release_grp): assistant
    tokens attend bidirectionally over the closed prefix and causally within the turn.

Token scheme:
  0 = PAD, 1 = USER, 2 = ASST   (reserved)
  keys   in [KEY_BASE, KEY_BASE + num_keys)
  values in [VAL_BASE, vocab_size)

Returns (x, y, loss_mask, prefix_mask, seg_ids, meta) with meta["release_grp"] the
(B, T) turn-group ids. Interface matches the other generators.
"""

from typing import Optional, Dict, Tuple
import torch

PAD_ID = 0
USER_ID = 1
ASST_ID = 2
N_RESERVED = 3
KEY_BASE = N_RESERVED


def make_chat_batch(
    batch_size: int,
    n_rounds: int,
    kv_per_round: int,
    q_per_round: int,
    num_keys: int,
    vocab_size: int = 512,
    device: str = "cpu",
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
    if kv_per_round > num_keys:
        raise ValueError(f"kv_per_round={kv_per_round} > num_keys={num_keys}")
    val_base = KEY_BASE + num_keys
    if vocab_size <= val_base + 1:
        raise ValueError(f"vocab_size={vocab_size} too small for {num_keys} keys.")

    round_len = 1 + 2 * kv_per_round + 1 + 2 * q_per_round
    T = n_rounds * round_len
    B = batch_size
    dev = torch.device(device)

    x = torch.full((B, T), PAD_ID, dtype=torch.long, device=dev)
    y = torch.zeros(B, T, dtype=torch.long, device=dev)
    loss_mask = torch.zeros(B, T, dtype=torch.float32, device=dev)
    seg_ids = torch.zeros(B, T, dtype=torch.long, device=dev)      # 0=context,1=assistant
    release_grp = torch.zeros(B, T, dtype=torch.long, device=dev)

    def randint(lo, hi, n=1):
        return torch.randint(lo, hi, (n,), device=dev, generator=generator)

    for b in range(B):
        keys = torch.arange(KEY_BASE, KEY_BASE + num_keys, device=dev)
        latest: Dict[int, int] = {}   # key -> latest value seen so far
        pos = 0
        for r in range(n_rounds):
            user_grp = 2 * r
            asst_grp = 2 * r + 1

            # ---- USER block: state kv_per_round facts (may overwrite) ----
            x[b, pos] = USER_ID
            release_grp[b, pos] = user_grp
            seg_ids[b, pos] = 0
            pos += 1
            sel = torch.randperm(num_keys, generator=generator, device=dev)[:kv_per_round]
            for i in range(kv_per_round):
                k = int(keys[sel[i]].item())
                val = int(randint(val_base, vocab_size).item())
                latest[k] = val
                x[b, pos] = k
                x[b, pos + 1] = val
                release_grp[b, pos] = user_grp
                release_grp[b, pos + 1] = user_grp
                pos += 2

            # ---- ASSISTANT block: recall latest value for queried keys ----
            x[b, pos] = ASST_ID
            release_grp[b, pos] = asst_grp
            seg_ids[b, pos] = 1
            pos += 1
            known = list(latest.keys())
            qsel = [known[int(randint(0, len(known)).item())] for _ in range(q_per_round)]
            for i in range(q_per_round):
                k = qsel[i]
                x[b, pos] = k                      # the query key (assistant restates it)
                x[b, pos + 1] = latest[k]          # target answer at this position
                y[b, pos + 1] = latest[k]
                loss_mask[b, pos + 1] = 1.0
                seg_ids[b, pos] = 1
                seg_ids[b, pos + 1] = 1
                release_grp[b, pos] = asst_grp
                release_grp[b, pos + 1] = asst_grp
                pos += 2

    # prefix_mask kept for compatibility (1 on non-assistant context).
    prefix_mask = (seg_ids == 0).float()

    meta = {
        "release_grp": release_grp,
        "seq_len": T,
        "round_len": round_len,
        "n_rounds": n_rounds,
        "num_keys": num_keys,
    }
    return x, y, loss_mask, prefix_mask, seg_ids, meta


def _self_test():
    x, y, lm, pm, seg, meta = make_chat_batch(
        batch_size=3, n_rounds=3, kv_per_round=4, q_per_round=3, num_keys=8,
        vocab_size=256)
    B, T = x.shape
    grp = meta["release_grp"]
    assert x.shape == y.shape == lm.shape == pm.shape == seg.shape == grp.shape == (B, T)
    assert (lm.sum(dim=1) == 3 * 3).all(), lm.sum(dim=1)   # q_per_round * n_rounds
    # release groups are non-decreasing along T
    assert (grp[:, 1:] - grp[:, :-1] >= 0).all()
    # loss only on assistant positions
    assert ((lm > 0) == ((lm > 0) & (seg == 1))).all()
    # answers are valid value tokens
    assert (y[lm.bool()] >= KEY_BASE + meta["num_keys"]).all()
    print("chat_data self-test passed:", {k: meta[k] for k in ("n_rounds", "round_len", "seq_len")})


if __name__ == "__main__":
    _self_test()
