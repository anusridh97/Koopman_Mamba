"""
pretokenize.py -- Pre-tokenize a mixed corpus into memory-mapped arrays with a
PARALLEL per-token recall-weight stream (dual-stream format).

440M rewrite changes:
  * Tokenizer default -> Llama-2 (meta-llama/Llama-2-7b-hf, 32000 vocab; use
    NousResearch/Llama-2-7b-hf for an ungated mirror with the identical
    tokenizer). uint16 packing still valid (vocab < 65536).
  * Three sources, flat-mixed by token budget: 55% FineWeb-Edu, 15% PG-19,
    30% SCROLLS.
  * SCROLLS examples are reformatted as context -> query -> answer; the ANSWER
    span (short answer for QA subtasks, the full summary for summarization
    subtasks) is up-weighted in the recall-weight stream. Everything else gets
    weight 1.0. This is the "recall pressure" that forces SKA to turn on
    (the MoE-load-balancing analogue) without changing the LM objective shape.

Outputs (in --output_dir):
  train.bin     uint16  flat token ids
  weights.bin   uint8   per-token recall weight (1 normal, RECALL_W on answers)
  meta.json     {n_tokens, vocab_size, tokenizer, dtype, weight_dtype, mix, ...}

The weight stream is read by MemmapPackedDataset (train_fast.py) and consumed
by KoopmanLM.forward(loss_weights=...).

Usage:
  python pretokenize.py --output_dir ./tok_440m --max_tokens 30_000_000_000
  python pretokenize.py --output_dir ./tok_440m --tokenizer NousResearch/Llama-2-7b-hf
"""

import argparse
import json
import os
import numpy as np

RECALL_W = 4          # up-weight factor for SCROLLS answer-span tokens
WEIGHT_DTYPE = np.uint8


def _interleave_quota(mix, chunk_tokens):
    """Given fractional mix, return how many tokens to pull from each source
    per round-robin cycle (integers summing ~chunk_tokens). A source with
    weight exactly 0 gets quota 0 (not the max(1, ...) floor) so a pure
    single-source mix (e.g. --mix 1.0 0.0 0.0) doesn't still pull a whole
    document from the zero-weighted sources every shard."""
    return {k: (max(1, int(round(v * chunk_tokens))) if v > 0 else 0)
            for k, v in mix.items()}


def _scrolls_format(ex, subset):
    """Reformat a SCROLLS example into (context_text, query_text, answer_text).
    SCROLLS fields vary by subtask; the loader exposes 'input' and 'output'.
    We treat 'input' as context(+embedded query) and 'output' as the answer
    span to up-weight. For QA subtasks the output is short; for summarization
    it's the full summary -- both are the thing the model must produce from
    long context, so both are valid recall spans.
    """
    inp = ex.get("input", "") or ""
    out = ex.get("output", "") or ""
    # Light structure so the answer boundary is unambiguous after tokenization.
    context = inp.strip()
    query = "\n\nAnswer:"
    answer = " " + out.strip()
    return context, query, answer


def write_synthetic_corpus(output_dir, n_tokens=200_000, vocab_size=32000,
                           recall_weight=4, seed=0):
    """Write a tiny self-contained dual-stream corpus (NO network/tokenizer).

    Produces train.bin (uint16), weights.bin (uint8), meta.json in the exact
    format MemmapPackedDataset reads. Periodic spans get the recall weight so the
    weighted-CE path is exercised. Used by the end-to-end smoke test so a fresh
    clone can train without downloading FineWeb/PG-19/SCROLLS.
    """
    assert vocab_size <= 65535, "uint16 packing requires vocab < 65536"
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.RandomState(seed)
    tokens = rng.randint(0, vocab_size, size=n_tokens, dtype=np.uint16)
    weights = np.ones(n_tokens, dtype=WEIGHT_DTYPE)
    # mark every 500th block of 20 tokens as an "answer span" (recall pressure)
    for s in range(0, n_tokens - 20, 500):
        weights[s:s + 20] = recall_weight
    tokens.tofile(os.path.join(output_dir, "train.bin"))
    weights.tofile(os.path.join(output_dir, "weights.bin"))
    with open(os.path.join(output_dir, "meta.json"), "w") as f:
        json.dump({"n_tokens": int(n_tokens), "vocab_size": int(vocab_size),
                   "tokenizer": "synthetic", "dtype": "uint16",
                   "weight_dtype": "uint8", "mix": "synthetic",
                   "recall_weight": int(recall_weight)}, f)
    return output_dir


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", type=str, default="./tok_440m")
    p.add_argument("--smoke", action="store_true",
                   help="write a tiny synthetic corpus (no network) for the e2e smoke test")
    p.add_argument("--smoke_tokens", type=int, default=200_000)
    p.add_argument("--tokenizer", type=str, default="meta-llama/Llama-2-7b-hf",
                   help="Llama-2 tokenizer (use NousResearch/Llama-2-7b-hf if gated)")
    p.add_argument("--fineweb", type=str, default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--fineweb_subset", type=str, default="sample-10BT")
    p.add_argument("--pg19", type=str, default="pg19")
    p.add_argument("--scrolls", type=str, default="tau/scrolls")
    p.add_argument("--scrolls_subsets", type=str, nargs="+",
                   default=["gov_report", "summ_screen_fd", "qasper",
                            "narrative_qa", "quality", "contract_nli"])
    p.add_argument("--mix", type=float, nargs=3, default=[0.55, 0.15, 0.30],
                   help="fractions for [fineweb, pg19, scrolls]")
    p.add_argument("--recall_weight", type=int, default=RECALL_W)
    p.add_argument("--max_tokens", type=int, default=None)
    p.add_argument("--shard_size", type=int, default=50_000_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip_docs", type=int, default=0,
                   help="skip this many FineWeb-Edu source documents before "
                        "shuffling (use to carve out a val shard disjoint "
                        "from a train run: skip past train's "
                        "fineweb_docs_consumed + a safety margin)")
    a = p.parse_args()

    if a.smoke:
        write_synthetic_corpus(a.output_dir, n_tokens=a.smoke_tokens, seed=a.seed)
        print(f"Wrote synthetic smoke corpus ({a.smoke_tokens:,} tokens) -> {a.output_dir}")
        return

    try:
        from datasets import load_dataset
        from transformers import AutoTokenizer
    except ImportError:
        raise SystemExit("need: pip install datasets transformers")

    os.makedirs(a.output_dir, exist_ok=True)
    bin_path = os.path.join(a.output_dir, "train.bin")
    w_path = os.path.join(a.output_dir, "weights.bin")
    meta_path = os.path.join(a.output_dir, "meta.json")

    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    eos = tok.eos_token_id if tok.eos_token_id is not None else 0
    assert len(tok) <= 65535, f"vocab {len(tok)} too big for uint16"

    mix = {"fineweb": a.mix[0], "pg19": a.mix[1], "scrolls": a.mix[2]}
    quota = _interleave_quota(mix, a.shard_size)

    # streaming iterators. .skip() must precede .shuffle() (shuffle only
    # locally reorders a rolling window, so it does not by itself make two
    # differently-seeded runs draw disjoint documents from the ~10B-token
    # source -- skip_docs carves out a val shard past everything a prior
    # train run consumed).
    fw_stream = load_dataset(a.fineweb, name=a.fineweb_subset, split="train",
                             streaming=True)
    if a.skip_docs:
        fw_stream = fw_stream.skip(a.skip_docs)
    fw = iter(fw_stream.shuffle(seed=a.seed, buffer_size=10000))
    fw_docs_consumed = 0
    pg = iter(load_dataset(a.pg19, split="train", streaming=True)
              .shuffle(seed=a.seed, buffer_size=1000))
    # SCROLLS: chain the chosen subsets
    def scrolls_stream():
        for sub in a.scrolls_subsets:
            try:
                ds = load_dataset(a.scrolls, sub, split="train", streaming=True)
            except Exception as e:
                print(f"  [scrolls:{sub}] skipped ({e})")
                continue
            for ex in ds:
                yield sub, ex
    sc = iter(scrolls_stream())

    def pull_fineweb(n):
        nonlocal fw_docs_consumed
        toks, w = [], []
        while len(toks) < n:
            try: ex = next(fw)
            except StopIteration: break
            fw_docs_consumed += 1
            ids = tok(ex.get("text", ""), add_special_tokens=False)["input_ids"]
            ids.append(eos); toks += ids; w += [1] * len(ids)
        return toks, w

    def pull_pg19(n):
        toks, w = [], []
        while len(toks) < n:
            try: ex = next(pg)
            except StopIteration: break
            ids = tok(ex.get("text", ""), add_special_tokens=False)["input_ids"]
            ids.append(eos); toks += ids; w += [1] * len(ids)
        return toks, w

    def pull_scrolls(n):
        toks, w = [], []
        while len(toks) < n:
            try: sub, ex = next(sc)
            except StopIteration: break
            ctx, q, ans = _scrolls_format(ex, sub)
            # tokenize segments SEPARATELY -> exact answer boundary (no BPE drift)
            cids = tok(ctx, add_special_tokens=False)["input_ids"]
            qids = tok(q, add_special_tokens=False)["input_ids"]
            aids = tok(ans, add_special_tokens=False)["input_ids"]
            ids = cids + qids + aids + [eos]
            wt = [1] * (len(cids) + len(qids)) + [a.recall_weight] * len(aids) + [1]
            toks += ids; w += wt
        return toks, w

    pullers = {"fineweb": pull_fineweb, "pg19": pull_pg19, "scrolls": pull_scrolls}

    total = 0
    fbin = open(bin_path, "wb"); fw_ = open(w_path, "wb")
    print(f"Tokenizer: {a.tokenizer} (vocab {len(tok)}) | mix {mix} | recall_w={a.recall_weight}")
    try:
        while True:
            shard_toks, shard_w = [], []
            for src, qn in quota.items():
                if qn == 0:
                    continue
                t, w = pullers[src](qn)
                shard_toks += t; shard_w += w
            if not shard_toks:
                break
            # shuffle at document granularity is already done per-source; here we
            # write the shard as-is (flat-mixed). Packing windows are drawn
            # randomly by the dataset, so intra-shard order is not critical.
            arr = np.asarray(shard_toks, dtype=np.uint16)
            warr = np.asarray(shard_w, dtype=WEIGHT_DTYPE)
            assert arr.shape == warr.shape
            arr.tofile(fbin); warr.tofile(fw_)
            total += arr.size
            print(f"  wrote {total/1e6:.1f}M tokens "
                  f"(scrolls answer-weighted={int((warr>1).sum())/1e3:.1f}K in shard)")
            if a.max_tokens and total >= a.max_tokens:
                break
    finally:
        fbin.close(); fw_.close()

    meta = {
        "n_tokens": total, "vocab_size": len(tok), "tokenizer": a.tokenizer,
        "dtype": "uint16", "weight_dtype": "uint8", "recall_weight": a.recall_weight,
        "mix": mix, "scrolls_subsets": a.scrolls_subsets,
        "skip_docs": a.skip_docs, "fineweb_docs_consumed": fw_docs_consumed,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nDone: {total:,} tokens -> {bin_path} (+ weights.bin)  "
          f"{total*2/1e9:.2f}GB tokens + {total/1e9:.2f}GB weights")


if __name__ == "__main__":
    main()
