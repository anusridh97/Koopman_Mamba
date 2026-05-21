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
    per round-robin cycle (integers summing ~chunk_tokens)."""
    return {k: max(1, int(round(v * chunk_tokens))) for k, v in mix.items()}


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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", type=str, default="./tok_440m")
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
    a = p.parse_args()

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

    # streaming iterators
    fw = iter(load_dataset(a.fineweb, name=a.fineweb_subset, split="train",
                           streaming=True).shuffle(seed=a.seed, buffer_size=10000))
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
        toks, w = [], []
        while len(toks) < n:
            try: ex = next(fw)
            except StopIteration: break
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
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nDone: {total:,} tokens -> {bin_path} (+ weights.bin)  "
          f"{total*2/1e9:.2f}GB tokens + {total/1e9:.2f}GB weights")


if __name__ == "__main__":
    main()
