"""
pretokenize.py -- Pre-tokenize a mixed corpus into memory-mapped arrays with a
PARALLEL per-token recall-weight stream (dual-stream format).

440M rewrite changes:
  * Tokenizer default -> Llama-2 (meta-llama/Llama-2-7b-hf, 32000 vocab; use
    NousResearch/Llama-2-7b-hf for an ungated mirror with the identical
    tokenizer). uint16 packing still valid (vocab < 65536).
  * SCROLLS examples are reformatted as context -> query -> answer; the ANSWER
    span (short answer for QA subtasks, the full summary for summarization
    subtasks) is up-weighted in the recall-weight stream. Everything else gets
    weight 1.0. This is the "recall pressure" that forces SKA to turn on
    (the MoE-load-balancing analogue) without changing the LM objective shape.

Source mixes (two ways to specify):
  * Legacy 3-float ``--mix f p s`` -> {fineweb, pg19, scrolls}. Default
    [0.55, 0.15, 0.30]. Kept so pretrain.sh and older docs are unchanged.
  * General ``--sources name=frac ...`` -> any subset of the registry in
    ``mix.py`` (fineweb, pg19, scrolls, code, math, cosmopedia). Fractions are
    renormalized to sum to 1. This is the path for the 4-bucket continued-
    pretraining corpus, e.g.
      --sources fineweb=0.40 code=0.125 math=0.125 cosmopedia=0.20 scrolls=0.15

Outputs (in --output_dir):
  train.bin     uint16  flat token ids
  weights.bin   uint8   per-token recall weight (1 normal, RECALL_W on answers)
  meta.json     {n_tokens, vocab_size, tokenizer, dtype, weight_dtype, mix, ...}

The weight stream is read by MemmapPackedDataset (dataset.py) and consumed by
KoopmanLM.forward(loss_weights=...).

Usage:
  python pretokenize.py --output_dir ./tok_440m --max_tokens 30_000_000_000
  python pretokenize.py --output_dir ./cpt --sources fineweb=0.40 code=0.125 \
      math=0.125 cosmopedia=0.20 scrolls=0.15 --max_tokens 2_500_000_000
"""

import argparse
import json
import os
import numpy as np

from koopman_lm.training.data.mix import (
    SOURCE_SPECS, parse_sources, normalize_mix, resolve_specs,
    interleave_quota as _interleave_quota,
)

RECALL_W = 4          # up-weight factor for SCROLLS answer-span tokens
WEIGHT_DTYPE = np.uint8


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


def _qa_context_text(name, ex):
    """Flatten an evidence-bearing QA example into plain causal-LM text.

    The retrieval-oriented LM bucket (Phase 1) trains on the natural
    co-occurrence of a question with its evidence + distractor paragraphs, so
    the model sees "question -> supporting passages" as ordinary text. It does
    NOT use the contrastive objective (that is Phase 2 / retrieval adaptation).

    Schemas differ per dataset; unknown/empty fields degrade to whatever text is
    present, so a schema drift yields shorter documents rather than a crash.
    """
    q = (ex.get("question") or "").strip()
    paras = []
    if name == "hotpotqa":
        ctx = ex.get("context") or {}
        titles = ctx.get("title") or []
        sents = ctx.get("sentences") or []
        for t, ss in zip(titles, sents):
            paras.append((f"{t}. " if t else "") + " ".join(ss or []))
    elif name == "musique":
        for p in (ex.get("paragraphs") or []):
            title = (p.get("title") or "").strip()
            body = (p.get("paragraph_text") or "").strip()
            paras.append((f"{title}. " if title else "") + body)
    elif name == "nq":
        # natural_questions: the document text is heavy/nested; best-effort pull
        # of the title + long-answer HTML-stripped tokens if present.
        doc = ex.get("document") or {}
        title = (doc.get("title") or "").strip()
        if title:
            paras.append(title)
        toks = (doc.get("tokens") or {}).get("token") or []
        if toks:
            paras.append(" ".join(toks[:4000]))   # cap: NQ docs can be enormous
    else:
        paras.append((ex.get("text") or "").strip())
    body = "\n".join(p for p in paras if p).strip()
    return (q + "\n" + body).strip() if q else body


def write_synthetic_corpus(output_dir, n_tokens=200_000, vocab_size=32000,
                           recall_weight=4, seed=0,
                           tokenizer="NousResearch/Llama-2-7b-hf"):
    """Write a tiny self-contained dual-stream corpus (NO network for the
    token generation itself -- tokens are drawn from a synthetic RNG, not
    produced by `tokenizer`).

    Produces train.bin (uint16), weights.bin (uint8), meta.json in the exact
    format MemmapPackedDataset reads. Periodic spans get the recall weight so the
    weighted-CE path is exercised. Used by the end-to-end smoke test so a fresh
    clone can train without downloading FineWeb/PG-19/SCROLLS.

    `tokenizer` is recorded into meta.json as-is and must be an id
    AutoTokenizer.from_pretrained can resolve: train.py calls
    AutoTokenizer.from_pretrained(args.tokenizer), and
    koopman_lm.run.data_verify compares that same string against meta.json --
    the literal "synthetic" satisfies neither, making the smoke path
    otherwise unusable through the run system. Default is
    NousResearch/Llama-2-7b-hf (ungated, vocab 32000, matching the default
    vocab_size here).
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
                   "tokenizer": tokenizer, "dtype": "uint16",
                   "weight_dtype": "uint8", "mix": "synthetic",
                   "recall_weight": int(recall_weight)}, f)
    return output_dir


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", type=str, default="./tok_440m")
    p.add_argument("--smoke", action="store_true",
                   help="write a tiny synthetic corpus (no network) for the e2e smoke test")
    p.add_argument("--smoke_tokens", type=int, default=200_000)
    p.add_argument("--smoke_tokenizer", type=str,
                   default="NousResearch/Llama-2-7b-hf",
                   help="tokenizer id recorded into the smoke corpus's "
                        "meta.json -- must be AutoTokenizer-resolvable so "
                        "train.py and koopman_lm.run.data_verify agree "
                        "(token generation itself stays synthetic/network-free)")
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
                   help="legacy fractions for [fineweb, pg19, scrolls] "
                        "(ignored when --sources is given)")
    p.add_argument("--sources", type=str, nargs="+", default=None,
                   help="general mix 'name=frac ...' (renormalized to sum 1; "
                        "OVERRIDES --mix). Known names: "
                        + ", ".join(sorted(SOURCE_SPECS)))
    p.add_argument("--skip_source", type=str, default="fineweb",
                   help="which source --skip_docs applies to (val-shard carving)")
    # per-source HF-coordinate overrides for the code / math / QA buckets
    p.add_argument("--code_path", type=str, default=None,
                   help="override HF path for the 'code' source (default StarCoder)")
    p.add_argument("--starcoder_data_dir", type=str, default=None,
                   help="data_dir/language subset for StarCoder (e.g. 'python')")
    p.add_argument("--math_path", type=str, default=None,
                   help="override HF path for the 'math' source (default OpenWebMath)")
    p.add_argument("--cosmopedia_path", type=str, default=None,
                   help="override HF path for the 'cosmopedia' source")
    p.add_argument("--cosmopedia_subset", type=str, default=None,
                   help="cosmopedia config (default web_samples_v2)")
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
        write_synthetic_corpus(a.output_dir, n_tokens=a.smoke_tokens, seed=a.seed,
                                tokenizer=a.smoke_tokenizer)
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

    # ---- resolve the source mix (general --sources overrides legacy --mix) ----
    if a.sources:
        mix = normalize_mix(parse_sources(a.sources))
    else:
        mix = {"fineweb": a.mix[0], "pg19": a.mix[1], "scrolls": a.mix[2]}
    overrides = {
        "fineweb":    {"path": a.fineweb, "name": a.fineweb_subset},
        "pg19":       {"path": a.pg19},
        "scrolls":    {"path": a.scrolls, "subsets": a.scrolls_subsets},
        "code":       {"path": a.code_path, "data_dir": a.starcoder_data_dir},
        "math":       {"path": a.math_path},
        "cosmopedia": {"path": a.cosmopedia_path, "name": a.cosmopedia_subset},
    }
    specs = resolve_specs(mix, overrides)
    quota = _interleave_quota(mix, a.shard_size)
    docs_consumed = {name: 0 for name in mix}

    # streaming iterators. .skip() must precede .shuffle() (shuffle only locally
    # reorders a rolling window, so it does not by itself make two differently-
    # seeded runs draw disjoint documents -- --skip_docs carves out a val shard
    # past everything a prior train run consumed, applied to --skip_source).
    def make_plain_stream(name, spec):
        kw = dict(split=spec["split"], streaming=True)
        if spec.get("name"):
            kw["name"] = spec["name"]
        if spec.get("data_dir"):
            kw["data_dir"] = spec["data_dir"]
        if spec.get("trust_remote_code"):
            kw["trust_remote_code"] = True
        ds = load_dataset(spec["path"], **kw)
        if name == a.skip_source and a.skip_docs:
            ds = ds.skip(a.skip_docs)
        buf = 10000 if name == "fineweb" else 1000
        return iter(ds.shuffle(seed=a.seed, buffer_size=buf))

    def make_scrolls_stream(spec):
        # chain the chosen SCROLLS subsets into one stream of (subset, example)
        def gen():
            for sub in spec["subsets"]:
                try:
                    ds = load_dataset(spec["path"], sub, split=spec["split"],
                                      streaming=True)
                except Exception as e:
                    print(f"  [scrolls:{sub}] skipped ({e})")
                    continue
                for ex in ds:
                    yield sub, ex
        return iter(gen())

    streams = {}
    for name, spec in specs.items():
        if quota.get(name, 0) <= 0:
            continue
        streams[name] = (make_scrolls_stream(spec) if spec["kind"] == "scrolls"
                         else make_plain_stream(name, spec))

    def pull_plain(name, n):
        it, tf = streams[name], specs[name]["text_field"]
        toks, w = [], []
        while len(toks) < n:
            try: ex = next(it)
            except StopIteration: break
            docs_consumed[name] += 1
            ids = tok(ex.get(tf, "") or "", add_special_tokens=False)["input_ids"]
            ids.append(eos); toks += ids; w += [1] * len(ids)
        return toks, w

    def pull_scrolls(name, n):
        it = streams[name]
        toks, w = [], []
        while len(toks) < n:
            try: sub, ex = next(it)
            except StopIteration: break
            docs_consumed[name] += 1
            ctx, q, ans = _scrolls_format(ex, sub)
            # tokenize segments SEPARATELY -> exact answer boundary (no BPE drift)
            cids = tok(ctx, add_special_tokens=False)["input_ids"]
            qids = tok(q, add_special_tokens=False)["input_ids"]
            aids = tok(ans, add_special_tokens=False)["input_ids"]
            ids = cids + qids + aids + [eos]
            wt = [1] * (len(cids) + len(qids)) + [a.recall_weight] * len(aids) + [1]
            toks += ids; w += wt
        return toks, w

    def pull_qa(name, n):
        # evidence-bearing QA record -> "question + evidence paragraphs" LM text
        it = streams[name]
        toks, w = [], []
        while len(toks) < n:
            try: ex = next(it)
            except StopIteration: break
            docs_consumed[name] += 1
            text = _qa_context_text(name, ex)
            if not text:
                continue
            ids = tok(text, add_special_tokens=False)["input_ids"]
            ids.append(eos); toks += ids; w += [1] * len(ids)
        return toks, w

    def pull(name, n):
        kind = specs[name]["kind"]
        fn = {"scrolls": pull_scrolls, "qa_context": pull_qa}.get(kind, pull_plain)
        return fn(name, n)

    total = 0
    fbin = open(bin_path, "wb"); fw_ = open(w_path, "wb")
    print(f"Tokenizer: {a.tokenizer} (vocab {len(tok)}) | mix {mix} | "
          f"recall_w={a.recall_weight}")
    try:
        while True:
            shard_toks, shard_w = [], []
            for src, qn in quota.items():
                if qn <= 0 or src not in streams:
                    continue
                t, w = pull(src, qn)
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
                  f"(answer-weighted={int((warr>1).sum())/1e3:.1f}K in shard)")
            if a.max_tokens and total >= a.max_tokens:
                break
    finally:
        fbin.close(); fw_.close()

    meta = {
        "n_tokens": total, "vocab_size": len(tok), "tokenizer": a.tokenizer,
        "dtype": "uint16", "weight_dtype": "uint8", "recall_weight": a.recall_weight,
        "mix": mix, "sources": {k: specs[k]["path"] for k in specs},
        "scrolls_subsets": specs.get("scrolls", {}).get("subsets"),
        "skip_docs": a.skip_docs, "skip_source": a.skip_source,
        "docs_consumed": docs_consumed,
        # back-compat: pretrain.sh reads fineweb_docs_consumed for val carving
        "fineweb_docs_consumed": docs_consumed.get("fineweb", 0),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nDone: {total:,} tokens -> {bin_path} (+ weights.bin)  "
          f"{total*2/1e9:.2f}GB tokens + {total/1e9:.2f}GB weights")


if __name__ == "__main__":
    main()
