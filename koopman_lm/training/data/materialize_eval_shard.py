"""materialize_eval_shard.py -- freeze an evaluation corpus into the packed
MemmapPackedDataset format so held-out perplexity is reproducible.

Why this exists
---------------
``evaluation/evaluate.py::WikiTextDataset`` streams WikiText-103 live from the
Hub and tokenizes on every eval. That is not reproducible: the dataset revision
is unpinned, the tokenizer revision is whatever is cached, and the eval needs
network access. ``eval_fineweb_ppl`` already consumes a frozen directory via
``--held_out_data_dir`` (MemmapPackedDataset, ``shuffle=False``), so writing the
eval corpus in that same format makes held-out PPL a deterministic function of
two checksummed artifacts.

Packing is deliberately byte-compatible with ``WikiTextDataset.__iter__``:
examples whose text is empty or all-whitespace are skipped, each example is
tokenized with ``add_special_tokens=False``, and one ``eos`` is appended per
example. The resulting flat token stream is what that class would have built in
memory, so reading it back through MemmapPackedDataset yields the same windows.

Output (same layout pretokenize.py writes, minus weights.bin, which
MemmapPackedDataset defaults to all-ones -> ordinary unweighted CE):
  train.bin   uint16 flat token ids
  meta.json   {n_tokens, vocab_size, tokenizer, dtype, dataset, revision, ...}

Example
-------
  python -m koopman_lm.training.data.materialize_eval_shard \
      --dataset Salesforce/wikitext --config wikitext-103-raw-v1 --split test \
      --revision b08601e04326c79dfdd32d625aee71d232d685c3 \
      --tokenizer /path/to/frozen/tokenizer --output_dir /path/to/out
"""

import argparse
import hashlib
import json
import os

import numpy as np

DTYPE = np.uint16


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="Salesforce/wikitext")
    p.add_argument("--config", type=str, default="wikitext-103-raw-v1")
    p.add_argument("--split", type=str, default="test")
    p.add_argument("--revision", type=str, default=None,
                   help="pin the dataset revision; REQUIRED for a frozen "
                        "scientific artifact")
    p.add_argument("--tokenizer", type=str, required=True,
                   help="hub id or a frozen local tokenizer directory")
    p.add_argument("--text_field", type=str, default="text")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--max_tokens", type=int, default=None)
    p.add_argument("--max_seq_len", type=int, default=2048,
                   help="only used to report the resulting window count")
    a = p.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer

    os.makedirs(a.output_dir, exist_ok=True)
    bin_path = os.path.join(a.output_dir, "train.bin")
    meta_path = os.path.join(a.output_dir, "meta.json")

    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    assert len(tok) <= 65535, f"vocab {len(tok)} too big for uint16"
    eos = tok.eos_token_id
    assert eos is not None, "tokenizer has no eos_token_id"

    kw = {"split": a.split}
    if a.config:
        kw["name"] = a.config
    if a.revision:
        kw["revision"] = a.revision
    ds = load_dataset(a.dataset, **kw)

    print(f"dataset={a.dataset} config={a.config} split={a.split} "
          f"revision={a.revision} rows={len(ds):,}")
    print(f"tokenizer={a.tokenizer} vocab={len(tok)} eos_id={eos}")

    tokens = []
    n_used = n_skipped = 0
    for ex in ds:
        text = ex.get(a.text_field, "")
        # identical filter to WikiTextDataset.__iter__
        if not text or text.isspace():
            n_skipped += 1
            continue
        ids = tok(text, truncation=False, add_special_tokens=False)["input_ids"]
        tokens.extend(ids)
        tokens.append(eos)
        n_used += 1
        if a.max_tokens and len(tokens) >= a.max_tokens:
            break

    if a.max_tokens:
        tokens = tokens[:a.max_tokens]

    arr = np.asarray(tokens, dtype=DTYPE)
    assert arr.size > 0, "produced an empty shard"
    assert int(arr.max()) < len(tok), "token id outside tokenizer vocab"
    arr.tofile(bin_path)

    sha = hashlib.sha256()
    with open(bin_path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            sha.update(block)

    n_windows = max(1, (arr.size - (a.max_seq_len - 1) - 1) // a.max_seq_len)
    meta = {
        "n_tokens": int(arr.size),
        "vocab_size": len(tok),
        "tokenizer": a.tokenizer,
        "dtype": "uint16",
        "dataset": a.dataset,
        "dataset_config": a.config,
        "split": a.split,
        "dataset_revision": a.revision,
        "rows_used": n_used,
        "rows_skipped_empty": n_skipped,
        "eos_token_id": int(eos),
        "packing": "concat_per_example_with_eos__skip_empty_or_whitespace",
        "packing_parity": "evaluation.evaluate.WikiTextDataset.__iter__",
        "train_bin_sha256": sha.hexdigest(),
        "windows_at_max_seq_len": int(n_windows),
        "max_seq_len_for_window_count": a.max_seq_len,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  rows used={n_used:,} skipped(empty/ws)={n_skipped:,}")
    print(f"  tokens={arr.size:,}  bytes={arr.size*2:,}")
    print(f"  train.bin sha256={sha.hexdigest()}")
    print(f"  windows at max_seq_len={a.max_seq_len}: {n_windows:,}")
    print(f"Done -> {a.output_dir}")


if __name__ == "__main__":
    main()
