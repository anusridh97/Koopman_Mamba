"""
pretokenize.py -- Pre-tokenize a HuggingFace dataset into memory-mapped numpy arrays.

This eliminates the single biggest training bottleneck: on-the-fly tokenization
in the DataLoader. The output is a single .bin file (uint16 token IDs) + .json
metadata that the fast DataLoader reads with zero-copy via np.memmap.

Usage:
  python pretokenize.py \
      --dataset_name HuggingFaceFW/fineweb-edu \
      --dataset_subset sample-10BT \
      --tokenizer mistralai/Mistral-7B-v0.1 \
      --output_dir ./tokenized_data \
      --max_tokens 10_000_000_000  # stop after 10B tokens (optional)

Output:
  ./tokenized_data/train.bin    -- flat uint16 array of token IDs
  ./tokenized_data/meta.json    -- {n_tokens, vocab_size, tokenizer, ...}
"""

import argparse
import json
import os
import numpy as np
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_name", type=str, default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--dataset_subset", type=str, default="sample-10BT")
    p.add_argument("--tokenizer", type=str, default="mistralai/Mistral-7B-v0.1")
    p.add_argument("--output_dir", type=str, default="./tokenized_data")
    p.add_argument("--max_tokens", type=int, default=None,
                   help="Stop after this many tokens (default: process all)")
    p.add_argument("--shard_size", type=int, default=100_000_000,
                   help="Write buffer size in tokens before flushing")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    bin_path = os.path.join(args.output_dir, "train.bin")
    meta_path = os.path.join(args.output_dir, "meta.json")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    eos_id = tokenizer.eos_token_id

    # Validate vocab fits in uint16 (max 65535). Token IDs are stored as
    # uint16 for 2x memory savings over uint32. Mistral (32000), Llama (32000),
    # and GPT-2 (50257) all fit. If you use a tokenizer with >65535 tokens,
    # change the dtype to np.uint32 here and in MemmapPackedDataset.
    assert len(tokenizer) <= 65535, (
        f"Vocab size {len(tokenizer)} exceeds uint16 max (65535). "
        f"Change dtype to np.uint32 in pretokenize.py and MemmapPackedDataset."
    )

    print(f"Loading dataset: {args.dataset_name}/{args.dataset_subset}")
    ds = load_dataset(args.dataset_name, args.dataset_subset,
                      split="train", streaming=True)

    # We write in append mode, flushing every shard_size tokens
    total_tokens = 0
    buffer = []

    # Use 'ab' (append binary) so we can flush in chunks
    with open(bin_path, 'wb') as f:
        for i, example in enumerate(ds):
            text = example.get("text", "")
            if not text:
                continue

            tokens = tokenizer(text, truncation=False,
                               add_special_tokens=False)["input_ids"]
            buffer.extend(tokens)
            buffer.append(eos_id)

            if len(buffer) >= args.shard_size:
                # Flush to disk
                arr = np.array(buffer, dtype=np.uint16)
                arr.tofile(f)
                total_tokens += len(buffer)
                buffer = []

                print(f"  Flushed {total_tokens:,} tokens "
                      f"({total_tokens / 1e9:.2f}B) ...")

                if args.max_tokens and total_tokens >= args.max_tokens:
                    print(f"  Reached max_tokens={args.max_tokens:,}, stopping.")
                    break

            if (i + 1) % 100_000 == 0:
                print(f"  Processed {i+1:,} documents, "
                      f"buffer={len(buffer):,} tokens ...")

        # Flush remaining
        if buffer:
            arr = np.array(buffer, dtype=np.uint16)
            arr.tofile(f)
            total_tokens += len(buffer)

    # Write metadata
    meta = {
        "n_tokens": total_tokens,
        "vocab_size": len(tokenizer),
        "tokenizer": args.tokenizer,
        "dataset_name": args.dataset_name,
        "dataset_subset": args.dataset_subset,
        "dtype": "uint16",
    }
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    size_gb = os.path.getsize(bin_path) / (1024**3)
    print(f"\nDone! {total_tokens:,} tokens ({total_tokens/1e9:.2f}B)")
    print(f"  {bin_path} ({size_gb:.2f} GB)")
    print(f"  {meta_path}")


if __name__ == "__main__":
    main()
