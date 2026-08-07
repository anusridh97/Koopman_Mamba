"""
prepare_data.py -- tokenize a HuggingFace text corpus into the flat uint16
token .bin that train_echo.py / TokenStream reads (nanoGPT-style).

Defaults to FineWeb-Edu with a 32k-vocab Llama tokenizer (matches echo_config's
vocab=32000; ids < 65536 so uint16 is exact). Documents are concatenated with the
EOS id as separator. Requires `datasets` and `transformers`:

    pip install datasets transformers

Examples:
    python prepare_data.py --out tokens.bin --tokens 3_000_000_000          # ~3B train tokens
    python prepare_data.py --out val.bin   --tokens 5_000_000 --split train --skip 3_000_000_000
    python prepare_data.py --dataset HuggingFaceFW/fineweb-edu --name sample-10BT \
        --tokenizer meta-llama/Llama-2-7b-hf --out tokens.bin --tokens 5e9
"""
import argparse, sys
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    p.add_argument("--name", default="sample-10BT", help="dataset config/subset name")
    p.add_argument("--split", default="train")
    p.add_argument("--text_key", default="text")
    p.add_argument("--tokenizer", default="meta-llama/Llama-2-7b-hf")
    p.add_argument("--out", required=True)
    p.add_argument("--tokens", type=float, required=True, help="approx number of tokens to write")
    p.add_argument("--skip", type=float, default=0, help="skip this many tokens first (for val split)")
    p.add_argument("--chunk", type=int, default=1 << 20, help="flush granularity (tokens)")
    a = p.parse_args()

    try:
        from datasets import load_dataset
        from transformers import AutoTokenizer
    except ImportError:
        sys.exit("need: pip install datasets transformers")

    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    eos = tok.eos_token_id if tok.eos_token_id is not None else 0
    assert tok.vocab_size < 65536, f"vocab {tok.vocab_size} too big for uint16"
    ds = load_dataset(a.dataset, name=a.name, split=a.split, streaming=True)

    target, skip = int(a.tokens), int(a.skip)
    buf, written, skipped = [], 0, 0
    with open(a.out, "wb") as f:
        for ex in ds:
            ids = tok(ex[a.text_key], add_special_tokens=False)["input_ids"]
            ids.append(eos)
            if skipped < skip:                                  # fast-forward for val split
                skipped += len(ids); continue
            buf.extend(ids)
            if len(buf) >= a.chunk:
                arr = np.asarray(buf, dtype=np.uint16); buf = []
                f.write(arr.tobytes()); written += arr.size
                print(f"\r{written/1e6:.1f}M / {target/1e6:.1f}M tokens", end="", flush=True)
                if written >= target:
                    break
        if buf and written < target:
            f.write(np.asarray(buf, dtype=np.uint16).tobytes()); written += len(buf)
    print(f"\nwrote {written} tokens -> {a.out}  (uint16, {written*2/1e9:.2f} GB)")


if __name__ == "__main__":
    main()
