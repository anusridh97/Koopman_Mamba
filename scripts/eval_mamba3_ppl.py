#!/usr/bin/env python
"""Held-out perplexity for a Mamba-3 release checkpoint, on OUR val shard.

Why this exists: a perplexity quoted in someone else's paper is measured on
their held-out split. Ours is measured on FineWeb-Edu documents 0-19,999 of
sample/100BT/000_00000.parquet. Those are different texts, so the two numbers
cannot be subtracted -- a matched tokenizer is necessary but not sufficient.
The only way to get a comparable figure is to run their weights over our shard,
which is what this does.

The loss accumulation deliberately mirrors
experimentation/evaluation/evaluate.py:eval_fineweb_ppl exactly -- same
MemmapPackedDataset, same seed, same shuffle=False ordering, same
sum(loss * n_tokens) / sum(n_tokens), same exp() -- so the number it prints is
directly subtractable from ours rather than merely similar in spirit.

Mamba-3 returns logits rather than a loss, so the cross-entropy is computed
here with the same shift and reduction HuggingFace-style heads use, which is
what KoopmanLM.forward does internally.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf_path", required=True)
    ap.add_argument("--held_out_data_dir", required=True)
    ap.add_argument("--tokenizer", default="NousResearch/Meta-Llama-3.1-8B")
    ap.add_argument("--max_seq_len", type=int, default=2048)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    from ruler_predict import load_mamba3          # applies the Mamba3 dispatch patch
    from experimentation.training.data.dataset import MemmapPackedDataset

    model, _, _ = load_mamba3(args.hf_path, args.tokenizer)
    model.eval()

    dataset = MemmapPackedDataset(args.held_out_data_dir, args.max_seq_len, seed=0)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].cuda()
            labels = batch["labels"].cuda()
            logits = model(input_ids).logits
            # Same shift + mean reduction the Koopman head applies internally,
            # so this is comparable to eval_fineweb_ppl rather than merely close.
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)).float(),
                labels.view(-1),
                ignore_index=-100,
            )
            n = labels.numel()
            total_loss += loss.item() * n
            total_tokens += n

    avg = total_loss / max(total_tokens, 1)
    ppl = math.exp(min(avg, 20))
    print(f"  Tokens: {total_tokens:,}")
    print(f"  Loss:   {avg:.4f}")
    print(f"  PPL:    {ppl:.4f}")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        json.dump({"checkpoint": args.hf_path,
                   "held_out_data_dir": args.held_out_data_dir,
                   "metrics": {"fineweb_ppl": {"loss": avg, "ppl": ppl,
                                               "n_tokens": total_tokens}}},
                  open(args.output, "w"), indent=2)
        print(f"  wrote {args.output}")


if __name__ == "__main__":
    main()
