"""
run_mqar.py -- MQAR prefix-mask taxonomy sweep.

Compares, at matched model size, how each architecture uses (or fails to use) the
context/query boundary that MQAR makes available in the data:

  mamba          pure SSM (no global retrieval)                    baseline
  attn/causal    attention, plain causal mask (ignores prefix)
  attn/prefixLM  attention, prefix-LM mask (prefix passed in)
  ska/none       SKA, no mask -- queries pollute their own operator (broken baseline)
  ska/prefix     SKA, hard context/query split from the SEP token  (paper "prefix mode")
  ska/soft       SKA, soft mask (queries down-weighted, not zeroed)
  ska/causal     SKA, chunk-causal streaming (no boundary handed over)

All models additionally receive a shared segment embedding (seg_ids), so the
"where do queries start" signal is available to everyone.

Example:
  python -m prefix_bench.run_mqar --device cuda --steps 4000 --seq-lens 64 128 256 512
"""

import argparse
import torch

from prefix_bench.mqar_data import make_mqar_batch
from prefix_bench.models import HybridLM, count_params
from prefix_bench.train_eval import train_and_eval

# (label, kind, ska_mode, attn_mode)
CONDITIONS = [
    ("mamba",         "mamba", "none",   "causal"),
    ("attn/causal",   "attn",  "none",   "causal"),
    ("attn/prefixLM", "attn",  "none",   "prefix_lm"),
    ("ska/none",      "ska",   "none",   "causal"),
    ("ska/prefix",    "ska",   "prefix", "causal"),
    ("ska/soft",      "ska",   "soft",   "causal"),
    ("ska/causal",    "ska",   "causal", "causal"),
]


def num_kv_for(seq_len: int) -> int:
    # roughly one KV pair per 16 context tokens, matching the Zoology-style grid
    return max(4, seq_len // 16)


def build_model(kind, args, vocab_size, n_heads):
    return HybridLM(
        vocab_size=vocab_size, d_model=args.d_model, n_layers=args.n_layers, kind=kind,
        n_heads=n_heads, d_state=args.d_state, ska_rank=args.ska_rank,
        use_seg_embed=not args.no_seg_embed, tie_embeddings=True,
        ska_kwargs=dict(power_K=args.power_K, chunk_size=args.chunk_size,
                        ridge=args.ridge, use_rho_gate=not args.no_rho_gate,
                        lag_value=args.lag_value),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seq-lens", type=int, nargs="+", default=[64, 128, 256])
    ap.add_argument("--vocab-size", type=int, default=256)
    ap.add_argument("--d-model", dest="d_model", type=int, default=128)
    ap.add_argument("--n-layers", dest="n_layers", type=int, default=4)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--d-state", dest="d_state", type=int, default=16)
    ap.add_argument("--ska-rank", dest="ska_rank", type=int, default=24)
    ap.add_argument("--power-K", dest="power_K", type=int, default=2)
    ap.add_argument("--chunk-size", dest="chunk_size", type=int, default=16)
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument("--no-rho-gate", action="store_true")
    ap.add_argument("--lag-value", action="store_true")
    ap.add_argument("--no-seg-embed", action="store_true")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--eval-batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--conditions", nargs="+", default=None,
                    help="subset of condition labels to run")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    conds = [c for c in CONDITIONS if args.conditions is None or c[0] in args.conditions]
    n_heads = args.n_heads
    results = {}

    for seq_len in args.seq_lens:
        num_kv = num_kv_for(seq_len)
        print(f"\n{'='*72}\nMQAR seq_len={seq_len}  num_kv={num_kv}  vocab={args.vocab_size}\n{'='*72}")

        def sample_fn(bs, _sl=seq_len, _kv=num_kv):
            return make_mqar_batch(bs, _sl, _kv, vocab_size=args.vocab_size, device="cpu")

        eval_gen = torch.Generator().manual_seed(10_000 + seq_len)
        eval_batch = make_mqar_batch(args.eval_batch, seq_len, num_kv,
                                     vocab_size=args.vocab_size, device="cpu",
                                     generator=eval_gen)
        results[seq_len] = {}
        for label, kind, ska_mode, attn_mode in conds:
            torch.manual_seed(args.seed)
            model = build_model(kind, args, args.vocab_size, n_heads)
            npar = count_params(model)
            out = train_and_eval(
                model, kind, sample_fn, eval_batch,
                steps=args.steps, batch_size=args.batch_size, lr=args.lr,
                ska_mode=ska_mode, attn_mode=attn_mode, device=args.device,
                label=label, log_every=0)
            results[seq_len][label] = out["best_acc"]
            print(f"  {label:14s} | params {npar:>9,} | best acc {out['best_acc']:.4f}")
            del model
            if args.device == "cuda":
                torch.cuda.empty_cache()

    # Summary table: rows = condition, cols = seq_len
    print(f"\n\n{'='*72}\nSUMMARY  (best query accuracy)\n{'='*72}")
    header = f"{'condition':14s} | " + " | ".join(f"{sl:>7d}" for sl in args.seq_lens)
    print(header)
    print("-" * len(header))
    for label, *_ in conds:
        row = f"{label:14s} | " + " | ".join(
            f"{results[sl][label]:>7.4f}" for sl in args.seq_lens)
        print(row)


if __name__ == "__main__":
    main()
