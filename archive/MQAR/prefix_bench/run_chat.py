"""
run_chat.py -- Method 1 (chat / stopped-prefix) sweep under Causal Structured
Prefixing. Multi-turn recall with loss only on assistant tokens.

Conditions exercise the release-time machinery (csp.py): SKA consumes per-token
release groups (turns); attention consumes the matching segment/turn causal mask.

  mamba           pure SSM baseline
  attn/causal     attention, plain causal
  attn/segment    attention, segment (turn) causal mask -- the CSP mask for attention
  ska/causal      SKA, fixed chunk-causal (the old mask)
  ska/release     SKA, release-time groups = turns (the CSP mask for SKA)

Example:
  python -m prefix_bench.run_chat --device cuda --steps 4000 \
      --n-rounds 4 --kv-per-round 6 --q-per-round 4 --num-keys 24
"""

import argparse
import torch

from prefix_bench.chat_data import make_chat_batch
from prefix_bench.models import HybridLM, count_params
from prefix_bench.train_eval import train_and_eval

# (label, kind, ska_mode, attn_mode)
CONDITIONS = [
    ("mamba",        "mamba", "release", "causal"),
    ("attn/causal",  "attn",  "release", "causal"),
    ("attn/segment", "attn",  "release", "segment"),
    ("ska/causal",   "ska",   "causal",  "causal"),
    ("ska/release",  "ska",   "release", "causal"),
]


def build_model(kind, args, vocab_size):
    return HybridLM(
        vocab_size=vocab_size, d_model=args.d_model, n_layers=args.n_layers, kind=kind,
        n_heads=args.n_heads, d_state=args.d_state, ska_rank=args.ska_rank,
        use_seg_embed=not args.no_seg_embed, tie_embeddings=True,
        ska_kwargs=dict(power_K=args.power_K, chunk_size=args.chunk_size,
                        ridge=args.ridge, use_rho_gate=not args.no_rho_gate,
                        lag_value=args.lag_value),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n-rounds", dest="n_rounds", type=int, default=4)
    ap.add_argument("--kv-per-round", dest="kv_per_round", type=int, default=6)
    ap.add_argument("--q-per-round", dest="q_per_round", type=int, default=4)
    ap.add_argument("--num-keys", dest="num_keys", type=int, default=24)
    ap.add_argument("--vocab-size", type=int, default=512)
    ap.add_argument("--d-model", dest="d_model", type=int, default=128)
    ap.add_argument("--n-layers", dest="n_layers", type=int, default=4)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--d-state", dest="d_state", type=int, default=16)
    ap.add_argument("--ska-rank", dest="ska_rank", type=int, default=24)
    ap.add_argument("--power-K", dest="power_K", type=int, default=2)
    ap.add_argument("--chunk-size", dest="chunk_size", type=int, default=32)
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument("--no-rho-gate", action="store_true")
    ap.add_argument("--lag-value", action="store_true")
    ap.add_argument("--no-seg-embed", action="store_true")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--eval-batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--conditions", nargs="+", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    conds = [c for c in CONDITIONS if args.conditions is None or c[0] in args.conditions]

    def sample_fn(bs):
        return make_chat_batch(bs, args.n_rounds, args.kv_per_round, args.q_per_round,
                               args.num_keys, vocab_size=args.vocab_size, device="cpu")

    eval_gen = torch.Generator().manual_seed(30_000)
    eval_batch = make_chat_batch(args.eval_batch, args.n_rounds, args.kv_per_round,
                                 args.q_per_round, args.num_keys,
                                 vocab_size=args.vocab_size, device="cpu", generator=eval_gen)

    print(f"chat: n_rounds={args.n_rounds} kv_per_round={args.kv_per_round} "
          f"q_per_round={args.q_per_round} num_keys={args.num_keys} "
          f"seq_len={eval_batch[5]['seq_len']}")

    results = {}
    for label, kind, ska_mode, attn_mode in conds:
        torch.manual_seed(args.seed)
        model = build_model(kind, args, args.vocab_size)
        npar = count_params(model)
        out = train_and_eval(
            model, kind, sample_fn, eval_batch,
            steps=args.steps, batch_size=args.batch_size, lr=args.lr,
            ska_mode=ska_mode, attn_mode=attn_mode, device=args.device,
            label=label, log_every=0)
        results[label] = out["best_acc"]
        print(f"  {label:14s} | params {npar:>9,} | best acc {out['best_acc']:.4f}")
        del model
        if args.device == "cuda":
            torch.cuda.empty_cache()

    print(f"\n{'='*50}\nSUMMARY (assistant-token recall accuracy)\n{'='*50}")
    for label, *_ in conds:
        print(f"  {label:14s} | {results[label]:.4f}")


if __name__ == "__main__":
    main()
