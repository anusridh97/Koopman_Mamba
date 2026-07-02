"""
run_toolcall.py -- large long-context tool-call retrieval sweep.

Trains each model once at a moderate context length, then evaluates it at a range
of much longer contexts (length generalization). The number of key->value bindings
is held fixed while the distractor gap grows, so this is a tool-trace analogue of
needle-in-a-haystack with recency (keys are overwritten; the latest binding is the
answer).

This is where the story diverges from MQAR:
  * SKA fits its operator from constant-size sufficient statistics, so accuracy
    should stay flat as context grows (O(1) inference state), and the Koopman
    dynamics have real temporal structure to exploit (rho > 0).
  * Attention pays O(T^2) and, trained at a short context, tends to degrade as the
    evaluation context grows well beyond training length.
  * Pure Mamba collapses (memory cliff).

Example (scale --eval-contexts up to your GPU budget):
  python -m prefix_bench.run_toolcall --device cuda --steps 4000 \
      --train-context 1024 --eval-contexts 1024 2048 4096 8192 16384
"""

import argparse
import torch

from prefix_bench.toolcall_data import make_toolcall_batch
from prefix_bench.models import HybridLM, count_params
from prefix_bench.train_eval import train_and_eval, evaluate

# (label, kind, ska_mode, attn_mode)
CONDITIONS = [
    ("mamba",         "mamba", "none",   "causal"),
    ("attn/causal",   "attn",  "none",   "causal"),
    ("attn/prefixLM", "attn",  "none",   "prefix_lm"),
    ("ska/prefix",    "ska",   "prefix", "causal"),
    ("ska/causal",    "ska",   "causal", "causal"),
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
    ap.add_argument("--train-context", dest="train_context", type=int, default=512)
    ap.add_argument("--eval-contexts", dest="eval_contexts", type=int, nargs="+",
                    default=[512, 1024, 2048, 4096])
    ap.add_argument("--num-keys", dest="num_keys", type=int, default=16)
    ap.add_argument("--num-query", dest="num_query", type=int, default=8)
    ap.add_argument("--overwrite-prob", dest="overwrite_prob", type=float, default=0.4)
    ap.add_argument("--vocab-size", type=int, default=512)
    ap.add_argument("--d-model", dest="d_model", type=int, default=128)
    ap.add_argument("--n-layers", dest="n_layers", type=int, default=4)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--d-state", dest="d_state", type=int, default=16)
    ap.add_argument("--ska-rank", dest="ska_rank", type=int, default=24)
    ap.add_argument("--power-K", dest="power_K", type=int, default=2)
    ap.add_argument("--chunk-size", dest="chunk_size", type=int, default=64)
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument("--no-rho-gate", action="store_true")
    ap.add_argument("--lag-value", action="store_true")
    ap.add_argument("--no-seg-embed", action="store_true")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--eval-batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--conditions", nargs="+", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    conds = [c for c in CONDITIONS if args.conditions is None or c[0] in args.conditions]

    def sample_fn(bs):
        return make_toolcall_batch(
            bs, args.train_context, args.num_keys, args.num_query,
            vocab_size=args.vocab_size, overwrite_prob=args.overwrite_prob, device="cpu")

    # Fixed held-out eval batches, one per evaluation context length.
    eval_batches = {}
    for ctx in args.eval_contexts:
        g = torch.Generator().manual_seed(20_000 + ctx)
        eval_batches[ctx] = make_toolcall_batch(
            args.eval_batch, ctx, args.num_keys, args.num_query,
            vocab_size=args.vocab_size, overwrite_prob=args.overwrite_prob,
            device="cpu", generator=g)

    print(f"train_context={args.train_context} num_keys={args.num_keys} "
          f"num_query={args.num_query} overwrite_prob={args.overwrite_prob}")

    results = {}
    for label, kind, ska_mode, attn_mode in conds:
        torch.manual_seed(args.seed)
        model = build_model(kind, args, args.vocab_size)
        npar = count_params(model)
        # Train at the (short) training context; eval batch here is just for monitoring.
        train_and_eval(
            model, kind, sample_fn, eval_batches[args.eval_contexts[0]],
            steps=args.steps, batch_size=args.batch_size, lr=args.lr,
            ska_mode=ska_mode, attn_mode=attn_mode, device=args.device,
            label=label, log_every=0)
        results[label] = {}
        for ctx in args.eval_contexts:
            acc = evaluate(model, kind, eval_batches[ctx], ska_mode, attn_mode,
                           0.1, args.device)
            results[label][ctx] = acc
        print(f"  {label:14s} | params {npar:>9,} | " +
              " ".join(f"{ctx}:{results[label][ctx]:.3f}" for ctx in args.eval_contexts))
        del model
        if args.device == "cuda":
            torch.cuda.empty_cache()

    print(f"\n\n{'='*72}\nSUMMARY  (accuracy vs eval context length; trained at "
          f"{args.train_context})\n{'='*72}")
    header = f"{'condition':14s} | " + " | ".join(f"{c:>7d}" for c in args.eval_contexts)
    print(header)
    print("-" * len(header))
    for label, *_ in conds:
        print(f"{label:14s} | " +
              " | ".join(f"{results[label][c]:>7.4f}" for c in args.eval_contexts))


if __name__ == "__main__":
    main()
