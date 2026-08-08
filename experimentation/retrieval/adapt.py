"""adapt.py -- Phase-2 retrieval adaptation for the Koopman LM.

Turns a pretrained (Phase-1 continued) checkpoint into a dense dual-encoder with
a short, compute-cheap contrastive run: short query/passage lengths (64/256), an
InfoNCE objective, and an LM anchor to protect general LM quality. No long
concatenated contexts.

  loss (schedule mode, default): interleave 4 retrieval batches : 1 LM batch.
      retrieval step -> InfoNCE(query, positive, hard-negs + in-batch).
      LM step        -> lm_anchor_weight * CE on the Phase-1 corpus.
  loss (combined mode): every step -> InfoNCE + lm_anchor_weight * CE.

Both realizations of the plan's "L = L_retrieval + 0.1 L_LM" are provided; the
default 4:1 schedule matches the plan's batch-schedule instruction.

Backbone and projection head are optimized with SEPARATE learning rates
(backbone ~8e-6, projection ~1e-4) via RetrievalEncoder.param_groups.

Recall@{1,5,20} on a held-out in-domain pool is measured every --eval_steps.
Zero-shot LM benchmarks (MMLU/ARC/PIQA/HellaSwag/WinoGrande) are intentionally
NOT run in-loop (too slow); checkpoints are saved every --eval_steps so
scripts/eval_sweep.sh can rank them on the full suite offline.

Usage:
  python -m experimentation.retrieval.adapt \
      --init_from runs/echo-180m_v2-cpt/final/model.pt --model_size 180m_v2 \
      --lm_data_dir data/mix_180m_v2_cpt_train --output_dir runs/echo-180m_v2-ret
"""

import argparse
import dataclasses
import math
import os
import time

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from koopman_lm.config import build_config, config_hash
from koopman_lm.models.koopman_lm import KoopmanLM
from experimentation.retrieval.encoder import RetrievalEncoder, info_nce
from experimentation.retrieval.data import build_iterable_dataset, extract_pairs, RETRIEVAL_SOURCE_SPECS
from experimentation.training.data.dataset import MemmapPackedDataset


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_backbone(checkpoint, model_size, tokenizer, device):
    cfg = build_config(model_size)
    cfg = dataclasses.replace(cfg, vocab_size=len(tokenizer))
    model = KoopmanLM(cfg)
    if checkpoint:
        if not os.path.exists(checkpoint):
            raise SystemExit(f"--init_from checkpoint not found: {checkpoint}")
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"  Loaded backbone from {checkpoint} "
              f"({len(missing)} missing, {len(unexpected)} unexpected keys)")
    return model.to(device), cfg


# ---------------------------------------------------------------------------
# Recall@k eval (in-domain held-out pool)
# ---------------------------------------------------------------------------

def _tok_batch(tok, texts, length, device):
    pad = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)
    ids, masks = [], []
    for t in texts:
        x = tok(t or "", add_special_tokens=False, truncation=True,
                max_length=length)["input_ids"]
        m = [1] * len(x)
        if len(x) < length:
            x = x + [pad] * (length - len(x)); m = m + [0] * (length - len(m))
        ids.append(x); masks.append(m)
    return (torch.tensor(ids, device=device),
            torch.tensor(masks, dtype=torch.float32, device=device))


@torch.no_grad()
def evaluate_recall(encoder, eval_pairs, tok, q_len, p_len, device,
                    ks=(1, 5, 20), bs=64):
    """Recall@k: rank each query's gold positive against the pool of all
    positives. eval_pairs: list[ContrastivePair]. Returns {k: recall}."""
    encoder.eval()
    queries = [p.query for p in eval_pairs]
    corpus = [p.positive for p in eval_pairs]      # gold index i == query i
    def embed(texts, length):
        out = []
        for s in range(0, len(texts), bs):
            ids, m = _tok_batch(tok, texts[s:s + bs], length, device)
            out.append(encoder.embed(ids, m))
        return torch.cat(out, 0)
    q = embed(queries, q_len)                       # (N, D)
    d = embed(corpus, p_len)                        # (N, D)
    sims = q @ d.t()                                # (N, N), cosine (unit norm)
    N = sims.shape[0]
    gold = torch.arange(N, device=device)
    ranks = (sims.argsort(dim=1, descending=True) == gold[:, None])
    out = {}
    for k in ks:
        out[k] = ranks[:, :min(k, N)].any(dim=1).float().mean().item()
    encoder.train()
    return out


def _collect_eval_pairs(mix, tok, q_len, p_len, n_hard, seed, n_eval):
    """Materialize a fixed held-out set of ContrastivePairs (distinct seed)."""
    from experimentation.retrieval._torch_data import _cycle
    import random
    rng = random.Random(seed)
    from experimentation.retrieval.data import sample_source
    specs = RETRIEVAL_SOURCE_SPECS
    streams = {name: _cycle(specs[name], seed + i) for i, name in enumerate(mix)}
    pairs = []
    while len(pairs) < n_eval:
        name = sample_source(mix, rng)
        try:
            ex = next(streams[name])
        except StopIteration:
            continue
        for p in extract_pairs(name, ex, rng):
            if p.positive:
                pairs.append(p)
    return pairs[:n_eval]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _encode_negs(encoder, n_ids, n_mask):
    B, K, L = n_ids.shape
    z = encoder.embed(n_ids.reshape(B * K, L), n_mask.reshape(B * K, L))
    return z.reshape(B, K, -1)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    backbone, cfg = load_backbone(args.init_from, args.model_size, tok, device)
    encoder = RetrievalEncoder(backbone, cfg.d_model, proj_dim=args.proj_dim,
                               pool=args.pool).to(device)
    encoder.train()

    # retrieval mix (name=frac), default HotpotQA/MuSiQue/Wikipedia self-sup.
    mix = _parse_mix(args.sources)
    ds = build_iterable_dataset(mix, tok, q_len=args.q_len, p_len=args.p_len,
                                n_hard=args.n_hard, seed=args.seed)
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers,
                        pin_memory=True,
                        prefetch_factor=2 if args.num_workers > 0 else None)
    ret_iter = iter(loader)

    # LM anchor data (the Phase-1 corpus); optional but recommended.
    lm_iter = None
    if args.lm_data_dir:
        lm_ds = MemmapPackedDataset(args.lm_data_dir, args.lm_seq_len, seed=args.seed)
        lm_loader = DataLoader(lm_ds, batch_size=args.lm_batch_size, shuffle=True,
                               num_workers=1, drop_last=True)
        lm_iter = iter(lm_loader)
        print(f"  LM anchor: {lm_ds.n_tokens/1e9:.2f}B tokens from {args.lm_data_dir}")
    elif args.lm_anchor_weight > 0:
        print("  WARNING: --lm_anchor_weight>0 but no --lm_data_dir; LM anchor disabled")

    opt = torch.optim.AdamW(
        encoder.param_groups(args.backbone_lr, args.proj_lr, args.weight_decay),
        betas=(0.9, 0.95))
    sched = get_cosine_schedule_with_warmup(opt, args.warmup_steps, args.max_steps)
    autocast = torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=args.bf16)

    eval_pairs = _collect_eval_pairs(mix, tok, args.q_len, args.p_len, args.n_hard,
                                     args.seed + 99991, args.n_eval)
    print(f"  Held-out eval pool: {len(eval_pairs)} pairs")

    def next_lm_loss():
        nonlocal lm_iter
        if lm_iter is None:
            return torch.zeros((), device=device)
        try:
            b = next(lm_iter)
        except StopIteration:
            lm_iter = iter(lm_loader); b = next(lm_iter)
        ids = b["input_ids"].to(device); labels = b["labels"].to(device)
        lw = b.get("loss_weights")
        lw = lw.to(device) if lw is not None else None
        return backbone(input_ids=ids, labels=labels, loss_weights=lw)["loss"]

    def next_ret_batch():
        nonlocal ret_iter
        try:
            return next(ret_iter)
        except StopIteration:
            ret_iter = iter(loader); return next(ret_iter)

    print(f"\nAdapting: {args.max_steps} steps | mode={args.lm_mode} | "
          f"q={args.q_len} p={args.p_len} K={args.n_hard} tau={args.temperature} "
          f"| lr(bb={args.backbone_lr}, proj={args.proj_lr})")
    os.makedirs(args.output_dir, exist_ok=True)
    t0 = time.time()
    period = args.retrieval_per_lm + 1                 # 4 retrieval : 1 LM

    for step in range(1, args.max_steps + 1):
        is_lm_step = (args.lm_mode == "schedule" and lm_iter is not None
                      and step % period == 0)
        opt.zero_grad(set_to_none=True)
        with autocast:
            if is_lm_step:
                loss = args.lm_anchor_weight * next_lm_loss()
                rloss = float("nan")
            else:
                b = next_ret_batch()
                q = encoder.embed(b["q_ids"].to(device), b["q_mask"].to(device))
                p = encoder.embed(b["p_ids"].to(device), b["p_mask"].to(device))
                dneg = _encode_negs(encoder, b["n_ids"].to(device),
                                    b["n_mask"].to(device))
                loss = info_nce(q, p, dneg, args.temperature)
                rloss = loss.item()
                if args.lm_mode == "combined" and lm_iter is not None:
                    loss = loss + args.lm_anchor_weight * next_lm_loss()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), args.max_grad_norm)
        opt.step(); sched.step()

        if step % args.logging_steps == 0:
            tps = step / (time.time() - t0)
            tag = "LM " if is_lm_step else "ret"
            print(f"step {step:>5d}/{args.max_steps} | {tag} loss {loss.item():.4f} "
                  f"| infonce {rloss:.4f} | {tps:.2f} it/s")
        if step % args.eval_steps == 0 or step == args.max_steps:
            r = evaluate_recall(encoder, eval_pairs, tok, args.q_len, args.p_len,
                                device)
            print(f"  [eval @ {step}] " + "  ".join(f"R@{k}={v:.3f}" for k, v in r.items()))
            _save(encoder, backbone, cfg, tok, step, args)

    print(f"\nDone in {(time.time()-t0)/60:.1f} min -> {args.output_dir}")


def _save(encoder, backbone, cfg, tok, step, args, dirname=None):
    d = os.path.join(args.output_dir, dirname or f"step_{step}")
    os.makedirs(d, exist_ok=True)
    # backbone state (loadable by eval_sweep.sh / lm_harness) + full encoder.
    torch.save(backbone.state_dict(), os.path.join(d, "model.pt"))
    torch.save(encoder.state_dict(), os.path.join(d, "retrieval_encoder.pt"))
    torch.save({"step": step, "cfg": cfg, "cfg_hash": config_hash(cfg),
                "proj_dim": args.proj_dim, "pool": args.pool,
                "model_type": "koopman", "model_size": args.model_size},
               os.path.join(d, "meta.pt"))
    tok.save_pretrained(d)
    print(f"  saved {d}")


def _parse_mix(pairs):
    mix = {}
    for e in pairs:
        name, frac = e.split("=")
        if name not in RETRIEVAL_SOURCE_SPECS:
            raise SystemExit(f"unknown retrieval source {name!r}; "
                             f"known: {sorted(RETRIEVAL_SOURCE_SPECS)}")
        mix[name] = float(frac)
    total = sum(mix.values())
    return {k: v / total for k, v in mix.items()}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--init_from", type=str, required=True,
                   help="Phase-1 continued-pretraining checkpoint model.pt")
    p.add_argument("--model_size", type=str, default="180m_v2")
    p.add_argument("--tokenizer", type=str, default="NousResearch/Llama-2-7b-hf")
    p.add_argument("--output_dir", type=str, default="./runs/echo-ret")
    # retrieval mix (fractions renormalized). NQ evidence ~ Wikipedia (covered).
    p.add_argument("--sources", type=str, nargs="+",
                   default=["hotpotqa=0.35", "wikipedia=0.45", "musique=0.20"],
                   help="name=frac over {hotpotqa, musique, wikipedia}")
    # objective / lengths
    p.add_argument("--q_len", type=int, default=64)
    p.add_argument("--p_len", type=int, default=256)
    p.add_argument("--n_hard", type=int, default=2)
    p.add_argument("--temperature", type=float, default=0.05)
    p.add_argument("--proj_dim", type=int, default=768)
    p.add_argument("--pool", type=str, default="mean", choices=["mean", "last"])
    # LM anchor
    p.add_argument("--lm_mode", type=str, default="schedule",
                   choices=["schedule", "combined"],
                   help="'schedule': 4 retrieval : 1 LM batch; "
                        "'combined': InfoNCE + w*LM every step")
    p.add_argument("--retrieval_per_lm", type=int, default=4)
    p.add_argument("--lm_anchor_weight", type=float, default=1.0,
                   help="weight on the LM loss (use ~0.1 with --lm_mode combined)")
    p.add_argument("--lm_data_dir", type=str, default=None,
                   help="Phase-1 tokenized corpus dir for the LM anchor")
    p.add_argument("--lm_seq_len", type=int, default=1024)
    p.add_argument("--lm_batch_size", type=int, default=4)
    # optimization
    p.add_argument("--max_steps", type=int, default=5000)
    p.add_argument("--warmup_steps", type=int, default=250)
    p.add_argument("--batch_size", type=int, default=128,
                   help="contrastive batch (in-batch negatives = this - 1)")
    p.add_argument("--backbone_lr", type=float, default=8e-6)
    p.add_argument("--proj_lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--no_bf16", action="store_false", dest="bf16")
    p.add_argument("--num_workers", type=int, default=2)
    # eval / logging
    p.add_argument("--eval_steps", type=int, default=500)
    p.add_argument("--n_eval", type=int, default=512)
    p.add_argument("--logging_steps", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    train(parse_args())


if __name__ == "__main__":
    main()
