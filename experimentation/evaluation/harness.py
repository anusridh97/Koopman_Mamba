#!/usr/bin/env python
"""Unified evaluation harness (scaling plan, Phase 0).

One command, one JSON:

    python eval_harness.py --checkpoint path/to/step_5000/model.pt --out results.json

Reconstructs the model from the checkpoint's embedded config (meta.pt), so the
model SCALE is auto-detected -- no manual config. Produces a single JSON with:
  * perplexity (wikitext)            -- koopman_lm.evaluate.eval_held_out_ppl
  * NIAH accuracy (1K..16K)          -- koopman_lm.evaluate.eval_niah
  * MQAR grid (seq_len x #kv pairs)  -- koopman_lm.evals.mqar
  * RULER subset (4K/8K)             -- koopman_lm.evals.ruler
  * BABILong QA1/QA2 (4K/8K)         -- koopman_lm.evals.babilong
  * SKA-zeroed PPL delta             -- model.ablate(zero_ska=True)
plus a wandb summary if --wandb_project is given.

The metric runners need a GPU (mamba_ssm backbone) and/or network (BABILong);
the scale-detection / plan-selection / JSON-assembly helpers below are pure and
unit-tested on CPU (tests/test_eval_harness.py).
"""
import argparse
import json
import os

import torch

from koopman_lm.config import config_hash, CONFIG_FACTORIES

# tasks the harness can run; --tasks selects a subset
ALL_TASKS = ["ppl", "niah", "mqar", "ruler", "babilong", "ska_delta"]


# --------------------------------------------------------------------------
# Pure helpers (CPU-testable): provenance, scale detection, eval plan, assembly
# --------------------------------------------------------------------------

def load_meta(checkpoint):
    """Load the meta.pt sidecar next to a model.pt checkpoint (or {} if absent)."""
    meta_path = checkpoint.replace("model.pt", "meta.pt")
    if os.path.exists(meta_path):
        return torch.load(meta_path, map_location="cpu", weights_only=False)
    return {}


def detect_scale(meta):
    """Identify the model scale from checkpoint metadata.

    Prefers the stored cfg (hashes it to confirm the recorded cfg_hash and to
    match it against a known factory). Returns a provenance dict.

    `cfg_hash` is the value the checkpoint recorded -- a provenance fact, kept
    verbatim. `cfg_hash_recomputed` is this code's hash of the stored cfg, and
    `cfg_hash_mismatch` is the verdict: True when the two disagree, False when
    they agree, None when the checkpoint recorded no hash to confirm.

    The disagreement case is real and used to be invisible: this function
    overwrote the recorded value with the recomputed one, so a checkpoint whose
    config schema drifted after it was written was silently accepted as
    self-consistent. Confirming a claim means reporting when it fails, not
    replacing the claim with the evidence.
    """
    recorded = meta.get("cfg_hash")
    info = {"model_size": meta.get("model_size"),
            "model_type": meta.get("model_type", "koopman"),
            "cfg_hash": recorded,
            "cfg_hash_recomputed": None,
            "cfg_hash_mismatch": None,
            "param_count": None,
            "matched_factory": None}
    cfg = meta.get("cfg")
    if cfg is not None:
        h = config_hash(cfg)
        info["cfg_hash_recomputed"] = h
        if recorded is None:
            # Nothing was claimed, so there is nothing to contradict; the
            # recomputed hash is the only identity available.
            info["cfg_hash"] = h
        else:
            info["cfg_hash_mismatch"] = recorded != h
        info["param_count"] = int(cfg.param_count_estimate())
        # match against a known scale by hash (auto-detect scale)
        for name, factory in CONFIG_FACTORIES.items():
            try:
                if config_hash(factory()) == h:
                    info["matched_factory"] = name
                    break
            except Exception:
                continue
    return info


def default_eval_plan(cfg, max_seq_len=None):
    """Choose eval sequence lengths / batch sizes for a config's scale.

    Longer contexts and smaller batches at larger d_model to avoid OOM; NIAH /
    RULER / BABILong context lengths are capped by the model's max_seq_len.
    """
    cap = max_seq_len or getattr(cfg, "max_seq_len", 8192)
    d = cfg.d_model
    # batch shrinks with width
    ppl_bs = 8 if d <= 1024 else (4 if d <= 1792 else 2)
    niah_ctx = [c for c in (1024, 2048, 4096, 8192, 16384) if c <= cap]
    long_ctx = [c for c in (4096, 8192) if c <= cap]
    mqar_seq = [c for c in (256, 512, 1024, 2048) if c <= cap]
    return {
        "ppl_batch_size": ppl_bs,
        "ppl_max_seq_len": min(cap, 2048),
        "niah_context_lens": niah_ctx,
        "niah_batch_size": max(1, ppl_bs // 2),
        "mqar_seq_lens": mqar_seq,
        "mqar_kv_pairs": [4, 8, 16, 32, 64],
        "ruler_context_lens": long_ctx,
        "babilong_context_lens": long_ctx,
    }


def assemble_results(checkpoint, scale, plan, metrics):
    """Final JSON: provenance + the eval plan that produced the numbers + metrics."""
    return {
        "checkpoint": checkpoint,
        "scale": scale,
        "eval_plan": plan,
        "metrics": metrics,
    }


# --------------------------------------------------------------------------
# Metric runners (need a model; GPU / network). Each is guarded in run().
# --------------------------------------------------------------------------

def _ppl(model, device, tokenizer, plan):
    from experimentation.evaluation.evaluate import eval_held_out_ppl
    return eval_held_out_ppl(model, device, tokenizer,
                             max_seq_len=plan["ppl_max_seq_len"],
                             batch_size=plan["ppl_batch_size"])


def _ska_delta(model, device, tokenizer, plan, base_ppl=None):
    """PPL(SKA-zeroed) - PPL(full). Positive + large => SKA is load-bearing."""
    from experimentation.evaluation.evaluate import eval_held_out_ppl
    if not hasattr(model, "ablate"):
        return {"supported": False}
    full = base_ppl if base_ppl is not None else _ppl(model, device, tokenizer, plan)["ppl"]
    with model.ablate(zero_ska=True):
        zeroed = eval_held_out_ppl(model, device, tokenizer,
                                   max_seq_len=plan["ppl_max_seq_len"],
                                   batch_size=plan["ppl_batch_size"])["ppl"]
    return {"supported": True, "ppl_full": full, "ppl_ska_zeroed": zeroed,
            "delta": zeroed - full}


def run(args):
    from experimentation.evaluation.evaluate import load_model, eval_niah

    device = torch.device(args.device if args.device else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    model, cfg, tokenizer, model_type = load_model(
        args.checkpoint, args.model_size, args.tokenizer)
    model = model.to(device).eval()

    meta = load_meta(args.checkpoint)
    meta.setdefault("cfg", cfg)
    meta.setdefault("model_type", model_type)
    scale = detect_scale(meta)
    plan = default_eval_plan(cfg, args.max_seq_len)
    tasks = args.tasks or ALL_TASKS
    metrics = {}

    if "ppl" in tasks or "ska_delta" in tasks:
        ppl = _ppl(model, device, tokenizer, plan)
        if "ppl" in tasks:
            metrics["perplexity"] = ppl
    if "niah" in tasks:
        metrics["niah"] = eval_niah(model, device, tokenizer, model_type,
                                    batch_size=plan["niah_batch_size"],
                                    n_examples=args.n_examples,
                                    context_lens=plan["niah_context_lens"])
    if "mqar" in tasks:
        from experimentation.evaluation.mqar.mqar import eval_mqar_grid
        metrics["mqar"] = eval_mqar_grid(
            model, cfg.vocab_size, device, batch=args.mqar_batch,
            seq_lens=tuple(plan["mqar_seq_lens"]),
            kv_pairs=tuple(plan["mqar_kv_pairs"]))
    if "ruler" in tasks:
        from experimentation.evaluation.ruler import eval_ruler_subset
        metrics["ruler"] = eval_ruler_subset(
            model, tokenizer, device,
            context_lens=tuple(plan["ruler_context_lens"]),
            n_examples=args.n_examples)
    if "babilong" in tasks:
        from experimentation.evaluation.babilong import eval_babilong_subset
        metrics["babilong"] = eval_babilong_subset(
            model, tokenizer, device,
            context_lens=tuple(plan["babilong_context_lens"]),
            n_examples=args.n_examples)
    if "ska_delta" in tasks:
        base = metrics.get("perplexity", {}).get("ppl")
        metrics["ska_zeroed_delta"] = _ska_delta(model, device, tokenizer, plan, base)

    results = assemble_results(args.checkpoint, scale, plan, metrics)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Wrote {args.out}")
    else:
        print(json.dumps(results, indent=2, default=str))

    if args.wandb_project:
        import wandb
        group = (f"eval-{scale.get('matched_factory') or scale.get('model_size')}"
                 f"-{(scale.get('cfg_hash') or '')[:8]}")
        wandb.init(project=args.wandb_project, group=group,
                   name=f"eval-{os.path.basename(os.path.dirname(args.checkpoint))}")
        wandb.summary.update({"eval": results})
        wandb.finish()
    return results


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True, help="path to model.pt")
    p.add_argument("--tasks", nargs="+", default=None, choices=ALL_TASKS,
                   help=f"subset of {ALL_TASKS} (default: all)")
    p.add_argument("--out", type=str, default=None, help="JSON output path")
    p.add_argument("--tokenizer", type=str, default="mistralai/Mistral-7B-v0.1")
    p.add_argument("--model_size", type=str, default="180m",
                   help="fallback scale if the checkpoint has no embedded cfg")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--max_seq_len", type=int, default=None,
                   help="cap eval context length (default: model's max_seq_len)")
    p.add_argument("--n_examples", type=int, default=50)
    p.add_argument("--mqar_batch", type=int, default=64)
    p.add_argument("--wandb_project", type=str, default=None)
    return p.parse_args()


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
