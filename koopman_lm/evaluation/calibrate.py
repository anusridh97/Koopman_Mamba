#!/usr/bin/env python
"""Phase-1 threshold calibration.

Runs the SKA health, gradient-flow, and four-mode load-bearing diagnostics on a
REFERENCE model and dumps the RAW continuous values to JSON, so the Phase-2
pruning thresholds ([0.3, 0.95] health band, 0.1 grad-norm ratio, 0.5 ska_delta)
are set from data instead of guessed. Preserve the raw numbers -- do not reduce
to pass/fail here.

Reference set (run once each; then diff the JSONs):
  * completed trained checkpoint   --checkpoint .../final/model.pt
  * untrained init (SAME arch)     --checkpoint .../final/model.pt --init_only
  * from-scratch scale             --init_only --model_size 50m
  * Mamba-only baseline            --checkpoint .../mamba_only/final/model.pt

Health + grad-flow reflect the INPUT distribution: --data wikitext uses real
text (default), --data synthetic uses random token ids (no network needed).

Example:
    python -m koopman_lm.evaluation.calibrate \\
        --checkpoint /labs/.../echo-50m-fineweb-3B/final/model.pt \\
        --out results/calib_echo50m_trained.json
    python -m koopman_lm.evaluation.calibrate \\
        --checkpoint /labs/.../echo-50m-fineweb-3B/final/model.pt --init_only \\
        --out results/calib_echo50m_untrained.json
"""
import argparse
import json
import math
import os

import torch


def _one_batch(tokenizer, cfg, device, data, seq_len, batch_size, seed):
    """A single (input_ids, labels) batch for the health/grad-flow probes."""
    if data == "wikitext":
        from torch.utils.data import DataLoader
        from koopman_lm.evaluation.evaluate import WikiTextDataset
        ds = WikiTextDataset(tokenizer=tokenizer, max_len=seq_len)
        b = next(iter(DataLoader(ds, batch_size=batch_size)))
        return b["input_ids"].to(device), b["labels"].to(device)
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, cfg.vocab_size, (batch_size, seq_len), generator=g)
    labels = torch.randint(0, cfg.vocab_size, (batch_size, seq_len), generator=g)
    return ids.to(device), labels.to(device)


def run(args):
    from koopman_lm.evaluation.evaluate import load_model, eval_held_out_ppl
    from koopman_lm.evaluation.harness import eval_load_bearing
    from koopman_lm.training.diagnostics import SKAHealthMonitor, GradFlowMonitor

    device = torch.device(args.device or
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    model, cfg, tokenizer, model_type = load_model(
        args.checkpoint, args.model_size, args.tokenizer,
        init_only=args.init_only)
    model = model.to(device).eval()

    ids, labels = _one_batch(tokenizer, cfg, device, args.data,
                             args.seq_len, args.batch_size, args.seed)

    out = {"provenance": {
        "checkpoint": args.checkpoint, "init_only": args.init_only,
        "model_type": model_type, "model_size": args.model_size,
        "data": args.data, "seq_len": args.seq_len,
        "batch_size": args.batch_size, "device": str(device)}}

    # --- SKA health (forward only) ---
    try:
        hmon = SKAHealthMonitor(model)
        with torch.no_grad(), hmon.capture():
            model(input_ids=ids)
        out["health"] = hmon.collect(wrap_histograms=False)
    except Exception as e:                                    # noqa: BLE001
        out["health_error"] = repr(e)

    # --- gradient flow (real fwd+bwd, snapshot the accumulated grads) ---
    try:
        gmon = GradFlowMonitor(model)
        model.zero_grad(set_to_none=True)
        model(input_ids=ids, labels=labels)["loss"].backward()
        gmon.snapshot()
        out["grad_flow"] = gmon.collect()
        model.zero_grad(set_to_none=True)
    except Exception as e:                                    # noqa: BLE001
        out["grad_flow_error"] = repr(e)

    # --- four-mode load-bearing (the shared model.ablate path) ---
    try:
        if args.data == "wikitext":
            def ppl_fn(m):
                return eval_held_out_ppl(m, device, tokenizer,
                                         max_seq_len=args.seq_len,
                                         batch_size=args.batch_size)["ppl"]
        else:
            def ppl_fn(m):
                with torch.no_grad():
                    loss = m(input_ids=ids, labels=labels)["loss"].item()
                return math.exp(min(loss, 20))
        out["load_bearing"] = eval_load_bearing(model, ppl_fn)
    except Exception as e:                                    # noqa: BLE001
        out["load_bearing_error"] = repr(e)

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2, default=str)
        print(f"Wrote {args.out}")
    else:
        print(json.dumps(out, indent=2, default=str))
    return out


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=str, default=None,
                   help="path to model.pt (omit with --init_only for a "
                        "from-scratch scale)")
    p.add_argument("--init_only", action="store_true",
                   help="skip load_state_dict: untrained reference")
    p.add_argument("--model_size", type=str, default="50m",
                   help="fallback scale when there is no checkpoint cfg")
    p.add_argument("--tokenizer", type=str, default="NousResearch/Llama-2-7b-hf")
    p.add_argument("--data", type=str, default="wikitext",
                   choices=["wikitext", "synthetic"])
    p.add_argument("--seq_len", type=int, default=2048)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default=None, help="JSON output path")
    return p.parse_args()


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
