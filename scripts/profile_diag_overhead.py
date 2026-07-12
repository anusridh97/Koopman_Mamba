#!/usr/bin/env python
"""Measure the amortized overhead of the SKA health diagnostics on the target GPU.

Phase-1 target: the diagnostics logging must cost < 3% of step time at the
chosen diag_every cadence (N=100 for 50M, N=500 for 440M+). Run on a GPU node:

    python scripts/profile_diag_overhead.py --model_size 50m --diag_every 100

Exits non-zero if the amortized overhead exceeds 3%.
"""
import argparse

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_size", default="50m")
    p.add_argument("--seq_len", type=int, default=2048)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--diag_every", type=int, default=100)
    args = p.parse_args()

    from koopman_lm.config import build_config
    from koopman_lm.models.koopman_lm import KoopmanLM
    from koopman_lm.training.diagnostics import SKAHealthMonitor, profile_overhead

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = build_config(args.model_size)
    model = KoopmanLM(cfg).to(device).train()

    def batch_fn():
        ids = torch.randint(0, cfg.vocab_size, (args.batch_size, args.seq_len),
                            device=device)
        return {"input_ids": ids, "labels": ids}

    monitor = SKAHealthMonitor(model)
    stats = profile_overhead(model, batch_fn, monitor,
                             n_warmup=2, n_iter=10, device=device)
    for k, v in stats.items():
        print(f"  {k}: {v}")

    plain = stats["plain_step_s"]
    frac = stats["diag_extra_s"] / args.diag_every / plain if plain else float("nan")
    print(f"\namortized diagnostics overhead at diag_every={args.diag_every}: "
          f"{frac * 100:.3f}%")
    if not frac < 0.03:
        raise SystemExit(f"FAIL: overhead {frac * 100:.2f}% exceeds the 3% target")
    print("OK: diagnostics overhead < 3% of step time")


if __name__ == "__main__":
    main()
