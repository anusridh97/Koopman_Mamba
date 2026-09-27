#!/usr/bin/env python
"""Measure training-time peak memory AND step time vs per_device_batch_size.

    python scripts/probe_pdbs.py --model_config <run_dir>/model_config.json --pdbs 1,2,4

Unlike scripts/measure_train_memory.py this takes the run's OWN
model_config.json rather than a name from a fixed size table, so it profiles
the geometry that is actually training.

Two numbers per microbatch, both around a REAL fwd+bwd+optimizer step:

  * torch.cuda.max_memory_allocated() -- the only figure CLAUDE.md accepts for
    choosing per_device_batch_size. DDP adds a gradient bucket roughly the size
    of the parameters (4 bytes/param fp32) on top of what this single process
    measures, so the printed peak is a FLOOR for the real per-rank figure. That
    bucket is reported separately and added into the headroom check.
  * tokens/s -- because the point of raising pdbs here is throughput. The run
    is at ~5% MFU with pdbs 1, where each GPU sees a single 2048-token sequence
    per micro-step and is launch-bound rather than compute-bound.

per_device_batch_size is NOT hashed into run_id, so a value chosen here can be
applied to the next chunk of an in-flight run without changing its identity.
"""

import argparse
import json
import time

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_config", required=True)
    ap.add_argument("--pdbs", default="1,2,4")
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--steps", type=int, default=6, help="timed steps after warmup")
    args = ap.parse_args()

    from koopman_lm.config import build_config
    from koopman_lm.models.koopman_lm import KoopmanLM

    cfg = build_config(args.model_config)
    total_gib = torch.cuda.get_device_properties(0).total_memory / 2**30
    print(f"device: {torch.cuda.get_device_name(0)}  {total_gib:.1f} GiB")
    print(f"geometry: d_model={cfg.d_model} n_layers={cfg.n_layers} "
          f"d_state={cfg.d_state} ska_rank={cfg.ska_rank} "
          f"ska_layers={len(cfg.ska_layer_indices)} backend={cfg.ska_backend}")

    model = KoopmanLM(cfg).cuda()
    nparam = sum(p.numel() for p in model.parameters())
    ddp_bucket = nparam * 4 / 2**30          # fp32 gradient bucket DDP allocates
    print(f"params: {nparam:,}   DDP grad bucket ~{ddp_bucket:.1f} GiB\n")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5, fused=True)

    print(f"{'pdbs':>5} {'peak GiB':>9} {'+DDP':>7} {'free':>7} "
          f"{'s/step':>8} {'tok/s/GPU':>11} {'vs pdbs1':>9}")
    base = None
    for pdbs in [int(x) for x in args.pdbs.split(",")]:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        ids = torch.randint(0, cfg.vocab_size, (pdbs, args.seq), device="cuda")
        try:
            for i in range(2 + args.steps):          # 2 warmup, then timed
                if i == 2:
                    torch.cuda.synchronize(); t0 = time.time()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = model(ids, labels=ids)
                out["loss"].backward()
                opt.step()
                opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            dt = (time.time() - t0) / args.steps
            peak = torch.cuda.max_memory_allocated() / 2**30
            tps = pdbs * args.seq / dt
            base = base or tps
            print(f"{pdbs:>5} {peak:>9.1f} {peak+ddp_bucket:>7.1f} "
                  f"{total_gib-peak-ddp_bucket:>7.1f} {dt:>8.3f} {tps:>11,.0f} "
                  f"{tps/base:>8.2f}x")
        except torch.cuda.OutOfMemoryError:
            print(f"{pdbs:>5} {'OOM':>9}")
            break
        finally:
            del ids
            opt.zero_grad(set_to_none=True)


if __name__ == "__main__":
    main()
