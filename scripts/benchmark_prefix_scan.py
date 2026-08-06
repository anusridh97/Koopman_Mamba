#!/usr/bin/env python3
"""Benchmark the fused exact scan against the legacy chunk-64 SKA branch.

This times the complete SKA module (projections, statistics/core, and output
projection) so the result reflects the actual model-level cost rather than an
isolated microkernel.
"""
from __future__ import annotations

import argparse
import statistics

import torch

from koopman_lm.globals.modules.ska.ska import SKAModule


def make_module(args: argparse.Namespace, *, fused: bool) -> SKAModule:
    return SKAModule(
        d_model=args.d_model,
        n_heads=args.heads,
        rank=24,
        head_dim=64,
        ridge_eps=1e-2,
        power_K=1,
        chunk_size=64,
        backend="cuda_prefix" if fused else "pytorch",
        eta_learnable=False,
        eta_value=1.0,
        gamma_learnable=False,
        gamma_value=1.0,
        layerscale=True,
        layerscale_init=0.01,
        out_proj_std=0.02,
        prefix_scan=fused,
        prefix_scan_block_size=32,
        prefix_scan_jitter=0.0,
        norm_clip_c=4.0,
    ).cuda().train()


def time_module(module: SKAModule, hidden: torch.Tensor, warmup: int, iters: int):
    times = []
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    for i in range(warmup + iters):
        module.zero_grad(set_to_none=True)
        hidden.grad = None
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        out = module(hidden)
        out.square().mean().backward()
        stop.record()
        stop.synchronize()
        if i >= warmup:
            times.append(start.elapsed_time(stop))
    return {
        "median_ms": statistics.median(times),
        "mean_ms": statistics.mean(times),
        "peak_extra_gib": max(0, torch.cuda.max_memory_allocated() - baseline) / (1024 ** 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--length", type=int, default=2048)
    parser.add_argument("--d-model", type=int, default=384)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()
    if args.d_model != args.heads * 64:
        raise SystemExit("this benchmark requires d_model == heads * 64")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    torch.manual_seed(0)
    hidden = torch.randn(
        args.batch, args.length, args.d_model,
        device="cuda", dtype=torch.bfloat16, requires_grad=True,
    )
    fused = make_module(args, fused=True)
    chunk = make_module(args, fused=False)
    chunk.load_state_dict(fused.state_dict(), strict=True)

    fused_result = time_module(fused, hidden, args.warmup, args.iters)
    chunk_result = time_module(chunk, hidden, args.warmup, args.iters)
    tokens = args.batch * args.length
    print({
        "device": torch.cuda.get_device_name(),
        "shape": [args.batch, args.length, args.d_model],
        "fused_exact": fused_result,
        "chunk64": chunk_result,
        "fused_tokens_per_s": tokens / (fused_result["median_ms"] / 1000.0),
        "chunk64_tokens_per_s": tokens / (chunk_result["median_ms"] / 1000.0),
        "exact_over_chunk64": fused_result["median_ms"] / chunk_result["median_ms"],
    })


if __name__ == "__main__":
    main()
