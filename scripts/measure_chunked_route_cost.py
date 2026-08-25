"""What the exact routes COST at the geometry of each config that is on the
chunked one -- and whether they fit at all.

`space.py`'s route table says `inverse_cholesky` is "0.92x at 4m, 1.41x at 50m"
against chunked, and that number is used to argue "exactness is free". Both
geometries are rank 24. Every config still on the chunked route is rank 24 (`1m`),
48 (the three 180m variants), 64 (`370m`), 96 (`440m`, `880m`) or 128 (`1p5b`,
`3b`), and the exact path's per-token statistics are (B,T,H,r,r) -- QUADRATIC in
rank and LINEAR in sequence length, where the chunked path's are
(B,T/S,H,r,r). So "0.92x" cannot be assumed to travel, in either time or memory.

This measures, per config geometry, on one real SKA layer at the config's own
rank / heads / value width / chunk size / sequence length:

  * chunked forward+backward wall time and peak memory;
  * inverse_cholesky ditto, or the reason it cannot run;
  * exact_intrachunk ditto;
  * the ratio, so the claim "exactness is free" gets a per-config answer.

It measures ONE LAYER, not a full model, and says so in the output: a full-model
ratio also carries the embedding/MLP time that is identical between routes and so
flatters the exact route. The one-layer ratio is the pessimistic, mechanism-level
number; `space.py`'s is the optimistic full-model one. Both are worth having and
confusing them is how "0.92x" came to sound like a free lunch.

Usage:  python scripts/measure_chunked_route_cost.py [--json OUT] [--batch 1]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from koopman_lm.config import CONFIG_REGISTRY, build_config  # noqa: E402
from koopman_lm.modules.seq.ska import SKAModule  # noqa: E402

ROUTES = ("chunked", "inverse_cholesky", "exact_intrachunk")


def route_kwargs(route):
    return {} if route == "chunked" else {route: True}


def describe(name):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c = build_config(name)
    return {
        "d_model": c.d_model, "n_heads": c.ska_n_heads, "rank": c.ska_rank,
        "chunk_size": c.ska_chunk_size, "seq_len": c.max_seq_len,
        "power_K": c.ska_power_K, "precision": c.ska_precision,
        "value_width": c.d_model // c.ska_n_heads,
        "n_ska_layers": (len(c.ska_layer_indices) if c.ska_layer_indices
                         else c.n_layers),
        "current_route": ("prefix_scan" if c.ska_prefix_scan else
                          "inverse_cholesky" if c.ska_inverse_cholesky else
                          "exact_intrachunk" if c.ska_exact_intrachunk else
                          "chunked"),
    }


def time_one(geom, route, device, batch, seq_len, iters=3, warmup=1):
    """(seconds per fwd+bwd, peak GiB) for one SKA layer, or an error string."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mod = SKAModule(
                d_model=geom["d_model"], n_heads=geom["n_heads"],
                rank=geom["rank"], chunk_size=geom["chunk_size"],
                power_K=geom["power_K"], precision=geom["precision"],
                **route_kwargs(route)).to(device)
    except AssertionError as e:
        return None, None, f"REFUSED at construction: {e}"

    x = torch.randn(batch, seq_len, geom["d_model"], device=device,
                    requires_grad=True)
    try:
        for i in range(warmup + iters):
            if i == warmup:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                t0 = time.time()
            out = mod(x)
            out.sum().backward()
            mod.zero_grad(set_to_none=True)
            x.grad = None
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = (time.time() - t0) / iters
        peak = (torch.cuda.max_memory_allocated() / 1024 ** 3
                if device.type == "cuda" else float("nan"))
        return dt, peak, None
    except torch.cuda.OutOfMemoryError as e:  # pragma: no cover - GPU only
        torch.cuda.empty_cache()
        return None, None, f"OOM: {str(e).splitlines()[0]}"
    except RuntimeError as e:  # pragma: no cover - GPU only
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return None, None, f"RuntimeError: {str(e).splitlines()[0]}"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--seq_len", type=int, default=None,
                    help="override every config's max_seq_len (for a cheap probe)")
    ap.add_argument("--configs", default=None,
                    help="comma-separated subset of CONFIG_REGISTRY")
    args = ap.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    names = (args.configs.split(",") if args.configs else list(CONFIG_REGISTRY))
    out = {"device": (torch.cuda.get_device_name(0) if device.type == "cuda"
                      else "cpu"),
           "batch": args.batch, "seq_len_override": args.seq_len,
           "note": "ONE SKA layer, fwd+bwd. Not a full-model ratio.",
           "configs": {}}

    for name in names:
        geom = describe(name)
        seq = args.seq_len or geom["seq_len"]
        rec = {"geometry": geom, "seq_len_used": seq, "routes": {}}
        print(f"\n=== {name}  (rank {geom['rank']}, heads {geom['n_heads']}, "
              f"vwidth {geom['value_width']}, CS {geom['chunk_size']}, "
              f"seq {seq}, currently {geom['current_route']}) ===", flush=True)
        for route in ROUTES:
            dt, peak, err = time_one(geom, route, device, args.batch, seq)
            rec["routes"][route] = ({"error": err} if err else
                                    {"sec_per_step": dt, "peak_gib": peak})
            if err:
                print(f"  {route:18s} {err}", flush=True)
            else:
                print(f"  {route:18s} {dt*1e3:9.2f} ms   peak {peak:7.2f} GiB",
                      flush=True)
        base = rec["routes"]["chunked"].get("sec_per_step")
        for route in ROUTES[1:]:
            got = rec["routes"][route].get("sec_per_step")
            if base and got:
                rec["routes"][route]["ratio_vs_chunked"] = got / base
                print(f"  -> {route} is {got/base:.2f}x chunked", flush=True)
        out["configs"][name] = rec

    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2) + "\n")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
