"""Does the chunked route's SKA earn anything? The paired measurement.

## Why this and not a loss comparison between two cells

`REVIEW.md` records the between-cell attempt: job 439883 (chunked) against 440183
(exact) moved every objective by at most 2.3e-4 against a 0.458 spread across
anchors, i.e. unresolvable, and REVIEW.md:291-296 lists the proper version as
"a now-possible measurement that has NOT been run".

It is worth saying why a between-cell loss comparison is nearly guaranteed to be
unresolvable here, because that is a fact about the instrument and not about the
route. On `proxy-256x17` at 1500 steps the SKA ablation delta is 0.0251 -- that is
the WHOLE contribution of SKA to held-out loss. So no route change can move
held-out loss by more than about 0.0251, whatever it does to the mechanism. The
trial-vs-trial resolvable effect at the same horizon is 0.0213 (sigma 7.54e-3, job
445994). A between-cell chunked-vs-exact contrast therefore has a ceiling only
1.18x its own noise floor, and would need dozens of seeds per cell to separate
"the route costs everything SKA earns" from "the route costs nothing".

The PAIRED quantity does not have that problem. `ska_ablation.loss_delta` is
measured within one run -- the same weights, the same held-out batches, SKA on
versus SKA zeroed -- and at 1500 steps it is 0.025139 +- 0.002976 SEM over five
seeds, t = 8.4. So the question "does SKA contribute on this route?" IS resolvable
at n = 3, while "which route reaches a lower loss?" is not resolvable at n = 20.

That reframes the whole question. If the chunked route's ablation delta is
statistically zero while the exact route's is 0.025 at t = 8.4, the finding is not
"the chunked route is slightly worse". It is "on the chunked route SKA is not
contributing at all, so a chunked run is a Mamba baseline wearing SKA's parameter
count" -- and that is a statement about what the config MEASURES, which no loss
comparison could have delivered.

## What this runs

Cells at the `proxy-256x17` geometry (`ska_inverse_cholesky: true`, rank 24,
value width 64, seq 1024, 1500 steps, the horizon sigma was measured at):

  exact-invchol      x3 seeds   the control, and a re-measurement of the 0.0251
  chunked-cs64       x3 seeds   the proxy's own ska_chunk_size
  chunked-cs16       x3 seeds   `1m`'s ska_chunk_size -- the Echo Table 2 config

Then, on EVERY finished checkpoint, all four of {route as trained, route swapped}
x {SKA on, SKA zeroed}. The route flag changes no parameter shape, so the swap is
a pure evaluation of the same weights under the other operator: another paired
contrast, and the cheapest one in the file.

Usage:
    python scripts/measure_chunked_route_ablation.py \
        --run_root $SCRATCH/chunked-route-ablation --json out.json
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import os
import subprocess
import sys
import time
import warnings
from pathlib import Path

import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

PROXY = REPO / "configs/runs/proxy-256x17.yaml"

#: (cell name, ska_inverse_cholesky, ska_chunk_size or None to keep the base's)
CELLS = (
    ("exact-invchol", True, None),
    ("chunked-cs64", False, 64),
    ("chunked-cs16", False, 16),
)
SEEDS = (42, 43, 44)


# --------------------------------------------------------------- spec build ----

def variant_spec(route_exact: bool, chunk_size, seed: int, max_steps: int,
                 eval_dir: str, out_path: Path, run_root: str) -> Path:
    """Write a proxy-256x17 variant differing ONLY in route / chunk / seed / steps."""
    raw = yaml.safe_load(PROXY.read_text())
    raw["model"]["ska_inverse_cholesky"] = bool(route_exact)
    if chunk_size is not None:
        raw["model"]["ska_chunk_size"] = int(chunk_size)
    raw["runtime"]["seed"] = int(seed)
    raw["optim"]["max_steps"] = int(max_steps)
    # warmup_ratio 0.02 of the new horizon, matching how the study driver scales it
    raw["optim"]["warmup_steps"] = max(1, int(round(0.04 * max_steps)))
    raw["name"] = out_path.stem
    out_path.write_text(yaml.safe_dump(raw, sort_keys=False))
    return out_path


# ------------------------------------------------------------------- train ----

def train_cell(spec_path: Path, run_dir: Path, eval_dir: str,
               logging_steps: int) -> Path:
    """Train one cell as a subprocess. Returns the final checkpoint path."""
    from experimentation.run.resolve import resolve_run_spec
    from experimentation.run.train_argv import build_train_argv

    spec = resolve_run_spec(spec_path)
    run_dir.mkdir(parents=True, exist_ok=True)
    argv = build_train_argv(spec, run_dir, eval_on_final=True,
                            eval_data_dir=eval_dir, logging_steps=logging_steps)
    cmd = [sys.executable, "-m", "experimentation.training.train"] + argv
    print(f"    $ {' '.join(cmd[:6])} ... ({len(cmd)} args)", flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(REPO))
    print(f"    train exited {proc.returncode} in {time.time()-t0:.0f}s", flush=True)
    if proc.returncode != 0:
        raise RuntimeError(f"training failed for {spec_path.name}")
    return run_dir / "final" / "model.pt"


# ------------------------------------------------- cross-route evaluation ----

def eval_both_routes(ckpt: Path, eval_dir: str, batch_size: int,
                     max_batches: int) -> dict:
    """{route: {full/ablated loss}} for the trained route AND the other one.

    The four SKA routes share every parameter -- they differ only in how the
    sufficient statistics are formed -- so the same state_dict loads under either
    flag and the difference is purely the operator.
    """
    from experimentation.evaluation.quick_eval import run_quick_eval
    from koopman_lm.models.koopman_lm import KoopmanLM

    meta = torch.load(str(ckpt).replace("model.pt", "meta.pt"),
                      map_location="cpu", weights_only=False)
    cfg = meta["cfg"]
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    state = state.get("model", state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out = {}
    for label, use_exact in (("as_trained", cfg.ska_inverse_cholesky),
                             ("route_swapped", not cfg.ska_inverse_cholesky)):
        vcfg = dataclasses.replace(cfg, ska_inverse_cholesky=bool(use_exact))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = KoopmanLM(vcfg)
        missing, unexpected = model.load_state_dict(state, strict=False)
        model = model.to(device).eval()
        m = run_quick_eval(model, device, data_dir=eval_dir,
                           max_seq_len=min(vcfg.max_seq_len, 1024),
                           batch_size=batch_size, max_batches=max_batches,
                           ska_ablation=True)
        out[label] = {
            "route": "inverse_cholesky" if use_exact else "chunked",
            "loss": m["full"]["loss"],
            "ablated_loss": m["ska_ablation"].get("loss"),
            "ska_loss_delta": m["ska_ablation"].get("loss_delta"),
            "missing_keys": len(missing), "unexpected_keys": len(unexpected),
        }
        print(f"      {label:14s} route={out[label]['route']:17s} "
              f"loss={out[label]['loss']:.5f} "
              f"ska_delta={out[label]['ska_loss_delta']:+.6f}", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return out


# -------------------------------------------------------------------- main ----

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_root", required=True)
    ap.add_argument("--eval_data_dir",
                    default="/scratch/m000151-pm06/jkli/fineweb_small_val")
    ap.add_argument("--json", default=None)
    ap.add_argument("--max_steps", type=int, default=1500)
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    ap.add_argument("--cells", default=",".join(c[0] for c in CELLS))
    ap.add_argument("--logging_steps", type=int, default=100)
    ap.add_argument("--eval_batches", type=int, default=64)
    ap.add_argument("--eval_batch_size", type=int, default=4)
    args = ap.parse_args(argv)

    root = Path(args.run_root)
    root.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",")]
    want = set(args.cells.split(","))
    cells = [c for c in CELLS if c[0] in want]

    results = {"max_steps": args.max_steps, "seeds": seeds,
               "eval_data_dir": args.eval_data_dir, "cells": {}}
    for name, exact, cs in cells:
        results["cells"][name] = {"ska_inverse_cholesky": exact,
                                  "ska_chunk_size": cs, "trials": {}}
        for seed in seeds:
            tag = f"{name}-seed{seed}"
            print(f"\n=== {tag} ===", flush=True)
            spec_p = variant_spec(exact, cs, seed, args.max_steps,
                                  args.eval_data_dir, root / f"{tag}.yaml",
                                  str(root))
            run_dir = root / tag
            try:
                ckpt = train_cell(spec_p, run_dir, args.eval_data_dir,
                                  args.logging_steps)
                ev = eval_both_routes(ckpt, args.eval_data_dir,
                                      args.eval_batch_size, args.eval_batches)
                results["cells"][name]["trials"][str(seed)] = ev
            except Exception as exc:                        # noqa: BLE001
                print(f"    FAILED: {type(exc).__name__}: {exc}", flush=True)
                results["cells"][name]["trials"][str(seed)] = {
                    "error": f"{type(exc).__name__}: {exc}"}
            if args.json:
                Path(args.json).write_text(json.dumps(results, indent=2) + "\n")

    # ---- summary ----
    print("\n" + "=" * 78)
    print(f"{'cell':16s} {'n':>2s} {'loss mean':>11s} {'ska_delta mean':>15s} "
          f"{'SEM':>10s} {'t':>7s}")
    for name, rec in results["cells"].items():
        ds, ls = [], []
        for t in rec["trials"].values():
            if "error" in t:
                continue
            ds.append(t["as_trained"]["ska_loss_delta"])
            ls.append(t["as_trained"]["loss"])
        if not ds:
            print(f"{name:16s}  0   (all trials failed)")
            continue
        n = len(ds)
        mean = sum(ds) / n
        if n > 1:
            var = sum((d - mean) ** 2 for d in ds) / (n - 1)
            sem = (var / n) ** 0.5
            tstat = mean / sem if sem else float("inf")
        else:
            sem, tstat = float("nan"), float("nan")
        print(f"{name:16s} {n:2d} {sum(ls)/n:11.5f} {mean:15.6f} "
              f"{sem:10.6f} {tstat:7.2f}")
    print("=" * 78)
    print("ska_delta = held-out loss with SKA zeroed MINUS with SKA on.")
    print("Positive and large => SKA is load-bearing. ~0 => it earns nothing.")

    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2) + "\n")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
