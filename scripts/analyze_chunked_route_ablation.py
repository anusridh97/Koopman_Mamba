"""Read the per-trial `route_ablation.json` files and report the per-route
ablation delta, the paired route-swap penalty, and the between-cell contrast.

Separate from `measure_chunked_route_ablation.py` because the measurement runs on
a GPU node, once, for hours, and the analysis is re-run whenever another seed
lands. Folding the analysis into the measurement would mean re-training to change
a statistic.

## What the columns are, and which one answers which question

`delta`     `ska_ablation.loss_delta` from THIS script's 64-batch cross-route
            pass: held-out loss with the SKA branch zeroed, minus with it live.
            Positive means SKA is load-bearing.
`tr_delta`  the same quantity from `train.py --eval_on_final`, which evaluates 8
            batches. Noisier, and the instrument the 0.025139 reference (job
            445994) used -- so that reference must be compared against THIS
            column, not `delta`. Reported side by side because a reader who
            compares the sharper number to the reference will conclude the
            harness disagrees with a known result when it does not.
`swap_pen`  held-out loss of the SAME weights evaluated under the OTHER route,
            minus under the trained route. Paired on weights and on eval batches,
            so it is far sharper than any between-cell loss difference -- whose
            resolvable floor here is 0.0213 (sigma 7.54e-3, job 445994).

Zeroing SKA makes the route irrelevant, so `ablated_loss` must be identical
between the two routes of a trial; the loader checks that, because if it ever
differs the swap is touching something other than the SKA operator.

Usage:
    python scripts/analyze_chunked_route_ablation.py \
        /scratch/.../chunked-route-*/route_ablation.json
"""
from __future__ import annotations

import argparse
import glob
import json
import math
from collections import defaultdict


def load(paths, expect_steps=1500):
    rows = defaultdict(list)
    used, skipped = [], []
    for p in sorted(paths):
        doc = json.loads(open(p).read())
        # A smoke run (MAX_STEPS=20) writes the same filename. Averaging a
        # deliberately-untrained trial in would be silent and wrong.
        if expect_steps is not None and doc.get("max_steps") != expect_steps:
            skipped.append((p, doc.get("max_steps")))
            continue
        used.append(p)
        for cell, rec in doc.get("cells", {}).items():
            for seed, t in rec.get("trials", {}).items():
                at = t.get("as_trained", {})
                sw = t.get("route_swapped", {})
                tr = t.get("train_side_quick_eval", {})
                if at and sw:
                    assert at["ablated_loss"] == sw["ablated_loss"], (
                        f"{p} {cell} seed{seed}: ablated_loss differs between "
                        f"routes ({at['ablated_loss']} vs {sw['ablated_loss']}); "
                        f"zeroing SKA should make the route irrelevant, so the "
                        f"swap is changing something else")
                    assert not (at["missing_keys"] or at["unexpected_keys"]), (
                        f"{p} {cell} seed{seed}: state_dict did not load cleanly")
                rows[cell].append(dict(
                    seed=int(seed), loss=at.get("loss"),
                    delta=at.get("ska_loss_delta"), swap_loss=sw.get("loss"),
                    tr_delta=tr.get("ska_loss_delta"), err=t.get("error")))
    return rows, used, skipped


def stats(vals):
    vals = [v for v in vals if v is not None]
    n = len(vals)
    if n == 0:
        return None
    mean = sum(vals) / n
    if n < 2:
        return n, mean, float("nan"), float("nan")
    var = sum((v - mean) ** 2 for v in vals) / (n - 1)
    sem = math.sqrt(var / n)
    return n, mean, sem, (mean / sem if sem else float("inf"))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="route_ablation.json files (globs ok)")
    ap.add_argument("--expect_steps", type=int, default=1500,
                    help="reject files whose max_steps differs (0 to disable)")
    ap.add_argument("--baseline_cell", default="exact-invchol")
    args = ap.parse_args(argv)

    files = []
    for p in args.paths:
        files.extend(glob.glob(p) if any(c in p for c in "*?[") else [p])
    rows, used, skipped = load(files, args.expect_steps or None)
    for p, steps in skipped:
        print(f"SKIPPED {p} (max_steps={steps})")
    print(f"{len(used)} trial file(s)\n")

    print("PER-TRIAL")
    print(f"{'cell':15s} {'seed':>4s} {'loss':>9s} {'delta':>10s} "
          f"{'swap_pen':>9s} {'tr_delta':>10s}")
    for cell in sorted(rows):
        for r in sorted(rows[cell], key=lambda r: r["seed"]):
            if r["loss"] is None:
                print(f"{cell:15s} {r['seed']:>4d}   ERROR: {r['err']}")
                continue
            print(f"{cell:15s} {r['seed']:>4d} {r['loss']:9.5f} "
                  f"{r['delta']:+10.6f} {r['swap_loss'] - r['loss']:+9.6f} "
                  f"{(r['tr_delta'] if r['tr_delta'] is not None else float('nan')):+10.6f}")

    print()
    print("PER-CELL  (delta > 0 => SKA is load-bearing; ~0 => it earns nothing)")
    print(f"{'cell':15s} {'n':>2s} {'loss':>9s} {'delta':>10s} {'SEM':>9s} "
          f"{'t':>6s} {'swap_pen':>10s} {'tr_delta':>10s}")
    summary = {}
    for cell in sorted(rows):
        ds = stats([r["delta"] for r in rows[cell]])
        if not ds:
            print(f"{cell:15s}  0  (no completed trials)")
            continue
        ls = stats([r["loss"] for r in rows[cell]])
        ps = stats([r["swap_loss"] - r["loss"] for r in rows[cell]
                    if r["swap_loss"] is not None and r["loss"] is not None])
        ts = stats([r["tr_delta"] for r in rows[cell]])
        summary[cell] = ds
        n, dm, dsem, dt = ds
        print(f"{cell:15s} {n:2d} {ls[1]:9.5f} {dm:+10.6f} {dsem:9.6f} "
              f"{dt:6.2f} {ps[1]:+10.6f} "
              f"{(ts[1] if ts else float('nan')):+10.6f}")

    base = args.baseline_cell
    if base in summary and summary[base][0] >= 2:
        print()
        print(f"BETWEEN-CELL contrast of the ablation delta vs {base} (Welch)")
        n0, m0, s0, _ = summary[base]
        for cell, (n1, m1, s1, _) in sorted(summary.items()):
            if cell == base or n1 < 2:
                continue
            sed = math.sqrt(s0 ** 2 + s1 ** 2)
            t = (m0 - m1) / sed if sed else float("nan")
            print(f"  {cell:15s} keeps {m1/m0*100:5.1f}% of it; "
                  f"difference {m0-m1:+.6f} SE {sed:.6f} t {t:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
