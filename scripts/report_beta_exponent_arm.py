#!/usr/bin/env python
"""The exponent arm's ledger: predeclared statistics plus the SKA ablation delta.

## What this reports and why each column is here

`docs/beta-exponent-preregistration.md` fixed the statistics before any run of
`key_linear_value_sqrt` or `key_sqrt_value_linear` existed. This script computes
exactly those and nothing else:

    grokked / grok_step / censored   the three-state outcome (never a zero for a
                                    censored run)
    lc_area                         normalised area under the accuracy curve --
                                    the predeclared sample-efficiency statistic
    final_acc / best_acc            reported separately, so a run that solved the
                                    task and then degraded is visible
    ska_acc_on / ska_acc_zeroed     retrieval with the SKA branch live, and with
    ska_delta_acc                   every SKA layer a pure residual passthrough

**`ska_delta_acc` is the primary discriminator**, not a footnote. Held-out LM loss
is proven non-responsive to SKA damage in this repo: 18 trials across three SKA
routes (jobs 446363-446420) cut the ablation delta from 0.024418 to 0.013693 --
destroying 44% of SKA's contribution -- and moved mean held-out loss from 4.39030
to 4.39014, i.e. DOWN, and two orders of magnitude below the 7.54e-3 floor. A
change can gut the mechanism and be invisible in aggregate loss. The ablation
delta is the instrument that is not blind to it: it separates conditions at
t = 7-22 in the same runs where loss does not move.

So a policy that groks with `ska_delta_acc` ~ 0 has learned MQAR in the Mamba
branch and its write gate is decoration. A policy that groks with a large
`ska_delta_acc` is the functional anchor. Those are different findings with the
same accuracy column, which is why the accuracy column alone cannot decide this.

## The provenance assertions

Each row also records what the CHECKPOINT says, not what the command line was
supposed to set: `policy`, `route`, `ridge`, `gamma`, `power_K`. A cell whose
resolved config took the chunked route is reported as `route=CHUNKED` and its
numbers are not to be used -- on a chunked route `beta_proj.bias` gradient cosine
is ~0.00, so a beta comparison there measures a gate receiving no usable
gradient.

Usage:
    python scripts/report_beta_exponent_arm.py <run_root> \\
        --kv 8 --gap 128 [--json_out out.json] [--no_ablation]

`<run_root>` is a directory of `mqar-<cell>-seed<N>/` run dirs alongside
`mqar-<cell>-seed<N>.log` files, as `scripts/run_beta_exponent_mqar.sbatch`
writes them.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from experimentation.evaluation.mqar_curves import (  # noqa: E402
    GROK_THRESHOLD, parse_curve, run_statistics, summarise_policy)

#: `mqar-<cell>-seed<N>`. `cell` rather than `policy`: the ridge controls are
#: `learned-ridge2x` and `linear-ridge0.5x`, which are the same policy at a
#: different ridge and must not collapse into the policy's own row.
_NAME = re.compile(r"^mqar-(?P<cell>.+)-seed(?P<seed>\d+)$")


def _latest_ckpt(run_dir: Path):
    steps = sorted(run_dir.glob("step_*"),
                   key=lambda p: int(p.name.split("_")[1]))
    return steps[-1] if steps else None


def _route_of(cfg) -> str:
    """The resolved backend, as a word. `CHUNKED` is the one that invalidates a
    beta measurement, so it is spelled loudly rather than as a False triple."""
    if getattr(cfg, "ska_prefix_scan", False):
        return "prefix_scan"
    if getattr(cfg, "ska_inverse_cholesky", False):
        return "inverse_cholesky"
    if getattr(cfg, "ska_exact_intrachunk", False):
        return "exact_intrachunk"
    return "CHUNKED"


def _measure_ablation(ckpt: Path, kv: int, gap: int, task_vocab: int,
                      batch: int, seed: int):
    """(acc_on, acc_zeroed, provenance) for one checkpoint.

    Both accuracies are measured on the SAME batch, from the same seed: a delta
    between two different token sets is not an ablation. That is the same
    discipline `quick_eval::run_quick_eval` applies by rebuilding its loader per
    pass.
    """
    import torch
    from experimentation.evaluation.ska_ablation import ablate_ska, ska_blocks
    from experimentation.experiments.curricula import eval_mqar
    from experimentation.experiments.mqar_finetune import (
        build_model, derived_seq_len)

    meta = torch.load(ckpt / "meta.pt", map_location="cpu", weights_only=False)
    cfg = meta["cfg"]
    model = build_model(meta.get("model_type", "mamba_ska_swiglu"), cfg)
    model.load_state_dict(
        torch.load(ckpt / "model.pt", map_location="cpu", weights_only=True))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    seq_len = derived_seq_len(kv, gap)
    kwargs = dict(batch=batch, seq_len=seq_len, num_kv_pairs=kv,
                  vocab_size=task_vocab, device=device, seed=seed)
    with torch.no_grad():
        acc_on = eval_mqar(model, **kwargs)
        n_blocks = len(ska_blocks(model))
        with ablate_ska(model):
            acc_zeroed = eval_mqar(model, **kwargs)

    prov = {
        "policy": cfg.ska_beta_policy,
        "route": _route_of(cfg),
        "ridge": float(cfg.ska_ridge),
        "gamma": float(cfg.ska_gamma_value),
        "power_K": int(cfg.ska_power_K),
        "layerscale_init": float(cfg.ska_layerscale_init),
        "rank": int(cfg.ska_rank),
        "n_ska_blocks": n_blocks,
        "step": int(ckpt.name.split("_")[1]),
    }
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return acc_on, acc_zeroed, prov


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("run_root", type=Path)
    p.add_argument("--kv", type=int, required=True)
    p.add_argument("--gap", type=int, required=True)
    p.add_argument("--task_vocab", type=int, default=128)
    p.add_argument("--eval_batch", type=int, default=64)
    p.add_argument("--threshold", type=float, default=GROK_THRESHOLD)
    p.add_argument("--json_out", type=Path, default=None)
    p.add_argument("--no_ablation", action="store_true",
                   help="skip the SKA-on/zeroed pass (needs torch + a GPU). The "
                        "curve statistics still print, so a login-node run of "
                        "this script is useful without pretending to have "
                        "measured the primary discriminator.")
    args = p.parse_args(argv)

    rows = []
    for log in sorted(args.run_root.glob("mqar-*.log")):
        m = _NAME.match(log.stem)
        if not m:
            print(f"SKIP {log.name}: does not match mqar-<cell>-seed<N>",
                  file=sys.stderr)
            continue
        curve = parse_curve(log.read_text(errors="replace"))
        row = {"cell": m.group("cell"), "seed": int(m.group("seed")),
               "run": log.stem, "curve": curve}
        row.update(run_statistics(curve, threshold=args.threshold))
        rows.append(row)

    if not rows:
        print(f"FATAL: no mqar-*.log under {args.run_root}", file=sys.stderr)
        return 2

    # --- the primary discriminator -------------------------------------------
    if not args.no_ablation:
        for row in rows:
            run_dir = args.run_root / row["run"]
            ckpt = _latest_ckpt(run_dir)
            if ckpt is None:
                row["ska_status"] = "no_checkpoint"
                continue
            try:
                on, zeroed, prov = _measure_ablation(
                    ckpt, args.kv, args.gap, args.task_vocab,
                    args.eval_batch, seed=999)
            except Exception as exc:                       # noqa: BLE001
                # An infrastructure failure here must not be recorded as a
                # scientific one, so it is named rather than folded into a 0.0.
                row["ska_status"] = f"error: {type(exc).__name__}: {exc}"
                continue
            row["ska_status"] = "ok"
            row["ska_acc_on"] = on
            row["ska_acc_zeroed"] = zeroed
            row["ska_delta_acc"] = on - zeroed
            row.update(prov)

    # --- per-run table --------------------------------------------------------
    print(f"# exponent arm: kv={args.kv} gap={args.gap} "
          f"threshold={args.threshold}")
    print(f"# lc_area = normalised area under the accuracy curve "
          f"(sample efficiency, NOT an accuracy)")
    print(f"# ska_delta = MQAR accuracy with SKA on MINUS with SKA zeroed. "
          f"PRIMARY DISCRIMINATOR.")
    print()
    print(f"# lc_area is normalised by each run's OWN observed span, so it is a "
          f"RATE.")
    print(f"# It is comparable only between runs with the same `last`. `n_ev` and "
          f"`last` are")
    print(f"# printed for that reason -- a truncated run's rate otherwise sits in "
          f"the same")
    print(f"# column as a complete run's with nothing marking the difference.")
    print()
    hdr = (f"{'cell':>24} {'seed':>4} {'grok':>5} {'step':>6} {'lc_area':>8} "
           f"{'n_ev':>5} {'last':>7} {'final':>7} {'best':>7} {'ska_on':>7} "
           f"{'ska_0':>7} {'ska_delta':>10} {'route':>17} {'ridge':>7}")
    print(hdr)
    print("-" * len(hdr))

    def fmt(v, w, prec=4):
        if v is None:
            return " " * (w - 1) + "-"
        if isinstance(v, bool):
            return f"{('yes' if v else 'CENS'):>{w}}"
        if isinstance(v, float):
            return f"{v:>{w}.{prec}f}"
        return f"{v:>{w}}"

    for r in sorted(rows, key=lambda r: (r["cell"], r["seed"])):
        print(f"{r['cell']:>24} {r['seed']:>4} {fmt(r['grokked'], 5)} "
              f"{fmt(r['grok_step'], 6)} {fmt(r['lc_area'], 8)} "
              f"{fmt(r['n_evals'], 5)} {fmt(r['last_step'], 7)} "
              f"{fmt(r['final_acc'], 7)} {fmt(r['best_acc'], 7)} "
              f"{fmt(r.get('ska_acc_on'), 7)} {fmt(r.get('ska_acc_zeroed'), 7)} "
              f"{fmt(r.get('ska_delta_acc'), 10)} "
              f"{str(r.get('route', '-')):>17} {fmt(r.get('ridge'), 7, 5)}")

    # --- per-cell summary: counts and ranges, never a mean accuracy -----------
    by_cell = defaultdict(list)
    for r in rows:
        by_cell[r["cell"]].append(r)

    print()
    print("# per-cell summary. NO mean accuracy is reported: with the budget "
          "straddling")
    print("# grokking, a mean over seeds is a grok RATE dressed as an accuracy.")
    print()
    print(f"{'cell':>24} {'n':>3} {'grok':>6} {'cens':>5} {'nodata':>6} "
          f"{'grok_steps':>20} {'lc_area range':>22} {'ska_delta range':>22} "
          f"{'spans':>18}")
    incomparable = []
    for cell in sorted(by_cell):
        runs = by_cell[cell]
        s = summarise_policy(runs)
        # NaN-safe: eval_mqar returns nan on an empty label mask, and a nan in
        # min()/max() would silently poison the printed range.
        deltas = [r["ska_delta_acc"] for r in runs
                  if r.get("ska_delta_acc") is not None
                  and r["ska_delta_acc"] == r["ska_delta_acc"]]
        lc = s["lc_area_range"]
        if not s["spans_comparable"]:
            incomparable.append(cell)
        print(f"{cell:>24} {s['n']:>3} {s['n_grokked']:>2}/{s['n']:<3} "
              f"{s['n_censored']:>5} {s['n_no_data']:>6} "
              f"{str(s['grok_steps']):>20} "
              f"{(f'{lc[0]:.4f} .. {lc[1]:.4f}' if lc else '-'):>22} "
              f"{(f'{min(deltas):+.4f} .. {max(deltas):+.4f}' if deltas else '-'):>22} "
              f"{str(s['last_steps'])[:18]:>18}")

    # --- gates ---------------------------------------------------------------
    # These set `rc`, they do not merely print. A caller wrapping this in
    # `|| echo WARN` cannot see a message.
    print()
    rc = 0

    if incomparable:
        print(f"WARNING: lc_area spans differ WITHIN these cells, so their "
              f"lc_area values are not mutually comparable: {incomparable}")
        print(f"         (expected if the allocation was cut short; extend or "
              f"drop the short runs before ranking)")

    bad_route = sorted({r["cell"] for r in rows if r.get("route") == "CHUNKED"})
    if bad_route:
        print(f"REFUSED: cells on the CHUNKED route -- on a chunked route "
              f"beta_proj.bias gradient cosine is ~0.00, so their beta numbers "
              f"are not evidence: {bad_route}")
        rc = 3

    # Keyed by RUN, not by cell. Keyed by cell, two seeds that resolved
    # DIFFERENT policies -- precisely the failure this block exists to catch --
    # would silently collapse to whichever row was last.
    policies = {r["run"]: r.get("policy") for r in rows if r.get("policy")}
    per_cell = defaultdict(set)
    for r in rows:
        if r.get("policy"):
            per_cell[r["cell"]].add(r["policy"])
    mixed = {c: sorted(v) for c, v in per_cell.items() if len(v) > 1}
    if mixed:
        print(f"REFUSED: these cells resolved MORE THAN ONE beta policy across "
              f"their seeds, so the cell is not one experiment: {mixed}")
        rc = 3
    if policies:
        print(f"resolved policies from checkpoints: "
              f"{json.dumps(policies, sort_keys=True)}")

    # A model with no SKA blocks gives ska_delta_acc == 0.0 with status "ok" --
    # the primary discriminator would read "the gate is decoration" for a model
    # that has no gate. `ablate_ska` yields the blocks it touched so this is
    # checkable; check it.
    no_ska = [r["run"] for r in rows if r.get("n_ska_blocks") == 0]
    if no_ska:
        print(f"REFUSED: these runs have ZERO SKA blocks, so their "
              f"ska_delta_acc is 0.0 by construction and is not a measurement: "
              f"{no_ska}")
        rc = 3

    nan_delta = [r["run"] for r in rows
                 if r.get("ska_delta_acc") is not None
                 and r["ska_delta_acc"] != r["ska_delta_acc"]]
    if nan_delta:
        print(f"NOTE: ska_delta_acc is NaN for {nan_delta} -- eval_mqar returns "
              f"NaN on an empty label mask. Excluded from the printed ranges.")

    no_ckpt = [r["run"] for r in rows if r.get("ska_status") == "no_checkpoint"]
    if no_ckpt:
        print(f"NOTE: no checkpoint (ablation not measured) for {no_ckpt}")
    errs = [(r["run"], r["ska_status"]) for r in rows
            if str(r.get("ska_status", "")).startswith("error")]
    for run, msg in errs:
        print(f"NOTE: ablation failed for {run}: {msg}  "
              f"(infrastructure, NOT a scientific result)")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(
            {"kv": args.kv, "gap": args.gap, "threshold": args.threshold,
             "run_root": str(args.run_root), "runs": rows,
             "cells": {c: summarise_policy(by_cell[c]) for c in by_cell}},
            indent=2, default=str))
        print(f"\nwrote {args.json_out}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
