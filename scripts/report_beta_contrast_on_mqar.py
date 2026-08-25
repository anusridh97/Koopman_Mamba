#!/usr/bin/env python
"""Did the write gate learn to write facts and suppress distractors -- and by
how much, in the units the operator actually sees?

## The measurement, and why it is the decisive one

MQAR sequences have a known, exact partition (`curricula.make_mqar`):

    [k1 v1 ... kP vP]        positions [0, 2P)          THE FACTS
    <random filler>          positions [2P, 2P+F)       THE DISTRACTORS
    [kq1 vq1 ...]            the tail                   THE QUERIES

A write gate that is doing its job puts a high weight on the fact block and a
low weight on the filler. So the quantity of interest is not beta's variance --
it is the CONTRAST between the two blocks, in KEY WEIGHT rather than in beta:

    key weight = sqrt(beta)   under `learned` / `head_scalar` / `one`
    key weight = beta         under `linear`

That distinction is the whole point. `beta_policy` changes the map from beta to
key weight, so two policies can reach the same beta contrast and different
OPERATOR contrast -- and the operator contrast is what suppresses a distractor
in G, M and C. Reporting only beta would compare the policies in units none of
them uses.

## What it can settle that an accuracy number cannot

The retrieval arm (job 446102, kv=8 gap=128, proxy geometry, seed 42) separates
sharply: `linear` and `one` reach 1.0000 by step 3999 while `learned` and
`head_scalar` sit at 0.36 and 0.33. Accuracy alone cannot say WHY, and there are
two very different explanations with the same accuracy signature:

  * the gate TRIED AND COULD NOT. beta collapses toward 0 on filler and 1 on
    facts, but sqrt() compresses the achieved key-weight contrast to the square
    root of what `linear` gets at the same logits (see
    code-tests/test_beta_policy_suppression_range.py), and that is not enough.
  * the gate NEVER TRIED. beta stays near its 0.5 initialisation, so the policy
    difference is irrelevant and the failure is an optimisation problem.

These call for opposite responses -- reparameterise, versus fix the training --
so distinguishing them is worth a script. The beta contrast column separates
them directly: near 1.0 means "never tried", far from 1.0 means "tried".

`one` is the control that makes the reading honest: it has NO gate, so its
contrast is exactly 1 by construction, and it still solves the task. Whatever
the gate is contributing, `one` demonstrates the task does not require it.

Usage:
    python scripts/report_beta_contrast_on_mqar.py \\
        --kv 8 --gap 128 <mqar_output_dir> [<mqar_output_dir> ...]

Each directory is an `mqar-<policy>-seed<N>` dir containing `step_<N>/model.pt`
and `step_<N>/meta.pt`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


def _latest_ckpt(run_dir: Path):
    steps = sorted(run_dir.glob("step_*"), key=lambda p: int(p.name.split("_")[1]))
    return steps[-1] if steps else None


def _beta_per_layer(model, input_ids):
    """(name, beta) for every SKA layer, captured with forward hooks.

    Hooks rather than a reimplementation of the projection stack: the gate reads
    the hidden state at its own depth, and recomputing that path is how a
    diagnostic ends up describing something the model does not do.
    """
    from koopman_lm.modules.seq.ska import SKAModule

    grabbed = []
    handles = []

    def make_hook(mod):
        def hook(_m, args, _out):
            grabbed.append((mod, mod._resolve_beta(args[0]).detach().float()))
        return hook

    skas = [m for _n, m in model.named_modules() if isinstance(m, SKAModule)]
    for mod in skas:
        handles.append(mod.register_forward_hook(make_hook(mod)))
    try:
        with torch.no_grad():
            model(input_ids=input_ids)
    finally:
        for h in handles:
            h.remove()
    return grabbed


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--kv", type=int, required=True)
    parser.add_argument("--gap", type=int, required=True)
    parser.add_argument("--task_vocab", type=int, default=128)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--json_out", type=Path, default=None)
    args = parser.parse_args(argv)

    from experimentation.experiments.curricula import make_mqar
    from experimentation.experiments.mqar_finetune import build_model, derived_seq_len

    seq_len = derived_seq_len(args.kv, args.gap)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ids, _labels = make_mqar(args.batch, seq_len, args.kv, args.task_vocab,
                             seed=12345)
    ids = ids.to(device)

    # The exact partition make_mqar builds. Derived, not guessed: the fact block
    # is 2*kv tokens, the query block is 2*num_queries (= 2*kv by default), and
    # the filler is everything between.
    n_fact = 2 * args.kv
    n_query = 2 * args.kv
    fact = slice(0, n_fact)
    filler = slice(n_fact, seq_len - n_query)
    query = slice(seq_len - n_query, seq_len)

    rows = []
    for run_dir in args.run_dirs:
        ckpt = _latest_ckpt(run_dir)
        if ckpt is None:
            print(f"SKIP {run_dir}: no step_* checkpoint", file=sys.stderr)
            continue
        meta = torch.load(ckpt / "meta.pt", map_location="cpu", weights_only=False)
        cfg = meta["cfg"]
        model = build_model(meta.get("model_type", "mamba_ska_swiglu"), cfg)
        model.load_state_dict(
            torch.load(ckpt / "model.pt", map_location="cpu", weights_only=True))
        model = model.to(device).eval()

        for layer, (mod, beta) in enumerate(_beta_per_layer(model, ids)):
            b_fact = float(beta[:, fact].mean())
            b_fill = float(beta[:, filler].mean())
            b_query = float(beta[:, query].mean())
            # In KEY WEIGHT, the units the operator sees. Read off the shipped
            # helper so it cannot drift from the policy's real weighting.
            unit = torch.ones(1, 1, mod.H, mod.rank)
            zero = torch.zeros(1, 1, mod.H, mod.P)

            def weight(b):
                x, _ = mod._weight_key_value(
                    unit, torch.full((1, 1, mod.H), b), zero)
                return float(x[0, 0, 0, 0])

            rows.append({
                "run": run_dir.name, "policy": cfg.ska_beta_policy,
                "step": int(ckpt.name.split("_")[1]), "layer": layer,
                "beta_fact": b_fact, "beta_filler": b_fill,
                "beta_query": b_query,
                "beta_contrast": b_fact / b_fill if b_fill else float("inf"),
                "weight_fact": weight(b_fact), "weight_filler": weight(b_fill),
                "weight_contrast": (weight(b_fact) / weight(b_fill)
                                    if weight(b_fill) else float("inf")),
            })
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    if not rows:
        print("FATAL: nothing measured", file=sys.stderr)
        return 2

    print(f"# gate contrast on MQAR  (kv={args.kv} gap={args.gap} "
          f"seq_len={seq_len})")
    print(f"# fact positions [0,{n_fact})  filler [{n_fact},{seq_len-n_query})"
          f"  query [{seq_len-n_query},{seq_len})")
    print()
    print(f"{'run':>26} {'L':>2} {'b_fact':>7} {'b_fill':>7} {'b_ratio':>8} "
          f"{'w_fact':>7} {'w_fill':>7} {'w_ratio':>8}")
    for r in rows:
        print(f"{r['run']:>26} {r['layer']:>2} {r['beta_fact']:>7.4f} "
              f"{r['beta_filler']:>7.4f} {r['beta_contrast']:>8.3f} "
              f"{r['weight_fact']:>7.4f} {r['weight_filler']:>7.4f} "
              f"{r['weight_contrast']:>8.3f}")

    print()
    by_run: dict[str, list[dict]] = {}
    for r in rows:
        by_run.setdefault(r["run"], []).append(r)
    print(f"{'run':>26} {'policy':>12} {'mean b_ratio':>13} "
          f"{'mean w_ratio':>13}  reading")
    for run, group in sorted(by_run.items()):
        br = sum(r["beta_contrast"] for r in group) / len(group)
        wr = sum(r["weight_contrast"] for r in group) / len(group)
        policy = group[0]["policy"]
        if policy == "one":
            reading = "no gate -- contrast is 1 by construction"
        elif abs(br - 1.0) < 0.05:
            reading = "NEVER TRIED (beta stayed near its init)"
        else:
            reading = "TRIED (beta separated facts from filler)"
        print(f"{run:>26} {policy:>12} {br:>13.3f} {wr:>13.3f}  {reading}")

    print()
    print("b_ratio near 1.0 => the gate did not separate the two blocks, so the")
    print("  policy's key-weight map is irrelevant and the failure is an")
    print("  optimisation problem.")
    print("b_ratio far from 1.0 with a small w_ratio => the gate DID separate")
    print("  them and the parameterisation threw the separation away.")
    print("Neither is evidence of BENEFIT -- see the accuracy table.")

    if args.json_out:
        args.json_out.write_text(json.dumps(rows, indent=2))
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
