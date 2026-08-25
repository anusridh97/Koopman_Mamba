#!/usr/bin/env python
"""The MQAR arm, summarised in a way that survives grokking.

## Why a final accuracy is not enough, and why an intermediate one is a trap

MQAR is learned through a phase transition, not a smooth curve. Observed in job
446102 at kv=8/gap=128 on the proxy geometry, seed 42:

    step 1333   learned 0.2969   one 0.2773   head_scalar 0.3125   linear 0.9922
    step 2666   learned    ...   one 1.0000   head_scalar 0.3164   linear 0.9980

Reading the step-1333 row as a ranking would have concluded that `linear`
massively outperforms and that the standard `learned` gate is no better than no
gate at all. By step 2666 `one` is at 1.0000. The step-1333 spread was
GROKKING TIME, and grokking time is exactly the kind of quantity that varies
with the seed.

So this script reports three things and refuses to collapse them:

  final       accuracy at the last eval. The quantity that matters if every
              policy eventually solves the cell -- and the quantity that is
              USELESS when they all reach 1.0, which is a CEILING and is
              reported as such rather than as a tie.
  best        max accuracy over the run, so a policy that solved the task and
              then degraded is visible rather than averaged away.
  grok_step   first eval at which accuracy >= 0.9. A real difference in sample
              efficiency, and the only thing the step-1333 row could ever have
              been evidence about -- but it is seed-noisy, so it is reported
              per seed and never as a single number per policy.

## The floor this arm does NOT have

The LM arm has a measured noise floor (sigma = 9.50e-3 over five seeds). This
arm does not: MQAR's seed-to-seed spread at this geometry has never been
measured, and grokking makes it large by construction. So a difference here is
reported with every seed visible and NO resolvable-effect threshold, because
inventing one would be worse than admitting there is none. With >= 3 seeds per
policy the script prints the observed range per cell, which is the honest
substitute: a policy difference smaller than the within-policy range across
seeds is not a finding.

Usage:
    python scripts/analyze_beta_policy_mqar.py <run_root> [--threshold 0.9]

<run_root> is a `beta-mqar-<jobid>` directory containing
`mqar-<policy>-seed<N>.log` files.
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

_ACC = re.compile(r"\[step\s+(\d+)\]\s+in-task accuracy \([^)]*\):\s+([0-9.]+)")
_NAME = re.compile(r"^mqar-(?P<policy>.+)-seed(?P<seed>\d+)$")


def _curve(log: Path):
    return [(int(s), float(a)) for s, a in _ACC.findall(log.read_text())]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--threshold", type=float, default=0.9,
                        help="accuracy defining 'grokked'")
    args = parser.parse_args(argv)

    logs = sorted(args.run_root.glob("mqar-*.log"))
    if not logs:
        print(f"FATAL: no mqar-*.log under {args.run_root}", file=sys.stderr)
        return 2

    rows = []
    for log in logs:
        m = _NAME.match(log.stem)
        if not m:
            print(f"SKIP {log.name}: unexpected name", file=sys.stderr)
            continue
        curve = _curve(log)
        if not curve:
            print(f"SKIP {log.name}: no accuracy lines yet", file=sys.stderr)
            continue
        grok = next((s for s, a in curve if a >= args.threshold), None)
        rows.append({
            "policy": m.group("policy"), "seed": int(m.group("seed")),
            "final": curve[-1][1], "best": max(a for _s, a in curve),
            "grok_step": grok, "last_step": curve[-1][0], "n_evals": len(curve),
        })

    print(f"# MQAR write-gate arm  ({args.run_root})")
    print()
    print(f"{'policy':>13} {'seed':>5} {'final':>7} {'best':>7} "
          f"{'grok@' + str(args.threshold):>10} {'last step':>10} {'evals':>6}")
    for row in sorted(rows, key=lambda r: (r["policy"], r["seed"])):
        print(f"{row['policy']:>13} {row['seed']:>5} {row['final']:>7.4f} "
              f"{row['best']:>7.4f} "
              f"{(str(row['grok_step']) if row['grok_step'] else 'never'):>10} "
              f"{row['last_step']:>10} {row['n_evals']:>6}")

    by_policy = defaultdict(list)
    for row in rows:
        by_policy[row["policy"]].append(row)

    print()
    print(f"{'policy':>13} {'n':>3} {'grokked':>8} {'final min':>10} "
          f"{'final max':>10} {'grok min':>9} {'grok max':>9}")
    for policy, group in sorted(by_policy.items()):
        finals = [r["final"] for r in group]
        groks = [r["grok_step"] for r in group if r["grok_step"] is not None]
        # GROKKED / n is the statistic this task actually has. Accuracy at a
        # fixed budget on a phase-transition task is very nearly Bernoulli --
        # "did this seed cross before the steps ran out" -- and averaging it
        # with the pre-transition values produces a number that describes
        # neither state. Reported as a fraction so the seed count is always
        # visible next to it: 2/3 and 200/300 are not the same evidence, and a
        # bare 0.67 hides which one you have.
        print(f"{policy:>13} {len(group):>3} "
              f"{f'{len(groks)}/{len(group)}':>8} {min(finals):>10.4f} "
              f"{max(finals):>10.4f} "
              f"{(str(min(groks)) if groks else '-'):>9} "
              f"{(str(max(groks)) if groks else '-'):>9}")

    finals = [r["final"] for r in rows]
    n_seeds = min(len(g) for g in by_policy.values())
    print()

    # RAGGED DATA IS NOT A RESULT. `final` is the last eval in the log, which for
    # a run still in flight is an early-training number. Comparing one run's
    # step-8000 accuracy against another's step-1333 accuracy measures how far
    # each got, and under grokking that difference is enormous -- so every
    # verdict below would be reporting scheduling as a policy effect.
    #
    # This fired for real: with job 446106 half finished, the script printed
    # "NOT RESOLVED" from seed 42 at step 8000 against seed 43 at step 1333. The
    # per-row `last step` column showed it and the verdict ignored it, which is
    # the wrong way round.
    last_steps = {r["last_step"] for r in rows}
    if len(last_steps) > 1:
        print(f"VERDICT: INCOMPLETE -- runs are at different steps "
              f"{sorted(last_steps)}. Refusing to compare them: `final` is the "
              f"last eval present, so a run still in flight contributes an "
              f"early-training number, and under grokking that dominates any "
              f"policy effect. Re-run when every log reaches the same step.")
        return 0
    if min(finals) > 0.95:
        print(f"VERDICT: CEILING (all finals > {min(finals):.4f}). Every policy "
              f"solves this cell, so FINAL accuracy separates nothing.")
        print("  If the grok_step columns differ by more than the within-policy")
        print("  range, sample efficiency is the live difference -- and it needs")
        print("  a harder cell (larger KV or GAP) to show up in final accuracy.")
    elif max(finals) < 0.05:
        print(f"VERDICT: FLOOR (all finals < {max(finals):.4f}). No policy "
              f"learns the cell; nothing is comparable. Easier cell or more "
              f"steps.")
    elif n_seeds < 2:
        print("VERDICT: ONE SEED PER POLICY. MQAR groks at a seed-dependent "
              "step, so a spread here is not a policy effect.")
        print("  Re-run with SEEDS='42 43 44' before reading any ordering.")
    else:
        # If ANY policy both grokked and failed to grok across its own seeds,
        # the budget straddles the phase transition and accuracy at that budget
        # is a coin flip about timing rather than a capability measurement.
        # Checked before the spread comparison, because a large between-policy
        # spread is exactly what a straddling budget produces.
        straddling = sorted(
            p for p, g in by_policy.items()
            if 0 < sum(1 for r in g if r["grok_step"] is not None) < len(g))
        if straddling:
            print(f"VERDICT: BUDGET STRADDLES THE TRANSITION for {straddling}. "
                  f"Those policies both grokked and failed to grok across their "
                  f"own seeds, so accuracy at this budget is measuring WHETHER "
                  f"each seed crossed in time -- not what the policy can do.")
            print("  Any ordering read from the means here is grokking-time "
                  "noise. This is not a")
            print("  small effect being missed: it is a large one that changes "
                  "sign with the seed.")
            print("  Fix by raising --max_steps until every seed of every "
                  "policy grokks (then final")
            print("  accuracy is a ceiling and grok_step is the live "
                  "quantity), or by using a cell")
            print("  hard enough that none of them does.")
            return 0
        # The honest substitute for a noise floor: compare the between-policy
        # spread of cell means against the largest WITHIN-policy spread.
        means = {p: sum(r["final"] for r in g) / len(g)
                 for p, g in by_policy.items()}
        within = max(max(r["final"] for r in g) - min(r["final"] for r in g)
                     for g in by_policy.values())
        between = max(means.values()) - min(means.values())
        print(f"between-policy spread of cell means: {between:.4f}")
        print(f"largest within-policy spread across seeds: {within:.4f}")
        if between <= within:
            print("VERDICT: NOT RESOLVED. The policy differences are no larger "
                  "than the seed-to-seed spread inside a single policy, so this "
                  "arm does not separate them.")
        else:
            print("VERDICT: SEPARATED. The between-policy spread exceeds the "
                  "within-policy spread. Report the per-seed table, not just "
                  "the means -- this arm has no calibrated noise floor, so "
                  "'exceeds the within-policy range' is the strongest claim "
                  "available.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
