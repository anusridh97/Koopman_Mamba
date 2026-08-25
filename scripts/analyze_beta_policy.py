#!/usr/bin/env python
"""The 4x5 write-gate comparison, with the denominator the design earns.

Reads a completed `beta-policy-*` study's `trials.csv` and reports, per policy
cell: n, mean held-out loss, sd, and the contrast against the `learned` control
stated in units of the POOLED sigma.

## Why this is not `analysis.anchor_contrasts`

`anchor_contrasts` contrasts a SINGLE trial against a replicate mean and
excludes replicate-group members from its rows. Every design in this study is a
group member, so it returns nothing -- by design, since the alternative
(one grouped control plus fifteen lone anchors) would have thrown away the
replicate structure of the three non-control cells.

The denominators differ, and that is the whole reason this file exists.
`anchor_contrasts`'s own comment makes the point better than a paraphrase: the
denominator is what decides whether a real effect is reported as resolvable, and
using a threshold 29% too high "cannot manufacture a finding" but silently
discards real ones. The three relevant standard deviations, for n = 5:

    single trial vs single trial   sigma*sqrt(2)       = 1.414*sigma
    single trial vs 5-mean         sigma*sqrt(1 + 1/5) = 1.095*sigma
    5-mean vs 5-mean               sigma*sqrt(2/5)     = 0.632*sigma   <- here

At the repo's 2-sd convention the resolvable effect is therefore 1.265*sigma,
against 2.83*sigma for the trial-vs-trial comparison the noise-floor studies
report. Using the wrong one would make this study look 2.2x less sensitive than
it is.

## Sigma is pooled, and pooled the right way

Four cells of five give 16 degrees of freedom, against the 4 that job 445832's
five-trial measurement had -- whose own header records that 4 dof leaves sigma
with wide error bars. Pooled by summing squared deviations within each cell and
dividing by the total within-cell dof, NOT by averaging the four cell sds: an
average would weight a cell of 2 the same as a cell of 5, which is the same
mistake `analysis.noise_floor` documents avoiding.

Pooling assumes the four cells share a variance. That is an assumption, not a
measurement, and it is reported: per-cell sds are printed so a reader can see
whether one cell is visibly noisier, and Levene-style heterogeneity would show
up there rather than being hidden inside the pooled number.

## What this script will not do

It will not pick a winner when the contrasts do not clear the floor. A cell mean
closer to zero than 1.265*sigma is reported as UNRESOLVED with the floor beside
it, because "the simpler model is indistinguishable from the learned gate,
therefore prefer the simpler model" is the correct reading of that outcome and
ranking four unresolved numbers would obscure it.

Usage:
    python scripts/analyze_beta_policy.py <trials.csv> [--control learned]
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

#: The repo's convention throughout `analysis.py`: an effect is resolvable at
#: two standard deviations of its OWN sampling distribution.
RESOLVE_SIGMAS = 2.0

#: Bonferroni-corrected threshold, used when there is more than one contrast.
#:
#: Three cells against one control is three tests. At 2 sigma each (alpha ~ 0.05
#: two-sided) the family-wise false-positive rate is 1 - 0.95^3 = 14%: about one
#: run in seven produces a spurious "RESOLVED". That bias points TOWARD
#: declaring a winner, which is the one direction this comparison must not be
#: biased in -- a null result is an acceptable and expected outcome here, and an
#: uncorrected threshold quietly trades it for a false positive.
#:
#: alpha/3 two-sided is 2.39 sigma. Applied automatically rather than offered as
#: a flag, and the uncorrected per-contrast sigma count is still printed beside
#: it so a reader can see both.
_BONFERRONI_SIGMAS = {1: 2.00, 2: 2.24, 3: 2.39, 4: 2.50, 5: 2.58}


def _threshold_sigmas(n_contrasts: int) -> float:
    """Sigma multiple a contrast must clear, corrected for how many were made."""
    if n_contrasts <= 1:
        return RESOLVE_SIGMAS
    return _BONFERRONI_SIGMAS.get(n_contrasts, 2.81)  # ~alpha/10 beyond the table

#: The column `report.py` writes for a trial's reference group, and the one that
#: identifies a cell member. Filtering on it (rather than on trial count) is what
#: makes the script correct in the presence of target overshoot: an overshoot
#: trial is a sampled point with no group, so it is excluded here for the same
#: reason `noise_floor` excludes it.
GROUP_COL = "attr_reference_group"
VALUE_COL = "objective"
POLICY_COL = "param_beta_policy"
SEED_COL = "attr_model_seed"


def _rows(path: Path):
    with path.open() as handle:
        for row in csv.DictReader(handle):
            if row.get("state") != "COMPLETE":
                continue
            group = (row.get(GROUP_COL) or "").strip()
            if not group:
                continue
            try:
                value = float(row[VALUE_COL])
            except (KeyError, TypeError, ValueError):
                continue
            if not math.isfinite(value):
                continue
            yield group, row.get(POLICY_COL) or group, value, row.get(SEED_COL)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("trials_csv", type=Path)
    parser.add_argument("--control", default="learned",
                        help="the policy every other cell is contrasted against")
    args = parser.parse_args(argv)

    if not args.trials_csv.exists():
        print(f"FATAL: {args.trials_csv} does not exist", file=sys.stderr)
        return 2

    cells: dict[str, list[float]] = defaultdict(list)
    seeds: dict[str, list[str]] = defaultdict(list)
    for _group, policy, value, seed in _rows(args.trials_csv):
        cells[policy].append(value)
        seeds[policy].append(str(seed))

    if not cells:
        print("FATAL: no COMPLETE trials with a reference_group in "
              f"{args.trials_csv}. Either the study has not finished a cell "
              "member yet, or the design file stopped declaring groups -- in "
              "which case there is no replicate structure and no floor.",
              file=sys.stderr)
        return 2

    # Duplicate seeds inside a cell are refused rather than pooled, for the
    # reason `analysis.noise_floor` refuses them: the base spec sets
    # `deterministic: true`, so two members at one seed land on the same loss and
    # contribute a zero to the sum of squares AND a degree of freedom -- driving
    # sigma toward zero and making every effect look resolved.
    for policy, members in sorted(seeds.items()):
        repeated = sorted({s for s in members if members.count(s) > 1})
        if repeated:
            print(f"FATAL: cell {policy!r} has repeated seed(s) {repeated}. "
                  f"Two members at one seed are one datapoint counted twice, "
                  f"and under `deterministic: true` they bias sigma toward "
                  f"zero. Refusing rather than pooling.", file=sys.stderr)
            return 2

    # Pooled within-cell sigma: sum of squared deviations over total dof.
    ss = 0.0
    dof = 0
    for values in cells.values():
        if len(values) < 2:
            continue
        mean = sum(values) / len(values)
        ss += sum((v - mean) ** 2 for v in values)
        dof += len(values) - 1
    sigma = math.sqrt(ss / dof) if dof else None

    print(f"# beta-policy comparison  ({args.trials_csv})")
    print()
    print(f"cells: {len(cells)}   completed members: "
          f"{sum(len(v) for v in cells.values())}")
    if sigma is None:
        print("pooled sigma: UNAVAILABLE (no cell has two or more members yet)")
        print()
    else:
        print(f"pooled sigma: {sigma:.4e}   within-cell dof: {dof}")
        print(f"  trial vs trial resolvable   2*sigma*sqrt(2)   = "
              f"{RESOLVE_SIGMAS * sigma * math.sqrt(2):.4e}")
    control = cells.get(args.control)
    control_mean = sum(control) / len(control) if control else None
    n_contrasts = max(0, len(cells) - (1 if control else 0))
    crit = _threshold_sigmas(n_contrasts)

    if sigma is not None:
        # The typical cell size, not a hardcoded 5: a cell that lost a member to
        # a failure must not be described as though it were complete. The
        # per-row arithmetic below already uses each cell's own n.
        sizes = sorted(len(v) for v in cells.values())
        n_typ = sizes[len(sizes) // 2]
        print(f"  mean vs mean resolvable     {crit:.2f}*sigma*sqrt(2/n) = "
              f"{crit * sigma * math.sqrt(2 / n_typ):.4e}   "
              f"(n={n_typ} per cell; THE relevant one here)")
        if n_contrasts > 1:
            print(f"  threshold is Bonferroni-corrected for {n_contrasts} "
                  f"contrasts ({crit:.2f} sigma, not {RESOLVE_SIGMAS:.2f}): at "
                  f"2 sigma each the")
            print(f"  family-wise false-positive rate would be "
                  f"{100 * (1 - 0.95 ** n_contrasts):.0f}%, and that bias points "
                  f"toward declaring a winner.")
        print()

    print(f"{'policy':>14}  {'n':>2}  {'mean':>12}  {'sd':>10}  "
          f"{'delta vs ' + args.control:>18}  {'sigmas':>7}  verdict")
    if control is None:
        print(f"  (no {args.control!r} cell present -- deltas omitted)")

    for policy in sorted(cells, key=lambda p: (p != args.control, p)):
        values = cells[policy]
        mean = sum(values) / len(values)
        sd = (math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))
              if len(values) > 1 else None)
        if policy == args.control or control_mean is None:
            delta = sigmas = None
        else:
            delta = mean - control_mean
            # The mean-vs-mean sd, using each cell's own n rather than assuming
            # 5: a cell that lost a member to a failure must not be scored as
            # though it were complete.
            contrast_sd = (sigma * math.sqrt(1.0 / len(values) + 1.0 / len(control))
                           if sigma else None)
            sigmas = abs(delta) / contrast_sd if contrast_sd else None
        if policy == args.control:
            verdict = "CONTROL"
        elif sigmas is None:
            verdict = "-"
        elif sigmas >= crit:
            verdict = "RESOLVED " + ("worse" if delta > 0 else "BETTER")
        else:
            verdict = "unresolved"
        print(f"{policy:>14}  {len(values):>2}  {mean:>12.6f}  "
              f"{('%10.3e' % sd) if sd is not None else '         -'}  "
              f"{('%18.4e' % delta) if delta is not None else '                 -'}  "
              f"{('%7.2f' % sigmas) if sigmas is not None else '      -'}  {verdict}")

    print()
    if sigma is not None and control_mean is not None:
        unresolved = [
            p for p in cells
            if p != args.control and len(cells[p]) > 0
            and abs(sum(cells[p]) / len(cells[p]) - control_mean)
            < crit * sigma * math.sqrt(1.0 / len(cells[p]) + 1.0 / len(control))
        ]
        # `len(cells) > 1` guards a vacuous verdict: with only the control
        # present, `unresolved` is empty and so is `len(cells) - 1`, so the
        # equality below held and the script announced "no policy is
        # distinguishable" having compared nothing. Caught by running it against
        # job 445832's single-group trials.csv.
        if len(cells) > 1 and len(unresolved) == len(cells) - 1:
            print("READING: no policy is distinguishable from the control above "
                  "the floor.")
            print("  That is a result, not a failed experiment. Prefer the "
                  "SIMPLEST policy among")
            print("  the indistinguishable ones -- `one` has no gate parameters "
                  "at all -- and")
            print("  report the floor alongside it, so the claim is 'no "
                  "difference larger than X'")
            print("  rather than 'no difference'.")
        elif unresolved:
            print(f"READING: {sorted(unresolved)} are indistinguishable from the "
                  f"control; the rest are not.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
