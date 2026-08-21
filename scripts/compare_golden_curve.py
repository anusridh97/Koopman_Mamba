#!/usr/bin/env python
"""Compare a training log against code-tests/golden_4m_curve.json.

Run after any change to the training loop -- the §6.2 TrainTask unification above
all -- to answer "did this change training?" by measurement rather than by
reading the diff.

    sbatch scripts/capture_golden_curve.sbatch        # produces two run logs
    python scripts/compare_golden_curve.py <train.log>

The threshold is not a guess. The golden was captured as TWO runs of the same
spec at the same seed, and their disagreement is recorded in the artifact as
`noise_floor`. Measured 2026-08-21 at commit b73a853: mean 8.2e-05, max 2.0e-04
over 40 logged steps.

That 2.0e-04 is the *log format's* precision, not a numerical one -- the trainer
prints `loss %.4f`, so two units in the last digit is the smallest difference
observable at all. The two runs are effectively bit-identical. Which is the useful
conclusion: a real regression has to clear only ~1e-3 to be unambiguous, so this
is a sharp instrument, not a rubber stamp.

Exit status is 1 on a regression, so this is usable as a gate.
"""
import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from experimentation.sweep.search.metrics import parse_progress  # noqa: E402

GOLDENS = pathlib.Path(__file__).resolve().parents[1] / "code-tests"
DEFAULT_GOLDEN = GOLDENS / "golden_4m_curve.json"
# 5x the observed noise floor. Generous enough that log-precision rounding and
# any genuine nondeterminism cannot trip it, tight enough that a changed
# optimizer, schedule, accumulation count or batch order cannot hide under it --
# those move the curve by 1e-2 or more, two orders of magnitude clear.
TOLERANCE_MULTIPLE = 5.0
# ...but a floor on the floor. golden_mqar_curve.json was captured with
# --deterministic and its measured spread is EXACTLY 0.0, so 5x it is 0.0 and the
# last digit of the log format would read as a regression. The log prints
# `loss %.4f`, so 1e-4 is the smallest difference that is even representable;
# anything at or below it is indistinguishable from equality.
MIN_TOLERANCE = 1e-4


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("log", type=pathlib.Path, help="a training log to check")
    ap.add_argument("--golden", type=pathlib.Path, default=DEFAULT_GOLDEN,
                    help="which baseline; see code-tests/golden_*_curve.json")
    ap.add_argument("--replicate", default="A", choices=("A", "B"))
    args = ap.parse_args()

    golden = json.loads(args.golden.read_text())
    baseline = {int(k): v for k, v in golden["replicates"][args.replicate].items()}
    floor = golden["noise_floor"]["max_abs_delta"]
    tol = max(floor * TOLERANCE_MULTIPLE, MIN_TOLERANCE)

    observed = {p.step: p.loss for p in parse_progress(args.log.read_text())}
    if not observed:
        print(f"no progress lines parsed from {args.log} -- did the run get anywhere?")
        return 1

    what = golden.get("spec") or golden.get("trainer", "<unknown>")
    print(f"golden captured at commit {golden['commit'][:9]} from {what}")
    print(f"noise floor {floor:.6f}  ->  tolerance {tol:.6f} ({TOLERANCE_MULTIPLE:g}x)")

    shared = sorted(set(baseline) & set(observed))
    missing = sorted(set(baseline) - set(observed))
    if missing:
        print(f"\nWARNING: {len(missing)} golden step(s) absent from this log: "
              f"{missing[:8]}{'...' if len(missing) > 8 else ''}")
    if not shared:
        print("no steps in common -- cannot compare")
        return 1

    over = []
    for s in shared:
        d = abs(observed[s] - baseline[s])
        flag = ""
        if d > tol:
            over.append((s, d))
            flag = "  <-- OVER TOLERANCE"
        print(f"  step {s:>5}  golden {baseline[s]:.6f}  now {observed[s]:.6f}  "
              f"d {d:.6f}{flag}")

    worst = max((abs(observed[s] - baseline[s]) for s in shared), default=0.0)
    print(f"\ncompared {len(shared)} steps; worst |delta| = {worst:.6f}")
    if over:
        print(f"REGRESSION: {len(over)} step(s) exceed tolerance, worst at "
              f"step {max(over, key=lambda t: t[1])[0]}")
        print("The training loop's behaviour changed. That may be intended -- if so,")
        print("recapture the golden and say in the commit message what changed and why.")
        return 1
    print("MATCH: the loop still trains identically within the measured noise floor.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
