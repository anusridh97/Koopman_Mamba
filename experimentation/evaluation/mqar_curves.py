"""MQAR learning curves, summarised in a way that survives a phase transition.

MQAR is learned through grokking, not along a smooth curve, and that makes the
obvious summaries actively misleading. From the earlier beta arm (job 446102,
kv=8 / gap=128, proxy geometry, seed 42):

    step 1333   learned 0.2969   one 0.2773   head_scalar 0.3125   linear 0.9922
    step 2666   learned    ...   one 1.0000   head_scalar 0.3164   linear 0.9980

Read the step-1333 row as a ranking and you conclude that `linear` massively
outperforms and the default gate is no better than no gate. By step 2666 `one`
is at 1.0000. The spread was GROKKING TIME, which varies with the seed.

So this module provides the statistics `docs/beta-exponent-preregistration.md`
fixed in advance, and deliberately provides no single scalar to rank policies on.
Three design decisions, each of which is a refusal:

  * **`lc_area` is trapezoidal, not a mean over evals.** Extending a censored run
    appends coarser eval points, so an arithmetic mean would let the DENSITY of
    the eval schedule change the number. Normalised by step span, so it is
    bounded in [0, 1] and comparable across runs with different schedules.
  * **Censoring is a state, not a zero.** A run that has not grokked by the
    budget is not a run that will not grok. `grokked` / `censored` / no-data are
    three distinct outcomes and `run_statistics` cannot emit a fourth.
  * **`summarise_policy` returns ranges and counts, never a mean accuracy.**
    Averaging pre-grok and post-grok accuracies is exactly the mistake above.
    The within-policy spread across seeds at this cell is larger than most
    between-policy gaps, so the range is the honest summary.

Pure arithmetic over `(step, accuracy)` pairs, with no torch and no filesystem,
so it is covered by the CPU suite
(`code-tests/test_beta_exponent_mqar_statistics.py`) rather than only exercised
on a GPU node.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple

Curve = Sequence[Tuple[int, float]]

#: The line `experiments/mqar_finetune.py` prints at every eval. Matched against
#: the real format rather than an approximation of it: a parser that silently
#: returns [] on a real log reports "no data", which reads as an infrastructure
#: failure and would be chased as one.
_ACC_LINE = re.compile(
    r"\[step\s+(\d+)\]\s+in-task accuracy\s*\([^)]*\):\s*([0-9.]+)")

#: The predeclared grok threshold. Named here so the analysis script, the
#: launcher's summary and the tests all read one constant; two copies would be
#: two analyses of the same run that can disagree.
GROK_THRESHOLD = 0.90


def parse_curve(text: str) -> List[Tuple[int, float]]:
    """`(step, accuracy)` pairs from a run log, ordered by step.

    Deduplicated to the LAST value seen for a given step, because a resumed run
    re-evaluates steps a preempted attempt already logged and the resumed value
    is the one that describes the surviving checkpoint.
    """
    seen: Dict[int, float] = {}
    for step, acc in _ACC_LINE.findall(text):
        seen[int(step)] = float(acc)
    return [(s, seen[s]) for s in sorted(seen)]


def lc_area(curve: Curve) -> Optional[float]:
    """Normalised area under the accuracy curve: the sample-efficiency statistic.

    `None` for an empty curve. That is not the same as 0.0 -- a run that produced
    no eval is missing data, and reporting 0.0 would make an infrastructure
    failure indistinguishable from a scientific one.

    A single eval has zero step span, so its area is that accuracy: the only
    answer that neither divides by zero nor reports 0.0 for a run sitting at 1.0.
    """
    pts = sorted(curve)
    if not pts:
        return None
    if len(pts) == 1:
        return float(pts[0][1])
    span = pts[-1][0] - pts[0][0]
    if span <= 0:
        return float(sum(a for _, a in pts) / len(pts))
    total = 0.0
    for (s0, a0), (s1, a1) in zip(pts, pts[1:]):
        total += 0.5 * (a0 + a1) * (s1 - s0)
    return total / span


def run_statistics(curve: Curve, *,
                   threshold: float = GROK_THRESHOLD) -> Dict[str, object]:
    """The predeclared per-run record. Never a single score.

    `grokked` is `True` / `False` / `None`, and the third value is load-bearing:
    `None` means no eval was produced, which is an infrastructure outcome and
    must not be counted in a grok rate. `censored` is `True` exactly when the run
    produced evals and none crossed the threshold -- so `grokked` and `censored`
    are never both true, and the grok-rate table cannot double-count a run.

    `threshold` is inclusive at the declared value: accuracy == 0.90 groks.
    Fixed in code because two analyses that disagree about the boundary are two
    different experiments.
    """
    pts = sorted(curve)
    if not pts:
        return {"n_evals": 0, "grokked": None, "grok_step": None,
                "censored": None, "lc_area": None, "final_acc": None,
                "best_acc": None, "first_step": None, "last_step": None}
    grok_step = next((s for s, a in pts if a >= threshold), None)
    grokked = grok_step is not None
    return {
        "n_evals": len(pts),
        "grokked": grokked,
        "grok_step": grok_step,
        # Not `not grokked`: that would also be True for the no-data case, which
        # returns above precisely so it cannot be mistaken for a censored run.
        "censored": not grokked,
        "lc_area": lc_area(pts),
        "final_acc": float(pts[-1][1]),
        "best_acc": float(max(a for _, a in pts)),
        "first_step": pts[0][0],
        "last_step": pts[-1][0],
    }


def summarise_policy(runs: Sequence[Dict[str, object]]) -> Dict[str, object]:
    """A cell's outcome across seeds: counts and ranges, no ranking scalar.

    There is deliberately no `mean_acc` / `final_acc_mean` key, and adding one
    would defeat the module. With the budget straddling grokking, a mean over
    seeds is a mean over "solved" and "not yet solved", which is a grok RATE
    dressed as an accuracy. The grok rate is reported as a rate.

    `grok_steps` lists only runs that actually grokked. Imputing a grok step for
    a censored run -- as the budget, or as infinity -- would make the mean grok
    time a function of how long the job was allowed to run.
    """
    scored = [r for r in runs if r.get("grokked") is not None]
    finals = [r["final_acc"] for r in scored]
    areas = [r["lc_area"] for r in scored if r["lc_area"] is not None]
    bests = [r["best_acc"] for r in scored]
    return {
        "n": len(scored),
        "n_no_data": len(runs) - len(scored),
        "n_grokked": sum(1 for r in scored if r["grokked"]),
        "n_censored": sum(1 for r in scored if r["censored"]),
        "grok_steps": sorted(r["grok_step"] for r in scored if r["grokked"]),
        "final_acc_range": (min(finals), max(finals)) if finals else None,
        "best_acc_range": (min(bests), max(bests)) if bests else None,
        "lc_area_range": (min(areas), max(areas)) if areas else None,
    }
