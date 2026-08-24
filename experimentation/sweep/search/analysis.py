"""Post-hoc analysis of a finished study: read-only, and careful about what it
is allowed to claim.

`report.py` writes what a study produced -- trials.csv, a ranked table, a
promotion sweep. This writes what a study MEANS, for the one question the
interaction study was designed to answer: which pairs of SKA hyperparameters
interact.

**The methodological problem, stated first because it shapes everything here.**
The study samples adaptively. A multivariate TPE concentrates its later proposals
in whatever region it currently believes is good, so by trial 200 the sampled
columns of `trials.csv` are correlated *with each other* for reasons that have
nothing to do with the model. `corr(ska_rank, ska_ridge)` over that table is a
description of where the sampler went. Reported as a finding it would look
exactly like an interaction -- same sign, same magnitude, same plausibility --
and it would be an artefact of the search strategy.

So the output separates three things a single correlation matrix conflates, and
labels each one in the artefact rather than only here:

  ``main-effect importance``
      Variance in the objective attributable to one axis, from an evaluator that
      accounts for the others. This IS evidence about the model, within the
      region the sampler explored.

  ``pairwise response structure``
      The mean objective in each cell of a prespecified 2-way table, with the
      cell count beside it. This is the quantity the study was designed to
      estimate, and the cell counts are what make it readable -- under adaptive
      sampling they are wildly uneven by construction, and a mean over 2 trials
      next to a mean over 40 invites reading noise as effect.

  ``sampler-induced correlation``
      Correlation between sampled columns. A diagnostic of the search path.
      Explicitly NOT evidence, and the artefact says so in those words.

**The pairs are prespecified.** Seven, fixed in `PRESPECIFIED_PAIRS`, named in
the study config before any trial ran. Nine axes admit 36 pairs; picking which
to report after looking at the data is how one of 36 comes out looking
significant. The objective is also deliberately absent from the correlation
table: a parameter-vs-objective correlation reads as a main effect and is not
one, for the same adaptive-sampling reason.

**No plots, by measurement not by preference.** sklearn and matplotlib are not
installed in this environment, so fANOVA and MeanDecreaseImpurity are
unavailable and there is nothing to draw with. `PedAnovaImportanceEvaluator` is
pure numpy and works. Everything ships as CSV, JSON or Markdown, and adding an
unverified dependency was out of scope. If even PedAnova becomes unavailable the
run degrades -- counts, pairwise tables and the Pareto front need no evaluator --
rather than failing.

**Read-only.** This may run against a journal other workers are still writing
to, so nothing here calls `tell`, `set_user_attr`, or `enqueue_trial`.
"""
from __future__ import annotations

import csv
import io
import math
import warnings
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import optuna

from experimentation.atomic_io import atomic_write_json, atomic_write_text
from experimentation.sweep.search.study import ANCHOR_ATTR

__all__ = ["PRESPECIFIED_PAIRS", "analyse", "pairwise_table", "pareto_front",
           "sampler_induced_correlations", "write_analysis"]

#: The interactions this study was designed to estimate, fixed before any trial
#: ran. See the module docstring for why the list is in code and not chosen after
#: looking at the data.
#:
#: Each pair has a reason:
#:   rank x ridge          the ridge conditions a Gram matrix whose size IS the
#:                         rank, so the useful loading plausibly scales with it
#:   rank x norm-clip      the clip bounds a whitened quantity whose natural
#:                         scale is sqrt(rank); the multiplier may not be enough
#:   rank x power_K        K composes the operator; more capacity per application
#:                         may need fewer applications
#:   layers x layerscale   N branches each scaled by alpha is a total contribution
#:                         of roughly N*alpha, so the two should trade off
#:   layers x placement    where adapters go can only matter given how many
#:   ridge x power_K       K is a matrix power, so conditioning errors compound
#:   layerscale x lr       both scale an update; the classic confound
PRESPECIFIED_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("ska_rank", "ska_ridge"),
    ("ska_rank", "norm_clip_multiplier"),
    ("ska_rank", "ska_power_K"),
    ("n_ska_layers", "ska_layerscale_init"),
    ("n_ska_layers", "placement"),
    ("ska_ridge", "ska_power_K"),
    ("ska_layerscale_init", "learning_rate"),
)

#: Bins for a continuous axis in a 2-way table. Three, not ten: cell counts are
#: uneven under adaptive sampling, and ten bins over 200 completed trials in a
#: 9-dimensional space leaves most cells with a handful of trials or none.
DEFAULT_BINS = 3

_CAVEAT = (
    "The sampler was ADAPTIVE. Later trials were proposed inside whatever region "
    "the sampler believed was good, so cell counts are uneven by construction "
    "and the sampled columns are correlated with each other for reasons that "
    "have nothing to do with the model. Read cell means WITH their n. These "
    "seven pairs were PRESPECIFIED before any trial ran; nine axes admit 36 "
    "pairs, and choosing which to report after looking is how one of 36 comes "
    "out looking significant.")


# --------------------------------------------------------------- trial views ----

def _require_minimize(study: optuna.study.Study) -> None:
    """Refuse a maximize study rather than silently inverting every conclusion.

    Every ranking here sorts ASCENDING and `pareto_front` sweeps for smaller-is-
    better. Pointed at a maximize journal -- which `scripts/analyze_interactions.py`
    will happily open, since it analyses whatever it is given -- the "front" comes
    back containing strictly dominated points and missing the best trial, with no
    error. Measured on a 3-trial maximize study in review.

    A hard failure, not a warning: the output is a table of numbers that looks
    exactly as authoritative when it is backwards.
    """
    if study.direction != optuna.study.StudyDirection.MINIMIZE:
        raise ValueError(
            f"this analysis assumes a MINIMIZE study and {study.study_name!r} is "
            f"{study.direction.name}. Every ranking here sorts ascending and the "
            f"Pareto sweep takes smaller-is-better, so the results would be "
            f"silently inverted rather than wrong-looking. StudySpec permits only "
            f"'minimize'; this journal was not produced by one.")


def _completed(study: optuna.study.Study) -> List[Any]:
    return [t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]


def _state_counts(study: optuna.study.Study) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for trial in study.trials:
        name = getattr(trial.state, "name", str(trial.state))
        counts[name] = counts.get(name, 0) + 1
    counts["total"] = len(study.trials)
    return counts


def _failure_tally(study: optuna.study.Study) -> Dict[str, int]:
    tally: Dict[str, int] = {}
    for trial in study.trials:
        if getattr(trial.state, "name", "") != "FAIL":
            continue
        reason = trial.user_attrs.get("failure") or "(no reason recorded)"
        tally[reason] = tally.get(reason, 0) + 1
    return tally


# ------------------------------------------------------- pairwise structure ----

def _is_continuous(study: optuna.study.Study, axis: str) -> bool:
    """Was this axis DECLARED continuous, per the journal's own distributions?

    Read off `trial.distributions` rather than inferred from the observed values,
    and that is not a stylistic preference -- inference gets it wrong. A first
    version treated an axis as continuous when it had more distinct values than
    `bins`, which binned `ska_rank`'s four declared choices into three and merged
    8 with 16: a table over a fabricated level, reported as if it were a rank.

    The journal records every trial's params WITH their distributions, so the
    declared kind is durable and does not have to be guessed. `study.py`'s
    `to_distribution` only ever emits CategoricalDistribution or
    FloatDistribution, so "not categorical" is a sufficient test and a
    FloatDistribution over a handful of observed values still gets binned --
    correctly, since the sampler could have drawn any value in the range.
    """
    for trial in study.trials:
        distribution = trial.distributions.get(axis)
        if distribution is None:
            continue
        return not isinstance(distribution,
                              optuna.distributions.CategoricalDistribution)
    return False


def _levels(values: Sequence[Any], *, bins: int, continuous: bool
            ) -> Tuple[List[Dict[str, Any]], bool]:
    """Distinct levels for an axis, binning it only if it is continuous.

    `continuous` comes from the journal's declared distribution (see
    `_is_continuous`), never from the observed values.
    """
    distinct = sorted({v for v in values if v is not None},
                      key=lambda v: (isinstance(v, str), v))
    numeric = all(isinstance(v, (int, float)) and not isinstance(v, bool)
                  for v in distinct)
    if not continuous or not numeric or len(distinct) <= 1:
        return [{"label": v, "value": v} for v in distinct], False

    lo, hi = float(distinct[0]), float(distinct[-1])
    # Log-spaced when the axis spans more than a factor of 10 and is positive:
    # ridge, layerscale and lr are all declared log, and linear bins over a
    # decade put almost every trial in the first bin.
    logarithmic = lo > 0 and hi / lo >= 10.0
    edges = []
    for index in range(bins + 1):
        fraction = index / bins
        if logarithmic:
            edges.append(math.exp(math.log(lo) + fraction * (math.log(hi) - math.log(lo))))
        else:
            edges.append(lo + fraction * (hi - lo))
    # BOTH ends repaired, not just the top. `exp(log(x))` is frequently ABOVE x
    # for a decimal literal -- measured: exp(log(0.002)) == 0.0020000000000000005
    # -- so a computed bottom edge excludes the minimum-valued trial, `_bucket`
    # returns None for it, and `pairwise_table` used to drop it silently.
    #
    # That was not a theoretical rounding worry. `ska_layerscale_init`'s declared
    # low is 0.002 and the `layerscale-low` anchor resolves to EXACTLY 0.002, so a
    # curated design endpoint vanished from two of the seven prespecified tables
    # on every single run. Only the top edge was repaired because only the top
    # edge's exclusion was obvious from the half-open comparison.
    edges[0] = lo
    edges[-1] = hi
    out = []
    for index in range(bins):
        out.append({"label": f"[{edges[index]:.4g}, {edges[index + 1]:.4g}]",
                    "lo": edges[index], "hi": edges[index + 1],
                    "scale": "log" if logarithmic else "linear"})
    return out, True


def _bucket(value: Any, levels: Sequence[Mapping[str, Any]], binned: bool
            ) -> Optional[int]:
    if value is None:
        return None
    if not binned:
        for index, level in enumerate(levels):
            if level["value"] == value:
                return index
        return None
    numeric = float(value)
    for index, level in enumerate(levels):
        # Half-open except for the last bin, so the maximum lands somewhere.
        if level["lo"] <= numeric < level["hi"]:
            return index
        if index == len(levels) - 1 and numeric == level["hi"]:
            return index
    # Belt and braces over the edge repair above: clamp rather than drop. A value
    # outside every bin can only come from float error at an edge, and silently
    # discarding a trial is the worst available outcome -- the table would look
    # complete and be missing a point.
    if numeric <= levels[0]["lo"]:
        return 0
    if numeric >= levels[-1]["hi"]:
        return len(levels) - 1
    return None


def pairwise_table(study: optuna.study.Study, left: str, right: str, *,
                   bins: int = DEFAULT_BINS) -> Dict[str, Any]:
    """The mean objective in each cell of a 2-way table over `left` x `right`.

    Only COMPLETED trials. A PRUNED trial has no final objective, and using its
    last intermediate value would mix two different measurements into one cell.

    Every cell carries its `n`. Under adaptive sampling the counts are uneven by
    construction, so a mean without its count is not interpretable -- this is the
    single most important thing the table reports.
    """
    _require_minimize(study)
    trials = _completed(study)
    left_values = [t.params.get(left) for t in trials]
    right_values = [t.params.get(right) for t in trials]
    left_levels, left_binned = _levels(
        left_values, bins=bins, continuous=_is_continuous(study, left))
    right_levels, right_binned = _levels(
        right_values, bins=bins, continuous=_is_continuous(study, right))

    degenerate = len(left_levels) < 2 or len(right_levels) < 2
    note = ""
    if degenerate:
        thin = [name for name, levels in ((left, left_levels),
                                          (right, right_levels))
                if len(levels) < 2]
        note = (f"DEGENERATE: {', '.join(thin)} took fewer than two distinct "
                f"values across the completed trials, so this table cannot show "
                f"structure. A fixed axis produces this; so does an axis the "
                f"sampler never varied.")

    buckets: Dict[Tuple[int, int], List[float]] = {}
    dropped = 0
    for trial, lv, rv in zip(trials, left_values, right_values):
        li = _bucket(lv, left_levels, left_binned)
        ri = _bucket(rv, right_levels, right_binned)
        if li is None or ri is None:
            # Counted, not just skipped. A silently dropped trial is how a
            # float-error bug at a bin edge hid a curated anchor endpoint from two
            # tables on every run -- the table looked complete and was not.
            dropped += 1
            continue
        buckets.setdefault((li, ri), []).append(float(trial.value))

    rows = []
    for li, level in enumerate(left_levels):
        cells = []
        for ri in range(len(right_levels)):
            values = buckets.get((li, ri), [])
            cells.append({
                "n": len(values),
                "mean": (sum(values) / len(values)) if values else None,
                "best": min(values) if values else None,
            })
        rows.append({"level": level["label"], "cells": cells})

    placed = sum(len(v) for v in buckets.values())
    return {"left": left, "right": right,
            "left_levels": left_levels, "right_levels": right_levels,
            "left_binned": left_binned, "right_binned": right_binned,
            "degenerate": degenerate, "note": note, "rows": rows,
            # Every completed trial must land somewhere. Reported so the
            # invariant is checkable from the artefact rather than trusted.
            "n_completed": len(trials), "n_placed": placed,
            "n_dropped": dropped,
            "caveat": _CAVEAT}


# ------------------------------------------------------- main-effect importance ----

def _importance_evaluator():
    """The best importance evaluator this environment can actually construct.

    A function so it can be monkeypatched in a test, and so the ImportError from
    a missing sklearn is raised HERE rather than from inside `analyse`.

    PedAnova first, and in this environment only: fANOVA and
    MeanDecreaseImpurity both import sklearn, which is not installed. PedAnova is
    pure numpy. It is flagged experimental by optuna and the warning is silenced
    at this one construction site, the same way `study.make_sampler` treats
    `constant_liar` and `multivariate`.
    """
    from optuna.importance import PedAnovaImportanceEvaluator

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",
                                category=optuna.exceptions.ExperimentalWarning)
        return PedAnovaImportanceEvaluator(), "PedAnovaImportanceEvaluator"


def _importances(study: optuna.study.Study) -> Dict[str, Any]:
    if len(_completed(study)) < 2:
        return {"available": False,
                "reason": "fewer than two completed trials"}
    try:
        evaluator, name = _importance_evaluator()
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=optuna.exceptions.ExperimentalWarning)
            values = optuna.importance.get_param_importances(
                study, evaluator=evaluator)
    except Exception as exc:                          # noqa: BLE001
        # Degrade, do not lose the run. Counts, pairwise tables and the Pareto
        # front need no evaluator, and an analysis that crashed on the one
        # optional part would throw away the parts that worked.
        return {"available": False,
                "reason": f"{type(exc).__name__}: {exc}"}
    return {
        "available": True,
        "evaluator": name,
        "values": {k: float(v) for k, v in values.items()},
        "interpretation": (
            "Variance in the objective attributable to each axis, accounting "
            "for the others. This IS evidence about the model -- but only WITHIN "
            "the region the sampler explored, which adaptive sampling makes "
            "narrower than the declared space. An axis with low importance may "
            "be unimportant, or may simply have been held near one value by the "
            "sampler after the startup window; check its spread in trials.csv "
            "before concluding the first."),
    }


# ------------------------------------------ sampler-induced correlation ----

def _pearson(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    n = len(xs)
    if n < 2:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0 or syy <= 0:
        return None                # a constant column: r is 0/0, not 0
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return sxy / math.sqrt(sxx * syy)


def sampler_induced_correlations(study: optuna.study.Study) -> Dict[str, Any]:
    """Correlation between SAMPLED COLUMNS. A diagnostic, not a finding.

    Reported because it is genuinely useful -- it shows where the sampler
    concentrated, which is what makes a low importance score ambiguous -- and
    labelled because it is the single easiest number here to misread. Two axes
    correlating at 0.6 across a TPE study says the sampler visited that diagonal,
    not that the model prefers it.

    The objective is deliberately excluded. A parameter-vs-objective correlation
    reads as a main effect and is not one: adaptive sampling concentrates good
    regions, so it conflates "this value helps" with "the sampler went here".
    Main effects come from the importance evaluator instead.

    Categorical string axes are excluded too -- Pearson's r over an arbitrary
    label ordering is a number with no meaning. Constant columns are reported by
    NAME rather than as a NaN, because a NaN in a CSV reads as a missing
    measurement rather than as a column that never varied.
    """
    trials = _completed(study)
    columns: Dict[str, List[float]] = {}
    skipped: List[str] = []
    names = sorted({k for t in trials for k in t.params})
    for name in names:
        values = [t.params.get(name) for t in trials]
        if any(v is None for v in values):
            skipped.append(name)                 # not sampled in every trial
            continue
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool)
                   for v in values):
            skipped.append(name)                 # categorical labels
            continue
        columns[name] = [float(v) for v in values]

    constant = sorted(n for n, v in columns.items() if len(set(v)) < 2)
    varying = [n for n in sorted(columns) if n not in set(constant)]

    pairs = []
    for i, a in enumerate(varying):
        for b in varying[i + 1:]:
            r = _pearson(columns[a], columns[b])
            if r is None:
                continue
            pairs.append({"a": a, "b": b, "r": r, "n": len(columns[a])})
    pairs.sort(key=lambda p: -abs(p["r"]))

    return {
        "is_evidence": False,
        "pairs": pairs,
        "constant_columns": constant,
        "excluded_columns": sorted(set(skipped)),
        "interpretation": (
            "SAMPLER DIAGNOSTIC, NOT EVIDENCE. These are correlations between "
            "sampled columns. The sampler was adaptive, so it concentrated its "
            "later proposals in a region it believed was good, and any "
            "correlation here is primarily a description of that search path. "
            "It is not a finding about the model and must not be reported as "
            "one. Its legitimate use is to qualify the importance table: an "
            "axis the sampler pinned near one value will show low importance "
            "whether or not it matters."),
    }


# --------------------------------------------------------- Pareto and rankings ----

def pareto_front(study: optuna.study.Study) -> List[Dict[str, Any]]:
    """Trials not dominated on (objective, parameter count), smallest first.

    Computed post hoc rather than searched for, which is the reason the study is
    single-objective: optuna 4.9 raises NotImplementedError from `Trial.report`
    under multiple directions, so a Pareto study cannot prune at all -- and
    pruning is what makes the study affordable. Everything needed for the front
    is in the journal.

    This is also where the parameter count is SUPPOSED to be traded against
    loss. A `parameter_penalty` bakes one exchange rate into every proposal the
    sampler makes afterwards, unrecoverably; a front lets the reader choose, and
    change their mind, without re-running anything.
    """
    _require_minimize(study)
    points = []
    for trial in _completed(study):
        count = trial.user_attrs.get("param_count")
        # `is None`, not truthiness: a param_count of 0 is absurd but would be
        # dropped as though absent, which is a different statement.
        if count is None or int(count) <= 0:
            continue
        points.append({"trial": trial.number, "objective": float(trial.value),
                       "param_count": int(count),
                       "anchor": trial.user_attrs.get(ANCHOR_ATTR) or "",
                       "run_id": trial.user_attrs.get("run_id") or ""})
    points.sort(key=lambda p: (p["param_count"], p["objective"]))

    front: List[Dict[str, Any]] = []
    best = math.inf
    for point in points:
        # Sorted by ascending size, so a point joins the front only by being
        # strictly better than everything smaller.
        if point["objective"] < best:
            front.append(point)
            best = point["objective"]
    return front


def _top_by_loss(study, k):
    rows = [{"trial": t.number, "objective": float(t.value),
             "param_count": t.user_attrs.get("param_count"),
             "anchor": t.user_attrs.get(ANCHOR_ATTR) or "",
             "run_id": t.user_attrs.get("run_id") or "",
             "worker_id": t.user_attrs.get("worker_id"),
             **{f"param_{n}": v for n, v in t.params.items()}}
            for t in _completed(study)]
    rows.sort(key=lambda r: r["objective"])
    return rows[:k]


def _top_by_ska_delta(study, k):
    """Ranked by how much zeroing the SKA branch HURT -- descending.

    The opposite direction from loss, and the more interesting ranking for this
    study: a config with a good loss whose SKA branch contributes nothing is a
    good Mamba model, not evidence for SKA.

    A trial with no recorded delta is EXCLUDED, not ranked as zero. Absent is not
    zero: the ablation is an optional part of quick_eval, and a study that never
    ran it would otherwise produce a full ranking of ties.
    """
    rows = []
    for trial in _completed(study):
        delta = trial.user_attrs.get("ska_delta")
        if delta is None:
            continue
        rows.append({"trial": trial.number, "ska_delta": float(delta),
                     "objective": float(trial.value),
                     "anchor": trial.user_attrs.get(ANCHOR_ATTR) or "",
                     "run_id": trial.user_attrs.get("run_id") or ""})
    rows.sort(key=lambda r: -r["ska_delta"])
    return rows[:k]


def _objective_vs_params(study):
    """The scatter, as the numbers behind it. No matplotlib in this env."""
    return [{"trial": t.number, "objective": float(t.value),
             "param_count": t.user_attrs.get("param_count"),
             "anchor": t.user_attrs.get(ANCHOR_ATTR) or ""}
            for t in _completed(study)]


def _rank_curve(study):
    """Best objective seen up to each completed trial, in trial order.

    The rank-plot equivalent. Monotone by construction, which is what makes it
    readable as progress -- and what makes a FLAT stretch after the startup
    window the thing to look at.
    """
    curve = []
    best = math.inf
    for trial in sorted(_completed(study), key=lambda t: t.number):
        best = min(best, float(trial.value))
        curve.append({"trial": trial.number, "objective": float(trial.value),
                      "best_so_far": best,
                      "anchor": trial.user_attrs.get(ANCHOR_ATTR) or ""})
    return curve


# ------------------------------------------------------------------- analyse ----

def analyse(study: optuna.study.Study, *, top_k: int = 15,
            bins: int = DEFAULT_BINS) -> Dict[str, Any]:
    """Everything the report needs, as plain data. Touches no filesystem."""
    _require_minimize(study)
    completed = _completed(study)
    return {
        "study_name": study.study_name,
        "counts": _state_counts(study),
        "failures": _failure_tally(study),
        "n_anchors": sum(1 for t in study.trials
                         if t.user_attrs.get(ANCHOR_ATTR)),
        "n_completed": len(completed),
        "importances": _importances(study),
        "pairwise": [pairwise_table(study, left, right, bins=bins)
                     for left, right in PRESPECIFIED_PAIRS],
        "sampler_correlations": sampler_induced_correlations(study),
        "pareto": pareto_front(study),
        "top_by_loss": _top_by_loss(study, top_k),
        "top_by_ska_delta": _top_by_ska_delta(study, top_k),
        "objective_vs_params": _objective_vs_params(study),
        "rank_curve": _rank_curve(study),
    }


# ------------------------------------------------------------------ artefacts ----

def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]],
               columns: Optional[Sequence[str]] = None) -> Path:
    """Rows as CSV, always with a header.

    A study that produced nothing still gets a header rather than a zero-byte
    file, because an empty file reads as "the analysis did not run" and a bare
    header reads as "there was nothing to report" -- which are different
    findings.
    """
    if columns is None:
        columns = sorted({k for row in rows for k in row}) or ["(no rows)"]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(columns),
                            extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(path, buffer.getvalue())
    return path


def _format_table(table: Mapping[str, Any]) -> List[str]:
    left, right = table["left"], table["right"]
    lines = [f"### {left} x {right}", ""]
    if table["degenerate"]:
        lines += [f"**{table['note']}**", ""]
        return lines
    headers = [str(level["label"]) for level in table["right_levels"]]
    lines.append(f"| {left} \\ {right} | " + " | ".join(headers) + " |")
    lines.append("|:--" + "|---:" * len(headers) + "|")
    for row in table["rows"]:
        cells = []
        for cell in row["cells"]:
            if cell["n"] == 0:
                cells.append("-")
            else:
                cells.append(f"{cell['mean']:.4f} (n={cell['n']})")
        lines.append(f"| {row['level']} | " + " | ".join(cells) + " |")
    lines.append("")
    if table["n_dropped"]:
        lines.append(f"**WARNING: {table['n_dropped']} of "
                     f"{table['n_completed']} completed trial(s) fell outside "
                     f"every cell and are NOT in this table.** That should be "
                     f"impossible -- every completed trial has a value on both "
                     f"axes and the bins tile their range. Treat the cell means "
                     f"as incomplete.")
        lines.append("")
    if table["right_binned"] or table["left_binned"]:
        binned = [name for name, flag in ((left, table["left_binned"]),
                                          (right, table["right_binned"])) if flag]
        lines.append(f"Binned: {', '.join(binned)}. Cell values are mean "
                     f"objective with the trial count.")
        lines.append("")
    return lines


def _interactions_markdown(result: Mapping[str, Any]) -> str:
    lines = [
        f"# Interaction analysis: {result['study_name']}",
        "",
        "## How to read this, and what it is not",
        "",
        _CAVEAT,
        "",
        "Three different kinds of number appear below and they are NOT "
        "interchangeable:",
        "",
        "1. **Main-effect importance** -- evidence about the model, within the "
        "region the sampler explored.",
        "2. **Pairwise response structure** -- the prespecified quantity this "
        "study was designed to estimate. Read every cell mean with its n.",
        "3. **Sampler-induced correlation** -- a diagnostic of where the sampler "
        "went. NOT evidence about the model.",
        "",
        "## Trial states",
        "",
        f"- completed: **{result['counts'].get('COMPLETE', 0)}**",
        f"- pruned: **{result['counts'].get('PRUNED', 0)}**",
        f"- failed: **{result['counts'].get('FAIL', 0)}**",
        f"- total recorded: **{result['counts']['total']}**",
        f"- of which anchors: **{result['n_anchors']}**",
        "",
    ]
    if result["failures"]:
        lines += ["### Failure reasons", "",
                  "One reason dominating is a bug in the harness; many distinct "
                  "reasons is a rough study. A count alone cannot tell them "
                  "apart.", ""]
        for reason, count in sorted(result["failures"].items(),
                                    key=lambda kv: -kv[1]):
            lines.append(f"- {count}x `{reason}`")
        lines.append("")

    lines += ["## 1. Main-effect importance", ""]
    importances = result["importances"]
    if importances["available"]:
        lines += [f"Evaluator: `{importances['evaluator']}`.", "",
                  importances["interpretation"], "",
                  "| axis | importance |", "|:--|---:|"]
        for axis, value in sorted(importances["values"].items(),
                                  key=lambda kv: -kv[1]):
            lines.append(f"| `{axis}` | {value:.4f} |")
    else:
        lines += [f"**Unavailable**: {importances['reason']}.", "",
                  "sklearn is not installed in this environment, so fANOVA and "
                  "MeanDecreaseImpurity cannot be constructed. The rest of this "
                  "report does not depend on an evaluator."]
    lines.append("")

    lines += ["## 2. Pairwise response structure (prespecified)", ""]
    for table in result["pairwise"]:
        lines += _format_table(table)

    lines += ["## 3. Sampler-induced correlation (DIAGNOSTIC, NOT EVIDENCE)", ""]
    correlations = result["sampler_correlations"]
    lines += [correlations["interpretation"], ""]
    if correlations["constant_columns"]:
        lines.append(f"Constant columns (no correlation is defined): "
                     + ", ".join(f"`{c}`" for c in correlations["constant_columns"]))
        lines.append("")
    if correlations["excluded_columns"]:
        lines.append(f"Excluded (categorical labels have no meaningful r): "
                     + ", ".join(f"`{c}`"
                                 for c in correlations["excluded_columns"]))
        lines.append("")
    if correlations["pairs"]:
        lines += ["| a | b | r | n |", "|:--|:--|---:|---:|"]
        for pair in correlations["pairs"][:15]:
            lines.append(f"| `{pair['a']}` | `{pair['b']}` | {pair['r']:+.3f} "
                         f"| {pair['n']} |")
        lines.append("")

    lines += ["## 4. Loss / parameter-count Pareto front", "",
              "Computed POST HOC, which is why the study is single-objective: "
              "optuna cannot prune a multi-objective study at all, and pruning "
              "is what makes the study affordable. This is also where parameter "
              "count is meant to be traded against loss -- a scalar "
              "`parameter_penalty` bakes one exchange rate into every "
              "subsequent proposal, unrecoverably.", ""]
    if result["pareto"]:
        lines += ["| params | objective | trial | anchor |",
                  "|---:|---:|---:|:--|"]
        for point in result["pareto"]:
            lines.append(f"| {point['param_count']:,} | "
                         f"{point['objective']:.5f} | {point['trial']} | "
                         f"{point['anchor']} |")
    else:
        lines.append("No completed trial carried a parameter count.")
    lines.append("")
    return "\n".join(lines) + "\n"


def write_analysis(study: optuna.study.Study, out_dir, *, top_k: int = 15,
                   bins: int = DEFAULT_BINS) -> Dict[str, Path]:
    """Write every artefact; returns a name -> path map.

    Idempotent and read-only: re-running overwrites rather than appending, and
    nothing here mutates the study, which matters because this may run against a
    journal other workers are still writing to.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = analyse(study, top_k=top_k, bins=bins)

    # `atomic_write_json` returns None, so the path is built here rather than
    # taken from its return value -- `f(...) or path` would work and would read
    # as though the writer sometimes returned something.
    written: Dict[str, Path] = {"summary": out_dir / "summary.json"}
    atomic_write_json(written["summary"], {
        "study_name": result["study_name"],
        "counts": result["counts"],
        "failures": result["failures"],
        "n_anchors": result["n_anchors"],
        "n_completed": result["n_completed"],
        "importances": result["importances"],
        "prespecified_pairs": [list(p) for p in PRESPECIFIED_PAIRS],
        "caveat": _CAVEAT,
    })

    written["importances"] = _write_csv(
        out_dir / "importances.csv",
        [{"axis": a, "importance": v}
         for a, v in sorted(result["importances"].get("values", {}).items(),
                            key=lambda kv: -kv[1])],
        columns=["axis", "importance"])
    written["interactions"] = out_dir / "interactions.md"
    atomic_write_text(written["interactions"], _interactions_markdown(result))
    written["pareto"] = _write_csv(
        out_dir / "pareto.csv", result["pareto"],
        columns=["param_count", "objective", "trial", "anchor", "run_id"])
    written["top_by_loss"] = _write_csv(out_dir / "top_by_loss.csv",
                                        result["top_by_loss"])
    written["top_by_ska_delta"] = _write_csv(
        out_dir / "top_by_ska_delta.csv", result["top_by_ska_delta"],
        columns=["trial", "ska_delta", "objective", "anchor", "run_id"])
    written["objective_vs_params"] = _write_csv(
        out_dir / "objective_vs_params.csv", result["objective_vs_params"],
        columns=["trial", "objective", "param_count", "anchor"])
    written["rank_curve"] = _write_csv(
        out_dir / "rank_curve.csv", result["rank_curve"],
        columns=["trial", "objective", "best_so_far", "anchor"])
    written["sampler_correlations"] = _write_csv(
        out_dir / "sampler_correlations.csv",
        result["sampler_correlations"]["pairs"],
        columns=["a", "b", "r", "n"])
    return written
