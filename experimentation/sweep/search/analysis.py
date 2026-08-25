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

So the output separates FIVE kinds of number that a single correlation matrix
would conflate, labels each one in the artefact rather than only here, and orders
them by how much of a claim about the MODEL each can support:

  ``the noise floor``
      The spread of one configuration across training seeds. Not a finding: the
      SCALE for every finding. Every magnitude below carries its size in units of
      this and a `resolved` / `unresolvable` verdict, and an effect smaller than
      it is unmeasurable by this study at any trial count. See `noise_floor`.

  ``controlled anchor contrasts``
      One factor moved from one reference point, verified after resolution so no
      second axis moved on the way through, differenced against the reference
      replicate mean. **The only causal statements in this module.** See
      `anchor_contrasts`.

  ``partial dependence and pairwise response structure``
      Mean objective per level, and per cell of a prespecified 2-way table, with
      the counts beside them. This is the quantity the study was designed to
      estimate, and the counts are what make it readable -- under adaptive
      sampling they are wildly uneven by construction, and a mean over 2 trials
      next to a mean over 40 invites reading noise as effect. DESCRIPTIVE, not
      causal: the sampler chose which trials exist, so the other axes are not
      balanced across the levels of any one of them.

  ``PED-ANOVA importance``
      A DIVERGENCE between a parameter's distribution among the study's
      top-quantile trials and its distribution in a reference set -- not a
      variance decomposition, and not evidence about the model. With optuna's
      default `evaluate_on_local=True` the reference set is the study's own
      remaining trials, i.e. an empirical distribution the SAMPLER produced, so
      the headline number is partly a description of the search path. On this
      repo's own analysis fixture a NULL axis outranks a planted main effect
      under that setting. Reported alongside its `evaluate_on_local=False`
      counterpart, and neither is called proof. See `_IMPORTANCE_NOTE`.

  ``sampler-induced correlation``
      Correlation between sampled columns. A diagnostic of the search path.
      Explicitly NOT evidence, and the artefact says so in those words.

**The pairs are prespecified.** Fourteen, fixed in `PRESPECIFIED_PAIRS` and split
into the seven this module was written with and the seven added when the study was
commissioned -- all of them before any trial of the 256-trial study ran. Nine axes
admit 36 pairs; picking which to report after looking at the data is how one of 36
comes out looking significant. Fourteen of 36 is still a real multiplicity burden
and `_MULTIPLICITY` says so in the artefact rather than leaving a reader to count
tables. The objective is also deliberately absent from the correlation table: a
parameter-vs-objective correlation reads as a main effect and is not one, for the
same adaptive-sampling reason.

**Everything is stated against a MEASURED noise floor.** `noise_floor` computes
the within-group spread of the study's designated reference replicates -- one
configuration, several training seeds -- and every magnitude in the report carries
its size in units of that spread plus a `resolved` / `unresolvable` verdict. This
is the section to read first, and the reason is that the effects being ranked are
SMALL.

The size of the effects is now measured rather than feared. Job 445689, four
trials on the proxy base at its real 600-step horizon crossing
`ska_layerscale_init`, gave losses 5.2687 / 5.2648 / 5.2933 / 5.3338 -- a total
span of 0.069, with the top two 0.004 apart -- and SKA ablation deltas of
1.17e-2 to 7.48e-2, all POSITIVE, i.e. removing SKA makes the model worse. Two
consequences:

  * The SKA branch is load-bearing at this scale. An earlier 4m-geometry smoke
    study measured an ablation delta of 1.17e-4 and raised the worry that the
    whole study was in a dead regime; at the same `ska_layerscale_init = 0.01`
    the proxy gives 1.17e-2, a hundred times larger. That was a SCALE artefact of
    the 4m geometry, not a property of the mechanism.
  * The differences the study must rank are of order 1e-2 to 1e-3 in held-out
    loss, and the top two configs in that probe differ by 4e-3. Whether an
    ordering that tight is real is a question about SEED NOISE and nothing else,
    which is exactly what `noise_floor` measures and why it is computed first and
    threaded through every magnitude below.

A report without a floor is a ranking without a scale, and `noise_floor` says so
in those words rather than returning zero.

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
from experimentation.sweep.search.study import ANCHOR_ATTR, REFERENCE_GROUP_ATTR

__all__ = ["CANNOT_ESTABLISH", "PRESPECIFIED_PAIRS", "analyse",
           "anchor_contrasts", "conditional_effects", "main_effects",
           "noise_floor", "pairwise_table", "pareto_front",
           "sampler_induced_correlations", "shortlist", "throughput_pareto",
           "write_analysis"]

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
#: Fourteen pairs in two blocks, and the split is recorded rather than smoothed
#: over. The first seven were written down when this module was first written;
#: the second seven were commissioned afterwards, still before any trial of the
#: 256-trial study had run. That is what keeps "prespecified" true rather than
#: decorative -- but it also means the reported set is 13 of the 36 pairs nine
#: axes admit, and `_MULTIPLICITY` states that in the artefact instead of leaving
#: a reader to count tables.
#:
#: Dropping the original seven to make room would have been the same error as
#: choosing pairs after looking at the data: it makes the reported set depend on
#: a later judgement. So both blocks are kept and neither is privileged.
_PAIRS_ORIGINAL: Tuple[Tuple[str, str], ...] = (
    ("ska_rank", "ska_ridge"),
    ("ska_rank", "norm_clip_multiplier"),
    ("ska_rank", "ska_power_K"),
    ("n_ska_layers", "ska_layerscale_init"),
    ("n_ska_layers", "placement"),
    ("ska_ridge", "ska_power_K"),
    ("ska_layerscale_init", "learning_rate"),
)

#: The second block. Each has a reason, in the same vocabulary as the first:
#:   rank x n_ska_layers        total SKA capacity is rank x count, so the two are
#:                              two ways of spending one budget
#:   K x gamma                  gamma scales the recurrence that K then composes,
#:                              so K amplifies whatever gamma did
#:   ridge x layerscale         conditioning the internal solve against scaling
#:                              its output: both change how loud a badly-
#:                              conditioned branch is
#:   gamma x norm_clip          the clip bounds a whitened quantity whose scale
#:                              gamma moves
#:   LR x rank / LR x layers    more capacity changes the curvature the step size
#:                              is crossing
#:   LR x K                     K composes the operator, so it multiplies the
#:                              effective depth an LR is tuned against
#:
#: `placement x n_ska_layers` and `K x ridge` were already in the first block, in
#: the other order, and are NOT repeated -- `(a, b)` and `(b, a)` are one table.
_PAIRS_ADDED: Tuple[Tuple[str, str], ...] = (
    ("ska_rank", "n_ska_layers"),
    ("ska_power_K", "gamma_value"),
    ("ska_ridge", "ska_layerscale_init"),
    ("gamma_value", "norm_clip_multiplier"),
    ("learning_rate", "ska_rank"),
    ("learning_rate", "n_ska_layers"),
    ("learning_rate", "ska_power_K"),
)

PRESPECIFIED_PAIRS: Tuple[Tuple[str, str], ...] = (
    _PAIRS_ORIGINAL + _PAIRS_ADDED)

#: Nine axes admit 9*8/2 = 36 pairs. Stated in the artefact because 14 tables is
#: a real multiple-comparisons burden and a reader should not have to count.
#: Derived from `PRESPECIFIED_PAIRS` rather than written out, because a hardcoded
#: count went stale the first time the list grew.
_N_POSSIBLE_PAIRS = 36
_MULTIPLICITY = (
    f"{len(PRESPECIFIED_PAIRS)} of {_N_POSSIBLE_PAIRS} possible pairs are reported. All {len(PRESPECIFIED_PAIRS)} were "
    f"named before any trial of this study ran -- {len(_PAIRS_ORIGINAL)} when "
    f"this module was written and {len(_PAIRS_ADDED)} when the study was "
    f"commissioned -- so none was chosen after looking at the data. That is the "
    f"protection prespecification buys and it is the only one: with 14 tables, "
    f"an uncorrected 'largest observed interaction' is still the largest of 14 "
    f"draws. Every magnitude below is therefore reported against the measured "
    f"noise floor rather than against zero.")

#: Bins for a continuous axis in a 2-way table. Three, not ten: cell counts are
#: uneven under adaptive sampling, and ten bins over 200 completed trials in a
#: 9-dimensional space leaves most cells with a handful of trials or none.
DEFAULT_BINS = 3

_CAVEAT = (
    "The sampler was ADAPTIVE. Later trials were proposed inside whatever region "
    "the sampler believed was good, so cell counts are uneven by construction "
    "and the sampled columns are correlated with each other for reasons that "
    "have nothing to do with the model. Read cell means WITH their n. These "
    f"{len(PRESPECIFIED_PAIRS)} pairs were PRESPECIFIED before any trial ran; "
    "nine axes admit 36 pairs, and choosing which to report after looking is "
    "how one of 36 comes out looking significant.")


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


# ------------------------------------------------------------ noise floor ----

#: How many sigmas of the DIFFERENCE of two single-trial measurements count as
#: resolvable. Two. Not three (a 600-step proxy screen is a filter, and a filter
#: that discards a real 2-sigma effect wastes the confirmation study it exists to
#: design), and not one (at one sigma roughly a third of pure-noise pairs would
#: be called effects, over 14 tables).
#:
#: A convention, stated as one, and NOT a significance test -- see `_CRITERION`.
_RESOLVE_SIGMAS = 2.0

_CRITERION = (
    "An effect is called RESOLVED when its magnitude exceeds "
    f"{_RESOLVE_SIGMAS:g} x sigma x sqrt(2), where sigma is the within-group "
    "standard deviation of the reference replicates. The sqrt(2) is there "
    "because almost every effect below is a DIFFERENCE of two single-trial "
    "measurements, and the difference of two independent draws of sd sigma has "
    "sd sigma*sqrt(2) -- comparing a difference against a bare sigma would "
    "over-resolve by 41%. This is a CONVENTION and not a p-value: n=5 "
    f"replicates estimate sigma to about +/-35%, the "
    f"{len(PRESPECIFIED_PAIRS)} reported tables are {len(PRESPECIFIED_PAIRS)} "
    "chances for noise to clear any fixed bar, and nothing here corrects for "
    "either. Read 'resolved' as 'larger than this study can explain by seed "
    "alone', never as 'significant'.")


def _sd(values: Sequence[float]) -> Optional[float]:
    """Sample standard deviation (ddof=1), or None below two observations.

    ddof=1 rather than 0: this is an ESTIMATE of a population spread from a small
    sample, and ddof=0 would bias the noise floor DOWN -- by 11% at n=5 -- which
    is the direction that turns noise into findings.
    """
    n = len(values)
    if n < 2:
        return None
    mean = sum(values) / n
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))


def noise_floor(study: optuna.study.Study) -> Dict[str, Any]:
    """The study's own reproducibility spread, from its reference replicates.

    **The single most important number this module produces.** Everything else
    here is a difference in held-out loss, and a difference is only a measurement
    if it is larger than what the same configuration produces twice.

    The scale of what has to be resolved is measured: job 445689, four trials on
    the proxy base at 600 steps, produced losses spanning 0.069 with the best two
    0.004 apart. So this study is being asked to rank differences of order 1e-2
    to 1e-3, and whether it can is entirely a question about how much one
    configuration moves when only its seed changes. If sigma is of that order the
    study cannot rank these axes at all -- which is a finding rather than a
    failure, but only if it is measured.

    Computed from `reference_group` members and from NOTHING else. In particular
    not from the spread of the whole study: that is a spread ACROSS
    configurations, i.e. the quantity being measured, and using it would make
    every effect trivially unresolvable.

    Pooled across groups when there is more than one, by pooling sums of squares
    rather than averaging standard deviations -- averaging sds weights a group of
    2 the same as a group of 8.

    Returns `available: False` with `sigma: None` when there are no usable
    replicates. **Never zero.** A floor of zero makes every difference resolvable,
    which is the most damaging way this could fail: the report would look complete
    and would rank noise.
    """
    _require_minimize(study)
    groups: Dict[str, List[Any]] = {}
    for trial in _completed(study):
        group = trial.user_attrs.get(REFERENCE_GROUP_ATTR)
        if group:
            groups.setdefault(str(group), []).append(trial)

    # A group with two members at ONE seed is not a replicate set, and pooling it
    # does not merely add noise -- it BIASES SIGMA DOWN. The proxy base sets
    # `deterministic: true`, so a duplicated member lands on an identical loss:
    # the pair contributes 0 to the sum of squares and 1 to the degrees of
    # freedom, so sigma shrinks toward zero and every effect becomes "resolved".
    # That is the exact failure this whole function exists to prevent, and it
    # would look like an excellent result.
    #
    # It is reachable. `study.enqueue_anchors` is a read-then-write name check
    # with no lock, so two concurrent supervisors -- a resubmission of a resumable
    # study, or a supervisor restarted while workers run -- can double-enqueue the
    # entire anchor set. `anchors.check_replicates_resolve` cannot see that: it
    # validates the design FILE, and the file is fine. Only the journal knows.
    #
    # Refused rather than deduplicated, because the two members are not
    # interchangeable copies of one datapoint from this function's point of view:
    # something upstream produced a state nobody designed, and quietly repairing
    # it here would hide that from the person who has to fix the cause.
    corrupted: Dict[str, List[int]] = {}
    for group, trials in sorted(groups.items()):
        seen: Dict[int, int] = {}
        for trial in trials:
            seed = trial.user_attrs.get("model_seed")
            if seed is None:
                # A journal written before `model_seed` existed has no seeds to
                # compare. Refusing there would delete the floor from every
                # archived study rather than protecting it.
                continue
            seen[int(seed)] = seen.get(int(seed), 0) + 1
        repeated = sorted(s for s, count in seen.items() if count > 1)
        if repeated:
            corrupted[group] = repeated
    if corrupted:
        detail = "; ".join(f"{group!r} has two or more COMPLETED trials at "
                           f"seed(s) {seeds}"
                           for group, seeds in sorted(corrupted.items()))
        return {
            "available": False,
            "reason": (
                f"a reference group is CORRUPTED: {detail}. Two members at one "
                f"seed are one datapoint counted twice, and because the base spec "
                f"sets `deterministic: true` they land on the same loss -- so "
                f"pooling them adds a zero to the sum of squares AND a degree of "
                f"freedom, which drives sigma toward zero and makes every effect "
                f"look resolved. This is the failure mode a noise floor exists to "
                f"prevent, so it is refused rather than pooled. The likeliest "
                f"cause is a double enqueue: `enqueue_anchors` keys idempotency "
                f"on the anchor name with no lock, so two concurrent supervisors "
                f"can queue the whole set twice. Check `trials.csv` for duplicate "
                f"`attr_anchor_name` values and re-run the analysis against a "
                f"journal with one trial per anchor."),
            "groups": [], "sigma": None, "dof": 0,
            "sd_of_difference": None, "min_resolvable_effect": None,
            "criterion": _CRITERION,
        }

    rows: List[Dict[str, Any]] = []
    pooled_ss, pooled_dof = 0.0, 0
    for group, trials in sorted(groups.items()):
        values = [float(t.value) for t in trials]
        mean = sum(values) / len(values)
        sd = _sd(values)
        rows.append({
            "group": group,
            "n": len(values),
            "seeds": sorted(int(t.user_attrs["model_seed"]) for t in trials
                            if t.user_attrs.get("model_seed") is not None),
            "trials": sorted(t.number for t in trials),
            "objectives": sorted(values),
            "mean": mean,
            "sd": sd,
            # `None`, not 0.0, for a single member -- for the same reason `sd` is:
            # a one-member group has no range, and 0.0 in `noise_floor.csv` reads
            # as "measured, perfectly reproducible".
            "range": (max(values) - min(values)) if len(values) >= 2 else None,
        })
        if sd is not None:
            pooled_ss += sum((v - mean) ** 2 for v in values)
            pooled_dof += len(values) - 1

    if pooled_dof < 1:
        why = ("no COMPLETED trial carries a `reference_group` attr, so this "
               "study measured no replicates and has no noise floor"
               if not groups else
               "every `reference_group` has fewer than two COMPLETED trials, so "
               "there is no spread to measure -- one observation is not a floor")
        return {
            "available": False,
            "reason": (
                f"{why}. Every effect below is therefore reported WITHOUT a "
                f"scale: a difference of 1e-4 and a difference of 1e-1 are "
                f"presented identically and neither can be called resolvable. "
                f"Add a `reference_group` of 3-5 designs differing only in "
                f"`seed` to the study's design file."),
            "groups": rows, "sigma": None, "dof": 0,
            "sd_of_difference": None, "min_resolvable_effect": None,
            "criterion": _CRITERION,
        }

    sigma = math.sqrt(pooled_ss / pooled_dof)
    sd_difference = sigma * math.sqrt(2.0)
    return {
        "available": True,
        "reason": "",
        "groups": rows,
        "sigma": sigma,
        "dof": pooled_dof,
        "sd_of_difference": sd_difference,
        "min_resolvable_effect": _RESOLVE_SIGMAS * sd_difference,
        "criterion": _CRITERION,
        "interpretation": (
            f"sigma = {sigma:.6g} is what ONE configuration's held-out loss does "
            f"when only the training seed changes, pooled over {pooled_dof} "
            f"degree(s) of freedom. Two single trials therefore have to differ by "
            f"more than {_RESOLVE_SIGMAS * sd_difference:.6g} before the "
            f"difference is anything but seed. Compare that number to the effect "
            f"sizes this study is searching for BEFORE reading any ranking "
            f"below: if the design effects are smaller, the ranking is a ranking "
            f"of seeds and no sampler can repair it."),
    }


def _threshold(floor: Optional[Mapping[str, Any]]) -> Optional[float]:
    if not floor or not floor.get("available"):
        return None
    return floor.get("min_resolvable_effect")


def _resolves(magnitude: Optional[float],
              threshold: Optional[float]) -> Optional[bool]:
    """True / False / None-for-unknowable. Three states, on purpose.

    `None` is not `False`. "This study has no noise floor, so the question cannot
    be asked" and "the effect is smaller than the noise" are different findings,
    and collapsing them would let a study with no replicates report every effect
    as unresolvable -- which reads as a measurement.
    """
    if threshold is None or magnitude is None:
        return None
    return bool(abs(magnitude) > threshold)


# -------------------------------------------------- controlled contrasts ----

def anchor_contrasts(study: optuna.study.Study,
                     floor: Optional[Mapping[str, Any]] = None
                     ) -> List[Dict[str, Any]]:
    """Each anchor's held-out loss minus the reference REPLICATE MEAN.

    The one part of this analysis that is a controlled experiment rather than an
    observational summary: the 24 screen anchors are one-factor-at-a-time moves
    from a single reference point, so each difference is attributable to the one
    factor its name claims -- which is exactly what
    `test_proxy_anchor_design.py::test_each_anchor_moves_only_its_named_factor`
    checks, after resolution.

    Against the replicate MEAN and not against `reference-k1` alone. The reference
    is estimated five times; using one of the five throws away four and picks
    whichever seed happened to be listed first. It also carries the reference's
    own standard error, so a reader can see that the comparison is between a
    single trial and a five-trial mean rather than between two single trials.

    Replicate members are excluded from the rows: an anchor's contrast against
    the mean of a set it belongs to is not a contrast.
    """
    _require_minimize(study)
    floor = floor if floor is not None else noise_floor(study)

    # The FIRST group, and `noise_floor` guarantees there is at most one usable
    # one worth contrasting against: a study with several replicate sets has
    # several reference points and "the" contrast is then ambiguous. Recorded as a
    # limitation rather than guessed at -- if a second group is ever added, this
    # has to become a per-group contrast rather than silently keep using the
    # alphabetically-first.
    reference = next((g for g in floor.get("groups") or []), None)
    reference_values = list(reference["objectives"]) if reference else []
    reference_mean = (sum(reference_values) / len(reference_values)
                      if reference_values else None)
    reference_sd = _sd(reference_values)
    reference_sem = (reference_sd / math.sqrt(len(reference_values))
                     if reference_sd is not None else None)

    # THE RIGHT DENOMINATOR, and it is not the one `min_resolvable_effect` uses.
    #
    # Every OTHER comparison in this module is a difference of two SINGLE trials,
    # whose sd is sigma*sqrt(2). A contrast is a single trial against the MEAN of
    # n replicates, whose sd is sigma*sqrt(1 + 1/n) -- 1.095*sigma at n=5, not
    # 1.414*sigma. Reusing the trial-vs-trial threshold here makes the bar 29% too
    # high and understates every `sigmas` figure by the same factor, so a real
    # 1.6-sigma anchor effect prints as 1.2 and is labelled unresolvable.
    #
    # That error is conservative -- it cannot manufacture a finding -- and it
    # discards real ones in the one table this module calls the only causal
    # statements it has, which is the worst place to be quietly conservative.
    contrast_sd = None
    if floor.get("sigma") is not None and reference_values:
        contrast_sd = float(floor["sigma"]) * math.sqrt(
            1.0 + 1.0 / len(reference_values))
    contrast_threshold = (None if contrast_sd is None
                          else _RESOLVE_SIGMAS * contrast_sd)

    rows: List[Dict[str, Any]] = []
    for trial in _completed(study):
        name = trial.user_attrs.get(ANCHOR_ATTR)
        if not name or trial.user_attrs.get(REFERENCE_GROUP_ATTR):
            continue
        delta = (float(trial.value) - reference_mean
                 if reference_mean is not None else None)
        rows.append({
            "anchor": name,
            "trial": trial.number,
            "objective": float(trial.value),
            "reference_mean": reference_mean,
            "reference_n": len(reference_values),
            "reference_sem": reference_sem,
            "delta": delta,
            "abs_delta": abs(delta) if delta is not None else None,
            "sd_of_contrast": contrast_sd,
            "threshold": contrast_threshold,
            "sigmas": (abs(delta) / contrast_sd
                       if delta is not None and contrast_sd else None),
            "resolved": _resolves(delta, contrast_threshold),
        })
    rows.sort(key=lambda r: -(r["abs_delta"] or 0.0))
    return rows


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


# -------------------------------- partial dependence and main effects ----

_MAIN_EFFECT_NOTE = (
    "Each row is a PARTIAL DEPENDENCE table: the mean objective of every "
    "completed trial at each level of one axis, marginalising over the others by "
    "averaging rather than by holding them fixed. `spread` is max-min of those "
    "level means and `variance_share` is the count-weighted between-level "
    "variance divided by the total variance of the objective.\n\n"
    "This is DESCRIPTIVE, not causal, and the reason is the sampler. It chose "
    "which trials exist, so the other axes are not balanced across the levels of "
    "this one -- a level the sampler visited mostly alongside a good ridge looks "
    "good. It also concentrated its later proposals, so an axis it pinned near "
    "one value shows a small spread whether or not it matters; check that axis's "
    "own spread in trials.csv before concluding it is unimportant. The "
    "one-factor ANCHOR CONTRASTS in section 0b are the controlled version of this "
    "question and should be read first; this table is what extends it to the "
    "sampled body, at the cost of the control."
)


def main_effects(study: optuna.study.Study,
                 floor: Optional[Mapping[str, Any]] = None, *,
                 bins: int = DEFAULT_BINS) -> Dict[str, Any]:
    """Per-axis partial dependence, with each spread stated against the floor.

    `variance_share` is the first-order fANOVA term computed directly: the
    count-weighted variance of the level means over the total variance. That IS
    what fANOVA's main-effect component estimates, on a discretised grid -- and
    computing it here rather than through optuna's `FanovaImportanceEvaluator`
    is not a preference: that evaluator imports sklearn, which is not installed
    (measured, see the module docstring). The arithmetic is a dozen lines and
    needs no random forest.

    Its limitation is stated in `_MAIN_EFFECT_NOTE` rather than in a footnote: the
    axes are not balanced across each other, because the sampler chose the
    trials.
    """
    _require_minimize(study)
    floor = floor if floor is not None else noise_floor(study)
    threshold = _threshold(floor)
    trials = _completed(study)
    values = [float(t.value) for t in trials]
    total_variance = 0.0
    if len(values) >= 2:
        grand = sum(values) / len(values)
        total_variance = sum((v - grand) ** 2 for v in values) / len(values)

    axes: List[Dict[str, Any]] = []
    for axis in sorted({k for t in trials for k in t.params}):
        observed = [t.params.get(axis) for t in trials]
        levels, binned = _levels(observed, bins=bins,
                                 continuous=_is_continuous(study, axis))
        cells: List[Dict[str, Any]] = []
        for index, level in enumerate(levels):
            picked = [float(t.value) for t, v in zip(trials, observed)
                      if _bucket(v, levels, binned) == index]
            cells.append({
                "label": str(level["label"]),
                "n": len(picked),
                "mean": (sum(picked) / len(picked)) if picked else None,
                "best": min(picked) if picked else None,
            })
        populated = [c for c in cells if c["mean"] is not None]
        means = [c["mean"] for c in populated]
        # `None`, not 0.0, when the axis took fewer than two levels. A fixed axis
        # was never MEASURED, and `spread = 0.0` + `unresolvable` reads as
        # "measured, and smaller than the noise floor" -- a different and much
        # stronger statement. Every other absent path in this module returns None
        # for exactly this reason; this one did not, and `_resolved_word` printed
        # `unresolvable` for the three singleton axes on every run.
        spread = (max(means) - min(means)) if len(means) >= 2 else None
        share = 0.0
        if total_variance > 0 and populated:
            weight_total = sum(c["n"] for c in populated)
            weighted = sum(c["n"] * c["mean"] for c in populated) / weight_total
            between = sum(c["n"] * (c["mean"] - weighted) ** 2
                          for c in populated) / weight_total
            share = min(1.0, between / total_variance)
        axes.append({
            "axis": axis,
            "binned": binned,
            "levels": cells,
            "spread": spread,
            "variance_share": share,
            # `_resolves` already returns None when either argument is None, so a
            # never-varied axis comes back None rather than False -- which is what
            # `_resolved_word` renders as DID NOT VARY.
            "resolved": _resolves(spread, threshold),
            "varied": spread is not None,
        })
    axes.sort(key=lambda a: -a["variance_share"])
    return {"axes": axes, "total_variance": total_variance,
            "n_completed": len(trials),
            "interpretation": _MAIN_EFFECT_NOTE}


# ----------------------------------------------------- conditional effects ----

def conditional_effects(study: optuna.study.Study,
                        floor: Optional[Mapping[str, Any]] = None, *,
                        bins: int = DEFAULT_BINS) -> List[Dict[str, Any]]:
    """For each prespecified pair: the effect of `left` WITHIN each level of
    `right`, and how much that effect changes across those levels.

    That last number -- `interaction_magnitude` -- is the quantity the study was
    launched to estimate, stated as one number per pair so 14 tables can be
    ranked. It is `max - min` over levels of `right` of (`max - min` of the
    `left`-conditional means). An interaction IS "the effect of A depends on B",
    so a pair whose conditional effects are all the same size has no interaction
    however large its main effects are.

    Compared against the noise floor, and that comparison is doing more work here
    than anywhere else in this module: an interaction magnitude is a difference OF
    differences, so it accumulates noise from four cell means rather than two. The
    threshold applied is still the two-single-trials one, which makes this the
    most OPTIMISTIC test in the file -- a pair that fails it is certainly
    unresolvable, while a pair that passes it by a small margin should not be
    trusted without the cell counts beside it.
    """
    _require_minimize(study)
    floor = floor if floor is not None else noise_floor(study)
    threshold = _threshold(floor)

    rows: List[Dict[str, Any]] = []
    for left, right in PRESPECIFIED_PAIRS:
        table = pairwise_table(study, left, right, bins=bins)
        conditionals = []
        for column, level in enumerate(table["right_levels"]):
            cells = [row["cells"][column] for row in table["rows"]]
            populated = [(index, cell["mean"])
                         for index, cell in enumerate(cells)
                         if cell["mean"] is not None]
            means = [mean for _index, mean in populated]
            # The SIGNED contrast: the objective at the LAST populated level of
            # `left` minus the objective at the FIRST. `_levels` returns levels in
            # a stable sorted order, so "first" and "last" mean the same thing at
            # every level of `right` -- which is what makes the signs comparable
            # across rows and therefore what makes a crossover detectable.
            contrast = (means[-1] - means[0]) if len(means) >= 2 else None
            conditionals.append({
                "level": str(level["label"]),
                "n": sum(cell["n"] for cell in cells),
                "n_cells": len(means),
                # How much `left` MOVES the objective at this level of `right`,
                # unsigned. Answers "does the effect change size".
                "effect": (max(means) - min(means)) if len(means) >= 2 else None,
                # ...and which DIRECTION it moves it. Answers "does the effect
                # change direction", which the unsigned range cannot.
                "contrast": contrast,
            })

        effects = [c["effect"] for c in conditionals if c["effect"] is not None]
        contrasts = [c["contrast"] for c in conditionals
                     if c["contrast"] is not None]
        magnitude = (max(effects) - min(effects)) if len(effects) >= 2 else None
        # THE FIX. `interaction_magnitude` is built from UNSIGNED ranges, so a
        # pure crossover -- `left` better at one level of `right` and worse at
        # another, by the same amount -- produces the SAME range at every level
        # and a magnitude of exactly 0.0. That is the maximal interaction there
        # is, reported as none at all, and it is the shape this study most exists
        # to find: `_SLOTS["best_k1"]` says K is the axis most likely to change
        # the SIGN of another axis's effect. Unfixed, the report would state "no
        # pair of these axes interacts by more than one configuration varies
        # against its own seed" -- a confident false negative.
        #
        # `signed_magnitude` is `max - min` over the SIGNED contrasts. It reduces
        # to the unsigned magnitude when every contrast has the same sign, and is
        # maximal (twice the effect size) under a clean crossover.
        signed = (max(contrasts) - min(contrasts)) if len(contrasts) >= 2 else None
        # A sign change is exactly "the contrasts do not all point the same way",
        # and it is worth a flag rather than left to be inferred from two columns:
        # it is the difference between "this axis matters more over here" and
        # "this axis wants the opposite thing over here", and only the second
        # invalidates a single recommended value.
        sign_change = (len(contrasts) >= 2
                       and min(contrasts) < 0.0 < max(contrasts))
        # The verdict reads the LARGER of the two. Using the unsigned one alone
        # would leave the crossover labelled unresolvable and the fix would buy
        # nothing -- `shortlist`'s interaction probe reads this field.
        best = max((m for m in (magnitude, signed) if m is not None),
                   default=None)
        thin = [c["level"] for c in conditionals if c["n_cells"] < 2]
        rows.append({
            "left": left, "right": right,
            "conditionals": conditionals,
            "interaction_magnitude": magnitude,
            "signed_magnitude": signed,
            "sign_change": sign_change,
            "sigmas": (best / floor["sd_of_difference"]
                       if best is not None and floor.get("sd_of_difference")
                       else None),
            "resolved": _resolves(best, threshold),
            "degenerate": table["degenerate"],
            "thin_levels": thin,
            "n_completed": table["n_completed"],
        })
    rows.sort(key=lambda r: -max(
        (m for m in (r["interaction_magnitude"], r["signed_magnitude"])
         if m is not None), default=-1.0))
    return rows


# ------------------------------------------------------- main-effect importance ----

def _importance_evaluator(*, evaluate_on_local: bool = True):
    """The best importance evaluator this environment can actually construct.

    A function so it can be monkeypatched in a test, and so the ImportError from
    a missing sklearn is raised HERE rather than from inside `analyse`.

    PedAnova first, and in this environment only: fANOVA and
    MeanDecreaseImpurity both import sklearn, which is not installed. PedAnova is
    pure numpy. It is flagged experimental by optuna and the warning is silenced
    at this one construction site, the same way `study.make_sampler` treats
    `constant_liar` and `multivariate`.

    `evaluate_on_local` is the parameter the framing turns on, so it is exposed
    rather than left at its default -- see `_importances`.
    """
    from optuna.importance import PedAnovaImportanceEvaluator

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",
                                category=optuna.exceptions.ExperimentalWarning)
        return (PedAnovaImportanceEvaluator(evaluate_on_local=evaluate_on_local),
                f"PedAnovaImportanceEvaluator(evaluate_on_local="
                f"{evaluate_on_local})")


#: What PED-ANOVA actually computes, said plainly, because the previous label
#: here overclaimed and a review caught it.
#:
#: The old text read "This IS evidence about the model". It is not, or at least
#: not in the way that phrasing implies. PED-ANOVA measures a divergence between
#: the distribution of a parameter among the study's TOP-quantile trials and its
#: distribution among some reference set. With optuna's default
#: `evaluate_on_local=True` that reference set is the study's own remaining
#: trials -- i.e. an empirical distribution the SAMPLER produced. Under adaptive
#: sampling that distribution is largely a description of where the search went,
#: so the headline importance is partly a function of the search path. On the
#: test fixture in `test_interaction_analysis.py` a null axis outranked a planted
#: main effect, which is the concrete demonstration.
#:
#: The fix is not to delete the number -- it is genuinely informative about which
#: axes the sampler's own belief distinguishes -- but to report it beside its
#: `evaluate_on_local=False` counterpart, which measures against the DECLARED
#: prior instead of the observed one, and to stop calling either proof. The
#: controlled anchor contrasts and the partial-dependence table are where a
#: claim about the MODEL should come from.
_IMPORTANCE_NOTE = (
    "WHAT THIS MEASURES. PED-ANOVA scores each axis by the DIVERGENCE between "
    "its distribution among the study's top-quantile trials and its distribution "
    "in a reference set. Two reference sets are reported and they answer "
    "different questions:\n"
    "  * `local` (optuna's default, `evaluate_on_local=True`) compares against "
    "the study's OWN remaining trials. Because the sampler was adaptive, that "
    "empirical distribution is largely a description of the search path -- so a "
    "high local score can mean 'this axis separates good from bad' or 'the "
    "sampler moved this axis a lot late on', and the number cannot tell them "
    "apart. On this repo's own analysis test fixture a NULL axis outranked a "
    "planted main effect under this setting.\n"
    "  * `global` (`evaluate_on_local=False`) compares against the DECLARED "
    "prior instead, so it is not a function of where the sampler went -- at the "
    "cost of being a comparison against a region the study may barely have "
    "sampled.\n\n"
    "Neither is proof about the model. Treat a large gap between the two columns "
    "as a warning that the axis's score is search-path dependent. For claims "
    "about the MODEL, read the controlled anchor contrasts first (one factor "
    "moved from one reference point, with the noise floor beside it) and the "
    "partial-dependence table second. An axis with a low score here may be "
    "unimportant, or may simply have been pinned near one value by the sampler "
    "after the startup window; its spread in trials.csv is what distinguishes "
    "those.")


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

    # The second reference distribution. Best-effort and separate: if only this
    # one fails the local column is still worth having, and the artefact says
    # which columns it got rather than pretending to both.
    global_values: Dict[str, float] = {}
    global_reason = ""
    try:
        other, _ = _importance_evaluator(evaluate_on_local=False)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=optuna.exceptions.ExperimentalWarning)
            global_values = {k: float(v) for k, v in
                             optuna.importance.get_param_importances(
                                 study, evaluator=other).items()}
    except Exception as exc:                          # noqa: BLE001
        global_reason = f"{type(exc).__name__}: {exc}"

    return {
        "available": True,
        "evaluator": name,
        "values": {k: float(v) for k, v in values.items()},
        "values_global": global_values,
        "global_reason": global_reason,
        "interpretation": _IMPORTANCE_NOTE,
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


#: Why a worker's first trial is kept off the throughput front, with the number.
_WARMUP_REASON = (
    "A worker process's FIRST trial (`attr_worker_trial_ordinal == 0`) is "
    "excluded from the throughput front. Measured, job 445689: four trials on "
    "the proxy base at 600 steps reported 17,818 / 111,416 / 104,760 / 112,685 "
    "tok/s, and the 6x outlier was trial 0. Those four configs differ only in "
    "`ska_layerscale_init` -- one scalar multiply on a gate -- so nothing "
    "architectural can produce a 6x throughput gap. It is first-trial CUDA "
    "context creation and kernel autotuning, and `tokens_per_sec` is measured "
    "during the eval pass, early enough in the process's life to still be inside "
    "that. At `concurrent_trials: 8` this is EIGHT poisoned points, not one, "
    "because every worker process pays its own warmup. The trials are NOT "
    "dropped from anything else -- their loss is a real measurement and only "
    "their throughput is suspect -- and `throughput_pareto(exclude_warmup=False)` "
    "returns them for anyone who wants to check the claim or who is on hardware "
    "where it does not hold. A journal written before this column existed has no "
    "ordinals and nothing is excluded, rather than everything.")


def throughput_pareto(study: optuna.study.Study, *,
                      exclude_warmup: bool = True) -> List[Dict[str, Any]]:
    """Trials not dominated on (held-out loss, tokens/sec). Fastest first.

    The other half of the cost question, and until `tokens_per_sec` was promoted
    to a trial attr it could not be computed at all: the number was measured in
    every `quick_eval.json` and reached neither `user_attrs` nor `trials.csv`.

    Distinct from the parameter-count front and not a substitute for it. Parameter
    count is what a 50m confirmation study inherits; throughput is what this
    geometry costs on this hardware at this microbatch, and a trial that descended
    the OOM ladder was measured at a smaller microbatch than its neighbours -- so
    read this front WITH `attr_per_device_batch_size`. That caveat is why the
    exchange rate stays the reader's rather than becoming a `throughput_penalty`.

    Two exclusions, both because a wrong number here looks exactly like a
    measurement:

      * **No recorded throughput.** Excluded, never zero-filled -- 0 tokens/sec
        would sit at the fast-loss end of nothing and the slow end of everything.
      * **A worker's first trial.** See `_WARMUP_REASON`; it is a measured 6x
        artefact of process warmup, and at 8 workers it is 8 points.
    """
    _require_minimize(study)
    points = []
    for trial in _completed(study):
        speed = trial.user_attrs.get("tokens_per_sec")
        if speed is None or float(speed) <= 0:
            continue
        ordinal = trial.user_attrs.get("worker_trial_ordinal")
        # `is not None` and not truthiness: ordinal 0 is the value being tested
        # for. And a MISSING ordinal excludes nothing -- an archived journal
        # written before this column existed would otherwise lose every trial
        # from its own front, which is a worse failure than the artefact.
        if exclude_warmup and ordinal is not None and int(ordinal) == 0:
            continue
        points.append({
            "trial": trial.number,
            "objective": float(trial.value),
            "tokens_per_sec": float(speed),
            "peak_memory_gib": trial.user_attrs.get("peak_memory_gib"),
            "per_device_batch_size": trial.user_attrs.get(
                "per_device_batch_size"),
            "param_count": trial.user_attrs.get("param_count"),
            "worker_trial_ordinal": ordinal,
            "anchor": trial.user_attrs.get(ANCHOR_ATTR) or "",
            "run_id": trial.user_attrs.get("run_id") or "",
        })
    # Fastest first, then best loss, so a point joins the front only by beating
    # every faster point's loss -- the same sweep the parameter front uses, with
    # the cost axis reversed because more tokens/sec is better.
    points.sort(key=lambda p: (-p["tokens_per_sec"], p["objective"]))
    front: List[Dict[str, Any]] = []
    best = math.inf
    for point in points:
        if point["objective"] < best:
            front.append(point)
            best = point["objective"]
    return front


def _throughput_notes(study: optuna.study.Study) -> Dict[str, Any]:
    """How many trials the throughput front dropped, and why.

    Reported rather than silent. A front whose point count nobody can reconcile
    with the trial count invites the reader to assume a bug somewhere else, and
    is a worse artefact than the outlier it removes.
    """
    completed = _completed(study)
    measured = [t for t in completed
                if t.user_attrs.get("tokens_per_sec") not in (None, 0, 0.0)]
    warmup = [t.number for t in measured
              if t.user_attrs.get("worker_trial_ordinal") is not None
              and int(t.user_attrs["worker_trial_ordinal"]) == 0]
    return {
        "n_completed": len(completed),
        "n_with_throughput": len(measured),
        "n_missing_throughput": len(completed) - len(measured),
        "n_excluded_warmup": len(warmup),
        "excluded_warmup_trials": sorted(warmup),
        "reason": _WARMUP_REASON,
    }


# ------------------------------------------------------------- shortlist ----

#: What a 600-step, 25M-parameter, single-seed-per-config proxy screen cannot
#: establish, however clean its arithmetic. Written into the artefact rather than
#: left to a reader's judgement, because the failure mode is a reader taking the
#: shortlist's first row as a result.
CANNOT_ESTABLISH: Tuple[str, ...] = (
    "**Which configuration is best.** 600 steps at 25.35M parameters is a "
    "SCREEN. Rankings at 600 steps and at convergence differ systematically, not "
    "randomly: a config that warms up fast is rewarded here whether or not it "
    "ends better, which is why `prune_after_step` is 450 of 600 rather than "
    "something cheaper. The output is a shortlist for confirmation and there is "
    "no defensible single winner in it.",
    "**That anything here transfers to 50m.** The proxy keeps 50m's depth (17) "
    "and SKA placement ([3,7,11,15]) and narrows d_model from 384 to 256, so "
    "every effect measured here is measured at 2/3 the width. Effects that "
    "interact with width -- which includes the norm-clip multiplier and the "
    "layerscale, both of which act on per-head quantities -- can change sign "
    "between the two. `d_state` is 48 here against 50m's 64, which is a SECOND "
    "unresolved difference and is flagged as an open question rather than "
    "quietly treated as immaterial.",
    "**How much SKA earns its place, beyond that it does.** This one is PARTLY "
    "settled and the correction is worth recording. An earlier 4m-geometry smoke "
    "study measured an ablation delta of 1.17e-4 and raised the worry that this "
    "study might be measuring SKA's hyperparameters without establishing that the "
    "branch does anything. Job 445689 settles the sign: on THIS base at 600 "
    "steps the delta is 1.17e-2 to 7.48e-2 depending on the layerscale, a "
    "hundred times larger, and positive throughout -- removing SKA makes the "
    "model worse. The 4m number was a scale artefact. What remains unestablished "
    "is the MAGNITUDE at any scale that matters: 25.35M parameters and 600 steps "
    "is not 50m and it is not convergence, and an ablation delta measured on a "
    "model that has barely left its warmup transient is not the delta at "
    "convergence. Compare the ska_delta column to the noise floor directly, and "
    "do not read its size as a claim about a trained model.",
    "**Any effect smaller than the measured noise floor.** Not 'probably not "
    "real' -- unmeasurable by this study, at any trial count, because it is "
    "inside what one configuration does to itself when the seed changes. More "
    "trials narrow the sampler's search, not this.",
    "**Non-determinism, or seed sensitivity at other configurations.** The base "
    "spec sets `deterministic: true` and the replicates are all at ONE "
    "configuration. So the floor is a deterministic-mode floor at the reference "
    "point; a non-deterministic run spreads more, and a badly-conditioned corner "
    "of the space (rank 8 with a light ridge, say) plausibly spreads more still. "
    "Applying one floor to the whole space is an assumption, and it is the "
    "optimistic one.",
    "**Statistical significance of anything.** Fourteen prespecified pairwise "
    "tables, nine axes, one seed per sampled trial, and cell counts that are "
    "uneven BY CONSTRUCTION because the sampler was adaptive. 'Resolved' here "
    "means 'larger than seed noise', with no multiplicity correction and no "
    "model of the sampling process. It is a filter for what to confirm, not a "
    "test.",
    "**Interactions among the fixed axes, or between them and anything.** "
    "`weight_decay`, `warmup_ratio` and `grad_clip` are singletons in this "
    "study, so their columns are constant and every table over them is "
    "degenerate by design, not by accident.",
)

#: The shortlist slots, and why each one exists. A list rather than a winner is
#: the whole deliverable -- see `shortlist`.
_SLOTS = (
    ("best_loss",
     "Lowest held-out loss. The obvious candidate and the most likely to be "
     "luck: it is the minimum of ~230 draws, so it carries the largest "
     "selection bias of any row here. Confirm it, do not believe it."),
    ("best_cost_adjusted",
     "Best loss among trials on the loss/throughput Pareto front, i.e. the best "
     "config that is not also the slowest. Kept separate from best_loss because "
     "the objective is pure loss on purpose -- a `throughput_penalty` would bake "
     "one exchange rate into every later proposal, unrecoverably."),
    ("best_k1",
     "Best config at ska_power_K=1. K is the axis most likely to change the SIGN "
     "of another axis's effect, so a confirmation study needs the best candidate "
     "at EACH K rather than whichever K happened to win here."),
    ("best_k2",
     "Best config at ska_power_K=2, for the same reason."),
    ("best_low_capacity",
     "Best config in the bottom half of the parameter range. If it is within the "
     "noise floor of the best overall, the capacity axis did not earn its "
     "parameters and the confirmation study should be cheaper."),
    ("interaction_probe",
     "A config chosen to TEST the largest resolved interaction rather than to "
     "win: it sits at the corner that interaction predicts is good, which is the "
     "only way a confirmation run can falsify it. If the interaction is an "
     "artefact of adaptive sampling, this is the row that says so."),
)


def _row(trial, extra: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    attrs = trial.user_attrs
    return {
        "trial": trial.number,
        "objective": float(trial.value),
        "param_count": attrs.get("param_count"),
        "tokens_per_sec": attrs.get("tokens_per_sec"),
        "peak_memory_gib": attrs.get("peak_memory_gib"),
        "ska_delta": attrs.get("ska_delta"),
        "anchor": attrs.get(ANCHOR_ATTR) or "",
        "run_id": attrs.get("run_id") or "",
        "model_seed": attrs.get("model_seed"),
        "ska_layer_indices": attrs.get("ska_layer_indices"),
        **{f"param_{k}": v for k, v in trial.params.items()},
        **(extra or {}),
    }


def shortlist(study: optuna.study.Study,
              floor: Optional[Mapping[str, Any]] = None
              ) -> Dict[str, Any]:
    """Six candidates for confirmation at larger scale. Never a single winner.

    **Why a list and not a winner.** The best-loss trial is the minimum of a few
    hundred draws whose spread is partly seed noise, so its lead over the
    runner-up is biased upward by selection -- and this is a 600-step screen at
    2/3 the width of the geometry anyone cares about, so even an unbiased ranking
    here is a hypothesis about 50m rather than a result at it. A shortlist spends
    a confirmation budget on the axes of DISAGREEMENT (which K, how much capacity,
    how much throughput) instead of on one row that happened to be first.

    Every entry carries `resolved_vs_reference`: whether its lead over the
    reference replicate mean exceeds the noise floor. A candidate that fails that
    is a candidate by luck, and saying so is the difference between a shortlist
    and a leaderboard.
    """
    _require_minimize(study)
    floor = floor if floor is not None else noise_floor(study)
    threshold = _threshold(floor)
    reference = next((g for g in floor.get("groups") or []), None)
    reference_mean = reference["mean"] if reference else None

    completed = _completed(study)
    entries: List[Dict[str, Any]] = []

    def add(slot: str, rationale: str, trial, *, empty_note: str = "",
            note: str = "", extra: Optional[Mapping[str, Any]] = None) -> None:
        """Fill one slot. `empty_note` explains an UNFILLABLE slot; `note` annotates
        a filled one.

        Two parameters rather than one because a single `note` printed both, so a
        filled `best_k1` row carried "no completed trial ran at ska_power_K=1"
        directly under the trial it had found -- a line that contradicts the four
        above it, which is the fastest way to teach a reader to stop reading.
        """
        if trial is None:
            entries.append({"slot": slot, "rationale": rationale, "trial": None,
                            "note": (empty_note or note
                                     or "no completed trial qualifies")})
            return
        lead = (reference_mean - float(trial.value)
                if reference_mean is not None else None)
        entries.append({
            "slot": slot, "rationale": rationale, "note": note,
            "lead_over_reference": lead,
            "resolved_vs_reference": _resolves(lead, threshold),
            **_row(trial, extra),
        })

    def best(pool) -> Optional[Any]:
        pool = list(pool)
        return min(pool, key=lambda t: float(t.value)) if pool else None

    add("best_loss", _dict(_SLOTS)["best_loss"], best(completed))

    front = {p["trial"] for p in throughput_pareto(study)}
    add("best_cost_adjusted", _dict(_SLOTS)["best_cost_adjusted"],
        best(t for t in completed if t.number in front),
        empty_note=("no trial carries a throughput measurement, so this slot "
                    "cannot be filled -- check that quick_eval wrote "
                    "metrics.full.tokens_per_sec and that "
                    "driver._stamp_measured_metrics ran"))

    for slot, k in (("best_k1", 1), ("best_k2", 2)):
        add(slot, _dict(_SLOTS)[slot],
            best(t for t in completed if t.params.get("ska_power_K") == k),
            empty_note=f"no completed trial ran at ska_power_K={k}")

    # `is not None`, not truthiness -- the discipline `pareto_front` documents.
    # A param_count of 0 is absurd and dropping it as though absent is a
    # different statement from dropping it as impossible.
    counts = [t.user_attrs["param_count"] for t in completed
              if t.user_attrs.get("param_count") is not None]
    if counts:
        midpoint = (min(counts) + max(counts)) / 2.0
        add("best_low_capacity", _dict(_SLOTS)["best_low_capacity"],
            best(t for t in completed
                 if t.user_attrs.get("param_count") is not None
                 and t.user_attrs["param_count"] <= midpoint),
            extra={"capacity_cutoff": midpoint})
    else:
        add("best_low_capacity", _dict(_SLOTS)["best_low_capacity"], None,
            empty_note="no completed trial carries a parameter count")

    interactions = [r for r in conditional_effects(study, floor)
                    if r["resolved"] and r["interaction_magnitude"] is not None]
    probe, probe_note = None, (
        "no prespecified interaction is resolvable against the noise floor, so "
        "there is nothing to probe. That is itself the study's answer to its own "
        "question: at 600 steps on this proxy, no pair of these axes interacts by "
        "more than one configuration varies against its own seed.")
    if interactions:
        top = interactions[0]
        # The best trial inside the pair's best-performing cell. Not a
        # synthesised config: a real trial is reproducible and already has a
        # run_id, and a synthesised one would have to be justified by the very
        # surface it is meant to test.
        table = pairwise_table(study, top["left"], top["right"])
        cells = [(row_index, column, cell)
                 for row_index, row in enumerate(table["rows"])
                 for column, cell in enumerate(row["cells"])
                 if cell["mean"] is not None and cell["n"] >= 2]
        if cells:
            row_index, column, _ = min(cells, key=lambda c: c[2]["mean"])
            left_levels, left_binned = _levels(
                [t.params.get(top["left"]) for t in completed],
                bins=DEFAULT_BINS,
                continuous=_is_continuous(study, top["left"]))
            right_levels, right_binned = _levels(
                [t.params.get(top["right"]) for t in completed],
                bins=DEFAULT_BINS,
                continuous=_is_continuous(study, top["right"]))
            pool = [t for t in completed
                    if _bucket(t.params.get(top["left"]), left_levels,
                               left_binned) == row_index
                    and _bucket(t.params.get(top["right"]), right_levels,
                                right_binned) == column]
            probe = best(pool)
            probe_note = (f"tests {top['left']} x {top['right']} "
                          f"(magnitude {top['interaction_magnitude']:.5g}, "
                          f"{top['sigmas']:.1f} sigma of a single-trial "
                          f"difference) at its best-performing cell "
                          f"{table['rows'][row_index]['level']} x "
                          f"{table['right_levels'][column]['label']}")
    # `probe_note` describes the interaction being probed when there IS one and
    # explains the absence when there is not, so it goes to whichever parameter
    # applies -- the note is informative in both cases, unlike the others above.
    add("interaction_probe", _dict(_SLOTS)["interaction_probe"], probe,
        note=probe_note if probe is not None else "",
        empty_note=probe_note)

    return {
        "entries": entries,
        "noise_floor": {
            "available": bool(floor.get("available")),
            "sigma": floor.get("sigma"),
            "min_resolvable_effect": floor.get("min_resolvable_effect"),
        },
        "cannot_establish": list(CANNOT_ESTABLISH),
        "interpretation": (
            "A SHORTLIST FOR CONFIRMATION, not a ranking and not a result. Each "
            "row answers a different question, and rows whose "
            "`resolved_vs_reference` is false lead the reference by less than "
            "this study can measure -- they are on the list because their SLOT "
            "matters, not because they won. Read `cannot_establish` before "
            "using any of them."),
    }


def _dict(pairs) -> Dict[str, str]:
    return {name: text for name, text in pairs}


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
    # Computed FIRST and threaded through everything that reports a magnitude,
    # rather than recomputed per section: one floor per analysis, so two sections
    # cannot disagree about what is resolvable.
    floor = noise_floor(study)
    return {
        "study_name": study.study_name,
        "counts": _state_counts(study),
        "failures": _failure_tally(study),
        "n_anchors": sum(1 for t in study.trials
                         if t.user_attrs.get(ANCHOR_ATTR)),
        "n_completed": len(completed),
        "noise_floor": floor,
        "anchor_contrasts": anchor_contrasts(study, floor),
        "main_effects": main_effects(study, floor, bins=bins),
        "conditional_effects": conditional_effects(study, floor, bins=bins),
        "importances": _importances(study),
        "pairwise": [pairwise_table(study, left, right, bins=bins)
                     for left, right in PRESPECIFIED_PAIRS],
        "sampler_correlations": sampler_induced_correlations(study),
        "pareto": pareto_front(study),
        "throughput_pareto": throughput_pareto(study),
        "throughput_notes": _throughput_notes(study),
        "shortlist": shortlist(study, floor),
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


def _fmt(value, spec=".5g", dash="-"):
    if value is None:
        return dash
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        return format(value, spec)
    return str(value)


def _resolved_word(flag, *, varied: bool = True):
    """The verdict, with THREE non-verdicts kept apart.

    `unresolvable` (measured, smaller than the floor), `DID NOT VARY` (never
    measured -- a singleton axis), and `NO FLOOR` (the study measured no
    replicates, so the question cannot be asked) are three different findings, and
    printing any of them as another is a false statement about what was observed.
    """
    if not varied:
        return "DID NOT VARY"
    if flag is None:
        return "NO FLOOR"
    return "RESOLVED" if flag else "unresolvable"


def _noise_floor_markdown(floor: Mapping[str, Any]) -> List[str]:
    """The floor, first, because every number after it is divided by it."""
    lines = ["## 0. The noise floor -- read this before any number below", ""]
    if not floor.get("available"):
        lines += [
            "**UNAVAILABLE.** " + floor.get("reason", ""), "",
            "Every magnitude in this report is therefore presented WITHOUT A "
            "SCALE. A difference of 1e-4 and a difference of 1e-1 look "
            "identical here -- and job 445689 measured the losses this study "
            "must separate as spanning 0.069 with the best two 0.004 apart, so "
            "that distinction is the whole question. Treat every ranking below "
            "as unvalidated ordering.", ""]
        return lines

    lines += [
        f"**sigma = {floor['sigma']:.6g}** held-out loss, from "
        f"{floor['dof']} degree(s) of freedom.", "",
        "This is what ONE configuration does when only `runtime.seed` changes. "
        "It is measured from designated reference replicates and from nothing "
        "else -- not from the spread of the study, which is a spread ACROSS "
        "configurations and is the quantity being measured.", "",
        f"**Minimum resolvable effect = "
        f"{floor['min_resolvable_effect']:.6g}.**", "",
        floor["criterion"], "",
        floor.get("interpretation", ""), "",
        "| group | n | seeds | mean | sd | range | trials |",
        "|:--|---:|:--|---:|---:|---:|:--|",
    ]
    for group in floor["groups"]:
        lines.append(
            f"| `{group['group']}` | {group['n']} | "
            f"{group['seeds']} | {_fmt(group['mean'], '.6g')} | "
            f"{_fmt(group['sd'], '.4g')} | {_fmt(group['range'], '.4g')} | "
            f"{group['trials']} |")
    lines.append("")
    return lines


def _contrasts_markdown(rows: Sequence[Mapping[str, Any]]) -> List[str]:
    lines = [
        "## 0b. Controlled anchor contrasts",
        "",
        "The only part of this report that is a CONTROLLED EXPERIMENT rather "
        "than an observational summary. Each anchor is a one-factor move from a "
        "single reference point -- verified after resolution, so no second axis "
        "moved on the way through -- and each delta is against the reference "
        "replicate MEAN rather than against one of its seeds.",
        "",
        "Read this table before the importance table and before the pairwise "
        "tables. It is the only place a difference is attributable to a named "
        "factor without an argument about the sampler.",
        "",
    ]
    if not rows:
        lines += ["No completed anchor trials outside the replicate set.", ""]
        return lines
    lines += ["| anchor | trial | objective | delta vs reference | sigmas | "
              "verdict |",
              "|:--|---:|---:|---:|---:|:--|"]
    for row in rows:
        lines.append(
            f"| `{row['anchor']}` | {row['trial']} | "
            f"{_fmt(row['objective'], '.6g')} | {_fmt(row['delta'], '+.4g')} | "
            f"{_fmt(row['sigmas'], '.1f')} | {_resolved_word(row['resolved'])} |")
    lines += ["",
              "`unresolvable` does NOT mean the factor does not matter. It means "
              "this study, at 600 steps, cannot tell that move apart from a "
              "change of seed -- and more trials would not change that, because "
              "the limit is the measurement and not the sample size.", ""]
    return lines


def _main_effects_markdown(result: Mapping[str, Any]) -> List[str]:
    lines = ["## 0c. Partial dependence / main effects", "",
             result["interpretation"], "",
             # The threshold caveat, because it is NOT the same threshold section
             # 0b and 0d use and the column header is the same word.
             "**The verdict column here is the weakest of the three that carry "
             "one.** A level mean averages over many trials, so its own "
             "seed-noise standard error is `sigma/sqrt(n)` rather than `sigma`, "
             "and the spread of two such means is correspondingly tighter than "
             "the trial-vs-trial threshold it is being compared against. So an "
             "axis marked `unresolvable` here may still have a resolvable effect "
             "-- read the controlled anchor contrasts in section 0b, which "
             "compare like with like. A `RESOLVED` here is safe in the other "
             "direction, and `DID NOT VARY` means the axis took one value and "
             "was never measured at all.",
             "",
             "| axis | variance share | spread of level means | verdict | "
             "levels (mean, n) |",
             "|:--|---:|---:|:--|:--|"]
    for axis in result["axes"]:
        cells = "; ".join(
            f"{level['label']}: {_fmt(level['mean'], '.5g')} (n={level['n']})"
            for level in axis["levels"])
        lines.append(
            f"| `{axis['axis']}` | {axis['variance_share']:.4f} | "
            f"{_fmt(axis['spread'], '.4g')} | "
            f"{_resolved_word(axis['resolved'], varied=axis['varied'])} "
            f"| {cells} |")
    lines.append("")
    return lines


def _conditional_markdown(rows: Sequence[Mapping[str, Any]]) -> List[str]:
    lines = [
        "## 0d. Conditional effects -- the interaction magnitudes",
        "",
        "How much the effect of `left` CHANGES across the levels of `right`. That "
        "is what an interaction is, so these are the one-number-per-pair "
        "summaries of the question the study was launched to answer. The full "
        "cell tables are in section 2.",
        "",
        "Both are differences OF differences, so they accumulate noise from four "
        "cell means while being tested against a two-single-trials threshold -- "
        "which makes this the most OPTIMISTIC comparison in this report. A pair "
        "that fails it is certainly unresolvable; a pair that passes it narrowly "
        "needs its cell counts read before it is believed.",
        "",
        "**Two magnitudes, because an interaction can change a SIZE or a "
        "DIRECTION and only one of those invalidates a single recommended "
        "value.** `size` is how much the unsigned effect of `left` varies across "
        "the levels of `right`. `signed` is the same statistic over the SIGNED "
        "contrast (last minus first level of `left`), and it is the one that "
        "catches a crossover: if `left` is better at one level of `right` and "
        "worse by the same amount at another, the unsigned ranges are identical "
        "and `size` is exactly 0.0 while the interaction is maximal. The verdict "
        "reads whichever is larger, and `sign change` marks the rows where the "
        "contrasts do not all point the same way -- those are the ones where "
        "there is no single best value of `left` to recommend.",
        "",
        "| left | right | size | signed | sign change | sigmas | verdict | "
        "conditional effects (level: effect, signed contrast, n) |",
        "|:--|:--|---:|---:|:--|---:|:--|:--|",
    ]
    for row in rows:
        cells = "; ".join(
            f"{c['level']}: {_fmt(c['effect'], '.4g')} / "
            f"{_fmt(c['contrast'], '+.4g')} (n={c['n']})"
            for c in row["conditionals"])
        lines.append(
            f"| `{row['left']}` | `{row['right']}` | "
            f"{_fmt(row['interaction_magnitude'], '.4g')} | "
            f"{_fmt(row['signed_magnitude'], '.4g')} | "
            f"{'**YES**' if row['sign_change'] else 'no'} | "
            f"{_fmt(row['sigmas'], '.1f')} | "
            f"{_resolved_word(row['resolved'])} | {cells} |")
    lines.append("")
    return lines


def _shortlist_markdown(result: Mapping[str, Any]) -> List[str]:
    lines = ["## 5. Shortlist for confirmation (NOT a winner)", "",
             result["interpretation"], ""]
    for entry in result["entries"]:
        lines.append(f"### `{entry['slot']}`")
        lines.append("")
        lines.append(entry["rationale"])
        lines.append("")
        if entry["trial"] is None:
            lines += [f"**Unfilled**: {entry.get('note', '')}", ""]
            continue
        # The verdict word is about MAGNITUDE, so it has to be qualified by the
        # SIGN here or a candidate that is resolvably WORSE than the reference
        # reads as endorsed. `interaction_probe` is routinely in that position by
        # design -- it is chosen to test a surface, not to win.
        lead = entry.get("lead_over_reference")
        verdict = _resolved_word(entry.get("resolved_vs_reference"))
        if lead is not None and verdict == "RESOLVED":
            verdict += " BETTER" if lead > 0 else " WORSE"
        lines += [
            f"- trial **{entry['trial']}**"
            + (f" (anchor `{entry['anchor']}`)" if entry["anchor"] else ""),
            f"- objective **{_fmt(entry['objective'], '.6g')}**, "
            f"lead over reference {_fmt(lead, '+.4g')} -> **{verdict}**",
            f"- run_id `{entry['run_id'] or '(not recorded)'}`, "
            f"seed {_fmt(entry.get('model_seed'))}, "
            f"params {_fmt(entry.get('param_count'), ',')}, "
            f"{_fmt(entry.get('tokens_per_sec'), '.4g')} tok/s, "
            f"{_fmt(entry.get('peak_memory_gib'), '.3g')} GiB peak",
            f"- SKA indices {entry.get('ska_layer_indices')}, "
            f"rank {_fmt(entry.get('param_ska_rank'))}, "
            f"K {_fmt(entry.get('param_ska_power_K'))}, "
            f"ridge {_fmt(entry.get('param_ska_ridge'), '.4g')}, "
            f"lr {_fmt(entry.get('param_learning_rate'), '.4g')}",
        ]
        if entry.get("note"):
            lines.append(f"- {entry['note']}")
        lines.append("")
    lines += ["## 6. What this study CANNOT establish", "",
              "Written down rather than left to judgement, because the failure "
              "mode is a reader taking the shortlist's first row as a result.",
              ""]
    for item in result["cannot_establish"]:
        lines += [f"- {item}", ""]
    return lines


def _interactions_markdown(result: Mapping[str, Any]) -> str:
    lines = [
        f"# Interaction analysis: {result['study_name']}",
        "",
        "## How to read this, and what it is not",
        "",
        _CAVEAT,
        "",
        _MULTIPLICITY,
        "",
        "Five different kinds of number appear below and they are NOT "
        "interchangeable:",
        "",
        "0. **The noise floor** -- the spread of one configuration across "
        "training seeds. Every magnitude below is stated against it, and one "
        "smaller than it is labelled `unresolvable` rather than ranked.",
        "1. **Controlled anchor contrasts** -- one factor moved from one "
        "reference point. The only causal statements here.",
        "2. **Partial dependence and pairwise response structure** -- "
        "descriptive, over trials the sampler chose. Read every cell mean with "
        "its n.",
        "3. **PED-ANOVA importance** -- a divergence between two distributions "
        "the sampler produced. Informative, not proof; see its own section.",
        "4. **Sampler-induced correlation** -- a diagnostic of where the sampler "
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

    lines += _noise_floor_markdown(result["noise_floor"])
    lines += _contrasts_markdown(result["anchor_contrasts"])
    lines += _main_effects_markdown(result["main_effects"])
    lines += _conditional_markdown(result["conditional_effects"])

    lines += ["## 1. PED-ANOVA importance (a divergence, not a decomposition)",
              ""]
    importances = result["importances"]
    if importances["available"]:
        lines += [f"Evaluator: `{importances['evaluator']}`.", "",
                  importances["interpretation"], "",
                  "| axis | local | global |", "|:--|---:|---:|"]
        global_values = importances.get("values_global") or {}
        for axis, value in sorted(importances["values"].items(),
                                  key=lambda kv: -kv[1]):
            other = global_values.get(axis)
            lines.append(f"| `{axis}` | {value:.4f} | "
                         f"{_fmt(other, '.4f')} |")
        if importances.get("global_reason"):
            lines += ["", f"The `global` column is empty: "
                          f"{importances['global_reason']}"]
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

    lines += ["## 4b. Loss / throughput Pareto front", "",
              "The other half of the cost question, and until `tokens_per_sec` "
              "was promoted onto the trial it could not be computed at all -- "
              "the number was measured in every `quick_eval.json` and reached "
              "neither `user_attrs` nor `trials.csv`.", "",
              "Read WITH `per_device_batch_size`: a trial that descended the OOM "
              "ladder was measured at a smaller microbatch than its neighbours, "
              "so its throughput is not comparable to theirs. That caveat is why "
              "the loss/throughput exchange rate stays the reader's rather than "
              "becoming a `throughput_penalty` baked into every proposal.", ""]
    notes = result["throughput_notes"]
    lines += [notes["reason"], "",
              f"Of {notes['n_completed']} completed trial(s): "
              f"{notes['n_with_throughput']} carry a throughput measurement, "
              f"{notes['n_missing_throughput']} do not, and "
              f"{notes['n_excluded_warmup']} were excluded as a worker's first "
              f"trial (trials {notes['excluded_warmup_trials']}).", ""]
    if result["throughput_pareto"]:
        lines += ["| tok/s | objective | trial | pdbs | peak GiB | params | "
                  "anchor |",
                  "|---:|---:|---:|---:|---:|---:|:--|"]
        for point in result["throughput_pareto"]:
            lines.append(
                f"| {point['tokens_per_sec']:,.0f} | "
                f"{point['objective']:.5f} | {point['trial']} | "
                f"{_fmt(point['per_device_batch_size'])} | "
                f"{_fmt(point['peak_memory_gib'], '.3g')} | "
                f"{_fmt(point['param_count'], ',')} | {point['anchor']} |")
    else:
        lines.append("No completed trial carried a throughput measurement. "
                     "Check that `quick_eval.json` recorded "
                     "`metrics.full.tokens_per_sec` and that "
                     "`driver._stamp_measured_metrics` ran.")
    lines.append("")

    lines += _shortlist_markdown(result["shortlist"])
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
        # First key after the counts on purpose: it is the scale for everything
        # else in this payload, and a consumer that reads only the top of the
        # file should not be able to miss it.
        "noise_floor": result["noise_floor"],
        "importances": result["importances"],
        "prespecified_pairs": [list(p) for p in PRESPECIFIED_PAIRS],
        "multiplicity": _MULTIPLICITY,
        "caveat": _CAVEAT,
        "shortlist": result["shortlist"]["entries"],
        "cannot_establish": list(CANNOT_ESTABLISH),
        "throughput_notes": result["throughput_notes"],
        # WITHOUT its label, `sampler_correlations.csv` is a bare `a,b,r,n`
        # table -- exactly the thing this module refuses to publish. A machine
        # consumer read the correlations and not the sentence saying they are
        # not evidence, which is the one reader most likely to quote them.
        "sampler_correlations": {
            "is_evidence": result["sampler_correlations"]["is_evidence"],
            "interpretation":
                result["sampler_correlations"]["interpretation"],
            "constant_columns":
                result["sampler_correlations"]["constant_columns"],
            "excluded_columns":
                result["sampler_correlations"]["excluded_columns"],
            "pairs": result["sampler_correlations"]["pairs"],
        },
    })

    written["noise_floor"] = _write_csv(
        out_dir / "noise_floor.csv",
        [{"group": g["group"], "n": g["n"], "mean": g["mean"], "sd": g["sd"],
          "range": g["range"], "seeds": g["seeds"], "trials": g["trials"],
          "objectives": g["objectives"],
          "sigma": result["noise_floor"].get("sigma"),
          "min_resolvable_effect":
              result["noise_floor"].get("min_resolvable_effect")}
         for g in result["noise_floor"]["groups"]],
        columns=["group", "n", "mean", "sd", "range", "sigma",
                 "min_resolvable_effect", "seeds", "trials", "objectives"])
    written["anchor_contrasts"] = _write_csv(
        out_dir / "anchor_contrasts.csv", result["anchor_contrasts"],
        columns=["anchor", "trial", "objective", "reference_mean",
                 "reference_n", "reference_sem", "delta", "abs_delta",
                 "sd_of_contrast", "threshold", "sigmas", "resolved"])
    written["main_effects"] = _write_csv(
        out_dir / "main_effects.csv",
        [{"axis": a["axis"], "variance_share": a["variance_share"],
          "spread": a["spread"], "resolved": a["resolved"],
          "varied": a["varied"], "binned": a["binned"],
          "levels": "; ".join(f"{c['label']}={c['mean']} (n={c['n']})"
                              for c in a["levels"])}
         for a in result["main_effects"]["axes"]],
        columns=["axis", "variance_share", "spread", "resolved", "varied",
                 "binned", "levels"])
    written["conditional_effects"] = _write_csv(
        out_dir / "conditional_effects.csv",
        [{"left": r["left"], "right": r["right"],
          "interaction_magnitude": r["interaction_magnitude"],
          "signed_magnitude": r["signed_magnitude"],
          "sign_change": r["sign_change"],
          "sigmas": r["sigmas"], "resolved": r["resolved"],
          "degenerate": r["degenerate"], "thin_levels": r["thin_levels"],
          "conditionals": "; ".join(
              f"{c['level']}={c['effect']}/{c['contrast']} (n={c['n']})"
              for c in r["conditionals"])}
         for r in result["conditional_effects"]],
        columns=["left", "right", "interaction_magnitude", "signed_magnitude",
                 "sign_change", "sigmas", "resolved", "degenerate",
                 "thin_levels", "conditionals"])
    written["throughput_pareto"] = _write_csv(
        out_dir / "throughput_pareto.csv", result["throughput_pareto"],
        columns=["tokens_per_sec", "objective", "trial",
                 "per_device_batch_size", "peak_memory_gib", "param_count",
                 "worker_trial_ordinal", "anchor", "run_id"])
    written["shortlist"] = _write_csv(
        out_dir / "shortlist.csv", result["shortlist"]["entries"],
        columns=["slot", "trial", "objective", "lead_over_reference",
                 "resolved_vs_reference", "param_count", "tokens_per_sec",
                 "peak_memory_gib", "ska_delta", "param_ska_rank",
                 "param_n_ska_layers", "param_placement", "param_ska_power_K",
                 "param_ska_ridge", "param_ska_layerscale_init",
                 "param_norm_clip_multiplier", "param_gamma_value",
                 "param_learning_rate", "ska_layer_indices", "model_seed",
                 "anchor", "run_id", "rationale", "note"])

    written["importances"] = _write_csv(
        out_dir / "importances.csv",
        [{"axis": a, "importance": v,
          "importance_global":
              (result["importances"].get("values_global") or {}).get(a)}
         for a, v in sorted(result["importances"].get("values", {}).items(),
                            key=lambda kv: -kv[1])],
        columns=["axis", "importance", "importance_global"])
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
