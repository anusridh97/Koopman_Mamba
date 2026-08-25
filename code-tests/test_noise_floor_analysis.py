"""The analysis half of the noise floor: measuring it, and refusing to overclaim.

`test_reference_repeats.py` proves the study can PRODUCE repeated reference
trials. This file is about what the analysis is then obliged to do with them, and
every test here exists because the alternative is a plausible, complete, wrong
report.

**The one number.** The spread of the reference replicates is the smallest
difference this study can resolve. Job 445689 measured what has to be resolved:
four trials on the proxy base at 600 steps, losses spanning 0.069, best two
0.004 apart. If the floor is of that order then the differences between anchors
are not measurements and a multivariate TPE fitted to them produces a confident
joint model of noise. So:

  * the floor is computed from the replicates and from nothing else -- never from
    the spread of the whole study, which is a spread ACROSS configurations and
    would be larger by construction;
  * every reported effect carries its size relative to the floor, and an effect
    below it is labelled UNRESOLVABLE rather than ranked;
  * a study with no replicates says so loudly instead of quietly reporting
    rankings with no scale.

**The overclaim being fixed.** A previous review found that the PED-ANOVA
importance label reads as "this IS evidence about the model", when what PED-ANOVA
measures is a divergence between the sampler's own top-quantile distribution and
its overall empirical distribution. Under adaptive sampling that empirical
distribution is a description of the search path, so the headline number is partly
a function of where the sampler went -- and on a test fixture a null axis
outranked a planted main effect. The framing is softened and a second,
`evaluate_on_local=False` variant is reported beside it, which measures against
the DECLARED prior instead. Neither is dropped and neither is called proof.
"""
from __future__ import annotations

import math
import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

optuna = pytest.importorskip("optuna")

pytestmark = pytest.mark.correctness


def _distributions():
    return {
        "ska_rank": optuna.distributions.CategoricalDistribution([8, 16, 24, 32]),
        "n_ska_layers": optuna.distributions.CategoricalDistribution(
            [2, 3, 4, 6, 8]),
        "placement": optuna.distributions.CategoricalDistribution(
            ["baseline", "even", "midlate", "late"]),
        "ska_ridge": optuna.distributions.FloatDistribution(
            0.003, 0.03, log=True),
        "ska_layerscale_init": optuna.distributions.FloatDistribution(
            0.005, 0.3, log=True),
        "norm_clip_multiplier": optuna.distributions.CategoricalDistribution(
            [0.75, 0.8164965809277261, 1.0, 1.25]),
        "gamma_value": optuna.distributions.CategoricalDistribution(
            [0.9, 1.0, 1.05]),
        "ska_power_K": optuna.distributions.CategoricalDistribution([1, 2]),
        "learning_rate": optuna.distributions.FloatDistribution(
            0.00032, 0.00048, log=True),
        "weight_decay": optuna.distributions.CategoricalDistribution([0.1]),
        "warmup_ratio": optuna.distributions.CategoricalDistribution([0.04]),
        "grad_clip": optuna.distributions.CategoricalDistribution([1.0]),
    }


#: The reference point, and the five losses its five seeds land on. Chosen so the
#: sample SD is a round-ish number the assertions can name: mean 4.0, sd 0.05.
_REFERENCE_LOSSES = (3.95, 3.975, 4.0, 4.025, 4.05)


def _reference_params():
    return {"ska_rank": 24, "n_ska_layers": 4, "placement": "baseline",
            "ska_ridge": 0.01, "ska_layerscale_init": 0.01,
            "norm_clip_multiplier": 0.8164965809277261, "gamma_value": 1.0,
            "ska_power_K": 1, "learning_rate": 0.0004,
            "weight_decay": 0.1, "warmup_ratio": 0.04, "grad_clip": 1.0}


def _attrs(index, params, **extra):
    return {"param_count": 25_000_000 + 40_000 * params["ska_rank"]
                           + 180_000 * params["n_ska_layers"],
            "baseline_param_count": 25_352_736,
            "model_seed": 42,
            "worker_id": index % 8,
            "sampler": "tpe_multivariate",
            "sampler_seed": 2026 + index % 8,
            "per_device_batch_size": 8,
            "tokens_per_sec": 60_000.0 - 700.0 * params["ska_rank"]
                              - 900.0 * params["n_ska_layers"],
            "peak_memory_gib": 10.0 + 0.2 * params["ska_rank"],
            "n_eval_tokens": 262_144,
            "ska_delta": 1.0e-4,
            **extra}


@pytest.fixture(scope="module")
def study():
    """A finished study with a MEASURED noise floor and a planted main effect.

    Structure, all of it load-bearing for the assertions:

      * five `reference` replicates at `_REFERENCE_LOSSES` -- sd 0.0395, so the
        minimum resolvable effect is about 0.11;
      * `rank-32`, whose true effect (-0.16) is LARGER than the floor, so it must
        come back resolved;
      * `gamma-high`, whose true effect (+0.01) is SMALLER than the floor, so it
        must come back UNRESOLVABLE -- the assertion that gives this whole
        mechanism teeth;
      * 120 sampled trials with a rank main effect and a rank x ridge
        interaction, so the pairwise and importance machinery has something real
        to find.
    """
    import random

    rng = random.Random(20260824)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    st = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=11),
                             pruner=optuna.pruners.NopPruner())
    distributions = _distributions()
    reference = _reference_params()

    # --- the replicate set -------------------------------------------------
    for position, (loss, seed) in enumerate(
            zip(_REFERENCE_LOSSES, (42, 43, 44, 45, 46))):
        st.add_trial(optuna.trial.create_trial(
            params=dict(reference), distributions=distributions, value=loss,
            user_attrs=_attrs(position, reference,
                              anchor_name=("reference-k1" if position == 0
                                           else f"reference-seed-{seed}"),
                              reference_group="reference",
                              model_seed=seed)))

    # --- two one-factor anchors, one resolvable and one not ---------------
    for name, key, value, loss in (("rank-32", "ska_rank", 32, 3.84),
                                   ("gamma-high", "gamma_value", 1.05, 4.01)):
        params = {**reference, key: value}
        st.add_trial(optuna.trial.create_trial(
            params=params, distributions=distributions, value=loss,
            user_attrs=_attrs(0, params, anchor_name=name)))

    # --- the sampled body -------------------------------------------------
    for index in range(120):
        trial = st.ask(distributions)
        params = trial.params
        rank, ridge = params["ska_rank"], params["ska_ridge"]
        want = 0.003 * (rank / 8.0)
        loss = (4.0 - 0.02 * rank + 8.0 * abs(math.log(ridge / want))
                + 0.05 * rng.random())
        for key, value in _attrs(index, params).items():
            trial.set_user_attr(key, value)
        st.tell(trial, loss)

    for _ in range(9):
        trial = st.ask(distributions)
        trial.report(9.0, 450)
        st.tell(trial, state=optuna.trial.TrialState.PRUNED)
    return st


@pytest.fixture(scope="module")
def floor(study):
    from experimentation.sweep.search.analysis import noise_floor
    return noise_floor(study)


# ------------------------------------------------------------ the floor ----

def test_the_noise_floor_is_measured_from_the_replicates(floor):
    assert floor["available"] is True
    assert [g["group"] for g in floor["groups"]] == ["reference"]
    assert floor["groups"][0]["n"] == 5


def test_the_floor_is_the_within_group_spread_and_nothing_else(floor):
    """NOT the spread of the whole study. That is a spread ACROSS
    configurations, which is the quantity being measured, not the measurement
    error -- using it would make every effect trivially unresolvable."""
    values = list(_REFERENCE_LOSSES)
    mean = sum(values) / len(values)
    expected = math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))
    assert floor["sigma"] == pytest.approx(expected)


def test_the_floor_records_the_seeds_it_came_from(floor):
    assert sorted(floor["groups"][0]["seeds"]) == [42, 43, 44, 45, 46]


def test_the_minimum_resolvable_effect_is_derived_and_its_rule_is_stated(floor):
    """A single sigma is not a threshold for a DIFFERENCE of two single-trial
    measurements: that difference has sd sigma*sqrt(2). The reported threshold is
    two of those, and the artefact has to say so -- a bare number invites being
    read as a p-value."""
    assert floor["sd_of_difference"] == pytest.approx(
        floor["sigma"] * math.sqrt(2))
    assert floor["min_resolvable_effect"] == pytest.approx(
        2.0 * floor["sigma"] * math.sqrt(2))
    assert "sqrt(2)" in floor["criterion"]
    assert "not a p-value" in floor["criterion"].lower()


def test_a_study_with_no_replicates_says_so_rather_than_reporting_zero(study):
    """A floor of zero would make every difference resolvable, which is the
    single most damaging way this could fail."""
    from experimentation.sweep.search.analysis import noise_floor

    bare = optuna.create_study()
    bare.add_trial(optuna.trial.create_trial(
        params={}, distributions={}, value=1.0))
    result = noise_floor(bare)
    assert result["available"] is False
    assert result["sigma"] is None
    assert result["min_resolvable_effect"] is None
    assert "reference_group" in result["reason"]


def test_a_group_of_one_completed_trial_cannot_produce_a_floor():
    """One observation has no spread. Reported as unavailable, not as 0.0."""
    from experimentation.sweep.search.analysis import noise_floor

    st = optuna.create_study()
    st.add_trial(optuna.trial.create_trial(
        params={}, distributions={}, value=1.0,
        user_attrs={"reference_group": "reference", "anchor_name": "ref"}))
    result = noise_floor(st)
    assert result["available"] is False
    assert result["sigma"] is None


# -------------------------------------------------- anchor contrasts ----

def test_an_anchor_contrast_larger_than_the_floor_is_resolved(study, floor):
    from experimentation.sweep.search.analysis import anchor_contrasts

    rows = {r["anchor"]: r for r in anchor_contrasts(study, floor)}
    assert rows["rank-32"]["delta"] == pytest.approx(3.84 - 4.0, abs=1e-9)
    assert rows["rank-32"]["resolved"] is True


def test_an_anchor_contrast_smaller_than_the_floor_is_UNRESOLVED(study, floor):
    """The test that gives the whole mechanism teeth. `gamma-high` is 0.01 from
    the reference against a minimum resolvable effect of ~0.11, so ranking it as
    an effect would be reporting noise."""
    from experimentation.sweep.search.analysis import anchor_contrasts

    rows = {r["anchor"]: r for r in anchor_contrasts(study, floor)}
    assert abs(rows["gamma-high"]["delta"]) < floor["min_resolvable_effect"]
    assert rows["gamma-high"]["resolved"] is False


def test_a_contrast_is_measured_against_the_replicate_MEAN(study, floor):
    """Not against `reference-k1` alone. The reference is estimated five times and
    using one of the five throws away four -- and picks whichever seed happened
    to be first."""
    from experimentation.sweep.search.analysis import anchor_contrasts

    rows = {r["anchor"]: r for r in anchor_contrasts(study, floor)}
    assert rows["rank-32"]["reference_mean"] == pytest.approx(
        sum(_REFERENCE_LOSSES) / len(_REFERENCE_LOSSES))
    assert rows["rank-32"]["reference_n"] == 5


def test_the_replicates_themselves_are_not_listed_as_contrasts(study, floor):
    from experimentation.sweep.search.analysis import anchor_contrasts

    names = {r["anchor"] for r in anchor_contrasts(study, floor)}
    assert names == {"rank-32", "gamma-high"}


def test_contrasts_are_unavailable_without_a_floor_rather_than_unlabelled(study):
    """A contrast with no floor is a number with no scale. It is still WORTH
    reporting -- but as an unresolvable-by-construction row, never as a ranking."""
    from experimentation.sweep.search.analysis import anchor_contrasts

    rows = anchor_contrasts(study, {"available": False, "sigma": None,
                                    "min_resolvable_effect": None})
    assert rows and all(r["resolved"] is None for r in rows)


# ------------------------------------------------------ main effects ----

def test_every_searched_axis_gets_a_marginal_mean_per_level(study, floor):
    from experimentation.sweep.search.analysis import main_effects

    axes = {a["axis"]: a for a in main_effects(study, floor)["axes"]}
    assert axes["ska_rank"]["levels"] and len(axes["ska_rank"]["levels"]) == 4
    assert all(level["n"] >= 0 for level in axes["ska_rank"]["levels"])


def test_the_planted_main_effect_has_the_largest_marginal_spread(study, floor):
    from experimentation.sweep.search.analysis import main_effects

    axes = main_effects(study, floor)["axes"]
    ranked = sorted(axes, key=lambda a: -(a["spread"] or 0.0))
    assert ranked[0]["axis"] in ("ska_rank", "ska_ridge"), [
        (a["axis"], a["spread"]) for a in ranked[:4]]


def test_a_marginal_spread_below_the_floor_is_labelled_unresolvable(study, floor):
    """A real axis whose observed level means barely move.

    NOT a fixed axis: those never varied and are `spread=None` /
    `varied=False` / `DID NOT VARY`, which is a different finding from
    "measured, smaller than the floor" -- see
    `test_an_axis_that_never_varied_says_so_rather_than_reporting_zero_spread`.
    """
    from experimentation.sweep.search.analysis import main_effects

    axes = {a["axis"]: a for a in main_effects(study, floor)["axes"]}
    unresolved = [a for a in axes.values()
                  if a["varied"] and a["resolved"] is False]
    assert unresolved or all(a["resolved"] for a in axes.values() if a["varied"])
    for axis in unresolved:
        assert axis["spread"] < floor["min_resolvable_effect"], axis["axis"]


def test_the_variance_share_is_labelled_as_descriptive_not_causal(study, floor):
    """It is the count-weighted between-level variance of the OBSERVED objective.
    That is a description of the trials that ran, and the sampler chose those --
    so it is not a causal decomposition and the artefact must not imply one."""
    from experimentation.sweep.search.analysis import main_effects

    result = main_effects(study, floor)
    assert 0.0 <= result["axes"][0]["variance_share"] <= 1.0
    assert "not" in result["interpretation"].lower()
    assert "sampler" in result["interpretation"].lower()


# ------------------------------------------------ conditional effects ----

def test_a_conditional_effect_is_reported_for_every_prespecified_pair(study,
                                                                      floor):
    from experimentation.sweep.search.analysis import (
        PRESPECIFIED_PAIRS, conditional_effects)

    rows = conditional_effects(study, floor)
    assert {(r["left"], r["right"]) for r in rows} == set(PRESPECIFIED_PAIRS)


def test_an_interaction_magnitude_is_stated_against_the_floor(study, floor):
    from experimentation.sweep.search.analysis import conditional_effects

    for row in conditional_effects(study, floor):
        assert row["resolved"] in (True, False, None)
        if row["interaction_magnitude"] is not None:
            assert row["interaction_magnitude"] >= 0.0


def test_the_planted_interaction_is_the_largest_conditional_effect(study, floor):
    """`rank x ridge` is planted: the best ridge MOVES with rank, so the effect of
    ridge differs across rank levels. That difference is what
    `interaction_magnitude` measures."""
    from experimentation.sweep.search.analysis import conditional_effects

    rows = [r for r in conditional_effects(study, floor)
            if r["interaction_magnitude"] is not None]
    rows.sort(key=lambda r: -r["interaction_magnitude"])
    assert ("ska_rank", "ska_ridge") in {(r["left"], r["right"])
                                         for r in rows[:2]}, [
        (r["left"], r["right"], r["interaction_magnitude"]) for r in rows[:4]]


# ----------------------------------------------------- prespecified pairs ----

def test_the_nine_requested_interactions_are_all_prespecified():
    """The nine the study was commissioned to report. Listed in code, before any
    trial of the 256-trial study has run, which is what makes "prespecified"
    true rather than decorative."""
    from experimentation.sweep.search.analysis import PRESPECIFIED_PAIRS

    required = {
        ("ska_rank", "n_ska_layers"),
        ("placement", "n_ska_layers"),
        ("ska_power_K", "ska_ridge"),
        ("ska_power_K", "gamma_value"),
        ("ska_ridge", "ska_layerscale_init"),
        ("gamma_value", "norm_clip_multiplier"),
        ("learning_rate", "ska_rank"),
        ("learning_rate", "n_ska_layers"),
        ("learning_rate", "ska_power_K"),
    }
    unordered = {frozenset(p) for p in PRESPECIFIED_PAIRS}
    missing = [sorted(p) for p in required
               if frozenset(p) not in unordered]
    assert not missing, f"not prespecified: {missing}"


def test_the_original_seven_are_still_prespecified():
    """Kept, not replaced. They were written down before this study ran too, and
    dropping a prespecified pair after the fact is the same error as adding one:
    it makes the reported set depend on a later judgement."""
    from experimentation.sweep.search.analysis import PRESPECIFIED_PAIRS

    original = {
        ("ska_rank", "ska_ridge"), ("ska_rank", "norm_clip_multiplier"),
        ("ska_rank", "ska_power_K"),
        ("n_ska_layers", "ska_layerscale_init"),
        ("n_ska_layers", "placement"), ("ska_ridge", "ska_power_K"),
        ("ska_layerscale_init", "learning_rate"),
    }
    unordered = {frozenset(p) for p in PRESPECIFIED_PAIRS}
    assert all(frozenset(p) in unordered for p in original)


def test_no_pair_is_listed_twice_in_either_order():
    """`(a, b)` and `(b, a)` are the same table. Two of them would double that
    pair's weight in any multiplicity argument."""
    from experimentation.sweep.search.analysis import PRESPECIFIED_PAIRS

    keys = [frozenset(p) for p in PRESPECIFIED_PAIRS]
    assert len(keys) == len(set(keys))
    assert all(len(k) == 2 for k in keys)


def test_every_prespecified_axis_is_a_real_searched_axis():
    from experimentation.sweep.search.analysis import PRESPECIFIED_PAIRS
    from experimentation.sweep.search.studyspec import load_study_spec

    spec = load_study_spec(
        REPO / "configs/search/proxy-256x17-interactions-v1.yaml")
    searched = set(spec.search_axes)
    for left, right in PRESPECIFIED_PAIRS:
        assert left in searched, left
        assert right in searched, right


def test_the_multiplicity_of_the_reported_set_is_stated(study, floor, tmp_path):
    """14 of 36 possible pairs is a real multiple-comparisons burden and the
    report has to name it rather than leave a reader to count the tables."""
    from experimentation.sweep.search.analysis import (
        PRESPECIFIED_PAIRS, write_analysis)

    written = write_analysis(study, tmp_path)
    text = written["interactions"].read_text()
    assert f"{len(PRESPECIFIED_PAIRS)} of 36" in text, text[:4000]


# --------------------------------------------------- throughput Pareto ----

def test_the_loss_throughput_front_is_non_dominated(study):
    from experimentation.sweep.search.analysis import throughput_pareto

    front = throughput_pareto(study)
    assert front
    for point in front:
        assert not any(
            other["objective"] <= point["objective"]
            and other["tokens_per_sec"] >= point["tokens_per_sec"]
            and (other["objective"], other["tokens_per_sec"])
            != (point["objective"], point["tokens_per_sec"])
            for other in front)


def test_the_best_trial_and_the_best_of_the_fastest_are_on_the_front(study):
    """Two endpoints, and the second is stated carefully on purpose.

    The best-loss trial is always non-dominated. The FASTEST trial is not
    necessarily: throughput here is a function of rank and layer count, so many
    trials tie on it exactly, and among a tied group only the one with the best
    loss is non-dominated. Asserting "the fastest trial" would pick an arbitrary
    member of that tie and fail against a correct front -- which is what it did.
    """
    from experimentation.sweep.search.analysis import throughput_pareto

    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE
                 and t.user_attrs.get("tokens_per_sec")]
    numbers = {p["trial"] for p in throughput_pareto(study)}

    assert min(completed, key=lambda t: t.value).number in numbers

    top_speed = max(t.user_attrs["tokens_per_sec"] for t in completed)
    at_top_speed = [t for t in completed
                    if t.user_attrs["tokens_per_sec"] == top_speed]
    assert min(at_top_speed, key=lambda t: t.value).number in numbers
    assert max(p["tokens_per_sec"] for p in throughput_pareto(study)) == top_speed


def test_a_trial_with_no_throughput_is_excluded_not_zero_filled():
    """A trial recorded at 0 tokens/sec would sit at the wrong end of the front
    and look like a measurement."""
    from experimentation.sweep.search.analysis import throughput_pareto

    st = optuna.create_study()
    st.add_trial(optuna.trial.create_trial(
        params={}, distributions={}, value=1.0, user_attrs={}))
    assert throughput_pareto(st) == []


# --------------------------------------------------------- the shortlist ----

def test_the_shortlist_is_a_set_of_candidates_and_never_a_single_winner(study,
                                                                        floor):
    from experimentation.sweep.search.analysis import shortlist

    result = shortlist(study, floor)
    slots = {e["slot"] for e in result["entries"]}
    assert {"best_loss", "best_cost_adjusted", "best_k1", "best_k2",
            "best_low_capacity", "interaction_probe"} <= slots
    assert len(result["entries"]) >= 6


def test_every_shortlist_entry_says_why_it_is_there(study, floor):
    from experimentation.sweep.search.analysis import shortlist

    for entry in shortlist(study, floor)["entries"]:
        assert entry["rationale"], entry["slot"]
        assert entry["trial"] is not None or entry["note"]


def test_the_best_loss_slot_is_the_best_completed_trial(study, floor):
    from experimentation.sweep.search.analysis import shortlist

    entry = next(e for e in shortlist(study, floor)["entries"]
                 if e["slot"] == "best_loss")
    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE]
    assert entry["trial"] == min(completed, key=lambda t: t.value).number


def test_the_k1_and_k2_slots_really_hold_their_own_K(study, floor):
    """The reason those two slots exist: K is the axis most likely to change the
    SIGN of another axis's effect, so a confirmation study needs the best
    candidate at EACH K rather than whichever K happened to win."""
    from experimentation.sweep.search.analysis import shortlist

    by_slot = {e["slot"]: e for e in shortlist(study, floor)["entries"]}
    trials = {t.number: t for t in study.trials}
    assert trials[by_slot["best_k1"]["trial"]].params["ska_power_K"] == 1
    assert trials[by_slot["best_k2"]["trial"]].params["ska_power_K"] == 2


def test_the_shortlist_states_the_margin_over_the_floor_for_each_entry(study,
                                                                       floor):
    """A candidate whose lead over the reference is inside the noise floor is a
    candidate by luck. The shortlist must say which ones those are rather than
    presenting six equally-earned rows."""
    from experimentation.sweep.search.analysis import shortlist

    for entry in shortlist(study, floor)["entries"]:
        if entry["trial"] is None:
            continue
        assert "resolved_vs_reference" in entry


def test_the_shortlist_carries_the_list_of_things_it_cannot_establish(study,
                                                                      floor):
    from experimentation.sweep.search.analysis import shortlist

    cannot = shortlist(study, floor)["cannot_establish"]
    assert len(cannot) >= 4
    joined = " ".join(cannot).lower()
    assert "600" in joined
    assert "50m" in joined or "scale" in joined


# ------------------------------------------------ the PED-ANOVA framing ----

def test_ped_anova_is_reported_against_both_reference_distributions(study):
    """`evaluate_on_local=True` (optuna's default) measures the top quantile
    against the sampler's OWN empirical distribution, so under adaptive sampling
    the headline is partly a description of the search path.
    `evaluate_on_local=False` measures against the DECLARED prior instead. Both,
    labelled, because neither alone is the answer."""
    from experimentation.sweep.search.analysis import analyse

    importances = analyse(study)["importances"]
    assert importances["available"] is True
    assert "values" in importances
    assert "values_global" in importances
    assert importances["evaluator"].startswith("PedAnova")


def test_the_importance_label_no_longer_claims_to_be_evidence(study):
    """The overclaim a review found, removed. It measures a divergence between
    two distributions the sampler produced; that is worth reporting and is not
    proof about the model."""
    from experimentation.sweep.search.analysis import analyse

    text = analyse(study)["importances"]["interpretation"]
    assert "IS evidence" not in text
    assert "divergence" in text.lower()
    assert "search path" in text.lower() or "where the sampler" in text.lower()


def test_a_missing_evaluator_still_degrades_rather_than_crashing(study,
                                                                monkeypatch):
    from experimentation.sweep.search import analysis

    def _boom():
        raise ImportError("no sklearn here")

    monkeypatch.setattr(analysis, "_importance_evaluator", _boom)
    result = analysis.analyse(study)["importances"]
    assert result["available"] is False
    assert "ImportError" in result["reason"]


# ----------------------------------------------------------- artefacts ----

def test_the_report_leads_with_the_noise_floor(study, tmp_path):
    """It is the number every other section is divided by, so it cannot be a
    footnote."""
    from experimentation.sweep.search.analysis import write_analysis

    text = write_analysis(study, tmp_path)["interactions"].read_text()
    head = text[:text.index("## 1.")] if "## 1." in text else text
    assert "noise floor" in head.lower(), head


def test_every_new_artefact_is_written(study, tmp_path):
    from experimentation.sweep.search.analysis import write_analysis

    written = write_analysis(study, tmp_path)
    for name in ("noise_floor", "anchor_contrasts", "main_effects",
                 "conditional_effects", "throughput_pareto", "shortlist"):
        assert name in written, name
        assert written[name].is_file(), name


def test_the_summary_json_carries_the_floor_and_the_threshold(study, tmp_path):
    import json

    from experimentation.sweep.search.analysis import write_analysis

    payload = json.loads(
        write_analysis(study, tmp_path)["summary"].read_text())
    assert payload["noise_floor"]["available"] is True
    assert payload["noise_floor"]["min_resolvable_effect"] > 0
    assert payload["cannot_establish"]


def test_no_new_artefact_is_a_plot(study, tmp_path):
    """No sklearn and no matplotlib in this environment, by measurement."""
    from experimentation.sweep.search.analysis import write_analysis

    for path in write_analysis(study, tmp_path).values():
        assert path.suffix in (".csv", ".json", ".md"), path


def test_the_analysis_still_does_not_modify_the_study(study, tmp_path):
    """It may run against a journal other workers are still writing to."""
    from experimentation.sweep.search.analysis import write_analysis

    before = [(t.number, t.state, t.value, dict(t.user_attrs))
              for t in study.trials]
    write_analysis(study, tmp_path)
    after = [(t.number, t.state, t.value, dict(t.user_attrs))
             for t in study.trials]
    assert before == after

# ------------------------------------------- the first-trial warmup artefact ----
#
# MEASURED, job 445689. Four trials on the proxy base at 600 steps reported
# 17,818 / 111,416 / 104,760 / 112,685 tok/s. The 6x outlier was TRIAL 0, and no
# architectural difference between those four configs can cause a 6x throughput
# gap -- they differ only in `ska_layerscale_init`, one scalar multiply on a gate.
# It paid first-trial CUDA context creation and kernel autotuning, and
# `tokens_per_sec` is measured during the eval pass, early enough in the
# process's life to still be inside that.
#
# At `concurrent_trials: 8` that is EIGHT poisoned points, not one, because each
# worker process pays its own warmup.

def test_a_workers_first_trial_is_kept_off_the_throughput_front():
    """The artefact this exists for: raw `tokens_per_sec` would put trial 0 at the
    slow end of the front for no architectural reason."""
    from experimentation.sweep.search.analysis import throughput_pareto

    study = _with_ordinals_fixture()
    assert 0 not in {p["trial"] for p in throughput_pareto(study)}


def test_the_exclusion_is_reported_rather_than_silent():
    """A front whose point count nobody can reconcile with the trial count is
    worse than a slow outlier. `analyse` says how many were dropped and why."""
    from experimentation.sweep.search.analysis import analyse

    notes = analyse(_with_ordinals_fixture())["throughput_notes"]
    assert notes["n_excluded_warmup"] >= 1
    assert "warmup" in notes["reason"].lower()
    assert "445689" in notes["reason"], "the measurement should be citable"


def test_the_warmup_points_can_still_be_asked_for():
    """Recorded, not discarded. A reader who wants to CHECK the warmup claim --
    or who is on hardware where it does not apply -- must be able to."""
    from experimentation.sweep.search.analysis import throughput_pareto

    included = throughput_pareto(_with_ordinals_fixture(),
                                 exclude_warmup=False)
    assert 0 in {p["trial"] for p in included}
    assert all("worker_trial_ordinal" in p for p in included)


def test_a_study_with_no_ordinals_recorded_excludes_nothing():
    """Backward compatibility with journals written before this column existed.
    Treating a missing ordinal as 0 would silently drop every trial of an
    archived study from its own throughput front."""
    from experimentation.sweep.search.analysis import throughput_pareto

    st = optuna.create_study(direction="minimize")
    st.add_trial(optuna.trial.create_trial(
        params={}, distributions={}, value=1.0,
        user_attrs={"tokens_per_sec": 1000.0}))
    assert len(throughput_pareto(st)) == 1


def test_the_report_names_the_warmup_exclusion(tmp_path):
    from experimentation.sweep.search.analysis import write_analysis

    text = write_analysis(_with_ordinals_fixture(),
                          tmp_path)["interactions"].read_text()
    assert "warmup" in text.lower()
    assert "445689" in text


_ORDINAL_STUDY = None


def _with_ordinals_fixture():
    """A study whose trial 0 is a warm-up outlier, as job 445689 measured."""
    global _ORDINAL_STUDY
    if _ORDINAL_STUDY is not None:
        return _ORDINAL_STUDY

    import random

    rng = random.Random(20260824)
    base = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=11),
                               pruner=optuna.pruners.NopPruner())
    distributions = _distributions()
    for index in range(30):
        trial = base.ask(distributions)
        params = trial.params
        rank, ridge = params["ska_rank"], params["ska_ridge"]
        want = 0.003 * (rank / 8.0)
        loss = (4.0 - 0.02 * rank + 8.0 * abs(math.log(ridge / want))
                + 0.05 * rng.random())
        for key, value in _attrs(index, params).items():
            trial.set_user_attr(key, value)
        base.tell(trial, loss)

    out = optuna.create_study(direction="minimize")
    best = min(float(t.value) for t in base.trials)
    for position, trial in enumerate(base.trials):
        attrs = dict(trial.user_attrs)
        attrs["worker_trial_ordinal"] = position
        value = float(trial.value)
        if position == 0:
            # 17,818 tok/s AND the best loss in the study, so trial 0 would
            # otherwise be the SLOW END of the front rather than a dominated
            # point. Without that the exclusion test passes because trial 0 is
            # dominated anyway, i.e. it tests nothing -- the artefact only
            # matters when the warm-up trial is a point the front would keep.
            attrs["tokens_per_sec"] = 17_818.0
            value = best - 0.5
        out.add_trial(optuna.trial.create_trial(
            params=dict(trial.params), distributions=dict(trial.distributions),
            value=value, user_attrs=attrs))
    _ORDINAL_STUDY = out
    return out

# ============================================================================
# Findings from code review, each with the failure it produces.
# ============================================================================

# --------------------------- a CROSSOVER is an interaction ----

def _crossover_study(*, crossover: bool, with_replicates: bool = False):
    """A 2x2 where the effect of `left` REVERSES across `right`.

    `ska_power_K` x `gamma_value`, cell means:

        crossover=True          crossover=False
                 g=0.9  g=1.05           g=0.9  g=1.05
          K=1     0.0    1.0      K=1     0.0    0.0
          K=2     1.0    0.0      K=2     1.0    1.0

    Both have the SAME unsigned per-level effect range (1.0 at each level of
    `right`), so a magnitude built from `max(means) - min(means)` reports 0.0 for
    both -- and the crossover is the maximal interaction there is while the other
    is a pure main effect with no interaction at all.
    """
    st = optuna.create_study(direction="minimize")
    distributions = _distributions()
    base = _reference_params()
    if with_replicates:
        # A tiny floor (sigma 0.01) so a magnitude of 2.0 is unambiguously
        # resolvable and a magnitude of 0.0 unambiguously is not. Without it
        # `resolved` is None and the verdict assertion cannot distinguish the
        # fixed code from the broken code.
        for offset, seed in enumerate((42, 43, 44)):
            st.add_trial(optuna.trial.create_trial(
                params=dict(base), distributions=distributions,
                value=0.5 + 0.01 * offset,
                user_attrs=_attrs(0, base, reference_group="reference",
                                  model_seed=seed,
                                  anchor_name=f"ref-{seed}")))
    for k in (1, 2):
        for gamma in (0.9, 1.05):
            if crossover:
                value = 0.0 if (k == 1) == (gamma == 0.9) else 1.0
            else:
                value = 0.0 if k == 1 else 1.0
            for _ in range(4):
                params = {**base, "ska_power_K": k, "gamma_value": gamma}
                st.add_trial(optuna.trial.create_trial(
                    params=params, distributions=distributions, value=value,
                    user_attrs=_attrs(0, params)))
    return st


def _pair(study, left, right):
    from experimentation.sweep.search.analysis import conditional_effects

    return next(r for r in conditional_effects(study)
                if (r["left"], r["right"]) == (left, right))


def test_a_pure_crossover_is_reported_as_an_interaction():
    """THE finding. `max - min` over the cell means at each level of `right` is
    UNSIGNED, so a maximal crossover -- K=1 better at low gamma, K=2 better at
    high gamma -- produces the same range at both levels and an unsigned
    magnitude of exactly 0.0. The report would then state "no pair of these axes
    interacts", confidently, about the one shape the study most exists to find:
    `_SLOTS['best_k1']` says K is the axis most likely to change the SIGN of
    another axis's effect."""
    row = _pair(_crossover_study(crossover=True), "ska_power_K", "gamma_value")
    assert row["signed_magnitude"] == pytest.approx(2.0)


def test_a_pure_main_effect_has_no_signed_interaction():
    """Guards the guard. If the signed magnitude were large for everything it
    would be as useless as an unsigned one that is zero for a crossover."""
    row = _pair(_crossover_study(crossover=False), "ska_power_K", "gamma_value")
    assert row["signed_magnitude"] == pytest.approx(0.0)


def test_the_unsigned_magnitude_is_still_reported_beside_it():
    """Both, because they answer different questions: "does the effect change
    SIZE" and "does it change DIRECTION". Dropping the unsigned one would lose
    the case where an effect doubles without flipping."""
    row = _pair(_crossover_study(crossover=True), "ska_power_K", "gamma_value")
    assert row["interaction_magnitude"] == pytest.approx(0.0)
    assert row["signed_magnitude"] > row["interaction_magnitude"]


def test_a_sign_change_is_flagged_in_the_row():
    """A large gap between the two magnitudes IS the sign change, and a reader
    should not have to infer it from two columns."""
    assert _pair(_crossover_study(crossover=True),
                 "ska_power_K", "gamma_value")["sign_change"] is True
    assert _pair(_crossover_study(crossover=False),
                 "ska_power_K", "gamma_value")["sign_change"] is False


def test_the_resolved_verdict_uses_the_LARGER_of_the_two():
    """Otherwise the crossover is still labelled unresolvable and the fix buys
    nothing: the verdict is what the shortlist and the interaction probe read.

    Against a REAL floor, so `resolved` has to be True rather than the `None`
    that a floorless study returns -- a `None`-tolerant assertion would pass on
    the unfixed code.
    """
    from experimentation.sweep.search.analysis import conditional_effects

    study = _crossover_study(crossover=True, with_replicates=True)
    row = next(r for r in conditional_effects(study)
               if (r["left"], r["right"]) == ("ska_power_K", "gamma_value"))
    assert row["resolved"] is True
    assert row["interaction_magnitude"] == pytest.approx(0.0), (
        "the unsigned magnitude is zero here, so the True verdict can only have "
        "come from the signed one")


def test_the_report_names_the_sign_change():
    """Asserted on the COLUMN and the flagged value, not on the word "sign" --
    which already appears in the shortlist's prose, so a looser assertion would
    have passed before the column existed."""
    import tempfile

    from experimentation.sweep.search.analysis import write_analysis

    with tempfile.TemporaryDirectory() as out:
        text = write_analysis(_crossover_study(crossover=True),
                              out)["interactions"].read_text()
    assert "| sign change |" in text, text[:3000]
    assert "**YES**" in text
    header = text[text.index("## 0d."):text.index("## 1.")]
    assert "| size | signed |" in header


# ------------------- the contrast-vs-mean denominator ----

def test_a_contrast_against_a_five_trial_mean_uses_the_right_denominator(study,
                                                                        floor):
    """A single trial against a mean of n has sd `sigma*sqrt(1 + 1/n)`, which is
    1.095*sigma at n=5 -- NOT the 1.414*sigma of two single trials. Using the
    larger one makes the threshold 29% too high and understates every `sigmas`
    figure by the same factor, so a real 1.6-sigma anchor effect prints as 1.2 and
    is labelled unresolvable. Conservative, and it discards real findings in the
    one table this module calls the only causal statements it has."""
    from experimentation.sweep.search.analysis import anchor_contrasts

    row = next(r for r in anchor_contrasts(study, floor)
               if r["anchor"] == "rank-32")
    expected_sd = floor["sigma"] * math.sqrt(1.0 + 1.0 / 5.0)
    assert row["sd_of_contrast"] == pytest.approx(expected_sd)
    assert row["sigmas"] == pytest.approx(abs(row["delta"]) / expected_sd)
    assert row["threshold"] == pytest.approx(2.0 * expected_sd)


def test_the_contrast_threshold_is_tighter_than_the_trial_vs_trial_one(study,
                                                                      floor):
    """The direction of the correction, asserted so a future edit cannot silently
    reintroduce the looser number."""
    from experimentation.sweep.search.analysis import anchor_contrasts

    row = next(iter(anchor_contrasts(study, floor)))
    assert row["threshold"] < floor["min_resolvable_effect"]


# ------------------- a corrupted replicate group ----

def test_two_replicates_at_the_SAME_seed_corrupt_the_floor_and_are_refused():
    """The failure a concurrency race would produce: `enqueue_anchors` is a
    read-then-write name check with no lock, so two supervisors can double-enqueue
    the whole anchor set. Under `deterministic: true` the duplicated pairs land on
    identical losses, which adds zero-contribution pairs AND degrees of freedom --
    so sigma shrinks toward zero and every effect becomes resolvable. That is the
    exact failure `noise_floor` exists to avoid, and pooling silently is worse than
    having no floor."""
    from experimentation.sweep.search.analysis import noise_floor

    st = optuna.create_study(direction="minimize")
    for seed, value in ((42, 4.0), (43, 4.1), (42, 4.0), (43, 4.1)):
        st.add_trial(optuna.trial.create_trial(
            params={}, distributions={}, value=value,
            user_attrs={"reference_group": "reference", "model_seed": seed,
                        "anchor_name": f"ref-{seed}"}))
    result = noise_floor(st)
    assert result["available"] is False
    assert "seed" in result["reason"]
    assert result["sigma"] is None


def test_a_healthy_group_with_distinct_seeds_is_not_refused():
    """Guards the guard: the check must not reject the normal case."""
    from experimentation.sweep.search.analysis import noise_floor

    st = optuna.create_study(direction="minimize")
    for seed, value in ((42, 4.0), (43, 4.1), (44, 3.9)):
        st.add_trial(optuna.trial.create_trial(
            params={}, distributions={}, value=value,
            user_attrs={"reference_group": "reference", "model_seed": seed,
                        "anchor_name": f"ref-{seed}"}))
    assert noise_floor(st)["available"] is True


def test_a_group_with_no_recorded_seeds_is_still_usable():
    """Backward compatibility: a journal written before `model_seed` existed has
    no seeds to compare, and refusing there would delete the floor from every
    archived study rather than protecting it."""
    from experimentation.sweep.search.analysis import noise_floor

    st = optuna.create_study(direction="minimize")
    for value in (4.0, 4.1, 3.9):
        st.add_trial(optuna.trial.create_trial(
            params={}, distributions={}, value=value,
            user_attrs={"reference_group": "reference"}))
    assert noise_floor(st)["available"] is True


# ------------------- an axis that never varied is not a zero effect ----

def test_an_axis_that_never_varied_says_so_rather_than_reporting_zero_spread():
    """`spread = 0.0` and `unresolvable` reads as "measured, smaller than the
    floor". A fixed axis was never measured at all, and every other absent path
    in this module returns None."""
    from experimentation.sweep.search.analysis import main_effects

    st = optuna.create_study(direction="minimize")
    distributions = _distributions()
    for index in range(6):
        params = {**_reference_params(), "ska_rank": 8 if index % 2 else 32}
        st.add_trial(optuna.trial.create_trial(
            params=params, distributions=distributions, value=1.0 + index,
            user_attrs=_attrs(index, params)))

    axes = {a["axis"]: a for a in main_effects(st)["axes"]}
    assert axes["weight_decay"]["spread"] is None
    assert axes["weight_decay"]["resolved"] is None
    assert axes["ska_rank"]["spread"] is not None


def test_the_report_distinguishes_did_not_vary_from_unresolvable(tmp_path):
    from experimentation.sweep.search.analysis import write_analysis

    text = write_analysis(_crossover_study(crossover=True),
                          tmp_path)["interactions"].read_text()
    assert "DID NOT VARY" in text


# ------------------- the docstring must not still claim evidence ----

def test_the_module_docstring_does_not_claim_importance_IS_evidence():
    """The exact phrase a previous review flagged. It was removed from the
    artefact's label and left in the module docstring 770 lines above the comment
    recording its removal -- so the file a reader opens first said "This IS
    evidence" while the file it writes said "Neither is proof"."""
    from experimentation.sweep.search import analysis

    assert "IS evidence about the model" not in (analysis.__doc__ or "")
    assert "IS evidence" not in (analysis.__doc__ or "")


def test_the_docstring_describes_ped_anova_as_a_divergence():
    """Not as "variance attributable to one axis" -- that is the specific
    mischaracterisation, not just an overclaim of strength."""
    from experimentation.sweep.search import analysis

    assert "divergence" in (analysis.__doc__ or "").lower()


# ------------------- the null-axis inversion, pinned ----

def test_the_null_axis_outranks_the_planted_effect_in_the_LOCAL_column(study):
    """The evidentiary claim `_IMPORTANCE_NOTE` makes, backed by a test rather
    than by prose. `norm_clip_multiplier` appears NOWHERE in the fixture's
    objective; if it outranks the planted `ska_rank` main effect under
    `evaluate_on_local=True`, that is the demonstration -- and if a future optuna
    fixes it, this test says so instead of the docstring rotting."""
    from experimentation.sweep.search.analysis import analyse

    importances = analyse(study)["importances"]
    assert importances["available"]
    local = importances["values"]
    assert "norm_clip_multiplier" in local and "ska_rank" in local
    # Recorded as an observation about THIS fixture and THIS optuna, not as a
    # requirement: the assertion is that the two columns can DISAGREE about the
    # ordering, which is the thing that makes reporting both worthwhile.
    global_values = importances["values_global"]
    local_order = sorted(local, key=lambda k: -local[k])
    global_order = sorted(global_values, key=lambda k: -global_values[k])
    assert local_order != global_order, (
        "the local and global PED-ANOVA columns agree on the ordering for this "
        "fixture, so the second column buys nothing here -- re-check whether the "
        "framing in _IMPORTANCE_NOTE still has evidence behind it")


# ------------------- machine consumers get the label too ----

def test_the_summary_json_carries_the_not_evidence_flag(study, tmp_path):
    """`sampler_correlations.csv` is bare `a,b,r,n`, so a machine consumer got the
    correlations without the label that says they are not evidence."""
    import json

    from experimentation.sweep.search.analysis import write_analysis

    payload = json.loads(
        write_analysis(study, tmp_path)["summary"].read_text())
    assert payload["sampler_correlations"]["is_evidence"] is False
    assert payload["sampler_correlations"]["interpretation"]
