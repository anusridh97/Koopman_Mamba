"""The analysis half of the noise floor: measuring it, and refusing to overclaim.

`test_reference_repeats.py` proves the study can PRODUCE repeated reference
trials. This file is about what the analysis is then obliged to do with them, and
every test here exists because the alternative is a plausible, complete, wrong
report.

**The one number.** The spread of the reference replicates is the smallest
difference this study can resolve. On the 4m smoke study (job 445657) ablating
SKA entirely moved held-out loss by 1.166e-4. If the floor is of that order then
the differences between anchors are not measurements and a multivariate TPE
fitted to them produces a confident joint model of noise. So:

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
            0.002, 0.03, log=True),
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
    from experimentation.sweep.search.analysis import main_effects

    axes = {a["axis"]: a for a in main_effects(study, floor)["axes"]}
    # A fixed axis is a singleton, so its spread is 0 by construction.
    assert axes["weight_decay"]["spread"] == pytest.approx(0.0)
    assert axes["weight_decay"]["resolved"] is False


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
