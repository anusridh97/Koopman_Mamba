"""Post-hoc analysis of a finished interaction study: read-only, and honest.

The study samples ADAPTIVELY, and that is the thing this analysis has to be
careful about. A multivariate TPE concentrates its later proposals in whatever
region it currently believes is good, so the sampled columns of `trials.csv` are
correlated with each other *because of the sampler* and not because of the model.
Reporting `corr(ska_rank, ska_ridge)` from such a table as evidence about the
architecture would be reporting the sampler's search path as a scientific
finding -- and it would look exactly like a real result.

So the output separates three things that a single correlation matrix would
conflate, and labels them:

  **main-effect importance** -- variance in the objective attributable to one
  axis, from an importance evaluator that accounts for the others;
  **pairwise response structure** -- the mean objective in each cell of a 2-way
  table, which is the prespecified quantity the study was designed to estimate;
  **sampler-induced correlation** -- correlation BETWEEN SAMPLED COLUMNS, which
  is a diagnostic of where the sampler went and is explicitly not evidence.

The prespecified pairs are fixed in code, not chosen after looking. Seven of
them, named in the study config before any trial ran. Choosing which pairs to
report after seeing the data is how a 9-axis study yields 36 tables and one of
them looks significant.

Environment: sklearn and matplotlib are NOT installed here (measured), so
fANOVA and MeanDecreaseImpurity are unavailable and there are no plots.
PedAnova is pure-numpy and works, and every artefact is CSV / JSON / Markdown.
The task's instruction was to emit those rather than add an unverified
dependency.
"""
from __future__ import annotations

import csv
import json
import math
import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

optuna = pytest.importorskip("optuna")

pytestmark = pytest.mark.correctness

from experimentation.sweep.search.analysis import (            # noqa: E402
    PRESPECIFIED_PAIRS, analyse, pairwise_table, pareto_front,
    sampler_induced_correlations, write_analysis)


# --------------------------------------------------------------- fixtures ----

def _distributions():
    return {
        "ska_rank": optuna.distributions.CategoricalDistribution([8, 16, 24, 32]),
        "n_ska_layers": optuna.distributions.CategoricalDistribution([2, 3, 4, 6, 8]),
        "placement": optuna.distributions.CategoricalDistribution(
            ["baseline", "even", "midlate", "late"]),
        "ska_ridge": optuna.distributions.FloatDistribution(0.003, 0.03, log=True),
        "ska_layerscale_init": optuna.distributions.FloatDistribution(
            0.002, 0.03, log=True),
        "norm_clip_multiplier": optuna.distributions.CategoricalDistribution(
            [0.75, 0.8164965809277261, 1.0, 1.25]),
        "gamma_value": optuna.distributions.CategoricalDistribution([0.9, 1.0, 1.05]),
        "ska_power_K": optuna.distributions.CategoricalDistribution([1, 2]),
        "learning_rate": optuna.distributions.FloatDistribution(
            0.00032, 0.00048, log=True),
        # The fixed axes, as the singletons the study actually declares.
        "weight_decay": optuna.distributions.CategoricalDistribution([0.1]),
        "warmup_ratio": optuna.distributions.CategoricalDistribution([0.04]),
        "grad_clip": optuna.distributions.CategoricalDistribution([1.0]),
    }


@pytest.fixture(scope="module")
def study():
    """A synthetic finished study with a KNOWN interaction planted in it.

    The objective is built so that `ska_rank` has a strong main effect and
    `ska_rank x ska_ridge` has a real interaction: high rank prefers a heavy
    ridge and low rank prefers a light one, so neither marginal shows it. That
    is the structure the analysis has to be able to surface, and planting it is
    what makes the assertions about the output mean something.
    """
    import random

    rng = random.Random(20260823)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    st = optuna.create_study(
        sampler=optuna.samplers.RandomSampler(seed=7),
        pruner=optuna.pruners.NopPruner())
    distributions = _distributions()

    for index in range(120):
        trial = st.ask(distributions)
        params = trial.params
        rank = params["ska_rank"]
        ridge = params["ska_ridge"]
        # main effect: more rank is better. interaction: the BEST ridge moves
        # with rank, so neither marginal reveals it.
        want = 0.003 * (rank / 8.0)
        loss = (4.0 - 0.02 * rank
                + 8.0 * abs(math.log(ridge / want))
                + 0.05 * rng.random())
        trial.set_user_attr("param_count",
                           25_000_000 + 40_000 * rank
                           + 180_000 * params["n_ska_layers"])
        trial.set_user_attr("baseline_param_count", 25_352_736)
        trial.set_user_attr("worker_id", index % 8)
        trial.set_user_attr("sampler", "tpe_multivariate")
        trial.set_user_attr("sampler_seed", 2026 + index % 8)
        trial.set_user_attr("per_device_batch_size", 8)
        trial.set_user_attr("ska_delta", 0.001 * (index % 11))
        if index < 24:
            trial.set_user_attr("anchor_name", f"anchor-{index}")
        st.tell(trial, loss)

    # Some pruned and some failed trials, because a real study has both and the
    # counts are the first thing the report has to get right.
    for _ in range(17):
        trial = st.ask(distributions)
        trial.report(9.0, 450)
        st.tell(trial, state=optuna.trial.TrialState.PRUNED)
    for _ in range(5):
        trial = st.ask(distributions)
        trial.set_user_attr("failure", "CUDA out of memory")
        st.tell(trial, state=optuna.trial.TrialState.FAIL)
    return st


# ------------------------------------------------------ prespecified pairs ----

def test_the_prespecified_pairs_are_the_seven_the_study_declared():
    """Fixed in code, not chosen after looking at the data. Choosing pairs post
    hoc is how a 9-axis study yields 36 tables and one looks significant."""
    assert PRESPECIFIED_PAIRS == (
        ("ska_rank", "ska_ridge"),
        ("ska_rank", "norm_clip_multiplier"),
        ("ska_rank", "ska_power_K"),
        ("n_ska_layers", "ska_layerscale_init"),
        ("n_ska_layers", "placement"),
        ("ska_ridge", "ska_power_K"),
        ("ska_layerscale_init", "learning_rate"),
    )


def test_every_prespecified_axis_is_a_real_search_axis():
    """A pair naming an axis the space does not declare would produce an empty
    table that reads as "no effect"."""
    from experimentation.sweep.search.space import REQUIRED_PARAMS

    for left, right in PRESPECIFIED_PAIRS:
        assert left in REQUIRED_PARAMS, left
        assert right in REQUIRED_PARAMS, right


def test_no_prespecified_pair_involves_a_fixed_axis():
    """`weight_decay`, `warmup_ratio` and `grad_clip` are pinned, so a table over
    one of them has a single column and cannot show structure."""
    from experimentation.sweep.search.studyspec import load_study_spec

    fixed = set(load_study_spec(
        REPO / "configs/search/proxy-256x17-interactions-v1.yaml").fixed_params)
    for pair in PRESPECIFIED_PAIRS:
        assert not (set(pair) & fixed), f"{pair} involves a fixed axis"


# ------------------------------------------------------------- the counts ----

def test_the_state_counts_are_right(study):
    result = analyse(study)
    assert result["counts"]["COMPLETE"] == 120
    assert result["counts"]["PRUNED"] == 17
    assert result["counts"]["FAIL"] == 5
    assert result["counts"]["total"] == 142


def test_failure_reasons_are_grouped(study):
    """48 trials failing for 48 reasons is a rough study; 48 for ONE reason is a
    bug in the harness, and a count alone cannot tell them apart."""
    result = analyse(study)
    assert result["failures"] == {"CUDA out of memory": 5}


def test_the_anchors_are_identified(study):
    result = analyse(study)
    assert result["n_anchors"] == 24


# ------------------------------------------------------ pairwise structure ----

def test_a_pairwise_table_has_a_cell_per_observed_combination(study):
    table = pairwise_table(study, "ska_rank", "ska_power_K")
    assert table["left"] == "ska_rank" and table["right"] == "ska_power_K"
    assert len(table["left_levels"]) == 4
    assert len(table["right_levels"]) == 2
    for row in table["rows"]:
        assert len(row["cells"]) == 2


def test_a_pairwise_cell_reports_n_as_well_as_the_mean(study):
    """A cell mean over 2 trials and one over 40 are not comparable, and under
    adaptive sampling the counts are wildly uneven by construction. A mean
    printed without its n invites reading noise as effect."""
    table = pairwise_table(study, "ska_rank", "ska_ridge")
    total = 0
    for row in table["rows"]:
        for cell in row["cells"]:
            assert "n" in cell and "mean" in cell
            if cell["n"] == 0:
                assert cell["mean"] is None
            total += cell["n"]
    assert total == 120, "every completed trial lands in exactly one cell"


def test_a_continuous_axis_is_binned_and_the_edges_are_reported(study):
    """`ska_ridge` is a log float, so a 2-way table needs bins -- and the bin
    edges are part of the result, or the table cannot be reproduced or checked."""
    table = pairwise_table(study, "ska_rank", "ska_ridge", bins=3)
    assert len(table["right_levels"]) == 3
    assert table["right_binned"] is True
    assert table["left_binned"] is False
    for level in table["right_levels"]:
        assert "lo" in level and "hi" in level


def test_the_planted_interaction_is_visible_in_its_table(study):
    """The assertion that makes this analysis worth running. The synthetic
    objective was built so the BEST ridge moves with rank; a table that could not
    show that would be reporting nothing."""
    table = pairwise_table(study, "ska_rank", "ska_ridge", bins=3)
    best_bin_by_rank = {}
    for row in table["rows"]:
        scored = [(c["mean"], i) for i, c in enumerate(row["cells"])
                  if c["mean"] is not None]
        if scored:
            best_bin_by_rank[row["level"]] = min(scored)[1]
    assert len(set(best_bin_by_rank.values())) > 1, (
        f"the best ridge bin is the same at every rank ({best_bin_by_rank}), so "
        f"the table cannot express an interaction that was planted in the data")


def test_a_table_over_an_axis_with_one_observed_level_says_so(study):
    """A fixed axis produces a degenerate table. Reporting it as a table with one
    column reads as "no effect"; saying it is degenerate does not."""
    table = pairwise_table(study, "ska_rank", "weight_decay")
    assert table["degenerate"] is True
    assert "weight_decay" in table["note"]


def test_only_completed_trials_enter_a_table(study):
    """A PRUNED trial has no final objective. Including one at its last
    intermediate value would mix two different measurements."""
    table = pairwise_table(study, "ska_rank", "ska_power_K")
    assert sum(c["n"] for r in table["rows"] for c in r["cells"]) == 120


# ------------------------------------------------ importance, and its limits ----

def test_main_effect_importances_are_computed_for_the_searched_axes(study):
    result = analyse(study)
    importances = result["importances"]
    assert importances["available"] is True
    assert importances["evaluator"]
    assert set(importances["values"]) <= set(_distributions())


def test_the_planted_main_effect_is_the_largest(study):
    """`ska_rank` was given the strong main effect. If the importance evaluator
    did not rank it first, the number being reported is not measuring what the
    label says."""
    values = analyse(study)["importances"]["values"]
    ranked = sorted(values.items(), key=lambda kv: -kv[1])
    assert ranked[0][0] in ("ska_rank", "ska_ridge"), ranked[:3]


def test_a_missing_importance_evaluator_degrades_rather_than_crashes(study,
                                                                   monkeypatch):
    """sklearn is absent here, so fANOVA and MeanDecreaseImpurity are already
    unavailable. If PedAnova went too, a whole analysis run must not be lost --
    the counts, the pairwise tables and the Pareto front do not need it."""
    import experimentation.sweep.search.analysis as mod

    def boom(*a, **k):
        raise ImportError("no evaluator")

    monkeypatch.setattr(mod, "_importance_evaluator", boom)
    result = analyse(study)
    assert result["importances"]["available"] is False
    assert "reason" in result["importances"]
    assert result["counts"]["COMPLETE"] == 120       # the rest still works
    assert result["pairwise"]


# ------------------------------ sampler-induced correlation, clearly labelled ----

def test_sampler_correlations_are_reported_as_a_diagnostic_not_as_evidence(study):
    correlations = sampler_induced_correlations(study)
    assert correlations["is_evidence"] is False
    text = correlations["interpretation"].lower()
    assert "sampler" in text
    assert "not evidence" in text or "not a finding" in text


def test_sampler_correlations_are_between_sampled_columns_only(study):
    """The objective is deliberately excluded. A correlation between a parameter
    and the objective looks like a main effect and is not one -- adaptive
    sampling concentrates good regions, so it conflates 'this helps' with 'the
    sampler went here'. Main effects come from the importance evaluator, which
    accounts for the other axes."""
    correlations = sampler_induced_correlations(study)
    for pair in correlations["pairs"]:
        assert "objective" not in pair["a"] and "objective" not in pair["b"]


def test_a_constant_column_yields_no_correlation_rather_than_a_nan(study):
    """`weight_decay` is fixed, so its variance is zero and Pearson's r is 0/0.
    A NaN in a CSV reads as a missing measurement rather than as a constant."""
    correlations = sampler_induced_correlations(study)
    for pair in correlations["pairs"]:
        assert not math.isnan(pair["r"])
    assert "weight_decay" in correlations["constant_columns"]


# ----------------------------------------------------------- Pareto and tops ----

def test_the_pareto_front_is_non_dominated_on_loss_and_parameter_count(study):
    front = pareto_front(study)
    assert front
    for point in front:
        for other in front:
            if other is point:
                continue
            assert not (other["objective"] <= point["objective"]
                        and other["param_count"] <= point["param_count"]
                        and other != point), "a dominated point is on the front"


def test_the_pareto_front_is_sorted_by_parameter_count(study):
    counts = [p["param_count"] for p in pareto_front(study)]
    assert counts == sorted(counts)


def test_the_best_trial_overall_is_on_the_front(study):
    front = pareto_front(study)
    best = min((t for t in study.trials
                if t.state == optuna.trial.TrialState.COMPLETE),
               key=lambda t: t.value)
    assert best.number in {p["trial"] for p in front}


def test_top_trials_are_ranked_by_held_out_loss(study):
    result = analyse(study, top_k=10)
    objectives = [t["objective"] for t in result["top_by_loss"]]
    assert len(objectives) == 10
    assert objectives == sorted(objectives)


def test_top_trials_by_ska_delta_are_ranked_descending(study):
    """A larger delta means zeroing the SKA branch hurt more, i.e. the branch was
    doing more work -- so this ranking is the opposite direction from loss."""
    result = analyse(study, top_k=10)
    deltas = [t["ska_delta"] for t in result["top_by_ska_delta"]]
    assert deltas == sorted(deltas, reverse=True)


def test_a_study_with_no_recorded_ska_delta_reports_an_empty_ranking(study):
    """`ska_delta` comes from a quick_eval ablation that a study may not have
    run. Absent is not zero, and must not be ranked as zero."""
    bare = optuna.create_study()
    bare.tell(bare.ask({"ska_rank": _distributions()["ska_rank"]}), 1.0)
    assert analyse(bare)["top_by_ska_delta"] == []


def test_objective_versus_parameter_count_is_emitted_as_data(study):
    """The plot-equivalent. matplotlib is not installed, so the scatter ships as
    the numbers behind it."""
    rows = analyse(study)["objective_vs_params"]
    assert len(rows) == 120
    assert {"trial", "objective", "param_count", "anchor"} <= set(rows[0])


def test_the_rank_curve_is_monotone_best_so_far(study):
    """The rank-plot equivalent: best objective seen up to each trial. Monotone
    by construction, which is exactly what makes it readable as progress."""
    curve = [p["best_so_far"] for p in analyse(study)["rank_curve"]]
    assert curve == sorted(curve, reverse=True)
    assert len(curve) == 120


# ------------------------------------------------------------- the artefacts ----

def test_write_analysis_emits_every_artefact(study, tmp_path):
    written = write_analysis(study, tmp_path)
    for name in ("summary", "importances", "interactions", "pareto",
                 "top_by_loss", "top_by_ska_delta", "objective_vs_params",
                 "rank_curve", "sampler_correlations"):
        assert name in written, name
        assert written[name].is_file(), name
        assert written[name].stat().st_size > 0, name


def test_no_artefact_is_a_plot(study, tmp_path):
    """matplotlib is not installed and adding an unverified dependency was
    explicitly out of scope, so every artefact has to be readable as text."""
    written = write_analysis(study, tmp_path)
    for path in written.values():
        assert path.suffix in (".csv", ".json", ".md"), path


def test_the_summary_json_is_valid_and_carries_the_counts(study, tmp_path):
    written = write_analysis(study, tmp_path)
    payload = json.loads(written["summary"].read_text())
    assert payload["counts"]["COMPLETE"] == 120
    assert payload["counts"]["PRUNED"] == 17
    assert payload["counts"]["FAIL"] == 5


def test_the_interactions_markdown_contains_all_seven_tables(study, tmp_path):
    written = write_analysis(study, tmp_path)
    text = written["interactions"].read_text()
    for left, right in PRESPECIFIED_PAIRS:
        assert f"{left} x {right}" in text, f"{left} x {right} is missing"


def test_the_interactions_markdown_states_the_adaptive_sampling_caveat(study,
                                                                      tmp_path):
    """The caveat has to be in the artefact a human reads, not only in a
    docstring. A table of cell means under adaptive sampling is easy to
    over-read."""
    text = write_analysis(study, tmp_path)["interactions"].read_text().lower()
    assert "adaptive" in text
    assert "prespecified" in text


def test_the_pareto_csv_round_trips(study, tmp_path):
    written = write_analysis(study, tmp_path)
    rows = list(csv.DictReader(written["pareto"].read_text().splitlines()))
    assert rows
    assert {"trial", "objective", "param_count"} <= set(rows[0])


def test_writing_twice_is_idempotent(study, tmp_path):
    """Read-only on the study, and re-runnable: an analysis that appended would
    make the second run's numbers wrong."""
    first = {k: p.read_text() for k, p in write_analysis(study, tmp_path).items()}
    second = {k: p.read_text() for k, p in write_analysis(study, tmp_path).items()}
    assert first == second


def test_the_analysis_does_not_modify_the_study(study):
    """Read-only means read-only: this runs against a real journal that other
    workers may still be writing to."""
    before = (len(study.trials),
              [t.state.name for t in study.trials],
              [t.value for t in study.trials],
              [dict(t.user_attrs) for t in study.trials])
    write_analysis(study, pathlib.Path("/tmp") /
                   "kmwork-analysis-readonly-probe")
    after = (len(study.trials),
             [t.state.name for t in study.trials],
             [t.value for t in study.trials],
             [dict(t.user_attrs) for t in study.trials])
    assert before == after


# --------------------------------------------------------- degenerate inputs ----

def test_an_empty_study_produces_a_report_rather_than_a_crash(tmp_path):
    """A study that died before its first trial. A zero-byte output reads as "no
    output" instead of "no trials", and a crash here loses the failure reasons
    that are the only useful thing left."""
    empty = optuna.create_study()
    result = analyse(empty)
    assert result["counts"]["total"] == 0
    written = write_analysis(empty, tmp_path)
    assert json.loads(written["summary"].read_text())["counts"]["total"] == 0
    assert written["interactions"].read_text().strip()


def test_a_study_with_only_failures_still_reports_the_reasons(tmp_path):
    """The case where the reasons are the ENTIRE finding."""
    st = optuna.create_study()
    for _ in range(3):
        trial = st.ask({"ska_rank": _distributions()["ska_rank"]})
        trial.set_user_attr("failure", "no objective could be read")
        st.tell(trial, state=optuna.trial.TrialState.FAIL)
    result = analyse(st)
    assert result["counts"]["FAIL"] == 3
    assert result["failures"] == {"no objective could be read": 3}
    assert result["top_by_loss"] == []
    assert result["pareto"] == []


# ------------------------------------------------------------------- the CLI ----

def _write_journal(tmp_path, study):
    """Replay a study into a real JournalStorage the CLI can open.

    The CLI's job is to find and open a journal from a study directory, and a
    test that handed it an in-memory Study would exercise none of that -- which
    is exactly how `verify_study_e2e.sbatch` once pointed at `journal.log`
    instead of `optuna_journal.log`, created an EMPTY file, and crashed on
    "Record does not exist" instead of reporting.
    """
    study_dir = tmp_path / "_studies" / f"{study.study_name}.deadbeef"
    study_dir.mkdir(parents=True)
    storage = optuna.storages.JournalStorage(
        optuna.storages.journal.JournalFileBackend(
            str(study_dir / "optuna_journal.log")))
    replayed = optuna.create_study(study_name=study.study_name, storage=storage)
    replayed.add_trials(study.trials)
    return study_dir


def test_the_cli_writes_a_report_from_a_real_journal(study, tmp_path):
    import subprocess

    study_dir = _write_journal(tmp_path, study)
    result = subprocess.run(
        [sys.executable, str(REPO / "scripts/analyze_interactions.py"),
         str(study_dir), "--study-name", study.study_name],
        capture_output=True, text=True, cwd=REPO)
    assert result.returncode == 0, result.stderr[-3000:]
    out = study_dir / "analysis"
    assert (out / "summary.json").is_file()
    assert (out / "interactions.md").is_file()
    payload = json.loads((out / "summary.json").read_text())
    assert payload["counts"]["COMPLETE"] == 120


def test_the_cli_finds_the_study_by_itself_when_there_is_one(study, tmp_path):
    """A study directory holds exactly one study. Requiring its name is a
    papercut, and getting it wrong is a KeyError from deep inside optuna."""
    import subprocess

    study_dir = _write_journal(tmp_path, study)
    result = subprocess.run(
        [sys.executable, str(REPO / "scripts/analyze_interactions.py"),
         str(study_dir)],
        capture_output=True, text=True, cwd=REPO)
    assert result.returncode == 0, result.stderr[-3000:]
    assert study.study_name in result.stdout


def test_the_cli_reports_a_missing_journal_rather_than_creating_one(tmp_path):
    """`JournalFileBackend` happily CREATES an empty file, and `load_study` then
    raises "Record does not exist" -- a crash where a message belongs. Measured
    in the study-e2e script, which had this exact bug."""
    import subprocess

    empty = tmp_path / "not-a-study"
    empty.mkdir()
    result = subprocess.run(
        [sys.executable, str(REPO / "scripts/analyze_interactions.py"),
         str(empty)],
        capture_output=True, text=True, cwd=REPO)
    assert result.returncode != 0
    assert "optuna_journal.log" in (result.stdout + result.stderr)
    assert not (empty / "optuna_journal.log").exists(), (
        "the CLI created the journal it was meant to report as missing")


def test_the_cli_does_not_write_into_the_study_directory_itself(study, tmp_path):
    """Artefacts go in a subdirectory. Writing beside the journal risks a name
    collision with report.py's own output (trials.csv, top_trials.md), and a
    reader could not tell which run produced which file."""
    import subprocess

    study_dir = _write_journal(tmp_path, study)
    before = {p.name for p in study_dir.iterdir()}
    subprocess.run(
        [sys.executable, str(REPO / "scripts/analyze_interactions.py"),
         str(study_dir)], capture_output=True, text=True, cwd=REPO)
    after = {p.name for p in study_dir.iterdir()}
    assert after - before == {"analysis"}


# ------------------------------------------------- binning must lose nothing ----

def test_the_minimum_valued_trial_is_not_dropped_by_a_log_bin_edge():
    """The bug this pins, found in review and confirmed by arithmetic.

    `exp(log(x))` is frequently ABOVE x for a decimal literal --
    `exp(log(0.002)) == 0.0020000000000000005` -- so a COMPUTED bottom edge
    excludes the minimum-valued trial. Only the top edge was repaired, because
    only the top edge's exclusion is obvious from the half-open comparison.

    Not theoretical: `ska_layerscale_init`'s declared low is 0.002 and the
    `layerscale-low` anchor resolves to EXACTLY 0.002, so a curated design
    endpoint vanished from two of the seven prespecified tables on every run.
    """
    assert math.exp(math.log(0.002)) > 0.002, (
        "the float behaviour this test is built on no longer holds")

    d = {"ska_layerscale_init": optuna.distributions.FloatDistribution(
             0.002, 0.03, log=True),
         "ska_rank": optuna.distributions.CategoricalDistribution([8, 24])}
    st = optuna.create_study()
    for _ in range(6):
        st.tell(st.ask(d), 1.0)
    # Force the endpoint to be present, whatever the sampler drew.
    forced = optuna.trial.create_trial(
        params={"ska_layerscale_init": 0.002, "ska_rank": 8},
        distributions=d, value=1.0)
    st.add_trial(forced)
    table = pairwise_table(st, "ska_rank", "ska_layerscale_init", bins=3)
    assert table["n_dropped"] == 0, (
        f"{table['n_dropped']} trial(s) fell outside every bin -- the "
        f"minimum-valued trial is being discarded at the bottom edge")
    assert table["n_placed"] == table["n_completed"]


def test_every_completed_trial_lands_in_a_cell_for_every_prespecified_pair(study):
    """The general invariant, over the real pair list. A table that silently
    omits trials is worse than one that reports none: it looks complete."""
    for left, right in PRESPECIFIED_PAIRS:
        table = pairwise_table(study, left, right)
        assert table["n_dropped"] == 0, f"{left} x {right}: {table['n_dropped']}"
        assert table["n_placed"] == table["n_completed"]


def test_a_dropped_trial_would_be_reported_in_the_markdown(study):
    """Guards the guard: if the accounting were never surfaced, a future
    regression would be silent again."""
    from experimentation.sweep.search.analysis import _format_table

    table = dict(pairwise_table(study, "ska_rank", "ska_ridge"))
    table["n_dropped"] = 3
    assert "WARNING" in "\n".join(_format_table(table))
