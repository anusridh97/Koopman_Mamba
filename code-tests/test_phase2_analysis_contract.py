"""Pure, offline contract tests for Phase 2a analysis artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import pytest

from koopman_lm.experiments.phase2.analysis import (
    CONTROLLER_WARNING,
    _parser,
    build_wandb_report_payload,
    conditional_trial_views,
    falsified_hypotheses_report,
    load_result_summaries,
    merge_record_metadata,
    pareto_front,
    promotion_shortlist,
    retrospective_control_false_pruning_risk,
    seeded_stability_summary,
    select_trials_for_importance,
    validate_wandb_trial_mapping,
)
from koopman_lm.experiments.phase2.manifest import (
    build_trial_manifest,
    reference_parameters,
)
from koopman_lm.experiments.phase2.spec import load_spec


pytestmark = pytest.mark.correctness
ROOT = Path(__file__).resolve().parents[1]


def _record(
    token: str,
    mqar: float,
    ppl: float,
    *,
    rank: int,
    placement: str,
    objective: str = "next_token_only",
    health: bool | None = True,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "schema_version": 1,
        "trial_hash": token * 64,
        "status": "COMPLETE",
        "mqar_accuracy": mqar,
        "wikitext_ppl": ppl,
        "parameters": {
            "architecture_mode": "updated_sweep",
            "ska_rank": rank,
            "ska_fraction": 0.25,
            "ska_placement": placement,
            "ska_chunk_size": 64,
            "qk_norm": rank >= 96,
            "objective_arm": objective,
        },
    }
    if health is not None:
        record["health_passed"] = health
    return record


@dataclass
class _FakeTrial:
    value: float | None
    params: dict[str, Any]
    user_attrs: dict[str, Any] = field(default_factory=dict)


def test_conditional_views_retain_birdie_only_parameters_and_skip_missing_ppl():
    trials = [
        _FakeTrial(
            0.8,
            {
                "objective_arm": "birdie_mix",
                "birdie_retrieval_fraction": 0.2,
                "birdie_copy_share": 0.5,
                "architecture_mode": "updated_sweep",
            },
            {"wikitext_ppl": 20.0},
        ),
        _FakeTrial(
            0.7,
            {
                "objective_arm": "birdie_mix",
                "birdie_retrieval_fraction": 0.4,
                "birdie_copy_share": 0.75,
                "architecture_mode": "updated_sweep",
            },
            {},
        ),
        _FakeTrial(
            0.6,
            {
                "objective_arm": "next_token_only",
                "architecture_mode": "paper_control",
            },
            {"wikitext_ppl": 25.0},
        ),
    ]

    views = conditional_trial_views(trials)

    assert len(views["global"]) == 3
    assert len(views["birdie_only"]) == 2
    assert all(
        "birdie_retrieval_fraction" in trial.params
        for trial in views["birdie_only"]
    )
    assert len(views["updated_sweep_only"]) == 2
    assert len(select_trials_for_importance(views["global"], "mqar_accuracy")) == 3
    assert len(select_trials_for_importance(views["global"], "wikitext_ppl")) == 2


def test_promotion_shortlist_is_stable_diverse_and_deterministic():
    records = [
        _record("a", 0.95, 22.0, rank=32, placement="uniform"),
        _record("b", 0.94, 21.0, rank=48, placement="uniform"),
        _record(
            "c",
            0.92,
            18.0,
            rank=128,
            placement="middle_clustered",
            objective="birdie_mix",
        ),
        _record("d", 0.90, 15.0, rank=32, placement="uniform"),
        _record(
            "e",
            0.97,
            24.0,
            rank=96,
            placement="late_biased",
            health=False,
        ),
    ]

    first = promotion_shortlist(records, cap=3)
    second = promotion_shortlist(list(reversed(records)), cap=3)
    selected = [item["trial_hash"] for item in first["selected"]]

    assert selected == [item["trial_hash"] for item in second["selected"]]
    assert selected[0] == "a" * 64
    assert "d" * 64 in selected  # PPL anchor
    assert "c" * 64 in selected  # diverse over the near-duplicate b arm
    assert "b" * 64 not in selected
    assert "e" * 64 not in selected
    assert first["candidate_counts"]["explicitly_unstable"] == 1
    assert first["selected"][0]["selection_reason"] == "highest_mqar_anchor"


def test_promotion_can_require_positive_stability_evidence():
    passed = _record("a", 0.8, 20.0, rank=32, placement="uniform")
    unknown = _record(
        "b", 0.9, 21.0, rank=64, placement="late_biased", health=None
    )

    permissive = promotion_shortlist([passed, unknown], cap=10)
    strict = promotion_shortlist(
        [passed, unknown], cap=10, require_stability_evidence=True
    )

    assert permissive["candidate_counts"]["eligible"] == 2
    assert strict["candidate_counts"]["eligible"] == 1
    assert strict["selected"][0]["trial_hash"] == "a" * 64


def test_promotion_shortlist_deduplicates_seed_specific_configuration_runs():
    first = _record("a", 0.8, 20.0, rank=48, placement="uniform")
    second = _record("b", 0.81, 19.5, rank=48, placement="uniform")
    other = _record("c", 0.79, 18.0, rank=96, placement="late_biased")
    for seed, record in ((42, first), (43, second)):
        record["promotion_config_hash"] = "f" * 64
        record["seed"] = seed
    other["promotion_config_hash"] = "e" * 64
    other["seed"] = 42

    shortlist = promotion_shortlist([second, other, first], cap=10)

    hashes = [
        record["promotion_config_hash"] for record in shortlist["selected"]
    ]
    assert len(hashes) == len(set(hashes))
    assert shortlist["candidate_counts"]["eligible_seed_specific_runs"] == 3
    assert shortlist["candidate_counts"]["eligible_unique_configurations"] == 2


def test_promotion_shortlist_excludes_all_fixed_controls():
    candidate = _record("a", 0.70, 20.0, rank=48, placement="uniform")
    updated_control = _record(
        "b", 0.99, 10.0, rank=48, placement="uniform"
    )
    updated_control["parameters"]["name"] = "updated_default_control"
    paper_control = _record(
        "c", 0.98, 11.0, rank=48, placement="middle_clustered"
    )
    paper_control["parameters"].update(
        {"name": "paper_control", "architecture_mode": "paper_control"}
    )
    mamba_control = _record(
        "d", 0.97, 12.0, rank=32, placement="uniform"
    )
    mamba_control["parameters"].update(
        {"name": "mamba_only_control", "architecture_mode": "mamba_only"}
    )

    shortlist = promotion_shortlist(
        [updated_control, paper_control, mamba_control, candidate],
        cap=10,
    )

    assert [item["trial_hash"] for item in shortlist["selected"]] == [
        candidate["trial_hash"]
    ]
    counts = shortlist["candidate_counts"]
    assert counts["updated_sweep_unnamed_candidates"] == 1
    assert counts["excluded_named_controls"] == 1
    assert counts["excluded_non_updated_sweep"] == 2
    assert shortlist["selection_policy"]["fixed_controls_excluded"] is True


def test_retrospective_control_pruning_risk_uses_candidate_step_500_median():
    candidates = [
        _record("a", 0.8, 20.0, rank=32, placement="uniform"),
        _record("b", 0.8, 20.0, rank=48, placement="uniform"),
        _record("c", 0.8, 20.0, rank=96, placement="late_biased"),
    ]
    for record, step_500 in zip(candidates, (0.4, 0.6, 0.8)):
        record["step_500_mqar_accuracy"] = step_500

    risky = _record("d", 0.8, 20.0, rank=48, placement="uniform")
    risky["parameters"]["name"] = "updated_default_control"
    risky["step_500_mqar_accuracy"] = 0.5
    safe = _record("e", 0.8, 20.0, rank=48, placement="middle_clustered")
    safe["parameters"].update(
        {"name": "paper_control", "architecture_mode": "paper_control"}
    )
    safe["step_500_mqar_accuracy"] = 0.7
    missing = _record("f", 0.8, 20.0, rank=32, placement="uniform")
    missing["parameters"].update(
        {"name": "mamba_only_control", "architecture_mode": "mamba_only"}
    )

    report = retrospective_control_false_pruning_risk(
        [*candidates, risky, safe, missing]
    )

    assert report["analysis_scope"] == "retrospective_diagnostic_only"
    assert report["not_an_actual_pruner_replay"] is True
    assert "does not assert" in report["warning"]
    assert (
        report["candidate_reference"]["step_500_mqar_accuracy_median"]
        == pytest.approx(0.6)
    )
    by_name = {item["control_name"]: item for item in report["controls"]}
    assert (
        by_name["updated_default_control"]["retrospective_risk_label"]
        == "potential_false_pruning_risk"
    )
    assert by_name["updated_default_control"]["below_candidate_median_count"] == 1
    assert (
        by_name["paper_control"]["retrospective_risk_label"]
        == "no_below_median_signal"
    )
    assert (
        by_name["mamba_only_control"]["retrospective_risk_label"]
        == "insufficient_control_step_500_evidence"
    )


def test_seeded_confirmation_requires_all_three_healthy_seeds():
    records = []
    for seed, mqar, ppl in (
        (42, 0.80, 20.0),
        (43, 0.82, 21.0),
        (44, 0.78, 19.0),
    ):
        record = _record(
            str(seed)[-1],
            mqar,
            ppl,
            rank=48,
            placement="uniform",
        )
        record["promotion_config_hash"] = "f" * 64
        record["seed"] = seed
        records.append(record)
    incomplete = _record("a", 0.9, 22.0, rank=96, placement="late_biased")
    incomplete["promotion_config_hash"] = "e" * 64
    incomplete["seed"] = 42
    records.append(incomplete)

    result = seeded_stability_summary(records)
    assert result["confirmed_configuration_count"] == 1
    confirmed = result["confirmed_records"][0]
    assert confirmed["promotion_config_hash"] == "f" * 64
    assert confirmed["mqar_accuracy"] == pytest.approx(0.8)
    incomplete_summary = next(
        item
        for item in result["configurations"]
        if item["promotion_config_hash"] == "e" * 64
    )
    assert incomplete_summary["missing_seeds"] == [43, 44]


def test_analysis_cli_requires_stability_by_default():
    parser = _parser()

    strict = parser.parse_args(
        [
            "--results-root",
            "results",
            "--storage",
            "postgresql+psycopg://example.invalid/phase2",
            "--study-name",
            "echo-phase2a-1m-v1",
            "--output",
            "out.json",
        ]
    )
    exploratory = parser.parse_args(
        [
            "--results-root",
            "results",
            "--storage",
            "postgresql+psycopg://example.invalid/phase2",
            "--study-name",
            "echo-phase2a-1m-v1",
            "--allow-missing-stability-evidence",
            "--output",
            "out.json",
        ]
    )

    assert strict.require_stability_evidence is True
    assert exploratory.require_stability_evidence is False


def test_hypothesis_scaffold_never_auto_declares_falsification():
    records = [
        _record("a", 0.8, 20.0, rank=32, placement="uniform"),
        _record(
            "b",
            0.7,
            18.0,
            rank=96,
            placement="middle_clustered",
            objective="birdie_mix",
        ),
    ]
    records[1]["parameters"].update(
        {"birdie_retrieval_fraction": 0.3, "birdie_copy_share": 0.5}
    )

    report = falsified_hypotheses_report(records)

    assert report["report_status"] == "scientist_review_required"
    assert report["hypotheses"]
    assert all(item["falsified"] is None for item in report["hypotheses"])
    rank = next(
        item for item in report["hypotheses"] if item["id"] == "rank_scales_with_width"
    )
    assert rank["review_status"] == "unreviewed"
    assert rank["observed_values"]["ska_rank"] == [32, 96]


def test_wandb_payload_is_offline_and_table_ready():
    records = [
        _record("a", 0.9, 20.0, rank=32, placement="uniform"),
        _record("b", 0.8, 15.0, rank=96, placement="late_biased"),
    ]
    front = pareto_front(records)
    shortlist = promotion_shortlist(records, cap=2)
    hypotheses = falsified_hypotheses_report(records)
    records[0]["step_500_mqar_accuracy"] = 0.4
    control = _record("c", 0.7, 21.0, rank=48, placement="uniform")
    control["parameters"]["name"] = "updated_default_control"
    control["step_500_mqar_accuracy"] = 0.3
    control_risk = retrospective_control_false_pruning_risk(
        [*records, control]
    )
    fanova = {
        "views": {
            "birdie_only": {
                "mqar_accuracy": {
                    "importance": {"birdie_copy_share": 0.75},
                    "n_trials_with_target": 12,
                }
            }
        }
    }

    payload = build_wandb_report_payload(
        front,
        shortlist,
        fanova,
        hypotheses,
        control_pruning_risk=control_risk,
    )

    assert payload["network_write_performed"] is False
    assert payload["publication_status"] == "manual_authorized_upload_required"
    assert payload["published_report_url"] is None
    assert payload["sampling_warning"] == CONTROLLER_WARNING
    assert payload["tables"]["pareto_front"]["columns"]
    assert len(payload["tables"]["promotion_shortlist"]["data"]) == 2
    assert payload["tables"]["fanova_importance"]["data"] == [
        ["birdie_only", "mqar_accuracy", "birdie_copy_share", 0.75, 12]
    ]
    control_table = payload["tables"][
        "retrospective_control_false_pruning_risk"
    ]
    assert control_table["data"][0][0] == "retrospective_diagnostic_only"
    assert control_table["data"][0][1] == "updated_default_control"
    assert any(
        panel["title"].startswith("RETROSPECTIVE ONLY")
        for panel in payload["recommended_panels"]
    )
    assert any(
        panel["type"] == "scatter" for panel in payload["recommended_panels"]
    )


def test_wandb_run_ids_are_injective_across_trial_hashes():
    first = _record("a", 0.9, 20.0, rank=32, placement="uniform")
    second = _record("b", 0.8, 18.0, rank=64, placement="uniform")
    first["wandb_run_id"] = "shared"
    second["wandb_run_id"] = "shared"
    with pytest.raises(ValueError, match="multiple trial hashes"):
        validate_wandb_trial_mapping([first, second])

    second["wandb_run_id"] = "other"
    validate_wandb_trial_mapping([first, second])


def test_result_summaries_are_enriched_with_optuna_parameters():
    summary = {
        "trial_hash": "a" * 64,
        "status": "COMPLETE",
        "mqar_accuracy": 0.8,
        "wikitext_ppl": None,
        "optuna_trial_id": 7,
        "health_passed": False,
    }
    metadata = {
        "trial_hash": "a" * 64,
        "optuna_trial_id": 7,
        "parameters": {"ska_rank": 96, "objective_arm": "birdie_mix"},
        "wikitext_ppl": 20.0,
        "step_500_mqar_accuracy": 0.65,
        "health_passed": True,
    }

    merged = merge_record_metadata([summary], [metadata])

    assert merged[0]["parameters"]["ska_rank"] == 96
    assert merged[0]["wikitext_ppl"] == 20.0
    assert merged[0]["step_500_mqar_accuracy"] == 0.65
    assert merged[0]["health_passed"] is False  # summary evidence wins


def test_pareto_rejects_nonfinite_and_boolean_metrics():
    valid = _record("a", 0.8, 20.0, rank=32, placement="uniform")
    invalid_bool = {**valid, "trial_hash": "b" * 64, "mqar_accuracy": True}
    invalid_nan = {**valid, "trial_hash": "c" * 64, "wikitext_ppl": float("nan")}

    assert pareto_front([invalid_bool, invalid_nan, valid]) == [valid]


def test_result_tree_collection_is_sorted_and_rejects_duplicate_hashes(tmp_path):
    first = tmp_path / "trial-a" / "trial_summary.json"
    second = tmp_path / "trial-b" / "trial_summary.json"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_text('{"trial_hash":"' + "a" * 64 + '","status":"COMPLETE"}\n')
    second.write_text('{"trial_hash":"' + "b" * 64 + '","status":"PRUNED"}\n')
    assert [record["trial_hash"] for record in load_result_summaries(tmp_path)] == [
        "a" * 64,
        "b" * 64,
    ]

    second.write_text('{"trial_hash":"' + "a" * 64 + '","status":"PRUNED"}\n')
    with pytest.raises(ValueError, match="Duplicate trial summary"):
        load_result_summaries(tmp_path)


def test_result_tree_retains_step_500_mqar_from_sibling_metrics(tmp_path):
    trial_hash = "a" * 64
    trial_dir = tmp_path / "trial-a"
    metrics_dir = trial_dir / "metrics"
    metrics_dir.mkdir(parents=True)
    (trial_dir / "trial_summary.json").write_text(
        json.dumps({"trial_hash": trial_hash, "status": "COMPLETE"}) + "\n"
    )
    (metrics_dir / "step_500.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "trial_hash": trial_hash,
                "status": "ok",
                "optimizer_step": 500,
                "mqar_accuracy": 0.625,
            }
        )
        + "\n"
    )

    records = load_result_summaries(tmp_path)

    assert records[0]["step_500_mqar_accuracy"] == pytest.approx(0.625)

    records_path = trial_dir / "trial_summary.json"
    records_path.write_text(
        json.dumps(
            {
                "trial_hash": trial_hash,
                "status": "COMPLETE",
                "step_500_mqar_accuracy": 0.5,
            }
        )
        + "\n"
    )
    with pytest.raises(ValueError, match="Summary/step-500 MQAR mismatch"):
        load_result_summaries(tmp_path)


def test_result_tree_rejects_summary_seed_or_promotion_identity_tampering(
    tmp_path,
):
    spec = load_spec(ROOT / "configs" / "phase2a_search.json")
    manifest = build_trial_manifest(
        spec,
        reference_parameters(),
        seed=42,
        code_identity={"commit": "a" * 40, "branch": "test", "dirty": False},
        data_identity={"revision": "frozen"},
        capability_identity={"integration_commit": "a" * 40},
    )
    trial_dir = tmp_path / "trial"
    trial_dir.mkdir()
    (trial_dir / "trial_manifest.json").write_text(
        json.dumps(manifest) + "\n"
    )
    summary = {
        "trial_hash": manifest["trial_hash"],
        "promotion_config_hash": "0" * 64,
        "seed": 43,
        "status": "COMPLETE",
    }
    (trial_dir / "trial_summary.json").write_text(json.dumps(summary) + "\n")

    with pytest.raises(ValueError, match="promotion_config_hash mismatch"):
        load_result_summaries(tmp_path)
