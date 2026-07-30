"""Focused contract tests for the non-scientific Phase 2a calibration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from koopman_lm.experiments.phase2.calibration import (
    calibration_parameter_matrix,
    materialize_calibration,
)
from koopman_lm.experiments.phase2.spec import load_spec, stable_hash


pytestmark = pytest.mark.correctness

ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "configs" / "phase2a_search.json"
DATA = {
    "tokenizer_revision": "calibration-tokenizer",
    "train_dataset_revision": "calibration-train",
    "validation_dataset_revision": "calibration-validation",
    "checksums": {
        "train": "1" * 64,
        "validation": "2" * 64,
    },
}
CAPABILITIES = {
    "schema_version": 1,
    "integration_commit": "a" * 40,
    "status": "calibration-only",
}


def test_calibration_parameter_matrix_is_exact() -> None:
    matrix = calibration_parameter_matrix()
    assert [name for name, _ in matrix] == [
        "default_updated_sweep",
        "worst_case_stability",
        "birdie_mix",
    ]
    assert len(matrix) == 3

    default = matrix[0][1]
    assert default["architecture_mode"] == "updated_sweep"
    assert default["objective_arm"] == "next_token_only"

    stability = matrix[1][1]
    assert stability["ska_rank"] == 128
    assert stability["ska_ridge"] == 0.0001
    assert stability["ska_chunk_size"] == 128
    assert stability["qk_norm"] is True

    birdie = matrix[2][1]
    assert birdie["objective_arm"] == "birdie_mix"
    assert birdie["birdie_retrieval_fraction"] == 0.3
    assert birdie["birdie_copy_share"] == 0.5


def test_calibration_materializes_three_screen_only_identity_bound_runs(
    tmp_path: Path,
) -> None:
    summary = materialize_calibration(
        SPEC_PATH,
        tmp_path,
        data_identity=DATA,
        capability_identity=CAPABILITIES,
        calibration_preflight_ready=True,
    )

    assert summary["calibration_manifest_count"] == 3
    assert summary["gpu_count_per_run"] == 1
    assert summary["required_stage_counts"] == {"screen": 3, "final": 0}
    assert summary["calibration_preflight_ready"] is True
    assert summary["scientific_launch_ready"] is False
    assert summary["data_hash"] == stable_hash(DATA)
    assert summary["capability_hash"] == stable_hash(CAPABILITIES)
    assert len(set(summary["trial_hashes"])) == 3
    assert all(
        entry["required_stage"] == "screen"
        for entry in summary["execution_plan"]
    )

    spec = load_spec(SPEC_PATH)
    public_spec = {
        key: value for key, value in spec.items() if not key.startswith("_")
    }
    assert summary["spec_hash"] == stable_hash(public_spec)
    for entry in summary["execution_plan"]:
        run_dir = tmp_path / entry["run_directory"]
        manifest = json.loads((run_dir / "trial_manifest.json").read_text())
        assert manifest["trial_hash"] == entry["trial_hash"]
        assert manifest["spec_hash"] == summary["spec_hash"]
        assert manifest["data"] == DATA
        assert manifest["capability_identity"] == CAPABILITIES

    persisted = json.loads((tmp_path / "calibration_summary.json").read_text())
    assert persisted == summary
