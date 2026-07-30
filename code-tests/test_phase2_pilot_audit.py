"""Dependency-light tests for the fixed pilot evidence contract."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

try:
    import pytest
except ImportError:  # Keeps ``python test_*.py`` dependency-free.
    pytest = None

if pytest is not None:
    pytestmark = pytest.mark.correctness


ROOT = Path(__file__).resolve().parents[1]
PHASE2 = ROOT / "koopman_lm" / "experiments" / "phase2"
sys.path.insert(0, str(PHASE2))

import pilot_audit as pilot_audit_module  # noqa: E402
from manifest import git_identity  # noqa: E402
from pilot import materialize_pilot  # noqa: E402
from pilot_audit import PilotAuditError, audit_pilot  # noqa: E402
from results import Phase2ResultError  # noqa: E402
from spec import load_spec, stable_hash  # noqa: E402


SPEC_PATH = ROOT / "configs" / "phase2a_search.json"


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def _stage_metrics(
    manifest: dict,
    *,
    optimizer_step: int,
    wandb_run_id: str,
    gpu_seconds: float,
    wikitext_ppl: float | None,
) -> dict:
    payload = {
        "schema_version": 1,
        "trial_hash": manifest["trial_hash"],
        "status": "ok",
        "optimizer_step": optimizer_step,
        "gpu_seconds_actual": gpu_seconds,
        "gpu_hours_actual": gpu_seconds / 3600.0,
        "mqar_accuracy": 0.625,
        "wikitext_ppl": wikitext_ppl,
        "wandb_run_id": wandb_run_id,
        "model_accounting": {
            "parameter_counts_actual": manifest["parameter_counts_estimated"],
            "pilot_fixture": True,
        },
        "optimizer_group_audit": {
            "pilot_fixture": True,
            "trial_hash": manifest["trial_hash"],
        },
    }
    prune_step = int(manifest["fidelity"]["prune_step"])
    global_sequence_index = (
        prune_step * int(manifest["protocol"]["effective_batch_sequences"])
    )
    if optimizer_step == prune_step:
        payload["checkpoint_provenance"] = {
            "checkpoint_sha256": manifest["trial_hash"],
            "checkpoint_bundle_path": "checkpoints/step_500",
            "checkpoint_bundle_byte_size": 123,
            "checkpoint_bundle_file_count": 4,
            "checkpoint_digest_algorithm": (
                "sha256(path_nul_size_nul_content_sha256_newline_v1)"
            ),
            "optimizer_step": prune_step,
            "global_sequence_index": global_sequence_index,
            "checkpoint_format_revision": "pilot-fixture-v1",
        }
    else:
        payload["resume_provenance"] = {
            "source_checkpoint_sha256": manifest["trial_hash"],
            "source_checkpoint_bundle_path": "checkpoints/step_500",
            "source_checkpoint_bundle_byte_size": 123,
            "source_checkpoint_bundle_file_count": 4,
            "source_checkpoint_digest_algorithm": (
                "sha256(path_nul_size_nul_content_sha256_newline_v1)"
            ),
            "source_optimizer_step": prune_step,
            "restored_global_sequence_index": global_sequence_index,
            "checkpoint_format_revision": "pilot-fixture-v1",
            "optimizer_state_restored": True,
            "scheduler_state_restored": True,
            "rng_state_restored": True,
            "data_stream_state_restored": True,
        }
    return payload


def _study_identity(manifest: dict) -> dict:
    return {
        "study_name": manifest["study_name"],
        "spec_hash": manifest["spec_hash"],
        "code_commit": manifest["code"]["commit"],
        "code_dirty": manifest["code"]["dirty"],
        "data_hash": stable_hash(manifest["data"]),
        "capability_hash": stable_hash(manifest["capability_identity"]),
    }


def _result_summary(
    manifest: dict,
    *,
    screen: dict,
    final: dict | None,
) -> dict:
    terminal = final or screen
    stages = [screen] + ([final] if final is not None else [])
    return {
        "schema_version": 1,
        "trial_hash": manifest["trial_hash"],
        "promotion_config_hash": manifest["promotion_config_hash"],
        "study_identity": _study_identity(manifest),
        "seed": manifest["protocol"]["seed"],
        "status": "COMPLETE" if final is not None else "SCREEN_COMPLETE",
        "mqar_accuracy": terminal["mqar_accuracy"],
        "step_500_mqar_accuracy": screen["mqar_accuracy"],
        "wikitext_ppl": terminal["wikitext_ppl"],
        "wandb_run_id": terminal["wandb_run_id"],
        "optuna_trial_id": None,
        "health_passed": True,
        "gpu_seconds_actual": sum(stage["gpu_seconds_actual"] for stage in stages),
        "gpu_hours_actual": sum(stage["gpu_hours_actual"] for stage in stages),
        "gpu_time_source": "controller_aggregated_stage_invocations",
        "failure": None,
        "model_accounting": terminal["model_accounting"],
        "optimizer_group_audit": terminal["optimizer_group_audit"],
    }


def _materialize_valid_pilot(
    tmp_path: Path,
) -> tuple[dict, dict, dict, dict]:
    spec = load_spec(SPEC_PATH)
    capabilities = {"integration_commit": git_identity(ROOT)["commit"]}
    data = {"schema_version": 1, "identity": "test"}
    materialize_pilot(
        SPEC_PATH,
        tmp_path,
        data_identity=data,
        capability_identity=capabilities,
        scientific_ready=True,
    )
    summary = json.loads((tmp_path / "pilot_summary.json").read_text())
    for entry in summary["execution_plan"]:
        run_dir = tmp_path / entry["run_directory"]
        manifest = json.loads((run_dir / "trial_manifest.json").read_text())
        wandb_run_id = f"pilot-{entry['trial_hash'][:16]}"
        screen = _stage_metrics(
            manifest,
            optimizer_step=manifest["fidelity"]["prune_step"],
            wandb_run_id=wandb_run_id,
            gpu_seconds=10.0,
            wikitext_ppl=None,
        )
        _write_json(run_dir / "metrics" / "step_500.json", screen)
        if entry["required_stage"] == "final":
            final = _stage_metrics(
                manifest,
                optimizer_step=manifest["fidelity"]["max_steps"],
                wandb_run_id=wandb_run_id,
                gpu_seconds=20.0,
                wikitext_ppl=18.0,
            )
            _write_json(run_dir / "metrics" / "final.json", final)
            _write_json(
                run_dir / "trial_summary.json",
                _result_summary(manifest, screen=screen, final=final),
            )
        else:
            _write_json(
                run_dir / "screen_summary.json",
                _result_summary(manifest, screen=screen, final=None),
            )
    return spec, capabilities, data, summary


def _fixture_validator(calls: list[tuple[int, bool]]) -> object:
    def validate_fixture_metrics(
        manifest,
        metrics,
        *,
        expected_step,
        require_full_mqar_grid,
        require_wikitext_ppl,
        **kwargs,
    ):  # noqa: ANN001, ANN003
        if metrics["trial_hash"] != manifest["trial_hash"]:
            raise AssertionError("Auditor passed metrics to the wrong manifest")
        if metrics["optimizer_step"] != expected_step:
            raise AssertionError("Auditor passed the wrong expected step")
        if bool(metrics["wikitext_ppl"] is not None) != require_wikitext_ppl:
            raise AssertionError("Auditor passed the wrong PPL requirement")
        if require_full_mqar_grid is not True:
            raise AssertionError("Pilot metrics must require the full MQAR grid")
        if not require_wikitext_ppl:
            if kwargs.get("diagnostic_names") != manifest[
                "screen_required_diagnostics"
            ]:
                raise AssertionError(
                    "Screen validation must use the screen diagnostic contract"
                )
        calls.append((expected_step, require_wikitext_ppl))

    return validate_fixture_metrics


class PilotAuditTests(unittest.TestCase):
    def _fixture(self) -> tuple[Path, dict, dict, dict, dict]:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        spec, capabilities, data, summary = _materialize_valid_pilot(root)
        return root, spec, capabilities, data, summary

    def _audit(
        self,
        root: Path,
        spec: dict,
        capabilities: dict,
        data: dict,
        *,
        calls: list[tuple[int, bool]] | None = None,
    ) -> dict:
        validation_calls = calls if calls is not None else []
        with mock.patch.object(
            pilot_audit_module,
            "validate_step_metrics",
            new=_fixture_validator(validation_calls),
        ):
            return audit_pilot(
                spec=spec,
                capabilities=capabilities,
                data_manifest=data,
                pilot_root=root,
            )

    def test_revalidates_metrics_accounting_and_wandb(self) -> None:
        root, spec, capabilities, data, summary = self._fixture()
        validation_calls: list[tuple[int, bool]] = []
        report = self._audit(
            root,
            spec,
            capabilities,
            data,
            calls=validation_calls,
        )

        self.assertEqual(report["status"], "pass")
        self.assertEqual(
            report["pilot_manifest_count"],
            len(summary["execution_plan"]),
        )
        expected_metric_count = (
            len(summary["execution_plan"])
            + summary["required_stage_counts"]["final"]
        )
        self.assertEqual(len(validation_calls), expected_metric_count)
        self.assertEqual(
            report["validated_metric_file_count"],
            expected_metric_count,
        )
        self.assertEqual(
            report["validated_accounting_summary_count"],
            len(summary["execution_plan"]),
        )
        self.assertEqual(
            report["validated_wandb_crosslink_count"],
            len(summary["execution_plan"]),
        )
        checks = {check["name"] for check in report["checks"]}
        self.assertTrue(
            {
                "validated_step_and_final_metrics",
                "validated_model_optimizer_gpu_accounting",
                "wandb_run_crosslinks",
            }
            <= checks
        )

    def test_requires_every_planned_stage(self) -> None:
        root, spec, capabilities, data, summary = self._fixture()
        missing = summary["execution_plan"][0]
        result_name = (
            "trial_summary.json"
            if missing["required_stage"] == "final"
            else "screen_summary.json"
        )
        (root / missing["run_directory"] / result_name).unlink()

        with self.assertRaises(PilotAuditError):
            self._audit(root, spec, capabilities, data)

    def test_rejects_summary_tampering(self) -> None:
        cases = [
            ("wandb_run_id", "different-run", "W&B run ID mismatch"),
            ("model_accounting", {"tampered": True}, "model_accounting mismatch"),
            ("study_identity", {"tampered": True}, "study_identity mismatch"),
            ("step_500_mqar_accuracy", 0.1, "step_500_mqar_accuracy mismatch"),
            ("gpu_seconds_actual", 99.0, "gpu_seconds_actual mismatch"),
            ("gpu_time_source", "forged", "gpu_time_source mismatch"),
        ]
        for field, tampered_value, message in cases:
            with self.subTest(field=field):
                root, spec, capabilities, data, summary = self._fixture()
                entry = summary["execution_plan"][0]
                result_name = (
                    "trial_summary.json"
                    if entry["required_stage"] == "final"
                    else "screen_summary.json"
                )
                result_path = root / entry["run_directory"] / result_name
                result = json.loads(result_path.read_text())
                result[field] = tampered_value
                _write_json(result_path, result)

                with self.assertRaisesRegex(PilotAuditError, message):
                    self._audit(root, spec, capabilities, data)

    def test_wraps_raw_metrics_validation_failure(self) -> None:
        root, spec, capabilities, data, _ = self._fixture()

        def reject_metrics(*args, **kwargs):  # noqa: ANN002, ANN003
            raise Phase2ResultError("broken diagnostic aggregate")

        with mock.patch.object(
            pilot_audit_module,
            "validate_step_metrics",
            new=reject_metrics,
        ):
            with self.assertRaisesRegex(
                PilotAuditError,
                "step-500 metrics failed validation",
            ):
                audit_pilot(
                    spec=spec,
                    capabilities=capabilities,
                    data_manifest=data,
                    pilot_root=root,
                )

    def test_rejects_cross_stage_identity_drift(self) -> None:
        cases = [
            (
                "wandb_run_id",
                "different-stage-run",
                "screen/final W&B run IDs disagree",
            ),
            (
                "model_accounting",
                {"different_stage_accounting": True},
                "screen/final model_accounting disagree",
            ),
        ]
        for field, replacement, message in cases:
            with self.subTest(field=field):
                root, spec, capabilities, data, pilot_summary = self._fixture()
                entry = next(
                    item
                    for item in pilot_summary["execution_plan"]
                    if item["required_stage"] == "final"
                )
                run_dir = root / entry["run_directory"]
                final_path = run_dir / "metrics" / "final.json"
                summary_path = run_dir / "trial_summary.json"
                final = json.loads(final_path.read_text())
                result = json.loads(summary_path.read_text())
                final[field] = replacement
                result[field] = replacement
                _write_json(final_path, final)
                _write_json(summary_path, result)

                with self.assertRaisesRegex(PilotAuditError, message):
                    self._audit(root, spec, capabilities, data)

    def test_requires_nonempty_metrics_wandb_id(self) -> None:
        root, spec, capabilities, data, pilot_summary = self._fixture()
        entry = next(
            item
            for item in pilot_summary["execution_plan"]
            if item["required_stage"] == "final"
        )
        run_dir = root / entry["run_directory"]
        for relative in (
            Path("metrics") / "step_500.json",
            Path("metrics") / "final.json",
            Path("trial_summary.json"),
        ):
            path = run_dir / relative
            payload = json.loads(path.read_text())
            payload["wandb_run_id"] = " "
            _write_json(path, payload)

        with self.assertRaisesRegex(PilotAuditError, "no nonempty W&B run ID"):
            self._audit(root, spec, capabilities, data)


if __name__ == "__main__":
    unittest.main()
