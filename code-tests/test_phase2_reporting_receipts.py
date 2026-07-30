"""Dependency-light tests for final Phase 2a reporting receipts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

try:
    import pytest
except ImportError:  # Keeps direct unittest execution dependency-free.
    pytest = None

if pytest is not None:
    pytestmark = pytest.mark.correctness


ROOT = Path(__file__).resolve().parents[1]
PHASE2 = ROOT / "koopman_lm" / "experiments" / "phase2"
sys.path.insert(0, str(PHASE2))

from reporting_receipts import (  # noqa: E402
    EXPECTED_HYPOTHESIS_IDS,
    ReportingReceiptError,
    create_reporting_receipts,
)
from analysis import _HYPOTHESES  # noqa: E402
from spec import stable_hash  # noqa: E402


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact(path: Path) -> dict[str, object]:
    return {
        "filename": path.name,
        "sha256": _sha(path),
        "byte_size": path.stat().st_size,
    }


def _stored_artifact(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": _sha(path),
    }


class ReportingReceiptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.identity = {
            "study_name": "echo-phase2a-1m-v1",
            "spec_hash": "a" * 64,
            "code_commit": "1234567890abcdef",
            "code_dirty": False,
            "data_hash": "b" * 64,
            "capability_hash": "c" * 64,
        }
        self.payload = {
            "schema_version": 1,
            "kind": "wandb_report_payload",
            "network_write_performed": False,
            "publication_status": "manual_authorized_upload_required",
            "published_report_url": None,
            "study_identity": self.identity,
            "sampling_warning": "MQAR-only controller; post-hoc Pareto analysis.",
            "tables": {
                "pareto_front": {
                    "columns": ["trial_hash", "wandb_run_id"],
                    "data": [
                        [f"{index + 1:064x}", f"run-{index + 1:02d}"]
                        for index in range(len(EXPECTED_HYPOTHESIS_IDS))
                    ],
                },
                "promotion_shortlist": {
                    "columns": ["trial_hash", "wandb_run_id"],
                    "data": [
                        ["1".zfill(64), "run-01"],
                        ["2".zfill(64), "run-02"],
                    ],
                },
            },
            "recommended_panels": [],
        }
        self.analysis = {
            "schema_version": 1,
            "study_identity": self.identity,
            "trial_hash_index": [
                f"{index + 1:064x}"
                for index in range(len(EXPECTED_HYPOTHESIS_IDS))
            ],
            "fanova_importance": {
                "study_identity": self.identity,
                "views": {
                    view_name: {
                        metric: {
                            "status": "ok",
                            "importance": {"ska_rank": 1.0},
                            **(
                                {
                                    "capacity_adjusted": {
                                        "status": "ok",
                                        "importance": {"ska_rank": 1.0},
                                    }
                                }
                                if view_name == "global"
                                else {}
                            ),
                        }
                        for metric in ("mqar_accuracy", "wikitext_ppl")
                    }
                    for view_name in (
                        "global",
                        "birdie_only",
                        "next_token_only",
                        "updated_sweep_only",
                    )
                },
            },
            "falsified_hypotheses": {
                "schema_version": 1,
                "report_status": "scientist_review_required",
                "decision_rule": "Review seeded evidence.",
                "study_identity": self.identity,
                "hypotheses": [
                    {"id": hypothesis_id, "falsified": None}
                    for hypothesis_id in EXPECTED_HYPOTHESIS_IDS
                ],
            },
            "wandb_report_payload": self.payload,
        }
        self.decisions = {
            "schema_version": 1,
            "decisions": [
                {
                    "id": hypothesis_id,
                    "falsified": index % 2 == 0,
                    "rationale": f"Seeded evidence supports decision {index}.",
                    "evidence_trial_hashes": [f"{index + 1:064x}"],
                }
                for index, hypothesis_id in enumerate(EXPECTED_HYPOTHESIS_IDS)
            ],
        }
        self.analysis_path = self.root / "phase2a_analysis.json"
        self.payload_path = self.root / "wandb_report_payload.json"
        self.decisions_path = self.root / "decisions.json"
        self.storage_receipt_path = self.root / "storage_snapshot_receipt.json"
        self.wandb_verification_path = self.root / "wandb_verification.json"
        self.report_url = (
            "https://wandb.ai/echo-lab/koopman/reports/"
            "Phase-2a-results--Vmlldzox"
        )
        self.analysis_path.write_text(
            json.dumps(self.analysis, indent=2, sort_keys=True) + "\n"
        )
        self.payload_path.write_text(
            json.dumps(self.payload, indent=2, sort_keys=True) + "\n"
        )
        self.decisions_path.write_text(
            json.dumps(self.decisions, indent=2, sort_keys=True) + "\n"
        )
        analysis_artifact = _artifact(self.analysis_path)
        self.claims_output_root = self.root / "runs"
        self.claims_dir = self.claims_output_root / "claims"
        self.claims_dir.mkdir(parents=True)
        for trial_hash in self.analysis["trial_hash_index"]:
            (self.claims_dir / f"{trial_hash}.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "trial_hash": trial_hash,
                        "state": "terminal",
                        "outcome": "COMPLETE",
                        "claim_kind": "optuna_study",
                    }
                )
                + "\n"
            )
        self.compute_budget = {
            "status": "approved",
            "maximum_total_gpu_hours": 10.0,
            "maximum_full_trial_gpu_hours": 1.0,
            "maximum_concurrent_trials": 16,
        }
        self.ledger = {
            "schema_version": 1,
            **self.identity,
            "claims_output_root": str(self.claims_output_root.resolve()),
            "storage_identity_sha256": "7" * 64,
            "compute_budget": self.compute_budget,
            "requested_trials": len(EXPECTED_HYPOTHESIS_IDS),
        }
        self.ledger["ledger_contract_sha256"] = stable_hash(self.ledger)
        self.ledger_path = (
            self.claims_output_root / "phase2_ledger_contract.json"
        )
        self.ledger_path.write_text(
            json.dumps(self.ledger, indent=2, sort_keys=True) + "\n"
        )
        self.status = {
            "schema_version": 1,
            "study_name": self.identity["study_name"],
            "study_identity": self.identity,
            "storage_backend": "postgresql",
            "storage_identity_sha256": self.ledger[
                "storage_identity_sha256"
            ],
            "ledger_contract_sha256": self.ledger[
                "ledger_contract_sha256"
            ],
            "claims_output_root": str(self.claims_output_root.resolve()),
            "requested_trials": len(EXPECTED_HYPOTHESIS_IDS),
            "optuna_trial_count": len(EXPECTED_HYPOTHESIS_IDS),
            "fresh_optuna_trial_count": len(EXPECTED_HYPOTHESIS_IDS),
            "fresh_trial_target_guard_count": 0,
            "optuna_states": {
                "COMPLETE": len(EXPECTED_HYPOTHESIS_IDS)
            },
            "claims": {
                "claim_count": len(EXPECTED_HYPOTHESIS_IDS),
                "states": {
                    "terminal": len(EXPECTED_HYPOTHESIS_IDS)
                },
                "outcomes": {
                    "COMPLETE": len(EXPECTED_HYPOTHESIS_IDS)
                },
                "terminal_gpu_hours_actual": 8.0,
                "active_gpu_hours_reserved": 0.0,
                "compute_budget_violation_count": 0,
            },
            "compute_budget": {
                **self.compute_budget,
                "terminal_gpu_hours_actual": 8.0,
                "active_gpu_hours_reserved": 0.0,
                "remaining_unreserved_gpu_hours": 2.0,
            },
            "hold_reasons": ["requested_trial_target_reached"],
        }
        self.status_path = self.root / "study_status.json"
        self.write_status()
        self.storage_receipt = {
            "schema_version": 1,
            "receipt_kind": "phase2_optuna_storage_snapshot",
            "study_identity": {
                "study_name": self.identity["study_name"],
                "requested_trials": len(EXPECTED_HYPOTHESIS_IDS),
                "spec_canonical_sha256": self.identity["spec_hash"],
                "spec_file": {
                    "path": str(self.root / "phase2a_search.json"),
                    "size_bytes": 100,
                    "sha256": "d" * 64,
                },
            },
            "analysis_artifact": {
                "path": str(self.analysis_path.resolve()),
                "size_bytes": analysis_artifact["byte_size"],
                "sha256": analysis_artifact["sha256"],
            },
            "ledger_contract_artifact": _stored_artifact(
                self.ledger_path
            ),
            "storage": {
                "backend": "postgresql",
                "snapshot_file": {
                    "path": str(self.root / "optuna.dump"),
                    "size_bytes": 200,
                    "sha256": "e" * 64,
                },
                "backend_validation": {
                    "method": "pg_restore_list",
                    "entry_count": 8,
                    "listing_sha256": "f" * 64,
                },
            },
            "quiescence_evidence": {
                "status_file": {
                    **_stored_artifact(self.status_path),
                },
                "completion_disposition": "requested_trial_target_reached",
                "budget_exhausted_early_stop_approval": None,
                "claim_count": len(EXPECTED_HYPOTHESIS_IDS),
                "running_claims": 0,
                "recovery_queued_claims": 0,
                "terminal_gpu_hours_actual": 8.0,
                "active_gpu_hours_reserved": 0.0,
                "remaining_unreserved_gpu_hours": 2.0,
                "maximum_full_trial_gpu_hours": 1.0,
                "compute_budget_violation_count": 0,
                "status_hold_reasons": ["requested_trial_target_reached"],
                "fresh_optuna_trial_count": len(EXPECTED_HYPOTHESIS_IDS),
                "fresh_trial_target_guard_count": 0,
                "optuna_trial_count": len(EXPECTED_HYPOTHESIS_IDS),
                "optuna_states": {
                    "COMPLETE": len(EXPECTED_HYPOTHESIS_IDS)
                },
            },
            "created_by": "Storage Operator",
            "created_at_utc": "2026-07-27T19:55:00Z",
        }
        self.wandb_run_ids = [
            f"run-{index + 1:02d}"
            for index in range(len(EXPECTED_HYPOTHESIS_IDS))
        ]
        self.wandb_verification = {
            "schema_version": 1,
            "kind": "phase2a_wandb_report_verification",
            "verification_status": "verified",
            "verification_method": "manual_wandb_report_run_set_audit",
            "analysis_artifact": analysis_artifact,
            "study_identity": self.identity,
            "wandb_report_url": self.report_url,
            "wandb_run_count": len(self.wandb_run_ids),
            "wandb_run_ids": self.wandb_run_ids,
            "verified_by": "W&B Auditor",
            "verified_at_utc": "2026-07-27T20:02:00Z",
        }
        self.write_storage_receipt()
        self.write_wandb_verification()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write_storage_receipt(self) -> None:
        self.storage_receipt_path.write_text(
            json.dumps(self.storage_receipt, indent=2, sort_keys=True) + "\n"
        )

    def write_status(self) -> None:
        self.status_path.write_text(
            json.dumps(self.status, indent=2, sort_keys=True) + "\n"
        )

    def sync_status_and_receipt_identity(self) -> None:
        quiescence = self.storage_receipt["quiescence_evidence"]
        self.status["optuna_trial_count"] = quiescence[
            "optuna_trial_count"
        ]
        self.status["fresh_optuna_trial_count"] = quiescence[
            "fresh_optuna_trial_count"
        ]
        self.status["fresh_trial_target_guard_count"] = quiescence[
            "fresh_trial_target_guard_count"
        ]
        self.status["optuna_states"] = quiescence["optuna_states"]
        self.status["hold_reasons"] = quiescence["status_hold_reasons"]
        self.status["claims"]["claim_count"] = quiescence["claim_count"]
        self.status["claims"]["states"] = {
            "terminal": quiescence["claim_count"]
        }
        self.status["claims"]["terminal_gpu_hours_actual"] = quiescence[
            "terminal_gpu_hours_actual"
        ]
        self.status["claims"]["active_gpu_hours_reserved"] = quiescence[
            "active_gpu_hours_reserved"
        ]
        self.status["claims"]["compute_budget_violation_count"] = (
            quiescence["compute_budget_violation_count"]
        )
        self.status["compute_budget"]["terminal_gpu_hours_actual"] = (
            quiescence["terminal_gpu_hours_actual"]
        )
        self.status["compute_budget"]["active_gpu_hours_reserved"] = (
            quiescence["active_gpu_hours_reserved"]
        )
        self.status["compute_budget"]["remaining_unreserved_gpu_hours"] = (
            quiescence["remaining_unreserved_gpu_hours"]
        )
        self.status["compute_budget"]["maximum_full_trial_gpu_hours"] = (
            quiescence["maximum_full_trial_gpu_hours"]
        )
        self.write_status()
        quiescence["status_file"] = _stored_artifact(self.status_path)

    def write_wandb_verification(self) -> None:
        self.wandb_verification_path.write_text(
            json.dumps(self.wandb_verification, indent=2, sort_keys=True) + "\n"
        )

    def test_expected_ids_match_the_analysis_evidence_scaffold(self) -> None:
        self.assertEqual(
            EXPECTED_HYPOTHESIS_IDS,
            tuple(item["id"] for item in _HYPOTHESES),
        )

    def test_wandb_verification_template_tracks_its_schema(self) -> None:
        template = json.loads(
            (
                ROOT
                / "configs"
                / "phase2a_wandb_verification.template.json"
            ).read_text()
        )
        schema = json.loads(
            (
                ROOT
                / "docs"
                / "phase2"
                / "schemas"
                / "wandb_report_verification.schema.json"
            ).read_text()
        )
        self.assertEqual(set(template), set(schema["required"]))
        self.assertEqual(set(template), set(schema["properties"]))

    def test_storage_receipt_fixture_tracks_reporting_dependency_schema(
        self,
    ) -> None:
        schema = json.loads(
            (
                ROOT
                / "docs"
                / "phase2"
                / "schemas"
                / "storage_snapshot_receipt.schema.json"
            ).read_text()
        )
        self.assertEqual(
            set(self.storage_receipt),
            set(schema["required"]),
        )
        self.assertEqual(
            set(self.storage_receipt),
            set(schema["properties"]),
        )
        quiescence_schema = schema["properties"]["quiescence_evidence"]
        self.assertEqual(
            set(self.storage_receipt["quiescence_evidence"]),
            set(quiescence_schema["required"]),
        )
        self.assertEqual(
            set(self.storage_receipt["quiescence_evidence"]),
            set(quiescence_schema["properties"]),
        )

    def create(self, **overrides: object) -> tuple[Path, Path]:
        arguments: dict[str, object] = {
            "analysis_path": self.analysis_path,
            "decisions_path": self.decisions_path,
            "wandb_payload_path": self.payload_path,
            "storage_snapshot_receipt_path": self.storage_receipt_path,
            "wandb_verification_path": self.wandb_verification_path,
            "reviewer": "Research Lead",
            "reviewed_at_utc": "2026-07-27T20:00:00Z",
            "publisher": "Research Lead",
            "published_at_utc": "2026-07-27T20:05:00Z",
            "output_dir": self.root / "receipts",
        }
        arguments.update(overrides)
        return create_reporting_receipts(**arguments)  # type: ignore[arg-type]

    def test_receipts_bind_exact_files_identity_and_canonical_decisions(self) -> None:
        reviewed_path, publication_path = self.create()
        reviewed = json.loads(reviewed_path.read_text())
        publication = json.loads(publication_path.read_text())

        self.assertEqual(
            reviewed["analysis_artifact"]["sha256"], _sha(self.analysis_path)
        )
        self.assertEqual(
            publication["analysis_artifact"], reviewed["analysis_artifact"]
        )
        self.assertEqual(
            publication["wandb_payload_artifact"]["sha256"],
            _sha(self.payload_path),
        )
        self.assertEqual(
            reviewed["storage_snapshot_receipt_artifact"]["sha256"],
            _sha(self.storage_receipt_path),
        )
        self.assertEqual(
            publication["wandb_verification_artifact"]["sha256"],
            _sha(self.wandb_verification_path),
        )
        self.assertEqual(publication["wandb_report_url"], self.report_url)
        self.assertEqual(publication["wandb_run_ids"], self.wandb_run_ids)
        self.assertEqual(
            publication["wandb_run_count"], len(self.wandb_run_ids)
        )
        self.assertEqual(reviewed["study_identity"], self.identity)
        self.assertEqual(publication["study_identity"], self.identity)
        self.assertEqual(
            [item["id"] for item in reviewed["decisions"]],
            list(EXPECTED_HYPOTHESIS_IDS),
        )
        self.assertEqual(reviewed["decision_count"], 10)
        self.assertEqual(publication["publication_status"], "published")
        for value, schema_name in (
            (
                reviewed,
                "reviewed_falsified_hypotheses.schema.json",
            ),
            (
                publication,
                "wandb_publication_receipt.schema.json",
            ),
            (
                self.wandb_verification,
                "wandb_report_verification.schema.json",
            ),
        ):
            schema = json.loads(
                (
                    ROOT
                    / "docs"
                    / "phase2"
                    / "schemas"
                    / schema_name
                ).read_text()
            )
            self.assertEqual(set(value), set(schema["required"]))
            self.assertEqual(set(value), set(schema["properties"]))

    def test_rejects_incomplete_decision_set(self) -> None:
        self.decisions["decisions"].pop()
        self.decisions_path.write_text(json.dumps(self.decisions))
        with self.assertRaisesRegex(
            ReportingReceiptError, "do not exactly match"
        ):
            self.create()

    def test_rejects_payload_that_is_not_the_embedded_artifact(self) -> None:
        self.payload["tables"] = {"changed": True}
        self.payload_path.write_text(json.dumps(self.payload))
        with self.assertRaisesRegex(ReportingReceiptError, "does not exactly match"):
            self.create()

    def test_rejects_evidence_not_present_in_claim_bound_analysis(self) -> None:
        self.decisions["decisions"][0]["evidence_trial_hashes"] = ["f" * 64]
        self.decisions_path.write_text(json.dumps(self.decisions))
        with self.assertRaisesRegex(
            ReportingReceiptError,
            "not present in the claim-bound analysis index",
        ):
            self.create()

    def test_rejects_storage_receipt_for_different_analysis_bytes(self) -> None:
        self.storage_receipt["analysis_artifact"]["sha256"] = "0" * 64
        self.write_storage_receipt()
        with self.assertRaisesRegex(
            ReportingReceiptError,
            "not bound to the exact analysis artifact",
        ):
            self.create()

    def test_rejects_storage_receipt_for_different_study(self) -> None:
        self.storage_receipt["study_identity"]["study_name"] = (
            "unrelated-study"
        )
        self.write_storage_receipt()
        with self.assertRaisesRegex(
            ReportingReceiptError,
            "study identity differs from the analysis",
        ):
            self.create()

    def test_rejects_tampered_bound_ledger_artifact(self) -> None:
        self.ledger["data_hash"] = "0" * 64
        self.ledger_path.write_text(json.dumps(self.ledger) + "\n")
        with self.assertRaisesRegex(
            ReportingReceiptError,
            "ledger_contract_artifact bytes do not match",
        ):
            self.create()

    def test_rejects_status_not_bound_to_immutable_ledger(self) -> None:
        self.status["storage_identity_sha256"] = "0" * 64
        self.write_status()
        self.storage_receipt["quiescence_evidence"]["status_file"] = (
            _stored_artifact(self.status_path)
        )
        self.write_storage_receipt()
        with self.assertRaisesRegex(
            ReportingReceiptError,
            "does not match its receipt or immutable ledger",
        ):
            self.create()

    def test_rejects_analysis_index_without_exact_terminal_claim_set(
        self,
    ) -> None:
        first_hash = self.analysis["trial_hash_index"][0]
        (self.claims_dir / f"{first_hash}.json").unlink()
        with self.assertRaisesRegex(
            ReportingReceiptError,
            "terminal claim hash set does not exactly match",
        ):
            self.create()

    def test_rejects_storage_receipt_without_quiescent_complete_study(self) -> None:
        self.storage_receipt["quiescence_evidence"]["running_claims"] = 1
        self.write_storage_receipt()
        with self.assertRaisesRegex(
            ReportingReceiptError,
            "does not prove a complete quiescent study",
        ):
            self.create()

    def test_accepts_explicitly_approved_budget_exhausted_early_stop(self) -> None:
        quiescence = self.storage_receipt["quiescence_evidence"]
        quiescence["completion_disposition"] = (
            "approved_budget_exhausted_early_stop"
        )
        quiescence["fresh_optuna_trial_count"] = (
            len(EXPECTED_HYPOTHESIS_IDS) - 1
        )
        quiescence["remaining_unreserved_gpu_hours"] = 0.5
        quiescence["maximum_full_trial_gpu_hours"] = 1.0
        quiescence["status_hold_reasons"] = [
            "insufficient_unreserved_gpu_hours_for_one_full_trial"
        ]
        quiescence["budget_exhausted_early_stop_approval"] = {
            "approved_by": "Research Lead",
            "approved_at_utc": "2026-07-27T19:54:00Z",
            "approved_fresh_optuna_trial_count": (
                len(EXPECTED_HYPOTHESIS_IDS) - 1
            ),
            "requested_trials": len(EXPECTED_HYPOTHESIS_IDS),
            "remaining_unreserved_gpu_hours": 0.5,
            "maximum_full_trial_gpu_hours": 1.0,
        }
        self.sync_status_and_receipt_identity()
        self.write_storage_receipt()
        reviewed, publication = self.create()
        self.assertTrue(reviewed.is_file())
        self.assertTrue(publication.is_file())

    def test_rejects_unapproved_partial_storage_receipt(self) -> None:
        quiescence = self.storage_receipt["quiescence_evidence"]
        quiescence["fresh_optuna_trial_count"] = (
            len(EXPECTED_HYPOTHESIS_IDS) - 1
        )
        quiescence["completion_disposition"] = (
            "approved_budget_exhausted_early_stop"
        )
        quiescence["status_hold_reasons"] = [
            "insufficient_unreserved_gpu_hours_for_one_full_trial"
        ]
        quiescence["remaining_unreserved_gpu_hours"] = 0.5
        quiescence["maximum_full_trial_gpu_hours"] = 1.0
        self.sync_status_and_receipt_identity()
        self.write_storage_receipt()
        with self.assertRaisesRegex(
            ReportingReceiptError,
            "approved early-stop disposition is inconsistent",
        ):
            self.create()

    def test_rejects_wandb_verification_with_mismatched_run_set(self) -> None:
        self.wandb_verification["wandb_run_ids"] = [
            *self.wandb_run_ids,
            "unrelated-run",
        ]
        self.wandb_verification["wandb_run_count"] += 1
        self.write_wandb_verification()
        with self.assertRaisesRegex(
            ReportingReceiptError,
            "run-ID set does not exactly match",
        ):
            self.create()

    def test_rejects_wandb_verification_for_different_analysis(self) -> None:
        self.wandb_verification["analysis_artifact"]["sha256"] = "0" * 64
        self.write_wandb_verification()
        with self.assertRaisesRegex(
            ReportingReceiptError,
            "not bound to the exact analysis artifact",
        ):
            self.create()

    def test_rejects_wandb_verification_for_different_study(self) -> None:
        self.wandb_verification["study_identity"] = {
            **self.identity,
            "study_name": "unrelated-study",
        }
        self.write_wandb_verification()
        with self.assertRaisesRegex(
            ReportingReceiptError,
            "study identity differs from the analysis",
        ):
            self.create()

    def test_rejects_non_wandb_or_non_report_url_in_verification(self) -> None:
        for url in (
            "http://wandb.ai/echo-lab/koopman/reports/result",
            "https://example.com/echo-lab/koopman/reports/result",
            "https://wandb.ai/echo-lab/koopman/runs/result",
        ):
            with self.subTest(url=url):
                self.wandb_verification["wandb_report_url"] = url
                self.write_wandb_verification()
                with self.assertRaisesRegex(ReportingReceiptError, "W&B report URL"):
                    self.create(
                        output_dir=self.root / hashlib.sha256(url.encode()).hexdigest(),
                    )

    def test_rejects_offline_payload_without_a_run_id_set(self) -> None:
        self.payload["tables"] = {
            "fanova_importance": {
                "columns": ["axis", "importance"],
                "data": [["ska_rank", 1.0]],
            }
        }
        self.analysis["wandb_report_payload"] = self.payload
        self.payload_path.write_text(json.dumps(self.payload))
        self.analysis_path.write_text(json.dumps(self.analysis))
        analysis_artifact = _artifact(self.analysis_path)
        self.storage_receipt["analysis_artifact"] = {
            "path": str(self.analysis_path.resolve()),
            "size_bytes": analysis_artifact["byte_size"],
            "sha256": analysis_artifact["sha256"],
        }
        self.wandb_verification["analysis_artifact"] = analysis_artifact
        self.write_storage_receipt()
        self.write_wandb_verification()
        with self.assertRaisesRegex(
            ReportingReceiptError,
            "contains no nonempty wandb_run_id set",
        ):
            self.create()

    def test_rejects_overwrite_of_immutable_receipts(self) -> None:
        self.create()
        with self.assertRaisesRegex(ReportingReceiptError, "immutable"):
            self.create()


if __name__ == "__main__":
    unittest.main()
