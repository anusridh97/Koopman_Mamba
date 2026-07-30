"""Standalone tests for immutable Phase 2 storage snapshot receipts."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PHASE2 = ROOT / "koopman_lm" / "experiments" / "phase2"
sys.path.insert(0, str(PHASE2))

from spec import stable_hash  # noqa: E402
import storage_snapshot as storage_snapshot_module  # noqa: E402
from storage_snapshot import (  # noqa: E402
    StorageSnapshotError,
    create_storage_snapshot_receipt,
)


class StorageSnapshotReceiptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(
            prefix=".phase2-snapshot-test-",
            dir=ROOT / "code-tests",
        )
        self.work = Path(self.temp.name)
        spec = json.loads(
            (ROOT / "configs" / "phase2a_search.json").read_text()
        )
        spec["status"] = "scientific_ready"
        spec["protocol"]["learning_rate_status"] = "approved"
        spec["protocol"]["training_protocol_status"] = "approved"
        spec["study"]["compute_budget"].update(
            {
                "maximum_total_gpu_hours": 100.0,
                "maximum_full_trial_gpu_hours": 1.0,
                "required_gpu_type": "NVIDIA B200",
                "status": "approved",
            }
        )
        spec["storage"]["backend"] = "postgresql"
        self.spec_payload = spec
        self.spec = self.work / "spec.json"
        self.spec.write_text(json.dumps(spec))
        public_spec = {
            key: value for key, value in spec.items() if not key.startswith("_")
        }
        self.study_identity = {
            "study_name": spec["study_name"],
            "spec_hash": stable_hash(public_spec),
            "code_commit": "a" * 40,
            "code_dirty": False,
            "data_hash": "b" * 64,
            "capability_hash": "c" * 64,
        }
        self.analysis = self.work / "phase2a_analysis.json"
        self.analysis.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "study_identity": self.study_identity,
                    "trial_hash_index": ["d" * 64, "e" * 64],
                }
            )
        )
        self.snapshot = self.work / "optuna.dump"
        self.snapshot.write_bytes(b"backend-native-database-snapshot")
        self.status = self.work / "study_status.json"
        runs_root = self.work / "runs"
        runs_root.mkdir()
        ledger_payload = {
            "schema_version": 1,
            **self.study_identity,
            "claims_output_root": str(runs_root.resolve()),
            "storage_identity_sha256": "f" * 64,
            "compute_budget": dict(spec["study"]["compute_budget"]),
            "requested_trials": spec["study"]["requested_trials"],
        }
        ledger_payload["ledger_contract_sha256"] = stable_hash(
            ledger_payload
        )
        self.ledger = runs_root / "phase2_ledger_contract.json"
        self.ledger.write_text(json.dumps(ledger_payload))
        self.status_payload = {
            "schema_version": 1,
            "study_name": spec["study_name"],
            "study_identity": self.study_identity,
            "storage_backend": "postgresql",
            "claims_output_root": str(runs_root.resolve()),
            "storage_identity_sha256": ledger_payload[
                "storage_identity_sha256"
            ],
            "ledger_contract_sha256": ledger_payload[
                "ledger_contract_sha256"
            ],
            "requested_trials": spec["study"]["requested_trials"],
            "optuna_trial_count": spec["study"]["requested_trials"],
            "fresh_optuna_trial_count": spec["study"]["requested_trials"],
            "fresh_trial_target_guard_count": 0,
            "optuna_states": {
                "COMPLETE": 2,
                "PRUNED": spec["study"]["requested_trials"] - 2,
            },
            "compute_budget": {
                "maximum_total_gpu_hours": 100.0,
                "maximum_full_trial_gpu_hours": 1.0,
                "maximum_concurrent_trials": 16,
                "terminal_gpu_hours_actual": 1.25,
                "active_gpu_hours_reserved": 0.0,
                "remaining_unreserved_gpu_hours": 98.75,
            },
            "claims": {
                "claim_count": 2,
                "states": {"terminal": 2},
                "outcomes": {"COMPLETE": 2},
                "terminal_gpu_hours_actual": 1.25,
                "active_gpu_hours_reserved": 0.0,
                "compute_budget_violation_count": 0,
            },
            "launch_allowed": False,
            "hold_reasons": ["requested_trial_target_reached"],
        }
        self.status.write_text(json.dumps(self.status_payload))
        self.output = self.work / "storage_snapshot_receipt.json"

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _create(self, **overrides):
        kwargs = {
            "spec_path": self.spec,
            "snapshot_path": self.snapshot,
            "backend": "postgresql",
            "created_by": "phase2-operator",
            "created_at_utc": "2026-07-27T18:30:00Z",
            "status_evidence_path": self.status,
            "analysis_path": self.analysis,
            "output_path": self.output,
        }
        kwargs.update(overrides)
        completed = storage_snapshot_module.subprocess.CompletedProcess(
            args=["pg_restore", "--list"],
            returncode=0,
            stdout="1; 0 0 TABLE public trials owner\n",
            stderr="",
        )
        with mock.patch.object(
            storage_snapshot_module.subprocess,
            "run",
            return_value=completed,
        ):
            return create_storage_snapshot_receipt(**kwargs)

    def test_receipt_binds_spec_snapshot_backend_and_quiescence(self) -> None:
        receipt = self._create()
        persisted = json.loads(self.output.read_text())
        self.assertEqual(persisted, receipt)
        public_spec = {
            key: value
            for key, value in self.spec_payload.items()
            if not key.startswith("_")
        }
        self.assertEqual(
            receipt["study_identity"]["spec_canonical_sha256"],
            stable_hash(public_spec),
        )
        self.assertEqual(
            receipt["storage"]["snapshot_file"]["sha256"],
            hashlib.sha256(self.snapshot.read_bytes()).hexdigest(),
        )
        self.assertEqual(receipt["storage"]["backend"], "postgresql")
        self.assertEqual(
            receipt["study_identity"]["study_name"],
            self.spec_payload["study_name"],
        )
        self.assertEqual(
            receipt["quiescence_evidence"]["running_claims"],
            0,
        )
        self.assertEqual(
            receipt["quiescence_evidence"]["completion_disposition"],
            "requested_trial_target_reached",
        )
        self.assertIsNone(
            receipt["quiescence_evidence"][
                "budget_exhausted_early_stop_approval"
            ]
        )
        with self.assertRaisesRegex(
            StorageSnapshotError,
            "will not be replaced",
        ):
            self._create()

    def test_rejects_nonquiescent_status(self) -> None:
        self.status_payload["claims"]["claim_count"] = 3
        self.status_payload["claims"]["states"]["running"] = 1
        self.status_payload["claims"]["active_gpu_hours_reserved"] = 0.5
        self.status.write_text(json.dumps(self.status_payload))
        with self.assertRaisesRegex(StorageSnapshotError, "not quiescent"):
            self._create()

    def test_rejects_backend_that_differs_from_frozen_spec(self) -> None:
        with self.assertRaisesRegex(StorageSnapshotError, "does not match"):
            self._create(backend="scg_validated_journal_storage")

    def test_rejects_unready_spec(self) -> None:
        self.spec_payload["status"] = "premerge_infrastructure_only"
        self.spec.write_text(json.dumps(self.spec_payload))
        with self.assertRaisesRegex(StorageSnapshotError, "scientific_ready"):
            self._create()

    def test_target_receipt_accepts_terminal_tail_guard_rows(self) -> None:
        self.status_payload["fresh_trial_target_guard_count"] = 15
        self.status_payload["optuna_trial_count"] += 15
        self.status_payload["optuna_states"]["PRUNED"] += 15
        self.status.write_text(json.dumps(self.status_payload))

        receipt = self._create()

        self.assertEqual(
            receipt["quiescence_evidence"]["fresh_optuna_trial_count"],
            self.spec_payload["study"]["requested_trials"],
        )
        self.assertEqual(
            receipt["quiescence_evidence"]["fresh_trial_target_guard_count"],
            15,
        )
        self.assertEqual(
            receipt["quiescence_evidence"]["optuna_trial_count"],
            self.spec_payload["study"]["requested_trials"] + 15,
        )

    def test_accepts_explicitly_approved_budget_exhausted_early_stop(
        self,
    ) -> None:
        self.status_payload.update(
            {
                "optuna_trial_count": 2,
                "fresh_optuna_trial_count": 2,
                "optuna_states": {"COMPLETE": 2},
                "hold_reasons": [
                    "insufficient_unreserved_gpu_hours_for_one_full_trial"
                ],
            }
        )
        self.status_payload["claims"]["terminal_gpu_hours_actual"] = 99.5
        self.status_payload["compute_budget"].update(
            {
                "terminal_gpu_hours_actual": 99.5,
                "remaining_unreserved_gpu_hours": 0.5,
            }
        )
        self.status.write_text(json.dumps(self.status_payload))

        receipt = self._create(
            budget_exhausted_approved_by="phase2-lead",
            budget_exhausted_approved_at_utc="2026-07-27T18:00:00Z",
        )

        evidence = receipt["quiescence_evidence"]
        self.assertEqual(
            evidence["completion_disposition"],
            "approved_budget_exhausted_early_stop",
        )
        self.assertEqual(
            evidence["budget_exhausted_early_stop_approval"],
            {
                "approved_by": "phase2-lead",
                "approved_at_utc": "2026-07-27T18:00:00Z",
                "approved_fresh_optuna_trial_count": 2,
                "requested_trials": self.spec_payload["study"][
                    "requested_trials"
                ],
                "remaining_unreserved_gpu_hours": 0.5,
                "maximum_full_trial_gpu_hours": 1.0,
            },
        )

    def test_rejects_unapproved_budget_exhausted_early_stop(self) -> None:
        self.status_payload.update(
            {
                "optuna_trial_count": 2,
                "fresh_optuna_trial_count": 2,
                "optuna_states": {"COMPLETE": 2},
                "hold_reasons": [
                    "insufficient_unreserved_gpu_hours_for_one_full_trial"
                ],
            }
        )
        self.status_payload["claims"]["terminal_gpu_hours_actual"] = 99.5
        self.status_payload["compute_budget"].update(
            {
                "terminal_gpu_hours_actual": 99.5,
                "remaining_unreserved_gpu_hours": 0.5,
            }
        )
        self.status.write_text(json.dumps(self.status_payload))

        with self.assertRaisesRegex(
            StorageSnapshotError,
            "explicit early-stop approval",
        ):
            self._create()

    def test_rejects_budget_approval_when_target_was_reached(self) -> None:
        with self.assertRaisesRegex(
            StorageSnapshotError,
            "not valid after the target is reached",
        ):
            self._create(
                budget_exhausted_approved_by="phase2-lead",
                budget_exhausted_approved_at_utc="2026-07-27T18:00:00Z",
            )


if __name__ == "__main__":
    unittest.main()
