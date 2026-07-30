"""Dependency-light tests for controller-owned checkpoint provenance."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PHASE2 = ROOT / "koopman_lm" / "experiments" / "phase2"
sys.path.insert(0, str(PHASE2))

from checkpoint import checkpoint_bundle_identity  # noqa: E402
from run_study import (  # noqa: E402
    SystemicAdapterFailure,
    _validate_checkpoint_bundle_output,
    terminal_artifact_bindings,
)


class CheckpointProvenanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(
            prefix=".phase2-checkpoint-test-",
            dir=ROOT / "code-tests",
        )
        self.run = Path(self.temp.name)
        self.bundle = self.run / "checkpoints" / "step_500"
        self.bundle.mkdir(parents=True)
        (self.bundle / "model.bin").write_bytes(b"model-state")
        (self.bundle / "optimizer.bin").write_bytes(b"optimizer-state")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _screen_provenance(self) -> dict:
        identity = checkpoint_bundle_identity(
            self.run,
            "checkpoints/step_500",
        )
        return {
            "checkpoint_bundle_path": identity["path"],
            "checkpoint_bundle_byte_size": identity["byte_size"],
            "checkpoint_bundle_file_count": identity["file_count"],
            "checkpoint_sha256": identity["sha256"],
            "checkpoint_digest_algorithm": identity["digest_algorithm"],
            "optimizer_step": 500,
            "global_sequence_index": 48_000,
            "checkpoint_format_revision": "test-v1",
        }

    def test_controller_rehashes_bundle_and_rejects_self_attestation(self) -> None:
        provenance = self._screen_provenance()
        _validate_checkpoint_bundle_output(
            {"checkpoint_provenance": provenance},
            run_dir=self.run,
            stage="screen",
        )
        provenance["checkpoint_sha256"] = "0" * 64
        with self.assertRaisesRegex(
            SystemicAdapterFailure,
            "controller hash",
        ):
            _validate_checkpoint_bundle_output(
                {"checkpoint_provenance": provenance},
                run_dir=self.run,
                stage="screen",
            )

    def test_terminal_claim_binding_includes_checkpoint_bundle(self) -> None:
        provenance = self._screen_provenance()
        metrics = self.run / "metrics"
        metrics.mkdir()
        (self.run / "trial_manifest.json").write_text(
            json.dumps({"trial_hash": "a" * 64}) + "\n"
        )
        (self.run / "trial_summary.json").write_text(
            json.dumps({"trial_hash": "a" * 64, "status": "PRUNED"}) + "\n"
        )
        (metrics / "step_500.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "trial_hash": "a" * 64,
                    "status": "ok",
                    "checkpoint_provenance": provenance,
                }
            )
            + "\n"
        )
        bindings = terminal_artifact_bindings(self.run)
        self.assertEqual(
            bindings["checkpoint_bundle"]["sha256"],
            provenance["checkpoint_sha256"],
        )

        (self.bundle / "optimizer.bin").write_bytes(b"mutated-state")
        with self.assertRaisesRegex(RuntimeError, "changed before"):
            terminal_artifact_bindings(self.run)


if __name__ == "__main__":
    unittest.main()
