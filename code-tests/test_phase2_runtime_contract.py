"""Distributed runtime contracts for Phase 2a study workers."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from koopman_lm.experiments.phase2 import run_manifest as fixed_runner
from koopman_lm.experiments.phase2.checkpoint import checkpoint_bundle_identity
from koopman_lm.experiments.phase2.run_study import (
    ComputeBudgetExhausted,
    SystemicAdapterFailure,
    TrialLocalFailure,
    _validate_checkpoint_bundle_output,
    _cumulative_claim_gpu_time,
    _summary,
    abandon_trial_claim_for_recovery,
    begin_claim_stage,
    claim_trial_hash,
    count_queued_trial_recoveries,
    finalize_trial_claim,
    heartbeat_trial_claim,
    queue_stale_trial_recoveries,
    record_failed_claim_stage,
    record_successful_claim_stage,
    release_trial_claim,
    run_adapter_stage,
)
from koopman_lm.experiments.phase2.manifest import (
    build_trial_manifest,
    reference_parameters,
    write_trial_directory,
)
from koopman_lm.experiments.phase2.spec import load_spec, stable_hash

pytestmark = pytest.mark.correctness

_TRIAL_HASH = "f" * 64


_ROOT = Path(__file__).resolve().parents[1]
_CODE = {"commit": "a" * 40, "branch": "test", "dirty": False}
_DATA = {
    "tokenizer_revision": "tokenizer-pinned",
    "train_dataset_revision": "train-pinned",
    "validation_dataset_revision": "validation-pinned",
    "checksums": {"train": "1" * 64, "validation": "2" * 64},
}
_CAPABILITIES = {"schema_version": 1, "status": "approved-for-test"}


class _RecordingStudy:
    def __init__(self, *, fail_enqueue: bool = False) -> None:
        self.enqueued: list[tuple[dict, dict]] = []
        self.fail_enqueue = fail_enqueue

    def enqueue_trial(self, params, *, user_attrs):  # noqa: ANN001
        if self.fail_enqueue:
            raise RuntimeError("storage unavailable")
        self.enqueued.append((dict(params), dict(user_attrs)))


def _stale_study_trial(
    tmp_path: Path,
    *,
    timestamp: float = 1_000.0,
) -> tuple[dict, dict, object]:
    spec = load_spec(_ROOT / "configs" / "phase2a_search.json")
    manifest = build_trial_manifest(
        spec,
        reference_parameters(),
        seed=int(spec["protocol"]["seed"]),
        trial_number=7,
        code_identity=_CODE,
        data_identity=_DATA,
        capability_identity=_CAPABILITIES,
    )
    run_dir = tmp_path / "preserved-run"
    write_trial_directory(run_dir, manifest)
    claim = claim_trial_hash(
        tmp_path,
        manifest["trial_hash"],
        7,
        run_dir=run_dir,
        stale_after_seconds=100,
        now=lambda: timestamp,
    )
    assert claim is not None
    return spec, manifest, claim


def test_claim_duplicate_stale_reclaim_reuses_run_dir_and_terminal_is_final(
    tmp_path: Path,
) -> None:
    digest = "a" * 64
    first_dir = tmp_path / "first-attempt"
    first = claim_trial_hash(
        tmp_path,
        digest,
        1,
        run_dir=first_dir,
        stale_after_seconds=100,
        now=lambda: 1_000.0,
    )
    assert first is not None
    assert first.run_dir == first_dir.resolve()
    assert first.reclaimed is False

    duplicate = claim_trial_hash(
        tmp_path,
        digest,
        2,
        run_dir=tmp_path / "duplicate-attempt",
        stale_after_seconds=100,
        now=lambda: 1_050.0,
    )
    assert duplicate is None

    reclaimed = claim_trial_hash(
        tmp_path,
        digest,
        3,
        run_dir=tmp_path / "new-attempt-must-not-be-used",
        stale_after_seconds=100,
        now=lambda: 1_101.0,
    )
    assert reclaimed is not None
    assert reclaimed.reclaimed is True
    assert reclaimed.run_dir == first.run_dir
    assert reclaimed.lease_id != first.lease_id

    payload = json.loads(reclaimed.path.read_text())
    assert payload["state"] == "running"
    assert payload["reclaimed_count"] == 1
    assert payload["run_dir"] == str(first.run_dir)
    assert payload["optuna_trial_id"] == 3

    finalize_trial_claim(reclaimed, "COMPLETE", now=lambda: 1_200.0)
    terminal = json.loads(reclaimed.path.read_text())
    assert terminal["state"] == "terminal"
    assert terminal["outcome"] == "COMPLETE"
    assert release_trial_claim(reclaimed) is False

    assert (
        claim_trial_hash(
            tmp_path,
            digest,
            4,
            run_dir=tmp_path / "after-terminal",
            stale_after_seconds=100,
            now=lambda: 1_000_000.0,
        )
        is None
    )


def test_only_current_running_lease_can_release_claim(tmp_path: Path) -> None:
    digest = "b" * 64
    first = claim_trial_hash(
        tmp_path,
        digest,
        1,
        stale_after_seconds=10,
        now=lambda: 10.0,
    )
    assert first is not None
    reclaimed = claim_trial_hash(
        tmp_path,
        digest,
        2,
        stale_after_seconds=10,
        now=lambda: 21.0,
    )
    assert reclaimed is not None
    assert release_trial_claim(first) is False
    assert reclaimed.path.exists()
    assert release_trial_claim(reclaimed) is True
    assert not reclaimed.path.exists()


def test_startup_recovery_is_atomically_queued_and_token_gated(
    tmp_path: Path,
) -> None:
    spec, manifest, original = _stale_study_trial(tmp_path)
    study = _RecordingStudy()

    queued = queue_stale_trial_recoveries(
        study,
        spec=spec,
        output_root=tmp_path,
        code_identity=_CODE,
        data_identity=_DATA,
        capability_identity=_CAPABILITIES,
        stale_after_seconds=100,
        now=lambda: 1_101.0,
    )

    assert len(queued) == 1
    assert queued[0].run_dir == original.run_dir
    assert count_queued_trial_recoveries(tmp_path) == 1
    fixed_params, user_attrs = study.enqueued[0]
    assert "architecture_mode" not in fixed_params
    assert fixed_params["ska_rank"] == manifest["parameters"]["ska_rank"]
    assert user_attrs["phase2_recovery_trial_hash"] == manifest["trial_hash"]

    claim_payload = json.loads(original.path.read_text())
    assert claim_payload["state"] == "recovery_queued"
    assert claim_payload["recovery_token"] == queued[0].recovery_token

    # Concurrent startup scans see a fresh reservation and do not enqueue it.
    assert (
        queue_stale_trial_recoveries(
            _RecordingStudy(),
            spec=spec,
            output_root=tmp_path,
            code_identity=_CODE,
            data_identity=_DATA,
            capability_identity=_CAPABILITIES,
            stale_after_seconds=100,
            now=lambda: 1_102.0,
        )
        == []
    )
    # Neither a random sample nor an old/forged recovery job can steal it.
    assert (
        claim_trial_hash(
            tmp_path,
            manifest["trial_hash"],
            8,
            stale_after_seconds=100,
            now=lambda: 1_102.0,
        )
        is None
    )
    assert (
        claim_trial_hash(
            tmp_path,
            manifest["trial_hash"],
            8,
            stale_after_seconds=100,
            recovery_token="wrong-token",
            now=lambda: 1_102.0,
        )
        is None
    )

    recovered = claim_trial_hash(
        tmp_path,
        manifest["trial_hash"],
        8,
        stale_after_seconds=100,
        recovery_token=queued[0].recovery_token,
        now=lambda: 1_102.0,
    )
    assert recovered is not None
    assert recovered.reclaimed is True
    assert recovered.run_dir == original.run_dir
    assert count_queued_trial_recoveries(tmp_path) == 0


def test_optuna_startup_recovery_ignores_fixed_manifest_claims(
    tmp_path: Path,
) -> None:
    fixed = claim_trial_hash(
        tmp_path,
        "7" * 64,
        -1,
        run_dir=tmp_path / "fixed-control",
        stale_after_seconds=10,
        claim_kind="fixed_manifest",
        now=lambda: 10.0,
    )
    assert fixed is not None

    queued = queue_stale_trial_recoveries(
        _RecordingStudy(),
        spec=load_spec(_ROOT / "configs" / "phase2a_search.json"),
        output_root=tmp_path,
        code_identity=_CODE,
        data_identity=_DATA,
        capability_identity=_CAPABILITIES,
        stale_after_seconds=10,
        now=lambda: 1_000.0,
    )

    assert queued == []
    payload = json.loads(fixed.path.read_text())
    assert payload["claim_kind"] == "fixed_manifest"
    assert payload["state"] == "running"


def test_failed_recovery_enqueue_restores_stale_claim(tmp_path: Path) -> None:
    spec, manifest, original = _stale_study_trial(tmp_path)
    with pytest.raises(RuntimeError, match="storage unavailable"):
        queue_stale_trial_recoveries(
            _RecordingStudy(fail_enqueue=True),
            spec=spec,
            output_root=tmp_path,
            code_identity=_CODE,
            data_identity=_DATA,
            capability_identity=_CAPABILITIES,
            stale_after_seconds=100,
            now=lambda: 1_101.0,
        )
    payload = json.loads(original.path.read_text())
    assert payload["state"] == "running"
    assert payload["updated_at_unix"] == 1_000.0
    assert payload["trial_hash"] == manifest["trial_hash"]


def test_claim_heartbeat_prevents_live_work_from_being_reclaimed(
    tmp_path: Path,
) -> None:
    digest = "c" * 64
    claim = claim_trial_hash(
        tmp_path,
        digest,
        1,
        stale_after_seconds=100,
        now=lambda: 1_000.0,
    )
    assert claim is not None
    assert heartbeat_trial_claim(claim, now=lambda: 1_050.0) is True
    payload = json.loads(claim.path.read_text())
    assert payload["heartbeat_count"] == 1
    assert payload["last_heartbeat_at_unix"] == 1_050.0

    assert (
        claim_trial_hash(
            tmp_path,
            digest,
            2,
            stale_after_seconds=100,
            now=lambda: 1_101.0,
        )
        is None
    )
    reclaimed = claim_trial_hash(
        tmp_path,
        digest,
        2,
        stale_after_seconds=100,
        now=lambda: 1_151.0,
    )
    assert reclaimed is not None
    assert heartbeat_trial_claim(claim, now=lambda: 1_152.0) is False


def test_shared_compute_budget_reservations_are_atomic_and_release_unused_hours(
    tmp_path: Path,
) -> None:
    common = {
        "maximum_total_gpu_hours": 2.0,
        "maximum_full_trial_gpu_hours": 1.0,
        "maximum_concurrent_trials": 2,
    }
    first = claim_trial_hash(
        tmp_path,
        "d" * 64,
        1,
        now=lambda: 1_000.0,
        **common,
    )
    second = claim_trial_hash(
        tmp_path,
        "e" * 64,
        2,
        now=lambda: 1_001.0,
        **common,
    )
    assert first is not None
    assert second is not None
    with pytest.raises(ComputeBudgetExhausted, match="ceiling exhausted"):
        claim_trial_hash(
            tmp_path,
            "9" * 64,
            3,
            now=lambda: 1_002.0,
            **common,
        )

    finalize_trial_claim(
        first,
        "PRUNED",
        gpu_hours_actual=0.25,
        now=lambda: 1_003.0,
    )
    with pytest.raises(ComputeBudgetExhausted):
        claim_trial_hash(
            tmp_path,
            "9" * 64,
            3,
            now=lambda: 1_004.0,
            **common,
        )
    finalize_trial_claim(
        second,
        "COMPLETE",
        gpu_hours_actual=0.5,
        now=lambda: 1_005.0,
    )
    third = claim_trial_hash(
        tmp_path,
        "9" * 64,
        3,
        now=lambda: 1_006.0,
        **common,
    )
    assert third is not None
    payload = json.loads(third.path.read_text())
    assert payload["gpu_hours_reserved"] == 1.0
    with pytest.raises(RuntimeError, match="gpu_hours_actual"):
        finalize_trial_claim(third, "COMPLETE")
    assert json.loads(third.path.read_text())["state"] == "running"
    finalize_trial_claim(
        third,
        "COMPLETE",
        gpu_hours_actual=1.25,
        now=lambda: 1_007.0,
    )
    overrun = json.loads(third.path.read_text())
    assert overrun["state"] == "terminal"
    assert overrun["gpu_hours_reservation_overrun"] == pytest.approx(0.25)
    assert (
        overrun["compute_budget_violation"]
        == "full_trial_gpu_hours_exceeded"
    )
    first_payload = json.loads(first.path.read_text())
    assert first_payload["gpu_hours_actual"] == 0.25
    assert first_payload["gpu_hours_reservation_released"] == 1.0


def test_stage_and_recovery_attempt_gpu_time_is_persisted_exactly_once(
    tmp_path: Path,
) -> None:
    limits = {
        "maximum_total_gpu_hours": 2.0,
        "maximum_full_trial_gpu_hours": 1.0,
        "maximum_concurrent_trials": 2,
    }
    claim = claim_trial_hash(
        tmp_path,
        "8" * 64,
        1,
        stale_after_seconds=10,
        now=lambda: 10.0,
        **limits,
    )
    assert claim is not None
    record_successful_claim_stage(
        claim,
        stage="screen",
        gpu_seconds_actual=36.0,
        gpu_hours_actual=0.01,
        now=lambda: 11.0,
    )
    abandon_trial_claim_for_recovery(
        claim,
        failure={"scope": "systemic", "code": "preempted", "detail": "lost"},
        gpu_seconds_actual=72.0,
        gpu_hours_actual=0.02,
        now=lambda: 12.0,
    )
    recovered = claim_trial_hash(
        tmp_path,
        "8" * 64,
        2,
        stale_after_seconds=10,
        now=lambda: 13.0,
        **limits,
    )
    assert recovered is not None
    record_successful_claim_stage(
        recovered,
        stage="final",
        gpu_seconds_actual=108.0,
        gpu_hours_actual=0.03,
        now=lambda: 14.0,
    )
    seconds, hours = _cumulative_claim_gpu_time(
        recovered,
        gpu_seconds_actual=0.0,
        gpu_hours_actual=0.0,
    )
    assert seconds == 216.0
    assert hours == pytest.approx(0.06)


def test_trial_local_failure_uses_heartbeat_max_without_double_charge(
    tmp_path: Path,
) -> None:
    claim = claim_trial_hash(
        tmp_path,
        "7" * 64,
        1,
        now=lambda: 100.0,
        maximum_total_gpu_hours=2.0,
        maximum_full_trial_gpu_hours=1.0,
        maximum_concurrent_trials=1,
    )
    assert claim is not None
    begin_claim_stage(
        claim,
        stage="screen",
        world_size=1,
        heartbeat_interval_seconds=60.0,
        now=lambda: 100.0,
    )
    assert heartbeat_trial_claim(claim, now=lambda: 200.0)
    record_failed_claim_stage(
        claim,
        stage="screen",
        failure={"scope": "trial", "code": "oom", "detail": "trial-local"},
        gpu_seconds_actual=1.0,
        gpu_hours_actual=1.0 / 3600.0,
        now=lambda: 201.0,
    )
    abandon_trial_claim_for_recovery(
        claim,
        failure={
            "scope": "systemic",
            "code": "summary_write_failed",
            "detail": "persistence failed after stage accounting",
        },
        gpu_seconds_actual=0.0,
        gpu_hours_actual=0.0,
        now=lambda: 202.0,
    )
    seconds, hours = _cumulative_claim_gpu_time(
        claim,
        gpu_seconds_actual=0.0,
        gpu_hours_actual=0.0,
    )
    assert seconds == pytest.approx(101.0)
    assert hours == pytest.approx(101.0 / 3600.0)


def test_hard_preemption_is_charged_through_last_durable_heartbeat(
    tmp_path: Path,
) -> None:
    limits = {
        "maximum_total_gpu_hours": 10.0,
        "maximum_full_trial_gpu_hours": 5.0,
        "maximum_concurrent_trials": 2,
    }
    claim = claim_trial_hash(
        tmp_path,
        "6" * 64,
        1,
        stale_after_seconds=10,
        now=lambda: 100.0,
        **limits,
    )
    assert claim is not None
    begin_claim_stage(
        claim,
        stage="screen",
        world_size=2,
        heartbeat_interval_seconds=5.0,
        now=lambda: 101.0,
    )
    assert heartbeat_trial_claim(claim, now=lambda: 106.0)

    recovered = claim_trial_hash(
        tmp_path,
        "6" * 64,
        2,
        stale_after_seconds=10,
        now=lambda: 117.0,
        **limits,
    )
    assert recovered is not None
    payload = json.loads(recovered.path.read_text())
    assert "inflight_stage" not in payload
    assert len(payload["systemic_attempts"]) == 1
    attempt = payload["systemic_attempts"][0]
    assert attempt["failure"]["code"] == "worker_lost_during_adapter_stage"
    assert attempt["gpu_seconds_actual"] == pytest.approx(10.0)
    assert attempt["gpu_hours_actual"] == pytest.approx(10.0 / 3600.0)
    assert attempt["maximum_unaccounted_gpu_seconds"] == pytest.approx(10.0)


def test_summary_binds_results_to_study_identity(tmp_path: Path) -> None:
    _, manifest, _ = _stale_study_trial(tmp_path)
    summary = _summary(manifest, "COMPLETE")
    assert summary["study_identity"] == {
        "study_name": manifest["study_name"],
        "spec_hash": manifest["spec_hash"],
        "code_commit": _CODE["commit"],
        "code_dirty": False,
        "data_hash": stable_hash(_DATA),
        "capability_hash": stable_hash(_CAPABILITIES),
    }


def _fixed_runner_fixture(tmp_path: Path) -> tuple[dict, dict]:
    spec = load_spec(_ROOT / "configs" / "phase2a_search.json")
    spec["study"]["compute_budget"].update(
        {
            "maximum_total_gpu_hours": 10.0,
            "maximum_full_trial_gpu_hours": 5.0,
            "maximum_concurrent_trials": 2,
        }
    )
    manifest = build_trial_manifest(
        spec,
        reference_parameters(),
        seed=int(spec["protocol"]["seed"]),
        trial_number=None,
        code_identity=_CODE,
        data_identity=_DATA,
        capability_identity=_CAPABILITIES,
    )
    run_dir = tmp_path / "fixed-run"
    write_trial_directory(run_dir, manifest)
    manifest["_manifest_path"] = str(run_dir / "trial_manifest.json")
    return spec, manifest


def test_budgeted_fixed_runner_aggregates_stages_and_reuses_only_terminal_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec, manifest = _fixed_runner_fixture(tmp_path)
    calls: list[str] = []

    def fake_stage(*args, stage, **kwargs):  # noqa: ANN002, ANN003
        calls.append(stage)
        if stage == "screen":
            return {
                "gpu_seconds_actual": 3600.0,
                "gpu_hours_actual": 1.0,
                "mqar_accuracy": 0.25,
                "wandb_run_id": "fixed-test-run",
                "checkpoint_provenance": {
                    "checkpoint_sha256": "c" * 64,
                    "checkpoint_bundle_path": "checkpoints/step_500",
                    "checkpoint_bundle_byte_size": 123,
                    "checkpoint_bundle_file_count": 4,
                    "checkpoint_digest_algorithm": (
                        "sha256(path_nul_size_nul_content_sha256_newline_v1)"
                    ),
                    "optimizer_step": 500,
                    "global_sequence_index": 48_000,
                    "checkpoint_format_revision": "test-v1",
                },
            }
        return {
            "gpu_seconds_actual": 7200.0,
            "gpu_hours_actual": 2.0,
            "mqar_accuracy": 0.0,
            "wikitext_ppl": 20.0,
            "model_accounting": {},
            "optimizer_group_audit": {},
            "wandb_run_id": "fixed-test-run",
            "resume_provenance": {
                "source_checkpoint_sha256": "c" * 64,
                "source_checkpoint_bundle_path": "checkpoints/step_500",
                "source_checkpoint_bundle_byte_size": 123,
                "source_checkpoint_bundle_file_count": 4,
                "source_checkpoint_digest_algorithm": (
                    "sha256(path_nul_size_nul_content_sha256_newline_v1)"
                ),
                "source_optimizer_step": 500,
                "restored_global_sequence_index": 48_000,
                "checkpoint_format_revision": "test-v1",
                "optimizer_state_restored": True,
                "scheduler_state_restored": True,
                "rng_state_restored": True,
                "data_stream_state_restored": True,
            },
        }

    monkeypatch.setattr(fixed_runner, "run_adapter_stage", fake_stage)
    monkeypatch.setattr(
        fixed_runner,
        "validate_step_metrics",
        lambda *args, **kwargs: None,
    )
    budget_root = tmp_path / "budget"
    summary = fixed_runner.execute_budgeted_manifest(
        spec=spec,
        manifest=manifest,
        worker_command=["fake"],
        budget_root=budget_root,
        heartbeat_interval=60.0,
    )
    assert summary["gpu_seconds_actual"] == 10800.0
    assert summary["gpu_hours_actual"] == 3.0
    claim_path = budget_root / "claims" / f"{manifest['trial_hash']}.json"
    claim = json.loads(claim_path.read_text())
    assert claim["claim_kind"] == "fixed_manifest"
    assert claim["state"] == "terminal"
    assert claim["outcome"] == "COMPLETE"
    assert claim["gpu_hours_actual"] == 3.0

    reused = fixed_runner.execute_budgeted_manifest(
        spec=spec,
        manifest=manifest,
        worker_command=["must-not-run"],
        budget_root=budget_root,
        heartbeat_interval=60.0,
    )
    assert reused == summary
    assert calls == ["screen", "final"]


def _patch_adapter_process(
    monkeypatch: pytest.MonkeyPatch,
    *,
    metrics_path: Path,
    output: object | None,
    returncode: int,
) -> None:
    (metrics_path.parents[1] / "trial_manifest.json").write_text(
        json.dumps(
            {
                "trial_hash": _TRIAL_HASH,
                "protocol": {"world_size": 1},
            }
        )
        + "\n"
    )

    def fake_run(*args, **kwargs):  # noqa: ANN002, ANN003
        if output is not None:
            payload = output
            if (
                isinstance(output, dict)
                and output.get("status") == "ok"
                and returncode == 0
            ):
                checkpoint_dir = metrics_path.parents[1] / "checkpoints" / "step_500"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                (checkpoint_dir / "state.bin").write_bytes(
                    b"controller-hashed-checkpoint"
                )
                identity = checkpoint_bundle_identity(
                    metrics_path.parents[1],
                    "checkpoints/step_500",
                )
                payload = dict(output)
                payload["checkpoint_provenance"] = {
                    "checkpoint_bundle_path": identity["path"],
                    "checkpoint_bundle_byte_size": identity["byte_size"],
                    "checkpoint_bundle_file_count": identity["file_count"],
                    "checkpoint_sha256": identity["sha256"],
                    "checkpoint_digest_algorithm": identity[
                        "digest_algorithm"
                    ],
                    "optimizer_step": 500,
                    "global_sequence_index": 48_000,
                    "checkpoint_format_revision": "test-v1",
                }
            metrics_path.write_text(json.dumps(payload) + "\n")
        return subprocess.CompletedProcess(args=args[0], returncode=returncode)

    monkeypatch.setattr(subprocess, "run", fake_run)


def test_adapter_stage_heartbeats_owned_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metrics_path = tmp_path / "metrics" / "screen.json"
    metrics_path.parent.mkdir()
    _patch_adapter_process(
        monkeypatch,
        metrics_path=metrics_path,
        returncode=0,
        output={"schema_version": 1, "trial_hash": _TRIAL_HASH, "status": "ok"},
    )
    claim = claim_trial_hash(
        tmp_path,
        _TRIAL_HASH,
        1,
        run_dir=tmp_path,
        now=lambda: 1_000.0,
    )
    assert claim is not None
    output = run_adapter_stage(
        ["fake-adapter"],
        manifest_path=tmp_path / "trial_manifest.json",
        run_dir=tmp_path,
        stage="screen",
        metrics_path=metrics_path,
        claim=claim,
        claim_heartbeat_interval_seconds=3_600.0,
    )
    assert output["status"] == "ok"
    payload = json.loads(claim.path.read_text())
    assert payload["heartbeat_count"] == 2
    assert payload["last_heartbeat_at_unix"] > 1_000.0


def test_controller_rejects_self_attested_checkpoint_digest(
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "checkpoints" / "step_500"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "state.bin").write_bytes(b"actual-checkpoint")
    identity = checkpoint_bundle_identity(tmp_path, "checkpoints/step_500")
    provenance = {
        "checkpoint_bundle_path": identity["path"],
        "checkpoint_bundle_byte_size": identity["byte_size"],
        "checkpoint_bundle_file_count": identity["file_count"],
        "checkpoint_sha256": "0" * 64,
        "checkpoint_digest_algorithm": identity["digest_algorithm"],
    }
    with pytest.raises(SystemicAdapterFailure, match="controller hash"):
        _validate_checkpoint_bundle_output(
            {"checkpoint_provenance": provenance},
            run_dir=tmp_path,
            stage="screen",
        )


def test_structured_trial_failure_is_the_only_local_adapter_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metrics_path = tmp_path / "metrics" / "screen.json"
    metrics_path.parent.mkdir()
    _patch_adapter_process(
        monkeypatch,
        metrics_path=metrics_path,
        returncode=17,
        output={
            "schema_version": 1,
            "trial_hash": _TRIAL_HASH,
            "status": "failed",
            "failure": {
                "scope": "trial",
                "code": "nonfinite_loss",
                "detail": "loss became non-finite at optimizer step 31",
            },
        },
    )
    with pytest.raises(TrialLocalFailure) as caught:
        run_adapter_stage(
            ["fake-adapter"],
            manifest_path=tmp_path / "trial_manifest.json",
            run_dir=tmp_path,
            stage="screen",
            metrics_path=metrics_path,
        )
    assert caught.value.failure is not None
    assert caught.value.failure["code"] == "nonfinite_loss"
    assert caught.value.returncode == 17


def test_budgeted_trial_failure_requires_stage_gpu_accounting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metrics_path = tmp_path / "metrics" / "screen.json"
    metrics_path.parent.mkdir()
    claim = claim_trial_hash(
        tmp_path,
        _TRIAL_HASH,
        1,
        run_dir=tmp_path,
        maximum_total_gpu_hours=2.0,
        maximum_full_trial_gpu_hours=1.0,
        maximum_concurrent_trials=2,
    )
    assert claim is not None
    _patch_adapter_process(
        monkeypatch,
        metrics_path=metrics_path,
        returncode=17,
        output={
            "schema_version": 1,
            "trial_hash": _TRIAL_HASH,
            "status": "failed",
            "gpu_seconds_actual": 36.0,
            "gpu_hours_actual": 0.01,
            "failure": {
                "scope": "trial",
                "code": "nonfinite_loss",
                "detail": "loss became non-finite",
            },
        },
    )
    with pytest.raises(TrialLocalFailure) as caught:
        run_adapter_stage(
            ["fake-adapter"],
            manifest_path=tmp_path / "trial_manifest.json",
            run_dir=tmp_path,
            stage="screen",
            metrics_path=metrics_path,
            claim=claim,
            claim_heartbeat_interval_seconds=3_600.0,
        )
    assert caught.value.gpu_seconds_actual == 36.0
    assert caught.value.gpu_hours_actual == 0.01
    finalize_trial_claim(
        claim,
        "FAIL",
        gpu_hours_actual=0.01,
    )
    second_claim = claim_trial_hash(
        tmp_path / "second-budget",
        _TRIAL_HASH,
        2,
        run_dir=tmp_path,
        maximum_total_gpu_hours=2.0,
        maximum_full_trial_gpu_hours=1.0,
        maximum_concurrent_trials=2,
    )
    assert second_claim is not None

    _patch_adapter_process(
        monkeypatch,
        metrics_path=metrics_path,
        returncode=17,
        output={
            "schema_version": 1,
            "trial_hash": _TRIAL_HASH,
            "status": "failed",
            "failure": {
                "scope": "trial",
                "code": "nonfinite_loss",
                "detail": "loss became non-finite",
            },
        },
    )
    with pytest.raises(SystemicAdapterFailure, match="gpu_seconds_actual"):
        run_adapter_stage(
            ["fake-adapter"],
            manifest_path=tmp_path / "trial_manifest.json",
            run_dir=tmp_path,
            stage="screen",
            metrics_path=metrics_path,
            claim=second_claim,
            claim_heartbeat_interval_seconds=3_600.0,
        )


@pytest.mark.parametrize(
    ("output", "returncode"),
    [
        (None, 23),
        ({"status": "ok"}, 23),
        ({"status": "failed"}, 1),
        (
            {
                "schema_version": 1,
                "trial_hash": _TRIAL_HASH,
                "status": "failed",
                "failure": {
                    "scope": "systemic",
                    "code": "bad_environment",
                    "detail": "CUDA runtime unavailable",
                },
            },
            1,
        ),
    ],
)
def test_unstructured_or_systemic_adapter_failure_stops_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    output: object | None,
    returncode: int,
) -> None:
    metrics_path = tmp_path / "metrics" / "screen.json"
    metrics_path.parent.mkdir()
    _patch_adapter_process(
        monkeypatch,
        metrics_path=metrics_path,
        output=output,
        returncode=returncode,
    )
    with pytest.raises(SystemicAdapterFailure):
        run_adapter_stage(
            ["fake-adapter"],
            manifest_path=tmp_path / "trial_manifest.json",
            run_dir=tmp_path,
            stage="screen",
            metrics_path=metrics_path,
        )
