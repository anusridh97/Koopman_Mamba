"""Execute one pre-materialized control/pilot manifest through the adapter."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

try:
    from .manifest import (
        atomic_write_json,
        git_identity,
        recompute_promotion_config_hash,
        recompute_trial_hash,
        stable_hash,
    )
    from .preflight import evaluate_preflight, unresolved_placeholder_paths
    from .results import validate_cross_stage_identity, validate_step_metrics
    from .run_study import (
        AdapterFailure,
        TrialLocalFailure,
        _cumulative_claim_gpu_time,
        abandon_trial_claim_for_recovery,
        build_ledger_contract,
        claim_trial_hash,
        ensure_ledger_contract,
        finalize_trial_claim,
        record_successful_claim_stage,
        record_failed_claim_stage,
        release_trial_claim,
        run_adapter_stage,
        terminal_artifact_bindings,
    )
    from .spec import load_spec
except ImportError:
    from manifest import (  # type: ignore
        atomic_write_json,
        git_identity,
        recompute_promotion_config_hash,
        recompute_trial_hash,
        stable_hash,
    )
    from preflight import evaluate_preflight, unresolved_placeholder_paths  # type: ignore
    from results import (  # type: ignore
        validate_cross_stage_identity,
        validate_step_metrics,
    )
    from run_study import (  # type: ignore
        AdapterFailure,
        TrialLocalFailure,
        _cumulative_claim_gpu_time,
        abandon_trial_claim_for_recovery,
        build_ledger_contract,
        claim_trial_hash,
        ensure_ledger_contract,
        finalize_trial_claim,
        record_successful_claim_stage,
        record_failed_claim_stage,
        release_trial_claim,
        run_adapter_stage,
        terminal_artifact_bindings,
    )
    from spec import load_spec  # type: ignore


def _read_object(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _public_spec(spec: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in spec.items() if not key.startswith("_")}


def _summary(
    manifest: dict[str, Any],
    status: str,
    metrics: dict[str, Any] | None = None,
    failure: BaseException | None = None,
    fallback_gpu_seconds: float = 0.0,
    gpu_seconds_actual: float | None = None,
    gpu_hours_actual: float | None = None,
    gpu_time_source: str | None = None,
    step_500_mqar_accuracy: float | None = None,
) -> dict[str, Any]:
    metrics = metrics or {}
    reported_gpu_seconds = metrics.get("gpu_seconds_actual")
    reported_gpu_hours = metrics.get("gpu_hours_actual")
    if gpu_seconds_actual is not None or gpu_hours_actual is not None:
        if gpu_seconds_actual is None or gpu_hours_actual is None:
            raise ValueError("GPU seconds and hours overrides must be paired")
        gpu_seconds = float(gpu_seconds_actual)
        gpu_hours = float(gpu_hours_actual)
        resolved_gpu_time_source = gpu_time_source or "controller_aggregated"
    elif (
        isinstance(reported_gpu_seconds, (int, float))
        and not isinstance(reported_gpu_seconds, bool)
        and math.isfinite(float(reported_gpu_seconds))
        and float(reported_gpu_seconds) >= 0.0
        and isinstance(reported_gpu_hours, (int, float))
        and not isinstance(reported_gpu_hours, bool)
        and math.isfinite(float(reported_gpu_hours))
        and float(reported_gpu_hours) >= 0.0
    ):
        gpu_seconds = float(reported_gpu_seconds)
        gpu_hours = float(reported_gpu_hours)
        resolved_gpu_time_source = "adapter_reported"
    else:
        gpu_seconds = max(0.0, float(fallback_gpu_seconds))
        gpu_hours = gpu_seconds / 3600.0
        resolved_gpu_time_source = "controller_wall_clock_fallback"
    summary = {
        "schema_version": 1,
        "trial_hash": manifest["trial_hash"],
        "promotion_config_hash": manifest["promotion_config_hash"],
        "study_identity": {
            "study_name": manifest["study_name"],
            "spec_hash": manifest["spec_hash"],
            "code_commit": manifest["code"].get("commit", "UNKNOWN"),
            "code_dirty": manifest["code"].get("dirty", True),
            "data_hash": stable_hash(manifest["data"]),
            "capability_hash": stable_hash(manifest["capability_identity"]),
        },
        "seed": manifest["protocol"]["seed"],
        "status": status,
        "mqar_accuracy": metrics.get("mqar_accuracy"),
        "step_500_mqar_accuracy": (
            metrics.get("mqar_accuracy")
            if metrics.get("optimizer_step")
            == manifest.get("fidelity", {}).get("prune_step")
            else step_500_mqar_accuracy
        ),
        "wikitext_ppl": metrics.get("wikitext_ppl"),
        "wandb_run_id": metrics.get("wandb_run_id"),
        "optuna_trial_id": None,
        "health_passed": status in {"COMPLETE", "SCREEN_COMPLETE"},
        "gpu_seconds_actual": gpu_seconds,
        "gpu_hours_actual": gpu_hours,
        "gpu_time_source": resolved_gpu_time_source,
        "failure": (
            {"code": type(failure).__name__, "detail": str(failure)}
            if failure
            else None
        ),
    }
    if isinstance(metrics.get("model_accounting"), dict):
        summary["model_accounting"] = metrics["model_accounting"]
    if isinstance(metrics.get("optimizer_group_audit"), dict):
        summary["optimizer_group_audit"] = metrics["optimizer_group_audit"]
    return summary


def _validate_reusable_terminal_summary(
    manifest: dict[str, Any],
    summary: dict[str, Any],
    claim_payload: dict[str, Any],
) -> None:
    """Reject stale, partial, or identity-mismatched fixed-run outcomes."""

    expected_identity = _summary(manifest, "COMPLETE")["study_identity"]
    required_equalities = {
        "schema_version": 1,
        "trial_hash": manifest["trial_hash"],
        "promotion_config_hash": manifest["promotion_config_hash"],
        "seed": manifest["protocol"]["seed"],
        "status": "COMPLETE",
        "study_identity": expected_identity,
        "health_passed": True,
    }
    for field, expected in required_equalities.items():
        if summary.get(field) != expected:
            raise RuntimeError(
                f"Existing fixed-run summary has invalid {field}: "
                f"{summary.get(field)!r} != {expected!r}"
            )
    mqar = summary.get("mqar_accuracy")
    if (
        not isinstance(mqar, (int, float))
        or isinstance(mqar, bool)
        or not math.isfinite(float(mqar))
        or not 0.0 <= float(mqar) <= 1.0
    ):
        raise RuntimeError(
            f"Existing fixed-run summary has invalid mqar_accuracy: {mqar!r}"
        )
    ppl = summary.get("wikitext_ppl")
    if (
        not isinstance(ppl, (int, float))
        or isinstance(ppl, bool)
        or not math.isfinite(float(ppl))
        or float(ppl) <= 0.0
    ):
        raise RuntimeError(
            f"Existing fixed-run summary has invalid wikitext_ppl: {ppl!r}"
        )
    for field in ("model_accounting", "optimizer_group_audit"):
        if not isinstance(summary.get(field), dict):
            raise RuntimeError(
                f"Existing fixed-run summary is missing {field}"
            )
    wandb_run_id = summary.get("wandb_run_id")
    if not isinstance(wandb_run_id, str) or not wandb_run_id.strip():
        raise RuntimeError(
            "Existing fixed-run summary is missing a stable W&B run ID"
        )
    seconds = summary.get("gpu_seconds_actual")
    hours = summary.get("gpu_hours_actual")
    if (
        not isinstance(seconds, (int, float))
        or isinstance(seconds, bool)
        or not math.isfinite(float(seconds))
        or float(seconds) < 0.0
        or not isinstance(hours, (int, float))
        or isinstance(hours, bool)
        or not math.isfinite(float(hours))
        or float(hours) < 0.0
        or not math.isclose(
            float(hours),
            float(seconds) / 3600.0,
            rel_tol=1e-9,
            abs_tol=1e-12,
        )
    ):
        raise RuntimeError("Existing fixed-run summary has invalid GPU accounting")
    if (
        claim_payload.get("state") != "terminal"
        or claim_payload.get("outcome") != "COMPLETE"
    ):
        raise RuntimeError(
            "A reusable fixed-run summary requires a terminal COMPLETE claim"
        )
    claimed_hours = claim_payload.get("gpu_hours_actual")
    if (
        not isinstance(claimed_hours, (int, float))
        or isinstance(claimed_hours, bool)
        or not math.isclose(
            float(claimed_hours),
            float(hours),
            rel_tol=1e-9,
            abs_tol=1e-12,
        )
    ):
        raise RuntimeError(
            "Fixed-run summary GPU-hours disagree with the terminal claim"
        )
    manifest_path = manifest.get("_manifest_path")
    if not isinstance(manifest_path, str):
        raise RuntimeError("Reusable manifest is missing its source path")
    current_artifacts = terminal_artifact_bindings(
        Path(manifest_path).resolve().parent
    )
    if claim_payload.get("terminal_artifacts") != current_artifacts:
        raise RuntimeError(
            "Fixed-run terminal artifact bindings do not match current files"
        )


def validate_manifest_identity(
    spec: dict[str, Any],
    manifest: dict[str, Any],
    capabilities: dict[str, Any],
    *,
    preflight_stage: str = "study",
) -> None:
    expected_spec_hash = stable_hash(_public_spec(spec))
    if manifest.get("spec_hash") != expected_spec_hash:
        raise ValueError(
            f"Manifest spec hash {manifest.get('spec_hash')} != {expected_spec_hash}"
        )
    recomputed = recompute_trial_hash(manifest)
    if manifest.get("trial_hash") != recomputed:
        raise ValueError(
            f"Manifest trial hash {manifest.get('trial_hash')} != {recomputed}"
        )
    promotion_hash = recompute_promotion_config_hash(manifest)
    if manifest.get("promotion_config_hash") != promotion_hash:
        raise ValueError(
            "Manifest promotion config hash "
            f"{manifest.get('promotion_config_hash')} != {promotion_hash}"
        )
    if manifest.get("capability_identity") != capabilities:
        raise ValueError("Manifest was not materialized with this capability manifest")
    placeholders = unresolved_placeholder_paths(
        {
            "data": manifest.get("data"),
            "capabilities": manifest.get("capability_identity"),
        }
    )
    if preflight_stage in {"calibration", "pilot"}:
        ignored_evidence = (
            {"pilot_report", "storage_concurrency_report"}
            if preflight_stage == "calibration"
            else {"pilot_report"}
        )
        placeholders = [
            path
            for path in placeholders
            if not any(
                path.startswith(f"capabilities.evidence.{name}")
                for name in ignored_evidence
            )
        ]
    if placeholders:
        raise ValueError(
            "Manifest contains unresolved placeholders: " + ", ".join(placeholders)
        )
    code = git_identity(spec["_repo_root"])
    manifest_code = manifest.get("code", {})
    if code.get("dirty"):
        raise ValueError("Current worktree is dirty")
    if manifest_code.get("commit") != code.get("commit"):
        raise ValueError(
            f"Manifest commit {manifest_code.get('commit')} != checkout {code.get('commit')}"
        )


def execute_manifest(
    *,
    spec: dict[str, Any],
    manifest: dict[str, Any],
    worker_command: list[str],
    stage: str,
) -> dict[str, Any]:
    run_dir = Path(manifest["_manifest_path"]).resolve().parent
    manifest_path = Path(manifest["_manifest_path"]).resolve()
    prune_step = int(manifest["fidelity"]["prune_step"])
    final_step = int(manifest["fidelity"]["max_steps"])

    world_size = int(manifest["protocol"]["world_size"])
    successful_gpu_seconds = 0.0
    successful_gpu_hours = 0.0
    stage_started: float | None = None
    stage_accounted = False
    stage_reported_gpu_seconds: float | None = None
    stage_reported_gpu_hours: float | None = None
    try:
        screen_path = run_dir / "metrics" / "step_500.json"
        stage_started = time.monotonic()
        stage_accounted = False
        stage_reported_gpu_seconds = None
        stage_reported_gpu_hours = None
        if stage in {"screen", "both"}:
            screen = run_adapter_stage(
                worker_command,
                manifest_path=manifest_path,
                run_dir=run_dir,
                stage="screen",
                metrics_path=screen_path,
            )
        else:
            screen = _read_object(screen_path)
        stage_reported_gpu_seconds = float(screen["gpu_seconds_actual"])
        stage_reported_gpu_hours = float(screen["gpu_hours_actual"])
        validate_step_metrics(
            manifest,
            screen,
            expected_step=prune_step,
            require_full_mqar_grid=True,
            require_wikitext_ppl=False,
            diagnostic_names=manifest["screen_required_diagnostics"],
        )
        successful_gpu_seconds += stage_reported_gpu_seconds
        successful_gpu_hours += stage_reported_gpu_hours
        stage_accounted = True
        if stage == "screen":
            summary = _summary(
                manifest,
                "SCREEN_COMPLETE",
                screen,
                gpu_seconds_actual=successful_gpu_seconds,
                gpu_hours_actual=successful_gpu_hours,
                gpu_time_source="controller_aggregated_stage_invocations",
            )
            atomic_write_json(run_dir / "screen_summary.json", summary)
            return summary

        final_path = run_dir / "metrics" / "final.json"
        stage_started = time.monotonic()
        stage_accounted = False
        stage_reported_gpu_seconds = None
        stage_reported_gpu_hours = None
        final = run_adapter_stage(
            worker_command,
            manifest_path=manifest_path,
            run_dir=run_dir,
            stage="final",
            metrics_path=final_path,
        )
        stage_reported_gpu_seconds = float(final["gpu_seconds_actual"])
        stage_reported_gpu_hours = float(final["gpu_hours_actual"])
        validate_step_metrics(
            manifest,
            final,
            expected_step=final_step,
            require_full_mqar_grid=True,
            require_wikitext_ppl=True,
        )
        validate_cross_stage_identity(screen, final)
        successful_gpu_seconds += stage_reported_gpu_seconds
        successful_gpu_hours += stage_reported_gpu_hours
        stage_accounted = True
        if not math.isclose(
            successful_gpu_hours,
            successful_gpu_seconds / 3600.0,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "Cumulative pilot GPU seconds/hours disagree across stages"
            )
        summary = _summary(
            manifest,
            "COMPLETE",
            final,
            gpu_seconds_actual=successful_gpu_seconds,
            gpu_hours_actual=successful_gpu_hours,
            gpu_time_source="controller_aggregated_stage_invocations",
            step_500_mqar_accuracy=float(screen["mqar_accuracy"]),
        )
        atomic_write_json(run_dir / "trial_summary.json", summary)
        return summary
    except BaseException as exc:
        if (
            not stage_accounted
            and stage_reported_gpu_seconds is not None
            and stage_reported_gpu_hours is not None
        ):
            successful_gpu_seconds += stage_reported_gpu_seconds
            successful_gpu_hours += stage_reported_gpu_hours
            gpu_time_source = "controller_aggregated_with_uncommitted_stage"
        elif stage_started is not None and not stage_accounted:
            successful_gpu_seconds += (
                max(0.0, time.monotonic() - stage_started) * world_size
            )
            successful_gpu_hours = successful_gpu_seconds / 3600.0
            gpu_time_source = "controller_aggregated_with_wall_clock_fallback"
        else:
            gpu_time_source = "controller_aggregated_stage_invocations"
        atomic_write_json(
            run_dir / "trial_summary.json",
            _summary(
                manifest,
                "FAIL",
                failure=exc,
                gpu_seconds_actual=successful_gpu_seconds,
                gpu_hours_actual=successful_gpu_hours,
                gpu_time_source=gpu_time_source,
            ),
        )
        raise


def execute_budgeted_manifest(
    *,
    spec: dict[str, Any],
    manifest: dict[str, Any],
    worker_command: list[str],
    budget_root: Path,
    heartbeat_interval: float,
) -> dict[str, Any]:
    """Run a post-pilot fixed manifest against the shared scientific budget."""

    run_dir = Path(manifest["_manifest_path"]).resolve().parent
    manifest_path = Path(manifest["_manifest_path"]).resolve()
    prune_step = int(manifest["fidelity"]["prune_step"])
    final_step = int(manifest["fidelity"]["max_steps"])
    budget = spec["study"]["compute_budget"]
    claim = claim_trial_hash(
        budget_root,
        manifest["trial_hash"],
        -1,
        run_dir=run_dir,
        stale_after_seconds=int(spec["storage"]["claim_stale_after_seconds"]),
        maximum_total_gpu_hours=float(budget["maximum_total_gpu_hours"]),
        maximum_full_trial_gpu_hours=float(
            budget["maximum_full_trial_gpu_hours"]
        ),
        maximum_concurrent_trials=int(budget["maximum_concurrent_trials"]),
        claim_kind="fixed_manifest",
    )
    if claim is None:
        claim_path = (
            budget_root.expanduser().resolve()
            / "claims"
            / f"{manifest['trial_hash']}.json"
        )
        claim_payload = _read_object(claim_path)
        summary_path = run_dir / "trial_summary.json"
        if summary_path.is_file():
            summary = _read_object(summary_path)
            _validate_reusable_terminal_summary(
                manifest,
                summary,
                claim_payload,
            )
            return summary
        raise RuntimeError(
            "Fixed trial already has a live or terminal content-addressed claim"
        )

    stage_started: float | None = None
    stage_accounted = False
    stage_reported_gpu_seconds: float | None = None
    stage_reported_gpu_hours: float | None = None
    try:
        screen_path = run_dir / "metrics" / "step_500.json"
        stage_started = time.monotonic()
        stage_reported_gpu_seconds = None
        stage_reported_gpu_hours = None
        screen = run_adapter_stage(
            worker_command,
            manifest_path=manifest_path,
            run_dir=run_dir,
            stage="screen",
            metrics_path=screen_path,
            claim=claim,
            claim_heartbeat_interval_seconds=heartbeat_interval,
        )
        stage_reported_gpu_seconds = float(screen["gpu_seconds_actual"])
        stage_reported_gpu_hours = float(screen["gpu_hours_actual"])
        validate_step_metrics(
            manifest,
            screen,
            expected_step=prune_step,
            require_full_mqar_grid=True,
            require_wikitext_ppl=False,
            diagnostic_names=manifest["screen_required_diagnostics"],
        )
        record_successful_claim_stage(
            claim,
            stage="screen",
            gpu_seconds_actual=stage_reported_gpu_seconds,
            gpu_hours_actual=stage_reported_gpu_hours,
        )
        stage_accounted = True

        final_path = run_dir / "metrics" / "final.json"
        stage_started = time.monotonic()
        stage_accounted = False
        stage_reported_gpu_seconds = None
        stage_reported_gpu_hours = None
        final = run_adapter_stage(
            worker_command,
            manifest_path=manifest_path,
            run_dir=run_dir,
            stage="final",
            metrics_path=final_path,
            claim=claim,
            claim_heartbeat_interval_seconds=heartbeat_interval,
        )
        stage_reported_gpu_seconds = float(final["gpu_seconds_actual"])
        stage_reported_gpu_hours = float(final["gpu_hours_actual"])
        validate_step_metrics(
            manifest,
            final,
            expected_step=final_step,
            require_full_mqar_grid=True,
            require_wikitext_ppl=True,
        )
        validate_cross_stage_identity(screen, final)
        record_successful_claim_stage(
            claim,
            stage="final",
            gpu_seconds_actual=stage_reported_gpu_seconds,
            gpu_hours_actual=stage_reported_gpu_hours,
        )
        stage_accounted = True
        gpu_seconds, gpu_hours = _cumulative_claim_gpu_time(
            claim,
            gpu_seconds_actual=0.0,
            gpu_hours_actual=0.0,
        )
        summary = _summary(
            manifest,
            "COMPLETE",
            final,
            gpu_seconds_actual=gpu_seconds,
            gpu_hours_actual=gpu_hours,
            gpu_time_source="controller_aggregated_invocations",
            step_500_mqar_accuracy=float(screen["mqar_accuracy"]),
        )
        summary_path = run_dir / "trial_summary.json"
        try:
            atomic_write_json(summary_path, summary)
            finalize_trial_claim(
                claim,
                "COMPLETE",
                gpu_hours_actual=gpu_hours,
                terminal_artifacts=terminal_artifact_bindings(run_dir),
            )
        except BaseException:
            summary_path.unlink(missing_ok=True)
            raise
        return summary
    except TrialLocalFailure as exc:
        failure_seconds = float(exc.gpu_seconds_actual or 0.0)
        failure_hours = float(exc.gpu_hours_actual or 0.0)
        failed_stage_recorded = False
        summary_path = run_dir / "trial_summary.json"
        try:
            record_failed_claim_stage(
                claim,
                stage=exc.stage,
                failure=exc.failure
                or {
                    "scope": "trial",
                    "code": type(exc).__name__,
                    "detail": str(exc),
                },
                gpu_seconds_actual=failure_seconds,
                gpu_hours_actual=failure_hours,
            )
            failed_stage_recorded = True
            gpu_seconds, gpu_hours = _cumulative_claim_gpu_time(
                claim,
                gpu_seconds_actual=0.0,
                gpu_hours_actual=0.0,
            )
            summary = _summary(
                manifest,
                "FAIL",
                failure=exc,
                gpu_seconds_actual=gpu_seconds,
                gpu_hours_actual=gpu_hours,
                gpu_time_source="controller_aggregated_invocations",
            )
            atomic_write_json(summary_path, summary)
            finalize_trial_claim(
                claim,
                "FAIL",
                gpu_hours_actual=gpu_hours,
                terminal_artifacts=terminal_artifact_bindings(run_dir),
            )
        except BaseException as persistence_exc:
            summary_path.unlink(missing_ok=True)
            abandon_trial_claim_for_recovery(
                claim,
                failure={
                    "scope": "systemic",
                    "code": type(persistence_exc).__name__,
                    "detail": str(persistence_exc),
                },
                # The failed stage was already reconciled and charged above.
                gpu_seconds_actual=(
                    0.0 if failed_stage_recorded else failure_seconds
                ),
                gpu_hours_actual=(
                    0.0 if failed_stage_recorded else failure_hours
                ),
            )
            raise
        raise
    except BaseException as exc:
        if (
            isinstance(exc, AdapterFailure)
            and exc.gpu_seconds_actual is not None
            and exc.gpu_hours_actual is not None
        ):
            failure_seconds = float(exc.gpu_seconds_actual)
            failure_hours = float(exc.gpu_hours_actual)
            gpu_time_source = "adapter_failure_envelope"
        elif (
            not stage_accounted
            and stage_reported_gpu_seconds is not None
            and stage_reported_gpu_hours is not None
        ):
            failure_seconds = stage_reported_gpu_seconds
            failure_hours = stage_reported_gpu_hours
            gpu_time_source = "adapter_reported_uncommitted_stage"
        else:
            failure_seconds = (
                0.0
                if stage_started is None or stage_accounted
                else max(0.0, time.monotonic() - stage_started)
            )
            failure_hours = failure_seconds / 3600.0
            gpu_time_source = "controller_wall_time_single_gpu"
        attempt = _summary(
            manifest,
            "FAIL",
            failure=exc,
            gpu_seconds_actual=failure_seconds,
            gpu_hours_actual=failure_hours,
            gpu_time_source=gpu_time_source,
        )
        if stage_started is None:
            attempt["recoverable_systemic_attempt"] = False
            try:
                atomic_write_json(run_dir / "trial_setup_failure.json", attempt)
            finally:
                release_trial_claim(claim)
            raise
        attempt["recoverable_systemic_attempt"] = True
        try:
            atomic_write_json(run_dir / "trial_attempt_failure.json", attempt)
        finally:
            abandon_trial_claim_for_recovery(
                claim,
                failure={
                    "scope": "systemic",
                    "code": type(exc).__name__,
                    "detail": str(exc),
                },
                gpu_seconds_actual=failure_seconds,
                gpu_hours_actual=failure_hours,
            )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", default="configs/phase2a_search.json")
    parser.add_argument("--capabilities", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--stage", choices=("screen", "final", "both"), default="both")
    parser.add_argument(
        "--preflight-stage",
        choices=("calibration", "pilot", "study"),
        default="pilot",
        help=(
            "Calibration is a non-scientific screen-only budget measurement; "
            "pilot omits only the not-yet-created pilot report; use study for "
            "post-pilot controls and promoted runs."
        ),
    )
    parser.add_argument(
        "--budget-root",
        help=(
            "Shared Phase 2a output root containing the scientific claim/budget "
            "ledger; required for study-stage controls and promotions."
        ),
    )
    parser.add_argument(
        "--storage",
        help=(
            "The same Optuna storage URL used by sampled workers; required "
            "with --preflight-stage study and hashed without credentials into "
            "the immutable shared-ledger contract."
        ),
    )
    parser.add_argument("--heartbeat-interval", type=int, default=60)
    parser.add_argument("worker_command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    worker_command = list(args.worker_command)
    if worker_command and worker_command[0] == "--":
        worker_command = worker_command[1:]
    if not worker_command:
        raise SystemExit("Pass the finalized training adapter after '--'")
    if args.heartbeat_interval <= 0:
        raise SystemExit("--heartbeat-interval must be positive")
    if args.preflight_stage == "study":
        if not args.budget_root:
            raise SystemExit(
                "--budget-root is required for study-stage fixed manifests"
            )
        if not args.storage:
            raise SystemExit(
                "--storage is required for study-stage fixed manifests"
            )
        if args.stage != "both":
            raise SystemExit(
                "Study-stage fixed manifests must run --stage both so their "
                "reservation reaches a terminal outcome"
            )
    elif args.preflight_stage == "calibration" and args.stage != "screen":
        raise SystemExit(
            "Calibration manifests are screen-only; pass --stage screen"
        )

    spec = load_spec(args.spec)
    capabilities = _read_object(args.capabilities)
    manifest = _read_object(args.manifest)
    preflight = evaluate_preflight(
        spec,
        capabilities,
        data_manifest=manifest.get("data"),
        stage=args.preflight_stage,
        require_clean=True,
    )
    if not preflight["ready"]:
        print(json.dumps(preflight, indent=2, sort_keys=True))
        raise SystemExit("Scientific launch preflight failed")
    manifest["_manifest_path"] = str(Path(args.manifest).expanduser().resolve())
    validate_manifest_identity(
        spec,
        manifest,
        capabilities,
        preflight_stage=args.preflight_stage,
    )
    if args.preflight_stage == "study":
        stale_after = int(spec["storage"]["claim_stale_after_seconds"])
        if stale_after <= args.heartbeat_interval * 3:
            raise SystemExit(
                "storage.claim_stale_after_seconds must exceed three heartbeat "
                "intervals"
            )
        budget_root = Path(args.budget_root).expanduser().resolve()
        ledger_contract = build_ledger_contract(
            spec,
            output_root=budget_root,
            storage=args.storage,
            code_identity=git_identity(spec["_repo_root"]),
            data_identity=manifest["data"],
            capability_identity=capabilities,
        )
        ensure_ledger_contract(
            budget_root,
            ledger_contract,
            create=False,
        )
        summary = execute_budgeted_manifest(
            spec=spec,
            manifest=manifest,
            worker_command=worker_command,
            budget_root=budget_root,
            heartbeat_interval=float(args.heartbeat_interval),
        )
    else:
        summary = execute_manifest(
            spec=spec,
            manifest=manifest,
            worker_command=worker_command,
            stage=args.stage,
        )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
