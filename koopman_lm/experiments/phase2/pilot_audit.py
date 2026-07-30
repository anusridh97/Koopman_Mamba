"""Audit the fixed Phase 2a pilot and emit its signed-off evidence report."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping

try:
    from .manifest import (
        atomic_write_json,
        build_trial_manifest,
        recompute_promotion_config_hash,
        recompute_trial_hash,
    )
    from .pilot import pilot_parameter_matrix
    from .preflight import evaluate_preflight
    from .results import (
        Phase2ResultError,
        validate_cross_stage_identity,
        validate_step_metrics,
    )
    from .spec import load_spec, stable_hash
except ImportError:
    from manifest import (  # type: ignore
        atomic_write_json,
        build_trial_manifest,
        recompute_promotion_config_hash,
        recompute_trial_hash,
    )
    from pilot import pilot_parameter_matrix  # type: ignore
    from preflight import evaluate_preflight  # type: ignore
    from results import (  # type: ignore
        Phase2ResultError,
        validate_cross_stage_identity,
        validate_step_metrics,
    )
    from spec import load_spec, stable_hash  # type: ignore


class PilotAuditError(ValueError):
    """The materialized pilot is incomplete or contains an invalid result."""


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise PilotAuditError(f"Could not read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PilotAuditError(f"Expected JSON object: {path}")
    return value


def _safe_run_dir(root: Path, relative: Any) -> Path:
    if not isinstance(relative, str):
        raise PilotAuditError(f"Invalid pilot run_directory: {relative!r}")
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise PilotAuditError(f"Pilot run directory escapes root: {relative}") from exc
    return path


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _expected_study_identity(manifest: Mapping[str, Any]) -> dict[str, Any]:
    code = manifest.get("code")
    if not isinstance(code, Mapping):
        raise PilotAuditError("Pilot manifest has no code identity")
    return {
        "study_name": manifest.get("study_name"),
        "spec_hash": manifest.get("spec_hash"),
        "code_commit": code.get("commit"),
        "code_dirty": code.get("dirty"),
        "data_hash": stable_hash(manifest.get("data")),
        "capability_hash": stable_hash(manifest.get("capability_identity")),
    }


def _expected_execution_plan(
    spec: Mapping[str, Any],
    capabilities: Mapping[str, Any],
    data_manifest: Mapping[str, Any],
    *,
    seed: int,
    code_identity: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Rebuild the immutable pilot plan instead of trusting its summary."""

    expected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, params in enumerate(pilot_parameter_matrix(spec)):
        manifest = build_trial_manifest(
            spec,
            params,
            seed=seed,
            trial_number=index,
            code_identity=code_identity,
            data_identity=data_manifest,
            capability_identity=capabilities,
        )
        trial_hash = manifest["trial_hash"]
        if trial_hash in seen:
            continue
        seen.add(trial_hash)
        reason = str(
            params.get("pilot_reason")
            or params.get("name")
            or f"pilot-{index:03d}"
        ).replace("/", "-")
        requires_final = bool(params.get("name")) or reason in {
            "rank128-ridge1e-4",
            "qknorm-on",
            "birdie-mix",
        }
        expected.append(
            {
                "trial_hash": trial_hash,
                "reason": reason,
                "run_directory": str(
                    Path("trials") / f"{index:03d}-{reason}"
                ),
                "required_stage": "final" if requires_final else "screen",
            }
        )
    return expected


def _validate_metrics(
    manifest: Mapping[str, Any],
    metrics: Mapping[str, Any],
    *,
    expected_step: int,
    require_wikitext_ppl: bool,
    screen: bool,
    trial_hash: str,
) -> None:
    try:
        validate_step_metrics(
            manifest,
            metrics,
            expected_step=expected_step,
            require_full_mqar_grid=True,
            require_wikitext_ppl=require_wikitext_ppl,
            **(
                {"diagnostic_names": manifest["screen_required_diagnostics"]}
                if screen
                else {}
            ),
        )
    except (Phase2ResultError, KeyError, TypeError, ValueError) as exc:
        stage = "step-500" if screen else "final"
        raise PilotAuditError(
            f"Pilot {stage} metrics failed validation for {trial_hash}: {exc}"
        ) from exc


def _validate_summary(
    manifest: Mapping[str, Any],
    summary: Mapping[str, Any],
    *,
    status: str,
    terminal_metrics: Mapping[str, Any],
    screen_metrics: Mapping[str, Any],
    final_metrics: Mapping[str, Any] | None,
    trial_hash: str,
) -> None:
    expected_equalities = {
        "schema_version": 1,
        "trial_hash": trial_hash,
        "promotion_config_hash": manifest.get("promotion_config_hash"),
        "study_identity": _expected_study_identity(manifest),
        "seed": manifest.get("protocol", {}).get("seed"),
        "status": status,
        "step_500_mqar_accuracy": screen_metrics.get("mqar_accuracy"),
        "health_passed": True,
        "failure": None,
        "optuna_trial_id": None,
        "gpu_time_source": "controller_aggregated_stage_invocations",
        "model_accounting": terminal_metrics.get("model_accounting"),
        "optimizer_group_audit": terminal_metrics.get("optimizer_group_audit"),
    }
    for field, expected in expected_equalities.items():
        if summary.get(field) != expected:
            raise PilotAuditError(
                f"Pilot summary {field} mismatch for {trial_hash}: "
                f"{summary.get(field)!r} != {expected!r}"
            )

    wandb_run_id = terminal_metrics.get("wandb_run_id")
    if not _nonempty_string(wandb_run_id):
        raise PilotAuditError(
            f"Pilot metrics have no nonempty W&B run ID for {trial_hash}"
        )
    if summary.get("wandb_run_id") != wandb_run_id:
        raise PilotAuditError(
            f"Pilot summary W&B run ID mismatch for {trial_hash}"
        )
    if screen_metrics.get("wandb_run_id") != wandb_run_id:
        raise PilotAuditError(
            f"Pilot screen/final W&B run IDs disagree for {trial_hash}"
        )

    if final_metrics is not None:
        for field in ("model_accounting", "optimizer_group_audit"):
            if screen_metrics.get(field) != final_metrics.get(field):
                raise PilotAuditError(
                    f"Pilot screen/final {field} disagree for {trial_hash}"
                )

    for field in ("mqar_accuracy", "wikitext_ppl"):
        expected = terminal_metrics.get(field)
        actual = summary.get(field)
        if expected is None:
            if actual is not None:
                raise PilotAuditError(
                    f"Pilot summary {field} must be null for {trial_hash}"
                )
        elif (
            not _finite_number(actual)
            or not math.isclose(
                float(actual),
                float(expected),
                rel_tol=1e-9,
                abs_tol=1e-12,
            )
        ):
            raise PilotAuditError(
                f"Pilot summary {field} mismatch for {trial_hash}"
            )

    stages = [screen_metrics]
    if final_metrics is not None:
        stages.append(final_metrics)
    expected_seconds = sum(float(metrics["gpu_seconds_actual"]) for metrics in stages)
    expected_hours = sum(float(metrics["gpu_hours_actual"]) for metrics in stages)
    if not math.isclose(
        expected_hours,
        expected_seconds / 3600.0,
        rel_tol=1e-9,
        abs_tol=1e-12,
    ):
        raise PilotAuditError(
            f"Pilot stage GPU seconds/hours disagree for {trial_hash}"
        )
    for field, expected in (
        ("gpu_seconds_actual", expected_seconds),
        ("gpu_hours_actual", expected_hours),
    ):
        actual = summary.get(field)
        if (
            not _finite_number(actual)
            or float(actual) < 0.0
            or not math.isclose(
                float(actual),
                expected,
                rel_tol=1e-9,
                abs_tol=1e-12,
            )
        ):
            raise PilotAuditError(
                f"Pilot cumulative {field} mismatch for {trial_hash}"
            )


def audit_pilot(
    *,
    spec: Mapping[str, Any],
    capabilities: Mapping[str, Any],
    data_manifest: Mapping[str, Any],
    pilot_root: str | Path,
) -> dict[str, Any]:
    """Return a pass report only when every planned pilot stage completed."""

    root = Path(pilot_root).expanduser().resolve()
    summary = _read_object(root / "pilot_summary.json")
    if not summary.get("scientific_launch_ready"):
        raise PilotAuditError(
            "Pilot was not materialized from a passing pilot-stage preflight"
        )
    if summary.get("study_name") != spec["study_name"]:
        raise PilotAuditError("Pilot study name does not match the search spec")
    plan = summary.get("execution_plan")
    if not isinstance(plan, list) or not plan:
        raise PilotAuditError("Pilot summary has no execution plan")
    seed = summary.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise PilotAuditError("Pilot summary must record its integer seed")
    code_identity = summary.get("code")
    if (
        not isinstance(code_identity, Mapping)
        or code_identity.get("commit")
        != capabilities.get("integration_commit")
        or not isinstance(code_identity.get("dirty"), bool)
    ):
        raise PilotAuditError(
            "Pilot summary code identity does not match the capability commit"
        )
    expected_plan = _expected_execution_plan(
        spec,
        capabilities,
        data_manifest,
        seed=seed,
        code_identity=code_identity,
    )
    if plan != expected_plan:
        raise PilotAuditError(
            "Pilot execution plan does not exactly match the fixed matrix"
        )
    if int(summary.get("pilot_manifest_count", -1)) != len(plan):
        raise PilotAuditError("Pilot manifest count does not match execution plan")
    if summary.get("trial_hashes") != [
        entry["trial_hash"] for entry in expected_plan
    ]:
        raise PilotAuditError(
            "Pilot summary trial_hashes do not exactly match its fixed plan"
        )

    seen: set[str] = set()
    result_identities: list[dict[str, Any]] = []
    public_spec = {
        key: value for key, value in spec.items() if not str(key).startswith("_")
    }
    expected_spec_hash = stable_hash(public_spec)
    completed_screen = 0
    completed_final = 0
    validated_metric_files = 0
    validated_accounting_summaries = 0
    validated_wandb_crosslinks = 0
    wandb_to_trial: dict[str, str] = {}
    for entry in plan:
        if not isinstance(entry, Mapping):
            raise PilotAuditError("Pilot execution entries must be objects")
        trial_hash = entry.get("trial_hash")
        if not isinstance(trial_hash, str) or len(trial_hash) != 64:
            raise PilotAuditError(f"Invalid pilot trial hash: {trial_hash!r}")
        if trial_hash in seen:
            raise PilotAuditError(f"Duplicate pilot trial hash: {trial_hash}")
        seen.add(trial_hash)
        run_dir = _safe_run_dir(root, entry.get("run_directory"))
        manifest = _read_object(run_dir / "trial_manifest.json")
        if (
            manifest.get("trial_hash") != trial_hash
            or recompute_trial_hash(manifest) != trial_hash
            or recompute_promotion_config_hash(manifest)
            != manifest.get("promotion_config_hash")
        ):
            raise PilotAuditError(f"Pilot manifest hash mismatch: {trial_hash}")
        if manifest.get("spec_hash") != expected_spec_hash:
            raise PilotAuditError(f"Pilot manifest spec mismatch: {trial_hash}")
        if manifest.get("capability_identity") != capabilities:
            raise PilotAuditError(
                f"Pilot manifest capability identity mismatch: {trial_hash}"
            )
        if manifest.get("data") != data_manifest:
            raise PilotAuditError(
                f"Pilot manifest data identity mismatch: {trial_hash}"
            )
        if manifest.get("code", {}).get("commit") != capabilities.get(
            "integration_commit"
        ):
            raise PilotAuditError(
                f"Pilot manifest integration commit mismatch: {trial_hash}"
            )

        prune_step = int(manifest["fidelity"]["prune_step"])
        final_step = int(manifest["fidelity"]["max_steps"])
        screen_metrics = _read_object(run_dir / "metrics" / "step_500.json")
        _validate_metrics(
            manifest,
            screen_metrics,
            expected_step=prune_step,
            require_wikitext_ppl=False,
            screen=True,
            trial_hash=trial_hash,
        )
        validated_metric_files += 1

        required_stage = entry.get("required_stage")
        final_metrics: dict[str, Any] | None = None
        if required_stage == "final":
            result = _read_object(run_dir / "trial_summary.json")
            if result.get("status") != "COMPLETE":
                raise PilotAuditError(
                    f"Final pilot arm {trial_hash} is {result.get('status')!r}"
                )
            final_metrics = _read_object(run_dir / "metrics" / "final.json")
            _validate_metrics(
                manifest,
                final_metrics,
                expected_step=final_step,
                require_wikitext_ppl=True,
                screen=False,
                trial_hash=trial_hash,
            )
            validated_metric_files += 1
            completed_final += 1
        elif required_stage == "screen":
            final_path = run_dir / "trial_summary.json"
            if final_path.is_file():
                result = _read_object(final_path)
                valid_status = result.get("status") == "COMPLETE"
                if valid_status:
                    final_metrics = _read_object(run_dir / "metrics" / "final.json")
                    _validate_metrics(
                        manifest,
                        final_metrics,
                        expected_step=final_step,
                        require_wikitext_ppl=True,
                        screen=False,
                        trial_hash=trial_hash,
                    )
                    validated_metric_files += 1
            else:
                result = _read_object(run_dir / "screen_summary.json")
                valid_status = result.get("status") == "SCREEN_COMPLETE"
            if not valid_status:
                raise PilotAuditError(
                    f"Screen pilot arm {trial_hash} is {result.get('status')!r}"
                )
            completed_screen += 1
        else:
            raise PilotAuditError(f"Unknown required pilot stage: {required_stage!r}")
        if result.get("trial_hash") != trial_hash:
            raise PilotAuditError(
                f"Pilot result hash does not match its plan: {trial_hash}"
            )

        terminal_metrics = final_metrics or screen_metrics
        _validate_summary(
            manifest,
            result,
            status="COMPLETE" if final_metrics is not None else "SCREEN_COMPLETE",
            terminal_metrics=terminal_metrics,
            screen_metrics=screen_metrics,
            final_metrics=final_metrics,
            trial_hash=trial_hash,
        )
        if final_metrics is not None:
            try:
                validate_cross_stage_identity(
                    screen_metrics,
                    final_metrics,
                )
            except Phase2ResultError as exc:
                raise PilotAuditError(
                    f"Pilot resume provenance failed for {trial_hash}: {exc}"
                ) from exc
        validated_accounting_summaries += 1
        validated_wandb_crosslinks += 1
        wandb_run_id = str(result["wandb_run_id"])
        prior_trial = wandb_to_trial.setdefault(wandb_run_id, trial_hash)
        if prior_trial != trial_hash:
            raise PilotAuditError(
                "One W&B run ID is linked to multiple pilot trial hashes: "
                f"{wandb_run_id!r} -> {prior_trial}, {trial_hash}"
            )
        result_identities.append(
            {
                "trial_hash": trial_hash,
                "required_stage": required_stage,
                "status": result.get("status"),
                "summary_sha256": stable_hash(result),
                "screen_metrics_sha256": stable_hash(screen_metrics),
                "final_metrics_sha256": (
                    stable_hash(final_metrics)
                    if final_metrics is not None
                    else None
                ),
                "accounting_sha256": stable_hash(
                    {
                        "model_accounting": terminal_metrics["model_accounting"],
                        "optimizer_group_audit": terminal_metrics[
                            "optimizer_group_audit"
                        ],
                        "gpu_seconds_actual": result["gpu_seconds_actual"],
                        "gpu_hours_actual": result["gpu_hours_actual"],
                    }
                ),
                "wandb_run_id": result["wandb_run_id"],
            }
        )

    coverage = summary.get("coverage", {})
    expected_coverage = {
        "architecture_modes": {"updated_sweep", "paper_control", "mamba_only"},
        "ranks": {32, 48, 128},
        "fractions": {0.15, 0.2, 0.25, 0.33},
        "chunks": {32, 64, 128},
        "placements": {"uniform", "late_biased", "middle_clustered"},
        "qk_norm": {False, True},
        "objective_arms": {"next_token_only", "birdie_mix"},
    }
    for name, expected in expected_coverage.items():
        actual = coverage.get(name)
        if not isinstance(actual, list) or set(actual) != expected:
            raise PilotAuditError(
                f"Pilot coverage {name}={actual!r} does not exactly equal "
                f"{sorted(expected)!r}"
            )
    if completed_final < 1:
        raise PilotAuditError("Pilot must complete at least one exact-resume survivor")

    return {
        "schema_version": 1,
        "status": "pass",
        "evidence_type": "pilot_report",
        "integration_commit": capabilities["integration_commit"],
        "spec_hash": expected_spec_hash,
        "data_manifest_sha256": stable_hash(data_manifest),
        "checks": [
            {"name": "fixed_matrix_coverage", "status": "pass"},
            {"name": "all_planned_screen_arms_complete", "status": "pass"},
            {"name": "required_full_survivors_complete", "status": "pass"},
            {"name": "trial_hash_integrity", "status": "pass"},
            {
                "name": "validated_step_and_final_metrics",
                "status": "pass",
                "detail": f"{validated_metric_files} metrics artifacts revalidated",
            },
            {
                "name": "validated_model_optimizer_gpu_accounting",
                "status": "pass",
                "detail": (
                    f"{validated_accounting_summaries} summaries matched their "
                    "validated accounting"
                ),
            },
            {
                "name": "checkpoint_resume_provenance",
                "status": "pass",
                "detail": (
                    f"{completed_final} final arms restored the exact "
                    "step-500 checkpoint identity and data-stream offset"
                ),
            },
            {
                "name": "wandb_run_crosslinks",
                "status": "pass",
                "detail": (
                    f"{validated_wandb_crosslinks} runs had one matching nonempty "
                    "W&B run ID across stages and summary"
                ),
            },
        ],
        "pilot_root": str(root),
        "pilot_summary_sha256": stable_hash(summary),
        "result_summary_set_sha256": stable_hash(
            sorted(result_identities, key=lambda item: item["trial_hash"])
        ),
        "pilot_manifest_count": len(plan),
        "screen_only_complete": completed_screen,
        "full_survivors_complete": completed_final,
        "validated_metric_file_count": validated_metric_files,
        "validated_accounting_summary_count": validated_accounting_summaries,
        "validated_wandb_crosslink_count": validated_wandb_crosslinks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", default="configs/phase2a_search.json")
    parser.add_argument("--capabilities", required=True)
    parser.add_argument("--data-manifest", required=True)
    parser.add_argument("--pilot-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()

    spec = load_spec(args.spec)
    capabilities = _read_object(Path(args.capabilities))
    data_manifest = _read_object(Path(args.data_manifest))
    preflight = evaluate_preflight(
        spec,
        capabilities,
        data_manifest=data_manifest,
        stage="pilot",
        require_clean=not args.allow_dirty,
    )
    if not preflight["ready"]:
        print(json.dumps(preflight, indent=2, sort_keys=True))
        raise SystemExit("Pilot-stage preflight failed")
    try:
        report = audit_pilot(
            spec=spec,
            capabilities=capabilities,
            data_manifest=data_manifest,
            pilot_root=args.pilot_root,
        )
    except PilotAuditError as exc:
        raise SystemExit(f"Pilot audit failed: {exc}") from exc
    atomic_write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
