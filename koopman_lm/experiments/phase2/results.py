"""Strict result validation shared by future Phase 2 training adapters."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping, Sequence


class Phase2ResultError(ValueError):
    """A trial result violates the study's immutable output contract."""


def validate_cross_stage_identity(
    screen_metrics: Mapping[str, Any],
    final_metrics: Mapping[str, Any],
) -> None:
    """Require one W&B run to span screen and resumed final execution."""

    screen_run_id = screen_metrics.get("wandb_run_id")
    final_run_id = final_metrics.get("wandb_run_id")
    if (
        not isinstance(screen_run_id, str)
        or not screen_run_id.strip()
        or final_run_id != screen_run_id
    ):
        raise Phase2ResultError(
            "Screen and final metrics must use the same nonempty wandb_run_id"
        )
    checkpoint = screen_metrics.get("checkpoint_provenance")
    resume = final_metrics.get("resume_provenance")
    if not isinstance(checkpoint, Mapping) or not isinstance(resume, Mapping):
        raise Phase2ResultError(
            "Screen checkpoint and final resume provenance must be objects"
        )
    comparisons = {
        "source_checkpoint_sha256": checkpoint.get("checkpoint_sha256"),
        "source_checkpoint_bundle_path": checkpoint.get(
            "checkpoint_bundle_path"
        ),
        "source_checkpoint_bundle_byte_size": checkpoint.get(
            "checkpoint_bundle_byte_size"
        ),
        "source_checkpoint_bundle_file_count": checkpoint.get(
            "checkpoint_bundle_file_count"
        ),
        "source_checkpoint_digest_algorithm": checkpoint.get(
            "checkpoint_digest_algorithm"
        ),
        "source_optimizer_step": checkpoint.get("optimizer_step"),
        "restored_global_sequence_index": checkpoint.get(
            "global_sequence_index"
        ),
        "checkpoint_format_revision": checkpoint.get(
            "checkpoint_format_revision"
        ),
    }
    for field, expected in comparisons.items():
        if resume.get(field) != expected:
            raise Phase2ResultError(
                f"Final resume provenance {field}={resume.get(field)!r} "
                f"does not match screen checkpoint value {expected!r}"
            )


_MQAR_LENGTHS = (256, 512, 1024, 2048)
_MQAR_PAIRS = (4, 8, 16, 32, 64)
_AGGREGATE_REL_TOL = 1e-6
_AGGREGATE_ABS_TOL = 1e-8
_FLOP_SCOPE = "training_forward_backward_per_optimizer_step"
_REQUIRED_HEAD_METRICS = {
    "spectral_radius_raw_preclamp_mean",
    "spectral_radius_raw_preclamp_max",
    "spectral_radius_normalized_pre_gamma_mean",
    "spectral_radius_normalized_pre_gamma_max",
    "spectral_radius_applied_post_gamma_mean",
    "spectral_radius_applied_post_gamma_max",
    "clamp_factor_mean",
    "clamp_fraction",
    "spectral_gap_mean",
    "spectral_gap_max",
    "lambda_min_gram_mean",
    "lambda_min_gram_min",
    "ridge_condition_number_mean",
    "ridge_condition_number_max",
    "ridge_escalation_count",
    "effective_ridge_min",
    "effective_ridge_max",
}
_REQUIRED_LAYER_METRICS = {
    "residual_norm_ratio",
    "gradient_norm_ratio",
    "effective_jacobian_rank",
    "beta_mean",
    "beta_saturation_fraction",
    "layerscale_magnitude_mean",
}


def _is_finite_number(value: Any) -> bool:
    """Return true for finite JSON numbers, excluding bool-as-int accidents."""

    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _require_finite_number(value: Any, label: str) -> float:
    if not _is_finite_number(value):
        raise Phase2ResultError(f"{label} must be a finite number")
    return float(value)


def _require_nonnegative(value: Any, label: str) -> float:
    number = _require_finite_number(value, label)
    if number < 0.0:
        raise Phase2ResultError(f"{label} must be nonnegative")
    return number


def _require_positive(value: Any, label: str) -> float:
    number = _require_finite_number(value, label)
    if number <= 0.0:
        raise Phase2ResultError(f"{label} must be positive")
    return number


def _require_fraction(value: Any, label: str) -> float:
    number = _require_finite_number(value, label)
    if not 0.0 <= number <= 1.0:
        raise Phase2ResultError(f"{label} must lie in [0, 1]")
    return number


def _require_nonnegative_integer(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise Phase2ResultError(f"{label} must be a nonnegative integer")
    return value


def _require_positive_integer(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise Phase2ResultError(f"{label} must be a positive integer")
    return value


def _require_boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise Phase2ResultError(f"{label} must be a boolean")
    return value


def _require_nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise Phase2ResultError(f"{label} must be a nonempty string")
    return value


def _require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise Phase2ResultError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _canonical_digest(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def optimizer_parameter_digest(parameters: Sequence[Mapping[str, Any]]) -> str:
    """Digest a sorted optimizer-group parameter inventory."""

    canonical = [
        {"name": item["name"], "numel": item["numel"]}
        for item in parameters
    ]
    return _canonical_digest(canonical)


def optimizer_coverage_digest(groups: Sequence[Mapping[str, Any]]) -> str:
    """Digest the exact parameter-to-group assignment and optimizer settings."""

    rows: list[dict[str, Any]] = []
    for group in groups:
        for parameter in group["parameters"]:
            rows.append(
                {
                    "name": parameter["name"],
                    "numel": parameter["numel"],
                    "group_name": group["group_name"],
                    "manifest_group_name": group["manifest_group_name"],
                    "lr": group["lr"],
                    "weight_decay": group["weight_decay"],
                }
            )
    rows.sort(key=lambda item: item["name"])
    return _canonical_digest(rows)


def _require_close(actual: Any, expected: float, label: str) -> None:
    number = _require_finite_number(actual, label)
    if not math.isclose(
        number,
        expected,
        rel_tol=_AGGREGATE_REL_TOL,
        abs_tol=_AGGREGATE_ABS_TOL,
    ):
        raise Phase2ResultError(
            f"{label} aggregate mismatch: reported={number}, derived={expected}"
        )


def expected_mqar_cells() -> set[str]:
    return {
        f"length_{length}/pairs_{pairs}"
        for length in _MQAR_LENGTHS
        for pairs in _MQAR_PAIRS
    }


def _relative_error(actual: float, reference: float) -> float:
    if reference <= 0.0:
        raise Phase2ResultError("Relative-error reference must be positive")
    return abs(actual - reference) / reference


def _validate_model_accounting(
    manifest: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> int:
    accounting = metrics.get("model_accounting")
    if not isinstance(accounting, Mapping):
        raise Phase2ResultError("model_accounting must be an object")

    counts = accounting.get("parameter_counts_actual")
    if not isinstance(counts, Mapping):
        raise Phase2ResultError(
            "model_accounting.parameter_counts_actual must be an object"
        )
    embedding = _require_nonnegative_integer(
        counts.get("embedding"),
        "Actual embedding parameter count",
    )
    core = _require_positive_integer(
        counts.get("non_embedding_core"),
        "Actual non-embedding core parameter count",
    )
    total = _require_positive_integer(
        counts.get("total"),
        "Actual total parameter count",
    )
    if total != embedding + core:
        raise Phase2ResultError(
            "Actual total parameter count must equal embedding + non-embedding core"
        )

    estimates = manifest.get("parameter_counts_estimated")
    if not isinstance(estimates, Mapping):
        raise Phase2ResultError(
            "Manifest lacks provisional parameter-count estimates"
        )
    estimated_embedding = _require_nonnegative_integer(
        estimates.get("embedding"),
        "Manifest estimated embedding parameter count",
    )
    estimated_core = _require_positive_integer(
        estimates.get("non_embedding_core"),
        "Manifest estimated non-embedding core parameter count",
    )
    estimated_total = _require_positive_integer(
        estimates.get("total"),
        "Manifest estimated total parameter count",
    )
    if embedding != estimated_embedding:
        raise Phase2ResultError(
            "Actual embedding count differs from the model-config estimate; "
            "tied embeddings may have been counted incorrectly"
        )

    estimator = accounting.get("parameter_estimator")
    if not isinstance(estimator, Mapping):
        raise Phase2ResultError(
            "model_accounting.parameter_estimator must be an object"
        )
    maximum_parameter_drift = _require_fraction(
        manifest["hard_failure_rules"][
            "parameter_estimator_relative_error_max"
        ],
        "Manifest parameter-estimator drift limit",
    )
    reported_parameter_drift_limit = _require_fraction(
        estimator.get("maximum_allowed_relative_error"),
        "Reported parameter-estimator drift limit",
    )
    _require_close(
        reported_parameter_drift_limit,
        maximum_parameter_drift,
        "Parameter-estimator drift limit",
    )
    core_drift = _relative_error(float(core), float(estimated_core))
    total_drift = _relative_error(float(total), float(estimated_total))
    _require_close(
        estimator.get("core_relative_error"),
        core_drift,
        "Core parameter-estimator relative error",
    )
    _require_close(
        estimator.get("total_relative_error"),
        total_drift,
        "Total parameter-estimator relative error",
    )
    expected_parameter_outcome = (
        "within_tolerance"
        if max(core_drift, total_drift) <= maximum_parameter_drift
        else "exceeded"
    )
    if estimator.get("outcome") != expected_parameter_outcome:
        raise Phase2ResultError(
            "Parameter-estimator outcome disagrees with actual/estimated counts"
        )
    if expected_parameter_outcome != "within_tolerance":
        raise Phase2ResultError(
            "Actual parameter counts exceed the manifest estimator-drift limit: "
            f"core={core_drift:.6f}, total={total_drift:.6f}, "
            f"limit={maximum_parameter_drift:.6f}"
        )

    scale = accounting.get("scale_band")
    if not isinstance(scale, Mapping):
        raise Phase2ResultError("model_accounting.scale_band must be an object")
    manifest_scale = manifest.get("scale_check")
    if not isinstance(manifest_scale, Mapping):
        raise Phase2ResultError("Manifest lacks scale_check")
    for name in ("label", "accounting"):
        if scale.get(name) != manifest_scale.get(name):
            raise Phase2ResultError(
                f"Reported scale-band {name} differs from the manifest"
            )
    if scale.get("accounting") != "non_embedding_trainable_parameters":
        raise Phase2ResultError(
            "This study only supports non-embedding trainable-parameter accounting"
        )
    accepted_min = _require_positive_integer(
        manifest_scale.get("accepted_min"),
        "Manifest scale accepted_min",
    )
    accepted_max = _require_positive_integer(
        manifest_scale.get("accepted_max"),
        "Manifest scale accepted_max",
    )
    if accepted_min > accepted_max:
        raise Phase2ResultError("Manifest scale band is inverted")
    for name, expected in (
        ("accepted_min", accepted_min),
        ("accepted_max", accepted_max),
        ("actual_value", core),
    ):
        if scale.get(name) != expected:
            raise Phase2ResultError(
                f"Reported scale-band {name}={scale.get(name)!r} "
                f"does not equal {expected!r}"
            )
    expected_scale_outcome = (
        "within_band" if accepted_min <= core <= accepted_max else "outside_band"
    )
    if scale.get("outcome") != expected_scale_outcome:
        raise Phase2ResultError(
            "Reported scale-band outcome disagrees with the actual core count"
        )
    if expected_scale_outcome != "within_band":
        raise Phase2ResultError(
            f"Actual core parameter count {core} is outside "
            f"[{accepted_min}, {accepted_max}]"
        )

    flops = accounting.get("flops")
    if not isinstance(flops, Mapping):
        raise Phase2ResultError("model_accounting.flops must be an object")
    if flops.get("scope") != _FLOP_SCOPE:
        raise Phase2ResultError(
            f"FLOP scope must equal {_FLOP_SCOPE!r}"
        )
    estimated_flops = _require_positive(
        flops.get("estimated"),
        "Estimated training FLOPs per optimizer step",
    )
    measured_flops = _require_positive(
        flops.get("measured"),
        "Measured training FLOPs per optimizer step",
    )
    _require_positive_integer(
        flops.get("measurement_steps"),
        "FLOP measurement_steps",
    )
    _require_nonempty_string(
        flops.get("estimator_method"),
        "FLOP estimator_method",
    )
    _require_nonempty_string(
        flops.get("measurement_method"),
        "FLOP measurement_method",
    )
    maximum_flop_drift = _require_fraction(
        manifest["hard_failure_rules"]["flop_estimator_relative_error_max"],
        "Manifest FLOP-estimator drift limit",
    )
    reported_flop_drift_limit = _require_fraction(
        flops.get("maximum_allowed_relative_error"),
        "Reported FLOP-estimator drift limit",
    )
    _require_close(
        reported_flop_drift_limit,
        maximum_flop_drift,
        "FLOP-estimator drift limit",
    )
    flop_drift = _relative_error(estimated_flops, measured_flops)
    _require_close(
        flops.get("relative_error"),
        flop_drift,
        "FLOP-estimator relative error",
    )
    expected_flop_outcome = (
        "within_tolerance"
        if flop_drift <= maximum_flop_drift
        else "exceeded"
    )
    if flops.get("outcome") != expected_flop_outcome:
        raise Phase2ResultError(
            "FLOP-estimator outcome disagrees with measured/estimated FLOPs"
        )
    if expected_flop_outcome != "within_tolerance":
        raise Phase2ResultError(
            "Architecture FLOP estimate exceeds measured-profile drift limit: "
            f"drift={flop_drift:.6f}, limit={maximum_flop_drift:.6f}"
        )
    return total


def _validate_optimizer_group_audit(
    manifest: Mapping[str, Any],
    metrics: Mapping[str, Any],
    *,
    actual_total_parameters: int,
) -> None:
    audit = metrics.get("optimizer_group_audit")
    if not isinstance(audit, Mapping):
        raise Phase2ResultError("optimizer_group_audit must be an object")
    groups = audit.get("groups")
    if not isinstance(groups, list) or not groups:
        raise Phase2ResultError(
            "optimizer_group_audit.groups must be a nonempty list"
        )
    group_names = [group.get("group_name") for group in groups if isinstance(group, Mapping)]
    if len(group_names) != len(groups):
        raise Phase2ResultError("Every optimizer group audit entry must be an object")
    if any(not isinstance(name, str) or not name.strip() for name in group_names):
        raise Phase2ResultError("Every optimizer group_name must be nonempty")
    if group_names != sorted(group_names):
        raise Phase2ResultError(
            "Optimizer groups must be sorted by group_name for deterministic auditing"
        )
    if len(set(group_names)) != len(group_names):
        raise Phase2ResultError("Optimizer group_name values must be unique")

    manifest_groups = manifest.get("optimizer", {}).get("groups")
    if not isinstance(manifest_groups, Mapping) or not manifest_groups:
        raise Phase2ResultError("Manifest has no optimizer group contract")
    protocol_weight_decay = _require_nonnegative(
        manifest["protocol"]["weight_decay"],
        "Manifest optimizer weight decay",
    )
    flattened: list[Mapping[str, Any]] = []
    all_names: list[str] = []
    realized_manifest_groups: set[str] = set()
    assigned_scalars = 0
    for group in groups:
        group_name = _require_nonempty_string(
            group.get("group_name"),
            "Optimizer group_name",
        )
        manifest_group_name = _require_nonempty_string(
            group.get("manifest_group_name"),
            f"Optimizer group {group_name} manifest_group_name",
        )
        if manifest_group_name not in manifest_groups:
            raise Phase2ResultError(
                f"Optimizer group {group_name} references unknown manifest group "
                f"{manifest_group_name!r}"
            )
        realized_manifest_groups.add(manifest_group_name)
        expected_group = manifest_groups[manifest_group_name]
        actual_lr = _require_positive(
            group.get("lr"),
            f"Optimizer group {group_name} lr",
        )
        _require_close(
            actual_lr,
            float(expected_group["lr"]),
            f"Optimizer group {group_name} lr",
        )
        weight_decay = _require_nonnegative(
            group.get("weight_decay"),
            f"Optimizer group {group_name} weight_decay",
        )
        explicit_decay = expected_group.get("weight_decay")
        if explicit_decay is not None:
            _require_close(
                weight_decay,
                float(explicit_decay),
                f"Optimizer group {group_name} weight_decay",
            )
        elif not (
            math.isclose(
                weight_decay,
                0.0,
                rel_tol=_AGGREGATE_REL_TOL,
                abs_tol=_AGGREGATE_ABS_TOL,
            )
            or math.isclose(
                weight_decay,
                protocol_weight_decay,
                rel_tol=_AGGREGATE_REL_TOL,
                abs_tol=_AGGREGATE_ABS_TOL,
            )
        ):
            raise Phase2ResultError(
                f"Optimizer group {group_name} weight_decay must be zero or "
                "the manifest's global weight decay"
            )

        parameters = group.get("parameters")
        if not isinstance(parameters, list) or not parameters:
            raise Phase2ResultError(
                f"Optimizer group {group_name} must list its parameters"
            )
        parameter_names: list[str] = []
        group_scalars = 0
        for parameter in parameters:
            if not isinstance(parameter, Mapping):
                raise Phase2ResultError(
                    f"Optimizer group {group_name} parameter entry must be an object"
                )
            name = _require_nonempty_string(
                parameter.get("name"),
                f"Optimizer group {group_name} parameter name",
            )
            numel = _require_positive_integer(
                parameter.get("numel"),
                f"Optimizer parameter {name} numel",
            )
            parameter_names.append(name)
            group_scalars += numel
            flattened.append(parameter)
            all_names.append(name)
        if parameter_names != sorted(parameter_names):
            raise Phase2ResultError(
                f"Optimizer group {group_name} parameters must be sorted by name"
            )
        if group.get("tensor_count") != len(parameters):
            raise Phase2ResultError(
                f"Optimizer group {group_name} tensor_count mismatch"
            )
        if group.get("scalar_parameter_count") != group_scalars:
            raise Phase2ResultError(
                f"Optimizer group {group_name} scalar_parameter_count mismatch"
            )
        reported_group_digest = _require_sha256(
            group.get("parameter_digest"),
            f"Optimizer group {group_name} parameter_digest",
        )
        expected_group_digest = optimizer_parameter_digest(parameters)
        if reported_group_digest != expected_group_digest:
            raise Phase2ResultError(
                f"Optimizer group {group_name} parameter_digest mismatch"
            )
        assigned_scalars += group_scalars

    required_manifest_groups = {
        name
        for name, definition in manifest_groups.items()
        if "conditional" not in definition
    }
    missing_manifest_groups = sorted(
        required_manifest_groups - realized_manifest_groups
    )
    if missing_manifest_groups:
        raise Phase2ResultError(
            "Optimizer audit omitted non-conditional manifest groups: "
            f"{missing_manifest_groups}"
        )

    duplicate_names = sorted(
        {name for name in all_names if all_names.count(name) > 1}
    )
    coverage = audit.get("coverage")
    if not isinstance(coverage, Mapping):
        raise Phase2ResultError(
            "optimizer_group_audit.coverage must be an object"
        )
    reported_unassigned = coverage.get("unassigned_parameter_names")
    reported_duplicates = coverage.get("duplicate_parameter_names")
    if not isinstance(reported_unassigned, list) or any(
        not isinstance(name, str) for name in reported_unassigned
    ):
        raise Phase2ResultError(
            "Optimizer unassigned_parameter_names must be a string list"
        )
    if not isinstance(reported_duplicates, list) or any(
        not isinstance(name, str) for name in reported_duplicates
    ):
        raise Phase2ResultError(
            "Optimizer duplicate_parameter_names must be a string list"
        )
    if reported_unassigned != sorted(set(reported_unassigned)):
        raise Phase2ResultError(
            "Optimizer unassigned_parameter_names must be sorted and unique"
        )
    if reported_duplicates != duplicate_names:
        raise Phase2ResultError(
            "Optimizer duplicate_parameter_names disagrees with group inventory"
        )
    if reported_unassigned:
        raise Phase2ResultError(
            "Optimizer audit has unassigned trainable parameters"
        )
    if duplicate_names:
        raise Phase2ResultError(
            "Optimizer audit assigns at least one parameter more than once"
        )

    assigned_tensors = len(flattened)
    for name, expected in (
        ("assigned_parameter_tensor_count", assigned_tensors),
        ("trainable_parameter_tensor_count", assigned_tensors),
        ("assigned_scalar_parameter_count", assigned_scalars),
        ("trainable_scalar_parameter_count", actual_total_parameters),
    ):
        if coverage.get(name) != expected:
            raise Phase2ResultError(
                f"Optimizer coverage {name}={coverage.get(name)!r} "
                f"does not equal {expected}"
            )
    if assigned_scalars != actual_total_parameters:
        raise Phase2ResultError(
            "Optimizer group scalar coverage does not equal actual trainable "
            "parameter count"
        )
    if not _require_boolean(
        coverage.get("complete"),
        "Optimizer coverage complete",
    ):
        raise Phase2ResultError("Optimizer coverage must be complete")
    if not _require_boolean(
        coverage.get("exclusive"),
        "Optimizer coverage exclusive",
    ):
        raise Phase2ResultError("Optimizer coverage must be exclusive")

    expected_parameter_digest = optimizer_parameter_digest(
        sorted(flattened, key=lambda item: item["name"])
    )
    if _require_sha256(
        audit.get("trainable_parameter_digest"),
        "Optimizer trainable_parameter_digest",
    ) != expected_parameter_digest:
        raise Phase2ResultError("Optimizer trainable_parameter_digest mismatch")
    if _require_sha256(
        audit.get("coverage_digest"),
        "Optimizer coverage_digest",
    ) != optimizer_coverage_digest(groups):
        raise Phase2ResultError("Optimizer coverage_digest mismatch")


def validate_step_metrics(
    manifest: Mapping[str, Any],
    metrics: Mapping[str, Any],
    *,
    expected_step: int,
    require_full_mqar_grid: bool = True,
    require_wikitext_ppl: bool = True,
    diagnostic_names: list[str] | None = None,
) -> None:
    """Fail before Optuna/W&B sees incomplete or nonfinite trial output."""

    if metrics.get("trial_hash") != manifest.get("trial_hash"):
        raise Phase2ResultError("Result trial_hash does not match its manifest")
    if metrics.get("schema_version") != 1:
        raise Phase2ResultError("Result schema_version must equal 1")
    if metrics.get("status") != "ok":
        raise Phase2ResultError("Only healthy results can be reported as scores")
    optimizer_step = metrics.get("optimizer_step")
    if (
        not isinstance(optimizer_step, int)
        or isinstance(optimizer_step, bool)
        or optimizer_step != expected_step
    ):
        raise Phase2ResultError(
            f"Expected optimizer step {expected_step}, got {metrics.get('optimizer_step')}"
        )
    tokens_seen = metrics.get("tokens_seen")
    if not isinstance(tokens_seen, int) or isinstance(tokens_seen, bool):
        raise Phase2ResultError("tokens_seen must be an integer")
    protocol = manifest["protocol"]
    expected_tokens = (
        expected_step
        * int(protocol["effective_batch_sequences"])
        * int(protocol["sequence_length"])
    )
    if tokens_seen != expected_tokens:
        raise Phase2ResultError(
            f"tokens_seen={tokens_seen} does not match fixed budget {expected_tokens}"
        )
    prune_step = int(manifest["fidelity"]["prune_step"])
    expected_global_sequence_index = (
        expected_step * int(protocol["effective_batch_sequences"])
    )
    if expected_step == prune_step:
        checkpoint = metrics.get("checkpoint_provenance")
        if not isinstance(checkpoint, Mapping):
            raise Phase2ResultError(
                "Screen metrics require checkpoint_provenance"
            )
        _require_sha256(
            checkpoint.get("checkpoint_sha256"),
            "checkpoint_provenance.checkpoint_sha256",
        )
        checkpoint_path = checkpoint.get("checkpoint_bundle_path")
        if (
            not isinstance(checkpoint_path, str)
            or not checkpoint_path.strip()
            or checkpoint_path.startswith("/")
            or ".." in checkpoint_path.split("/")
        ):
            raise Phase2ResultError(
                "checkpoint_provenance.checkpoint_bundle_path must be relative"
            )
        for field in (
            "checkpoint_bundle_byte_size",
            "checkpoint_bundle_file_count",
        ):
            value = checkpoint.get(field)
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value <= 0
            ):
                raise Phase2ResultError(
                    f"checkpoint_provenance.{field} must be a positive integer"
                )
        if checkpoint.get("checkpoint_digest_algorithm") != (
            "sha256(path_nul_size_nul_content_sha256_newline_v1)"
        ):
            raise Phase2ResultError(
                "Screen checkpoint_digest_algorithm is not canonical"
            )
        if checkpoint.get("optimizer_step") != prune_step:
            raise Phase2ResultError(
                "Screen checkpoint optimizer_step must equal the prune step"
            )
        if checkpoint.get("global_sequence_index") != (
            expected_global_sequence_index
        ):
            raise Phase2ResultError(
                "Screen checkpoint global_sequence_index does not match the "
                "deterministic data stream"
            )
        checkpoint_revision = checkpoint.get("checkpoint_format_revision")
        if (
            not isinstance(checkpoint_revision, str)
            or not checkpoint_revision.strip()
        ):
            raise Phase2ResultError(
                "Screen checkpoint_format_revision must be nonempty"
            )
    elif expected_step > prune_step:
        resume = metrics.get("resume_provenance")
        if not isinstance(resume, Mapping):
            raise Phase2ResultError(
                "Final metrics require resume_provenance"
            )
        _require_sha256(
            resume.get("source_checkpoint_sha256"),
            "resume_provenance.source_checkpoint_sha256",
        )
        source_path = resume.get("source_checkpoint_bundle_path")
        if (
            not isinstance(source_path, str)
            or not source_path.strip()
            or source_path.startswith("/")
            or ".." in source_path.split("/")
        ):
            raise Phase2ResultError(
                "resume_provenance.source_checkpoint_bundle_path must be relative"
            )
        for field in (
            "source_checkpoint_bundle_byte_size",
            "source_checkpoint_bundle_file_count",
        ):
            value = resume.get(field)
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value <= 0
            ):
                raise Phase2ResultError(
                    f"resume_provenance.{field} must be a positive integer"
                )
        if resume.get("source_checkpoint_digest_algorithm") != (
            "sha256(path_nul_size_nul_content_sha256_newline_v1)"
        ):
            raise Phase2ResultError(
                "Final source_checkpoint_digest_algorithm is not canonical"
            )
        if resume.get("source_optimizer_step") != prune_step:
            raise Phase2ResultError(
                "Final resume source_optimizer_step must equal the prune step"
            )
        expected_source_index = (
            prune_step * int(protocol["effective_batch_sequences"])
        )
        if resume.get("restored_global_sequence_index") != expected_source_index:
            raise Phase2ResultError(
                "Final restored_global_sequence_index does not match the "
                "screen checkpoint"
            )
        checkpoint_revision = resume.get("checkpoint_format_revision")
        if (
            not isinstance(checkpoint_revision, str)
            or not checkpoint_revision.strip()
        ):
            raise Phase2ResultError(
                "Final checkpoint_format_revision must be nonempty"
            )
        for field in (
            "optimizer_state_restored",
            "scheduler_state_restored",
            "rng_state_restored",
            "data_stream_state_restored",
        ):
            if resume.get(field) is not True:
                raise Phase2ResultError(
                    f"resume_provenance.{field} must be true"
                )

    gpu_seconds_actual = _require_positive(
        metrics.get("gpu_seconds_actual"),
        "gpu_seconds_actual",
    )
    gpu_hours_actual = _require_positive(
        metrics.get("gpu_hours_actual"),
        "gpu_hours_actual",
    )
    _require_close(
        gpu_hours_actual,
        gpu_seconds_actual / 3600.0,
        "gpu_hours_actual",
    )
    wandb_run_id = metrics.get("wandb_run_id")
    if not isinstance(wandb_run_id, str) or not wandb_run_id.strip():
        raise Phase2ResultError(
            "wandb_run_id must be a nonempty stable run identifier"
        )

    actual_total_parameters = _validate_model_accounting(manifest, metrics)
    _validate_optimizer_group_audit(
        manifest,
        metrics,
        actual_total_parameters=actual_total_parameters,
    )

    mqar = _require_fraction(metrics.get("mqar_accuracy"), "mqar_accuracy")
    ppl = metrics.get("wikitext_ppl")
    if require_wikitext_ppl:
        _require_positive(ppl, "wikitext_ppl")
    elif ppl is not None:
        _require_positive(ppl, "Optional wikitext_ppl")

    diagnostics = metrics.get("diagnostics")
    if not isinstance(diagnostics, Mapping):
        raise Phase2ResultError("diagnostics must be an object")
    raw_not_applicable = metrics.get("not_applicable_diagnostics", [])
    if (
        not isinstance(raw_not_applicable, list)
        or any(not isinstance(name, str) for name in raw_not_applicable)
        or len(set(raw_not_applicable)) != len(raw_not_applicable)
    ):
        raise Phase2ResultError(
            "not_applicable_diagnostics must be a list of unique strings"
        )
    not_applicable = set(raw_not_applicable)
    required_diagnostics = (
        list(diagnostic_names)
        if diagnostic_names is not None
        else list(manifest["required_diagnostics"])
    )
    unknown_required = set(required_diagnostics) - set(manifest["required_diagnostics"])
    if unknown_required:
        raise Phase2ResultError(
            f"Unknown required diagnostics: {sorted(unknown_required)}"
        )
    for name in required_diagnostics:
        if name not in diagnostics:
            raise Phase2ResultError(f"Missing required diagnostic: {name}")
        value = diagnostics[name]
        if name in not_applicable:
            if value is not None:
                raise Phase2ResultError(
                    f"Not-applicable diagnostic {name} must be null"
                )
        else:
            _require_finite_number(value, f"Diagnostic {name}")

    mode = manifest["parameters"]["architecture_mode"]
    allowed_na: set[str] = set()
    if mode in {"paper_control", "mamba_only", "transformer"}:
        allowed_na.update(
            {
                "beta_mean",
                "beta_saturation_fraction",
                "layerscale_magnitude_mean",
            }
        )
    if mode in {"mamba_only", "transformer"}:
        allowed_na.update(
            {
                "ska_zero_ppl_delta",
                "spectral_radius_raw_preclamp",
                "spectral_radius_normalized_pre_gamma",
                "spectral_radius_applied_post_gamma",
                "clamp_factor_mean",
                "clamp_fraction",
                "ridge_condition_number",
                "ridge_escalation_count",
                "ridge_requested",
                "effective_ridge_min",
                "effective_ridge_max",
                "layerscale_magnitude_mean",
                "ska_branch_norm_ratio",
                "spectral_gap_mean",
                "spectral_gap_max",
                "lambda_min_gram",
                "ska_gradient_to_mamba_ratio",
                "effective_jacobian_rank",
            }
        )
    illegal_na = not_applicable - allowed_na
    if illegal_na:
        raise Phase2ResultError(
            f"Diagnostics incorrectly declared not applicable: {sorted(illegal_na)}"
        )
    for name in not_applicable:
        if name not in diagnostics or diagnostics[name] is not None:
            raise Phase2ResultError(
                f"Not-applicable diagnostic {name} must be present and null"
            )
    missing_mode_na = (
        allowed_na & set(required_diagnostics)
    ) - not_applicable
    if missing_mode_na:
        raise Phase2ResultError(
            "Mode-required N/A diagnostics were not declared: "
            f"{sorted(missing_mode_na)}"
        )

    for name in (
        "spectral_radius_raw_preclamp",
        "spectral_radius_normalized_pre_gamma",
        "spectral_radius_applied_post_gamma",
        "spectral_gap_mean",
        "spectral_gap_max",
        "lambda_min_gram",
        "ska_branch_norm_ratio",
        "ska_gradient_to_mamba_ratio",
        "within_chunk_parallel_decode_drift_max_abs_error",
        "state_rebuild_decode_prefill_max_abs_error",
        "chunk_boundary_parallel_decode_max_abs_error",
    ):
        value = diagnostics.get(name)
        if value is not None:
            _require_nonnegative(value, f"Diagnostic {name}")

    clamp_factor = diagnostics.get("clamp_factor_mean")
    if clamp_factor is not None:
        clamp_factor = _require_positive(
            clamp_factor, "Diagnostic clamp_factor_mean"
        )
        if clamp_factor > 1.0:
            raise Phase2ResultError(
                "Diagnostic clamp_factor_mean must lie in (0, 1]"
            )

    for name in ("clamp_fraction", "beta_mean", "beta_saturation_fraction"):
        value = diagnostics.get(name)
        if value is not None:
            _require_fraction(value, f"Diagnostic {name}")

    layerscale = diagnostics.get("layerscale_magnitude_mean")
    if layerscale is not None:
        _require_nonnegative(
            layerscale, "Diagnostic layerscale_magnitude_mean"
        )

    condition_number = diagnostics.get("ridge_condition_number")
    if condition_number is not None:
        condition_number = _require_finite_number(
            condition_number, "Diagnostic ridge_condition_number"
        )
        if condition_number < 1.0:
            raise Phase2ResultError(
                "Diagnostic ridge_condition_number must be at least 1"
            )

    escalation_count = diagnostics.get("ridge_escalation_count")
    if escalation_count is not None:
        _require_nonnegative_integer(
            escalation_count, "Diagnostic ridge_escalation_count"
        )

    for name in ("ridge_requested", "effective_ridge_min", "effective_ridge_max"):
        value = diagnostics.get(name)
        if value is not None:
            _require_positive(value, f"Diagnostic {name}")
    requested_ridge = diagnostics.get("ridge_requested")
    effective_ridge_min = diagnostics.get("effective_ridge_min")
    effective_ridge_max = diagnostics.get("effective_ridge_max")
    if requested_ridge is not None:
        configured_ridge = float(manifest["model_config"]["ska_ridge"])
        _require_close(
            requested_ridge,
            configured_ridge,
            "Diagnostic ridge_requested",
        )
        if (
            effective_ridge_min is None
            or effective_ridge_max is None
            or escalation_count is None
        ):
            raise Phase2ResultError(
                "Ridge diagnostics must report requested/min/max/escalation together"
            )
        ridge_min = float(effective_ridge_min)
        ridge_max = float(effective_ridge_max)
        if ridge_min > ridge_max:
            raise Phase2ResultError(
                "Diagnostic effective_ridge_min cannot exceed effective_ridge_max"
            )
        if ridge_min + _AGGREGATE_ABS_TOL < configured_ridge:
            raise Phase2ResultError(
                "Effective ridge cannot be lower than the requested ridge"
            )
        escalations = int(escalation_count)
        changed = not math.isclose(
            ridge_max,
            configured_ridge,
            rel_tol=_AGGREGATE_REL_TOL,
            abs_tol=_AGGREGATE_ABS_TOL,
        )
        if changed != (escalations > 0):
            raise Phase2ResultError(
                "Effective ridge may exceed the request only when explicitly "
                "counted as a Cholesky escalation"
            )

    d_model = int(manifest["model_config"]["d_model"])
    effective_rank = diagnostics.get("effective_jacobian_rank")
    if effective_rank is not None:
        effective_rank = _require_nonnegative(
            effective_rank, "Diagnostic effective_jacobian_rank"
        )
        if effective_rank > d_model:
            raise Phase2ResultError(
                "Diagnostic effective_jacobian_rank cannot exceed d_model"
            )

    step_time = diagnostics.get("step_time_ms")
    if step_time is not None:
        _require_positive(step_time, "Diagnostic step_time_ms")
    peak_memory = diagnostics.get("peak_memory_bytes")
    if peak_memory is not None:
        _require_positive_integer(peak_memory, "Diagnostic peak_memory_bytes")

    for name in ("ppl_full", "ppl_ska_zero", "ppl_mamba_zero", "ppl_both_zero"):
        value = diagnostics.get(name)
        if value is not None:
            _require_positive(value, f"Diagnostic {name}")

    gap_mean = diagnostics.get("spectral_gap_mean")
    gap_max = diagnostics.get("spectral_gap_max")
    if gap_mean is not None and gap_max is not None:
        if float(gap_mean) > float(gap_max):
            raise Phase2ResultError(
                "Diagnostic spectral_gap_mean cannot exceed spectral_gap_max"
            )

    post_radius = diagnostics.get("spectral_radius_normalized_pre_gamma")
    if post_radius is not None:
        maximum = float(
            manifest["hard_failure_rules"][
                "normalized_pre_gamma_spectral_radius_max"
            ]
        )
        if float(post_radius) > maximum:
            raise Phase2ResultError(
                "normalized pre-gamma spectral radius "
                f"{post_radius} exceeds {maximum}"
            )

    by_layer = metrics.get("diagnostics_by_layer")
    if not isinstance(by_layer, Mapping):
        raise Phase2ResultError("diagnostics_by_layer must be an object")
    expected_layers = {
        str(index)
        for index in manifest["model_extensions"]["realized_layout"][
            "ska_layer_indices_zero_based"
        ]
    }
    if set(by_layer) != expected_layers:
        raise Phase2ResultError(
            "diagnostics_by_layer keys must exactly match realized SKA indices: "
            f"expected={sorted(expected_layers)}, got={sorted(by_layer)}"
        )
    structure = manifest["diagnostic_structure"]
    head_metrics = list(structure["per_layer_head_metrics"])
    layer_metrics = list(structure["per_layer_metrics"])
    missing_head_metrics = sorted(_REQUIRED_HEAD_METRICS - set(head_metrics))
    missing_layer_metrics = sorted(_REQUIRED_LAYER_METRICS - set(layer_metrics))
    if missing_head_metrics or missing_layer_metrics:
        raise Phase2ResultError(
            "Manifest diagnostic_structure is incomplete: "
            f"missing_head={missing_head_metrics}, "
            f"missing_layer={missing_layer_metrics}"
        )
    n_heads = int(manifest["model_config"]["ska_n_heads"])
    maximum_radius = float(
        manifest["hard_failure_rules"]["normalized_pre_gamma_spectral_radius_max"]
    )
    per_head_records: list[Mapping[str, Any]] = []
    per_layer_records: list[Mapping[str, Any]] = []
    for layer, layer_values in by_layer.items():
        if not isinstance(layer_values, Mapping):
            raise Phase2ResultError(f"Layer {layer} diagnostics must be an object")
        per_head = layer_values.get("per_head")
        if not isinstance(per_head, list) or len(per_head) != n_heads:
            raise Phase2ResultError(
                f"Layer {layer} must contain {n_heads} per-head records"
            )
        for head_index, head_values in enumerate(per_head):
            if not isinstance(head_values, Mapping):
                raise Phase2ResultError(
                    f"Layer {layer} head {head_index} must be an object"
                )
            for name in head_metrics:
                value = head_values.get(name)
                _require_finite_number(
                    value,
                    f"Layer {layer} head {head_index} metric {name}",
                )

            head_label = f"Layer {layer} head {head_index}"
            raw_mean = _require_nonnegative(
                head_values["spectral_radius_raw_preclamp_mean"],
                f"{head_label} spectral_radius_raw_preclamp_mean",
            )
            raw_max = _require_nonnegative(
                head_values["spectral_radius_raw_preclamp_max"],
                f"{head_label} spectral_radius_raw_preclamp_max",
            )
            normalized_mean = _require_nonnegative(
                head_values["spectral_radius_normalized_pre_gamma_mean"],
                f"{head_label} spectral_radius_normalized_pre_gamma_mean",
            )
            normalized_max = _require_nonnegative(
                head_values["spectral_radius_normalized_pre_gamma_max"],
                f"{head_label} spectral_radius_normalized_pre_gamma_max",
            )
            applied_mean = _require_nonnegative(
                head_values["spectral_radius_applied_post_gamma_mean"],
                f"{head_label} spectral_radius_applied_post_gamma_mean",
            )
            applied_max = _require_nonnegative(
                head_values["spectral_radius_applied_post_gamma_max"],
                f"{head_label} spectral_radius_applied_post_gamma_max",
            )
            head_clamp_factor = _require_positive(
                head_values["clamp_factor_mean"],
                f"{head_label} clamp_factor_mean",
            )
            if head_clamp_factor > 1.0:
                raise Phase2ResultError(
                    f"{head_label} clamp_factor_mean must lie in (0, 1]"
                )
            _require_fraction(
                head_values["clamp_fraction"],
                f"{head_label} clamp_fraction",
            )
            head_gap_mean = _require_nonnegative(
                head_values["spectral_gap_mean"],
                f"{head_label} spectral_gap_mean",
            )
            head_gap_max = _require_nonnegative(
                head_values["spectral_gap_max"],
                f"{head_label} spectral_gap_max",
            )
            lambda_mean = _require_nonnegative(
                head_values["lambda_min_gram_mean"],
                f"{head_label} lambda_min_gram_mean",
            )
            lambda_min = _require_nonnegative(
                head_values["lambda_min_gram_min"],
                f"{head_label} lambda_min_gram_min",
            )
            condition_mean = _require_finite_number(
                head_values["ridge_condition_number_mean"],
                f"{head_label} ridge_condition_number_mean",
            )
            condition_max = _require_finite_number(
                head_values["ridge_condition_number_max"],
                f"{head_label} ridge_condition_number_max",
            )
            if condition_mean < 1.0 or condition_max < 1.0:
                raise Phase2ResultError(
                    f"{head_label} ridge condition numbers must be at least 1"
                )
            if condition_mean > condition_max:
                raise Phase2ResultError(
                    f"{head_label} ridge_condition_number_mean "
                    "cannot exceed its maximum"
                )
            _require_nonnegative_integer(
                head_values["ridge_escalation_count"],
                f"{head_label} ridge_escalation_count",
            )
            head_effective_ridge_min = _require_positive(
                head_values["effective_ridge_min"],
                f"{head_label} effective_ridge_min",
            )
            head_effective_ridge_max = _require_positive(
                head_values["effective_ridge_max"],
                f"{head_label} effective_ridge_max",
            )
            if head_effective_ridge_min > head_effective_ridge_max:
                raise Phase2ResultError(
                    f"{head_label} effective_ridge_min cannot exceed its maximum"
                )
            for mean, maximum_value, metric_name in (
                (raw_mean, raw_max, "spectral_radius_raw_preclamp"),
                (
                    normalized_mean,
                    normalized_max,
                    "spectral_radius_normalized_pre_gamma",
                ),
                (
                    applied_mean,
                    applied_max,
                    "spectral_radius_applied_post_gamma",
                ),
                (head_gap_mean, head_gap_max, "spectral_gap"),
            ):
                if mean > maximum_value:
                    raise Phase2ResultError(
                        f"{head_label} {metric_name}_mean cannot exceed its maximum"
                    )
            if lambda_min > lambda_mean:
                raise Phase2ResultError(
                    f"{head_label} lambda_min_gram_min cannot exceed its mean"
                )
            if (
                normalized_max
                > maximum_radius
            ):
                raise Phase2ResultError(
                    f"Layer {layer} head {head_index} exceeds spectral-radius gate"
                )
            ridge = float(manifest["model_config"]["ska_ridge"])
            if head_effective_ridge_min + _AGGREGATE_ABS_TOL < ridge:
                raise Phase2ResultError(
                    f"{head_label} effective ridge is below the requested ridge"
                )
            head_escalations = int(head_values["ridge_escalation_count"])
            head_ridge_changed = not math.isclose(
                head_effective_ridge_max,
                ridge,
                rel_tol=_AGGREGATE_REL_TOL,
                abs_tol=_AGGREGATE_ABS_TOL,
            )
            if head_ridge_changed != (head_escalations > 0):
                raise Phase2ResultError(
                    f"{head_label} effective ridge change must be explicitly "
                    "counted as a Cholesky escalation"
                )
            minimum_ratio = float(
                manifest["hard_failure_rules"]["lambda_min_over_ridge_min"]
            )
            if lambda_min < head_effective_ridge_min * minimum_ratio:
                raise Phase2ResultError(
                    f"Layer {layer} head {head_index} violates lambda-min/ridge gate"
                )
            per_head_records.append(head_values)
        per_layer = layer_values.get("layer")
        if not isinstance(per_layer, Mapping):
            raise Phase2ResultError(f"Layer {layer} needs layer-level diagnostics")
        candidate_only_layer_metrics = {
            "beta_mean",
            "beta_saturation_fraction",
            "layerscale_magnitude_mean",
        }
        for name in layer_metrics:
            value = per_layer.get(name)
            if mode == "paper_control" and name in candidate_only_layer_metrics:
                if value is not None:
                    raise Phase2ResultError(
                        f"Paper-control layer {layer} metric {name} must be null"
                    )
            else:
                _require_finite_number(value, f"Layer {layer} metric {name}")
        for name in ("residual_norm_ratio", "gradient_norm_ratio"):
            _require_nonnegative(
                per_layer[name], f"Layer {layer} metric {name}"
            )
        layer_rank = _require_nonnegative(
            per_layer["effective_jacobian_rank"],
            f"Layer {layer} metric effective_jacobian_rank",
        )
        if layer_rank > d_model:
            raise Phase2ResultError(
                f"Layer {layer} effective_jacobian_rank cannot exceed d_model"
            )
        if mode != "paper_control":
            _require_fraction(
                per_layer["beta_mean"], f"Layer {layer} metric beta_mean"
            )
            _require_fraction(
                per_layer["beta_saturation_fraction"],
                f"Layer {layer} metric beta_saturation_fraction",
            )
            _require_nonnegative(
                per_layer["layerscale_magnitude_mean"],
                f"Layer {layer} metric layerscale_magnitude_mean",
            )
        per_layer_records.append(per_layer)

    if per_head_records:
        # These reductions are part of the wire contract. Each head contributes
        # equally because every head record summarizes the same sample set.
        aggregate_reductions = {
            "spectral_radius_raw_preclamp": max(
                float(record["spectral_radius_raw_preclamp_max"])
                for record in per_head_records
            ),
            "spectral_radius_normalized_pre_gamma": max(
                float(record["spectral_radius_normalized_pre_gamma_max"])
                for record in per_head_records
            ),
            "spectral_radius_applied_post_gamma": max(
                float(record["spectral_radius_applied_post_gamma_max"])
                for record in per_head_records
            ),
            "clamp_factor_mean": sum(
                float(record["clamp_factor_mean"]) for record in per_head_records
            )
            / len(per_head_records),
            "clamp_fraction": sum(
                float(record["clamp_fraction"]) for record in per_head_records
            )
            / len(per_head_records),
            "spectral_gap_mean": sum(
                float(record["spectral_gap_mean"]) for record in per_head_records
            )
            / len(per_head_records),
            "spectral_gap_max": max(
                float(record["spectral_gap_max"]) for record in per_head_records
            ),
            "lambda_min_gram": min(
                float(record["lambda_min_gram_min"])
                for record in per_head_records
            ),
            "ridge_condition_number": max(
                float(record["ridge_condition_number_max"])
                for record in per_head_records
            ),
            "ridge_escalation_count": sum(
                int(record["ridge_escalation_count"])
                for record in per_head_records
            ),
            "effective_ridge_min": min(
                float(record["effective_ridge_min"])
                for record in per_head_records
            ),
            "effective_ridge_max": max(
                float(record["effective_ridge_max"])
                for record in per_head_records
            ),
        }
        for name, derived in aggregate_reductions.items():
            reported = diagnostics.get(name)
            if reported is not None:
                _require_close(
                    reported,
                    derived,
                    f"Diagnostic {name}",
                )

    if per_layer_records:
        layer_aggregate_reductions = {
            "ska_branch_norm_ratio": sum(
                float(record["residual_norm_ratio"])
                for record in per_layer_records
            )
            / len(per_layer_records),
            "ska_gradient_to_mamba_ratio": sum(
                float(record["gradient_norm_ratio"])
                for record in per_layer_records
            )
            / len(per_layer_records),
            "effective_jacobian_rank": sum(
                float(record["effective_jacobian_rank"])
                for record in per_layer_records
            )
            / len(per_layer_records),
        }
        if mode != "paper_control":
            layer_aggregate_reductions.update(
                {
                    "beta_mean": sum(
                        float(record["beta_mean"])
                        for record in per_layer_records
                    )
                    / len(per_layer_records),
                    "beta_saturation_fraction": sum(
                        float(record["beta_saturation_fraction"])
                        for record in per_layer_records
                    )
                    / len(per_layer_records),
                    "layerscale_magnitude_mean": sum(
                        float(record["layerscale_magnitude_mean"])
                        for record in per_layer_records
                    )
                    / len(per_layer_records),
                }
            )
        for name, derived in layer_aggregate_reductions.items():
            reported = diagnostics.get(name)
            if reported is not None:
                _require_close(
                    reported,
                    derived,
                    f"Diagnostic {name}",
                )

    ablation_names = {
        "ppl_full",
        "ppl_ska_zero",
        "ppl_mamba_zero",
        "ppl_both_zero",
        "ska_zero_ppl_delta",
    }
    if ablation_names <= set(required_diagnostics):
        derived_delta = float(diagnostics["ppl_ska_zero"]) - float(
            diagnostics["ppl_full"]
        )
        reported_delta = diagnostics["ska_zero_ppl_delta"]
        if reported_delta is not None and not math.isclose(
            float(reported_delta),
            derived_delta,
            rel_tol=_AGGREGATE_REL_TOL,
            abs_tol=_AGGREGATE_ABS_TOL,
        ):
            raise Phase2ResultError(
                "ska_zero_ppl_delta must equal ppl_ska_zero - ppl_full"
            )
        if ppl is not None and not math.isclose(
            float(diagnostics["ppl_full"]),
            float(ppl),
            rel_tol=_AGGREGATE_REL_TOL,
            abs_tol=_AGGREGATE_ABS_TOL,
        ):
            raise Phase2ResultError(
                "ppl_full must equal the primary WikiText perplexity"
            )

    parity_gates = (
        (
            "state_rebuild_decode_prefill_max_abs_error",
            "state_rebuild_decode_prefill_max_abs_error",
        ),
        (
            "chunk_boundary_parallel_decode_max_abs_error",
            "chunk_boundary_parallel_decode_max_abs_error",
        ),
    )
    for diagnostic_name, rule_name in parity_gates:
        if diagnostic_name in required_diagnostics:
            parity = float(diagnostics[diagnostic_name])
            maximum_parity = float(manifest["hard_failure_rules"][rule_name])
            if parity > maximum_parity:
                raise Phase2ResultError(
                    f"{diagnostic_name} {parity} exceeds {maximum_parity}"
                )

    if require_full_mqar_grid:
        grid = metrics.get("mqar_grid")
        if not isinstance(grid, Mapping):
            raise Phase2ResultError("mqar_grid must contain all 20 fixed cells")
        missing = sorted(expected_mqar_cells() - set(grid))
        extra = sorted(set(grid) - expected_mqar_cells())
        if missing or extra:
            raise Phase2ResultError(
                f"MQAR grid mismatch: missing={missing}, extra={extra}"
            )
        for cell, value in grid.items():
            try:
                _require_fraction(value, f"MQAR cell {cell}")
            except Phase2ResultError as exc:
                raise Phase2ResultError(
                    f"Invalid MQAR cell {cell}: {value!r}"
                ) from exc
        macro = sum(float(value) for value in grid.values()) / len(grid)
        if not math.isclose(
            mqar,
            macro,
            rel_tol=_AGGREGATE_REL_TOL,
            abs_tol=_AGGREGATE_ABS_TOL,
        ):
            raise Phase2ResultError(
                "mqar_accuracy must equal the macro-average of all 20 cells: "
                f"reported={metrics['mqar_accuracy']}, derived={macro}"
            )
        hard_cells = {
            cell: float(grid[cell])
            for cell in (
                "length_1024/pairs_32",
                "length_1024/pairs_64",
                "length_2048/pairs_32",
                "length_2048/pairs_64",
            )
        }
        hard_accuracy = sum(hard_cells.values()) / len(hard_cells)
        _require_close(
            metrics.get("mqar_hard_accuracy"),
            hard_accuracy,
            "mqar_hard_accuracy",
        )
        worst_cell = min(hard_cells, key=lambda cell: (hard_cells[cell], cell))
        _require_close(
            metrics.get("mqar_worst_cell_accuracy"),
            hard_cells[worst_cell],
            "mqar_worst_cell_accuracy",
        )
        if metrics.get("mqar_worst_cell") != worst_cell:
            raise Phase2ResultError(
                "mqar_worst_cell must identify the deterministic minimum hard "
                f"cell: reported={metrics.get('mqar_worst_cell')!r}, "
                f"derived={worst_cell!r}"
            )
        mqar_identity = manifest.get("data", {}).get("mqar_screening", {})
        expected_metadata = {
            "mqar_seed": mqar_identity.get("seed"),
            "mqar_generator_revision": mqar_identity.get(
                "generator_revision"
            ),
            "mqar_samples_per_cell": mqar_identity.get("samples_per_cell"),
        }
        for name, expected in expected_metadata.items():
            if metrics.get(name) != expected:
                raise Phase2ResultError(
                    f"{name}={metrics.get(name)!r} does not match frozen "
                    f"manifest value {expected!r}"
                )


__all__ = [
    "Phase2ResultError",
    "expected_mqar_cells",
    "validate_cross_stage_identity",
    "validate_step_metrics",
]
