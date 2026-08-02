"""Load, validate, and sample the immutable Phase 2a search contract.

The module is deliberately standard-library-only. Search-space validation and
dry runs must work before PyTorch, CUDA, or Optuna are installed.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any, Mapping


class Phase2SpecError(ValueError):
    """Raised when the Phase 2 contract or a materialized trial is invalid."""


_MANDATED_VALUES = {
    "ska_rank": [32, 48, 64, 96, 128],
    "ska_fraction": [0.15, 0.2, 0.25, 0.33],
    "ska_placement": ["uniform", "late_biased", "middle_clustered"],
    "ska_chunk_size": [32, 64, 96, 128],
    "ska_ridge": [0.0001, 0.001, 0.01],
    "beta_init_probability": [0.1, 0.3, 0.5],
    "layerscale_init": [0.00001, 0.0001, 0.001],
    "short_conv_kernel": [4, 8, 16],
    "short_conv_gate_init": [0.001, 0.01, 0.1],
    "qk_norm": [False, True],
    "gamma_eta_lr_multiplier": [5, 10, 25, 50],
    "objective_arm": ["next_token_only", "birdie_mix"],
}

_MANDATED_DIAGNOSTICS = {
    "spectral_radius_raw_preclamp",
    "spectral_radius_normalized_pre_gamma",
    "spectral_radius_applied_post_gamma",
    "clamp_factor_mean",
    "clamp_fraction",
    "spectral_gap_mean",
    "spectral_gap_max",
    "lambda_min_gram",
    "ridge_condition_number",
    "ridge_escalation_count",
    "ridge_requested",
    "effective_ridge_min",
    "effective_ridge_max",
    "beta_mean",
    "beta_saturation_fraction",
    "layerscale_magnitude_mean",
    "ska_branch_norm_ratio",
    "ska_gradient_to_mamba_ratio",
    "effective_jacobian_rank",
    "step_time_ms",
    "peak_memory_bytes",
    "ppl_full",
    "ppl_ska_zero",
    "ppl_mamba_zero",
    "ppl_both_zero",
    "ska_zero_ppl_delta",
    "state_rebuild_decode_prefill_max_abs_error",
    "chunk_boundary_parallel_decode_max_abs_error",
    "within_chunk_parallel_decode_drift_max_abs_error",
}
_MANDATED_HEAD_DIAGNOSTICS = {
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
_MANDATED_LAYER_DIAGNOSTICS = {
    "residual_norm_ratio",
    "gradient_norm_ratio",
    "effective_jacobian_rank",
    "beta_mean",
    "beta_saturation_fraction",
    "layerscale_magnitude_mean",
}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def stable_hash(value: Any) -> str:
    """Return a deterministic SHA-256 digest for a JSON-compatible value."""

    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise Phase2SpecError(f"Could not find repo root above {start}")


def load_spec(path: str | Path) -> dict[str, Any]:
    """Load and validate a Phase 2 JSON spec.

    Relative `base_model_config` paths are resolved against the repository root,
    not the caller's current directory.
    """

    spec_path = Path(path).expanduser().resolve()
    try:
        spec = json.loads(spec_path.read_text())
    except FileNotFoundError as exc:
        raise Phase2SpecError(f"Search spec does not exist: {spec_path}") from exc
    except json.JSONDecodeError as exc:
        raise Phase2SpecError(f"Invalid JSON in {spec_path}: {exc}") from exc

    if not isinstance(spec, dict):
        raise Phase2SpecError("Search spec must be a JSON object")
    repo_root = _find_repo_root(spec_path.parent)
    spec["_spec_path"] = str(spec_path)
    spec["_repo_root"] = str(repo_root)
    base_path = Path(spec.get("base_model_config", ""))
    if not base_path.is_absolute():
        base_path = repo_root / base_path
    spec["_base_model_path"] = str(base_path.resolve())
    validate_spec(spec)
    return spec


def _axis_values(spec: Mapping[str, Any], name: str) -> list[Any]:
    axis = spec.get("axes", {}).get(name)
    if not isinstance(axis, Mapping):
        raise Phase2SpecError(f"Missing axis: {name}")
    values = axis.get("values")
    if axis.get("kind") != "categorical" or not isinstance(values, list):
        raise Phase2SpecError(f"Axis {name} must be categorical with a values list")
    return values


def validate_spec(spec: Mapping[str, Any]) -> None:
    """Fail on scaling-plan drift or an internally inconsistent search spec."""

    required_sections = {
        "schema_version",
        "study_name",
        "architecture_base",
        "base_model_config",
        "scale",
        "protocol",
        "fidelity",
        "study",
        "architecture_modes",
        "axes",
        "fixed_controls",
        "required_diagnostics",
        "screen_required_diagnostics",
        "diagnostic_structure",
        "required_capabilities",
        "required_approvals",
        "required_evidence_checks",
        "hard_failure_rules",
        "storage",
        "outputs",
    }
    missing = sorted(required_sections - set(spec))
    if missing:
        raise Phase2SpecError(f"Missing spec sections: {missing}")
    if spec["schema_version"] != 1:
        raise Phase2SpecError(f"Unsupported schema_version={spec['schema_version']!r}")
    if spec.get("status") not in {
        "premerge_infrastructure_only",
        "scientific_ready",
    }:
        raise Phase2SpecError(f"Unknown Phase 2 spec status={spec.get('status')!r}")

    architecture_base = spec["architecture_base"]
    if not isinstance(architecture_base, Mapping):
        raise Phase2SpecError("architecture_base must be an object")
    base_branch = architecture_base.get("branch")
    base_commit = architecture_base.get("commit")
    if not isinstance(base_branch, str) or not base_branch.strip():
        raise Phase2SpecError("architecture_base.branch must be nonempty")
    if (
        not isinstance(base_commit, str)
        or len(base_commit) != 40
        or any(character not in "0123456789abcdef" for character in base_commit)
    ):
        raise Phase2SpecError(
            "architecture_base.commit must be a full lowercase 40-character Git SHA"
        )

    hard_failure_rules = spec["hard_failure_rules"]
    required_result_gates = {
        "state_rebuild_decode_prefill_max_abs_error",
        "chunk_boundary_parallel_decode_max_abs_error",
        "parameter_estimator_relative_error_max",
        "flop_estimator_relative_error_max",
    }
    missing_result_gates = sorted(
        required_result_gates - set(hard_failure_rules)
    )
    if missing_result_gates:
        raise Phase2SpecError(
            f"Missing result-accounting/parity gates: {missing_result_gates}"
        )
    for name in (
        "state_rebuild_decode_prefill_max_abs_error",
        "chunk_boundary_parallel_decode_max_abs_error",
    ):
        if float(hard_failure_rules[name]) != 1e-4:
            raise Phase2SpecError(f"{name} must remain hard-gated at 1e-4")
    for name in (
        "parameter_estimator_relative_error_max",
        "flop_estimator_relative_error_max",
    ):
        value = float(hard_failure_rules[name])
        if not 0.0 < value <= 1.0:
            raise Phase2SpecError(f"{name} must lie in (0, 1]")

    for name, expected in _MANDATED_VALUES.items():
        actual = _axis_values(spec, name)
        if actual != expected:
            raise Phase2SpecError(
                f"Scaling-plan axis drift for {name}: expected {expected}, got {actual}"
            )

    axes = spec["axes"]
    for name, axis in axes.items():
        if not isinstance(axis, Mapping):
            raise Phase2SpecError(f"Axis {name} must be an object")
        kind = axis.get("kind")
        if kind == "categorical":
            values = axis.get("values")
            if not isinstance(values, list) or not values:
                raise Phase2SpecError(f"Axis {name} needs nonempty values")
        elif kind == "float":
            if not all(k in axis for k in ("low", "high")):
                raise Phase2SpecError(f"Float axis {name} needs low/high")
            if float(axis["low"]) > float(axis["high"]):
                raise Phase2SpecError(f"Float axis {name} has low > high")
        else:
            raise Phase2SpecError(f"Axis {name} has unsupported kind={kind!r}")

        condition = axis.get("active_when")
        if condition is not None:
            if not isinstance(condition, Mapping) or len(condition) != 1:
                raise Phase2SpecError(
                    f"Axis {name} active_when must contain exactly one equality"
                )
            parent, wanted = next(iter(condition.items()))
            if parent not in axes:
                raise Phase2SpecError(f"Axis {name} depends on unknown axis {parent}")
            if wanted not in _axis_values(spec, parent):
                raise Phase2SpecError(
                    f"Axis {name} condition {parent}={wanted!r} is unreachable"
                )

    fidelity = spec["fidelity"]
    if fidelity.get("prune_step") != 500:
        raise Phase2SpecError("Scaling plan requires pruning at optimizer step 500")
    if fidelity.get("pruner") != "completed_trial_median":
        raise Phase2SpecError("Phase 2a must use completed-trial median pruning")
    if fidelity.get("prune_metric") != "mqar_accuracy":
        raise Phase2SpecError("Phase 2a pruning metric must be MQAR accuracy")
    if int(fidelity.get("max_steps", 0)) < 500:
        raise Phase2SpecError("max_steps must reach the step-500 pruning rung")
    if fidelity.get("max_steps") != 6000:
        raise Phase2SpecError("The initial 1M-core survivor budget must be 6,000 steps")
    if fidelity.get("checkpoint_steps") != [500, 2000, 6000]:
        raise Phase2SpecError("Phase 2 checkpoint rungs have drifted")
    if (
        not isinstance(fidelity.get("minimum_completed_trials"), int)
        or isinstance(fidelity.get("minimum_completed_trials"), bool)
        or int(fidelity["minimum_completed_trials"]) < 20
    ):
        raise Phase2SpecError("Median pruning requires at least 20 startup trials")

    protocol = spec["protocol"]
    if protocol.get("sequence_length") != 2048:
        raise Phase2SpecError("Initial Phase 2a sequence length must be 2048")
    if protocol.get("effective_batch_sequences") != 96:
        raise Phase2SpecError("Initial Phase 2a effective batch must be 96 sequences")
    if protocol.get("precision") != "bf16":
        raise Phase2SpecError("Initial Phase 2a precision must be BF16")
    effective = (
        int(protocol["world_size"])
        * int(protocol["per_device_microbatch_sequences"])
        * int(protocol["gradient_accumulation_steps"])
    )
    if effective != int(protocol["effective_batch_sequences"]):
        raise Phase2SpecError(
            f"Training batch contract is inconsistent: derived={effective}, "
            f"declared={protocol['effective_batch_sequences']}"
        )
    if protocol.get("optimizer") != "AdamW":
        raise Phase2SpecError("Phase 2a optimizer must be explicitly pinned to AdamW")
    if protocol.get("scheduler") != "cosine":
        raise Phase2SpecError("Phase 2a scheduler must be explicitly pinned")
    if protocol.get("adam_betas") != [0.9, 0.95]:
        raise Phase2SpecError("Phase 2a Adam betas must be [0.9, 0.95]")
    if float(protocol.get("adam_epsilon", 0.0)) <= 0.0:
        raise Phase2SpecError("Phase 2a Adam epsilon must be positive")
    if protocol.get("fused_adamw") is not True:
        raise Phase2SpecError("Phase 2a fused AdamW setting must be explicit")
    if float(protocol.get("base_learning_rate", 0.0)) <= 0.0:
        raise Phase2SpecError("Phase 2a base learning rate must be positive")
    if float(protocol.get("weight_decay", -1.0)) < 0.0:
        raise Phase2SpecError("Phase 2a weight decay cannot be negative")
    if not 0.0 <= float(protocol.get("warmup_fraction", -1.0)) < 1.0:
        raise Phase2SpecError("Phase 2a warmup fraction must lie in [0, 1)")
    if not 0.0 <= float(protocol.get("minimum_learning_rate_ratio", -1.0)) <= 1.0:
        raise Phase2SpecError("Phase 2a minimum LR ratio must lie in [0, 1]")
    if float(protocol.get("max_gradient_norm", 0.0)) <= 0.0:
        raise Phase2SpecError("Phase 2a gradient clipping norm must be positive")
    if protocol.get("loss_normalization") != (
        "mean_over_supervised_tokens_then_weighted_objective_sum"
    ):
        raise Phase2SpecError("Phase 2a loss normalization has drifted")
    if protocol.get("optimizer_step_order") != (
        "clip_then_optimizer_then_scheduler"
    ):
        raise Phase2SpecError("Phase 2a optimizer-step order has drifted")
    if protocol.get("gradient_checkpointing") is not False:
        raise Phase2SpecError(
            "Initial 1M-core runs must explicitly disable gradient checkpointing"
        )
    data_stream = protocol.get("data_stream")
    expected_stream_fields = {
        "sampler",
        "sampler_revision",
        "shuffle_seed",
        "shard_order_policy",
        "shard_order_manifest_sha256",
        "epoch_policy",
        "repeat_policy",
        "worker_partitioning",
    }
    if not isinstance(data_stream, Mapping) or set(data_stream) != (
        expected_stream_fields
    ):
        raise Phase2SpecError(
            "Phase 2a protocol.data_stream must freeze exactly "
            f"{sorted(expected_stream_fields)}"
        )
    if data_stream.get("sampler") != "deterministic_global_sequence_index":
        raise Phase2SpecError("Phase 2a data sampler semantics have drifted")
    if data_stream.get("worker_partitioning") != (
        "global_sequence_index modulo world_size after deterministic ordering"
    ):
        raise Phase2SpecError("Phase 2a worker partitioning semantics have drifted")

    controller_objective = spec["study"].get("controller_objective", {})
    if spec["study"].get("sampler") != "TPESampler":
        raise Phase2SpecError("Phase 2a sampler must be TPESampler")
    if spec["study"].get("sampler_options") != {
        "multivariate": True,
        "group": True,
    }:
        raise Phase2SpecError(
            "Conditional Phase 2 axes require grouped multivariate TPE"
        )
    if controller_objective.get("name") != "mqar_accuracy":
        raise Phase2SpecError("Optuna controller objective must be MQAR accuracy")
    if controller_objective.get("direction") != "maximize":
        raise Phase2SpecError("MQAR controller objective must be maximized")
    analysis = spec["study"].get("analysis_objectives")
    expected_analysis = [
        {"name": "mqar_accuracy", "direction": "maximize"},
        {"name": "wikitext_ppl", "direction": "minimize"},
    ]
    if analysis != expected_analysis:
        raise Phase2SpecError(
            f"Analysis objectives must be {expected_analysis}, got {analysis}"
        )
    if spec["study"].get("requested_trials") != 2000:
        raise Phase2SpecError("The Scaling Plan initial target is 2,000 trials")
    if spec["study"].get("promotion_seeds") != [42, 43, 44]:
        raise Phase2SpecError("Promotion seeds must remain [42, 43, 44]")
    compute_budget = spec["study"].get("compute_budget")
    if not isinstance(compute_budget, Mapping):
        raise Phase2SpecError("study.compute_budget must be declared")
    concurrency = compute_budget.get("maximum_concurrent_trials")
    if (
        not isinstance(concurrency, int)
        or isinstance(concurrency, bool)
        or concurrency <= 0
    ):
        raise Phase2SpecError("Compute concurrency must be a positive integer")
    reserve = compute_budget.get("failure_retry_reserve_fraction")
    if not isinstance(reserve, (int, float)) or isinstance(reserve, bool) or not (
        0.0 <= float(reserve) < 1.0
    ):
        raise Phase2SpecError("Compute retry reserve must lie in [0, 1)")
    resolved_hour_limits: dict[str, float] = {}
    for name in (
        "maximum_total_gpu_hours",
        "maximum_full_trial_gpu_hours",
    ):
        value = compute_budget.get(name)
        if value == "REQUIRED_BEFORE_SCIENTIFIC_RUN":
            continue
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or float(value) <= 0.0
        ):
            raise Phase2SpecError(f"{name} must be positive when resolved")
        resolved_hour_limits[name] = float(value)
    if (
        len(resolved_hour_limits) == 2
        and resolved_hour_limits["maximum_full_trial_gpu_hours"]
        > resolved_hour_limits["maximum_total_gpu_hours"]
    ):
        raise Phase2SpecError(
            "A full-trial GPU-hour reservation cannot exceed the total ceiling"
        )
    required_gpu_type = compute_budget.get("required_gpu_type")
    if not isinstance(required_gpu_type, str):
        raise Phase2SpecError("Required GPU type must be explicit")
    if (
        required_gpu_type != "REQUIRED_BEFORE_SCIENTIFIC_RUN"
        and not required_gpu_type.strip()
    ):
        raise Phase2SpecError("Resolved required GPU type cannot be empty")
    if spec.get("status") == "scientific_ready":
        if protocol.get("learning_rate_status") != "approved":
            raise Phase2SpecError(
                "Scientific-ready specs require approved base learning rate"
            )
        if protocol.get("training_protocol_status") != "approved":
            raise Phase2SpecError(
                "Scientific-ready specs require an approved training protocol"
            )
        if compute_budget.get("status") != "approved":
            raise Phase2SpecError(
                "Scientific-ready specs require an approved compute budget"
            )
        if len(resolved_hour_limits) != 2:
            raise Phase2SpecError(
                "Scientific-ready specs require resolved total and full-trial "
                "GPU-hour ceilings"
            )

    diagnostics = set(spec["required_diagnostics"])
    missing_diagnostics = sorted(_MANDATED_DIAGNOSTICS - diagnostics)
    if missing_diagnostics:
        raise Phase2SpecError(
            f"Missing required Phase 2 diagnostics: {missing_diagnostics}"
        )
    screen_diagnostics = set(spec["screen_required_diagnostics"])
    if not screen_diagnostics <= diagnostics:
        raise Phase2SpecError(
            "screen_required_diagnostics must be a subset of required_diagnostics"
        )
    structure = spec["diagnostic_structure"]
    head_diagnostics = structure.get("per_layer_head_metrics")
    layer_diagnostics = structure.get("per_layer_metrics")
    if (
        not isinstance(head_diagnostics, list)
        or any(not isinstance(name, str) for name in head_diagnostics)
        or len(set(head_diagnostics)) != len(head_diagnostics)
    ):
        raise Phase2SpecError("Per-layer/per-head diagnostics must be declared")
    if (
        not isinstance(layer_diagnostics, list)
        or any(not isinstance(name, str) for name in layer_diagnostics)
        or len(set(layer_diagnostics)) != len(layer_diagnostics)
    ):
        raise Phase2SpecError("Per-layer diagnostics must be declared")
    missing_head_diagnostics = sorted(
        _MANDATED_HEAD_DIAGNOSTICS - set(head_diagnostics)
    )
    missing_layer_diagnostics = sorted(
        _MANDATED_LAYER_DIAGNOSTICS - set(layer_diagnostics)
    )
    if missing_head_diagnostics or missing_layer_diagnostics:
        raise Phase2SpecError(
            "Diagnostic structure is incomplete: "
            f"missing_head={missing_head_diagnostics}, "
            f"missing_layer={missing_layer_diagnostics}"
        )

    required_evidence = spec["required_evidence_checks"]
    expected_evidence_types = {
        "correctness_report",
        "resume_parity_report",
        "data_validation_report",
        "storage_concurrency_report",
        "pilot_report",
    }
    if not isinstance(required_evidence, Mapping) or set(required_evidence) != (
        expected_evidence_types
    ):
        raise Phase2SpecError(
            "required_evidence_checks must define exactly "
            f"{sorted(expected_evidence_types)}"
        )
    for evidence_type, checks in required_evidence.items():
        if (
            not isinstance(checks, list)
            or not checks
            or any(not isinstance(check, str) or not check for check in checks)
            or len(checks) != len(set(checks))
        ):
            raise Phase2SpecError(
                f"Evidence checks for {evidence_type} must be unique strings"
            )

    storage = spec["storage"]
    stale_seconds = storage.get("claim_stale_after_seconds")
    if (
        not isinstance(stale_seconds, int)
        or isinstance(stale_seconds, bool)
        or stale_seconds <= 0
    ):
        raise Phase2SpecError(
            "storage.claim_stale_after_seconds must be a positive integer"
        )
    if storage.get("shared_sqlite_allowed") is not False:
        raise Phase2SpecError("Phase 2 shared SQLite must remain disabled")

    base_path = Path(spec.get("_base_model_path", spec["base_model_config"]))
    if not base_path.is_file():
        raise Phase2SpecError(f"Base model config does not exist: {base_path}")
    try:
        base = json.loads(base_path.read_text())
    except json.JSONDecodeError as exc:
        raise Phase2SpecError(f"Invalid base model JSON: {exc}") from exc
    n_layers = int(base.get("n_layers", 0))
    realized_counts = [round_ska_count(n_layers, f) for f in _axis_values(spec, "ska_fraction")]
    if len(set(realized_counts)) != len(realized_counts):
        raise Phase2SpecError(
            f"SKA fractions collapse at n_layers={n_layers}: {realized_counts}"
        )


def round_ska_count(n_layers: int, fraction: float) -> int:
    """Map a fraction to a count using documented round-half-up semantics."""

    if n_layers < 3:
        raise Phase2SpecError("At least three layers are required")
    if not 0.0 <= fraction <= 1.0:
        raise Phase2SpecError(f"Invalid SKA fraction: {fraction}")
    count = math.floor(n_layers * fraction + 0.5)
    return min(max(count, 0), n_layers - 2)


def _spread_indices(low: int, high: int, count: int) -> list[int]:
    if count == 0:
        return []
    candidates = list(range(low, high + 1))
    if count > len(candidates):
        raise Phase2SpecError(
            f"Cannot choose {count} unique layers from [{low}, {high}]"
        )
    if count == 1:
        return [(low + high) // 2]

    targets = [low + (high - low) * i / (count - 1) for i in range(count)]
    chosen: list[int] = []
    available = set(candidates)
    for target in targets:
        index = min(available, key=lambda x: (abs(x - target), x))
        chosen.append(index)
        available.remove(index)
    return sorted(chosen)


def placement_indices(n_layers: int, count: int, strategy: str) -> list[int]:
    """Materialize deterministic, zero-based candidate SKA layer indices."""

    if n_layers < 4:
        raise Phase2SpecError("Candidate placement requires at least four layers")
    if not 0 <= count <= n_layers - 2:
        raise Phase2SpecError(
            f"count={count} violates the interior-only policy for {n_layers} layers"
        )
    if count == 0:
        return []

    interior_low, interior_high = 1, n_layers - 2
    if strategy == "uniform":
        low, high = interior_low, interior_high
    elif strategy == "late_biased":
        low = max(interior_low, math.ceil(0.40 * (n_layers - 1)))
        high = interior_high
    elif strategy == "middle_clustered":
        low = max(interior_low, math.floor(0.33 * n_layers))
        high = min(interior_high, math.floor(0.75 * n_layers))
    else:
        raise Phase2SpecError(f"Unknown placement strategy: {strategy!r}")

    if high - low + 1 < count:
        raise Phase2SpecError(
            f"{strategy} window [{low}, {high}] cannot fit {count} layers"
        )
    result = _spread_indices(low, high, count)
    if result != sorted(set(result)):
        raise AssertionError(f"Placement produced duplicate/unsorted indices: {result}")
    if result and (result[0] == 0 or result[-1] == n_layers - 1):
        raise AssertionError(f"Placement violated first/last Mamba policy: {result}")
    return result


def _is_active(axis: Mapping[str, Any], params: Mapping[str, Any]) -> bool:
    condition = axis.get("active_when")
    if not condition:
        return True
    parent, wanted = next(iter(condition.items()))
    return params.get(parent) == wanted


def _float_grid(axis: Mapping[str, Any]) -> list[float]:
    low = float(axis["low"])
    high = float(axis["high"])
    step = axis.get("step")
    if step is None:
        return [low, high]
    step = float(step)
    count = int(round((high - low) / step))
    return [round(low + i * step, 12) for i in range(count + 1)]


def sample_parameters(
    spec: Mapping[str, Any], rng: random.Random | None = None
) -> dict[str, Any]:
    """Draw one legal conditional configuration without requiring Optuna."""

    rng = rng or random.Random()
    params: dict[str, Any] = {"architecture_mode": "updated_sweep"}
    for name, axis in spec["axes"].items():
        if not _is_active(axis, params):
            continue
        if axis["kind"] == "categorical":
            params[name] = rng.choice(axis["values"])
        else:
            params[name] = rng.choice(_float_grid(axis))
    validate_trial_parameters(spec, params)
    return params


def suggest_parameters(spec: Mapping[str, Any], trial: Any) -> dict[str, Any]:
    """Materialize parameters through an Optuna-like trial interface."""

    params: dict[str, Any] = {"architecture_mode": "updated_sweep"}
    for name, axis in spec["axes"].items():
        if not _is_active(axis, params):
            continue
        if axis["kind"] == "categorical":
            params[name] = trial.suggest_categorical(name, axis["values"])
        else:
            kwargs: dict[str, Any] = {}
            if axis.get("step") is not None:
                kwargs["step"] = axis["step"]
            if axis.get("log"):
                kwargs["log"] = True
            params[name] = trial.suggest_float(
                name, float(axis["low"]), float(axis["high"]), **kwargs
            )
    validate_trial_parameters(spec, params)
    return params


def validate_trial_parameters(
    spec: Mapping[str, Any], params: Mapping[str, Any]
) -> None:
    mode = params.get("architecture_mode")
    if mode not in {"updated_sweep", "paper_control", "mamba_only", "transformer"}:
        raise Phase2SpecError(f"Unknown architecture_mode={mode!r}")
    if mode != "updated_sweep":
        return

    for name, axis in spec["axes"].items():
        active = _is_active(axis, params)
        if active and name not in params:
            raise Phase2SpecError(f"Active axis {name} is missing")
        if not active and name in params:
            raise Phase2SpecError(f"Inactive conditional axis {name} must be omitted")
        if not active:
            continue
        value = params[name]
        if axis["kind"] == "categorical" and value not in axis["values"]:
            raise Phase2SpecError(f"Illegal value for {name}: {value!r}")
        if axis["kind"] == "float":
            if not float(axis["low"]) <= float(value) <= float(axis["high"]):
                raise Phase2SpecError(f"Value for {name} is outside its range: {value}")

    probability = float(params["beta_init_probability"])
    if not 0.0 < probability < 1.0:
        raise Phase2SpecError("Beta initialization probability must lie in (0, 1)")


def derive_objective_weights(params: Mapping[str, Any]) -> dict[str, float]:
    """Return normalized next-token/copy/infilling weights."""

    if params.get("objective_arm") == "next_token_only":
        return {"next_token": 1.0, "selective_copy": 0.0, "infilling": 0.0}
    if params.get("objective_arm") != "birdie_mix":
        raise Phase2SpecError(f"Unknown objective_arm={params.get('objective_arm')!r}")
    retrieval = float(params["birdie_retrieval_fraction"])
    copy_share = float(params["birdie_copy_share"])
    weights = {
        "next_token": 1.0 - retrieval,
        "selective_copy": retrieval * copy_share,
        "infilling": retrieval * (1.0 - copy_share),
    }
    if abs(sum(weights.values()) - 1.0) > 1e-12:
        raise AssertionError(f"Objective weights do not sum to one: {weights}")
    return weights


def estimate_parameter_counts(
    model: Mapping[str, Any], architecture_mode: str | None = None
) -> dict[str, int]:
    """Estimate embedding/core counts for Echo or a standard baseline.

    Baseline modes use the canonical builders from ``models.baselines``:
    all-Mamba-2 or all-attention sequence mixers and SwiGLU channel mixers.
    The estimator remains provisional until checked against a built CUDA model.
    """

    d = int(model["d_model"])
    vocab = int(model["vocab_size"])
    n_layers = int(model["n_layers"])
    indices = list(model.get("ska_layer_indices", []))
    baseline_mode = architecture_mode in {"mamba_only", "transformer"}
    n_ska = 0 if baseline_mode else len(indices)
    n_mamba = n_layers - n_ska
    n_heads = int(model["ska_n_heads"])
    rank = int(model["ska_rank"])
    head_dim = d // n_heads

    embedding = vocab * d * (1 if model.get("tie_embeddings", True) else 2)
    d_inner = d * int(model.get("mamba_expand", 2))
    per_mamba = (
        d * d_inner * 2
        + d_inner * int(model["d_state"]) * 2
        + d_inner * int(model.get("d_conv", 4))
        + d_inner
        + d_inner * d
    )
    if architecture_mode == "transformer":
        # qkv (3*d*d) + output projection (d*d); norms are counted below.
        mamba_total = 4 * d * d * n_layers
    else:
        mamba_total = per_mamba * n_mamba

    per_ska = (
        d * n_heads * rank * 2
        + d * n_heads * head_dim
        + n_heads * head_dim * d
        + d * n_heads
        + n_heads
        + (d if model.get("ska_layerscale", False) else 0)
        + (
            d * (int(model.get("ska_short_conv_kernel", 4)) + 1) + d
            if model.get("ska_short_conv", False)
            else 0
        )
        + 2
    )
    ska_total = per_ska * n_ska

    expanded = int(d * float(model.get("mlp_expand", 2.667)))
    d_k = ((expanded + 63) // 64) * 64
    if baseline_mode:
        # SwiGLU: gate/up/down, all bias-free; its LayerNorm is counted below.
        mlp_total = 3 * d * d_k * n_layers
    else:
        projections = 3 if model.get("mlp_gated", False) else 2
        mlp_total = (d * d_k * projections + d_k) * n_layers
    norms = n_layers * d * 2 + d

    core = mamba_total + ska_total + mlp_total + norms
    return {
        "embedding": embedding,
        "non_embedding_core": core,
        "total": embedding + core,
    }
