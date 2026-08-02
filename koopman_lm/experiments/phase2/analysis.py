"""Offline Pareto, conditional fANOVA, and promotion analysis for Phase 2a."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    from .checkpoint import checkpoint_bundle_identity
    from .manifest import (
        atomic_write_json,
        recompute_promotion_config_hash,
        recompute_trial_hash,
        stable_hash,
    )
    from .results import validate_cross_stage_identity, validate_step_metrics
except ImportError:
    from checkpoint import checkpoint_bundle_identity  # type: ignore
    from manifest import (  # type: ignore
        atomic_write_json,
        recompute_promotion_config_hash,
        recompute_trial_hash,
        stable_hash,
    )
    from results import (  # type: ignore
        validate_cross_stage_identity,
        validate_step_metrics,
    )


CONTROLLER_WARNING = (
    "The Optuna controller maximizes MQAR only. WikiText perplexity is attached "
    "after training and the MQAR-driven sampler may under-sample useful regions "
    "of the post-hoc MQAR/PPL Pareto front."
)

DEFAULT_ARCHITECTURE_AXES = (
    "architecture_mode",
    "ska_rank",
    "ska_fraction",
    "ska_placement",
    "ska_chunk_size",
    "ska_ridge",
    "beta_init_probability",
    "layerscale_init",
    "short_conv_kernel",
    "short_conv_gate_init",
    "qk_norm",
)

_STABILITY_FIELDS = (
    "stable",
    "stability_passed",
    "health_passed",
    "diagnostics_passed",
    "numerically_stable",
)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


_HYPOTHESES = (
    {
        "id": "rank_scales_with_width",
        "statement": "SKA rank must increase with model width to preserve retrieval quality.",
        "axes": ("ska_rank",),
        "falsification_rule": (
            "Falsify if lower-rank arms remain Pareto-competitive across promotion "
            "seeds and larger model widths."
        ),
    },
    {
        "id": "ska_fraction_and_placement",
        "statement": "Approximately 25% middle-to-deep SKA placement is optimal.",
        "axes": ("ska_fraction", "ska_placement"),
        "falsification_rule": (
            "Falsify if another fraction/placement family wins with seeded "
            "uncertainty intervals that exclude the reference family."
        ),
    },
    {
        "id": "chunk_size_matters",
        "statement": "The causal chunk size materially changes retrieval or language modeling.",
        "axes": ("ska_chunk_size",),
        "falsification_rule": (
            "Falsify if the chunk axis has negligible conditional importance and "
            "seeded matched comparisons are practically equivalent."
        ),
    },
    {
        "id": "differential_lr_ratios_transfer",
        "statement": "The hand-tuned differential learning-rate ratios transfer to 3M-total.",
        "axes": (
            "ska_projection_lr_multiplier",
            "gamma_eta_lr_multiplier",
            "koopman_eigen_lr_multiplier",
            "norm_lr_multiplier",
            "mlp_lr_multiplier",
            "backbone_lr_multiplier",
        ),
        "falsification_rule": (
            "Falsify when promoted configurations consistently select materially "
            "different ratios without losing numerical health."
        ),
    },
    {
        "id": "ridge_1e3_is_sufficient",
        "statement": "Ridge epsilon 1e-3 is sufficient throughout the rank sweep.",
        "axes": ("ska_ridge", "ska_rank"),
        "falsification_rule": (
            "Falsify if high-rank 1e-3 arms violate conditioning gates or a larger "
            "ridge improves stability without a Pareto penalty."
        ),
    },
    {
        "id": "beta_initialization",
        "statement": "The beta write-gate initialization materially affects the updated SKA path.",
        "axes": ("beta_init_probability",),
        "falsification_rule": (
            "Falsify if beta initialization is conditionally unimportant and seeded "
            "matched arms are practically equivalent."
        ),
    },
    {
        "id": "layerscale_initialization",
        "statement": "LayerScale initialization is load-bearing for stable SKA optimization.",
        "axes": ("layerscale_init",),
        "falsification_rule": (
            "Falsify if all initializations pass health gates and remain "
            "performance-equivalent across seeds."
        ),
    },
    {
        "id": "short_conv_design",
        "statement": "Short-convolution width and gate initialization improve the updated SKA path.",
        "axes": ("short_conv_kernel", "short_conv_gate_init"),
        "falsification_rule": (
            "Falsify if the conditional axes are negligible and ablations match the "
            "best seeded frontier."
        ),
    },
    {
        "id": "qknorm_required",
        "statement": "QKNorm/BCNorm is required for stable training at scale.",
        "axes": ("qk_norm",),
        "falsification_rule": (
            "Falsify at this scale if QKNorm-off arms pass all health gates and "
            "remain Pareto-competitive; re-test before extrapolating to larger scales."
        ),
    },
    {
        "id": "birdie_objectives_dominate",
        "statement": "The Birdie-style objective mix matters more than architecture choices.",
        "axes": (
            "objective_arm",
            "birdie_retrieval_fraction",
            "birdie_copy_share",
        ),
        "falsification_rule": (
            "Falsify if objective-arm and Birdie-conditional importances are small "
            "and seeded objective-matched comparisons do not move the frontier."
        ),
    },
)


def _finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _valid_record(record: Mapping[str, Any]) -> bool:
    return (
        record.get("status") == "COMPLETE"
        and _finite_number(record.get("mqar_accuracy"))
        and _finite_number(record.get("wikitext_ppl"))
        and 0.0 <= float(record["mqar_accuracy"]) <= 1.0
        and float(record["wikitext_ppl"]) > 0.0
    )


def dominates(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    """True when ``a`` is no worse on both metrics and strictly better on one."""

    a_mqar, b_mqar = float(a["mqar_accuracy"]), float(b["mqar_accuracy"])
    a_ppl, b_ppl = float(a["wikitext_ppl"]), float(b["wikitext_ppl"])
    return (
        a_mqar >= b_mqar
        and a_ppl <= b_ppl
        and (a_mqar > b_mqar or a_ppl < b_ppl)
    )


def pareto_front(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return the deterministic MQAR-max/PPL-min nondominated frontier."""

    valid = [dict(record) for record in records if _valid_record(record)]
    front = [
        record
        for index, record in enumerate(valid)
        if not any(
            dominates(other, record)
            for other_index, other in enumerate(valid)
            if index != other_index
        )
    ]
    return sorted(
        front,
        key=lambda record: (
            -float(record["mqar_accuracy"]),
            float(record["wikitext_ppl"]),
            str(record.get("trial_hash", "")),
        ),
    )


def _trial_params(trial: Any) -> Mapping[str, Any]:
    attrs = getattr(trial, "user_attrs", None)
    if isinstance(attrs, Mapping):
        materialized = attrs.get("materialized_parameters")
        if isinstance(materialized, Mapping):
            return materialized
    params = getattr(trial, "params", None)
    return params if isinstance(params, Mapping) else {}


def conditional_trial_views(trials: Iterable[Any]) -> dict[str, list[Any]]:
    """Split completed trials into fANOVA subspaces with common active axes."""

    materialized = list(trials)
    return {
        "global": materialized,
        "birdie_only": [
            trial
            for trial in materialized
            if _trial_params(trial).get("objective_arm") == "birdie_mix"
        ],
        "next_token_only": [
            trial
            for trial in materialized
            if _trial_params(trial).get("objective_arm") == "next_token_only"
        ],
        "updated_sweep_only": [
            trial
            for trial in materialized
            if _trial_params(trial).get("architecture_mode") == "updated_sweep"
        ],
    }


def _trial_metric_value(trial: Any, metric: str) -> float | None:
    if metric == "mqar_accuracy":
        value = getattr(trial, "value", None)
        return float(value) if _finite_number(value) else None
    if metric == "wikitext_ppl":
        attrs = getattr(trial, "user_attrs", None)
        value = attrs.get("wikitext_ppl") if isinstance(attrs, Mapping) else None
        if _finite_number(value) and float(value) > 0.0:
            return float(value)
        return None
    raise ValueError(f"Unknown importance metric: {metric}")


def select_trials_for_importance(trials: Iterable[Any], metric: str) -> list[Any]:
    """Filter a view to trials with a usable target, including missing-PPL safety."""

    return [
        trial
        for trial in trials
        if _trial_metric_value(trial, metric) is not None
    ]


def _trial_capacity_covariates(trial: Any) -> tuple[float, float] | None:
    attrs = getattr(trial, "user_attrs", None)
    if not isinstance(attrs, Mapping):
        return None
    core = attrs.get("actual_non_embedding_core_params")
    flops = attrs.get("measured_flops_per_optimizer_step")
    if (
        not _finite_number(core)
        or float(core) <= 0.0
        or not _finite_number(flops)
        or float(flops) <= 0.0
    ):
        return None
    return math.log(float(core)), math.log(float(flops))


def _capacity_adjusted_fanova(
    trials: Sequence[Any],
    *,
    metric: str,
    optuna: Any,
) -> dict[str, Any]:
    """Residualize capacity/compute before ranking sampled architecture axes."""

    selected = [
        trial
        for trial in select_trials_for_importance(trials, metric)
        if _trial_capacity_covariates(trial) is not None
    ]
    result: dict[str, Any] = {
        "method": (
            "ridge residualization on log(actual non-embedding parameters) "
            "and log(measured FLOPs), followed by fANOVA"
        ),
        "n_trials": len(selected),
        "importance": {},
    }
    coverage = _parameter_coverage(selected)
    result.update(coverage)
    if len(selected) < 5:
        result["status"] = "insufficient_trials_with_capacity_evidence"
        return result
    if not coverage["varying_common_parameters"]:
        result["status"] = "no_varying_common_parameters"
        return result
    try:
        from sklearn.linear_model import Ridge
        from optuna.importance import (
            FanovaImportanceEvaluator,
            get_param_importances,
        )

        features = [_trial_capacity_covariates(trial) for trial in selected]
        targets = [
            float(_trial_metric_value(trial, metric)) for trial in selected
        ]
        model = Ridge(alpha=1e-6).fit(features, targets)
        view_study = optuna.create_study(direction="maximize")
        view_study.add_trials(selected)
        evaluator = FanovaImportanceEvaluator(seed=42)

        def residual_target(trial: Any) -> float:
            covariates = _trial_capacity_covariates(trial)
            if covariates is None:
                raise ValueError("Missing capacity covariates in adjusted fANOVA")
            observed = float(_trial_metric_value(trial, metric))
            predicted = float(model.predict([covariates])[0])
            return observed - predicted

        importance = get_param_importances(
            view_study,
            evaluator=evaluator,
            target=residual_target,
            normalize=True,
        )
        result.update(
            {
                "status": "ok",
                "capacity_model_r2": float(model.score(features, targets)),
                "importance": {
                    name: float(value) for name, value in importance.items()
                },
                "actual_core_parameter_range": [
                    min(math.exp(row[0]) for row in features),
                    max(math.exp(row[0]) for row in features),
                ],
                "measured_flop_range": [
                    min(math.exp(row[1]) for row in features),
                    max(math.exp(row[1]) for row in features),
                ],
            }
        )
    except Exception as exc:
        result.update(
            {
                "status": "capacity_adjusted_fanova_unavailable",
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
    return result


def _parameter_coverage(trials: Sequence[Any]) -> dict[str, list[str]]:
    if not trials:
        return {
            "common_parameters": [],
            "varying_common_parameters": [],
            "conditional_or_missing_parameters": [],
        }
    param_sets = [set(_trial_params(trial)) for trial in trials]
    common = set.intersection(*param_sets)
    union = set.union(*param_sets)
    varying = {
        name
        for name in common
        if len(
            {
                json.dumps(
                    _trial_params(trial)[name],
                    sort_keys=True,
                    separators=(",", ":"),
                )
                for trial in trials
            }
        )
        > 1
    }
    return {
        "common_parameters": sorted(common),
        "varying_common_parameters": sorted(varying),
        "conditional_or_missing_parameters": sorted(union - common),
    }


def _importance_for_view(
    trials: Sequence[Any],
    *,
    metric: str,
    optuna: Any,
) -> dict[str, Any]:
    selected = select_trials_for_importance(trials, metric)
    coverage = _parameter_coverage(selected)
    result: dict[str, Any] = {
        "n_trials_in_view": len(trials),
        "n_trials_with_target": len(selected),
        "n_trials_missing_target": len(trials) - len(selected),
        **coverage,
        "importance": {},
        "capacity_adjusted": _capacity_adjusted_fanova(
            trials,
            metric=metric,
            optuna=optuna,
        ),
    }
    if len(selected) < 2:
        result["status"] = "insufficient_trials"
        return result
    if not coverage["varying_common_parameters"]:
        result["status"] = "no_varying_common_parameters"
        return result

    from optuna.importance import FanovaImportanceEvaluator, get_param_importances

    view_study = optuna.create_study(direction="maximize")
    view_study.add_trials(selected)
    evaluator = FanovaImportanceEvaluator(seed=42)
    try:
        importance = get_param_importances(
            view_study,
            evaluator=evaluator,
            target=lambda trial: float(_trial_metric_value(trial, metric)),
            normalize=True,
        )
    except Exception as exc:  # Sparse/degenerate fANOVA views should not erase others.
        result.update(
            {
                "status": "fanova_unavailable_for_view",
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        return result
    result["status"] = "ok"
    result["importance"] = {
        str(name): float(value) for name, value in importance.items()
    }
    return result


def fanova_importance(study: Any) -> dict[str, Any]:
    """Calculate global and conditional fANOVA views for both final metrics.

    Optuna normally intersects parameter names over all completed trials. That
    silently drops Birdie-only parameters from a mixed global study. The
    explicit ``birdie_only`` view keeps those parameters active, while the
    global view remains available for unconditional comparisons.
    """

    try:
        import optuna
        from optuna.trial import TrialState
    except ImportError as exc:
        raise RuntimeError("Install the optional 'nas' dependencies for fANOVA") from exc

    completed = list(
        study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
    )
    views = conditional_trial_views(completed)
    return {
        "controller_objective": {
            "name": "mqar_accuracy",
            "direction": "maximize",
            "single_objective": True,
        },
        "sampling_warning": CONTROLLER_WARNING,
        "views": {
            view_name: {
                metric: _importance_for_view(
                    view_trials,
                    metric=metric,
                    optuna=optuna,
                )
                for metric in ("mqar_accuracy", "wikitext_ppl")
            }
            for view_name, view_trials in views.items()
        },
    }


def _record_parameters(record: Mapping[str, Any]) -> Mapping[str, Any]:
    params = record.get("parameters")
    return params if isinstance(params, Mapping) else record


_CONTROL_NAME_FIELDS = (
    "control",
    "control_name",
    "fixed_control",
    "fixed_control_name",
)


def _named_control(record: Mapping[str, Any]) -> str | None:
    """Return the fixed-control label carried by a result or its manifest params."""

    parameters = _record_parameters(record)
    for source in (record, parameters):
        for field in _CONTROL_NAME_FIELDS:
            value = source.get(field)
            if isinstance(value, str) and value:
                return value
    # ``control_parameters`` preserves the fixed-control spec's ``name`` field
    # in the immutable manifest. Ordinary Optuna suggestions have no such field.
    name = parameters.get("name")
    if isinstance(name, str) and name:
        return name
    return None


def _is_unnamed_updated_sweep(record: Mapping[str, Any]) -> bool:
    parameters = _record_parameters(record)
    return (
        parameters.get("architecture_mode") == "updated_sweep"
        and _named_control(record) is None
    )


def architecture_signature(
    record: Mapping[str, Any],
    axes: Sequence[str] = DEFAULT_ARCHITECTURE_AXES,
) -> dict[str, Any]:
    """Extract the categorical architecture signature used for diversity."""

    params = _record_parameters(record)
    return {name: params[name] for name in axes if name in params}


def capacity_covariates(record: Mapping[str, Any]) -> dict[str, Any]:
    """Extract actual capacity/compute evidence for analysis and W&B tables."""

    accounting = record.get("model_accounting")
    if isinstance(accounting, Mapping):
        counts = accounting.get("parameter_counts_actual")
        flops = accounting.get("flops")
        if isinstance(counts, Mapping) and isinstance(flops, Mapping):
            return {
                "actual_non_embedding_core_params": counts.get(
                    "non_embedding_core"
                ),
                "actual_total_params": counts.get("total"),
                "estimated_flops_per_optimizer_step": flops.get("estimated"),
                "measured_flops_per_optimizer_step": flops.get("measured"),
                "gpu_hours_actual": record.get("gpu_hours_actual"),
            }
    value = record.get("capacity_covariates")
    return dict(value) if isinstance(value, Mapping) else {}


def _passed_marker(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"pass", "passed", "stable", "healthy", "ok", "complete"}:
            return True
        if normalized in {
            "fail",
            "failed",
            "unstable",
            "unhealthy",
            "error",
            "invalid",
        }:
            return False
    return None


def stability_evidence(record: Mapping[str, Any]) -> dict[str, Any]:
    """Summarize explicit health fields without inventing evidence when absent."""

    observed: dict[str, Any] = {}
    verdicts: list[bool] = []
    for name in _STABILITY_FIELDS:
        if name in record:
            observed[name] = record[name]
            marker = _passed_marker(record[name])
            verdicts.append(False if marker is None else marker)
    for container_name in ("health", "stability"):
        if container_name not in record:
            continue
        container = record[container_name]
        if isinstance(container, Mapping):
            for field in ("passed", "status", "stable", "healthy"):
                if field in container:
                    key = f"{container_name}.{field}"
                    observed[key] = container[field]
                    marker = _passed_marker(container[field])
                    verdicts.append(False if marker is None else marker)
        else:
            observed[container_name] = container
            marker = _passed_marker(container)
            verdicts.append(False if marker is None else marker)
    if not verdicts:
        status = "not_recorded"
    elif all(verdicts):
        status = "passed"
    else:
        status = "failed"
    return {"status": status, "observed": observed}


def validate_wandb_trial_mapping(
    records: Iterable[Mapping[str, Any]],
) -> None:
    """Require healthy scientific trials to map injectively to W&B runs."""

    by_run_id: dict[str, str] = {}
    for record in records:
        status = record.get("status")
        if status not in {"COMPLETE", "PRUNED", "SCREEN_COMPLETE"}:
            continue
        trial_hash = record.get("trial_hash")
        run_id = record.get("wandb_run_id")
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError(
                f"Healthy trial {trial_hash!r} has no nonempty wandb_run_id"
            )
        if not isinstance(trial_hash, str) or not trial_hash:
            raise ValueError(f"W&B run {run_id!r} has no trial hash")
        previous = by_run_id.setdefault(run_id, trial_hash)
        if previous != trial_hash:
            raise ValueError(
                "One W&B run ID maps to multiple trial hashes: "
                f"{run_id!r} -> {previous}, {trial_hash}"
            )


def _architecture_distance(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    axes: Sequence[str],
) -> float:
    left_signature = architecture_signature(left, axes)
    right_signature = architecture_signature(right, axes)
    shared = sorted(set(left_signature) & set(right_signature))
    if not shared:
        return 0.0
    mismatches = sum(
        left_signature[name] != right_signature[name] for name in shared
    )
    return mismatches / len(shared)


def promotion_shortlist(
    records: Iterable[Mapping[str, Any]],
    *,
    cap: int = 10,
    require_stability_evidence: bool = False,
    architecture_axes: Sequence[str] = DEFAULT_ARCHITECTURE_AXES,
) -> dict[str, Any]:
    """Select stable Pareto candidates with deterministic architecture diversity."""

    if cap < 1:
        raise ValueError("promotion cap must be at least one")
    materialized = [dict(record) for record in records]
    valid = [record for record in materialized if _valid_record(record)]
    candidate_records = [
        record for record in valid if _is_unnamed_updated_sweep(record)
    ]
    evidence = [stability_evidence(record) for record in candidate_records]
    eligible_runs: list[dict[str, Any]] = []
    for record, record_evidence in zip(candidate_records, evidence):
        status = record_evidence["status"]
        if status == "failed":
            continue
        if require_stability_evidence and status != "passed":
            continue
        eligible_runs.append(record)

    # Promotion is configuration-level. Controls and already-promoted candidates
    # can contribute several seed-specific runs to one results tree; selecting
    # those runs separately would waste shortlist slots and later create a
    # partially materialized promotion directory.
    by_configuration: dict[str, dict[str, Any]] = {}
    for record in eligible_runs:
        promotion_hash = record.get("promotion_config_hash")
        trial_hash = record.get("trial_hash")
        key = (
            f"config:{promotion_hash}"
            if isinstance(promotion_hash, str)
            else f"trial:{trial_hash}"
        )
        existing = by_configuration.get(key)
        if existing is not None and dict(_record_parameters(existing)) != dict(
            _record_parameters(record)
        ):
            raise ValueError(
                f"Configuration identity {key} has inconsistent parameters"
            )
        record_seed = record.get("seed")
        existing_seed = existing.get("seed") if existing is not None else None
        record_order = (
            record_seed if isinstance(record_seed, int) else 2**31 - 1,
            str(record.get("trial_hash", "")),
        )
        existing_order = (
            existing_seed if isinstance(existing_seed, int) else 2**31 - 1,
            str(existing.get("trial_hash", "")) if existing is not None else "",
        )
        if existing is None or record_order < existing_order:
            by_configuration[key] = record
    eligible = list(by_configuration.values())

    candidates = pareto_front(eligible)
    selected: list[tuple[dict[str, Any], str]] = []

    def add(record: dict[str, Any], reason: str) -> None:
        if len(selected) >= cap or any(item is record for item, _ in selected):
            return
        selected.append((record, reason))

    if candidates:
        add(candidates[0], "highest_mqar_anchor")
    if len(selected) < cap and candidates:
        ppl_anchor = min(
            candidates,
            key=lambda record: (
                float(record["wikitext_ppl"]),
                -float(record["mqar_accuracy"]),
                str(record.get("trial_hash", "")),
            ),
        )
        add(ppl_anchor, "lowest_ppl_anchor")

    while len(selected) < min(cap, len(candidates)):
        remaining = [
            record
            for record in candidates
            if not any(item is record for item, _ in selected)
        ]
        choice = min(
            remaining,
            key=lambda record: (
                -min(
                    _architecture_distance(record, chosen, architecture_axes)
                    for chosen, _ in selected
                ),
                -float(record["mqar_accuracy"]),
                float(record["wikitext_ppl"]),
                str(record.get("trial_hash", "")),
            ),
        )
        distance = min(
            _architecture_distance(choice, chosen, architecture_axes)
            for chosen, _ in selected
        )
        add(choice, f"architecture_diversity_min_distance={distance:.6f}")

    promoted: list[dict[str, Any]] = []
    for rank, (record, reason) in enumerate(selected, start=1):
        promoted.append(
            {
                **record,
                "promotion_rank": rank,
                "selection_reason": reason,
                "architecture_signature": architecture_signature(
                    record, architecture_axes
                ),
                "stability_evidence": stability_evidence(record),
            }
        )

    return {
        "schema_version": 1,
        "selection_policy": {
            "metrics": {
                "mqar_accuracy": "maximize",
                "wikitext_ppl": "minimize",
            },
            "pareto_required": True,
            "candidate_scope": (
                "architecture_mode=updated_sweep with no fixed-control name"
            ),
            "fixed_controls_excluded": True,
            "explicit_health_failure_excluded": True,
            "require_stability_evidence": require_stability_evidence,
            "cap": cap,
            "architecture_axes": list(architecture_axes),
            "diversity_method": (
                "MQAR and PPL anchors, then greedy maximum minimum categorical "
                "distance with deterministic quality/hash tie-breaks"
            ),
        },
        "candidate_counts": {
            "input": len(materialized),
            "complete_with_both_metrics": len(valid),
            "updated_sweep_unnamed_candidates": len(candidate_records),
            "excluded_non_updated_sweep": sum(
                _record_parameters(record).get("architecture_mode")
                != "updated_sweep"
                for record in valid
            ),
            "excluded_named_controls": sum(
                _record_parameters(record).get("architecture_mode")
                == "updated_sweep"
                and _named_control(record) is not None
                for record in valid
            ),
            "explicitly_unstable": sum(
                item["status"] == "failed" for item in evidence
            ),
            "without_stability_evidence": sum(
                item["status"] == "not_recorded" for item in evidence
            ),
            "eligible": len(eligible),
            "eligible_seed_specific_runs": len(eligible_runs),
            "eligible_unique_configurations": len(eligible),
            "stable_pareto": len(candidates),
            "selected": len(promoted),
        },
        "selected": promoted,
    }


def _step_500_mqar(record: Mapping[str, Any]) -> float | None:
    value = record.get("step_500_mqar_accuracy")
    if _finite_number(value) and 0.0 <= float(value) <= 1.0:
        return float(value)
    return None


def retrospective_control_false_pruning_risk(
    records: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare named controls with the aggregate candidate step-500 median.

    This is deliberately not an order-aware replay of Optuna's MedianPruner.
    It is a retrospective diagnostic for controls that would sit below the
    corpus-wide candidate median if they had been exposed to that threshold.
    """

    materialized = [dict(record) for record in records]
    eligible_statuses = {"COMPLETE", "PRUNED", "SCREEN_COMPLETE"}
    candidate_evidence = sorted(
        (
            {
                "trial_hash": record.get("trial_hash"),
                "seed": record.get("seed"),
                "status": record.get("status"),
                "step_500_mqar_accuracy": value,
            }
            for record in materialized
            if record.get("status") in eligible_statuses
            and _is_unnamed_updated_sweep(record)
            and (value := _step_500_mqar(record)) is not None
        ),
        key=lambda item: (
            item["seed"] if isinstance(item["seed"], int) else 2**31 - 1,
            str(item["trial_hash"] or ""),
        ),
    )
    candidate_observations = [
        float(item["step_500_mqar_accuracy"]) for item in candidate_evidence
    ]
    candidate_median = (
        float(statistics.median(candidate_observations))
        if candidate_observations
        else None
    )

    grouped_controls: dict[str, list[dict[str, Any]]] = {}
    for record in materialized:
        control_name = _named_control(record)
        if control_name is not None:
            grouped_controls.setdefault(control_name, []).append(record)

    controls: list[dict[str, Any]] = []
    total_observations = 0
    total_below = 0
    for control_name, members in sorted(grouped_controls.items()):
        ordered = sorted(
            members,
            key=lambda record: (
                record.get("seed")
                if isinstance(record.get("seed"), int)
                else 2**31 - 1,
                str(record.get("trial_hash", "")),
            ),
        )
        runs: list[dict[str, Any]] = []
        observed_values: list[float] = []
        below_count = 0
        architecture_modes = sorted(
            {
                str(_record_parameters(record).get("architecture_mode"))
                for record in ordered
                if _record_parameters(record).get("architecture_mode") is not None
            }
        )
        for record in ordered:
            value = (
                _step_500_mqar(record)
                if record.get("status") in eligible_statuses
                else None
            )
            below = (
                value < candidate_median
                if value is not None and candidate_median is not None
                else None
            )
            if value is not None:
                observed_values.append(value)
            if below is True:
                below_count += 1
            runs.append(
                {
                    "trial_hash": record.get("trial_hash"),
                    "seed": record.get("seed"),
                    "status": record.get("status"),
                    "step_500_mqar_accuracy": value,
                    "below_retrospective_candidate_median": below,
                }
            )

        total_observations += len(observed_values)
        total_below += below_count
        if candidate_median is None:
            risk_label = "insufficient_candidate_reference"
        elif not observed_values:
            risk_label = "insufficient_control_step_500_evidence"
        elif below_count:
            risk_label = "potential_false_pruning_risk"
        else:
            risk_label = "no_below_median_signal"
        controls.append(
            {
                "control_name": control_name,
                "architecture_modes": architecture_modes,
                "run_count": len(ordered),
                "step_500_observation_count": len(observed_values),
                "step_500_mqar_accuracy_median": (
                    float(statistics.median(observed_values))
                    if observed_values
                    else None
                ),
                "below_candidate_median_count": below_count,
                "retrospective_risk_label": risk_label,
                "runs": runs,
            }
        )

    return {
        "schema_version": 1,
        "analysis_type": "retrospective_control_false_pruning_risk",
        "analysis_scope": "retrospective_diagnostic_only",
        "not_an_actual_pruner_replay": True,
        "warning": (
            "This compares controls with one corpus-wide candidate median. It "
            "does not reproduce trial ordering, startup-trial rules, or the "
            "historical threshold seen by any Optuna trial, and therefore does "
            "not assert that a control was actually or falsely pruned."
        ),
        "comparison_rule": (
            "Potential risk means a named control's step_500_mqar_accuracy is "
            "strictly below the median across eligible unnamed updated-sweep "
            "candidate observations."
        ),
        "candidate_reference": {
            "scope": (
                "unnamed architecture_mode=updated_sweep records in "
                "COMPLETE, PRUNED, or SCREEN_COMPLETE state"
            ),
            "step_500_observation_count": len(candidate_observations),
            "step_500_mqar_accuracy_median": candidate_median,
            "observations": candidate_evidence,
        },
        "control_count": len(controls),
        "control_step_500_observation_count": total_observations,
        "control_observations_below_candidate_median": total_below,
        "controls": controls,
    }


def seeded_stability_summary(
    records: Iterable[Mapping[str, Any]],
    *,
    required_seeds: Sequence[int] = (42, 43, 44),
) -> dict[str, Any]:
    """Aggregate numerically healthy three-seed confirmation runs.

    This never manufactures a seedless identity from displayed parameters:
    every record must carry the content-addressed ``promotion_config_hash``
    emitted by the trial manifest.
    """

    required = tuple(int(seed) for seed in required_seeds)
    if not required or len(set(required)) != len(required):
        raise ValueError("required_seeds must be nonempty and unique")
    groups: dict[str, list[dict[str, Any]]] = {}
    ignored = 0
    for source in records:
        record = dict(source)
        config_hash = record.get("promotion_config_hash")
        seed = record.get("seed")
        if (
            not _valid_record(record)
            or not _is_sha256(config_hash)
            or not isinstance(seed, int)
            or isinstance(seed, bool)
        ):
            ignored += 1
            continue
        groups.setdefault(config_hash, []).append(record)

    summaries: list[dict[str, Any]] = []
    confirmed_records: list[dict[str, Any]] = []
    required_set = set(required)
    for config_hash, members in sorted(groups.items()):
        by_seed: dict[int, dict[str, Any]] = {}
        duplicate_seeds: set[int] = set()
        for record in members:
            seed = int(record["seed"])
            if seed in by_seed:
                duplicate_seeds.add(seed)
            else:
                by_seed[seed] = record
        missing = sorted(required_set - set(by_seed))
        health = {
            seed: stability_evidence(record)["status"]
            for seed, record in by_seed.items()
            if seed in required_set
        }
        health_passed = (
            not missing
            and not duplicate_seeds
            and all(health.get(seed) == "passed" for seed in required)
        )
        requested_members = [
            by_seed[seed] for seed in required if seed in by_seed
        ]
        mqar_values = [
            float(record["mqar_accuracy"]) for record in requested_members
        ]
        ppl_values = [
            float(record["wikitext_ppl"]) for record in requested_members
        ]

        def metric_summary(values: Sequence[float]) -> dict[str, float] | None:
            if not values:
                return None
            mean = statistics.mean(values)
            standard_deviation = statistics.stdev(values) if len(values) > 1 else 0.0
            # Student-t 95% critical value for n=3; summaries with fewer seeds
            # are incomplete and the interval is descriptive only.
            critical = 4.303 if len(values) == 3 else 1.96
            half_width = (
                critical * standard_deviation / math.sqrt(len(values))
                if len(values) > 1
                else 0.0
            )
            return {
                "mean": mean,
                "standard_deviation": standard_deviation,
                "minimum": min(values),
                "maximum": max(values),
                "ci95_low": mean - half_width,
                "ci95_high": mean + half_width,
            }

        status = (
            "confirmed"
            if health_passed
            else (
                "duplicate_seed"
                if duplicate_seeds
                else "incomplete_or_unhealthy"
            )
        )
        summary = {
            "promotion_config_hash": config_hash,
            "status": status,
            "required_seeds": list(required),
            "observed_seeds": sorted(by_seed),
            "missing_seeds": missing,
            "duplicate_seeds": sorted(duplicate_seeds),
            "health_by_seed": health,
            "constituent_trial_hashes": [
                record.get("trial_hash") for record in requested_members
            ],
            "mqar_accuracy": metric_summary(mqar_values),
            "wikitext_ppl": metric_summary(ppl_values),
            "parameters": (
                dict(_record_parameters(requested_members[0]))
                if requested_members
                else {}
            ),
        }
        summaries.append(summary)
        if status == "confirmed":
            study_identities = {
                json.dumps(
                    record.get("study_identity"),
                    sort_keys=True,
                    separators=(",", ":"),
                )
                for record in requested_members
                if isinstance(record.get("study_identity"), Mapping)
            }
            if len(study_identities) > 1:
                raise ValueError(
                    f"Seed group {config_hash} spans multiple study identities"
                )
            confirmed_records.append(
                {
                    "schema_version": 1,
                    "promotion_config_hash": config_hash,
                    "status": "COMPLETE",
                    "mqar_accuracy": summary["mqar_accuracy"]["mean"],
                    "wikitext_ppl": summary["wikitext_ppl"]["mean"],
                    "health_passed": True,
                    "parameters": summary["parameters"],
                    "seed_confirmation": summary,
                    **(
                        {"study_identity": dict(requested_members[0]["study_identity"])}
                        if requested_members
                        and isinstance(
                            requested_members[0].get("study_identity"), Mapping
                        )
                        else {}
                    ),
                }
            )
    return {
        "schema_version": 1,
        "required_seeds": list(required),
        "input_records_ignored": ignored,
        "configuration_count": len(summaries),
        "confirmed_configuration_count": len(confirmed_records),
        "configurations": summaries,
        "confirmed_records": confirmed_records,
    }


def _axis_values(
    records: Sequence[Mapping[str, Any]], axes: Sequence[str]
) -> dict[str, list[Any]]:
    values: dict[str, list[Any]] = {}
    for axis in axes:
        encoded: dict[str, Any] = {}
        for record in records:
            params = _record_parameters(record)
            if axis in params:
                value = params[axis]
                encoded[
                    json.dumps(value, sort_keys=True, separators=(",", ":"))
                ] = value
        values[axis] = [encoded[key] for key in sorted(encoded)]
    return values


def falsified_hypotheses_report(
    records: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build an evidence scaffold; conclusions remain explicit scientist decisions."""

    completed = [dict(record) for record in records if _valid_record(record)]
    hypotheses: list[dict[str, Any]] = []
    for definition in _HYPOTHESES:
        coverage = _axis_values(completed, definition["axes"])
        sufficiently_varied = all(len(values) >= 2 for values in coverage.values())
        hypotheses.append(
            {
                **definition,
                "axes": list(definition["axes"]),
                "observed_values": coverage,
                "review_status": (
                    "unreviewed"
                    if sufficiently_varied
                    else "insufficient_axis_coverage"
                ),
                "falsified": None,
                "evidence_trial_hashes": [],
                "seeded_effect_summary": None,
                "reviewer_conclusion": None,
            }
        )
    return {
        "schema_version": 1,
        "report_status": "scientist_review_required",
        "completed_trials_with_both_metrics": len(completed),
        "decision_rule": (
            "Do not infer falsification from fANOVA rank alone. Record seeded "
            "matched comparisons, uncertainty, numerical-health evidence, and "
            "larger-scale checks where the statement extrapolates across scale."
        ),
        "hypotheses": hypotheses,
    }


def _table(columns: Sequence[str], rows: Iterable[Sequence[Any]]) -> dict[str, Any]:
    return {"columns": list(columns), "data": [list(row) for row in rows]}


def _fanova_rows(fanova: Mapping[str, Any] | None) -> list[list[Any]]:
    rows: list[list[Any]] = []
    if not isinstance(fanova, Mapping):
        return rows
    views = fanova.get("views")
    if not isinstance(views, Mapping):
        return rows
    for view_name, metrics in sorted(views.items()):
        if not isinstance(metrics, Mapping):
            continue
        for metric_name, result in sorted(metrics.items()):
            if not isinstance(result, Mapping):
                continue
            variants: list[tuple[str, Mapping[str, Any], int]] = []
            importance = result.get("importance")
            if isinstance(importance, Mapping):
                variants.append(
                    (
                        metric_name,
                        importance,
                        int(result.get("n_trials_with_target", 0)),
                    )
                )
            adjusted = result.get("capacity_adjusted")
            if isinstance(adjusted, Mapping) and isinstance(
                adjusted.get("importance"), Mapping
            ):
                variants.append(
                    (
                        f"{metric_name}_capacity_adjusted",
                        adjusted["importance"],
                        int(adjusted.get("n_trials", 0)),
                    )
                )
            for variant_name, values, trial_count in variants:
                for parameter, value in sorted(values.items()):
                    rows.append(
                        [
                            view_name,
                            variant_name,
                            parameter,
                            float(value),
                            trial_count,
                        ]
                    )
    return rows


def build_wandb_report_payload(
    pareto: Sequence[Mapping[str, Any]],
    shortlist: Mapping[str, Any],
    fanova: Mapping[str, Any] | None,
    hypotheses: Mapping[str, Any],
    seeded_stability: Mapping[str, Any] | None = None,
    confirmed_shortlist: Mapping[str, Any] | None = None,
    study_identity: Mapping[str, Any] | None = None,
    control_pruning_risk: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return JSON tables/panel specs consumable by W&B without contacting W&B."""

    selected = shortlist.get("selected", [])
    if not isinstance(selected, list):
        selected = []
    hypothesis_entries = hypotheses.get("hypotheses", [])
    if not isinstance(hypothesis_entries, list):
        hypothesis_entries = []
    seeded_configurations = (
        seeded_stability.get("configurations", [])
        if isinstance(seeded_stability, Mapping)
        else []
    )
    if not isinstance(seeded_configurations, list):
        seeded_configurations = []
    confirmed_selected = (
        confirmed_shortlist.get("selected", [])
        if isinstance(confirmed_shortlist, Mapping)
        else []
    )
    if not isinstance(confirmed_selected, list):
        confirmed_selected = []
    control_risk_entries = (
        control_pruning_risk.get("controls", [])
        if isinstance(control_pruning_risk, Mapping)
        else []
    )
    if not isinstance(control_risk_entries, list):
        control_risk_entries = []
    candidate_step_500_median = (
        control_pruning_risk.get("candidate_reference", {}).get(
            "step_500_mqar_accuracy_median"
        )
        if isinstance(control_pruning_risk, Mapping)
        and isinstance(control_pruning_risk.get("candidate_reference"), Mapping)
        else None
    )
    return {
        "schema_version": 1,
        "kind": "wandb_report_payload",
        "network_write_performed": False,
        "publication_status": "manual_authorized_upload_required",
        "published_report_url": None,
        "study_identity": (
            dict(study_identity) if isinstance(study_identity, Mapping) else None
        ),
        "sampling_warning": CONTROLLER_WARNING,
        "tables": {
            "pareto_front": _table(
                (
                    "trial_hash",
                    "wandb_run_id",
                    "step_500_mqar_accuracy",
                    "mqar_accuracy",
                    "wikitext_ppl",
                    "actual_core_params",
                    "actual_total_params",
                    "measured_flops_per_optimizer_step",
                    "gpu_hours_actual",
                    "stability_status",
                    "architecture_signature_json",
                ),
                (
                    (
                        record.get("trial_hash"),
                        record.get("wandb_run_id"),
                        _step_500_mqar(record),
                        float(record["mqar_accuracy"]),
                        float(record["wikitext_ppl"]),
                        capacity_covariates(record).get(
                            "actual_non_embedding_core_params"
                        ),
                        capacity_covariates(record).get("actual_total_params"),
                        capacity_covariates(record).get(
                            "measured_flops_per_optimizer_step"
                        ),
                        capacity_covariates(record).get("gpu_hours_actual"),
                        stability_evidence(record)["status"],
                        json.dumps(
                            architecture_signature(record),
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                    )
                    for record in pareto
                ),
            ),
            "promotion_shortlist": _table(
                (
                    "promotion_rank",
                    "trial_hash",
                    "wandb_run_id",
                    "step_500_mqar_accuracy",
                    "mqar_accuracy",
                    "wikitext_ppl",
                    "actual_core_params",
                    "measured_flops_per_optimizer_step",
                    "selection_reason",
                    "stability_status",
                    "architecture_signature_json",
                ),
                (
                    (
                        record.get("promotion_rank"),
                        record.get("trial_hash"),
                        record.get("wandb_run_id"),
                        _step_500_mqar(record),
                        float(record["mqar_accuracy"]),
                        float(record["wikitext_ppl"]),
                        capacity_covariates(record).get(
                            "actual_non_embedding_core_params"
                        ),
                        capacity_covariates(record).get(
                            "measured_flops_per_optimizer_step"
                        ),
                        record.get("selection_reason"),
                        (
                            record.get("stability_evidence", {}).get("status")
                            if isinstance(record.get("stability_evidence"), Mapping)
                            else None
                        ),
                        json.dumps(
                            record.get("architecture_signature", {}),
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                    )
                    for record in selected
                ),
            ),
            "fanova_importance": _table(
                ("view", "metric", "parameter", "importance", "n_trials"),
                _fanova_rows(fanova),
            ),
            "falsified_hypotheses": _table(
                (
                    "id",
                    "review_status",
                    "falsified",
                    "statement",
                    "falsification_rule",
                ),
                (
                    (
                        item.get("id"),
                        item.get("review_status"),
                        item.get("falsified"),
                        item.get("statement"),
                        item.get("falsification_rule"),
                    )
                    for item in hypothesis_entries
                    if isinstance(item, Mapping)
                ),
            ),
            "seed_confirmation": _table(
                (
                    "promotion_config_hash",
                    "status",
                    "observed_seeds_json",
                    "missing_seeds_json",
                    "mqar_mean",
                    "mqar_ci95_low",
                    "mqar_ci95_high",
                    "ppl_mean",
                    "ppl_ci95_low",
                    "ppl_ci95_high",
                ),
                (
                    (
                        item.get("promotion_config_hash"),
                        item.get("status"),
                        json.dumps(item.get("observed_seeds", [])),
                        json.dumps(item.get("missing_seeds", [])),
                        (
                            item.get("mqar_accuracy", {}).get("mean")
                            if isinstance(item.get("mqar_accuracy"), Mapping)
                            else None
                        ),
                        (
                            item.get("mqar_accuracy", {}).get("ci95_low")
                            if isinstance(item.get("mqar_accuracy"), Mapping)
                            else None
                        ),
                        (
                            item.get("mqar_accuracy", {}).get("ci95_high")
                            if isinstance(item.get("mqar_accuracy"), Mapping)
                            else None
                        ),
                        (
                            item.get("wikitext_ppl", {}).get("mean")
                            if isinstance(item.get("wikitext_ppl"), Mapping)
                            else None
                        ),
                        (
                            item.get("wikitext_ppl", {}).get("ci95_low")
                            if isinstance(item.get("wikitext_ppl"), Mapping)
                            else None
                        ),
                        (
                            item.get("wikitext_ppl", {}).get("ci95_high")
                            if isinstance(item.get("wikitext_ppl"), Mapping)
                            else None
                        ),
                    )
                    for item in seeded_configurations
                    if isinstance(item, Mapping)
                ),
            ),
            "confirmed_promotion_shortlist": _table(
                (
                    "promotion_rank",
                    "promotion_config_hash",
                    "mqar_accuracy_mean",
                    "wikitext_ppl_mean",
                    "selection_reason",
                ),
                (
                    (
                        item.get("promotion_rank"),
                        item.get("promotion_config_hash"),
                        item.get("mqar_accuracy"),
                        item.get("wikitext_ppl"),
                        item.get("selection_reason"),
                    )
                    for item in confirmed_selected
                    if isinstance(item, Mapping)
                ),
            ),
            "retrospective_control_false_pruning_risk": _table(
                (
                    "analysis_scope",
                    "control_name",
                    "architecture_modes_json",
                    "candidate_step_500_median",
                    "control_step_500_median",
                    "step_500_observation_count",
                    "below_candidate_median_count",
                    "retrospective_risk_label",
                ),
                (
                    (
                        "retrospective_diagnostic_only",
                        item.get("control_name"),
                        json.dumps(item.get("architecture_modes", [])),
                        candidate_step_500_median,
                        item.get("step_500_mqar_accuracy_median"),
                        item.get("step_500_observation_count"),
                        item.get("below_candidate_median_count"),
                        item.get("retrospective_risk_label"),
                    )
                    for item in control_risk_entries
                    if isinstance(item, Mapping)
                ),
            ),
        },
        "recommended_panels": [
            {
                "title": "MQAR accuracy vs. WikiText perplexity",
                "type": "scatter",
                "table": "pareto_front",
                "x": "wikitext_ppl",
                "y": "mqar_accuracy",
            },
            {
                "title": "Conditional fANOVA importance",
                "type": "bar",
                "table": "fanova_importance",
                "x": "parameter",
                "y": "importance",
                "facet": ["view", "metric"],
            },
            {
                "title": "Phase 2b promotion shortlist",
                "type": "table",
                "table": "promotion_shortlist",
            },
            {
                "title": "Falsified hypotheses review",
                "type": "table",
                "table": "falsified_hypotheses",
            },
            {
                "title": "Three-seed confirmation",
                "type": "table",
                "table": "seed_confirmation",
            },
            {
                "title": "Confirmed Phase 2b promotion shortlist",
                "type": "table",
                "table": "confirmed_promotion_shortlist",
            },
            {
                "title": (
                    "RETROSPECTIVE ONLY: fixed-control false-pruning risk"
                ),
                "type": "table",
                "table": "retrospective_control_false_pruning_risk",
            },
        ],
    }


def merge_record_metadata(
    records: Iterable[Mapping[str, Any]],
    metadata_records: Iterable[Mapping[str, Any]],
    *,
    require_exact_optuna_complete_set: bool = False,
) -> list[dict[str, Any]]:
    """Enrich result summaries with Optuna parameters without replacing evidence."""

    by_hash: dict[str, Mapping[str, Any]] = {}
    for metadata in metadata_records:
        trial_hash = metadata.get("trial_hash")
        if isinstance(trial_hash, str):
            if trial_hash in by_hash:
                raise ValueError(
                    f"Duplicate Optuna metadata for trial hash {trial_hash}"
                )
            by_hash[trial_hash] = metadata

    materialized_records = [dict(record) for record in records]
    if require_exact_optuna_complete_set:
        result_hashes = {
            str(record.get("trial_hash"))
            for record in materialized_records
            if record.get("claim_kind") == "optuna_study"
            and record.get("status") == "COMPLETE"
        }
        metadata_hashes = set(by_hash)
        if result_hashes != metadata_hashes:
            raise ValueError(
                "Claim-bound COMPLETE Optuna hashes do not exactly match "
                f"storage: missing_results={sorted(metadata_hashes - result_hashes)}, "
                f"missing_storage={sorted(result_hashes - metadata_hashes)}"
            )

    merged: list[dict[str, Any]] = []
    for source in materialized_records:
        record = dict(source)
        match = None
        trial_hash = record.get("trial_hash")
        if isinstance(trial_hash, str):
            match = by_hash.get(trial_hash)
        if match is not None:
            for name in ("promotion_config_hash", "seed", "study_identity"):
                left = record.get(name)
                right = match.get(name)
                if left is not None and right is not None and left != right:
                    raise ValueError(
                        f"Result/Optuna {name} mismatch for trial {trial_hash}"
                    )
            if not isinstance(record.get("parameters"), Mapping):
                parameters = match.get("parameters")
                if isinstance(parameters, Mapping):
                    record["parameters"] = dict(parameters)
            elif isinstance(match.get("parameters"), Mapping) and dict(
                record["parameters"]
            ) != dict(match["parameters"]):
                raise ValueError(
                    f"Result/Optuna parameter mismatch for trial {trial_hash}"
                )
            if (
                require_exact_optuna_complete_set
                and record.get("claim_kind") == "optuna_study"
                and record.get("status") == "COMPLETE"
            ):
                for name in (
                    "mqar_accuracy",
                    "wikitext_ppl",
                    "step_500_mqar_accuracy",
                    "wandb_run_id",
                    "health_passed",
                ):
                    left = record.get(name)
                    right = match.get(name)
                    if left != right:
                        raise ValueError(
                            f"Result/Optuna {name} mismatch for trial {trial_hash}: "
                            f"{left!r} != {right!r}"
                        )
                accounting = record.get("model_accounting")
                counts = (
                    accounting.get("parameter_counts_actual")
                    if isinstance(accounting, Mapping)
                    else None
                )
                flops = (
                    accounting.get("flops")
                    if isinstance(accounting, Mapping)
                    else None
                )
                expected_covariates = {
                    "actual_non_embedding_core_params": (
                        counts.get("non_embedding_core")
                        if isinstance(counts, Mapping)
                        else None
                    ),
                    "actual_total_params": (
                        counts.get("total")
                        if isinstance(counts, Mapping)
                        else None
                    ),
                    "estimated_flops_per_optimizer_step": (
                        flops.get("estimated")
                        if isinstance(flops, Mapping)
                        else None
                    ),
                    "measured_flops_per_optimizer_step": (
                        flops.get("measured")
                        if isinstance(flops, Mapping)
                        else None
                    ),
                    "gpu_hours_actual": record.get("gpu_hours_actual"),
                }
                observed_covariates = match.get("capacity_covariates")
                if (
                    not isinstance(observed_covariates, Mapping)
                    or dict(observed_covariates) != expected_covariates
                ):
                    raise ValueError(
                        "Result/Optuna capacity covariates mismatch for "
                        f"trial {trial_hash}"
                    )
            for name in (*_STABILITY_FIELDS, "health", "stability"):
                if name not in record and name in match:
                    record[name] = match[name]
            for name in (
                "wikitext_ppl",
                "step_500_mqar_accuracy",
                "wandb_run_id",
                "promotion_config_hash",
                "seed",
                "study_identity",
                "capacity_covariates",
            ):
                if record.get(name) is None and match.get(name) is not None:
                    record[name] = match[name]
        merged.append(record)
    return merged


def _manifest_study_identity(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Derive the immutable study identity shared by results and Optuna."""

    code = manifest.get("code")
    if not isinstance(code, Mapping):
        raise ValueError("Trial manifest is missing code identity")
    return {
        "study_name": manifest.get("study_name"),
        "spec_hash": manifest.get("spec_hash"),
        "code_commit": code.get("commit"),
        "code_dirty": code.get("dirty"),
        "data_hash": stable_hash(manifest.get("data")),
        "capability_hash": stable_hash(manifest.get("capability_identity")),
    }


_STUDY_IDENTITY_FIELDS = (
    "study_name",
    "spec_hash",
    "code_commit",
    "code_dirty",
    "data_hash",
    "capability_hash",
)


def _validated_study_identity(value: Any, *, source: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{source} has no bound study identity")
    identity = {name: value.get(name) for name in _STUDY_IDENTITY_FIELDS}
    if not isinstance(identity["study_name"], str) or not identity["study_name"]:
        raise ValueError(f"{source} has an invalid study_name")
    for name in ("spec_hash", "data_hash", "capability_hash"):
        if not _is_sha256(identity[name]):
            raise ValueError(f"{source} has an invalid {name}")
    if (
        not isinstance(identity["code_commit"], str)
        or len(identity["code_commit"]) < 7
    ):
        raise ValueError(f"{source} has an invalid code_commit")
    if identity["code_dirty"] is not False:
        raise ValueError(f"{source} is not bound to a clean integration commit")
    return identity


def _study_identity_from_optuna(study: Any) -> dict[str, Any]:
    contract = study.system_attrs.get("echo_phase2_contract_v1")
    return _validated_study_identity(contract, source="Optuna study")


def resolve_study_identity(
    records: Sequence[Mapping[str, Any]], study: Any | None = None
) -> dict[str, Any]:
    """Require every analyzed record and optional Optuna study to match."""

    identities: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        identities.append(
            _validated_study_identity(
                record.get("study_identity"),
                source=f"result record {index}",
            )
        )
    if study is not None:
        identities.append(_study_identity_from_optuna(study))
    if not identities:
        raise ValueError("No study identity is available for analysis")
    expected = identities[0]
    for identity in identities[1:]:
        if identity != expected:
            raise ValueError(
                "Analysis inputs span multiple study identities: "
                f"{expected!r} != {identity!r}"
            )
    return expected


def _retain_sibling_step_500_mqar(
    summary: dict[str, Any],
    *,
    summary_path: Path,
    trial_hash: str,
) -> None:
    """Retain the immutable screen metric alongside a terminal summary."""

    declared = summary.get("step_500_mqar_accuracy")
    if declared is not None and (
        not _finite_number(declared) or not 0.0 <= float(declared) <= 1.0
    ):
        raise ValueError(
            f"Invalid step_500_mqar_accuracy in trial summary {summary_path}"
        )

    step_path = summary_path.parent / "metrics" / "step_500.json"
    if not step_path.is_file():
        return
    try:
        step_metrics = json.loads(step_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid step-500 metrics {step_path}: {exc}") from exc
    if not isinstance(step_metrics, Mapping):
        raise ValueError(f"Step-500 metrics must be an object: {step_path}")
    if (
        step_metrics.get("schema_version") != 1
        or step_metrics.get("trial_hash") != trial_hash
        or step_metrics.get("status") not in {"ok", "failed"}
    ):
        raise ValueError(
            f"Step-500 metrics identity/status mismatch: {step_path}"
        )
    if step_metrics.get("status") == "failed":
        if declared is not None:
            raise ValueError(
                f"Failed step-500 metrics cannot support a summary MQAR: {step_path}"
            )
        return
    if step_metrics.get("optimizer_step") != 500:
        raise ValueError(f"Step-500 optimizer-step mismatch: {step_path}")
    observed = step_metrics.get("mqar_accuracy")
    if not _finite_number(observed) or not 0.0 <= float(observed) <= 1.0:
        raise ValueError(f"Invalid step-500 MQAR accuracy: {step_path}")
    observed_value = float(observed)
    if declared is not None and not math.isclose(
        float(declared),
        observed_value,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError(
            f"Summary/step-500 MQAR mismatch: {summary_path}"
        )
    summary["step_500_mqar_accuracy"] = observed_value


def _file_identity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        before = os.fstat(handle.fileno())
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
        after = os.fstat(handle.fileno())
    current = path.stat()
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    current_identity = (
        current.st_dev,
        current.st_ino,
        current.st_size,
        current.st_mtime_ns,
    )
    if before_identity != after_identity or before_identity != current_identity:
        raise ValueError(f"Artifact changed while hashing: {path}")
    return {
        "byte_size": int(before.st_size),
        "sha256": digest.hexdigest(),
    }


def _validate_terminal_artifact_bindings(
    claim: Mapping[str, Any],
    *,
    run_dir: Path,
    outcome: str,
) -> None:
    claimed_run_dir = claim.get("run_dir")
    if (
        not isinstance(claimed_run_dir, str)
        or Path(claimed_run_dir).expanduser().resolve() != run_dir.resolve()
    ):
        raise ValueError(
            f"Terminal claim run_dir does not match result directory: {run_dir}"
        )
    bindings = claim.get("terminal_artifacts")
    if not isinstance(bindings, Mapping):
        raise ValueError("Terminal claim has no immutable artifact bindings")
    required = {"trial_manifest.json", "trial_summary.json"}
    if outcome in {"COMPLETE", "PRUNED"}:
        required.add("metrics/step_500.json")
        required.add("checkpoint_bundle")
    if outcome == "COMPLETE":
        required.add("metrics/final.json")
    if not required.issubset(bindings):
        raise ValueError(
            f"Terminal claim is missing artifact bindings: "
            f"{sorted(required - set(bindings))}"
        )
    allowed = {
        "trial_manifest.json",
        "trial_summary.json",
        "metrics/step_500.json",
        "metrics/final.json",
        "checkpoint_bundle",
    }
    if not set(bindings).issubset(allowed):
        raise ValueError("Terminal claim contains an unknown artifact binding")
    for relative, expected in bindings.items():
        if relative == "checkpoint_bundle":
            if not isinstance(expected, Mapping):
                raise ValueError("Checkpoint bundle binding must be an object")
            try:
                actual = checkpoint_bundle_identity(
                    run_dir,
                    expected.get("path"),
                )
            except ValueError as exc:
                raise ValueError(str(exc)) from exc
            if dict(expected) != actual:
                raise ValueError(
                    "Terminal checkpoint bundle identity mismatch"
                )
            continue
        path = (run_dir / str(relative)).resolve()
        try:
            path.relative_to(run_dir.resolve())
        except ValueError as exc:
            raise ValueError(
                f"Terminal artifact escapes its run directory: {relative}"
            ) from exc
        if not path.is_file() or not isinstance(expected, Mapping):
            raise ValueError(f"Missing terminal artifact: {path}")
        actual = _file_identity(path)
        if dict(expected) != actual:
            raise ValueError(
                f"Terminal artifact identity mismatch: {path}"
            )


def _validate_summary_against_metrics(
    summary: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    run_dir: Path,
) -> None:
    status = summary.get("status")
    if status == "FAIL":
        return
    screen_path = run_dir / "metrics" / "step_500.json"
    try:
        screen = json.loads(screen_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid bound screen metrics {screen_path}: {exc}") from exc
    if not isinstance(screen, dict):
        raise ValueError(f"Screen metrics must be an object: {screen_path}")
    validate_step_metrics(
        manifest,
        screen,
        expected_step=int(manifest["fidelity"]["prune_step"]),
        require_full_mqar_grid=True,
        require_wikitext_ppl=False,
        diagnostic_names=manifest["screen_required_diagnostics"],
    )
    final: Mapping[str, Any] = screen
    if status == "COMPLETE":
        final_path = run_dir / "metrics" / "final.json"
        try:
            parsed_final = json.loads(final_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Invalid bound final metrics {final_path}: {exc}"
            ) from exc
        if not isinstance(parsed_final, dict):
            raise ValueError(f"Final metrics must be an object: {final_path}")
        validate_step_metrics(
            manifest,
            parsed_final,
            expected_step=int(manifest["fidelity"]["max_steps"]),
            require_full_mqar_grid=True,
            require_wikitext_ppl=True,
        )
        validate_cross_stage_identity(screen, parsed_final)
        final = parsed_final
    expected_fields = {
        "mqar_accuracy": final.get("mqar_accuracy"),
        "wikitext_ppl": final.get("wikitext_ppl"),
        "wandb_run_id": final.get("wandb_run_id"),
        "model_accounting": final.get("model_accounting"),
        "optimizer_group_audit": final.get("optimizer_group_audit"),
        "step_500_mqar_accuracy": screen.get("mqar_accuracy"),
        "health_passed": True,
    }
    for field, expected in expected_fields.items():
        if summary.get(field) != expected:
            raise ValueError(
                f"Summary/bound-metrics {field} mismatch: {run_dir}"
            )


def load_result_summaries(
    root: str | Path,
    *,
    require_terminal_claims: bool = False,
    claims_root: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Load one immutable trial summary per hash from a Phase 2 run tree."""

    directory = Path(root).expanduser().resolve()
    if not directory.is_dir():
        raise ValueError(f"Results root is not a directory: {directory}")
    claim_directory = (
        Path(claims_root).expanduser().resolve()
        if claims_root is not None
        else directory
    ) / "claims"
    if require_terminal_claims and not claim_directory.is_dir():
        raise ValueError(
            f"Scientific analysis requires a claim ledger: {claim_directory}"
        )
    records: list[dict[str, Any]] = []
    seen: dict[str, Path] = {}
    for path in sorted(directory.rglob("trial_summary.json")):
        try:
            value = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid trial summary {path}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"Trial summary must be a JSON object: {path}")
        trial_hash = value.get("trial_hash")
        if not isinstance(trial_hash, str):
            raise ValueError(f"Trial summary has no trial_hash: {path}")
        if trial_hash in seen:
            raise ValueError(
                f"Duplicate trial summary for {trial_hash}: "
                f"{seen[trial_hash]} and {path}"
            )
        seen[trial_hash] = path
        manifest_path = path.parent / "trial_manifest.json"
        if require_terminal_claims and not manifest_path.is_file():
            raise ValueError(
                f"Scientific analysis requires a sibling manifest: {manifest_path}"
            )
        manifest: dict[str, Any] | None = None
        if manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"Invalid sibling trial manifest {manifest_path}: {exc}"
                ) from exc
            if (
                not isinstance(manifest, dict)
                or manifest.get("trial_hash") != trial_hash
                or recompute_trial_hash(manifest) != trial_hash
            ):
                raise ValueError(
                    f"Sibling trial manifest hash mismatch: {manifest_path}"
                )
            promotion_hash = recompute_promotion_config_hash(manifest)
            if manifest.get("promotion_config_hash") != promotion_hash:
                raise ValueError(
                    f"Sibling promotion config hash mismatch: {manifest_path}"
                )
            parameters = manifest.get("parameters")
            if isinstance(parameters, Mapping):
                if isinstance(value.get("parameters"), Mapping) and dict(
                    value["parameters"]
                ) != dict(parameters):
                    raise ValueError(
                        f"Summary/manifest parameters mismatch: {manifest_path}"
                    )
                value["parameters"] = dict(parameters)
            authoritative = {
                "promotion_config_hash": promotion_hash,
                "seed": manifest.get("protocol", {}).get("seed"),
                "study_identity": _manifest_study_identity(manifest),
            }
            for name, expected in authoritative.items():
                if value.get(name) is not None and value.get(name) != expected:
                    raise ValueError(
                        f"Summary/manifest {name} mismatch: {manifest_path}"
                    )
                value[name] = expected
        _retain_sibling_step_500_mqar(
            value,
            summary_path=path,
            trial_hash=trial_hash,
        )
        if require_terminal_claims:
            claim_path = claim_directory / f"{trial_hash}.json"
            try:
                claim = json.loads(claim_path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"Missing/invalid terminal claim {claim_path}: {exc}"
                ) from exc
            expected_outcome = {
                "COMPLETE": "COMPLETE",
                "PRUNED": "PRUNED",
                "FAIL": "FAIL",
            }.get(value.get("status"))
            if (
                not isinstance(claim, Mapping)
                or claim.get("trial_hash") != trial_hash
                or claim.get("state") != "terminal"
                or claim.get("outcome") != expected_outcome
            ):
                raise ValueError(
                    f"Summary has no matching terminal claim: {path}"
                )
            claim_kind = claim.get("claim_kind")
            if claim_kind not in {"optuna_study", "fixed_manifest"}:
                raise ValueError(
                    f"Terminal claim has invalid claim_kind: {claim_path}"
                )
            value["claim_kind"] = claim_kind
            summary_hours = value.get("gpu_hours_actual")
            claim_hours = claim.get("gpu_hours_actual")
            if (
                not _finite_number(summary_hours)
                or not _finite_number(claim_hours)
                or not math.isclose(
                    float(summary_hours),
                    float(claim_hours),
                    rel_tol=1e-9,
                    abs_tol=1e-12,
                )
            ):
                raise ValueError(
                    f"Summary/claim GPU accounting mismatch: {path}"
                )
            if claim.get("compute_budget_violation") is not None:
                raise ValueError(
                    f"Compute-budget-violating study cannot be analyzed: {claim_path}"
                )
            _validate_terminal_artifact_bindings(
                claim,
                run_dir=path.parent,
                outcome=str(expected_outcome),
            )
            if manifest is None:
                raise ValueError(
                    f"Scientific analysis has no parsed manifest: {manifest_path}"
                )
            try:
                _validate_summary_against_metrics(
                    value,
                    manifest,
                    run_dir=path.parent,
                )
            except Exception as exc:
                raise ValueError(
                    f"Summary/metrics validation failed for {path}: {exc}"
                ) from exc
        records.append(value)
    return records


def _records_from_study(study: Any) -> list[dict[str, Any]]:
    try:
        from optuna.trial import TrialState
    except ImportError as exc:
        raise RuntimeError("Install the optional 'nas' dependencies") from exc
    study_identity = _study_identity_from_optuna(study)
    records: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    for trial in study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,)):
        trial_hash = trial.user_attrs.get("trial_hash")
        if not _is_sha256(trial_hash):
            raise ValueError(
                f"Optuna COMPLETE trial {trial.number} has no valid trial_hash"
            )
        if trial_hash in seen_hashes:
            raise ValueError(
                f"Optuna contains duplicate COMPLETE trial_hash {trial_hash}"
            )
        seen_hashes.add(trial_hash)
        materialized_parameters = trial.user_attrs.get(
            "materialized_parameters"
        )
        parameters = (
            dict(materialized_parameters)
            if isinstance(materialized_parameters, Mapping)
            else {
                "architecture_mode": "updated_sweep",
                **dict(trial.params),
            }
        )
        intermediate_values = getattr(trial, "intermediate_values", {})
        step_500_mqar = (
            intermediate_values.get(500)
            if isinstance(intermediate_values, Mapping)
            else None
        )
        if step_500_mqar is not None and (
            not _finite_number(step_500_mqar)
            or not 0.0 <= float(step_500_mqar) <= 1.0
        ):
            raise ValueError(
                f"Optuna trial {trial.number} has invalid step-500 MQAR"
            )
        record = {
            "schema_version": 1,
            "trial_hash": trial_hash,
            "promotion_config_hash": trial.user_attrs.get(
                "promotion_config_hash"
            ),
            "seed": trial.user_attrs.get("seed"),
            "status": "COMPLETE",
            "mqar_accuracy": trial.value,
            "step_500_mqar_accuracy": (
                float(step_500_mqar) if step_500_mqar is not None else None
            ),
            "wikitext_ppl": trial.user_attrs.get("wikitext_ppl"),
            "wandb_run_id": trial.user_attrs.get("wandb_run_id"),
            "optuna_trial_id": trial.number,
            "parameters": parameters,
            "study_identity": dict(study_identity),
            "capacity_covariates": {
                "actual_non_embedding_core_params": trial.user_attrs.get(
                    "actual_non_embedding_core_params"
                ),
                "actual_total_params": trial.user_attrs.get(
                    "actual_total_params"
                ),
                "estimated_flops_per_optimizer_step": trial.user_attrs.get(
                    "estimated_flops_per_optimizer_step"
                ),
                "measured_flops_per_optimizer_step": trial.user_attrs.get(
                    "measured_flops_per_optimizer_step"
                ),
                "gpu_hours_actual": trial.user_attrs.get("gpu_hours_actual"),
            },
        }
        for name in (*_STABILITY_FIELDS, "health", "stability"):
            if name in trial.user_attrs:
                record[name] = trial.user_attrs[name]
        records.append(record)
    return records


def validate_optuna_terminal_claim_alignment(
    records: Sequence[Mapping[str, Any]],
    study: Any,
) -> None:
    """Require every scientific Optuna row to resolve to one terminal claim."""

    try:
        from optuna.trial import TrialState
    except ImportError as exc:
        raise RuntimeError("Install the optional 'nas' dependencies") from exc
    claims = {
        str(record.get("trial_hash")): str(record.get("status"))
        for record in records
        if record.get("claim_kind") == "optuna_study"
    }
    states_by_hash: dict[str, set[str]] = {}
    for trial in study.get_trials(deepcopy=False):
        state = getattr(trial, "state", None)
        state_name = str(getattr(state, "name", "UNKNOWN"))
        if state not in {
            TrialState.COMPLETE,
            TrialState.PRUNED,
            TrialState.FAIL,
        }:
            raise ValueError(
                f"Optuna trial {trial.number} is not terminal: {state_name}"
            )
        if trial.user_attrs.get("duplicate_trial_hash") is True:
            continue
        trial_hash = trial.user_attrs.get("trial_hash")
        if not _is_sha256(trial_hash):
            raise ValueError(
                f"Optuna terminal trial {trial.number} has invalid trial_hash"
            )
        if trial_hash not in claims:
            raise ValueError(
                f"Optuna terminal trial {trial.number}/{trial_hash} has no "
                "claim-bound terminal summary"
            )
        states_by_hash.setdefault(str(trial_hash), set()).add(state_name)
    expected_state = {
        "COMPLETE": "COMPLETE",
        "PRUNED": "PRUNED",
        "FAIL": "FAIL",
    }
    for trial_hash, outcome in claims.items():
        required = expected_state.get(outcome)
        if required is None or required not in states_by_hash.get(trial_hash, set()):
            raise ValueError(
                f"Claim-bound Optuna outcome {outcome} for {trial_hash} has "
                "no matching terminal storage row"
            )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        required=True,
        help=(
            "Recursively collect manifest-bound trial summaries from a run "
            "tree and require matching terminal claims."
        ),
    )
    parser.add_argument(
        "--claims-root",
        help=(
            "Shared budget root containing claims/; defaults to --results-root. "
            "Scientific directory analysis requires terminal claim matches."
        ),
    )
    parser.add_argument(
        "--storage",
        required=True,
        help="Frozen Optuna storage URL used to calculate mandatory fANOVA",
    )
    parser.add_argument(
        "--study-name",
        required=True,
        help="Exact frozen Optuna study name",
    )
    parser.add_argument("--shortlist-cap", type=int, default=10)
    stability = parser.add_mutually_exclusive_group()
    stability.add_argument(
        "--require-stability-evidence",
        dest="require_stability_evidence",
        action="store_true",
        default=True,
        help="Require an explicit passing health marker (default)",
    )
    stability.add_argument(
        "--allow-missing-stability-evidence",
        dest="require_stability_evidence",
        action="store_false",
        help="Exploratory only: retain records with no health marker",
    )
    parser.add_argument(
        "--artifact-dir",
        help="Optionally write each requested study artifact as a separate JSON file.",
    )
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.shortlist_cap < 1:
        raise SystemExit("--shortlist-cap must be at least one")

    records: list[dict[str, Any]] = []
    study = None
    try:
        records = load_result_summaries(
            args.results_root,
            require_terminal_claims=True,
            claims_root=args.claims_root,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if not records:
        raise SystemExit(
            "No terminal claim-bound trial summaries found under --results-root"
        )
    try:
        import optuna
    except ImportError as exc:
        raise SystemExit("Install optional nas dependencies for fANOVA") from exc
    try:
        from .run_study import build_storage
    except ImportError:
        from run_study import build_storage  # type: ignore
    if args.storage.startswith("postgresql+psycopg://"):
        approved_backend = "postgresql"
    elif args.storage.startswith("journal://"):
        approved_backend = "scg_validated_journal_storage"
    else:
        raise SystemExit(
            "Analysis storage must use postgresql+psycopg:// or journal://"
        )
    storage = build_storage(
        args.storage,
        approved_backend=approved_backend,
    )
    study = optuna.load_study(study_name=args.study_name, storage=storage)
    try:
        validate_optuna_terminal_claim_alignment(records, study)
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    study_records = _records_from_study(study)
    records = merge_record_metadata(
        records,
        study_records,
        require_exact_optuna_complete_set=True,
    )
    try:
        validate_wandb_trial_mapping(records)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    try:
        study_identity = resolve_study_identity(records, study)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    trial_hash_index = sorted(
        {
            str(record.get("trial_hash"))
            for record in records
            if _is_sha256(record.get("trial_hash"))
        }
    )
    if len(trial_hash_index) != len(records):
        raise SystemExit(
            "Every scientific result must have one unique lowercase SHA-256 "
            "trial_hash before final analysis"
        )

    front = pareto_front(records)
    shortlist = promotion_shortlist(
        records,
        cap=args.shortlist_cap,
        require_stability_evidence=bool(args.require_stability_evidence),
    )
    fanova = fanova_importance(study)
    hypotheses = falsified_hypotheses_report(records)
    control_pruning_risk = retrospective_control_false_pruning_risk(records)
    seeded = seeded_stability_summary(
        records,
        required_seeds=(42, 43, 44),
    )
    confirmed_shortlist = promotion_shortlist(
        seeded["confirmed_records"],
        cap=args.shortlist_cap,
        require_stability_evidence=True,
    )
    shortlist["study_identity"] = dict(study_identity)
    seeded["study_identity"] = dict(study_identity)
    confirmed_shortlist["study_identity"] = dict(study_identity)
    control_pruning_risk["study_identity"] = dict(study_identity)
    hypotheses["study_identity"] = dict(study_identity)
    if isinstance(fanova, dict):
        fanova["study_identity"] = dict(study_identity)
    output: dict[str, Any] = {
        "schema_version": 1,
        "study_identity": study_identity,
        "trial_hash_index": trial_hash_index,
        "study_semantics": {
            "optuna_controller_objective": "maximize_mqar_accuracy",
            "final_selection": "posthoc_mqar_max_wikitext_ppl_min_pareto",
            "sampling_warning": CONTROLLER_WARNING,
        },
        "pareto_front": front,
        "promotion_shortlist": shortlist,
        "seeded_stability": seeded,
        "confirmed_promotion_shortlist": confirmed_shortlist,
        "retrospective_control_false_pruning_risk": control_pruning_risk,
        "fanova_importance": fanova,
        "falsified_hypotheses": hypotheses,
    }
    output["wandb_report_payload"] = build_wandb_report_payload(
        front,
        shortlist,
        fanova,
        hypotheses,
        seeded,
        confirmed_shortlist,
        study_identity,
        control_pruning_risk,
    )
    atomic_write_json(args.output, output)
    if args.artifact_dir:
        artifact_dir = Path(args.artifact_dir).expanduser().resolve()
        artifacts = {
            "analysis_report.json": output,
            "pareto_front.json": front,
            "fanova_importance.json": fanova,
            "promotion_shortlist.json": shortlist,
            "seeded_stability.json": seeded,
            "confirmed_promotion_shortlist.json": confirmed_shortlist,
            "retrospective_control_false_pruning_risk.json": (
                control_pruning_risk
            ),
            "falsified_hypotheses.json": hypotheses,
            "wandb_report_payload.json": output["wandb_report_payload"],
        }
        for filename, value in artifacts.items():
            atomic_write_json(artifact_dir / filename, value)
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
