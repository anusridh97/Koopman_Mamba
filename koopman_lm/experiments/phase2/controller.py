"""Optuna controller helpers for step-500 MQAR median pruning."""

from __future__ import annotations

import statistics
from typing import Any, Iterable, Mapping


_STUDY_CONTRACT_KEY = "echo_phase2_contract_v1"


def should_prune_at_median(
    current_mqar: float,
    completed_mqar: Iterable[float],
    *,
    minimum_completed_trials: int,
) -> bool:
    """Apply the literal below-median rule after enough completed trials."""

    completed = [float(value) for value in completed_mqar]
    if len(completed) < minimum_completed_trials:
        return False
    return float(current_mqar) < statistics.median(completed)


def create_optuna_study(
    *,
    study_name: str,
    storage: Any,
    seed: int,
    minimum_completed_trials: int,
    load_if_exists: bool = True,
) -> Any:
    """Create the single-objective study used for valid native pruning.

    WikiText perplexity is stored as a trial user attribute and analyzed
    post-hoc; using a native multiobjective study would disable Trial.report /
    MedianPruner semantics in supported Optuna versions.
    """

    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError("Install the optional 'nas' dependencies for Optuna") from exc
    return optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(
            seed=seed,
            multivariate=True,
            group=True,
        ),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=minimum_completed_trials,
            n_warmup_steps=500,
            interval_steps=500,
        ),
        load_if_exists=load_if_exists,
    )


def bind_study_contract(study: Any, identity: Mapping[str, Any]) -> None:
    """Bind shared storage to one spec/code/data/capability identity."""

    expected = dict(identity)
    existing = study.system_attrs.get(_STUDY_CONTRACT_KEY)
    if existing is None:
        study.set_system_attr(_STUDY_CONTRACT_KEY, expected)
        existing = study.system_attrs.get(_STUDY_CONTRACT_KEY)
    if existing != expected:
        raise RuntimeError(
            "Optuna study contract mismatch; create a new versioned study name. "
            f"stored={existing!r}, requested={expected!r}"
        )


def report_step_500(trial: Any, metrics: Mapping[str, Any]) -> None:
    """Validate and report the fixed MQAR screening metric to Optuna."""

    if int(metrics.get("optimizer_step", -1)) != 500:
        raise ValueError("Pruning report must come from optimizer step 500")
    if metrics.get("status") != "ok":
        raise ValueError("Failed/numerically invalid trials must not be reported as scores")
    mqar = metrics.get("mqar_accuracy")
    if not isinstance(mqar, (int, float)) or not 0.0 <= float(mqar) <= 1.0:
        raise ValueError(f"Invalid MQAR accuracy: {mqar!r}")
    trial.report(float(mqar), step=500)


__all__ = [
    "bind_study_contract",
    "create_optuna_study",
    "report_step_500",
    "should_prune_at_median",
]
