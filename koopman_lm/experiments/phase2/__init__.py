"""Portable Phase 2a search contracts and orchestration helpers.

These modules intentionally avoid importing model/training code. They remain
usable while the canonical SKA architecture is being integrated.
"""

from .spec import (
    Phase2SpecError,
    derive_objective_weights,
    estimate_parameter_counts,
    load_spec,
    placement_indices,
    round_ska_count,
    sample_parameters,
    suggest_parameters,
    validate_spec,
    validate_trial_parameters,
)
from .manifest import (
    build_trial_manifest,
    recompute_promotion_config_hash,
    recompute_trial_hash,
    stable_hash,
)
from .results import (
    Phase2ResultError,
    expected_mqar_cells,
    validate_cross_stage_identity,
    validate_step_metrics,
)

__all__ = [
    "Phase2SpecError",
    "Phase2ResultError",
    "build_trial_manifest",
    "derive_objective_weights",
    "estimate_parameter_counts",
    "expected_mqar_cells",
    "load_spec",
    "placement_indices",
    "recompute_promotion_config_hash",
    "recompute_trial_hash",
    "round_ska_count",
    "sample_parameters",
    "stable_hash",
    "suggest_parameters",
    "validate_cross_stage_identity",
    "validate_spec",
    "validate_step_metrics",
    "validate_trial_parameters",
]
