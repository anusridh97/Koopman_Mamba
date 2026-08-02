"""Standard-library-only tests for Phase 2 diagnostic result semantics."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

try:
    import pytest
except ImportError:  # Keeps `python test_*.py` dependency-free.
    pytest = None

if pytest is not None:
    pytestmark = pytest.mark.correctness


_RESULTS_PATH = (
    Path(__file__).resolve().parents[1]
    / "koopman_lm"
    / "experiments"
    / "phase2"
    / "results.py"
)
_MODULE_SPEC = importlib.util.spec_from_file_location(
    "phase2_results_contract_under_test", _RESULTS_PATH
)
if _MODULE_SPEC is None or _MODULE_SPEC.loader is None:
    raise RuntimeError(f"Cannot import Phase 2 results contract from {_RESULTS_PATH}")
_RESULTS = importlib.util.module_from_spec(_MODULE_SPEC)
_MODULE_SPEC.loader.exec_module(_RESULTS)

Phase2ResultError = _RESULTS.Phase2ResultError
expected_mqar_cells = _RESULTS.expected_mqar_cells
validate_step_metrics = _RESULTS.validate_step_metrics
validate_cross_stage_identity = _RESULTS.validate_cross_stage_identity
optimizer_coverage_digest = _RESULTS.optimizer_coverage_digest
optimizer_parameter_digest = _RESULTS.optimizer_parameter_digest


_REQUIRED_DIAGNOSTICS = [
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
]

_HEAD_METRICS = [
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
]

_LAYER_METRICS = [
    "residual_norm_ratio",
    "gradient_norm_ratio",
    "effective_jacobian_rank",
    "beta_mean",
    "beta_saturation_fraction",
    "layerscale_magnitude_mean",
]


def _head(
    *,
    raw_mean: float,
    raw_max: float,
    normalized_mean: float,
    normalized_max: float,
    applied_mean: float,
    applied_max: float,
    clamp_mean: float,
    clamp_fraction: float,
    gap_mean: float,
    gap_max: float,
    lambda_mean: float,
    lambda_min: float,
    condition_mean: float,
    condition_max: float,
    escalation_count: int = 0,
    effective_ridge_min: float = 0.001,
    effective_ridge_max: float = 0.001,
) -> dict[str, float]:
    return {
        "spectral_radius_raw_preclamp_mean": raw_mean,
        "spectral_radius_raw_preclamp_max": raw_max,
        "spectral_radius_normalized_pre_gamma_mean": normalized_mean,
        "spectral_radius_normalized_pre_gamma_max": normalized_max,
        "spectral_radius_applied_post_gamma_mean": applied_mean,
        "spectral_radius_applied_post_gamma_max": applied_max,
        "clamp_factor_mean": clamp_mean,
        "clamp_fraction": clamp_fraction,
        "spectral_gap_mean": gap_mean,
        "spectral_gap_max": gap_max,
        "lambda_min_gram_mean": lambda_mean,
        "lambda_min_gram_min": lambda_min,
        "ridge_condition_number_mean": condition_mean,
        "ridge_condition_number_max": condition_max,
        "ridge_escalation_count": escalation_count,
        "effective_ridge_min": effective_ridge_min,
        "effective_ridge_max": effective_ridge_max,
    }


def _manifest(mode: str = "updated_sweep") -> dict:
    indices = [] if mode in {"mamba_only", "transformer"} else [2, 5]
    return {
        "trial_hash": "a" * 64,
        "parameters": {"architecture_mode": mode},
        "protocol": {
            "effective_batch_sequences": 2,
            "sequence_length": 4,
            "weight_decay": 0.1,
        },
        "data": {
            "mqar_screening": {
                "seed": 42,
                "generator_revision": "test-mqar-v1",
                "samples_per_cell": 8,
            }
        },
        "fidelity": {
            "prune_step": 3,
            "max_steps": 6,
        },
        "model_config": {
            "d_model": 64,
            "ska_n_heads": 2,
            "ska_ridge": 0.001,
        },
        "model_extensions": {
            "realized_layout": {
                "ska_layer_indices_zero_based": indices,
            }
        },
        "parameter_counts_estimated": {
            "embedding": 2048,
            "non_embedding_core": 1000,
            "total": 3048,
        },
        "scale_check": {
            "label": "3m-total",
            "accounting": "total_trainable_parameters",
            "accepted_min": 2800,
            "accepted_max": 3250,
            "parameter_count_key": "total",
            "estimated_value": 3048,
            "estimated_within_band": True,
        },
        "optimizer": {
            "groups": {
                "backbone": {
                    "lr": 1e-5,
                    "owns": ["test parameters"],
                }
            }
        },
        "required_diagnostics": list(_REQUIRED_DIAGNOSTICS),
        "diagnostic_structure": {
            "per_layer_head_metrics": list(_HEAD_METRICS),
            "per_layer_metrics": list(_LAYER_METRICS),
        },
        "hard_failure_rules": {
            "normalized_pre_gamma_spectral_radius_max": 1.0001,
            "lambda_min_over_ridge_min": 0.999,
            "state_rebuild_decode_prefill_max_abs_error": 0.0001,
            "chunk_boundary_parallel_decode_max_abs_error": 0.0001,
            "parameter_estimator_relative_error_max": 0.05,
            "flop_estimator_relative_error_max": 0.15,
        },
    }


def _model_accounting() -> dict:
    return {
        "parameter_counts_actual": {
            "embedding": 2048,
            "non_embedding_core": 1000,
            "total": 3048,
        },
        "parameter_estimator": {
            "core_relative_error": 0.0,
            "total_relative_error": 0.0,
            "maximum_allowed_relative_error": 0.05,
            "outcome": "within_tolerance",
        },
        "flops": {
            "scope": "training_forward_backward_per_optimizer_step",
            "estimated": 1_000_000.0,
            "measured": 1_000_000.0,
            "measurement_steps": 3,
            "estimator_method": "analytic-test",
            "measurement_method": "profiler-test",
            "relative_error": 0.0,
            "maximum_allowed_relative_error": 0.15,
            "outcome": "within_tolerance",
        },
        "scale_band": {
            "label": "3m-total",
            "accounting": "total_trainable_parameters",
            "accepted_min": 2800,
            "accepted_max": 3250,
            "actual_value": 3048,
            "outcome": "within_band",
        },
    }


def _optimizer_group_audit() -> dict:
    parameters = [
        {"name": "backbone.weight", "numel": 1000},
        {"name": "embedding.weight", "numel": 2048},
    ]
    groups = [
        {
            "group_name": "backbone_decay",
            "manifest_group_name": "backbone",
            "lr": 1e-5,
            "weight_decay": 0.1,
            "parameters": parameters,
            "tensor_count": 2,
            "scalar_parameter_count": 3048,
            "parameter_digest": optimizer_parameter_digest(parameters),
        }
    ]
    return {
        "groups": groups,
        "coverage": {
            "trainable_parameter_tensor_count": 2,
            "assigned_parameter_tensor_count": 2,
            "trainable_scalar_parameter_count": 3048,
            "assigned_scalar_parameter_count": 3048,
            "unassigned_parameter_names": [],
            "duplicate_parameter_names": [],
            "complete": True,
            "exclusive": True,
        },
        "trainable_parameter_digest": optimizer_parameter_digest(parameters),
        "coverage_digest": optimizer_coverage_digest(groups),
    }


def _metrics() -> dict:
    diagnostics = {
        "spectral_radius_raw_preclamp": 1.2,
        "spectral_radius_normalized_pre_gamma": 0.9,
        "spectral_radius_applied_post_gamma": 1.3,
        "clamp_factor_mean": 0.85,
        "clamp_fraction": 0.25,
        "spectral_gap_mean": 0.15,
        "spectral_gap_max": 0.4,
        "lambda_min_gram": 0.001,
        "ridge_condition_number": 10.0,
        "ridge_escalation_count": 0,
        "ridge_requested": 0.001,
        "effective_ridge_min": 0.001,
        "effective_ridge_max": 0.001,
        "beta_mean": 0.3,
        "beta_saturation_fraction": 0.01,
        "layerscale_magnitude_mean": 0.0001,
        "ska_branch_norm_ratio": 0.2,
        "ska_gradient_to_mamba_ratio": 1.0,
        "effective_jacobian_rank": 9.0,
        "step_time_ms": 12.5,
        "peak_memory_bytes": 4096,
        "ppl_full": 100.0,
        "ppl_ska_zero": 101.0,
        "ppl_mamba_zero": 103.0,
        "ppl_both_zero": 104.0,
        "ska_zero_ppl_delta": 1.0,
        "state_rebuild_decode_prefill_max_abs_error": 0.00001,
        "chunk_boundary_parallel_decode_max_abs_error": 0.00001,
        "within_chunk_parallel_decode_drift_max_abs_error": 0.05,
    }
    by_layer = {
        "2": {
            "per_head": [
                _head(
                    raw_mean=0.5,
                    raw_max=0.8,
                    normalized_mean=0.4,
                    normalized_max=0.7,
                    applied_mean=0.6,
                    applied_max=0.9,
                    clamp_mean=0.9,
                    clamp_fraction=0.1,
                    gap_mean=0.1,
                    gap_max=0.2,
                    lambda_mean=0.0015,
                    lambda_min=0.0011,
                    condition_mean=4.0,
                    condition_max=5.0,
                ),
                _head(
                    raw_mean=0.6,
                    raw_max=1.1,
                    normalized_mean=0.5,
                    normalized_max=0.8,
                    applied_mean=0.7,
                    applied_max=1.2,
                    clamp_mean=0.8,
                    clamp_fraction=0.2,
                    gap_mean=0.2,
                    gap_max=0.3,
                    lambda_mean=0.0014,
                    lambda_min=0.001,
                    condition_mean=6.0,
                    condition_max=8.0,
                ),
            ],
            "layer": {
                "residual_norm_ratio": 0.1,
                "gradient_norm_ratio": 0.9,
                "effective_jacobian_rank": 8.0,
                "beta_mean": 0.2,
                "beta_saturation_fraction": 0.0,
                "layerscale_magnitude_mean": 0.00005,
            },
        },
        "5": {
            "per_head": [
                _head(
                    raw_mean=0.4,
                    raw_max=0.7,
                    normalized_mean=0.3,
                    normalized_max=0.6,
                    applied_mean=0.5,
                    applied_max=0.8,
                    clamp_mean=1.0,
                    clamp_fraction=0.3,
                    gap_mean=0.05,
                    gap_max=0.15,
                    lambda_mean=0.0018,
                    lambda_min=0.0012,
                    condition_mean=8.0,
                    condition_max=10.0,
                ),
                _head(
                    raw_mean=0.7,
                    raw_max=1.2,
                    normalized_mean=0.6,
                    normalized_max=0.9,
                    applied_mean=0.8,
                    applied_max=1.3,
                    clamp_mean=0.7,
                    clamp_fraction=0.4,
                    gap_mean=0.25,
                    gap_max=0.4,
                    lambda_mean=0.0016,
                    lambda_min=0.00105,
                    condition_mean=5.0,
                    condition_max=7.0,
                ),
            ],
            "layer": {
                "residual_norm_ratio": 0.3,
                "gradient_norm_ratio": 1.1,
                "effective_jacobian_rank": 10.0,
                "beta_mean": 0.4,
                "beta_saturation_fraction": 0.02,
                "layerscale_magnitude_mean": 0.00015,
            },
        },
    }
    return {
        "schema_version": 1,
        "trial_hash": "a" * 64,
        "status": "ok",
        "optimizer_step": 3,
        "tokens_seen": 24,
        "gpu_seconds_actual": 3600.0,
        "gpu_hours_actual": 1.0,
        "wandb_run_id": "test-run",
        "checkpoint_provenance": {
            "checkpoint_sha256": "c" * 64,
            "checkpoint_bundle_path": "checkpoints/step_500",
            "checkpoint_bundle_byte_size": 123,
            "checkpoint_bundle_file_count": 4,
            "checkpoint_digest_algorithm": (
                "sha256(path_nul_size_nul_content_sha256_newline_v1)"
            ),
            "optimizer_step": 3,
            "global_sequence_index": 6,
            "checkpoint_format_revision": "test-checkpoint-v1",
        },
        "mqar_accuracy": 0.5,
        "mqar_hard_accuracy": 0.5,
        "mqar_worst_cell_accuracy": 0.5,
        "mqar_worst_cell": "length_1024/pairs_32",
        "mqar_seed": 42,
        "mqar_generator_revision": "test-mqar-v1",
        "mqar_samples_per_cell": 8,
        "wikitext_ppl": 100.0,
        "mqar_grid": {cell: 0.5 for cell in expected_mqar_cells()},
        "model_accounting": _model_accounting(),
        "optimizer_group_audit": _optimizer_group_audit(),
        "diagnostics": diagnostics,
        "diagnostics_by_layer": by_layer,
        "not_applicable_diagnostics": [],
    }


def _validate(manifest: dict, metrics: dict) -> None:
    validate_step_metrics(manifest, metrics, expected_step=3)


class Phase2ResultsContractTests(unittest.TestCase):
    def test_valid_updated_result_has_consistent_aggregates(self) -> None:
        _validate(_manifest(), _metrics())

    def test_invalid_aggregate_domains_are_rejected(self) -> None:
        cases = {
            "negative_radius": ("spectral_radius_raw_preclamp", -0.1),
            "zero_clamp_factor": ("clamp_factor_mean", 0.0),
            "fraction_above_one": ("clamp_fraction", 1.1),
            "impossible_condition_number": ("ridge_condition_number", 0.9),
            "fractional_escalation_count": ("ridge_escalation_count", 0.5),
            "beta_above_one": ("beta_mean", 1.1),
            "negative_layerscale_magnitude": (
                "layerscale_magnitude_mean",
                -0.1,
            ),
            "negative_norm_ratio": ("ska_branch_norm_ratio", -0.1),
            "rank_above_d_model": ("effective_jacobian_rank", 65.0),
            "zero_step_time": ("step_time_ms", 0.0),
            "zero_memory": ("peak_memory_bytes", 0),
            "zero_perplexity": ("ppl_full", 0.0),
            "negative_parity_error": (
                "state_rebuild_decode_prefill_max_abs_error",
                -0.1,
            ),
        }
        for label, (name, value) in cases.items():
            with self.subTest(label=label):
                metrics = _metrics()
                metrics["diagnostics"][name] = value
                with self.assertRaises(Phase2ResultError):
                    _validate(_manifest(), metrics)

    def test_per_head_and_per_layer_domains_are_rejected(self) -> None:
        mutations = {
            "mean_above_max": (
                ("2", "per_head", 0, "spectral_radius_raw_preclamp_mean"),
                0.9,
            ),
            "negative_gap": (
                ("2", "per_head", 0, "spectral_gap_mean"),
                -0.1,
            ),
            "normalized_radius_gate": (
                (
                    "2",
                    "per_head",
                    0,
                    "spectral_radius_normalized_pre_gamma_max",
                ),
                1.01,
            ),
            "lambda_below_ridge": (
                ("2", "per_head", 0, "lambda_min_gram_min"),
                0.0005,
            ),
            "zero_head_clamp_factor": (
                ("2", "per_head", 0, "clamp_factor_mean"),
                0.0,
            ),
            "invalid_head_clamp_fraction": (
                ("2", "per_head", 0, "clamp_fraction"),
                1.1,
            ),
            "condition_mean_above_max": (
                ("2", "per_head", 0, "ridge_condition_number_mean"),
                6.0,
            ),
            "fractional_head_escalation_count": (
                ("2", "per_head", 0, "ridge_escalation_count"),
                0.5,
            ),
            "negative_layer_ratio": (
                ("2", "layer", "residual_norm_ratio"),
                -0.1,
            ),
            "layer_rank_above_d_model": (
                ("2", "layer", "effective_jacobian_rank"),
                65.0,
            ),
        }
        for label, (path, value) in mutations.items():
            with self.subTest(label=label):
                metrics = _metrics()
                target = metrics["diagnostics_by_layer"]
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = value
                with self.assertRaises(Phase2ResultError):
                    _validate(_manifest(), metrics)

    def test_aggregate_reductions_must_match_per_head_records(self) -> None:
        inconsistent = {
            "spectral_radius_raw_preclamp": 1.19,
            "spectral_radius_normalized_pre_gamma": 0.89,
            "spectral_radius_applied_post_gamma": 1.29,
            "clamp_factor_mean": 0.84,
            "clamp_fraction": 0.24,
            "spectral_gap_mean": 0.16,
            "spectral_gap_max": 0.39,
            "lambda_min_gram": 0.0011,
            "ridge_condition_number": 9.0,
            "ska_branch_norm_ratio": 0.19,
            "ska_gradient_to_mamba_ratio": 0.99,
            "effective_jacobian_rank": 8.9,
            "beta_mean": 0.29,
            "beta_saturation_fraction": 0.011,
            "layerscale_magnitude_mean": 0.00011,
        }
        for name, value in inconsistent.items():
            with self.subTest(metric=name):
                metrics = _metrics()
                metrics["diagnostics"][name] = value
                with self.assertRaisesRegex(
                    Phase2ResultError, "aggregate mismatch"
                ):
                    _validate(_manifest(), metrics)

    def test_aggregate_roundoff_tolerance_is_explicit(self) -> None:
        metrics = _metrics()
        metrics["diagnostics"]["spectral_radius_normalized_pre_gamma"] += 5e-8
        _validate(_manifest(), metrics)

    def test_boolean_numbers_are_rejected(self) -> None:
        for name in ("spectral_radius_raw_preclamp", "peak_memory_bytes"):
            with self.subTest(metric=name):
                metrics = _metrics()
                metrics["diagnostics"][name] = True
                with self.assertRaises(Phase2ResultError):
                    _validate(_manifest(), metrics)

    def test_actual_model_and_compute_accounting_is_centrally_checked(self) -> None:
        metrics = _metrics()
        metrics["gpu_hours_actual"] = 0.9
        with self.assertRaisesRegex(Phase2ResultError, "gpu_hours_actual"):
            _validate(_manifest(), metrics)

        metrics = _metrics()
        metrics["model_accounting"]["parameter_counts_actual"]["total"] = 3049
        with self.assertRaisesRegex(Phase2ResultError, "embedding \\+"):
            _validate(_manifest(), metrics)

        metrics = _metrics()
        metrics["model_accounting"]["parameter_counts_actual"]["embedding"] = 2049
        metrics["model_accounting"]["parameter_counts_actual"]["total"] = 3049
        with self.assertRaisesRegex(Phase2ResultError, "embedding count differs"):
            _validate(_manifest(), metrics)

        metrics = _metrics()
        metrics["model_accounting"]["scale_band"]["outcome"] = "outside_band"
        with self.assertRaisesRegex(Phase2ResultError, "scale-band outcome"):
            _validate(_manifest(), metrics)

        metrics = _metrics()
        accounting = metrics["model_accounting"]
        accounting["parameter_counts_actual"]["non_embedding_core"] = 1100
        accounting["parameter_counts_actual"]["total"] = 3148
        accounting["parameter_estimator"]["core_relative_error"] = 0.1
        accounting["parameter_estimator"]["total_relative_error"] = 100 / 3048
        accounting["parameter_estimator"]["outcome"] = "exceeded"
        accounting["scale_band"]["actual_value"] = 3148
        with self.assertRaisesRegex(Phase2ResultError, "estimator-drift limit"):
            _validate(_manifest(), metrics)

        metrics = _metrics()
        flops = metrics["model_accounting"]["flops"]
        flops["estimated"] = 2_000_000.0
        flops["relative_error"] = 1.0
        flops["outcome"] = "exceeded"
        with self.assertRaisesRegex(Phase2ResultError, "FLOP estimate"):
            _validate(_manifest(), metrics)

    def test_optimizer_group_coverage_exclusivity_and_digests_are_checked(
        self,
    ) -> None:
        metrics = _metrics()
        metrics["optimizer_group_audit"]["coverage_digest"] = "0" * 64
        with self.assertRaisesRegex(Phase2ResultError, "coverage_digest"):
            _validate(_manifest(), metrics)

        manifest = _manifest()
        manifest["optimizer"]["groups"]["norms"] = {
            "lr": 5e-6,
            "weight_decay": 0.0,
            "owns": ["normalization scales"],
        }
        with self.assertRaisesRegex(Phase2ResultError, "omitted non-conditional"):
            _validate(manifest, _metrics())

        metrics = _metrics()
        metrics["optimizer_group_audit"]["groups"][0]["lr"] = 2e-5
        with self.assertRaisesRegex(Phase2ResultError, "lr aggregate mismatch"):
            _validate(_manifest(), metrics)

        metrics = _metrics()
        metrics["optimizer_group_audit"]["coverage"][
            "unassigned_parameter_names"
        ] = ["orphan.weight"]
        with self.assertRaisesRegex(Phase2ResultError, "unassigned"):
            _validate(_manifest(), metrics)

        metrics = _metrics()
        group = metrics["optimizer_group_audit"]["groups"][0]
        group["parameters"][1]["name"] = "backbone.weight"
        group["parameter_digest"] = optimizer_parameter_digest(
            group["parameters"]
        )
        metrics["optimizer_group_audit"]["coverage"][
            "duplicate_parameter_names"
        ] = ["backbone.weight"]
        metrics["optimizer_group_audit"][
            "trainable_parameter_digest"
        ] = optimizer_parameter_digest(group["parameters"])
        metrics["optimizer_group_audit"]["coverage_digest"] = (
            optimizer_coverage_digest(metrics["optimizer_group_audit"]["groups"])
        )
        with self.assertRaisesRegex(Phase2ResultError, "more than once"):
            _validate(_manifest(), metrics)

    def test_requested_and_effective_ridge_are_literal_and_auditable(self) -> None:
        metrics = _metrics()
        metrics["diagnostics"]["effective_ridge_max"] = 0.0011
        with self.assertRaisesRegex(Phase2ResultError, "explicitly counted"):
            _validate(_manifest(), metrics)

        metrics = _metrics()
        head = metrics["diagnostics_by_layer"]["2"]["per_head"][0]
        head["effective_ridge_max"] = 0.002
        head["ridge_escalation_count"] = 1
        metrics["diagnostics"]["effective_ridge_max"] = 0.002
        metrics["diagnostics"]["ridge_escalation_count"] = 1
        _validate(_manifest(), metrics)

        metrics["diagnostics"]["effective_ridge_max"] = 0.003
        with self.assertRaisesRegex(Phase2ResultError, "aggregate mismatch"):
            _validate(_manifest(), metrics)

        metrics = _metrics()
        metrics["diagnostics"]["ridge_requested"] = 0.0011
        with self.assertRaisesRegex(Phase2ResultError, "ridge_requested"):
            _validate(_manifest(), metrics)

    def test_parity_gates_are_boundary_specific(self) -> None:
        for name in (
            "state_rebuild_decode_prefill_max_abs_error",
            "chunk_boundary_parallel_decode_max_abs_error",
        ):
            with self.subTest(metric=name):
                metrics = _metrics()
                metrics["diagnostics"][name] = 0.0002
                with self.assertRaisesRegex(Phase2ResultError, "exceeds"):
                    _validate(_manifest(), metrics)

        metrics = _metrics()
        metrics[
            "diagnostics"
        ]["within_chunk_parallel_decode_drift_max_abs_error"] = 1.0
        _validate(_manifest(), metrics)

    def test_paper_and_mamba_na_contracts(self) -> None:
        paper_metrics = _metrics()
        paper_na = [
            "beta_mean",
            "beta_saturation_fraction",
            "layerscale_magnitude_mean",
        ]
        for name in paper_na:
            paper_metrics["diagnostics"][name] = None
        for layer_values in paper_metrics["diagnostics_by_layer"].values():
            for name in paper_na:
                layer_values["layer"][name] = None
        paper_metrics["not_applicable_diagnostics"] = paper_na
        _validate(_manifest("paper_control"), paper_metrics)

        mamba_metrics = _metrics()
        mamba_na = [
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
            "beta_mean",
            "beta_saturation_fraction",
            "layerscale_magnitude_mean",
            "ska_branch_norm_ratio",
            "spectral_gap_mean",
            "spectral_gap_max",
            "lambda_min_gram",
            "ska_gradient_to_mamba_ratio",
            "effective_jacobian_rank",
        ]
        for name in mamba_na:
            mamba_metrics["diagnostics"][name] = None
        mamba_metrics["not_applicable_diagnostics"] = mamba_na
        mamba_metrics["diagnostics_by_layer"] = {}
        _validate(_manifest("mamba_only"), mamba_metrics)

        transformer_metrics = _metrics()
        for name in mamba_na:
            transformer_metrics["diagnostics"][name] = None
        transformer_metrics["not_applicable_diagnostics"] = mamba_na
        transformer_metrics["diagnostics_by_layer"] = {}
        _validate(_manifest("transformer"), transformer_metrics)

    def test_na_cannot_hide_an_applicable_metric(self) -> None:
        metrics = _metrics()
        metrics["diagnostics"]["step_time_ms"] = None
        metrics["not_applicable_diagnostics"] = ["step_time_ms"]
        with self.assertRaisesRegex(Phase2ResultError, "incorrectly declared"):
            _validate(_manifest(), metrics)

    def test_wandb_run_identity_is_mandatory(self) -> None:
        metrics = _metrics()
        metrics["wandb_run_id"] = ""
        with self.assertRaisesRegex(Phase2ResultError, "wandb_run_id"):
            _validate(_manifest(), metrics)

    def test_hard_mqar_and_generator_identity_are_recomputed(self) -> None:
        metrics = _metrics()
        metrics["mqar_hard_accuracy"] = 0.4
        with self.assertRaisesRegex(Phase2ResultError, "mqar_hard_accuracy"):
            _validate(_manifest(), metrics)

        metrics = _metrics()
        metrics["mqar_generator_revision"] = "wrong-revision"
        with self.assertRaisesRegex(
            Phase2ResultError, "mqar_generator_revision"
        ):
            _validate(_manifest(), metrics)

    def test_final_resume_provenance_must_match_screen_checkpoint(self) -> None:
        manifest = _manifest()
        screen = _metrics()
        final = _metrics()
        final["optimizer_step"] = 6
        final["tokens_seen"] = 48
        final.pop("checkpoint_provenance")
        final["resume_provenance"] = {
            "source_checkpoint_sha256": "c" * 64,
            "source_checkpoint_bundle_path": "checkpoints/step_500",
            "source_checkpoint_bundle_byte_size": 123,
            "source_checkpoint_bundle_file_count": 4,
            "source_checkpoint_digest_algorithm": (
                "sha256(path_nul_size_nul_content_sha256_newline_v1)"
            ),
            "source_optimizer_step": 3,
            "restored_global_sequence_index": 6,
            "checkpoint_format_revision": "test-checkpoint-v1",
            "optimizer_state_restored": True,
            "scheduler_state_restored": True,
            "rng_state_restored": True,
            "data_stream_state_restored": True,
        }
        validate_step_metrics(manifest, final, expected_step=6)
        validate_cross_stage_identity(screen, final)

        final["resume_provenance"]["source_checkpoint_sha256"] = "d" * 64
        with self.assertRaisesRegex(
            Phase2ResultError, "source_checkpoint_sha256"
        ):
            validate_cross_stage_identity(screen, final)

if __name__ == "__main__":
    unittest.main()
