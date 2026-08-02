"""Portable correctness tests for the Phase 2a experiment contract."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from pathlib import Path

import pytest

from koopman_lm.experiments.phase2.analysis import pareto_front
from koopman_lm.experiments.phase2.controller import (
    bind_study_contract,
    report_step_500,
    should_prune_at_median,
)
from koopman_lm.experiments.phase2.controls import materialize_controls
from koopman_lm.experiments.phase2.dry_run import materialize_dry_run
from koopman_lm.experiments.phase2.manifest import (
    build_trial_manifest,
    control_parameters,
    git_identity,
    recompute_promotion_config_hash,
    recompute_trial_hash,
    reference_parameters,
    write_trial_directory,
)
from koopman_lm.experiments.phase2.promotions import materialize_promotions
from koopman_lm.experiments.phase2.preflight import (
    capability_template,
    evaluate_preflight,
)
from koopman_lm.experiments.phase2.results import (
    Phase2ResultError,
    expected_mqar_cells,
    optimizer_coverage_digest,
    optimizer_parameter_digest,
    validate_step_metrics,
)
from koopman_lm.experiments.phase2.runtime_fingerprint import (
    _CRITICAL_IMPORTS,
    build_runtime_lock,
)
from koopman_lm.experiments.phase2.run_study import (
    claim_trial_hash,
    validate_storage_url,
)
from koopman_lm.experiments.phase2.spec import (
    Phase2SpecError,
    derive_objective_weights,
    load_spec,
    placement_indices,
    round_ska_count,
    sample_parameters,
    stable_hash,
)


pytestmark = pytest.mark.correctness

ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "configs" / "phase2a_search.json"
CODE = {"commit": "a" * 40, "branch": "test", "dirty": False}
DATA = {
    "tokenizer_revision": "pinned",
    "train_dataset_revision": "pinned",
    "validation_dataset_revision": "pinned",
    "checksums": {"train": "1" * 64, "validation": "2" * 64},
    "mqar_screening": {
        "seed": 42,
        "generator_revision": "test-mqar-v1",
        "samples_per_cell": 8,
    },
}


def _test_runtime_fingerprint():
    return {
        "python": {
            "implementation": "CPython",
            "version": "3.11.9",
            "executable": "/test/venv/bin/python",
        },
        "platform": {
            "system": "Linux",
            "release": "test",
            "machine": "x86_64",
            "libc": ["glibc", "2.31"],
        },
        "packages": {
            "koopman_lm": "0.5.0",
            "torch": "2.6.0",
            "numpy": "2.0.0",
            "mamba-ssm": "2.2.4",
            "causal-conv1d": "1.5.0",
            "triton": "3.2.0",
            "transformers": "4.48.0",
            "datasets": "3.2.0",
            "safetensors": "0.5.0",
            "wandb": "0.19.0",
            "PyYAML": "6.0.2",
            "optuna": "4.2.0",
            "scikit-learn": "1.6.0",
            "psycopg": "3.2.0",
        },
        "critical_imports": {
            module_name: {"ok": True} for module_name in _CRITICAL_IMPORTS
        },
        "torch": {
            "available": True,
            "version": "2.6.0",
            "cuda_build": "12.4",
            "cuda_available": True,
            "device_count": 1,
            "devices": [{"name": "test-gpu"}],
            "gpu_smoke": {
                "ok": True,
                "operation": "cuda_float32_matmul_2x2",
                "result": 16.0,
            },
        },
        "cuda_toolkit": {
            "cuda_home": "/cuda",
            "cuda_path": None,
            "nvcc_version_output": "release 12.4",
            "driver_versions": ["550"],
        },
    }


def test_architecture_base_is_exactly_pinned(spec):
    assert spec["architecture_base"] == {
        "branch": "claude/cholesky-smaller-rank-0dcu89",
        "commit": "e69f087a7eafffcfaee0bb946f2213c87059dde0",
        "status": "upstream_work_in_progress",
    }


def _diagnostic_payload(manifest, ppl=100.0):
    diagnostics = {name: 0.0 for name in manifest["required_diagnostics"]}
    diagnostics.update(
        {
            "spectral_radius_raw_preclamp": 0.5,
            "spectral_radius_normalized_pre_gamma": 0.5,
            "spectral_radius_applied_post_gamma": 0.6,
            "clamp_factor_mean": 1.0,
            "clamp_fraction": 0.0,
            "spectral_gap_mean": 0.1,
            "spectral_gap_max": 0.1,
            "lambda_min_gram": 0.001,
            "ridge_condition_number": 1.0,
            "ridge_escalation_count": 0,
            "ridge_requested": manifest["model_config"]["ska_ridge"],
            "effective_ridge_min": manifest["model_config"]["ska_ridge"],
            "effective_ridge_max": manifest["model_config"]["ska_ridge"],
            "beta_mean": 0.0,
            "beta_saturation_fraction": 0.0,
            "layerscale_magnitude_mean": 0.0,
            "ska_branch_norm_ratio": 0.1,
            "ska_gradient_to_mamba_ratio": 1.0,
            "effective_jacobian_rank": 1.0,
            "step_time_ms": 1.0,
            "peak_memory_bytes": 1,
            "ppl_full": ppl,
            "ppl_ska_zero": ppl,
            "ppl_mamba_zero": ppl,
            "ppl_both_zero": ppl,
            "ska_zero_ppl_delta": 0.0,
        }
    )
    head = {
        "spectral_radius_raw_preclamp_mean": 0.5,
        "spectral_radius_raw_preclamp_max": 0.5,
        "spectral_radius_normalized_pre_gamma_mean": 0.5,
        "spectral_radius_normalized_pre_gamma_max": 0.5,
        "spectral_radius_applied_post_gamma_mean": 0.6,
        "spectral_radius_applied_post_gamma_max": 0.6,
        "clamp_factor_mean": 1.0,
        "clamp_fraction": 0.0,
        "spectral_gap_mean": 0.1,
        "spectral_gap_max": 0.1,
        "lambda_min_gram_mean": 0.001,
        "lambda_min_gram_min": 0.001,
        "ridge_condition_number_mean": 1.0,
        "ridge_condition_number_max": 1.0,
        "ridge_escalation_count": 0,
        "effective_ridge_min": manifest["model_config"]["ska_ridge"],
        "effective_ridge_max": manifest["model_config"]["ska_ridge"],
    }
    layer = {
        "residual_norm_ratio": 0.1,
        "gradient_norm_ratio": 1.0,
        "effective_jacobian_rank": 1.0,
        "beta_mean": 0.0,
        "beta_saturation_fraction": 0.0,
        "layerscale_magnitude_mean": 0.0,
    }
    by_layer = {
        str(index): {
            "per_head": [
                dict(head) for _ in range(manifest["model_config"]["ska_n_heads"])
            ],
            "layer": dict(layer),
        }
        for index in manifest["model_config"]["ska_layer_indices"]
    }
    return diagnostics, by_layer


def _result_accounting(manifest):
    counts = dict(manifest["parameter_counts_estimated"])
    required_groups = sorted(
        name
        for name, definition in manifest["optimizer"]["groups"].items()
        if "conditional" not in definition
    )
    groups = []
    remaining = counts["total"]
    for index, manifest_group_name in enumerate(required_groups):
        numel = (
            remaining
            if index == len(required_groups) - 1
            else 1
        )
        remaining -= numel
        parameter = {
            "name": f"{manifest_group_name}.test_parameter",
            "numel": numel,
        }
        definition = manifest["optimizer"]["groups"][manifest_group_name]
        group = {
            "group_name": f"{manifest_group_name}_test",
            "manifest_group_name": manifest_group_name,
            "lr": definition["lr"],
            "weight_decay": definition.get(
                "weight_decay",
                manifest["protocol"]["weight_decay"],
            ),
            "parameters": [parameter],
            "tensor_count": 1,
            "scalar_parameter_count": numel,
            "parameter_digest": optimizer_parameter_digest([parameter]),
        }
        groups.append(group)
    parameters = sorted(
        [
            parameter
            for group in groups
            for parameter in group["parameters"]
        ],
        key=lambda item: item["name"],
    )
    return {
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
            "optimizer_step": manifest["fidelity"]["prune_step"],
            "global_sequence_index": (
                manifest["fidelity"]["prune_step"]
                * manifest["protocol"]["effective_batch_sequences"]
            ),
            "checkpoint_format_revision": "test-checkpoint-v1",
        },
        "model_accounting": {
            "parameter_counts_actual": counts,
            "parameter_estimator": {
                "core_relative_error": 0.0,
                "total_relative_error": 0.0,
                "maximum_allowed_relative_error": manifest[
                    "hard_failure_rules"
                ]["parameter_estimator_relative_error_max"],
                "outcome": "within_tolerance",
            },
            "flops": {
                "scope": "training_forward_backward_per_optimizer_step",
                "estimated": 1_000_000.0,
                "measured": 1_000_000.0,
                "measurement_steps": 1,
                "estimator_method": "analytic-test",
                "measurement_method": "profiler-test",
                "relative_error": 0.0,
                "maximum_allowed_relative_error": manifest[
                    "hard_failure_rules"
                ]["flop_estimator_relative_error_max"],
                "outcome": "within_tolerance",
            },
            "scale_band": {
                "label": manifest["scale_check"]["label"],
                "accounting": manifest["scale_check"]["accounting"],
                "accepted_min": manifest["scale_check"]["accepted_min"],
                "accepted_max": manifest["scale_check"]["accepted_max"],
                "actual_value": counts["non_embedding_core"],
                "outcome": "within_band",
            },
        },
        "optimizer_group_audit": {
            "groups": groups,
            "coverage": {
                "trainable_parameter_tensor_count": len(parameters),
                "assigned_parameter_tensor_count": len(parameters),
                "trainable_scalar_parameter_count": counts["total"],
                "assigned_scalar_parameter_count": counts["total"],
                "unassigned_parameter_names": [],
                "duplicate_parameter_names": [],
                "complete": True,
                "exclusive": True,
            },
            "trainable_parameter_digest": optimizer_parameter_digest(parameters),
            "coverage_digest": optimizer_coverage_digest(groups),
        },
    }


@pytest.fixture(scope="module")
def spec():
    return load_spec(SPEC_PATH)


def test_phase2_fraction_counts_are_distinct(spec):
    fractions = spec["axes"]["ska_fraction"]["values"]
    assert [round_ska_count(16, fraction) for fraction in fractions] == [2, 3, 4, 5]


def test_placement_algorithms_are_exact_and_interior():
    assert placement_indices(16, 4, "uniform") == [1, 5, 10, 14]
    assert placement_indices(16, 4, "late_biased") == [6, 9, 11, 14]
    assert placement_indices(16, 4, "middle_clustered") == [5, 7, 10, 12]
    for strategy in ("uniform", "late_biased", "middle_clustered"):
        for count in (2, 3, 4, 5):
            indices = placement_indices(16, count, strategy)
            assert len(indices) == count
            assert indices == sorted(set(indices))
            assert 0 not in indices and 15 not in indices


def test_conditional_birdie_axes_never_leak(spec):
    rng = random.Random(123)
    saw_birdie = saw_ntp = False
    for _ in range(200):
        params = sample_parameters(spec, rng)
        if params["objective_arm"] == "birdie_mix":
            saw_birdie = True
            assert "birdie_retrieval_fraction" in params
            assert "birdie_copy_share" in params
            assert math.isclose(sum(derive_objective_weights(params).values()), 1.0)
        else:
            saw_ntp = True
            assert "birdie_retrieval_fraction" not in params
            assert "birdie_copy_share" not in params
            assert derive_objective_weights(params)["next_token"] == 1.0
    assert saw_birdie and saw_ntp


@pytest.mark.parametrize("probability", [0.1, 0.3, 0.5])
def test_beta_probability_materializes_as_logit_bias(spec, probability):
    params = reference_parameters({"beta_init_probability": probability})
    manifest = build_trial_manifest(
        spec, params, code_identity=CODE, data_identity=DATA
    )
    bias = manifest["model_extensions"]["beta_init_bias"]
    assert math.isclose(1.0 / (1.0 + math.exp(-bias)), probability, abs_tol=1e-12)


def test_every_rank_fraction_kernel_combination_stays_in_core_band(spec):
    low, high = spec["scale"]["accepted_min"], spec["scale"]["accepted_max"]
    for rank in spec["axes"]["ska_rank"]["values"]:
        for fraction in spec["axes"]["ska_fraction"]["values"]:
            for kernel in spec["axes"]["short_conv_kernel"]["values"]:
                params = reference_parameters(
                    {
                        "ska_rank": rank,
                        "ska_fraction": fraction,
                        "short_conv_kernel": kernel,
                    }
                )
                manifest = build_trial_manifest(
                    spec, params, code_identity=CODE, data_identity=DATA
                )
                core = manifest["parameter_counts_estimated"]["non_embedding_core"]
                assert low <= core <= high


def test_trial_hash_covers_non_model_inputs(spec):
    base = reference_parameters()
    first = build_trial_manifest(
        spec, base, seed=42, code_identity=CODE, data_identity=DATA
    )
    lr_changed = build_trial_manifest(
        spec,
        reference_parameters({"mlp_lr_multiplier": 3.0}),
        seed=42,
        code_identity=CODE,
        data_identity=DATA,
    )
    seed_changed = build_trial_manifest(
        spec, base, seed=43, code_identity=CODE, data_identity=DATA
    )
    data_changed = build_trial_manifest(
        spec,
        base,
        seed=42,
        code_identity=CODE,
        data_identity={**DATA, "tokenizer_revision": "different"},
    )
    capability_changed = build_trial_manifest(
        spec,
        base,
        seed=42,
        code_identity=CODE,
        data_identity=DATA,
        capability_identity={"integration_commit": "c" * 40},
    )
    code_changed = build_trial_manifest(
        spec,
        base,
        seed=42,
        code_identity={**CODE, "commit": "d" * 40},
        data_identity=DATA,
    )
    assert first["model_hash"] == lr_changed["model_hash"]
    assert first["promotion_config_hash"] == seed_changed["promotion_config_hash"]
    assert first["promotion_config_hash"] != lr_changed["promotion_config_hash"]
    assert recompute_trial_hash(first) == first["trial_hash"]
    assert (
        recompute_promotion_config_hash(first)
        == first["promotion_config_hash"]
    )
    assert len(
        {
            first["trial_hash"],
            lr_changed["trial_hash"],
            seed_changed["trial_hash"],
            data_changed["trial_hash"],
            capability_changed["trial_hash"],
            code_changed["trial_hash"],
        }
    ) == 6

    qknorm_changed = build_trial_manifest(
        spec,
        reference_parameters({"qk_norm": True}),
        seed=42,
        code_identity=CODE,
        data_identity=DATA,
    )
    assert first["model_hash"] != qknorm_changed["model_hash"]


def test_fixed_controls_keep_architecture_modes_separate(spec):
    paper = build_trial_manifest(
        spec,
        control_parameters(spec, "paper_control"),
        code_identity=CODE,
        data_identity=DATA,
    )
    candidate = build_trial_manifest(
        spec,
        control_parameters(spec, "updated_default_control"),
        code_identity=CODE,
        data_identity=DATA,
    )
    mamba = build_trial_manifest(
        spec,
        control_parameters(spec, "mamba_only_control"),
        code_identity=CODE,
        data_identity=DATA,
    )
    transformer = build_trial_manifest(
        spec,
        control_parameters(spec, "transformer_control"),
        code_identity=CODE,
        data_identity=DATA,
    )
    assert paper["model_extensions"]["stats_mode"] == "paper_sequence_max"
    assert paper["model_extensions"]["write_gate"] == "none"
    assert "beta_init_probability" not in paper["model_extensions"]
    assert "beta_init_probability" not in paper["parameters"]
    assert candidate["model_extensions"]["stats_mode"] == "candidate_beta_gated"
    assert mamba["model_config"]["ska_layer_indices"] == []
    assert "ska_rank" not in mamba["parameters"]
    assert mamba["model_extensions"]["baseline_builder"] == "build_mamba_only"
    assert mamba["model_extensions"]["channel_mixer"] == "swiglu"
    assert transformer["model_config"]["ska_layer_indices"] == []
    assert transformer["model_extensions"]["baseline_builder"] == "build_transformer"
    assert transformer["model_extensions"]["echo_components_present"] is False
    assert transformer["architecture_base"]["commit"] == (
        "e69f087a7eafffcfaee0bb946f2213c87059dde0"
    )


def test_step_500_pruning_contract():
    assert not should_prune_at_median(0.1, [0.5] * 19, minimum_completed_trials=20)
    assert should_prune_at_median(0.49, [0.5] * 20, minimum_completed_trials=20)
    assert not should_prune_at_median(0.5, [0.5] * 20, minimum_completed_trials=20)

    class Trial:
        reports = []

        def report(self, value, step):
            self.reports.append((value, step))

    trial = Trial()
    report_step_500(
        trial,
        {"status": "ok", "optimizer_step": 500, "mqar_accuracy": 0.75},
    )
    assert trial.reports == [(0.75, 500)]
    with pytest.raises(ValueError):
        report_step_500(
            trial,
            {"status": "ok", "optimizer_step": 499, "mqar_accuracy": 0.75},
        )


def test_study_contract_rejects_spec_code_or_data_drift():
    class Study:
        system_attrs = {}

        def set_system_attr(self, key, value):
            self.system_attrs[key] = value

    study = Study()
    identity = {
        "spec_hash": "a" * 64,
        "code_commit": "b" * 40,
        "data_hash": "c" * 64,
    }
    bind_study_contract(study, identity)
    bind_study_contract(study, identity)
    with pytest.raises(RuntimeError, match="study contract mismatch"):
        bind_study_contract(study, {**identity, "data_hash": "d" * 64})


def test_distributed_trial_hash_claim_is_atomic(tmp_path):
    digest = "f" * 64
    first = claim_trial_hash(tmp_path, digest, 1)
    second = claim_trial_hash(tmp_path, digest, 2)
    assert first is not None and first.is_file()
    assert second is None


def test_storage_backend_must_match_approved_spec():
    validate_storage_url(
        "postgresql+psycopg://host/study", approved_backend="postgresql"
    )
    validate_storage_url(
        "journal:///absolute/study.log",
        approved_backend="scg_validated_journal_storage",
    )
    with pytest.raises(ValueError):
        validate_storage_url("sqlite:///study.db", approved_backend="postgresql")
    with pytest.raises(ValueError):
        validate_storage_url(
            "journal://relative.log",
            approved_backend="scg_validated_journal_storage",
        )


def test_pareto_front_uses_both_metrics_without_scalarization():
    records = [
        {"trial_hash": "a", "status": "COMPLETE", "mqar_accuracy": 0.8, "wikitext_ppl": 20},
        {"trial_hash": "b", "status": "COMPLETE", "mqar_accuracy": 0.7, "wikitext_ppl": 18},
        {"trial_hash": "c", "status": "COMPLETE", "mqar_accuracy": 0.6, "wikitext_ppl": 22},
        {"trial_hash": "d", "status": "PRUNED", "mqar_accuracy": 0.9, "wikitext_ppl": 17},
    ]
    assert [item["trial_hash"] for item in pareto_front(records)] == ["a", "b"]


def test_result_contract_requires_all_20_mqar_cells_including_256_by_64(spec):
    manifest = build_trial_manifest(
        spec, reference_parameters(), code_identity=CODE, data_identity=DATA
    )
    diagnostics, by_layer = _diagnostic_payload(manifest)
    metrics = {
        "schema_version": 1,
        "trial_hash": manifest["trial_hash"],
        "status": "ok",
        "optimizer_step": 500,
        "tokens_seen": 98_304_000,
        "mqar_accuracy": 0.5,
        "mqar_hard_accuracy": 0.5,
        "mqar_worst_cell_accuracy": 0.5,
        "mqar_worst_cell": "length_1024/pairs_32",
        "mqar_seed": 42,
        "mqar_generator_revision": "test-mqar-v1",
        "mqar_samples_per_cell": 8,
        "wikitext_ppl": 100.0,
        "diagnostics": diagnostics,
        "diagnostics_by_layer": by_layer,
        "mqar_grid": {cell: 0.5 for cell in expected_mqar_cells()},
        **_result_accounting(manifest),
    }
    assert len(metrics["mqar_grid"]) == 20
    assert "length_256/pairs_64" in metrics["mqar_grid"]
    validate_step_metrics(manifest, metrics, expected_step=500)

    metrics["mqar_grid"].pop("length_256/pairs_64")
    with pytest.raises(Phase2ResultError, match="MQAR grid mismatch"):
        validate_step_metrics(manifest, metrics, expected_step=500)


def test_result_contract_rejects_inconsistent_mqar_macro(spec):
    manifest = build_trial_manifest(
        spec, reference_parameters(), code_identity=CODE, data_identity=DATA
    )
    diagnostics, by_layer = _diagnostic_payload(manifest)
    metrics = {
        "schema_version": 1,
        "trial_hash": manifest["trial_hash"],
        "status": "ok",
        "optimizer_step": 500,
        "tokens_seen": 98_304_000,
        "mqar_accuracy": 1.0,
        "mqar_hard_accuracy": 0.0,
        "mqar_worst_cell_accuracy": 0.0,
        "mqar_worst_cell": "length_1024/pairs_32",
        "mqar_seed": 42,
        "mqar_generator_revision": "test-mqar-v1",
        "mqar_samples_per_cell": 8,
        "wikitext_ppl": None,
        "diagnostics": diagnostics,
        "diagnostics_by_layer": by_layer,
        "mqar_grid": {cell: 0.0 for cell in expected_mqar_cells()},
        **_result_accounting(manifest),
    }
    with pytest.raises(Phase2ResultError, match="macro-average"):
        validate_step_metrics(
            manifest,
            metrics,
            expected_step=500,
            require_wikitext_ppl=False,
        )


def test_preflight_fails_closed_then_can_pass_a_resolved_contract(spec, tmp_path):
    template = capability_template(spec)
    failed = evaluate_preflight(
        spec, template, data_manifest={}, require_clean=False
    )
    assert not failed["ready"]
    assert any(item["code"] == "MISSING_CAPABILITY" for item in failed["failures"])

    resolved_spec = copy.deepcopy(spec)
    resolved_spec["status"] = "scientific_ready"
    resolved_spec["protocol"]["tokenizer_revision"] = "pinned"
    resolved_spec["protocol"]["train_dataset_revision"] = "pinned"
    resolved_spec["protocol"]["validation_dataset_revision"] = "pinned"
    resolved_spec["protocol"]["learning_rate_status"] = "approved"
    resolved_spec["protocol"]["training_protocol_status"] = "approved"
    resolved_spec["protocol"]["data_stream"].update(
        {
            "sampler_revision": "sampler-v1",
            "shuffle_seed": 42,
            "shard_order_policy": "frozen-manifest-order",
            "shard_order_manifest_sha256": "a" * 64,
            "epoch_policy": "single-pass-then-repeat",
            "repeat_policy": "restart-identical-order",
        }
    )
    resolved_spec["study"]["compute_budget"].update(
        {
            "maximum_total_gpu_hours": 100,
            "maximum_full_trial_gpu_hours": 1,
            "required_gpu_type": "test-gpu",
            "status": "approved",
        }
    )
    resolved_spec["storage"]["backend"] = "postgresql"
    train_tokens = tmp_path / "train.tokens"
    validation_tokens = tmp_path / "validation.tokens"
    mqar_samples = tmp_path / "mqar.jsonl"
    train_tokens.write_bytes(b"train")
    validation_tokens.write_bytes(b"validation")
    mqar_samples.write_bytes(b"mqar")
    auxiliary_paths = {}
    for index, name in enumerate(
        (
            "tokenizer_fingerprint",
            "training_shard_order_manifest",
            "mqar_oracle_fixture",
            "mqar_sample_ids",
            "mqar_vocabulary_identity",
            "mqar_token_map",
            "birdie_selective_copy_fixture",
            "birdie_infilling_fixture",
            "birdie_sample_identity_manifest",
        )
    ):
        path = tmp_path / f"{name}.json"
        path.write_bytes(f"frozen-{index}-{name}".encode())
        auxiliary_paths[name] = path
    resolved_spec["protocol"]["data_stream"][
        "shard_order_manifest_sha256"
    ] = hashlib.sha256(
        auxiliary_paths["training_shard_order_manifest"].read_bytes()
    ).hexdigest()
    data_manifest = {
        "schema_version": 1,
        "tokenizer": {
            "name": "NousResearch/Llama-2-7b-hf",
            "revision": "pinned",
            "vocab_size": 32000,
            "fingerprint_sha256": hashlib.sha256(
                auxiliary_paths["tokenizer_fingerprint"].read_bytes()
            ).hexdigest(),
        },
        "training": {
            "dataset": "FineWeb-Edu",
            "revision": "pinned",
            "split": "train",
            "document_range": "0:100",
            "packing": {
                "sequence_length": 2048,
                "eos_between_documents": True,
                "drop_remainder": True,
            },
            "stream": copy.deepcopy(
                resolved_spec["protocol"]["data_stream"]
            ),
            "token_file": str(train_tokens),
            "token_count": 2048,
            "byte_size": train_tokens.stat().st_size,
            "sha256": hashlib.sha256(train_tokens.read_bytes()).hexdigest(),
        },
        "wikitext_validation": {
            "dataset": "WikiText-103",
            "revision": "pinned",
            "split": "validation",
            "maximum_tokens": 2048,
            "packing": "contiguous",
            "token_file": str(validation_tokens),
            "byte_size": validation_tokens.stat().st_size,
            "sha256": hashlib.sha256(validation_tokens.read_bytes()).hexdigest(),
        },
        "mqar_screening": {
            "generator": "pinned oracle",
            "generator_revision": "oracle-v1",
            "oracle_fixture_sha256": hashlib.sha256(
                auxiliary_paths["mqar_oracle_fixture"].read_bytes()
            ).hexdigest(),
            "sequence_lengths": [256, 512, 1024, 2048],
            "key_value_pairs": [4, 8, 16, 32, 64],
            "samples_per_cell": 8,
            "seed": 42,
            "sample_ids_sha256": hashlib.sha256(
                auxiliary_paths["mqar_sample_ids"].read_bytes()
            ).hexdigest(),
            "artifact_file": str(mqar_samples),
            "byte_size": mqar_samples.stat().st_size,
            "sha256": hashlib.sha256(mqar_samples.read_bytes()).hexdigest(),
            "vocabulary": {
                "size": 128,
                "identity_sha256": hashlib.sha256(
                    auxiliary_paths["mqar_vocabulary_identity"].read_bytes()
                ).hexdigest(),
            },
            "token_map": {
                "revision": "token-map-v1",
                "sha256": hashlib.sha256(
                    auxiliary_paths["mqar_token_map"].read_bytes()
                ).hexdigest(),
            },
        },
        "birdie_training_objectives": {
            "mixer_semantics": "static Optuna mixture fixed for the whole trial",
            "mixer_revision": "mixer-v1",
            "selective_copy": {
                "generator": "copy-v1",
                "generator_revision": "copy-rev",
                "seed_policy": "trial-seed-derived",
                "fixture_sha256": hashlib.sha256(
                    auxiliary_paths["birdie_selective_copy_fixture"].read_bytes()
                ).hexdigest(),
            },
            "infilling": {
                "generator": "infill-v1",
                "generator_revision": "infill-rev",
                "seed_policy": "trial-seed-derived",
                "fixture_sha256": hashlib.sha256(
                    auxiliary_paths["birdie_infilling_fixture"].read_bytes()
                ).hexdigest(),
            },
            "sample_identity_manifest_sha256": hashlib.sha256(
                auxiliary_paths["birdie_sample_identity_manifest"].read_bytes()
            ).hexdigest(),
            "loss_normalization": (
                "mean_over_supervised_tokens_then_weighted_objective_sum"
            ),
            "equal_token_accounting": True,
        },
        "auxiliary_artifacts": {
            name: {
                "path": str(path),
                "byte_size": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for name, path in auxiliary_paths.items()
        },
    }
    data_hash = stable_hash(data_manifest)
    public_spec = {
        key: value for key, value in resolved_spec.items()
        if not key.startswith("_")
    }
    spec_hash = stable_hash(public_spec)
    integration_commit = "b" * 40
    evidence = {}
    for name in (
        "correctness_report",
        "resume_parity_report",
        "data_validation_report",
        "storage_concurrency_report",
        "pilot_report",
    ):
        path = tmp_path / f"{name}.json"
        report = {
            "schema_version": 1,
            "status": "pass",
            "evidence_type": name,
            "integration_commit": integration_commit,
            "spec_hash": spec_hash,
            "checks": [
                {"name": check, "status": "pass"}
                for check in resolved_spec["required_evidence_checks"][name]
            ],
        }
        if name in {"data_validation_report", "pilot_report"}:
            report["data_manifest_sha256"] = data_hash
        path.write_text(
            json.dumps(report, sort_keys=True, allow_nan=False) + "\n"
        )
        evidence[name] = {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    runtime_fingerprint = _test_runtime_fingerprint()
    environment_lock = tmp_path / "environment.lock.json"
    environment_lock.write_text(
        json.dumps(
            build_runtime_lock(runtime_fingerprint),
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    evidence["environment_lock"] = {
        "path": str(environment_lock),
        "sha256": hashlib.sha256(environment_lock.read_bytes()).hexdigest(),
    }
    capabilities = {
        "schema_version": 1,
        "spec_hash": spec_hash,
        "integration_commit": integration_commit,
        "approved_data_manifest_sha256": data_hash,
        "capabilities": {
            name: True for name in resolved_spec["required_capabilities"]
        },
        "approvals": {
            name: True for name in resolved_spec["required_approvals"]
        },
        "approval_metadata": {
            "approved_by": "test",
            "approved_at_utc": "2026-07-27T00:00:00Z",
        },
        "evidence": evidence,
    }
    passed = evaluate_preflight(
        resolved_spec,
        capabilities,
        data_manifest=data_manifest,
        require_clean=False,
        verify_commit=False,
        runtime_fingerprint=runtime_fingerprint,
    )
    assert passed == {
        "study_name": spec["study_name"],
        "stage": "study",
        "spec_hash": spec_hash,
        "data_manifest_sha256": data_hash,
        "ready": True,
        "failure_count": 0,
        "failures": [],
    }
    drifted_runtime = copy.deepcopy(runtime_fingerprint)
    drifted_runtime["packages"]["triton"] = "unexpected"
    runtime_failed = evaluate_preflight(
        resolved_spec,
        capabilities,
        data_manifest=data_manifest,
        require_clean=False,
        verify_commit=False,
        runtime_fingerprint=drifted_runtime,
    )
    assert not runtime_failed["ready"]
    assert any(
        item["code"] == "ACTIVE_RUNTIME_MISMATCH"
        for item in runtime_failed["failures"]
    )

    pilot_capabilities = copy.deepcopy(capabilities)
    pilot_capabilities["evidence"]["pilot_report"] = {
        "path": "REQUIRED_BEFORE_SCIENTIFIC_RUN",
        "sha256": "REQUIRED_BEFORE_SCIENTIFIC_RUN",
    }
    pilot_passed = evaluate_preflight(
        resolved_spec,
        pilot_capabilities,
        data_manifest=data_manifest,
        stage="pilot",
        require_clean=False,
        verify_commit=False,
        runtime_fingerprint=runtime_fingerprint,
    )
    assert pilot_passed["ready"]
    study_failed = evaluate_preflight(
        resolved_spec,
        pilot_capabilities,
        data_manifest=data_manifest,
        stage="study",
        require_clean=False,
        verify_commit=False,
        runtime_fingerprint=runtime_fingerprint,
    )
    assert not study_failed["ready"]

    calibration_spec = copy.deepcopy(resolved_spec)
    calibration_spec["status"] = "premerge_infrastructure_only"
    calibration_spec["study"]["compute_budget"].update(
        {
            "maximum_total_gpu_hours": "REQUIRED_BEFORE_SCIENTIFIC_RUN",
            "maximum_full_trial_gpu_hours": "REQUIRED_BEFORE_SCIENTIFIC_RUN",
            "required_gpu_type": "REQUIRED_BEFORE_SCIENTIFIC_RUN",
            "status": "provisional_team_confirmation_required",
        }
    )
    calibration_spec["storage"]["backend"] = (
        "REQUIRED_BEFORE_SCIENTIFIC_RUN"
    )
    calibration_public_spec = {
        key: value
        for key, value in calibration_spec.items()
        if not key.startswith("_")
    }
    calibration_spec_hash = stable_hash(calibration_public_spec)
    calibration_capabilities = copy.deepcopy(capabilities)
    calibration_capabilities["spec_hash"] = calibration_spec_hash
    for name in (
        "survivor_step_budget",
        "distributed_storage_backend",
        "compute_budget",
    ):
        calibration_capabilities["approvals"][name] = False
    for name in (
        "correctness_report",
        "resume_parity_report",
        "data_validation_report",
    ):
        report_path = tmp_path / f"calibration_{name}.json"
        report = json.loads(
            Path(capabilities["evidence"][name]["path"]).read_text()
        )
        report["spec_hash"] = calibration_spec_hash
        report_path.write_text(
            json.dumps(report, sort_keys=True, allow_nan=False) + "\n"
        )
        calibration_capabilities["evidence"][name] = {
            "path": str(report_path),
            "sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
        }
    for name in ("storage_concurrency_report", "pilot_report"):
        calibration_capabilities["evidence"][name] = {
            "path": "REQUIRED_BEFORE_SCIENTIFIC_RUN",
            "sha256": "REQUIRED_BEFORE_SCIENTIFIC_RUN",
        }
    calibration_passed = evaluate_preflight(
        calibration_spec,
        calibration_capabilities,
        data_manifest=data_manifest,
        stage="calibration",
        require_clean=False,
        verify_commit=False,
        runtime_fingerprint=runtime_fingerprint,
    )
    assert calibration_passed["ready"]


def test_dry_run_writes_strict_unique_manifests(spec, tmp_path):
    result = materialize_dry_run(
        SPEC_PATH, trials=12, seed=7, output=tmp_path, include_controls=True
    )
    summary = result["summary"]
    assert summary["total_manifests"] == 15
    assert summary["unique_trial_hashes"] == 15
    assert not summary["scientific_launch_ready"]
    parsed = json.loads((tmp_path / "dry_run_summary.json").read_text())
    assert parsed == summary


def test_fixed_controls_materialize_at_all_three_promotion_seeds(tmp_path):
    summary = materialize_controls(SPEC_PATH, tmp_path)
    assert summary["control_run_count"] == 12
    assert summary["seeds"] == [42, 43, 44]
    hashes = {entry["trial_hash"] for entry in summary["execution_plan"]}
    assert len(hashes) == 12
    assert all(
        entry["required_stage"] == "final"
        for entry in summary["execution_plan"]
    )


def test_tampered_promotion_hash_is_rejected(spec, tmp_path):
    manifest = build_trial_manifest(
        spec,
        reference_parameters(),
        seed=42,
        code_identity=CODE,
        data_identity=DATA,
    )
    manifest["promotion_config_hash"] = "0" * 64
    assert recompute_trial_hash(manifest) == manifest["trial_hash"]
    assert (
        recompute_promotion_config_hash(manifest)
        != manifest["promotion_config_hash"]
    )
    with pytest.raises(Phase2SpecError, match="promotion config hash"):
        write_trial_directory(tmp_path, manifest)


def test_promotion_materializer_reuses_source_seed_and_builds_two_missing(
    spec, tmp_path
):
    params = reference_parameters()
    source = build_trial_manifest(
        spec,
        params,
        seed=42,
        code_identity=git_identity(ROOT),
    )
    shortlist = {
        "selected": [
            {
                "parameters": params,
                "seed": 42,
                "trial_hash": source["trial_hash"],
                "promotion_config_hash": source["promotion_config_hash"],
                "status": "COMPLETE",
                "health_passed": True,
            }
        ]
    }
    summary = materialize_promotions(SPEC_PATH, shortlist, tmp_path)
    assert summary["new_run_count"] == 2
    assert {entry["seed"] for entry in summary["execution_plan"]} == {43, 44}


def test_promotion_materializer_requires_complete_hashed_source(spec, tmp_path):
    params = reference_parameters()
    source = build_trial_manifest(
        spec,
        params,
        seed=42,
        code_identity=git_identity(ROOT),
    )
    incomplete = {
        "selected": [
            {
                "parameters": params,
                "seed": 42,
                "trial_hash": source["trial_hash"],
                "status": "COMPLETE",
                "health_passed": True,
            }
        ]
    }
    with pytest.raises(
        ValueError, match="requires a valid promotion_config_hash"
    ):
        materialize_promotions(SPEC_PATH, incomplete, tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_invalid_conditional_axis_is_rejected(spec):
    params = reference_parameters()
    params["birdie_retrieval_fraction"] = 0.3
    with pytest.raises(Phase2SpecError, match="Inactive conditional"):
        build_trial_manifest(
            spec, params, code_identity=CODE, data_identity=DATA
        )
