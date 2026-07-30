"""Materialize complete, content-addressed Phase 2 trial manifests."""

from __future__ import annotations

import json
import math
import os
import subprocess
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

try:
    from .spec import (
        Phase2SpecError,
        derive_objective_weights,
        estimate_parameter_counts,
        placement_indices,
        round_ska_count,
        stable_hash,
        validate_trial_parameters,
    )
except ImportError:  # Direct execution in a minimal environment.
    from spec import (  # type: ignore
        Phase2SpecError,
        derive_objective_weights,
        estimate_parameter_counts,
        placement_indices,
        round_ska_count,
        stable_hash,
        validate_trial_parameters,
    )


_REFERENCE_DEFAULTS: dict[str, Any] = {
    "architecture_mode": "updated_sweep",
    "ska_rank": 48,
    "ska_fraction": 0.25,
    "ska_placement": "uniform",
    "ska_chunk_size": 64,
    "ska_ridge": 0.001,
    "beta_init_probability": 0.3,
    "layerscale_init": 0.0001,
    "short_conv_kernel": 8,
    "short_conv_gate_init": 0.01,
    "qk_norm": False,
    "ska_projection_lr_multiplier": 5.0,
    "gamma_eta_lr_multiplier": 25.0,
    "koopman_eigen_lr_multiplier": 10.0,
    "norm_lr_multiplier": 0.5,
    "mlp_lr_multiplier": 1.5,
    "backbone_lr_multiplier": 1.0,
    "objective_arm": "next_token_only",
}


def _public_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in spec.items() if not key.startswith("_")}


def _load_base_model(spec: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(spec.get("_base_model_path", spec["base_model_config"]))
    try:
        model = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise Phase2SpecError(f"Base model config does not exist: {path}") from exc
    if not isinstance(model, dict):
        raise Phase2SpecError("Base model config must be a JSON object")
    return model


def git_identity(repo_root: str | Path) -> dict[str, Any]:
    """Return commit and dirty state without mutating the repository."""

    root = Path(repo_root)

    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    try:
        commit = run("rev-parse", "HEAD")
        status = run("status", "--porcelain")
        branch = run("rev-parse", "--abbrev-ref", "HEAD")
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "UNKNOWN", "branch": "UNKNOWN", "dirty": True}
    return {"commit": commit, "branch": branch, "dirty": bool(status)}


def reference_parameters(overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return the explicit candidate defaults, optionally overridden."""

    params = dict(_REFERENCE_DEFAULTS)
    if overrides:
        params.update(overrides)
    if params.get("objective_arm") != "birdie_mix":
        params.pop("birdie_retrieval_fraction", None)
        params.pop("birdie_copy_share", None)
    return params


def control_parameters(spec: Mapping[str, Any], name: str) -> dict[str, Any]:
    """Materialize one fixed control from the search spec."""

    matches = [control for control in spec["fixed_controls"] if control["name"] == name]
    if len(matches) != 1:
        raise Phase2SpecError(f"Expected exactly one fixed control named {name!r}")
    params = reference_parameters(matches[0])
    mode = params["architecture_mode"]
    candidate_only = {
        "beta_init_probability",
        "layerscale_init",
        "short_conv_kernel",
        "short_conv_gate_init",
    }
    if mode != "updated_sweep":
        for key in candidate_only:
            params.pop(key, None)
    if mode == "mamba_only":
        for key in {
            "ska_rank",
            "ska_placement",
            "ska_chunk_size",
            "ska_ridge",
            "qk_norm",
            "ska_projection_lr_multiplier",
            "gamma_eta_lr_multiplier",
        }:
            params.pop(key, None)
    return params


def _effective_lrs(
    spec: Mapping[str, Any], params: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    base_lr = float(spec["protocol"]["base_learning_rate"])
    mode = params["architecture_mode"]
    groups = {
        "backbone": {
            "lr": base_lr * float(params.get("backbone_lr_multiplier", 1.0)),
            "owns": ["Mamba/DeltaNet parameters when present"],
        },
        "koopman_mlp": {
            "lr": base_lr * float(params.get("mlp_lr_multiplier", 1.5)),
            "owns": ["Koopman MLP lift and readout"],
        },
        "koopman_eigenvalues": {
            "lr": base_lr * float(params.get("koopman_eigen_lr_multiplier", 10.0)),
            "owns": ["Koopman MLP eigenvalue parameters"],
            "weight_decay": 0.0,
        },
        "norms": {
            "lr": base_lr * float(params.get("norm_lr_multiplier", 0.5)),
            "owns": [
                "RMSNorm/LayerNorm and QKNorm scale parameters",
                "LayerScale",
            ],
            "weight_decay": 0.0,
        },
    }
    if mode != "mamba_only":
        groups["ska_projections"] = {
            "lr": base_lr * float(params.get("ska_projection_lr_multiplier", 5.0)),
            "owns": [
                "SKA key/query/value/output projections",
                "beta projection when present",
                "short-conv weights",
            ],
            "bias_weight_decay": 0.0,
        }
        groups["ska_gamma_eta"] = {
            "lr": base_lr * float(params.get("gamma_eta_lr_multiplier", 25.0)),
            "owns": ["learnable gamma/eta only"],
            "conditional": "exclude fixed or absent parameters",
            "weight_decay": 0.0,
        }
        groups["ska_auxiliary_gates"] = {
            "lr": base_lr,
            "owns": ["short-conv scalar gate"],
            "conditional": "exclude absent parameters",
            "weight_decay": 0.0,
        }
    return groups


def _materialize_model(
    spec: Mapping[str, Any], params: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    model = _load_base_model(spec)
    mode = params["architecture_mode"]
    n_layers = int(model["n_layers"])

    if mode == "mamba_only":
        count = 0
        indices: list[int] = []
    else:
        count = round_ska_count(n_layers, float(params["ska_fraction"]))
        indices = placement_indices(n_layers, count, str(params["ska_placement"]))

    if mode != "mamba_only":
        model.update(
            {
                "ska_rank": int(params["ska_rank"]),
                "ska_chunk_size": int(params["ska_chunk_size"]),
                "ska_ridge": float(params["ska_ridge"]),
                "ska_layer_indices": indices,
            }
        )
    else:
        model["ska_layer_indices"] = []

    extensions: dict[str, Any] = {
        "architecture_mode": mode,
        "stats_mode": "beta_causal" if mode == "updated_sweep" else mode,
        "qk_norm": False,
    }
    if mode == "updated_sweep":
        probability = float(params["beta_init_probability"])
        model.update(
            {
                "ska_layerscale": True,
                "ska_layerscale_init": float(params["layerscale_init"]),
                "ska_short_conv": True,
                "ska_short_conv_kernel": int(params["short_conv_kernel"]),
                "ska_short_conv_gate_init": float(params["short_conv_gate_init"]),
                "ska_exact_intrachunk": False,
            }
        )
        extensions.update(
            {
                "stats_mode": "candidate_beta_gated",
                "beta_init_probability": probability,
                "beta_init_bias": math.log(probability / (1.0 - probability)),
                "qk_norm": bool(params["qk_norm"]),
                "chunk_semantics": "strict_causal_chunked",
            }
        )
    elif mode == "paper_control":
        model.update(
            {
                "ska_layerscale": False,
                "ska_short_conv": False,
                "ska_exact_intrachunk": False,
                "ska_power_K": 2,
                "ska_ridge": 0.001,
                "ska_eta_learnable": False,
                "ska_eta_value": 1.5,
                "ska_gamma_learnable": True,
            }
        )
        extensions.update(
            {
                "stats_mode": "paper_sequence_max",
                "write_gate": "none",
                "whitening": "two_sided",
                "cross_chunk_boundary": "exclusive",
                "statistics_dtype": "float32",
            }
        )

    realized = {
        "requested_ska_fraction": float(params.get("ska_fraction", 0.0)),
        "realized_ska_count": count,
        "realized_ska_fraction": count / n_layers,
        "ska_layer_indices_zero_based": indices,
        "candidate_first_last_mamba_policy": mode == "updated_sweep",
    }
    extensions["realized_layout"] = realized
    return model, extensions


def build_trial_manifest(
    spec: Mapping[str, Any],
    params: Mapping[str, Any],
    *,
    seed: int | None = None,
    trial_number: int | None = None,
    code_identity: Mapping[str, Any] | None = None,
    data_identity: Mapping[str, Any] | None = None,
    capability_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a complete manifest and a hash over every outcome-affecting field."""

    mode = params.get("architecture_mode")
    if mode == "updated_sweep":
        validate_trial_parameters(spec, params)
    elif mode not in {"paper_control", "mamba_only"}:
        raise Phase2SpecError(f"Unsupported architecture mode: {mode!r}")

    model, extensions = _materialize_model(spec, params)
    weights = derive_objective_weights(params)
    counts = estimate_parameter_counts(model)
    scale = spec["scale"]
    core = counts["non_embedding_core"]
    within_band = int(scale["accepted_min"]) <= core <= int(scale["accepted_max"])
    if not within_band:
        raise Phase2SpecError(
            f"Materialized core count {core:,} is outside "
            f"[{int(scale['accepted_min']):,}, {int(scale['accepted_max']):,}]"
        )

    protocol = deepcopy(spec["protocol"])
    run_seed = int(protocol["seed"] if seed is None else seed)
    protocol["seed"] = run_seed
    optimizer = {
        "name": protocol["optimizer"],
        "base_learning_rate": protocol["base_learning_rate"],
        "weight_decay": protocol["weight_decay"],
        "groups": _effective_lrs(spec, params),
        "assignment_invariants": [
            "mutually_exclusive",
            "collectively_exhaustive",
            "exactly_once",
        ],
    }
    objective = {
        "arm": params["objective_arm"],
        "weights": weights,
        "equal_token_budget_required": True,
    }
    public_spec = _public_spec(spec)
    code = dict(code_identity or git_identity(spec.get("_repo_root", Path.cwd())))
    code_hash_identity = {
        "commit": code.get("commit", "UNKNOWN"),
        "dirty": code.get("dirty", True),
    }
    capabilities = dict(
        capability_identity
        or {
            "status": "REQUIRED_BEFORE_SCIENTIFIC_RUN",
        }
    )
    data = dict(
        data_identity
        or {
            "train_dataset_revision": protocol["train_dataset_revision"],
            "validation_dataset_revision": protocol["validation_dataset_revision"],
            "tokenizer_revision": protocol["tokenizer_revision"],
            "checksums": "REQUIRED_BEFORE_SCIENTIFIC_RUN",
        }
    )

    identity_inputs = {
        "schema_version": 1,
        "study_name": spec["study_name"],
        "spec_hash": stable_hash(public_spec),
        "model_config": model,
        "model_extensions": extensions,
        "optimizer": optimizer,
        "objective": objective,
        "protocol": protocol,
        "fidelity": spec["fidelity"],
        "code": code_hash_identity,
        "capabilities": capabilities,
        "data": data,
    }
    seedless_identity_inputs = deepcopy(identity_inputs)
    seedless_identity_inputs["protocol"].pop("seed", None)
    manifest = {
        "trial_manifest_schema_version": 1,
        "study_name": spec["study_name"],
        "trial_number": trial_number,
        "trial_hash": stable_hash(identity_inputs),
        "promotion_config_hash": stable_hash(seedless_identity_inputs),
        "model_hash": stable_hash(
            {"model_config": model, "model_extensions": extensions}
        ),
        "model_config_hash": stable_hash(model),
        "base_model_config_hash": stable_hash(_load_base_model(spec)),
        "spec_hash": identity_inputs["spec_hash"],
        "status": "MATERIALIZED_NOT_RUN",
        "parameters": dict(params),
        "model_config": model,
        "model_extensions": extensions,
        "parameter_counts_estimated": counts,
        "scale_check": {
            "label": scale["label"],
            "accounting": scale["accounting"],
            "accepted_min": scale["accepted_min"],
            "accepted_max": scale["accepted_max"],
            "estimated_within_band": within_band,
        },
        "optimizer": optimizer,
        "objective": objective,
        "protocol": protocol,
        "fidelity": deepcopy(spec["fidelity"]),
        "required_diagnostics": list(spec["required_diagnostics"]),
        "screen_required_diagnostics": list(spec["screen_required_diagnostics"]),
        "diagnostic_structure": deepcopy(spec["diagnostic_structure"]),
        "hard_failure_rules": deepcopy(spec["hard_failure_rules"]),
        "required_capabilities": list(spec["required_capabilities"]),
        "capability_identity": capabilities,
        "code": code,
        "data": data,
    }
    return manifest


def _stored_identity_inputs(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Return the outcome-affecting identity stored in a trial manifest."""

    code = manifest["code"]
    return {
        "schema_version": 1,
        "study_name": manifest["study_name"],
        "spec_hash": manifest["spec_hash"],
        "model_config": manifest["model_config"],
        "model_extensions": manifest["model_extensions"],
        "optimizer": manifest["optimizer"],
        "objective": manifest["objective"],
        "protocol": manifest["protocol"],
        "fidelity": manifest["fidelity"],
        "code": {
            "commit": code.get("commit", "UNKNOWN"),
            "dirty": code.get("dirty", True),
        },
        "capabilities": manifest["capability_identity"],
        "data": manifest["data"],
    }


def recompute_trial_hash(manifest: Mapping[str, Any]) -> str:
    """Recompute the seed-specific trial hash from a stored manifest."""

    return stable_hash(_stored_identity_inputs(manifest))


def recompute_promotion_config_hash(manifest: Mapping[str, Any]) -> str:
    """Recompute the cross-seed configuration hash from a stored manifest."""

    identity_inputs = deepcopy(_stored_identity_inputs(manifest))
    identity_inputs["protocol"].pop("seed", None)
    return stable_hash(identity_inputs)


def atomic_write_json(path: str | Path, value: Any) -> None:
    """Atomically write strict JSON so partial results cannot look complete."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    fd, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", dir=destination.parent, text=True
    )
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def write_trial_directory(directory: str | Path, manifest: Mapping[str, Any]) -> Path:
    """Write the immutable manifest and trainer-compatible model config."""

    trial_dir = Path(directory)
    trial_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = trial_dir / "trial_manifest.json"
    model_path = trial_dir / "model_config.json"
    if manifest_path.is_file():
        try:
            existing = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise Phase2SpecError(
                f"Existing trial manifest is unreadable: {manifest_path}"
            ) from exc
        if (
            not isinstance(existing, Mapping)
            or existing.get("trial_hash") != manifest.get("trial_hash")
            or recompute_trial_hash(existing) != manifest.get("trial_hash")
            or existing.get("promotion_config_hash")
            != manifest.get("promotion_config_hash")
            or recompute_promotion_config_hash(existing)
            != manifest.get("promotion_config_hash")
        ):
            raise Phase2SpecError(
                f"Refusing to overwrite a different/corrupt trial: {manifest_path}"
            )
    if recompute_trial_hash(manifest) != manifest.get("trial_hash"):
        raise Phase2SpecError("Refusing to write a manifest with an invalid trial hash")
    if recompute_promotion_config_hash(manifest) != manifest.get(
        "promotion_config_hash"
    ):
        raise Phase2SpecError(
            "Refusing to write a manifest with an invalid promotion config hash"
        )
    if model_path.is_file():
        try:
            existing_model = json.loads(model_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise Phase2SpecError(
                f"Existing model config is unreadable: {model_path}"
            ) from exc
        if stable_hash(existing_model) != stable_hash(manifest["model_config"]):
            raise Phase2SpecError(
                f"Refusing to overwrite a different model config: {model_path}"
            )
    atomic_write_json(manifest_path, manifest)
    atomic_write_json(model_path, manifest["model_config"])
    return trial_dir


__all__ = [
    "atomic_write_json",
    "build_trial_manifest",
    "control_parameters",
    "git_identity",
    "reference_parameters",
    "recompute_promotion_config_hash",
    "recompute_trial_hash",
    "stable_hash",
    "write_trial_directory",
]
