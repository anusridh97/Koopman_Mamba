"""Materialize three non-scientific, screen-only Phase 2a calibration arms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

try:
    from .manifest import (
        atomic_write_json,
        build_trial_manifest,
        git_identity,
        reference_parameters,
        write_trial_directory,
    )
    from .preflight import evaluate_preflight
    from .spec import load_spec, stable_hash
except ImportError:
    from manifest import (  # type: ignore
        atomic_write_json,
        build_trial_manifest,
        git_identity,
        reference_parameters,
        write_trial_directory,
    )
    from preflight import evaluate_preflight  # type: ignore
    from spec import load_spec, stable_hash  # type: ignore


def calibration_parameter_matrix() -> list[tuple[str, dict[str, Any]]]:
    """Return the exact three fixed calibration arms."""

    return [
        (
            "default_updated_sweep",
            reference_parameters(),
        ),
        (
            "worst_case_stability",
            reference_parameters(
                {
                    "ska_rank": 128,
                    "ska_ridge": 0.0001,
                    "ska_chunk_size": 128,
                    "qk_norm": True,
                }
            ),
        ),
        (
            "birdie_mix",
            reference_parameters(
                {
                    "objective_arm": "birdie_mix",
                    "birdie_retrieval_fraction": 0.3,
                    "birdie_copy_share": 0.5,
                }
            ),
        ),
    ]


def materialize_calibration(
    spec_path: str | Path,
    output: str | Path,
    *,
    data_identity: Mapping[str, Any],
    capability_identity: Mapping[str, Any],
    calibration_preflight_ready: bool = False,
) -> dict[str, Any]:
    """Write exactly three identity-bound manifests for one-GPU timing checks."""

    spec = load_spec(spec_path)
    if int(spec["protocol"]["world_size"]) != 1:
        raise ValueError("Calibration is restricted to exactly one GPU")
    if not isinstance(data_identity, Mapping) or not data_identity:
        raise ValueError("Calibration requires a nonempty frozen data identity")
    if not isinstance(capability_identity, Mapping) or not capability_identity:
        raise ValueError("Calibration requires a nonempty capability identity")

    code = git_identity(spec["_repo_root"])
    root = Path(output)
    entries: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    for index, (name, parameters) in enumerate(calibration_parameter_matrix()):
        manifest = build_trial_manifest(
            spec,
            parameters,
            seed=int(spec["protocol"]["seed"]),
            trial_number=index,
            code_identity=code,
            data_identity=data_identity,
            capability_identity=capability_identity,
        )
        trial_hash = str(manifest["trial_hash"])
        if trial_hash in seen_hashes:
            raise RuntimeError(
                f"Calibration arms are not outcome-distinct: {name} -> {trial_hash}"
            )
        seen_hashes.add(trial_hash)
        relative_run_dir = Path("trials") / f"{index:02d}-{name}"
        write_trial_directory(root / relative_run_dir, manifest)
        manifests.append(manifest)
        entries.append(
            {
                "arm": name,
                "trial_hash": trial_hash,
                "run_directory": str(relative_run_dir),
                "required_stage": "screen",
            }
        )

    if len(manifests) != 3:
        raise AssertionError("Calibration must materialize exactly three manifests")
    spec_hashes = {str(manifest["spec_hash"]) for manifest in manifests}
    if len(spec_hashes) != 1:
        raise AssertionError("Calibration manifests disagree on spec identity")

    summary = {
        "schema_version": 1,
        "study_name": spec["study_name"],
        "calibration_manifest_count": 3,
        "gpu_count_per_run": 1,
        "required_stage_counts": {"screen": 3, "final": 0},
        "trial_hashes": [manifest["trial_hash"] for manifest in manifests],
        "execution_plan": entries,
        "spec_hash": next(iter(spec_hashes)),
        "data_hash": stable_hash(data_identity),
        "capability_hash": stable_hash(capability_identity),
        "code_commit": code.get("commit", "UNKNOWN"),
        "code_dirty": code.get("dirty", True),
        "preflight_stage": "calibration",
        "calibration_preflight_ready": bool(calibration_preflight_ready),
        "scientific_launch_ready": False,
        "non_scientific_notice": (
            "Calibration results are for one-GPU runtime and stability planning "
            "only; they are not Phase 2a scientific sweep results."
        ),
    }
    atomic_write_json(root / "calibration_summary.json", summary)
    return summary


def _read_object(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", default="configs/phase2a_search.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--capabilities", required=True)
    parser.add_argument("--data-manifest", required=True)
    args = parser.parse_args()

    spec = load_spec(args.spec)
    capabilities = _read_object(args.capabilities)
    data = _read_object(args.data_manifest)
    report = evaluate_preflight(
        spec,
        capabilities,
        data_manifest=data,
        stage="calibration",
        require_clean=True,
    )
    if not report["ready"]:
        print(json.dumps(report, indent=2, sort_keys=True))
        raise SystemExit("Calibration preflight failed")
    summary = materialize_calibration(
        args.spec,
        args.output,
        data_identity=data,
        capability_identity=capabilities,
        calibration_preflight_ready=True,
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
