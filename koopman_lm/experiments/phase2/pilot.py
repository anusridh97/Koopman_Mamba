"""Build the fixed pipeline-validation pilot matrix required before Phase 2a."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

try:
    from .manifest import (
        atomic_write_json,
        build_trial_manifest,
        control_parameters,
        git_identity,
        reference_parameters,
        write_trial_directory,
    )
    from .spec import load_spec
    from .preflight import evaluate_preflight
except ImportError:
    from manifest import (  # type: ignore
        atomic_write_json,
        build_trial_manifest,
        control_parameters,
        git_identity,
        reference_parameters,
        write_trial_directory,
    )
    from spec import load_spec  # type: ignore
    from preflight import evaluate_preflight  # type: ignore


def pilot_parameter_matrix(spec: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Cover controls, all layouts, risky extremes, QKNorm, and Birdie."""

    matrix: list[dict[str, Any]] = [
        control_parameters(spec, control["name"])
        for control in spec["fixed_controls"]
    ]

    # All fraction/placement combinations ensure every layer-layout algorithm
    # is exercised at every requested count.
    for fraction in spec["axes"]["ska_fraction"]["values"]:
        for placement in spec["axes"]["ska_placement"]["values"]:
            matrix.append(
                reference_parameters(
                    {
                        "pilot_reason": f"layout-f{fraction}-{placement}",
                        "ska_fraction": fraction,
                        "ska_placement": placement,
                    }
                )
            )

    risky = [
        {"pilot_reason": "minimum-rank", "ska_rank": 32},
        {"pilot_reason": "maximum-rank", "ska_rank": 128},
        {"pilot_reason": "minimum-chunk", "ska_chunk_size": 32},
        {"pilot_reason": "maximum-chunk", "ska_chunk_size": 128},
        {
            "pilot_reason": "rank128-ridge1e-4",
            "ska_rank": 128,
            "ska_ridge": 0.0001,
        },
        {"pilot_reason": "qknorm-on", "qk_norm": True},
        {
            "pilot_reason": "birdie-mix",
            "objective_arm": "birdie_mix",
            "birdie_retrieval_fraction": 0.3,
            "birdie_copy_share": 0.5,
        },
    ]
    matrix.extend(reference_parameters(overrides) for overrides in risky)
    return matrix


def materialize_pilot(
    spec_path: str | Path,
    output: str | Path,
    *,
    seed: int | None = None,
    data_identity: Mapping[str, Any] | None = None,
    capability_identity: Mapping[str, Any] | None = None,
    scientific_ready: bool = False,
) -> dict[str, Any]:
    spec = load_spec(spec_path)
    code = git_identity(spec["_repo_root"])
    root = Path(output)
    effective_seed = int(spec["protocol"]["seed"] if seed is None else seed)
    manifests: list[dict[str, Any]] = []
    execution_plan: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, params in enumerate(pilot_parameter_matrix(spec)):
        manifest = build_trial_manifest(
            spec,
            params,
            seed=effective_seed,
            trial_number=index,
            code_identity=code,
            data_identity=data_identity,
            capability_identity=capability_identity,
        )
        if manifest["trial_hash"] in seen:
            continue
        seen.add(manifest["trial_hash"])
        reason = str(
            params.get("pilot_reason")
            or params.get("name")
            or f"pilot-{index:03d}"
        ).replace("/", "-")
        relative_run_dir = Path("trials") / f"{index:03d}-{reason}"
        write_trial_directory(root / relative_run_dir, manifest)
        manifests.append(manifest)
        requires_final = (
            bool(params.get("name"))
            or reason
            in {
                "rank128-ridge1e-4",
                "qknorm-on",
                "birdie-mix",
            }
        )
        execution_plan.append(
            {
                "trial_hash": manifest["trial_hash"],
                "reason": reason,
                "run_directory": str(relative_run_dir),
                "required_stage": "final" if requires_final else "screen",
            }
        )

    summary = {
        "study_name": spec["study_name"],
        "seed": effective_seed,
        "code": code,
        "pilot_manifest_count": len(manifests),
        "trial_hashes": [manifest["trial_hash"] for manifest in manifests],
        "execution_plan": execution_plan,
        "required_stage_counts": {
            "screen": sum(
                entry["required_stage"] == "screen"
                for entry in execution_plan
            ),
            "final": sum(
                entry["required_stage"] == "final"
                for entry in execution_plan
            ),
        },
        "coverage": {
            "architecture_modes": sorted(
                {m["parameters"]["architecture_mode"] for m in manifests}
            ),
            "ranks": sorted(
                {
                    m["parameters"]["ska_rank"]
                    for m in manifests
                    if m["parameters"]["architecture_mode"] == "updated_sweep"
                }
            ),
            "fractions": sorted(
                {
                    m["parameters"]["ska_fraction"]
                    for m in manifests
                    if m["parameters"]["architecture_mode"] == "updated_sweep"
                }
            ),
            "chunks": sorted(
                {
                    m["parameters"]["ska_chunk_size"]
                    for m in manifests
                    if m["parameters"]["architecture_mode"] == "updated_sweep"
                }
            ),
            "placements": sorted(
                {
                    m["parameters"]["ska_placement"]
                    for m in manifests
                    if m["parameters"]["architecture_mode"] == "updated_sweep"
                }
            ),
            "qk_norm": sorted(
                {
                    m["parameters"]["qk_norm"]
                    for m in manifests
                    if m["parameters"]["architecture_mode"] == "updated_sweep"
                }
            ),
            "objective_arms": sorted(
                {m["parameters"]["objective_arm"] for m in manifests}
            ),
        },
        "scientific_launch_ready": scientific_ready,
        "next_gate": (
            "Execute through the finalized adapter."
            if scientific_ready
            else "Run finalized capability preflight before executing this pilot."
        ),
    }
    atomic_write_json(root / "pilot_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", default="configs/phase2a_search.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--capabilities")
    parser.add_argument("--data-manifest")
    args = parser.parse_args()
    if bool(args.capabilities) != bool(args.data_manifest):
        raise SystemExit("--capabilities and --data-manifest must be supplied together")
    data_identity = capability_identity = None
    scientific_ready = False
    if args.capabilities:
        spec = load_spec(args.spec)
        capability_identity = json.loads(Path(args.capabilities).read_text())
        data_identity = json.loads(Path(args.data_manifest).read_text())
        preflight = evaluate_preflight(
            spec,
            capability_identity,
            data_manifest=data_identity,
            stage="pilot",
            require_clean=True,
        )
        if not preflight["ready"]:
            print(json.dumps(preflight, indent=2, sort_keys=True))
            raise SystemExit("Scientific launch preflight failed")
        scientific_ready = True
    summary = materialize_pilot(
        args.spec,
        args.output,
        seed=args.seed,
        data_identity=data_identity,
        capability_identity=capability_identity,
        scientific_ready=scientific_ready,
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
