"""Materialize Phase 2a trials without importing or running the model."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

try:
    from .manifest import (
        atomic_write_json,
        build_trial_manifest,
        control_parameters,
        git_identity,
        write_trial_directory,
    )
    from .spec import Phase2SpecError, load_spec, sample_parameters
except ImportError:  # Supports `python path/to/dry_run.py`.
    from manifest import (  # type: ignore
        atomic_write_json,
        build_trial_manifest,
        control_parameters,
        git_identity,
        write_trial_directory,
    )
    from spec import Phase2SpecError, load_spec, sample_parameters  # type: ignore


def materialize_dry_run(
    spec_path: str | Path,
    *,
    trials: int,
    seed: int,
    output: str | Path | None = None,
    include_controls: bool = True,
) -> dict[str, Any]:
    if trials < 0:
        raise Phase2SpecError("trials must be nonnegative")
    spec = load_spec(spec_path)
    rng = random.Random(seed)
    code = git_identity(spec["_repo_root"])
    manifests: list[dict[str, Any]] = []
    hashes: set[str] = set()

    def add(params: dict[str, Any], number: int | None) -> None:
        manifest = build_trial_manifest(
            spec,
            params,
            seed=int(spec["protocol"]["seed"]),
            trial_number=number,
            code_identity=code,
        )
        if manifest["trial_hash"] in hashes:
            raise Phase2SpecError(
                f"Duplicate trial hash in dry run: {manifest['trial_hash']}"
            )
        hashes.add(manifest["trial_hash"])
        manifests.append(manifest)

    if include_controls:
        for control in spec["fixed_controls"]:
            add(control_parameters(spec, control["name"]), None)

    for trial_number in range(trials):
        # A large categorical space makes collision unlikely. Still retry so a
        # dry-run request has deterministic cardinality.
        for _ in range(100):
            params = sample_parameters(spec, rng)
            manifest = build_trial_manifest(
                spec,
                params,
                seed=int(spec["protocol"]["seed"]),
                trial_number=trial_number,
                code_identity=code,
            )
            if manifest["trial_hash"] not in hashes:
                hashes.add(manifest["trial_hash"])
                manifests.append(manifest)
                break
        else:
            raise Phase2SpecError("Could not draw a unique trial after 100 attempts")

    layouts = {
        (
            manifest["parameters"]["architecture_mode"],
            tuple(
                manifest["model_extensions"]["realized_layout"][
                    "ska_layer_indices_zero_based"
                ]
            ),
        )
        for manifest in manifests
    }
    cores = [
        manifest["parameter_counts_estimated"]["non_embedding_core"]
        for manifest in manifests
    ]
    totals = [
        manifest["parameter_counts_estimated"]["total"] for manifest in manifests
    ]
    birdie_count = sum(
        manifest["objective"]["arm"] == "birdie_mix" for manifest in manifests
    )
    summary = {
        "study_name": spec["study_name"],
        "spec_hash": manifests[0]["spec_hash"] if manifests else None,
        "sampled_trials": trials,
        "fixed_controls": len(spec["fixed_controls"]) if include_controls else 0,
        "total_manifests": len(manifests),
        "unique_trial_hashes": len(hashes),
        "unique_realized_layouts": len(layouts),
        "birdie_trials": birdie_count,
        "core_parameter_range": [min(cores), max(cores)] if cores else None,
        "total_parameter_range": [min(totals), max(totals)] if totals else None,
        "scientific_launch_ready": False,
        "scientific_launch_note": (
            "Dry-run success validates materialization only. Run preflight against "
            "the finalized capability manifest before training."
        ),
    }

    if output is not None:
        root = Path(output)
        for index, manifest in enumerate(manifests):
            label = (
                f"control-{manifest['parameters'].get('name', index)}"
                if manifest["trial_number"] is None
                else f"trial-{manifest['trial_number']:05d}"
            )
            write_trial_directory(root / "trials" / label, manifest)
        atomic_write_json(root / "dry_run_summary.json", summary)
    return {"summary": summary, "manifests": manifests}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", default="configs/phase2a_search.json")
    parser.add_argument("--trials", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output")
    parser.add_argument("--no-controls", action="store_true")
    parser.add_argument(
        "--show-manifests",
        action="store_true",
        help="Print all manifests instead of only the summary.",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = materialize_dry_run(
        args.spec,
        trials=args.trials,
        seed=args.seed,
        output=args.output,
        include_controls=not args.no_controls,
    )
    print(
        json.dumps(
            result if args.show_manifests else result["summary"],
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
