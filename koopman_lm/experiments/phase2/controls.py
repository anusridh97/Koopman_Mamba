"""Materialize the four fixed Phase 2a controls at all promotion seeds."""

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
        write_trial_directory,
    )
    from .preflight import evaluate_preflight
    from .spec import load_spec
except ImportError:
    from manifest import (  # type: ignore
        atomic_write_json,
        build_trial_manifest,
        control_parameters,
        git_identity,
        write_trial_directory,
    )
    from preflight import evaluate_preflight  # type: ignore
    from spec import load_spec  # type: ignore


def materialize_controls(
    spec_path: str | Path,
    output: str | Path,
    *,
    data_identity: Mapping[str, Any] | None = None,
    capability_identity: Mapping[str, Any] | None = None,
    scientific_ready: bool = False,
) -> dict[str, Any]:
    spec = load_spec(spec_path)
    code = git_identity(spec["_repo_root"])
    root = Path(output)
    entries: list[dict[str, Any]] = []
    for seed in spec["study"]["promotion_seeds"]:
        for control in spec["fixed_controls"]:
            name = control["name"]
            manifest = build_trial_manifest(
                spec,
                control_parameters(spec, name),
                seed=int(seed),
                code_identity=code,
                data_identity=data_identity,
                capability_identity=capability_identity,
            )
            relative = Path(f"seed-{seed}") / name
            write_trial_directory(root / relative, manifest)
            entries.append(
                {
                    "seed": seed,
                    "control": name,
                    "trial_hash": manifest["trial_hash"],
                    "run_directory": str(relative),
                    "required_stage": "final",
                }
            )
    summary = {
        "schema_version": 1,
        "study_name": spec["study_name"],
        "scientific_launch_ready": scientific_ready,
        "control_run_count": len(entries),
        "seeds": list(spec["study"]["promotion_seeds"]),
        "controls": [control["name"] for control in spec["fixed_controls"]],
        "execution_plan": entries,
    }
    atomic_write_json(root / "control_summary.json", summary)
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
    parser.add_argument("--capabilities")
    parser.add_argument("--data-manifest")
    args = parser.parse_args()
    if bool(args.capabilities) != bool(args.data_manifest):
        raise SystemExit("--capabilities and --data-manifest must be supplied together")

    data = capabilities = None
    scientific_ready = False
    if args.capabilities:
        spec = load_spec(args.spec)
        capabilities = _read_object(args.capabilities)
        data = _read_object(args.data_manifest)
        report = evaluate_preflight(
            spec,
            capabilities,
            data_manifest=data,
            stage="study",
            require_clean=True,
        )
        if not report["ready"]:
            print(json.dumps(report, indent=2, sort_keys=True))
            raise SystemExit("Study-stage preflight failed")
        scientific_ready = True
    summary = materialize_controls(
        args.spec,
        args.output,
        data_identity=data,
        capability_identity=capabilities,
        scientific_ready=scientific_ready,
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
