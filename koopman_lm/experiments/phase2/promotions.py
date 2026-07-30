"""Materialize missing promotion seeds from a Phase 2a candidate shortlist."""

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
        write_trial_directory,
    )
    from .preflight import evaluate_preflight
    from .spec import load_spec
except ImportError:
    from manifest import (  # type: ignore
        atomic_write_json,
        build_trial_manifest,
        git_identity,
        write_trial_directory,
    )
    from preflight import evaluate_preflight  # type: ignore
    from spec import load_spec  # type: ignore


class PromotionMaterializationError(ValueError):
    """A shortlist cannot be reproduced under the finalized study identity."""


def _selected_records(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, Mapping) and isinstance(
        value.get("promotion_shortlist"), Mapping
    ):
        value = value["promotion_shortlist"]
    if isinstance(value, Mapping):
        value = value.get("selected")
    if not isinstance(value, list) or not value:
        raise PromotionMaterializationError(
            "Shortlist must contain a nonempty selected list"
        )
    if not all(isinstance(record, Mapping) for record in value):
        raise PromotionMaterializationError("Every shortlist item must be an object")
    return [dict(record) for record in value]


def materialize_promotions(
    spec_path: str | Path,
    shortlist: Mapping[str, Any] | list[Any],
    output: str | Path,
    *,
    data_identity: Mapping[str, Any] | None = None,
    capability_identity: Mapping[str, Any] | None = None,
    scientific_ready: bool = False,
) -> dict[str, Any]:
    spec = load_spec(spec_path)
    selected = _selected_records(shortlist)
    if len(selected) > 12:
        raise PromotionMaterializationError(
            "Promotion shortlist exceeds the Scaling Plan's 8-12 candidate range"
        )
    root = Path(output)
    code = git_identity(spec["_repo_root"])
    required_seeds = list(spec["study"]["promotion_seeds"])
    execution: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    seen_configs: set[str] = set()
    prepared: list[
        tuple[int, Mapping[str, Any], str, int, str, list[dict[str, Any]]]
    ] = []

    # Validate the entire shortlist before creating any run directory. This
    # prevents a duplicate or corrupt later item from leaving a partial tree.
    for index, record in enumerate(selected, start=1):
        params = record.get("parameters")
        if not isinstance(params, Mapping):
            raise PromotionMaterializationError(
                f"Shortlist item {index} has no materialized parameters"
            )
        expected_config_hash = record.get("promotion_config_hash")
        if (
            not isinstance(expected_config_hash, str)
            or len(expected_config_hash) != 64
            or any(
                character not in "0123456789abcdef"
                for character in expected_config_hash
            )
        ):
            raise PromotionMaterializationError(
                f"Shortlist item {index} requires a valid promotion_config_hash"
            )
        source_seed = record.get("seed")
        source_hash = record.get("trial_hash")
        if (
            source_seed not in required_seeds
            or not isinstance(source_seed, int)
            or isinstance(source_seed, bool)
            or not isinstance(source_hash, str)
            or len(source_hash) != 64
            or any(character not in "0123456789abcdef" for character in source_hash)
            or record.get("status") != "COMPLETE"
            or record.get("health_passed") is not True
        ):
            raise PromotionMaterializationError(
                f"Shortlist item {index} must identify a healthy COMPLETE "
                "source run at a required seed"
            )
        manifests: list[dict[str, Any]] = []
        for seed in required_seeds:
            manifest = build_trial_manifest(
                spec,
                params,
                seed=seed,
                code_identity=code,
                data_identity=data_identity,
                capability_identity=capability_identity,
            )
            config_hash = manifest["promotion_config_hash"]
            if config_hash != expected_config_hash:
                raise PromotionMaterializationError(
                    "Shortlist promotion_config_hash does not match the current "
                    f"spec/code/data/capability identity for rank {index}"
                )
            manifests.append(manifest)
        source_manifest = next(
            manifest
            for manifest in manifests
            if manifest["protocol"]["seed"] == source_seed
        )
        if source_manifest["trial_hash"] != source_hash:
            raise PromotionMaterializationError(
                f"Shortlist item {index} source trial hash does not match its "
                "current spec/code/data/capability identity"
            )
        if expected_config_hash in seen_configs:
            raise PromotionMaterializationError(
                f"Duplicate promoted configuration: {expected_config_hash}"
            )
        seen_configs.add(expected_config_hash)
        prepared.append(
            (
                index,
                params,
                expected_config_hash,
                source_seed,
                source_hash,
                manifests,
            )
        )

    for index, _, config_hash, source_seed, source_hash, manifests in prepared:
        sources.append(
            {
                "promotion_rank": index,
                "promotion_config_hash": config_hash,
                "seed": source_seed,
                "trial_hash": source_hash,
                "status": "existing_complete_required",
            }
        )
        for manifest in manifests:
            seed = int(manifest["protocol"]["seed"])
            if seed == source_seed:
                continue
            relative = Path(f"candidate-{index:02d}-{config_hash[:12]}") / (
                f"seed-{seed}"
            )
            write_trial_directory(root / relative, manifest)
            execution.append(
                {
                    "promotion_rank": index,
                    "promotion_config_hash": config_hash,
                    "seed": seed,
                    "trial_hash": manifest["trial_hash"],
                    "run_directory": str(relative),
                    "required_stage": "final",
                }
            )

    summary = {
        "schema_version": 1,
        "study_name": spec["study_name"],
        "scientific_launch_ready": scientific_ready,
        "selected_configuration_count": len(selected),
        "required_seeds": required_seeds,
        "existing_source_runs": sources,
        "new_run_count": len(execution),
        "execution_plan": execution,
    }
    atomic_write_json(root / "promotion_run_summary.json", summary)
    return summary


def _read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", default="configs/phase2a_search.json")
    parser.add_argument("--shortlist", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--capabilities")
    parser.add_argument("--data-manifest")
    args = parser.parse_args()
    if bool(args.capabilities) != bool(args.data_manifest):
        raise SystemExit("--capabilities and --data-manifest must be supplied together")

    shortlist = _read_json(args.shortlist)
    data = capabilities = None
    scientific_ready = False
    if args.capabilities:
        spec = load_spec(args.spec)
        capabilities = _read_json(args.capabilities)
        data = _read_json(args.data_manifest)
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
    try:
        summary = materialize_promotions(
            args.spec,
            shortlist,
            args.output,
            data_identity=data,
            capability_identity=capabilities,
            scientific_ready=scientific_ready,
        )
    except PromotionMaterializationError as exc:
        raise SystemExit(f"Could not materialize promotions: {exc}") from exc
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
