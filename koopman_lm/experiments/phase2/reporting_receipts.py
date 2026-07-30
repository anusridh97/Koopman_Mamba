"""Finalize scientist review and W&B publication receipts for Phase 2a.

This module is deliberately standard-library-only. It does not upload to W&B.
It accepts only a completed storage-snapshot receipt bound to the exact
analysis bytes and a separate W&B verification artifact whose report URL and
run-ID set exactly match the offline publication payload.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import stat
from typing import Any, Mapping
from urllib.parse import urlsplit

try:
    from .manifest import atomic_write_json
    from .spec import stable_hash
except ImportError:  # Direct execution in a minimal environment.
    from manifest import atomic_write_json  # type: ignore
    from spec import stable_hash  # type: ignore


EXPECTED_HYPOTHESIS_IDS = (
    "rank_scales_with_width",
    "ska_fraction_and_placement",
    "chunk_size_matters",
    "differential_lr_ratios_transfer",
    "ridge_1e3_is_sufficient",
    "beta_initialization",
    "layerscale_initialization",
    "short_conv_design",
    "qknorm_required",
    "birdie_objectives_dominate",
)

_STUDY_IDENTITY_FIELDS = {
    "study_name",
    "spec_hash",
    "code_commit",
    "code_dirty",
    "data_hash",
    "capability_hash",
}
_SHA256_FIELDS = {"spec_hash", "data_hash", "capability_hash"}
_REVIEWED_FILENAME = "reviewed_falsified_hypotheses.json"
_PUBLICATION_FILENAME = "wandb_publication_receipt.json"
_STORAGE_RECEIPT_KIND = "phase2_optuna_storage_snapshot"
_WANDB_VERIFICATION_KIND = "phase2a_wandb_report_verification"
_STORAGE_BACKENDS = {"postgresql", "scg_validated_journal_storage"}


class ReportingReceiptError(ValueError):
    """Raised when a completion receipt cannot be issued safely."""


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _read_json_object(
    path: str | Path,
    *,
    label: str,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    try:
        with source.open("rb") as handle:
            before = os.fstat(handle.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise ReportingReceiptError(
                    f"{label} must be a regular file: {source}"
                )
            raw = handle.read()
            after = os.fstat(handle.fileno())
    except FileNotFoundError as exc:
        raise ReportingReceiptError(f"{label} does not exist: {source}") from exc
    except OSError as exc:
        raise ReportingReceiptError(f"{label} cannot be read: {source}: {exc}") from exc
    identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    if identity != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise ReportingReceiptError(f"{label} changed while it was being read")
    try:
        current = source.stat()
    except OSError as exc:
        raise ReportingReceiptError(
            f"{label} changed while it was being read: {source}: {exc}"
        ) from exc
    if identity != (
        current.st_dev,
        current.st_ino,
        current.st_size,
        current.st_mtime_ns,
    ):
        raise ReportingReceiptError(f"{label} path changed while it was being read")
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ReportingReceiptError(
            f"{label} is not valid JSON: {source}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise ReportingReceiptError(f"{label} must be a JSON object: {source}")
    if not raw:
        raise ReportingReceiptError(f"{label} must not be empty: {source}")
    binding = {
        "filename": source.name,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "byte_size": len(raw),
    }
    return source, value, binding


def _study_identity(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ReportingReceiptError("analysis.study_identity must be an object")
    identity = dict(value)
    fields = set(identity)
    if fields != _STUDY_IDENTITY_FIELDS:
        missing = sorted(_STUDY_IDENTITY_FIELDS - fields)
        extra = sorted(fields - _STUDY_IDENTITY_FIELDS)
        raise ReportingReceiptError(
            "analysis.study_identity fields do not match the frozen contract: "
            f"missing={missing}, extra={extra}"
        )
    if not isinstance(identity["study_name"], str) or not identity[
        "study_name"
    ].strip():
        raise ReportingReceiptError("study_identity.study_name must be non-empty")
    if (
        not isinstance(identity["code_commit"], str)
        or len(identity["code_commit"]) < 7
    ):
        raise ReportingReceiptError(
            "study_identity.code_commit must contain at least seven characters"
        )
    if identity["code_dirty"] is not False:
        raise ReportingReceiptError(
            "final reporting is forbidden for a dirty code identity"
        )
    for field in sorted(_SHA256_FIELDS):
        if not _is_sha256(identity[field]):
            raise ReportingReceiptError(
                f"study_identity.{field} must be a lowercase SHA-256"
            )
    return identity


def _hypothesis_ids(
    analysis: Mapping[str, Any],
    *,
    study_identity: Mapping[str, Any],
) -> tuple[str, ...]:
    report = analysis.get("falsified_hypotheses")
    if not isinstance(report, Mapping):
        raise ReportingReceiptError(
            "analysis.falsified_hypotheses must be an object"
        )
    if report.get("report_status") != "scientist_review_required":
        raise ReportingReceiptError(
            "analysis falsified-hypotheses scaffold has an unexpected status"
        )
    if report.get("study_identity") != dict(study_identity):
        raise ReportingReceiptError(
            "falsified-hypotheses scaffold study identity differs from the analysis"
        )
    hypotheses = report.get("hypotheses")
    if not isinstance(hypotheses, list):
        raise ReportingReceiptError(
            "analysis.falsified_hypotheses.hypotheses must be an array"
        )
    ids: list[str] = []
    for index, item in enumerate(hypotheses):
        if not isinstance(item, Mapping) or not isinstance(item.get("id"), str):
            raise ReportingReceiptError(
                f"analysis hypothesis {index} must have a string id"
            )
        if item.get("falsified") is not None:
            raise ReportingReceiptError(
                "the source analysis must remain an unreviewed evidence scaffold"
            )
        ids.append(item["id"])
    if len(ids) != len(set(ids)):
        raise ReportingReceiptError("analysis hypothesis ids contain duplicates")
    if set(ids) != set(EXPECTED_HYPOTHESIS_IDS):
        raise ReportingReceiptError(
            "analysis hypothesis ids do not match the Phase 2a contract: "
            f"expected={list(EXPECTED_HYPOTHESIS_IDS)}, observed={ids}"
        )
    return tuple(ids)


def _validate_offline_payload(
    payload: Mapping[str, Any],
    *,
    study_identity: Mapping[str, Any],
) -> None:
    if payload.get("schema_version") != 1:
        raise ReportingReceiptError(
            "offline W&B payload schema_version must equal 1"
        )
    if payload.get("kind") != "wandb_report_payload":
        raise ReportingReceiptError(
            "offline W&B artifact is not a wandb_report_payload"
        )
    if payload.get("network_write_performed") is not False:
        raise ReportingReceiptError(
            "offline W&B payload must state network_write_performed=false"
        )
    if payload.get("publication_status") != "manual_authorized_upload_required":
        raise ReportingReceiptError(
            "offline W&B payload publication_status is not awaiting manual upload"
        )
    if payload.get("published_report_url") is not None:
        raise ReportingReceiptError(
            "offline W&B payload must not claim a published report URL"
        )
    if payload.get("study_identity") != dict(study_identity):
        raise ReportingReceiptError(
            "offline W&B payload study identity differs from the analysis"
        )


def _payload_wandb_run_ids(payload: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the canonical run-ID set referenced by the offline report."""

    tables = payload.get("tables")
    if not isinstance(tables, Mapping):
        raise ReportingReceiptError(
            "offline W&B payload tables must be an object"
        )
    run_ids: set[str] = set()
    run_id_table_count = 0
    for table_name, table in tables.items():
        if not isinstance(table, Mapping):
            raise ReportingReceiptError(
                f"offline W&B table {table_name!r} must be an object"
            )
        columns = table.get("columns")
        rows = table.get("data")
        if (
            not isinstance(columns, list)
            or not all(isinstance(column, str) for column in columns)
            or len(columns) != len(set(columns))
            or not isinstance(rows, list)
        ):
            raise ReportingReceiptError(
                f"offline W&B table {table_name!r} has an invalid table shape"
            )
        if "wandb_run_id" not in columns:
            continue
        run_id_table_count += 1
        run_id_index = columns.index("wandb_run_id")
        for row_index, row in enumerate(rows):
            if not isinstance(row, list) or len(row) != len(columns):
                raise ReportingReceiptError(
                    f"offline W&B table {table_name!r} row {row_index} "
                    "does not match its columns"
                )
            run_id = row[run_id_index]
            if (
                not isinstance(run_id, str)
                or not run_id.strip()
                or run_id != run_id.strip()
            ):
                raise ReportingReceiptError(
                    f"offline W&B table {table_name!r} row {row_index} "
                    "has no canonical wandb_run_id"
                )
            run_ids.add(run_id)
    if run_id_table_count == 0 or not run_ids:
        raise ReportingReceiptError(
            "offline W&B payload contains no nonempty wandb_run_id set"
        )
    return tuple(sorted(run_ids))


def _validate_analysis(
    analysis: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> tuple[dict[str, Any], frozenset[str], tuple[str, ...]]:
    if analysis.get("schema_version") != 1:
        raise ReportingReceiptError("analysis schema_version must equal 1")
    identity = _study_identity(analysis.get("study_identity"))
    fanova = analysis.get("fanova_importance")
    if not isinstance(fanova, Mapping):
        raise ReportingReceiptError(
            "analysis is not final: fanova_importance must be present"
        )
    if fanova.get("study_identity") != identity:
        raise ReportingReceiptError(
            "fANOVA study identity differs from the analysis"
        )
    views = fanova.get("views")
    if not isinstance(views, Mapping):
        raise ReportingReceiptError("fANOVA views are missing from the analysis")
    for view_name in (
        "global",
        "birdie_only",
        "next_token_only",
        "updated_sweep_only",
    ):
        view = views.get(view_name)
        if not isinstance(view, Mapping):
            raise ReportingReceiptError(
                f"final fANOVA is missing the {view_name} view"
            )
        for metric in ("mqar_accuracy", "wikitext_ppl"):
            result = view.get(metric)
            importance = (
                result.get("importance")
                if isinstance(result, Mapping)
                else None
            )
            if (
                not isinstance(result, Mapping)
                or result.get("status") != "ok"
                or not isinstance(importance, Mapping)
                or not importance
                or any(
                    not isinstance(value, (int, float))
                    or isinstance(value, bool)
                    or not math.isfinite(float(value))
                    for value in importance.values()
                )
            ):
                raise ReportingReceiptError(
                    f"final fANOVA {view_name}/{metric} is unavailable or empty"
                )
    for metric in ("mqar_accuracy", "wikitext_ppl"):
        adjusted = views["global"][metric].get("capacity_adjusted")
        importance = (
            adjusted.get("importance")
            if isinstance(adjusted, Mapping)
            else None
        )
        if (
            not isinstance(adjusted, Mapping)
            or adjusted.get("status") != "ok"
            or not isinstance(importance, Mapping)
            or not importance
        ):
            raise ReportingReceiptError(
                f"final capacity-adjusted global fANOVA for {metric} "
                "is unavailable or empty"
            )
    trial_hashes = analysis.get("trial_hash_index")
    if (
        not isinstance(trial_hashes, list)
        or not trial_hashes
        or not all(_is_sha256(item) for item in trial_hashes)
        or len(trial_hashes) != len(set(trial_hashes))
    ):
        raise ReportingReceiptError(
            "analysis.trial_hash_index must be a nonempty unique SHA-256 list"
        )
    _hypothesis_ids(analysis, study_identity=identity)
    embedded_payload = analysis.get("wandb_report_payload")
    if not isinstance(embedded_payload, Mapping):
        raise ReportingReceiptError(
            "analysis.wandb_report_payload must be an object"
        )
    if dict(embedded_payload) != dict(payload):
        raise ReportingReceiptError(
            "offline W&B payload does not exactly match the payload embedded "
            "in the analysis"
        )
    _validate_offline_payload(payload, study_identity=identity)
    return (
        identity,
        frozenset(trial_hashes),
        _payload_wandb_run_ids(payload),
    )


def _nonempty_text(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ReportingReceiptError(f"{label} must be a non-empty string")
    return value.strip()


def _utc_timestamp(value: Any, *, label: str) -> str:
    timestamp = _nonempty_text(value, label=label)
    if not timestamp.endswith("Z"):
        raise ReportingReceiptError(f"{label} must be an RFC 3339 UTC timestamp")
    try:
        parsed = datetime.fromisoformat(f"{timestamp[:-1]}+00:00")
    except ValueError as exc:
        raise ReportingReceiptError(
            f"{label} must be an RFC 3339 UTC timestamp"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ReportingReceiptError(f"{label} must use the UTC Z designator")
    return timestamp


def _review_decisions(
    value: Mapping[str, Any],
    *,
    allowed_trial_hashes: frozenset[str],
) -> list[dict[str, Any]]:
    if set(value) != {"schema_version", "decisions"}:
        raise ReportingReceiptError(
            "review input must contain exactly schema_version and decisions"
        )
    if value.get("schema_version") != 1:
        raise ReportingReceiptError("review input schema_version must equal 1")
    raw_decisions = value.get("decisions")
    if not isinstance(raw_decisions, list):
        raise ReportingReceiptError("review input decisions must be an array")

    indexed: dict[str, dict[str, Any]] = {}
    required_fields = {
        "id",
        "falsified",
        "rationale",
        "evidence_trial_hashes",
    }
    for index, raw in enumerate(raw_decisions):
        if not isinstance(raw, Mapping) or set(raw) != required_fields:
            raise ReportingReceiptError(
                f"review decision {index} must contain exactly "
                f"{sorted(required_fields)}"
            )
        hypothesis_id = raw.get("id")
        if not isinstance(hypothesis_id, str):
            raise ReportingReceiptError(f"review decision {index} has no string id")
        if hypothesis_id in indexed:
            raise ReportingReceiptError(
                f"duplicate review decision for {hypothesis_id}"
            )
        if not isinstance(raw.get("falsified"), bool):
            raise ReportingReceiptError(
                f"{hypothesis_id}.falsified must be a boolean"
            )
        rationale = _nonempty_text(
            raw.get("rationale"), label=f"{hypothesis_id}.rationale"
        )
        evidence = raw.get("evidence_trial_hashes")
        if not isinstance(evidence, list) or not evidence:
            raise ReportingReceiptError(
                f"{hypothesis_id}.evidence_trial_hashes must be non-empty"
            )
        if not all(_is_sha256(item) for item in evidence):
            raise ReportingReceiptError(
                f"{hypothesis_id}.evidence_trial_hashes must contain SHA-256 values"
            )
        if len(evidence) != len(set(evidence)):
            raise ReportingReceiptError(
                f"{hypothesis_id}.evidence_trial_hashes contains duplicates"
            )
        unknown = sorted(set(evidence) - allowed_trial_hashes)
        if unknown:
            raise ReportingReceiptError(
                f"{hypothesis_id}.evidence_trial_hashes are not present in "
                f"the claim-bound analysis index: {unknown}"
            )
        indexed[hypothesis_id] = {
            "id": hypothesis_id,
            "falsified": raw["falsified"],
            "rationale": rationale,
            "evidence_trial_hashes": list(evidence),
        }

    if set(indexed) != set(EXPECTED_HYPOTHESIS_IDS):
        raise ReportingReceiptError(
            "review decision ids do not exactly match the Phase 2a contract: "
            f"expected={list(EXPECTED_HYPOTHESIS_IDS)}, "
            f"observed={sorted(indexed)}"
        )
    return [indexed[hypothesis_id] for hypothesis_id in EXPECTED_HYPOTHESIS_IDS]


def _wandb_report_url(value: Any) -> str:
    url = _nonempty_text(value, label="W&B report URL")
    parsed = urlsplit(url)
    try:
        port = parsed.port
    except ValueError as exc:
        raise ReportingReceiptError("W&B report URL has an invalid port") from exc
    path_parts = [part for part in parsed.path.split("/") if part]
    report_index = path_parts.index("reports") if "reports" in path_parts else -1
    if (
        parsed.scheme != "https"
        or parsed.hostname != "wandb.ai"
        or parsed.username is not None
        or parsed.password is not None
        or port is not None
        or report_index < 1
        or report_index == len(path_parts) - 1
        or parsed.fragment
    ):
        raise ReportingReceiptError(
            "W&B report URL must be an https://wandb.ai/.../reports/... URL "
            "without credentials, a port, or a fragment"
        )
    return url


def _stored_file_identity(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ReportingReceiptError(f"{label} must be an object")
    identity = dict(value)
    if set(identity) != {"path", "size_bytes", "sha256"}:
        raise ReportingReceiptError(
            f"{label} must contain exactly path, size_bytes, and sha256"
        )
    path = identity["path"]
    size = identity["size_bytes"]
    if (
        not isinstance(path, str)
        or not Path(path).is_absolute()
        or not isinstance(size, int)
        or isinstance(size, bool)
        or size <= 0
        or not _is_sha256(identity["sha256"])
    ):
        raise ReportingReceiptError(f"{label} has an invalid file identity")
    return identity


def _read_bound_json_artifact(
    value: Any,
    *,
    label: str,
) -> tuple[Path, dict[str, Any]]:
    """Read a JSON artifact and require its bytes to match a receipt identity."""

    expected = _stored_file_identity(value, label=label)
    path, payload, binding = _read_json_object(expected["path"], label=label)
    observed = {
        "path": str(path),
        "size_bytes": binding["byte_size"],
        "sha256": binding["sha256"],
    }
    if observed != expected:
        raise ReportingReceiptError(
            f"{label} bytes do not match the storage-snapshot receipt"
        )
    return path, payload


def _validate_terminal_claim_set(
    *,
    claims_output_root: str,
    expected_trial_hashes: frozenset[str],
) -> tuple[dict[str, int], dict[str, int]]:
    """Re-read the terminal claim ledger used to build the final analysis."""

    claims_dir = Path(claims_output_root) / "claims"
    if not claims_dir.is_dir():
        raise ReportingReceiptError(
            "storage-snapshot ledger has no readable claims directory"
        )
    before = tuple(sorted(claims_dir.glob("*.json")))
    observed_hashes: set[str] = set()
    outcomes: dict[str, int] = {}
    claim_kinds: dict[str, int] = {}
    for claim_path in before:
        if claim_path.is_symlink():
            raise ReportingReceiptError(
                f"terminal claim must not be a symbolic link: {claim_path}"
            )
        trial_hash = claim_path.stem
        _, claim, _ = _read_json_object(
            claim_path,
            label=f"terminal claim {trial_hash}",
        )
        if (
            not _is_sha256(trial_hash)
            or claim.get("schema_version") != 1
            or claim.get("trial_hash") != trial_hash
            or claim.get("state") != "terminal"
            or claim.get("outcome") not in {"COMPLETE", "PRUNED", "FAIL"}
            or claim.get("claim_kind", "optuna_study")
            not in {"optuna_study", "fixed_manifest"}
        ):
            raise ReportingReceiptError(
                f"terminal claim does not satisfy the frozen contract: {claim_path}"
            )
        observed_hashes.add(trial_hash)
        outcome = str(claim["outcome"])
        outcomes[outcome] = outcomes.get(outcome, 0) + 1
        claim_kind = str(claim.get("claim_kind", "optuna_study"))
        claim_kinds[claim_kind] = claim_kinds.get(claim_kind, 0) + 1
    after = tuple(sorted(claims_dir.glob("*.json")))
    if after != before:
        raise ReportingReceiptError(
            "terminal claim directory changed during reporting validation"
        )
    if observed_hashes != set(expected_trial_hashes):
        missing = sorted(set(expected_trial_hashes) - observed_hashes)
        extra = sorted(observed_hashes - set(expected_trial_hashes))
        raise ReportingReceiptError(
            "terminal claim hash set does not exactly match the analysis index: "
            f"missing={missing}, extra={extra}"
        )
    return outcomes, claim_kinds


def _validate_storage_snapshot_receipt(
    receipt: Mapping[str, Any],
    *,
    analysis_path: Path,
    analysis_binding: Mapping[str, Any],
    study_identity: Mapping[str, Any],
    analysis_trial_hashes: frozenset[str],
) -> None:
    required = {
        "schema_version",
        "receipt_kind",
        "study_identity",
        "analysis_artifact",
        "ledger_contract_artifact",
        "storage",
        "quiescence_evidence",
        "created_by",
        "created_at_utc",
    }
    if set(receipt) != required:
        raise ReportingReceiptError(
            "storage-snapshot receipt fields do not match the frozen contract"
        )
    if (
        receipt.get("schema_version") != 1
        or receipt.get("receipt_kind") != _STORAGE_RECEIPT_KIND
    ):
        raise ReportingReceiptError(
            "storage-snapshot receipt kind/schema is invalid"
        )
    expected_analysis_artifact = {
        "path": str(analysis_path),
        "size_bytes": analysis_binding["byte_size"],
        "sha256": analysis_binding["sha256"],
    }
    observed_analysis_artifact = _stored_file_identity(
        receipt.get("analysis_artifact"),
        label="storage-snapshot analysis_artifact",
    )
    if observed_analysis_artifact != expected_analysis_artifact:
        raise ReportingReceiptError(
            "storage-snapshot receipt is not bound to the exact analysis artifact"
        )

    receipt_identity = receipt.get("study_identity")
    if (
        not isinstance(receipt_identity, Mapping)
        or set(receipt_identity)
        != {
            "study_name",
            "requested_trials",
            "spec_canonical_sha256",
            "spec_file",
        }
    ):
        raise ReportingReceiptError(
            "storage-snapshot receipt study_identity is invalid"
        )
    requested_trials = receipt_identity.get("requested_trials")
    if (
        receipt_identity.get("study_name") != study_identity["study_name"]
        or receipt_identity.get("spec_canonical_sha256")
        != study_identity["spec_hash"]
        or not isinstance(requested_trials, int)
        or isinstance(requested_trials, bool)
        or requested_trials <= 0
    ):
        raise ReportingReceiptError(
            "storage-snapshot receipt study identity differs from the analysis"
        )
    _stored_file_identity(
        receipt_identity.get("spec_file"),
        label="storage-snapshot spec_file",
    )

    ledger_path, ledger = _read_bound_json_artifact(
        receipt.get("ledger_contract_artifact"),
        label="storage-snapshot ledger_contract_artifact",
    )
    ledger_fields = {
        "schema_version",
        "study_name",
        "spec_hash",
        "code_commit",
        "code_dirty",
        "data_hash",
        "capability_hash",
        "claims_output_root",
        "storage_identity_sha256",
        "compute_budget",
        "requested_trials",
        "ledger_contract_sha256",
    }
    if set(ledger) != ledger_fields:
        raise ReportingReceiptError(
            "storage-snapshot ledger contract fields are invalid"
        )
    ledger_payload = dict(ledger)
    declared_ledger_hash = ledger_payload.pop(
        "ledger_contract_sha256",
        None,
    )
    claims_output_root = ledger.get("claims_output_root")
    expected_ledger_identity = {
        "study_name": study_identity["study_name"],
        "spec_hash": study_identity["spec_hash"],
        "code_commit": study_identity["code_commit"],
        "code_dirty": False,
        "data_hash": study_identity["data_hash"],
        "capability_hash": study_identity["capability_hash"],
    }
    if (
        ledger.get("schema_version") != 1
        or any(
            ledger.get(field) != expected
            for field, expected in expected_ledger_identity.items()
        )
        or ledger.get("requested_trials") != requested_trials
        or not isinstance(claims_output_root, str)
        or not Path(claims_output_root).is_absolute()
        or ledger_path
        != Path(claims_output_root).resolve()
        / "phase2_ledger_contract.json"
        or not _is_sha256(ledger.get("storage_identity_sha256"))
        or not isinstance(ledger.get("compute_budget"), Mapping)
        or declared_ledger_hash != stable_hash(ledger_payload)
    ):
        raise ReportingReceiptError(
            "storage-snapshot ledger contract does not match the final study"
        )

    storage = receipt.get("storage")
    if (
        not isinstance(storage, Mapping)
        or set(storage)
        != {"backend", "snapshot_file", "backend_validation"}
        or storage.get("backend") not in _STORAGE_BACKENDS
    ):
        raise ReportingReceiptError(
            "storage-snapshot receipt has an invalid storage backend"
        )
    _stored_file_identity(
        storage.get("snapshot_file"),
        label="storage-snapshot snapshot_file",
    )
    backend_validation = storage.get("backend_validation")
    if (
        not isinstance(backend_validation, Mapping)
        or set(backend_validation)
        != {"method", "entry_count", "listing_sha256"}
        or backend_validation.get("method")
        not in {"pg_restore_list", "safe_tar_member_inventory"}
        or not isinstance(backend_validation.get("entry_count"), int)
        or isinstance(backend_validation.get("entry_count"), bool)
        or int(backend_validation["entry_count"]) <= 0
        or not _is_sha256(backend_validation.get("listing_sha256"))
    ):
        raise ReportingReceiptError(
            "storage-snapshot receipt backend validation is incomplete"
        )

    quiescence = receipt.get("quiescence_evidence")
    expected_quiescence_fields = {
        "status_file",
        "completion_disposition",
        "budget_exhausted_early_stop_approval",
        "claim_count",
        "running_claims",
        "recovery_queued_claims",
        "terminal_gpu_hours_actual",
        "active_gpu_hours_reserved",
        "remaining_unreserved_gpu_hours",
        "maximum_full_trial_gpu_hours",
        "compute_budget_violation_count",
        "status_hold_reasons",
        "fresh_optuna_trial_count",
        "fresh_trial_target_guard_count",
        "optuna_trial_count",
        "optuna_states",
    }
    if (
        not isinstance(quiescence, Mapping)
        or set(quiescence) != expected_quiescence_fields
    ):
        raise ReportingReceiptError(
            "storage-snapshot receipt quiescence_evidence is invalid"
        )
    _, status = _read_bound_json_artifact(
        quiescence.get("status_file"),
        label="storage-snapshot status_file",
    )
    claim_count = quiescence.get("claim_count")
    fresh_count = quiescence.get("fresh_optuna_trial_count")
    target_guard_count = quiescence.get("fresh_trial_target_guard_count")
    optuna_count = quiescence.get("optuna_trial_count")
    if (
        claim_count != len(analysis_trial_hashes)
        or not isinstance(fresh_count, int)
        or isinstance(fresh_count, bool)
        or not 0 <= fresh_count <= requested_trials
        or not isinstance(target_guard_count, int)
        or isinstance(target_guard_count, bool)
        or target_guard_count < 0
        or not isinstance(optuna_count, int)
        or isinstance(optuna_count, bool)
        or optuna_count <= 0
        or optuna_count < fresh_count + target_guard_count
        or quiescence.get("running_claims") != 0
        or quiescence.get("recovery_queued_claims") != 0
        or quiescence.get("active_gpu_hours_reserved") != 0
        or quiescence.get("compute_budget_violation_count") != 0
    ):
        raise ReportingReceiptError(
            "storage-snapshot receipt does not prove a complete quiescent study"
        )
    states = quiescence.get("optuna_states")
    if (
        not isinstance(states, Mapping)
        or any(
            state not in {"COMPLETE", "PRUNED", "FAIL"}
            or not isinstance(count, int)
            or isinstance(count, bool)
            or count < 0
            for state, count in states.items()
        )
        or sum(int(count) for count in states.values()) != optuna_count
    ):
        raise ReportingReceiptError(
            "storage-snapshot receipt has invalid Optuna terminal states"
        )
    hold_reasons = quiescence.get("status_hold_reasons")
    if (
        not isinstance(hold_reasons, list)
        or any(not isinstance(reason, str) for reason in hold_reasons)
        or len(hold_reasons) != len(set(hold_reasons))
    ):
        raise ReportingReceiptError(
            "storage-snapshot receipt has invalid final hold reasons"
        )
    disposition = quiescence.get("completion_disposition")
    approval = quiescence.get("budget_exhausted_early_stop_approval")
    terminal_actual = quiescence.get("terminal_gpu_hours_actual")
    remaining = quiescence.get("remaining_unreserved_gpu_hours")
    maximum_full = quiescence.get("maximum_full_trial_gpu_hours")
    if (
        not isinstance(terminal_actual, (int, float))
        or isinstance(terminal_actual, bool)
        or not math.isfinite(float(terminal_actual))
        or float(terminal_actual) < 0.0
        or not isinstance(remaining, (int, float))
        or isinstance(remaining, bool)
        or not math.isfinite(float(remaining))
        or float(remaining) < 0.0
        or not isinstance(maximum_full, (int, float))
        or isinstance(maximum_full, bool)
        or not math.isfinite(float(maximum_full))
        or float(maximum_full) <= 0.0
    ):
        raise ReportingReceiptError(
            "storage-snapshot receipt has invalid GPU-hour completion evidence"
        )
    if disposition == "requested_trial_target_reached":
        if (
            fresh_count != requested_trials
            or approval is not None
            or "requested_trial_target_reached" not in hold_reasons
        ):
            raise ReportingReceiptError(
                "storage-snapshot target-reached disposition is inconsistent"
            )
    elif disposition == "approved_budget_exhausted_early_stop":
        expected_approval_fields = {
            "approved_by",
            "approved_at_utc",
            "approved_fresh_optuna_trial_count",
            "requested_trials",
            "remaining_unreserved_gpu_hours",
            "maximum_full_trial_gpu_hours",
        }
        if (
            fresh_count >= requested_trials
            or float(remaining) >= float(maximum_full)
            or "insufficient_unreserved_gpu_hours_for_one_full_trial"
            not in hold_reasons
            or not isinstance(approval, Mapping)
            or set(approval) != expected_approval_fields
            or approval.get("approved_fresh_optuna_trial_count")
            != fresh_count
            or approval.get("requested_trials") != requested_trials
            or approval.get("remaining_unreserved_gpu_hours") != remaining
            or approval.get("maximum_full_trial_gpu_hours") != maximum_full
        ):
            raise ReportingReceiptError(
                "storage-snapshot approved early-stop disposition is inconsistent"
            )
        _nonempty_text(
            approval.get("approved_by"),
            label="budget-exhausted approval approved_by",
        )
        _utc_timestamp(
            approval.get("approved_at_utc"),
            label="budget-exhausted approval approved_at_utc",
        )
    else:
        raise ReportingReceiptError(
            "storage-snapshot receipt has no valid completion disposition"
        )

    status_claims = status.get("claims")
    status_claim_states = (
        status_claims.get("states")
        if isinstance(status_claims, Mapping)
        else None
    )
    status_claim_outcomes = (
        status_claims.get("outcomes")
        if isinstance(status_claims, Mapping)
        else None
    )
    status_budget = status.get("compute_budget")
    expected_status_identity = {
        "schema_version": 1,
        "study_name": study_identity["study_name"],
        "study_identity": dict(study_identity),
        "storage_backend": storage["backend"],
        "storage_identity_sha256": ledger["storage_identity_sha256"],
        "ledger_contract_sha256": declared_ledger_hash,
        "claims_output_root": claims_output_root,
        "requested_trials": requested_trials,
        "optuna_trial_count": optuna_count,
        "fresh_optuna_trial_count": fresh_count,
        "fresh_trial_target_guard_count": target_guard_count,
        "optuna_states": dict(states),
        "hold_reasons": hold_reasons,
    }
    if (
        any(
            status.get(field) != expected
            for field, expected in expected_status_identity.items()
        )
        or not isinstance(status_claims, Mapping)
        or status_claims.get("claim_count") != claim_count
        or not isinstance(status_claim_states, Mapping)
        or status_claim_states.get("terminal") != claim_count
        or set(status_claim_states) != {"terminal"}
        or not isinstance(status_claim_outcomes, Mapping)
        or any(
            outcome not in {"COMPLETE", "PRUNED", "FAIL"}
            or not isinstance(count, int)
            or isinstance(count, bool)
            or count < 0
            for outcome, count in status_claim_outcomes.items()
        )
        or sum(status_claim_outcomes.values()) != claim_count
        or status_claims.get("terminal_gpu_hours_actual")
        != terminal_actual
        or status_claims.get("active_gpu_hours_reserved") != 0
        or status_claims.get("compute_budget_violation_count") != 0
        or not isinstance(status_budget, Mapping)
        or status_budget.get("maximum_full_trial_gpu_hours")
        != maximum_full
        or status_budget.get("terminal_gpu_hours_actual")
        != terminal_actual
        or status_budget.get("active_gpu_hours_reserved") != 0
        or status_budget.get("remaining_unreserved_gpu_hours") != remaining
    ):
        raise ReportingReceiptError(
            "storage-snapshot status evidence does not match its receipt "
            "or immutable ledger"
        )
    observed_outcomes, _ = _validate_terminal_claim_set(
        claims_output_root=claims_output_root,
        expected_trial_hashes=analysis_trial_hashes,
    )
    if observed_outcomes != dict(status_claim_outcomes):
        raise ReportingReceiptError(
            "terminal claim outcomes do not match the revalidated status evidence"
        )
    _nonempty_text(receipt.get("created_by"), label="snapshot created_by")
    _utc_timestamp(receipt.get("created_at_utc"), label="snapshot created_at_utc")


def _validate_wandb_verification(
    verification: Mapping[str, Any],
    *,
    analysis_binding: Mapping[str, Any],
    study_identity: Mapping[str, Any],
    expected_run_ids: tuple[str, ...],
) -> dict[str, Any]:
    required = {
        "schema_version",
        "kind",
        "verification_status",
        "verification_method",
        "analysis_artifact",
        "study_identity",
        "wandb_report_url",
        "wandb_run_count",
        "wandb_run_ids",
        "verified_by",
        "verified_at_utc",
    }
    if set(verification) != required:
        raise ReportingReceiptError(
            "W&B verification fields do not match the frozen contract"
        )
    if (
        verification.get("schema_version") != 1
        or verification.get("kind") != _WANDB_VERIFICATION_KIND
        or verification.get("verification_status") != "verified"
        or verification.get("verification_method")
        != "manual_wandb_report_run_set_audit"
    ):
        raise ReportingReceiptError(
            "W&B verification kind/status/method is invalid"
        )
    if verification.get("analysis_artifact") != dict(analysis_binding):
        raise ReportingReceiptError(
            "W&B verification is not bound to the exact analysis artifact"
        )
    if verification.get("study_identity") != dict(study_identity):
        raise ReportingReceiptError(
            "W&B verification study identity differs from the analysis"
        )
    report_url = _wandb_report_url(verification.get("wandb_report_url"))
    observed_run_ids = verification.get("wandb_run_ids")
    if (
        not isinstance(observed_run_ids, list)
        or any(
            not isinstance(run_id, str)
            or not run_id.strip()
            or run_id != run_id.strip()
            for run_id in observed_run_ids
        )
        or observed_run_ids != sorted(set(observed_run_ids))
        or tuple(observed_run_ids) != expected_run_ids
        or verification.get("wandb_run_count") != len(expected_run_ids)
    ):
        raise ReportingReceiptError(
            "W&B verification run-ID set does not exactly match the "
            "offline publication payload"
        )
    verified_by = _nonempty_text(
        verification.get("verified_by"),
        label="W&B verified_by",
    )
    verified_at = _utc_timestamp(
        verification.get("verified_at_utc"),
        label="W&B verified_at_utc",
    )
    return {
        "wandb_report_url": report_url,
        "wandb_run_ids": list(expected_run_ids),
        "wandb_run_count": len(expected_run_ids),
        "verified_by": verified_by,
        "verified_at_utc": verified_at,
    }


def create_reporting_receipts(
    *,
    analysis_path: str | Path,
    decisions_path: str | Path,
    wandb_payload_path: str | Path,
    storage_snapshot_receipt_path: str | Path,
    wandb_verification_path: str | Path,
    reviewer: str,
    reviewed_at_utc: str,
    publisher: str,
    published_at_utc: str,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """Validate and atomically write both Phase 2a completion receipts."""

    resolved_analysis_path, analysis, analysis_binding = _read_json_object(
        analysis_path, label="Phase 2a analysis"
    )
    _, payload, payload_binding = _read_json_object(
        wandb_payload_path, label="offline W&B payload"
    )
    _, review_input, decisions_binding = _read_json_object(
        decisions_path, label="hypothesis review input"
    )
    _, storage_receipt, storage_receipt_binding = _read_json_object(
        storage_snapshot_receipt_path,
        label="storage-snapshot receipt",
    )
    _, wandb_verification, wandb_verification_binding = _read_json_object(
        wandb_verification_path,
        label="W&B verification",
    )
    identity, allowed_trial_hashes, expected_run_ids = _validate_analysis(
        analysis,
        payload,
    )
    _validate_storage_snapshot_receipt(
        storage_receipt,
        analysis_path=resolved_analysis_path,
        analysis_binding=analysis_binding,
        study_identity=identity,
        analysis_trial_hashes=allowed_trial_hashes,
    )
    verified_wandb = _validate_wandb_verification(
        wandb_verification,
        analysis_binding=analysis_binding,
        study_identity=identity,
        expected_run_ids=expected_run_ids,
    )
    decisions = _review_decisions(
        review_input,
        allowed_trial_hashes=allowed_trial_hashes,
    )

    identity_hash = stable_hash(identity)
    reviewed = {
        "schema_version": 1,
        "kind": "phase2a_reviewed_falsified_hypotheses",
        "analysis_artifact": analysis_binding,
        "storage_snapshot_receipt_artifact": storage_receipt_binding,
        "review_input_artifact": decisions_binding,
        "study_identity": identity,
        "study_identity_sha256": identity_hash,
        "expected_hypothesis_ids": list(EXPECTED_HYPOTHESIS_IDS),
        "reviewer": _nonempty_text(reviewer, label="reviewer"),
        "reviewed_at_utc": _utc_timestamp(
            reviewed_at_utc, label="reviewed_at_utc"
        ),
        "decision_count": len(decisions),
        "decisions": decisions,
    }
    publication = {
        "schema_version": 1,
        "kind": "wandb_publication_receipt",
        "publication_status": "published",
        "analysis_artifact": analysis_binding,
        "storage_snapshot_receipt_artifact": storage_receipt_binding,
        "wandb_payload_artifact": payload_binding,
        "wandb_verification_artifact": wandb_verification_binding,
        "study_identity": identity,
        "study_identity_sha256": identity_hash,
        "wandb_report_url": verified_wandb["wandb_report_url"],
        "wandb_run_count": verified_wandb["wandb_run_count"],
        "wandb_run_ids": verified_wandb["wandb_run_ids"],
        "wandb_verified_by": verified_wandb["verified_by"],
        "wandb_verified_at_utc": verified_wandb["verified_at_utc"],
        "publisher": _nonempty_text(publisher, label="publisher"),
        "published_at_utc": _utc_timestamp(
            published_at_utc, label="published_at_utc"
        ),
    }

    destination = Path(output_dir).expanduser().resolve()
    reviewed_path = destination / _REVIEWED_FILENAME
    publication_path = destination / _PUBLICATION_FILENAME
    if destination.exists():
        raise ReportingReceiptError(
            "completion receipts are immutable; use a new versioned output "
            f"directory instead of overwriting {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        destination.mkdir()
    except FileExistsError as exc:
        raise ReportingReceiptError(
            "completion receipts are immutable; use a new versioned output "
            f"directory instead of overwriting {destination}"
        ) from exc
    atomic_write_json(reviewed_path, reviewed)
    atomic_write_json(publication_path, publication)
    return reviewed_path, publication_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", required=True)
    parser.add_argument("--decisions", required=True)
    parser.add_argument("--wandb-payload", required=True)
    parser.add_argument("--storage-snapshot-receipt", required=True)
    parser.add_argument("--wandb-verification", required=True)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--reviewed-at-utc", required=True)
    parser.add_argument("--publisher", required=True)
    parser.add_argument("--published-at-utc", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    try:
        reviewed, publication = create_reporting_receipts(
            analysis_path=args.analysis,
            decisions_path=args.decisions,
            wandb_payload_path=args.wandb_payload,
            storage_snapshot_receipt_path=args.storage_snapshot_receipt,
            wandb_verification_path=args.wandb_verification,
            reviewer=args.reviewer,
            reviewed_at_utc=args.reviewed_at_utc,
            publisher=args.publisher,
            published_at_utc=args.published_at_utc,
            output_dir=args.output_dir,
        )
    except ReportingReceiptError as exc:
        raise SystemExit(str(exc)) from exc
    print(
        json.dumps(
            {
                "reviewed_falsified_hypotheses": str(reviewed),
                "wandb_publication_receipt": str(publication),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
