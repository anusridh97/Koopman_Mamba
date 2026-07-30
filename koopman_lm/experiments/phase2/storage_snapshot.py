"""Create an immutable, identity-bound receipt for an Optuna storage snapshot.

This command does not create or inspect the database backup itself.  An operator
must first quiesce the study, generate a study-status report, and create the
backend-native snapshot.  The command then verifies those files and records
their hashes without importing Optuna or database drivers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import subprocess
import tarfile
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

try:
    from .spec import load_spec, stable_hash
except ImportError:
    from spec import load_spec, stable_hash  # type: ignore


_BACKENDS = {"postgresql", "scg_validated_journal_storage"}
_READINESS_HOLDS = {
    "spec_not_scientific_ready",
    "compute_budget_not_approved",
    "gpu_hour_limits_unresolved",
    "compute_budget_violation_requires_reapproval",
}
_FINAL_HOLDS = {
    "requested_trial_target_reached",
    "insufficient_unreserved_gpu_hours_for_one_full_trial",
}


class StorageSnapshotError(ValueError):
    """Raised when a storage snapshot cannot receive a valid receipt."""


def _hash_stable_file(path: str | Path, *, label: str) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    try:
        with resolved.open("rb") as handle:
            before = os.fstat(handle.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise StorageSnapshotError(f"{label} must be a regular file: {resolved}")
            digest = hashlib.sha256()
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
            after = os.fstat(handle.fileno())
    except FileNotFoundError as exc:
        raise StorageSnapshotError(f"{label} does not exist: {resolved}") from exc
    except OSError as exc:
        raise StorageSnapshotError(f"Could not read {label} {resolved}: {exc}") from exc

    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if identity_before != identity_after:
        raise StorageSnapshotError(f"{label} changed while it was being hashed")
    if before.st_size <= 0:
        raise StorageSnapshotError(f"{label} must not be empty: {resolved}")
    return {
        "path": str(resolved),
        "size_bytes": int(before.st_size),
        "sha256": digest.hexdigest(),
    }


def _load_stable_json(
    path: str | Path,
    *,
    label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    before = _hash_stable_file(path, label=label)
    try:
        payload = json.loads(Path(before["path"]).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StorageSnapshotError(f"{label} is not valid UTF-8 JSON: {exc}") from exc
    after = _hash_stable_file(path, label=label)
    if before != after:
        raise StorageSnapshotError(f"{label} changed while it was being validated")
    if not isinstance(payload, dict):
        raise StorageSnapshotError(f"{label} must contain a JSON object")
    return payload, before


def _require_nonnegative_int(value: Any, *, label: str) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < 0
    ):
        raise StorageSnapshotError(f"{label} must be a nonnegative integer")
    return value


def _require_finite_nonnegative(value: Any, *, label: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise StorageSnapshotError(
            f"{label} must be a finite nonnegative number"
        )
    return float(value)


def _validated_quiescence(
    status: Mapping[str, Any],
    *,
    study_name: str,
    requested_trials: int,
    compute_budget: Mapping[str, Any],
    study_identity: Mapping[str, Any],
    backend: str,
    analysis_trial_count: int,
    budget_exhausted_approval: Mapping[str, str] | None,
) -> dict[str, Any]:
    if status.get("schema_version") != 1:
        raise StorageSnapshotError("Status evidence must use schema_version=1")
    if status.get("study_name") != study_name:
        raise StorageSnapshotError(
            "Status evidence study_name does not match the frozen search spec"
        )
    if status.get("requested_trials") != requested_trials:
        raise StorageSnapshotError(
            "Status evidence requested_trials does not match the frozen search spec"
        )
    if status.get("study_identity") != dict(study_identity):
        raise StorageSnapshotError(
            "Status evidence study identity does not match the final analysis"
        )
    if status.get("storage_backend") != backend:
        raise StorageSnapshotError(
            "Status evidence storage backend does not match the snapshot backend"
        )
    fresh_count = _require_nonnegative_int(
        status.get("fresh_optuna_trial_count"),
        label="fresh_optuna_trial_count",
    )
    if fresh_count > requested_trials:
        raise StorageSnapshotError(
            "Status evidence exceeds the approved fresh-trial target"
        )
    target_guard_count = _require_nonnegative_int(
        status.get("fresh_trial_target_guard_count"),
        label="fresh_trial_target_guard_count",
    )
    optuna_count = _require_nonnegative_int(
        status.get("optuna_trial_count"),
        label="optuna_trial_count",
    )
    if optuna_count < fresh_count + target_guard_count:
        raise StorageSnapshotError(
            "Status evidence optuna_trial_count is below the sum of approved "
            "fresh and target-guard rows"
        )
    optuna_states = status.get("optuna_states")
    if not isinstance(optuna_states, Mapping):
        raise StorageSnapshotError("Status evidence optuna_states must be an object")
    optuna_state_counts = {
        str(name): _require_nonnegative_int(
            count,
            label=f"optuna_states.{name}",
        )
        for name, count in optuna_states.items()
    }
    if sum(optuna_state_counts.values()) != optuna_count:
        raise StorageSnapshotError(
            "Optuna state counts do not sum to optuna_trial_count"
        )
    nonterminal_states = sorted(
        name
        for name, count in optuna_state_counts.items()
        if count and name not in {"COMPLETE", "PRUNED", "FAIL"}
    )
    if nonterminal_states:
        raise StorageSnapshotError(
            "Optuna study is not terminal: " + ", ".join(nonterminal_states)
        )
    status_budget = status.get("compute_budget")
    if not isinstance(status_budget, Mapping):
        raise StorageSnapshotError(
            "Status evidence is missing the compute_budget object"
        )
    for key in (
        "maximum_total_gpu_hours",
        "maximum_full_trial_gpu_hours",
        "maximum_concurrent_trials",
    ):
        if status_budget.get(key) != compute_budget.get(key):
            raise StorageSnapshotError(
                f"Status evidence compute_budget.{key} does not match "
                "the frozen search spec"
            )
    maximum_total_gpu_hours = _require_finite_nonnegative(
        status_budget.get("maximum_total_gpu_hours"),
        label="compute_budget.maximum_total_gpu_hours",
    )
    maximum_full_trial_gpu_hours = _require_finite_nonnegative(
        status_budget.get("maximum_full_trial_gpu_hours"),
        label="compute_budget.maximum_full_trial_gpu_hours",
    )
    if maximum_total_gpu_hours <= 0.0 or maximum_full_trial_gpu_hours <= 0.0:
        raise StorageSnapshotError(
            "Final snapshot requires positive approved GPU-hour limits"
        )
    claims = status.get("claims")
    if not isinstance(claims, Mapping):
        raise StorageSnapshotError("Status evidence is missing the claims object")
    states = claims.get("states")
    if not isinstance(states, Mapping):
        raise StorageSnapshotError("Status evidence claims.states must be an object")
    state_counts = {
        str(name): _require_nonnegative_int(
            count,
            label=f"claims.states.{name}",
        )
        for name, count in states.items()
    }
    claim_count = _require_nonnegative_int(
        claims.get("claim_count"),
        label="claims.claim_count",
    )
    if sum(state_counts.values()) != claim_count:
        raise StorageSnapshotError(
            "Status evidence claim_count does not equal the sum of claims.states"
        )
    if set(state_counts) - {"terminal"} or state_counts.get("terminal", 0) != claim_count:
        raise StorageSnapshotError(
            "Study is not quiescent: final snapshot requires every claim "
            "to be terminal"
        )
    if claim_count != analysis_trial_count:
        raise StorageSnapshotError(
            "Status claim count does not match the claim-bound analysis index"
        )
    outcomes = claims.get("outcomes")
    if not isinstance(outcomes, Mapping):
        raise StorageSnapshotError("Status evidence claims.outcomes must be an object")
    outcome_counts = {
        str(name): _require_nonnegative_int(
            count,
            label=f"claims.outcomes.{name}",
        )
        for name, count in outcomes.items()
    }
    if (
        sum(outcome_counts.values()) != claim_count
        or set(outcome_counts) - {"COMPLETE", "PRUNED", "FAIL"}
    ):
        raise StorageSnapshotError(
            "Claim outcome counts are incomplete or contain unknown outcomes"
        )
    terminal_gpu_hours_actual = _require_finite_nonnegative(
        claims.get("terminal_gpu_hours_actual"),
        label="claims.terminal_gpu_hours_actual",
    )
    running = state_counts.get("running", 0)
    recovery_queued = state_counts.get("recovery_queued", 0)
    if running or recovery_queued:
        raise StorageSnapshotError(
            "Study is not quiescent: running and recovery_queued claims must both be zero"
        )

    reserved = _require_finite_nonnegative(
        claims.get("active_gpu_hours_reserved"),
        label="claims.active_gpu_hours_reserved",
    )
    if reserved != 0.0:
        raise StorageSnapshotError(
            "Status evidence active_gpu_hours_reserved must equal zero"
        )
    status_terminal_gpu_hours = _require_finite_nonnegative(
        status_budget.get("terminal_gpu_hours_actual"),
        label="compute_budget.terminal_gpu_hours_actual",
    )
    status_active_reserved = _require_finite_nonnegative(
        status_budget.get("active_gpu_hours_reserved"),
        label="compute_budget.active_gpu_hours_reserved",
    )
    remaining_gpu_hours = _require_finite_nonnegative(
        status_budget.get("remaining_unreserved_gpu_hours"),
        label="compute_budget.remaining_unreserved_gpu_hours",
    )
    if not math.isclose(
        status_terminal_gpu_hours,
        terminal_gpu_hours_actual,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise StorageSnapshotError(
            "Status compute-budget terminal actual does not match claims"
        )
    if not math.isclose(
        status_active_reserved,
        reserved,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise StorageSnapshotError(
            "Status compute-budget active reservation does not match claims"
        )
    expected_remaining = max(
        0.0,
        maximum_total_gpu_hours - terminal_gpu_hours_actual - reserved,
    )
    if not math.isclose(
        remaining_gpu_hours,
        expected_remaining,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise StorageSnapshotError(
            "Status remaining GPU-hours do not match the frozen budget and claims"
        )
    if terminal_gpu_hours_actual + reserved > maximum_total_gpu_hours:
        raise StorageSnapshotError(
            "Recorded GPU-hours exceed the approved total budget"
        )
    violations = _require_nonnegative_int(
        claims.get("compute_budget_violation_count"),
        label="claims.compute_budget_violation_count",
    )
    if violations:
        raise StorageSnapshotError(
            "A compute-budget violation must be reapproved before snapshot receipt"
        )
    if status.get("launch_allowed") is not False:
        raise StorageSnapshotError(
            "Final snapshot requires launch_allowed=false"
        )

    hold_reasons = status.get("hold_reasons")
    if not isinstance(hold_reasons, list) or any(
        not isinstance(reason, str) for reason in hold_reasons
    ):
        raise StorageSnapshotError("Status evidence hold_reasons must be a string list")
    if len(hold_reasons) != len(set(hold_reasons)):
        raise StorageSnapshotError("Status evidence hold_reasons must be unique")
    readiness_holds = sorted(_READINESS_HOLDS.intersection(hold_reasons))
    if readiness_holds:
        raise StorageSnapshotError(
            "Status evidence is not preflight-ready: " + ", ".join(readiness_holds)
        )
    unexpected_holds = sorted(set(hold_reasons) - _FINAL_HOLDS)
    if unexpected_holds:
        raise StorageSnapshotError(
            "Status evidence contains a non-final hold: "
            + ", ".join(unexpected_holds)
        )

    target_reached = fresh_count == requested_trials
    target_hold = "requested_trial_target_reached" in hold_reasons
    budget_hold = (
        "insufficient_unreserved_gpu_hours_for_one_full_trial"
        in hold_reasons
    )
    if target_reached:
        if not target_hold:
            raise StorageSnapshotError(
                "Final snapshot at target requires requested_trial_target_reached"
            )
        if budget_exhausted_approval is not None:
            raise StorageSnapshotError(
                "Budget-exhaustion approval is not valid after the target is reached"
            )
        completion_disposition = "requested_trial_target_reached"
        recorded_approval: dict[str, Any] | None = None
    else:
        if target_hold:
            raise StorageSnapshotError(
                "Status cannot claim requested_trial_target_reached below target"
            )
        if not budget_hold or remaining_gpu_hours >= maximum_full_trial_gpu_hours:
            raise StorageSnapshotError(
                "Below-target snapshot requires genuine exhaustion of the "
                "approved full-trial GPU-hour budget"
            )
        if budget_exhausted_approval is None:
            raise StorageSnapshotError(
                "Below-target budget exhaustion requires explicit early-stop approval"
            )
        completion_disposition = "approved_budget_exhausted_early_stop"
        recorded_approval = {
            "approved_by": budget_exhausted_approval["approved_by"],
            "approved_at_utc": budget_exhausted_approval["approved_at_utc"],
            "approved_fresh_optuna_trial_count": fresh_count,
            "requested_trials": requested_trials,
            "remaining_unreserved_gpu_hours": remaining_gpu_hours,
            "maximum_full_trial_gpu_hours": maximum_full_trial_gpu_hours,
        }

    return {
        "completion_disposition": completion_disposition,
        "budget_exhausted_early_stop_approval": recorded_approval,
        "claim_count": claim_count,
        "running_claims": running,
        "recovery_queued_claims": recovery_queued,
        "terminal_gpu_hours_actual": terminal_gpu_hours_actual,
        "active_gpu_hours_reserved": reserved,
        "remaining_unreserved_gpu_hours": remaining_gpu_hours,
        "maximum_full_trial_gpu_hours": maximum_full_trial_gpu_hours,
        "compute_budget_violation_count": violations,
        "status_hold_reasons": sorted(hold_reasons),
        "fresh_optuna_trial_count": fresh_count,
        "fresh_trial_target_guard_count": target_guard_count,
        "optuna_trial_count": optuna_count,
        "optuna_states": dict(sorted(optuna_state_counts.items())),
    }


def _validate_backend_snapshot(
    snapshot_path: Path,
    *,
    backend: str,
) -> dict[str, Any]:
    if backend == "postgresql":
        try:
            completed = subprocess.run(
                ["pg_restore", "--list", str(snapshot_path)],
                check=False,
                capture_output=True,
                text=True,
                timeout=120,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise StorageSnapshotError(
                f"Could not validate PostgreSQL custom dump with pg_restore: {exc}"
            ) from exc
        listing = completed.stdout.strip()
        if completed.returncode != 0 or not listing:
            raise StorageSnapshotError(
                "PostgreSQL snapshot is not a valid nonempty custom-format dump: "
                f"{completed.stderr.strip()}"
            )
        entries = [
            line
            for line in listing.splitlines()
            if line.strip() and not line.lstrip().startswith(";")
        ]
        if not entries:
            raise StorageSnapshotError(
                "PostgreSQL snapshot contains no restorable entries"
            )
        return {
            "method": "pg_restore_list",
            "entry_count": len(entries),
            "listing_sha256": hashlib.sha256(
                listing.encode("utf-8")
            ).hexdigest(),
        }
    try:
        with tarfile.open(snapshot_path, mode="r:*") as archive:
            regular = []
            for member in archive.getmembers():
                parts = Path(member.name).parts
                if (
                    member.name.startswith("/")
                    or ".." in parts
                    or member.issym()
                    or member.islnk()
                ):
                    raise StorageSnapshotError(
                        "Journal snapshot contains an unsafe archive member"
                    )
                if member.isfile():
                    regular.append((member.name, int(member.size)))
    except (OSError, tarfile.TarError) as exc:
        raise StorageSnapshotError(
            f"Journal snapshot is not a readable tar archive: {exc}"
        ) from exc
    if not regular:
        raise StorageSnapshotError(
            "Journal snapshot contains no regular storage files"
        )
    return {
        "method": "safe_tar_member_inventory",
        "entry_count": len(regular),
        "listing_sha256": hashlib.sha256(
            json.dumps(
                sorted(regular),
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("utf-8")
        ).hexdigest(),
    }


def _validate_utc_timestamp(value: str, *, option: str) -> str:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise StorageSnapshotError(
            f"{option} must be an RFC 3339 UTC timestamp"
        )
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise StorageSnapshotError(
            f"{option} must be an RFC 3339 UTC timestamp"
        ) from exc
    if parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
        raise StorageSnapshotError(f"{option} must use UTC")
    return value


def _write_json_exclusive(path: str | Path, payload: Mapping[str, Any]) -> Path:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise StorageSnapshotError(
            f"Receipt already exists and will not be replaced: {output}"
        )
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    temp_fd, temp_name = tempfile.mkstemp(
        prefix=f".{output.name}.",
        suffix=".tmp",
        dir=output.parent,
    )
    try:
        with os.fdopen(temp_fd, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temp_name, output)
        except FileExistsError as exc:
            raise StorageSnapshotError(
                f"Receipt already exists and will not be replaced: {output}"
            ) from exc
    finally:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
    return output


def create_storage_snapshot_receipt(
    *,
    spec_path: str | Path,
    snapshot_path: str | Path,
    backend: str,
    created_by: str,
    created_at_utc: str,
    status_evidence_path: str | Path,
    analysis_path: str | Path,
    output_path: str | Path,
    budget_exhausted_approved_by: str | None = None,
    budget_exhausted_approved_at_utc: str | None = None,
) -> dict[str, Any]:
    """Validate a quiescent snapshot and write its no-clobber receipt."""

    if backend not in _BACKENDS:
        raise StorageSnapshotError(f"Unsupported storage backend: {backend!r}")
    if not isinstance(created_by, str) or not created_by.strip():
        raise StorageSnapshotError("--created-by must be a nonempty identity")
    if created_by != created_by.strip():
        raise StorageSnapshotError("--created-by must not have surrounding whitespace")
    created_at_utc = _validate_utc_timestamp(
        created_at_utc,
        option="--created-at-utc",
    )
    approval_values_present = (
        budget_exhausted_approved_by is not None,
        budget_exhausted_approved_at_utc is not None,
    )
    if any(approval_values_present) and not all(approval_values_present):
        raise StorageSnapshotError(
            "--budget-exhausted-approved-by and "
            "--budget-exhausted-approved-at-utc must be provided together"
        )
    budget_exhausted_approval: dict[str, str] | None = None
    if all(approval_values_present):
        assert budget_exhausted_approved_by is not None
        assert budget_exhausted_approved_at_utc is not None
        if (
            not isinstance(budget_exhausted_approved_by, str)
            or not budget_exhausted_approved_by.strip()
            or budget_exhausted_approved_by
            != budget_exhausted_approved_by.strip()
        ):
            raise StorageSnapshotError(
                "--budget-exhausted-approved-by must be a nonempty identity "
                "without surrounding whitespace"
            )
        approved_at_utc = _validate_utc_timestamp(
            budget_exhausted_approved_at_utc,
            option="--budget-exhausted-approved-at-utc",
        )
        approved_at = datetime.fromisoformat(
            approved_at_utc[:-1] + "+00:00"
        )
        created_at = datetime.fromisoformat(
            created_at_utc[:-1] + "+00:00"
        )
        if approved_at > created_at:
            raise StorageSnapshotError(
                "Budget-exhausted early-stop approval cannot postdate the receipt"
            )
        budget_exhausted_approval = {
            "approved_by": budget_exhausted_approved_by,
            "approved_at_utc": approved_at_utc,
        }

    spec_file_before = _hash_stable_file(spec_path, label="search spec")
    spec = load_spec(spec_file_before["path"])
    spec_file_after = _hash_stable_file(spec_path, label="search spec")
    if spec_file_before != spec_file_after:
        raise StorageSnapshotError("Search spec changed while it was being validated")
    if spec.get("status") != "scientific_ready":
        raise StorageSnapshotError(
            "Search spec must have status=scientific_ready before snapshot receipt"
        )
    storage = spec.get("storage")
    if not isinstance(storage, Mapping):
        raise StorageSnapshotError("Search spec storage section must be an object")
    acceptable = storage.get("acceptable_backends")
    if not isinstance(acceptable, list) or backend not in acceptable:
        raise StorageSnapshotError(
            f"Backend {backend!r} is not acceptable under the search spec"
        )
    if storage.get("backend") != backend:
        raise StorageSnapshotError(
            f"Backend {backend!r} does not match frozen spec backend "
            f"{storage.get('backend')!r}"
        )

    public_spec = {
        key: value for key, value in spec.items() if not key.startswith("_")
    }
    spec_hash = stable_hash(public_spec)
    analysis, analysis_file = _load_stable_json(
        analysis_path,
        label="final analysis",
    )
    analysis_identity = analysis.get("study_identity")
    if not isinstance(analysis_identity, Mapping):
        raise StorageSnapshotError("Final analysis has no study identity")
    if (
        analysis_identity.get("study_name") != spec["study_name"]
        or analysis_identity.get("spec_hash") != spec_hash
        or analysis_identity.get("code_dirty") is not False
    ):
        raise StorageSnapshotError(
            "Final analysis identity does not match the scientific-ready spec"
        )
    trial_hash_index = analysis.get("trial_hash_index")
    if (
        not isinstance(trial_hash_index, list)
        or not trial_hash_index
        or len(trial_hash_index) != len(set(trial_hash_index))
        or any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in trial_hash_index
        )
    ):
        raise StorageSnapshotError(
            "Final analysis has no valid claim-bound trial hash index"
        )
    status, status_file = _load_stable_json(
        status_evidence_path,
        label="status evidence",
    )
    claims_output_root = status.get("claims_output_root")
    if (
        not isinstance(claims_output_root, str)
        or not Path(claims_output_root).is_absolute()
    ):
        raise StorageSnapshotError(
            "Status evidence has no absolute claims_output_root"
        )
    ledger_path = (
        Path(claims_output_root).expanduser().resolve()
        / "phase2_ledger_contract.json"
    )
    ledger_contract, ledger_file = _load_stable_json(
        ledger_path,
        label="shared ledger contract",
    )
    ledger_payload = dict(ledger_contract)
    declared_ledger_hash = ledger_payload.pop(
        "ledger_contract_sha256",
        None,
    )
    if (
        declared_ledger_hash != stable_hash(ledger_payload)
        or status.get("ledger_contract_sha256") != declared_ledger_hash
        or status.get("storage_identity_sha256")
        != ledger_contract.get("storage_identity_sha256")
    ):
        raise StorageSnapshotError(
            "Status evidence does not match the immutable shared-ledger contract"
        )
    expected_ledger_fields = {
        "schema_version": 1,
        "study_name": spec["study_name"],
        "spec_hash": spec_hash,
        "code_commit": analysis_identity.get("code_commit"),
        "code_dirty": False,
        "data_hash": analysis_identity.get("data_hash"),
        "capability_hash": analysis_identity.get("capability_hash"),
        "claims_output_root": str(Path(claims_output_root).resolve()),
        "compute_budget": dict(spec["study"]["compute_budget"]),
        "requested_trials": int(spec["study"]["requested_trials"]),
    }
    for field, expected in expected_ledger_fields.items():
        if ledger_contract.get(field) != expected:
            raise StorageSnapshotError(
                f"Shared ledger contract {field} does not match final evidence"
            )
    if not isinstance(
        ledger_contract.get("storage_identity_sha256"),
        str,
    ) or len(str(ledger_contract["storage_identity_sha256"])) != 64:
        raise StorageSnapshotError(
            "Shared ledger contract has no storage-instance identity hash"
        )
    quiescence = _validated_quiescence(
        status,
        study_name=str(spec["study_name"]),
        requested_trials=int(spec["study"]["requested_trials"]),
        compute_budget=spec["study"]["compute_budget"],
        study_identity=analysis_identity,
        backend=backend,
        analysis_trial_count=len(trial_hash_index),
        budget_exhausted_approval=budget_exhausted_approval,
    )
    snapshot_file = _hash_stable_file(snapshot_path, label="storage snapshot")
    backend_validation = _validate_backend_snapshot(
        Path(snapshot_file["path"]),
        backend=backend,
    )
    if _hash_stable_file(snapshot_path, label="storage snapshot") != snapshot_file:
        raise StorageSnapshotError(
            "Storage snapshot changed during backend-native validation"
        )

    input_paths = {
        spec_file_before["path"],
        status_file["path"],
        analysis_file["path"],
        ledger_file["path"],
        snapshot_file["path"],
    }
    output = Path(output_path).expanduser().resolve()
    if len(input_paths) != 5:
        raise StorageSnapshotError(
            "Search spec, final analysis, status evidence, shared ledger, and "
            "snapshot must be distinct files"
        )
    if str(output) in input_paths:
        raise StorageSnapshotError("Receipt output must not alias an input file")

    receipt = {
        "schema_version": 1,
        "receipt_kind": "phase2_optuna_storage_snapshot",
        "study_identity": {
            "study_name": spec["study_name"],
            "requested_trials": spec["study"]["requested_trials"],
            "spec_canonical_sha256": spec_hash,
            "spec_file": spec_file_before,
        },
        "analysis_artifact": analysis_file,
        "ledger_contract_artifact": ledger_file,
        "storage": {
            "backend": backend,
            "snapshot_file": snapshot_file,
            "backend_validation": backend_validation,
        },
        "quiescence_evidence": {
            "status_file": status_file,
            **quiescence,
        },
        "created_by": created_by,
        "created_at_utc": created_at_utc,
    }
    _write_json_exclusive(output, receipt)
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--backend", required=True, choices=sorted(_BACKENDS))
    parser.add_argument("--created-by", required=True)
    parser.add_argument("--created-at-utc", required=True)
    parser.add_argument("--status-evidence", required=True)
    parser.add_argument("--analysis", required=True)
    parser.add_argument(
        "--budget-exhausted-approved-by",
        help=(
            "Explicit approver identity for a below-target stop caused solely "
            "by exhaustion of the frozen GPU-hour budget"
        ),
    )
    parser.add_argument(
        "--budget-exhausted-approved-at-utc",
        help=(
            "RFC 3339 UTC time of the explicit below-target "
            "budget-exhaustion approval"
        ),
    )
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    try:
        receipt = create_storage_snapshot_receipt(
            spec_path=args.spec,
            snapshot_path=args.snapshot,
            backend=args.backend,
            created_by=args.created_by,
            created_at_utc=args.created_at_utc,
            status_evidence_path=args.status_evidence,
            analysis_path=args.analysis,
            output_path=args.output,
            budget_exhausted_approved_by=args.budget_exhausted_approved_by,
            budget_exhausted_approved_at_utc=(
                args.budget_exhausted_approved_at_utc
            ),
        )
    except (StorageSnapshotError, ValueError) as exc:
        raise SystemExit(f"Snapshot receipt rejected: {exc}") from exc
    print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
