"""Run Phase 2a Optuna trials through a finalized external training adapter.

The adapter is intentionally a separate executable. It receives an immutable
manifest and must implement the canonical training/evaluation APIs after the
architecture branches are integrated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import socket
import subprocess
import threading
import time
import uuid
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence
from urllib.parse import parse_qsl, urlsplit

try:
    from .checkpoint import checkpoint_bundle_identity
    from .controller import (
        bind_study_contract,
        create_optuna_study,
        report_step_500,
    )
    from .manifest import (
        atomic_write_json,
        build_trial_manifest,
        git_identity,
        recompute_promotion_config_hash,
        recompute_trial_hash,
        write_trial_directory,
    )
    from .preflight import evaluate_preflight
    from .results import validate_cross_stage_identity, validate_step_metrics
    from .spec import load_spec, stable_hash, suggest_parameters
except ImportError:
    from checkpoint import checkpoint_bundle_identity  # type: ignore
    from controller import (  # type: ignore
        bind_study_contract,
        create_optuna_study,
        report_step_500,
    )
    from manifest import (  # type: ignore
        atomic_write_json,
        build_trial_manifest,
        git_identity,
        recompute_promotion_config_hash,
        recompute_trial_hash,
        write_trial_directory,
    )
    from preflight import evaluate_preflight  # type: ignore
    from results import (  # type: ignore
        validate_cross_stage_identity,
        validate_step_metrics,
    )
    from spec import load_spec, stable_hash, suggest_parameters  # type: ignore


DEFAULT_CLAIM_STALE_AFTER_SECONDS = 10 * 60
_CLAIM_LOCK_STALE_AFTER_SECONDS = 5 * 60
_TERMINAL_CLAIM_OUTCOMES = frozenset({"COMPLETE", "PRUNED", "FAIL"})
_ACTIVE_CLAIM_STATES = frozenset({"running", "recovery_queued"})
_CLAIM_KINDS = frozenset({"optuna_study", "fixed_manifest"})
_TERMINAL_ARTIFACT_NAMES = frozenset(
    {
        "trial_manifest.json",
        "trial_summary.json",
        "metrics/step_500.json",
        "metrics/final.json",
    }
)
_TERMINAL_BINDING_NAMES = _TERMINAL_ARTIFACT_NAMES | {"checkpoint_bundle"}


class AdapterFailure(RuntimeError):
    """Base class for failures reported by or attributed to the adapter."""

    def __init__(
        self,
        message: str,
        *,
        stage: str,
        returncode: int | None = None,
        failure: Mapping[str, Any] | None = None,
        gpu_seconds_actual: float | None = None,
        gpu_hours_actual: float | None = None,
    ) -> None:
        super().__init__(message)
        self.stage = stage
        self.returncode = returncode
        self.failure = dict(failure) if failure is not None else None
        self.gpu_seconds_actual = gpu_seconds_actual
        self.gpu_hours_actual = gpu_hours_actual


class TrialLocalFailure(AdapterFailure):
    """A valid, structured failure caused by one sampled trial.

    Optuna is explicitly configured to catch this exception and continue.
    """


class SystemicAdapterFailure(AdapterFailure):
    """An adapter/process/protocol failure that must stop the worker."""


class ComputeBudgetExhausted(RuntimeError):
    """The shared GPU-hour ledger cannot reserve another full trial."""


def terminal_artifact_bindings(run_dir: str | Path) -> dict[str, dict[str, Any]]:
    """Hash stable terminal artifacts before committing a claim."""

    root = Path(run_dir).expanduser().resolve()
    bindings: dict[str, dict[str, Any]] = {}
    for relative in sorted(_TERMINAL_ARTIFACT_NAMES):
        path = root / relative
        if not path.is_file():
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            before = os.fstat(handle.fileno())
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
            after = os.fstat(handle.fileno())
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
        current = path.stat()
        identity_current = (
            current.st_dev,
            current.st_ino,
            current.st_size,
            current.st_mtime_ns,
        )
        if identity_before != identity_after or identity_before != identity_current:
            raise RuntimeError(f"Terminal artifact changed while hashing: {path}")
        if before.st_size <= 0:
            raise RuntimeError(f"Terminal artifact must not be empty: {path}")
        bindings[relative] = {
            "byte_size": int(before.st_size),
            "sha256": digest.hexdigest(),
        }
    required = {"trial_manifest.json", "trial_summary.json"}
    if not required.issubset(bindings):
        raise RuntimeError(
            "Terminal claim requires manifest and summary artifact bindings"
        )
    screen_path = root / "metrics" / "step_500.json"
    if screen_path.is_file():
        screen = _read_json(screen_path)
        checkpoint = screen.get("checkpoint_provenance")
        if not isinstance(checkpoint, Mapping):
            raise RuntimeError(
                "Healthy screen metrics require checkpoint provenance"
            )
        try:
            checkpoint_identity = checkpoint_bundle_identity(
                root,
                checkpoint.get("checkpoint_bundle_path"),
            )
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc
        expected_identity = {
            "path": checkpoint.get("checkpoint_bundle_path"),
            "byte_size": checkpoint.get("checkpoint_bundle_byte_size"),
            "file_count": checkpoint.get("checkpoint_bundle_file_count"),
            "sha256": checkpoint.get("checkpoint_sha256"),
            "digest_algorithm": checkpoint.get(
                "checkpoint_digest_algorithm"
            ),
        }
        if checkpoint_identity != expected_identity:
            raise RuntimeError(
                "Screen checkpoint bundle identity changed before claim finalization"
            )
        bindings["checkpoint_bundle"] = checkpoint_identity
    return bindings


@dataclass(frozen=True)
class TrialClaim:
    """One ownership lease for a content-addressed trial outcome."""

    path: Path
    trial_hash: str
    trial_number: int
    run_dir: Path
    lease_id: str
    reclaimed: bool = False

    def is_file(self) -> bool:
        """Preserve the former Path-like assertion used by contract tests."""

        return self.path.is_file()


@dataclass(frozen=True)
class QueuedTrialRecovery:
    """A stale manifest atomically reserved and enqueued for recovery."""

    trial_hash: str
    manifest_path: Path
    run_dir: Path
    recovery_token: str


def validate_storage_url(storage: str, approved_backend: str) -> None:
    """Ensure the runtime URL matches the backend approved in the spec."""

    if approved_backend == "postgresql":
        if not storage.startswith("postgresql+psycopg://"):
            raise ValueError(
                "PostgreSQL Phase 2 storage must use postgresql+psycopg://"
            )
        return
    if approved_backend == "scg_validated_journal_storage":
        if not storage.startswith("journal://"):
            raise ValueError("Approved JournalStorage requires a journal:// URL")
        path = Path(storage.removeprefix("journal://")).expanduser()
        if not path.is_absolute():
            raise ValueError("JournalStorage path must be absolute")
        return
    raise ValueError(f"Unsupported approved storage backend: {approved_backend!r}")


def credential_free_storage_identity(
    storage: str,
    *,
    approved_backend: str,
) -> dict[str, Any]:
    """Identify one storage instance without persisting credentials."""

    validate_storage_url(storage, approved_backend)
    if storage.startswith("journal://"):
        path = Path(storage.removeprefix("journal://")).expanduser().resolve()
        return {
            "backend": approved_backend,
            "journal_path": str(path),
        }
    parsed = urlsplit(storage)
    safe_query = [
        (name, value)
        for name, value in parse_qsl(parsed.query, keep_blank_values=True)
        if not any(
            secret in name.lower()
            for secret in ("password", "passwd", "secret", "token", "key")
        )
    ]
    return {
        "backend": approved_backend,
        "scheme": parsed.scheme,
        "hostname": parsed.hostname,
        "port": parsed.port,
        "database": parsed.path.lstrip("/"),
        "username": parsed.username,
        "safe_query": sorted(safe_query),
    }


def build_ledger_contract(
    spec: Mapping[str, Any],
    *,
    output_root: Path,
    storage: str,
    code_identity: Mapping[str, Any],
    data_identity: Mapping[str, Any],
    capability_identity: Mapping[str, Any],
) -> dict[str, Any]:
    public_spec = {
        key: value for key, value in spec.items() if not str(key).startswith("_")
    }
    storage_identity = credential_free_storage_identity(
        storage,
        approved_backend=str(spec["storage"]["backend"]),
    )
    contract = {
        "schema_version": 1,
        "study_name": spec["study_name"],
        "spec_hash": stable_hash(public_spec),
        "code_commit": code_identity.get("commit"),
        "code_dirty": code_identity.get("dirty"),
        "data_hash": stable_hash(data_identity),
        "capability_hash": stable_hash(capability_identity),
        "claims_output_root": str(output_root.expanduser().resolve()),
        "storage_identity_sha256": stable_hash(storage_identity),
        "compute_budget": dict(spec["study"]["compute_budget"]),
        "requested_trials": int(spec["study"]["requested_trials"]),
    }
    return {
        **contract,
        "ledger_contract_sha256": stable_hash(contract),
    }


def ensure_ledger_contract(
    output_root: Path,
    expected: Mapping[str, Any],
    *,
    create: bool,
) -> Path:
    """Create once for sampled workers; fixed runners may only consume it."""

    root = output_root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    path = root / "phase2_ledger_contract.json"
    with _compute_budget_lock(root):
        if not path.exists():
            if not create:
                raise RuntimeError(
                    "Shared Phase 2 ledger contract is missing; start/validate "
                    "the Optuna study at the exact --budget-root first"
                )
            atomic_write_json(path, dict(expected))
            path.chmod(0o444)
        try:
            actual = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError("Shared Phase 2 ledger contract is invalid") from exc
        if actual != dict(expected):
            raise RuntimeError(
                "Shared Phase 2 ledger contract does not match this "
                "spec/code/data/capability/storage/output identity"
            )
    return path


def build_storage(
    storage: str, *, approved_backend: str, heartbeat_interval: int = 60
) -> Any:
    """Build an Optuna storage while rejecting unsafe shared SQLite."""

    validate_storage_url(storage, approved_backend)
    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError("Install optional nas dependencies to run the study") from exc
    if storage.startswith("journal://"):
        path = storage.removeprefix("journal://")
        backend = optuna.storages.journal.JournalFileBackend(path)
        return optuna.storages.JournalStorage(backend)
    return optuna.storages.RDBStorage(
        url=storage,
        heartbeat_interval=heartbeat_interval,
        grace_period=heartbeat_interval * 3,
    )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Training adapter did not produce valid JSON at {path}"
        ) from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"Training adapter output must be an object: {path}")
    return value


def _claim_payload(path: Path, expected_hash: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError) as exc:
        raise RuntimeError(f"Invalid distributed trial claim: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"Distributed trial claim is not an object: {path}")
    if payload.get("schema_version") != 1:
        raise RuntimeError(f"Unsupported distributed trial claim schema: {path}")
    if payload.get("trial_hash") != expected_hash:
        raise RuntimeError(f"Distributed trial claim hash mismatch: {path}")
    if payload.get("state") not in {*_ACTIVE_CLAIM_STATES, "terminal"}:
        raise RuntimeError(f"Distributed trial claim has invalid state: {path}")
    if not isinstance(payload.get("lease_id"), str) or not payload["lease_id"]:
        raise RuntimeError(f"Distributed trial claim has no lease ID: {path}")
    run_dir = payload.get("run_dir")
    if not isinstance(run_dir, str) or not Path(run_dir).is_absolute():
        raise RuntimeError(f"Distributed trial claim has invalid run_dir: {path}")
    if not isinstance(payload.get("updated_at_unix"), (int, float)):
        raise RuntimeError(f"Distributed trial claim has invalid timestamp: {path}")
    if payload["state"] == "terminal":
        if payload.get("outcome") not in _TERMINAL_CLAIM_OUTCOMES:
            raise RuntimeError(f"Distributed trial claim has invalid outcome: {path}")
    if payload["state"] == "recovery_queued":
        token = payload.get("recovery_token")
        if not isinstance(token, str) or not token:
            raise RuntimeError(
                f"Distributed recovery claim has no recovery token: {path}"
            )
    claim_kind = payload.get("claim_kind", "optuna_study")
    if claim_kind not in _CLAIM_KINDS:
        raise RuntimeError(f"Distributed claim has invalid claim_kind: {path}")
    payload["claim_kind"] = claim_kind
    return payload


def _write_exclusive_json(path: Path, value: Mapping[str, Any]) -> bool:
    payload = (
        json.dumps(value, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        return False
    with os.fdopen(fd, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return True


@contextmanager
def _claim_mutation_lock(
    claim_path: Path,
    *,
    timeout_seconds: float = 5.0,
) -> Iterator[None]:
    """Serialize terminal/reclaim/release mutations with an O_EXCL lock."""

    lock_path = claim_path.with_suffix(".lock")
    deadline = time.monotonic() + timeout_seconds
    while True:
        lock_payload = {
            "schema_version": 1,
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "created_at_unix": time.time(),
        }
        if _write_exclusive_json(lock_path, lock_payload):
            break
        try:
            lock_age = time.time() - lock_path.stat().st_mtime
        except FileNotFoundError:
            continue
        if lock_age > _CLAIM_LOCK_STALE_AFTER_SECONDS:
            try:
                lock_path.unlink()
            except FileNotFoundError:
                pass
            continue
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"Timed out acquiring distributed claim lock: {lock_path}"
            )
        time.sleep(0.01)
    try:
        yield
    finally:
        lock_path.unlink(missing_ok=True)


def _running_claim_value(
    *,
    trial_hash: str,
    trial_number: int,
    run_dir: Path,
    timestamp: float,
    lease_id: str,
    claim_kind: str,
    created_at: float | None = None,
    reclaimed_count: int = 0,
    previous_lease_id: str | None = None,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "schema_version": 1,
        "trial_hash": trial_hash,
        "state": "running",
        "outcome": None,
        "optuna_trial_id": trial_number,
        "run_dir": str(run_dir),
        "lease_id": lease_id,
        "claim_kind": claim_kind,
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "created_at_unix": timestamp if created_at is None else created_at,
        "updated_at_unix": timestamp,
        "reclaimed_count": reclaimed_count,
    }
    if previous_lease_id is not None:
        value["previous_lease_id"] = previous_lease_id
        value["reclaimed_at_unix"] = timestamp
    return value


def _positive_finite_gpu_hours(value: Any, *, field: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise RuntimeError(f"{field} must be a positive finite number")
    return float(value)


def _nonnegative_finite_gpu_hours(value: Any, *, field: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise RuntimeError(f"{field} must be a nonnegative finite number")
    return float(value)


def _compute_budget_lock(output_root: Path):
    claims = output_root / "claims"
    claims.mkdir(parents=True, exist_ok=True)
    return _claim_mutation_lock(
        claims / ".compute-budget.json",
        timeout_seconds=30.0,
    )


def _recorded_active_gpu_hours(payload: Mapping[str, Any]) -> float:
    recorded = 0.0
    for collection_name in (
        "successful_stages",
        "failed_stages",
        "systemic_attempts",
    ):
        entries = payload.get(collection_name, [])
        if not isinstance(entries, list):
            raise RuntimeError(f"Claim {collection_name} is corrupt")
        for entry in entries:
            if not isinstance(entry, Mapping):
                raise RuntimeError(f"Claim {collection_name} entry is corrupt")
            recorded += _nonnegative_finite_gpu_hours(
                entry.get("gpu_hours_actual"),
                field=f"{collection_name} gpu_hours_actual",
            )
    inflight = payload.get("inflight_stage")
    if inflight is not None:
        if not isinstance(inflight, Mapping):
            raise RuntimeError("Claim inflight_stage is corrupt")
        recorded += _nonnegative_finite_gpu_hours(
            inflight.get("gpu_hours_accounted"),
            field="inflight_stage gpu_hours_accounted",
        )
    return recorded


def _accrue_inflight_stage(
    payload: dict[str, Any],
    *,
    timestamp: float,
) -> None:
    inflight = payload.get("inflight_stage")
    if inflight is None:
        return
    if not isinstance(inflight, dict):
        raise RuntimeError("Claim inflight_stage is corrupt")
    last = inflight.get("last_accounted_at_unix")
    world_size = inflight.get("world_size")
    seconds = inflight.get("gpu_seconds_accounted")
    if (
        not isinstance(last, (int, float))
        or not isinstance(world_size, int)
        or isinstance(world_size, bool)
        or world_size <= 0
    ):
        raise RuntimeError("Claim inflight_stage timing metadata is corrupt")
    accounted_seconds = _nonnegative_finite_gpu_hours(
        seconds,
        field="inflight_stage gpu_seconds_accounted",
    )
    delta = max(0.0, timestamp - float(last)) * world_size
    accounted_seconds += delta
    inflight.update(
        {
            "last_accounted_at_unix": timestamp,
            "gpu_seconds_accounted": accounted_seconds,
            "gpu_hours_accounted": accounted_seconds / 3600.0,
        }
    )


def begin_claim_stage(
    claim: TrialClaim,
    *,
    stage: str,
    world_size: int,
    heartbeat_interval_seconds: float,
    now: Callable[[], float] = time.time,
) -> None:
    """Persist a bounded-error GPU-time meter before launching an adapter."""

    if stage not in {"screen", "final"}:
        raise ValueError(f"Unknown adapter stage: {stage}")
    if (
        not isinstance(world_size, int)
        or isinstance(world_size, bool)
        or world_size <= 0
    ):
        raise ValueError("world_size must be a positive integer")
    if heartbeat_interval_seconds <= 0:
        raise ValueError("heartbeat_interval_seconds must be positive")
    with _claim_mutation_lock(claim.path):
        payload = _claim_payload(claim.path, claim.trial_hash)
        if payload["state"] != "running" or payload["lease_id"] != claim.lease_id:
            raise RuntimeError("Distributed trial claim lease was lost")
        if payload.get("inflight_stage") is not None:
            raise RuntimeError("Distributed claim already has an in-flight stage")
        timestamp = float(now())
        payload["inflight_stage"] = {
            "stage": stage,
            "lease_id": claim.lease_id,
            "started_at_unix": timestamp,
            "last_accounted_at_unix": timestamp,
            "world_size": world_size,
            "heartbeat_interval_seconds": float(heartbeat_interval_seconds),
            "gpu_seconds_accounted": 0.0,
            "gpu_hours_accounted": 0.0,
            "accounting_method": "heartbeat_wall_clock_times_world_size",
        }
        atomic_write_json(claim.path, payload)


def _roll_inflight_into_systemic_attempt(
    payload: dict[str, Any],
    *,
    timestamp: float,
) -> None:
    """Convert a preempted stage meter into immutable recovery accounting."""

    inflight = payload.pop("inflight_stage", None)
    if inflight is None:
        return
    if not isinstance(inflight, dict):
        raise RuntimeError("Claim inflight_stage is corrupt")
    seconds = _nonnegative_finite_gpu_hours(
        inflight.get("gpu_seconds_accounted"),
        field="interrupted inflight gpu_seconds_accounted",
    )
    hours = _nonnegative_finite_gpu_hours(
        inflight.get("gpu_hours_accounted"),
        field="interrupted inflight gpu_hours_accounted",
    )
    attempts = payload.get("systemic_attempts", [])
    if not isinstance(attempts, list):
        raise RuntimeError("Claim systemic_attempts is corrupt")
    attempts.append(
        {
            "failed_at_unix": timestamp,
            "failure": {
                "scope": "systemic",
                "code": "worker_lost_during_adapter_stage",
                "detail": (
                    "GPU time estimated through the last durable claim "
                    "heartbeat before recovery"
                ),
            },
            "stage": inflight.get("stage"),
            "gpu_seconds_actual": seconds,
            "gpu_hours_actual": hours,
            "accounting_method": inflight.get("accounting_method"),
            "maximum_unaccounted_gpu_seconds": (
                float(inflight.get("heartbeat_interval_seconds", 0.0))
                * int(inflight.get("world_size", 1))
            ),
        }
    )
    payload["systemic_attempts"] = attempts


def _compute_budget_usage(
    output_root: Path,
    *,
    exclude_path: Path | None = None,
) -> tuple[float, float, int]:
    """Return terminal actual GPU-hours, active reservations, and active count."""

    terminal_actual = 0.0
    active_reserved = 0.0
    active_count = 0
    claims_dir = output_root / "claims"
    for path in sorted(claims_dir.glob("*.json")):
        if exclude_path is not None and path == exclude_path:
            continue
        payload = _claim_payload(path, path.stem)
        if payload["state"] == "terminal":
            terminal_actual += _nonnegative_finite_gpu_hours(
                payload.get("gpu_hours_actual"),
                field=f"{path}: gpu_hours_actual",
            )
        else:
            active_count += 1
            reservation = _positive_finite_gpu_hours(
                payload.get("gpu_hours_reserved"),
                field=f"{path}: gpu_hours_reserved",
            )
            active_reserved += max(
                reservation,
                _recorded_active_gpu_hours(payload),
            )
    return terminal_actual, active_reserved, active_count


def _reserve_compute_budget(
    output_root: Path,
    *,
    maximum_total_gpu_hours: float,
    maximum_full_trial_gpu_hours: float,
    maximum_concurrent_trials: int,
    exclude_path: Path | None = None,
) -> tuple[float, float, float, int]:
    """Validate room for one more reservation under the held budget lock."""

    maximum_total = _positive_finite_gpu_hours(
        maximum_total_gpu_hours,
        field="maximum_total_gpu_hours",
    )
    full_trial = _positive_finite_gpu_hours(
        maximum_full_trial_gpu_hours,
        field="maximum_full_trial_gpu_hours",
    )
    if full_trial > maximum_total:
        raise RuntimeError(
            "maximum_full_trial_gpu_hours exceeds maximum_total_gpu_hours"
        )
    if (
        not isinstance(maximum_concurrent_trials, int)
        or isinstance(maximum_concurrent_trials, bool)
        or maximum_concurrent_trials <= 0
    ):
        raise RuntimeError("maximum_concurrent_trials must be a positive integer")
    terminal_actual, active_reserved, active_count = _compute_budget_usage(
        output_root,
        exclude_path=exclude_path,
    )
    if active_count >= maximum_concurrent_trials:
        raise ComputeBudgetExhausted(
            "Concurrent-trial ceiling exhausted: "
            f"active={active_count}, maximum={maximum_concurrent_trials}"
        )
    projected = terminal_actual + active_reserved + full_trial
    if projected > maximum_total + 1e-12:
        raise ComputeBudgetExhausted(
            "GPU-hour ceiling exhausted: "
            f"actual={terminal_actual:.6f}, "
            f"reserved={active_reserved:.6f}, "
            f"next={full_trial:.6f}, ceiling={maximum_total:.6f}"
        )
    return terminal_actual, active_reserved, full_trial, active_count


def claim_trial_hash(
    output_root: Path,
    trial_hash: str,
    trial_number: int,
    *,
    run_dir: Path | None = None,
    stale_after_seconds: int = DEFAULT_CLAIM_STALE_AFTER_SECONDS,
    recovery_token: str | None = None,
    maximum_total_gpu_hours: float | None = None,
    maximum_full_trial_gpu_hours: float | None = None,
    maximum_concurrent_trials: int | None = None,
    claim_kind: str = "optuna_study",
    now: Callable[[], float] = time.time,
) -> TrialClaim | None:
    """Atomically acquire or reclaim one content-addressed outcome lease.

    A live running claim and every terminal claim are deduplicated. A running
    claim older than ``stale_after_seconds`` may be reclaimed under an O_EXCL
    mutation lock. A startup-queued recovery can be reclaimed immediately, but
    only with its one-time token. Reclamation preserves the original run
    directory so a checkpoint left by a preempted worker can be resumed.
    """

    if stale_after_seconds <= 0:
        raise ValueError("stale_after_seconds must be positive")
    output_root = output_root.expanduser().resolve()
    claims = output_root / "claims"
    claims.mkdir(parents=True, exist_ok=True)
    path = claims / f"{trial_hash}.json"
    requested_run_dir = (
        run_dir.expanduser().resolve()
        if run_dir is not None
        else (
            output_root / f"trial-{trial_number:07d}-{trial_hash[:12]}"
        ).resolve()
    )
    timestamp = float(now())
    lease_id = uuid.uuid4().hex
    if claim_kind not in _CLAIM_KINDS:
        raise ValueError(f"Unknown claim_kind: {claim_kind!r}")
    budget_requested = any(
        value is not None
        for value in (
            maximum_total_gpu_hours,
            maximum_full_trial_gpu_hours,
            maximum_concurrent_trials,
        )
    )
    if budget_requested and (
        maximum_total_gpu_hours is None
        or maximum_full_trial_gpu_hours is None
        or maximum_concurrent_trials is None
    ):
        raise ValueError(
            "Total GPU-hours, full-trial GPU-hours, and maximum concurrency "
            "are all required for a budgeted claim"
        )
    budget_context = (
        _compute_budget_lock(output_root)
        if budget_requested
        else nullcontext()
    )
    with budget_context:
        initial = _running_claim_value(
            trial_hash=trial_hash,
            trial_number=trial_number,
            run_dir=requested_run_dir,
            timestamp=timestamp,
            lease_id=lease_id,
            claim_kind=claim_kind,
        )
        if recovery_token is None and not path.exists():
            if budget_requested:
                _, _, reservation, _ = _reserve_compute_budget(
                    output_root,
                    maximum_total_gpu_hours=float(maximum_total_gpu_hours),
                    maximum_full_trial_gpu_hours=float(
                        maximum_full_trial_gpu_hours
                    ),
                    maximum_concurrent_trials=int(maximum_concurrent_trials),
                )
                initial["gpu_hours_reserved"] = reservation
                initial["compute_budget_maximum_total_gpu_hours"] = float(
                    maximum_total_gpu_hours
                )
                initial["compute_budget_maximum_concurrent_trials"] = int(
                    maximum_concurrent_trials
                )
            if _write_exclusive_json(path, initial):
                return TrialClaim(
                    path=path,
                    trial_hash=trial_hash,
                    trial_number=trial_number,
                    run_dir=requested_run_dir,
                    lease_id=lease_id,
                )
        if recovery_token is not None and not path.is_file():
            raise RuntimeError(
                f"Startup recovery claim disappeared before acquisition: {path}"
            )

        with _claim_mutation_lock(path):
            existing = _claim_payload(path, trial_hash)
            if existing["claim_kind"] != claim_kind:
                raise RuntimeError(
                    "Distributed claim kind mismatch: "
                    f"{existing['claim_kind']} != {claim_kind}"
                )
            if existing["state"] == "terminal":
                return None
            if existing["state"] == "recovery_queued":
                if recovery_token != existing["recovery_token"]:
                    return None
            else:
                if recovery_token is not None:
                    return None
                age = timestamp - float(existing["updated_at_unix"])
                if age <= stale_after_seconds:
                    return None

            _roll_inflight_into_systemic_attempt(
                existing,
                timestamp=timestamp,
            )
            reservation = existing.get("gpu_hours_reserved")
            if reservation is not None and not budget_requested:
                raise RuntimeError(
                    "Budgeted trial claims require explicit GPU-hour limits "
                    "when reclaimed"
                )
            if budget_requested:
                consumed = _recorded_active_gpu_hours(existing)
                cumulative_recovery_reservation = consumed + float(
                    maximum_full_trial_gpu_hours
                )
                _, _, reservation, _ = _reserve_compute_budget(
                    output_root,
                    maximum_total_gpu_hours=float(
                        maximum_total_gpu_hours
                    ),
                    maximum_full_trial_gpu_hours=(
                        cumulative_recovery_reservation
                    ),
                    maximum_concurrent_trials=int(
                        maximum_concurrent_trials
                    ),
                    exclude_path=path,
                )
            preserved_run_dir = Path(existing["run_dir"])
            reclaimed = _running_claim_value(
                trial_hash=trial_hash,
                trial_number=trial_number,
                run_dir=preserved_run_dir,
                timestamp=timestamp,
                lease_id=lease_id,
                claim_kind=claim_kind,
                created_at=float(existing.get("created_at_unix", timestamp)),
                reclaimed_count=int(existing.get("reclaimed_count", 0)) + 1,
                previous_lease_id=str(existing["lease_id"]),
            )
            if reservation is not None:
                reclaimed["gpu_hours_reserved"] = reservation
                reclaimed["compute_budget_maximum_total_gpu_hours"] = float(
                    maximum_total_gpu_hours
                )
                reclaimed["compute_budget_maximum_concurrent_trials"] = int(
                    maximum_concurrent_trials
                )
            for accounting_key in (
                "successful_stages",
                "failed_stages",
                "systemic_attempts",
            ):
                if isinstance(existing.get(accounting_key), list):
                    reclaimed[accounting_key] = list(existing[accounting_key])
            atomic_write_json(path, reclaimed)
    return TrialClaim(
        path=path,
        trial_hash=trial_hash,
        trial_number=trial_number,
        run_dir=preserved_run_dir,
        lease_id=lease_id,
        reclaimed=True,
    )


def finalize_trial_claim(
    claim: TrialClaim,
    outcome: str,
    *,
    gpu_hours_actual: float | None = None,
    terminal_artifacts: Mapping[str, Mapping[str, Any]] | None = None,
    now: Callable[[], float] = time.time,
) -> None:
    """Atomically mark an owned claim terminal and release its reservation."""

    if outcome not in _TERMINAL_CLAIM_OUTCOMES:
        raise ValueError(f"Invalid terminal claim outcome: {outcome!r}")
    output_root = claim.path.parent.parent
    with _compute_budget_lock(output_root):
        with _claim_mutation_lock(claim.path):
            payload = _claim_payload(claim.path, claim.trial_hash)
            reservation = payload.get("gpu_hours_reserved")
            actual: float | None = None
            if reservation is not None:
                reservation = _positive_finite_gpu_hours(
                    reservation,
                    field=f"{claim.path}: gpu_hours_reserved",
                )
                actual = _nonnegative_finite_gpu_hours(
                    gpu_hours_actual,
                    field="gpu_hours_actual",
                )
            elif gpu_hours_actual is not None:
                actual = _nonnegative_finite_gpu_hours(
                    gpu_hours_actual,
                    field="gpu_hours_actual",
                )
            if payload["state"] == "terminal":
                if (
                    payload.get("lease_id") == claim.lease_id
                    and payload.get("outcome") == outcome
                    and (
                        actual is None
                        or payload.get("gpu_hours_actual") == actual
                    )
                    and (
                        terminal_artifacts is None
                        or payload.get("terminal_artifacts")
                        == {
                            str(name): dict(identity)
                            for name, identity in terminal_artifacts.items()
                        }
                    )
                ):
                    return
                raise RuntimeError("Distributed trial claim is already terminal")
            if payload.get("lease_id") != claim.lease_id:
                raise RuntimeError("Distributed trial claim lease was lost")
            timestamp = float(now())
            payload.update(
                {
                    "state": "terminal",
                    "outcome": outcome,
                    "updated_at_unix": timestamp,
                    "terminal_at_unix": timestamp,
                }
            )
            payload.pop("inflight_stage", None)
            if terminal_artifacts is not None:
                normalized_artifacts = {
                    str(name): dict(identity)
                    for name, identity in terminal_artifacts.items()
                }
                if (
                    not {"trial_manifest.json", "trial_summary.json"}.issubset(
                        normalized_artifacts
                    )
                    or not set(normalized_artifacts).issubset(
                        _TERMINAL_BINDING_NAMES
                    )
                ):
                    raise ValueError("Invalid terminal artifact binding set")
                for name, identity in normalized_artifacts.items():
                    if (
                        not isinstance(identity.get("byte_size"), int)
                        or isinstance(identity.get("byte_size"), bool)
                        or int(identity["byte_size"]) <= 0
                        or not isinstance(identity.get("sha256"), str)
                        or len(str(identity["sha256"])) != 64
                    ):
                        raise ValueError(
                            f"Invalid terminal artifact identity: {name}"
                        )
                    if name == "checkpoint_bundle" and (
                        not isinstance(identity.get("path"), str)
                        or not isinstance(identity.get("file_count"), int)
                        or isinstance(identity.get("file_count"), bool)
                        or int(identity["file_count"]) <= 0
                        or identity.get("digest_algorithm")
                        != (
                            "sha256(path_nul_size_nul_content_sha256_newline_v1)"
                        )
                    ):
                        raise ValueError(
                            "Invalid terminal checkpoint bundle identity"
                        )
                payload["terminal_artifacts"] = normalized_artifacts
            if actual is not None:
                payload["gpu_hours_actual"] = actual
                payload["gpu_hours_reservation_released"] = reservation
                if (
                    reservation is not None
                    and actual > reservation + 1e-12
                ):
                    payload["gpu_hours_reservation_overrun"] = (
                        actual - reservation
                    )
                    payload["compute_budget_violation"] = (
                        "full_trial_gpu_hours_exceeded"
                    )
            atomic_write_json(claim.path, payload)


def release_trial_claim(claim: TrialClaim) -> bool:
    """Release an owned running claim after a systemic failure."""

    output_root = claim.path.parent.parent
    with _compute_budget_lock(output_root):
        with _claim_mutation_lock(claim.path):
            try:
                payload = _claim_payload(claim.path, claim.trial_hash)
            except RuntimeError:
                if not claim.path.exists():
                    return False
                raise
            if (
                payload["state"] == "terminal"
                or payload["lease_id"] != claim.lease_id
            ):
                return False
            claim.path.unlink()
            return True


def abandon_trial_claim_for_recovery(
    claim: TrialClaim,
    *,
    failure: Mapping[str, Any],
    gpu_seconds_actual: float,
    gpu_hours_actual: float,
    now: Callable[[], float] = time.time,
) -> None:
    """Preserve a budgeted checkpoint but make a systemic attempt recoverable."""

    seconds = _nonnegative_finite_gpu_hours(
        gpu_seconds_actual,
        field="gpu_seconds_actual",
    )
    hours = _nonnegative_finite_gpu_hours(
        gpu_hours_actual,
        field="gpu_hours_actual",
    )
    if not math.isclose(
        hours,
        seconds / 3600.0,
        rel_tol=1e-9,
        abs_tol=1e-12,
    ):
        raise RuntimeError(
            "gpu_hours_actual must equal gpu_seconds_actual / 3600"
        )
    output_root = claim.path.parent.parent
    with _compute_budget_lock(output_root):
        with _claim_mutation_lock(claim.path):
            payload = _claim_payload(claim.path, claim.trial_hash)
            if payload["state"] != "running":
                raise RuntimeError("Only a running claim can be abandoned")
            if payload["lease_id"] != claim.lease_id:
                raise RuntimeError("Distributed trial claim lease was lost")
            if payload.get("gpu_hours_reserved") is None:
                raise RuntimeError(
                    "Only a compute-budgeted claim can be preserved for recovery"
                )
            timestamp = float(now())
            controller_seconds = 0.0
            controller_hours = 0.0
            inflight = payload.get("inflight_stage")
            if inflight is not None:
                if not isinstance(inflight, dict):
                    raise RuntimeError("Claim inflight_stage is corrupt")
                _accrue_inflight_stage(payload, timestamp=timestamp)
                controller_seconds = _nonnegative_finite_gpu_hours(
                    inflight.get("gpu_seconds_accounted"),
                    field="inflight_stage gpu_seconds_accounted",
                )
                controller_hours = _nonnegative_finite_gpu_hours(
                    inflight.get("gpu_hours_accounted"),
                    field="inflight_stage gpu_hours_accounted",
                )
            billed_seconds = max(seconds, controller_seconds)
            billed_hours = billed_seconds / 3600.0
            payload.pop("inflight_stage", None)
            attempts = payload.get("systemic_attempts", [])
            if not isinstance(attempts, list):
                raise RuntimeError("Claim systemic_attempts is corrupt")
            attempts.append(
                {
                    "failed_at_unix": timestamp,
                    "failure": dict(failure),
                    "gpu_seconds_actual": billed_seconds,
                    "gpu_hours_actual": billed_hours,
                    "adapter_gpu_seconds_reported": seconds,
                    "adapter_gpu_hours_reported": hours,
                    "controller_gpu_seconds_accounted": controller_seconds,
                    "controller_gpu_hours_accounted": controller_hours,
                    "accounting_method": (
                        "max_adapter_report_and_heartbeat_wall_clock"
                        if inflight is not None
                        else "adapter_report_no_inflight_meter"
                    ),
                }
            )
            payload.update(
                {
                    "updated_at_unix": 0.0,
                    "abandoned_for_recovery": True,
                    "systemic_attempts": attempts,
                }
            )
            atomic_write_json(claim.path, payload)


def record_successful_claim_stage(
    claim: TrialClaim,
    *,
    stage: str,
    gpu_seconds_actual: float,
    gpu_hours_actual: float,
    now: Callable[[], float] = time.time,
) -> None:
    """Persist one validated adapter invocation before later work can crash."""

    if stage not in {"screen", "final"}:
        raise ValueError(f"Unknown adapter stage: {stage}")
    seconds = _nonnegative_finite_gpu_hours(
        gpu_seconds_actual,
        field="gpu_seconds_actual",
    )
    hours = _nonnegative_finite_gpu_hours(
        gpu_hours_actual,
        field="gpu_hours_actual",
    )
    if not math.isclose(
        hours,
        seconds / 3600.0,
        rel_tol=1e-9,
        abs_tol=1e-12,
    ):
        raise RuntimeError(
            "gpu_hours_actual must equal gpu_seconds_actual / 3600"
        )
    output_root = claim.path.parent.parent
    with _compute_budget_lock(output_root):
        with _claim_mutation_lock(claim.path):
            payload = _claim_payload(claim.path, claim.trial_hash)
            if (
                payload["state"] != "running"
                or payload["lease_id"] != claim.lease_id
            ):
                raise RuntimeError("Distributed trial claim lease was lost")
            stages = payload.get("successful_stages", [])
            if not isinstance(stages, list):
                raise RuntimeError("Claim successful_stages is corrupt")
            timestamp = float(now())
            controller_seconds = 0.0
            controller_hours = 0.0
            inflight = payload.get("inflight_stage")
            if inflight is not None:
                if (
                    not isinstance(inflight, dict)
                    or inflight.get("stage") != stage
                    or inflight.get("lease_id") != claim.lease_id
                ):
                    raise RuntimeError(
                        "Claim in-flight stage does not match successful stage"
                    )
                _accrue_inflight_stage(payload, timestamp=timestamp)
                controller_seconds = _nonnegative_finite_gpu_hours(
                    inflight.get("gpu_seconds_accounted"),
                    field="inflight_stage gpu_seconds_accounted",
                )
                controller_hours = _nonnegative_finite_gpu_hours(
                    inflight.get("gpu_hours_accounted"),
                    field="inflight_stage gpu_hours_accounted",
                )
            billed_seconds = max(seconds, controller_seconds)
            billed_hours = billed_seconds / 3600.0
            stages.append(
                {
                    "stage": stage,
                    "recorded_at_unix": timestamp,
                    "lease_id": claim.lease_id,
                    "gpu_seconds_actual": billed_seconds,
                    "gpu_hours_actual": billed_hours,
                    "adapter_gpu_seconds_reported": seconds,
                    "adapter_gpu_hours_reported": hours,
                    "controller_gpu_seconds_accounted": controller_seconds,
                    "controller_gpu_hours_accounted": controller_hours,
                    "accounting_method": (
                        "max_adapter_report_and_heartbeat_wall_clock"
                        if inflight is not None
                        else "adapter_report_no_inflight_meter"
                    ),
                }
            )
            payload["successful_stages"] = stages
            payload.pop("inflight_stage", None)
            atomic_write_json(claim.path, payload)


def record_failed_claim_stage(
    claim: TrialClaim,
    *,
    stage: str,
    failure: Mapping[str, Any],
    gpu_seconds_actual: float,
    gpu_hours_actual: float,
    now: Callable[[], float] = time.time,
) -> None:
    """Durably charge a terminal trial-local stage before writing its summary."""

    if stage not in {"screen", "final"}:
        raise ValueError(f"Unknown adapter stage: {stage}")
    seconds = _nonnegative_finite_gpu_hours(
        gpu_seconds_actual,
        field="gpu_seconds_actual",
    )
    hours = _nonnegative_finite_gpu_hours(
        gpu_hours_actual,
        field="gpu_hours_actual",
    )
    if not math.isclose(
        hours,
        seconds / 3600.0,
        rel_tol=1e-9,
        abs_tol=1e-12,
    ):
        raise RuntimeError(
            "gpu_hours_actual must equal gpu_seconds_actual / 3600"
        )
    output_root = claim.path.parent.parent
    with _compute_budget_lock(output_root):
        with _claim_mutation_lock(claim.path):
            payload = _claim_payload(claim.path, claim.trial_hash)
            if (
                payload["state"] != "running"
                or payload["lease_id"] != claim.lease_id
            ):
                raise RuntimeError("Distributed trial claim lease was lost")
            timestamp = float(now())
            controller_seconds = 0.0
            controller_hours = 0.0
            inflight = payload.get("inflight_stage")
            if inflight is not None:
                if (
                    not isinstance(inflight, dict)
                    or inflight.get("stage") != stage
                    or inflight.get("lease_id") != claim.lease_id
                ):
                    raise RuntimeError(
                        "Claim in-flight stage does not match failed stage"
                    )
                _accrue_inflight_stage(payload, timestamp=timestamp)
                controller_seconds = _nonnegative_finite_gpu_hours(
                    inflight.get("gpu_seconds_accounted"),
                    field="inflight_stage gpu_seconds_accounted",
                )
                controller_hours = _nonnegative_finite_gpu_hours(
                    inflight.get("gpu_hours_accounted"),
                    field="inflight_stage gpu_hours_accounted",
                )
            billed_seconds = max(seconds, controller_seconds)
            failed = payload.get("failed_stages", [])
            if not isinstance(failed, list):
                raise RuntimeError("Claim failed_stages is corrupt")
            failed.append(
                {
                    "stage": stage,
                    "recorded_at_unix": timestamp,
                    "lease_id": claim.lease_id,
                    "failure": dict(failure),
                    "gpu_seconds_actual": billed_seconds,
                    "gpu_hours_actual": billed_seconds / 3600.0,
                    "adapter_gpu_seconds_reported": seconds,
                    "adapter_gpu_hours_reported": hours,
                    "controller_gpu_seconds_accounted": controller_seconds,
                    "controller_gpu_hours_accounted": controller_hours,
                    "accounting_method": (
                        "max_adapter_report_and_heartbeat_wall_clock"
                        if inflight is not None
                        else "adapter_report_no_inflight_meter"
                    ),
                }
            )
            payload["failed_stages"] = failed
            payload.pop("inflight_stage", None)
            atomic_write_json(claim.path, payload)


def heartbeat_trial_claim(
    claim: TrialClaim,
    *,
    now: Callable[[], float] = time.time,
) -> bool:
    """Refresh an owned running lease without reviving a lost claim."""

    with _claim_mutation_lock(claim.path):
        try:
            payload = _claim_payload(claim.path, claim.trial_hash)
        except RuntimeError:
            if not claim.path.exists():
                return False
            raise
        if payload["state"] != "running" or payload["lease_id"] != claim.lease_id:
            return False
        timestamp = float(now())
        _accrue_inflight_stage(payload, timestamp=timestamp)
        payload.update(
            {
                "updated_at_unix": timestamp,
                "last_heartbeat_at_unix": timestamp,
                "heartbeat_count": int(payload.get("heartbeat_count", 0)) + 1,
            }
        )
        atomic_write_json(claim.path, payload)
        return True


@contextmanager
def _claim_heartbeat(
    claim: TrialClaim | None,
    *,
    interval_seconds: float,
) -> Iterator[list[BaseException]]:
    """Keep an adapter lease live while its blocking subprocess runs."""

    errors: list[BaseException] = []
    if claim is None:
        yield errors
        return
    if interval_seconds <= 0:
        raise ValueError("claim heartbeat interval must be positive")
    if not heartbeat_trial_claim(claim):
        raise RuntimeError("Distributed trial claim lease was lost before adapter start")

    stopped = threading.Event()

    def refresh() -> None:
        while not stopped.wait(interval_seconds):
            try:
                if not heartbeat_trial_claim(claim):
                    raise RuntimeError(
                        "Distributed trial claim lease was lost during adapter run"
                    )
            except BaseException as exc:
                errors.append(exc)
                stopped.set()
                return

    thread = threading.Thread(
        target=refresh,
        name=f"phase2-claim-heartbeat-{claim.trial_hash[:12]}",
        daemon=True,
    )
    thread.start()
    try:
        yield errors
    finally:
        stopped.set()
        thread.join(timeout=max(1.0, min(interval_seconds * 2, 10.0)))
        if not errors:
            try:
                if not heartbeat_trial_claim(claim):
                    raise RuntimeError(
                        "Distributed trial claim lease was lost after adapter run"
                    )
            except BaseException as exc:
                errors.append(exc)


def _recovery_parameters(
    spec: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    expected_hash: str,
    code_identity: Mapping[str, Any],
    data_identity: Mapping[str, Any],
    capability_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a stale manifest against the current scientific identity."""

    if manifest.get("trial_hash") != expected_hash:
        raise RuntimeError(
            f"Stale trial manifest hash does not match claim {expected_hash}"
        )
    try:
        recomputed_hash = recompute_trial_hash(manifest)
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Stale trial manifest is incomplete for claim {expected_hash}"
        ) from exc
    if recomputed_hash != expected_hash:
        raise RuntimeError(
            f"Stale trial manifest failed hash validation: {expected_hash}"
        )
    if recompute_promotion_config_hash(manifest) != manifest.get(
        "promotion_config_hash"
    ):
        raise RuntimeError(
            f"Stale trial manifest failed promotion hash validation: {expected_hash}"
        )
    parameters = manifest.get("parameters")
    if not isinstance(parameters, Mapping):
        raise RuntimeError(
            f"Stale trial manifest has no parameters: {expected_hash}"
        )
    if parameters.get("architecture_mode") != "updated_sweep":
        raise RuntimeError(
            f"Study recovery only supports updated_sweep trials: {expected_hash}"
        )
    rebuilt = build_trial_manifest(
        spec,
        parameters,
        seed=int(spec["protocol"]["seed"]),
        trial_number=None,
        code_identity=code_identity,
        data_identity=data_identity,
        capability_identity=capability_identity,
    )
    if rebuilt["trial_hash"] != expected_hash:
        raise RuntimeError(
            "Stale trial belongs to a different spec/code/data/capability "
            f"identity: {expected_hash}"
        )
    return {
        name: parameters[name]
        for name in spec["axes"]
        if name in parameters
    }


def queue_stale_trial_recoveries(
    study: Any,
    *,
    spec: Mapping[str, Any],
    output_root: Path,
    code_identity: Mapping[str, Any],
    data_identity: Mapping[str, Any],
    capability_identity: Mapping[str, Any],
    stale_after_seconds: int = DEFAULT_CLAIM_STALE_AFTER_SECONDS,
    now: Callable[[], float] = time.time,
) -> list[QueuedTrialRecovery]:
    """Atomically queue stale manifests before Optuna samples new parameters.

    The claim is first moved to ``recovery_queued`` with a one-time token.
    Concurrent startup workers therefore cannot enqueue the same recovery.
    The queued Optuna trial carries that token and is the only trial allowed to
    reclaim the preserved checkpoint directory.
    """

    if stale_after_seconds <= 0:
        raise ValueError("stale_after_seconds must be positive")
    output_root = output_root.expanduser().resolve()
    claims_dir = output_root / "claims"
    if not claims_dir.is_dir():
        return []

    queued: list[QueuedTrialRecovery] = []
    timestamp = float(now())
    for claim_path in sorted(claims_dir.glob("*.json")):
        trial_hash = claim_path.stem
        previous_payload: dict[str, Any] | None = None
        recovery_token: str | None = None
        manifest_path: Path | None = None
        run_dir: Path | None = None
        fixed_parameters: dict[str, Any] | None = None
        with _claim_mutation_lock(claim_path):
            payload = _claim_payload(claim_path, trial_hash)
            if payload["claim_kind"] != "optuna_study":
                continue
            if payload["state"] == "terminal":
                continue
            if payload["state"] == "recovery_queued":
                # The Optuna WAITING row already owns this one-time token.
                # Rotating it would invalidate a legitimate queued recovery.
                continue
            age = timestamp - float(payload["updated_at_unix"])
            if age <= stale_after_seconds:
                continue

            run_dir = Path(payload["run_dir"])
            manifest_path = run_dir / "trial_manifest.json"
            try:
                manifest = json.loads(manifest_path.read_text())
            except (FileNotFoundError, OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise RuntimeError(
                    f"Cannot recover stale claim without a valid manifest: {claim_path}"
                ) from exc
            if not isinstance(manifest, Mapping):
                raise RuntimeError(
                    f"Stale trial manifest is not an object: {manifest_path}"
                )
            fixed_parameters = _recovery_parameters(
                spec,
                manifest,
                expected_hash=trial_hash,
                code_identity=code_identity,
                data_identity=data_identity,
                capability_identity=capability_identity,
            )
            previous_payload = dict(payload)
            recovery_token = uuid.uuid4().hex
            payload.update(
                {
                    "state": "recovery_queued",
                    "outcome": None,
                    "recovery_token": recovery_token,
                    "recovery_queued_at_unix": timestamp,
                    "updated_at_unix": timestamp,
                    "recovery_queue_count": int(
                        payload.get("recovery_queue_count", 0)
                    )
                    + 1,
                }
            )
            atomic_write_json(claim_path, payload)

        assert previous_payload is not None
        assert recovery_token is not None
        assert manifest_path is not None
        assert run_dir is not None
        assert fixed_parameters is not None
        try:
            study.enqueue_trial(
                fixed_parameters,
                user_attrs={
                    "phase2_recovery_token": recovery_token,
                    "phase2_recovery_trial_hash": trial_hash,
                    "phase2_recovery_run_dir": str(run_dir),
                },
            )
        except BaseException:
            with _claim_mutation_lock(claim_path):
                current = _claim_payload(claim_path, trial_hash)
                if (
                    current["state"] == "recovery_queued"
                    and current.get("recovery_token") == recovery_token
                ):
                    atomic_write_json(claim_path, previous_payload)
            raise
        queued.append(
            QueuedTrialRecovery(
                trial_hash=trial_hash,
                manifest_path=manifest_path,
                run_dir=run_dir,
                recovery_token=recovery_token,
            )
        )
    return queued


def count_queued_trial_recoveries(output_root: Path) -> int:
    """Count currently reserved startup recoveries for worker budget gating."""

    claims_dir = output_root.expanduser().resolve() / "claims"
    if not claims_dir.is_dir():
        return 0
    queued = 0
    for claim_path in sorted(claims_dir.glob("*.json")):
        payload = _claim_payload(claim_path, claim_path.stem)
        queued += (
            payload["claim_kind"] == "optuna_study"
            and payload["state"] == "recovery_queued"
        )
    return queued


def is_recovery_optuna_trial(trial: Any) -> bool:
    """Return whether an Optuna row replays an existing trial identity."""

    attrs = getattr(trial, "user_attrs", {})
    return isinstance(attrs, Mapping) and (
        attrs.get("phase2_trial_kind") == "recovery"
        or isinstance(attrs.get("phase2_recovery_trial_hash"), str)
        or isinstance(attrs.get("phase2_recovery_token"), str)
    )


def is_fresh_trial_target_guard(trial: Any) -> bool:
    """Return whether a row was pruned before sampling because the quota was full.

    The row itself is useful storage evidence that a concurrent tail worker
    stopped safely, but it is not one of the approved sampled configurations.
    """

    attrs = getattr(trial, "user_attrs", {})
    return (
        isinstance(attrs, Mapping)
        and attrs.get("compute_budget_guard") == "fresh_trial_target_reached"
    )


def count_fresh_optuna_trials(trials: Sequence[Any]) -> int:
    """Count approved sampled configurations.

    Recovery replays and target-guard rows are both Optuna rows, but neither
    consumes the frozen fresh-configuration quota.  Unknown/in-flight rows are
    counted conservatively so concurrent workers cannot all assume the same
    remaining slot before their user attributes become visible.
    """

    return sum(
        not is_recovery_optuna_trial(trial)
        and not is_fresh_trial_target_guard(trial)
        for trial in trials
    )


def count_fresh_trial_target_guards(trials: Sequence[Any]) -> int:
    """Count terminal/in-flight rows created only by the fresh-target guard."""

    return sum(is_fresh_trial_target_guard(trial) for trial in trials)


def _validate_checkpoint_bundle_output(
    output: Mapping[str, Any],
    *,
    run_dir: Path,
    stage: str,
) -> None:
    """Independently hash the checkpoint named by healthy adapter output."""

    if stage == "screen":
        provenance_name = "checkpoint_provenance"
        path_field = "checkpoint_bundle_path"
        size_field = "checkpoint_bundle_byte_size"
        count_field = "checkpoint_bundle_file_count"
        sha_field = "checkpoint_sha256"
        algorithm_field = "checkpoint_digest_algorithm"
    else:
        provenance_name = "resume_provenance"
        path_field = "source_checkpoint_bundle_path"
        size_field = "source_checkpoint_bundle_byte_size"
        count_field = "source_checkpoint_bundle_file_count"
        sha_field = "source_checkpoint_sha256"
        algorithm_field = "source_checkpoint_digest_algorithm"
    provenance = output.get(provenance_name)
    if not isinstance(provenance, Mapping):
        raise SystemicAdapterFailure(
            f"Healthy {stage} output requires {provenance_name}",
            stage=stage,
        )
    try:
        actual = checkpoint_bundle_identity(run_dir, provenance.get(path_field))
    except ValueError as exc:
        raise SystemicAdapterFailure(str(exc), stage=stage) from exc
    expected = {
        "path": provenance.get(path_field),
        "byte_size": provenance.get(size_field),
        "file_count": provenance.get(count_field),
        "sha256": provenance.get(sha_field),
        "digest_algorithm": provenance.get(algorithm_field),
    }
    if expected != actual:
        raise SystemicAdapterFailure(
            f"{stage} checkpoint bundle identity does not match controller hash",
            stage=stage,
        )


def run_adapter_stage(
    command: Sequence[str],
    *,
    manifest_path: Path,
    run_dir: Path,
    stage: str,
    metrics_path: Path,
    claim: TrialClaim | None = None,
    claim_heartbeat_interval_seconds: float = 60.0,
) -> dict[str, Any]:
    """Invoke the adapter and classify its strict JSON result.

    Only ``{"status": "failed", "failure": {"scope": "trial", ...}}`` is
    considered trial-local. Missing/malformed output, an unstructured nonzero
    exit, and every other failure scope are systemic.
    """

    if stage not in {"screen", "final"}:
        raise ValueError(f"Unknown adapter stage: {stage}")
    try:
        expected_manifest = _read_json(manifest_path)
    except RuntimeError as exc:
        raise SystemicAdapterFailure(
            str(exc),
            stage=stage,
        ) from exc
    expected_trial_hash = expected_manifest.get("trial_hash")
    if not isinstance(expected_trial_hash, str):
        raise SystemicAdapterFailure(
            "Trial manifest has no trial_hash",
            stage=stage,
        )
    argv = [
        *command,
        "--manifest",
        str(manifest_path),
        "--run-dir",
        str(run_dir),
        "--stage",
        stage,
        "--metrics-out",
        str(metrics_path),
    ]
    env = os.environ.copy()
    env["PHASE2_TRIAL_MANIFEST"] = str(manifest_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    # A reclaimed run may contain an output from the interrupted attempt. Never
    # classify stale JSON as the result of the process launched below.
    metrics_path.unlink(missing_ok=True)
    log_path = run_dir / f"adapter-{stage}.log"
    requires_gpu_hours = False
    if claim is not None:
        claim_payload = _claim_payload(claim.path, claim.trial_hash)
        requires_gpu_hours = claim_payload.get("gpu_hours_reserved") is not None
    if requires_gpu_hours:
        protocol = expected_manifest.get("protocol")
        world_size = (
            protocol.get("world_size")
            if isinstance(protocol, Mapping)
            else None
        )
        try:
            begin_claim_stage(
                claim,
                stage=stage,
                world_size=int(world_size),
                heartbeat_interval_seconds=(
                    claim_heartbeat_interval_seconds
                ),
            )
        except (RuntimeError, TypeError, ValueError) as exc:
            raise SystemicAdapterFailure(
                str(exc),
                stage=stage,
            ) from exc
    try:
        with _claim_heartbeat(
            claim,
            interval_seconds=claim_heartbeat_interval_seconds,
        ) as heartbeat_errors:
            with log_path.open("a") as log:
                completed = subprocess.run(
                    argv,
                    check=False,
                    cwd=run_dir,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
    except RuntimeError as exc:
        raise SystemicAdapterFailure(str(exc), stage=stage) from exc
    if heartbeat_errors:
        raise SystemicAdapterFailure(
            str(heartbeat_errors[0]),
            stage=stage,
            returncode=completed.returncode,
        ) from heartbeat_errors[0]
    try:
        output = _read_json(metrics_path)
    except RuntimeError as exc:
        raise SystemicAdapterFailure(
            str(exc),
            stage=stage,
            returncode=completed.returncode,
        ) from exc

    gpu_seconds_actual: float | None = None
    gpu_hours_actual: float | None = None
    if requires_gpu_hours:
        try:
            gpu_seconds_actual = _nonnegative_finite_gpu_hours(
                output.get("gpu_seconds_actual"),
                field="adapter gpu_seconds_actual",
            )
            gpu_hours_actual = _nonnegative_finite_gpu_hours(
                output.get("gpu_hours_actual"),
                field="adapter gpu_hours_actual",
            )
            if not math.isclose(
                gpu_hours_actual,
                gpu_seconds_actual / 3600.0,
                rel_tol=1e-9,
                abs_tol=1e-12,
            ):
                raise RuntimeError(
                    "adapter gpu_hours_actual does not equal "
                    "gpu_seconds_actual / 3600"
                )
        except RuntimeError as exc:
            raise SystemicAdapterFailure(
                str(exc),
                stage=stage,
                returncode=completed.returncode,
            ) from exc

    if output.get("status") == "failed":
        if output.get("schema_version") != 1:
            raise SystemicAdapterFailure(
                "Adapter failure output schema_version must equal 1",
                stage=stage,
                returncode=completed.returncode,
            )
        if output.get("trial_hash") != expected_trial_hash:
            raise SystemicAdapterFailure(
                "Adapter failure output trial_hash does not match its manifest",
                stage=stage,
                returncode=completed.returncode,
            )
        failure = output.get("failure")
        if not isinstance(failure, dict):
            raise SystemicAdapterFailure(
                "Adapter failure output must contain a failure object",
                stage=stage,
                returncode=completed.returncode,
            )
        scope = failure.get("scope")
        code = failure.get("code")
        detail = failure.get("detail")
        if (
            not isinstance(code, str)
            or not code
            or not isinstance(detail, str)
            or not detail
        ):
            raise SystemicAdapterFailure(
                "Adapter failure requires nonempty failure.code and failure.detail",
                stage=stage,
                returncode=completed.returncode,
                failure=failure,
            )
        if scope == "trial":
            raise TrialLocalFailure(
                f"{code}: {detail}",
                stage=stage,
                returncode=completed.returncode,
                failure=failure,
                gpu_seconds_actual=gpu_seconds_actual,
                gpu_hours_actual=gpu_hours_actual,
            )
        raise SystemicAdapterFailure(
            f"{code}: {detail}",
            stage=stage,
            returncode=completed.returncode,
            failure=failure,
            gpu_seconds_actual=gpu_seconds_actual,
            gpu_hours_actual=gpu_hours_actual,
        )

    if completed.returncode != 0:
        raise SystemicAdapterFailure(
            f"Adapter exited with status {completed.returncode} without a "
            "structured trial-local failure",
            stage=stage,
            returncode=completed.returncode,
        )
    _validate_checkpoint_bundle_output(output, run_dir=run_dir, stage=stage)
    return output


def _summary(
    manifest: Mapping[str, Any],
    status: str,
    *,
    step_metrics: Mapping[str, Any] | None = None,
    final_metrics: Mapping[str, Any] | None = None,
    failure: Mapping[str, Any] | None = None,
    optuna_trial_id: int | None = None,
    gpu_seconds_actual: float | None = None,
    gpu_hours_actual: float | None = None,
    gpu_time_source: str | None = None,
) -> dict[str, Any]:
    metrics = final_metrics or step_metrics or {}
    seconds = (
        metrics.get("gpu_seconds_actual")
        if gpu_seconds_actual is None
        else gpu_seconds_actual
    )
    hours = (
        metrics.get("gpu_hours_actual")
        if gpu_hours_actual is None
        else gpu_hours_actual
    )
    summary = {
        "schema_version": 1,
        "trial_hash": manifest["trial_hash"],
        "promotion_config_hash": manifest["promotion_config_hash"],
        "study_identity": {
            "study_name": manifest["study_name"],
            "spec_hash": manifest["spec_hash"],
            "code_commit": manifest["code"].get("commit", "UNKNOWN"),
            "code_dirty": manifest["code"].get("dirty", True),
            "data_hash": stable_hash(manifest["data"]),
            "capability_hash": stable_hash(manifest["capability_identity"]),
        },
        "seed": manifest["protocol"]["seed"],
        "status": status,
        "mqar_accuracy": metrics.get("mqar_accuracy"),
        "step_500_mqar_accuracy": (
            step_metrics.get("mqar_accuracy")
            if step_metrics is not None
            else None
        ),
        "wikitext_ppl": metrics.get("wikitext_ppl"),
        "wandb_run_id": metrics.get("wandb_run_id"),
        "optuna_trial_id": optuna_trial_id,
        "gpu_seconds_actual": seconds,
        "gpu_hours_actual": hours,
        "health_passed": status in {"COMPLETE", "PRUNED"},
        "failure": dict(failure) if failure else None,
    }
    if metrics.get("model_accounting") is not None:
        summary["model_accounting"] = metrics["model_accounting"]
    if metrics.get("optimizer_group_audit") is not None:
        summary["optimizer_group_audit"] = metrics["optimizer_group_audit"]
    if gpu_time_source is not None:
        summary["gpu_time_source"] = gpu_time_source
    return summary


def _cumulative_claim_gpu_time(
    claim: TrialClaim,
    *,
    gpu_seconds_actual: float,
    gpu_hours_actual: float,
) -> tuple[float, float]:
    """Add measured prior systemic attempts retained across recovery."""

    seconds = _nonnegative_finite_gpu_hours(
        gpu_seconds_actual,
        field="gpu_seconds_actual",
    )
    hours = _nonnegative_finite_gpu_hours(
        gpu_hours_actual,
        field="gpu_hours_actual",
    )
    payload = _claim_payload(claim.path, claim.trial_hash)
    for collection_name in (
        "successful_stages",
        "failed_stages",
        "systemic_attempts",
    ):
        entries = payload.get(collection_name, [])
        if not isinstance(entries, list):
            raise RuntimeError(f"Claim {collection_name} is corrupt")
        for entry in entries:
            if not isinstance(entry, Mapping):
                raise RuntimeError(f"Claim {collection_name} entry is corrupt")
            seconds += _nonnegative_finite_gpu_hours(
                entry.get("gpu_seconds_actual"),
                field=f"{collection_name} gpu_seconds_actual",
            )
            hours += _nonnegative_finite_gpu_hours(
                entry.get("gpu_hours_actual"),
                field=f"{collection_name} gpu_hours_actual",
            )
    if not math.isclose(
        hours,
        seconds / 3600.0,
        rel_tol=1e-9,
        abs_tol=1e-12,
    ):
        raise RuntimeError("Cumulative GPU seconds/hours disagree")
    return seconds, hours


def _set_accounting_user_attrs(
    trial: Any,
    metrics: Mapping[str, Any],
) -> None:
    """Expose centrally validated capacity/compute facts to Optuna analysis."""

    accounting = metrics["model_accounting"]
    counts = accounting["parameter_counts_actual"]
    flops = accounting["flops"]
    trial.set_user_attr(
        "actual_non_embedding_core_params",
        int(counts["non_embedding_core"]),
    )
    trial.set_user_attr("actual_total_params", int(counts["total"]))
    trial.set_user_attr(
        "estimated_flops_per_optimizer_step",
        float(flops["estimated"]),
    )
    trial.set_user_attr(
        "measured_flops_per_optimizer_step",
        float(flops["measured"]),
    )
    trial.set_user_attr(
        "gpu_hours_actual",
        float(metrics["gpu_hours_actual"]),
    )


def objective_factory(
    *,
    spec: Mapping[str, Any],
    output_root: Path,
    worker_command: Sequence[str],
    code_identity: Mapping[str, Any],
    data_identity: Mapping[str, Any],
    capability_identity: Mapping[str, Any],
    claim_heartbeat_interval_seconds: float = 60.0,
):
    """Return an Optuna objective with a checkpoint/evaluation boundary at 500."""

    try:
        import optuna
    except ImportError as exc:
        raise RuntimeError("Install optional nas dependencies to run the study") from exc

    prune_step = int(spec["fidelity"]["prune_step"])
    final_step = int(spec["fidelity"]["max_steps"])
    requested_trials = int(spec["study"]["requested_trials"])
    compute_budget = spec["study"]["compute_budget"]
    maximum_total_gpu_hours = _positive_finite_gpu_hours(
        compute_budget.get("maximum_total_gpu_hours"),
        field="study.compute_budget.maximum_total_gpu_hours",
    )
    maximum_full_trial_gpu_hours = _positive_finite_gpu_hours(
        compute_budget.get("maximum_full_trial_gpu_hours"),
        field="study.compute_budget.maximum_full_trial_gpu_hours",
    )
    maximum_concurrent_trials = compute_budget.get("maximum_concurrent_trials")
    if (
        not isinstance(maximum_concurrent_trials, int)
        or isinstance(maximum_concurrent_trials, bool)
        or maximum_concurrent_trials <= 0
    ):
        raise RuntimeError(
            "study.compute_budget.maximum_concurrent_trials must be a "
            "positive integer"
        )

    def objective(trial: Any) -> float:
        recovery_token = trial.user_attrs.get("phase2_recovery_token")
        if recovery_token is not None and (
            not isinstance(recovery_token, str) or not recovery_token
        ):
            raise SystemicAdapterFailure(
                "Queued recovery has an invalid one-time token",
                stage="startup_recovery",
            )
        if recovery_token is None:
            trial.set_user_attr("phase2_trial_kind", "fresh")
            fresh_ordinal = count_fresh_optuna_trials(
                [
                    existing
                    for existing in trial.study.get_trials(deepcopy=False)
                    if existing.number <= trial.number
                ]
            )
            trial.set_user_attr("phase2_fresh_trial_ordinal", fresh_ordinal)
            if fresh_ordinal > requested_trials:
                trial.set_user_attr(
                    "compute_budget_guard",
                    "fresh_trial_target_reached",
                )
                raise optuna.TrialPruned(
                    f"fresh trial ordinal {fresh_ordinal} exceeds approved "
                    f"target {requested_trials}"
                )
        else:
            trial.set_user_attr("phase2_trial_kind", "recovery")
        params = suggest_parameters(spec, trial)
        manifest = build_trial_manifest(
            spec,
            params,
            seed=int(spec["protocol"]["seed"]),
            trial_number=trial.number,
            code_identity=code_identity,
            data_identity=data_identity,
            capability_identity=capability_identity,
        )
        requested_run_dir = (output_root / (
            f"trial-{trial.number:07d}-{manifest['trial_hash'][:12]}"
        )).resolve()
        trial.set_user_attr("trial_hash", manifest["trial_hash"])
        trial.set_user_attr("model_hash", manifest["model_hash"])
        trial.set_user_attr(
            "materialized_parameters", manifest["parameters"]
        )
        trial.set_user_attr(
            "promotion_config_hash", manifest["promotion_config_hash"]
        )
        trial.set_user_attr("seed", manifest["protocol"]["seed"])
        trial.set_user_attr(
            "non_embedding_core_params",
            manifest["parameter_counts_estimated"]["non_embedding_core"],
        )
        claim = claim_trial_hash(
            output_root,
            manifest["trial_hash"],
            trial.number,
            run_dir=requested_run_dir,
            stale_after_seconds=int(
                spec["storage"].get(
                    "claim_stale_after_seconds",
                    DEFAULT_CLAIM_STALE_AFTER_SECONDS,
                )
            ),
            recovery_token=recovery_token,
            maximum_total_gpu_hours=maximum_total_gpu_hours,
            maximum_full_trial_gpu_hours=maximum_full_trial_gpu_hours,
            maximum_concurrent_trials=maximum_concurrent_trials,
            claim_kind="optuna_study",
        )
        if claim is None:
            trial.set_user_attr("duplicate_trial_hash", True)
            raise optuna.TrialPruned(
                f"duplicate outcome identity {manifest['trial_hash']}"
            )
        run_dir = claim.run_dir
        trial.set_user_attr("reclaimed_trial_hash_lease", claim.reclaimed)
        trial.set_user_attr("startup_recovery", recovery_token is not None)
        trial.set_user_attr("run_dir", str(run_dir))

        stage_gpu_started: float | None = None
        stage_gpu_accounted = False
        stage_reported_gpu_seconds: float | None = None
        stage_reported_gpu_hours: float | None = None
        try:
            write_trial_directory(run_dir, manifest)
            manifest_path = run_dir / "trial_manifest.json"
            screen_path = run_dir / "metrics" / "step_500.json"
            stage_gpu_started = time.monotonic()
            stage_gpu_accounted = False
            stage_reported_gpu_seconds = None
            stage_reported_gpu_hours = None
            screen = run_adapter_stage(
                worker_command,
                manifest_path=manifest_path,
                run_dir=run_dir,
                stage="screen",
                metrics_path=screen_path,
                claim=claim,
                claim_heartbeat_interval_seconds=claim_heartbeat_interval_seconds,
            )
            stage_reported_gpu_seconds = float(screen["gpu_seconds_actual"])
            stage_reported_gpu_hours = float(screen["gpu_hours_actual"])
            validate_step_metrics(
                manifest,
                screen,
                expected_step=prune_step,
                require_full_mqar_grid=True,
                require_wikitext_ppl=False,
                diagnostic_names=manifest["screen_required_diagnostics"],
            )
            record_successful_claim_stage(
                claim,
                stage="screen",
                gpu_seconds_actual=stage_reported_gpu_seconds,
                gpu_hours_actual=stage_reported_gpu_hours,
            )
            stage_gpu_accounted = True
            screen_gpu_seconds, screen_gpu_hours = _cumulative_claim_gpu_time(
                claim,
                gpu_seconds_actual=0.0,
                gpu_hours_actual=0.0,
            )
            _set_accounting_user_attrs(trial, screen)
            trial.set_user_attr("gpu_hours_actual", screen_gpu_hours)
            report_step_500(trial, screen)
            if screen.get("wikitext_ppl") is not None:
                trial.set_user_attr(
                    "step_500_wikitext_ppl", screen["wikitext_ppl"]
                )

            if trial.should_prune():
                summary_path = run_dir / "trial_summary.json"
                try:
                    atomic_write_json(
                        summary_path,
                        _summary(
                            manifest,
                            "PRUNED",
                            step_metrics=screen,
                            optuna_trial_id=trial.number,
                            gpu_seconds_actual=screen_gpu_seconds,
                            gpu_hours_actual=screen_gpu_hours,
                            gpu_time_source=(
                                "controller_aggregated_invocations"
                            ),
                        ),
                    )
                    finalize_trial_claim(
                        claim,
                        "PRUNED",
                        gpu_hours_actual=screen_gpu_hours,
                        terminal_artifacts=terminal_artifact_bindings(run_dir),
                    )
                except BaseException:
                    summary_path.unlink(missing_ok=True)
                    raise
                raise optuna.TrialPruned(
                    f"step-500 MQAR={screen['mqar_accuracy']:.6f} below median"
                )

            if final_step == prune_step:
                final = screen
                final_gpu_seconds = screen_gpu_seconds
                final_gpu_hours = screen_gpu_hours
            else:
                final_path = run_dir / "metrics" / "final.json"
                stage_gpu_started = time.monotonic()
                stage_gpu_accounted = False
                stage_reported_gpu_seconds = None
                stage_reported_gpu_hours = None
                final = run_adapter_stage(
                    worker_command,
                    manifest_path=manifest_path,
                    run_dir=run_dir,
                    stage="final",
                    metrics_path=final_path,
                    claim=claim,
                    claim_heartbeat_interval_seconds=claim_heartbeat_interval_seconds,
                )
                stage_reported_gpu_seconds = float(final["gpu_seconds_actual"])
                stage_reported_gpu_hours = float(final["gpu_hours_actual"])
                validate_step_metrics(
                    manifest,
                    final,
                    expected_step=final_step,
                    require_full_mqar_grid=True,
                )
                validate_cross_stage_identity(screen, final)
                record_successful_claim_stage(
                    claim,
                    stage="final",
                    gpu_seconds_actual=stage_reported_gpu_seconds,
                    gpu_hours_actual=stage_reported_gpu_hours,
                )
                stage_gpu_accounted = True
                final_gpu_seconds, final_gpu_hours = (
                    _cumulative_claim_gpu_time(
                        claim,
                        gpu_seconds_actual=0.0,
                        gpu_hours_actual=0.0,
                    )
                )

            _set_accounting_user_attrs(trial, final)
            trial.set_user_attr("gpu_hours_actual", final_gpu_hours)
            trial.set_user_attr("wikitext_ppl", float(final["wikitext_ppl"]))
            trial.set_user_attr("health_passed", True)
            if final.get("wandb_run_id") is not None:
                trial.set_user_attr("wandb_run_id", final["wandb_run_id"])
            summary_path = run_dir / "trial_summary.json"
            try:
                atomic_write_json(
                    summary_path,
                    _summary(
                        manifest,
                        "COMPLETE",
                        step_metrics=screen,
                        final_metrics=final,
                        optuna_trial_id=trial.number,
                        gpu_seconds_actual=final_gpu_seconds,
                        gpu_hours_actual=final_gpu_hours,
                        gpu_time_source=(
                            "controller_aggregated_invocations"
                        ),
                    ),
                )
                finalize_trial_claim(
                    claim,
                    "COMPLETE",
                    gpu_hours_actual=final_gpu_hours,
                    terminal_artifacts=terminal_artifact_bindings(run_dir),
                )
            except BaseException:
                summary_path.unlink(missing_ok=True)
                raise
            return float(final["mqar_accuracy"])
        except optuna.TrialPruned:
            raise
        except TrialLocalFailure as exc:
            failure = exc.failure or {
                "scope": "trial",
                "code": type(exc).__name__,
                "detail": str(exc),
            }
            failed_stage_recorded = False
            try:
                record_failed_claim_stage(
                    claim,
                    stage=exc.stage,
                    failure=failure,
                    gpu_seconds_actual=float(exc.gpu_seconds_actual),
                    gpu_hours_actual=float(exc.gpu_hours_actual),
                )
                failed_stage_recorded = True
                failure_gpu_seconds, failure_gpu_hours = (
                    _cumulative_claim_gpu_time(
                        claim,
                        gpu_seconds_actual=0.0,
                        gpu_hours_actual=0.0,
                    )
                )
                atomic_write_json(
                    run_dir / "trial_summary.json",
                    _summary(
                        manifest,
                        "FAIL",
                        failure=failure,
                        optuna_trial_id=trial.number,
                        gpu_seconds_actual=failure_gpu_seconds,
                        gpu_hours_actual=failure_gpu_hours,
                        gpu_time_source=(
                            "controller_aggregated_with_adapter_failure"
                        ),
                    ),
                )
                finalize_trial_claim(
                    claim,
                    "FAIL",
                    gpu_hours_actual=failure_gpu_hours,
                    terminal_artifacts=terminal_artifact_bindings(run_dir),
                )
            except BaseException:
                (run_dir / "trial_summary.json").unlink(missing_ok=True)
                failure_of_failure = {
                    "scope": "systemic",
                    "code": "TrialFailureFinalizationError",
                    "detail": "Could not persist/finalize a trial-local failure",
                }
                abandon_trial_claim_for_recovery(
                    claim,
                    failure=failure_of_failure,
                    # record_failed_claim_stage already reconciled and charged
                    # the adapter envelope against the in-flight heartbeat.
                    gpu_seconds_actual=(
                        0.0
                        if failed_stage_recorded
                        else float(exc.gpu_seconds_actual or 0.0)
                    ),
                    gpu_hours_actual=(
                        0.0
                        if failed_stage_recorded
                        else float(exc.gpu_hours_actual or 0.0)
                    ),
                )
                raise
            raise
        except BaseException as exc:
            failure = {
                "scope": "systemic",
                "code": type(exc).__name__,
                "detail": str(exc),
            }
            if (
                isinstance(exc, AdapterFailure)
                and exc.gpu_seconds_actual is not None
                and exc.gpu_hours_actual is not None
            ):
                gpu_seconds_actual = float(exc.gpu_seconds_actual)
                gpu_hours_actual = float(exc.gpu_hours_actual)
                gpu_time_source = "adapter_failure_envelope"
            elif (
                not stage_gpu_accounted
                and stage_reported_gpu_seconds is not None
                and stage_reported_gpu_hours is not None
            ):
                gpu_seconds_actual = stage_reported_gpu_seconds
                gpu_hours_actual = stage_reported_gpu_hours
                gpu_time_source = "adapter_reported_uncommitted_stage"
            else:
                failed_stage_gpu_seconds = (
                    0.0
                    if stage_gpu_started is None or stage_gpu_accounted
                    else max(0.0, time.monotonic() - stage_gpu_started)
                )
                gpu_seconds_actual = failed_stage_gpu_seconds
                gpu_hours_actual = gpu_seconds_actual / 3600.0
                gpu_time_source = "controller_wall_time_single_gpu"
            attempt_summary = _summary(
                manifest,
                "FAIL",
                failure=failure,
                optuna_trial_id=trial.number,
                gpu_seconds_actual=gpu_seconds_actual,
                gpu_hours_actual=gpu_hours_actual,
                gpu_time_source=gpu_time_source,
            )
            if stage_gpu_started is None:
                attempt_summary["recoverable_systemic_attempt"] = False
                if claim.reclaimed:
                    attempt_summary["recoverable_systemic_attempt"] = True
                    try:
                        atomic_write_json(
                            run_dir / "trial_setup_failure.json",
                            attempt_summary,
                        )
                    finally:
                        abandon_trial_claim_for_recovery(
                            claim,
                            failure=failure,
                            gpu_seconds_actual=0.0,
                            gpu_hours_actual=0.0,
                        )
                else:
                    try:
                        atomic_write_json(
                            run_dir / "trial_setup_failure.json",
                            attempt_summary,
                        )
                    finally:
                        release_trial_claim(claim)
                raise
            attempt_summary["recoverable_systemic_attempt"] = True
            try:
                atomic_write_json(
                    run_dir / "trial_attempt_failure.json",
                    attempt_summary,
                )
            finally:
                abandon_trial_claim_for_recovery(
                    claim,
                    failure=failure,
                    gpu_seconds_actual=gpu_seconds_actual,
                    gpu_hours_actual=gpu_hours_actual,
                )
            raise

    return objective


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", default="configs/phase2a_search.json")
    parser.add_argument("--capabilities", required=True)
    parser.add_argument("--data-manifest", required=True)
    parser.add_argument("--storage", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--heartbeat-interval", type=int, default=60)
    parser.add_argument(
        "--worker-index",
        type=int,
        help="Unique distributed-worker index; defaults to SLURM_ARRAY_TASK_ID.",
    )
    parser.add_argument(
        "worker_command",
        nargs=argparse.REMAINDER,
        help="Adapter executable and arguments after '--'.",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    worker_command = list(args.worker_command)
    if worker_command and worker_command[0] == "--":
        worker_command = worker_command[1:]
    if not worker_command:
        raise SystemExit("Pass the finalized training adapter after '--'")
    if args.trials < 1:
        raise SystemExit("--trials must be positive")
    if args.heartbeat_interval <= 0:
        raise SystemExit("--heartbeat-interval must be positive")
    worker_index = (
        args.worker_index
        if args.worker_index is not None
        else int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    )
    if worker_index < 0:
        raise SystemExit("--worker-index must be nonnegative")

    spec = load_spec(args.spec)
    claim_stale_after_seconds = int(
        spec["storage"].get(
            "claim_stale_after_seconds",
            DEFAULT_CLAIM_STALE_AFTER_SECONDS,
        )
    )
    if claim_stale_after_seconds <= args.heartbeat_interval * 3:
        raise SystemExit(
            "storage.claim_stale_after_seconds must exceed three heartbeat "
            "intervals"
        )
    slurm_task_count = os.environ.get("SLURM_ARRAY_TASK_COUNT")
    if slurm_task_count is not None and int(slurm_task_count) > int(
        spec["study"]["compute_budget"]["maximum_concurrent_trials"]
    ):
        raise SystemExit(
            "Slurm array exceeds the approved maximum_concurrent_trials"
        )
    capabilities = _read_json(Path(args.capabilities))
    data_identity = _read_json(Path(args.data_manifest))
    preflight = evaluate_preflight(
        spec,
        capabilities,
        data_manifest=data_identity,
        stage="study",
        require_clean=True,
    )
    if not preflight["ready"]:
        print(json.dumps(preflight, indent=2, sort_keys=True))
        raise SystemExit("Scientific launch preflight failed")
    code = git_identity(spec["_repo_root"])
    output_root = Path(args.output_root).expanduser().resolve()
    ledger_contract = build_ledger_contract(
        spec,
        output_root=output_root,
        storage=args.storage,
        code_identity=code,
        data_identity=data_identity,
        capability_identity=capabilities,
    )
    ensure_ledger_contract(output_root, ledger_contract, create=True)
    storage = build_storage(
        args.storage,
        approved_backend=spec["storage"]["backend"],
        heartbeat_interval=args.heartbeat_interval,
    )
    study = create_optuna_study(
        study_name=spec["study_name"],
        storage=storage,
        seed=int(spec["study"]["sampler_seed"]) + worker_index,
        minimum_completed_trials=int(
            spec["fidelity"]["minimum_completed_trials"]
        ),
    )
    bind_study_contract(
        study,
        {
            key: ledger_contract[key]
            for key in (
                "study_name",
                "spec_hash",
                "code_commit",
                "code_dirty",
                "data_hash",
                "capability_hash",
                "claims_output_root",
                "storage_identity_sha256",
                "ledger_contract_sha256",
            )
        },
    )
    requested_trials = int(spec["study"]["requested_trials"])
    existing_rows_before_recovery = list(study.get_trials(deepcopy=False))
    existing_fresh_trials = count_fresh_optuna_trials(
        existing_rows_before_recovery
    )
    queued_recoveries = queue_stale_trial_recoveries(
        study,
        spec=spec,
        output_root=output_root,
        code_identity=code,
        data_identity=data_identity,
        capability_identity=capabilities,
        stale_after_seconds=int(
            spec["storage"].get(
                "claim_stale_after_seconds",
                DEFAULT_CLAIM_STALE_AFTER_SECONDS,
            )
        ),
    )
    pending_recoveries = count_queued_trial_recoveries(output_root)
    if queued_recoveries:
        print(
            f"Queued {len(queued_recoveries)} stale trial manifest(s) "
            "for checkpoint recovery before new sampling."
        )
    fresh_trial_budget = max(
        0, requested_trials - existing_fresh_trials
    )
    if fresh_trial_budget == 0 and pending_recoveries == 0:
        print(
            f"Study already contains {existing_fresh_trials} fresh sampled "
            f"configurations; the approved target is {requested_trials}."
        )
        return
    trials_this_worker = min(
        args.trials,
        fresh_trial_budget + pending_recoveries,
    )
    objective = objective_factory(
        spec=spec,
        output_root=output_root,
        worker_command=worker_command,
        code_identity=code,
        data_identity=data_identity,
        capability_identity=capabilities,
        claim_heartbeat_interval_seconds=float(args.heartbeat_interval),
    )
    # Only a valid structured trial-local failure is safe to consume and
    # continue. Systemic adapter/config/protocol failures stop the worker.
    study.optimize(
        objective,
        n_trials=trials_this_worker,
        gc_after_trial=True,
        catch=(TrialLocalFailure,),
    )


if __name__ == "__main__":
    main()
