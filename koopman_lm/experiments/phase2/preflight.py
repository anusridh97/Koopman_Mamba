"""Fail-closed scientific-launch preflight for Phase 2a."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
from pathlib import Path
from typing import Any, Mapping

try:
    from .manifest import atomic_write_json
    from .runtime_fingerprint import verify_runtime_lock
    from .spec import Phase2SpecError, load_spec, stable_hash
except ImportError:  # Supports direct script execution.
    from manifest import atomic_write_json  # type: ignore
    from runtime_fingerprint import verify_runtime_lock  # type: ignore
    from spec import Phase2SpecError, load_spec, stable_hash  # type: ignore


_PLACEHOLDER = "REQUIRED_BEFORE_SCIENTIFIC_RUN"
_EVIDENCE_FILES = (
    "correctness_report",
    "resume_parity_report",
    "data_validation_report",
    "storage_concurrency_report",
    "pilot_report",
)
_PILOT_OPTIONAL_EVIDENCE = frozenset({"pilot_report"})
_CALIBRATION_OPTIONAL_EVIDENCE = frozenset(
    {"pilot_report", "storage_concurrency_report"}
)
_CALIBRATION_OPTIONAL_APPROVALS = frozenset(
    {
        "survivor_step_budget",
        "distributed_storage_backend",
        "compute_budget",
    }
)
_CALIBRATION_ALLOWED_SPEC_PLACEHOLDERS = frozenset(
    {
        "study.compute_budget.maximum_total_gpu_hours",
        "study.compute_budget.maximum_full_trial_gpu_hours",
        "study.compute_budget.required_gpu_type",
        "storage.backend",
    }
)


def unresolved_placeholder_paths(value: Any, prefix: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).startswith("_"):
                continue
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            found.extend(unresolved_placeholder_paths(child, child_prefix))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(unresolved_placeholder_paths(child, f"{prefix}[{index}]"))
    elif value == _PLACEHOLDER:
        found.append(prefix)
    return found


def capability_template(spec: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "spec_hash": _PLACEHOLDER,
        "integration_commit": _PLACEHOLDER,
        "approved_data_manifest_sha256": _PLACEHOLDER,
        "capabilities": {
            name: False for name in spec["required_capabilities"]
        },
        "approvals": {
            name: False for name in spec.get("required_approvals", [])
        },
        "approval_metadata": {
            "approved_by": _PLACEHOLDER,
            "approved_at_utc": _PLACEHOLDER,
        },
        "evidence": {
            **{
                name: {"path": _PLACEHOLDER, "sha256": _PLACEHOLDER}
                for name in _EVIDENCE_FILES
            },
            "environment_lock": {
                "path": _PLACEHOLDER,
                "sha256": _PLACEHOLDER,
            },
        },
    }


def _stable_file_identity(path: Path) -> dict[str, Any]:
    """Hash one immutable regular file without accepting a rename/write race."""

    if path.is_symlink():
        raise OSError(f"Frozen artifact must not be a symlink: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        before = os.fstat(handle.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise OSError(f"Frozen artifact is not a regular file: {path}")
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
        after = os.fstat(handle.fileno())
    current = path.stat()
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    current_identity = (
        current.st_dev,
        current.st_ino,
        current.st_size,
        current.st_mtime_ns,
    )
    if before_identity != after_identity or before_identity != current_identity:
        raise OSError(f"Frozen artifact changed while hashing: {path}")
    return {
        "path": str(path.resolve()),
        "device": int(before.st_dev),
        "inode": int(before.st_ino),
        "byte_size": int(before.st_size),
        "mtime_ns": int(before.st_mtime_ns),
        "mode": int(stat.S_IMODE(before.st_mode)),
        "sha256": digest.hexdigest(),
    }


def _file_sha256(path: Path) -> str:
    return str(_stable_file_identity(path)["sha256"])


def _read_json_object(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and bool(re.fullmatch(r"[0-9a-f]{64}", value))


def _nested(value: Mapping[str, Any], *path: str) -> Any:
    current: Any = value
    for name in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(name)
    return current


def _normalized_gpu_name(value: Any) -> str:
    return re.sub(
        r"[^a-z0-9]+",
        "",
        str(value).lower().replace("nvidia", ""),
    )


def _data_contract_failures(
    spec: Mapping[str, Any],
    data_manifest: Mapping[str, Any],
) -> list[dict[str, str]]:
    """Cross-check frozen data identity against the training/search protocol."""

    failures: list[dict[str, str]] = []

    def require_equal(path: tuple[str, ...], expected: Any) -> None:
        actual = _nested(data_manifest, *path)
        if actual != expected:
            failures.append(
                {
                    "code": "DATA_PROTOCOL_MISMATCH",
                    "detail": (
                        f"{'.'.join(path)}={actual!r}, expected={expected!r}"
                    ),
                }
            )

    protocol = spec["protocol"]
    require_equal(("tokenizer", "name"), protocol["tokenizer"])
    require_equal(
        ("tokenizer", "revision"), protocol["tokenizer_revision"]
    )
    require_equal(("tokenizer", "vocab_size"), 32000)
    require_equal(("training", "dataset"), protocol["train_dataset"])
    require_equal(
        ("training", "revision"), protocol["train_dataset_revision"]
    )
    require_equal(
        ("training", "packing", "sequence_length"),
        protocol["sequence_length"],
    )
    require_equal(("training", "split"), "train")
    require_equal(("training", "packing", "eos_between_documents"), True)
    require_equal(("training", "packing", "drop_remainder"), True)
    require_equal(("training", "stream"), protocol["data_stream"])
    require_equal(
        ("wikitext_validation", "dataset"),
        protocol["validation_dataset"],
    )
    require_equal(
        ("wikitext_validation", "revision"),
        protocol["validation_dataset_revision"],
    )
    require_equal(("wikitext_validation", "split"), "validation")
    require_equal(
        ("birdie_training_objectives", "mixer_semantics"),
        "static Optuna mixture fixed for the whole trial",
    )
    require_equal(
        ("birdie_training_objectives", "loss_normalization"),
        protocol["loss_normalization"],
    )
    require_equal(
        ("birdie_training_objectives", "equal_token_accounting"),
        True,
    )
    require_equal(
        ("mqar_screening", "sequence_lengths"),
        [256, 512, 1024, 2048],
    )
    require_equal(
        ("mqar_screening", "key_value_pairs"),
        [4, 8, 16, 32, 64],
    )

    for path in (
        ("tokenizer", "fingerprint_sha256"),
        ("training", "sha256"),
        ("training", "stream", "shard_order_manifest_sha256"),
        ("wikitext_validation", "sha256"),
        ("mqar_screening", "generator_revision"),
        ("mqar_screening", "sha256"),
        ("mqar_screening", "oracle_fixture_sha256"),
        ("mqar_screening", "sample_ids_sha256"),
        ("mqar_screening", "vocabulary", "identity_sha256"),
        ("mqar_screening", "token_map", "sha256"),
        (
            "birdie_training_objectives",
            "mixer_revision",
        ),
        (
            "birdie_training_objectives",
            "selective_copy",
            "generator_revision",
        ),
        (
            "birdie_training_objectives",
            "selective_copy",
            "fixture_sha256",
        ),
        (
            "birdie_training_objectives",
            "infilling",
            "generator_revision",
        ),
        (
            "birdie_training_objectives",
            "infilling",
            "fixture_sha256",
        ),
        (
            "birdie_training_objectives",
            "sample_identity_manifest_sha256",
        ),
    ):
        value = _nested(data_manifest, *path)
        if path[-1].endswith("sha256"):
            valid = _is_sha256(value)
        else:
            valid = isinstance(value, str) and bool(value.strip())
        if not valid:
            failures.append(
                {
                    "code": "INVALID_DATA_FINGERPRINT",
                    "detail": f"{'.'.join(path)}={value!r}",
                }
            )

    for path in (
        ("training", "token_count"),
        ("training", "byte_size"),
        ("wikitext_validation", "maximum_tokens"),
        ("wikitext_validation", "byte_size"),
        ("mqar_screening", "samples_per_cell"),
        ("mqar_screening", "byte_size"),
        ("mqar_screening", "vocabulary", "size"),
        ("training", "stream", "shuffle_seed"),
    ):
        value = _nested(data_manifest, *path)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            failures.append(
                {
                    "code": "INVALID_DATA_COUNT",
                    "detail": f"{'.'.join(path)}={value!r}",
                }
            )
    seed = _nested(data_manifest, "mqar_screening", "seed")
    if not isinstance(seed, int) or isinstance(seed, bool):
        failures.append(
            {
                "code": "INVALID_DATA_SEED",
                "detail": f"mqar_screening.seed={seed!r}",
            }
        )

    artifact_identities: dict[str, dict[str, Any]] = {}
    for section, file_name in (
        ("training", "token_file"),
        ("wikitext_validation", "token_file"),
        ("mqar_screening", "artifact_file"),
    ):
        file_value = _nested(data_manifest, section, file_name)
        byte_size = _nested(data_manifest, section, "byte_size")
        expected_sha256 = _nested(data_manifest, section, "sha256")
        candidate = (
            Path(file_value).expanduser()
            if isinstance(file_value, str)
            else None
        )
        if (
            candidate is None
            or not candidate.is_absolute()
            or not candidate.is_file()
            or candidate.is_symlink()
        ):
            failures.append(
                {
                    "code": "MISSING_FROZEN_DATA_FILE",
                    "detail": f"{section}.{file_name}={file_value!r}",
                }
            )
            continue
        try:
            identity = _stable_file_identity(candidate)
        except OSError as exc:
            failures.append(
                {
                    "code": "UNSTABLE_FROZEN_DATA_FILE",
                    "detail": f"{section}: {exc}",
                }
            )
            continue
        actual_size = identity["byte_size"]
        actual_sha256 = identity["sha256"]
        artifact_identities[section] = identity
        if identity["mode"] & 0o222:
            failures.append(
                {
                    "code": "WRITABLE_FROZEN_DATA_FILE",
                    "detail": (
                        f"{section}: mode={identity['mode']:04o}, "
                        f"path={candidate}"
                    ),
                }
            )
        if (
            isinstance(byte_size, int)
            and not isinstance(byte_size, bool)
            and byte_size > 0
            and actual_size != byte_size
        ):
            failures.append(
                {
                    "code": "DATA_ARTIFACT_SIZE_MISMATCH",
                    "detail": (
                        f"{section}: expected={byte_size}, actual={actual_size}, "
                        f"path={candidate}"
                    ),
                }
            )
        if _is_sha256(expected_sha256):
            if actual_sha256 != expected_sha256:
                failures.append(
                    {
                        "code": "DATA_ARTIFACT_HASH_MISMATCH",
                        "detail": (
                            f"{section}: expected={expected_sha256}, "
                            f"actual={actual_sha256}, path={candidate}"
                        ),
                    }
                )

    auxiliary_hash_sources = {
        "tokenizer_fingerprint": ("tokenizer", "fingerprint_sha256"),
        "training_shard_order_manifest": (
            "training",
            "stream",
            "shard_order_manifest_sha256",
        ),
        "mqar_oracle_fixture": (
            "mqar_screening",
            "oracle_fixture_sha256",
        ),
        "mqar_sample_ids": ("mqar_screening", "sample_ids_sha256"),
        "mqar_vocabulary_identity": (
            "mqar_screening",
            "vocabulary",
            "identity_sha256",
        ),
        "mqar_token_map": (
            "mqar_screening",
            "token_map",
            "sha256",
        ),
        "birdie_selective_copy_fixture": (
            "birdie_training_objectives",
            "selective_copy",
            "fixture_sha256",
        ),
        "birdie_infilling_fixture": (
            "birdie_training_objectives",
            "infilling",
            "fixture_sha256",
        ),
        "birdie_sample_identity_manifest": (
            "birdie_training_objectives",
            "sample_identity_manifest_sha256",
        ),
    }
    auxiliary = data_manifest.get("auxiliary_artifacts")
    for name, source_path in auxiliary_hash_sources.items():
        entry = auxiliary.get(name) if isinstance(auxiliary, Mapping) else None
        file_value = entry.get("path") if isinstance(entry, Mapping) else None
        byte_size = (
            entry.get("byte_size") if isinstance(entry, Mapping) else None
        )
        declared_sha256 = (
            entry.get("sha256") if isinstance(entry, Mapping) else None
        )
        expected_sha256 = _nested(data_manifest, *source_path)
        candidate = (
            Path(file_value).expanduser()
            if isinstance(file_value, str)
            else None
        )
        if (
            candidate is None
            or not candidate.is_absolute()
            or not candidate.is_file()
            or candidate.is_symlink()
        ):
            failures.append(
                {
                    "code": "MISSING_AUXILIARY_DATA_FILE",
                    "detail": f"auxiliary_artifacts.{name}.path={file_value!r}",
                }
            )
            continue
        try:
            identity = _stable_file_identity(candidate)
        except OSError as exc:
            failures.append(
                {
                    "code": "UNSTABLE_AUXILIARY_DATA_FILE",
                    "detail": f"{name}: {exc}",
                }
            )
            continue
        actual_size = identity["byte_size"]
        actual_sha256 = identity["sha256"]
        artifact_identities[f"auxiliary_artifacts.{name}"] = identity
        if identity["mode"] & 0o222:
            failures.append(
                {
                    "code": "WRITABLE_AUXILIARY_DATA_FILE",
                    "detail": (
                        f"{name}: mode={identity['mode']:04o}, "
                        f"path={candidate}"
                    ),
                }
            )
        try:
            auxiliary_payload = json.loads(candidate.read_text())
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            failures.append(
                {
                    "code": "INVALID_AUXILIARY_DATA_FORMAT",
                    "detail": f"{name}: expected UTF-8 JSON: {exc}",
                }
            )
        else:
            if not (
                (isinstance(auxiliary_payload, dict) and auxiliary_payload)
                or (isinstance(auxiliary_payload, list) and auxiliary_payload)
            ):
                failures.append(
                    {
                        "code": "INVALID_AUXILIARY_DATA_FORMAT",
                        "detail": f"{name}: JSON must be a nonempty object or list",
                    }
                )
        if (
            not isinstance(byte_size, int)
            or isinstance(byte_size, bool)
            or byte_size <= 0
        ):
            failures.append(
                {
                    "code": "INVALID_AUXILIARY_DATA_SIZE",
                    "detail": (
                        f"auxiliary_artifacts.{name}.byte_size="
                        f"{byte_size!r}"
                    ),
                }
            )
        if not _is_sha256(declared_sha256):
            failures.append(
                {
                    "code": "INVALID_AUXILIARY_DATA_HASH",
                    "detail": (
                        f"auxiliary_artifacts.{name}.sha256="
                        f"{declared_sha256!r}"
                    ),
                }
            )
        elif declared_sha256 != expected_sha256:
            failures.append(
                {
                    "code": "AUXILIARY_HASH_REFERENCE_MISMATCH",
                    "detail": (
                        f"auxiliary_artifacts.{name}.sha256="
                        f"{declared_sha256}, "
                        f"{'.'.join(source_path)}={expected_sha256}"
                    ),
                }
            )
        if (
            isinstance(byte_size, int)
            and not isinstance(byte_size, bool)
            and byte_size > 0
            and actual_size != byte_size
        ):
            failures.append(
                {
                    "code": "AUXILIARY_DATA_SIZE_MISMATCH",
                    "detail": (
                        f"{name}: expected={byte_size}, "
                        f"actual={actual_size}, path={candidate}"
                    ),
                }
            )
        if _is_sha256(declared_sha256) and actual_sha256 != declared_sha256:
            failures.append(
                {
                    "code": "AUXILIARY_DATA_HASH_MISMATCH",
                    "detail": (
                        f"{name}: expected={declared_sha256}, "
                        f"actual={actual_sha256}, path={candidate}"
                    ),
                }
            )

    artifact_sections = sorted(artifact_identities)
    for index, left_name in enumerate(artifact_sections):
        for right_name in artifact_sections[index + 1 :]:
            left = artifact_identities[left_name]
            right = artifact_identities[right_name]
            aliased_by = []
            if left["path"] == right["path"]:
                aliased_by.append("resolved_path")
            if (left["device"], left["inode"]) == (
                right["device"],
                right["inode"],
            ):
                aliased_by.append("filesystem_inode")
            if left["sha256"] == right["sha256"]:
                aliased_by.append("content_sha256")
            if aliased_by:
                failures.append(
                    {
                        "code": "DATA_SPLIT_ALIAS",
                        "detail": (
                            f"{left_name} and {right_name} overlap by "
                            f"{','.join(aliased_by)}"
                        ),
                    }
                )

    for path in (
        ("training", "document_range"),
        ("wikitext_validation", "packing"),
        ("mqar_screening", "generator"),
        ("mqar_screening", "token_map", "revision"),
        (
            "birdie_training_objectives",
            "selective_copy",
            "generator",
        ),
        (
            "birdie_training_objectives",
            "selective_copy",
            "seed_policy",
        ),
        ("birdie_training_objectives", "infilling", "generator"),
        ("birdie_training_objectives", "infilling", "seed_policy"),
    ):
        value = _nested(data_manifest, *path)
        if value is None or (isinstance(value, str) and not value.strip()):
            failures.append(
                {
                    "code": "MISSING_DATA_PROTOCOL_FIELD",
                    "detail": ".".join(path),
                }
            )
    return failures


def _worktree_dirty(repo_root: str | Path) -> bool:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return True
    return bool(result.stdout.strip())


def _git_head(repo_root: str | Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def evaluate_preflight(
    spec: Mapping[str, Any],
    capability_manifest: Mapping[str, Any],
    *,
    data_manifest: Mapping[str, Any] | None = None,
    stage: str = "study",
    require_clean: bool = True,
    verify_commit: bool = True,
    runtime_fingerprint: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate the exact scientific inputs approved for a pilot or study.

    ``stage="calibration"`` is a non-scientific, screen-only budget measurement
    that may omit only the budget/storage decisions it exists to inform.
    ``stage="pilot"`` deliberately omits only the pilot report, because that
    report cannot exist until the fixed pilot has run. ``stage="study"``
    requires the final immutable contract and every evidence report.
    """

    if stage not in {"calibration", "pilot", "study"}:
        raise ValueError(f"Unknown preflight stage: {stage!r}")
    failures: list[dict[str, str]] = []

    if stage != "calibration" and spec.get("status") != "scientific_ready":
        failures.append(
            {
                "code": "SPEC_STATUS_NOT_SCIENTIFIC_READY",
                "detail": str(spec.get("status")),
            }
        )

    for path in unresolved_placeholder_paths(spec):
        if (
            stage == "calibration"
            and path in _CALIBRATION_ALLOWED_SPEC_PLACEHOLDERS
        ):
            continue
        failures.append(
            {
                "code": "UNRESOLVED_SPEC_PLACEHOLDER",
                "detail": path,
            }
        )
    if stage == "calibration":
        ignored_placeholder_prefixes = tuple(
            f"evidence.{name}" for name in _CALIBRATION_OPTIONAL_EVIDENCE
        )
    elif stage == "pilot":
        ignored_placeholder_prefixes = ("evidence.pilot_report",)
    else:
        ignored_placeholder_prefixes = ()
    for path in unresolved_placeholder_paths(capability_manifest):
        if any(
            path == prefix or path.startswith(prefix + ".")
            for prefix in ignored_placeholder_prefixes
        ):
            continue
        failures.append(
            {
                "code": "UNRESOLVED_CAPABILITY_PLACEHOLDER",
                "detail": path,
            }
        )

    if capability_manifest.get("schema_version") != 1:
        failures.append(
            {
                "code": "INVALID_CAPABILITY_SCHEMA_VERSION",
                "detail": repr(capability_manifest.get("schema_version")),
            }
        )

    capabilities = capability_manifest.get("capabilities", {})
    for name in spec["required_capabilities"]:
        if capabilities.get(name) is not True:
            failures.append({"code": "MISSING_CAPABILITY", "detail": name})

    approvals = capability_manifest.get("approvals", {})
    for name in spec.get("required_approvals", []):
        if (
            stage == "calibration"
            and name in _CALIBRATION_OPTIONAL_APPROVALS
        ):
            continue
        if approvals.get(name) is not True:
            failures.append({"code": "MISSING_APPROVAL", "detail": name})

    public_spec = {
        key: value for key, value in spec.items() if not str(key).startswith("_")
    }
    expected_spec_hash = stable_hash(public_spec)
    if capability_manifest.get("spec_hash") != expected_spec_hash:
        failures.append(
            {
                "code": "CAPABILITY_SPEC_HASH_MISMATCH",
                "detail": (
                    f"manifest={capability_manifest.get('spec_hash')}, "
                    f"expected={expected_spec_hash}"
                ),
            }
        )

    approved_data_hash = capability_manifest.get(
        "approved_data_manifest_sha256"
    )
    if not _is_sha256(approved_data_hash):
        failures.append(
            {
                "code": "INVALID_APPROVED_DATA_MANIFEST_HASH",
                "detail": repr(approved_data_hash),
            }
        )
    if data_manifest is None:
        failures.append(
            {
                "code": "MISSING_DATA_MANIFEST",
                "detail": "Pass the exact immutable data manifest to preflight",
            }
        )
    else:
        if data_manifest.get("schema_version") != 1:
            failures.append(
                {
                    "code": "INVALID_DATA_MANIFEST_SCHEMA_VERSION",
                    "detail": repr(data_manifest.get("schema_version")),
                }
            )
        data_placeholders = unresolved_placeholder_paths(data_manifest)
        for path in data_placeholders:
            failures.append(
                {
                    "code": "UNRESOLVED_DATA_PLACEHOLDER",
                    "detail": path,
                }
            )
        actual_data_hash = stable_hash(data_manifest)
        if approved_data_hash != actual_data_hash:
            failures.append(
                {
                    "code": "DATA_MANIFEST_HASH_MISMATCH",
                    "detail": (
                        f"approved={approved_data_hash}, "
                        f"actual={actual_data_hash}"
                    ),
                }
            )
        failures.extend(_data_contract_failures(spec, data_manifest))

    approval_metadata = capability_manifest.get("approval_metadata", {})
    for name in ("approved_by", "approved_at_utc"):
        value = approval_metadata.get(name)
        if not isinstance(value, str) or not value.strip():
            failures.append(
                {"code": "MISSING_APPROVAL_METADATA", "detail": name}
            )

    evidence = capability_manifest.get("evidence", {})
    for name in _EVIDENCE_FILES:
        if stage == "calibration" and name in _CALIBRATION_OPTIONAL_EVIDENCE:
            continue
        if stage == "pilot" and name in _PILOT_OPTIONAL_EVIDENCE:
            continue
        entry = evidence.get(name)
        value = entry.get("path") if isinstance(entry, Mapping) else None
        expected_digest = (
            entry.get("sha256") if isinstance(entry, Mapping) else None
        )
        path = Path(value).expanduser() if isinstance(value, str) else None
        if path is None or not path.is_absolute() or not path.is_file():
            failures.append(
                {
                    "code": "MISSING_EVIDENCE_FILE",
                    "detail": f"{name}.path={value!r}",
                }
            )
            continue
        if not _is_sha256(expected_digest):
            failures.append(
                {
                    "code": "INVALID_EVIDENCE_HASH",
                    "detail": f"{name}.sha256={expected_digest!r}",
                }
            )
            continue
        actual_digest = _file_sha256(path)
        if actual_digest != expected_digest:
            failures.append(
                {
                    "code": "EVIDENCE_HASH_MISMATCH",
                    "detail": (
                        f"{name}: expected={expected_digest}, "
                        f"actual={actual_digest}"
                    ),
                }
            )
            continue
        report = _read_json_object(path)
        if report is None:
            failures.append(
                {
                    "code": "INVALID_EVIDENCE_REPORT",
                    "detail": f"{name}: not a JSON object",
                }
            )
            continue
        semantic_expectations = {
            "schema_version": 1,
            "status": "pass",
            "evidence_type": name,
            "integration_commit": capability_manifest.get(
                "integration_commit"
            ),
            "spec_hash": expected_spec_hash,
        }
        for key, expected in semantic_expectations.items():
            if report.get(key) != expected:
                failures.append(
                    {
                        "code": "EVIDENCE_CONTEXT_MISMATCH",
                        "detail": (
                            f"{name}.{key}={report.get(key)!r}, "
                            f"expected={expected!r}"
                        ),
                    }
                )
        checks = report.get("checks")
        if (
            not isinstance(checks, list)
            or not checks
            or any(
                not isinstance(check, Mapping)
                or not isinstance(check.get("name"), str)
                or not check["name"].strip()
                or check.get("status") != "pass"
                for check in checks
            )
        ):
            failures.append(
                {
                    "code": "INVALID_EVIDENCE_CHECKS",
                    "detail": f"{name}.checks must be a nonempty all-pass list",
                }
            )
        else:
            check_names = [str(check["name"]) for check in checks]
            if len(check_names) != len(set(check_names)):
                failures.append(
                    {
                        "code": "DUPLICATE_EVIDENCE_CHECK",
                        "detail": name,
                    }
                )
            required_checks = set(
                spec.get("required_evidence_checks", {}).get(name, [])
            )
            missing_checks = sorted(required_checks - set(check_names))
            if missing_checks:
                failures.append(
                    {
                        "code": "MISSING_EVIDENCE_CHECK",
                        "detail": f"{name}: {missing_checks}",
                    }
                )
        if name in {"data_validation_report", "pilot_report"}:
            if report.get("data_manifest_sha256") != approved_data_hash:
                failures.append(
                    {
                        "code": "EVIDENCE_DATA_MISMATCH",
                        "detail": (
                            f"{name}.data_manifest_sha256="
                            f"{report.get('data_manifest_sha256')!r}, "
                            f"expected={approved_data_hash!r}"
                        ),
                    }
                )
    environment_entry = evidence.get("environment_lock")
    environment_value = (
        environment_entry.get("path")
        if isinstance(environment_entry, Mapping)
        else None
    )
    environment_hash = (
        environment_entry.get("sha256")
        if isinstance(environment_entry, Mapping)
        else None
    )
    environment_path = (
        Path(environment_value).expanduser()
        if isinstance(environment_value, str)
        else None
    )
    if (
        environment_path is None
        or not environment_path.is_absolute()
        or not environment_path.is_file()
    ):
        failures.append(
            {
                "code": "MISSING_ENVIRONMENT_LOCK",
                "detail": repr(environment_value),
            }
        )
    elif not _is_sha256(environment_hash):
        failures.append(
            {
                "code": "INVALID_ENVIRONMENT_FINGERPRINT",
                "detail": repr(environment_hash),
            }
        )
    else:
        actual_environment_hash = _file_sha256(environment_path)
        if actual_environment_hash != environment_hash:
            failures.append(
                {
                    "code": "ENVIRONMENT_FINGERPRINT_MISMATCH",
                    "detail": (
                        f"expected={environment_hash}, "
                        f"actual={actual_environment_hash}"
                    ),
                }
            )
        else:
            environment_lock = _read_json_object(environment_path)
            if environment_lock is None:
                failures.append(
                    {
                        "code": "INVALID_ENVIRONMENT_LOCK",
                        "detail": f"{environment_path}: not a JSON object",
                    }
                )
            else:
                failures.extend(
                    verify_runtime_lock(
                        environment_lock,
                        current_fingerprint=runtime_fingerprint,
                    )
                )
                locked_fingerprint = environment_lock.get("fingerprint", {})
                locked_torch = (
                    locked_fingerprint.get("torch", {})
                    if isinstance(locked_fingerprint, Mapping)
                    else {}
                )
                devices = (
                    locked_torch.get("devices", [])
                    if isinstance(locked_torch, Mapping)
                    else []
                )
                if (
                    int(spec["protocol"].get("world_size", 0)) != 1
                    or not isinstance(devices, list)
                    or len(devices) != 1
                ):
                    failures.append(
                        {
                            "code": "ONE_GPU_TRIAL_CONTRACT_MISMATCH",
                            "detail": (
                                "world_size="
                                f"{spec['protocol'].get('world_size')!r}, "
                                "locked_devices="
                                f"{len(devices) if isinstance(devices, list) else 'invalid'}"
                            ),
                        }
                    )
                elif stage != "calibration":
                    required_gpu = spec["study"]["compute_budget"].get(
                        "required_gpu_type"
                    )
                    actual_gpu = (
                        devices[0].get("name")
                        if isinstance(devices[0], Mapping)
                        else None
                    )
                    required_normalized = _normalized_gpu_name(required_gpu)
                    actual_normalized = _normalized_gpu_name(actual_gpu)
                    if (
                        not required_normalized
                        or required_normalized
                        == _normalized_gpu_name(_PLACEHOLDER)
                        or not actual_normalized
                        or (
                            required_normalized not in actual_normalized
                            and actual_normalized not in required_normalized
                        )
                    ):
                        failures.append(
                            {
                                "code": "GPU_TYPE_MISMATCH",
                                "detail": (
                                    f"required={required_gpu!r}, "
                                    f"runtime={actual_gpu!r}"
                                ),
                            }
                        )

    storage = spec.get("storage", {})
    if storage.get("shared_sqlite_allowed") is not False:
        failures.append(
            {
                "code": "UNSAFE_STORAGE_POLICY",
                "detail": "shared_sqlite_allowed must be false",
            }
        )
    backend = storage.get("backend")
    if (
        stage != "calibration"
        and backend not in storage.get("acceptable_backends", [])
    ):
        failures.append(
            {
                "code": "UNAPPROVED_STORAGE_BACKEND",
                "detail": str(backend),
            }
        )

    commit = capability_manifest.get("integration_commit")
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        failures.append(
            {
                "code": "INVALID_INTEGRATION_COMMIT",
                "detail": repr(commit),
            }
        )
    elif verify_commit:
        current_head = _git_head(spec["_repo_root"])
        if current_head != commit:
            failures.append(
                {
                    "code": "INTEGRATION_COMMIT_MISMATCH",
                    "detail": f"manifest={commit}, checkout={current_head}",
                }
            )
    if require_clean and _worktree_dirty(spec["_repo_root"]):
        failures.append(
            {
                "code": "DIRTY_WORKTREE",
                "detail": str(spec["_repo_root"]),
            }
        )

    return {
        "study_name": spec["study_name"],
        "stage": stage,
        "spec_hash": expected_spec_hash,
        "data_manifest_sha256": (
            stable_hash(data_manifest) if data_manifest is not None else None
        ),
        "ready": not failures,
        "failure_count": len(failures),
        "failures": failures,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", default="configs/phase2a_search.json")
    parser.add_argument("--capabilities")
    parser.add_argument("--data-manifest")
    parser.add_argument(
        "--stage",
        choices=("calibration", "pilot", "study"),
        default="study",
    )
    parser.add_argument("--write-template")
    parser.add_argument("--print-data-hash", action="store_true")
    parser.add_argument("--output")
    parser.add_argument("--allow-dirty", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    spec = load_spec(args.spec)
    if args.write_template:
        atomic_write_json(args.write_template, capability_template(spec))
        print(f"Wrote capability template: {args.write_template}")
        return
    if args.print_data_hash:
        if not args.data_manifest:
            raise SystemExit("--print-data-hash requires --data-manifest")
        try:
            data_manifest = json.loads(Path(args.data_manifest).read_text())
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            raise SystemExit(f"Could not read data manifest: {exc}") from exc
        if not isinstance(data_manifest, dict):
            raise SystemExit("Data manifest must be a JSON object")
        placeholders = unresolved_placeholder_paths(data_manifest)
        if placeholders:
            raise SystemExit(
                "Data manifest contains unresolved placeholders: "
                + ", ".join(placeholders)
            )
        contract_failures = _data_contract_failures(spec, data_manifest)
        if contract_failures:
            raise SystemExit(
                "Data manifest does not match the study protocol: "
                + json.dumps(contract_failures, sort_keys=True)
            )
        print(stable_hash(data_manifest))
        return
    if not args.capabilities:
        raise SystemExit("--capabilities is required unless --write-template is used")
    try:
        capabilities = json.loads(Path(args.capabilities).read_text())
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Could not read capability manifest: {exc}") from exc
    if not args.data_manifest:
        raise SystemExit("--data-manifest is required for scientific preflight")
    try:
        data_manifest = json.loads(Path(args.data_manifest).read_text())
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Could not read data manifest: {exc}") from exc
    if not isinstance(capabilities, dict) or not isinstance(data_manifest, dict):
        raise SystemExit("Capability and data manifests must be JSON objects")
    report = evaluate_preflight(
        spec,
        capabilities,
        data_manifest=data_manifest,
        stage=args.stage,
        require_clean=not args.allow_dirty,
    )
    if args.output:
        atomic_write_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    if not report["ready"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
