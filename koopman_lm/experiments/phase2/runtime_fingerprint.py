"""Capture and verify the exact runtime used by a Phase 2a worker.

The lock is deliberately JSON. Package versions, critical import/extension
health, and a minimal CUDA tensor operation are all captured so a broken ABI
cannot pass merely because distribution metadata exists.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping


_DISTRIBUTIONS = (
    "koopman_lm",
    "torch",
    "numpy",
    "mamba-ssm",
    "causal-conv1d",
    "triton",
    "transformers",
    "datasets",
    "safetensors",
    "wandb",
    "PyYAML",
    "optuna",
    "scikit-learn",
    "psycopg",
)
_SCIENTIFIC_REQUIRED_DISTRIBUTIONS = (
    "koopman_lm",
    "torch",
    "numpy",
    "mamba-ssm",
    "causal-conv1d",
    "triton",
    "transformers",
    "datasets",
    "safetensors",
    "wandb",
    "PyYAML",
    "optuna",
    "scikit-learn",
    "psycopg",
)
_CRITICAL_IMPORTS = (
    "koopman_lm",
    "numpy",
    "mamba_ssm",
    "mamba_ssm.ops.triton.ssd_combined",
    "causal_conv1d",
    "causal_conv1d_cuda",
    "triton",
    "transformers",
    "datasets",
    "safetensors",
    "wandb",
    "yaml",
    "optuna",
    "sklearn",
    "psycopg",
)


def stable_hash(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _distribution_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in _DISTRIBUTIONS:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _installed_distributions() -> dict[str, list[str]]:
    """Capture the complete environment, not only the training hot path."""

    versions: dict[str, set[str]] = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if not isinstance(name, str) or not name.strip():
            continue
        canonical = re.sub(r"[-_.]+", "-", name).lower()
        versions.setdefault(canonical, set()).add(str(distribution.version))
    return {
        name: sorted(found_versions)
        for name, found_versions in sorted(versions.items())
    }


def _command_output(command: list[str]) -> str | None:
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() or result.stderr.strip() or None


def _cuda_driver_versions() -> list[str] | None:
    output = _command_output(
        [
            "nvidia-smi",
            "--query-gpu=driver_version",
            "--format=csv,noheader",
        ]
    )
    if output is None:
        return None
    return sorted({line.strip() for line in output.splitlines() if line.strip()})


def _torch_fingerprint() -> dict[str, Any]:
    try:
        torch = importlib.import_module("torch")
    except Exception as exc:  # Import failures are recorded and rejected later.
        return {
            "available": False,
            "import_error_type": type(exc).__name__,
        }

    cuda = getattr(torch, "cuda", None)
    cuda_available = bool(cuda is not None and cuda.is_available())
    device_count = int(cuda.device_count()) if cuda_available else 0
    devices: list[dict[str, Any]] = []
    for index in range(device_count):
        properties = cuda.get_device_properties(index)
        devices.append(
            {
                "name": str(properties.name),
                "compute_capability": [
                    int(properties.major),
                    int(properties.minor),
                ],
                "total_memory_bytes": int(properties.total_memory),
            }
        )
    try:
        compiled_arches = list(cuda.get_arch_list()) if cuda is not None else []
    except Exception:
        compiled_arches = []
    try:
        cudnn_version = torch.backends.cudnn.version()
    except Exception:
        cudnn_version = None
    try:
        cxx11_abi = bool(torch._C._GLIBCXX_USE_CXX11_ABI)
    except Exception:
        cxx11_abi = None
    gpu_smoke: dict[str, Any]
    if cuda_available and device_count == 1:
        try:
            left = torch.ones((2, 2), device="cuda", dtype=torch.float32)
            right = torch.full((2, 2), 2.0, device="cuda", dtype=torch.float32)
            smoke_value = float((left @ right).sum().item())
            cuda.synchronize()
            gpu_smoke = {
                "ok": smoke_value == 16.0,
                "operation": "cuda_float32_matmul_2x2",
                "result": smoke_value,
            }
        except Exception as exc:
            gpu_smoke = {
                "ok": False,
                "operation": "cuda_float32_matmul_2x2",
                "error_type": type(exc).__name__,
            }
    else:
        gpu_smoke = {
            "ok": False,
            "operation": "cuda_float32_matmul_2x2",
            "error_type": "CudaDeviceUnavailable",
        }
    version_namespace = getattr(torch, "version", None)
    return {
        "available": True,
        "version": str(getattr(torch, "__version__", "")),
        "git_version": getattr(version_namespace, "git_version", None),
        "cuda_build": getattr(version_namespace, "cuda", None),
        "hip_build": getattr(version_namespace, "hip", None),
        "cuda_available": cuda_available,
        "device_count": device_count,
        "devices": devices,
        "compiled_cuda_arches": compiled_arches,
        "cudnn_version": cudnn_version,
        "cxx11_abi": cxx11_abi,
        "gpu_smoke": gpu_smoke,
    }


def _critical_import_health() -> dict[str, dict[str, Any]]:
    """Import the packages/extensions needed by the worker hot path."""

    health: dict[str, dict[str, Any]] = {}
    for module_name in _CRITICAL_IMPORTS:
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            health[module_name] = {
                "ok": False,
                "error_type": type(exc).__name__,
            }
        else:
            health[module_name] = {"ok": True}
    return health


def capture_runtime_fingerprint() -> dict[str, Any]:
    """Return deterministic runtime identity, excluding host/job-specific IDs."""

    executable = Path(sys.executable).expanduser().resolve()
    return {
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "executable": str(executable),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "libc": list(platform.libc_ver()),
        },
        "packages": _distribution_versions(),
        "installed_distributions": _installed_distributions(),
        "critical_imports": _critical_import_health(),
        "torch": _torch_fingerprint(),
        "cuda_toolkit": {
            "cuda_home": os.environ.get("CUDA_HOME"),
            "cuda_path": os.environ.get("CUDA_PATH"),
            "nvcc_version_output": _command_output(["nvcc", "--version"]),
            "driver_versions": _cuda_driver_versions(),
        },
        "toolchain_environment": {
            "loaded_modules": sorted(
                item
                for item in os.environ.get("LOADEDMODULES", "").split(":")
                if item
            ),
            "cc": os.environ.get("CC"),
            "cxx": os.environ.get("CXX"),
            "pythonpath": os.environ.get("PYTHONPATH"),
        },
    }


def build_runtime_lock(
    fingerprint: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    captured = dict(fingerprint or capture_runtime_fingerprint())
    return {
        "schema_version": 1,
        "fingerprint_sha256": stable_hash(captured),
        "fingerprint": captured,
    }


def runtime_health_failures(
    fingerprint: Mapping[str, Any],
) -> list[dict[str, str]]:
    failures: list[dict[str, str]] = []
    packages = fingerprint.get("packages", {})
    for name in _SCIENTIFIC_REQUIRED_DISTRIBUTIONS:
        version = packages.get(name) if isinstance(packages, Mapping) else None
        if not isinstance(version, str) or not version.strip():
            failures.append(
                {
                    "code": "MISSING_RUNTIME_PACKAGE",
                    "detail": name,
                }
            )
    critical_imports = fingerprint.get("critical_imports")
    for module_name in _CRITICAL_IMPORTS:
        status = (
            critical_imports.get(module_name)
            if isinstance(critical_imports, Mapping)
            else None
        )
        if not isinstance(status, Mapping) or status.get("ok") is not True:
            failures.append(
                {
                    "code": "CRITICAL_RUNTIME_IMPORT_FAILED",
                    "detail": f"{module_name}: {status!r}",
                }
            )
    torch = fingerprint.get("torch", {})
    if not isinstance(torch, Mapping) or torch.get("available") is not True:
        failures.append(
            {
                "code": "TORCH_RUNTIME_UNAVAILABLE",
                "detail": repr(torch),
            }
        )
    elif torch.get("cuda_available") is not True:
        failures.append(
            {
                "code": "CUDA_RUNTIME_UNAVAILABLE",
                "detail": repr(torch.get("cuda_available")),
            }
        )
    elif not isinstance(torch.get("cuda_build"), str) or not torch[
        "cuda_build"
    ].strip():
        failures.append(
            {
                "code": "TORCH_CUDA_BUILD_MISSING",
                "detail": repr(torch.get("cuda_build")),
            }
        )
    elif (
        not isinstance(torch.get("device_count"), int)
        or isinstance(torch.get("device_count"), bool)
        or int(torch["device_count"]) != 1
    ):
        failures.append(
            {
                "code": "EXACTLY_ONE_VISIBLE_CUDA_DEVICE_REQUIRED",
                "detail": (
                    f"device_count={torch.get('device_count')!r}; "
                    "Phase 2a trials are one GPU each"
                ),
            }
        )
    elif (
        not isinstance(torch.get("devices"), list)
        or len(torch["devices"]) != int(torch["device_count"])
    ):
        failures.append(
            {
                "code": "CUDA_DEVICE_FINGERPRINT_INCOMPLETE",
                "detail": repr(torch.get("devices")),
            }
        )
    gpu_smoke = torch.get("gpu_smoke") if isinstance(torch, Mapping) else None
    if not isinstance(gpu_smoke, Mapping) or gpu_smoke.get("ok") is not True:
        failures.append(
            {
                "code": "CUDA_SMOKE_TEST_FAILED",
                "detail": repr(gpu_smoke),
            }
        )
    return failures


def _differences(expected: Any, actual: Any, path: str = "fingerprint") -> list[str]:
    if isinstance(expected, Mapping) and isinstance(actual, Mapping):
        differences: list[str] = []
        for key in sorted(set(expected) | set(actual)):
            child = f"{path}.{key}"
            if key not in expected:
                differences.append(f"{child}: unexpected={actual[key]!r}")
            elif key not in actual:
                differences.append(f"{child}: missing (expected={expected[key]!r})")
            else:
                differences.extend(_differences(expected[key], actual[key], child))
        return differences
    if expected != actual:
        return [f"{path}: expected={expected!r}, actual={actual!r}"]
    return []


def verify_runtime_lock(
    lock: Mapping[str, Any],
    *,
    current_fingerprint: Mapping[str, Any] | None = None,
) -> list[dict[str, str]]:
    """Return fail-closed lock integrity, health, and active-runtime failures."""

    failures: list[dict[str, str]] = []
    if lock.get("schema_version") != 1:
        failures.append(
            {
                "code": "INVALID_RUNTIME_LOCK_SCHEMA",
                "detail": repr(lock.get("schema_version")),
            }
        )
    expected = lock.get("fingerprint")
    expected_hash = lock.get("fingerprint_sha256")
    if not isinstance(expected, Mapping):
        failures.append(
            {
                "code": "INVALID_RUNTIME_LOCK",
                "detail": "fingerprint must be a JSON object",
            }
        )
        return failures
    if not isinstance(expected_hash, str) or expected_hash != stable_hash(expected):
        failures.append(
            {
                "code": "RUNTIME_LOCK_SELF_HASH_MISMATCH",
                "detail": repr(expected_hash),
            }
        )
    failures.extend(runtime_health_failures(expected))

    current = dict(current_fingerprint or capture_runtime_fingerprint())
    failures.extend(runtime_health_failures(current))
    for difference in _differences(expected, current):
        failures.append(
            {
                "code": "ACTIVE_RUNTIME_MISMATCH",
                "detail": difference,
            }
        )
    return failures


def _write_json(path: str | Path, value: Mapping[str, Any]) -> None:
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    temporary.write_text(payload)
    os.replace(temporary, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--write-lock")
    action.add_argument("--verify-lock")
    args = parser.parse_args()

    if args.write_lock:
        lock = build_runtime_lock()
        failures = runtime_health_failures(lock["fingerprint"])
        if failures:
            print(json.dumps({"ready": False, "failures": failures}, indent=2))
            raise SystemExit(2)
        _write_json(args.write_lock, lock)
        print(f"Wrote runtime lock: {Path(args.write_lock).expanduser().resolve()}")
        return
    if args.verify_lock:
        try:
            lock = json.loads(Path(args.verify_lock).expanduser().read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit(f"Could not read runtime lock: {exc}") from exc
        if not isinstance(lock, dict):
            raise SystemExit("Runtime lock must be a JSON object")
        failures = verify_runtime_lock(lock)
        report = {"ready": not failures, "failures": failures}
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
        if failures:
            raise SystemExit(2)
        return
    print(
        json.dumps(
            build_runtime_lock(),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
