#!/usr/bin/env python3
"""Compile and numerically validate the fused SKA prefix-scan extension."""
from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
from typing import Iterable

import torch
from torch.utils.cpp_extension import CUDA_HOME

from koopman_lm.modules.kernels.cuda_prefix_scan import (
    fused_ska_prefix_scan,
    launch_info,
    load_prefix_scan_ext,
)
from koopman_lm.modules.kernels.prefix_scan import dense_exact_oracle


def _version_tuple(version: str | None) -> tuple[int, int]:
    if not version:
        return (0, 0)
    fields = version.split(".")
    return (int(fields[0]), int(fields[1]))


def _nvcc_version() -> tuple[int, int]:
    if not CUDA_HOME:
        raise SystemExit("CUDA_HOME is unset; a local CUDA toolkit with nvcc is required")
    nvcc = Path(CUDA_HOME) / "bin" / "nvcc"
    if not nvcc.is_file():
        raise SystemExit(f"nvcc was not found at {nvcc}")
    text = subprocess.check_output([str(nvcc), "--version"], text=True)
    match = re.search(r"release\s+(\d+)\.(\d+)", text)
    if match is None:
        raise SystemExit(f"could not parse nvcc version from: {text}")
    version = (int(match.group(1)), int(match.group(2)))
    print(f"nvcc={version[0]}.{version[1]} path={nvcc}")
    return version


def _max_errors(length: int, batch: int = 1, heads: int = 1) -> tuple[float, list[float]]:
    torch.manual_seed(9300 + length + 17 * batch + 31 * heads)
    device = torch.device("cuda")
    shape = (batch, length, heads)
    x = (0.05 * torch.randn(*shape, 24, device=device, dtype=torch.float32)).requires_grad_()
    q = torch.randn(*shape, 24, device=device, dtype=torch.float32, requires_grad=True)
    v = (0.05 * torch.randn(*shape, 64, device=device, dtype=torch.float32)).requires_grad_()
    weight = torch.randn(*shape, 64, device=device, dtype=torch.float32)

    y = fused_ska_prefix_scan(x, q, v, ridge=1e-2)
    fused_grads = torch.autograd.grad((y * weight).sum(), (x, q, v))

    xr = x.detach().clone().requires_grad_()
    qr = q.detach().clone().requires_grad_()
    vr = v.detach().clone().requires_grad_()
    ref = dense_exact_oracle(xr, qr, vr, ridge=1e-2, power_k=1)
    ref_grads = torch.autograd.grad((ref * weight).sum(), (xr, qr, vr))

    forward_error = float((y.detach() - ref.detach()).abs().max())
    grad_errors = [
        float((got.detach() - expected.detach()).abs().max())
        for got, expected in zip(fused_grads, ref_grads)
    ]
    return forward_error, grad_errors


def _validate_lengths(lengths: Iterable[int]) -> None:
    max_forward = 0.0
    max_grads = [0.0, 0.0, 0.0]
    for length in lengths:
        forward_error, grad_errors = _max_errors(length)
        max_forward = max(max_forward, forward_error)
        max_grads = [max(a, b) for a, b in zip(max_grads, grad_errors)]
        print(
            f"length={length:3d} forward_max={forward_error:.3e} "
            f"dx={grad_errors[0]:.3e} dq={grad_errors[1]:.3e} dv={grad_errors[2]:.3e}"
        )

    # Fast-math FP32 thresholds are intentionally strict enough to catch an
    # indexing/causality error while allowing normal Cholesky roundoff.
    if max_forward > 3.0e-4:
        raise RuntimeError(f"forward validation failed: max error {max_forward:.3e}")
    if max(max_grads) > 1.2e-3:
        raise RuntimeError(f"backward validation failed: max errors {max_grads}")


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA-enabled PyTorch and a visible NVIDIA GPU are required")

    name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    cuda_version = torch.version.cuda
    nvcc_version = _nvcc_version()
    print(f"torch={torch.__version__} cuda={cuda_version}")
    print(f"device={name} capability={capability}")

    if os.environ.get("SKA_REQUIRE_B200", "0") == "1":
        if capability != (10, 0):
            raise SystemExit(
                f"B200 build requested, but device 0 has capability {capability}; expected (10, 0)"
            )
        if _version_tuple(cuda_version) < (12, 8) or nvcc_version < (12, 8):
            raise SystemExit(
                "native B200 compilation requires both a CUDA 12.8+ PyTorch "
                f"build and CUDA 12.8+ nvcc; got torch CUDA {cuda_version}, "
                f"nvcc {nvcc_version[0]}.{nvcc_version[1]}"
            )

    load_prefix_scan_ext(verbose=True)
    _validate_lengths((7, 33, 65))
    native_forward, native_grads = _max_errors(35, batch=2, heads=3)
    print(
        "native-layout B=2,H=3,T=35 "
        f"forward_max={native_forward:.3e} "
        f"dx={native_grads[0]:.3e} dq={native_grads[1]:.3e} dv={native_grads[2]:.3e}"
    )
    if native_forward > 3.0e-4 or max(native_grads) > 1.2e-3:
        raise RuntimeError("native [B,T,H,W] layout validation failed")
    torch.cuda.synchronize()
    print("fused prefix scan compiled and matched the exact dense forward/backward oracle")
    print("launch_info(N=96,T=2048):", launch_info(96, 2048))


if __name__ == "__main__":
    main()
