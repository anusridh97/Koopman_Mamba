"""Lazy CUDA backend for the exact rank-24 SKA prefix scan.

Production geometry: rank 24, value/head width 64, exact 32-token scheduling
blocks, eight-token backward checkpoints, and operator power K=1. The extension
consumes the model's native contiguous [B,T,H,W] layout directly.
"""
from __future__ import annotations

import os
from pathlib import Path
import threading
from typing import Final

import torch

_RANK: Final = 24
_VALUE: Final = 64
_BLOCK: Final = 32
_CHECKPOINT: Final = 8

_ext = None
_ext_error: BaseException | None = None
_lock = threading.Lock()


def load_prefix_scan_ext(*, verbose: bool | None = None):
    """Build once and return the fused CUDA extension."""
    global _ext, _ext_error
    if _ext is not None:
        return _ext
    if _ext_error is not None:
        raise RuntimeError("the fused SKA CUDA extension previously failed to build") from _ext_error
    if not torch.cuda.is_available():
        raise RuntimeError("the fused SKA prefix scan requires CUDA-enabled PyTorch")

    with _lock:
        if _ext is not None:
            return _ext
        if _ext_error is not None:
            raise RuntimeError("the fused SKA CUDA extension previously failed to build") from _ext_error
        try:
            from torch.utils.cpp_extension import load
            source = Path(__file__).with_name("csrc") / "prefix_scan_ext.cu"
            if not source.is_file():
                raise FileNotFoundError(f"missing CUDA source: {source}")
            if verbose is None:
                verbose = os.environ.get("SKA_PREFIX_SCAN_BUILD_VERBOSE", "0") == "1"
            cuda_flags = [
                "-O3",
                "--use_fast_math",
                "--lineinfo",
                "--extra-device-vectorization",
                "-std=c++17",
                "-DNDEBUG",
                "-Xptxas=-O3",
                "-Xptxas=-warn-spills",
            ]
            if os.environ.get("SKA_PREFIX_SCAN_PTXAS_VERBOSE", "0") == "1":
                cuda_flags.append("-Xptxas=-v")
            _ext = load(
                name="ska_prefix_scan_r24p64_v3_native",
                sources=[str(source)],
                extra_cuda_cflags=cuda_flags,
                extra_cflags=["-O3", "-std=c++17"],
                with_cuda=True,
                verbose=bool(verbose),
            )
            return _ext
        except BaseException as exc:
            _ext_error = exc
            raise


def is_supported(
    x: torch.Tensor,
    q: torch.Tensor,
    vbar: torch.Tensor,
    *,
    power_k: int,
    block_size: int,
) -> bool:
    return (
        x.is_cuda and q.is_cuda and vbar.is_cuda
        and x.dtype == q.dtype == vbar.dtype == torch.float32
        and x.ndim == q.ndim == vbar.ndim == 4
        and x.shape == q.shape
        and x.shape[:-1] == vbar.shape[:-1]
        and x.shape[-1] == _RANK
        and vbar.shape[-1] == _VALUE
        and int(power_k) == 1
        and int(block_size) == _BLOCK
    )



class _FusedPrefixScanFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, q, vbar, ridge, power_k, block_size):
        if not is_supported(x, q, vbar, power_k=power_k, block_size=block_size):
            raise ValueError(
                "fused CUDA requires FP32 CUDA tensors, rank=24, value width=64, "
                "power_k=1, and block_size=32"
            )
        ext = load_prefix_scan_ext()
        x = x.contiguous()
        q = q.contiguous()
        vbar = vbar.contiguous()
        y, checkpoints = ext.forward(
            x, q, vbar, float(ridge), int(power_k), int(block_size), _CHECKPOINT
        )
        ctx.save_for_backward(x, q, vbar, checkpoints)
        ctx.power_k = int(power_k)
        ctx.block_size = int(block_size)
        return y

    @staticmethod
    def backward(ctx, dy):
        x, q, vbar, checkpoints = ctx.saved_tensors
        dx, dq, dv = load_prefix_scan_ext().backward(
            x, q, vbar, dy.contiguous(), checkpoints,
            ctx.power_k, ctx.block_size, _CHECKPOINT,
        )
        return dx, dq, dv, None, None, None


def fused_ska_prefix_scan(
    x: torch.Tensor,
    q: torch.Tensor,
    vbar: torch.Tensor,
    ridge: float,
    power_k: int = 1,
    block_size: int = 32,
) -> torch.Tensor:
    return _FusedPrefixScanFn.apply(
        x, q, vbar, float(ridge), int(power_k), int(block_size)
    )


def launch_info(N: int, T: int) -> list[int]:
    return list(load_prefix_scan_ext().launch_info(int(N), int(T)))


def warmup(device: torch.device | str = "cuda") -> None:
    device = torch.device(device)
    x = torch.randn(1, 32, 1, _RANK, device=device, dtype=torch.float32, requires_grad=True)
    q = torch.randn_like(x, requires_grad=True)
    v = torch.randn(1, 32, 1, _VALUE, device=device, dtype=torch.float32, requires_grad=True)
    fused_ska_prefix_scan(x, q, v, ridge=1e-2).square().mean().backward()
