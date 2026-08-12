"""Exact low-rank symmetric CUDA backend using an ``(L, C=A_w-I)`` state.

The common symmetric update is

    G+ = G + z z^T,
    M+ = M + z z^T,
    A_w = L^{-1} M L^{-T},
    G = L L^T.

Because ``D=M-G`` is invariant, the hot state can be reduced to

    C = A_w - I = L^{-1} D L^{-T}.

If ``[L | z] Q = [L+ | 0]`` is the ordinary Givens Cholesky update, then

    C+ = top_left_r(Q^T diag(C, 0) Q).

The CUDA extension implements the Cholesky update and this symmetric
congruence in one launch, with no triangular solve, inverse factor, prefix
scan, rank-one correction, descriptor pass, or global scratch tensor.

Persistent matrices use 32x32 lower-tile storage.  Diagonal tiles retain only
their lower triangles.  This is the hot representation; dense conversion is
provided only for initialization, validation, refresh, and API boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

import torch


TILE = 32
BUCKETS = (32, 64, 96, 128, 160, 192, 224, 256)
UpdateMode = Literal["auto", "resident", "staged"]


def rank_bucket(rank: int) -> int:
    """Return the compile-time storage bucket for ``1 <= rank <= 256``."""
    rank = int(rank)
    if rank <= 0:
        raise ValueError("rank must be positive")
    for bucket in BUCKETS:
        if rank <= bucket:
            return bucket
    raise ValueError(f"rank={rank} exceeds the low-rank limit of 256")


def tile_count(bucket: int) -> int:
    if bucket not in BUCKETS:
        raise ValueError(f"unsupported bucket: {bucket}")
    n = bucket // TILE
    return n * (n + 1) // 2


def _as_batched_matrix(matrix: torch.Tensor, name: str) -> tuple[torch.Tensor, bool]:
    if matrix.ndim == 2:
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"{name} must be square")
        return matrix.unsqueeze(0), True
    if matrix.ndim == 3:
        if matrix.shape[-2] != matrix.shape[-1]:
            raise ValueError(f"{name} must be square")
        if matrix.shape[0] <= 0:
            raise ValueError(f"{name} batch must be positive")
        return matrix, False
    raise ValueError(f"{name} must have shape (r,r) or (batch,r,r)")


def _as_batched_vector(
    vector: torch.Tensor,
    *,
    batch: int,
    rank: int,
    name: str,
) -> tuple[torch.Tensor, bool]:
    if vector.ndim == 1:
        if batch != 1 or vector.shape[0] != rank:
            raise ValueError(
                f"{name} must have shape ({rank},) only for a single state"
            )
        return vector.unsqueeze(0), True
    if vector.ndim == 2 and vector.shape == (batch, rank):
        return vector, False
    raise ValueError(f"{name} must have shape ({rank},) or ({batch},{rank})")


def pack_lower_tiles(
    matrix: torch.Tensor,
    *,
    bucket: int | None = None,
) -> torch.Tensor:
    """Pack a dense lower-triangular/symmetric matrix into 32x32 lower tiles.

    The returned tensor always has shape ``(batch, tile_count, 32, 32)``.
    Off-diagonal lower tiles are stored in full.  Upper entries of diagonal
    tiles are explicitly zeroed so they can never leak into later kernels.
    """
    batched, _ = _as_batched_matrix(matrix, "matrix")
    rank = batched.shape[-1]
    bucket = rank_bucket(rank) if bucket is None else int(bucket)
    if bucket not in BUCKETS or bucket < rank:
        raise ValueError(f"bucket={bucket} cannot hold rank={rank}")

    batch = batched.shape[0]
    packed = batched.new_zeros((batch, tile_count(bucket), TILE, TILE))
    tile = 0
    for tile_i in range(bucket // TILE):
        row0 = tile_i * TILE
        row_count = max(0, min(TILE, rank - row0))
        for tile_j in range(tile_i + 1):
            col0 = tile_j * TILE
            col_count = max(0, min(TILE, rank - col0))
            if row_count and col_count:
                packed[:, tile, :row_count, :col_count].copy_(
                    batched[:, row0 : row0 + row_count, col0 : col0 + col_count]
                )
                if tile_i == tile_j:
                    packed[:, tile].tril_()
            tile += 1
    return packed.contiguous()


def unpack_lower_tiles(
    packed: torch.Tensor,
    rank: int,
    *,
    symmetric: bool,
) -> torch.Tensor:
    """Convert packed lower tiles to a dense matrix.

    ``symmetric=False`` produces a lower-triangular matrix (for ``L``).
    ``symmetric=True`` mirrors the stored lower triangle (for ``C``).
    A batch dimension is always retained.
    """
    rank = int(rank)
    bucket = rank_bucket(rank)
    expected_tiles = tile_count(bucket)
    if packed.ndim != 4 or packed.shape[1:] != (expected_tiles, TILE, TILE):
        raise ValueError(
            "packed must have shape "
            f"(batch,{expected_tiles},{TILE},{TILE}) for rank={rank}"
        )

    batch = packed.shape[0]
    dense = packed.new_zeros((batch, rank, rank))
    tile = 0
    for tile_i in range(bucket // TILE):
        row0 = tile_i * TILE
        row_count = max(0, min(TILE, rank - row0))
        for tile_j in range(tile_i + 1):
            col0 = tile_j * TILE
            col_count = max(0, min(TILE, rank - col0))
            if row_count and col_count:
                block = packed[:, tile, :row_count, :col_count]
                if tile_i == tile_j:
                    lower = torch.tril(block)
                    if symmetric:
                        diagonal = torch.diagonal(lower, dim1=-2, dim2=-1)
                        symmetric_block = (
                            lower
                            + lower.transpose(-1, -2)
                            - torch.diag_embed(diagonal)
                        )
                        dense[
                            :, row0 : row0 + row_count, col0 : col0 + col_count
                        ].copy_(symmetric_block)
                    else:
                        dense[
                            :, row0 : row0 + row_count, col0 : col0 + col_count
                        ].copy_(lower)
                else:
                    dense[
                        :, row0 : row0 + row_count, col0 : col0 + col_count
                    ].copy_(block)
                    if symmetric:
                        dense[
                            :, col0 : col0 + col_count, row0 : row0 + row_count
                        ].copy_(block.transpose(-1, -2))
            tile += 1
    return dense


@lru_cache(maxsize=1)
def _extension():
    try:
        from . import _C as _lowrank_residual_ext  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on CUDA build host
        raise RuntimeError(
            "The lowrank_residual_cuda CUDA extension is not built. "
            "Install from the package root with:\n"
            "  TORCH_CUDA_ARCH_LIST=10.0 pip install ."
        ) from exc
    return _lowrank_residual_ext


def kernel_layout_info(rank: int) -> dict[str, int | str]:
    """Return the persistent and shared-memory layout selected for ``rank``."""
    bucket = rank_bucket(rank)
    ntiles = tile_count(bucket)
    shared_elements = ntiles * TILE * TILE
    resident_bytes = (2 * shared_elements + bucket) * 4
    staged_bytes = (shared_elements + 3 * bucket) * 4
    apply_bytes = (shared_elements + bucket) * 4
    return {
        "rank": int(rank),
        "bucket": bucket,
        "tiles": ntiles,
        "persistent_bytes_per_matrix": ntiles * TILE * TILE * 4,
        "resident_shared_bytes": resident_bytes,
        "staged_shared_bytes": staged_bytes,
        "apply_shared_bytes": apply_bytes,
        "default_update_kernel": "resident" if bucket <= 224 else "staged",
        "threads": 32 if bucket <= 64 else (128 if bucket <= 128 else 256),
    }


def residual_update_reference(
    L: torch.Tensor,
    C: torch.Tensor,
    z: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dense correctness oracle for the exact ``(L,C)`` recurrence.

    This is intentionally not optimized and is suitable only for tests.
    It supports single or batched states and preserves the input dtype.
    """
    Lb, L_single = _as_batched_matrix(L, "L")
    Cb, C_single = _as_batched_matrix(C, "C")
    if L_single != C_single or Lb.shape != Cb.shape:
        raise ValueError("L and C must have identical shapes")
    batch, rank, _ = Lb.shape
    zb, z_single = _as_batched_vector(z, batch=batch, rank=rank, name="z")
    if z_single != L_single:
        raise ValueError("z batching must match L and C")

    L_work = Lb.clone()
    C_work = Cb.clone()
    z_work = zb.clone()
    cs = Lb.new_empty((batch, rank))
    ss = Lb.new_empty((batch, rank))

    for k in range(rank):
        a = L_work[:, k, k]
        b = z_work[:, k]
        rho = torch.sqrt(a * a + b * b)
        nonzero = rho != 0
        safe = torch.where(nonzero, rho, torch.ones_like(rho))
        c = torch.where(nonzero, a / safe, torch.ones_like(a))
        s = torch.where(nonzero, b / safe, torch.zeros_like(b))
        cs[:, k] = c
        ss[:, k] = s

        old_l = L_work[:, k:, k].clone()
        old_z = z_work[:, k:].clone()
        L_work[:, k:, k] = c[:, None] * old_l + s[:, None] * old_z
        z_work[:, k:] = -s[:, None] * old_l + c[:, None] * old_z

    padded = Cb.new_zeros((batch, rank))
    delta = Cb.new_zeros((batch,))
    for k in range(rank):
        c = cs[:, k]
        s = ss[:, k]
        old_row = C_work[:, k, :].clone()
        old_padded = padded.clone()
        old_a = old_row[:, k]
        old_b = old_padded[:, k]

        new_row = c[:, None] * old_row + s[:, None] * old_padded
        new_padded = -s[:, None] * old_row + c[:, None] * old_padded

        c2 = c * c
        s2 = s * s
        cs_value = c * s
        new_a = c2 * old_a + 2 * cs_value * old_b + s2 * delta
        new_b = (c2 - s2) * old_b + cs_value * (delta - old_a)
        new_delta = s2 * old_a - 2 * cs_value * old_b + c2 * delta

        new_row[:, k] = new_a
        new_padded[:, k] = new_b
        C_work[:, k, :] = new_row
        C_work[:, :, k] = new_row
        padded = new_padded
        delta = new_delta

    if L_single:
        return L_work[0], C_work[0]
    return L_work, C_work


@dataclass
class LowRankSymmetricState:
    """Packed exact state for the optimized symmetric low-rank CUDA path."""

    L_tiles: torch.Tensor
    C_tiles: torch.Tensor
    rank: int
    bucket: int
    single: bool = False

    def __post_init__(self) -> None:
        self.rank = int(self.rank)
        self.bucket = int(self.bucket)
        expected_bucket = rank_bucket(self.rank)
        if self.bucket != expected_bucket:
            raise ValueError(
                f"bucket={self.bucket} does not match rank bucket {expected_bucket}"
            )
        expected_shape = (tile_count(self.bucket), TILE, TILE)
        if self.L_tiles.ndim != 4 or self.L_tiles.shape[1:] != expected_shape:
            raise ValueError(f"L_tiles must have trailing shape {expected_shape}")
        if self.C_tiles.shape != self.L_tiles.shape:
            raise ValueError("C_tiles must have the same shape as L_tiles")
        if self.L_tiles.dtype != torch.float32 or self.C_tiles.dtype != torch.float32:
            raise ValueError("optimized state must use float32")
        if self.L_tiles.device != self.C_tiles.device:
            raise ValueError("L_tiles and C_tiles must share a device")
        if not self.L_tiles.is_contiguous() or not self.C_tiles.is_contiguous():
            raise ValueError("packed state tensors must be contiguous")
        if self.single and self.batch_size != 1:
            raise ValueError("single=True requires batch size one")

    @classmethod
    def from_dense(
        cls,
        L: torch.Tensor,
        Aw: torch.Tensor,
        *,
        check_symmetric: bool = False,
        symmetry_rtol: float = 1e-5,
        symmetry_atol: float = 1e-6,
    ) -> "LowRankSymmetricState":
        """Create packed state from dense ``L`` and symmetric ``A_w``.

        Conversion is an initialization/refresh operation, not part of the hot
        update path.  The CUDA kernel requires float32 state.
        """
        Lb, L_single = _as_batched_matrix(L, "L")
        Awb, Aw_single = _as_batched_matrix(Aw, "Aw")
        if L_single != Aw_single or Lb.shape != Awb.shape:
            raise ValueError("L and Aw must have identical shapes")
        if Lb.device != Awb.device or Lb.dtype != Awb.dtype:
            raise ValueError("L and Aw must share dtype and device")
        if Lb.dtype != torch.float32:
            raise ValueError("optimized state initialization requires float32")
        if check_symmetric and not torch.allclose(
            Awb, Awb.transpose(-1, -2), rtol=symmetry_rtol, atol=symmetry_atol
        ):
            raise ValueError("Aw is not symmetric within the requested tolerance")

        rank = Lb.shape[-1]
        bucket = rank_bucket(rank)
        C = Awb.clone()
        torch.diagonal(C, dim1=-2, dim2=-1).sub_(1.0)
        return cls(
            L_tiles=pack_lower_tiles(Lb, bucket=bucket),
            C_tiles=pack_lower_tiles(C, bucket=bucket),
            rank=rank,
            bucket=bucket,
            single=L_single,
        )

    @property
    def batch_size(self) -> int:
        return int(self.L_tiles.shape[0])

    @property
    def device(self) -> torch.device:
        return self.L_tiles.device

    @property
    def storage_bytes(self) -> int:
        return (
            self.L_tiles.numel() * self.L_tiles.element_size()
            + self.C_tiles.numel() * self.C_tiles.element_size()
        )

    def clone(self) -> "LowRankSymmetricState":
        return LowRankSymmetricState(
            self.L_tiles.clone(),
            self.C_tiles.clone(),
            self.rank,
            self.bucket,
            self.single,
        )

    def update_(self, z: torch.Tensor, *, mode: UpdateMode = "auto") -> "LowRankSymmetricState":
        """Apply one exact symmetric rank-one update in place.

        No tensor is allocated or copied by this method.  ``z`` must already
        be contiguous float32 CUDA storage on the state device.
        """
        zb, _ = _as_batched_vector(
            z, batch=self.batch_size, rank=self.rank, name="z"
        )
        self._validate_hot_tensor(zb, "z")
        ext = _extension()
        if mode == "auto":
            ext.update_(self.L_tiles, self.C_tiles, zb, self.rank)
        elif mode == "resident":
            if self.bucket > 224:
                raise ValueError("resident mode supports rank buckets through 224")
            ext.update_resident_(self.L_tiles, self.C_tiles, zb, self.rank)
        elif mode == "staged":
            ext.update_staged_(self.L_tiles, self.C_tiles, zb, self.rank)
        else:
            raise ValueError(f"unknown update mode: {mode}")
        return self

    def update_sequence_(self, Z: torch.Tensor) -> "LowRankSymmetricState":
        """Fuse a sequence of exact updates into one launch for rank <= 224."""
        if self.bucket > 224:
            raise ValueError("sequence fusion supports rank buckets through 224")
        if Z.ndim == 2:
            if not self.single or Z.shape[1] != self.rank:
                raise ValueError(
                    f"Z must have shape (T,{self.rank}) only for a single state"
                )
            Zb = Z.unsqueeze(0)
        elif Z.ndim == 3 and Z.shape[0] == self.batch_size and Z.shape[2] == self.rank:
            Zb = Z
        else:
            raise ValueError(
                f"Z must have shape (T,{self.rank}) or "
                f"({self.batch_size},T,{self.rank})"
            )
        if Zb.shape[1] <= 0:
            raise ValueError("sequence length must be positive")
        self._validate_hot_tensor(Zb, "Z")
        _extension().update_sequence_(
            self.L_tiles, self.C_tiles, Zb, self.rank
        )
        return self

    def apply_out_(self, x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        """Write ``out = A_w x = x + Cx`` without materializing dense ``A_w``."""
        xb, x_single = _as_batched_vector(
            x, batch=self.batch_size, rank=self.rank, name="x"
        )
        yb, y_single = _as_batched_vector(
            out, batch=self.batch_size, rank=self.rank, name="out"
        )
        if x_single != y_single:
            raise ValueError("x and out must use the same batching convention")
        self._validate_hot_tensor(xb, "x")
        self._validate_hot_tensor(yb, "out")
        if xb.data_ptr() == yb.data_ptr():
            raise ValueError("in-place apply is not supported; x and out must differ")
        _extension().apply_residual_out(self.C_tiles, xb, yb, self.rank)
        return out

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """Allocate and return ``A_w x``.  Use :meth:`apply_out_` in hot code."""
        out = torch.empty_like(x)
        return self.apply_out_(x, out)

    def dense_L(self) -> torch.Tensor:
        dense = unpack_lower_tiles(self.L_tiles, self.rank, symmetric=False)
        return dense[0] if self.single else dense

    def dense_C(self) -> torch.Tensor:
        dense = unpack_lower_tiles(self.C_tiles, self.rank, symmetric=True)
        return dense[0] if self.single else dense

    def dense_Aw(self) -> torch.Tensor:
        dense = self.dense_C()
        torch.diagonal(dense, dim1=-2, dim2=-1).add_(1.0)
        return dense

    def _validate_hot_tensor(self, tensor: torch.Tensor, name: str) -> None:
        if tensor.device != self.device:
            raise ValueError(f"{name} must be on {self.device}")
        if tensor.dtype != torch.float32:
            raise ValueError(f"{name} must be float32")
        if not tensor.is_contiguous():
            raise ValueError(
                f"{name} must be contiguous; the hot path never inserts a copy"
            )
        if tensor.device.type != "cuda":
            raise ValueError("optimized updates require CUDA")


__all__ = [
    "BUCKETS",
    "LowRankSymmetricState",
    "kernel_layout_info",
    "pack_lower_tiles",
    "rank_bucket",
    "residual_update_reference",
    "tile_count",
    "unpack_lower_tiles",
]
