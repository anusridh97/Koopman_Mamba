"""Optimized exact low-rank symmetric Cholesky residual updates for CUDA."""

from .api import (
    BUCKETS,
    TILE,
    LowRankSymmetricState,
    kernel_layout_info,
    pack_lower_tiles,
    rank_bucket,
    residual_update_reference,
    tile_count,
    unpack_lower_tiles,
)

__all__ = [
    "BUCKETS",
    "TILE",
    "LowRankSymmetricState",
    "kernel_layout_info",
    "pack_lower_tiles",
    "rank_bucket",
    "residual_update_reference",
    "tile_count",
    "unpack_lower_tiles",
]

__version__ = "0.1.0"
