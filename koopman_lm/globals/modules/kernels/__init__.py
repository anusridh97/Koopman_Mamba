"""Numerical kernels shared across the model -- NOT SKA-specific.

Chunk sufficient statistics, the whitened SKA core, factored Cholesky scans,
and rank-1 Cholesky updates. Consumed by the SKA token mixer (forward), the
recurrent decode path, and the last-layer ridge memory. Import the submodules
directly (e.g. `from ...kernels.core import ska_core`).
"""
