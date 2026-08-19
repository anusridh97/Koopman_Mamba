"""What a precision name means: one mapping, one autocast helper, one rule.

Step 1 of the precision-policy design, and deliberately inert -- nothing imports
it yet. The config fields that will consume it move `config_hash` for all 11
registry configs, so they land separately and on purpose.

Lives at the package root beside `config.py` and `pooling.py`: the category of
thing every subpackage may use and none owns. It must live under `koopman_lm/`
because the dependency edge runs `experimentation -> koopman_lm` only, enforced
by `code-tests/test_package_boundary.py`, and both sides need this.

**The policy it will serve.** `compute_precision` sets the floor for the whole
network; `ska_precision` and `mlp_precision` raise it for components whose math
needs more. `compute_precision` names the dtype ops *run in* -- weights are
always fp32, and storage is a separate concern.

**Why the two domains differ, and why that removes a check.**
`COMPUTE_PRECISIONS` excludes fp64 and `COMPONENT_PRECISIONS` excludes bf16/fp16.
Between them, "components may only ever raise precision" becomes unsatisfiable to
violate rather than something to validate: the global floor can never exceed fp32,
and fp32 is the minimum either component field allows. Widen either domain -- add
fp64 to compute, or bf16 to a component -- and an explicit ordering check becomes
necessary. Both domains are asserted by test for that reason.

The component domains exclude bf16 concretely, not fastidiously: the SKA core
takes a Cholesky of a Gram matrix, and bf16's 8 mantissa bits make that
unreliable, which is part of why `ska_ridge` exists. A field whose purpose is
protecting a fragile component must not be able to break it.

fp64 is not hypothetical on the component side. The exact prefix-scan path already
accepts float64, and fp64 is already used for numerical validation across
`kernels/ska_operator.py`, `kernels/chunk_stats_exact.py` and
`kernels/factor_scan.py`.
"""
from __future__ import annotations

import contextlib
from typing import Any, Dict, Iterator, Tuple

import torch

__all__ = ["DTYPES", "COMPUTE_PRECISIONS", "COMPONENT_PRECISIONS",
           "dtype_of", "rank_of", "needs_grad_scaler", "autocast"]

DTYPES: Dict[str, torch.dtype] = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
    "fp64": torch.float64,
}

# Ordering by width, for asserting the "only raise" invariant over the domains
# themselves. fp16 and bf16 rank equally: bf16 has fp32's exponent range with
# fewer mantissa bits, so neither strictly dominates, and nothing here needs them
# to.
_RANKS: Dict[str, int] = {"fp16": 0, "bf16": 0, "fp32": 1, "fp64": 2}

# The global autocast dtype. fp64 is excluded -- see the module docstring.
COMPUTE_PRECISIONS: Tuple[str, ...] = ("fp32", "bf16", "fp16")

# Per-component overrides, which may only raise precision.
COMPONENT_PRECISIONS: Tuple[str, ...] = ("fp32", "fp64")


def _validate(name: Any) -> str:
    if not isinstance(name, str):
        raise TypeError(
            f"precision must be a name like 'bf16', not {type(name).__name__} "
            f"({name!r}). Passing a torch.dtype here would make the config "
            f"surface two things.")
    if name not in DTYPES:
        raise ValueError(
            f"unknown precision {name!r}; expected one of {sorted(DTYPES)}")
    return name


def dtype_of(name: str) -> torch.dtype:
    """A precision name -> the torch dtype it denotes."""
    return DTYPES[_validate(name)]


def rank_of(name: str) -> int:
    """Relative width, for comparing two precisions."""
    return _RANKS[_validate(name)]


def needs_grad_scaler(compute_precision: str) -> bool:
    """Does this compute precision require a GradScaler?

    Derived rather than configured, so "fp16 without a scaler" -- which silently
    produces NaNs once gradients underflow -- is not a reachable state. bf16 has
    fp32's exponent range and needs no scaling.
    """
    return _validate(compute_precision) == "fp16"


@contextlib.contextmanager
def autocast(device_type: str, precision: str, *,
             enabled: bool = True) -> Iterator[None]:
    """Autocast to `precision`, or do nothing when that would be meaningless.

    fp32 disables rather than configures: autocasting *to* fp32 changes nothing
    and costs the dispatcher work, so "compute_precision: fp32" must mean "no
    autocast" and not "autocast to the default dtype".

    `enabled` exists so the call sites that currently read a `--bf16` CLI flag can
    keep that behaviour while taking the dtype from config.
    """
    dtype = dtype_of(precision)
    if not enabled or precision == "fp32":
        yield
        return
    with torch.autocast(device_type=device_type, dtype=dtype):
        yield
