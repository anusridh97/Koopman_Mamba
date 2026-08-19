"""Where SKA layers go, and how many: pure arithmetic over layer indices.

The bottom of the search package. No optuna, no torch, no config object -- it
takes integers and returns integers, which is why it can be pinned exhaustively
on a CPU box in milliseconds.

Two decisions are encoded here rather than in the sampler, because they are
modelling choices and not search-strategy choices.

**Layer 0 and the final layer stay free.** An SKA adapter at depth 0 reads
embeddings no local mixer has touched yet, and one at the final depth leaves no
Mamba-only cleanup layer after it. So the usable window is
`range(1, n_layers - 1)`, and capacity is `n_layers - 2`.

**The space contains the baseline.** `layer_count_choices` always includes the
base count, and `placement="baseline"` reproduces the base indices *exactly*
when the count matches. A search whose space cannot express the config you
already run cannot tell you whether you beat it, and an approximate
reproduction is worse than none -- it looks like the reference and is not.
"""
from __future__ import annotations

from typing import Sequence

__all__ = ["clamp", "nearest", "layer_count_choices", "make_layer_indices",
           "PLACEMENTS"]

PLACEMENTS = ("baseline", "even", "midlate", "late")

# `midlate` starts here, as a fraction of depth: deep enough to read composed
# features, shallow enough to leave several layers of processing after it.
_MIDLATE_START_FRACTION = 0.30

# `late` skews its uniform spacing by u -> 1 - (1-u)**EXPONENT. Above 1 biases
# toward depth; 1.70 was the value carried over from the original design.
_LATE_SKEW_EXPONENT = 1.70


def clamp(value: float, lo: float, hi: float) -> float:
    return min(hi, max(lo, value))


def nearest(value: float, choices: Sequence[float]) -> float:
    """The element of `choices` closest to `value`, ties going to the smaller.

    The tie-break is deterministic on purpose. A sampler that proposes an exact
    midpoint must resolve it the same way every time, or two trials that
    proposed identical parameters would get different configs -- and therefore
    different run_ids, which would silently split one experiment in two.
    """
    return min(choices, key=lambda c: (abs(float(c) - float(value)), float(c)))


def layer_count_choices(base_count: int, n_layers: int) -> list[int]:
    """Candidate SKA-layer counts around `base_count`: half, three-quarters,
    baseline, one-and-a-half, double -- clamped to the usable window.

    Multiplicative rather than additive because the question being asked is
    "does more SKA depth help?", and the answer scales with the backbone.
    """
    capacity = max(1, n_layers - 2)
    raw = [
        max(2, round(base_count * 0.50)),
        max(2, round(base_count * 0.75)),
        base_count,
        round(base_count * 1.50),
        base_count * 2,
    ]
    return sorted({min(capacity, max(1, int(x))) for x in raw})


def _nearest_unused(target: float, available: set[int]) -> int:
    return min(available, key=lambda x: (abs(x - target), x))


def make_layer_indices(n_layers: int, count: int, placement: str,
                       base_indices: Sequence[int]) -> list[int]:
    """`count` SKA depths in an `n_layers` backbone, arranged by `placement`.

    Returns sorted, unique indices inside the free window. `count` is clamped to
    the window's size rather than raising: a sampler that proposes more adapters
    than there are slots should get the deepest legal configuration, not a
    failed trial.
    """
    if placement not in PLACEMENTS:
        raise ValueError(
            f"unknown placement={placement!r}; expected one of {list(PLACEMENTS)}")

    window = list(range(1, max(2, n_layers - 1)))
    if not window:                      # a backbone too small to have an interior
        window = list(range(n_layers))
    count = min(max(1, count), len(window))
    first, last = float(window[0]), float(window[-1])

    if placement == "baseline":
        # Exact reproduction when the count matches and every base index is
        # legal; otherwise interpolate across the base indices' own span.
        if count == len(base_indices):
            valid = sorted({int(x) for x in base_indices if x in window})
            if len(valid) == count:
                return valid
        lo = max(first, float(min(base_indices)))
        hi = min(last, float(max(base_indices)))
        if hi <= lo:
            lo, hi = first, last
        targets = _spread(lo, hi, count)
    elif placement == "even":
        targets = _spread(first, last, count)
    elif placement == "midlate":
        lo = max(first, float(round(_MIDLATE_START_FRACTION * (n_layers - 1))))
        targets = _spread(lo, last, count)
    else:                               # "late"
        targets = [
            first + (last - first) * (1.0 - (1.0 - (i + 1) / (count + 1)) ** _LATE_SKEW_EXPONENT)
            for i in range(count)
        ]

    unused = set(window)
    chosen: list[int] = []
    for target in targets:
        index = _nearest_unused(target, unused)
        unused.remove(index)
        chosen.append(index)
    return sorted(chosen)


def _spread(lo: float, hi: float, count: int) -> list[float]:
    """`count` evenly spaced points from `lo` to `hi` inclusive."""
    return [lo + (hi - lo) * i / max(count - 1, 1) for i in range(count)]
