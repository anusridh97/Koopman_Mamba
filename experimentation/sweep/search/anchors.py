"""Curated anchor designs: hand-reasoned points, written relative to the base.

A design says "three-quarters the SKA depth, placed late, ridge x3" rather than
"ska_layer_indices: [5, 9, 12], ska_ridge: 0.03". Relative is the right
vocabulary for two reasons: a design set stays meaningful when pointed at a
different base model, and the intent survives in the file ("x3 ridge") where an
absolute value loses it.

`resolve_design` produces the same params dict shape the sampler produces, and
snaps it onto the declared space. That interchangeability is the point:

  * anchors can seed a study through `study.enqueue_trial`, so the first trials
    are chosen by a person and TPE models from real results rather than noise;
  * the same anchors can be materialized as an ordinary static `cells:` sweep,
    which is how a curated design set ships before any sampler exists.

Snapping matters for the first of those. `enqueue_trial` requires values that
exist in the study's distributions -- an unsnapped 20 against choices
[8, 16, 24, 32] is rejected at enqueue time, not silently coerced.

The minimum design count is the caller's policy, not this loader's. The original
harness hardcoded `if len(designs) < 15: raise`, which coupled a general
validator to one particular study; `load_designs(path, minimum=15)` says the
same thing where the requirement actually lives.
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Union

import yaml

from koopman_lm.config import KoopmanLMConfig
from experimentation.sweep.search.geometry import clamp, nearest

__all__ = ["Design", "load_designs", "resolve_design", "designs_to_cells"]

_BASELINE = "baseline"


@dataclass(frozen=True)
class Design:
    """One curated point. Every factor defaults to 1.0, so a design that sets
    nothing IS the base config -- which makes the reference anchor free and
    makes each design's diff from the baseline readable at a glance."""
    name: str
    rank: Union[int, str] = _BASELINE
    layer_factor: float = 1.0
    placement: str = _BASELINE
    ridge_factor: float = 1.0
    layerscale_factor: float = 1.0
    lr_factor: float = 1.0
    norm_clip_multiplier: Union[float, str] = _BASELINE
    gamma_value: Union[float, str] = _BASELINE
    #: The SKA matrix power. `"baseline"` (the default) inherits the base
    #: config's own `ska_power_K`, which is what every anchor did before this
    #: field existed -- so no committed design file changes meaning.
    #:
    #: An explicit value is treated differently from an inherited one, and
    #: deliberately: an inherited K SNAPS onto the declared choices (the base
    #: config's K is always a declared choice, so it never actually moves),
    #: while an explicit K must be declared EXACTLY or it raises. K is a matrix
    #: power over a set of two or three small integers, so snapping 3 -> 2 would
    #: leave a design named `reference-k3` running at K=2 -- a trial whose name
    #: asserts something its params deny, which is the one failure mode a
    #: named design set exists to prevent.
    power_K: Union[int, str] = _BASELINE
    weight_decay: float = 0.1
    warmup_ratio: float = 0.02
    grad_clip: float = 1.0


_FIELDS = {f.name for f in dataclasses.fields(Design)}


def load_designs(path, *, minimum: int = 1) -> List[Design]:
    """Read a design file: a top-level `designs:` list of named mappings."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text()) or {}
    entries = raw.get("designs") if isinstance(raw, Mapping) else None
    if not isinstance(entries, list):
        raise ValueError(
            f"{path} must contain a top-level 'designs' list; got keys "
            f"{sorted(raw) if isinstance(raw, Mapping) else type(raw).__name__}")

    designs: List[Design] = []
    seen: set[str] = set()
    for position, entry in enumerate(entries, start=1):
        if not isinstance(entry, Mapping):
            raise ValueError(f"{path}: design #{position} is not a mapping")
        unknown = sorted(set(entry) - _FIELDS)
        if unknown:
            # Silently ignoring a typo is the worst outcome available: the trial
            # would run as the baseline while its name claims otherwise.
            raise ValueError(
                f"{path}: design #{position} has unknown field(s) {unknown}; "
                f"known fields are {sorted(_FIELDS)}")
        name = entry.get("name")
        if not name:
            raise ValueError(
                f"{path}: design #{position} has no 'name'; names are how a "
                f"trial is traced back to the reasoning that proposed it")
        if name in seen:
            raise ValueError(f"{path}: duplicate design name {name!r}")
        seen.add(name)
        designs.append(Design(**entry))

    if len(designs) < minimum:
        raise ValueError(
            f"{path} declares {len(designs)} design(s); this study requires at "
            f"least {minimum}")
    return designs


def _baseline_norm_clip_multiplier(base_model: KoopmanLMConfig) -> float:
    clip_c = base_model.ska_norm_clip_c
    if clip_c is None:
        clip_c = math.sqrt(base_model.ska_rank)
    return float(clip_c) / math.sqrt(base_model.ska_rank)


def resolve_design(design: Design, base_model: KoopmanLMConfig,
                   space: Mapping[str, Mapping[str, Any]], *,
                   base_lr: float) -> Dict[str, Any]:
    """One design -> a params dict, snapped and clamped onto `space`.

    Categoricals snap to the nearest declared choice; floats clamp into the
    declared bounds. Both are lossy on purpose: the study's space is what
    exists, and a design asking for something outside it should get the closest
    available point rather than an error, since the design was written against
    the base config and not against this particular study's bounds.
    """
    placements = space["placement"]["choices"]
    if design.placement not in placements:
        raise ValueError(
            f"design {design.name!r} asks for placement={design.placement!r}; "
            f"this study declares {list(placements)}")

    rank = base_model.ska_rank if design.rank == _BASELINE else int(design.rank)
    base_count = len(base_model.ska_layer_indices)
    multiplier = (_baseline_norm_clip_multiplier(base_model)
                  if design.norm_clip_multiplier == _BASELINE
                  else float(design.norm_clip_multiplier))
    gamma = (base_model.ska_gamma_value if design.gamma_value == _BASELINE
             else float(design.gamma_value))

    ridge_low, ridge_high = space["ska_ridge"]["low"], space["ska_ridge"]["high"]
    ls_low, ls_high = (space["ska_layerscale_init"]["low"],
                       space["ska_layerscale_init"]["high"])
    lr_low, lr_high = space["learning_rate"]["low"], space["learning_rate"]["high"]

    return {
        "ska_rank": int(nearest(rank, space["ska_rank"]["choices"])),
        "n_ska_layers": int(nearest(max(1, round(base_count * design.layer_factor)),
                                    space["n_ska_layers"]["choices"])),
        "placement": design.placement,
        "ska_ridge": clamp(base_model.ska_ridge * design.ridge_factor,
                           ridge_low, ridge_high),
        "ska_layerscale_init": clamp(
            base_model.ska_layerscale_init * design.layerscale_factor, ls_low, ls_high),
        "norm_clip_multiplier": nearest(multiplier,
                                        space["norm_clip_multiplier"]["choices"]),
        "gamma_value": nearest(gamma, space["gamma_value"]["choices"]),
        "learning_rate": clamp(base_lr * design.lr_factor, lr_low, lr_high),
        "weight_decay": nearest(design.weight_decay, space["weight_decay"]["choices"]),
        "warmup_ratio": nearest(design.warmup_ratio, space["warmup_ratio"]["choices"]),
        "grad_clip": nearest(design.grad_clip, space["grad_clip"]["choices"]),
        "ska_power_K": _resolve_power_k(design, base_model, space),
    }


def _resolve_power_k(design: Design, base_model: KoopmanLMConfig,
                     space: Mapping[str, Any]) -> int:
    """`design.power_K` -> a declared choice, or a loud failure.

    Inherited (`"baseline"`) SNAPS: an anchor is defined relative to the config
    you already run, and `space._with_value_int` guarantees that config's own K
    is a declared choice, so the snap never actually moves anything.

    Explicit does NOT snap, for the reason `Design.power_K` documents: silently
    resolving a requested 3 to 2 leaves a design whose NAME claims one thing and
    whose params say another. Placement is refused the same way, and for the
    same reason.
    """
    choices = list(space["ska_power_K"]["choices"])
    if design.power_K == _BASELINE:
        return int(nearest(base_model.ska_power_K, choices))
    try:
        as_float = float(design.power_K)
    except (TypeError, ValueError):
        raise ValueError(
            f"design {design.name!r} asks for power_K={design.power_K!r}, which "
            f"is not a number. Use an integer, or 'baseline' to inherit the base "
            f"config's own ska_power_K.") from None
    if as_float != int(as_float):
        raise ValueError(
            f"design {design.name!r} asks for power_K={design.power_K!r}; K is a "
            f"MATRIX POWER and must be a whole number -- int({design.power_K!r}) "
            f"would silently round it")
    requested = int(as_float)
    if requested not in choices:
        raise ValueError(
            f"design {design.name!r} asks for power_K={requested}; this study "
            f"declares {choices}. Unlike a ridge or a learning rate, K is not "
            f"clamped onto the nearest declared value: a design named for the K "
            f"it runs at must actually run at it. Either widen the ska_power_K "
            f"axis in the study's search_axes, or name a declared value.")
    return requested


def designs_to_cells(designs, base_model: KoopmanLMConfig,
                     space: Mapping[str, Mapping[str, Any]], *,
                     base_lr: float,
                     max_steps: int,
                     backend_policy: str = "exact_invchol",
                     seq_len: int | None = None) -> List[Dict[str, Any]]:
    """A design set -> a `cells:` list for configs/sweeps/<name>.yaml.

    Composes resolve_design with space.params_to_overrides, which is the whole
    trick: a curated design set becomes an ordinary static sweep, and therefore
    launchable through the unmodified run system, with no sampler and no optuna
    anywhere in the path.

    Order is preserved so a generated sweep's Nth cell is the Nth design, which
    is what lets a companion name->run_id mapping be built positionally against
    the same expand_cells output the launcher will use.
    """
    from experimentation.sweep.search.space import params_to_overrides

    return [
        params_to_overrides(
            resolve_design(design, base_model, space, base_lr=base_lr),
            base_model, max_steps=max_steps, seq_len=seq_len,
            backend_policy=backend_policy)
        for design in designs
    ]
