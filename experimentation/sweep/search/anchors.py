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
from typing import Any, Dict, List, Mapping, Optional, Union

import yaml

from koopman_lm.config import KoopmanLMConfig
from experimentation.sweep.search.geometry import clamp, nearest

__all__ = ["Design", "check_space_is_resolvable", "check_replicates_resolve",
           "load_designs", "reference_groups", "resolve_design",
           "designs_to_cells"]

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
    #: The TRAINING seed -- `runtime.seed`, which seeds initialisation and batch
    #: order. `"baseline"` (the default) inherits the base spec's own, which is
    #: what every design did before this field existed, so no committed design
    #: file changes meaning.
    #:
    #: NOT a search parameter, and it cannot become one: `resolve_design` emits
    #: the params dict optuna's declared distributions have to accept, and a key
    #: with no distribution is rejected at `enqueue_trial` time. So the seed
    #: travels as trial METADATA (`study.SEED_ATTR`) and the driver applies it as
    #: a `runtime.seed` override.
    #:
    #: `runtime.seed` IS inside `run/spec.py::_scientific_payload(include_seed=
    #: True)` and outside `group_id`, so two designs differing only here produce
    #: distinct `run_id`s in distinct directories under one shared `group_id` --
    #: which is how the run system already spells "one experiment, N datapoints".
    seed: Union[int, str] = _BASELINE
    #: Names the replicate set this design belongs to, or None.
    #:
    #: A `reference_group` is a PROMISE, checked by `load_designs`: every member
    #: is identical in every scientific factor and differs only in `seed`. That
    #: is what makes the spread of their objectives an estimate of the study's
    #: own reproducibility noise rather than of some factor nobody controlled --
    #: and a group that quietly varied a factor would inflate the noise floor and
    #: therefore label real effects unresolvable, which is the most damaging
    #: thing this mechanism could do.
    reference_group: Optional[str] = None


_FIELDS = {f.name for f in dataclasses.fields(Design)}

#: Fields that may differ between members of one `reference_group`. Everything
#: else must be identical, or the group is not a controlled replicate set.
_REPLICATE_FREE_FIELDS = frozenset({"name", "seed", "reference_group"})


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
        entry = _normalise_seed(path, position, dict(entry))
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
    _check_reference_groups(path, designs)
    return designs


def _normalise_seed(path, position: int, entry: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce one entry's `seed` to an int or `"baseline"`. Raise on anything else.

    Two shapes YAML makes easy and both were wrong further downstream:

      * `seed:` with nothing after it parses as `None`. `enqueue_anchors` treated
        that as "inherit", `resolved_seed` did `int(None)`, and the result was a
        bare `TypeError` from inside `check_replicates_resolve` instead of one of
        this module's messages. Normalised to `"baseline"` here, which is what the
        author of a bare `seed:` meant.
      * `seed: 43.5` reached `int()` and TRUNCATED to 43 -- silently, and possibly
        onto a sibling's seed, which is the collision `check_replicates_resolve`
        exists to catch and would then have caught for an unintelligible reason.
        Refused, the same way `power_K` refuses a non-integral matrix power.
    """
    if "seed" not in entry:
        return entry
    seed = entry["seed"]
    if seed is None:
        entry["seed"] = _BASELINE
        return entry
    if seed == _BASELINE:
        return entry
    if isinstance(seed, bool) or not isinstance(seed, (int, float, str)):
        raise ValueError(
            f"{path}: design #{position} has seed={seed!r}; a training seed must "
            f"be a whole number, or 'baseline' to inherit the base spec's own")
    try:
        as_float = float(seed)
    except (TypeError, ValueError):
        raise ValueError(
            f"{path}: design #{position} has seed={seed!r}, which is not a "
            f"number. Use an integer, or 'baseline' to inherit the base spec's "
            f"own `runtime.seed`.") from None
    if as_float != int(as_float):
        raise ValueError(
            f"{path}: design #{position} has seed={seed!r}; a seed is an INDEX "
            f"into a random stream and must be a whole number. int({seed!r}) "
            f"would silently truncate it -- possibly onto a sibling's seed, "
            f"which would collapse a replicate pair into one datapoint.")
    entry["seed"] = int(as_float)
    return entry


def _check_reference_groups(path, designs: List[Design]) -> None:
    """A `reference_group` must be a controlled replicate set. Enforce it.

    Three requirements, and the reason each one is a hard error rather than a
    warning is the same: the group's spread becomes the study's NOISE FLOOR, and
    every reported effect is compared against it. A wrong noise floor does not
    look wrong -- it produces a complete, plausible analysis in which real
    effects are labelled unresolvable (if inflated) or noise is labelled an
    effect (if deflated).

      * **At least two members.** One observation has no spread. A group of one
        is a declaration wearing a measurement's clothes.
      * **Distinct seeds.** Two members at one seed are one datapoint recorded
        twice, and `deterministic: true` on the proxy spec means they would
        report a spread of ~0 -- i.e. claim perfect reproducibility from one run.
      * **Identical in every other factor.** This is the one that matters most.
        A group whose members differ in rank measures the RANK effect and reports
        it as noise.
    """
    groups: Dict[str, List[Design]] = {}
    for design in designs:
        if design.reference_group is None:
            continue
        groups.setdefault(str(design.reference_group), []).append(design)

    for group, members in sorted(groups.items()):
        if len(members) < 2:
            raise ValueError(
                f"{path}: reference_group {group!r} has {len(members)} member(s); "
                f"a replicate set needs at least two, because its whole purpose "
                f"is to produce a SPREAD and one observation has none. Add "
                f"another design with the same factors and a different seed, or "
                f"drop the reference_group key.")
        seeds = [d.seed for d in members]
        if len(set(map(repr, seeds))) != len(seeds):
            raise ValueError(
                f"{path}: reference_group {group!r} has two members at the same "
                f"seed ({sorted(map(repr, seeds))}). That is one datapoint "
                f"recorded twice, not two datapoints: the proxy base sets "
                f"`deterministic: true`, so both would land on the same loss and "
                f"the group would report a spread of ~0 -- a claim of perfect "
                f"reproducibility drawn from a single run. Give each member its "
                f"own seed.")
        reference = members[0]
        for member in members[1:]:
            differing = sorted(
                field for field in sorted(_FIELDS - _REPLICATE_FREE_FIELDS)
                if getattr(member, field) != getattr(reference, field))
            if differing:
                raise ValueError(
                    f"{path}: reference_group {group!r} is not a controlled "
                    f"replicate set -- {member.name!r} differs from "
                    f"{reference.name!r} in {differing}. A reference group's "
                    f"spread becomes the study's NOISE FLOOR, against which "
                    f"every reported effect is judged; a group that also varies "
                    f"{differing} measures that factor's effect and reports it "
                    f"as noise, which would label real effects unresolvable. "
                    f"Members may differ in `seed` and nothing else.")


def reference_groups(designs) -> Dict[str, List[Design]]:
    """`{group name: members}` for every design that names a `reference_group`."""
    groups: Dict[str, List[Design]] = {}
    for design in designs:
        if getattr(design, "reference_group", None) is None:
            continue
        groups.setdefault(str(design.reference_group), []).append(design)
    return groups


def resolved_seed(design: Design, base_seed: int) -> int:
    """The `runtime.seed` this design will actually train at."""
    if getattr(design, "seed", _BASELINE) == _BASELINE:
        return int(base_seed)
    return int(design.seed)


def check_replicates_resolve(designs, base_seed: int) -> None:
    """Do the members of each `reference_group` reach DISTINCT training seeds?

    `load_designs` cannot answer this and says so: it compares the declared
    `seed` fields, and `"baseline"` is not comparable to `42` without knowing the
    base spec. So a group containing `reference-k1` (inheriting 42) alongside a
    design that wrote `seed: 42` explicitly passes the load check and then trains
    the same model twice -- two identical runs, one `run_id`, one datapoint
    recorded as two, and a noise floor of exactly zero.

    Pure arithmetic over committed files, so `__main__` calls it during
    `--dry_run`. The alternative -- discovering it from the analysis after 256
    GPU-trials -- is not a discovery, because a spread of zero looks like an
    excellent result.
    """
    for group, members in sorted(reference_groups(designs).items()):
        seeds: Dict[int, List[str]] = {}
        for design in members:
            seeds.setdefault(resolved_seed(design, base_seed),
                             []).append(design.name)
        collisions = {seed: names for seed, names in seeds.items()
                      if len(names) > 1}
        if collisions:
            detail = "; ".join(f"seed {seed} <- {sorted(names)}"
                               for seed, names in sorted(collisions.items()))
            raise ValueError(
                f"reference_group {group!r} does not reach distinct training "
                f"seeds against this base spec (base runtime.seed="
                f"{int(base_seed)}): {detail}. Two members at one seed are ONE "
                f"datapoint recorded twice -- identical params and identical "
                f"seed hash to one `run_id`, so the pair contributes a spread of "
                f"exactly zero to the study's noise floor, which reads as "
                f"perfect reproducibility rather than as a mistake. A design "
                f"that omits `seed` inherits the base spec's, so it must not "
                f"also be written out explicitly by a sibling.")


def _baseline_norm_clip_multiplier(base_model: KoopmanLMConfig) -> float:
    clip_c = base_model.ska_norm_clip_c
    if clip_c is None:
        clip_c = math.sqrt(base_model.ska_rank)
    return float(clip_c) / math.sqrt(base_model.ska_rank)


#: Which shape `resolve_design` reads off each axis. It reads `choices` for the
#: categoricals and `low`/`high` for the floats, so a study that changes an
#: axis's KIND in `search_axes` breaks it -- and discretising a ridge into a
#: 3-level grid is an entirely plausible thing for an interaction study to want.
#:
#: Declared as data so the failure can name the axis and the shape it needed.
#: It used to be a bare `KeyError: 'low'` from inside a dict subscript, raised
#: only when `enqueue_anchors` ran -- which is past `--dry_run`'s return, so the
#: check that exists to catch this could not see it.
_AXIS_SHAPE = {
    "ska_rank": "choices", "n_ska_layers": "choices", "placement": "choices",
    "norm_clip_multiplier": "choices", "gamma_value": "choices",
    "ska_power_K": "choices", "weight_decay": "choices",
    "warmup_ratio": "choices", "grad_clip": "choices",
    "ska_ridge": "bounds", "ska_layerscale_init": "bounds",
    "learning_rate": "bounds",
}


def check_space_is_resolvable(space: Mapping[str, Any]) -> None:
    """Can `resolve_design` read every axis it needs? Raise naming what is wrong.

    Pure and cheap, so `__main__` calls it during `--dry_run`. The alternative --
    discovering it from `enqueue_anchors` -- happens after the git gate and after
    `create_study`, i.e. on the cluster, which is precisely what the dry run
    exists to prevent.
    """
    for axis, shape in sorted(_AXIS_SHAPE.items()):
        declaration = space.get(axis)
        if declaration is None:
            raise ValueError(
                f"the search space declares no {axis!r}, which anchor resolution "
                f"requires. Restriction can narrow or pin an axis; it cannot "
                f"remove one.")
        needed = ("choices",) if shape == "choices" else ("low", "high")
        missing = [k for k in needed if k not in declaration]
        if missing:
            kind = declaration.get("kind", "?")
            wanted = ("a categorical (choices)" if shape == "choices"
                      else "a float range (low/high)")
            raise ValueError(
                f"anchor resolution reads {list(needed)} off {axis!r}, but this "
                f"study declares it as kind={kind!r} (missing {missing}). "
                f"`resolve_design` expects {wanted} for this axis: a categorical "
                f"snaps to its nearest choice and a float clamps into its "
                f"bounds, and those are different operations, so changing an "
                f"axis's kind in `search_axes` changes which one applies. Either "
                f"keep {axis!r} as {wanted}, or drop `design_file` for this "
                f"study.")


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
    check_space_is_resolvable(space)
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
