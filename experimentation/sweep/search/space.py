"""The searchable space, declared exactly once, plus the map from a sampled
point to RunSpec overrides.

`sweep/spec.py` requires a sweep's grid to be declared in one place. This is that
rule one level up: the bounds and choices live here and nowhere else reads them.

Two design commitments.

**Distributions are plain data.** `search_space()` returns dicts like
``{"kind": "categorical", "choices": [...]}`` and ``{"kind": "float", "low": ...,
"high": ..., "log": True}``. `study.py` translates them into optuna
distributions. So the entire space stays declarable, inspectable and testable
with no optuna installed, and replacing the sampler later touches one module.

**`params_to_overrides` speaks `"<section>.<field>"`**, the vocabulary
`sweep/spec.py::build_cell_run_spec` already accepts. That one choice is why a
trial inherits every guarantee the run system provides -- a content-hashed
`run_id`, the dirty-tree gate, `verify_shard`, atomic materialization, the Slurm
array -- rather than reimplementing them. Three sampled parameters are not 1:1
config fields, and deriving them is this module's real work:

    n_ska_layers + placement  ->  model.ska_layer_indices
    norm_clip_multiplier      ->  model.ska_norm_clip_c   (x sqrt(rank))
    warmup_ratio              ->  optim.warmup_steps      (x max_steps)

**Baseline containment.** `search_space()` folds the base config's own values
into the choices -- notably `ska_norm_clip_c / sqrt(ska_rank)`, without which
rank 24 could never reproduce `configs/50m.yaml`'s `ska_norm_clip_c: 4.0`. A
space that cannot express the config you already run cannot tell you whether you
improved on it.

**The eta/gamma policy is inherited, not pinned.** It used to be pinned: eight
policy fields were written into every trial's overrides, because the
`KoopmanLMConfig` field defaults still encoded the *superseded* regime
(learnable eta and gamma, clamped) and five tier-2 configs picked that up by
omission. A search that inherited whatever its base happened to imply would
have been comparing across two parameterisations. But the fix for that belongs
in the defaults, not here -- pinning eight fields inside a parameter-mapping
function is config composition in the wrong place, and it silently overrode any
base config that deliberately chose otherwise. `KoopmanLMConfig` now defaults to
the modern policy and the five legacy configs state the old one explicitly, so
a trial simply inherits its base config and this module maps only what the
search actually samples or derives.

**Which route computes SKA, and what it costs.** ``SKAModule`` dispatches on
three independent booleans in an ``if/elif`` chain -- ``ska_prefix_scan``,
``ska_inverse_cholesky``, ``ska_exact_intrachunk`` -- and the *approximate*
route is what you get when none of them is set. So "exact" is the negation of a
disjunction of implementation flags and has no field of its own. That is what a
`backend_policy` exists to hide, and the policy list is short because the four
routes were measured against ``prefix_scan.dense_exact_oracle`` in fp64 on an
H100 (jobs **440122** correctness and **440135** cost):

===================  =====================  ==============================
route                fp64 err vs oracle     full-model step vs chunked
===================  =====================  ==============================
chunked              0.92 - 1.52            1.00x  (the baseline)
prefix_scan ref      <= 4.7e-13             160x at 4m
prefix_scan fused    fp32 only, 1e-5        1.01x, rank 24 + width 64 ONLY
inverse_cholesky     <= 5.1e-13             0.92x at 4m, 1.41x at 50m
exact_intrachunk     <= 5.0e-13             57x at 4m, 87x at 50m
===================  =====================  ==============================

Three consequences, each of which shaped the list above.

*``exact_invchol`` is the default because exactness is free.* Routes 2, 3 and 4
are the same function to 1e-13; route 3 is the one that costs what the
approximation costs. Nothing has to be traded.

*``exact_auto`` is a cliff, not a slope.* ``cuda_prefix_scan.is_supported``
requires rank **exactly** 24, and this space samples ``{8, 16, 24, 32}``. So
inside a single study ``exact_auto`` measured 0.0043 s at rank 24 and 0.72-3.19 s
at the other three -- a **167x-738x discontinuity correlated with a searched
variable**. A sampler comparing trials on wall-clock is being told rank 24 is
free. It is kept as an explicit slow-reference option, not as a default.

*Do not mix routes inside one study.* ``chunk_stats`` and ``exact_stats`` add a
``1e-4 * I`` jitter on top of ``ska_ridge``; ``ska_prefix_scan`` does not. Since
``ska_ridge`` is *sampled* over [3e-3, 3e-2], switching route between trials
perturbs the effective ridge by 0.3%-10% -- measured as 0.13%-2.9% in the output
and 0.25%-6% in the gradients. It is not a bug in either route, and it is exactly
why a policy resolves to ONE route for the whole study rather than picking per
trial.

**Backend hazard.** Two of the three policies here set
``ska_prefix_scan=True``, and per `run/train_argv.py` that architecture does not
tolerate ``torch.compile`` or gradient checkpointing: cudagraph capture plus
activation-checkpoint recomputation re-entering ``_SKAPrefixScanFn`` raises
``cudaErrorStreamCaptureInvalidated`` -- reproduced on H100 in job 415208, and
identically with ``ska_backend='pytorch'``, so it is not the CUDA kernel. The run
layer already passes ``--no_compile --no_gradient_checkpointing``
unconditionally; a searcher must not undo that to escape an OOM. Shrink the
per-device batch instead.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional, Sequence

from koopman_lm.config import KoopmanLMConfig
from experimentation.sweep.search.geometry import (
    PLACEMENTS, layer_count_choices, make_layer_indices)

__all__ = ["BACKEND_POLICIES", "REQUIRED_PARAMS", "base_reference_point",
           "default_base_lr", "dropped_base_values", "restrict_space",
           "search_space", "params_to_overrides"]

# Ordered by what a study should reach for first. See the module docstring for
# the measurement (jobs 440122 / 440135) that put inverse_cholesky at the front.
BACKEND_POLICIES = ("exact_invchol", "fused_only", "exact_auto")

# Policies that existed, were measured, and lost. Kept as named rejections
# rather than deleted, because the caller most likely to pass one is someone
# re-running an archived config -- and "unknown backend policy 'proxy_chunked'"
# reads as a rename when it was a retraction.
_RETIRED_POLICIES = {
    "proxy_chunked": (
        "backend policy 'proxy_chunked' is retired: it was never a cheap "
        "screen. Measured against prefix_scan.dense_exact_oracle in fp64 (job "
        "440122, H100) the chunked route is 92%-152% wrong in the FORWARD and "
        "93%-101% wrong in the GRADIENTS, at both the 4m and 50m geometries, "
        "on random inputs and on a lag-3 recall task, with no dependence on "
        "sequence length, ridge or power_K. It is not an approximation of the "
        "exact operator, it is a different operator -- and ska_rank, ska_ridge "
        "and ska_norm_clip_c reach the model ONLY through it, so it cannot "
        "screen the parameters this space samples. Its one remaining argument "
        "was speed, and that is gone too: at the 4m geometry 'exact_invchol' "
        "measured 0.0224 s per full-model micro-step against chunked's 0.0244 s "
        "(job 440135). Use 'exact_invchol'."),
}

# The fused SM100 prefix-scan kernel is specialised to this rank (and value
# width 64); see configs/50m.yaml's header.
_FUSED_RANK = 24
# Kept in step with cuda_prefix_scan._VALUE by test, not by hope: the fused
# kernel is compiled for exactly this value width and silently unreachable at
# any other.
_FUSED_VALUE = 64

DEFAULT_RANKS = (8, 16, 24, 32)
DEFAULT_WEIGHT_DECAYS = (0.05, 0.10, 0.15)
DEFAULT_WARMUP_RATIOS = (0.02, 0.04, 0.06)
DEFAULT_GRAD_CLIPS = (0.5, 1.0)
DEFAULT_GAMMAS = (0.90, 1.00, 1.05)
DEFAULT_POWER_KS = (1, 2)
DEFAULT_NORM_CLIP_MULTIPLIERS = (0.75, 1.00, 1.25)

RIDGE_BOUNDS = (3e-3, 3e-2)
LAYERSCALE_BOUNDS = (2e-3, 3e-2)
LR_FACTOR_BOUNDS = (0.65, 1.35)


def default_base_lr(cfg: KoopmanLMConfig, name: str = "") -> float:
    """The learning rate the search centres its range on.

    Prefers the production recipes by name -- configs/runs/50m-fineweb-3b.yaml
    uses 4e-4 -- and otherwise falls back to width, since a wider model wants a
    smaller rate.
    """
    lowered = name.lower()
    if "50m" in lowered:
        return 4e-4
    if "180m" in lowered:
        return 3e-4
    if cfg.d_model <= 384:
        return 4e-4
    if cfg.d_model <= 768:
        return 3e-4
    return 2.5e-4


def _with_value(choices: Sequence[float], value: float) -> list[float]:
    """`choices` plus `value`, deduplicated -- baseline containment."""
    merged = {float(c) for c in choices}
    merged.add(float(value))
    return sorted(merged)


def _with_value_int(choices: Sequence[int], value: int) -> list[int]:
    """`_with_value` for parameters that index something rather than scale it.

    Separate from `_with_value` because that one coerces to float, which is
    right for a ridge and wrong for a matrix power: optuna records the sampled
    value verbatim, so choices of [1.0, 2.0] put floats in the journal and make
    1.0 a different category from a hand-written 1 in an anchor design. Both
    int-valued parameters here (ska_rank, ska_power_K) use this.
    """
    return sorted({*(int(c) for c in choices), int(value)})


def _bounds_containing(bounds: tuple[float, float], value: float) -> tuple[float, float]:
    return (min(bounds[0], float(value)), max(bounds[1], float(value)))


def search_space(base_model: KoopmanLMConfig, *,
                 base_name: str = "",
                 ranks: Sequence[int] = DEFAULT_RANKS,
                 placements: Sequence[str] = PLACEMENTS,
                 base_lr: float | None = None) -> Dict[str, Dict[str, Any]]:
    """The declared space, as plain data. `study.py` turns it into optuna
    distributions; nothing else needs to know the bounds."""
    base_multiplier = base_model.ska_norm_clip_c
    if base_multiplier is None:
        base_multiplier = math.sqrt(base_model.ska_rank)
    base_multiplier = float(base_multiplier) / math.sqrt(base_model.ska_rank)

    lr = base_lr if base_lr is not None else default_base_lr(base_model, base_name)

    return {
        "ska_rank": {"kind": "categorical",
                     "choices": _with_value_int(ranks, base_model.ska_rank)},
        "n_ska_layers": {"kind": "categorical",
                         "choices": layer_count_choices(
                             len(base_model.ska_layer_indices), base_model.n_layers)},
        "placement": {"kind": "categorical", "choices": list(placements)},
        "ska_ridge": {"kind": "float", "log": True,
                      **dict(zip(("low", "high"),
                                 _bounds_containing(RIDGE_BOUNDS, base_model.ska_ridge)))},
        "ska_layerscale_init": {"kind": "float", "log": True,
                                **dict(zip(("low", "high"),
                                           _bounds_containing(LAYERSCALE_BOUNDS,
                                                              base_model.ska_layerscale_init)))},
        "norm_clip_multiplier": {"kind": "categorical",
                                 "choices": _with_value(DEFAULT_NORM_CLIP_MULTIPLIERS,
                                                        base_multiplier)},
        "gamma_value": {"kind": "categorical",
                        "choices": _with_value(DEFAULT_GAMMAS, base_model.ska_gamma_value)},
        # Searched rather than pinned. It used to be pinned to 1 while
        # configs/runs/4m-golden.yaml pins 2, so a study on that base compared
        # every trial at K=1 against a baseline at K=2 -- internally consistent
        # trials, but "we beat the baseline" confounded on one axis. Searching it
        # removes the confound rather than picking a side, and it is nearly free:
        # job 440135 measured route 3 at 0.00542 s (K=1) vs 0.00559 s (K=2).
        #
        # Choices are the two values any config in this repo actually uses, plus
        # whatever the base pins (baseline containment). Not widened further: K
        # is a matrix power, so K=3+ is both untested here and monotonically
        # more work.
        "ska_power_K": {"kind": "categorical",
                        "choices": _with_value_int(DEFAULT_POWER_KS,
                                                  base_model.ska_power_K)},
        "learning_rate": {"kind": "float", "log": True,
                          "low": lr * LR_FACTOR_BOUNDS[0], "high": lr * LR_FACTOR_BOUNDS[1]},
        "weight_decay": {"kind": "categorical", "choices": list(DEFAULT_WEIGHT_DECAYS)},
        "warmup_ratio": {"kind": "categorical", "choices": list(DEFAULT_WARMUP_RATIOS)},
        "grad_clip": {"kind": "categorical", "choices": list(DEFAULT_GRAD_CLIPS)},
    }


# --------------------------------------------------------------------------
# Per-study restriction of the declared space.
# --------------------------------------------------------------------------
#
# `search_space()` is the space the REPO declares; a study may narrow it. Two
# operations, and only two:
#
#   search_axes   replace one axis's declaration outright
#   fixed_params  pin one axis to a single value
#
# There is deliberately no third. A generic "overrides" dict was the obvious
# design and is the wrong one: `params_to_overrides` requires twelve named
# parameters and `KoopmanLMConfig` asserts on four of them, so an unvalidated
# passthrough turns a one-line typo in a YAML file into a failed trial on a GPU,
# after a run directory has been claimed. Every value that arrives here is
# checked against the axis's real domain before it can reach a sampler.
#
# **Replacement, not intersection.** An entry in `search_axes` discards the base
# declaration including the baseline containment `search_space()` folds in. That
# is a decision, not an oversight: an interaction study over
# ridge in [3e-3, 3e-2] must be able to say exactly that, and a study whose base
# config sits outside its own declared range is a legitimate thing to want (the
# base becomes a reference point outside the search rather than a member of it).
# What is NOT legitimate is doing it by accident, so `restrict_space` reports
# every axis whose base value it dropped and `__main__` prints that in the plan.

#: Every parameter `params_to_overrides` reads. Named so a restriction can be
#: checked for completeness rather than discovered incomplete by a KeyError on
#: trial 0. `ska_power_K` is read with `.get` (report.py replays archived
#: journals that predate it) but is still required of a live space.
REQUIRED_PARAMS = (
    "ska_rank", "n_ska_layers", "placement", "ska_ridge", "ska_layerscale_init",
    "norm_clip_multiplier", "gamma_value", "ska_power_K", "learning_rate",
    "weight_decay", "warmup_ratio", "grad_clip",
)

#: Axes whose values index or count something. Coerced to `int`, for the reason
#: `_with_value_int` gives: optuna records the sampled value verbatim, so a
#: choice of 1.0 is a different category from a hand-written 1 in an anchor.
_INT_AXES = frozenset({"ska_rank", "n_ska_layers", "ska_power_K"})
#: Axes whose values name something. Left as strings.
_STR_AXES = frozenset({"placement"})


def _coerce(axis: str, value: Any) -> Any:
    # `bool` first, and before `_STR_AXES`, because Python makes it invisible
    # otherwise: `bool` IS an `int`, so `float(True)` is 1.0 and `int(1.0)` is 1.
    # A study writing `ska_power_K: true` would have been silently accepted as
    # K=1, and `str(True)` is "True", which is not a placement. YAML makes this
    # reachable rather than theoretical -- `yes`, `on` and `true` all parse to
    # True. `StudySpec` also rejects a bool in `fixed_params`, but this function
    # is the public entry point for a caller who did not come through a StudySpec,
    # and two layers that disagree about what is legal is how the weaker one
    # becomes the real contract.
    if isinstance(value, bool):
        raise ValueError(
            f"{axis}={value!r} is a boolean. YAML parses `true`, `yes` and `on` "
            f"that way, and Python's bool is an int -- so this would silently "
            f"become {int(value)} rather than being rejected. Write the number.")
    if axis in _STR_AXES:
        return str(value)
    if axis in _INT_AXES:
        as_float = float(value)
        if as_float != int(as_float):
            raise ValueError(
                f"{axis}={value!r} must be a whole number -- it counts or indexes "
                f"something, and int({value!r}) would silently round it")
        return int(as_float)
    return float(value)


def _check_rank(value: int, base_model: KoopmanLMConfig) -> None:
    if value < 8 or value % 8 != 0:
        raise ValueError(
            f"ska_rank={value} must be a positive multiple of 8 -- "
            f"KoopmanLMConfig asserts it, so this would fail on the GPU after a "
            f"run directory was materialized")


def _check_layer_count(value: int, base_model: KoopmanLMConfig) -> None:
    capacity = max(1, base_model.n_layers - 2)
    if value < 1:
        raise ValueError(f"n_ska_layers={value} must be >= 1")
    if value > capacity:
        raise ValueError(
            f"n_ska_layers={value} exceeds the usable window of a "
            f"{base_model.n_layers}-layer backbone, which is {capacity} "
            f"(geometry.py keeps layer 0 and the final layer free). "
            f"make_layer_indices CLAMPS rather than raising, so this would not "
            f"fail -- it would silently collapse onto the same indices as "
            f"n_ska_layers={capacity} and give the study a DEAD AXIS: two "
            f"choices, one resolved config, and a parameter importance computed "
            f"over a difference that does not exist.")


def _check_placement(value: str, base_model: KoopmanLMConfig) -> None:
    if value not in PLACEMENTS:
        raise ValueError(
            f"placement={value!r}; geometry.py declares {list(PLACEMENTS)}")


def _positive(name: str):
    def check(value: float, base_model: KoopmanLMConfig) -> None:
        if not float(value) > 0.0:
            raise ValueError(f"{name}={value} must be > 0")
    return check


def _check_warmup_ratio(value: float, base_model: KoopmanLMConfig) -> None:
    if not 0.0 <= float(value) <= 1.0:
        raise ValueError(
            f"warmup_ratio={value} must lie in [0, 1] -- it is multiplied by "
            f"max_steps to get optim.warmup_steps, so a ratio above 1 warms up "
            f"for the whole run and never decays")


def _check_weight_decay(value: float, base_model: KoopmanLMConfig) -> None:
    if float(value) < 0.0:
        raise ValueError(f"weight_decay={value} must be >= 0")


#: One domain check per axis. Every axis in `search_space()` has an entry, and
#: `test_search_axis_restriction.py` asserts the two sets are equal -- so adding
#: an axis without a validator fails a test rather than creating a hole.
_AXIS_DOMAINS = {
    "ska_rank": _check_rank,
    "n_ska_layers": _check_layer_count,
    "placement": _check_placement,
    "ska_ridge": _positive("ska_ridge"),
    "ska_layerscale_init": _positive("ska_layerscale_init"),
    "norm_clip_multiplier": _positive("norm_clip_multiplier"),
    "gamma_value": _positive("gamma_value"),
    "ska_power_K": _positive("ska_power_K"),
    "learning_rate": _positive("learning_rate"),
    "weight_decay": _check_weight_decay,
    "warmup_ratio": _check_warmup_ratio,
    "grad_clip": _positive("grad_clip"),
}


def _validated_declaration(axis: str, declaration: Mapping[str, Any],
                           base_model: KoopmanLMConfig) -> Dict[str, Any]:
    """One replacement declaration -> the same declaration, coerced and checked.

    `StudySpec` has already checked the SHAPE (kind, required keys, low < high).
    What it could not check is the DOMAIN, because that needs the base model --
    the layer-index capacity and `d_model % ska_n_heads` live there. So the two
    validations are split along the line of what each module can see, and this
    half is the one that would otherwise surface on a GPU.
    """
    check = _AXIS_DOMAINS[axis]
    kind = declaration["kind"]
    if kind == "categorical":
        choices = [_coerce(axis, c) for c in declaration["choices"]]
        for choice in choices:
            check(choice, base_model)
        if len(set(choices)) != len(choices):
            raise ValueError(
                f"search_axes[{axis!r}]: choices {choices} contain duplicates "
                f"after coercion to the axis's own type -- e.g. 1 and 1.0 are "
                f"one category once coerced, and a duplicate doubles that "
                f"value's prior weight")
        return {"kind": "categorical", "choices": choices}

    low, high = float(declaration["low"]), float(declaration["high"])
    if low >= high:
        # `StudySpec` rejects this too, with a longer message pointing at
        # `fixed_params`. Repeated here for the same reason `_coerce` rejects
        # bools: this is the entry point for a caller who did not come through a
        # StudySpec, and low == high would otherwise become a degenerate
        # FloatDistribution -- legal to optuna, `single()`, and indistinguishable
        # in the journal from an axis the sampler simply never varied.
        raise ValueError(
            f"search_axes[{axis!r}]: low={low} must be strictly less than "
            f"high={high}. low == high is a fixed value, not a range -- put it "
            f"in fixed_params, where it becomes a validated singleton "
            f"categorical instead of a degenerate interval.")
    check(low, base_model)
    check(high, base_model)
    if axis in _INT_AXES or axis in _STR_AXES:
        raise ValueError(
            f"search_axes[{axis!r}] declares a 'float' range, but {axis} counts "
            f"or names something and is categorical everywhere else. A "
            f"FloatDistribution here would put non-integral values in the "
            f"journal that no anchor could ever match.")
    out: Dict[str, Any] = {"kind": "float", "low": low, "high": high}
    if declaration.get("log"):
        out["log"] = True
    return out


def _contains(declaration: Mapping[str, Any], value: Any) -> bool:
    if declaration["kind"] == "categorical":
        return any(choice == value for choice in declaration["choices"])
    return float(declaration["low"]) <= float(value) <= float(declaration["high"])


def _describe(declaration: Mapping[str, Any]) -> str:
    if declaration["kind"] == "categorical":
        return f"choices {list(declaration['choices'])}"
    scale = "log" if declaration.get("log") else "linear"
    return f"{scale} range [{declaration['low']}, {declaration['high']}]"


def restrict_space(space: Mapping[str, Mapping[str, Any]],
                   base_model: KoopmanLMConfig, *,
                   axes: Optional[Mapping[str, Mapping[str, Any]]] = None,
                   fixed: Optional[Mapping[str, Any]] = None,
                   ) -> Dict[str, Dict[str, Any]]:
    """`search_space()`'s output, narrowed by a study's own declarations.

    Returns a NEW space; `space` is not mutated. With both arguments empty the
    result is `dict(space)` -- byte-for-byte the behaviour that existed before
    this function, which is what lets every committed study keep its current
    space by saying nothing.

    Raises on: an axis this space does not declare, a declaration whose values
    fall outside the axis's real domain, a fixed value outside the declaration
    it is being fixed within, and a restriction that leaves
    `params_to_overrides` without a parameter it requires.
    """
    axes = dict(axes or {})
    fixed = dict(fixed or {})
    restricted: Dict[str, Dict[str, Any]] = {
        name: dict(decl) for name, decl in space.items()}

    unknown = sorted((set(axes) | set(fixed)) - set(restricted))
    if unknown:
        raise ValueError(
            f"unknown search axis/axes {unknown}. This study declares "
            f"{sorted(restricted)}. Axis names are the SAMPLED parameter names "
            f"from space.search_space(), not RunSpec field paths -- "
            f"'model.ska_rank' is a field override and 'ska_rank' is an axis.")

    for name, declaration in sorted(axes.items()):
        restricted[name] = _validated_declaration(name, declaration, base_model)

    for name, value in sorted(fixed.items()):
        coerced = _coerce(name, value)
        _AXIS_DOMAINS[name](coerced, base_model)
        if not _contains(restricted[name], coerced):
            raise ValueError(
                f"fixed_params[{name!r}]={value!r} lies outside this axis's "
                f"declaration ({_describe(restricted[name])}). Fixing is meant "
                f"to REMOVE a degree of freedom, not to smuggle in a value the "
                f"space says is out of range -- if the value is what you want, "
                f"widen the axis in search_axes and say so where a reader can "
                f"see it.")
        # A validated SINGLETON, not a deletion. See StudySpec.fixed_params for
        # the three properties that depend on the axis still being in the space.
        restricted[name] = {"kind": "categorical", "choices": [coerced]}

    missing = [name for name in REQUIRED_PARAMS if name not in restricted]
    if missing:
        raise ValueError(
            f"the restricted space is missing {missing}, which "
            f"params_to_overrides requires. Restriction can narrow an axis or "
            f"pin it; it cannot delete one, because every trial still has to "
            f"resolve to a complete RunSpec.")
    return restricted


def base_reference_point(base_model: KoopmanLMConfig, *,
                         base_lr: Optional[float] = None,
                         base_optim: Optional[Mapping[str, Any]] = None
                         ) -> Dict[str, Any]:
    """The base config expressed in the space's own vocabulary.

    This is what "baseline containment" is containment OF: the point a sampler
    would have to propose in order to reproduce the config the study is trying
    to beat. `search_space()` folds these values into its declarations for that
    reason; a restriction can drop them, and this is what makes the drop visible.

    `warmup_ratio` is deliberately absent. It is a ratio of the STUDY's
    `max_steps`, which is not the base spec's, so the base has no warmup_ratio to
    contain -- only a warmup_steps against a different run length.
    """
    optim = dict(base_optim or {})
    multiplier = base_model.ska_norm_clip_c
    if multiplier is None:
        multiplier = math.sqrt(base_model.ska_rank)
    point: Dict[str, Any] = {
        "ska_rank": int(base_model.ska_rank),
        "n_ska_layers": len(base_model.ska_layer_indices),
        "placement": "baseline",
        "ska_ridge": float(base_model.ska_ridge),
        "ska_layerscale_init": float(base_model.ska_layerscale_init),
        "norm_clip_multiplier": float(multiplier) / math.sqrt(base_model.ska_rank),
        "gamma_value": float(base_model.ska_gamma_value),
        "ska_power_K": int(base_model.ska_power_K),
    }
    lr = base_lr if base_lr is not None else optim.get("lr")
    if lr is not None:
        point["learning_rate"] = float(lr)
    for axis, key in (("weight_decay", "weight_decay"), ("grad_clip", "grad_clip")):
        if key in optim:
            point[axis] = float(optim[key])
    return point


def dropped_base_values(restricted: Mapping[str, Mapping[str, Any]],
                        base_model: KoopmanLMConfig, *,
                        base_lr: Optional[float] = None,
                        base_optim: Optional[Mapping[str, Any]] = None
                        ) -> Dict[str, str]:
    """Axes on which this space can no longer express the base config.

    Reported rather than rejected. Dropping the baseline is sometimes exactly
    right -- an interaction study may deliberately search only around, not
    including, its reference -- and it is always worth printing, because the
    alternative is a study that quietly cannot reproduce the config it is
    claiming to improve on.
    """
    lost: Dict[str, str] = {}
    for axis, value in base_reference_point(
            base_model, base_lr=base_lr, base_optim=base_optim).items():
        declaration = restricted.get(axis)
        if declaration is None or _contains(declaration, value):
            continue
        lost[axis] = (f"the base config's own {axis}={value!r} is outside "
                      f"{_describe(declaration)}")
    return lost


def _backend_overrides(policy: str, rank: int, *,
                       head_dim: Optional[int] = None) -> Dict[str, Any]:
    if policy in _RETIRED_POLICIES:
        raise ValueError(_RETIRED_POLICIES[policy])
    if policy not in BACKEND_POLICIES:
        raise ValueError(
            f"unknown backend policy {policy!r}; expected one of {list(BACKEND_POLICIES)}")
    if policy == "exact_invchol":
        # Exact, and free. Route 3 agrees with the reference prefix scan to
        # 5.1e-13 in fp64 and costs 0.92x the chunked approximation at 4m /
        # 1.41x at 50m per full-model micro-step.
        #
        # No `model.ska_backend` here on purpose. That string is read by exactly
        # one route: ska.py passes the RAW value to ska_prefix_scan and the
        # resolved one feeds only extra_repr(). Emitting it under a policy that
        # does not run the prefix scan would put a setting in spec.yaml that
        # looks like it selected something and did not.
        return {"model.ska_prefix_scan": False,
                "model.ska_inverse_cholesky": True,
                "model.ska_exact_intrachunk": False}
    if policy == "fused_only":
        # Rank AND value width. This checked rank alone, and value width is the
        # one that actually bites: cuda_prefix_scan.is_supported requires
        # vbar.shape[-1] == 64, and value width is d_model/ska_n_heads -- a
        # property of the BASE SPEC that no sampled rank can change. So
        # fused_only on a base spec with head_dim 32 passed this check and then
        # died at prefix_scan.py's RuntimeError on the first forward, on a GPU,
        # after materializing a run directory and queueing a job.
        problems = []
        if rank != _FUSED_RANK:
            problems.append(f"ska_rank={rank} (needs {_FUSED_RANK})")
        if head_dim is not None and head_dim != _FUSED_VALUE:
            problems.append(
                f"value width={head_dim} (needs {_FUSED_VALUE}; it is "
                f"d_model/ska_n_heads, so no sampled rank can fix it)")
        if problems:
            raise ValueError(
                "backend policy 'fused_only' pins the fused SM100 kernel, which "
                "is specialised to one geometry: " + "; ".join(problems)
                + ". Use 'exact_invchol', which is exact at ANY geometry and "
                  "measured 0.92x-1.41x the chunked cost; 'exact_auto' is also "
                  "exact here but falls back to the Python reference scan at "
                  "160x when the fused kernel does not apply.")
    return {"model.ska_prefix_scan": True,
            "model.ska_backend": "cuda_prefix" if policy == "fused_only" else "auto",
            "model.ska_inverse_cholesky": False,
            "model.ska_exact_intrachunk": False,
            "model.ska_prefix_scan_block_size": 32,
            "model.ska_prefix_scan_jitter": 0.0}


def params_to_overrides(params: Mapping[str, Any], base_model: KoopmanLMConfig, *,
                        max_steps: int,
                        seq_len: int | None = None,
                        backend_policy: str = "exact_invchol") -> Dict[str, Any]:
    """One sampled point -> `{"<section>.<field>": value}`, ready for
    `sweep/spec.py::build_cell_run_spec`.

    Pure: no filesystem, no optuna, no model construction. `max_steps` is needed
    because `warmup_ratio` is only meaningful against a run length, and the run
    length belongs to the base spec rather than to the space.
    """
    rank = int(params["ska_rank"])
    if rank % 8 != 0:
        raise ValueError(
            f"ska_rank must be a multiple of 8 (KoopmanLMConfig asserts it); got {rank}")

    indices = make_layer_indices(base_model.n_layers, int(params["n_ska_layers"]),
                                 str(params["placement"]),
                                 list(base_model.ska_layer_indices))

    overrides: Dict[str, Any] = {
        "model.ska_rank": rank,
        "model.ska_layer_indices": indices,
        "model.ska_ridge": float(params["ska_ridge"]),
        "model.ska_layerscale_init": float(params["ska_layerscale_init"]),
        "model.ska_norm_clip_c": round(
            math.sqrt(rank) * float(params["norm_clip_multiplier"]), 8),
        "model.ska_gamma_value": float(params["gamma_value"]),
        # `.get`, not `[...]`: report.py replays trial.params out of a journal,
        # and trials recorded before this became searchable have no such key.
        # They ran at the pinned value 1 -- so the fallback is 1 and NOT
        # base_model.ska_power_K, or promoting an archived study would confirm
        # a K the trial never ran at.
        "model.ska_power_K": int(params.get("ska_power_K", 1)),
        "optim.lr": float(params["learning_rate"]),
        "optim.weight_decay": float(params["weight_decay"]),
        "optim.grad_clip": float(params["grad_clip"]),
        "optim.warmup_steps": max(1, round(max_steps * float(params["warmup_ratio"]))),
    }
    if seq_len is not None:
        overrides["model.max_seq_len"] = int(seq_len)
    overrides.update(_backend_overrides(
        backend_policy, rank,
        head_dim=base_model.d_model // max(1, base_model.ska_n_heads)))
    return overrides
