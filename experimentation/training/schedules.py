"""
schedules.py -- §6 of the extension-mechanisms design: anything time-varying.

`model` describes the model at step 0; `schedules` describes how it moves.
Covers ridge-epsilon annealing, freeze-then-unfreeze, and the sequence-length
curriculum.

**The load-bearing rule (§6.2): a schedule is a pure function of global step,
with no internal state.** That single constraint means resume needs nothing
extra -- the step is already in resume.pt, so replaying a schedule is free and
exact, and the bit-exact resume test cannot be broken by one. It does rule out
anything adaptive ("reduce ridge when loss plateaus"); that is a deliberate
trade of adaptivity for exactness of resume.

Two module-level halves, deliberately split:

  * kind factories + `validate_schedules` are **pure and torch-free**, so
    koopman_lm.run.spec can validate a RunSpec at construction without
    importing torch;
  * `ScheduleApplier` needs a live model, and resolves the whitelist's module
    classes lazily on first use.
"""
from __future__ import annotations

import math
from fnmatch import fnmatchcase
from importlib import import_module
from typing import Any, Callable, Dict, Optional

# --------------------------------------------------------------------------
# The whitelist (§6.4). A whitelist, not reflection: the silent-failure mode
# is scheduling something the forward never re-reads -- you would see a
# beautiful annealing curve in wandb and the model would have ignored it
# entirely. Every entry here must name an attribute the forward reads at call
# time (SKAModule stores ridge_eps/power_K as plain Python numbers and passes
# them down as function arguments, including on the fused CUDA path, where
# ridge is a kernel argument rather than compiled in).
#
#   spec target -> (module class import path, attribute, cast)
# --------------------------------------------------------------------------
SCHEDULABLE = {
    "model.ska_ridge":   ("koopman_lm.modules.seq.ska:SKAModule", "ridge_eps", float),
    "model.ska_power_K": ("koopman_lm.modules.seq.ska:SKAModule", "power_K", int),
}

#: Targets handled by their own machinery rather than by setattr (§6.5).
FREEZE_TARGET = "freeze"
SEQ_LEN_TARGET = "data.seq_len"

#: Targets that are deliberately NOT schedulable, with the reason (§6.6).
_REFUSED = {
    "optim.lr": (
        "the existing LambdaLR owns the learning rate; two mechanisms fighting "
        "over param_group['lr'] is a bug generator. A custom LR shape belongs "
        "in OptimSpec.schedule, not here."),
    "model.ska_rank": (
        "dimensions are fixed at construction -- schedules only reach values "
        "the forward pass reads at runtime. Use a scalar-or-sequence config "
        "field (design §3)."),
    "model.d_model": (
        "dimensions are fixed at construction (design §6.6)."),
    "model.d_state": (
        "dimensions are fixed at construction (design §6.6)."),
    "model.ska_eta": (
        "eta/gamma are nn.Parameter when learnable and are read through "
        "resolver properties; scheduling them means writing into .data, which "
        "only makes sense when they are not learnable. Deferred (design §6.6)."),
    "model.ska_gamma": (
        "eta/gamma are nn.Parameter when learnable and are read through "
        "resolver properties; scheduling them means writing into .data, which "
        "only makes sense when they are not learnable. Deferred (design §6.6)."),
}


# --------------------------------------------------------------------------
# Kinds
# --------------------------------------------------------------------------

SCHEDULE_KINDS: Dict[str, Callable[[Dict[str, Any]], Callable[[int], float]]] = {}


def register_schedule(name: str):
    """Register a schedule kind. Anything exotic goes in this registry keyed
    by name, exactly like probes -- and promotes to a declarative kind if it
    earns it (§6.3)."""
    def _decorator(factory):
        SCHEDULE_KINDS[name] = factory
        return factory
    return _decorator


def _args(spec: Dict[str, Any], required, optional=()) -> Dict[str, Any]:
    """Pull `required`/`optional` keys out of a kind spec, rejecting anything
    else. An unknown key is an error, not a no-op: `until:` where the kind
    wants `over:` would otherwise silently run an unscheduled experiment."""
    kind = spec.get("kind")
    known = {"kind", *required, *optional}
    unknown = sorted(set(spec) - known)
    if unknown:
        raise ValueError(
            f"schedule kind {kind!r} got unknown key(s) {unknown}; "
            f"expected some of {sorted(known - {'kind'})}")
    missing = sorted(k for k in required if k not in spec)
    if missing:
        raise ValueError(
            f"schedule kind {kind!r} is missing required key(s) {missing}")
    return {k: spec[k] for k in spec if k != "kind"}


def _window(spec: Dict[str, Any]):
    over = spec["over"]
    if not isinstance(over, (list, tuple)) or len(over) != 2:
        raise ValueError(f"`over` must be a [start, end] pair, got {over!r}")
    start, end = int(over[0]), int(over[1])
    if end <= start:
        raise ValueError(
            f"`over` must be a forward interval with end > start, got "
            f"[{start}, {end}] -- a zero-length window has no interpolation "
            f"to do; use kind 'piecewise' for an instantaneous change.")
    return start, end


def _progress(step: int, start: int, end: int) -> float:
    """Fraction through the window, clamped to [0, 1]. A schedule is defined
    at every step, not only inside `over`."""
    return min(1.0, max(0.0, (step - start) / (end - start)))


@register_schedule("constant")
def _constant(spec):
    a = _args(spec, required=("value",))
    value = a["value"]
    return lambda step: value


@register_schedule("linear")
def _linear(spec):
    a = _args(spec, required=("from", "to", "over"))
    lo, hi = float(a["from"]), float(a["to"])
    start, end = _window(spec)
    return lambda step: lo + (hi - lo) * _progress(step, start, end)


@register_schedule("cosine")
def _cosine(spec):
    a = _args(spec, required=("from", "to", "over"))
    lo, hi = float(a["from"]), float(a["to"])
    start, end = _window(spec)

    def _at(step):
        # Half-period cosine: 1 at p=0, 0 at p=1, monotone in between.
        decay = 0.5 * (1.0 + math.cos(math.pi * _progress(step, start, end)))
        return hi + (lo - hi) * decay
    return _at


@register_schedule("piecewise")
def _piecewise(spec):
    """A staircase: `values[i]` holds from `at[i]` until `at[i+1]`.

    Deliberately NOT interpolated. The design's motivating example is a
    sequence-length curriculum (512 -> 1024 -> 2048), which must yield exactly
    those values -- an interpolated 768 at step 6000 is not a tensor shape
    anyone asked for. Use `linear`/`cosine` when you want a ramp.
    """
    a = _args(spec, required=("at", "values"))
    at = [int(x) for x in a["at"]]
    values = list(a["values"])
    if len(at) != len(values):
        raise ValueError(
            f"piecewise `at` has {len(at)} entries but `values` has "
            f"{len(values)}; they must be the same length")
    if not at:
        raise ValueError("piecewise `at` must be non-empty")
    if at[0] != 0:
        raise ValueError(
            f"piecewise `at` must start at 0 (the schedule has to be defined "
            f"from step 0), got {at[0]}")
    if any(b <= x for x, b in zip(at, at[1:])):
        raise ValueError(f"piecewise `at` must be strictly increasing, got {at}")

    def _at_step(step):
        out = values[0]
        for bp, v in zip(at, values):
            if step >= bp:
                out = v
            else:
                break
        return out
    return _at_step


@register_schedule("step")
def _step(spec):
    """Geometric decay: ``from * gamma ** (step // every)`` -- the classic
    StepLR shape, applied to a schedulable attribute rather than the LR."""
    a = _args(spec, required=("from", "gamma", "every"))
    base, gamma, every = float(a["from"]), float(a["gamma"]), int(a["every"])
    if every <= 0:
        raise ValueError(f"step `every` must be positive, got {every}")
    if gamma <= 0:
        raise ValueError(f"step `gamma` must be positive, got {gamma}")
    return lambda step: base * (gamma ** (step // every))


def make_schedule(spec: Dict[str, Any]) -> Callable[[int], float]:
    """Build a pure function of global step from a schedule spec dict."""
    if not isinstance(spec, dict):
        raise TypeError(f"a schedule spec must be a mapping, got {type(spec).__name__}")
    kind = spec.get("kind")
    if kind not in SCHEDULE_KINDS:
        raise ValueError(
            f"unknown schedule kind {kind!r}; known kinds are "
            f"{sorted(SCHEDULE_KINDS)}. Register an exotic one with "
            f"@register_schedule(name).")
    return SCHEDULE_KINDS[kind](spec)


# --------------------------------------------------------------------------
# Validation (torch-free, so the spec layer can call it at construction)
# --------------------------------------------------------------------------

def _validate_freeze(spec: Dict[str, Any]) -> None:
    if not isinstance(spec, dict):
        raise TypeError(f"`freeze` must be a mapping, got {type(spec).__name__}")
    unknown = sorted(set(spec) - {"match", "frozen_until"})
    if unknown:
        raise ValueError(f"`freeze` got unknown key(s) {unknown}")
    for key in ("match", "frozen_until"):
        if key not in spec:
            raise ValueError(f"`freeze` is missing required key {key!r}")
    if not spec["match"]:
        raise ValueError("`freeze.match` must be a non-empty fnmatch pattern")
    if int(spec["frozen_until"]) < 0:
        raise ValueError("`freeze.frozen_until` must be >= 0")


def validate_schedules(schedules: Optional[Dict[str, Any]]) -> None:
    """Check every target against the whitelist and every spec against its
    kind. Raises at *startup* rather than letting a typo become a silent
    no-op (§6.4)."""
    if not schedules:
        return
    if not isinstance(schedules, dict):
        raise TypeError(
            f"`schedules` must be a mapping of target -> spec, got "
            f"{type(schedules).__name__}")
    for target, spec in schedules.items():
        if target in _REFUSED:
            raise ValueError(
                f"schedule target {target!r} is deliberately not schedulable: "
                f"{_REFUSED[target]}")
        if target == FREEZE_TARGET:
            _validate_freeze(spec)
            continue
        if target == SEQ_LEN_TARGET:
            make_schedule(spec)
            continue
        if target not in SCHEDULABLE:
            raise ValueError(
                f"schedule target {target!r} is not in the SCHEDULABLE "
                f"whitelist. Known targets: "
                f"{sorted([*SCHEDULABLE, FREEZE_TARGET, SEQ_LEN_TARGET])}. "
                f"A whitelist is deliberate (design §6.4): scheduling a value "
                f"the forward never re-reads would give you a beautiful "
                f"annealing curve and a model that ignored it.")
        make_schedule(spec)


def _resolve(path: str):
    module_path, _, cls_name = path.partition(":")
    return getattr(import_module(module_path), cls_name)


# --------------------------------------------------------------------------
# The applier
# --------------------------------------------------------------------------

class ScheduleApplier:
    """Binds a schedules dict to a live model, then pushes values each step.

    One call site in the training loop, before the forward:
    ``sched_values = applier.apply(step)``. The returned dict is every
    scheduled value at that step -- log it, because without it you cannot
    distinguish "the schedule ran" from "the schedule was silently a no-op"
    (§6.4).
    """

    def __init__(self, model, schedules: Optional[Dict[str, Any]]):
        validate_schedules(schedules)
        self.bindings = []          # (target, fn, sites, attr, cast)
        self.freeze = None          # (fn_match, frozen_until, [params])
        self.seq_len = None         # callable(step) -> int
        for target, spec in (schedules or {}).items():
            if target == FREEZE_TARGET:
                self.freeze = self._bind_freeze(model, spec)
            elif target == SEQ_LEN_TARGET:
                fn = make_schedule(spec)
                self.seq_len = lambda step, _fn=fn: int(_fn(step))
            else:
                self.bindings.append(self._bind_attr(model, target, spec))

    @staticmethod
    def _bind_attr(model, target, spec):
        path, attr, cast = SCHEDULABLE[target]
        cls = _resolve(path)
        sites = [m for m in model.modules() if isinstance(m, cls)]
        if not sites:
            # An empty match must raise, not warn (§6.4).
            raise ValueError(
                f"schedule target {target!r} matched no modules of type "
                f"{cls.__name__} in this model -- the schedule would have been "
                f"a silent no-op.")
        return (target, make_schedule(spec), sites, attr, cast)

    @staticmethod
    def _bind_freeze(model, spec):
        pattern = spec["match"]
        until = int(spec["frozen_until"])
        params = [p for n, p in model.named_parameters() if fnmatchcase(n, pattern)]
        if not params:
            raise ValueError(
                f"freeze pattern {pattern!r} matched no parameters -- the "
                f"freeze would have been a silent no-op. Patterns are fnmatch "
                f"over named_parameters(), e.g. 'seq_layers.*.ska.*'.")
        return (pattern, until, params)

    def apply(self, step: int) -> Dict[str, float]:
        """Push every scheduled value for `step` and return them for logging."""
        values: Dict[str, float] = {}
        for target, fn, sites, attr, cast in self.bindings:
            v = cast(fn(step))
            for m in sites:
                setattr(m, attr, v)
            values[target] = v
        if self.freeze is not None:
            _pattern, until, params = self.freeze
            trainable = step >= until
            for p in params:
                p.requires_grad_(trainable)
            values[FREEZE_TARGET] = 0.0 if trainable else 1.0
        # data.seq_len is deliberately ABSENT from this dict. apply() reports
        # what it just made true, and seq_len is not per-step: the dataset is
        # only rebuilt at an epoch boundary, so a step past a breakpoint is
        # still feeding the old length. Reporting the raw schedule value here
        # would log a change that has not taken effect -- the epoch loop owns
        # both the rebuild and its reporting (§6.5).
        return values

    def seq_len_at(self, step: int) -> Optional[int]:
        """The sequence length for `step`, or None when no seq_len schedule is
        declared. Consumed by the epoch loop when it rebuilds epoch_loader --
        it is not settable on a module, and anything finer than per-epoch would
        mean rebuilding mid-epoch and breaking the resume index arithmetic
        (§6.5)."""
        return None if self.seq_len is None else self.seq_len(step)

    @property
    def freezes_parameters(self) -> bool:
        """True when a freeze schedule is present, so the caller knows the
        optimizer must be built over ALL parameters (§6.5): a parameter frozen
        at step 0 would otherwise never enter the optimizer and could never
        unfreeze."""
        return self.freeze is not None
