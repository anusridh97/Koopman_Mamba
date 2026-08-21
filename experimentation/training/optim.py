"""
optim.py -- Shared AdamW parameter-group policy, plus §7 per-group overrides.

Extracted from training/train.py so experiment scripts (table2.py,
mqar_finetune.py) can apply the same decay/no-decay split instead of each
re-deriving (or, historically, skipping) it. Kept dependency-light on
purpose: importing this module must not require torch, transformers, DDP, or
any GPU-only extras, so CPU-only experiment scripts, the run-spec layer, and
tests can use it freely.

§7 of docs/superpowers/specs/2026-08-08-extension-mechanisms-design.md adds
`optim.groups`: an ordered list of (match, lr_mult, weight_decay) overrides
applied on top of the decay policy, not instead of it.
"""
from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatchcase
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple


@dataclass(frozen=True)
class ParamGroupSpec:
    """One entry of `optim.groups` (§7).

    `match` is an fnmatch pattern over parameter names as
    ``model.named_parameters()`` spells them (e.g. ``"*.ska.*"``,
    ``"embed.weight"``).

    `lr_mult` is a MULTIPLIER, never an absolute lr.
    ``get_cosine_schedule_with_warmup`` is a ``LambdaLR``: it snapshots each
    group's lr as ``initial_lr`` and multiplies every group by the same
    factor each step. A multiplier therefore composes cleanly with the shared
    warmup+cosine, whereas an absolute per-group lr would fight it.

    `weight_decay` is None by default, meaning "inherit the decay/no-decay
    policy". A number overrides that policy for the matched parameters --
    including forcing decay onto a parameter the policy would have exempted,
    which is deliberate: an explicit override is a deliberate act.
    """
    match: str
    lr_mult: float = 1.0
    weight_decay: Optional[float] = None

    def __post_init__(self):
        if not self.match:
            raise ValueError("ParamGroupSpec.match is required and must be non-empty")
        if not isinstance(self.lr_mult, (int, float)) or isinstance(self.lr_mult, bool):
            raise TypeError(
                f"ParamGroupSpec.lr_mult must be numeric, got "
                f"{type(self.lr_mult).__name__} ({self.lr_mult!r}) -- check your "
                f"YAML: PyYAML parses '1e-1' as a string (no decimal point in "
                f"the mantissa); use '1.0e-1' or '0.1'.")
        if self.lr_mult <= 0:
            raise ValueError(
                f"ParamGroupSpec.lr_mult must be > 0, got {self.lr_mult}. A zero "
                f"multiplier is a freeze; express that as a `freeze` schedule "
                f"(design §6.5) so it can be lifted at a known step.")
        if self.weight_decay is not None:
            if not isinstance(self.weight_decay, (int, float)) or \
                    isinstance(self.weight_decay, bool):
                raise TypeError(
                    f"ParamGroupSpec.weight_decay must be numeric or null, got "
                    f"{type(self.weight_decay).__name__} ({self.weight_decay!r})")
            if self.weight_decay < 0:
                raise ValueError(
                    f"ParamGroupSpec.weight_decay must be >= 0, got {self.weight_decay}")


def parse_group_specs(raw: Optional[Iterable[Any]]) -> Tuple[ParamGroupSpec, ...]:
    """Coerce YAML's list-of-dicts into ParamGroupSpecs (idempotent on
    already-parsed specs), so OptimSpec and train.py validate identically.

    An unknown key raises rather than being ignored: `lr_multiplier:` instead
    of `lr_mult:` would otherwise silently run the unmodified experiment.
    """
    if raw is None:
        return ()
    out = []
    for i, item in enumerate(raw):
        if isinstance(item, ParamGroupSpec):
            out.append(item)
            continue
        if not isinstance(item, dict):
            raise TypeError(
                f"optim.groups[{i}] must be a mapping, got {type(item).__name__}")
        allowed = {"match", "lr_mult", "weight_decay"}
        unknown = sorted(set(item) - allowed)
        if unknown:
            raise ValueError(
                f"optim.groups[{i}] has unknown key(s) {unknown}; "
                f"expected some of {sorted(allowed)}")
        if "match" not in item:
            raise ValueError(f"optim.groups[{i}] is missing the required key 'match'")
        out.append(ParamGroupSpec(**item))
    return tuple(out)


def _should_decay(name, p, explicit_skip, special_leaves) -> bool:
    leaf = name.rsplit('.', 1)[-1]
    is_embedding = name == 'embed.weight' or '.embed.' in name or 'embedding' in name
    skip = (
        name in explicit_skip
        or p.ndim < 2
        or leaf == 'bias'
        or leaf in special_leaves
        or is_embedding
        or bool(getattr(p, '_no_weight_decay', False))
    )
    return not skip


def param_groups(raw_model, weight_decay, groups: Optional[Sequence[Any]] = None,
                 *, lr: Optional[float] = None, include_frozen: bool = False):
    """Return AdamW parameter groups: the standard decay/no-decay split, with
    §7 per-group overrides layered on top.

    A flat ``model.parameters()`` call decays every parameter unless a rare
    Koopman-v2 option is enabled. That includes norm scales, biases,
    embeddings, Mamba state-space parameters, LayerScale, and
    spectral/geometry variables. At long schedules ``weight_decay=0.1`` (or
    even table2.py's 0.01) can shrink those parameters by several-fold even
    before gradients are considered. Matrix weights are decayed; state,
    scale, bias, embedding, and explicitly geometric parameters are not.

    `groups` (§7) is an ordered sequence of ParamGroupSpec (or the dicts
    parse_group_specs accepts). Parameters are bucketed by the pair
    ``(should_decay, first_matching_spec)``, so an override refines the decay
    policy rather than replacing it: one ``"ska.*"`` pattern still yields a
    decayed bucket for ``ska.weight`` and a no-decay bucket for ``ska.bias``.
    First match wins. **An empty or absent `groups` leaves the output
    bit-identical to the historical two-bucket policy** -- that equivalence is
    pinned by code-tests/test_optim_groups.py.

    `lr` is the base learning rate, required only when some spec carries a
    non-unit `lr_mult` (there is nothing to multiply otherwise). When given
    alongside overrides, every returned group carries an explicit
    ``lr = lr * lr_mult`` so a LambdaLR picks it up as that group's
    ``initial_lr``.

    `include_frozen` (§6.5) builds groups over parameters with
    ``requires_grad=False`` too. Default False keeps every existing caller
    bit-identical; a freeze *schedule* must pass True, because a parameter
    frozen at step 0 would otherwise never enter the optimizer and could never
    unfreeze. Including them is numerically inert while they stay frozen --
    AdamW skips any parameter whose ``.grad`` is None.
    """
    specs = parse_group_specs(groups)
    if lr is None and any(s.lr_mult != 1.0 for s in specs):
        raise ValueError(
            "optim.groups uses lr_mult but no base lr was passed to "
            "param_groups(); lr_mult is a multiplier and needs one to "
            "multiply. Pass lr=<base learning rate>.")

    fn = getattr(raw_model, 'no_weight_decay_param_names', None)
    explicit_skip = set(fn() if fn is not None else ())
    special_leaves = {
        # Mamba state-space / discretization parameters.
        'A_log', 'D', 'dt_bias',
        # Koopman/SKA geometry and residual controls (some are >1D).
        'lift_v', 'lift_g', 'A_raw', 's', 'theta', 'gamma', 'omega',
        'eta', 'eta_raw', 'ssn_gamma', 'layerscale_gate', 'short_conv_gate',
    }

    # Bucket key -> params, insertion-ordered so the no-override buckets keep
    # their historical (decay, no_decay) position when nothing matches.
    buckets: Dict[Tuple[bool, Optional[int]], list] = {}
    matched_any = [False] * len(specs)
    seen = set()
    for name, p in raw_model.named_parameters():
        if (not include_frozen and not p.requires_grad) or id(p) in seen:
            continue
        seen.add(id(p))
        idx = None
        for i, spec in enumerate(specs):
            if fnmatchcase(name, spec.match):
                idx, matched_any[i] = i, True
                break
        buckets.setdefault((_should_decay(name, p, explicit_skip, special_leaves), idx),
                           []).append(p)

    dead = [specs[i].match for i, hit in enumerate(matched_any) if not hit]
    if dead:
        raise ValueError(
            f"optim.groups pattern(s) {dead} matched no parameters. A "
            f"silently-dead pattern is how you believe you ran an experiment "
            f"you did not (design §7). Patterns are fnmatch over "
            f"named_parameters(), e.g. 'seq_layers.*.ska.*'.")

    # Historical ordering: decay bucket first, then no-decay.
    out = []
    for key in sorted(buckets, key=lambda k: (not k[0], -1 if k[1] is None else k[1])):
        should_decay, idx = key
        spec = None if idx is None else specs[idx]
        wd = weight_decay if should_decay else 0.0
        if spec is not None and spec.weight_decay is not None:
            wd = spec.weight_decay
        group = {'params': buckets[key], 'weight_decay': wd}
        if lr is not None and specs:
            group['lr'] = lr * (1.0 if spec is None else spec.lr_mult)
        out.append(group)
    return out
