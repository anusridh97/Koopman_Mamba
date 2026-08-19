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

**The eta/gamma policy is pinned, not inherited.** Five tier-2 configs get the
*old* policy (learnable eta and gamma, clamped) by omission, because the
`KoopmanLMConfig` field defaults still encode it. A search that inherited
whatever its base config happened to imply would be comparing across two
parameterisations, so the modern policy is written explicitly into every trial.

**Backend hazard.** Every policy here except ``proxy_chunked`` sets
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
from typing import Any, Dict, Mapping, Sequence

from koopman_lm.config import KoopmanLMConfig
from experimentation.sweep.search.geometry import (
    PLACEMENTS, layer_count_choices, make_layer_indices)

__all__ = ["BACKEND_POLICIES", "default_base_lr", "search_space",
           "params_to_overrides"]

BACKEND_POLICIES = ("exact_auto", "fused_only", "proxy_chunked")

# The fused SM100 prefix-scan kernel is specialised to this rank (and value
# width 64); see configs/50m.yaml's header.
_FUSED_RANK = 24

DEFAULT_RANKS = (8, 16, 24, 32)
DEFAULT_WEIGHT_DECAYS = (0.05, 0.10, 0.15)
DEFAULT_WARMUP_RATIOS = (0.02, 0.04, 0.06)
DEFAULT_GRAD_CLIPS = (0.5, 1.0)
DEFAULT_GAMMAS = (0.90, 1.00, 1.05)
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
                     "choices": sorted({*(int(r) for r in ranks), base_model.ska_rank})},
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
        "learning_rate": {"kind": "float", "log": True,
                          "low": lr * LR_FACTOR_BOUNDS[0], "high": lr * LR_FACTOR_BOUNDS[1]},
        "weight_decay": {"kind": "categorical", "choices": list(DEFAULT_WEIGHT_DECAYS)},
        "warmup_ratio": {"kind": "categorical", "choices": list(DEFAULT_WARMUP_RATIOS)},
        "grad_clip": {"kind": "categorical", "choices": list(DEFAULT_GRAD_CLIPS)},
    }


def _backend_overrides(policy: str, rank: int) -> Dict[str, Any]:
    if policy not in BACKEND_POLICIES:
        raise ValueError(
            f"unknown backend policy {policy!r}; expected one of {list(BACKEND_POLICIES)}")
    if policy == "proxy_chunked":
        # The cheap approximate screen: no exact prefix scan.
        return {"model.ska_prefix_scan": False,
                "model.ska_backend": "auto",
                "model.ska_inverse_cholesky": False,
                "model.ska_exact_intrachunk": False}
    if policy == "fused_only" and rank != _FUSED_RANK:
        raise ValueError(
            f"backend policy 'fused_only' pins the fused SM100 kernel, which is "
            f"specialised to ska_rank={_FUSED_RANK}; got {rank}. Use 'exact_auto' "
            f"to let the backend fall back per rank.")
    return {"model.ska_prefix_scan": True,
            "model.ska_backend": "cuda_prefix" if policy == "fused_only" else "auto",
            "model.ska_inverse_cholesky": False,
            "model.ska_exact_intrachunk": False,
            "model.ska_prefix_scan_block_size": 32,
            "model.ska_prefix_scan_jitter": 0.0}


def params_to_overrides(params: Mapping[str, Any], base_model: KoopmanLMConfig, *,
                        max_steps: int,
                        seq_len: int | None = None,
                        backend_policy: str = "exact_auto") -> Dict[str, Any]:
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
        # Pinned, not inherited -- see the module docstring on the tier-2 configs.
        "model.ska_layerscale": True,
        "model.ska_norm_clip": True,
        "model.ska_eta_learnable": False,
        "model.ska_eta_value": 1.0,
        "model.ska_eta_bounds": None,
        "model.ska_gamma_learnable": False,
        "model.ska_gamma_clamp": None,
        "model.ska_gamma_bounds": None,
        "model.ska_power_K": 1,
        "optim.lr": float(params["learning_rate"]),
        "optim.weight_decay": float(params["weight_decay"]),
        "optim.grad_clip": float(params["grad_clip"]),
        "optim.warmup_steps": max(1, round(max_steps * float(params["warmup_ratio"]))),
    }
    if seq_len is not None:
        overrides["model.max_seq_len"] = int(seq_len)
    overrides.update(_backend_overrides(backend_policy, rank))
    return overrides
