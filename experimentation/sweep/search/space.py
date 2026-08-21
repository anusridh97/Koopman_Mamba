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

__all__ = ["BACKEND_POLICIES", "default_base_lr", "search_space",
           "params_to_overrides"]

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
        # Pinned, not inherited -- see the module docstring on the tier-2 configs.
        "model.ska_layerscale": True,
        "model.ska_norm_clip": True,
        "model.ska_eta_learnable": False,
        "model.ska_eta_value": 1.0,
        "model.ska_eta_bounds": None,
        "model.ska_gamma_learnable": False,
        "model.ska_gamma_clamp": None,
        "model.ska_gamma_bounds": None,
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
