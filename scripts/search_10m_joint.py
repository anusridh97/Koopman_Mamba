#!/usr/bin/env python3
"""The 10M joint architecture + optimizer search, as one standalone driver.

    python scripts/search_10m_joint.py configs/search/10m-joint-v1.yaml --dry_run
    python scripts/search_10m_joint.py configs/search/10m-joint-v1.yaml
    python scripts/search_10m_joint.py <study> --report_only   # re-read a journal

`--dry_run` needs no GPU and no optuna: it prints the plan and verifies the
parameter band exhaustively. Set `KOOPMAN_SEARCH_WORKER_LOG_DIR` to give each
worker its own log file instead of eight processes interleaving one stream.
Resubmitting is resuming -- `n_trials` is the study's target size, so a study
that already finished 1400 trials runs 600 more, not 2000.

## Why this is a script and not a change to experimentation/sweep/search/

`space.py` searches SKA and optimizer settings around ONE fixed backbone. It has
no depth, no Mamba expansion, no state size and no head-count axis, so the study
this file exists for -- "at a fixed ~10M parameter budget and a fixed token
budget, which ARCHITECTURE wins, and does the answer depend on the optimizer?" --
cannot be expressed as a study YAML alone.

Adding those axes to `space.py` would have been three functions and a validator
table. It is deliberately NOT done that way: several people are running studies
off this checkout right now, and `space.search_space()` is the space EVERY study
inherits by omission. Widening it silently adds dimensions to their studies and
changes `study_id` for archived ones. So the macro axes live here, in a file
nobody else's study imports, and everything else is the unmodified shared
machinery: `restrict_space` still validates the space, `run_trial` still does
ask -> materialize -> launch -> prune -> tell, `report`/`analysis` still produce
trials.csv and the Pareto fronts.

The one seam that makes this work with zero shared-code change: `run_trial` takes
`base_sections` and `base_model` as ARGUMENTS. So a macro sample is applied by
handing it a per-trial base -- a base whose depth, expansion, state size and head
count are the ones the sampler just drew. `params_to_overrides` then derives SKA
layer placement from the SAMPLED depth and the backend geometry from the SAMPLED
head count, because those are properties of the base model it is given. Nothing
had to learn about macro architecture; it just had to be told a different base.

## The macro axes, and why they are shaped this way

At 10M TOTAL parameters with a tied 32K-vocab embedding, width buys embedding
before it buys model. `vocab * d_model` is 4.096M at d_model=128 and 6.144M at
192, so the width choice is really a choice about how much of the budget is a
lookup table. Enumerated against `KoopmanLMConfig.param_count_estimate()`:

    d_model  embedding  depths inside 9.2M-11.0M   verdict
      128      40%      18-28 across expansions    CHOSEN
      160      48%      12-18                      viable, more embedding
      192      58%      6-10                       REJECTED, see below
       96      31%      39 (expansion 3 only)      REJECTED, one expansion

d_model=192 is excluded even though it is in band, for the reason d_model=80 was
excluded at 3M. `geometry.make_layer_indices` reserves layer 0 and the final
layer, so a 7-layer backbone has a 5-slot window: `n_ska_layers: 6` CLAMPS to 5,
collapsing onto the level below it, and `placement` resolves to near-identical
indices at every level. Three axes would be dead and a parameter importance
would be computed over a difference that does not exist. d_model=96 reaches the
band only at expansion 3, which would kill the `mamba_expand` contrast outright.

So width is FIXED at 128 rather than searched, and depth and Mamba expansion
trade off against each other at constant parameter count exactly as at 3M. The
axes are `mamba_expand` and a two-level `depth_tier`, and DEPTH IS DERIVED from
the pair (see `DEPTHS`). Sampling depth independently would propose (expand 3,
depth 28) = 13.4M and spend much of the study rejecting its own draws.

That shape buys two separable readings from one 6-cell table. Within a tier the
three expansions are matched in mean parameter count to under 1%, so
`mamba_expand` is a genuine iso-parameter contrast. Across tiers, at constant
expansion, the step is ~2% in capacity -- so `depth_tier` MEASURES this study's
own loss-per-parameter exchange rate. That second reading is not a nicety: rank,
head count and SKA layer count each move the parameter count by 1-2.5% as a side
effect, and without a measured slope there is no way to say whether a win on one
of them was architecture or capacity.

`d_state` and `ska_n_heads` are separable: every one of their values is in band
at every geometry, verified exhaustively by `--dry_run` before anything launches.

## What is fixed, and where those values come from

Fixed by the base RunSpec (`configs/runs/10m-joint-base.yaml`) and by this study
file, not by this script: NTP-only FineWeb-Edu, the Llama-2 tokenizer, tied 32K
embeddings, seq_len 2048, effective batch 96, cosine-to-zero, fused AdamW with
betas (0.9, 0.95), one exact inverse-Cholesky SKA route for every trial, seed 42,
and a token budget held CONSTANT across architectures. Constant tokens is the
load-bearing one: the band spans ~9% in parameter count, and giving the larger
candidates proportionally more data would confound architecture with data.

## What this script does NOT do

It DOES enqueue anchors, and that is the one deliberate difference from the 3M
driver. That study shipped without a design file on the argument that a
2000-trial random start is its own reference population -- which was true for the
sampler and false for the analysis: `analysis.noise_floor` needs two COMPLETED
trials sharing a `reference_group`, found none, and so every effect the 3M study
printed was UNCALIBRATED. The floor only existed afterwards, by accident, from
the Mamba-2 baseline seed replicates (sigma = 0.00142). A confirmatory study
cannot afford that: at a few hundred trials the whole point is to say which
differences are real, and that sentence has no meaning without a floor.

Anchors here sit at the BASE backbone. `Design` has no macro fields, so
`_enqueue_macro_anchors` merges the base's macro values into every resolved
anchor before enqueueing -- the same seam `apply_macro` uses for sampled trials.
A reference group is therefore N seeds of the base geometry, which is exactly
the quantity `noise_floor` wants.

It does not do multi-objective optimization. Optuna 4.9 cannot prune a
multi-objective study, and pruning is what makes 2000 trials affordable. TPE
minimizes held-out loss; parameter count and throughput are recorded per trial
and become Pareto fronts post hoc via `analysis.pareto_front` /
`analysis.throughput_pareto`, where the exchange rate is the reader's to choose.

`analysis.PRESPECIFIED_PAIRS` predates the macro axes, so `write_analysis` gives
main effects and importances for them but no macro pairwise tables. This script
writes those separately (`macro_pairwise.json`) using the same public function.

And it does not make the base RunSpec reachable as a sampled point. With depth
searched there is no "the base architecture": `configs/runs/10m-joint-base.yaml`
is where every trial's fixed fields come from, not a config the study can
rediscover and compare itself against. If you want that comparison, launch the
base spec by hand and read it beside `top_trials.md`.

## Before the real launch

  1. `--dry_run`, anywhere. Space, band and SKA-clamp checks, no GPU.
  2. One GPU smoke per (mamba_expand, depth_tier) cell -- six trials, a few
     dozen steps. `mamba_headdim` is pinned at 16, so expansion 1/2/3 give 4/8/12
     Mamba-2 heads, and only a real instantiation proves mamba_ssm's kernels
     accept all three. A kernel that refuses nheads=4 fails a third of the study,
     as ~660 identical FAILs.
  3. Confirm `run_root` exists and `eval_data_dir` is disjoint from the training
     shard. Held-out loss read off the training shard is not a ranking.
  4. Time one full-length trial, replace the study file's cost estimate with the
     measurement, and get the GPU-hour approval.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import traceback
from dataclasses import replace
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experimentation.run.launchers import LocalLauncher, SlurmLauncher
from experimentation.run.provenance import check_git_clean
from experimentation.run.resolve import resolve_model_config
from experimentation.sweep.search.anchors import (
    check_replicates_resolve, check_space_is_resolvable, load_designs,
    reference_groups, resolve_design)
from experimentation.sweep.search.geometry import make_layer_indices
from experimentation.sweep.search.space import restrict_space, search_space
from experimentation.sweep.search.studyspec import (
    load_study_spec, study_id as compute_study_id)
# Private in both cases, and imported rather than copied for the reason those
# modules give for existing: a base spec must be read the same way by every
# launcher, and the fanout must pin GPUs the same way as the shared CLI. A copy
# here would be a second answer to the same question.
from experimentation.sweep.search.__main__ import (
    OPTUNA_WORKER_ENV, WORKER_ENV, WORKER_LOG_DIR_ENV, should_fanout,
    visible_gpus, worker_index_from_env)
from experimentation.sweep.spec import _base_sections as load_base_sections

# --------------------------------------------------------------- the macro axes

#: Backbone depth as a function of (Mamba expansion, size tier).
#:
#: Depth is DERIVED rather than sampled, because at a fixed parameter budget it
#: is not free: an expansion-3 layer costs roughly what one and a half
#: expansion-2 layers cost, so (expand 3, depth 17) is 3.9M parameters and
#: sampling depth independently would spend a quarter of the study proposing
#: models outside the band.
#:
#: The two tiers are what make the expansion axis interpretable. Measured with
#: `param_count_estimate` over the full joint product, the mean parameter count
#: of each cell is:
#:
#:     tier   expand 1       expand 2       expand 3      spread
#:     lean   9.865M (27)    9.961M (22)    9.855M (18)   1.08%
#:     full   10.070M (28)   10.217M (23)   10.162M (19)  1.46%
#:
#: The within-tier match is looser than 3M's 0.2%/0.7%, and that is a granularity
#: limit rather than a choice: one layer at d_model 128 is ~205-307K parameters
#: depending on expansion, so the depth grid cannot land three expansions closer
#: than this anywhere that also leaves room for the searched axes. The one tight
#: triple that does exist (expand 1 d30 / 2 d24 / 3 d20, 0.11% spread) sits at
#: 10.47M, where the richest corner would reach 11.28M and leave the band.
#:
#: So within a tier, the three expansions are matched to under 1% -- the
#: expansion axis is a genuine ISO-PARAMETER contrast, and "expand 3 won" cannot
#: mean "expand 3 was bigger". And the tier axis is deliberately a ~2% capacity
#: step at constant expansion, which makes it the study's own measured exchange
#: rate between parameters and loss. That number is what calibrates every OTHER
#: axis here: rank, head count and SKA layer count all move the parameter count
#: by 1-2.5% as a side effect, and without a measured slope there is no way to
#: say whether a win on one of them was capacity or architecture.
#:
#: The searched axes swing the parameter count by 1.07M -- 10.3% of the base --
#: so the tiers are deliberately centred near 9.9M/10.2M rather than at the
#: tightest available match. `--dry_run` enumerates the exhaustive size-bearing
#: product (1,920 combinations) and confirms 0 violations over a 9.59M-10.83M
#: span, comparable to the 3M study's 12.3%.
DEPTHS = {
    (1, "lean"): 27, (1, "full"): 28,
    (2, "lean"): 22, (2, "full"): 23,
    (3, "lean"): 18, (3, "full"): 19,
}

#: Declared in `space.py`'s own declaration format, so `study.to_distribution`
#: converts them and a reader does not have to hold two notations at once.
MACRO_AXES = {
    # Mamba-2's inner expansion, d_inner = 64 * expand. With mamba_headdim
    # pinned at 16 this is 4, 8 or 12 SSM heads.
    "mamba_expand": {"kind": "categorical", "choices": [1, 2, 3]},
    # The capacity step, ~2% at constant expansion. See DEPTHS.
    "depth_tier": {"kind": "categorical", "choices": ["lean", "full"]},
    # Mamba-2's SSM state width. A multiple of 8 because KoopmanLMConfig asserts
    # it; 8 is mamba_ssm's own small end and 32 is where this band runs out.
    "d_state": {"kind": "categorical", "choices": [8, 16, 24, 32]},
    # SKA head geometry. head_dim = 128 // ska_n_heads, so this is {128, 64, 32, 16}
    # value width. 1 is included because head_dim 64 IS the production geometry
    # AGENTS.md records ("rank 24, value/head width 64"), and at d_model=64 that
    # is one head -- excluding it would make the study unable to propose the
    # configuration the repo currently recommends.
    "ska_n_heads": {"kind": "categorical", "choices": [1, 2, 4, 8]},
}

#: The approved total-parameter band, tied embedding counted once.
PARAM_BAND = (9_200_000, 11_000_000)

#: The shallowest backbone any geometry can produce. The space is validated
#: against THIS depth, not against the base spec's: `make_layer_indices` clamps a
#: too-large SKA layer count instead of raising, so an `n_ska_layers` choice that
#: is legal at depth 16 and clamped at depth 10 would be a dead axis level for a
#: third of the study with nothing in the output saying so.
MIN_DEPTH = min(DEPTHS.values())

#: Macro pairs worth a response surface, given the question this study asks.
#: `analysis.PRESPECIFIED_PAIRS` covers the SKA axes and predates these.
#:
#: `(mamba_expand, depth_tier)` is not here on purpose: together they DETERMINE
#: depth, so their cells are the six-cell table above and reading it as an
#: interaction would be reading the design, not a result.
MACRO_PAIRS = (
    ("mamba_expand", "learning_rate"),
    ("mamba_expand", "ska_rank"),
    ("mamba_expand", "n_ska_layers"),
    ("mamba_expand", "d_state"),
    ("depth_tier", "learning_rate"),
    ("ska_n_heads", "ska_rank"),
    ("d_state", "learning_rate"),
)


def macro_id() -> str:
    """A content hash of the macro declaration, for the study directory name.

    `studyspec.study_id` hashes the study YAML, and the macro axes are not in it
    -- they are in this file. So editing `GEOMETRIES` or `MACRO_AXES` would leave
    `study_id` unchanged and a resubmission would attach to a journal whose
    trials were drawn from a DIFFERENT space. Folding this into the directory name
    is what stops that: a changed space is a changed study, and it gets its own
    journal.
    """
    blob = json.dumps({"depths": {f"{k[0]}:{k[1]}": v for k, v in DEPTHS.items()},
                       "axes": MACRO_AXES, "band": list(PARAM_BAND)},
                      sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:8]


def macro_fields(params) -> dict:
    """One sampled point's macro axes -> KoopmanLMConfig field values.

    `n_layers` is derived from `(mamba_expand, depth_tier)` rather than sampled,
    so it is recorded on the trial as a user attr by the caller. Deriving it
    silently and recording nothing is the shape of bug this repo has had
    repeatedly: a resolved value that no column in trials.csv reports.
    """
    expand = int(params["mamba_expand"])
    tier = str(params["depth_tier"])
    return {"mamba_expand": expand,
            "n_layers": int(DEPTHS[(expand, tier)]),
            "d_state": int(params["d_state"]),
            "ska_n_heads": int(params["ska_n_heads"])}


def apply_macro(base_sections, base_model, params):
    """The per-trial base that `run_trial` is handed: `(sections, model)`.

    `ska_layer_indices` is resolved HERE as well as inside
    `params_to_overrides`, and that redundancy is required rather than sloppy:
    `KoopmanLMConfig.__post_init__` validates the indices against `n_layers`, so
    `replace(base_model, n_layers=10)` alone raises on the base spec's own
    `[2, 5, 8, 11]`. The two resolutions agree by construction -- same function,
    same arguments, and for a non-`baseline` placement the base indices are not
    an input at all.
    """
    fields = macro_fields(params)
    indices = make_layer_indices(int(fields["n_layers"]),
                                 int(params["n_ska_layers"]),
                                 str(params["placement"]),
                                 list(base_model.ska_layer_indices))
    model = replace(base_model, ska_layer_indices=tuple(indices), **fields)
    sections = {name: dict(values) for name, values in base_sections.items()}
    sections["model"] = {**sections["model"], **fields,
                         "ska_layer_indices": list(indices)}
    return sections, model


def size_bearing_product(base_model, space):
    """Every combination of the axes that CHANGE the parameter count.

    Yields `(params, resolved_indices, model)`. Ridge, LayerScale, gamma, K and
    every optimizer axis are absent because none of them adds a tensor, and
    `placement` is fixed to one representative because `make_layer_indices`
    returns `count` indices whatever the pattern -- so where they land cannot
    change a count.
    """
    placement = space["placement"]["choices"][0]
    for expand in MACRO_AXES["mamba_expand"]["choices"]:
        for tier in MACRO_AXES["depth_tier"]["choices"]:
            for d_state in MACRO_AXES["d_state"]["choices"]:
                for heads in MACRO_AXES["ska_n_heads"]["choices"]:
                    for rank in space["ska_rank"]["choices"]:
                        for count in space["n_ska_layers"]["choices"]:
                            for beta in space["beta_policy"]["choices"]:
                                params = {"mamba_expand": expand,
                                          "depth_tier": tier,
                                          "d_state": d_state,
                                          "ska_n_heads": heads,
                                          "ska_rank": rank,
                                          "n_ska_layers": count,
                                          "beta_policy": beta,
                                          "placement": placement}
                                fields = macro_fields(params)
                                indices = make_layer_indices(
                                    fields["n_layers"], int(count), placement,
                                    list(base_model.ska_layer_indices))
                                yield params, indices, replace(
                                    base_model,
                                    ska_layer_indices=tuple(indices),
                                    ska_rank=int(rank),
                                    ska_beta_policy=str(beta), **fields)


def check_band(base_model, space):
    """`(violations, checked, lo, hi)` over `size_bearing_product`.

    Exhaustive, not sampled, and run by `--dry_run` on a laptop with no GPU: a
    study that discovers an out-of-band architecture on trial 1400 has already
    spent the allocation. Two distinct faults are looked for -- a total outside
    `PARAM_BAND`, and an `n_ska_layers` choice that CLAMPED against the sampled
    depth, which would silently merge two axis levels into one config.
    """
    floor, ceiling = PARAM_BAND
    bad, checked, lo, hi = [], 0, None, None
    for params, indices, model in size_bearing_product(base_model, space):
        total = int(model.param_count_estimate())
        checked += 1
        lo = total if lo is None else min(lo, total)
        hi = total if hi is None else max(hi, total)
        label = (f"expand{params['mamba_expand']}/{params['depth_tier']}"
                 f"(depth {model.n_layers}) d_state={params['d_state']} "
                 f"heads={params['ska_n_heads']} rank={params['ska_rank']} "
                 f"n_ska={params['n_ska_layers']} beta={params['beta_policy']}")
        if not floor <= total <= ceiling:
            bad.append(f"{label}: {total:,} parameters")
        if len(indices) != int(params["n_ska_layers"]):
            bad.append(f"{label}: n_ska_layers clamped to {len(indices)}")
    return bad, checked, lo, hi


def tier_table(base_model, space):
    """Mean parameter count per (expand, tier) cell -- the iso-parameter claim.

    Printed by `--dry_run` rather than asserted, because "matched to under 1%" is
    a design goal with no bright line: what matters is that the number is in
    front of whoever approves the launch, beside the effect sizes it has to be
    compared against.
    """
    cells = {}
    for params, _indices, model in size_bearing_product(base_model, space):
        key = (int(params["mamba_expand"]), str(params["depth_tier"]))
        cells.setdefault(key, []).append(int(model.param_count_estimate()))
    return {key: (DEPTHS[key], sum(v) / len(v)) for key, v in cells.items()}


# ------------------------------------------------------------------- the driver

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="10M joint architecture + optimizer search",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("study", help="path to configs/search/10m-joint-*.yaml")
    p.add_argument("--dry_run", action="store_true",
                   help="print the plan, verify the parameter band, materialize "
                        "and submit nothing")
    p.add_argument("--allow-dirty", dest="allow_dirty", action="store_true")
    p.add_argument("--force", action="store_true",
                   help="reuse a run directory that is already claimed")
    p.add_argument("--report_only", action="store_true",
                   help="write the report and analysis for an existing journal "
                        "and run no trials")
    return p.parse_args(argv)


def _terminal_count(study):
    import optuna
    done = {optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED,
            optuna.trial.TrialState.FAIL}
    return sum(1 for t in study.trials if t.state in done)


def _fanout(args, spec, *, gpus, log_dir):
    """`spec.concurrent_trials` copies of THIS script, one per visible GPU.

    Shaped after `search/__main__._fanout` and for its reasons -- the supervisor
    does not run a trial itself, each child gets a line-buffered log file rather
    than a shared stream, and never a PIPE, because a pipe nobody drains blocks
    the child once it fills and the study stops with no error at all.
    """
    procs, logs = [], []
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
    for index in range(spec.concurrent_trials):
        env = {**os.environ, WORKER_ENV: str(index),
               OPTUNA_WORKER_ENV: str(index)}
        if gpus:
            env["CUDA_VISIBLE_DEVICES"] = gpus[index % len(gpus)]
        cmd = [sys.executable, str(Path(__file__).resolve()), args.study]
        if args.allow_dirty:
            cmd.append("--allow-dirty")
        if args.force:
            cmd.append("--force")
        if log_dir is not None:
            log = open(log_dir / f"worker-{index}.log", "w", buffering=1)
            logs.append(log)
            env["PYTHONUNBUFFERED"] = "1"
            proc = subprocess.Popen(cmd, env=env, stdout=log,
                                    stderr=subprocess.STDOUT)
        else:
            proc = subprocess.Popen(cmd, env=env)
        procs.append(proc)
        print(f"[10m-joint]   worker {index}  "
              f"GPU {env.get('CUDA_VISIBLE_DEVICES', '(none)')}  pid {proc.pid}")
    codes = [p.wait() for p in procs]
    for log in logs:
        log.close()
    for index, code in enumerate(codes):
        if code != 0:
            print(f"[10m-joint] worker {index} exited {code}")
    return 0 if all(c == 0 for c in codes) else 1


def _print_plan(spec, space, base_model, base_sections, *, study_dir, gpus,
                envelope):
    tokens = (int(base_sections["optim"]["effective_batch"])
              * int(spec.seq_len or base_model.max_seq_len) * int(spec.max_steps))
    lo, hi = envelope
    print(f"[10m-joint] study      {spec.name}")
    print(f"[10m-joint] base       {spec.base}")
    print(f"[10m-joint] study_dir  {study_dir}")
    print(f"[10m-joint] trials     {spec.n_trials} x {spec.max_steps} steps, "
          f"{spec.concurrent_trials} worker(s), GPUs {gpus or '(none)'}")
    print(f"[10m-joint] tokens     {tokens:,} per trial "
          f"({tokens / max(1, base_model.param_count_estimate()):.1f} per base param)")
    print(f"[10m-joint] param band {PARAM_BAND[0]:,}-{PARAM_BAND[1]:,}; "
          f"space spans {lo:,}-{hi:,} "
          f"({100.0 * (hi - lo) / lo:.1f}% spread)")
    print(f"[10m-joint] backend    {spec.backend_policy}   "
          f"sampler {spec.sampler} (startup {spec.sampler_startup_trials})")
    print(f"[10m-joint] pruning    after step {spec.prune_after_step} of "
          f"{spec.max_steps}, once {spec.prune_startup_trials} trials complete")
    print("[10m-joint] macro axes (declared in this script, hashed into study_dir):")
    for name, decl in MACRO_AXES.items():
        print(f"[10m-joint]   {name:16} {decl['choices']}")
    print("[10m-joint] derived depth, and the mean parameter count of each cell:")
    for (expand, tier), (depth, mean) in sorted(tier_table(base_model, space).items()):
        print(f"[10m-joint]   expand {expand}  {tier:5}  depth {depth:2}  "
              f"{mean:>11,.0f} mean parameters")
    print("[10m-joint] declared axes (space.py, restricted by the study file):")
    for name in sorted(space):
        decl = space[name]
        shown = (decl.get("choices") if decl["kind"] == "categorical"
                 else f"[{decl.get('low')}, {decl.get('high')}]"
                      f"{' log' if decl.get('log') else ''}")
        print(f"[10m-joint]   {name:22} {shown}")
    dims = len(MACRO_AXES) + len(space)
    print(f"[10m-joint] dimensions {dims} "
          f"({len(MACRO_AXES)} macro + {len(space)} declared)")


def _write_outputs(study, study_dir, spec, base_model):
    from experimentation.sweep.search.analysis import pairwise_table, write_analysis
    from experimentation.sweep.search.report import write_report

    written = write_report(study, study_dir, base_spec=spec.base,
                           base_model=base_model,
                           promotion_max_steps=spec.max_steps,
                           promotion_seq_len=spec.seq_len or base_model.max_seq_len)
    for name, path in sorted(written.items()):
        if path is not None:
            print(f"[10m-joint] wrote {name:16} {path}")
    for name, path in sorted(write_analysis(study, study_dir).items()):
        print(f"[10m-joint] wrote {name:16} {path}")
    # The macro response surfaces. `analysis.PRESPECIFIED_PAIRS` covers the SKA
    # axes and predates the macro ones, so these are computed here with the same
    # public function rather than by widening that constant.
    macro = {f"{left}__{right}": pairwise_table(study, left, right)
             for left, right in MACRO_PAIRS}
    path = Path(study_dir) / "macro_pairwise.json"
    path.write_text(json.dumps(macro, indent=2, default=str))
    print(f"[10m-joint] wrote {'macro_pairwise':16} {path}")


def _enqueue_macro_anchors(study, designs, base_model, base_sections, space,
                           *, base_lr):
    """`study.enqueue_anchors`, plus the macro axes it cannot know about.

    The shared `study.enqueue_anchors` resolves a Design against `space` and
    enqueues exactly those params. That is complete for a study whose space IS
    space.py's, and incomplete here: this driver adds four macro axes, and
    `apply_macro` reads `params["mamba_expand"]` on every trial. An anchor
    enqueued without them raises a KeyError the moment a worker picks it up --
    after the queue is written, so the study would have to be rebuilt.

    Anchors therefore sit at the BASE backbone: the base spec's own macro values
    are merged into every resolved point. That is the right shape for what
    anchors are for here. A reference group becomes N seeds of one fixed
    geometry, which is the quantity `analysis.noise_floor` pools; a screen anchor
    becomes a one-factor move against a fixed backbone, which is what makes
    `anchor_contrasts` readable.

    Mirrors the shared function's contract exactly otherwise: idempotent on the
    anchor NAME (not the params, which a reference group makes identical by
    construction), and `skip_if_exists=False` for the same reason.
    """
    from experimentation.sweep.search.study import (
        ANCHOR_ATTR, REFERENCE_GROUP_ATTR, SEED_ATTR, enqueued_anchor_names)

    macro = {"mamba_expand": int(base_model.mamba_expand),
             "d_state": int(base_model.d_state),
             "ska_n_heads": int(base_model.ska_n_heads),
             "depth_tier": _tier_of(int(base_model.n_layers),
                                    int(base_model.mamba_expand))}
    existing = enqueued_anchor_names(study)
    added = 0
    for design in designs:
        if design.name in existing:
            continue
        params = dict(resolve_design(design, base_model, space, base_lr=base_lr))
        params.update(macro)
        attrs = {ANCHOR_ATTR: design.name}
        seed = getattr(design, "seed", "baseline")
        if seed != "baseline" and seed is not None:
            attrs[SEED_ATTR] = int(seed)
        group = getattr(design, "reference_group", None)
        if group is not None:
            attrs[REFERENCE_GROUP_ATTR] = str(group)
        study.enqueue_trial(params, user_attrs=attrs, skip_if_exists=False)
        existing.add(design.name)
        added += 1
    return added


def _tier_of(n_layers, expand):
    """Which `depth_tier` the base spec's own depth corresponds to.

    Derived rather than declared so the base config and DEPTHS cannot drift into
    disagreeing: if someone edits one, this raises instead of silently pinning
    the anchors to a backbone the study never samples.
    """
    for (e, tier), depth in DEPTHS.items():
        if e == expand and depth == n_layers:
            return tier
    raise SystemExit(
        f"the base spec is depth {n_layers} at expansion {expand}, which is not "
        f"a cell of DEPTHS {sorted(DEPTHS.items())}. Anchors sit at the base "
        f"backbone, so the base must BE one of the sampled geometries.")


def main(argv=None):
    args = parse_args(argv)
    spec = load_study_spec(args.study)

    # A parameter penalty cannot mean what it says in this study, so it is
    # refused rather than silently neutered. `driver._record_trial_attrs` stamps
    # `baseline_param_count` from the base model it is GIVEN, and this script
    # gives it the per-trial macro base -- so baseline and actual are the same
    # number and any penalty computed from their difference is identically zero.
    # The loss/size trade lives in `analysis.pareto_front` instead, where the
    # exchange rate is the reader's and can be changed without re-running.
    if spec.objective:
        raise SystemExit(
            f"{args.study}: `objective:` must be empty in this study. "
            f"{sorted(spec.objective)} would be computed against a per-trial "
            f"baseline that equals the trial itself. Parameter count and "
            f"throughput are recorded per trial and traded off post hoc.")

    base_sections = load_base_sections(spec.base)
    base_model = resolve_model_config(base_sections["model"])
    # Validated against the SHALLOWEST geometry, not the base spec's depth -- see
    # MIN_DEPTH for the dead-axis this prevents.
    shallowest = replace(
        base_model, n_layers=MIN_DEPTH,
        ska_layer_indices=tuple(make_layer_indices(
            MIN_DEPTH, len(base_model.ska_layer_indices), "even",
            list(base_model.ska_layer_indices))))
    space = restrict_space(search_space(shallowest, base_name=spec.base),
                           shallowest, axes=spec.search_axes,
                           fixed=spec.fixed_params)
    collisions = sorted(set(MACRO_AXES) & set(space))
    if collisions:
        raise SystemExit(
            f"{collisions} are declared both here and by space.py. The macro "
            f"axes exist because space.py has none; if it has grown them, "
            f"delete them from this script rather than sampling twice.")
    if "baseline" in space["placement"]["choices"]:
        raise SystemExit(
            f"{args.study}: `placement: baseline` is not meaningful when depth "
            f"is searched -- there is no single base backbone for it to "
            f"reproduce, and make_layer_indices falls back to interpolating "
            f"across the base indices clipped to the sampled depth, which is a "
            f"near-duplicate of `even`. Declare [even, midlate, late].")

    study_dir = (Path(spec.run_root) / "_studies"
                 / f"{spec.name}.{compute_study_id(spec)}.{macro_id()}")
    raw_worker = os.environ.get(WORKER_ENV)
    is_worker = raw_worker is not None
    worker_index = worker_index_from_env()
    gpus = visible_gpus()

    if not is_worker:
        bad, checked, lo, hi = check_band(base_model, space)
        _print_plan(spec, space, base_model, base_sections,
                    study_dir=study_dir, gpus=gpus, envelope=(lo, hi))
        print(f"[10m-joint] band check {checked} size-bearing combination(s), "
              f"{len(bad)} violation(s)")
        if bad:
            for row in bad[:10]:
                print(f"[10m-joint]   REJECTED {row}")
            raise SystemExit(
                f"{len(bad)} combination(s) leave {PARAM_BAND} or clamp the SKA "
                f"layer count. Narrow DEPTHS or the study's axes; do not launch "
                f"a study that rejects its own proposals.")

    # Anchors. Loaded and RESOLVED here rather than inside the enqueue, so a
    # malformed design file dies on a laptop instead of on the cluster: an
    # undeclared placement, an undeclared power_K and an axis-kind change all
    # fail at resolution, which is past create_study and past the git gate.
    designs = []
    if spec.design_file:
        designs = load_designs(spec.design_file, minimum=1)
        check_space_is_resolvable(space)
        base_lr = float(base_sections["optim"]["lr"])
        resolved = [resolve_design(d, base_model, space, base_lr=base_lr)
                    for d in designs]
        check_replicates_resolve(designs, int(base_sections["runtime"]["seed"]))
        if not is_worker:
            groups = reference_groups(designs)
            print(f"[10m-joint] anchors    {len(designs)} from {spec.design_file}, "
                  f"{len(resolved)} resolve")
            if groups:
                for name, members in sorted(groups.items()):
                    seeds = [getattr(m, "seed", "baseline") for m in members]
                    print(f"[10m-joint]   noise floor '{name}': {len(members)} "
                          f"replicates, seeds {seeds}")
            else:
                print("[10m-joint]   WARNING: no reference_group -- "
                      "analysis.noise_floor will report nothing, exactly as the "
                      "3M study did. Every effect would be UNCALIBRATED.")
    elif not is_worker:
        print("[10m-joint] anchors    none -- no design_file, so there will be "
              "NO noise floor (see this script's docstring)")

    if args.dry_run:
        print("[10m-joint] --dry_run: nothing materialized, nothing submitted.")
        return 0

    # Imported here, not at module scope, for the reason the shared package
    # splits along this line: everything above works with optuna absent, which is
    # what lets `--dry_run` check the space and the parameter band on a laptop.
    from experimentation.sweep.search.driver import (
        FATAL_EXCEPTIONS, FatalTrialError, run_trial)
    from experimentation.sweep.search.metrics import wait_for_objective
    from experimentation.sweep.search.study import (
        create_study, sampler_seed_for, to_distributions)

    # One dict, so the supervisor's study and the post-fanout reattachment cannot
    # disagree about the sampler, the pruner or the space -- optuna accepts a
    # reopen with a different pruner SILENTLY.
    study_kwargs = dict(
        study_name=spec.name, study_dir=study_dir, seed=spec.seed,
        concurrent_trials=spec.concurrent_trials, sampler=spec.sampler,
        sampler_startup_trials=spec.sampler_startup_trials,
        worker_id=worker_index, prune_after_step=spec.prune_after_step,
        prune_startup_trials=spec.prune_startup_trials, n_trials=spec.n_trials,
        logging_steps=spec.logging_steps, direction=spec.direction,
        storage_url=spec.storage)
    study = create_study(**study_kwargs)

    # Before the dirty-tree check on purpose: reading a finished journal spends
    # nothing and cannot contaminate a result, and refusing to report because the
    # working tree moved since the launch is how a study's own output becomes
    # unreachable.
    if args.report_only:
        _write_outputs(study, study_dir, spec, base_model)
        return 0

    dirty = check_git_clean(allow_dirty=args.allow_dirty)

    # THE SUPERVISOR ONLY, for the reason the shared CLI gives: `enqueue_anchors`
    # is a read-then-write with no lock -- JournalStorage offers no atomic
    # reservation -- so N processes calling it concurrently CAN double-enqueue.
    # A double-enqueued reference group is the worst available outcome: under
    # `deterministic: true` the duplicates land on identical losses, pooling them
    # drives sigma toward zero, and every effect in the analysis becomes
    # "resolved". Not creating that state beats detecting it.
    if designs and not is_worker:
        added = _enqueue_macro_anchors(
            study, designs, base_model, base_sections, space,
            base_lr=float(base_sections["optim"]["lr"]))
        print(f"[10m-joint] enqueued {added} anchor(s) "
              f"({len(designs) - added} already present)")

    if should_fanout(spec.concurrent_trials, raw_worker):
        print(f"[10m-joint] launching {spec.concurrent_trials} worker(s)")
        code = _fanout(args, spec, gpus=gpus,
                       log_dir=os.environ.get(WORKER_LOG_DIR_ENV))
        # Reattach: this process built its Study before the workers ran, so its
        # trial list is a stale snapshot and a report from it describes an empty
        # study.
        _write_outputs(create_study(**study_kwargs), study_dir, spec, base_model)
        return code

    # `eval_on_final` is what makes a trial scoreable at all -- it puts
    # --eval_on_final on the training command so the run writes
    # eval/final/quick_eval.json. On the LAUNCHER and not the spec: whether a run
    # scores itself is a property of who launched it, and anything on the spec is
    # hashed into run_id.
    scoring = {"eval_on_final": True, "eval_data_dir": spec.eval_data_dir,
               "logging_steps": spec.logging_steps}
    launcher = (LocalLauncher(**scoring) if spec.launcher == "local"
                else SlurmLauncher(repo_root=str(REPO), **scoring))

    def objective_reader_for(study_, trial):
        def read(run_dir):
            return wait_for_objective(study_, trial, run_dir,
                                      timeout_seconds=spec.trial_timeout_seconds)
        return read

    distributions = to_distributions({**MACRO_AXES, **space})
    sampler_seed = sampler_seed_for(spec.seed, worker_index)
    states, ordinal, stalled = {}, 0, 0
    while _terminal_count(study) < spec.n_trials:
        before = _terminal_count(study)
        trial = study.ask(distributions)
        try:
            # THE macro seam: a per-trial base, so `params_to_overrides` derives
            # SKA placement from the sampled depth and backend geometry from the
            # sampled head count without knowing macro architecture exists.
            sections, model = apply_macro(base_sections, base_model, trial.params)
            # The two DERIVED quantities, as their own trials.csv columns.
            # `mamba_expand`, `depth_tier`, `d_state` and `ska_n_heads` are in
            # trial.params already; depth is a lookup and d_inner is a product,
            # and a resolved value that no column reports is the shape of bug
            # this repo has had several times.
            trial.set_user_attr("n_layers", int(model.n_layers))
            trial.set_user_attr("d_inner", int(model.d_model * model.mamba_expand))
            outcome = run_trial(
                study, trial, base_sections=sections, base_model=model,
                space=space, max_steps=spec.max_steps, run_root=spec.run_root,
                study_name=spec.name, launcher=launcher,
                objective_reader_for=objective_reader_for,
                backend_policy=spec.backend_policy, seq_len=spec.seq_len,
                force=args.force, dirty=dirty, batch_ladder=spec.batch_ladder,
                worker_id=worker_index, sampler_name=spec.sampler,
                sampler_seed=sampler_seed, worker_trial_ordinal=ordinal)
            states[outcome.state] = states.get(outcome.state, 0) + 1
        except FATAL_EXCEPTIONS:
            _fail(study, trial, sys.exc_info()[1])
            raise
        except FatalTrialError as exc:
            _fail(study, trial, exc)
            raise
        except Exception as exc:                        # noqa: BLE001
            # This trial's problem as far as anything here can tell: recorded,
            # told, and the worker moves on. A claimed directory or a transient
            # /scratch error must not leave a trial RUNNING forever while the
            # remaining hours run one worker short with nothing saying why.
            _fail(study, trial, exc)
            states["failed"] = states.get("failed", 0) + 1
        ordinal += 1
        if _terminal_count(study) > before:
            stalled = 0
            continue
        stalled += 1
        if stalled >= 3:
            raise SystemExit(
                f"[10m-joint] {stalled} trials finished without advancing the "
                f"study's terminal count ({before} of {spec.n_trials}). The "
                f"journal is not recording states -- most likely it has become "
                f"unwritable -- so this loop would spin forever.")

    print(f"[10m-joint] {sum(states.values())} trial(s): "
          + ", ".join(f"{n} {s}" for s, n in sorted(states.items())))
    _write_outputs(study, study_dir, spec, base_model)
    return 0


def _fail(study, trial, exc):
    """Record why a trial died and tell optuna, so it does not sit RUNNING."""
    import optuna
    try:
        trial.set_user_attr("failure", f"{type(exc).__name__}: {exc}")
        trial.set_user_attr("traceback", traceback.format_exc())
        study.tell(trial, state=optuna.trial.TrialState.FAIL)
    except Exception:                                   # noqa: BLE001
        # The storage itself is failing. Raising here would mask the original.
        pass


if __name__ == "__main__":
    sys.exit(main())
