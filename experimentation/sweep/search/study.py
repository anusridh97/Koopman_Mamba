"""The optuna boundary: plain-data declarations in, a configured Study out.

One of only two modules in this package that imports optuna (the other is
driver.py). Everything below the boundary -- geometry, space, anchors -- stays
usable with optuna absent, which is what lets a curated design set ship as a
static sweep before any of this exists.

Three construction choices, each with a reason.

**A file-backed journal, not an in-memory study.** `JournalStorage` over
`JournalFileBackend` means a launcher crash loses no completed trial, a study can
be resumed, and several processes can attach to the same study concurrently. That
last property is the reason to prefer N single-GPU processes over
`study.optimize(n_jobs=N)`'s thread pool, which modern optuna deprecates.
The spec field that sizes our fleet is `concurrent_trials`, deliberately named
neither `n_jobs` (optuna's thread pool, which we reject) nor `workers`
(`RuntimeSpec.workers` is the dataloader count), so the three ideas cannot be
confused.

**`constant_liar` when running in parallel.** With N workers in flight and none
finished, every worker proposes from the same history and they converge on nearly
the same point -- N GPUs computing one answer. `constant_liar` pessimistically
imputes running trials so proposals repel each other. It is flagged experimental
by optuna; we take it knowingly, because the failure it prevents is worse than
the risk of an interface change, and the warning is silenced only at the one
construction site that opts in.

**MedianPruner, reconciled against the original.** This was first written from
the one visible signal (`--prune-after-step 180`) because the original's
`create_study` sat in a truncated part of the file. The full file since confirmed
the class and `n_warmup_steps`, and supplied three details the inference missed:

    MedianPruner(n_startup_trials=min(6, max(3, n_trials // 3)),
                 n_warmup_steps=args.prune_after_step,
                 interval_steps=max(1, args.logging_steps),
                 n_min_trials=3)

`interval_steps` matters more than it looks. `train.py` prints only every
`logging_steps` steps, so reports arrive at 10, 20, 30...; consulting the pruner
at every integer step between them compares a trial against steps no other trial
ever reported. `n_min_trials=3` stops one unlucky reference run from deciding the
median alone. And the startup count scales with study size rather than being
fixed, so a 15-trial study waits for 5 completions while a small study still waits
for 3.

The cold start is not a bug and is pinned by test: MedianPruner prunes nothing
until `n_startup_trials` trials have COMPLETED, because before that there is no
median to compare against. Six rising loss reports returning
`should_prune() == False` looks broken and is not. The practical consequence is
that the curated anchors are what establish the reference set, so they must be
allowed to finish -- which is why `is_anchor` exists and why the driver exempts
them from pruning by default.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import optuna

from koopman_lm.config import KoopmanLMConfig
from experimentation.sweep.search.anchors import resolve_design
from experimentation.sweep.search.studyspec import (
    DEFAULT_STARTUP_TRIALS, SAMPLERS, derived_prune_startup_trials)

__all__ = ["SAMPLERS", "to_distribution", "to_distributions", "make_sampler",
           "make_pruner", "prune_startup_trials_for", "sampler_seed_for",
           "make_storage", "create_study", "enqueue_anchors", "is_anchor",
           "ANCHOR_ATTR"]

ANCHOR_ATTR = "anchor_name"

# TPE samples randomly until this many trials have FINISHED (optuna counts
# COMPLETE and PRUNED here, not COMPLETE alone -- verified, not assumed); with an
# anchor set the anchors themselves fill the startup window, so TPE begins
# modelling from chosen points rather than from noise. Re-exported from
# studyspec, which owns it because __main__ needs it without optuna.

# The trainer's default logging cadence (train.py --logging_steps). The pruner is
# consulted on this interval because that is when progress reports actually exist.
DEFAULT_LOGGING_STEPS = 10

# At least this many trials must have reported at a step before the median there
# is worth acting on.
PRUNE_MIN_TRIALS = 3

# Loss curves cross constantly in the first stretch of training; pruning inside
# that window kills configs for being slow to warm up rather than bad.
DEFAULT_PRUNE_AFTER_STEP = 180


def to_distribution(declaration: Mapping[str, Any]) -> optuna.distributions.BaseDistribution:
    """One `space.py` declaration -> the optuna distribution it describes."""
    kind = declaration.get("kind")
    if kind == "categorical":
        return optuna.distributions.CategoricalDistribution(
            list(declaration["choices"]))
    if kind == "float":
        return optuna.distributions.FloatDistribution(
            float(declaration["low"]), float(declaration["high"]),
            log=bool(declaration.get("log", False)))
    raise ValueError(
        f"unsupported declaration kind {kind!r}; space.py emits 'categorical' "
        f"or 'float'")


def to_distributions(space: Mapping[str, Mapping[str, Any]]
                     ) -> Dict[str, optuna.distributions.BaseDistribution]:
    return {name: to_distribution(decl) for name, decl in space.items()}


def sampler_seed_for(seed: int, worker_id: Optional[int]) -> int:
    """The SAMPLER's seed for one worker. Never the model's or the data's.

    The gap this closes. `_fanout` spawns N identical processes and each builds
    its own sampler; with one seed for all of them, every worker's TPE draws the
    *same* random stream, so the whole startup window is N copies of one sequence
    and `constant_liar` is left to repel proposals that were identical by
    construction. Deriving a distinct stream per worker is what makes N workers
    explore N times as much rather than N times as often.

    `seed + worker_id` and not a hash, for one reason: a study's provenance
    should be arithmetic a human can check against `trials.csv`. Worker 3 of the
    2026-seeded study sampled with 2029, and the trial metadata says so.

    **This must not touch the model or the data.** `runtime.seed` is what seeds
    initialisation and batch order, it lives on the base run spec, and nothing
    here writes it -- so two trials with identical params still have identical
    `run_id`s and are still the same experiment. Varying the model seed per
    worker would make every trial's result depend on which worker happened to
    pull it, which is the one thing a search must never do.
    """
    return int(seed) + int(worker_id or 0)


def make_sampler(*, seed: int, concurrent_trials: int = 1,
                 sampler: str = "tpe",
                 startup_trials: Optional[int] = None,
                 worker_id: Optional[int] = None
                 ) -> optuna.samplers.BaseSampler:
    """The sampler this study declares, seeded for this worker.

    `concurrent_trials` is the study spec's fleet size, which is also what
    spawns the processes -- so constant_liar is on exactly when a fleet actually
    exists. It used to come from `n_jobs`, a field that launched nothing, so a
    study could run 8 workers with constant_liar off and have all 8 propose from
    an identical history.

    `sampler` and `startup_trials` come from the StudySpec, so every worker
    reconstructs the same sampler. Defaults reproduce exactly what existed
    before those fields: independent TPE at `DEFAULT_STARTUP_TRIALS`.
    """
    if sampler not in SAMPLERS:
        raise ValueError(
            f"unknown sampler {sampler!r}; expected one of {list(SAMPLERS)}")
    resolved_seed = sampler_seed_for(seed, worker_id)
    if sampler == "random":
        if startup_trials is not None:
            raise ValueError(
                "RandomSampler has no n_startup_trials -- every trial is drawn "
                "the same way. Passing one would be silently ignored, so it is "
                "refused instead.")
        return optuna.samplers.RandomSampler(seed=resolved_seed)

    startup = (DEFAULT_STARTUP_TRIALS if startup_trials is None
               else int(startup_trials))
    if startup < 1:
        raise ValueError(f"sampler startup_trials must be >= 1, got {startup}")
    kwargs = {"seed": resolved_seed, "n_startup_trials": startup}
    if sampler == "tpe_multivariate":
        # `group=False` deliberately. `group=True` partitions the space by which
        # parameters co-occur, which is the right answer for a DYNAMIC space and
        # the wrong one here: this space is static and complete, so grouping
        # would only split the joint estimate `multivariate=True` exists to fit
        # -- and `sample_relative` drops `single()` distributions under
        # group=True, which is precisely how our fixed axes are spelled.
        kwargs["multivariate"] = True
    if concurrent_trials > 1:
        kwargs["constant_liar"] = True
    # BOTH `multivariate` and `constant_liar` are flagged experimental by optuna.
    # Opting in deliberately -- see the module docstring -- and scoping the filter
    # to this one construction so no other optuna warning is hidden. Silencing it
    # at the one opt-in site rather than globally is the whole point: a study that
    # does not ask for either still sees every warning optuna raises.
    #
    # This is unconditional now. It used to wrap only the constant_liar branch,
    # which meant `sampler: tpe_multivariate` at `concurrent_trials: 1` printed an
    # ExperimentalWarning per worker per launch -- noise that trains people to
    # ignore the output the plan printer is trying to be read in.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
        return optuna.samplers.TPESampler(**kwargs)


def prune_startup_trials_for(n_trials: Optional[int]) -> int:
    """How many completions to wait for before pruning anything.

    Delegates to `studyspec.derived_prune_startup_trials`, which is where the
    formula now lives: `__main__`'s plan printer needs the same number and must
    not import optuna, so the pure arithmetic sits below the optuna line and this
    stays as the name the rest of this module already used.
    """
    return derived_prune_startup_trials(n_trials)


def make_pruner(*, prune_after_step: int = DEFAULT_PRUNE_AFTER_STEP,
                prune_startup_trials: Optional[int] = None,
                n_trials: Optional[int] = None,
                logging_steps: int = DEFAULT_LOGGING_STEPS
                ) -> optuna.pruners.BasePruner:
    """A median pruner matched to the trainer's actual reporting cadence.

    `prune_startup_trials` overrides the size-derived default when a caller wants
    to pin it (the tests do).
    """
    startup = (prune_startup_trials if prune_startup_trials is not None
               else prune_startup_trials_for(n_trials))
    return optuna.pruners.MedianPruner(
        n_startup_trials=startup,
        n_warmup_steps=prune_after_step,
        # Only when a report can exist -- see the module docstring.
        interval_steps=max(1, int(logging_steps)),
        n_min_trials=PRUNE_MIN_TRIALS,
    )


def make_storage(study_dir, storage_url: Optional[str] = None):
    """A journal file under `study_dir`, or an explicit RDB URL if given."""
    if storage_url:
        return storage_url
    study_dir = Path(study_dir)
    study_dir.mkdir(parents=True, exist_ok=True)
    journal = study_dir / "optuna_journal.log"
    try:
        backend = optuna.storages.journal.JournalFileBackend(str(journal))
    except AttributeError as exc:                          # pragma: no cover
        raise RuntimeError(
            "JournalStorage requires optuna 4.x (optuna 3 spelled it "
            "JournalFileStorage). Install a 4.x release, or pass an explicit "
            "storage URL.") from exc
    return optuna.storages.JournalStorage(backend)


def create_study(*, study_name: str, study_dir, seed: int = 2026,
                 concurrent_trials: int = 1,
                 sampler: str = "tpe",
                 sampler_startup_trials: Optional[int] = None,
                 worker_id: Optional[int] = None,
                 prune_after_step: int = DEFAULT_PRUNE_AFTER_STEP,
                 prune_startup_trials: Optional[int] = None,
                 n_trials: Optional[int] = None,
                 logging_steps: int = DEFAULT_LOGGING_STEPS,
                 direction: str = "minimize",
                 storage_url: Optional[str] = None) -> optuna.study.Study:
    """Create or reattach to the study named `study_name` under `study_dir`.

    `load_if_exists` is what makes a search resumable: rerunning the driver
    against the same directory continues the study rather than starting a second
    one that knows nothing about the GPU-hours already spent.

    `worker_id` reaches only the SAMPLER's seed (see `sampler_seed_for`), so two
    workers attached to this journal agree about the space, the sampler class,
    the pruner and the model seed, and differ only in which random stream their
    proposals come from.
    """
    return optuna.create_study(
        study_name=study_name,
        storage=make_storage(study_dir, storage_url),
        sampler=make_sampler(seed=seed, concurrent_trials=concurrent_trials,
                             sampler=sampler,
                             startup_trials=sampler_startup_trials,
                             worker_id=worker_id),
        pruner=make_pruner(prune_after_step=prune_after_step,
                           prune_startup_trials=prune_startup_trials,
                           n_trials=n_trials,
                           logging_steps=logging_steps),
        # From the caller, not hardcoded. It WAS hardcoded, which made
        # `StudySpec.direction` inert -- a study declaring `maximize` was
        # silently minimized. `StudySpec` now permits only "minimize", so this
        # is defence for the day that changes rather than a live knob.
        direction=direction,
        load_if_exists=True,
    )


def is_anchor(trial) -> bool:
    """Was this trial one of the curated designs rather than a sampled point?

    Anchors are the reference set a median pruner needs, so the driver lets them
    finish by default. Reading it off a user attr means a resumed study can still
    tell which of its completed trials were hand-chosen.
    """
    return bool(getattr(trial, "user_attrs", {}).get(ANCHOR_ATTR))


def enqueue_anchors(study: optuna.study.Study, designs: Iterable[Any],
                    base_model: KoopmanLMConfig,
                    space: Mapping[str, Mapping[str, Any]], *,
                    base_lr: float) -> int:
    """Queue each design ahead of any sampled trial. Returns how many were added.

    `skip_if_exists=True` makes this idempotent, so resuming a study does not
    re-run anchors that already ran -- the expensive mistake this guards.
    """
    count = 0
    for design in designs:
        params = resolve_design(design, base_model, space, base_lr=base_lr)
        study.enqueue_trial(params, user_attrs={ANCHOR_ATTR: design.name},
                            skip_if_exists=True)
        count += 1
    return count
