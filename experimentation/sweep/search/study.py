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

__all__ = ["to_distribution", "to_distributions", "make_sampler", "make_pruner",
           "prune_startup_trials_for",
           "make_storage", "create_study", "enqueue_anchors", "is_anchor",
           "ANCHOR_ATTR"]

ANCHOR_ATTR = "anchor_name"

# TPE samples randomly until this many trials have completed; with a curated
# anchor set the anchors themselves fill the startup window, so TPE begins
# modelling from hand-chosen points rather than from noise.
DEFAULT_STARTUP_TRIALS = 4

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


def make_sampler(*, seed: int, n_jobs: int = 1) -> optuna.samplers.BaseSampler:
    """A seeded TPE sampler, made parallel-safe when more than one worker runs."""
    if n_jobs <= 1:
        return optuna.samplers.TPESampler(seed=seed,
                                          n_startup_trials=DEFAULT_STARTUP_TRIALS)
    with warnings.catch_warnings():
        # constant_liar is marked experimental. Opting in deliberately -- see the
        # module docstring -- and scoping the filter to this one call so no other
        # optuna warning is hidden.
        warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
        return optuna.samplers.TPESampler(seed=seed,
                                          n_startup_trials=DEFAULT_STARTUP_TRIALS,
                                          constant_liar=True)


def prune_startup_trials_for(n_trials: Optional[int]) -> int:
    """How many completions to wait for before pruning anything.

    `min(6, max(3, n_trials // 3))`, matching the original harness: scales with
    the study so a long study does not spend a third of itself unprunable, and
    floors at 3 so a short one never prunes off a single datapoint.
    """
    if not n_trials or n_trials < 1:
        return DEFAULT_STARTUP_TRIALS
    return min(6, max(3, n_trials // 3))


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
                 n_jobs: int = 1,
                 prune_after_step: int = DEFAULT_PRUNE_AFTER_STEP,
                 prune_startup_trials: Optional[int] = None,
                 n_trials: Optional[int] = None,
                 logging_steps: int = DEFAULT_LOGGING_STEPS,
                 storage_url: Optional[str] = None) -> optuna.study.Study:
    """Create or reattach to the study named `study_name` under `study_dir`.

    `load_if_exists` is what makes a search resumable: rerunning the driver
    against the same directory continues the study rather than starting a second
    one that knows nothing about the GPU-hours already spent.
    """
    return optuna.create_study(
        study_name=study_name,
        storage=make_storage(study_dir, storage_url),
        sampler=make_sampler(seed=seed, n_jobs=n_jobs),
        pruner=make_pruner(prune_after_step=prune_after_step,
                           prune_startup_trials=prune_startup_trials,
                           n_trials=n_trials,
                           logging_steps=logging_steps),
        direction="minimize",
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
