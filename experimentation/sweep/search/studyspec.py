"""A study, written down: `configs/search/<name>.yaml` -> `StudySpec`.

Mirrors `sweep/spec.py`'s `SweepSpec`, for the same reason and with the same
discipline: a frozen dataclass, `__post_init__` validation, and a content hash
used for **provenance, not identity**.

**Why this file has to exist, given the optuna journal already records so much.**

Measured against optuna 4.9.0's `JournalStorage`: the journal records
`study_name`, `directions`, per-trial params **with their distributions**,
`values`, `intermediate_value` and `step` for every pruning report, `state`,
`user_attr`, timestamps and `worker_id`. So the search space and every trial's
full trajectory are already durable, and this spec deliberately does NOT
re-declare the space -- `space.py` owns that, once.

What the journal does not record is everything optuna never sees. From optuna's
side a trial is a black box from params to a float; it does not know a model is
being trained. So `max_steps`, which base spec, which shard, the launcher --
constants of *our* objective function -- are invisible to it by construction, not
by omission.

And one thing it does use but does not persist: **the sampler and pruner**. Those
are reconstructed by whoever opens the study, and reopening with a different
pruner is accepted silently. That is a live footgun, and pinning them in a
committed file is the fix.

**The sharper argument is multi-worker agreement.** Parallelism here is N
processes sharing one journal file (`study.py` rejects `optimize(n_jobs=N)`'s
thread pool for exactly this reason). If `max_steps` lives in a command-line
argument, worker 3 can silently disagree with worker 1 -- and optuna will happily
mix trials trained for 300 steps with trials trained for 3000 into one study,
because the journal records no step budget to notice the disagreement with. A
committed spec is what makes N workers *provably* the same study rather than
hopefully.

**Above the optuna line.** This module imports no optuna, so a study can be
authored, validated and diffed on a machine that has never installed it -- the
same property that lets `anchors.py` ship a curated design set as an ordinary
`cells:` sweep.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

__all__ = ["DEFAULT_STARTUP_TRIALS", "SAMPLERS", "StudySpec",
           "derived_prune_startup_trials", "load_study_spec", "study_id"]

#: TPE samples randomly until this many trials have finished, and MedianPruner's
#: floor when a study size is unknown. Lives here rather than in `study.py`
#: because `__main__`'s plan printer needs it and `__main__` must not import
#: optuna -- that is what lets `--dry_run` print a full plan on a machine that
#: has never installed it.
DEFAULT_STARTUP_TRIALS = 4


def derived_prune_startup_trials(n_trials: Optional[int]) -> int:
    """How many completions to wait for before pruning, when unspecified.

    `min(6, max(3, n_trials // 3))`, matching the original harness: scales with
    the study so a long study does not spend a third of itself unprunable, and
    floors at 3 so a short one never prunes off a single datapoint. The cap at 6
    is why `_print_plan` reports WAVES rather than workers-vs-startup -- any
    fleet of 6+ would compare unfavourably to this no matter how large the study.
    """
    if not n_trials or n_trials < 1:
        return DEFAULT_STARTUP_TRIALS
    return min(6, max(3, n_trials // 3))

# Only "minimize" is meaningful for a loss objective, and multi-objective is a
# distinct decision rather than a knob: optuna 4.9 raises NotImplementedError
# from Trial.report under multiple directions, so a Pareto study cannot prune at
# all. Since pruning is what makes a study affordable, the recommended route is
# to search single-objective and compute the front post hoc from trials.csv,
# which already records every metric per trial.
_DIRECTIONS = ("minimize", "maximize")

_LAUNCHERS = ("slurm", "local")

#: The samplers a study may name. Deliberately short and deliberately explicit.
#:
#: ``tpe`` is what every study got before this field existed -- optuna's default
#: *independent* TPE, which models one marginal per parameter. It stays the
#: default so no committed study changes behaviour by omission.
#:
#: ``tpe_multivariate`` is the one an INTERACTION study wants, and the reason
#: this field exists at all. Independent TPE cannot represent "rank 32 wants a
#: heavier ridge": it fits p(ridge | good) and p(rank | good) separately, so a
#: joint effect appears in neither marginal. `multivariate=True` fits the joint
#: KDE instead. It costs more per proposal and needs more completed trials
#: before the joint estimate means anything, which is what
#: `sampler_startup_trials` is for.
#:
#: ``random`` is the control. A study that cannot beat its own random baseline
#: has not learned anything, and having the baseline be one word in the study
#: file -- rather than a different code path -- is what makes it cheap enough to
#: actually run.
_SAMPLERS = SAMPLERS = ("tpe", "tpe_multivariate", "random")

#: The weights `metrics.objective_from_metrics` accepts. Duplicated here rather
#: than introspected, because importing metrics.py from a module that must stay
#: importable with optuna absent is exactly the coupling this package's layering
#: exists to prevent -- and `test_objective_weights_stay_in_step.py` pins the two
#: lists together, so the duplicate cannot drift silently.
_OBJECTIVE_WEIGHTS = {
    "parameter_penalty",
    "throughput_penalty",
    "target_tokens_per_sec",
    "ska_delta_reward",
    "ska_delta_cap",
}

#: Accepted by `objective_from_metrics` and NOT declarable in a study file: they
#: are measured per trial, not chosen once. Named separately so the error can say
#: why rather than "unknown key".
_OBJECTIVE_DERIVED = {"param_count", "baseline_param_count"}

#: Keys that existed and no longer do. Named explicitly so a study written
#: against the old spelling fails with the reason rather than with "unknown
#: key", which reads like a typo and invites deleting the line instead of
#: renaming it.
_RENAMED = {"n_jobs": "concurrent_trials", "workers": "concurrent_trials"}
_RENAME_REASONS = {
    "n_jobs": (
        "`n_jobs` only flipped the sampler's constant_liar and launched nothing, "
        "so a study could declare n_jobs: 1 while 8 workers ran against it -- all "
        "8 proposing from the same history. `concurrent_trials` is the fleet size: "
        "it spawns the processes AND sets constant_liar, so the two cannot "
        "disagree. Rename the key and set it to the number of trials you want in "
        "flight at once."),
    "workers": (
        "`workers` was this field's spelling for one day and collided with a "
        "field that already existed: `RuntimeSpec.workers` is the DATALOADER "
        "worker count (experimentation/run/spec.py). A study that also carries a "
        "runtime section would have had the two, meaning entirely different "
        "things, a few lines apart:\n"
        "    workers: 8          # trials in flight\n"
        "    runtime:\n"
        "      workers: 4        # torch DataLoader workers\n"
        "which is exactly the confusion the rename away from `n_jobs` existed to "
        "prevent. `concurrent_trials` names the thing it counts and collides with "
        "nothing. Rename the key; the value means the same."),
}


@dataclass(frozen=True)
class StudySpec:
    """A parsed configs/search/<name>.yaml. Pure data; touches no filesystem."""

    name: str
    #: Path to the base run spec every trial is a perturbation of. Left as the
    #: literal string from the YAML and resolved later against the process cwd,
    #: matching `python -m experimentation.run <spec.yaml>`'s own convention.
    base: str
    #: The study's TARGET size, not "this many more" -- resuming a 15-trial study
    #: that finished 10 runs 5. Matches drive()'s own semantics.
    n_trials: int
    #: Steps per trial. The single most important field to pin: it is invisible
    #: to the journal, and two workers disagreeing about it produce trials that
    #: are not comparable while looking like one study.
    max_steps: int
    run_root: str = "runs"
    launcher: str = "slurm"
    direction: str = "minimize"
    #: Where the shared journal lives. In the spec rather than on the command
    #: line so two workers cannot open two different journals and believe they
    #: are collaborating.
    storage: Optional[str] = None
    seed: int = 2026
    #: How many trials run CONCURRENTLY -- and, unlike the `n_jobs` this
    #: replaces, an actual fleet rather than a hint. `__main__` spawns this many
    #: worker processes and the sampler's `constant_liar` follows from it, so the
    #: two can no longer disagree.
    #:
    #: The name has now been wrong twice, which is worth recording. `n_jobs`
    #: borrowed optuna's spelling for a thread pool `study.py` explicitly
    #: REJECTS. `workers` then collided with `RuntimeSpec.workers`, the
    #: DATALOADER worker count -- so a study carrying a runtime section would
    #: have had two unrelated `workers` a few lines apart. `concurrent_trials`
    #: names what it counts, and nothing else in the repo counts that.
    #:
    #: Independent of the GPU count on purpose. At the 4m geometry a trial runs
    #: at roughly 1% of an H100 (128 GFLOP per micro-step against a measured
    #: 0.0224 s), so more concurrent trials than GPUs is a reasonable setting
    #: rather than oversubscription; placement is derived, never declared. The
    #: worker processes are long-lived and pull the next trial as soon as one
    #: ends, so the trial queue -- not the GPU assignment -- balances the load.
    concurrent_trials: int = 1
    prune_after_step: int = 200
    logging_steps: int = 10
    #: Optional anchor designs to enqueue before adaptive sampling starts.
    design_file: Optional[str] = None
    #: How long to wait for a trial's objective before giving up on it.
    #:
    #: Load-bearing once the driver stops blocking on submit. With a blocking
    #: submit, a crashed trial surfaced as a non-zero exit from subprocess.run;
    #: without one, it surfaces only as "the objective never appeared". No
    #: timeout means a single dead trial hangs the whole study indefinitely.
    #:
    #: 2 hours by default -- generous next to the minutes a small trial takes, and
    #: still far short of a night.
    trial_timeout_seconds: float = 7200.0
    #: Held-out shard for a trial's own end-of-run scoring. Defaults to the base
    #: spec's training shard, which is fine for a proxy objective but means the
    #: score is not held out -- set it for anything whose ranking you trust.
    eval_data_dir: Optional[str] = None
    backend_policy: str = "exact_invchol"
    seq_len: Optional[int] = None
    batch_ladder: bool = False
    #: Objective weights, passed through to metrics.objective_from_metrics.
    #: Every one defaults to zero there, so an empty dict means "minimise the
    #: measured loss and nothing else".
    objective: Dict[str, Any] = field(default_factory=dict)
    #: Which sampler reconstructs this study. See `_SAMPLERS`.
    #:
    #: In the SPEC rather than on the command line for the reason the module
    #: docstring gives about the pruner: the sampler is reconstructed by whoever
    #: opens the study and optuna accepts a different one silently. Two workers
    #: attached to one journal with different samplers are not running one
    #: study, and nothing in the journal records the disagreement.
    sampler: str = "tpe"
    #: TPE's `n_startup_trials`: how many finished trials before TPE stops
    #: drawing at random and starts modelling. None keeps `study.py`'s own
    #: default (4), which is what every study got before this field existed.
    #:
    #: Worth setting explicitly for `tpe_multivariate`, because the joint KDE
    #: needs far more evidence than a product of marginals does -- with 4
    #: startup trials in a 9-dimensional space the "joint" estimate is noise
    #: wearing a joint distribution's clothes.
    #:
    #: NOTE optuna semantics, verified rather than assumed (test_sampler_and_
    #: pruner_startup.py): TPE counts trials in states COMPLETE **and PRUNED**.
    #: So a pruned trial advances TPE past its startup window even though it
    #: never produced a final objective.
    sampler_startup_trials: Optional[int] = None
    #: MedianPruner's `n_startup_trials`. None keeps the size-derived default
    #: `min(6, max(3, n_trials // 3))`.
    #:
    #: NOTE optuna semantics, and NOT the same rule as the sampler's: the pruner
    #: counts **COMPLETE only**. A study whose first wave is mostly pruned
    #: therefore advances the sampler and not the pruner.
    prune_startup_trials: Optional[int] = None
    #: Per-study replacements for declared search axes: `{axis: declaration}`,
    #: where a declaration is the same plain data `space.py` emits --
    #: `{"kind": "categorical", "choices": [...]}` or
    #: `{"kind": "float", "low": ..., "high": ..., "log": true}`.
    #:
    #: An entry REPLACES the base declaration wholesale, including the baseline
    #: containment `search_space()` folds in. That is the point -- a study that
    #: wants ridge over [3e-3, 3e-2] and nothing wider has to be able to say so
    #: -- and it is also the hazard, so `space.restrict_space` re-validates every
    #: replacement against the axis's real domain (rank % 8, the layer-index
    #: capacity, the declared placement names) instead of trusting the file.
    #:
    #: Absent by default: an empty dict resolves to exactly the space that
    #: existed before this field.
    search_axes: Dict[str, Any] = field(default_factory=dict)
    #: Parameters pinned to a single value for this study: `{axis: value}`.
    #:
    #: Becomes a validated SINGLETON CATEGORICAL distribution rather than being
    #: dropped from the space. Three things follow from that choice and none of
    #: them does from dropping it: `params_to_overrides` still receives every
    #: parameter it requires, the value is recorded in the journal and therefore
    #: in trials.csv, and `anchors.resolve_design` snaps to it automatically so
    #: an anchor cannot disagree with the study about a fixed axis.
    #:
    #: Deliberately NOT a generic override dict. An unvalidated escape hatch is
    #: exactly how a study would come to declare `ska_rank: 20` -- accepted here,
    #: fatal inside KoopmanLMConfig's own assertion, on a GPU, after a run
    #: directory had been materialized.
    fixed_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.name:
            raise ValueError("StudySpec.name is required")
        if not self.base:
            raise ValueError(
                "StudySpec.base is required -- a study is a perturbation of some "
                "base run spec, and without one there is nothing to perturb")
        if self.n_trials < 1:
            raise ValueError(f"n_trials must be >= 1, got {self.n_trials}")
        if self.max_steps < 1:
            raise ValueError(f"max_steps must be >= 1, got {self.max_steps}")
        if self.direction not in _DIRECTIONS:
            raise ValueError(
                f"direction={self.direction!r}; expected one of {list(_DIRECTIONS)}. "
                "Multiple objectives are not a list here on purpose: optuna 4.9 "
                "cannot prune a multi-objective study at all, so a Pareto front "
                "is computed post hoc from trials.csv instead.")
        if self.launcher not in _LAUNCHERS:
            raise ValueError(
                f"launcher={self.launcher!r}; expected one of {list(_LAUNCHERS)}")
        if self.concurrent_trials < 1:
            raise ValueError(
                f"concurrent_trials must be >= 1, got {self.concurrent_trials}")
        if self.prune_after_step < 0:
            raise ValueError("prune_after_step must be >= 0")
        if self.logging_steps < 1:
            raise ValueError(
                "logging_steps must be >= 1 -- it becomes the pruner's "
                "interval_steps, which optuna requires to be positive")
        if self.trial_timeout_seconds <= 0:
            raise ValueError(
                "trial_timeout_seconds must be positive -- without a timeout one "
                "dead trial hangs the study forever, and the driver no longer "
                "blocks on submit, so a crash has no other way to surface")
        if self.prune_after_step >= self.max_steps:
            raise ValueError(
                f"prune_after_step={self.prune_after_step} >= max_steps="
                f"{self.max_steps}: no trial could ever be pruned, so every "
                "trial would run to completion while appearing to be pruned-"
                "enabled. Lower it, or say so by setting it to 0.")
        self._validate_sampler()
        self._validate_axis_declarations()
        self._validate_objective()

    # ---------------------------------------------------------- objective ----

    def _validate_objective(self):
        if not isinstance(self.objective, dict):
            raise ValueError(
                f"objective must be a mapping of weight name -> number, got "
                f"{type(self.objective).__name__}")
        unknown = sorted(set(self.objective) - _OBJECTIVE_WEIGHTS - _OBJECTIVE_DERIVED)
        if unknown:
            raise ValueError(
                f"objective has unknown key(s) {unknown}. Known weights: "
                f"{sorted(_OBJECTIVE_WEIGHTS)}. They are splatted straight into "
                f"metrics.objective_from_metrics, which takes keyword arguments "
                f"only -- so a misspelled weight used to raise TypeError deep "
                f"inside the objective reader, which the driver's own "
                f"`except Exception` turns into a FAILED trial with no visible "
                f"cause.")
        derived = sorted(set(self.objective) & _OBJECTIVE_DERIVED)
        if derived:
            raise ValueError(
                f"objective declares {derived}, which are DERIVED PER TRIAL and "
                f"not weights: a parameter count is a property of the config the "
                f"sampler proposed, so a constant written here would apply the "
                f"same count to every trial and make the penalty a constant "
                f"offset -- i.e. no penalty at all. Set parameter_penalty and "
                f"the driver supplies both counts from the resolved spec.")
        for key, value in sorted(self.objective.items()):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(
                    f"objective[{key!r}]={value!r} must be numeric")

    # ------------------------------------------------------------ sampler ----

    def _validate_sampler(self):
        if self.sampler not in _SAMPLERS:
            raise ValueError(
                f"sampler={self.sampler!r}; expected one of {list(_SAMPLERS)}. "
                "'tpe' is optuna's independent TPE (one marginal per parameter), "
                "'tpe_multivariate' fits the joint and is what an INTERACTION "
                "study needs, 'random' is the control a study should be able to "
                "beat.")
        if self.sampler_startup_trials is not None:
            if self.sampler == "random":
                raise ValueError(
                    "sampler_startup_trials is meaningless with sampler='random': "
                    "RandomSampler has no startup window -- every trial is drawn "
                    "the same way -- so optuna would ignore the number silently "
                    "and the study would look configured when it was not. Drop "
                    "the key, or choose 'tpe'/'tpe_multivariate'.")
            if self.sampler_startup_trials < 1:
                raise ValueError(
                    f"sampler_startup_trials must be >= 1, got "
                    f"{self.sampler_startup_trials}")
            if self.sampler_startup_trials >= self.n_trials:
                raise ValueError(
                    f"sampler_startup_trials={self.sampler_startup_trials} >= "
                    f"n_trials={self.n_trials}: TPE would still be drawing at "
                    f"random when the study ended, so this is a random search "
                    f"that reports itself as TPE. Raise n_trials or lower the "
                    f"startup count.")
        if self.prune_startup_trials is not None:
            if self.prune_startup_trials < 1:
                raise ValueError(
                    f"prune_startup_trials must be >= 1, got "
                    f"{self.prune_startup_trials}")
            if self.prune_startup_trials >= self.n_trials:
                raise ValueError(
                    f"prune_startup_trials={self.prune_startup_trials} >= "
                    f"n_trials={self.n_trials}: MedianPruner needs that many "
                    f"COMPLETED trials before it prunes anything, so nothing "
                    f"could ever be pruned -- the same failure "
                    f"prune_after_step >= max_steps is rejected for.")

    # ------------------------------------------------- axes and fixed values ----

    #: The shape of one declaration, per kind. Structural only: whether a rank of
    #: 20 is legal is a question about KoopmanLMConfig, and it is answered in
    #: `space.restrict_space`, which is the module that has the base model.
    _DECLARATION_KEYS = {
        "categorical": ({"kind", "choices"}, {"kind", "choices"}),
        "float": ({"kind", "low", "high"}, {"kind", "low", "high", "log"}),
    }

    def _validate_axis_declarations(self):
        for holder, value in (("search_axes", self.search_axes),
                              ("fixed_params", self.fixed_params)):
            if not isinstance(value, dict):
                raise ValueError(
                    f"{holder} must be a mapping of axis name -> "
                    f"{'declaration' if holder == 'search_axes' else 'value'}, "
                    f"got {type(value).__name__}")

        both = sorted(set(self.search_axes) & set(self.fixed_params))
        if both:
            raise ValueError(
                f"{both} appear in BOTH search_axes and fixed_params. Fixing a "
                f"parameter already means it takes exactly one value; declaring "
                f"a distribution for it as well states two different intentions "
                f"and there is no reading of the pair that is not a mistake.")

        for name, declaration in sorted(self.search_axes.items()):
            self._validate_one_declaration(name, declaration)

        for name, value in sorted(self.fixed_params.items()):
            if isinstance(value, bool) or not isinstance(value, (int, float, str)):
                raise ValueError(
                    f"fixed_params[{name!r}] must be a single number or string, "
                    f"got {type(value).__name__} ({value!r}). A list here would "
                    f"be a search axis wearing the wrong key -- use search_axes.")

    def _validate_one_declaration(self, name, declaration):
        where = f"search_axes[{name!r}]"
        if not isinstance(declaration, dict):
            raise ValueError(
                f"{where} must be a mapping like "
                f"{{'kind': 'categorical', 'choices': [...]}}, got "
                f"{type(declaration).__name__}")
        kind = declaration.get("kind")
        if kind not in self._DECLARATION_KEYS:
            raise ValueError(
                f"{where}: kind={kind!r}; expected 'categorical' or 'float' -- "
                f"the two shapes space.py emits and study.py can translate into "
                f"optuna distributions.")
        required, allowed = self._DECLARATION_KEYS[kind]
        missing = sorted(required - set(declaration))
        if missing:
            raise ValueError(f"{where}: {kind} declaration is missing {missing}")
        unknown = sorted(set(declaration) - allowed)
        if unknown:
            raise ValueError(
                f"{where}: unknown key(s) {unknown} for a {kind} declaration; "
                f"allowed keys are {sorted(allowed)}")

        if kind == "categorical":
            choices = declaration["choices"]
            if not isinstance(choices, list) or not choices:
                raise ValueError(
                    f"{where}: 'choices' must be a non-empty list, got "
                    f"{choices!r}. An empty list is not 'search nothing' -- "
                    f"optuna's CategoricalDistribution raises on it, mid-study.")
            if len(set(map(repr, choices))) != len(choices):
                raise ValueError(
                    f"{where}: 'choices' contains duplicates ({choices!r}). "
                    f"A duplicate silently doubles that value's prior weight, "
                    f"which is a change to the experiment nobody wrote down.")
            return

        low, high = declaration["low"], declaration["high"]
        for key, bound in (("low", low), ("high", high)):
            if isinstance(bound, bool) or not isinstance(bound, (int, float)):
                raise ValueError(
                    f"{where}: {key}={bound!r} must be numeric. PyYAML parses "
                    f"'3e-3' as a STRING (no decimal point in the mantissa); "
                    f"write 3.0e-3 or 0.003.")
        if float(low) >= float(high):
            raise ValueError(
                f"{where}: low={low} must be strictly less than high={high}. "
                f"low == high is a fixed value, not a range -- put it in "
                f"fixed_params, where it becomes a validated singleton instead "
                f"of a degenerate interval optuna has to special-case.")
        if declaration.get("log"):
            if not isinstance(declaration["log"], bool):
                raise ValueError(f"{where}: 'log' must be true or false")
            if float(low) <= 0.0:
                raise ValueError(
                    f"{where}: log=true requires low > 0, got low={low}")


def load_study_spec(path) -> StudySpec:
    """Read configs/search/<name>.yaml into a StudySpec.

    Unknown keys are a hard error rather than a shrug: a study is expensive, and
    a misspelled `max_step` silently taking the default would produce a whole
    study at the wrong budget.
    """
    raw = yaml.safe_load(Path(path).read_text()) or {}
    known = {f for f in StudySpec.__dataclass_fields__}
    for old, new in _RENAMED.items():
        if old in raw:
            raise ValueError(
                f"{path}: `{old}` was renamed to `{new}`. {_RENAME_REASONS[old]}")
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(
            f"{path}: unknown key(s) {unknown}. Known keys: {sorted(known)}")
    missing = [k for k in ("name", "base", "n_trials", "max_steps") if k not in raw]
    if missing:
        raise ValueError(f"{path}: missing required key(s) {missing}")
    return StudySpec(**raw)


def study_id(study_spec: StudySpec) -> str:
    """A content hash of the study's own declaration.

    Stamped into each trial's materialized spec.yaml alongside `study_name`, so
    results can be grouped by which study produced them. Deliberately NOT part
    of run_id or group_id, for the reason `sweep/spec.py::sweep_id` gives: study
    membership is metadata about how a run was launched, not a scientific input.
    Two studies that happen to propose the same config must produce the same
    run_id, or the content-addressed run directory stops being content-addressed.
    """
    blob = json.dumps(asdict(study_spec), sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:8]
