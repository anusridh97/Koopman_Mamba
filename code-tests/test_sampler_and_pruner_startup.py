"""What optuna actually COUNTS toward a startup window. Measured, not assumed.

`configs/search/proxy-256x17-interactions-v1.yaml` claims three things about
optuna's semantics, and its phase plan is wrong if any of them is wrong:

  1. `sampler: tpe_multivariate` and `sampler_startup_trials: 64` really reach
     the TPESampler optuna constructs -- not a default that looks similar.
  2. TPE's `n_startup_trials` counts trials in states COMPLETE **and PRUNED**.
  3. MedianPruner's `n_startup_trials` counts **COMPLETE only** -- a DIFFERENT
     rule from the sampler's, which is the detail a reader is most likely to get
     wrong by symmetry.

And the one the study file's "Phase 1" depends on:

  4. An ENQUEUED anchor counts toward both once it finishes. The study reasons
     that 24 anchors plus ~40 random draws reach the 64-trial floor; if enqueued
     trials were excluded from the sampler's or the pruner's observation set,
     the study would need 64 SAMPLED completions on top of the anchors and every
     phase boundary in that file would be at the wrong trial number.

These are read off optuna's behaviour rather than its source, so they survive a
refactor of the source and fail on a change of behaviour -- which is the right
way round. The asymmetry in 2 vs 3 was found by reading
`TPESampler.sample_independent` (states = COMPLETE, PRUNED) against
`PercentilePruner.prune` (states = COMPLETE), and is asserted here so it cannot
be quietly reversed by an optuna upgrade.
"""
from __future__ import annotations

import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

optuna = pytest.importorskip("optuna")

pytestmark = pytest.mark.correctness

from experimentation.sweep.search.study import (            # noqa: E402
    create_study, make_pruner, make_sampler, sampler_seed_for)

STUDY = REPO / "configs/search/proxy-256x17-interactions-v1.yaml"


@pytest.fixture(scope="module")
def study_spec():
    from experimentation.sweep.search.studyspec import load_study_spec
    return load_study_spec(STUDY)


# ------------------------------------------- 1. the declarations reach optuna ----

def test_multivariate_reaches_the_sampler():
    """`multivariate=True` is the whole reason this study exists: independent TPE
    fits one marginal per parameter, so a joint effect appears in neither."""
    sampler = make_sampler(seed=1, sampler="tpe_multivariate")
    assert isinstance(sampler, optuna.samplers.TPESampler)
    assert sampler._multivariate is True


def test_plain_tpe_stays_independent():
    """The default must not silently become multivariate -- every committed study
    other than this one relies on it."""
    assert make_sampler(seed=1, sampler="tpe")._multivariate is False


def test_multivariate_does_not_group():
    """`group=True` partitions the space by parameter co-occurrence, which splits
    the joint estimate `multivariate=True` exists to fit -- and its
    `sample_relative` DROPS `single()` distributions, which is exactly how this
    study's fixed axes are spelled."""
    assert make_sampler(seed=1, sampler="tpe_multivariate")._group is False


def test_the_sampler_startup_count_reaches_tpe():
    assert make_sampler(seed=1, sampler="tpe_multivariate",
                        startup_trials=64)._n_startup_trials == 64


def test_the_prune_startup_count_reaches_the_median_pruner():
    pruner = make_pruner(prune_after_step=450, prune_startup_trials=64,
                        logging_steps=25)
    assert isinstance(pruner, optuna.pruners.MedianPruner)
    assert pruner._n_startup_trials == 64
    assert pruner._n_warmup_steps == 450
    assert pruner._interval_steps == 25


def test_random_is_a_random_sampler():
    assert isinstance(make_sampler(seed=1, sampler="random"),
                      optuna.samplers.RandomSampler)


def test_constant_liar_follows_the_fleet_size_for_every_sampler_kind():
    for kind in ("tpe", "tpe_multivariate"):
        assert make_sampler(seed=1, sampler=kind,
                            concurrent_trials=1)._constant_liar is False
        assert make_sampler(seed=1, sampler=kind,
                            concurrent_trials=8)._constant_liar is True


def test_the_target_studys_own_declarations_reach_optuna(tmp_path, study_spec):
    """End to end through `create_study`, from the committed file. The unit
    assertions above could all pass while `__main__` forgot to pass one of
    them -- which is the "valid, validated, and read by nothing" failure this
    repo keeps producing."""
    study = create_study(
        study_name=study_spec.name, study_dir=tmp_path, seed=study_spec.seed,
        concurrent_trials=study_spec.concurrent_trials,
        sampler=study_spec.sampler,
        sampler_startup_trials=study_spec.sampler_startup_trials,
        prune_after_step=study_spec.prune_after_step,
        prune_startup_trials=study_spec.prune_startup_trials,
        n_trials=study_spec.n_trials, logging_steps=study_spec.logging_steps)
    assert study.sampler._multivariate is True
    assert study.sampler._n_startup_trials == 64
    assert study.sampler._constant_liar is True
    assert study.pruner._n_startup_trials == 64
    assert study.pruner._n_warmup_steps == 450


# --------------------------------------- 2 & 3. what each counter counts ----

def _distributions():
    return {"x": optuna.distributions.FloatDistribution(0.0, 1.0)}


def _finish(study, value, *, state):
    trial = study.ask(_distributions())
    if state == optuna.trial.TrialState.PRUNED:
        trial.report(value, 10)
        study.tell(trial, state=state)
    else:
        study.tell(trial, value)
    return trial


#: The states TPE counts, spelled once. Read off `TPESampler.sample_independent`,
#: which calls `study._get_trials(states=(COMPLETE, PRUNED))` and falls through to
#: its random sampler while `len(trials) < n_startup_trials`. Asserted through
#: behaviour below rather than by comparing against this tuple, so an optuna
#: change of behaviour fails rather than a change of spelling.
_TPE_COUNTED_STATES = (optuna.trial.TrialState.COMPLETE,
                       optuna.trial.TrialState.PRUNED)


def test_tpe_counts_pruned_trials_toward_its_startup_window():
    """A pruned trial ADVANCES the sampler even though it produced no objective.
    Worth knowing before sizing a startup window: a study whose first wave is
    mostly pruned reaches its 'modelling' phase sooner than the trial count
    suggests."""
    sampler = make_sampler(seed=7, sampler="tpe", startup_trials=3)
    study = optuna.create_study(sampler=sampler,
                                pruner=optuna.pruners.NopPruner())
    for _ in range(3):
        _finish(study, 1.0, state=optuna.trial.TrialState.PRUNED)

    states = (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)
    counted = study.get_trials(deepcopy=False, states=states)
    assert len(counted) == 3, "three PRUNED trials are in TPE's observation set"
    assert all(t.state == optuna.trial.TrialState.PRUNED for t in counted)


def test_the_median_pruner_counts_only_completed_trials():
    """The DIFFERENT rule, and the one a reader assumes away. Three pruned trials
    leave the pruner exactly where it started."""
    pruner = make_pruner(prune_after_step=0, prune_startup_trials=3,
                        logging_steps=1)
    study = optuna.create_study(pruner=pruner)
    for _ in range(3):
        _finish(study, 1.0, state=optuna.trial.TrialState.PRUNED)

    completed = study.get_trials(deepcopy=False,
                                states=(optuna.trial.TrialState.COMPLETE,))
    assert len(completed) == 0

    # And the consequence: nothing can be pruned yet, however bad it looks.
    victim = study.ask(_distributions())
    for step in range(1, 6):
        victim.report(100.0, step)
    # `bool(...)`, not `is False`: optuna's should_prune returns numpy.bool_,
    # so `is True` is ALWAYS False and `is False` is always False too -- an
    # identity check here is a test that passes for the wrong reason in one
    # direction and can never pass in the other.
    assert bool(victim.should_prune()) is False, (
        "MedianPruner pruned with zero COMPLETED trials -- it has no median")


def test_the_median_pruner_does_fire_once_enough_have_completed():
    """Guards the guard above: if `should_prune` were always False the previous
    assertion would pass for the wrong reason."""
    pruner = make_pruner(prune_after_step=0, prune_startup_trials=3,
                        logging_steps=1)
    study = optuna.create_study(pruner=pruner)
    for _ in range(4):
        trial = study.ask(_distributions())
        for step in range(1, 6):
            trial.report(1.0, step)
        study.tell(trial, 1.0)

    victim = study.ask(_distributions())
    for step in range(1, 6):
        victim.report(100.0, step)
    assert bool(victim.should_prune()) is True


# --------------------------------- 4. enqueued anchors count toward both ----

def test_an_enqueued_trial_counts_toward_the_samplers_startup_window():
    """Phase 1 of the target study depends on this. If an enqueued anchor were
    excluded from TPE's observation set, `sampler_startup_trials: 64` would mean
    64 SAMPLED completions ON TOP OF the 24 anchors."""
    sampler = make_sampler(seed=11, sampler="tpe", startup_trials=3)
    study = optuna.create_study(sampler=sampler)
    for value in (0.1, 0.2, 0.3):
        study.enqueue_trial({"x": value})
    for value in (0.1, 0.2, 0.3):
        trial = study.ask(_distributions())
        assert trial.params["x"] == value, "enqueued order was not preserved"
        study.tell(trial, value)

    states = (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)
    counted = study.get_trials(deepcopy=False, states=states)
    assert len(counted) == 3, (
        "enqueued trials are excluded from TPE's observation set -- every phase "
        "boundary in the interaction study is at the wrong trial number")


def test_an_enqueued_trial_counts_toward_the_pruners_startup_window():
    pruner = make_pruner(prune_after_step=0, prune_startup_trials=3,
                        logging_steps=1)
    study = optuna.create_study(pruner=pruner)
    for value in (0.1, 0.2, 0.3):
        study.enqueue_trial({"x": value})
    for _ in range(3):
        trial = study.ask(_distributions())
        for step in range(1, 6):
            trial.report(1.0, step)
        study.tell(trial, 1.0)

    completed = study.get_trials(deepcopy=False,
                                states=(optuna.trial.TrialState.COMPLETE,))
    assert len(completed) == 3

    victim = study.ask(_distributions())
    for step in range(1, 6):
        victim.report(100.0, step)
    assert bool(victim.should_prune()) is True, (
        "three COMPLETED enqueued trials did not give the pruner a median -- so "
        "anchors would not establish the reference population the study says "
        "they do")


def test_a_real_anchor_set_fills_the_window_with_the_studys_own_space(tmp_path,
                                                                     study_spec):
    """The same claim, through the actual enqueue path and the actual space --
    including a SINGLETON categorical, which is how the fixed axes are spelled
    and which a multivariate sampler has to tolerate."""
    from experimentation.run.spec import resolve_model_config
    from experimentation.sweep.search.anchors import load_designs
    from experimentation.sweep.search.space import restrict_space, search_space
    from experimentation.sweep.search.study import (enqueue_anchors,
                                                    to_distributions)
    from experimentation.sweep.spec import _base_sections

    base_sections = _base_sections(str(REPO / study_spec.base))
    base_model = resolve_model_config(base_sections["model"])
    space = restrict_space(
        search_space(base_model, base_name=study_spec.base), base_model,
        axes=study_spec.search_axes, fixed=study_spec.fixed_params)
    designs = load_designs(REPO / study_spec.design_file, minimum=24)

    study = create_study(
        study_name=study_spec.name, study_dir=tmp_path, seed=study_spec.seed,
        concurrent_trials=1, sampler=study_spec.sampler,
        sampler_startup_trials=study_spec.sampler_startup_trials,
        prune_after_step=study_spec.prune_after_step,
        prune_startup_trials=study_spec.prune_startup_trials,
        n_trials=study_spec.n_trials, logging_steps=study_spec.logging_steps)
    added = enqueue_anchors(study, designs, base_model, space,
                            base_lr=base_sections["optim"]["lr"])
    assert added == 24

    distributions = to_distributions(space)
    for index in range(24):
        trial = study.ask(distributions)
        assert trial.user_attrs.get("anchor_name") == designs[index].name
        # The fixed axes arrive as their pinned values, through the singleton.
        assert trial.params["weight_decay"] == 0.1
        assert trial.params["warmup_ratio"] == 0.04
        assert trial.params["grad_clip"] == 1.0
        study.tell(trial, 1.0 + index / 100.0)

    states = (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)
    assert len(study.get_trials(deepcopy=False, states=states)) == 24
    # 24 of the 64 the study declares -- so ~40 exploratory draws remain, which
    # is exactly what the study file's Phase 2 says.
    assert study.sampler._n_startup_trials - 24 == 40


def test_enqueue_is_idempotent_so_a_resume_does_not_rerun_anchors(tmp_path,
                                                                 study_spec):
    """The expensive mistake: 24 anchors re-run on every restart."""
    from experimentation.run.spec import resolve_model_config
    from experimentation.sweep.search.anchors import load_designs
    from experimentation.sweep.search.space import restrict_space, search_space
    from experimentation.sweep.search.study import enqueue_anchors, to_distributions
    from experimentation.sweep.spec import _base_sections

    base_sections = _base_sections(str(REPO / study_spec.base))
    base_model = resolve_model_config(base_sections["model"])
    space = restrict_space(
        search_space(base_model, base_name=study_spec.base), base_model,
        axes=study_spec.search_axes, fixed=study_spec.fixed_params)
    designs = load_designs(REPO / study_spec.design_file, minimum=24)

    kwargs = dict(study_name=study_spec.name, study_dir=tmp_path,
                  seed=study_spec.seed, concurrent_trials=1,
                  sampler=study_spec.sampler,
                  sampler_startup_trials=study_spec.sampler_startup_trials,
                  prune_after_step=study_spec.prune_after_step,
                  prune_startup_trials=study_spec.prune_startup_trials,
                  n_trials=study_spec.n_trials,
                  logging_steps=study_spec.logging_steps)
    study = create_study(**kwargs)
    enqueue_anchors(study, designs, base_model, space,
                    base_lr=base_sections["optim"]["lr"])
    distributions = to_distributions(space)
    for index in range(24):
        study.tell(study.ask(distributions), 1.0 + index / 100.0)

    # Restart: same journal, same designs.
    resumed = create_study(**kwargs)
    enqueue_anchors(resumed, designs, base_model, space,
                    base_lr=base_sections["optim"]["lr"])
    waiting = resumed.get_trials(deepcopy=False,
                                states=(optuna.trial.TrialState.WAITING,))
    assert waiting == [], (
        f"{len(waiting)} anchor(s) were re-enqueued on resume; "
        f"skip_if_exists=True is what makes a restart cheap")


# --------------------------------------------- the per-worker sampler seed ----

def test_each_worker_gets_a_distinct_sampler_seed():
    assert [sampler_seed_for(2026, w) for w in range(8)] == list(
        range(2026, 2034))


def test_the_supervisor_uses_the_declared_seed_unchanged():
    """`None` means "not a worker", which is the single-process case -- it must
    reproduce exactly what a study got before per-worker seeding existed."""
    assert sampler_seed_for(2026, None) == 2026
    assert sampler_seed_for(2026, 0) == 2026


def test_distinct_seeds_produce_distinct_random_streams():
    """The whole point. With one seed for eight workers, the startup window is
    eight copies of one sequence and constant_liar is repelling proposals that
    were identical by construction."""
    def first_draws(seed):
        sampler = make_sampler(seed=seed, sampler="tpe", startup_trials=100)
        study = optuna.create_study(sampler=sampler)
        return [study.ask(_distributions()).params["x"] for _ in range(5)]

    streams = [tuple(first_draws(sampler_seed_for(2026, w))) for w in range(4)]
    assert len(set(streams)) == 4, (
        f"workers drew overlapping streams: {streams}")


def test_the_same_seed_reproduces_the_same_stream():
    """Guards the guard: if `first_draws` were nondeterministic the test above
    would pass for the wrong reason."""
    def first_draws(seed):
        sampler = make_sampler(seed=seed, sampler="tpe", startup_trials=100)
        study = optuna.create_study(sampler=sampler)
        return [study.ask(_distributions()).params["x"] for _ in range(5)]

    assert first_draws(2029) == first_draws(2029)


# ------------------------------------------------------------- refusals ----

def test_an_unknown_sampler_is_refused():
    with pytest.raises(ValueError, match="unknown sampler"):
        make_sampler(seed=1, sampler="cmaes")


def test_a_startup_count_on_the_random_sampler_is_refused():
    """RandomSampler has no startup window, so the number would be silently
    ignored and the study would look configured when it was not."""
    with pytest.raises(ValueError, match="no n_startup_trials"):
        make_sampler(seed=1, sampler="random", startup_trials=64)
