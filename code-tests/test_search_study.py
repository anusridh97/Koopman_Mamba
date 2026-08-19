"""experimentation/sweep/search/study.py -- the optuna boundary.

One of only two modules in the search package that imports optuna. Its job is
translation and construction: plain-data distributions in, an optuna Study out.

Everything here is skipped unless optuna is importable, which is the design
working as intended -- geometry/space/anchors are useful without it, so the CPU
suite stays green on a box that has never heard of optuna.

    PYTHONPATH=/users/jkli/.venvs/koopman-optuna/site \\
        /users/jkli/.venvs/koopman-cpu/bin/pytest code-tests/test_search_study.py

The pruner behaviour pinned at the bottom is the counter-intuitive one, and it
cost a probe to establish: MedianPruner will not prune anything until
`n_startup_trials` trials have COMPLETED, because before that there is no median
to compare against. A reasonable person watching six rising loss reports get
`should_prune() == False` would conclude pruning was broken.
"""
import pytest

optuna = pytest.importorskip("optuna", reason="optuna is an optional dependency")

pytestmark = pytest.mark.correctness


@pytest.fixture(autouse=True)
def _quiet_optuna():
    """optuna logs "A new study created in Journal with name: ..." at INFO for
    every study; a dozen of those buries a real warning."""
    previous = optuna.logging.get_verbosity()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    yield
    optuna.logging.set_verbosity(previous)


def _space():
    from koopman_lm.config import build_config
    from experimentation.sweep.search.space import search_space
    return search_space(build_config("50m"), base_name="50m")


# --------------------------------------------------------- translation ----

def test_categorical_declaration_becomes_a_categorical_distribution():
    from experimentation.sweep.search.study import to_distribution

    dist = to_distribution({"kind": "categorical", "choices": [8, 16, 24]})
    assert isinstance(dist, optuna.distributions.CategoricalDistribution)
    assert list(dist.choices) == [8, 16, 24]


def test_float_declaration_becomes_a_float_distribution_preserving_log():
    from experimentation.sweep.search.study import to_distribution

    dist = to_distribution({"kind": "float", "low": 1e-3, "high": 1e-1, "log": True})
    assert isinstance(dist, optuna.distributions.FloatDistribution)
    assert dist.low == pytest.approx(1e-3) and dist.high == pytest.approx(1e-1)
    assert dist.log is True, "a log-scaled range sampled linearly wastes most trials"


def test_float_declaration_defaults_to_linear_when_log_is_absent():
    from experimentation.sweep.search.study import to_distribution

    assert to_distribution({"kind": "float", "low": 0.0, "high": 1.0}).log is False


def test_an_unknown_declaration_kind_is_rejected():
    from experimentation.sweep.search.study import to_distribution

    with pytest.raises(ValueError, match="kind"):
        to_distribution({"kind": "gaussian", "mu": 0.0})


def test_the_whole_space_translates_and_optuna_accepts_it(tmp_path):
    """The real check: every declaration in space.py must be something optuna
    can actually sample from."""
    from experimentation.sweep.search.study import create_study, to_distributions

    space = _space()
    distributions = to_distributions(space)
    assert set(distributions) == set(space)

    study = create_study(study_name="translate", study_dir=tmp_path, seed=1)
    trial = study.ask(distributions)
    assert set(trial.params) == set(space)


# ------------------------------------------------------- construction ----

def test_sampler_enables_constant_liar_only_when_running_in_parallel():
    """Without it, N parallel workers all propose from the same history and
    compute nearly the same point -- N GPUs producing one answer."""
    from experimentation.sweep.search.study import make_sampler

    assert make_sampler(seed=1, n_jobs=1)._constant_liar is False
    assert make_sampler(seed=1, n_jobs=4)._constant_liar is True


def test_sampler_is_seeded_so_a_study_is_reproducible():
    from experimentation.sweep.search.study import make_sampler, to_distributions

    space = to_distributions(_space())
    first = optuna.create_study(sampler=make_sampler(seed=99, n_jobs=1))
    second = optuna.create_study(sampler=make_sampler(seed=99, n_jobs=1))
    a = [first.ask(space).params for _ in range(3)]
    b = [second.ask(space).params for _ in range(3)]
    assert a == b


def test_storage_is_a_journal_file_that_survives_reload(tmp_path):
    """A file-backed journal, not an in-memory study: a search that cannot be
    resumed after a launcher crash would lose every completed trial's result."""
    from experimentation.sweep.search.study import create_study

    study = create_study(study_name="resumable", study_dir=tmp_path, seed=1)
    trial = study.ask()
    study.tell(trial, 1.0)
    assert (tmp_path / "optuna_journal.log").is_file()

    reopened = create_study(study_name="resumable", study_dir=tmp_path, seed=1)
    assert len(reopened.trials) == 1, "reopening must attach to the same study"


def test_study_minimises(tmp_path):
    from experimentation.sweep.search.study import create_study

    study = create_study(study_name="direction", study_dir=tmp_path, seed=1)
    assert study.direction == optuna.study.StudyDirection.MINIMIZE


# ------------------------------------------------------------ anchors ----

def test_anchors_are_enqueued_in_order_and_come_back_from_ask(tmp_path):
    """The property the driver depends on: an enqueued anchor is returned by
    ask() with its exact parameters, so a curated point and a sampled point are
    handled by identical code downstream."""
    from koopman_lm.config import build_config
    from experimentation.sweep.search.anchors import Design
    from experimentation.sweep.search.study import (
        create_study, enqueue_anchors, is_anchor, to_distributions)

    cfg = build_config("50m")
    space = _space()
    designs = [Design(name="baseline"), Design(name="thin", rank=8)]

    study = create_study(study_name="anchored", study_dir=tmp_path, seed=1)
    count = enqueue_anchors(study, designs, cfg, space, base_lr=4e-4)
    assert count == 2

    distributions = to_distributions(space)
    first = study.ask(distributions)
    assert is_anchor(first) is True
    assert first.user_attrs["anchor_name"] == "baseline"
    assert first.params["ska_rank"] == cfg.ska_rank
    study.tell(first, 1.0)

    second = study.ask(distributions)
    assert second.user_attrs["anchor_name"] == "thin"
    assert second.params["ska_rank"] == 8


def test_a_sampled_trial_is_not_an_anchor(tmp_path):
    from experimentation.sweep.search.study import (
        create_study, is_anchor, to_distributions)

    study = create_study(study_name="mixed", study_dir=tmp_path, seed=1)
    assert is_anchor(study.ask(to_distributions(_space()))) is False


def test_re_enqueueing_the_same_anchors_does_not_duplicate_them(tmp_path):
    """Resuming a study must not re-run the anchors that already ran."""
    from koopman_lm.config import build_config
    from experimentation.sweep.search.anchors import Design
    from experimentation.sweep.search.study import create_study, enqueue_anchors

    cfg, space = build_config("50m"), _space()
    designs = [Design(name="baseline"), Design(name="thin", rank=8)]
    study = create_study(study_name="idempotent", study_dir=tmp_path, seed=1)
    enqueue_anchors(study, designs, cfg, space, base_lr=4e-4)
    enqueue_anchors(study, designs, cfg, space, base_lr=4e-4)
    assert len(study.trials) == 2


# ------------------------------------------------------------- pruner ----

def test_the_pruner_is_inert_until_enough_trials_have_completed(tmp_path):
    """Pinned because it looks like a bug. MedianPruner needs n_startup_trials
    COMPLETED trials before it has a median to compare against, so early rising
    losses are not pruned however bad they look."""
    from experimentation.sweep.search.study import create_study, to_distributions

    study = create_study(study_name="cold", study_dir=tmp_path, seed=1,
                         prune_after_step=2, prune_startup_trials=3)
    trial = study.ask(to_distributions(_space()))
    for step in range(8):
        trial.report(10.0 + step, step)
    # bool(): should_prune returns a numpy bool_, so `is False` would fail on a
    # correct result.
    assert bool(trial.should_prune()) is False, (
        "with no completed trials there is no median; the anchors are what "
        "establish the reference set")


def test_a_clearly_worse_trial_is_pruned_once_a_reference_set_exists(tmp_path):
    from experimentation.sweep.search.study import create_study, to_distributions

    space = to_distributions(_space())
    study = create_study(study_name="warm", study_dir=tmp_path, seed=1,
                         prune_after_step=1, prune_startup_trials=2)
    for good in (1.0, 1.1):
        trial = study.ask(space)
        for step in range(5):
            trial.report(good, step)
        study.tell(trial, good)

    bad = study.ask(space)
    for step in range(5):
        bad.report(50.0, step)
    assert bool(bad.should_prune()) is True


def test_prune_after_step_defers_pruning_through_the_warmup(tmp_path):
    from experimentation.sweep.search.study import create_study, to_distributions

    space = to_distributions(_space())
    study = create_study(study_name="warmup", study_dir=tmp_path, seed=1,
                         prune_after_step=100, prune_startup_trials=1)
    trial = study.ask(space)
    for step in range(5):
        trial.report(1.0, step)
    study.tell(trial, 1.0)

    bad = study.ask(space)
    bad.report(999.0, 3)
    assert bool(bad.should_prune()) is False, "step 3 is inside a 100-step warmup"
