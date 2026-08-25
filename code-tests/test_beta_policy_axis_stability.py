"""Adding a `beta_policy` value must not break a study that already searched it.

## The bug this file pins, measured rather than reasoned about

`space.py` declared `"choices": sorted(BETA_POLICIES)`, so the set a study
SAMPLES was the same object as the set a config may legally HOLD. Adding
`key_linear_value_sqrt` and `key_sqrt_value_linear` for the exponent arm
therefore widened the search axis too -- and optuna does not tolerate that:

    recorded on disk: CategoricalDistribution(('head_scalar','learned','linear','one'))
    code declared   : CategoricalDistribution(('head_scalar','key_linear_value_sqrt',
                                               'key_sqrt_value_linear','learned',
                                               'linear','one'))
    study.ask(...)  -> ValueError: CategoricalDistribution does not
                       support dynamic value space.

**A hard raise, not a warning**, and it lands at `ask()` -- after a worker has
started, claimed a run root and queued a job.

**Exactly ONE archived journal was affected**, which is worth stating precisely
because the initial diagnosis said two. Measured on disk:

    smoke-fanout.5f93952c      ('head_scalar','learned','linear','one')   BROKEN
    beta-policy-1500.bff5ef39  ('learned','one','head_scalar','linear')   SAFE

`beta-policy-1500` is safe because `configs/search/beta-policy-1500.yaml`
declares the axis itself under `search_axes`, in that order, and `search_axes`
REPLACES the declaration -- so the study carries its own support and never reads
the default. `proxy-256x17-interactions-v1.yaml` is safe for the adjacent reason:
it PINS the axis. `smoke-fanout` declared neither, so it sampled the default and
the default moved under it.

That distinction is not a footnote, it is the argument for the fix: the opt-in
mechanism already protects any study that says what it wants, so the only thing
the shared constant needed to stop doing was changing silently for studies that
did not.

Note also that ORDER is part of the support. `CategoricalDistribution.__eq__`
compares `choices` as a TUPLE, so a reordering raises the same error as an
addition -- which is why the default is frozen as an ordered tuple and not as a
set.

An in-memory probe does NOT catch this. `optuna.create_study()` with a trial
added via `add_trial` under the old support accepts the new distribution silently,
because `add_trial` records the distribution it is handed rather than reconciling
it. Only a study whose trials were WRITTEN under the narrow support and then
re-attached reproduces it. That is why the test below builds a journal on disk,
closes it, and re-attaches -- and why the first version of this investigation
concluded "optuna tolerates the widened support" and was wrong.

## The fix these tests pin

Two sets, not one:

  * `BETA_POLICIES` -- what a config may legally hold. WIDE. All six.
  * `DEFAULT_BETA_POLICIES` -- what a study samples unless it says otherwise.
    FROZEN at the original four.

A study that wants the mixed cells opts in via `search_axes`, which replaces the
declaration outright and validates each choice against the axis's real domain
(`BETA_POLICIES`), not against the default tuple. So widening is supported by the
mechanism that already exists; what is no longer supported is widening
ACCIDENTALLY, for every study, by editing a shared constant.
"""
from __future__ import annotations

import pathlib

import pytest

from experimentation.sweep.search.space import (
    DEFAULT_BETA_POLICIES, restrict_space, search_space)
from koopman_lm.config import BETA_POLICIES, build_config

optuna = pytest.importorskip("optuna", reason="needs the side-loaded optuna 4.9")

pytestmark = pytest.mark.correctness

#: What a DEFAULT-axis study recorded, in the order it recorded it. A literal,
#: not derived from any current constant: the point is to compare today's code
#: against what is on disk, and a derived value would move when the code moves.
#:
#: Read off `study-smoke-446055/_studies/smoke-fanout.5f93952c`, which did not
#: declare `search_axes` and so sampled the declared default.
ARCHIVED_DEFAULT_CHOICES = ("head_scalar", "learned", "linear", "one")

#: What `configs/search/beta-policy-1500.yaml` recorded. DIFFERENT ORDER, because
#: that study declares the axis itself:
#:
#:     search_axes:
#:       beta_policy:
#:         choices: [learned, one, head_scalar, linear]
#:
#: which is why it was NEVER at risk from widening the default -- `search_axes`
#: REPLACES the declaration, so the study carries its own support. The same
#: mechanism protects `proxy-256x17-interactions-v1.yaml`, which pins the axis.
#: Recorded here because "two journals were broken" was the initial diagnosis and
#: only one of them actually was.
ARCHIVED_EXPLICIT_CHOICES = ("learned", "one", "head_scalar", "linear")


def _base():
    return build_config("1m")


# ---------------------------------------------------------------------------
# The two sets are distinct, and each is the right width.
# ---------------------------------------------------------------------------

def test_the_default_search_axis_is_frozen_at_the_archived_four_IN_ORDER():
    """ORDER, not just membership.

    `optuna.distributions.CategoricalDistribution.__eq__` compares `choices` as a
    TUPLE, so a reordering is as fatal as an addition -- same
    `ValueError: does not support dynamic value space`. A `set()` comparison here
    would pass while every resumed default-axis study still raised.
    """
    assert tuple(DEFAULT_BETA_POLICIES) == ARCHIVED_DEFAULT_CHOICES


def test_the_legal_config_domain_still_contains_all_six():
    """Narrowing the SEARCH axis must not narrow what a config may hold -- the
    exponent arm sets these policies directly, not through a study."""
    assert {"key_linear_value_sqrt", "key_sqrt_value_linear"} <= BETA_POLICIES
    assert len(BETA_POLICIES) == 6


def test_the_declared_space_offers_exactly_the_archived_default_in_order():
    space = search_space(_base(), base_lr=1e-3)
    assert tuple(space["beta_policy"]["choices"]) == ARCHIVED_DEFAULT_CHOICES


def test_categorical_equality_is_order_sensitive_so_the_order_test_is_needed():
    """The guard on the guard above. If optuna ever compared choices as sets, the
    order assertion would be over-strict and someone would relax it; while it
    compares tuples, relaxing it reintroduces the bug."""
    a = optuna.distributions.CategoricalDistribution(["x", "y"])
    b = optuna.distributions.CategoricalDistribution(["y", "x"])
    assert a != b, "optuna now compares choices order-insensitively"


def test_the_mixed_cells_are_not_in_the_default_space():
    """The regression, stated directly. This is the assertion that fails if
    someone 'tidies' the declaration back to `sorted(BETA_POLICIES)`."""
    choices = search_space(_base(), base_lr=1e-3)["beta_policy"]["choices"]
    assert "key_linear_value_sqrt" not in choices
    assert "key_sqrt_value_linear" not in choices


# ---------------------------------------------------------------------------
# The behaviour that actually broke: re-attaching to a journal on disk.
# ---------------------------------------------------------------------------

def _journal_study(tmp_path, choices):
    """Write a study to a journal under `choices`, then RE-ATTACH to it.

    The round trip through the file is the whole point. An in-memory study with
    `add_trial` does not reproduce the failure, because `add_trial` stores the
    distribution it is given instead of reconciling it against the study's.
    """
    from experimentation.sweep.search.study import make_storage

    study_dir = tmp_path / "s"
    storage = make_storage(study_dir)
    dist = optuna.distributions.CategoricalDistribution(list(choices))
    study = optuna.create_study(study_name="fixture", storage=storage,
                                sampler=optuna.samplers.TPESampler(seed=0))
    for i in range(3):
        t = study.ask({"beta_policy": dist})
        study.tell(t, 4.39 + 0.001 * i)
    del study, storage
    # Fresh storage object over the same file: this is what a resumed worker does.
    return optuna.load_study(study_name="fixture",
                             storage=make_storage(study_dir))


def test_reattaching_to_a_default_axis_journal_does_not_raise(tmp_path):
    """The regression test, and the case that was really broken.

    `smoke-fanout` sampled the DECLARED DEFAULT, so widening that default made
    `ask()` raise `ValueError: CategoricalDistribution does not support dynamic
    value space` -- after a worker had started and claimed a run root.
    """
    study = _journal_study(tmp_path, ARCHIVED_DEFAULT_CHOICES)
    space = search_space(_base(), base_lr=1e-3)
    dist = optuna.distributions.CategoricalDistribution(
        list(space["beta_policy"]["choices"]))
    trial = study.ask({"beta_policy": dist})          # must not raise
    assert trial.params["beta_policy"] in ARCHIVED_DEFAULT_CHOICES
    study.tell(trial, 4.39)
    assert len(study.trials) == 4


def test_reattaching_to_a_study_that_declared_its_own_axis_does_not_raise(tmp_path):
    """`beta-policy-1500`'s case: it carries its own support through
    `search_axes`, in its own order, and resuming it must reproduce that support
    rather than the default -- which is why it was never at risk."""
    study = _journal_study(tmp_path, ARCHIVED_EXPLICIT_CHOICES)
    resumed = restrict_space(
        search_space(_base(), base_lr=1e-3), _base(),
        axes={"beta_policy": {"kind": "categorical",
                              "choices": list(ARCHIVED_EXPLICIT_CHOICES)}})
    dist = optuna.distributions.CategoricalDistribution(
        list(resumed["beta_policy"]["choices"]))
    trial = study.ask({"beta_policy": dist})          # must not raise
    assert trial.params["beta_policy"] in ARCHIVED_EXPLICIT_CHOICES


def test_a_widened_axis_still_raises_so_this_test_is_evidence(tmp_path):
    """The guard on the guard.

    If optuna ever stopped raising, the test above would pass for a reason that
    has nothing to do with the fix, and the frozen default would look
    unnecessary. This asserts the hazard is still real.
    """
    study = _journal_study(tmp_path, ARCHIVED_DEFAULT_CHOICES)
    widened = optuna.distributions.CategoricalDistribution(sorted(BETA_POLICIES))
    with pytest.raises(ValueError, match="dynamic value space"):
        study.ask({"beta_policy": widened})


def test_the_committed_journals_on_scratch_recorded_the_archived_support():
    """Reads the real artefacts this fix exists for, when they are present.

    Skipped rather than failed off the cluster: the assertion is about two files
    on `/scratch`, and a developer laptop legitimately has neither.
    """
    from experimentation.sweep.search.study import make_storage

    roots = [
        pathlib.Path("/scratch/m000151-pm06/jkli/study-smoke-446074/_studies/"
                     "beta-policy-1500.bff5ef39"),
        pathlib.Path("/scratch/m000151-pm06/jkli/study-smoke-446055/_studies/"
                     "smoke-fanout.5f93952c"),
    ]
    present = [r for r in roots if (r / "optuna_journal.log").exists()]
    if not present:
        pytest.skip("neither archived journal is on this filesystem")
    allowed = {ARCHIVED_DEFAULT_CHOICES, ARCHIVED_EXPLICIT_CHOICES}
    for root in present:
        storage = make_storage(root)
        for name in optuna.study.get_all_study_names(storage=storage):
            study = optuna.load_study(study_name=name, storage=storage)
            for trial in study.trials:
                dist = trial.distributions.get("beta_policy")
                if dist is None:
                    continue
                assert tuple(dist.choices) in allowed, (
                    f"{root.name}/{name} recorded {dist.choices}, which is "
                    f"neither the frozen default nor a known explicit axis")
                break


# ---------------------------------------------------------------------------
# The opt-in path: a study that WANTS the mixed cells.
# ---------------------------------------------------------------------------

def test_a_study_can_opt_into_the_mixed_cells_through_search_axes():
    """`restrict_space` REPLACES a declaration and validates each choice against
    the axis's real domain, so widening is already supported -- the answer to
    "does restrict_space only narrow?" is no, it replaces."""
    space = search_space(_base(), base_lr=1e-3)
    widened = restrict_space(
        space, _base(),
        axes={"beta_policy": {"kind": "categorical",
                              "choices": sorted(BETA_POLICIES)}})
    assert sorted(widened["beta_policy"]["choices"]) == sorted(BETA_POLICIES)
    # ... and the base space is not mutated, or one study would widen another's.
    assert tuple(space["beta_policy"]["choices"]) == ARCHIVED_DEFAULT_CHOICES


def test_the_exponent_arms_five_cells_are_declarable_as_an_axis():
    """The arm's own comparison set, as a study would spell it. `learned-ridge2x`
    and `linear-ridge0.5x` are ridge variants, not policies, so the policy axis
    has five members."""
    cells = ["learned", "one", "linear",
             "key_linear_value_sqrt", "key_sqrt_value_linear"]
    got = restrict_space(
        search_space(_base(), base_lr=1e-3), _base(),
        axes={"beta_policy": {"kind": "categorical", "choices": cells}})
    assert sorted(got["beta_policy"]["choices"]) == sorted(cells)


def test_opting_in_to_an_illegal_policy_is_still_refused():
    """Widening is permitted only within the axis's real domain. Otherwise
    `search_axes` would be a hole rather than a mechanism."""
    with pytest.raises(ValueError, match="beta_policy"):
        restrict_space(
            search_space(_base(), base_lr=1e-3), _base(),
            axes={"beta_policy": {"kind": "categorical",
                                  "choices": ["learned", "sqrt_beta"]}})


def test_a_study_pinning_the_policy_is_unaffected_by_any_of_this():
    """`configs/search/proxy-256x17-interactions-v1.yaml` pins
    `beta_policy: learned` rather than searching it, which is why it was never at
    risk. Asserted so that stays true."""
    fixed = restrict_space(search_space(_base(), base_lr=1e-3), _base(),
                           fixed={"beta_policy": "learned"})
    assert fixed["beta_policy"] == {"kind": "categorical", "choices": ["learned"]}
