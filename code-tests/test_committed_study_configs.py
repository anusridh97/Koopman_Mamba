"""Every committed study config parses, and names things that exist.

A `configs/search/*.yaml` is the one artifact in this repo whose only consumer is
a GPU job. `StudySpec.__post_init__` rejects unknown keys, so a typo is caught --
but only when something parses the file, and until now nothing did except an
sbatch script at 3am.

The specific hole this closes: `StudySpec` does NOT validate `backend_policy`.
It is a plain `str` field, so a config naming a policy that no longer exists
parses cleanly, materialises a run directory, queues a job, and dies inside
`params_to_overrides` on the first trial. When `proxy_chunked` was retired
(90d17cb) two committed configs named it, and both had to be found by grep. This
would have found them, and will find the next one.

Deliberately NOT asserted here: that a study is well-sized, that its budget is
sensible, or that its scratch paths exist. Scratch is not present on a CPU box
and sizing is a research judgement. This checks only what is checkable
everywhere: it parses, and the repo-relative files it points at are committed.
"""

import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

SEARCH_DIR = REPO / "configs/search"

# A design file, not a study: it has a `designs:` list and no `name`/`base`.
# Named explicitly rather than pattern-matched so a NEW study config cannot skip
# this suite by accident -- an unrecognised file is a failure, not a skip.
NOT_STUDIES = {"example_anchors.yaml"}


def _study_configs():
    return sorted(p for p in SEARCH_DIR.glob("*.yaml")
                  if p.name not in NOT_STUDIES)


def test_there_is_at_least_one_committed_study():
    """Guards the guard: a glob that matches nothing makes every
    parametrised test below vacuously green."""
    assert _study_configs(), f"no study configs found under {SEARCH_DIR}"


@pytest.mark.parametrize("path", _study_configs(), ids=lambda p: p.name)
def test_it_parses_as_a_study_spec(path):
    from experimentation.sweep.search.studyspec import load_study_spec

    study_spec = load_study_spec(path)
    assert study_spec.name, f"{path.name} has no study name"
    assert study_spec.n_trials > 0 and study_spec.max_steps > 0


@pytest.mark.parametrize("path", _study_configs(), ids=lambda p: p.name)
def test_its_backend_policy_still_exists(path):
    """The hole described in the module docstring. `backend_policy` is a plain
    str on StudySpec, so a retired policy survives parsing and fails on a GPU."""
    from experimentation.sweep.search.space import (BACKEND_POLICIES,
                                                    _RETIRED_POLICIES)
    from experimentation.sweep.search.studyspec import load_study_spec

    policy = load_study_spec(path).backend_policy
    assert policy not in _RETIRED_POLICIES, (
        f"{path.name} names the retired policy {policy!r}:\n"
        f"{_RETIRED_POLICIES.get(policy, '')}")
    assert policy in BACKEND_POLICIES, (
        f"{path.name} names unknown policy {policy!r}; "
        f"expected one of {list(BACKEND_POLICIES)}")


@pytest.mark.parametrize("path", _study_configs(), ids=lambda p: p.name)
def test_the_repo_relative_files_it_points_at_are_committed(path):
    """`base` and `design_file` are repo-relative and resolved against cwd at run
    time, so a rename in configs/runs/ silently strands a study. run_root,
    storage and eval_data_dir are deliberately exempt: they are scratch paths
    that do not exist on a CPU box."""
    from experimentation.sweep.search.studyspec import load_study_spec

    study_spec = load_study_spec(path)
    for field in ("base", "design_file"):
        value = getattr(study_spec, field)
        if not value:
            continue
        assert (REPO / value).is_file(), \
            f"{path.name}: {field} -> {value!r} is not a committed file"


@pytest.mark.parametrize("path", _study_configs(), ids=lambda p: p.name)
def test_pruning_and_a_multi_objective_are_not_both_requested(path):
    """optuna's `Trial.report` raises NotImplementedError under multiple
    directions, so a study cannot both prune and optimise several objectives.
    The failure surfaces mid-study, after trials have already been spent."""
    from experimentation.sweep.search.studyspec import load_study_spec

    study_spec = load_study_spec(path)
    weights = [k for k, v in (study_spec.objective or {}).items() if v]
    if len(weights) > 1:
        assert study_spec.prune_after_step <= 0, (
            f"{path.name} weights {weights} AND prunes after step "
            f"{study_spec.prune_after_step}; optuna cannot do both")
