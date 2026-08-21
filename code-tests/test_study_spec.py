"""StudySpec: the study's intent, written down where the journal cannot hold it.

Measured what optuna 4.9's journal actually persists: params with distributions,
values, per-step intermediate_values, state, user_attrs, timestamps, worker_id.
So the space and every trajectory are already durable, and this study_spec must NOT
re-declare the space -- space.py owns it once.

What the journal cannot hold: anything optuna never sees (max_steps, the base
study_spec, the shard -- constants of our objective, invisible to a black-box
optimiser), and one thing it does use but does not persist (the sampler and
pruner, reconstructed by whoever opens the study, with a different pruner
accepted silently).

The load-bearing case is multi-worker agreement. Parallelism is N processes on
one journal. With max_steps on a command line, worker 3 can disagree with worker
1, and optuna will mix 300-step and 3000-step trials into one study because the
journal records no budget to notice the disagreement with.

This module imports no optuna, so all of it runs in the plain CPU suite.
"""

import pathlib
import sys

import pytest
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from experimentation.sweep.search.studyspec import (  # noqa: E402
    StudySpec, load_study_spec, study_id)

MINIMAL = {"name": "ska-depth", "base": "configs/runs/50m-first-real.yaml",
           "n_trials": 12, "max_steps": 600}


def _write(tmp_path, **overrides):
    payload = dict(MINIMAL)
    payload.update(overrides)
    path = tmp_path / "study.yaml"
    path.write_text(yaml.safe_dump(payload))
    return path


# ------------------------------------------------------------- above the line ----

def test_the_module_does_not_import_optuna():
    """A study must be authorable and diffable where optuna is absent -- the same
    property that lets anchors ship as an ordinary cells: sweep."""
    src = (pathlib.Path(__file__).resolve().parents[1]
           / "experimentation/sweep/search/studyspec.py").read_text()
    for line in src.splitlines():
        stripped = line.strip()
        if stripped.startswith(("import ", "from ")):
            assert "optuna" not in stripped, f"optuna leaked in: {stripped}"


def test_it_does_not_redeclare_the_search_space():
    """space.py declares the space exactly once. A second declaration here is how
    two sources of truth start, and the journal records distributions anyway."""
    fields = set(StudySpec.__dataclass_fields__)
    for forbidden in ("space", "axes", "distributions", "params"):
        assert forbidden not in fields, (
            f"StudySpec.{forbidden} would duplicate space.py's declaration")


# ------------------------------------------------------------------- loading ----

def test_a_minimal_spec_loads(tmp_path):
    study_spec = load_study_spec(_write(tmp_path))
    assert study_spec.name == "ska-depth"
    assert (study_spec.n_trials, study_spec.max_steps) == (12, 600)
    assert study_spec.launcher == "slurm"          # the default worth having
    assert study_spec.objective == {}             # loss only, until told otherwise


@pytest.mark.parametrize("missing", ["name", "base", "n_trials", "max_steps"])
def test_the_four_required_keys_are_required(tmp_path, missing):
    payload = {k: v for k, v in MINIMAL.items() if k != missing}
    path = tmp_path / "study.yaml"
    path.write_text(yaml.safe_dump(payload))
    with pytest.raises(ValueError, match=missing):
        load_study_spec(path)


def test_an_unknown_key_is_an_error_not_a_shrug(tmp_path):
    """A misspelled max_step silently taking the default would run an entire
    study at the wrong budget -- expensive, and invisible until the numbers make
    no sense."""
    path = _write(tmp_path, max_step=600)
    with pytest.raises(ValueError, match="max_step"):
        load_study_spec(path)


def test_it_is_frozen(tmp_path):
    study_spec = load_study_spec(_write(tmp_path))
    with pytest.raises(Exception):
        study_spec.n_trials = 99


# ---------------------------------------------------------------- validation ----

@pytest.mark.parametrize("field,value", [
    ("n_trials", 0), ("max_steps", 0), ("n_jobs", 0), ("logging_steps", 0),
    ("prune_after_step", -1),
])
def test_nonsense_numbers_are_rejected(tmp_path, field, value):
    with pytest.raises(ValueError, match=field):
        load_study_spec(_write(tmp_path, **{field: value}))


def test_an_unknown_launcher_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="launcher"):
        load_study_spec(_write(tmp_path, launcher="pbs"))


def test_a_list_of_directions_is_rejected_with_the_reason(tmp_path):
    """Multi-objective is a decision, not a knob: optuna 4.9's Trial.report
    raises NotImplementedError under multiple directions, so a Pareto study
    cannot prune at all."""
    with pytest.raises(ValueError, match="prune"):
        load_study_spec(_write(tmp_path, direction=["minimize", "minimize"]))


def test_a_pruner_that_could_never_fire_is_rejected(tmp_path):
    """prune_after_step >= max_steps means every trial runs to completion while
    the study looks pruning-enabled. Silent waste is the failure mode worth
    catching at parse time."""
    with pytest.raises(ValueError, match="pruned"):
        load_study_spec(_write(tmp_path, max_steps=100, prune_after_step=100))


def test_prune_after_step_zero_is_allowed(tmp_path):
    """Saying 'never mind, prune from the start' explicitly must stay legal."""
    study_spec = load_study_spec(_write(tmp_path, prune_after_step=0))
    assert study_spec.prune_after_step == 0


# ------------------------------------------------------------------ study_id ----

def test_study_id_is_stable_and_content_addressed(tmp_path):
    a = load_study_spec(_write(tmp_path))
    b = load_study_spec(_write(tmp_path))
    assert study_id(a) == study_id(b)
    assert len(study_id(a)) == 8


def test_changing_any_field_moves_the_study_id(tmp_path):
    base = study_id(load_study_spec(_write(tmp_path)))
    for field, value in [("n_trials", 24), ("max_steps", 1200),
                         ("seed", 7), ("design_file", "configs/search/x.yaml")]:
        assert study_id(load_study_spec(_write(tmp_path, **{field: value}))) != base, \
            f"{field} did not affect study_id"


def test_study_id_is_not_a_run_identity():
    """Recorded as a test because it is the mistake sweep_id's docstring exists
    to prevent: two studies proposing the same config must produce the same
    run_id, or the content-addressed run directory stops being content-addressed."""
    from experimentation.run import study_spec as run_spec
    src = (pathlib.Path(__file__).resolve().parents[1]
           / "experimentation/run/spec.py").read_text()
    assert "study_id" not in src, (
        "run/spec.py references study_id -- study membership is provenance, not "
        "a scientific input, and must not perturb run_id or group_id")
    assert hasattr(run_spec, "group_id")
