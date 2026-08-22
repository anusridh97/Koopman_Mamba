"""A trial must train for the STUDY's budget, not the base spec's.

Found by the first end-to-end study on real hardware (job 439754), which is
exactly what that run was for. `configs/search/smoke-4m.yaml` asked for 200-step
trials and the log said `Training: 400 steps`, because
`configs/runs/4m-golden.yaml` says 400 and nothing overrode it.

`space.py` deliberately does not own run length -- its docstring says the run
length "belongs to the base spec rather than to the space", and that is the right
call: the space samples architecture and optimizer hyperparameters, not how long
to train. It accepts `max_steps` only because `warmup_ratio` is meaningless
without a run length to be a ratio OF.

The gap was that nothing then applied it. So `StudySpec.max_steps` -- documented
as "steps per trial, the single most important field to pin" -- silently scaled
warmup alone.

Scale of it: against `configs/runs/50m-fineweb-3b.yaml`, whose `max_steps` is
15000, a 12-trial study declaring 600 steps would have run 180,000 steps instead
of 7,200. A 25x overspend with nothing in the output saying so.

It also corrupted the schedule twice over. Warmup was computed against 200 while
the run was 400, so the LR ramp covered half the intended fraction of training;
and `StudySpec`'s `prune_after_step < max_steps` validation was checking against a
number the run did not use, so a study could pass validation and still be
unprunable.
"""

import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "code-tests"))

optuna = pytest.importorskip("optuna")

from test_search_driver import _FakeLauncher, _context  # noqa: E402

from experimentation.sweep.search.driver import run_trial  # noqa: E402
from experimentation.sweep.search.metrics import fixed_reader  # noqa: E402
from experimentation.sweep.search.space import (  # noqa: E402
    params_to_overrides, search_space)


@pytest.fixture(autouse=True)
def _quiet():
    optuna.logging.set_verbosity(optuna.logging.WARNING)


def _drive_one(tmp_path, max_steps):
    """Run one trial and hand back (outcome, the RunSpec the launcher saw)."""
    from experimentation.sweep.search.study import to_distributions
    context = _context(tmp_path, max_steps=max_steps)
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    trial = study.ask(to_distributions(context["space"]))
    launcher = _FakeLauncher()
    outcome = run_trial(study, trial, launcher=launcher,
                        objective_reader_for=fixed_reader(lambda rd: 1.0), **context)
    assert launcher.submitted, "nothing was submitted"
    return outcome, launcher.submitted[-1][0], trial, context


def test_the_trial_trains_for_the_studys_budget(tmp_path):
    _, spec, _, _ = _drive_one(tmp_path, 200)
    assert spec.optim.max_steps == 200, (
        f"trial trained for {spec.optim.max_steps} steps, not the study's 200 -- "
        "the base spec's budget leaked through")


def test_the_base_specs_budget_does_not_leak_through(tmp_path):
    """test_search_driver's fixture declares optim.max_steps=100, so a study
    asking for 700 has something to actually override."""
    _, spec, _, context = _drive_one(tmp_path, 700)
    assert context["base_sections"]["optim"]["max_steps"] != 700, (
        "fixture no longer distinguishes the two; this would pass vacuously")
    assert spec.optim.max_steps == 700


def test_warmup_is_a_ratio_of_the_budget_actually_used(tmp_path):
    """The second half of the bug: warmup was computed against the study's
    max_steps while the run used the base spec's, so the LR ramp covered the
    wrong fraction of training. Now both refer to one number."""
    _, spec, trial, _ = _drive_one(tmp_path, 1000)
    ratio = trial.params["warmup_ratio"]
    assert spec.optim.max_steps == 1000
    assert spec.optim.warmup_steps == max(1, round(1000 * ratio))
    assert spec.optim.warmup_steps < spec.optim.max_steps


def test_the_space_still_does_not_own_run_length():
    """The fix belongs in the driver, not the space. space.py commits in its
    docstring to not owning run length, and putting the budget there would make
    the space responsible for something it does not sample."""
    from koopman_lm.config import build_config
    from experimentation.sweep.search.study import to_distributions

    cfg = build_config("50m")
    space = search_space(cfg, base_name="50m")
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=1))
    trial = study.ask(to_distributions(space))

    overrides = params_to_overrides(trial.params, cfg, max_steps=321)
    assert "optim.max_steps" not in overrides, (
        "params_to_overrides should not set the budget; the driver does")
    assert "optim.warmup_steps" in overrides


def test_the_budget_is_recorded_in_the_materialized_spec(tmp_path):
    """A run directory must be able to answer 'how long was this trial meant to
    train?' without the study config in hand."""
    import yaml
    outcome, _, _, _ = _drive_one(tmp_path, 250)
    spec_yaml = pathlib.Path(outcome.run_dir) / "spec.yaml"
    assert spec_yaml.exists(), outcome.run_dir
    raw = yaml.safe_load(spec_yaml.read_text())
    assert raw["optim"]["max_steps"] == 250


def test_two_studies_at_different_budgets_get_different_run_ids(tmp_path):
    """A consequence worth pinning: max_steps is in the optim section, which IS
    hashed into group_id. So the same architecture at 200 and 400 steps are
    correctly different experiments -- and before the fix they collided, because
    both inherited the base spec's budget."""
    for sub in ("a", "b"):
        (tmp_path / sub).mkdir()          # _context materializes a shard dir under it
    _, a, _, _ = _drive_one(tmp_path / "a", 200)
    _, b, _, _ = _drive_one(tmp_path / "b", 400)
    from experimentation.run.spec import group_id
    assert a.model.ska_rank == b.model.ska_rank, "same seed should sample alike"
    assert group_id(a) != group_id(b), (
        "runs at different step budgets must not share an identity")
