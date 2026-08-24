"""The trainer's logging cadence and the pruner's interval must be ONE number.

`study.py`'s own docstring explains why `interval_steps` exists: "train.py prints
only every `logging_steps` steps, so reports arrive at 10, 20, 30...; consulting
the pruner at every integer step between them compares a trial against steps no
other trial ever reported."

That reasoning was right and the wiring was half-done. `StudySpec.logging_steps`
reached `make_pruner(interval_steps=...)` and NOTHING passed `--logging_steps` to
the trainer, so `train.py`'s default of 10 stayed in force. With the interaction
study's `logging_steps: 25` and `prune_after_step: 450`, reports arrived at
multiples of 10 while MedianPruner only consulted where
`(step - 450) % 25 == 0` -- so the real prune opportunities were
{450, 500, 550, 600} instead of the intended {450, 475, 500, ...}: a third of
them, and the plan printer said "reporting every 25", which was false.

The cadence belongs on the LAUNCHER, not the spec, for the reason `_ScoresRuns`
already gives about `eval_on_final`: how often a run PRINTS is a property of who
launched it, and anything on the spec would be hashed into `run_id` -- two runs
that differ only in log verbosity are the same experiment.
"""
from __future__ import annotations

import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

pytestmark = pytest.mark.correctness


@pytest.fixture(scope="module")
def spec():
    from experimentation.run.resolve import resolve_run_spec
    return resolve_run_spec(REPO / "configs/runs/proxy-256x17.yaml")


def _argv(spec, tmp_path, **kw):
    from experimentation.run.train_argv import build_train_argv
    return build_train_argv(spec, tmp_path, **kw)


def _flag(argv, name):
    return argv[argv.index(name) + 1] if name in argv else None


# ------------------------------------------------ it reaches the training argv ----

def test_the_logging_cadence_reaches_the_training_argv(spec, tmp_path):
    """The line whose absence was the whole bug."""
    assert _flag(_argv(spec, tmp_path, logging_steps=25), "--logging_steps") == "25"


def test_a_different_cadence_reaches_it_too(spec, tmp_path):
    """Guards against a hardcoded 25."""
    assert _flag(_argv(spec, tmp_path, logging_steps=7), "--logging_steps") == "7"


def test_omitting_it_leaves_the_flag_off_entirely(spec, tmp_path):
    """Absence must mean "trainer default", not a value this layer invented. A
    hand-launched run has no pruner to agree with."""
    assert "--logging_steps" not in _argv(spec, tmp_path)


def test_a_non_positive_cadence_is_refused(spec, tmp_path):
    """`logging_steps=0` would make train.py's `step % logging_steps` a
    ZeroDivisionError, and MedianPruner requires interval_steps >= 1."""
    for bad in (0, -1):
        with pytest.raises(ValueError, match="logging_steps"):
            _argv(spec, tmp_path, logging_steps=bad)


def test_the_flag_it_emits_is_one_train_py_accepts():
    """A flag the trainer does not know is an argparse error on the compute node,
    after the queue wait."""
    text = (REPO / "experimentation/training/train.py").read_text()
    assert '"--logging_steps"' in text or "'--logging_steps'" in text


# --------------------------------------------------- through both launchers ----

@pytest.mark.parametrize("kind", ["local", "slurm"])
def test_the_launcher_threads_the_cadence_into_the_command(kind, spec, tmp_path):
    from experimentation.run.launchers import LocalLauncher, SlurmLauncher

    launcher = (LocalLauncher(logging_steps=25) if kind == "local"
                else SlurmLauncher(repo_root=str(REPO), logging_steps=25))
    command = launcher.build_command(spec, tmp_path)
    joined = " ".join(str(c) for c in command)
    assert "--logging_steps 25" in joined, joined[-400:]


@pytest.mark.parametrize("kind", ["local", "slurm"])
def test_a_launcher_with_no_cadence_omits_the_flag(kind, spec, tmp_path):
    from experimentation.run.launchers import LocalLauncher, SlurmLauncher

    launcher = (LocalLauncher() if kind == "local"
                else SlurmLauncher(repo_root=str(REPO)))
    joined = " ".join(str(c) for c in launcher.build_command(spec, tmp_path))
    assert "--logging_steps" not in joined


# ------------------------------------- the study's number is the ONE number ----

def test_the_studys_cadence_reaches_both_the_launcher_and_the_pruner(tmp_path):
    """The invariant. If these two ever disagree, the pruner is consulted at
    steps no trial reported -- which is the failure `study.py`'s docstring
    describes and which this wiring existed to prevent."""
    pytest.importorskip("optuna")
    import textwrap

    from experimentation.sweep.search import __main__ as cli
    from experimentation.sweep.search import driver as driver_mod

    study = tmp_path / "study.yaml"
    study.write_text(textwrap.dedent(f"""
        name: cadence-probe
        base: configs/runs/proxy-256x17.yaml
        n_trials: 2
        max_steps: 600
        prune_after_step: 450
        logging_steps: 25
        launcher: local
        run_root: {tmp_path / 'runs'}
    """).strip())

    captured = {}

    def fake_drive(study_, **kwargs):
        captured["study"] = study_
        captured.update(kwargs)
        return []

    original, gate = driver_mod.drive, cli.check_git_clean
    driver_mod.drive = fake_drive
    cli.check_git_clean = lambda allow_dirty=False: False
    try:
        cli.main([str(study), "--force-no-eval"])
    finally:
        driver_mod.drive = original
        cli.check_git_clean = gate

    assert captured["launcher"].logging_steps == 25, (
        "the study's logging_steps did not reach the launcher, so train.py "
        "keeps its default of 10 while the pruner consults every 25")
    assert captured["study"].pruner._interval_steps == 25
    assert captured["launcher"].logging_steps == \
        captured["study"].pruner._interval_steps


def test_the_prune_opportunities_are_what_the_study_intends():
    """The arithmetic the bug corrupted, stated as the property that matters.

    With reports every 25 from step 450 of 600 there are 7 chances to prune. With
    reports every 10 -- the trainer default the study could not override -- the
    pruner's `(step-450) % 25 == 0` filter matches only 4 of them, so a third of
    the intended prune points silently did not exist.
    """
    reports_25 = [s for s in range(25, 601, 25) if s >= 450]
    aligned_25 = [s for s in reports_25 if (s - 450) % 25 == 0]
    assert aligned_25 == [450, 475, 500, 525, 550, 575, 600]

    reports_10 = [s for s in range(10, 601, 10) if s >= 450]
    aligned_10 = [s for s in reports_10 if (s - 450) % 25 == 0]
    assert aligned_10 == [450, 500, 550, 600], (
        "the mismatch this fix removes")
    assert len(aligned_10) < len(aligned_25)
