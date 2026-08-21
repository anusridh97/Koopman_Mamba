"""The search CLI: the invocation the package has been advertising all along.

`search/__init__.py` documents `python -m experimentation.sweep.search
<study.yaml>`. Until now that raised "No module named
experimentation.sweep.search.__main__" -- a docstring describing an interface that
did not exist.

Two properties get most of the attention here.

**--dry_run must be a complete, GPU-free view.** That is `sweep/__main__.py`'s
guarantee and the whole reason a study can be checked before it is paid for. It
means the plan is printed BEFORE anything is materialized, and that no optuna
import happens on the dry path -- so the check runs on a machine that has never
installed it.

**A study with no objective producer must refuse loudly.** Nothing in the launch
path writes `run_dir/eval/<ckpt>/quick_eval.json`, so every trial would train
successfully and be recorded FAIL. Discovering that after N GPU jobs is the
expensive failure; a startup error is the cheap one.
"""

import pathlib
import subprocess
import sys

import pytest
import yaml

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

CPU_PY = "/users/jkli/.venvs/koopman-cpu/bin/python"

MINIMAL = {"name": "smoke", "base": "configs/runs/4m-golden.yaml",
           "n_trials": 4, "max_steps": 200, "prune_after_step": 50,
           "logging_steps": 10, "launcher": "local"}


def _spec(tmp_path, **overrides):
    payload = dict(MINIMAL)
    payload.update(overrides)
    path = tmp_path / "study.yaml"
    path.write_text(yaml.safe_dump(payload))
    return path


def _run(*args, optuna=False):
    """Invoke the CLI as a subprocess, so argv parsing and exit codes are real."""
    pp = str(REPO)
    if optuna:
        pp += ":/users/jkli/.venvs/koopman-optuna/site"
    return subprocess.run(
        [CPU_PY, "-m", "experimentation.sweep.search", *map(str, args)],
        capture_output=True, text=True, cwd=str(REPO),
        env={"PYTHONPATH": pp, "PATH": "/usr/bin:/bin", "HOME": str(pathlib.Path.home())},
        timeout=300)


# ------------------------------------------------------------- it now exists ----

def test_the_advertised_invocation_resolves():
    """The bug this file's subject is: __init__.py promised this command."""
    r = _run("--help")
    assert r.returncode == 0, r.stderr
    assert "experimentation.sweep.search" in r.stdout


def test_the_package_docstring_advertises_what_now_exists():
    src = (REPO / "experimentation/sweep/search/__init__.py").read_text()
    assert "python -m experimentation.sweep.search" in src
    assert (REPO / "experimentation/sweep/search/__main__.py").exists(), (
        "__init__.py advertises the CLI; if this file is gone, stop advertising it")


# ----------------------------------------------------------------- dry_run ----

def test_dry_run_prints_a_complete_plan(tmp_path):
    r = _run(_spec(tmp_path), "--dry_run")
    assert r.returncode == 0, r.stderr
    for expected in ("study", "base spec", "budget", "launcher", "journal",
                     "pruning", "objective"):
        assert expected in r.stdout, f"plan omits {expected!r}:\n{r.stdout}"
    assert "nothing materialized" in r.stdout


def test_dry_run_materializes_nothing(tmp_path):
    run_root = tmp_path / "runs"
    r = _run(_spec(tmp_path), "--dry_run", "--run_root", run_root)
    assert r.returncode == 0
    assert not run_root.exists(), "a dry run created a run root"


def test_dry_run_does_not_need_optuna(tmp_path):
    """Deliberate: everything above the optuna line has to work without it, and
    checking a study before paying for it is exactly when you might not have it."""
    r = _run(_spec(tmp_path), "--dry_run")
    assert r.returncode == 0
    assert "optuna" not in r.stderr.lower(), r.stderr


def test_dry_run_warns_rather_than_refuses(tmp_path):
    """A dry run spends nothing, so refusing to print the plan would make the
    missing objective harder to discover rather than easier."""
    r = _run(_spec(tmp_path), "--dry_run")
    assert r.returncode == 0
    assert "WARNING" in r.stdout
    assert "REFUSING" not in r.stdout


# ------------------------------------------------- the missing objective gate ----

def test_a_real_launch_refuses_without_an_objective_producer(tmp_path):
    """The expensive failure is N trained trials scoring nothing. Refuse at
    startup instead."""
    r = _run(_spec(tmp_path), "--run_root", tmp_path / "runs")
    assert r.returncode == 2, f"expected refusal, got {r.returncode}\n{r.stdout}"
    assert "REFUSING TO LAUNCH" in r.stdout
    assert "quick_eval.json" in r.stdout


def test_the_refusal_names_where_the_fix_is_tracked(tmp_path):
    """A refusal that does not say what unblocks it just looks broken."""
    r = _run(_spec(tmp_path), "--run_root", tmp_path / "runs")
    assert "traintask-design" in r.stdout


def test_force_no_eval_says_warning_not_refusing(tmp_path):
    """Printing REFUSING and then launching anyway is how people learn to stop
    reading the output."""
    r = _run(_spec(tmp_path), "--force-no-eval", "--allow-dirty",
             "--run_root", tmp_path / "runs", optuna=True)
    assert "REFUSING" not in r.stdout, r.stdout
    assert "WARNING" in r.stdout


# ------------------------------------------------------------- spec plumbing ----

def test_the_plan_reports_anchors_when_a_design_file_is_given(tmp_path):
    r = _run(_spec(tmp_path, design_file="configs/search/example_anchors.yaml"),
             "--dry_run")
    assert r.returncode == 0
    assert "4 from configs/search/example_anchors.yaml" in r.stdout


def test_no_design_file_says_the_first_trials_are_unprunable(tmp_path):
    """Not decoration: MedianPruner needs COMPLETED trials, so an anchorless
    study's opening trials are both random and unprunable."""
    r = _run(_spec(tmp_path), "--dry_run")
    assert "unprunable" in r.stdout


def test_n_jobs_says_it_spawns_nothing(tmp_path):
    """The single most misreadable knob in the package -- n_jobs only flips
    constant_liar; the fleet is the operator's job."""
    r = _run(_spec(tmp_path, n_jobs=4), "--dry_run")
    assert "spawns" in r.stdout and "4x" in r.stdout


def test_cli_overrides_beat_the_spec(tmp_path):
    r = _run(_spec(tmp_path, n_trials=4), "--dry_run", "--n_trials", "9")
    assert "9 trials" in r.stdout


def test_a_bad_spec_fails_before_anything_is_touched(tmp_path):
    r = _run(_spec(tmp_path, max_steps=10, prune_after_step=10), "--dry_run")
    assert r.returncode != 0
    assert "pruned" in (r.stdout + r.stderr)


# ------------------------------------------------------------- the dependency ----

def test_optuna_is_a_declared_dependency():
    """It was side-loaded via PYTHONPATH and absent from pyproject entirely, so
    a fresh install could not run a study at all."""
    text = (REPO / "pyproject.toml").read_text()
    assert "optuna>=4.0" in text, "optuna must be declared in the lab extra"


def test_the_optuna_floor_is_4_not_3():
    """study.py::make_storage raises on optuna 3's JournalFileStorage spelling,
    so >=3 would install something that cannot create a study."""
    text = (REPO / "pyproject.toml").read_text()
    assert "optuna>=3" not in text
