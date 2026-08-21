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

**The objective gate must keep its teeth.** Optuna reads a trial's score from
`run_dir/eval/<ckpt>/quick_eval.json`. When nothing in the launch path wrote it,
every trial trained successfully and was recorded FAIL -- N GPU jobs for nothing --
so the CLI refused at startup. `train.py --eval_on_final` now writes it, so the
refusal no longer fires; what is tested is the MECHANISM, because removing the
hook must bring the refusal back rather than silently restoring N-trials-N-FAILs.
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


def test_the_objective_producer_is_now_detected():
    """The gate's SUBJECT changed under it: train.py now writes
    quick_eval.json via --eval_on_final, so the refusal must no longer fire.

    Three tests here previously pinned the refusal itself. That was pinning a
    transient state -- when the state flipped, they failed, correctly. Rewritten
    to test the MECHANISM, which is what has to keep working: the gate must
    detect a producer when there is one, and refuse when there is not."""
    import importlib.util as u
    spec = u.spec_from_file_location(
        "_search_main", REPO / "experimentation/sweep/search/__main__.py")
    mod = u.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod._objective_producer_exists(REPO) is True, (
        "train.py's --eval_on_final should satisfy the gate")


def test_the_gate_still_refuses_when_no_producer_exists(tmp_path):
    """The gate has to keep teeth. Point it at a tree with a train.py that does
    not write quick_eval.json and it must still say no -- otherwise removing the
    hook would silently reintroduce N-trials-N-FAILs."""
    import importlib.util as u
    spec = u.spec_from_file_location(
        "_search_main2", REPO / "experimentation/sweep/search/__main__.py")
    mod = u.module_from_spec(spec)
    spec.loader.exec_module(mod)

    fake = tmp_path / "repo"
    (fake / "experimentation/training").mkdir(parents=True)
    (fake / "experimentation/run").mkdir(parents=True)
    (fake / "experimentation/training/train.py").write_text("def train(): pass\n")
    (fake / "experimentation/run/launchers.py").write_text("class L: pass\n")
    assert mod._objective_producer_exists(fake) is False


def test_a_real_launch_now_proceeds_past_the_gate(tmp_path):
    """The end of the search's one hard blocker. It no longer exits 2; it gets
    as far as needing optuna, which is a dependency problem rather than a
    design one."""
    r = _run(_spec(tmp_path), "--run_root", tmp_path / "runs", "--allow-dirty")
    assert r.returncode != 2, (
        f"still refusing at the objective gate:\n{r.stdout}")
    assert "REFUSING TO LAUNCH" not in r.stdout


def test_dry_run_no_longer_warns_about_a_missing_objective(tmp_path):
    r = _run(_spec(tmp_path), "--dry_run")
    assert r.returncode == 0
    assert "quick_eval.json" not in r.stdout, (
        "the objective warning should be gone now that a producer exists")


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
