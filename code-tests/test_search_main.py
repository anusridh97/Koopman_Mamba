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
    # run_root comes from the spec: it is no longer overridable on the command
    # line, because a flag that redirects the journal without changing study_id
    # lets two workers believe they are collaborating when they are not.
    r = _run(_spec(tmp_path, run_root=str(run_root)), "--dry_run")
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
    r = _run(_spec(tmp_path, run_root=str(tmp_path / "runs")), "--allow-dirty")
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


def test_the_plan_reports_the_fleet_and_its_placement(tmp_path):
    """`concurrent_trials` replaced `n_jobs`, and the difference is that it launches.

    The old test here pinned the OPPOSITE property -- that the knob "spawns
    nothing" -- which was correct then and is the bug now: a study could declare
    n_jobs: 1 while eight hand-started workers proposed from one history.
    """
    r = _run(_spec(tmp_path, concurrent_trials=4), "--dry_run")
    assert "fleet" in r.stdout and "4 concurrent" in r.stdout
    assert "constant_liar ON" in r.stdout
    assert "placement" in r.stdout


def test_a_single_worker_study_says_nothing_about_a_fleet(tmp_path):
    """Guard the guard: the fleet lines must be CONDITIONAL, or the assertions
    above would pass for any spec at all."""
    r = _run(_spec(tmp_path, concurrent_trials=1), "--dry_run")
    assert "concurrent" not in r.stdout
    # "gpu pinning", not "placement": `placement` is now also a SEARCH AXIS name
    # and is printed unconditionally in the space block, so asserting on the bare
    # word would test the axis list rather than the fleet block.
    assert "gpu pinning" not in r.stdout
    assert "sampler seeds" not in r.stdout


def test_a_fleet_too_wide_for_its_budget_is_warned_about(tmp_path):
    """8 workers on a 4-trial study is half a wave: nearly every trial is
    proposed before any finishes, so TPE has nothing to learn from and the study
    is random search wearing a sampler. Said before the GPU time is spent."""
    r = _run(_spec(tmp_path, concurrent_trials=8, n_trials=4), "--dry_run")
    assert "WARNING" in r.stdout
    assert "sequential wave" in r.stdout


def test_a_well_sized_fleet_is_not_warned_about(tmp_path):
    """Guard the guard, and a real defect this caught: the first version compared
    the fleet size against the pruner's n_startup_trials, which CAPS at 6 -- so every
    fleet of 6+ warned no matter how large the study, including the 150-trial
    config this feature exists for. A warning that always fires is noise."""
    r = _run(_spec(tmp_path, concurrent_trials=8, n_trials=150), "--dry_run")
    assert "WARNING" not in r.stdout, r.stdout
    # ...but the unprunable first wave is still stated, because that is always true.
    assert "cannot be pruned" in r.stdout


def test_the_removed_overrides_are_really_gone(tmp_path):
    """Each of these resolved BEFORE study_id was computed but was not part of
    it, so the flag changed the study while leaving its identity alone -- two
    workers, one flagged, sharing a journal and disagreeing. argparse must now
    reject them rather than the spec silently losing."""
    for flag, value in (("--n_trials", "9"), ("--launcher", "slurm"),
                        ("--run_root", "/tmp/whatever")):
        r = _run(_spec(tmp_path), "--dry_run", flag, value)
        assert r.returncode != 0, f"{flag} was still accepted"
        assert "unrecognized arguments" in r.stderr, (
            f"{flag} failed for the wrong reason:\n{r.stderr}")


def test_the_spec_is_what_sets_the_budget(tmp_path):
    """The positive half of the above: with no flag able to override it, the
    number in the file is the number in the plan."""
    r = _run(_spec(tmp_path, n_trials=7), "--dry_run")
    assert "7 trials" in r.stdout


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


def test_the_plan_prints_the_journal_filename_the_code_actually_writes(tmp_path):
    """These drifted apart: the plan said journal.log and study.py writes
    optuna_journal.log. Cosmetic until you act on it -- a second worker told to
    open "the journal in the plan output" gets an empty file and believes it is
    collaborating on a study it cannot see."""
    src = (REPO / "experimentation/sweep/search/study.py").read_text()
    assert "optuna_journal.log" in src, "study.py's journal name moved"
    r = _run(_spec(tmp_path), "--dry_run")
    assert r.returncode == 0
    assert "optuna_journal.log" in r.stdout, (
        "the plan must name the file make_storage actually creates")


# ------------------------------------------------- the entry point's kwargs ----

def test_every_keyword_the_cli_hands_drive_is_one_drive_accepts():
    """The CLI's `drive(...)` call is not executed by any test.

    `--dry_run` returns before reaching it, and
    test_a_real_launch_now_proceeds_past_the_gate only asserts the run gets as
    far as needing optuna. So a keyword renamed in driver.py leaves this call
    site stale and the whole suite stays green -- which is exactly what happened
    when `read_objective_factory` was collapsed into `objective_reader_for`: the
    dry run reported success while a real study would have died at trial 0 on a
    TypeError.

    Same drift that made 22 tests fail on this branch when `launcher.submit`
    grew `wait`: a signature moved and its callers did not.

    Read from the SOURCE rather than by importing driver.py, which needs optuna
    at module scope. The mismatch is a static property, so the guard should hold
    in the CPU environment too -- and this file's whole point is that the CLI's
    plan is inspectable without optuna installed.
    """
    import ast

    def _signature_names(tree, func):
        node = next(n for n in ast.walk(tree)
                    if isinstance(n, ast.FunctionDef) and n.name == func)
        a = node.args
        names = {p.arg for p in (*a.posonlyargs, *a.args, *a.kwonlyargs)}
        return names, a.kwarg is not None

    driver = ast.parse(
        (REPO / "experimentation/sweep/search/driver.py").read_text())
    drive_names, drive_takes_kwargs = _signature_names(driver, "drive")
    run_trial_names, _ = _signature_names(driver, "run_trial")
    assert drive_takes_kwargs, (
        "drive() no longer takes **kwargs; this test's union with run_trial's "
        "parameters is now the wrong model of how the CLI reaches it")
    accepted = drive_names | run_trial_names

    src = (REPO / "experimentation/sweep/search/__main__.py").read_text()
    calls = [n for n in ast.walk(ast.parse(src))
             if isinstance(n, ast.Call)
             and getattr(n.func, "id", None) == "drive"]
    assert calls, "__main__.py no longer calls drive() -- has the CLI moved?"

    for call in calls:
        passed = {k.arg for k in call.keywords if k.arg}
        unknown = sorted(passed - accepted)
        assert not unknown, (
            f"__main__.py passes {unknown} to drive(), which accepts "
            f"{sorted(accepted)}. A real study would fail at trial 0; --dry_run "
            f"returns before this call and would not notice.")
