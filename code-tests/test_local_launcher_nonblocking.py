"""A locally-launched trial has to be cancellable, or pruning is theatre.

`wait_for_objective` prunes by tailing a running job's log and then cancelling it.
That works for the Slurm launcher because `submit` returns as soon as `sbatch`
does, leaving the job running and cancellable by job id.

`LocalLauncher.submit` used `subprocess.run`, which blocks until training
finishes. The driver's sequence is `submit(...)` then `read_objective(run_dir)`,
so with a blocking submit the reader only started AFTER training ended -- nothing
to watch, nothing to prune. Every locally-launched trial ran to `max_steps`,
silently. No error, no warning.

That mode matters: holding one GPU allocation and running trials back-to-back
inside it avoids paying a queue wait per trial, and it is the natural way to run a
study when queue latency is comparable to trial length.

Two things make it work, and both are pinned here.

**Blocking stays the default.** `run/__main__.py:56` documents depending on it --
its run-directory claim is released before hand-off *because* submit blocks for
the whole run -- and `sweep/__main__.py:120` submits one cell at a time expecting
each to finish. Flipping the default would launch every cell of a sweep at once.

**Cancellation goes through the run directory, not a process handle.** A poller
gets only a `run_dir`; that is the whole interface (see metrics.py's module
docstring). So the launcher records its PID there, exactly as the Slurm path is
cancellable from its `slurm-*.out` filenames alone.
"""

import ast
import os
import pathlib
import signal
import subprocess
import sys
import time

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from experimentation.run.launchers import LocalLauncher  # noqa: E402
from experimentation.sweep.search import metrics as M  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]


def _spawn_group(seconds=60):
    """A child in its own session, standing in for a training subprocess."""
    return subprocess.Popen([sys.executable, "-c", f"import time; time.sleep({seconds})"],
                            start_new_session=True)


def _alive(pid):
    """For a process that is NOT our child (a reparented grandchild)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _running(proc):
    """For our own child. os.kill(pid, 0) succeeds on a zombie, so a dead-but-
    unreaped child would look alive forever; poll() reaps and reports."""
    return proc.poll() is None


# ------------------------------------------------------------ the default ----

def test_submit_still_defaults_to_blocking():
    """Read the signature rather than run a job: the default is the contract two
    other modules depend on, and it must not drift."""
    src = (REPO / "experimentation/run/launchers.py").read_text()
    tree = ast.parse(src)
    cls = next(n for n in ast.walk(tree)
               if isinstance(n, ast.ClassDef) and n.name == "LocalLauncher")
    fn = next(n for n in cls.body
              if isinstance(n, ast.FunctionDef) and n.name == "submit")
    kwonly = {a.arg: d for a, d in zip(fn.args.kwonlyargs, fn.args.kw_defaults)}
    assert "wait" in kwonly, "submit must expose an explicit wait= switch"
    assert kwonly["wait"].value is True, (
        "wait must default to True -- run/__main__.py and sweep/__main__.py both "
        "rely on submit blocking for the whole run")


def test_run_main_still_documents_the_blocking_assumption():
    """If someone flips the default, this is the comment that explains why the
    run directory claim was safe. Fail loudly rather than let it rot."""
    src = (REPO / "experimentation/run/__main__.py").read_text()
    assert "blocks for the whole" in src, (
        "run/__main__.py's rationale for releasing its claim before hand-off is "
        "gone; re-check whether the claim window is still correct")


# --------------------------------------------------------- the PID channel ----

def test_local_pid_round_trips(tmp_path):
    (tmp_path / LocalLauncher.PID_FILE).write_text("12345\n")
    assert M._local_pid(tmp_path) == 12345


@pytest.mark.parametrize("body", ["", "not-a-pid", "  \n"])
def test_local_pid_is_none_when_unreadable(tmp_path, body):
    """A missing or corrupt pidfile must not raise inside a pruning poll -- the
    poll would die and the trial would hang instead of being scored."""
    if body:
        (tmp_path / LocalLauncher.PID_FILE).write_text(body)
    assert M._local_pid(tmp_path) is None


def test_local_pid_absent_directory(tmp_path):
    assert M._local_pid(tmp_path / "nope") is None


# ------------------------------------------------------------ cancellation ----

def test_default_cancel_kills_a_local_run(tmp_path):
    proc = _spawn_group()
    (tmp_path / LocalLauncher.PID_FILE).write_text(f"{proc.pid}\n")
    assert _running(proc)

    M._default_cancel(tmp_path)

    deadline = time.time() + 10
    while time.time() < deadline and _running(proc):
        time.sleep(0.1)
    assert not _running(proc), "a pruned local run must actually stop"


def test_default_cancel_signals_the_whole_group(tmp_path):
    """A torchrun launch has one worker per GPU. Killing only the parent leaves
    them holding the hardware, so pruning would free nothing -- the entire point
    of pruning is the GPU-hours."""
    parent = subprocess.Popen(
        [sys.executable, "-c",
         "import subprocess,sys,time;"
         "c=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']);"
         "print(c.pid, flush=True); time.sleep(60)"],
        start_new_session=True, stdout=subprocess.PIPE, text=True)
    child_pid = int(parent.stdout.readline().strip())
    assert _running(parent) and _alive(child_pid)

    (tmp_path / LocalLauncher.PID_FILE).write_text(f"{parent.pid}\n")
    M._default_cancel(tmp_path)

    deadline = time.time() + 10
    while time.time() < deadline and (_running(parent) or _alive(child_pid)):
        time.sleep(0.1)
    assert not _running(parent), "parent survived"
    assert not _alive(child_pid), "grandchild survived -- it would still hold a GPU"


def test_default_cancel_tolerates_a_stale_pid(tmp_path):
    """A pidfile outliving its process is normal -- the run finished. Cancelling
    must be a no-op, not an exception inside a poll loop."""
    proc = _spawn_group(seconds=0)
    proc.wait(timeout=5)          # reaped, so the pid is genuinely stale
    (tmp_path / LocalLauncher.PID_FILE).write_text(f"{proc.pid}\n")
    M._default_cancel(tmp_path)          # must not raise


def test_default_cancel_with_no_pidfile_is_a_noop(tmp_path):
    M._default_cancel(tmp_path)          # the Slurm-only path; must not raise


def test_default_cancel_still_reaches_slurm(tmp_path, monkeypatch):
    """The local path is additive -- it must not have displaced scancel."""
    (tmp_path / "slurm-987654.out").write_text("step 10/100 | loss 1.0 | ppl 2 | lr 1e-4 | 1.0K tok/s\n")
    calls = []
    monkeypatch.setattr(M.subprocess, "run",
                        lambda cmd, **kw: calls.append(cmd) or None)
    M._default_cancel(tmp_path)
    assert calls == [["scancel", "987654"]]


# ------------------------------------- the log the pruner has to be able to read ----

def test_the_log_filename_matches_what_the_pruner_globs():
    """The gap job 439827 exposed. Not blocking was necessary and not sufficient:
    a local run left NO log in its run dir, so read_progress's globs matched
    nothing and pruning could never fire. The study completed 4 trials with real
    objectives and 'trials with reported intermediate steps: 0'.

    Slurm gets this free -- its sbatch template routes stdout to
    run_dir/slurm-%j.out. Local execution had to be taught."""
    from experimentation.sweep.search.metrics import _LOG_GLOBS
    import fnmatch
    assert any(fnmatch.fnmatch(LocalLauncher.LOG_FILE, g) for g in _LOG_GLOBS), (
        f"{LocalLauncher.LOG_FILE!r} matches none of {_LOG_GLOBS}, so a poller "
        "cannot find a local run's progress")


def test_a_non_blocking_local_run_writes_that_log(tmp_path, monkeypatch):
    """Exercises submit's wait=False branch with a stub command, so the file and
    the pidfile are both real without needing a GPU or a RunSpec."""
    from experimentation.run import launchers as L

    monkeypatch.setattr(L, "write_model_config", lambda spec, run_dir: None)
    launcher = LocalLauncher()
    monkeypatch.setattr(
        launcher, "build_command",
        lambda spec, run_dir, resume=False: [
            sys.executable, "-c",
            "print('step     10/200 | loss 1.0 | ppl 2.7 | lr 1e-4 | 1.0K tok/s')"])

    proc = launcher.submit(object(), tmp_path, wait=False)
    proc.wait(timeout=30)

    log = tmp_path / LocalLauncher.LOG_FILE
    assert log.exists(), f"no {LocalLauncher.LOG_FILE} written"
    assert (tmp_path / LocalLauncher.PID_FILE).exists()

    # And the pruner's own parser must be able to read it back.
    from experimentation.sweep.search.metrics import read_progress
    points = read_progress(tmp_path)
    assert [p.step for p in points] == [10], f"read_progress got {points}"


def test_a_blocking_local_run_still_inherits_stdout(tmp_path, monkeypatch):
    """The asymmetry is deliberate: an interactive run wants its output on the
    terminal. Redirecting by default would silently take that away from
    `python -m experimentation.run --launcher local`."""
    from experimentation.run import launchers as L

    monkeypatch.setattr(L, "write_model_config", lambda spec, run_dir: None)
    launcher = LocalLauncher()
    monkeypatch.setattr(
        launcher, "build_command",
        lambda spec, run_dir, resume=False: [sys.executable, "-c", "print('hi')"])

    launcher.submit(object(), tmp_path, wait=True)
    assert not (tmp_path / LocalLauncher.LOG_FILE).exists(), (
        "wait=True must not redirect; the terminal is the point")


def test_slurm_submit_accepts_wait_without_caring():
    """So a caller can pass wait=False to any launcher without first asking which
    one it holds. sbatch is asynchronous whether or not anyone asks."""
    import inspect
    from experimentation.run.launchers import SlurmLauncher
    params = inspect.signature(SlurmLauncher.submit).parameters
    assert "wait" in params, "SlurmLauncher.submit must accept wait"
    assert params["wait"].default is True


def test_the_driver_submits_without_waiting():
    """The other half. A tailable log is useless if the driver still blocks until
    training is over before it starts reading."""
    src = (REPO / "experimentation/sweep/search/driver.py").read_text()
    assert "wait=False" in src, (
        "run_trial must submit with wait=False, or the objective reader starts "
        "after training ends and there is nothing left to prune")
