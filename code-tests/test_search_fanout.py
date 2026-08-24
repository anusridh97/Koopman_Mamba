"""The fleet: `concurrent_trials: N` launches N processes, pinned, once.

`concurrent_trials` replaced `n_jobs` (and, for one day, `workers` -- which
collided with `RuntimeSpec.workers`, the dataloader count). The whole point of
the replacement is that the number now *does* something. `n_jobs` only flipped the sampler's `constant_liar`
while the operator started workers by hand, so a study could declare `n_jobs: 1`
and have eight workers running against it, all proposing from one history. Two
numbers that could disagree became one that cannot.

What has to hold, and is tested here rather than argued:

  * N workers are actually started, one per `concurrent_trials`.
  * Each is pinned to a GPU round-robin, so they do not all pile onto cuda:0 --
    nothing in the repo set CUDA_VISIBLE_DEVICES before this, so eight workers
    shared one GPU and left seven idle.
  * A worker does NOT fan out again. Without the env marker, each child would
    spawn N grandchildren and the process count would be N**depth.
  * Only invocation-intent flags reach a child. There is nothing else to pass --
    the study-redefining flags were removed precisely so a child cannot be
    handed a different study than its siblings.

`_fanout` is exercised with `subprocess.Popen` patched: what matters is the argv
and the environment each worker gets, and actually starting eight trainers to
observe that is neither cheap nor more convincing.
"""
from __future__ import annotations

import pathlib
import sys
from dataclasses import dataclass

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from experimentation.sweep.search.__main__ import (  # noqa: E402
    WORKER_ENV, _fanout, should_fanout, visible_gpus)

pytestmark = pytest.mark.correctness


@dataclass
class _Args:
    study: str = "configs/search/smoke-4m.yaml"
    allow_dirty: bool = False
    force: bool = False
    force_no_eval: bool = False


@dataclass
class _Spec:
    concurrent_trials: int = 1


class _FakePopen:
    """Records what it was asked to launch; exits 0 without running anything."""

    started: list = []

    def __init__(self, cmd, env=None, **kwargs):
        self.cmd = list(cmd)
        self.env = dict(env or {})
        # Recorded so the log wiring is observable: `stdout` is the whole point
        # of the per-worker log, and a test that only checked argv could not see
        # 8 children sharing one stream.
        self.stdout_arg = kwargs.get("stdout")
        self.stderr_arg = kwargs.get("stderr")
        self.stdout_path = getattr(self.stdout_arg, "name", None)
        self.pid = 1000 + len(_FakePopen.started)
        self.returncode = 0
        _FakePopen.started.append(self)

    def wait(self):
        return self.returncode


@pytest.fixture
def spawned(monkeypatch):
    _FakePopen.started = []
    monkeypatch.setattr(
        "experimentation.sweep.search.__main__.subprocess.Popen", _FakePopen)
    return _FakePopen.started


# --------------------------------------------------------------- the fleet ----

@pytest.mark.parametrize("concurrent_trials", [2, 4, 8])
def test_it_starts_one_process_per_worker(spawned, concurrent_trials):
    code, codes = _fanout(_Args(), _Spec(concurrent_trials=concurrent_trials),
                          gpus=["0", "1"])
    assert len(spawned) == concurrent_trials
    assert code == 0
    assert codes == [0] * concurrent_trials


def test_each_worker_is_pinned_round_robin_over_the_visible_gpus(spawned):
    _fanout(_Args(), _Spec(concurrent_trials=4), gpus=["0", "1", "2", "3"])
    pinned = [p.env["CUDA_VISIBLE_DEVICES"] for p in spawned]
    assert pinned == ["0", "1", "2", "3"]


def test_more_workers_than_gpus_wraps_rather_than_crowding_gpu_zero(spawned):
    """The failure this prevents: with no pinning at all, every worker inherits
    the same CUDA_VISIBLE_DEVICES and they all land on one device."""
    _fanout(_Args(), _Spec(concurrent_trials=6), gpus=["0", "1", "2"])
    pinned = [p.env["CUDA_VISIBLE_DEVICES"] for p in spawned]
    assert pinned == ["0", "1", "2", "0", "1", "2"]
    # Two per GPU, not six on one.
    assert sorted(set(pinned)) == ["0", "1", "2"]


def test_with_no_gpu_detected_it_does_not_invent_a_pin(spawned):
    """Pinning to device 0 on a CPU box would be a lie, and CUDA_VISIBLE_DEVICES=""
    disables CUDA entirely -- either would be worse than leaving it alone."""
    _fanout(_Args(), _Spec(concurrent_trials=2), gpus=[])
    for p in spawned:
        assert "CUDA_VISIBLE_DEVICES" not in p.env


# ------------------------------------------------------ no infinite fanout ----

def test_every_worker_is_marked_as_a_worker(spawned):
    """Without this marker each child fans out again: N**depth processes."""
    _fanout(_Args(), _Spec(concurrent_trials=3), gpus=["0"])
    indices = sorted(p.env[WORKER_ENV] for p in spawned)
    assert indices == ["0", "1", "2"]


def test_the_supervisor_fans_out_and_a_worker_never_does():
    """`worker_index is None` is the supervisor; any string is already a worker."""
    assert should_fanout(8, None) is True
    for index in ("0", "1", "7"):
        assert should_fanout(8, index) is False, (
            f"worker {index} would fan out again, one extra process generation")


def test_worker_zero_counts_as_a_worker():
    """The whole reason `should_fanout` is a function. "0" is a non-empty string
    and so truthy, but deciding on truthiness instead of `is None` is an easy
    slip -- and it would make exactly one worker per launch spawn a second
    generation, which looks like a mysterious 2x in the process count."""
    assert should_fanout(8, "0") is False


def test_a_single_worker_study_never_fans_out():
    """concurrent_trials: 1 must run in-process, not spawn one child and
    supervise it -- the child would be the only worker and the parent would
    idle."""
    assert should_fanout(1, None) is False


# ------------------------------------------------------ what reaches a child ----

def test_a_child_is_handed_the_same_study_file(spawned):
    _fanout(_Args(study="configs/search/4m-adaptive.yaml"),
            _Spec(concurrent_trials=2), gpus=["0"])
    for p in spawned:
        assert p.cmd[-1] == "configs/search/4m-adaptive.yaml"
        assert p.cmd[1:3] == ["-m", "experimentation.sweep.search"]


def test_invocation_intent_flags_propagate(spawned):
    _fanout(_Args(allow_dirty=True, force=True, force_no_eval=True),
            _Spec(concurrent_trials=1), gpus=["0"])
    cmd = spawned[0].cmd
    for flag in ("--allow-dirty", "--force", "--force-no-eval"):
        assert flag in cmd


def test_nothing_else_is_passed(spawned):
    """A child's study must be defined entirely by the file. If a flag could
    redefine the budget or the launcher, siblings could disagree while sharing a
    journal -- which is the drift the spec-only rule exists to prevent."""
    _fanout(_Args(), _Spec(concurrent_trials=1), gpus=["0"])
    cmd = spawned[0].cmd
    assert cmd == [sys.executable, "-m", "experimentation.sweep.search",
                   "configs/search/smoke-4m.yaml"]


def test_a_failing_worker_is_reported_not_swallowed(spawned, monkeypatch):
    class _Failing(_FakePopen):
        def wait(self):
            return 3

    monkeypatch.setattr(
        "experimentation.sweep.search.__main__.subprocess.Popen", _Failing)
    code, codes = _fanout(_Args(), _Spec(concurrent_trials=2), gpus=["0"])
    assert code == 1, "a non-zero worker must not report success"
    assert codes == [3, 3]


# ------------------------------------------------------------- gpu discovery ----

def test_an_inherited_allocation_is_respected(monkeypatch):
    """Slurm sets CUDA_VISIBLE_DEVICES for the allocation. Asking the driver
    instead would return every GPU on the node and pin workers to devices this
    job does not own."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    assert visible_gpus() == ["2", "3"]


def test_an_empty_allocation_is_not_read_as_all_gpus(monkeypatch):
    """CUDA_VISIBLE_DEVICES="" means no device. Falling through to nvidia-smi
    here would pin workers onto GPUs the allocation explicitly excluded."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    assert visible_gpus() == []


# ------------------------------------------------ each worker gets its own log ----
#
# `_fanout` called `Popen(cmd, env=env)` with no `stdout=`, so all N children
# inherited ONE stream and interleaved -- a traceback's first line landing under
# another worker's progress. The sbatch script's `WORKER_LOG_DIR` therefore gave
# no worker a log at all, and its comment claiming otherwise was false.
#
# The pattern is `LocalLauncher.submit`'s existing one for its non-blocking path,
# and for the same reasons: a line-buffered file object here, plus
# PYTHONUNBUFFERED in the CHILD env, because `buffering=1` only affects this
# process's object -- the child inherits a raw fd and CPython block-buffers at
# 8 KB when stdout is a file, so without it a `tail -f` shows an empty file until
# the worker exits.


def test_each_worker_gets_a_distinct_log_file(spawned, tmp_path):
    _fanout(_Args(), _Spec(concurrent_trials=4), gpus=["0", "1"],
            log_dir=tmp_path)
    paths = [p.stdout_path for p in spawned]
    assert len(set(paths)) == 4, paths
    for index, path in enumerate(sorted(paths)):
        assert pathlib.Path(path).parent == tmp_path


def test_the_log_name_identifies_the_worker(spawned, tmp_path):
    """`worker-3.log` beats `log.3`: a human greps for the worker index, and the
    `.log` suffix is what `metrics._LOG_GLOBS` matches -- not needed here, but
    consistency with the run-directory logs costs nothing."""
    _fanout(_Args(), _Spec(concurrent_trials=3), gpus=["0"], log_dir=tmp_path)
    names = sorted(pathlib.Path(p.stdout_path).name for p in spawned)
    assert names == ["worker-0.log", "worker-1.log", "worker-2.log"]


def test_stderr_is_folded_into_the_same_file(spawned, tmp_path):
    """A traceback split from the progress line it followed is much harder to
    read, and two files per worker doubles what an operator has to open."""
    import subprocess as _sp

    _fanout(_Args(), _Spec(concurrent_trials=2), gpus=["0"], log_dir=tmp_path)
    for proc in spawned:
        assert proc.stderr_arg == _sp.STDOUT


def test_the_child_gets_pythonunbuffered(spawned, tmp_path):
    """Without it the log is empty until the worker exits, so a multi-hour job
    cannot be told from a hung one."""
    _fanout(_Args(), _Spec(concurrent_trials=2), gpus=["0"], log_dir=tmp_path)
    for proc in spawned:
        assert proc.env["PYTHONUNBUFFERED"] == "1"


def test_the_log_files_are_actually_created(spawned, tmp_path):
    """Opened, not just named: a path the parent never opened is a path nothing
    writes to."""
    _fanout(_Args(), _Spec(concurrent_trials=3), gpus=["0"], log_dir=tmp_path)
    for index in range(3):
        assert (tmp_path / f"worker-{index}.log").exists()


def test_the_parent_does_not_hold_the_pipe(spawned, tmp_path):
    """The failure mode this replaces. If the parent passed `stdout=PIPE` and
    never read it, a worker writing more than the pipe buffer would BLOCK
    forever -- a study that stops making progress with no error at all."""
    import subprocess as _sp

    _fanout(_Args(), _Spec(concurrent_trials=2), gpus=["0"], log_dir=tmp_path)
    for proc in spawned:
        assert proc.stdout_arg is not _sp.PIPE
        assert proc.stdout_arg is not None, (
            "inheriting the parent's stdout is what made 8 workers interleave")


def test_the_files_are_closed_by_the_parent(spawned, tmp_path):
    """`_fanout` waits for every child, so it must not leak N file objects for
    the length of a 12-hour study."""
    _fanout(_Args(), _Spec(concurrent_trials=3), gpus=["0"], log_dir=tmp_path)
    for proc in spawned:
        assert proc.stdout_arg.closed, "a worker log file was left open"


def test_with_no_log_dir_the_behaviour_is_unchanged(spawned):
    """Absent means inherit, which is what an interactive `python -m ... ` run
    wants -- the same asymmetry `LocalLauncher.submit` draws between its waiting
    and non-waiting branches."""
    _fanout(_Args(), _Spec(concurrent_trials=2), gpus=["0"])
    for proc in spawned:
        assert proc.stdout_arg is None


def test_the_log_directory_is_created_if_absent(spawned, tmp_path):
    """The sbatch script mkdir's it, but `_fanout` is also called directly and a
    missing directory would be an IOError after the plan printed."""
    target = tmp_path / "deep" / "nested"
    _fanout(_Args(), _Spec(concurrent_trials=2), gpus=["0"], log_dir=target)
    assert target.is_dir()
    assert (target / "worker-0.log").exists()
