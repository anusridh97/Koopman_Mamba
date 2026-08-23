"""The fleet: `workers: N` launches N processes, pinned, and no worker re-launches.

`workers` replaced `n_jobs`, and the whole point of the replacement is that the
number now *does* something. `n_jobs` only flipped the sampler's `constant_liar`
while the operator started workers by hand, so a study could declare `n_jobs: 1`
and have eight workers running against it, all proposing from one history. Two
numbers that could disagree became one that cannot.

What has to hold, and is tested here rather than argued:

  * N workers are actually started, one per `workers`.
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
    workers: int = 1


class _FakePopen:
    """Records what it was asked to launch; exits 0 without running anything."""

    started: list = []

    def __init__(self, cmd, env=None, **kwargs):
        self.cmd = list(cmd)
        self.env = dict(env or {})
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

@pytest.mark.parametrize("workers", [2, 4, 8])
def test_it_starts_one_process_per_worker(spawned, workers):
    code, codes = _fanout(_Args(), _Spec(workers=workers), gpus=["0", "1"])
    assert len(spawned) == workers
    assert code == 0
    assert codes == [0] * workers


def test_each_worker_is_pinned_round_robin_over_the_visible_gpus(spawned):
    _fanout(_Args(), _Spec(workers=4), gpus=["0", "1", "2", "3"])
    pinned = [p.env["CUDA_VISIBLE_DEVICES"] for p in spawned]
    assert pinned == ["0", "1", "2", "3"]


def test_more_workers_than_gpus_wraps_rather_than_crowding_gpu_zero(spawned):
    """The failure this prevents: with no pinning at all, every worker inherits
    the same CUDA_VISIBLE_DEVICES and they all land on one device."""
    _fanout(_Args(), _Spec(workers=6), gpus=["0", "1", "2"])
    pinned = [p.env["CUDA_VISIBLE_DEVICES"] for p in spawned]
    assert pinned == ["0", "1", "2", "0", "1", "2"]
    # Two per GPU, not six on one.
    assert sorted(set(pinned)) == ["0", "1", "2"]


def test_with_no_gpu_detected_it_does_not_invent_a_pin(spawned):
    """Pinning to device 0 on a CPU box would be a lie, and CUDA_VISIBLE_DEVICES=""
    disables CUDA entirely -- either would be worse than leaving it alone."""
    _fanout(_Args(), _Spec(workers=2), gpus=[])
    for p in spawned:
        assert "CUDA_VISIBLE_DEVICES" not in p.env


# ------------------------------------------------------ no infinite fanout ----

def test_every_worker_is_marked_as_a_worker(spawned):
    """Without this marker each child fans out again: N**depth processes."""
    _fanout(_Args(), _Spec(workers=3), gpus=["0"])
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
    """workers: 1 must run in-process, not spawn one child and supervise it --
    the child would be the only worker and the parent would idle."""
    assert should_fanout(1, None) is False


# ------------------------------------------------------ what reaches a child ----

def test_a_child_is_handed_the_same_study_file(spawned):
    _fanout(_Args(study="configs/search/4m-adaptive.yaml"), _Spec(workers=2),
            gpus=["0"])
    for p in spawned:
        assert p.cmd[-1] == "configs/search/4m-adaptive.yaml"
        assert p.cmd[1:3] == ["-m", "experimentation.sweep.search"]


def test_invocation_intent_flags_propagate(spawned):
    _fanout(_Args(allow_dirty=True, force=True, force_no_eval=True),
            _Spec(workers=1), gpus=["0"])
    cmd = spawned[0].cmd
    for flag in ("--allow-dirty", "--force", "--force-no-eval"):
        assert flag in cmd


def test_nothing_else_is_passed(spawned):
    """A child's study must be defined entirely by the file. If a flag could
    redefine the budget or the launcher, siblings could disagree while sharing a
    journal -- which is the drift the spec-only rule exists to prevent."""
    _fanout(_Args(), _Spec(workers=1), gpus=["0"])
    cmd = spawned[0].cmd
    assert cmd == [sys.executable, "-m", "experimentation.sweep.search",
                   "configs/search/smoke-4m.yaml"]


def test_a_failing_worker_is_reported_not_swallowed(spawned, monkeypatch):
    class _Failing(_FakePopen):
        def wait(self):
            return 3

    monkeypatch.setattr(
        "experimentation.sweep.search.__main__.subprocess.Popen", _Failing)
    code, codes = _fanout(_Args(), _Spec(workers=2), gpus=["0"])
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
