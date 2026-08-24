"""`configs/search/smoke-fanout.yaml` and the gate that runs it.

**The gap this closes.** `concurrent_trials: 8` has never run under sbatch. Job
445657 proved the ask -> materialize -> launch -> quick_eval -> tell loop closes
on real hardware, but at ONE worker: the supervisor drove the trials itself, so
`should_fanout` returned False and `_fanout` was never called. The entire fleet
path is therefore covered only by tests that patch `subprocess.Popen` -- which
verify argv, env and GPU pinning, and cannot see two OS processes appending to one
journal on a cluster filesystem, a child that cannot import the side-loaded
optuna, `visible_gpus` reading a real allocation, or the post-fanout
reattachment that writes the report.

So there is now a 6-trial / 200-step / 2-worker study whose only purpose is to
run that path for minutes rather than hours, and this file checks the two things
a CPU suite CAN check about it: that the config is shaped so the fanout is
actually exercised, and that the verification job asserts the fanout-specific
properties rather than the single-worker ones that would pass either way.

**What two workers do not prove** is in the config's own header and is repeated
here because it is the honest half: no GPU contention worth the name (both
workers round-robin onto one visible device and time-slice), no host-memory
pressure from 8 trainers, no OOM ladder under that pressure, no `constant_liar`
diversity effect at fleet size 8, and not the 7-trial target overshoot, which
needs 8 workers to reach its bound. It is the cheapest experiment that can
FALSIFY "the fanout works under a scheduler", which is the only claim currently
resting on mocks.
"""
from __future__ import annotations

import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

pytestmark = pytest.mark.correctness

CONFIG = REPO / "configs/search/smoke-fanout.yaml"
GATE = REPO / "scripts/verify_study_e2e.sbatch"


@pytest.fixture(scope="module")
def spec():
    from experimentation.sweep.search.studyspec import load_study_spec
    return load_study_spec(CONFIG)


@pytest.fixture(scope="module")
def gate():
    return GATE.read_text()


# --------------------------------------------------------------- the config ----

def test_it_declares_more_than_one_concurrent_trial(spec):
    """The entire point. At 1 the supervisor drives the trials itself and
    `_fanout` is never reached, which is the state job 445657 verified."""
    assert spec.concurrent_trials >= 2


def test_the_supervisor_would_actually_fan_out(spec):
    """Asserted through the real predicate rather than by reading the number, so
    a change to `should_fanout`'s contract fails here."""
    from experimentation.sweep.search.__main__ import should_fanout

    assert should_fanout(spec.concurrent_trials, None) is True
    # ...and a child does not fan out again, whatever its index. Deciding on
    # truthiness instead would make worker 0 spawn a second generation.
    for raw in ("0", "1", "7", "garbage"):
        assert should_fanout(spec.concurrent_trials, raw) is False


def test_it_is_the_local_launcher(spec):
    """The fleet is `_fanout`'s Popen loop inside ONE allocation. Under `slurm`
    each trial becomes its own job, so the fanout would be waiting on a queue --
    that tests the scheduler, not the fleet."""
    assert spec.launcher == "local"


def test_it_has_more_trials_than_anchors_so_both_paths_are_exercised(spec):
    """A fleet has two ways to get a trial and both can break under concurrency:
    pulling a WAITING enqueued anchor (two workers racing for the same one) and
    asking for a fresh sampled point. With n_trials == n_anchors every trial is
    an anchor and the second path never runs -- which is `smoke-4m.yaml`'s
    situation and why this is a separate file."""
    from experimentation.sweep.search.anchors import load_designs

    designs = load_designs(REPO / spec.design_file, minimum=1)
    assert spec.n_trials > len(designs)


def test_it_buys_more_than_one_sequential_wave(spec):
    """One wave would mean every trial proposed against an empty history, i.e. a
    fanout that never has to reconcile concurrent writes to the journal."""
    waves = spec.n_trials / spec.concurrent_trials
    assert waves >= 2.0, f"{waves} wave(s) is a single round of proposals"


def test_it_is_cheap_enough_to_be_run_before_the_real_thing(spec):
    """A validation step that costs as much as the thing it validates is not a
    validation step. 6 x 200 steps at 4m is minutes; the real study is 256 x 600
    at 25M."""
    assert spec.n_trials * spec.max_steps <= 2000
    assert "4m-golden" in spec.base


def test_its_objective_is_held_out(spec):
    """`train.py` is `val_dir = args.eval_data_dir or args.data_dir`, so omitting
    this silently scores on the training shard -- and a fanout gate that ranks on
    training loss would still pass every fanout assertion while being the wrong
    experiment."""
    from experimentation.sweep.spec import _base_sections

    assert spec.eval_data_dir
    assert spec.eval_data_dir != _base_sections(
        str(REPO / spec.base))["data"]["shard_dir"]


def test_it_pins_the_exact_backend_like_every_other_smoke_study(spec):
    """`exact_auto` sends the 4m geometry down the 160x Python reference scan
    (job 439754 sat 21 minutes without logging a step); the retired
    `proxy_chunked` was 92%-152% wrong in the forward."""
    assert spec.backend_policy == "exact_invchol"


def test_its_run_root_is_distinct_from_the_other_smoke_studies(spec):
    """Sharing a root with `smoke-4m` would mean sharing `_studies/`, and while
    `study_id` differs so the journals differ, the RUN directories are content-
    addressed and would collide across studies that propose the same config --
    which is correct behaviour and makes a gate's output ambiguous."""
    from experimentation.sweep.search.studyspec import load_study_spec

    others = [load_study_spec(p).run_root
              for p in sorted((REPO / "configs/search").glob("smoke-*.yaml"))
              if p != CONFIG]
    assert spec.run_root not in others


def test_the_dry_run_resolves_the_whole_plan():
    """The GPU-free surface. Anchor resolution, space restriction and the fleet
    plan all have to succeed before a single second of allocation is spent."""
    import subprocess

    result = subprocess.run(
        [sys.executable, "-m", "experimentation.sweep.search", str(CONFIG),
         "--dry_run"],
        capture_output=True, text=True, cwd=REPO,
        env=_env())
    assert result.returncode == 0, result.stdout + result.stderr
    assert "fleet        2 concurrent (constant_liar ON)" in result.stdout
    assert "4 anchor(s) resolve" in result.stdout


def test_the_dry_run_says_this_study_has_no_noise_floor():
    """It genuinely does not, and that must be visible rather than inferred: this
    is a WIRING gate, so it declares no `reference_group` and cannot state any
    effect against a measured spread. A gate that silently looked like a
    scientific study would be the worse artefact."""
    import subprocess

    result = subprocess.run(
        [sys.executable, "-m", "experimentation.sweep.search", str(CONFIG),
         "--dry_run"],
        capture_output=True, text=True, cwd=REPO, env=_env())
    assert "noise floor  NONE" in result.stdout, result.stdout


def _env():
    import os
    return {**os.environ,
            "PYTHONPATH": f"{REPO}:/users/jkli/.venvs/koopman-optuna/site"}


# ----------------------------------------------------------------- the gate ----

def test_the_gate_can_be_pointed_at_this_study(gate):
    """`STUDY_CONFIG` is the seam, and it already existed -- so the fanout gate
    needs no new script, which is the reason this config is the whole deliverable
    rather than a second sbatch file to keep in step."""
    assert 'STUDY_CONFIG="${STUDY_CONFIG:-configs/search/smoke-4m.yaml}"' in gate
    assert "smoke-fanout.yaml" in gate, (
        "the gate does not mention how to run the fanout study, so nobody will")


def test_the_gate_sets_a_per_worker_log_directory(gate):
    """`_fanout` writes `worker-<index>.log` only when
    `KOOPMAN_SEARCH_WORKER_LOG_DIR` is set. Without it every child inherits ONE
    stream and a traceback's first line lands under another worker's progress."""
    assert 'export KOOPMAN_SEARCH_WORKER_LOG_DIR=' in gate
    from experimentation.sweep.search.__main__ import WORKER_LOG_DIR_ENV

    assert WORKER_LOG_DIR_ENV in gate, (
        f"the gate exports a name that {WORKER_LOG_DIR_ENV} does not read -- "
        f"which is what `WORKER_LOG_DIR` was for a while")


@pytest.mark.parametrize("needle,why", [
    ("duplicate numbers",
     "two processes on one journal must not hand one trial to both"),
    ("sampler_seeds",
     "each worker must sample from its own stream"),
    ("model_seeds",
     "the MODEL seed must NOT move with the worker"),
    ("anchor trial(s), run more than once",
     "two workers racing for one WAITING anchor is the fanout-specific loss"),
    ("worker logs:",
     "per-worker logs must be separate files"),
    ("trials.csv rows",
     "the report must come from a REATTACHED study, not the stale snapshot"),
])
def test_the_gate_checks_each_fanout_specific_property(gate, needle, why):
    assert needle in gate, why


def test_the_fanout_verdict_reaches_the_gates_exit_code(gate):
    """A check that prints and does not fail is a check nobody acts on. This
    file's own history is the argument: the venv sanity check printed
    `STATUS venv=1` and ran the whole study anyway, producing "4 trials, 4
    failed" -- indistinguishable from a real wiring failure."""
    assert "fanout_ok" in gate
    assert "and fanout_ok" in gate, (
        "the fanout checks are computed and not folded into `ok`, so the job "
        "would report PASS with a broken fleet")


def test_the_gate_says_nothing_about_a_fanout_when_there_is_none(gate):
    """The default study is single-worker, and these assertions must not fail it
    -- a gate that fails on its own default is a gate that gets disabled."""
    assert "no fanout to check" in gate
    assert "if spec.concurrent_trials <= 1:" in gate
