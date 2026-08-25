"""The four method guarantees whose existing coverage was documented as thin.

An audit of `code-tests/` against the study's ten method claims found seven
STRONG, and four where the test either restated the rule it was testing or
stopped one hop short of the thing that can actually break. This file closes
those four. Each section says which claim it is and what the previous coverage
could not see.

Deliberately NOT parallel files: where a stronger assertion belongs in an
existing test it was strengthened there (see
`test_sampler_and_pruner_startup.py::test_enqueue_is_idempotent_...`). What is
here is what needed machinery the existing files do not have -- threads, a real
journal, an argv, a study file read at runtime.
"""
from __future__ import annotations

import json
import pathlib
import sys
import threading

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

optuna = pytest.importorskip("optuna")

pytestmark = pytest.mark.correctness

STUDY = REPO / "configs/search/proxy-256x17-interactions-v1.yaml"


@pytest.fixture(autouse=True)
def _quiet_optuna():
    previous = optuna.logging.get_verbosity()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    yield
    optuna.logging.set_verbosity(previous)


# ============================================================================
# Claim 6: eight GPU-pinned workers share one study without duplicate trials.
#
# Previous coverage, and its own docstring says so: `test_search_fanout.py`
# patches `subprocess.Popen`, so it verifies argv, env and GPU pinning but never
# that two LIVE workers attached to one journal get disjoint trials.
# `test_search_fleet_overshoot.py` re-implements `drive`'s loop condition rather
# than calling it, and records the real property as "not done rather than
# implied" -- naming exactly this test: "`drive` run in W threads against a
# lock-protected fake storage, with the runner telling PRUNED and FAIL".
#
# So that is what this is. Threads rather than processes because the property
# under test is about the SHARED STORAGE, and threads share a file-backed
# JournalStorage the same way processes do while staying debuggable in one
# pytest run. Every worker runs the real `drive` against the real study object.
# ============================================================================

class _CountingRunner:
    """Stands in for `run_trial`: records the trial, tells the study, moves on.

    Injected in place of the launch path because what is under test is the
    ask/tell contract under concurrency, not materialization -- and 8 threads
    each materializing run directories would test the filesystem instead.

    Tells a mix of COMPLETE, PRUNED and FAIL, because `_finished` counts all
    three and a runner that only ever completes would not exercise the state
    machine the loop condition reads.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.seen: list = []
        self.by_worker: dict = {}

    def __call__(self, study, trial, *, worker_id=None, **_kwargs):
        with self.lock:
            self.seen.append(trial.number)
            self.by_worker.setdefault(worker_id, []).append(trial.number)
            position = len(self.seen)
        if position % 7 == 0:
            trial.report(9.0, 10)
            study.tell(trial, state=optuna.trial.TrialState.PRUNED)
        elif position % 11 == 0:
            trial.set_user_attr("failure", "simulated")
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
        else:
            study.tell(trial, 1.0 + position / 100.0)
        from experimentation.sweep.search.driver import TrialOutcome
        return TrialOutcome(trial_number=trial.number, state="complete",
                            run_id="", group_id="",
                            run_dir=pathlib.Path("."), objective=1.0,
                            anchor=trial.user_attrs.get("anchor_name"))


def _fleet(tmp_path, *, workers, n_trials, designs=None, monkeypatch=None):
    """Run `drive` in `workers` threads against ONE file-backed journal."""
    from experimentation.sweep.search import driver as driver_module
    from experimentation.sweep.search.study import create_study, enqueue_anchors

    space = {"x": {"kind": "categorical", "choices": [1, 2, 3, 4, 5, 6, 7, 8]},
             "y": {"kind": "float", "low": 0.0, "high": 1.0}}
    runner = _CountingRunner()
    monkeypatch.setattr(driver_module, "run_trial", runner)

    kwargs = dict(study_name="fleet", study_dir=tmp_path, seed=2026,
                  concurrent_trials=workers, sampler="tpe",
                  n_trials=n_trials, prune_after_step=5, logging_steps=1)
    supervisor = create_study(**kwargs)
    if designs is not None:
        from koopman_lm.config import build_config
        enqueue_anchors(supervisor, designs, build_config("50m"),
                        _real_space(), base_lr=4e-4)

    errors: list = []

    def work(index):
        try:
            study = create_study(worker_id=index, **kwargs)
            driver_module.drive(study, n_trials=n_trials, space=space,
                                worker_id=index)
        except BaseException as exc:                    # noqa: BLE001
            errors.append(f"worker {index}: {type(exc).__name__}: {exc}")

    threads = [threading.Thread(target=work, args=(i,)) for i in range(workers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=180)
    assert not any(t.is_alive() for t in threads), "a worker thread hung"
    assert not errors, errors
    return create_study(**kwargs), runner


def _real_space():
    from experimentation.run.spec import resolve_model_config
    from experimentation.sweep.search.space import restrict_space, search_space
    from experimentation.sweep.search.studyspec import load_study_spec
    from experimentation.sweep.spec import _base_sections

    spec = load_study_spec(STUDY)
    sections = _base_sections(str(REPO / spec.base))
    model = resolve_model_config(sections["model"])
    return restrict_space(search_space(model, base_name=spec.base), model,
                          axes=spec.search_axes, fixed=spec.fixed_params)


def test_eight_workers_never_run_one_trial_twice(tmp_path, monkeypatch):
    """THE claim. `study.ask` against a shared journal must hand each trial to
    exactly one worker; a duplicate would mean two GPUs training the same config
    and one of the two results silently overwriting the other's directory."""
    study, runner = _fleet(tmp_path, workers=8, n_trials=40,
                           monkeypatch=monkeypatch)

    assert len(runner.seen) == len(set(runner.seen)), (
        f"a trial was handed to two workers: "
        f"{sorted(n for n in set(runner.seen) if runner.seen.count(n) > 1)}")
    numbers = [t.number for t in study.trials]
    assert len(numbers) == len(set(numbers))


def test_the_work_is_actually_shared_rather_than_serialised(tmp_path,
                                                            monkeypatch):
    """Guards the guard. A harness in which one worker did everything would
    satisfy the no-duplicates test perfectly, so the disjointness above only
    means something alongside this."""
    _study, runner = _fleet(tmp_path, workers=8, n_trials=40,
                            monkeypatch=monkeypatch)

    busy = [w for w, trials in runner.by_worker.items() if trials]
    assert len(busy) >= 2, (
        f"only worker(s) {busy} ever ran a trial, so this fleet is serial and "
        f"the disjointness assertion is vacuous")


def test_the_fleet_reaches_its_target_and_overshoots_only_within_the_bound(
        tmp_path, monkeypatch):
    """`drive` loops on `_finished(study) < n_trials` and W workers evaluate that
    independently, so up to W-1 trials can be in flight when the target is hit.
    Asserted against the REAL loop rather than a re-implementation of it, which
    is the gap `test_search_fleet_overshoot.py` records as not covered."""
    workers, target = 8, 40
    study, _runner = _fleet(tmp_path, workers=workers, n_trials=target,
                            monkeypatch=monkeypatch)

    terminal = {optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED,
                optuna.trial.TrialState.FAIL}
    finished = [t for t in study.trials if t.state in terminal]
    assert len(finished) >= target
    assert len(finished) <= target + workers - 1, (
        f"{len(finished)} finished trials against a target of {target} and a "
        f"bound of {target + workers - 1}")


def test_every_state_the_loop_counts_is_actually_produced(tmp_path,
                                                          monkeypatch):
    """The overshoot bound is only interesting if PRUNED and FAIL really advance
    the counter. If the runner had only ever completed trials this whole file
    would be testing one branch."""
    study, _runner = _fleet(tmp_path, workers=8, n_trials=40,
                            monkeypatch=monkeypatch)

    states = {t.state.name for t in study.trials}
    assert {"COMPLETE", "PRUNED", "FAIL"} <= states, states


def test_an_anchor_is_pulled_by_exactly_one_worker(tmp_path, monkeypatch):
    """Anchors are enqueued by the supervisor before the fleet starts, so from
    then on a worker only ever PULLS. Two workers pulling one anchor would be two
    GPU-trials on one design and one fewer reference for the pruner -- and with
    the reference replicates sharing one params dict, a params-based dedup could
    not detect it."""
    from experimentation.sweep.search.anchors import load_designs

    designs = load_designs(REPO / "configs/search/proxy-256x17-anchors.yaml",
                           minimum=24)
    study, runner = _fleet(tmp_path, workers=8, n_trials=40, designs=designs,
                           monkeypatch=monkeypatch)

    names = [t.user_attrs.get("anchor_name") for t in study.trials]
    pulled = [n for n in names if n]
    assert sorted(pulled) == sorted(d.name for d in designs)
    assert len(pulled) == len(set(pulled)), "an anchor ran twice"
    # And the five replicates all survived, which is the case a params-keyed
    # dedup gets wrong.
    assert sum(1 for n in pulled if n.startswith("reference-")) == 6, (
        "reference-k1, four reference-seed-* and reference-k2")


def test_a_second_supervisor_pass_over_a_live_journal_adds_nothing(tmp_path,
                                                                  monkeypatch):
    """`__main__` calls `enqueue_anchors` in EVERY process, not only the
    supervisor. That is safe because the supervisor enqueues before the fanout,
    so each worker's call finds the names already present -- but "safe by
    ordering" is worth an assertion, since the name check is a read-then-write
    with no lock."""
    from experimentation.sweep.search.anchors import load_designs
    from experimentation.sweep.search.study import create_study, enqueue_anchors

    designs = load_designs(REPO / "configs/search/proxy-256x17-anchors.yaml",
                           minimum=24)
    study, _runner = _fleet(tmp_path, workers=4, n_trials=12, designs=designs,
                            monkeypatch=monkeypatch)
    reopened = create_study(study_name="fleet", study_dir=tmp_path, seed=2026,
                            concurrent_trials=4, sampler="tpe", n_trials=12,
                            prune_after_step=5, logging_steps=1)
    from koopman_lm.config import build_config
    added = enqueue_anchors(reopened, designs, build_config("50m"),
                            _real_space(), base_lr=4e-4)
    assert added == 0


# ============================================================================
# Claim 4c: the remaining startup trials are EXPLORATORY, not TPE-directed, and
# multivariate TPE begins only after the startup phase.
#
# Previous coverage asserted `sampler._n_startup_trials == 64` and
# `sampler._multivariate is True` -- introspection of two private attributes.
# `test_search_fields_are_applied.py` even DOCUMENTS the behaviour in a comment
# ("With no completed trials TPE is inside its startup window and delegates to
# its own seeded random sampler") and uses it as a tool rather than asserting it.
# So if a future optuna redefined `n_startup_trials`, or if `multivariate=True`
# stopped fitting a joint, the suite stayed green.
#
# The behavioural version: below the boundary a seeded TPE must draw exactly what
# a seeded RandomSampler draws; above it, it must stop.
# ============================================================================

def _draw(sampler, count, *, feed=True):
    """`count` successive draws from `sampler`, telling each one back."""
    distributions = {
        "a": optuna.distributions.CategoricalDistribution([1, 2, 3, 4]),
        "b": optuna.distributions.FloatDistribution(0.0, 1.0),
    }
    study = optuna.create_study(sampler=sampler,
                                pruner=optuna.pruners.NopPruner())
    drawn = []
    for index in range(count):
        trial = study.ask(distributions)
        drawn.append((trial.params["a"], round(trial.params["b"], 12)))
        if feed:
            # A real objective with structure, so that IF the sampler were
            # modelling, its draws would move -- a constant objective would let
            # a modelling sampler look random.
            study.tell(trial, float(trial.params["a"]) + trial.params["b"])
        else:
            study.tell(trial, 1.0)
    return drawn


def test_below_the_startup_boundary_tpe_draws_exactly_what_random_draws():
    """The behavioural statement of "the startup phase is exploratory": inside the
    window a seeded TPE delegates to its own seeded RandomSampler, so the two
    sequences are identical. Not "looks random" -- identical."""
    from experimentation.sweep.search.study import make_sampler

    startup = 12
    tpe = _draw(make_sampler(seed=99, sampler="tpe_multivariate",
                             startup_trials=startup), startup)
    random_only = _draw(optuna.samplers.RandomSampler(seed=99), startup)
    assert tpe == random_only


def test_past_the_startup_boundary_tpe_stops_agreeing_with_random():
    """Guards the guard. If TPE agreed with random forever, the test above would
    pass while the study was a random search reporting itself as TPE -- which is
    the failure `StudySpec._validate_sampler` refuses `sampler_startup_trials >=
    n_trials` to prevent."""
    from experimentation.sweep.search.study import make_sampler

    startup, total = 12, 40
    tpe = _draw(make_sampler(seed=99, sampler="tpe_multivariate",
                             startup_trials=startup), total)
    random_only = _draw(optuna.samplers.RandomSampler(seed=99), total)
    assert tpe[:startup] == random_only[:startup]
    assert tpe[startup:] != random_only[startup:], (
        "TPE never left its startup window, so this study is a random search "
        "wearing TPE's name")


def test_the_boundary_is_exactly_where_the_study_declares_it():
    """Off by one here would mean the study's phase plan is wrong by a trial in
    every statement it makes about its own schedule."""
    from experimentation.sweep.search.study import make_sampler

    startup = 12
    tpe = _draw(make_sampler(seed=99, sampler="tpe_multivariate",
                             startup_trials=startup), startup + 1)
    random_only = _draw(optuna.samplers.RandomSampler(seed=99), startup + 1)
    assert tpe[:startup] == random_only[:startup]
    assert tpe[startup] != random_only[startup]


def test_a_pruned_trial_advances_the_samplers_window_observably():
    """The study file claims TPE counts COMPLETE **and PRUNED**, and the previous
    test of it asserted `len(get_trials(states=(COMPLETE, PRUNED))) == 3` -- i.e.
    it re-implemented the rule and checked that optuna recorded three PRUNED
    trials, which is tautological with respect to TPESampler.

    The behavioural version: fill the whole window with PRUNED trials, then draw.
    If PRUNED counted for nothing the sampler would still be inside its window
    and would still agree with random.
    """
    from experimentation.sweep.search.study import make_sampler

    startup = 6
    distributions = {"a": optuna.distributions.FloatDistribution(0.0, 1.0)}

    def stream(sampler, prune_first):
        study = optuna.create_study(sampler=sampler,
                                    pruner=optuna.pruners.NopPruner())
        for _ in range(prune_first):
            trial = study.ask(distributions)
            trial.report(1.0, 10)
            study.tell(trial, state=optuna.trial.TrialState.PRUNED)
        return round(study.ask(distributions).params["a"], 12)

    pruned_window = stream(make_sampler(seed=5, sampler="tpe",
                                        startup_trials=startup), startup)
    empty = stream(make_sampler(seed=5, sampler="tpe",
                                startup_trials=startup), 0)
    assert pruned_window != empty, (
        "the sampler's draw did not change after `startup` PRUNED trials, so "
        "PRUNED did not advance its window -- the study file's Phase 2 "
        "arithmetic is wrong")


# ============================================================================
# Claim 9: the objective is computed from HELD-OUT data, not training loss.
#
# Previous coverage stopped at the launcher: `eval_data_dir` was asserted to
# reach `LocalLauncher`/`SlurmLauncher`. The remaining hops were untested, and
# `training/train.py:468` is `val_dir = args.eval_data_dir or args.data_dir` --
# so ONE dropped argv entry silently reverts the whole study to a training-loss
# ranking, with no error and no visible difference in the output.
# ============================================================================

def test_the_studys_eval_shard_is_not_its_training_shard():
    """The cheapest possible check and nothing did it: if these were equal, the
    entire ranking would be a training-loss ranking no matter how correct every
    hop below is."""
    from experimentation.sweep.search.studyspec import load_study_spec
    from experimentation.sweep.spec import _base_sections

    spec = load_study_spec(STUDY)
    sections = _base_sections(str(REPO / spec.base))
    assert spec.eval_data_dir, (
        "the study declares no eval_data_dir, so train.py's "
        "`args.eval_data_dir or args.data_dir` falls back to the TRAINING shard")
    assert spec.eval_data_dir != sections["data"]["shard_dir"]


def test_the_eval_shard_becomes_an_argv_entry_on_the_training_command():
    """The hop the launcher test could not see. `train_argv` is where
    `eval_data_dir` becomes `--eval_data_dir`, and train.py falls back to the
    training shard when the flag is absent."""
    from experimentation.run.train_argv import build_train_argv

    argv = _argv_for(eval_on_final=True, eval_data_dir="/scratch/held-out")
    assert "--eval_on_final" in argv
    assert "--eval_data_dir" in argv
    assert argv[argv.index("--eval_data_dir") + 1] == "/scratch/held-out"
    assert build_train_argv is not None      # imported for the name in the error


def test_without_an_eval_shard_no_flag_is_emitted_and_that_is_the_fallback():
    """Guards the guard, and documents the hazard: no flag means train.py scores
    on `--data_dir`, i.e. on the training shard. That is a legitimate default for
    a hand-launched run and is exactly what a study must not do."""
    argv = _argv_for(eval_on_final=True, eval_data_dir=None)
    assert "--eval_on_final" in argv
    assert "--eval_data_dir" not in argv


def test_the_study_command_puts_the_held_out_shard_on_its_launcher():
    """End to end from the committed file: the value the study declares is the
    value the launcher will hand to `train_argv`."""
    from experimentation.run.launchers import LocalLauncher
    from experimentation.sweep.search.studyspec import load_study_spec

    spec = load_study_spec(STUDY)
    launcher = LocalLauncher(eval_on_final=True,
                             eval_data_dir=spec.eval_data_dir,
                             logging_steps=spec.logging_steps)
    assert launcher.eval_data_dir == spec.eval_data_dir
    assert launcher.eval_on_final is True


def _argv_for(**scoring):
    """A training argv for the proxy spec, with the given scoring kwargs."""
    from experimentation.run.resolve import resolve_run_spec
    from experimentation.run.train_argv import build_train_argv

    spec = resolve_run_spec(REPO / "configs/runs/proxy-256x17.yaml")
    return build_train_argv(spec, pathlib.Path("/tmp/run-dir"), **scoring)


# ============================================================================
# Claims 1 and 2: all nine axes vary as specified, and every placement resolves
# to distinct indices at every layer count.
#
# Both are STRONGLY covered -- but through a HARDCODED COPY of the study's
# `search_axes` in `test_search_axis_restriction.py` (`_TARGET_AXES`) and
# hardcoded `(2, 3, 4, 6, 8)` / `[3, 7, 11, 15]`. Nothing asserted that copy
# equals the committed file. So ADDING a value -- a fifth rank, a sixth layer
# count -- would slip past both: the dead-axis check would not be re-run at the
# new count, and the anchor extremes would be unchanged.
#
# These two tests read the committed file at runtime, so the guard grows with the
# study instead of with the test.
# ============================================================================

def test_the_hardcoded_target_axes_still_match_the_committed_study():
    """The seam. Every assertion in `test_search_axis_restriction.py`'s target
    section is against `_TARGET_AXES`; this is what ties that to reality."""
    import test_search_axis_restriction as axis_tests

    from experimentation.sweep.search.studyspec import load_study_spec

    spec = load_study_spec(STUDY)
    assert axis_tests._TARGET_FIXED == spec.fixed_params
    assert set(axis_tests._TARGET_AXES) == set(spec.search_axes)
    for axis, declaration in spec.search_axes.items():
        mirrored = axis_tests._TARGET_AXES[axis]
        assert mirrored.get("kind") == declaration.get("kind"), axis
        if declaration["kind"] == "categorical":
            assert [pytest.approx(c) if isinstance(c, float) else c
                    for c in mirrored["choices"]] == declaration["choices"], axis
        else:
            for key in ("low", "high", "log"):
                assert mirrored.get(key) == declaration.get(key), (axis, key)


def test_placement_is_live_at_every_layer_count_the_committed_study_declares():
    """The dead-axis regression guard, driven from the FILE rather than from a
    copy of it. `placement` x `n_ska_layers` must resolve to as many distinct
    index tuples as there are placements, at every declared count.

    On `4m-golden` all four placements collapse onto one pair -- the failure that
    made `4m-adaptive` useless, where `n_ska_layers` had one choice and every
    placement resolved to `(1, 2)`. Adding a sixth layer count to the study file
    now re-runs this check at that count automatically, which is the whole point
    of reading the file.
    """
    from experimentation.run.spec import resolve_model_config
    from experimentation.sweep.search.studyspec import load_study_spec
    from experimentation.sweep.search.geometry import make_layer_indices
    from experimentation.sweep.spec import _base_sections

    spec = load_study_spec(STUDY)
    sections = _base_sections(str(REPO / spec.base))
    model = resolve_model_config(sections["model"])
    counts = spec.search_axes["n_ska_layers"]["choices"]
    placements = spec.search_axes["placement"]["choices"]
    assert len(counts) >= 2 and len(placements) >= 2

    for count in counts:
        resolved = {
            placement: tuple(make_layer_indices(
                model.n_layers, int(count), str(placement),
                list(model.ska_layer_indices)))
            for placement in placements
        }
        assert len(set(resolved.values())) == len(placements), (
            f"at n_ska_layers={count} the {len(placements)} placements resolve "
            f"to only {len(set(resolved.values()))} distinct index tuple(s): "
            f"{resolved}. That is a DEAD AXIS -- the sampler would spend trials "
            f"on a difference that does not exist and the importance table would "
            f"be computed over it.")
        for placement, indices in resolved.items():
            assert len(indices) == int(count), (
                f"{placement} at n_ska_layers={count} resolved to "
                f"{len(indices)} indices; make_layer_indices CLAMPS rather than "
                f"raising, so a count past the usable window becomes a silent "
                f"duplicate of a smaller one")
            assert len(set(indices)) == len(indices)
            assert 0 < min(indices) and max(indices) < model.n_layers


def test_all_nine_declared_axes_move_the_resolved_spec():
    """Claim 1, from the file. `restrict_space` validates the declarations;
    this asserts each one still CHANGES something downstream, which is the
    property a dead axis violates."""
    from experimentation.run.spec import resolve_model_config
    from experimentation.sweep.search.space import (
        base_reference_point, params_to_overrides)
    from experimentation.sweep.search.studyspec import load_study_spec
    from experimentation.sweep.spec import _base_sections

    spec = load_study_spec(STUDY)
    sections = _base_sections(str(REPO / spec.base))
    model = resolve_model_config(sections["model"])
    space = _real_space()
    point = base_reference_point(model, base_lr=sections["optim"]["lr"],
                                base_optim=sections["optim"])
    point = {axis: point.get(axis, _first_value(space[axis]))
             for axis in space}
    baseline = params_to_overrides(point, model, max_steps=spec.max_steps,
                                  backend_policy=spec.backend_policy)

    dead = []
    for axis in sorted(spec.search_axes):
        moved = False
        for value in _candidates(space[axis]):
            if value == point[axis]:
                continue
            other = params_to_overrides({**point, axis: value}, model,
                                        max_steps=spec.max_steps,
                                        backend_policy=spec.backend_policy)
            if other != baseline:
                moved = True
                break
        if not moved:
            dead.append(axis)
    assert dead == [], f"axes that change nothing in the resolved spec: {dead}"
    assert len(spec.search_axes) == 9


def _first_value(declaration):
    if declaration["kind"] == "categorical":
        return declaration["choices"][0]
    return declaration["low"]


def _candidates(declaration):
    if declaration["kind"] == "categorical":
        return list(declaration["choices"])
    return [declaration["low"], declaration["high"]]
