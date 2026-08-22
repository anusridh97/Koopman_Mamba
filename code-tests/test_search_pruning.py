"""Pruning a queued run: report progress from its log, kill it if hopeless.

The original harness pruned by holding a `subprocess.PIPE` and regex-matching the
child's stdout as it arrived. That works, and it is also the single thing that
forced the whole design into one long-lived process and off Slurm.

The observation that unlocks the async version: `run/launchers.py`'s sbatch
template already routes training stdout to `run_dir/slurm-%j.out`. So the *same*
regex works against a durable file, readable from anywhere, at any time, by a
process that has never met the training job. No pipe, no parent-child
relationship, no long-lived owner.

`wait_for_objective` is therefore shaped to be the `read_objective` the driver
already accepts -- the injected seam from the driver commit pays for itself here,
because pruning needed no driver change at all. It polls the log, reports each
parsed step to the trial, asks the pruner, cancels and raises `TrialPruned` when
told to, and otherwise returns the objective once the eval result lands.

`sleep` and `cancel` are injected, so the whole loop is exercised without waiting
and without a scheduler.
"""
import json

import pytest

optuna = pytest.importorskip("optuna", reason="optuna is an optional dependency")

pytestmark = pytest.mark.correctness


@pytest.fixture(autouse=True)
def _quiet_optuna():
    previous = optuna.logging.get_verbosity()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    yield
    optuna.logging.set_verbosity(previous)


# The exact shape train.py:415-416 prints.
def _log_line(step, loss, total=600, ppl=1234.5, lr=4.0e-4, ktps=12.3):
    return (f"step {step:>6d}/{total} | loss {loss:.4f} | ppl {ppl:.1f} | "
            f"lr {lr:.2e} | {ktps:.1f}K tok/s")


def _space():
    from koopman_lm.config import build_config
    from experimentation.sweep.search.space import search_space
    return search_space(build_config("50m"), base_name="50m")


def _write_quick_eval(run_dir, loss):
    path = run_dir / "eval" / "final" / "quick_eval.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"metrics": {"full": {"loss": loss},
                                            "ska_ablation": {"supported": False}}}))


# ------------------------------------------------------------ parsing ----

def test_the_regex_matches_the_line_train_py_actually_prints():
    """Coupled to a print statement, so pinned against its exact format --
    including the right-aligned step width, which a naive pattern misses."""
    from experimentation.sweep.search.metrics import parse_progress

    text = "\n".join([_log_line(10, 7.1234), _log_line(20, 6.5000)])
    progress = parse_progress(text)
    assert [p.step for p in progress] == [10, 20]
    assert progress[0].loss == pytest.approx(7.1234)
    assert progress[1].loss == pytest.approx(6.5)


def test_throughput_is_converted_out_of_the_printed_kilo_units():
    from experimentation.sweep.search.metrics import parse_progress

    progress = parse_progress(_log_line(10, 7.0, ktps=12.5))
    assert progress[0].tokens_per_sec == pytest.approx(12_500.0)


def test_non_matching_lines_are_ignored():
    from experimentation.sweep.search.metrics import parse_progress

    text = "\n".join(["Training: 600 steps, eff_batch=64", _log_line(10, 7.0),
                      "  Saved checkpoint to runs/x/final  (cfg_hash=abcd1234)"])
    assert len(parse_progress(text)) == 1


def test_a_diverged_run_printing_nan_is_skipped_rather_than_crashing():
    """math.exp(min(avg, 20)) still prints, but a nan loss renders as 'nan',
    which is not a number the regex should pretend to parse."""
    from experimentation.sweep.search.metrics import parse_progress

    text = "step     10/600 | loss nan | ppl nan | lr 4.00e-04 | 12.3K tok/s"
    assert parse_progress(text) == []


def test_progress_is_read_from_the_slurm_log_in_the_run_directory(tmp_path):
    from experimentation.sweep.search.metrics import read_progress

    (tmp_path / "slurm-99887.out").write_text(_log_line(30, 5.0) + "\n")
    progress = read_progress(tmp_path)
    assert [p.step for p in progress] == [30]


def test_progress_from_several_logs_is_ordered_by_step(tmp_path):
    """A requeued job writes a second slurm-<jobid>.out; steps must not
    interleave out of order or the pruner sees a jagged curve."""
    from experimentation.sweep.search.metrics import read_progress

    (tmp_path / "slurm-2.out").write_text(_log_line(40, 4.0) + "\n")
    (tmp_path / "slurm-1.out").write_text(_log_line(20, 5.0) + "\n")
    assert [p.step for p in read_progress(tmp_path)] == [20, 40]


# --------------------------------------------------------------- ema ----

def test_the_ema_smooths_a_noisy_curve():
    """Raw step loss is noisy enough that a single bad step could prune a good
    config; the original harness smoothed at 0.45/0.55 and this matches it."""
    from experimentation.sweep.search.metrics import ema_losses

    smoothed = ema_losses([10.0, 0.0, 10.0, 0.0], alpha=0.45)
    assert smoothed[0] == pytest.approx(10.0)
    assert all(0.0 < value < 10.0 for value in smoothed[1:])
    assert max(smoothed[1:]) < 10.0


def test_the_ema_of_a_flat_curve_is_that_value():
    from experimentation.sweep.search.metrics import ema_losses

    assert ema_losses([3.0, 3.0, 3.0], alpha=0.45) == pytest.approx([3.0, 3.0, 3.0])


# ------------------------------------------------------ report + prune ----

def _study(tmp_path, **kwargs):
    from experimentation.sweep.search.study import create_study
    return create_study(study_name="prune", study_dir=tmp_path, seed=1, **kwargs)


def test_waiting_returns_the_objective_once_the_eval_lands(tmp_path):
    from experimentation.sweep.search.metrics import wait_for_objective
    from experimentation.sweep.search.study import to_distributions

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    study = _study(tmp_path)
    trial = study.ask(to_distributions(_space()))
    _write_quick_eval(run_dir, 2.5)

    value = wait_for_objective(study, trial, run_dir, sleep=lambda s: None)
    assert value == pytest.approx(2.5)


def test_waiting_reports_every_parsed_step_to_the_trial(tmp_path):
    """The reports are what a median pruner compares across trials; without them
    should_prune can never fire."""
    from experimentation.sweep.search.metrics import wait_for_objective
    from experimentation.sweep.search.study import to_distributions

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "slurm-1.out").write_text(
        "\n".join(_log_line(s, 5.0) for s in (10, 20, 30)) + "\n")
    _write_quick_eval(run_dir, 2.0)

    study = _study(tmp_path)
    trial = study.ask(to_distributions(_space()))
    wait_for_objective(study, trial, run_dir, sleep=lambda s: None)
    assert set(trial.storage.get_trial(trial._trial_id).intermediate_values) == {10, 20, 30}


def test_a_hopeless_trial_is_cancelled_and_pruned(tmp_path):
    from experimentation.sweep.search.metrics import wait_for_objective
    from experimentation.sweep.search.study import to_distributions

    distributions = to_distributions(_space())
    # THREE completed trials, and logging_steps=1: the pruner carries
    # n_min_trials=3 and an interval matched to the trainer's logging cadence, so
    # a median over two reports is deliberately not actionable.
    study = _study(tmp_path, prune_after_step=1, prune_startup_trials=3,
                   logging_steps=1)

    for good in (1.0, 1.1, 1.05):
        reference = study.ask(distributions)
        for step in (10, 20, 30):
            reference.report(good, step)
        study.tell(reference, good)

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "slurm-4242.out").write_text(
        "\n".join(_log_line(s, 99.0) for s in (10, 20, 30)) + "\n")

    cancelled = []
    trial = study.ask(distributions)
    with pytest.raises(optuna.TrialPruned):
        # timeout_seconds is a safety net, not part of the assertion: without it a
        # pruner that declines to fire turns this into an infinite poll loop that
        # hangs the suite instead of failing it. That is exactly what happened
        # when n_min_trials arrived.
        wait_for_objective(study, trial, run_dir, sleep=lambda s: None,
                           timeout_seconds=0.0, clock=lambda: 1.0,
                           cancel=cancelled.append)
    assert cancelled == [run_dir], "a pruned job must actually be stopped"


def test_an_anchor_is_not_pruned_by_default(tmp_path):
    """Anchors ARE the reference set a median pruner needs. Pruning them removes
    the thing later trials are judged against."""
    from experimentation.sweep.search.metrics import wait_for_objective
    from experimentation.sweep.search.study import to_distributions

    distributions = to_distributions(_space())
    study = _study(tmp_path, prune_after_step=1, prune_startup_trials=3,
                   logging_steps=1)
    for good in (1.0, 1.1, 1.05):
        reference = study.ask(distributions)
        for step in (10, 20):
            reference.report(good, step)
        study.tell(reference, good)

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "slurm-1.out").write_text(
        "\n".join(_log_line(s, 99.0) for s in (10, 20)) + "\n")
    _write_quick_eval(run_dir, 99.0)

    trial = study.ask(distributions)
    trial.set_user_attr("anchor_name", "baseline")
    value = wait_for_objective(study, trial, run_dir, sleep=lambda s: None)
    assert value == pytest.approx(99.0), "the anchor ran to completion"


def test_an_anchor_can_be_pruned_when_explicitly_allowed(tmp_path):
    from experimentation.sweep.search.metrics import wait_for_objective
    from experimentation.sweep.search.study import to_distributions

    distributions = to_distributions(_space())
    study = _study(tmp_path, prune_after_step=1, prune_startup_trials=3,
                   logging_steps=1)
    for good in (1.0, 1.1, 1.05):
        reference = study.ask(distributions)
        for step in (10, 20):
            reference.report(good, step)
        study.tell(reference, good)

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "slurm-1.out").write_text(
        "\n".join(_log_line(s, 99.0) for s in (10, 20)) + "\n")

    trial = study.ask(distributions)
    trial.set_user_attr("anchor_name", "baseline")
    with pytest.raises(optuna.TrialPruned):
        wait_for_objective(study, trial, run_dir, sleep=lambda s: None,
                           prune_anchors=True,
                           timeout_seconds=0.0, clock=lambda: 1.0)


def test_waiting_gives_up_after_the_timeout(tmp_path):
    """A job that never produces a result must not hang the study forever."""
    from experimentation.sweep.search.metrics import wait_for_objective
    from experimentation.sweep.search.study import to_distributions

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    study = _study(tmp_path)
    trial = study.ask(to_distributions(_space()))

    elapsed = {"t": 0.0}

    def _sleep(seconds):
        elapsed["t"] += seconds

    value = wait_for_objective(study, trial, run_dir, sleep=_sleep,
                               poll_seconds=30, timeout_seconds=120,
                               clock=lambda: elapsed["t"])
    assert value is None, "None becomes a FAIL in the driver, not a fake number"
    assert elapsed["t"] >= 120


def test_the_same_step_is_never_reported_twice(tmp_path):
    """optuna raises on a duplicate report step, and a poll loop re-reads the
    whole log every pass."""
    from experimentation.sweep.search.metrics import wait_for_objective
    from experimentation.sweep.search.study import to_distributions

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    log = run_dir / "slurm-1.out"
    log.write_text(_log_line(10, 5.0) + "\n")

    study = _study(tmp_path)
    trial = study.ask(to_distributions(_space()))
    polls = {"n": 0}

    def _sleep(seconds):
        polls["n"] += 1
        # The log grows between polls, and the earlier line is still in it.
        log.write_text("\n".join([_log_line(10, 5.0), _log_line(20, 4.0)]) + "\n")
        if polls["n"] >= 2:
            _write_quick_eval(run_dir, 3.0)

    value = wait_for_objective(study, trial, run_dir, sleep=_sleep)
    assert value == pytest.approx(3.0)
    reported = trial.storage.get_trial(trial._trial_id).intermediate_values
    assert set(reported) == {10, 20}


# ---------------------------------------------------- driver integration ----

def test_the_driver_records_a_pruned_trial_as_pruned(tmp_path):
    """No driver change was needed for any of this -- pruning arrives through the
    read_objective seam. This pins that the outcome is reported honestly."""
    import dataclasses

    from koopman_lm.config import build_config
    from experimentation.sweep.search.driver import run_trial
    from experimentation.sweep.search.study import create_study, to_distributions

    shard = tmp_path / "shard"
    shard.mkdir()
    (shard / "meta.json").write_text(json.dumps({
        "n_tokens": 1000, "tokenizer": "NousResearch/Llama-2-7b-hf",
        "mix": {"fineweb": 1.0}}))
    cfg = build_config("50m")
    context = {
        "base_sections": {
            "model": dataclasses.asdict(cfg),
            "data": {"kind": "shard", "shard_dir": str(shard),
                     "tokenizer": "NousResearch/Llama-2-7b-hf",
                     "mix": {"fineweb": 1.0}, "n_tokens": 1000},
            "optim": {"lr": 4.0e-4, "warmup_steps": 10, "max_steps": 100,
                      "effective_batch": 64},
            "runtime": {"per_device_batch_size": 16, "seed": 42},
        },
        "base_model": cfg,
        "space": _space(),
        "max_steps": 100,
        "run_root": tmp_path / "runs",
        "study_name": "prune",
    }

    class _Launcher:
        def submit(self, spec, run_dir, dry_run=False, resume=False, wait=False):
            return ["ok"]

    def _prune(run_dir):
        raise optuna.TrialPruned("hopeless")

    study = create_study(study_name="prune", study_dir=tmp_path, seed=1)
    trial = study.ask(to_distributions(context["space"]))
    outcome = run_trial(study, trial, launcher=_Launcher(),
                        read_objective=_prune, **context)

    assert outcome.state == "pruned"
    assert outcome.objective is None
    assert study.trials[0].state.name == "PRUNED"
