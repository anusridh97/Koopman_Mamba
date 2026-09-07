"""The health checks must FAIL on the situations that actually cost GPU-hours.

A monitoring script that cannot be shown to fire is not monitoring. Each test
here reconstructs one real incident in miniature and asserts the check catches
it, and one asserts a clean study stays quiet so the thing is not noise.
"""
import json

import pytest

from scripts.study_health import (
    FAIL, PASS, WARN, Report, check_duplicate_configs, check_failure_rate,
    check_nonfinite_loss, check_sampler_seeds, check_shared_run_dir,
    check_stalled)


class _T:
    """Minimal stand-in for an optuna FrozenTrial."""

    def __init__(self, number, state="RUNNING", params=None, attrs=None,
                 start=None, inter=None, value=None):
        self.number = number
        self.state = type("S", (), {"name": state})
        self.params = params or {}
        self.user_attrs = attrs or {}
        self.datetime_start = start
        self.intermediate_values = inter or {}
        self.value = value


def _status(rows, name):
    return next(s for n, s, _ in rows if n == name)


def test_duplicate_configs_fires_on_identical_params_and_seed():
    """The 180M incident: 16 single-worker jobs, one sampler stream, 24 copies."""
    dup = {"lr": 0.0033, "ska_rank": 32}
    trials = [_T(i, "RUNNING", dict(dup), {"model_seed": 42}) for i in range(5)]
    rep = Report()
    check_duplicate_configs(trials, rep)
    assert _status(rep.rows, "DUPLICATE_CONFIGS") == FAIL


def test_reference_replicates_are_not_flagged_as_duplicates():
    """Anchors legitimately share params across the reference group.

    Only the SEED differs, and that is the whole point of a noise floor -- so a
    check that flagged it would fire on every correctly-built study and get
    switched off.
    """
    same = {"lr": 0.0046}
    trials = [_T(i, "COMPLETE", dict(same),
                 {"model_seed": s, "anchor_name": f"reference-k{i}",
                  "reference_group": "reference"})
              for i, s in enumerate([42, 43, 44, 45, 46])]
    rep = Report()
    check_duplicate_configs(trials, rep)
    assert _status(rep.rows, "DUPLICATE_CONFIGS") == PASS


def test_two_identical_trials_warn_rather_than_fail():
    dup = {"lr": 0.0033}
    trials = [_T(i, "RUNNING", dict(dup), {"model_seed": 42}) for i in range(2)]
    trials.append(_T(9, "RUNNING", {"lr": 0.009}, {"model_seed": 42}))
    rep = Report()
    check_duplicate_configs(trials, rep)
    assert _status(rep.rows, "DUPLICATE_CONFIGS") == WARN


def test_sampler_seeds_fires_when_the_whole_fleet_shares_one_seed():
    trials = [_T(i, "RUNNING", {"lr": 0.003}, {"sampler_seed": 2026})
              for i in range(4)]
    rep = Report()
    check_sampler_seeds(trials, rep)
    assert _status(rep.rows, "SAMPLER_SEEDS") == FAIL


def test_sampler_seeds_quiet_when_streams_differ():
    trials = [_T(i, "RUNNING", {"lr": 0.003}, {"sampler_seed": 2026 + i})
              for i in range(4)]
    rep = Report()
    check_sampler_seeds(trials, rep)
    assert _status(rep.rows, "SAMPLER_SEEDS") == PASS


def test_shared_run_dir_fires_when_two_jobs_write_one_directory(tmp_path):
    """13 job_ids were writing one attempts.jsonl on the 180M study."""
    d = tmp_path / "_studies" / "s.aaaa"
    d.mkdir(parents=True)
    run = tmp_path / "grp.1234" / "seed42.abcd"
    run.mkdir(parents=True)
    run.joinpath("attempts.jsonl").write_text(
        json.dumps({"job_id": "1"}) + "\n" + json.dumps({"job_id": "2"}) + "\n")
    rep = Report()
    check_shared_run_dir(d, rep)
    assert _status(rep.rows, "SHARED_RUN_DIR") == FAIL


def test_shared_run_dir_quiet_on_a_requeued_single_job(tmp_path):
    """A requeue writes several lines with the SAME job_id; that is not a clash."""
    d = tmp_path / "_studies" / "s.aaaa"
    d.mkdir(parents=True)
    run = tmp_path / "grp.1234" / "seed42.abcd"
    run.mkdir(parents=True)
    run.joinpath("attempts.jsonl").write_text(
        "\n".join(json.dumps({"job_id": "7"}) for _ in range(3)))
    rep = Report()
    check_shared_run_dir(d, rep)
    assert _status(rep.rows, "SHARED_RUN_DIR") == PASS


def test_nonfinite_loss_fires_on_a_diverged_run(tmp_path):
    """`lr-3.85x` trained 33,000 further steps on NaN before anyone looked."""
    d = tmp_path / "_studies" / "s.aaaa"
    d.mkdir(parents=True)
    run = tmp_path / "grp.1234" / "seed42.abcd"
    run.mkdir(parents=True)
    run.joinpath("train.log").write_text(
        "step 17925/50862 | loss 2.9547\nstep 17950/50862 | loss nan\n")
    rep = Report()
    check_nonfinite_loss(d, rep)
    assert _status(rep.rows, "NONFINITE_LOSS") == FAIL


def test_failure_rate_fires_on_the_torchrun_wipeout(tmp_path):
    """25 of 25 trials died on a PATH lookup while Slurm reported COMPLETED."""
    trials = [_T(i, "FAIL", {}, {"failure": "FileNotFoundError: 'torchrun'"})
              for i in range(25)]
    rep = Report()
    check_failure_rate(trials, rep)
    assert _status(rep.rows, "FAILURE_RATE") == FAIL


def test_stalled_fires_on_a_trial_reporting_nothing():
    import datetime
    old = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None) \
        - datetime.timedelta(hours=3)
    rep = Report()
    check_stalled([_T(0, "RUNNING", {}, {}, start=old, inter={})], rep)
    assert _status(rep.rows, "STALLED") == FAIL


def test_report_verdict_is_the_worst_status():
    rep = Report()
    rep.add("a", PASS, "")
    rep.add("b", WARN, "")
    assert rep.worst() == WARN
    rep.add("c", FAIL, "")
    assert rep.worst() == FAIL
