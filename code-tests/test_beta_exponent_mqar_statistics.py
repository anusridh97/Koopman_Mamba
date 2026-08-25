"""The predeclared statistics of the exponent arm, and what they refuse to do.

`docs/beta-exponent-preregistration.md` fixes six per-run quantities before any
result exists. This file pins the two that have real arithmetic in them
(`lc_area`, and the grok/censoring triple) plus the refusals that keep the
report honest.

## The specific mistakes being prevented

MQAR is learned through a phase transition. That makes three plausible-looking
summaries actively misleading, and each has a test here:

  * **Averaging accuracy across the transition.** The earlier arm's step-1333 row
    read `learned 0.2969 / one 0.2773 / head_scalar 0.3125 / linear 0.9922`, and
    by step 2666 `one` was at 1.0000. A mean over evals would have reported a
    `linear` blowout that was really a difference in grokking TIME. `lc_area` is
    the sample-efficiency statistic precisely because it is honest about being
    one -- it is not offered as an accuracy.
  * **Scoring a censored run as a failure.** A run that has not grokked by the
    budget is not a run that will not grok. `grokked=False` with
    `censored=True` is a distinct state from `grokked=False, censored=False`,
    which cannot occur -- and the code must make that impossible rather than
    merely unlikely.
  * **Reading a policy ordering off means when the within-policy spread across
    seeds is larger.** `summarise_policy` reports the observed range and the
    grok count, and deliberately does NOT return a single scalar to rank on.

`lc_area` is normalised by the step span so it is bounded in [0, 1] and
comparable across runs whose eval schedules differ -- which they will, once
censored runs are extended.
"""
import json
import math

import pytest

from experimentation.evaluation.mqar_curves import (
    lc_area, run_statistics, summarise_policy, parse_curve)

pytestmark = pytest.mark.correctness


# ---------------------------------------------------------------------------
# lc_area: the sample-efficiency statistic.
# ---------------------------------------------------------------------------

def test_lc_area_of_a_run_that_is_perfect_from_the_first_eval_is_one():
    assert lc_area([(100, 1.0), (200, 1.0), (300, 1.0)]) == pytest.approx(1.0)


def test_lc_area_of_a_run_that_never_learns_is_zero():
    assert lc_area([(100, 0.0), (200, 0.0), (300, 0.0)]) == pytest.approx(0.0)


def test_lc_area_is_the_trapezoidal_mean_not_the_arithmetic_mean():
    """Evals are not always equally spaced -- extending a censored run appends
    coarser points -- so the statistic has to weight by step interval. A plain
    mean would let the density of the eval schedule change the number.
    """
    curve = [(0, 0.0), (100, 0.0), (1000, 1.0)]
    # trapezoid: 0 over [0,100], then mean 0.5 over [100,1000] -> 450/1000
    assert lc_area(curve) == pytest.approx(0.45)
    arithmetic = sum(a for _, a in curve) / len(curve)
    assert not math.isclose(lc_area(curve), arithmetic)


def test_lc_area_rewards_grokking_earlier():
    """The property that makes it a sample-efficiency statistic at all."""
    early = [(0, 0.0), (1000, 1.0), (2000, 1.0)]
    late = [(0, 0.0), (1000, 0.0), (2000, 1.0)]
    assert lc_area(early) > lc_area(late)


def test_lc_area_is_bounded_in_the_unit_interval():
    for curve in ([(0, 0.0), (1000, 1.0)], [(0, 1.0), (500, 0.0), (1000, 1.0)],
                  [(0, 0.3), (7, 0.9)]):
        assert 0.0 <= lc_area(curve) <= 1.0


def test_lc_area_of_a_single_eval_is_that_accuracy():
    """A run killed after one eval has zero step span. Returning the accuracy is
    the only answer that does not divide by zero or silently report 0.0 for a
    run that was at 1.0."""
    assert lc_area([(500, 0.75)]) == pytest.approx(0.75)


def test_lc_area_of_an_empty_curve_is_none_rather_than_zero():
    """A run that produced no eval at all is missing data. Reporting 0.0 would
    make an infrastructure failure look like a scientific one -- the exact
    distinction the job discipline requires be kept."""
    assert lc_area([]) is None


# ---------------------------------------------------------------------------
# grok / censoring: the three-state outcome.
# ---------------------------------------------------------------------------

def test_a_run_crossing_the_threshold_is_grokked_at_the_first_crossing():
    """`grok_step` is the FIRST crossing, not the last and not the best."""
    st = run_statistics([(500, 0.4), (1000, 0.91), (1500, 0.88), (2000, 0.99)],
                        threshold=0.9)
    assert st["grokked"] is True
    assert st["grok_step"] == 1000
    assert st["censored"] is False


def test_a_run_that_never_crosses_is_censored_and_not_scored_as_zero():
    st = run_statistics([(500, 0.3), (1000, 0.5), (1500, 0.62)], threshold=0.9)
    assert st["grokked"] is False
    assert st["grok_step"] is None
    assert st["censored"] is True, (
        "a run that has not grokked by the budget is censored, not a failure")
    # ... and its learning is still recorded, so extending it has a baseline.
    assert st["best_acc"] == pytest.approx(0.62)
    assert st["lc_area"] > 0


def test_censoring_and_grokking_are_never_both_true():
    """The state space is exactly three: grokked; censored; no data. A run that
    is both would be counted twice in the grok-rate table."""
    for curve in ([(1, 0.95)], [(1, 0.1)], [(1, 0.9)], [(1, 0.899)]):
        st = run_statistics(curve, threshold=0.9)
        assert not (st["grokked"] and st["censored"])


def test_the_threshold_is_inclusive_at_exactly_the_declared_value():
    """0.90 is the predeclared threshold. Whether 0.90 itself counts must be
    fixed in code, or two analyses of the same run can disagree."""
    assert run_statistics([(1, 0.9)], threshold=0.9)["grokked"] is True
    assert run_statistics([(1, 0.8999)], threshold=0.9)["grokked"] is False


def test_best_and_final_are_reported_separately():
    """A run that solved the task and then degraded must be visible as such
    rather than averaged away."""
    st = run_statistics([(500, 0.2), (1000, 1.0), (1500, 0.4)], threshold=0.9)
    assert st["best_acc"] == pytest.approx(1.0)
    assert st["final_acc"] == pytest.approx(0.4)
    assert st["grokked"] is True and st["grok_step"] == 1000


def test_a_run_with_no_evals_reports_missing_rather_than_failure():
    st = run_statistics([], threshold=0.9)
    assert st["grokked"] is None, "no data is not 'did not grok'"
    assert st["censored"] is None
    assert st["lc_area"] is None
    assert st["n_evals"] == 0


# ---------------------------------------------------------------------------
# The refusals.
# ---------------------------------------------------------------------------

def test_summarise_policy_reports_the_range_and_refuses_a_single_ranking_scalar():
    """The earlier arm's analysis refused to collapse across grokking and was
    right. A policy summary that returned one number would be ranked on, and the
    within-policy seed spread here is larger than most between-policy gaps.
    """
    runs = [
        run_statistics([(0, 0.0), (1000, 1.0), (2000, 1.0)], threshold=0.9),
        run_statistics([(0, 0.0), (1000, 0.1), (2000, 0.2)], threshold=0.9),
        run_statistics([(0, 0.0), (1000, 0.1), (2000, 0.95)], threshold=0.9),
    ]
    s = summarise_policy(runs)
    assert s["n"] == 3
    assert s["n_grokked"] == 2
    assert s["n_censored"] == 1
    # The range is reported, and it is what makes the summary readable as
    # "2 of 3, and the spread is enormous" rather than as a mean.
    assert s["final_acc_range"] == (pytest.approx(0.2), pytest.approx(1.0))
    assert s["lc_area_range"][0] < s["lc_area_range"][1]
    # No mean accuracy is offered. Averaging pre-grok and post-grok accuracies
    # is the thing the pre-registration explicitly refuses.
    assert "mean_acc" not in s
    assert "final_acc_mean" not in s


def test_summarise_policy_reports_grok_steps_only_for_runs_that_grokked():
    """Imputing a grok step for a censored run -- as the budget, or as
    infinity -- would make the mean grok time a function of the budget."""
    runs = [
        run_statistics([(0, 0.0), (500, 1.0)], threshold=0.9),
        run_statistics([(0, 0.0), (500, 0.1)], threshold=0.9),
    ]
    s = summarise_policy(runs)
    assert s["grok_steps"] == [500]
    assert s["n"] == 2


def test_summarise_policy_of_an_empty_cell_does_not_pretend_to_have_measured():
    s = summarise_policy([])
    assert s["n"] == 0
    assert s["n_grokked"] == 0
    assert s["final_acc_range"] is None


# ---------------------------------------------------------------------------
# Parsing the real log format.
# ---------------------------------------------------------------------------

def test_parse_curve_reads_the_format_mqar_finetune_actually_prints():
    """The exact line `mqar_finetune.py:199` emits. A parser tested against a
    hand-written approximation is a parser that silently returns [] on a real
    log, which reports as "no data" and looks like an infrastructure failure.
    """
    text = (
        "some preamble\n"
        "  [step 250] in-task accuracy (seq=160, kv=8): 0.2969\n"
        "  [step 500] in-task accuracy (seq=160, kv=8): 0.9922\n"
        "\nDone. Final in-task accuracy: 0.9922\n"
    )
    assert parse_curve(text) == [(250, 0.2969), (500, 0.9922)]


def test_parse_curve_returns_empty_for_a_log_with_no_evals():
    assert parse_curve("Traceback (most recent call last):\nRuntimeError: OOM\n") == []


def test_parse_curve_keeps_evals_in_step_order_even_if_the_log_does_not():
    """Two eval sequence lengths interleave in `--eval_seq_lens` mode, and a
    resumed run appends steps that restart. Order is by step, deduplicated to
    the LAST value seen for a step, so a resume overrides a preempted partial.
    """
    text = ("  [step 500] in-task accuracy (seq=160, kv=8): 0.5000\n"
            "  [step 250] in-task accuracy (seq=160, kv=8): 0.2000\n"
            "  [step 500] in-task accuracy (seq=160, kv=8): 0.7000\n")
    assert parse_curve(text) == [(250, 0.2), (500, 0.7)]
