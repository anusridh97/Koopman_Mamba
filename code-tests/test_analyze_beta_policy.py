"""`scripts/analyze_beta_policy.py` on inputs whose answers are known.

This script produces the headline number of the write-gate LM arm -- a bound of
the form "no policy differs from the control by more than X" -- so its
arithmetic needs checking against cases computed by hand rather than only
against a real study, where a wrong denominator looks like a result.

Four properties, in the order that would hurt most to have wrong:

  1. A KNOWN effect at a KNOWN sigma is resolved or not, correctly and on the
     right side of the Bonferroni threshold. This is the whole output.
  2. Duplicate seeds inside a cell are REFUSED. Under `deterministic: true` two
     members at one seed land on the same loss, contributing 0 to the sum of
     squares and 1 to the dof -- driving sigma toward zero and making every
     effect look resolved. Silently pooling them is the most damaging way this
     script could fail, because the output would look excellent.
  3. Trials without a reference group are IGNORED. Target overshoot produces
     sampled trials at the base seed, and including them would mix a different
     configuration into a cell mean.
  4. Non-COMPLETE trials are ignored, so a FAIL does not enter a mean as a
     zero or a partial loss.
"""
import csv
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.correctness

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "analyze_beta_policy.py"

_COLUMNS = ["state", "objective", "attr_reference_group", "param_beta_policy",
            "attr_model_seed", "attr_anchor_name"]


def _csv(tmp_path, rows):
    path = tmp_path / "trials.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def _cell(policy, values, seeds=None, group=None, state="COMPLETE"):
    seeds = seeds or list(range(42, 42 + len(values)))
    return [{"state": state, "objective": f"{v:.10f}",
             "attr_reference_group": group or f"beta_{policy}",
             "param_beta_policy": policy, "attr_model_seed": str(s),
             "attr_anchor_name": f"beta-{policy}-seed-{s}"}
            for v, s in zip(values, seeds)]


def _run(path, *args):
    out = subprocess.run(
        [sys.executable, str(SCRIPT), str(path), *args],
        capture_output=True, text=True, cwd=str(REPO),
        env={"PYTHONPATH": str(REPO), "PATH": "/usr/bin:/bin"})
    return out


def test_a_known_effect_at_a_known_sigma_is_reported_correctly(tmp_path):
    """Constructed so the arithmetic is checkable by hand.

    Every cell has the SAME within-cell spread (+-0.01 around its mean), so the
    pooled sigma is that spread exactly and does not depend on which cell you
    look at. Then the deltas are placed either side of the threshold.

    Within-cell values {m-0.01, m, m, m, m+0.01}: sum of squares = 2e-4 over 4
    dof per cell, 8e-4 over 16 dof pooled -> sigma = sqrt(8e-4/16) = 7.0711e-3.
    Contrast sd at n=5 vs n=5 = sigma*sqrt(2/5) = 4.4721e-3, and the threshold
    for three contrasts is 2.39 sigma_contrast = 1.0688e-2.

    So a delta of 0.05 is resolved (11.2 sigma) and one of 0.002 is not
    (0.45 sigma).
    """
    def spread(mean):
        return [mean - 0.01, mean, mean, mean, mean + 0.01]

    rows = (_cell("learned", spread(4.0))
            + _cell("one", spread(4.002))       # +0.002  -> unresolved
            + _cell("linear", spread(4.05))     # +0.05   -> resolved worse
            + _cell("head_scalar", spread(3.95)))   # -0.05 -> resolved BETTER
    out = _run(_csv(tmp_path, rows))
    assert out.returncode == 0, out.stderr
    text = out.stdout

    assert "pooled sigma: 7.0711e-03" in text, text
    assert "within-cell dof: 16" in text, text
    # Bonferroni applied and announced.
    assert "2.39" in text, text
    assert "Bonferroni-corrected for 3 contrasts" in text, text

    lines = {line.split()[0]: line for line in text.splitlines()
             if line.strip() and line.split()[0] in
             {"learned", "one", "linear", "head_scalar"}}
    assert "CONTROL" in lines["learned"]
    assert "unresolved" in lines["one"], lines["one"]
    assert "RESOLVED worse" in lines["linear"], lines["linear"]
    assert "RESOLVED BETTER" in lines["head_scalar"], lines["head_scalar"]
    # ... and the sigma counts, to two decimals.
    assert "0.45" in lines["one"], lines["one"]
    assert "11.18" in lines["linear"], lines["linear"]


def test_all_unresolved_prints_the_prefer_the_simpler_model_reading(tmp_path):
    """The null is the expected outcome, so the script has to say what it means
    rather than printing three dashes."""
    def spread(mean):
        return [mean - 0.01, mean, mean, mean, mean + 0.01]

    rows = (_cell("learned", spread(4.0)) + _cell("one", spread(4.001))
            + _cell("linear", spread(3.999)) + _cell("head_scalar", spread(4.002)))
    out = _run(_csv(tmp_path, rows))
    assert out.returncode == 0, out.stderr
    assert "no policy is distinguishable from the control" in out.stdout
    assert "Prefer the SIMPLEST policy" in out.stdout


def test_a_duplicate_seed_in_a_cell_is_refused_not_pooled(tmp_path):
    """The failure that would look like an excellent result.

    Two members at one seed land on the same loss under `deterministic: true`,
    so they add 0 to the sum of squares and 1 to the dof -- sigma shrinks and
    every effect becomes 'resolved'. Refused with exit 2.
    """
    rows = (_cell("learned", [4.0, 4.0, 4.01, 4.02, 4.03],
                  seeds=[42, 42, 43, 44, 45])
            + _cell("one", [4.0, 4.01, 4.02, 4.03, 4.04]))
    out = _run(_csv(tmp_path, rows))
    assert out.returncode == 2, out.stdout
    # `['42']`, not `[42]`: seeds arrive from csv.DictReader as strings and the
    # script keeps them that way, which is correct -- comparing them as strings
    # cannot silently equate 42 with 42.0.
    assert "repeated seed(s) ['42']" in out.stderr, out.stderr
    assert "one datapoint counted twice" in out.stderr


def test_trials_without_a_reference_group_are_ignored(tmp_path):
    """Target overshoot produces sampled trials at the base seed. They carry no
    group, and mixing them into a cell mean would average a different
    configuration in."""
    rows = _cell("learned", [4.0, 4.0, 4.0, 4.0, 4.0])
    rows += [{"state": "COMPLETE", "objective": "9.99",
              "attr_reference_group": "", "param_beta_policy": "learned",
              "attr_model_seed": "42", "attr_anchor_name": ""}]
    out = _run(_csv(tmp_path, rows))
    assert out.returncode == 0, out.stderr
    assert "completed members: 5" in out.stdout, out.stdout
    assert "9.99" not in out.stdout


def test_non_complete_trials_are_ignored(tmp_path):
    rows = _cell("learned", [4.0, 4.0, 4.0, 4.0, 4.0])
    rows += _cell("learned", [0.0], seeds=[99], state="FAIL")
    out = _run(_csv(tmp_path, rows))
    assert out.returncode == 0, out.stderr
    assert "completed members: 5" in out.stdout, out.stdout


def test_an_empty_or_groupless_csv_refuses_rather_than_reporting_nothing(tmp_path):
    rows = [{"state": "COMPLETE", "objective": "4.0",
             "attr_reference_group": "", "param_beta_policy": "learned",
             "attr_model_seed": "42", "attr_anchor_name": ""}]
    out = _run(_csv(tmp_path, rows))
    assert out.returncode == 2
    assert "no COMPLETE trials with a reference_group" in out.stderr
