"""The trace artifact and the viewer that reads it.

Split deliberately, and the split is what makes this file possible. An earlier
version computed log probs AND emitted HTML in one script, so the renderer could
only be tested against a stubbed model. As a pure function of a JSON file it is
testable from a literal fixture -- no torch, no GPU, no model.

The artifact goes through the SAME envelope quick_eval uses
(`evaluation/result.py::write_result`), so a trace lands beside the score it
explains and one filesystem walk finds both.

What is worth pinning here is not "it renders". It is:

  * the viewer imports no torch, because that is the property that lets it run
    on a laptop against a trace produced on a GPU node;
  * ink is chosen per cell so every colour in both ramps carries readable text --
    the ramp step that failed this was removed, and a regression would put it
    back silently;
  * the diverging scale is symmetric about zero, or colour misreports the SIGN of
    whether SKA helped;
  * the SKA delta's sign convention (positive = SKA helped), which is one
    subtraction away from being inverted and would read as confidently wrong.
"""

import json
import pathlib
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from experimentation.evaluation.inspect_html import (  # noqa: E402
    DIVERGING, SEQUENTIAL, _bucket, ink_for, render_html)

CPU_PY = "/users/jkli/.venvs/koopman-cpu/bin/python"


def _fixture(with_ablation=True):
    """A minimal token_trace envelope, written by hand rather than computed."""
    tokens = [
        {"pos": 0, "token": "the", "logprob": -0.12, "rank": 1,
         "top": [["the", 0.887], ["a", 0.041]]},
        {"pos": 1, "token": " quick", "logprob": -6.40, "rank": 91,
         "top": [["most", 0.201], ["first", 0.104]]},
        {"pos": 2, "token": " fox", "logprob": -2.75, "rank": 4,
         "top": [["dog", 0.310], ["fox", 0.064]]},
    ]
    if with_ablation:
        for tok, d in zip(tokens, (0.0, 0.51, -0.22)):
            tok["ska_delta"] = d
    lps = sorted(t["logprob"] for t in tokens)
    deltas = [t["ska_delta"] for t in tokens] if with_ablation else []
    return {
        "run_id": "2e63f16e", "task": "token_trace", "checkpoint": "final",
        "git_commit": "abc1234", "created_at": "2026-08-21T07:00:00Z",
        "metrics": {
            "sequences": [{"tokens": tokens}],
            "summary": {
                "n_sequences": 1, "n_tokens": len(tokens), "top_k": 2,
                "source": "/scratch/shard", "mean_logprob": -3.09,
                "logprob_p05": lps[0], "logprob_p95": lps[-1],
                "has_ablation": with_ablation,
                "ska_delta_absmax": max((abs(d) for d in deltas), default=0.0),
                "ska_delta_mean": (sum(deltas) / len(deltas)) if deltas else None,
                "bytes_estimate": 1234,
            },
        },
    }


# ------------------------------------------------------ the split itself ----

def test_the_viewer_imports_no_torch():
    """The property the whole split exists for: a trace computed on a GPU node
    can be read anywhere. Checked in a FRESH interpreter, because importing the
    trace module in the same process would load torch and mask this."""
    r = subprocess.run(
        [CPU_PY, "-c",
         "import sys; from experimentation.evaluation import inspect_html;"
         " sys.exit(1 if 'torch' in sys.modules else 0)"],
        cwd=str(REPO), env={"PYTHONPATH": str(REPO), "PATH": "/usr/bin:/bin",
                            "HOME": str(pathlib.Path.home())},
        capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, f"inspect_html pulled in torch:\n{r.stderr}"


def test_the_trace_module_writes_through_the_shared_envelope():
    """Not its own format. A trace has to be findable by the same walk that finds
    quick_eval.json, or the detail and the score drift apart."""
    src = (REPO / "experimentation/evaluation/token_trace.py").read_text()
    assert "from experimentation.evaluation.result import write_result" in src
    assert "find_run_dir" in src, "must land in the run dir like quick_eval does"


# ------------------------------------------------------------- rendering ----

def test_it_renders_a_self_contained_page():
    page = render_html(_fixture())
    assert page.startswith("<!DOCTYPE html>")
    for absent in ("<script src", "<link rel=\"stylesheet\"", "http://", "https://"):
        assert absent not in page, f"page reaches outside itself: {absent!r}"


def test_every_token_appears_with_its_tooltip_payload():
    page = render_html(_fixture())
    assert page.count('class="tk"') == 6, "3 tokens x 2 views"
    assert "&nbsp;quick" in page, "leading space should be visible, not collapsed"
    assert "&quot;rank&quot;: 91" in page, "top-k payload must reach the tooltip"


def test_a_trace_without_ablation_disables_that_view():
    page = render_html(_fixture(with_ablation=False))
    assert page.count('class="tk"') == 3, "only the logprob view"
    assert "const hasSka = false" in page
    assert 'data-view="ska"' not in page


# ------------------------------------------------------------------ colour ----

def test_the_unreadable_ramp_step_is_absent():
    """#2a78d6 measures 4.46 against dark ink and 4.42 against white, so NEITHER
    clears WCAG 4.5:1 and a token landing there would be unreadable. It was
    removed; putting it back would be silent."""
    assert "#2a78d6" not in SEQUENTIAL


@pytest.mark.parametrize("ramp,name", [(SEQUENTIAL, "sequential"),
                                       (DIVERGING, "diverging")])
def test_every_ramp_step_carries_readable_text(ramp, name):
    """Text sits ON the colour, which is why ink is chosen per cell rather than
    once. A fixed ink is exactly what makes the middle of a ramp illegible."""
    from experimentation.evaluation.inspect_html import _contrast
    for step in ramp:
        got = _contrast(step, ink_for(step))
        assert got >= 4.5, f"{name} step {step} only reaches {got:.2f}:1"


def test_the_diverging_scale_is_symmetric_about_zero():
    """Otherwise the midpoint stops meaning 'no effect' and colour misreports the
    sign of whether SKA helped."""
    mid = _bucket(0.0, -1.0, 1.0, DIVERGING)
    assert mid == DIVERGING[len(DIVERGING) // 2]
    assert _bucket(-0.5, -1.0, 1.0, DIVERGING) != _bucket(0.5, -1.0, 1.0, DIVERGING)


def test_the_diverging_midpoint_is_neutral_not_a_hue():
    """A hue in the middle reads as a third category rather than as nothing --
    and 'nothing' is what most tokens are on a lightly-trained model."""
    mid = DIVERGING[len(DIVERGING) // 2]
    r, g, b = (int(mid[i:i + 2], 16) for i in (1, 3, 5))
    assert max(r, g, b) - min(r, g, b) < 20, f"{mid} is not neutral"


def test_the_sequential_ramp_is_one_hue_darkening():
    """Magnitude gets one hue light-to-dark; a rainbow implies categories."""
    from experimentation.evaluation.inspect_html import _luminance
    lums = [_luminance(c) for c in SEQUENTIAL]
    assert lums == sorted(lums, reverse=True), "steps must darken monotonically"


# ---------------------------------------------------------- the comparison ----

def test_two_traces_can_share_one_scale():
    """The point of scale_override: 'step 200 vs step 400' is only a fair read if
    both pages use the same bounds. Self-normalising each would make a worse model
    look identical to a better one."""
    a, b = _fixture(), _fixture()
    b["metrics"]["summary"]["logprob_p05"] = -20.0
    shared = {"logprob_p05": -20.0, "logprob_p95": -0.12, "ska_delta_absmax": 0.51}
    page_a = render_html(a, scale_override=shared)
    page_b = render_html(b, scale_override=shared)
    assert "surprising (-20.00)" in page_a and "surprising (-20.00)" in page_b
    # Same value, same bounds -> same colour. Without the override, a's own p05
    # of -6.40 would have painted it differently.
    assert render_html(a) != page_a


def test_the_sign_convention_is_documented_where_it_is_computed():
    """positive = SKA helped. One subtraction from being inverted, and an
    inverted sign would render confidently backwards with nothing to catch it."""
    src = (REPO / "experimentation/evaluation/token_trace.py").read_text()
    assert "POSITIVE means SKA helped" in src
    assert 'lp - float(ablated[i, target])' in src
