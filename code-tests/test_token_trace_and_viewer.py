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


# ------------------------------------- what only a real checkpoint showed ----
#
# Everything below was added after running the trace against a 200-step 5.2M
# checkpoint and a real fineweb shard for the first time. None of it was
# reachable from the hand-written fixture above, which is the point: the fixture
# has no whitespace tokens, no tokenizer, and no filesystem.


def test_whitespace_tokens_stay_distinguishable():
    """' ', '\\n' and '\\n\\n' are three different tokens and HTML renders all
    three as one collapsed blank. Real shard text is mostly whitespace-adjacent;
    the fixture has none, which is why this needed a real trace to find."""
    from experimentation.evaluation.inspect_html import label_for
    got = {label_for(t) for t in (" ", "\n", "\n\n", "\t", "")}
    assert len(got) == 5, f"whitespace tokens collided: {got}"
    assert " " not in label_for(" "), "a literal space would collapse"
    assert "\n" not in label_for("\n"), "a literal newline would collapse"


def test_a_newline_token_breaks_the_line_it_renders_on():
    """Otherwise a 512-token trace is one unbroken block and the paragraph
    structure of the source document is invisible."""
    trace = _fixture()
    trace["metrics"]["sequences"][0]["tokens"][1]["token"] = "\n"
    page = render_html(trace)
    assert page.count("</span><br>") == 2, "both views break the line"
    # Counted as </span><br> specifically: the tooltip script contains a bare
    # <br> of its own, so a plain count of "<br>" is never zero.
    assert render_html(_fixture()).count("</span><br>") == 0


def test_the_summary_says_WHY_there_is_no_ablation():
    """quick_eval's docstring: "`supported: False` is distinct from
    `loss_delta: 0.0`". A lone has_ablation bool makes --no_ablation
    indistinguishable from a model that cannot be ablated at all."""
    from experimentation.evaluation import token_trace as tt

    reports = [{"tokens": [{"pos": 0, "token": "a", "logprob": -1.0,
                            "rank": 1, "top": [["a", 0.4]]}]}]
    for state in ("skipped", "unsupported"):
        summary = tt.assemble_trace(reports, top_k=1, source="s",
                                    ablation=state)["summary"]
        assert summary["ablation"] == state
        assert summary["has_ablation"] is False
    with pytest.raises(ValueError, match="no token carries a ska_delta"):
        tt.assemble_trace(reports, top_k=1, source="s", ablation="measured")


def test_bytes_estimate_is_measured_not_modelled():
    """It was `n * (90 + 26*top_k)`, which read 2.8x LOW against a real trace:
    it costed the numbers and not the indent=2 both write paths use. Too low is
    the wrong direction for a number whose job is to warn you."""
    from experimentation.evaluation import token_trace as tt

    tokens = [{"pos": i, "token": "tok", "logprob": -7.0, "rank": 900,
               "top": [["a", 0.01]] * 5, "ska_delta": 0.0} for i in range(40)]
    payload = tt.assemble_trace([{"tokens": tokens}], top_k=5, source="s")
    actual = len(json.dumps(payload, indent=2))
    estimate = payload["summary"]["bytes_estimate"]
    assert 0.7 * actual <= estimate <= 1.3 * actual, (estimate, actual)


def test_top_k_below_one_is_rejected_at_the_boundary():
    """--top_k 0 ran a full GPU pass and wrote a trace whose every `top` was [],
    which is a page of coloured cells with nothing behind the tooltip."""
    from experimentation.evaluation import token_trace as tt
    with pytest.raises(SystemExit):
        tt.main(["--checkpoint", "x", "--data_dir", "y", "--top_k", "0"])


def test_the_out_path_is_written_atomically_through_a_created_parent(tmp_path):
    """Path.write_text raised FileNotFoundError on a missing parent -- after
    every forward pass had already been paid for."""
    src = (REPO / "experimentation/evaluation/token_trace.py").read_text()
    assert "atomic_write_json(written, payload)" in src
    assert "Path(args.out).write_text" not in src
    from experimentation.atomic_io import atomic_write_json
    target = tmp_path / "a" / "b" / "trace.json"
    atomic_write_json(target, {"ok": True})
    assert json.loads(target.read_text()) == {"ok": True}


def test_a_single_token_id_decodes_with_its_leading_space():
    """`tokenizer.decode([id])` drops SentencePiece's boundary marker, so a
    trace of "in India" rendered as "inIndia" -- no word boundaries anywhere.
    Checked against a stand-in with the same contract as the real tokenizer,
    since the real one is a 500MB download."""
    from experimentation.evaluation.token_trace import _decode

    class SPLike:
        pieces = {7: "▁India", 8: "ving", 9: "▁"}

        def decode(self, ids):
            return self.pieces[ids[0]].replace("▁", "")

        def convert_ids_to_tokens(self, tid):
            return self.pieces[tid]

    tok = SPLike()
    assert _decode(tok, 7) == " India"
    assert _decode(tok, 8) == "ving"
    assert _decode(tok, 9) == " ", "the bare boundary piece IS a space"


def test_the_ablation_ramp_is_bounded_by_a_percentile_not_the_outlier():
    """The logprob ramp already uses p05/p95 for this reason; the diverging one
    used raw |max| and so painted 78.5% of a real trace's tokens with the
    neutral gray that means "SKA did nothing" -- while 96% of them had moved by
    more than 1e-4. The view was hiding its own finding."""
    from experimentation.evaluation import token_trace as tt

    deltas = [0.004] * 99 + [0.9]            # one outlier, 99 real signals
    reports = [{"tokens": [
        {"pos": i, "token": "t", "logprob": -7.0, "rank": 9, "top": [["a", 0.1]],
         "ska_delta": d} for i, d in enumerate(deltas)]}]
    summary = tt.assemble_trace(reports, top_k=1, source="s")["summary"]
    assert summary["ska_delta_absmax"] == 0.9, "the outlier bound is still recorded"
    assert summary["ska_delta_p95abs"] == 0.004

    trace = {"metrics": {"sequences": reports, "summary": summary}}
    page = render_html(trace)
    mid = DIVERGING[len(DIVERGING) // 2]
    assert page.count(f"background:{mid}") <= 2, \
        "the 99 real signals must not all collapse into the neutral bucket"


def test_also_shares_every_bound_the_renderer_might_read(tmp_path):
    """--also builds the shared scale from the summaries. Miss the key the
    renderer actually prefers and each page silently self-normalises again."""
    import inspect

    from experimentation.evaluation import inspect_html as ih
    src = inspect.getsource(ih.main)
    for key in ("ska_delta_absmax", "ska_delta_p95abs",
                "logprob_p05", "logprob_p95"):
        assert key in src, f"--also drops {key} from the shared scale"
