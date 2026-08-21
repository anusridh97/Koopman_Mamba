"""Every trainer's progress line must be readable by the search layer.

`sweep/search/metrics.py::TRAIN_RE` is how an optuna trial is pruned: the driver
tails the run's log, parses progress lines, EMA-smooths the losses, calls
`trial.report(value, step)` and then `trial.should_prune()`. If the regex cannot
read a trainer's output, `parse_progress` returns `[]`, `wait_for_objective` sees
zero progress points, and that trial becomes **unprunable -- silently**. No error,
no warning; it simply always runs to `max_steps`.

That was the real state of `mqar_finetune.py`, which printed

    step     10/400  loss 3.1234  ppl 22.7  lr 1.00e-04  12s

-- double-spaced, no pipes, elapsed seconds instead of throughput. `TRAIN_RE`
requires pipe separators and a `K tok/s` field, so the synthetic path was
invisible to pruning. It also meant the golden-curve comparator could not read it,
which is how this was found: trying to capture an MQAR baseline before the §6.2
TrainTask unification.

The same regex is shared by the pruner and `scripts/compare_golden_curve.py` on
purpose -- if the format drifts, both drift together instead of one silently
disagreeing with the other. That only holds while every trainer emits the same
shape, which is what this file pins.

The check reads the f-strings out of the source with `ast` rather than comparing
against a copied literal, so it stays honest if someone reformats the line.
"""

import ast
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from experimentation.sweep.search.metrics import TRAIN_RE, parse_progress  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]

# Every module that prints a per-step training progress line.
TRAINERS = [
    "experimentation/training/train.py",
    "experimentation/experiments/mqar_finetune.py",
]


def _progress_skeletons(path):
    """Reconstruct each f-string's literal shape, with '10' for every field.

    '10' rather than a float because TRAIN_RE requires \\d+ for the step and
    accepts [0-9.eE+-]+ elsewhere, so one token satisfies both.
    """
    tree = ast.parse((REPO / path).read_text())
    out = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)
                and getattr(node.func, "id", None) == "print"):
            continue
        parts = []
        for arg in node.args:
            if isinstance(arg, ast.JoinedStr):
                for piece in arg.values:
                    if isinstance(piece, ast.Constant):
                        parts.append(str(piece.value))
                    else:
                        parts.append("10")
            elif isinstance(arg, ast.Constant):
                parts.append(str(arg.value))
        rendered = "".join(parts)
        if "step" in rendered and "loss" in rendered:
            out.append(rendered)
    return out


@pytest.mark.parametrize("path", TRAINERS)
def test_trainer_progress_line_is_parseable(path):
    skeletons = _progress_skeletons(path)
    assert skeletons, f"no progress-shaped print found in {path}"
    matched = [s for s in skeletons if TRAIN_RE.search(s)]
    assert matched, (
        f"{path} prints a progress line the search layer cannot read, so any "
        f"optuna trial running it is silently unprunable. Candidates:\n  "
        + "\n  ".join(repr(s) for s in skeletons))


@pytest.mark.parametrize("path", TRAINERS)
def test_trainer_progress_line_has_a_pipe_and_throughput(path):
    """The two things the old mqar format lacked, named explicitly so a failure
    says what to fix rather than just 'regex did not match'."""
    skeletons = [s for s in _progress_skeletons(path) if TRAIN_RE.search(s)]
    assert skeletons
    for s in skeletons:
        assert "|" in s, f"{path}: progress line needs pipe separators: {s!r}"
        assert "tok/s" in s, f"{path}: progress line needs a tok/s field: {s!r}"


def test_all_trainers_agree_on_one_shape():
    """Not merely 'each is parseable' -- they must be the SAME shape, since one
    regex serves both the pruner and the golden comparator."""
    shapes = {}
    for path in TRAINERS:
        for s in _progress_skeletons(path):
            if TRAIN_RE.search(s):
                # Collapse the substituted values; compare only the skeleton.
                shapes.setdefault(path, s.replace("10", "N"))
    assert len(set(shapes.values())) == 1, (
        "trainers disagree on the progress format; one regex cannot serve "
        f"both the pruner and the golden comparator:\n"
        + "\n".join(f"  {k}: {v!r}" for k, v in shapes.items()))


def test_the_old_mqar_format_would_have_failed_this():
    """A control. If this ever passes, the regex has loosened enough that the
    original defect would slip through again."""
    old = "step     10/400  loss 3.1234  ppl 22.7  lr 1.00e-04  12s"
    assert parse_progress(old) == [], (
        "TRAIN_RE now accepts the double-spaced, throughput-less format that "
        "made synthetic trials unprunable -- this test no longer guards anything")


def test_the_current_format_parses_to_real_values():
    line = "step     10/400 | loss 3.1234 | ppl 22.7 | lr 1.00e-04 | 50.0K tok/s"
    (p,) = parse_progress(line)
    assert (p.step, p.loss, p.ppl) == (10, 3.1234, 22.7)
    assert p.tokens_per_sec == 50000.0
