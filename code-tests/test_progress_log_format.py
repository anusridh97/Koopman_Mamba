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

def _discover_trainers():
    """Find every module that prints a per-step progress line, by scanning.

    This was a hardcoded list of two, and table2.py was the third -- so the guard
    passed while a trainer carried the exact defect it exists to prevent. A
    hardcoded roster of "things that must conform" fails the moment someone adds
    a fourth, and it fails silently, which is the worst available outcome for a
    test.

    Discovery is by the same AST walk the assertions use: a print whose f-string
    mentions both "step" and "loss" is a progress line. That will occasionally
    pick up something unintended, which is the right direction to be wrong in --
    a false positive gets looked at, a false negative does not.
    """
    found = []
    for pkg in ("experimentation",):
        for path in sorted((REPO / pkg).rglob("*.py")):
            if "__pycache__" in path.parts or "/data/" in str(path):
                continue
            try:
                if _progress_skeletons(path.relative_to(REPO)):
                    found.append(str(path.relative_to(REPO)))
            except SyntaxError:
                continue
    return found


def _render(node):
    """A print argument's literal shape, with "10" for every interpolated field.

    Handles BinOp concatenation, not just a lone f-string. table2.py writes
    f"..." + "  ".join(parts) + f"..." -- three nodes, and a scanner that only
    understood JoinedStr skipped the whole call silently.
    """
    if isinstance(node, ast.JoinedStr):
        return "".join(_render(v) for v in node.values)
    if isinstance(node, ast.Constant):
        return str(node.value)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _render(node.left) + _render(node.right)
    return "10"


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
        rendered = "".join(_render(arg) for arg in node.args)
        # "step" alone, not "step" AND "loss": table2 builds its loss labels at
        # runtime ("  ".join(parts) where parts are f"{name}_loss ..."), so the
        # literal word never appears in the source. Requiring both is what made
        # this scanner miss the very file that motivated writing it.
        if "step" in rendered and "/" in rendered:
            out.append(rendered)
    return out


#: Must conform: these are the trainers a study can launch, so their logs are
#: what the pruner tails. train.py today; the other two once SyntheticDataSpec
#: is wired (§6.2), and getting the format right before that is cheaper than
#: after.
SEARCH_LAUNCHABLE = (
    "experimentation/training/train.py",
    "experimentation/experiments/mqar_finetune.py",
    "experimentation/experiments/table2.py",
)

#: Discovered, and deliberately NOT required to conform, with the reason. Both
#: are fine-tuning/eval entry points that no study launches, so their progress
#: lines are for humans and nothing parses them.
EXEMPT = {
    "experimentation/evaluation/evaluate_retrieval.py":
        "retrieval fine-tune, not search-launchable",
    "experimentation/retrieval/adapt.py":
        "adapter fine-tune, not search-launchable",
}

DISCOVERED = _discover_trainers()
TRAINERS = [p for p in DISCOVERED if p in SEARCH_LAUNCHABLE]


def test_discovery_sees_every_search_launchable_trainer():
    """A discovery-based roster can silently find nothing, or miss one. table2
    was missed exactly once, because its line is a BinOp concatenation."""
    for known in SEARCH_LAUNCHABLE:
        assert known in DISCOVERED, f"discovery missed {known}: {DISCOVERED}"


def test_no_undeclared_trainer_appeared():
    """A new module printing a progress line must be classified, not ignored.
    Silence here is how table2 stayed invisible while carrying the defect."""
    unclassified = sorted(set(DISCOVERED) - set(SEARCH_LAUNCHABLE) - set(EXEMPT))
    assert not unclassified, (
        f"{unclassified} print progress lines and are neither declared "
        "search-launchable nor exempted. Decide which, with a reason.")


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


def test_all_trainers_agree_on_the_canonical_prefix():
    """Not merely 'each is parseable' -- the part TRAIN_RE consumes must be
    IDENTICAL across trainers, since one regex serves both the pruner and the
    golden comparator.

    Compares `TRAIN_RE.search(...).group(0)` rather than the whole line, because
    a trailing suffix is both harmless (search matches a prefix) and sometimes
    load-bearing: table2.py appends per-task `toolcall_loss` / `sysprompt_loss`
    fields, and the comment on its loss_sum records that this visibility caught
    two real bugs a single number would have hidden. Demanding whole-line
    equality would force deleting that.
    """
    shapes = {}
    for path in TRAINERS:
        for s in _progress_skeletons(path):
            m = TRAIN_RE.search(s)
            if m:
                shapes.setdefault(path, m.group(0).replace("10", "N"))
    assert len(shapes) == len(TRAINERS), (
        f"only {sorted(shapes)} produced a parseable line, of {TRAINERS}")
    assert len(set(shapes.values())) == 1, (
        "trainers disagree on the canonical prefix; one regex cannot serve "
        f"both the pruner and the golden comparator:\n"
        + "\n".join(f"  {k}: {v!r}" for k, v in shapes.items()))


def test_a_trailing_suffix_does_not_break_parsing():
    """The property that makes table2's per-task fields safe."""
    line = ("step    10/400 | loss 1.5 | ppl 4.5 | lr 1e-4 | 9.9K tok/s"
            "  toolcall_loss 1.4  sysprompt_loss 1.6")
    (pt,) = parse_progress(line)
    assert (pt.step, pt.loss) == (10, 1.5)


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
