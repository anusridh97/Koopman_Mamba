"""Every trainer has a golden curve, and every golden curve is usable.

`docs/superpowers/specs/2026-08-21-traintask-design.md` §10 gates the three-loop
unification on "all four bit-exact instruments". That gate is only real if
something checks the instruments exist. Nothing did: the goldens were captured
one at a time by hand, and the roster of which trainers needed one lived in my
head.

That is the same failure `_discover_trainers` in test_progress_log_format.py was
written to fix one level up -- a hardcoded roster of "things that must conform"
fails silently the moment someone adds a fourth. So the trainer list here is
*discovered*, by importing that module's own scanner rather than reimplementing
it. Two scanners that are supposed to agree eventually will not.

What this does NOT do is check that a golden still MATCHES. That needs a GPU and
40 minutes; it is `scripts/compare_golden_curve.py`'s job, run from the capture
sbatch. This checks only that the instrument is present, parseable, and has the
fields the comparator reads -- so a missing or malformed golden fails on a laptop
in 0.2 s instead of at the end of a GPU job.
"""

import json
import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "code-tests"))

from test_progress_log_format import _discover_trainers  # noqa: E402

GOLDENS = sorted(REPO.glob("code-tests/golden_*_curve.json"))

# Trainers with no golden curve, and why. A written reason is required: the whole
# point of discovery is that nothing drops off the roster silently.
#
# BOTH ENTRIES ARE A FINDING, not housekeeping. The TrainTask design doc
# (§6, §9) is written about "the three loops" -- train.py, mqar_finetune.py,
# table2.py -- and its out-of-scope list says nothing about retrieval. But these
# two modules each run a full training loop: an AdamW, a scheduler,
# `loss.backward()` and `opt.step()`. So there are FIVE training loops in this
# repo, the unification was scoped against a census of three, and whether these
# two belong in it is a decision that document did not anticipate.
#
# Recorded here rather than quietly excluded, and rather than halting step 4 over
# it: table2 is fully specified and unaffected. See REVIEW.md.
NO_CURVE_EXPECTED = {
    "experimentation.retrieval.adapt": (
        "A fourth training loop (AdamW + backward + step, contrastive retrieval "
        "adapter), outside the TrainTask design doc's declared three-loop scope. "
        "No golden captured because it is not being refactored yet -- and it must "
        "not be refactored until one is."),
    "experimentation.evaluation.evaluate_retrieval": (
        "A fifth training loop, embedded in something named evaluate_*: it "
        "builds an optimizer and calls loss.backward() (see ~line 471-507). "
        "Same status as retrieval/adapt.py."),
}


def _goldens_by_trainer():
    out = {}
    for path in GOLDENS:
        payload = json.loads(path.read_text())
        out.setdefault(payload.get("trainer", ""), []).append(path)
    return out


def test_the_discovery_finds_the_trainers_we_know_about():
    """Guards the guard. If discovery silently returns [] every parametrised test
    below is vacuously green, which is how the hardcoded-roster bug survived."""
    found = _discover_trainers()
    assert len(found) >= 3, f"expected at least 3 trainers, discovered {found}"
    joined = " ".join(found)
    for expected in ("training/train.py", "mqar_finetune.py", "table2.py"):
        assert expected in joined, f"{expected} not discovered; got {found}"


@pytest.mark.parametrize("trainer_path", _discover_trainers())
def test_every_trainer_has_a_golden_curve(trainer_path):
    """A trainer with no golden cannot be refactored safely: "the loss still goes
    down" is not evidence the loop is unchanged. Both loop swaps done so far
    landed at 0.0001 and 0.000000 against their goldens -- that is the standard,
    and it needs an instrument per loop to hold to it."""
    module = trainer_path.replace("/", ".").removesuffix(".py")
    if module in NO_CURVE_EXPECTED:
        pytest.skip(NO_CURVE_EXPECTED[module])

    by_trainer = _goldens_by_trainer()
    assert module in by_trainer, (
        f"{trainer_path} prints a progress line but has no golden curve.\n"
        f"Goldens present: {sorted(k for k in by_trainer if k)}\n"
        f"Capture one with a scripts/capture_*_golden.sbatch, or add "
        f"{module!r} to NO_CURVE_EXPECTED with a written reason.")


@pytest.mark.parametrize("path", GOLDENS, ids=lambda p: p.name)
def test_a_golden_carries_what_the_comparator_reads(path):
    """compare_golden_curve.py reads replicates[<rep>] and noise_floor; a golden
    missing either fails at the end of a GPU job instead of here."""
    payload = json.loads(path.read_text())
    assert payload.get("commit"), f"{path.name} records no commit"
    # `args` OR `spec`: the synthetic goldens are reproduced from CLI args, the
    # shard golden from a run spec file. Both say how to re-capture it; requiring
    # `args` specifically just encoded the format I happened to look at first.
    assert payload.get("args") or payload.get("spec"), (
        f"{path.name} records neither args nor spec, so nothing says how to "
        f"reproduce it")

    reps = payload.get("replicates") or {}
    assert set(reps) >= {"A", "B"}, (
        f"{path.name} has replicates {sorted(reps)}; two are needed because the "
        f"noise floor IS their disagreement -- one replicate measures nothing")
    for name, curve in reps.items():
        assert curve, f"{path.name} replicate {name} is empty"
        assert all(int(k) >= 0 for k in curve), f"{path.name}: non-step key"

    floor = payload.get("noise_floor") or {}
    for field in ("max_abs_delta", "mean_abs_delta", "n_shared_steps"):
        assert field in floor, f"{path.name} noise_floor lacks {field}"
    assert floor["n_shared_steps"] > 1, (
        f"{path.name} shares {floor['n_shared_steps']} step(s) between "
        f"replicates; a floor measured on one point is not a floor")


@pytest.mark.parametrize("path", GOLDENS, ids=lambda p: p.name)
def test_the_two_replicates_actually_agree_to_the_recorded_floor(path):
    """The floor is a claim about the file's own contents, so it is checkable
    here with no GPU. A golden whose recorded floor disagrees with its own
    replicates would set the tolerance for every future comparison from a number
    that was never true."""
    payload = json.loads(path.read_text())
    a, b = payload["replicates"]["A"], payload["replicates"]["B"]
    shared = sorted(set(a) & set(b), key=int)
    deltas = [abs(a[k] - b[k]) for k in shared]

    assert len(shared) == payload["noise_floor"]["n_shared_steps"]
    recorded = payload["noise_floor"]["max_abs_delta"]
    assert max(deltas) == pytest.approx(recorded, abs=1e-9), (
        f"{path.name} records max_abs_delta={recorded} but its replicates "
        f"actually differ by {max(deltas)}")


@pytest.mark.parametrize("path", GOLDENS, ids=lambda p: p.name)
def test_the_args_record_can_actually_reproduce_the_step_grid(path):
    """`args` claims to say how to re-capture the curve. For the CLI-driven
    goldens it did not: the step grid is set by --log_every, whose defaults are
    100 (mqar_finetune) and 200 (table2), while both goldens log every 10. So
    "reproduce from the recorded args" would have produced 4 points and compared
    them against 40, and the comparator would have reported 36 missing steps as
    though the run were broken.

    Skipped for spec-driven goldens: their logging cadence lives in the run spec,
    not in an args dict.
    """
    payload = json.loads(path.read_text())
    args = payload.get("args")
    if not args:
        pytest.skip(f"{path.name} is spec-driven ({payload.get('spec')})")

    steps = sorted(int(k) for k in payload["replicates"]["A"])
    strides = {b - a for a, b in zip(steps, steps[1:])}
    assert len(strides) == 1, f"{path.name} has an uneven step grid: {sorted(strides)}"
    stride = strides.pop()

    assert args.get("log_every") == stride, (
        f"{path.name} logs every {stride} steps but its args record "
        f"log_every={args.get('log_every')!r}. The args are what a re-capture "
        f"is driven from, so an omitted one silently changes the grid.")

