"""§6 schedules and §7 param groups are WIRED, not merely present.

The failure this exists to catch has a shape, and this branch has hit it twice
already:

  * `ska_backend` was accepted, resolved, and read by exactly one of four SKA
    routes -- so setting it on the other three configured nothing while looking
    like it configured something;
  * `chunk_strategy` / `overlap_fraction` / `decay_alpha` survive on
    `KoopmanLMConfig` with no code path reading them at all.

Both were *valid*, *validated*, and *inert*. §6/§7 arrived in the same state when
they were re-landed: `read_spec_extensions` defined and never called,
`_param_groups` never given `groups=`, `loop.py` never applying a schedule -- with
93 tests passing the whole time, because unit tests of a mechanism cannot see that
nothing calls it.

So these are wiring assertions, deliberately made against the SOURCE. That is
normally a weak form of test, and here it is the right one: the alternative is a
full training run, which is a GPU golden's job, and a unit test of
`ScheduleApplier` passes whether or not anything invokes it. What is asserted is
narrow -- that the call exists -- because a call site is exactly what goes missing.
"""

import ast
import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

TRAIN = REPO / "experimentation/training/train.py"
LOOP = REPO / "experimentation/training/loop.py"


def _calls(path):
    """Every function name called anywhere in the module."""
    tree = ast.parse(path.read_text())
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            out.append(getattr(f, "id", None) or getattr(f, "attr", None))
    return [n for n in out if n]


def _kwargs_of_calls_to(path, name):
    """The keyword names passed at each call site of `name`."""
    tree = ast.parse(path.read_text())
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            if (getattr(f, "id", None) or getattr(f, "attr", None)) == name:
                out.append({k.arg for k in node.keywords if k.arg})
    return out


# ------------------------------------------------------------------ §7 ----

def test_the_spec_sections_are_actually_read():
    """`read_spec_extensions` defined but never called is the whole failure
    mode: the flag parses, the file is found, and both sections are dropped."""
    assert "read_spec_extensions" in _calls(TRAIN), (
        "train.py defines read_spec_extensions and never calls it, so --spec is "
        "accepted and both optim.groups and schedules are silently discarded")


def test_param_groups_receives_the_group_specs():
    """§7's entire effect is this one keyword. Without it the optimizer is built
    from the default decay policy and every `optim.groups` entry -- validated at
    spec construction, hashed into run_id -- changes nothing."""
    sites = _kwargs_of_calls_to(TRAIN, "_param_groups")
    assert sites, "train.py no longer calls _param_groups at all"
    assert any("groups" in kw for kw in sites), (
        f"no _param_groups call passes groups=; call sites pass {sites}. "
        f"optim.groups is hashed into run_id, so the run_id would claim a "
        f"per-group policy that never reached the optimizer")


# ------------------------------------------------------------------ §6 ----

def test_the_schedule_applier_is_built():
    assert "ScheduleApplier" in _calls(TRAIN), (
        "train.py imports ScheduleApplier and never constructs one, so no "
        "schedule is ever bound to the live model")


def test_the_loop_applies_schedules_every_step():
    """§6.3 is "one line, before the forward". A schedule is a pure function of
    global step, so applying it is the whole of its resume support too -- and
    skipping it leaves ska_ridge at its step-0 value for the entire run while
    the spec, the run_id and the log all say it annealed."""
    src = LOOP.read_text()
    assert "applier" in src, (
        "loop.py never mentions an applier, so schedules are inert. The loop is "
        "where they must be applied: they are per-step, and the loop owns the "
        "step")
    assert "apply(" in src, "loop.py holds an applier but never calls apply()"


def test_the_epoch_start_step_is_checkpointed():
    """The one thing a schedule needs in resume.pt (§6.5): a seq_len curriculum
    is keyed on the step its EPOCH began, which cannot be recomputed after a
    mid-epoch restart. Everything else is a pure function of global step."""
    src = TRAIN.read_text() + LOOP.read_text()
    assert "epoch_start_step" in src
    assert '"epoch_start_step"' in src, (
        "epoch_start_step is computed but never written into resume state, so a "
        "run resumed mid-epoch rebuilds its dataset at a different seq_len than "
        "its uninterrupted twin")


# ------------------------------------------------ the log tells you it ran ----

def test_scheduled_values_reach_the_progress_line():
    """§6.4: without this you cannot tell "the schedule ran" from "the schedule
    was silently a no-op" -- which is the entire class of bug this file is about,
    one level down."""
    src = LOOP.read_text()
    assert "sched_txt" in src or "sched_values" in src, (
        "no scheduled value reaches the progress line, so a no-op schedule is "
        "indistinguishable from a working one in the log")


def test_both_mechanisms_stay_inert_without_a_spec():
    """The other half of the contract, and the reason this is safe to wire: a
    bare `python -m experimentation.training.train` with no --spec must behave
    exactly as before. Checked by executing it, not by reading it."""
    from experimentation.training.train import read_spec_extensions

    groups, schedules = read_spec_extensions(None)
    assert groups == () and schedules == {}


def test_wiring_section_seven_is_identity_when_no_groups_are_declared():
    """The property that makes the wiring safe to land on a branch with four
    committed golden curves.

    train.py's optimizer call changed from `_param_groups(model, wd)` to
    `_param_groups(model, wd, groups=..., lr=..., include_frozen=...)`. With no
    `optim.groups` declared -- which is every config in this repo -- the two must
    produce the SAME groups, or every existing run's optimizer changed and all
    four goldens move for a reason having nothing to do with schedules.

    Checked on the partition that actually matters: decayed vs not, since the
    §7 change is precisely a refinement of the decay bucketing.
    """
    import torch.nn as nn

    from experimentation.training.optim import param_groups

    class _M(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(8, 8)
            self.norm = nn.LayerNorm(8)
            self.emb = nn.Embedding(4, 8)
            self.ska = nn.Linear(8, 8)

    model = _M()

    def _shape(gs):
        return [(g.get("weight_decay"), g.get("lr", "unset"),
                 tuple(sorted(int(p.numel()) for p in g["params"]))) for g in gs]

    old = _shape(param_groups(model, 0.1))
    new = _shape(param_groups(model, 0.1, groups=(), lr=1e-3,
                              include_frozen=False))
    assert old == new, (
        f"wiring §7 changed the optimizer for a config with no groups:\n"
        f"  before {old}\n  after  {new}")

