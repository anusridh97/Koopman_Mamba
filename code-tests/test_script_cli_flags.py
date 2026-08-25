"""Every flag a committed script passes to one of our CLIs must exist.

The guard that would have caught a removal. `--run_root`, `--launcher` and
`--n_trials` were removed from `experimentation.sweep.search` for a good reason --
a run_root that changes without changing `study_id` sends two workers to
different journals while each believes it is collaborating -- and nobody grepped
for callers. `scripts/verify_study_e2e.sbatch` kept passing `--run_root` and would
have died with "unrecognized arguments" after the queue wait, on the gate job
whose whole purpose is to catch that class of problem.

Nothing on the CPU side could see it: an sbatch script's only consumer is Slurm.
So this parses the flags out of every `scripts/*.sbatch` and `scripts/*.sh`
invocation of a `python -m experimentation.*` CLI and compares them against that
CLI's actual argparse parser.

**Per CLI, not globally.** `experimentation.run` still accepts `--run_root` and
`--launcher`; only `sweep.search` removed them. Five scripts legitimately pass
those to `experimentation.run`, and a global "these flags are banned" check would
have failed all five and taught people to delete the check.

**The parser is the source of truth**, obtained by importing the module and
reading its `parse_args`. A hardcoded list of known flags is the roster that goes
stale -- which is the failure this whole file is about.
"""
from __future__ import annotations

import argparse
import importlib
import pathlib
import re
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

pytestmark = pytest.mark.correctness

#: CLIs whose flags we can check. Each must expose `parse_args`, and must import
#: without optional dependencies -- `sweep.search` deliberately does.
#:
#: `experiments.mqar_finetune` is here because two committed launchers drive it
#: (`run_beta_policy_mqar.sbatch`, `run_beta_exponent_mqar.sbatch`) and it was
#: previously unchecked: a flag typo in either would have surfaced as
#: "unrecognized arguments" per run, after the queue wait and after a GPU was
#: claimed, on a job whose runs are the experiment. It imports cleanly in the CPU
#: venv (torch only; the mamba_ssm import is inside the model builders), which is
#: what makes it checkable at all.
CHECKABLE = ("experimentation.run", "experimentation.sweep",
             "experimentation.sweep.search",
             "experimentation.experiments.mqar_finetune")


def _scripts():
    return sorted([*(REPO / "scripts").glob("*.sbatch"),
                   *(REPO / "scripts").glob("*.sh")])


def _accepted_flags(module_name):
    """Every option string the CLI's parser accepts.

    Read off the parser rather than the `--help` text: help output wraps, and a
    long option can be split across lines by argparse's formatter, which would
    make a flag look absent.
    """
    # `python -m pkg` runs `pkg.__main__`; `python -m pkg.mod` runs `pkg.mod`
    # itself. Both spellings appear in the launchers, so try the package form
    # first and fall back to the module -- guessing one would silently return
    # None for the other, and None means "unchecked", which is the state this
    # whole file exists to eliminate.
    try:
        module = importlib.import_module(module_name + ".__main__")
    except ModuleNotFoundError:
        module = importlib.import_module(module_name)
    captured = {}

    real_parse = argparse.ArgumentParser.parse_args

    def spy(self, *args, **kwargs):
        captured.setdefault("parser", self)
        raise _Stop()

    class _Stop(Exception):
        pass

    argparse.ArgumentParser.parse_args = spy
    try:
        try:
            module.parse_args([])
        except _Stop:
            pass
    finally:
        argparse.ArgumentParser.parse_args = real_parse

    parser = captured.get("parser")
    if parser is None:
        return None
    flags = set()
    for action in parser._actions:
        flags.update(action.option_strings)
    return flags


def _commands(text):
    """Executable lines with continuations joined and comments dropped.

    Both matter and both have bitten. `verify_study_e2e.sbatch` writes the
    invocation and the flag on DIFFERENT physical lines, so a per-line scan
    missed it; and these scripts explain at length why a removed flag is absent,
    so a whole-text scan finds the explanation and calls it a violation.
    """
    joined = re.sub(r"\\\n\s*", " ", text)
    return [line for line in joined.splitlines()
            if line.strip() and not line.lstrip().startswith("#")]


def _invocations(text):
    """(module, [flags]) for every `python -m experimentation.<x>` command."""
    found = []
    for command in _commands(text):
        for match in re.finditer(
                r"python\d?\s+-m\s+(experimentation[\w.]*)", command):
            module = match.group(1)
            tail = command[match.end():]
            # Stop at a pipe or redirect: what follows belongs to another command.
            tail = re.split(r"[|>]", tail)[0]
            flags = re.findall(r"(?<![\w-])(--[A-Za-z][\w-]*)", tail)
            found.append((module, flags))
    return found


# ------------------------------------------------------------ guard the guard ----

def test_there_are_scripts_to_check():
    assert len(_scripts()) > 5


def test_at_least_one_script_invokes_a_checkable_cli():
    """A regex that matched nothing would make every assertion below vacuous."""
    total = 0
    for path in _scripts():
        total += sum(1 for module, _ in _invocations(path.read_text())
                     if module in CHECKABLE)
    assert total >= 3, f"only {total} checkable invocation(s) found"


@pytest.mark.parametrize("module", CHECKABLE)
def test_each_cli_exposes_its_flags(module):
    """If `_accepted_flags` silently returned None or an empty set, every script
    would pass or every script would fail -- both useless."""
    flags = _accepted_flags(module)
    assert flags, module
    assert "--dry_run" in flags, (module, sorted(flags))


def test_the_search_cli_really_dropped_the_three_flags():
    """The premise of this whole file. If they came back, the check would still
    pass and would be testing nothing."""
    flags = _accepted_flags("experimentation.sweep.search")
    for gone in ("--run_root", "--launcher", "--n_trials"):
        assert gone not in flags, (
            f"{gone} is accepted again -- either that was deliberate (update this "
            f"test and say why) or it was reintroduced by accident")


def test_the_run_cli_still_accepts_run_root_and_launcher():
    """Why the check is PER CLI. Five scripts legitimately pass these to
    `experimentation.run`; a global ban would fail all five."""
    flags = _accepted_flags("experimentation.run")
    assert "--run_root" in flags
    assert "--launcher" in flags


# ------------------------------------------------------------- the real check ----

@pytest.mark.parametrize("path", _scripts(), ids=lambda p: p.name)
def test_every_flag_a_script_passes_is_one_its_cli_accepts(path):
    """The check itself. A flag the CLI does not know is an argparse error on the
    compute node, after the queue wait, on a job whose output nobody reads until
    it has already failed."""
    problems = []
    for module, flags in _invocations(path.read_text()):
        if module not in CHECKABLE:
            continue
        accepted = _accepted_flags(module)
        if not accepted:
            continue
        for flag in flags:
            if flag not in accepted:
                problems.append(
                    f"{module} does not accept {flag} "
                    f"(accepts: {sorted(accepted)})")
    assert not problems, (
        f"{path.name} passes flag(s) its CLI removed; the job will die at "
        f"argparse:\n  " + "\n  ".join(problems))


# ------------------- per-job isolation, now that a flag cannot supply it ----
#
# `verify_study_e2e.sbatch` used to pass `--run_root $SCRATCH/study-smoke-$JOBID`.
# The flag is gone, and the PER-JOB part of it was load-bearing in a way an
# argparse error would not have revealed: `smoke-4m.yaml` has a FIXED run_root and
# a study is RESUMABLE (`n_trials` is the target size, not "this many more"). So a
# second submission against a fixed root finds its 4 trials already COMPLETE, runs
# ZERO trials, and reports PASS -- a gate that passes for the wrong reason, which
# is worse than one that fails.
#
# The run root therefore moved into a GENERATED per-job spec. These tests exist
# because reverting the invocation to the committed config would restore that
# silent failure with no flag error to notice, and a mutation proved nothing
# caught it.

E2E = REPO / "scripts/verify_study_e2e.sbatch"


def _search_invocations(text):
    return [c for c in _commands(text)
            if "-m experimentation.sweep.search" in c]


def test_the_e2e_gate_invokes_the_generated_spec_not_the_committed_one():
    """Catches reverting `"$GENERATED_STUDY"` to `"$STUDY_CONFIG"`.

    That mutation passes every flag check -- the flags are fine -- and silently
    removes per-job isolation, so a resubmitted gate would run zero trials and
    pass.
    """
    invocations = _search_invocations(E2E.read_text())
    assert invocations, "the gate no longer invokes the search CLI"
    for command in invocations:
        assert "$GENERATED_STUDY" in command, (
            f"this invocation uses the COMMITTED study config, which has a FIXED "
            f"run_root -- a resubmission would find its trials already COMPLETE, "
            f"run zero, and report PASS: {command.strip()}")
        assert "$STUDY_CONFIG" not in command, command.strip()


def test_the_generated_spec_is_written_before_it_is_invoked():
    """Order matters and is checkable: invoking a file that does not exist yet is
    a startup error, and the generation block is what makes the run root
    per-job."""
    commands = _commands(E2E.read_text())
    generated_at = next(i for i, c in enumerate(commands)
                        if "GENERATED_STUDY=" in c)
    first_use = next(i for i, c in enumerate(commands)
                     if "$GENERATED_STUDY" in c and "GENERATED_STUDY=" not in c)
    assert generated_at < first_use


def test_the_generated_run_root_is_per_job():
    """The property the flag used to provide. Without `SLURM_JOB_ID` in it, two
    submissions share a run root and therefore a journal."""
    text = E2E.read_text()
    roots = [c for c in _commands(text) if "RUN_ROOT=" in c]
    assert roots
    assert any("SLURM_JOB_ID" in c for c in roots), roots


def test_the_generation_step_aborts_on_failure():
    """An unparseable generated spec must stop the job, not leave it to fail at
    the dry run with a less obvious message."""
    text = E2E.read_text()
    block = [c for c in _commands(text) if "python - <<'PY'" in c]
    assert block, "the generation heredoc is gone"
    assert "exit 1" in block[0], (
        f"the generation step does not abort on failure: {block[0]}")


def test_the_committed_config_is_never_mutated_in_place():
    """The generated file goes to the job's own scratch. Rewriting the committed
    YAML would make the repo state depend on whether a job had run."""
    text = E2E.read_text()
    assert 'GENERATED_STUDY="$RUN_ROOT/' in text, (
        "the generated spec must live under the per-job run root")
    for command in _commands(text):
        assert not re.search(r">\s*\$STUDY_CONFIG", command), command
        assert not re.search(r">\s*configs/", command), command
