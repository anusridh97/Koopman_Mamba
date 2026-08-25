"""No sbatch script may run `python` before it activates a venv.

Job 445656 died in two seconds with `python: command not found`. The cause was
ordering, not logic: `verify_study_e2e.sbatch` generated its per-job study spec
at line 90 and activated the venv at line 133, and the compute nodes have no
system `python` on PATH. Cheap to lose, but it burned a submission and the
failure was invisible to every existing test.

It was invisible for a specific reason worth naming: the block had been verified
by extracting it and running it in a shell that already had a python. That is
exactly the condition which does not hold under sbatch, so the verification
could not have failed. This file checks the property the extraction cannot --
*where* in the script the interpreter is first needed.

Deliberately syntactic. Actually running a job to find this out costs a queue
wait and a node; reading the file costs nothing and catches the whole class.
"""
from __future__ import annotations

import pathlib
import re

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
SCRIPTS = sorted((REPO / "scripts").glob("*.sbatch")) + \
          sorted((REPO / "scripts").glob("*.sh"))

pytestmark = pytest.mark.correctness

#: Lines that put a python on PATH. `conda activate` is not used here but is
#: cheap to accept, so a future script does not fail this test for being right.
_ACTIVATES = re.compile(r"(source|\.)\s+\S*/bin/activate|conda activate|module load .*python")

#: A bare `python` / `python3` invocation at the start of a command. Anchored so
#: `$PYTHON`, `/abs/path/python`, `KOOPMAN_PY=python` and the word "python" in
#: prose do not match -- an absolute path or a variable is a script saying which
#: interpreter it means, which is the thing this test is not worried about.
#:
#: WIDENED 2026-08-24. This required the next token to be `-`, `-m` or `-c`, so
#: `python path/to/script.py` -- just as dependent on a python being on PATH --
#: was invisible and the case SKIPPED as "invokes no bare python". Found when
#: `report_beta_contrast.sbatch` was added and skipped rather than checked: a
#: guard that silently declines to check a whole invocation style is worse than
#: one that fails, because the skip reads as "nothing to check here".
#:
#: `\S` rather than an enumerated set, so the next style of invocation is
#: covered without a third revision.
_BARE_PYTHON = re.compile(r"^\s*(python3?)\s+\S")


def test_the_script_inventory_is_not_empty():
    """Guard the guard. A glob that matches nothing makes every parametrised
    case below vacuously green, which is how a file like this rots into
    decoration."""
    assert SCRIPTS, "no scripts/*.sbatch or *.sh found -- the glob is wrong"


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_python_is_not_invoked_before_a_venv_is_activated(script):
    lines = script.read_text().splitlines()

    first_activate = None
    first_python = None
    for number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if first_activate is None and _ACTIVATES.search(stripped):
            first_activate = number
        if first_python is None and _BARE_PYTHON.match(line):
            first_python = number

    if first_python is None:
        pytest.skip(f"{script.name} invokes no bare python")
    if first_activate is None:
        pytest.skip(f"{script.name} activates no venv (may use an explicit "
                    f"interpreter path elsewhere)")

    assert first_activate < first_python, (
        f"{script.name} runs `python` at line {first_python} but does not "
        f"activate its venv until line {first_activate}. Compute nodes have no "
        f"system python, so this dies with 'python: command not found' before "
        f"doing any work -- see job 445656. Move the python block after the "
        f"activation, or call an explicit interpreter path.")
