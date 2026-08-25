"""The exponent arm's launchers, checked against the design they claim to run.

## The failure this file exists to prevent

An earlier version of this file extracted the gate block from
`run_beta_exponent_mqar.sbatch` and drove it through mutations -- good -- but it
ALSO hardcoded its own copy of the cell list. So the assertion "7 cells, 7
distinct configs" said nothing about the arm the sbatch would launch: adding an
eighth cell, or a duplicate, or a typo'd policy to the sbatch's own `CELLS=( … )`
left every test passing. That was not hypothetical; `STEPS` and `SEEDS` were
edited in that file and no test noticed.

The fix was to delete the duplication rather than test around it. Both launchers
now read `experimentation/experiments/beta_exponent_cells.py`, which is tested
directly in `test_beta_exponent_cells.py` -- gates, cell list, budget and
array-index map. What is left for this file is the part that module cannot check:
**that the shell scripts actually go through it, and that the preparer CLI works
end to end.**

A test that greps a shell script is a weak test. It is the right strength here:
the strong assertions live against the module, and the only remaining risk is a
launcher that stops consulting it.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
WHOLE_NODE = REPO / "scripts" / "run_beta_exponent_mqar.sbatch"
ARRAY = REPO / "scripts" / "run_beta_exponent_array.sbatch"
PREPARE = REPO / "scripts" / "prepare_beta_exponent_arm.py"

pytestmark = pytest.mark.correctness


def _text(p: Path) -> str:
    assert p.exists(), f"{p} is missing"
    return p.read_text()


# ---------------------------------------------------------------------------
# Neither launcher may carry its own copy of the design.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("script", [WHOLE_NODE, ARRAY], ids=lambda p: p.name)
def test_the_launcher_reads_the_design_from_the_tested_module(script):
    t = _text(script)
    assert "beta_exponent_cells" in t, (
        f"{script.name} does not consult beta_exponent_cells.py, so nothing "
        f"ties the arm it launches to the arm the tests check")


@pytest.mark.parametrize("script", [WHOLE_NODE, ARRAY], ids=lambda p: p.name)
def test_no_launcher_carries_a_literal_cell_list(script):
    """The specific duplication that made the old assertions vacuous.

    Scans for the SHAPE of a hardcoded cell list rather than for policy names:
    `learned`, `linear` and `one` are ordinary English words that appear in these
    scripts' prose ("one run per GPU", "one wave per SEED"), so a name scan is
    all false positives. What cannot be prose is a `name:policy:ridge` triple, or
    a bash array of them, or the two distinctive mixed-cell names.

    Comments are skipped: the scripts explain the design at length, and the
    python heredocs name the module they import from, which is the point.
    """
    offending = []
    for line in _text(script).splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "beta_exponent_cells" in s:
            continue
        # A literal `something:something:0.0…` triple -- the launcher's own
        # encoding of a cell.
        if re.search(r'["\']?\w[\w.-]*:\w[\w.-]*:0?\.\d', s):
            offending.append(("literal cell triple", s))
        # The mixed-cell names are not English words, so a bare occurrence in an
        # executable line is a hardcoded cell.
        for pol in ("key_linear_value_sqrt", "key_sqrt_value_linear"):
            if re.search(rf'(?<![\w$"]){re.escape(pol)}(?![\w])', s):
                offending.append((pol, s))
    assert not offending, (
        f"{script.name} carries a literal cell list, so it is a second copy that "
        f"can drift from the tested module: {offending}")


@pytest.mark.parametrize("script", [WHOLE_NODE, ARRAY], ids=lambda p: p.name)
def test_no_launcher_hardcodes_the_step_budget_or_the_seeds(script):
    """`STEPS` and `SEEDS` were edited in the whole-node script once with no test
    noticing. They now come from the module, so a literal here is a regression."""
    for line in _text(script).splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "beta_exponent_cells" in s:
            continue
        assert not re.search(r"STEPS\s*=\s*\$?\{?STEPS:-\s*\d", s), s
        assert not re.search(r"SEEDS\s*=\s*\$?\{?SEEDS:-\s*\d", s), s


def test_both_launchers_are_valid_bash():
    for script in (WHOLE_NODE, ARRAY):
        r = subprocess.run(["bash", "-n", str(script)], capture_output=True,
                           text=True)
        assert r.returncode == 0, f"{script.name}: {r.stderr}"


def test_the_array_launcher_refuses_to_run_without_a_prepared_run_root():
    """Defaulting RUN_ROOT would let a task write into a root nobody prepared,
    against model YAMLs that do not exist or describe another arm."""
    t = _text(ARRAY)
    assert 'if [ -z "${RUN_ROOT:-}" ]' in t
    assert "exit 1" in t.split('if [ -z "${RUN_ROOT:-}" ]')[1][:400]


def test_the_array_launcher_resolves_its_cell_by_task_index():
    t = _text(ARRAY)
    assert "task_for_index" in t, (
        "the array task must resolve its own cell from the manifest, so an index "
        "off the end fails loudly instead of running the wrong cell")
    assert "preflight_failures" in t, (
        "each task must re-run the gates for its own cell; the preparer's "
        "check does not travel with the task")


# ---------------------------------------------------------------------------
# The preparer, end to end.
# ---------------------------------------------------------------------------

def _prepare(tmp_path, extra=()):
    env = dict(os.environ, PYTHONPATH=str(REPO))
    return subprocess.run(
        [sys.executable, str(PREPARE), "--run_root", str(tmp_path), *extra],
        capture_output=True, text=True, env=env, timeout=600)


def test_the_preparer_passes_and_writes_one_yaml_per_cell(tmp_path):
    from experimentation.experiments.beta_exponent_cells import CELLS, task_count
    r = _prepare(tmp_path)
    assert r.returncode == 0, f"{r.stdout}\n{r.stderr}"
    assert "all gates passed" in r.stdout
    written = {p.name for p in tmp_path.glob("model-*.yaml")}
    assert written == {f"model-{c.name}.yaml" for c in CELLS}
    assert f"ARRAY_WIDTH=0-{task_count() - 1}" in r.stdout


def test_the_preparer_writes_a_manifest_matching_the_module(tmp_path):
    from experimentation.experiments.beta_exponent_cells import manifest
    r = _prepare(tmp_path)
    assert r.returncode == 0, r.stderr
    payload = json.loads((tmp_path / "manifest.json").read_text())
    expected = manifest()
    assert len(payload["tasks"]) == len(expected)
    for got, want in zip(payload["tasks"], expected):
        assert got["index"] == want.index
        assert got["cell"] == want.cell.name
        assert got["seed"] == want.seed
        assert got["run_name"] == want.run_name


def test_the_printed_array_width_matches_the_manifest_length(tmp_path):
    """A width that disagrees with the cell list submits tasks that either never
    run a cell or run none. `task_for_index` raises on the latter, but the width
    should simply be right."""
    r = _prepare(tmp_path)
    payload = json.loads((tmp_path / "manifest.json").read_text())
    m = re.search(r"ARRAY_WIDTH=0-(\d+)", r.stdout)
    assert m, r.stdout
    assert int(m.group(1)) == len(payload["tasks"]) - 1


def test_the_preparer_dry_run_writes_nothing(tmp_path):
    r = _prepare(tmp_path, ["--dry_run"])
    assert r.returncode == 0, r.stderr
    assert list(tmp_path.glob("*")) == []


def test_the_preparer_refuses_a_chunked_proxy(tmp_path):
    """The gate the task brief makes mandatory, exercised through the CLI a human
    actually types rather than through the function."""
    import yaml
    spec = yaml.safe_load((REPO / "configs/runs/proxy-256x17.yaml").read_text())
    spec["model"]["ska_inverse_cholesky"] = False
    spec["model"]["ska_prefix_scan"] = False
    spec["model"]["ska_exact_intrachunk"] = False
    bad = tmp_path / "chunked.yaml"
    bad.write_text(yaml.safe_dump(spec))
    r = _prepare(tmp_path / "out", ["--proxy", str(bad)])
    assert r.returncode != 0, "a CHUNKED proxy was accepted"
    assert "CHUNKED" in r.stderr and "REFUSED" in r.stderr
    assert not (tmp_path / "out").exists() or not list(
        (tmp_path / "out").glob("model-*.yaml")), (
        "a refused arm must not leave model YAMLs behind for a task to find")
