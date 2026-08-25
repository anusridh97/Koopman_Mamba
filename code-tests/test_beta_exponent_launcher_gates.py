"""The exponent arm's pre-flight gates, run as code rather than trusted.

`scripts/run_beta_exponent_mqar.sbatch` embeds a python block that refuses to
launch a cell whose resolved config is wrong. The task brief requires that to be
"a gate, not a comment", and a gate nobody executes on the CPU side is a comment
with an `exit 1` in it: its only consumer is Slurm, so a mistake in it surfaces
after the queue wait, on the job whose runs are the experiment.

So this file EXTRACTS that block from the sbatch and runs it, once with the real
cell list (must pass) and once per mutation that should make it fail. The
mutations are the point -- a gate that passes on good input proves nothing about
whether it would catch bad input.

Extraction rather than a second copy of the checks, deliberately: two copies of a
gate are two chances for the copy that runs on the login node to disagree with
the copy that runs on the GPU, and the one that matters is the one in the sbatch.
If the marker this file greps for ever moves, the test errors loudly rather than
silently checking nothing.

## The four gates, and what each one is protecting

  1. **Route.** `mqar_finetune`'s default `--model_size 50m` is prefix_scan and
     table2's default `1m` is CHUNKED. On a chunked route `beta_proj.bias`
     gradient cosine is ~0.00, so a beta comparison there measures a gate that
     receives no usable gradient -- it would return numbers, and they would mean
     nothing.
  2. **gamma == 1, K == 1.** gamma > 1 could rescue a weak configuration by
     amplification, which confounds function with scale.
  3. **Ridge.** The ridge-matched control is the arm's mandatory confound
     control. A control that silently ran at the base ridge would be a duplicate
     of its own treatment, and the arm would report "ridge does not explain it"
     having never varied ridge.
  4. **The cells exist.** A checkout predating the exponent decomposition would
     run a 5-cell arm and the log would call it 7.
"""
from __future__ import annotations

import os
import pathlib
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
SBATCH = REPO / "scripts" / "run_beta_exponent_mqar.sbatch"

pytestmark = pytest.mark.correctness

#: The line the gate block starts after. Kept as a constant so a rename gives a
#: clear failure here instead of a silently empty extraction.
_MARKER = "python - <<'PY' || { echo \"FATAL: pre-flight gate FAILED"

#: The arm's real cells, in the launcher's `name:policy:ridge` encoding.
CELLS = "\n".join([
    "learned:learned:0.01",
    "one:one:0.01",
    "linear:linear:0.01",
    "key_linear_value_sqrt:key_linear_value_sqrt:0.01",
    "key_sqrt_value_linear:key_sqrt_value_linear:0.01",
    "learned-ridge2x:learned:0.02",
    "linear-ridge0.5x:linear:0.005",
])


def _gate_source() -> str:
    text = SBATCH.read_text()
    assert _MARKER in text, (
        f"the gate block's marker moved; this test would otherwise extract "
        f"nothing and pass. Looked for {_MARKER!r} in {SBATCH}")
    body = text[text.index("\n", text.index(_MARKER)) + 1:]
    end = body.index("\nPY\n")
    src = body[:end]
    assert "SystemExit(1)" in src, "the extracted block does not exit nonzero"
    return src


def _run_gate(src: str, cells: str, tmp_path) -> subprocess.CompletedProcess:
    script = tmp_path / "gate.py"
    script.write_text(src)
    env = dict(os.environ)
    env.update(REPO_ROOT=str(REPO), RUN_ROOT=str(tmp_path),
               CELL_SPEC=cells, PYTHONPATH=str(REPO))
    return subprocess.run([sys.executable, str(script)], env=env,
                          capture_output=True, text=True, timeout=600)


# ---------------------------------------------------------------------------
# The gate passes on the arm it is going to launch.
# ---------------------------------------------------------------------------

def test_the_real_cell_list_passes_every_gate(tmp_path):
    r = _run_gate(_gate_source(), CELLS, tmp_path)
    assert r.returncode == 0, f"gate rejected the real arm:\n{r.stdout}\n{r.stderr}"
    assert "all gates passed: 7 cells, 7 distinct configs" in r.stdout


def test_every_cell_resolves_to_an_exact_route_and_gamma_one(tmp_path):
    """Read off the gate's own report, so the arm's route/gamma/K claim is
    checked against the resolved config rather than against this file's belief
    about `configs/runs/proxy-256x17.yaml`."""
    r = _run_gate(_gate_source(), CELLS, tmp_path)
    lines = [ln for ln in r.stdout.splitlines() if "route=" in ln]
    assert len(lines) == 7, r.stdout
    for ln in lines:
        assert "route=CHUNKED" not in ln, ln
        assert "gamma=1.0" in ln, ln
        assert "K=1" in ln, ln


def test_the_gate_writes_a_round_tripping_model_yaml_per_cell(tmp_path):
    """Each cell trains from a generated flat model YAML. A malformed extraction
    has to fail on the login node, not after a GPU is claimed."""
    r = _run_gate(_gate_source(), CELLS, tmp_path)
    assert r.returncode == 0, r.stderr
    written = sorted(p.name for p in tmp_path.glob("model-*.yaml"))
    assert written == sorted([
        "model-key_linear_value_sqrt.yaml", "model-key_sqrt_value_linear.yaml",
        "model-learned-ridge2x.yaml", "model-learned.yaml",
        "model-linear-ridge0.5x.yaml", "model-linear.yaml", "model-one.yaml"])


def test_the_ridge_controls_differ_from_their_own_treatment(tmp_path):
    """The control's whole purpose. If `learned-ridge2x` hashed the same as
    `learned`, the arm would report "ridge does not explain it" having never
    varied ridge."""
    r = _run_gate(_gate_source(), CELLS, tmp_path)
    ridges = {}
    for ln in r.stdout.splitlines():
        if "ridge=" not in ln:
            continue
        name = ln.split()[0]
        ridges[name] = ln.split("ridge=")[1].split()[0]
    assert ridges["learned"] == "0.01"
    assert ridges["learned-ridge2x"] == "0.02"
    assert ridges["linear-ridge0.5x"] == "0.005"


# ---------------------------------------------------------------------------
# The mutations. A gate that only passes proves nothing.
# ---------------------------------------------------------------------------

def test_the_gate_refuses_the_chunked_route(tmp_path):
    """Gate 1, the one the task brief makes mandatory."""
    src = _gate_source().replace(
        "base = spec.model",
        "base = dataclasses.replace(spec.model, ska_inverse_cholesky=False, "
        "ska_prefix_scan=False, ska_exact_intrachunk=False)")
    r = _run_gate(src, "learned:learned:0.01", tmp_path)
    assert r.returncode != 0, "a CHUNKED route was allowed to launch"
    assert "CHUNKED" in r.stderr and "REFUSED" in r.stderr


def test_the_gate_refuses_power_K_other_than_one(tmp_path):
    src = _gate_source().replace(
        "base = spec.model",
        "base = dataclasses.replace(spec.model, ska_power_K=2)")
    r = _run_gate(src, "learned:learned:0.01", tmp_path)
    assert r.returncode != 0
    assert "power_K is 2" in r.stderr


def test_the_gate_refuses_gamma_other_than_one(tmp_path):
    src = _gate_source().replace(
        "base = spec.model",
        "base = dataclasses.replace(spec.model, ska_gamma_value=1.5)")
    r = _run_gate(src, "learned:learned:0.01", tmp_path)
    assert r.returncode != 0
    assert "gamma is 1.5" in r.stderr


def test_the_gate_refuses_an_unknown_policy(tmp_path):
    r = _run_gate(_gate_source(), "typo:sqrt_beta:0.01", tmp_path)
    assert r.returncode != 0
    assert "not a BETA_POLICIES member" in r.stderr


def test_the_gate_refuses_two_cells_that_are_the_same_experiment(tmp_path):
    """Two cells with one config_hash would claim one run directory, hence one
    set of checkpoints, and the second would overwrite the first."""
    r = _run_gate(_gate_source(),
                  "a:learned:0.01\nb:learned:0.01", tmp_path)
    assert r.returncode != 0
    assert "share config_hash" in r.stderr


def test_the_gate_refuses_a_checkout_without_the_new_cells(tmp_path):
    """Gate 4. Simulated by narrowing BETA_POLICIES, which is exactly what an
    older checkout would present."""
    src = _gate_source().replace(
        "from koopman_lm.config import BETA_POLICIES, config_hash, load_config",
        "from koopman_lm.config import config_hash, load_config\n"
        "BETA_POLICIES = frozenset({'learned', 'one', 'head_scalar', 'linear'})")
    r = _run_gate(src, "learned:learned:0.01", tmp_path)
    assert r.returncode != 0
    assert "predates the exponent decomposition" in r.stderr
