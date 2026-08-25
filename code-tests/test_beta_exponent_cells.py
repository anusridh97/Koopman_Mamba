"""The exponent arm's cell list, its gates, and the array-index mapping.

## Why this is a module and not an embedded shell block

The arm was first written as one whole-node sbatch with the gates inline. On a
GPU-saturated cluster that is the wrong shape: every node reports 8/8 GRES
allocated, so a whole-node request queued ~12 hours out while a 1-GPU request
BACKFILLED in ten minutes (job 446640, observed). A slurm array of 1-GPU tasks
backfills; one 8-GPU job waits.

That splits the work across N processes that never see each other, which changes
what the gates have to be. An inline block checked the cells once, in the job
that then ran them. An array task must re-check ITS OWN cell, from the manifest,
because nothing guarantees it is running the arm that was prepared -- a stale
RUN_ROOT, a re-submitted array against an edited cell list, or a task index off
the end of the manifest all produce a silently wrong run rather than an error.

So the gate logic lives here, is imported by the preparer and by each task, and
is tested. One copy: the login-node check cannot drift from the GPU one.

## What each gate is protecting

  * **Route.** On a CHUNKED route `beta_proj.bias` gradient cosine is ~0.00, so a
    beta comparison there measures a gate receiving no usable gradient. It would
    return numbers and they would mean nothing.
  * **gamma == 1, K == 1.** gamma > 1 could rescue a weak cell by amplification,
    confounding function with scale.
  * **Ridge.** The ridge-matched control is the arm's mandatory confound control.
    A control that silently ran at the base ridge would be a duplicate of its own
    treatment, and the arm would report "ridge does not explain it" having never
    varied ridge.
  * **Distinct identity.** Two cells sharing a config_hash would claim one run
    directory and the second would overwrite the first.
  * **The cells exist.** A checkout predating the exponent decomposition would run
    a 5-cell arm and the log would call it 7.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from experimentation.experiments.beta_exponent_cells import (
    CELLS, SEEDS, Cell, manifest, preflight_failures, resolve_cell,
    task_count, write_model_yaml)
from koopman_lm.config import BETA_POLICIES, config_hash, load_config

pytestmark = pytest.mark.correctness

REPO = Path(__file__).resolve().parents[1]


def _base():
    from experimentation.run.resolve import resolve_run_spec
    return resolve_run_spec(REPO / "configs/runs/proxy-256x17.yaml").model


# ---------------------------------------------------------------------------
# The design, as data.
# ---------------------------------------------------------------------------

def test_the_arm_is_the_seven_cells_the_preregistration_names():
    names = [c.name for c in CELLS]
    assert names == ["learned", "one", "linear", "key_linear_value_sqrt",
                     "key_sqrt_value_linear", "learned-ridge2x",
                     "linear-ridge0.5x"]


def test_head_scalar_is_not_in_the_arm():
    """Excluded by instruction. Asserted so it cannot drift back in via a
    "complete the policy list" edit."""
    assert all(c.policy != "head_scalar" for c in CELLS)


def test_both_mixed_exponent_cells_are_present():
    policies = {c.policy for c in CELLS}
    assert "key_linear_value_sqrt" in policies
    assert "key_sqrt_value_linear" in policies


def test_the_ridge_controls_bracket_the_base_ridge_from_both_sides():
    """`learned@2x` matches `linear`'s effective-ridge ratio from below and
    `linear@0.5x` matches `learned`'s from above. One-sided would leave the
    convergence test dependent on which direction was chosen."""
    by = {c.name: c for c in CELLS}
    assert by["learned"].ridge == 0.01
    assert by["learned-ridge2x"].ridge == pytest.approx(0.02)
    assert by["learned-ridge2x"].policy == "learned"
    assert by["linear-ridge0.5x"].ridge == pytest.approx(0.005)
    assert by["linear-ridge0.5x"].policy == "linear"


def test_every_declared_policy_is_a_real_one():
    for c in CELLS:
        assert c.policy in BETA_POLICIES, c


# ---------------------------------------------------------------------------
# Resolution and the gates.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cell", CELLS, ids=lambda c: c.name)
def test_each_cell_resolves_to_the_exact_route_with_gamma_one_and_K_one(cell):
    cfg = resolve_cell(cell, _base())
    assert (cfg.ska_prefix_scan or cfg.ska_inverse_cholesky
            or cfg.ska_exact_intrachunk), f"{cell.name} resolved to CHUNKED"
    assert float(cfg.ska_gamma_value) == 1.0
    assert not cfg.ska_gamma_learnable
    assert int(cfg.ska_power_K) == 1
    assert float(cfg.ska_ridge) == pytest.approx(cell.ridge)
    assert cfg.ska_beta_policy == cell.policy


def test_the_whole_arm_passes_preflight():
    assert preflight_failures(CELLS, _base()) == []


def test_every_cell_has_a_distinct_identity():
    """Two cells sharing a config_hash would claim one run directory."""
    seen = {}
    for c in CELLS:
        h = config_hash(resolve_cell(c, _base()))
        assert h not in seen, f"{c.name} and {seen[h]} are the same experiment"
        seen[h] = c.name


def test_preflight_refuses_the_chunked_route():
    """The gate the task brief makes mandatory. Simulated by clearing every exact
    flag on the base, which is exactly what `configs/1m.yaml` presents."""
    base = dataclasses.replace(
        _base(), ska_inverse_cholesky=False, ska_prefix_scan=False,
        ska_exact_intrachunk=False)
    fails = preflight_failures(CELLS, base)
    assert fails, "a CHUNKED base was accepted"
    assert all("CHUNKED" in f for f in fails)
    assert len(fails) == len(CELLS), "every cell must be refused, not just one"


def test_preflight_refuses_gamma_above_one():
    fails = preflight_failures(
        CELLS, dataclasses.replace(_base(), ska_gamma_value=1.5))
    assert any("gamma" in f for f in fails)


def test_preflight_refuses_gamma_bounds_even_when_gamma_is_not_learnable():
    """A second, independent way to make gamma trainable.

    `modules/seq/ska.py` promotes gamma to an nn.Parameter whenever
    `gamma_bounds is not None`, REGARDLESS of `gamma_learnable`. So a config with
    bounds set and learnable false would pass a gate whose message says "the arm
    pins gamma = 1" and then train a drifting gamma -- confounding function with
    amplification, which is the exact thing gamma = 1 is pinned to prevent.
    """
    base = dataclasses.replace(
        _base(), ska_gamma_bounds=(0.5, 2.0), ska_gamma_learnable=False,
        ska_gamma_value=1.0)
    fails = preflight_failures(CELLS, base)
    assert fails, "gamma_bounds slipped past a gate that claims to pin gamma"
    assert all("gamma_bounds" in f for f in fails)


def test_preflight_refuses_power_K_above_one():
    fails = preflight_failures(
        CELLS, dataclasses.replace(_base(), ska_power_K=2))
    assert any("power_K" in f for f in fails)


def test_preflight_refuses_an_unknown_policy():
    fails = preflight_failures([Cell("bogus", "sqrt_beta", 0.01)], _base())
    assert any("BETA_POLICIES" in f for f in fails)


def test_preflight_refuses_duplicate_cells():
    dup = [Cell("a", "learned", 0.01), Cell("b", "learned", 0.01)]
    fails = preflight_failures(dup, _base())
    assert any("config_hash" in f for f in fails)


def test_preflight_refuses_duplicate_cell_names():
    """Two cells with one NAME would write to one run directory even with
    different configs -- the name is what the run path is built from."""
    dup = [Cell("a", "learned", 0.01), Cell("a", "linear", 0.01)]
    fails = preflight_failures(dup, _base())
    assert any("name" in f for f in fails)


# ---------------------------------------------------------------------------
# The model YAML each task trains from.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cell", CELLS, ids=lambda c: c.name)
def test_the_written_model_yaml_round_trips_to_the_resolved_config(cell, tmp_path):
    """A malformed extraction has to fail on the login node, not after a GPU is
    claimed. `load_config` takes a FLAT model config and the proxy spec is a run
    spec, so something has to bridge them -- and the bridge is what round-trips."""
    cfg = resolve_cell(cell, _base())
    path = write_model_yaml(cell, cfg, tmp_path)
    assert path.exists()
    assert load_config(path) == cfg


def test_the_yaml_names_the_cell_so_a_task_cannot_load_the_wrong_one(tmp_path):
    base = _base()
    for c in CELLS:
        write_model_yaml(c, resolve_cell(c, base), tmp_path)
    written = {p.name for p in tmp_path.glob("model-*.yaml")}
    assert written == {f"model-{c.name}.yaml" for c in CELLS}


# ---------------------------------------------------------------------------
# The array index mapping. Getting this wrong silently runs the wrong cell.
# ---------------------------------------------------------------------------

def test_the_manifest_covers_every_cell_x_seed_exactly_once():
    m = manifest(CELLS, SEEDS)
    assert len(m) == len(CELLS) * len(SEEDS) == task_count()
    pairs = [(t.cell.name, t.seed) for t in m]
    assert len(set(pairs)) == len(pairs)
    assert set(pairs) == {(c.name, s) for c in CELLS for s in SEEDS}


def test_array_indices_are_contiguous_from_zero():
    """`--array=0-N` must map onto the manifest with no gap: a gap is a task that
    exits without running, which reads as a missing seed."""
    m = manifest(CELLS, SEEDS)
    assert [t.index for t in m] == list(range(len(m)))


def test_indices_are_grouped_by_seed_so_a_truncated_array_leaves_whole_seeds():
    """If the array is cut short -- cancelled, or the allocation ends -- the
    completed prefix must be N COMPLETE paired seeds across all seven cells, not
    a ragged set with some cells at six seeds and others at one. A ragged set is
    not a paired design, and pairing is what the arm's comparison rests on.
    """
    m = manifest(CELLS, SEEDS)
    n = len(CELLS)
    for k, seed in enumerate(SEEDS):
        block = m[k * n:(k + 1) * n]
        assert {t.seed for t in block} == {seed}
        assert [t.cell.name for t in block] == [c.name for c in CELLS]


def test_a_task_index_off_the_end_of_the_manifest_is_an_error():
    """An array submitted wider than the manifest must fail loudly. Silently
    exiting 0 would report as a completed seed with no data."""
    from experimentation.experiments.beta_exponent_cells import task_for_index
    m = manifest(CELLS, SEEDS)
    assert task_for_index(len(m) - 1).index == len(m) - 1
    with pytest.raises(IndexError):
        task_for_index(len(m))
    with pytest.raises(IndexError):
        task_for_index(-1)


def test_the_run_name_matches_what_the_report_script_parses():
    """`report_beta_exponent_arm.py` greps `mqar-<cell>-seed<N>`. A task that
    wrote a different name would be invisible to the report -- which reads as a
    missing run rather than as a naming bug."""
    import re
    pattern = re.compile(r"^mqar-(?P<cell>.+)-seed(?P<seed>\d+)$")
    for t in manifest(CELLS, SEEDS):
        m = pattern.match(t.run_name)
        assert m, t.run_name
        assert m.group("cell") == t.cell.name
        assert int(m.group("seed")) == t.seed
