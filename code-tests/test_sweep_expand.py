"""Sweep expansion (§4.2/§3.3): each cell becomes a full RunSpec through the
exact same run_id/group_id/run_dir_path machinery a lone
`python -m koopman_lm.run` launch uses -- sweep membership must not change
how identity is computed. Seed is the subtle axis: sweeping runtime.seed
must produce distinct run_ids that share one group_id.
"""
import textwrap

import pytest

pytestmark = pytest.mark.correctness


def _write_base(tmp_path, name="50m-fineweb-3b"):
    shard_dir = tmp_path / "shard"
    shard_dir.mkdir()
    base_path = tmp_path / "base.yaml"
    base_path.write_text(textwrap.dedent(f"""
        name: {name}
        model: 50m
        data:
          kind: shard
          shard_dir: {shard_dir}
          tokenizer: NousResearch/Llama-2-7b-hf
          mix: {{fineweb: 1.0}}
          n_tokens: 3000000000
        optim:
          lr: 4.0e-4
          warmup_steps: 300
          max_steps: 15000
          effective_batch: 96
          per_device_batch_size: 16
        runtime:
          seed: 42
    """))
    return base_path


def test_axes_expand_to_the_cartesian_product(tmp_path):
    from koopman_lm.sweep.spec import SweepSpec, expand_cells

    base = _write_base(tmp_path)
    sweep = SweepSpec(name="ska-rank-lr", base=str(base),
                       axes={"model.ska_rank": [16, 24],
                             "optim.lr": [1e-4, 2e-4]})
    cells = expand_cells(sweep)
    assert len(cells) == 4
    ranks_lrs = {(c.spec.model.ska_rank, c.spec.optim.lr) for c in cells}
    assert ranks_lrs == {(16, 1e-4), (16, 2e-4), (24, 1e-4), (24, 2e-4)}


def test_exclude_drops_matching_cells(tmp_path):
    from koopman_lm.sweep.spec import SweepSpec, expand_cells

    base = _write_base(tmp_path)
    sweep = SweepSpec(name="x", base=str(base),
                       axes={"model.ska_rank": [16, 24],
                             "optim.lr": [1e-4, 2e-4]},
                       exclude=[{"model.ska_rank": 16, "optim.lr": 2e-4}])
    cells = expand_cells(sweep)
    ranks_lrs = {(c.spec.model.ska_rank, c.spec.optim.lr) for c in cells}
    assert (16, 2e-4) not in ranks_lrs
    assert len(cells) == 3


def test_exclude_matches_across_multiple_predicates_as_or(tmp_path):
    from koopman_lm.sweep.spec import SweepSpec, expand_cells

    base = _write_base(tmp_path)
    sweep = SweepSpec(name="x", base=str(base),
                       axes={"model.ska_rank": [16, 24, 32]},
                       exclude=[{"model.ska_rank": 16}, {"model.ska_rank": 32}])
    cells = expand_cells(sweep)
    assert {c.spec.model.ska_rank for c in cells} == {24}


def test_explicit_cells_list_is_not_a_cartesian_product(tmp_path):
    from koopman_lm.sweep.spec import SweepSpec, expand_cells

    base = _write_base(tmp_path)
    sweep = SweepSpec(name="x", base=str(base), cells=[
        {"model.ska_rank": 16, "optim.lr": 1e-4},
        {"model.ska_rank": 32, "optim.lr": 2e-4},
    ])
    cells = expand_cells(sweep)
    ranks_lrs = {(c.spec.model.ska_rank, c.spec.optim.lr) for c in cells}
    assert ranks_lrs == {(16, 1e-4), (32, 2e-4)}   # not the full 2x2 product
    assert len(cells) == 2


def test_seed_axis_gives_distinct_run_ids_sharing_one_group_id(tmp_path):
    """The subtle case (§3.3): a different seed is a different datapoint
    (distinct run_id) but three seeds are one experiment (shared group_id)."""
    from koopman_lm.run.spec import group_id, run_id
    from koopman_lm.sweep.spec import SweepSpec, expand_cells

    base = _write_base(tmp_path)
    sweep = SweepSpec(name="x", base=str(base), axes={"runtime.seed": [42, 43, 44]})
    cells = expand_cells(sweep)
    assert len(cells) == 3
    run_ids = {run_id(c.spec) for c in cells}
    group_ids = {group_id(c.spec) for c in cells}
    assert len(run_ids) == 3     # three distinct datapoints
    assert len(group_ids) == 1   # one shared experiment


def test_model_axis_overrides_registry_named_base_model(tmp_path):
    """The base spec's model: field can be a bare registry name (e.g. '50m'),
    not an inline dict -- model.<field> overrides must still apply."""
    from koopman_lm.sweep.spec import SweepSpec, expand_cells

    base = _write_base(tmp_path)   # model: 50m (a registry name, not a dict)
    sweep = SweepSpec(name="x", base=str(base), axes={"model.ska_rank": [16]})
    cells = expand_cells(sweep)
    assert len(cells) == 1
    assert cells[0].spec.model.ska_rank == 16
    assert cells[0].spec.model.d_model == 384   # everything else still from '50m'


def test_run_dir_path_groups_seeds_under_one_prefix_with_distinct_suffixes(tmp_path):
    from koopman_lm.run.spec import run_dir_path
    from koopman_lm.sweep.spec import SweepSpec, expand_cells

    base = _write_base(tmp_path)
    sweep = SweepSpec(name="ska-rank-lr", base=str(base),
                       axes={"runtime.seed": [42, 43]})
    cells = expand_cells(sweep)
    dirs = [run_dir_path("/runs", c.spec) for c in cells]
    assert dirs[0].parent == dirs[1].parent   # same <name>.<group_id> dir
    assert dirs[0].name != dirs[1].name       # different seed<seed>.<run_id>


def test_zero_cells_after_exclude_is_allowed_by_expand_cells(tmp_path):
    """expand_cells itself doesn't police "did exclude eat the whole grid" --
    that belongs to the CLI (a clearer error where a human is looking at
    output), not the pure expansion function."""
    from koopman_lm.sweep.spec import SweepSpec, expand_cells

    base = _write_base(tmp_path)
    sweep = SweepSpec(name="x", base=str(base), axes={"model.ska_rank": [16]},
                       exclude=[{"model.ska_rank": 16}])
    assert expand_cells(sweep) == []


def test_unknown_axis_section_is_rejected(tmp_path):
    from koopman_lm.sweep.spec import SweepSpec, expand_cells

    base = _write_base(tmp_path)
    sweep = SweepSpec(name="x", base=str(base), axes={"bogus.field": [1]})
    with pytest.raises(ValueError, match="bogus"):
        expand_cells(sweep)


def test_axis_key_without_a_dot_is_rejected(tmp_path):
    from koopman_lm.sweep.spec import SweepSpec, expand_cells

    base = _write_base(tmp_path)
    sweep = SweepSpec(name="x", base=str(base), axes={"lr": [1e-4]})
    with pytest.raises(ValueError):
        expand_cells(sweep)
