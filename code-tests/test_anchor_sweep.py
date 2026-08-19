"""A curated design set -> a real `cells:` sweep, launchable today.

This is the piece that makes the design work pay off before any sampler exists:
`python -m experimentation.sweep configs/sweeps/<name>.yaml` takes the generated
file and produces one materialized run per anchor, with content-hashed
identities, the dirty-tree gate, verify_shard and a single Slurm array -- none of
which needed writing.

One wrinkle worth pinning. `sweep/spec.py::build_cell_run_spec` sets
`RunSpec.name` to the *sweep's* name for every cell (spec.py:184), because a
cell's name plays no part in identity. So a design's name has nowhere to live in
a `cells:` entry and would be lost -- and "which anchor produced this run?" is
exactly the question a curated study needs to answer. The generator therefore
also emits a companion mapping from design name to run_id, computed from the same
RunSpecs the sweep will build, so the join is exact rather than positional.
"""
import json
import textwrap

import pytest
import yaml

pytestmark = pytest.mark.correctness


def _write_base_spec(tmp_path):
    """A base run spec, as configs/runs/50m-fineweb-3b.yaml is shaped."""
    shard_dir = tmp_path / "shard"
    shard_dir.mkdir(exist_ok=True)
    path = tmp_path / "base.yaml"
    path.write_text(textwrap.dedent(f"""
        name: 50m-fineweb-3b
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
          partition: batch
          account: marlowe-m000151-pm06
          qos: medium
    """))
    return path


def _write_designs(tmp_path, count=3):
    """Build the mapping and dump it, rather than hand-indenting YAML text --
    the first version of this helper produced inconsistently indented entries
    and failed in the parser instead of in the code under test."""
    path = tmp_path / "designs.yaml"
    designs = [{"name": "baseline"}]
    for i in range(1, count):
        designs.append({
            "name": f"design-{i}",
            "layer_factor": 1.0 + 0.25 * i,
            "placement": "late",
            "ridge_factor": 1.0 + i,
        })
    path.write_text(yaml.safe_dump({"designs": designs}, sort_keys=False))
    return path


# ------------------------------------------------------ designs -> cells ----

def test_designs_to_cells_emits_one_cell_per_design_in_order():
    from koopman_lm.config import build_config
    from experimentation.sweep.search.anchors import Design, designs_to_cells
    from experimentation.sweep.search.space import search_space

    cfg = build_config("50m")
    designs = [Design(name="a"), Design(name="b", layer_factor=1.5),
               Design(name="c", placement="late")]
    cells = designs_to_cells(designs, cfg, search_space(cfg, base_name="50m"),
                             base_lr=4e-4, max_steps=15000)
    assert len(cells) == 3
    assert all(isinstance(c, dict) for c in cells)


def test_every_cell_key_is_a_valid_section_dot_field():
    from koopman_lm.config import build_config
    from experimentation.sweep.search.anchors import Design, designs_to_cells
    from experimentation.sweep.search.space import search_space

    cfg = build_config("50m")
    cells = designs_to_cells([Design(name="a")], cfg,
                             search_space(cfg, base_name="50m"),
                             base_lr=4e-4, max_steps=15000)
    for key in cells[0]:
        section, _, field = key.partition(".")
        assert section in ("model", "data", "optim", "runtime") and field


def test_cells_expand_into_distinct_runspecs(tmp_path):
    """The end-to-end claim: generated cells go through the unmodified sweep
    machinery and come out as validated RunSpecs with distinct identities."""
    from koopman_lm.config import build_config
    from experimentation.sweep.search.anchors import Design, designs_to_cells
    from experimentation.sweep.search.space import search_space
    from experimentation.sweep.spec import SweepSpec, expand_cells
    from experimentation.run.spec import run_id

    cfg = build_config("50m")
    designs = [Design(name="baseline"),
               Design(name="deep-late", layer_factor=2.0, placement="late"),
               Design(name="thin", rank=8)]
    cells = designs_to_cells(designs, cfg, search_space(cfg, base_name="50m"),
                             base_lr=4e-4, max_steps=15000)
    sweep = SweepSpec(name="ska-anchors", base=str(_write_base_spec(tmp_path)),
                      cells=cells)
    expanded = expand_cells(sweep)
    assert len(expanded) == 3
    assert len({run_id(c.spec) for c in expanded}) == 3


def test_the_baseline_design_cell_reproduces_the_shipped_config(tmp_path):
    from koopman_lm.config import build_config
    from experimentation.sweep.search.anchors import Design, designs_to_cells
    from experimentation.sweep.search.space import search_space
    from experimentation.sweep.spec import SweepSpec, expand_cells

    cfg = build_config("50m")
    cells = designs_to_cells([Design(name="baseline")], cfg,
                             search_space(cfg, base_name="50m"),
                             base_lr=4e-4, max_steps=15000)
    sweep = SweepSpec(name="ska-anchors", base=str(_write_base_spec(tmp_path)),
                      cells=cells)
    spec = expand_cells(sweep)[0].spec
    assert spec.model.ska_rank == cfg.ska_rank
    assert list(spec.model.ska_layer_indices) == list(cfg.ska_layer_indices)
    assert spec.model.param_count_estimate() == cfg.param_count_estimate()


# ----------------------------------------------------------- generator ----

def _run_generator(argv):
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    spec = importlib.util.spec_from_file_location(
        "gen_anchor_sweep", root / "scripts" / "gen_anchor_sweep.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main(argv)


def test_generator_writes_a_sweep_the_sweep_loader_accepts(tmp_path):
    from experimentation.sweep.spec import expand_cells, load_sweep_spec

    out = tmp_path / "ska-anchors.yaml"
    _run_generator([
        "--designs", str(_write_designs(tmp_path, count=4)),
        "--base", str(_write_base_spec(tmp_path)),
        "--name", "ska-anchors",
        "--out", str(out),
    ])
    sweep = load_sweep_spec(out)
    assert sweep.name == "ska-anchors"
    assert sweep.cells is not None and len(sweep.cells) == 4
    assert not sweep.axes, "an anchor set is a curated list, not a rectangle"
    assert len(expand_cells(sweep)) == 4


def test_generator_writes_a_design_name_to_run_id_mapping(tmp_path):
    """Without this the study cannot answer "which anchor was that?" --
    build_cell_run_spec gives every cell the sweep's name."""
    out = tmp_path / "ska-anchors.yaml"
    _run_generator([
        "--designs", str(_write_designs(tmp_path, count=3)),
        "--base", str(_write_base_spec(tmp_path)),
        "--name", "ska-anchors",
        "--out", str(out),
    ])
    mapping = json.loads((tmp_path / "ska-anchors.anchors.json").read_text())
    assert [a["design"] for a in mapping["anchors"]] == ["baseline", "design-1", "design-2"]
    for anchor in mapping["anchors"]:
        assert len(anchor["run_id"]) == 8
        assert len(anchor["group_id"]) == 8
    assert mapping["sweep_name"] == "ska-anchors"


def test_generated_run_ids_match_what_the_sweep_will_actually_build(tmp_path):
    """A mapping computed from a different code path than the launcher uses
    would be plausible and wrong. Compute it from the same RunSpecs."""
    from experimentation.run.spec import run_id
    from experimentation.sweep.spec import expand_cells, load_sweep_spec

    out = tmp_path / "ska-anchors.yaml"
    _run_generator([
        "--designs", str(_write_designs(tmp_path, count=3)),
        "--base", str(_write_base_spec(tmp_path)),
        "--name", "ska-anchors",
        "--out", str(out),
    ])
    mapping = json.loads((tmp_path / "ska-anchors.anchors.json").read_text())
    expected = [run_id(c.spec) for c in expand_cells(load_sweep_spec(out))]
    assert [a["run_id"] for a in mapping["anchors"]] == expected


def test_generator_enforces_a_minimum_design_count(tmp_path):
    out = tmp_path / "ska-anchors.yaml"
    with pytest.raises(ValueError, match="at least 15"):
        _run_generator([
            "--designs", str(_write_designs(tmp_path, count=3)),
            "--base", str(_write_base_spec(tmp_path)),
            "--name", "ska-anchors",
            "--out", str(out),
            "--minimum", "15",
        ])
    assert not out.exists(), "a rejected generation must leave nothing behind"


def test_generator_records_its_provenance_in_the_header(tmp_path):
    """The file is generated, and a reader who does not know that will hand-edit
    it and lose the edit on the next regeneration."""
    out = tmp_path / "ska-anchors.yaml"
    _run_generator([
        "--designs", str(_write_designs(tmp_path, count=3)),
        "--base", str(_write_base_spec(tmp_path)),
        "--name", "ska-anchors",
        "--out", str(out),
    ])
    header = out.read_text().split("\n\n")[0]
    assert "gen_anchor_sweep" in header
    assert "designs.yaml" in header


def test_generator_passes_through_max_concurrent(tmp_path):
    from experimentation.sweep.spec import load_sweep_spec

    out = tmp_path / "ska-anchors.yaml"
    _run_generator([
        "--designs", str(_write_designs(tmp_path, count=3)),
        "--base", str(_write_base_spec(tmp_path)),
        "--name", "ska-anchors",
        "--out", str(out),
        "--max-concurrent", "8",
    ])
    assert load_sweep_spec(out).max_concurrent == 8
