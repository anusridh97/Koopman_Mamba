"""configs/sweeps/<name>.yaml (§4.2): the sweep grid must be declared exactly
once. `axes:` is a cartesian product (the default, and the right shape for a
rectangular hyperparameter grid); `cells:` is an explicit list of override
dicts for designs that are not a rectangle. Declaring both, or neither, is a
schema error -- there must be exactly one place the grid lives.
"""
import pytest

pytestmark = pytest.mark.correctness


def test_axes_only_is_valid():
    from koopman_lm.sweep.spec import SweepSpec

    s = SweepSpec(name="x", base="configs/runs/50m-fineweb-3b.yaml",
                  axes={"optim.lr": [1e-4, 2e-4]})
    assert s.axes == {"optim.lr": [1e-4, 2e-4]}
    assert s.cells is None


def test_cells_only_is_valid():
    from koopman_lm.sweep.spec import SweepSpec

    s = SweepSpec(name="x", base="b.yaml",
                  cells=[{"optim.lr": 1e-4}, {"optim.lr": 2e-4}])
    assert s.cells == [{"optim.lr": 1e-4}, {"optim.lr": 2e-4}]
    assert s.axes == {}


def test_axes_and_cells_together_is_rejected():
    from koopman_lm.sweep.spec import SweepSpec

    with pytest.raises(ValueError, match="not both"):
        SweepSpec(name="x", base="b.yaml", axes={"optim.lr": [1e-4]},
                  cells=[{"optim.lr": 1e-4}])


def test_neither_axes_nor_cells_is_rejected():
    from koopman_lm.sweep.spec import SweepSpec

    with pytest.raises(ValueError, match="axes"):
        SweepSpec(name="x", base="b.yaml")


def test_missing_name_or_base_is_rejected():
    from koopman_lm.sweep.spec import SweepSpec

    with pytest.raises(ValueError):
        SweepSpec(name="", base="b.yaml", axes={"optim.lr": [1e-4]})
    with pytest.raises(ValueError):
        SweepSpec(name="x", base="", axes={"optim.lr": [1e-4]})


def test_max_concurrent_must_be_positive():
    from koopman_lm.sweep.spec import SweepSpec

    with pytest.raises(ValueError):
        SweepSpec(name="x", base="b.yaml", axes={"optim.lr": [1e-4]},
                  max_concurrent=0)
    SweepSpec(name="x", base="b.yaml", axes={"optim.lr": [1e-4]},
              max_concurrent=8)   # does not raise


def test_load_sweep_spec_reads_yaml(tmp_path):
    from koopman_lm.sweep.spec import load_sweep_spec

    path = tmp_path / "s.yaml"
    path.write_text(
        "name: ska-rank-lr\n"
        "base: configs/runs/50m-fineweb-3b.yaml\n"
        "max_concurrent: 8\n"
        "axes:\n"
        "  model.ska_rank: [16, 24, 32]\n"
        "  optim.lr: [2.0e-4, 4.0e-4]\n"
        "  runtime.seed: [42, 43]\n"
        "exclude:\n"
        "  - model.ska_rank: 16\n"
        "    optim.lr: 4.0e-4\n"
    )
    sweep = load_sweep_spec(path)
    assert sweep.name == "ska-rank-lr"
    assert sweep.base == "configs/runs/50m-fineweb-3b.yaml"
    assert sweep.max_concurrent == 8
    assert sweep.axes["model.ska_rank"] == [16, 24, 32]
    assert sweep.exclude == [{"model.ska_rank": 16, "optim.lr": 4.0e-4}]


def test_load_sweep_spec_defaults_exclude_to_empty_list(tmp_path):
    from koopman_lm.sweep.spec import load_sweep_spec

    path = tmp_path / "s.yaml"
    path.write_text("name: x\nbase: b.yaml\naxes:\n  optim.lr: [1.0e-4]\n")
    sweep = load_sweep_spec(path)
    assert sweep.exclude == []
    assert sweep.max_concurrent is None


def test_sweep_id_is_stable_and_content_sensitive():
    from koopman_lm.sweep.spec import SweepSpec, sweep_id

    a = SweepSpec(name="x", base="b.yaml", axes={"optim.lr": [1e-4, 2e-4]})
    b = SweepSpec(name="x", base="b.yaml", axes={"optim.lr": [1e-4, 2e-4]})
    c = SweepSpec(name="x", base="b.yaml", axes={"optim.lr": [1e-4, 3e-4]})
    d = SweepSpec(name="y", base="b.yaml", axes={"optim.lr": [1e-4, 2e-4]})
    assert sweep_id(a) == sweep_id(b)
    assert sweep_id(a) != sweep_id(c)
    assert sweep_id(a) != sweep_id(d)
