"""extends: resolution and spec.yaml materialization (§3.2).

Every downstream consumer (eval, resume, analysis) reads only the
materialized spec.yaml; base configs are never re-read after launch.
"""
import textwrap

import pytest
import yaml

pytestmark = pytest.mark.correctness


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text))


def test_resolve_run_spec_follows_extends_chain(tmp_path):
    from koopman_lm.run.resolve import resolve_run_spec
    from koopman_lm.run.spec import SyntheticDataSpec

    _write(tmp_path / "base.yaml", """
        optim:
          lr: 0.0004
          warmup_steps: 100
          max_steps: 1000
          effective_batch: 64
          per_device_batch_size: 8
        runtime:
          seed: 42
    """)
    _write(tmp_path / "leaf.yaml", """
        extends: base.yaml
        name: 50m-mqar-smoke
        model: 50m
        data:
          kind: synthetic
          generator: mqar
          params:
            num_kv_pairs: 8
        runtime:
          seed: 1337
    """)

    spec = resolve_run_spec(tmp_path / "leaf.yaml")
    assert spec.name == "50m-mqar-smoke"
    assert spec.optim.lr == 0.0004          # inherited from base
    assert spec.optim.max_steps == 1000     # inherited from base
    assert spec.runtime.seed == 1337        # overridden by leaf
    assert isinstance(spec.data, SyntheticDataSpec)
    assert spec.data.generator == "mqar"
    assert spec.model.d_model == 384        # from the "50m" registry entry


def test_resolve_run_spec_rejects_pyyaml_lr_footgun(tmp_path):
    from koopman_lm.run.resolve import resolve_run_spec

    _write(tmp_path / "bad.yaml", """
        name: bad-lr
        model: 50m
        data: {kind: synthetic, generator: mqar, params: {}}
        optim: {lr: 4e-4, warmup_steps: 10, max_steps: 100}
        runtime: {}
    """)
    with pytest.raises(TypeError):
        resolve_run_spec(tmp_path / "bad.yaml")


def test_to_flat_dict_inlines_the_model_and_has_no_extends_key():
    from koopman_lm.config import build_config
    from koopman_lm.run.resolve import to_flat_dict
    from koopman_lm.run.spec import OptimSpec, RuntimeSpec, RunSpec, SyntheticDataSpec

    spec = RunSpec(
        name="x", model=build_config("50m"),
        data=SyntheticDataSpec(generator="mqar", params={}),
        optim=OptimSpec(lr=4e-4, warmup_steps=10, max_steps=100),
        runtime=RuntimeSpec(),
    )
    flat = to_flat_dict(spec)
    assert "extends" not in flat
    assert flat["model"]["d_model"] == 384
    assert flat["run_id"] and flat["group_id"]


def test_materialize_writes_spec_yaml_with_provenance(tmp_path):
    from koopman_lm.config import build_config
    from koopman_lm.run.resolve import load_materialized_spec, materialize
    from koopman_lm.run.spec import (
        OptimSpec, RuntimeSpec, RunSpec, SyntheticDataSpec, run_id,
    )

    spec = RunSpec(
        name="50m-mqar-smoke", model=build_config("50m"),
        data=SyntheticDataSpec(generator="mqar", params={"num_kv_pairs": 8}),
        optim=OptimSpec(lr=4e-4, warmup_steps=10, max_steps=100),
        runtime=RuntimeSpec(),
    )
    run_dir = tmp_path / "run"
    out_path = materialize(spec, run_dir)
    assert out_path == run_dir / "spec.yaml"

    raw = yaml.safe_load(out_path.read_text())
    assert "extends" not in raw
    assert raw["run_id"] == run_id(spec)
    assert "provenance" in raw
    assert "git_commit" in raw["provenance"]
    assert "torch_version" in raw["provenance"]

    reloaded = load_materialized_spec(out_path)
    assert reloaded.name == spec.name
    assert run_id(reloaded) == run_id(spec)


def test_materialize_writes_code_id_distinct_from_run_id(tmp_path):
    """code_id (§4.2/§3.7 provenance) records what code actually executed,
    separately from run_id (declared science). A code change must not alter
    run_id -- that would make every commit spawn a new run identity -- but it
    must be recoverable from spec.yaml so two runs that collide on run_id but
    ran different code can be told apart."""
    from koopman_lm.config import build_config
    from koopman_lm.run.resolve import git_commit, materialize
    from koopman_lm.run.spec import (
        OptimSpec, RuntimeSpec, RunSpec, SyntheticDataSpec, run_id,
    )

    spec = RunSpec(
        name="50m-mqar-smoke", model=build_config("50m"),
        data=SyntheticDataSpec(generator="mqar", params={"num_kv_pairs": 8}),
        optim=OptimSpec(lr=4e-4, warmup_steps=10, max_steps=100),
        runtime=RuntimeSpec(),
    )
    run_dir = tmp_path / "run"
    out_path = materialize(spec, run_dir)
    raw = yaml.safe_load(out_path.read_text())

    assert raw["code_id"] == git_commit()
    assert raw["code_id"] != run_id(spec)
