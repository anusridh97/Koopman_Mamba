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


def _fake_run(stdout):
    def _inner(*args, **kwargs):
        class _Result:
            pass
        r = _Result()
        r.stdout = stdout
        r.returncode = 0
        return r
    return _inner


def test_git_dirty_paths_parses_porcelain_output(monkeypatch):
    from koopman_lm.run.resolve import git_dirty_paths

    monkeypatch.setattr(
        "koopman_lm.run.resolve.subprocess.run",
        _fake_run(" M koopman_lm/run/resolve.py\n?? scratch/junk.txt\n"))
    paths = git_dirty_paths()
    assert len(paths) == 2
    assert any("resolve.py" in p for p in paths)
    assert any("junk.txt" in p for p in paths)


def test_git_dirty_paths_empty_when_clean(monkeypatch):
    from koopman_lm.run.resolve import git_dirty_paths

    monkeypatch.setattr("koopman_lm.run.resolve.subprocess.run", _fake_run(""))
    assert git_dirty_paths() == []


def test_check_git_clean_raises_and_lists_paths_when_dirty(monkeypatch):
    from koopman_lm.run.resolve import DirtyTreeError, check_git_clean

    monkeypatch.setattr(
        "koopman_lm.run.resolve.subprocess.run",
        _fake_run(" M koopman_lm/run/resolve.py\n"))
    with pytest.raises(DirtyTreeError, match="resolve.py"):
        check_git_clean(allow_dirty=False)


def test_check_git_clean_allow_dirty_warns_and_returns_true(monkeypatch, capsys):
    from koopman_lm.run.resolve import check_git_clean

    monkeypatch.setattr(
        "koopman_lm.run.resolve.subprocess.run",
        _fake_run(" M koopman_lm/run/resolve.py\n"))
    dirty = check_git_clean(allow_dirty=True)
    assert dirty is True
    out = capsys.readouterr().out
    assert "WARNING" in out
    assert "resolve.py" in out


def test_check_git_clean_returns_false_when_clean(monkeypatch):
    from koopman_lm.run.resolve import check_git_clean

    monkeypatch.setattr("koopman_lm.run.resolve.subprocess.run", _fake_run(""))
    assert check_git_clean(allow_dirty=False) is False


def test_load_materialized_spec_rejects_missing_model_field(tmp_path):
    """A spec.yaml written before a field was added must not silently receive
    the new dataclass default on load -- that would make an old run appear
    to have declared a value it never had. Missing keys must raise, listing
    what's missing."""
    import dataclasses

    from koopman_lm.config import KoopmanLMConfig, build_config
    from koopman_lm.run.resolve import load_materialized_spec, to_flat_dict
    from koopman_lm.run.spec import OptimSpec, RuntimeSpec, RunSpec, SyntheticDataSpec

    spec = RunSpec(
        name="x", model=build_config("50m"),
        data=SyntheticDataSpec(generator="mqar", params={}),
        optim=OptimSpec(lr=4e-4, warmup_steps=10, max_steps=100),
        runtime=RuntimeSpec(),
    )
    flat = to_flat_dict(spec)
    removed_field = next(iter(sorted(flat["model"])))
    del flat["model"][removed_field]

    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(yaml.safe_dump(flat, sort_keys=False))

    with pytest.raises(ValueError, match="does not match this code's config schema") as excinfo:
        load_materialized_spec(spec_path)
    assert removed_field in str(excinfo.value)
    assert "missing" in str(excinfo.value)


def test_load_materialized_spec_rejects_unknown_model_field(tmp_path):
    """A spec.yaml key absent from the current KoopmanLMConfig dataclass
    (e.g. a field that was renamed or removed) must raise rather than being
    silently ignored or crashing with an opaque TypeError."""
    from koopman_lm.config import build_config
    from koopman_lm.run.resolve import load_materialized_spec, to_flat_dict
    from koopman_lm.run.spec import OptimSpec, RuntimeSpec, RunSpec, SyntheticDataSpec

    spec = RunSpec(
        name="x", model=build_config("50m"),
        data=SyntheticDataSpec(generator="mqar", params={}),
        optim=OptimSpec(lr=4e-4, warmup_steps=10, max_steps=100),
        runtime=RuntimeSpec(),
    )
    flat = to_flat_dict(spec)
    flat["model"]["some_field_that_was_removed_long_ago"] = 123

    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(yaml.safe_dump(flat, sort_keys=False))

    with pytest.raises(ValueError, match="does not match this code's config schema") as excinfo:
        load_materialized_spec(spec_path)
    assert "some_field_that_was_removed_long_ago" in str(excinfo.value)
    assert "unknown" in str(excinfo.value)


def test_load_materialized_spec_accepts_exact_key_set(tmp_path):
    from koopman_lm.config import build_config
    from koopman_lm.run.resolve import load_materialized_spec, materialize
    from koopman_lm.run.spec import OptimSpec, RuntimeSpec, RunSpec, SyntheticDataSpec

    spec = RunSpec(
        name="x", model=build_config("50m"),
        data=SyntheticDataSpec(generator="mqar", params={}),
        optim=OptimSpec(lr=4e-4, warmup_steps=10, max_steps=100),
        runtime=RuntimeSpec(),
    )
    out_path = materialize(spec, tmp_path)
    reloaded = load_materialized_spec(out_path)
    assert reloaded.model == spec.model
