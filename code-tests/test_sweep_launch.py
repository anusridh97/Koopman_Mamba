"""experimentation/sweep/launch.py: the materialize-one-cell primitive, shared.

It was `_materialize_cell`, private inside sweep/__main__.py. The adaptive
searcher needs exactly this sequence -- verify, claim, create, materialize,
record -- and importing it from a `__main__` module is not a thing to do, so it
moves out and loses its sweep-specific coupling: it now takes a RunSpec plus an
opaque `extra` dict rather than a SweepCell plus a SweepSpec.

That generalisation is the whole change. The sequence itself, and the order of
its steps, is byte-for-byte what run/__main__.py does for a single run, and the
existing sweep tests are what prove the move was inert.
"""
import json
import textwrap
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.correctness


def _shard_spec(tmp_path, *, seed=42):
    """A resolvable RunSpec over a real (tiny) shard meta.json."""
    from experimentation.run.resolve import resolve_run_spec

    shard_dir = tmp_path / "shard"
    shard_dir.mkdir(exist_ok=True)
    (shard_dir / "meta.json").write_text(json.dumps({
        "n_tokens": 1000, "tokenizer": "NousResearch/Llama-2-7b-hf",
        "mix": {"fineweb": 1.0},
    }))
    spec_path = tmp_path / f"spec-{seed}.yaml"
    spec_path.write_text(textwrap.dedent(f"""
        name: 50m-smoke
        model: 50m
        data:
          kind: shard
          shard_dir: {shard_dir}
          tokenizer: NousResearch/Llama-2-7b-hf
          mix: {{fineweb: 1.0}}
          n_tokens: 1000
        optim:
          lr: 0.0004
          warmup_steps: 10
          max_steps: 100
          effective_batch: 16
          per_device_batch_size: 16
        runtime:
          seed: {seed}
    """))
    return resolve_run_spec(spec_path)


def test_materialize_cell_writes_spec_and_one_attempt(tmp_path):
    from experimentation.sweep.launch import materialize_cell
    from experimentation.run.spec import run_dir_path, run_id

    spec = _shard_spec(tmp_path)
    run_dir = materialize_cell(spec, tmp_path / "runs", dry_run=True)

    assert run_dir == run_dir_path(tmp_path / "runs", spec)
    written = yaml.safe_load((run_dir / "spec.yaml").read_text())
    assert written["run_id"] == run_id(spec)
    attempts = (run_dir / "attempts.jsonl").read_text().strip().splitlines()
    assert len(attempts) == 1
    assert json.loads(attempts[0])["code_id"]


def test_materialize_cell_stamps_arbitrary_extra_keys(tmp_path):
    """`extra` is opaque on purpose. A static sweep stamps sweep_id/sweep_name;
    the searcher will stamp its own study identifiers through the same door,
    without launch.py needing to know either vocabulary."""
    from experimentation.sweep.launch import materialize_cell

    spec = _shard_spec(tmp_path)
    run_dir = materialize_cell(spec, tmp_path / "runs", dry_run=True,
                               extra={"study_name": "ska-depth", "trial_number": 7})
    written = yaml.safe_load((run_dir / "spec.yaml").read_text())
    assert written["study_name"] == "ska-depth"
    assert written["trial_number"] == 7


def test_materialize_cell_works_without_extra(tmp_path):
    from experimentation.sweep.launch import materialize_cell

    spec = _shard_spec(tmp_path)
    run_dir = materialize_cell(spec, tmp_path / "runs", dry_run=True)
    written = yaml.safe_load((run_dir / "spec.yaml").read_text())
    assert "sweep_id" not in written and "sweep_name" not in written


def test_materialize_cell_rejects_an_unlaunchable_spec_before_writing(tmp_path):
    """The gate PR #23 established: a spec with no launch path must not leave a
    run directory, a spec.yaml, or a fabricated attempts.jsonl entry behind."""
    from experimentation.run.data_verify import UnlaunchableDataSpecError
    from experimentation.run.spec import (OptimSpec, RunSpec, RuntimeSpec,
                                          SyntheticDataSpec)
    from experimentation.sweep.launch import materialize_cell
    from koopman_lm.config import build_config

    spec = RunSpec(name="mqar-smoke", model=build_config("50m"),
                   data=SyntheticDataSpec(kind="synthetic", generator="mqar"),
                   optim=OptimSpec(lr=4e-4, warmup_steps=10, max_steps=100,
                                   effective_batch=16, per_device_batch_size=16),
                   runtime=RuntimeSpec(seed=42))
    run_root = tmp_path / "runs"
    with pytest.raises(UnlaunchableDataSpecError):
        materialize_cell(spec, run_root, dry_run=True)
    assert not list(run_root.rglob("spec.yaml"))
    assert not list(run_root.rglob("attempts.jsonl"))


def test_materialize_cell_refuses_a_claimed_run_dir(tmp_path):
    from experimentation.run.spec import run_dir_path
    from experimentation.run.write_policy import CLAIM_SENTINEL, RunDirClaimedError
    from experimentation.sweep.launch import materialize_cell

    spec = _shard_spec(tmp_path)
    run_dir = run_dir_path(tmp_path / "runs", spec)
    (run_dir / CLAIM_SENTINEL).mkdir(parents=True)
    with pytest.raises(RunDirClaimedError):
        materialize_cell(spec, tmp_path / "runs", dry_run=True)


def test_materialize_cell_records_dirty_when_told_to(tmp_path):
    from experimentation.sweep.launch import materialize_cell

    spec = _shard_spec(tmp_path)
    run_dir = materialize_cell(spec, tmp_path / "runs", dirty=True, dry_run=True)
    assert yaml.safe_load((run_dir / "spec.yaml").read_text())["dirty"] is True


def test_sweep_main_no_longer_defines_its_own_materialize(tmp_path):
    """Guards the point of the move: one implementation, not two that drift."""
    import experimentation.sweep.__main__ as sweep_main

    assert not hasattr(sweep_main, "_materialize_cell"), (
        "sweep/__main__.py still defines _materialize_cell; it should call "
        "experimentation.sweep.launch.materialize_cell")
