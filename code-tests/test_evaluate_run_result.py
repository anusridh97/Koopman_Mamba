"""experimentation/evaluation/evaluate.py wiring to the result envelope (§4.2):
experimentation.evaluation.result.write_result and the <run_dir>/eval/<checkpoint>/
<task>.json layout. Before this, evaluate.py wrote raw JSON to a
caller-supplied --output with no envelope, so the first real training run
had to hand-wrap its own output.

Only the path-resolution and envelope-writing logic is exercised here --
no model is instantiated, so this runs without torch touching a GPU or
mamba_ssm being installed.
"""
import textwrap

import pytest
import yaml

from experimentation.evaluation.evaluate import find_run_dir, write_checkpoint_result
from experimentation.evaluation.result import read_result

pytestmark = pytest.mark.correctness


def _write_spec_yaml(run_dir, *, run_id="deadbeef", git_commit="traincommit"):
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "spec.yaml").write_text(textwrap.dedent(f"""
        name: 50m-fineweb-3b
        run_id: {run_id}
        group_id: gA
        model:
          d_model: 384
        data:
          kind: shard
        optim:
          lr: 0.0004
        runtime:
          seed: 42
        provenance:
          git_commit: {git_commit}
    """))


def test_find_run_dir_locates_spec_yaml_one_level_up(tmp_path):
    run_dir = tmp_path / "runs" / "50m-fineweb-3b.gA" / "seed42.deadbeef"
    _write_spec_yaml(run_dir)
    ckpt_dir = run_dir / "final"
    ckpt_dir.mkdir()
    checkpoint = ckpt_dir / "model.pt"
    checkpoint.write_text("not a real checkpoint")

    assert find_run_dir(checkpoint) == run_dir


def test_find_run_dir_locates_spec_yaml_several_levels_up(tmp_path):
    """A checkpoint can be nested deeper than one level below the run dir
    (e.g. a future checkpoints/step_5000/ layout) -- walking up must not
    stop after a single hop."""
    run_dir = tmp_path / "runs" / "50m-fineweb-3b.gA" / "seed42.deadbeef"
    _write_spec_yaml(run_dir)
    ckpt_dir = run_dir / "checkpoints" / "step_5000"
    ckpt_dir.mkdir(parents=True)
    checkpoint = ckpt_dir / "model.pt"
    checkpoint.write_text("not a real checkpoint")

    assert find_run_dir(checkpoint) == run_dir


def test_find_run_dir_returns_none_outside_any_run_directory(tmp_path):
    ckpt_dir = tmp_path / "some" / "adhoc" / "checkpoint_dir"
    ckpt_dir.mkdir(parents=True)
    checkpoint = ckpt_dir / "model.pt"
    checkpoint.write_text("not a real checkpoint")

    assert find_run_dir(checkpoint) is None


def test_write_checkpoint_result_returns_none_outside_a_run_directory(tmp_path):
    ckpt_dir = tmp_path / "some" / "adhoc" / "checkpoint_dir"
    ckpt_dir.mkdir(parents=True)
    checkpoint = ckpt_dir / "model.pt"
    checkpoint.write_text("not a real checkpoint")

    assert write_checkpoint_result(checkpoint, "ppl", {"ppl": 12.3}) is None


def test_write_checkpoint_result_writes_the_envelope_at_the_default_path(tmp_path):
    run_dir = tmp_path / "runs" / "50m-fineweb-3b.gA" / "seed42.deadbeef"
    _write_spec_yaml(run_dir, run_id="deadbeef")
    ckpt_dir = run_dir / "final"
    ckpt_dir.mkdir()
    checkpoint = ckpt_dir / "model.pt"
    checkpoint.write_text("not a real checkpoint")

    path = write_checkpoint_result(checkpoint, "ppl", {"ppl": 12.3, "loss": 2.5})

    assert path == run_dir / "eval" / "final" / "ppl.json"
    result = read_result(path)
    assert result["run_id"] == "deadbeef"
    assert result["checkpoint"] == "final"
    assert result["task"] == "ppl"
    assert result["metrics"] == {"ppl": 12.3, "loss": 2.5}


def test_write_checkpoint_result_reads_run_id_from_spec_yaml_not_recomputed(tmp_path):
    """spec.yaml's run_id is authoritative -- write_checkpoint_result must
    read it back rather than recomputing experimentation.run.spec.run_id(spec)
    from a RunSpec it never reconstructs. A deliberately "wrong" run_id
    (one that would never come out of the hash) proves it was read, not
    recomputed."""
    run_dir = tmp_path / "runs" / "50m-fineweb-3b.gA" / "seed42.notahash"
    _write_spec_yaml(run_dir, run_id="not-a-real-hash-at-all")
    ckpt_dir = run_dir / "final"
    ckpt_dir.mkdir()
    checkpoint = ckpt_dir / "model.pt"
    checkpoint.write_text("not a real checkpoint")

    path = write_checkpoint_result(checkpoint, "ppl", {"ppl": 1.0})
    assert read_result(path)["run_id"] == "not-a-real-hash-at-all"


def test_write_checkpoint_result_uses_the_current_eval_git_commit(tmp_path, monkeypatch):
    """The envelope's git_commit is eval-time code provenance (matching
    experimentation.results' `eval_git_commit` column) -- it must NOT be silently
    swapped for the training run's own provenance.git_commit that happens to
    sit right there in spec.yaml."""
    run_dir = tmp_path / "runs" / "50m-fineweb-3b.gA" / "seed42.deadbeef"
    _write_spec_yaml(run_dir, git_commit="traincommit")
    ckpt_dir = run_dir / "final"
    ckpt_dir.mkdir()
    checkpoint = ckpt_dir / "model.pt"
    checkpoint.write_text("not a real checkpoint")

    monkeypatch.setattr(
        "experimentation.evaluation.evaluate.git_commit", lambda: "evalcommit")

    path = write_checkpoint_result(checkpoint, "ppl", {"ppl": 1.0})
    assert read_result(path)["git_commit"] == "evalcommit"


def test_write_checkpoint_result_overwrites_on_rescoring(tmp_path):
    run_dir = tmp_path / "runs" / "50m-fineweb-3b.gA" / "seed42.deadbeef"
    _write_spec_yaml(run_dir)
    ckpt_dir = run_dir / "final"
    ckpt_dir.mkdir()
    checkpoint = ckpt_dir / "model.pt"
    checkpoint.write_text("not a real checkpoint")

    write_checkpoint_result(checkpoint, "ppl", {"ppl": 12.3})
    path = write_checkpoint_result(checkpoint, "ppl", {"ppl": 11.9})

    assert read_result(path)["metrics"]["ppl"] == 11.9
