"""Artifact write policy (§3.7): earned bytes (checkpoints, results) refuse to
clobber; derived bytes (spec.yaml, sbatch, attempts.jsonl entries) are cheap
and use plain atomic writes.
"""
import json

import pytest

pytestmark = pytest.mark.correctness


def test_atomic_write_text_creates_parents_and_no_tmp_leftover(tmp_path):
    from experimentation.run.artifacts import atomic_write_text

    target = tmp_path / "nested" / "spec.yaml"
    atomic_write_text(target, "hello: world\n")
    assert target.read_text() == "hello: world\n"
    leftovers = list(tmp_path.rglob("*.tmp*"))
    assert leftovers == []


def test_atomic_write_json_round_trips(tmp_path):
    from experimentation.run.artifacts import atomic_write_json

    target = tmp_path / "result.json"
    atomic_write_json(target, {"a": 1, "b": [1, 2, 3]})
    assert json.loads(target.read_text()) == {"a": 1, "b": [1, 2, 3]}


def test_create_run_dir_happy_path(tmp_path):
    from experimentation.run.artifacts import create_run_dir

    run_dir = tmp_path / "50m-fineweb-3b.abcd1234" / "seed42.deadbeef"
    result = create_run_dir(run_dir)
    assert result == run_dir
    assert run_dir.is_dir()


def test_create_run_dir_refuses_to_clobber_a_finished_run(tmp_path):
    from experimentation.run.artifacts import RunDirConflictError, create_run_dir

    run_dir = tmp_path / "run"
    (run_dir / "final").mkdir(parents=True)
    with pytest.raises(RunDirConflictError):
        create_run_dir(run_dir)


def test_create_run_dir_allows_resume_or_force(tmp_path):
    from experimentation.run.artifacts import create_run_dir

    run_dir = tmp_path / "run"
    (run_dir / "final").mkdir(parents=True)
    create_run_dir(run_dir, resume=True)
    create_run_dir(run_dir, force=True)


def test_append_attempt_is_append_only(tmp_path):
    from experimentation.run.artifacts import append_attempt

    run_dir = tmp_path / "run"
    append_attempt(run_dir, {"host": "node01", "job_id": "1"})
    append_attempt(run_dir, {"host": "node01", "job_id": "2"})
    lines = (run_dir / "attempts.jsonl").read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["job_id"] == "1"
    assert json.loads(lines[1])["job_id"] == "2"


def test_make_attempt_record_shape():
    from experimentation.run.artifacts import make_attempt_record

    record = make_attempt_record(host="node01", job_id="12345",
                                   git_commit="deadbeef", forced=False)
    assert record["host"] == "node01"
    assert record["job_id"] == "12345"
    assert record["git_commit"] == "deadbeef"
    assert record["forced"] is False
    assert "timestamp" in record


def test_make_attempt_record_includes_code_id():
    """Every attempts.jsonl entry must carry code_id (the commit that actually
    executed), independently of run_id -- so an attempt against a stale
    code_id is identifiable directly from the audit trail."""
    from experimentation.run.artifacts import make_attempt_record

    record = make_attempt_record(host="node01", job_id="12345",
                                   git_commit="deadbeef", forced=False)
    assert record["code_id"] == "deadbeef"


def test_create_run_dir_reports_code_id_mismatch_explicitly(tmp_path):
    """If the target dir is finished and its spec.yaml recorded a different
    code_id, the collision error must say so explicitly -- not just report a
    generic clobber conflict -- so a code-only change that collides on run_id
    (a real bug this fix corrects) is not mistaken for relaunching the exact
    same experiment."""
    import yaml

    from experimentation.run.artifacts import RunDirConflictError, create_run_dir

    run_dir = tmp_path / "run"
    (run_dir / "final").mkdir(parents=True)
    (run_dir / "spec.yaml").write_text(yaml.safe_dump({"code_id": "aaaaaaa"}))

    with pytest.raises(RunDirConflictError, match="code_id"):
        create_run_dir(run_dir, code_id="bbbbbbb")


def test_create_run_dir_same_code_id_gives_generic_conflict(tmp_path):
    """Same code_id (a genuine identical relaunch) keeps the plain, generic
    clobber message -- the explicit code_id language is reserved for actual
    mismatches."""
    import yaml

    from experimentation.run.artifacts import RunDirConflictError, create_run_dir

    run_dir = tmp_path / "run"
    (run_dir / "final").mkdir(parents=True)
    (run_dir / "spec.yaml").write_text(yaml.safe_dump({"code_id": "aaaaaaa"}))

    with pytest.raises(RunDirConflictError) as excinfo:
        create_run_dir(run_dir, code_id="aaaaaaa")
    assert "different code_id" not in str(excinfo.value)


def test_atomic_torch_save_round_trips_and_leaves_no_tmp(tmp_path):
    import torch
    from experimentation.run.artifacts import atomic_torch_save

    target = tmp_path / "resume.pt"
    atomic_torch_save(target, {"step": 7, "tensor": torch.arange(4)})
    loaded = torch.load(target, map_location="cpu", weights_only=False)
    assert loaded["step"] == 7
    assert torch.equal(loaded["tensor"], torch.arange(4))
    leftovers = list(tmp_path.rglob("*.tmp*"))
    assert leftovers == []


def test_atomic_torch_save_failure_does_not_corrupt_existing_file(tmp_path, monkeypatch):
    import torch
    from experimentation.run.artifacts import atomic_torch_save

    target = tmp_path / "resume.pt"
    atomic_torch_save(target, {"step": 1})

    def _boom(*a, **k):
        raise RuntimeError("simulated kill mid-write")

    monkeypatch.setattr("torch.save", _boom)
    with pytest.raises(RuntimeError):
        atomic_torch_save(target, {"step": 2})
    # old contents survive -- a kill mid-write must not corrupt the rolling file
    loaded = torch.load(target, map_location="cpu", weights_only=False)
    assert loaded["step"] == 1
    assert list(tmp_path.rglob("*.tmp*")) == []
