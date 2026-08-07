"""Artifact write policy (§3.7): earned bytes (checkpoints, results) refuse to
clobber; derived bytes (spec.yaml, sbatch, attempts.jsonl entries) are cheap
and use plain atomic writes.
"""
import json

import pytest

pytestmark = pytest.mark.correctness


def test_atomic_write_text_creates_parents_and_no_tmp_leftover(tmp_path):
    from koopman_lm.run.artifacts import atomic_write_text

    target = tmp_path / "nested" / "spec.yaml"
    atomic_write_text(target, "hello: world\n")
    assert target.read_text() == "hello: world\n"
    leftovers = list(tmp_path.rglob("*.tmp*"))
    assert leftovers == []


def test_atomic_write_json_round_trips(tmp_path):
    from koopman_lm.run.artifacts import atomic_write_json

    target = tmp_path / "result.json"
    atomic_write_json(target, {"a": 1, "b": [1, 2, 3]})
    assert json.loads(target.read_text()) == {"a": 1, "b": [1, 2, 3]}


def test_create_run_dir_happy_path(tmp_path):
    from koopman_lm.run.artifacts import create_run_dir

    run_dir = tmp_path / "50m-fineweb-3b.abcd1234" / "seed42.deadbeef"
    result = create_run_dir(run_dir)
    assert result == run_dir
    assert run_dir.is_dir()


def test_create_run_dir_refuses_to_clobber_a_finished_run(tmp_path):
    from koopman_lm.run.artifacts import RunDirConflictError, create_run_dir

    run_dir = tmp_path / "run"
    (run_dir / "final").mkdir(parents=True)
    with pytest.raises(RunDirConflictError):
        create_run_dir(run_dir)


def test_create_run_dir_allows_resume_or_force(tmp_path):
    from koopman_lm.run.artifacts import create_run_dir

    run_dir = tmp_path / "run"
    (run_dir / "final").mkdir(parents=True)
    create_run_dir(run_dir, resume=True)
    create_run_dir(run_dir, force=True)


def test_append_attempt_is_append_only(tmp_path):
    from koopman_lm.run.artifacts import append_attempt

    run_dir = tmp_path / "run"
    append_attempt(run_dir, {"host": "node01", "job_id": "1"})
    append_attempt(run_dir, {"host": "node01", "job_id": "2"})
    lines = (run_dir / "attempts.jsonl").read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["job_id"] == "1"
    assert json.loads(lines[1])["job_id"] == "2"


def test_make_attempt_record_shape():
    from koopman_lm.run.artifacts import make_attempt_record

    record = make_attempt_record(host="node01", job_id="12345",
                                   git_commit="deadbeef", forced=False)
    assert record["host"] == "node01"
    assert record["job_id"] == "12345"
    assert record["git_commit"] == "deadbeef"
    assert record["forced"] is False
    assert "timestamp" in record
