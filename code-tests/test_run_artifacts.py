"""Artifact write policy (§3.7): earned bytes (checkpoints, results) refuse to
clobber; derived bytes (spec.yaml, sbatch, attempts.jsonl entries) are cheap
and use plain atomic writes.
"""
import json

import pytest

pytestmark = pytest.mark.correctness


def test_atomic_write_text_creates_parents_and_no_tmp_leftover(tmp_path):
    from experimentation.atomic_io import atomic_write_text

    target = tmp_path / "nested" / "spec.yaml"
    atomic_write_text(target, "hello: world\n")
    assert target.read_text() == "hello: world\n"
    leftovers = list(tmp_path.rglob("*.tmp*"))
    assert leftovers == []


def test_atomic_write_json_round_trips(tmp_path):
    from experimentation.atomic_io import atomic_write_json

    target = tmp_path / "result.json"
    atomic_write_json(target, {"a": 1, "b": [1, 2, 3]})
    assert json.loads(target.read_text()) == {"a": 1, "b": [1, 2, 3]}


def test_create_run_dir_happy_path(tmp_path):
    from experimentation.run.write_policy import create_run_dir

    run_dir = tmp_path / "50m-fineweb-3b.abcd1234" / "seed42.deadbeef"
    result = create_run_dir(run_dir)
    assert result == run_dir
    assert run_dir.is_dir()


def test_create_run_dir_refuses_to_clobber_a_finished_run(tmp_path):
    from experimentation.run.write_policy import RunDirConflictError, create_run_dir

    run_dir = tmp_path / "run"
    (run_dir / "final").mkdir(parents=True)
    with pytest.raises(RunDirConflictError):
        create_run_dir(run_dir)


def test_create_run_dir_allows_resume_or_force(tmp_path):
    from experimentation.run.write_policy import create_run_dir

    run_dir = tmp_path / "run"
    (run_dir / "final").mkdir(parents=True)
    create_run_dir(run_dir, resume=True)
    create_run_dir(run_dir, force=True)


def test_append_attempt_is_append_only(tmp_path):
    from experimentation.run.write_policy import append_attempt

    run_dir = tmp_path / "run"
    append_attempt(run_dir, {"host": "node01", "job_id": "1"})
    append_attempt(run_dir, {"host": "node01", "job_id": "2"})
    lines = (run_dir / "attempts.jsonl").read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["job_id"] == "1"
    assert json.loads(lines[1])["job_id"] == "2"


def test_make_attempt_record_shape():
    from experimentation.run.write_policy import make_attempt_record

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
    from experimentation.run.write_policy import make_attempt_record

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

    from experimentation.run.write_policy import RunDirConflictError, create_run_dir

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

    from experimentation.run.write_policy import RunDirConflictError, create_run_dir

    run_dir = tmp_path / "run"
    (run_dir / "final").mkdir(parents=True)
    (run_dir / "spec.yaml").write_text(yaml.safe_dump({"code_id": "aaaaaaa"}))

    with pytest.raises(RunDirConflictError) as excinfo:
        create_run_dir(run_dir, code_id="aaaaaaa")
    assert "different code_id" not in str(excinfo.value)


def test_atomic_torch_save_round_trips_and_leaves_no_tmp(tmp_path):
    import torch
    from experimentation.atomic_io import atomic_torch_save

    target = tmp_path / "resume.pt"
    atomic_torch_save(target, {"step": 7, "tensor": torch.arange(4)})
    loaded = torch.load(target, map_location="cpu", weights_only=False)
    assert loaded["step"] == 7
    assert torch.equal(loaded["tensor"], torch.arange(4))
    leftovers = list(tmp_path.rglob("*.tmp*"))
    assert leftovers == []


def test_atomic_torch_save_failure_does_not_corrupt_existing_file(tmp_path, monkeypatch):
    import torch
    from experimentation.atomic_io import atomic_torch_save

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


# ---- concurrent-launch claim (§3.7 / reading-progress finding 4.11) ----

def test_claim_run_dir_refuses_a_second_concurrent_claim(tmp_path):
    """create_run_dir's guard keys on `final/`, i.e. on *completed* runs, so two
    launchers materializing the same spec at the same moment both proceed into
    one directory and interleave their spec.yaml / attempts.jsonl writes. The
    claim is the mutual exclusion that guard cannot provide."""
    from experimentation.run.write_policy import RunDirClaimedError, claim_run_dir

    run_dir = tmp_path / "run"
    with claim_run_dir(run_dir):
        with pytest.raises(RunDirClaimedError):
            with claim_run_dir(run_dir):
                pass


def test_claim_run_dir_releases_so_sequential_relaunch_is_unblocked(tmp_path):
    """A held claim must not outlive the launcher that took it -- a run that
    failed early has to be relaunchable without --force."""
    from experimentation.run.write_policy import claim_run_dir

    run_dir = tmp_path / "run"
    with claim_run_dir(run_dir):
        pass
    with claim_run_dir(run_dir):
        pass


def test_claim_run_dir_releases_when_the_body_raises(tmp_path):
    from experimentation.run.write_policy import claim_run_dir

    run_dir = tmp_path / "run"
    with pytest.raises(RuntimeError):
        with claim_run_dir(run_dir):
            raise RuntimeError("materialization blew up")
    # the claim is gone, so the next launcher is not blocked by a corpse
    with claim_run_dir(run_dir):
        pass


def test_claim_run_dir_records_who_holds_it(tmp_path):
    """A stale claim is only actionable if it says who left it."""
    import os

    from experimentation.run.write_policy import claim_run_dir

    run_dir = tmp_path / "run"
    with claim_run_dir(run_dir):
        claim = json.loads((run_dir / ".running" / "claim.json").read_text())
    assert claim["pid"] == os.getpid()
    assert claim["host"]
    assert claim["timestamp"]


def test_claim_run_dir_force_steals_a_stale_claim(tmp_path):
    """--force already means "proceed anyway, and it is recorded"; a claim left
    by a launcher that died mid-materialize must not be harder to clear than a
    completed run's final/."""
    from experimentation.run.write_policy import claim_run_dir

    run_dir = tmp_path / "run"
    (run_dir / ".running").mkdir(parents=True)     # a corpse from a killed launcher
    with claim_run_dir(run_dir, force=True):
        pass


def test_claim_run_dir_leaves_no_sentinel_behind(tmp_path):
    from experimentation.run.write_policy import claim_run_dir

    run_dir = tmp_path / "run"
    with claim_run_dir(run_dir):
        pass
    assert not (run_dir / ".running").exists()


# ---- atomic_write_bytes (run-provenance design 4.2 needs a binary writer) ----

def test_atomic_write_bytes_round_trips_non_utf8_and_creates_parents(tmp_path):
    """The caller is source.tar.gz, so the payload is gzip -- bytes that are not
    valid UTF-8. Routing it through atomic_write_text would raise or mangle it."""
    from experimentation.atomic_io import atomic_write_bytes

    payload = b"\x1f\x8b\x08\x00\x00\x00\x00\x00\xff\xfe not utf-8 \x00"
    target = tmp_path / "nested" / "source.tar.gz"
    atomic_write_bytes(target, payload)
    assert target.read_bytes() == payload
    assert list(tmp_path.rglob("*.tmp*")) == []


def test_atomic_write_bytes_failure_does_not_corrupt_the_existing_file(tmp_path, monkeypatch):
    """Same guarantee atomic_torch_save gives: a kill mid-write leaves the
    previous good file intact and no temp behind. An archive written beside a
    finished run's spec.yaml is provenance -- a truncated one is worse than none,
    because its hash would silently name the wrong code."""
    from pathlib import Path

    from experimentation.atomic_io import atomic_write_bytes

    target = tmp_path / "source.tar.gz"
    atomic_write_bytes(target, b"good")

    # Patch the write step, not os.replace: os.replace is used by pytest's own
    # machinery, so stubbing it globally would break the test run rather than
    # the code under test. This mirrors how the atomic_torch_save test stubs
    # torch.save.
    def _boom(self, data):
        raise RuntimeError("simulated kill mid-write")

    monkeypatch.setattr(Path, "write_bytes", _boom)
    with pytest.raises(RuntimeError):
        atomic_write_bytes(target, b"doomed")
    monkeypatch.undo()

    assert target.read_bytes() == b"good"
    assert list(tmp_path.rglob("*.tmp*")) == []


def test_atomic_write_bytes_overwrites_in_place(tmp_path):
    from experimentation.atomic_io import atomic_write_bytes

    target = tmp_path / "source.tar.gz"
    atomic_write_bytes(target, b"first")
    atomic_write_bytes(target, b"second")
    assert target.read_bytes() == b"second"
