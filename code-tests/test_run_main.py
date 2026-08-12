"""python -m experimentation.run <spec.yaml>: the full orchestration order (§3.4) --
resolve -> verify data -> materialize spec.yaml + attempt record -> hand off
to a Launcher. Always invoked with --dry_run so no real training subprocess
or GPU is ever touched.
"""
import json
import subprocess
import sys
import textwrap

import pytest
import yaml

pytestmark = pytest.mark.correctness


@pytest.fixture(autouse=True)
def _pretend_clean_tree(monkeypatch):
    """main() now refuses to launch from a dirty git tree (Fix 3). These
    tests exercise orchestration, not the state of the developer's actual
    working tree at test time, so default to "clean" here; the tests that
    specifically cover dirty-tree behavior override this explicitly."""
    monkeypatch.setattr("experimentation.run.__main__.check_git_clean",
                         lambda allow_dirty: False)


def _write_shard_run_spec(tmp_path):
    shard_dir = tmp_path / "shard"
    shard_dir.mkdir()
    (shard_dir / "meta.json").write_text(json.dumps({
        "n_tokens": 1000, "tokenizer": "NousResearch/Llama-2-7b-hf",
        "mix": {"fineweb": 1.0},
    }))
    spec_path = tmp_path / "spec.yaml"
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
          seed: 42
    """))
    return spec_path


def test_main_dry_run_materializes_and_appends_one_attempt(tmp_path):
    from experimentation.run.__main__ import main
    from experimentation.run.spec import run_dir_path
    from experimentation.run.resolve import resolve_run_spec

    spec_path = _write_shard_run_spec(tmp_path)
    run_root = tmp_path / "runs"
    cmd = main([str(spec_path), "--run_root", str(run_root), "--dry_run"])
    assert cmd[1:3] == ["-m", "experimentation.training.train"]

    spec = resolve_run_spec(spec_path)
    run_dir = run_dir_path(run_root, spec)
    assert (run_dir / "spec.yaml").is_file()
    attempts = (run_dir / "attempts.jsonl").read_text().splitlines()
    assert len(attempts) == 1
    assert json.loads(attempts[0])["forced"] is False


def test_main_resume_flag_reaches_train_py_argv(tmp_path):
    """`--resume` must survive the whole orchestrator path -- resolve ->
    create_run_dir -> build_train_argv -- and land on train.py's own
    `--resume` flag. Exercising build_train_argv and create_run_dir in
    isolation (as the rest of this suite historically did) cannot catch a
    dropped flag between them; only composing the real entry point can.
    """
    from experimentation.run.__main__ import main

    spec_path = _write_shard_run_spec(tmp_path)
    run_root = tmp_path / "runs"

    # First launch: no --resume, run dir doesn't exist yet, nothing to resume.
    main([str(spec_path), "--run_root", str(run_root), "--dry_run"])

    # Second invocation: --resume must be forwarded to train.py's argv.
    cmd = main([str(spec_path), "--run_root", str(run_root), "--dry_run", "--resume"])
    assert "--resume" in cmd


def test_main_dry_run_twice_appends_a_second_attempt_without_conflict(tmp_path):
    from experimentation.run.__main__ import main
    from experimentation.run.spec import run_dir_path
    from experimentation.run.resolve import resolve_run_spec

    spec_path = _write_shard_run_spec(tmp_path)
    run_root = tmp_path / "runs"
    main([str(spec_path), "--run_root", str(run_root), "--dry_run"])
    main([str(spec_path), "--run_root", str(run_root), "--dry_run"])

    spec = resolve_run_spec(spec_path)
    run_dir = run_dir_path(run_root, spec)
    attempts = (run_dir / "attempts.jsonl").read_text().splitlines()
    assert len(attempts) == 2


def test_main_refuses_to_clobber_a_finished_run(tmp_path):
    from experimentation.run.__main__ import main
    from experimentation.run.artifacts import RunDirConflictError
    from experimentation.run.spec import run_dir_path
    from experimentation.run.resolve import resolve_run_spec

    spec_path = _write_shard_run_spec(tmp_path)
    run_root = tmp_path / "runs"
    spec = resolve_run_spec(spec_path)
    run_dir = run_dir_path(run_root, spec)
    (run_dir / "final").mkdir(parents=True)

    with pytest.raises(RunDirConflictError):
        main([str(spec_path), "--run_root", str(run_root), "--dry_run"])

    # --force proceeds and is recorded.
    main([str(spec_path), "--run_root", str(run_root), "--dry_run", "--force"])
    attempts = [json.loads(l) for l in (run_dir / "attempts.jsonl").read_text().splitlines()]
    assert attempts[-1]["forced"] is True


def test_main_verifies_data_before_creating_the_run_dir(tmp_path):
    from experimentation.run.__main__ import main
    from experimentation.run.data_verify import DataVerificationError
    from experimentation.run.spec import run_dir_path
    from experimentation.run.resolve import resolve_run_spec

    spec_path = _write_shard_run_spec(tmp_path)
    # Corrupt the shard's meta.json after writing the spec so verification fails.
    shard_dir = tmp_path / "shard"
    (shard_dir / "meta.json").write_text(json.dumps({
        "n_tokens": 999999, "tokenizer": "NousResearch/Llama-2-7b-hf",
        "mix": {"fineweb": 1.0},
    }))
    run_root = tmp_path / "runs"
    with pytest.raises(DataVerificationError):
        main([str(spec_path), "--run_root", str(run_root), "--dry_run"])

    spec = resolve_run_spec(spec_path)
    run_dir = run_dir_path(run_root, spec)
    assert not run_dir.exists()   # verification failed before any dir was created


def test_main_dry_run_does_not_crash_when_the_shard_does_not_exist_yet(tmp_path):
    """Same regression as experimentation.sweep: a dry run must be inspectable
    before the data shard exists (typically from a login node). verify_shard
    raising DataVerificationError on a missing meta.json used to turn
    `python -m experimentation.run --dry_run` into a nonzero exit. Exercised as a
    real subprocess so the actual CLI exit code is checked."""
    shard_dir = tmp_path / "shard_not_pretokenized_yet"
    spec_path = tmp_path / "spec.yaml"
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
          seed: 42
    """))
    run_root = tmp_path / "runs"

    proc = subprocess.run(
        [sys.executable, "-m", "experimentation.run", str(spec_path),
         "--run_root", str(run_root), "--dry_run", "--allow-dirty"],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, (
        f"expected clean exit 0, got {proc.returncode}\n"
        f"stdout={proc.stdout}\nstderr={proc.stderr}"
    )
    assert not shard_dir.exists()  # dry_run must not have tried to create it


def test_main_refuses_to_launch_from_a_dirty_tree(tmp_path, monkeypatch):
    """A dirty working tree must block launch before anything is
    materialized -- the default is refusal, so launching uncommitted is a
    deliberate act (--allow-dirty) rather than an accident."""
    from experimentation.run.__main__ import main
    from experimentation.run.provenance import DirtyTreeError
    from experimentation.run.spec import run_dir_path
    from experimentation.run.resolve import resolve_run_spec

    spec_path = _write_shard_run_spec(tmp_path)
    run_root = tmp_path / "runs"
    monkeypatch.setattr("experimentation.run.__main__.check_git_clean",
                         lambda allow_dirty: (_ for _ in ()).throw(
                             DirtyTreeError("dirty: experimentation/run/resolve.py")))

    with pytest.raises(DirtyTreeError):
        main([str(spec_path), "--run_root", str(run_root), "--dry_run"])

    spec = resolve_run_spec(spec_path)
    run_dir = run_dir_path(run_root, spec)
    assert not run_dir.exists()   # nothing materialized before the dirty check


def test_main_allow_dirty_records_dirty_true_in_spec_yaml(tmp_path, monkeypatch):
    """--allow-dirty is a loud, explicit escape hatch: it must not silently
    bypass the check -- the resulting spec.yaml records dirty: true."""
    import yaml

    from experimentation.run.__main__ import main
    from experimentation.run.spec import run_dir_path
    from experimentation.run.resolve import resolve_run_spec

    spec_path = _write_shard_run_spec(tmp_path)
    run_root = tmp_path / "runs"
    monkeypatch.setattr("experimentation.run.__main__.check_git_clean",
                         lambda allow_dirty: True)

    main([str(spec_path), "--run_root", str(run_root), "--dry_run", "--allow-dirty"])

    spec = resolve_run_spec(spec_path)
    run_dir = run_dir_path(run_root, spec)
    raw = yaml.safe_load((run_dir / "spec.yaml").read_text())
    assert raw["dirty"] is True
