"""python -m koopman_lm.run <spec.yaml>: the full orchestration order (§3.4) --
resolve -> verify data -> materialize spec.yaml + attempt record -> hand off
to a Launcher. Always invoked with --dry_run so no real training subprocess
or GPU is ever touched.
"""
import json
import textwrap

import pytest
import yaml

pytestmark = pytest.mark.correctness


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
    from koopman_lm.run.__main__ import main
    from koopman_lm.run.spec import run_dir_path
    from koopman_lm.run.resolve import resolve_run_spec

    spec_path = _write_shard_run_spec(tmp_path)
    run_root = tmp_path / "runs"
    cmd = main([str(spec_path), "--run_root", str(run_root), "--dry_run"])
    assert cmd[1:3] == ["-m", "koopman_lm.training.train"]

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
    from koopman_lm.run.__main__ import main

    spec_path = _write_shard_run_spec(tmp_path)
    run_root = tmp_path / "runs"

    # First launch: no --resume, run dir doesn't exist yet, nothing to resume.
    main([str(spec_path), "--run_root", str(run_root), "--dry_run"])

    # Second invocation: --resume must be forwarded to train.py's argv.
    cmd = main([str(spec_path), "--run_root", str(run_root), "--dry_run", "--resume"])
    assert "--resume" in cmd


def test_main_dry_run_twice_appends_a_second_attempt_without_conflict(tmp_path):
    from koopman_lm.run.__main__ import main
    from koopman_lm.run.spec import run_dir_path
    from koopman_lm.run.resolve import resolve_run_spec

    spec_path = _write_shard_run_spec(tmp_path)
    run_root = tmp_path / "runs"
    main([str(spec_path), "--run_root", str(run_root), "--dry_run"])
    main([str(spec_path), "--run_root", str(run_root), "--dry_run"])

    spec = resolve_run_spec(spec_path)
    run_dir = run_dir_path(run_root, spec)
    attempts = (run_dir / "attempts.jsonl").read_text().splitlines()
    assert len(attempts) == 2


def test_main_refuses_to_clobber_a_finished_run(tmp_path):
    from koopman_lm.run.__main__ import main
    from koopman_lm.run.artifacts import RunDirConflictError
    from koopman_lm.run.spec import run_dir_path
    from koopman_lm.run.resolve import resolve_run_spec

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
    from koopman_lm.run.__main__ import main
    from koopman_lm.run.data_verify import DataVerificationError
    from koopman_lm.run.spec import run_dir_path
    from koopman_lm.run.resolve import resolve_run_spec

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
