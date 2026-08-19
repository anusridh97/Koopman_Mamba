"""python -m experimentation.sweep <sweep.yaml> (§4.2/§4.3): expand -> materialize
each cell exactly like a lone `python -m experimentation.run` launch would ->
hand off to a Launcher. This is a NEW entry point (experimentation/sweep/), not a
modification of experimentation/run/__main__.py.

--dry_run is the GPU-free testing surface: it always prints the expanded
cell list (with run_id) and still materializes spec.yaml/attempts.jsonl (so
the orchestration itself is exercised), but never calls subprocess/sbatch.
"""
import json
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.correctness


@pytest.fixture(autouse=True)
def _pretend_clean_tree(monkeypatch):
    monkeypatch.setattr("experimentation.sweep.__main__.check_git_clean",
                         lambda allow_dirty: False)


def _write_sweep(tmp_path, *, axes_yaml, max_concurrent=None):
    shard_dir = tmp_path / "shard"
    shard_dir.mkdir()
    (shard_dir / "meta.json").write_text(json.dumps({
        "n_tokens": 3000000000, "tokenizer": "NousResearch/Llama-2-7b-hf",
        "mix": {"fineweb": 1.0},
    }))
    base_path = tmp_path / "base.yaml"
    base_path.write_text(textwrap.dedent(f"""
        name: 50m-fineweb-3b
        model: 50m
        data:
          kind: shard
          shard_dir: {shard_dir}
          tokenizer: NousResearch/Llama-2-7b-hf
          mix: {{fineweb: 1.0}}
          n_tokens: 3000000000
        optim:
          lr: 4.0e-4
          warmup_steps: 300
          max_steps: 15000
          effective_batch: 96
        runtime:
          per_device_batch_size: 16
          per_device_batch_size: 16
          seed: 42
    """))
    sweep_path = tmp_path / "sweep.yaml"
    concurrency_line = f"max_concurrent: {max_concurrent}\n" if max_concurrent else ""
    sweep_path.write_text(
        f"name: ska-rank-lr\n"
        f"base: {base_path}\n"
        f"{concurrency_line}"
        f"axes:\n{axes_yaml}"
    )
    return sweep_path


def test_dry_run_prints_every_cell_with_a_distinct_run_id(tmp_path, capsys):
    from experimentation.sweep.__main__ import main

    sweep_path = _write_sweep(tmp_path, axes_yaml=(
        "  model.ska_rank: [16, 24]\n"
        "  runtime.seed: [42, 43]\n"))
    run_root = tmp_path / "runs"
    main([str(sweep_path), "--run_root", str(run_root), "--dry_run"])
    out = capsys.readouterr().out
    assert "4 cell(s)" in out
    run_ids = set(re.findall(r"run_id=(\w+)", out))
    assert len(run_ids) == 4


def test_dry_run_materializes_spec_yaml_with_sweep_id_and_name(tmp_path):
    from experimentation.run.spec import run_dir_path
    from experimentation.sweep.__main__ import main
    from experimentation.sweep.spec import expand_cells, load_sweep_spec
    from experimentation.sweep.spec import sweep_id as compute_sweep_id

    sweep_path = _write_sweep(tmp_path, axes_yaml="  optim.lr: [1.0e-4, 2.0e-4]\n")
    run_root = tmp_path / "runs"
    main([str(sweep_path), "--run_root", str(run_root), "--dry_run"])

    sweep = load_sweep_spec(sweep_path)
    cells = expand_cells(sweep)
    assert len(cells) == 2
    for cell in cells:
        run_dir = run_dir_path(run_root, cell.spec)
        raw = yaml.safe_load((run_dir / "spec.yaml").read_text())
        assert raw["sweep_id"] == compute_sweep_id(sweep)
        assert raw["sweep_name"] == "ska-rank-lr"
        assert (run_dir / "attempts.jsonl").is_file()


def test_skip_done_skips_cells_with_a_final_dir(tmp_path):
    from experimentation.run.spec import run_dir_path
    from experimentation.sweep.__main__ import main
    from experimentation.sweep.spec import expand_cells, load_sweep_spec

    sweep_path = _write_sweep(tmp_path, axes_yaml="  optim.lr: [1.0e-4, 2.0e-4]\n")
    run_root = tmp_path / "runs"

    sweep = load_sweep_spec(sweep_path)
    cells = expand_cells(sweep)
    done_dir = run_dir_path(run_root, cells[0].spec)
    (done_dir / "final").mkdir(parents=True)

    main([str(sweep_path), "--run_root", str(run_root), "--dry_run", "--skip_done"])

    pending_dir = run_dir_path(run_root, cells[1].spec)
    assert (pending_dir / "spec.yaml").is_file()
    # The already-done cell must not have been touched by this invocation --
    # it only has the final/ dir the test created, no spec.yaml.
    assert not (done_dir / "spec.yaml").is_file()


def test_all_cells_done_is_a_clean_no_op(tmp_path, capsys):
    from experimentation.run.spec import run_dir_path
    from experimentation.sweep.__main__ import main
    from experimentation.sweep.spec import expand_cells, load_sweep_spec

    sweep_path = _write_sweep(tmp_path, axes_yaml="  optim.lr: [1.0e-4]\n")
    run_root = tmp_path / "runs"
    sweep = load_sweep_spec(sweep_path)
    cells = expand_cells(sweep)
    (run_dir_path(run_root, cells[0].spec) / "final").mkdir(parents=True)

    result = main([str(sweep_path), "--run_root", str(run_root), "--dry_run", "--skip_done"])
    assert result == []


def test_local_launcher_dry_run_returns_one_command_per_cell(tmp_path):
    from experimentation.sweep.__main__ import main

    sweep_path = _write_sweep(tmp_path, axes_yaml="  optim.lr: [1.0e-4, 2.0e-4]\n")
    run_root = tmp_path / "runs"
    cmds = main([str(sweep_path), "--run_root", str(run_root), "--dry_run",
                 "--launcher", "local"])
    assert len(cmds) == 2
    for cmd in cmds:
        assert cmd[1:3] == ["-m", "experimentation.training.train"]


def test_slurm_launcher_dry_run_writes_one_array_script_for_the_whole_sweep(tmp_path):
    import subprocess

    from experimentation.sweep.__main__ import main

    sweep_path = _write_sweep(
        tmp_path, axes_yaml="  optim.lr: [1.0e-4, 2.0e-4, 3.0e-4]\n",
        max_concurrent=2)
    run_root = tmp_path / "runs"
    array_path = main([str(sweep_path), "--run_root", str(run_root), "--dry_run",
                       "--launcher", "slurm"])

    text = array_path.read_text()
    assert "#SBATCH --array=0-2%2" in text

    cells_txt_lines = (array_path.parent / "cells.txt").read_text().splitlines()
    assert len(cells_txt_lines) == 3
    for run_dir_line in cells_txt_lines:
        assert (Path(run_dir_line) / "launch_line.sh").is_file()
        assert (Path(run_dir_line) / "spec.yaml").is_file()

    result = subprocess.run(["bash", "-n", str(array_path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_dry_run_does_not_crash_when_the_shard_does_not_exist_yet(tmp_path):
    """A dry run's whole point is to inspect the expanded plan *before* the
    data exists -- typically from a login node. Regression: verify_shard's
    "no meta.json found" used to propagate as an uncaught
    DataVerificationError, turning `--dry_run` into a nonzero exit and
    failing any CI check that shells out to it (the cell list had already
    printed correctly by then). Exercised as a real subprocess so the
    actual CLI exit code is checked, not just whether main() raises
    in-process."""
    shard_dir = tmp_path / "shard_not_pretokenized_yet"
    base_path = tmp_path / "base.yaml"
    base_path.write_text(textwrap.dedent(f"""
        name: 50m-fineweb-3b
        model: 50m
        data:
          kind: shard
          shard_dir: {shard_dir}
          tokenizer: NousResearch/Llama-2-7b-hf
          mix: {{fineweb: 1.0}}
          n_tokens: 3000000000
        optim:
          lr: 4.0e-4
          warmup_steps: 300
          max_steps: 15000
          effective_batch: 96
        runtime:
          seed: 42
    """))
    sweep_path = tmp_path / "sweep.yaml"
    sweep_path.write_text(
        f"name: ska-rank-lr\n"
        f"base: {base_path}\n"
        f"axes:\n  optim.lr: [1.0e-4, 2.0e-4]\n"
    )
    run_root = tmp_path / "runs"

    proc = subprocess.run(
        [sys.executable, "-m", "experimentation.sweep", str(sweep_path),
         "--run_root", str(run_root), "--dry_run", "--allow-dirty"],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode == 0, (
        f"expected clean exit 0, got {proc.returncode}\n"
        f"stdout={proc.stdout}\nstderr={proc.stderr}"
    )
    assert "2 cell(s)" in proc.stdout
    assert "run_id=" in proc.stdout
    assert not shard_dir.exists()  # dry_run must not have tried to create it


def test_refuses_to_launch_from_a_dirty_tree(tmp_path, monkeypatch):
    from experimentation.run.provenance import DirtyTreeError
    from experimentation.sweep.__main__ import main

    sweep_path = _write_sweep(tmp_path, axes_yaml="  optim.lr: [1.0e-4]\n")
    run_root = tmp_path / "runs"
    monkeypatch.setattr(
        "experimentation.sweep.__main__.check_git_clean",
        lambda allow_dirty: (_ for _ in ()).throw(DirtyTreeError("dirty")))

    with pytest.raises(DirtyTreeError):
        main([str(sweep_path), "--run_root", str(run_root), "--dry_run"])
