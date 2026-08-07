"""SlurmLauncher (§3.4): launch.sbatch is a generated artifact, never
hand-edited. Account/QoS are real RuntimeSpec fields (§3.4); gpu_arch
defaults to Marlowe's actual H100 (sm_90), not the B200/sm_100 bug in
scripts/train_50m.sh.
"""
import pytest

pytestmark = pytest.mark.correctness


def _shard_spec(**runtime_overrides):
    from koopman_lm.config import build_config
    from koopman_lm.run.spec import OptimSpec, RuntimeSpec, RunSpec, ShardDataSpec

    return RunSpec(
        name="50m-fineweb-3b",
        model=build_config("50m"),
        data=ShardDataSpec(
            shard_dir="/scratch/data/fineweb_50m_train",
            tokenizer="NousResearch/Llama-2-7b-hf",
            mix={"fineweb": 1.0}, n_tokens=3_000_000_000,
        ),
        optim=OptimSpec(lr=4e-4, warmup_steps=300, max_steps=15000,
                          effective_batch=96, per_device_batch_size=16),
        runtime=RuntimeSpec(**runtime_overrides),
    )


def test_render_sbatch_contains_correct_account_qos_and_h100_arch(tmp_path):
    from koopman_lm.run.slurm import SlurmLauncher

    spec = _shard_spec()
    text = SlurmLauncher().render_sbatch(spec, tmp_path)
    assert "#SBATCH --account=marlowe-m000151-pm06" in text
    assert "#SBATCH --qos=medium" in text
    assert "#SBATCH --partition=batch" in text
    assert 'TORCH_CUDA_ARCH_LIST="9.0"' in text
    # Regression test against the scripts/train_50m.sh bug this design fixes.
    assert "10.0" not in text
    assert "SKA_REQUIRE_B200" not in text


def test_render_sbatch_single_node_uses_python_dash_m(tmp_path):
    from koopman_lm.run.slurm import SlurmLauncher

    spec = _shard_spec()
    text = SlurmLauncher().render_sbatch(spec, tmp_path)
    assert "-m koopman_lm.training.train" in text
    assert "torchrun" not in text


def test_render_sbatch_multi_gpu_uses_torchrun(tmp_path):
    from koopman_lm.run.slurm import SlurmLauncher

    spec = _shard_spec(ddp=True, gpus=2, nodes=1)   # 96 = 16 * 2 * 3: divides evenly
    text = SlurmLauncher().render_sbatch(spec, tmp_path)
    assert "torchrun" in text
    assert "--nproc_per_node=2" in text


def test_render_sbatch_requests_preemption_signal_and_requeue(tmp_path):
    """§5.4: Slurm must warn the job 300s before killing it (SIGUSR1) so
    train.py's handler can write resume.pt and exit cleanly, and --requeue so
    a preempted/timed-out job is resubmitted rather than lost."""
    from koopman_lm.run.slurm import SlurmLauncher

    spec = _shard_spec()
    text = SlurmLauncher().render_sbatch(spec, tmp_path)
    assert "#SBATCH --signal=B:USR1@300" in text
    assert "#SBATCH --requeue" in text


def test_submit_dry_run_writes_sbatch_without_calling_sbatch(tmp_path, monkeypatch):
    from koopman_lm.run.slurm import SlurmLauncher

    def _boom(*a, **k):
        raise AssertionError("subprocess.run must not be called under dry_run")

    monkeypatch.setattr("koopman_lm.run.slurm.subprocess.run", _boom)
    spec = _shard_spec()
    path = SlurmLauncher().submit(spec, tmp_path, dry_run=True)
    assert path == tmp_path / "launch.sbatch"
    assert path.is_file()
    assert (tmp_path / "model_config.json").is_file()


# ---------------------------------------------------------------------------
# Array support (§4.2/§4.3): "the sweep grid is declared exactly once ... the
# array job indexes into that materialized list. Bash never knows the grid."
# koopman_lm.sweep materializes the (RunSpec, run_dir) pairs; this module
# only renders/writes the array script and the per-cell launch lines.
# ---------------------------------------------------------------------------

def test_render_array_sbatch_uses_shared_runtime_and_array_range(tmp_path):
    from koopman_lm.run.slurm import render_array_sbatch

    cells = [(_shard_spec(), tmp_path / f"cell{i}") for i in range(3)]
    text = render_array_sbatch("ska-rank-lr", cells, tmp_path, concurrency=2)
    assert "#SBATCH --array=0-2%2" in text
    assert "#SBATCH --account=marlowe-m000151-pm06" in text
    assert "SLURM_ARRAY_TASK_ID" in text
    assert str(tmp_path / "cells.txt") in text


def test_render_array_sbatch_without_concurrency_omits_percent_cap(tmp_path):
    from koopman_lm.run.slurm import render_array_sbatch

    cells = [(_shard_spec(), tmp_path / "cell0")]
    text = render_array_sbatch("x", cells, tmp_path)
    array_line = next(l for l in text.splitlines() if l.startswith("#SBATCH --array="))
    assert array_line == "#SBATCH --array=0-0"


def test_render_array_sbatch_rejects_heterogeneous_runtime_across_cells(tmp_path):
    from koopman_lm.run.slurm import render_array_sbatch

    cells = [(_shard_spec(), tmp_path / "cell0"),
             (_shard_spec(gpus=2), tmp_path / "cell1")]
    with pytest.raises(ValueError, match="gpus"):
        render_array_sbatch("x", cells, tmp_path)


def test_render_array_sbatch_rejects_empty_cell_list(tmp_path):
    from koopman_lm.run.slurm import render_array_sbatch

    with pytest.raises(ValueError):
        render_array_sbatch("x", [], tmp_path)


def test_submit_array_dry_run_writes_cells_txt_and_per_cell_launch_lines(tmp_path, monkeypatch):
    from koopman_lm.run.slurm import SlurmLauncher

    def _boom(*a, **k):
        raise AssertionError("subprocess.run must not be called under dry_run")

    monkeypatch.setattr("koopman_lm.run.slurm.subprocess.run", _boom)

    cells = [(_shard_spec(), tmp_path / "cell0"), (_shard_spec(), tmp_path / "cell1")]
    sweep_dir = tmp_path / "sweep"
    array_path = SlurmLauncher().submit_array("ska-rank-lr", cells, sweep_dir, dry_run=True)

    assert array_path == sweep_dir / "launch_array.sbatch"
    assert array_path.is_file()
    cells_txt = (sweep_dir / "cells.txt").read_text().splitlines()
    assert cells_txt == [str(tmp_path / "cell0"), str(tmp_path / "cell1")]
    for run_dir in (tmp_path / "cell0", tmp_path / "cell1"):
        assert (run_dir / "launch_line.sh").is_file()
        assert (run_dir / "model_config.json").is_file()
        assert "-m koopman_lm.training.train" in (run_dir / "launch_line.sh").read_text()


def test_submit_array_dry_run_script_is_valid_bash(tmp_path):
    import subprocess

    from koopman_lm.run.slurm import SlurmLauncher

    cells = [(_shard_spec(), tmp_path / "cell0"), (_shard_spec(), tmp_path / "cell1")]
    array_path = SlurmLauncher().submit_array("x", cells, tmp_path / "sweep", dry_run=True)
    result = subprocess.run(["bash", "-n", str(array_path)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

    for run_dir in (tmp_path / "cell0", tmp_path / "cell1"):
        r = subprocess.run(["bash", "-n", str(run_dir / "launch_line.sh")],
                            capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
