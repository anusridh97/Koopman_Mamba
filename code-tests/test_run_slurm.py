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
