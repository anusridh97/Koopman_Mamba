"""Launcher ABC + LocalLauncher (§3.4). DDP arithmetic as a pure, unit-tested
function -- replacing pretrain.sh's untested shell division.
"""
import sys

import pytest

pytestmark = pytest.mark.correctness


def _shard_spec(**runtime_overrides):
    from koopman_lm.config import build_config
    from experimentation.run.spec import OptimSpec, RuntimeSpec, RunSpec, ShardDataSpec

    return RunSpec(
        name="50m-fineweb-3b",
        model=build_config("50m"),
        data=ShardDataSpec(
            shard_dir="/scratch/data/fineweb_50m_train",
            tokenizer="NousResearch/Llama-2-7b-hf",
            mix={"fineweb": 1.0}, n_tokens=3_000_000_000,
        ),
        optim=OptimSpec(lr=4e-4, warmup_steps=300, max_steps=15000,
                          effective_batch=96),
        runtime=RuntimeSpec(per_device_batch_size=16, **runtime_overrides),
    )


def test_ddp_grad_accum_holds_effective_batch_constant():
    from experimentation.run.train_argv import ddp_grad_accum

    assert ddp_grad_accum(96, 16, world_size=1) == 6
    assert ddp_grad_accum(96, 16, world_size=2) == 3
    assert ddp_grad_accum(96, 16, world_size=6) == 1


def test_ddp_grad_accum_rejects_non_divisible_batch():
    from experimentation.run.train_argv import ddp_grad_accum

    with pytest.raises(ValueError):
        ddp_grad_accum(96, 16, world_size=4)   # 16*4=64 does not divide 96


def test_ddp_grad_accum_rejects_bad_world_size():
    from experimentation.run.train_argv import ddp_grad_accum

    with pytest.raises(ValueError):
        ddp_grad_accum(96, 16, world_size=0)


def test_build_train_argv_maps_shard_spec_onto_train_py_cli(tmp_path):
    from experimentation.run.train_argv import build_train_argv

    spec = _shard_spec()
    argv = build_train_argv(spec, tmp_path, world_size=1)
    assert "--model_size" in argv
    assert str(tmp_path / "model_config.json") in argv
    assert "--data_dir" in argv and "/scratch/data/fineweb_50m_train" in argv
    assert "--tokenizer" in argv and "NousResearch/Llama-2-7b-hf" in argv
    assert "--gradient_accumulation_steps" in argv
    ga_idx = argv.index("--gradient_accumulation_steps") + 1
    assert argv[ga_idx] == "6"
    assert "--bf16" in argv
    assert "--ddp" not in argv


def test_build_train_argv_adds_ddp_flag_under_multi_gpu():
    from experimentation.run.train_argv import build_train_argv

    spec = _shard_spec(ddp=True, gpus=2)
    argv = build_train_argv(spec, "/tmp/run", world_size=2)
    assert "--ddp" in argv
    ga_idx = argv.index("--gradient_accumulation_steps") + 1
    assert argv[ga_idx] == "3"


def test_build_train_argv_rejects_synthetic_data():
    from experimentation.run.train_argv import build_train_argv
    from koopman_lm.config import build_config
    from experimentation.run.spec import OptimSpec, RuntimeSpec, RunSpec, SyntheticDataSpec

    spec = RunSpec(
        name="x", model=build_config("50m"),
        data=SyntheticDataSpec(generator="mqar", params={}),
        optim=OptimSpec(lr=4e-4, warmup_steps=10, max_steps=100),
        runtime=RuntimeSpec(),
    )
    with pytest.raises(ValueError, match="synthetic"):
        build_train_argv(spec, "/tmp/run")


def test_local_launcher_build_command_single_gpu(tmp_path):
    from experimentation.run.launchers import LocalLauncher

    spec = _shard_spec()
    cmd = LocalLauncher().build_command(spec, tmp_path)
    assert cmd[0] == sys.executable
    assert cmd[1:3] == ["-m", "experimentation.training.train"]
    assert "--ddp" not in cmd


def test_local_launcher_build_command_multi_gpu_uses_torchrun(tmp_path):
    from experimentation.run.launchers import LocalLauncher

    spec = _shard_spec(ddp=True, gpus=2)   # 96 = 16 * 2 * 3: divides evenly
    cmd = LocalLauncher().build_command(spec, tmp_path)
    assert cmd[0] == "torchrun"
    assert "--nproc_per_node=2" in cmd
    assert "--ddp" in cmd


def test_local_launcher_submit_dry_run_does_not_call_subprocess(tmp_path, monkeypatch):
    from experimentation.run.launchers import LocalLauncher

    def _boom(*a, **k):
        raise AssertionError("subprocess.run must not be called under dry_run")

    monkeypatch.setattr("experimentation.run.launchers.subprocess.run", _boom)
    spec = _shard_spec()
    cmd = LocalLauncher().submit(spec, tmp_path, dry_run=True)
    assert cmd[1:3] == ["-m", "experimentation.training.train"]
    assert (tmp_path / "model_config.json").is_file()
