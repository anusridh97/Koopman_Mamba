"""Launcher ABC + LocalLauncher (§3.4). DDP arithmetic as a pure, unit-tested
function -- replacing pretrain.sh's untested shell division.
"""
import os
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
    assert cmd[:3] == [sys.executable, "-m", "torch.distributed.run"]
    assert "--nproc_per_node=2" in cmd
    assert "--ddp" in cmd


def test_the_ddp_launch_does_not_depend_on_the_PATH(tmp_path):
    """A DDP command's argv[0] must be a real, absolute, executable file.

    Regression. The DDP branch used to emit the bare string "torchrun", which
    resolves through PATH at exec time. That console script is installed in the
    venv's bin/, which is on PATH in an interactive login shell but NOT inside
    a Slurm job step -- so the 180M study raised
    `FileNotFoundError: [Errno 2] No such file or directory: 'torchrun'` and
    lost 25 of 25 trials in 22 seconds.

    Asserting the argv SHAPE is what let that through: a dry run printed a
    perfectly well-formed `torchrun --standalone --nproc_per_node=4 ...` and the
    command was still unrunnable. So assert the property that was actually
    violated -- argv[0] exists and is executable -- rather than its spelling.
    """
    from experimentation.run.launchers import LocalLauncher

    # 1, 2, 3, 6: _shard_spec has per_device_batch_size 16, and train_argv
    # requires effective_batch 96 to be a multiple of 16 * world_size.
    for gpus in (1, 2, 3, 6):
        spec = _shard_spec(ddp=True, gpus=gpus) if gpus > 1 else _shard_spec()
        argv0 = LocalLauncher().build_command(spec, tmp_path)[0]
        assert os.path.isabs(argv0), f"gpus={gpus}: argv[0] {argv0!r} is not absolute"
        assert os.path.exists(argv0), f"gpus={gpus}: argv[0] {argv0!r} does not exist"
        assert os.access(argv0, os.X_OK), f"gpus={gpus}: argv[0] {argv0!r} is not executable"


def test_local_launcher_submit_dry_run_does_not_call_subprocess(tmp_path, monkeypatch):
    from experimentation.run.launchers import LocalLauncher

    def _boom(*a, **k):
        raise AssertionError("subprocess.run must not be called under dry_run")

    monkeypatch.setattr("experimentation.run.launchers.subprocess.run", _boom)
    spec = _shard_spec()
    cmd = LocalLauncher().submit(spec, tmp_path, dry_run=True)
    assert cmd[1:3] == ["-m", "experimentation.training.train"]
    assert (tmp_path / "model_config.json").is_file()


# --------------------------------------------------- reproducible launches ----
#
# Until now nothing launched through experimentation.run could be reproduced.
# RuntimeSpec carried `seed` but no way to ask for deterministic kernels, and
# train_argv never passed --deterministic -- so every production run was
# nondeterministic even at a fixed seed.
#
# Measured consequence (job 439605, mqar_finetune): two runs at the SAME seed
# diverged 0.53 in loss by step 400 without --deterministic, and 0.000000 with
# it. That is why golden_4m_curve.json carries a 2.0e-4 "noise floor" while the
# two synthetic goldens reach 0.000000: the synthetic capture scripts invoke a
# trainer CLI directly and could pass the flag; the shard golden goes through the
# run system and could not. One of three instruments was ~2000x blunter than the
# others, and the cause was a missing field.

def test_runtime_can_request_determinism_and_does_not_by_default():
    """Default False, because turning determinism on globally would silently slow
    every existing run and change nothing about correctness."""
    from experimentation.run.spec import RuntimeSpec

    assert RuntimeSpec().deterministic is False
    assert RuntimeSpec(deterministic=True).deterministic is True


def test_the_flag_reaches_the_trainer_only_when_asked():
    from experimentation.run.train_argv import build_train_argv

    plain = build_train_argv(_shard_spec(), "/tmp/run")
    assert "--deterministic" not in plain

    repro = build_train_argv(_shard_spec(deterministic=True), "/tmp/run")
    assert "--deterministic" in repro


def test_determinism_does_not_move_run_id_or_group_id():
    """THE constraint that makes this change safe to land on a branch with
    committed goldens and archived run directories.

    run_id is sha256(model + data + optim + seed) and group_id is
    sha256(model + data + optim) -- runtime is deliberately excluded apart from
    the seed, because a microbatch size or a worker count is a memory detail
    rather than a scientific input. So a new runtime field cannot renumber
    anything. If this test ever fails, the field was added to the wrong spec.
    """
    from experimentation.run.spec import group_id, run_id

    a, b = _shard_spec(), _shard_spec(deterministic=True)
    assert run_id(a) == run_id(b)
    assert group_id(a) == group_id(b)


def test_an_archived_spec_without_the_field_still_resolves(tmp_path):
    """Every spec.yaml already on disk predates this field. resolve_run_spec
    validates field-for-field, so a new REQUIRED field would strand every
    archived run; a defaulted one must not."""
    import yaml

    from experimentation.run.resolve import resolve_run_spec

    spec = _shard_spec()
    raw = {
        "name": spec.name,
        # A bare preset name, exactly as configs/runs/50m-fineweb-3b.yaml
        # writes it.
        "model": "50m",
        "data": {"kind": "shard", "shard_dir": spec.data.shard_dir,
                 "tokenizer": spec.data.tokenizer, "mix": spec.data.mix,
                 "n_tokens": spec.data.n_tokens},
        "optim": {"lr": 4e-4, "warmup_steps": 300, "max_steps": 15000,
                  "effective_batch": 96},
        # NOTE: no `deterministic` key, exactly like every committed spec.
        "runtime": {"per_device_batch_size": 16, "seed": 42},
    }
    path = tmp_path / "archived.yaml"
    path.write_text(yaml.safe_dump(raw))

    resolved = resolve_run_spec(path)
    assert resolved.runtime.deterministic is False, \
        "an archived spec must resolve, and must keep its original behaviour"

