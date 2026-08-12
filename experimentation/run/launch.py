"""Launcher ABC (§3.4): the spec says *what* to run, the launcher says
*where*. build_train_argv maps a RunSpec onto experimentation.training.train's
existing CLI -- train.py itself is frozen and not modified by this design.
"""
from __future__ import annotations

import abc
import dataclasses
import subprocess
import sys
from pathlib import Path
from typing import List

from experimentation.run.artifacts import atomic_write_json
from experimentation.run.spec import RunSpec, ShardDataSpec


def ddp_grad_accum(effective_batch: int, per_device_batch_size: int,
                    world_size: int) -> int:
    """Gradient-accumulation steps that hold effective batch size constant as
    world_size (GPU count) changes. Mirrors pretrain.sh's
    `GA_USE=$(( GA / NPROC ))`, but as a pure, unit-tested function instead
    of untested shell arithmetic (§3.4) -- and unlike the shell version,
    raises rather than silently truncating when the division isn't exact.
    """
    if world_size < 1:
        raise ValueError("world_size must be >= 1")
    per_step = per_device_batch_size * world_size
    if effective_batch % per_step != 0:
        raise ValueError(
            f"effective_batch={effective_batch} is not a multiple of "
            f"per_device_batch_size*world_size={per_step}")
    return max(1, effective_batch // per_step)


def write_model_config(spec: RunSpec, run_dir) -> Path:
    """Write the resolved model config as a standalone JSON file so the
    (frozen) train.py CLI's --model_size can load it via build_config's
    file-path branch, without re-reading configs/*.yaml or touching
    CONFIG_REGISTRY."""
    path = Path(run_dir) / "model_config.json"
    atomic_write_json(path, dataclasses.asdict(spec.model))
    return path


def build_train_argv(spec: RunSpec, run_dir, *, world_size: int = 1,
                      resume: bool = False) -> List[str]:
    """Map a RunSpec onto experimentation.training.train's existing CLI flags.
    Only kind='shard' is supported: kind='synthetic' needs TrainTask/
    SyntheticTask (§6.2), a separate, later plan."""
    if not isinstance(spec.data, ShardDataSpec):
        raise ValueError(
            "build_train_argv only supports data.kind='shard' -- synthetic "
            "runs need TrainTask/SyntheticTask (design §6), not yet "
            "implemented; experimentation.training.train's CLI has no synthetic "
            "data path.")
    run_dir = Path(run_dir)
    model_config_path = run_dir / "model_config.json"
    grad_accum = ddp_grad_accum(spec.optim.effective_batch,
                                 spec.optim.per_device_batch_size, world_size)
    # train.py's --save_steps default (5000) is silently unreachable for any
    # RunSpec with fewer max_steps than that -- e.g. a short proof-of-pipeline
    # run never writes a single step_<N>/ checkpoint, which also makes
    # --resume untestable (resume.pt is written at the same cadence, §5.3).
    # Scale it off max_steps (three checkpoints over the run, at least one)
    # instead of leaving the CLI default in force unconditionally.
    save_steps = max(1, spec.optim.max_steps // 3)
    argv = [
        "--model_size", str(model_config_path),
        "--data_dir", spec.data.shard_dir,
        "--tokenizer", spec.data.tokenizer,
        "--max_seq_len", str(spec.model.max_seq_len),
        "--per_device_train_batch_size", str(spec.optim.per_device_batch_size),
        "--gradient_accumulation_steps", str(grad_accum),
        "--max_steps", str(spec.optim.max_steps),
        "--learning_rate", str(spec.optim.lr),
        "--warmup_steps", str(spec.optim.warmup_steps),
        "--weight_decay", str(spec.optim.weight_decay),
        "--max_grad_norm", str(spec.optim.grad_clip),
        "--num_workers", str(spec.runtime.workers),
        "--save_steps", str(save_steps),
        "--output_dir", str(run_dir),
        "--seed", str(spec.runtime.seed),
        "--phase_tag", spec.name,
    ]
    argv.append("--bf16" if spec.runtime.precision == "bf16" else "--no_bf16")
    # train.py's CLI defaults --compile=True and --gradient_checkpointing=True,
    # but the fused-prefix-scan architecture (ska_prefix_scan=True, any
    # backend) does not tolerate that combination: torch.compile's
    # cudagraph-trees mode captures the whole step, and activation-checkpoint
    # recomputation re-entering the custom autograd Function
    # (_SKAPrefixScanFn) mid-capture raises
    # `torch.AcceleratorError: CUDA error: operation failed due to a previous
    # error during capture` (cudaErrorStreamCaptureInvalidated) -- reproduced
    # on H100 in run 50m-e2e-gpu-smoke (job 415208, run2-resume log), which
    # crashed identically even with ska_backend='pytorch', so this is not
    # scoped to the CUDA kernel specifically. scripts/pretrain.sh already
    # carries `--no_compile --no_gradient_checkpointing` as the known-safe
    # combination for every production 50m/180m run; the run system must not
    # regress behind that by omission.
    argv.append("--no_compile")
    argv.append("--no_gradient_checkpointing")
    if spec.runtime.ddp and world_size > 1:
        argv.append("--ddp")
    if resume:
        argv.append("--resume")
    return argv


class Launcher(abc.ABC):
    """The spec says *what* to run; the launcher says *where* (§3.4)."""

    @abc.abstractmethod
    def build_command(self, spec: RunSpec, run_dir, *, resume: bool = False) -> List[str]:
        ...

    @abc.abstractmethod
    def submit(self, spec: RunSpec, run_dir, dry_run: bool = False, *, resume: bool = False):
        ...


class LocalLauncher(Launcher):
    """Runs training in-process via subprocess: `python -m ...` for a single
    GPU, `torchrun --standalone` when runtime.ddp and runtime.gpus > 1."""

    def build_command(self, spec: RunSpec, run_dir, *, resume: bool = False) -> List[str]:
        world_size = spec.runtime.gpus if (spec.runtime.ddp and spec.runtime.gpus > 1) else 1
        train_args = build_train_argv(spec, run_dir, world_size=world_size, resume=resume)
        if world_size > 1:
            return ["torchrun", "--standalone", f"--nproc_per_node={world_size}",
                     "-m", "experimentation.training.train", *train_args]
        return [sys.executable, "-m", "experimentation.training.train", *train_args]

    def submit(self, spec: RunSpec, run_dir, dry_run: bool = False, *, resume: bool = False):
        write_model_config(spec, run_dir)
        cmd = self.build_command(spec, run_dir, resume=resume)
        if dry_run:
            return cmd
        return subprocess.run(cmd, check=True)
