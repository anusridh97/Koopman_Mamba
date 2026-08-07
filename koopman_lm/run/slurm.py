"""SlurmLauncher (§3.4): generates launch.sbatch (an artifact, not a
hand-edited script) and submits it via `sbatch`. Marlowe H100 nodes are
compute capability 9.0 (sm_90) -- NOT B200/sm_100; RuntimeSpec.gpu_arch
defaults to "9.0" for exactly this reason (see the scripts/train_50m.sh bug
this design corrects: TORCH_CUDA_ARCH_LIST=10.0, SKA_REQUIRE_B200=1).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import List

from koopman_lm.run.artifacts import atomic_write_text
from koopman_lm.run.launch import Launcher, build_train_argv, write_model_config
from koopman_lm.run.spec import RunSpec

_SBATCH_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --account={account}
#SBATCH --partition={partition}
#SBATCH --qos={qos}
#SBATCH --nodes={nodes}
#SBATCH --gpus-per-node={gpus}
#SBATCH --time={time_limit}
#SBATCH --output={run_dir}/slurm-%j.out

set -euo pipefail

# Marlowe H100 nodes are compute capability 9.0 (sm_90) -- NOT B200/sm_100.
export TORCH_CUDA_ARCH_LIST="{gpu_arch}"

cd {repo_root}
{launch_line}
"""


class SlurmLauncher(Launcher):
    """Generates run_dir/launch.sbatch and submits it with `sbatch`."""

    def __init__(self, repo_root: str = "."):
        self.repo_root = repo_root

    def build_command(self, spec: RunSpec, run_dir) -> List[str]:
        world_size = spec.runtime.gpus * spec.runtime.nodes
        train_args = build_train_argv(spec, run_dir, world_size=world_size)
        if world_size > 1:
            return ["torchrun", f"--nnodes={spec.runtime.nodes}",
                     f"--nproc_per_node={spec.runtime.gpus}",
                     "-m", "koopman_lm.training.train", *train_args]
        return [sys.executable, "-m", "koopman_lm.training.train", *train_args]

    def render_sbatch(self, spec: RunSpec, run_dir) -> str:
        run_dir = Path(run_dir)
        launch_line = " ".join(self.build_command(spec, run_dir))
        return _SBATCH_TEMPLATE.format(
            job_name=spec.name,
            account=spec.runtime.account,
            partition=spec.runtime.partition,
            qos=spec.runtime.qos,
            nodes=spec.runtime.nodes,
            gpus=spec.runtime.gpus,
            time_limit=spec.runtime.time_limit,
            run_dir=run_dir,
            gpu_arch=spec.runtime.gpu_arch,
            repo_root=self.repo_root,
            launch_line=launch_line,
        )

    def submit(self, spec: RunSpec, run_dir, dry_run: bool = False):
        run_dir = Path(run_dir)
        write_model_config(spec, run_dir)
        sbatch_path = run_dir / "launch.sbatch"
        atomic_write_text(sbatch_path, self.render_sbatch(spec, run_dir))
        if dry_run:
            return sbatch_path
        result = subprocess.run(["sbatch", str(sbatch_path)],
                                 check=True, capture_output=True, text=True)
        return result.stdout.strip()
