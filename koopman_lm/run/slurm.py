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
from typing import List, Optional, Tuple

from koopman_lm.run.artifacts import atomic_write_text
from koopman_lm.run.launch import Launcher, build_train_argv, write_model_config
from koopman_lm.run.spec import RunSpec, RuntimeSpec

_SBATCH_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --account={account}
#SBATCH --partition={partition}
#SBATCH --qos={qos}
#SBATCH --nodes={nodes}
#SBATCH --gpus-per-node={gpus}
#SBATCH --time={time_limit}
#SBATCH --signal=B:USR1@300
#SBATCH --requeue
#SBATCH --output={run_dir}/slurm-%j.out

set -euo pipefail

# Marlowe H100 nodes are compute capability 9.0 (sm_90) -- NOT B200/sm_100.
export TORCH_CUDA_ARCH_LIST="{gpu_arch}"

cd {repo_root}
{launch_line}
"""

# One array job indexing into a materialized cell list (§4.2/§4.3): "the
# sweep grid is declared exactly once ... a generator materializes one
# RunSpec per cell, and the array job indexes into that materialized list.
# Bash never knows the grid." cells.txt has one run_dir per line (the array
# index), and every value inside that run_dir's launch_line.sh was already
# expanded by Python (build_train_argv) when koopman_lm.sweep materialized
# it -- this template only does line lookup + exec.
_ARRAY_SBATCH_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --account={account}
#SBATCH --partition={partition}
#SBATCH --qos={qos}
#SBATCH --nodes={nodes}
#SBATCH --gpus-per-node={gpus}
#SBATCH --time={time_limit}
#SBATCH --signal=B:USR1@300
#SBATCH --requeue
#SBATCH --array=0-{max_index}{concurrency_suffix}
#SBATCH --output={sweep_dir}/slurm-%A_%a.out

set -euo pipefail

# Marlowe H100 nodes are compute capability 9.0 (sm_90) -- NOT B200/sm_100.
export TORCH_CUDA_ARCH_LIST="{gpu_arch}"

cd {repo_root}

# Bash never sees the grid: it looks up its row in a file materialized by
# koopman_lm.sweep and execs that row's own fully-expanded command.
CELL_LIST="{cell_list_path}"
RUN_DIR=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$CELL_LIST")
exec bash "$RUN_DIR/launch_line.sh"
"""

# Fields a single #SBATCH resource request must share across every index of
# one array job. If a sweep's axes vary one of these, it cannot be expressed
# as one array job (see _require_uniform_array_runtime).
_ARRAY_RUNTIME_FIELDS = ("partition", "account", "qos", "gpus", "nodes",
                          "time_limit", "gpu_arch")


def _require_uniform_array_runtime(cells: List[Tuple[RunSpec, "Path"]]) -> RuntimeSpec:
    if not cells:
        raise ValueError("cannot render a Slurm array job for an empty cell list")
    first = cells[0][0].runtime
    for spec, _ in cells[1:]:
        for f in _ARRAY_RUNTIME_FIELDS:
            a, b = getattr(first, f), getattr(spec.runtime, f)
            if a != b:
                raise ValueError(
                    f"Slurm array jobs share one #SBATCH resource request "
                    f"across every index, but runtime.{f} varies across this "
                    f"sweep's cells ({a!r} vs {b!r}). Split into separate "
                    f"sweeps/launches, or stop sweeping runtime.{f}.")
    return first


def render_array_sbatch(sweep_name: str, cells: List[Tuple[RunSpec, "Path"]],
                         sweep_dir, *, concurrency: Optional[int] = None,
                         repo_root: str = ".") -> str:
    """Render (but do not write) the array sbatch text for `cells` -- a list
    of (RunSpec, run_dir) pairs already materialized by the caller
    (koopman_lm.sweep). `concurrency` is the array's `%K` cap (a real
    RuntimeSpec-adjacent sweep-spec field, since Marlowe's `batch` partition
    caps at 16 nodes -- an uncapped array can starve every other job on the
    account)."""
    runtime = _require_uniform_array_runtime(cells)
    if concurrency is not None and concurrency < 1:
        raise ValueError("concurrency (the array's %K cap) must be >= 1")
    sweep_dir = Path(sweep_dir)
    concurrency_suffix = f"%{concurrency}" if concurrency else ""
    return _ARRAY_SBATCH_TEMPLATE.format(
        job_name=sweep_name,
        account=runtime.account,
        partition=runtime.partition,
        qos=runtime.qos,
        nodes=runtime.nodes,
        gpus=runtime.gpus,
        time_limit=runtime.time_limit,
        max_index=len(cells) - 1,
        concurrency_suffix=concurrency_suffix,
        sweep_dir=sweep_dir,
        gpu_arch=runtime.gpu_arch,
        repo_root=repo_root,
        cell_list_path=sweep_dir / "cells.txt",
    )


class SlurmLauncher(Launcher):
    """Generates run_dir/launch.sbatch and submits it with `sbatch`."""

    def __init__(self, repo_root: str = "."):
        self.repo_root = repo_root

    def build_command(self, spec: RunSpec, run_dir, *, resume: bool = False) -> List[str]:
        world_size = spec.runtime.gpus * spec.runtime.nodes
        train_args = build_train_argv(spec, run_dir, world_size=world_size, resume=resume)
        if world_size > 1:
            return ["torchrun", f"--nnodes={spec.runtime.nodes}",
                     f"--nproc_per_node={spec.runtime.gpus}",
                     "-m", "koopman_lm.training.train", *train_args]
        return [sys.executable, "-m", "koopman_lm.training.train", *train_args]

    def render_sbatch(self, spec: RunSpec, run_dir, *, resume: bool = False) -> str:
        run_dir = Path(run_dir)
        launch_line = " ".join(self.build_command(spec, run_dir, resume=resume))
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

    def submit(self, spec: RunSpec, run_dir, dry_run: bool = False, *, resume: bool = False):
        run_dir = Path(run_dir)
        write_model_config(spec, run_dir)
        sbatch_path = run_dir / "launch.sbatch"
        atomic_write_text(sbatch_path, self.render_sbatch(spec, run_dir, resume=resume))
        if dry_run:
            return sbatch_path
        result = subprocess.run(["sbatch", str(sbatch_path)],
                                 check=True, capture_output=True, text=True)
        return result.stdout.strip()

    def submit_array(self, sweep_name: str, cells: List[Tuple[RunSpec, "Path"]],
                      sweep_dir, *, concurrency: Optional[int] = None,
                      dry_run: bool = False):
        """One array job for a whole sweep (§4.2/§4.3). `cells` is a list of
        (RunSpec, run_dir) pairs -- one per surviving cell, already
        materialized (spec.yaml, attempts.jsonl) by the caller
        (koopman_lm.sweep). Writes `sweep_dir/cells.txt` (one run_dir per
        line, the array index) plus `run_dir/launch_line.sh` for every cell
        -- each fully expanded, so bash never sees ska_rank/lr/seed/etc, only
        a line number to look up."""
        sweep_dir = Path(sweep_dir)
        run_dirs: List[str] = []
        for spec, run_dir in cells:
            run_dir = Path(run_dir)
            write_model_config(spec, run_dir)
            launch_line = " ".join(self.build_command(spec, run_dir))
            atomic_write_text(run_dir / "launch_line.sh",
                               f"#!/bin/bash\nset -euo pipefail\n{launch_line}\n")
            run_dirs.append(str(run_dir))
        atomic_write_text(sweep_dir / "cells.txt", "\n".join(run_dirs) + "\n")
        text = render_array_sbatch(sweep_name, cells, sweep_dir,
                                    concurrency=concurrency, repo_root=self.repo_root)
        array_path = sweep_dir / "launch_array.sbatch"
        atomic_write_text(array_path, text)
        if dry_run:
            return array_path
        result = subprocess.run(["sbatch", str(array_path)],
                                 check=True, capture_output=True, text=True)
        return result.stdout.strip()
