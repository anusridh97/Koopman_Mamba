"""Launchers (§3.4): the spec says *what* to run, the launcher says *where*.

launch.py's Launcher ABC + LocalLauncher merged with all of slurm.py. They were
split across two files, which forced slurm.py to import three names from a module
it was otherwise unrelated to; the actual seam was never local-vs-slurm but
"build the command" (now run/train_argv.py) versus "run it somewhere" (here).

Marlowe H100 nodes are compute capability 9.0 (sm_90) -- NOT B200/sm_100;
RuntimeSpec.gpu_arch defaults to "9.0" for exactly this reason (see the
scripts/train_50m.sh bug this design corrects: TORCH_CUDA_ARCH_LIST=10.0,
SKA_REQUIRE_B200=1).
"""
from __future__ import annotations

import abc
import os
import signal
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from experimentation.atomic_io import atomic_write_text
from experimentation.run.spec import RunSpec, RuntimeSpec
from experimentation.run.train_argv import build_train_argv, write_model_config

__all__ = ["Launcher", "LocalLauncher", "SlurmLauncher", "render_array_sbatch"]


_SBATCH_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --account={account}
#SBATCH --partition={partition}
{qos_directive}
#SBATCH --nodes={nodes}
{ntasks_directive}{gpu_directive}
#SBATCH --time={time_limit}
{signal_directive}#SBATCH --requeue
#SBATCH --output={run_dir}/slurm-%j.out

set -euo pipefail

# Marlowe H100 nodes are compute capability 9.0 (sm_90) -- NOT B200/sm_100.
export TORCH_CUDA_ARCH_LIST="{gpu_arch}"

# Triton JIT cache MUST be node-local. Its default is $HOME/.triton/cache,
# which is shared NFS here, so on a multi-node job every rank compiles the same
# kernels into the same directory at once and they tear each other's files out
# from under themselves: "FileNotFoundError: ..._layer_norm_bwd_kernel.ttir"
# and "OSError: [Errno 116] Stale file handle: ..._chunk_state_fwd_kernel.cubin".
# That killed ska-1p5b-matched (job 497324) 99 seconds in, after all 64 ranks
# had built the model successfully. Earlier multi-node runs survived only
# because their kernel variants were already warm in that cache; a new geometry
# forces a fresh compile and loses the race. $HOME is also small here.
export TRITON_CACHE_DIR="${{TMPDIR:-/tmp}}/triton-cache-${{SLURM_JOB_ID:-local}}"
mkdir -p "$TRITON_CACHE_DIR"

cd {repo_root}
{launch_line}
"""

# One array job indexing into a materialized cell list (§4.2/§4.3): "the
# sweep grid is declared exactly once ... a generator materializes one
# RunSpec per cell, and the array job indexes into that materialized list.
# Bash never knows the grid." cells.txt has one run_dir per line (the array
# index), and every value inside that run_dir's launch_line.sh was already
# expanded by Python (build_train_argv) when experimentation.sweep materialized
# it -- this template only does line lookup + exec.
_ARRAY_SBATCH_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --account={account}
#SBATCH --partition={partition}
{qos_directive}
#SBATCH --nodes={nodes}
{ntasks_directive}{gpu_directive}
#SBATCH --time={time_limit}
{signal_directive}#SBATCH --requeue
#SBATCH --array=0-{max_index}{concurrency_suffix}
#SBATCH --output={sweep_dir}/slurm-%A_%a.out

set -euo pipefail

# Marlowe H100 nodes are compute capability 9.0 (sm_90) -- NOT B200/sm_100.
export TORCH_CUDA_ARCH_LIST="{gpu_arch}"

# Triton JIT cache MUST be node-local. Its default is $HOME/.triton/cache,
# which is shared NFS here, so on a multi-node job every rank compiles the same
# kernels into the same directory at once and they tear each other's files out
# from under themselves: "FileNotFoundError: ..._layer_norm_bwd_kernel.ttir"
# and "OSError: [Errno 116] Stale file handle: ..._chunk_state_fwd_kernel.cubin".
# That killed ska-1p5b-matched (job 497324) 99 seconds in, after all 64 ranks
# had built the model successfully. Earlier multi-node runs survived only
# because their kernel variants were already warm in that cache; a new geometry
# forces a fresh compile and loses the race. $HOME is also small here.
export TRITON_CACHE_DIR="${{TMPDIR:-/tmp}}/triton-cache-${{SLURM_JOB_ID:-local}}"
mkdir -p "$TRITON_CACHE_DIR"

cd {repo_root}

# Bash never sees the grid: it looks up its row in a file materialized by
# experimentation.sweep and execs that row's own fully-expanded command.
CELL_LIST="{cell_list_path}"
RUN_DIR=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$CELL_LIST")
exec bash "$RUN_DIR/launch_line.sh"
"""

# Fields a single #SBATCH resource request must share across every index of
# one array job. If a sweep's axes vary one of these, it cannot be expressed
# as one array job (see _require_uniform_array_runtime).
_ARRAY_RUNTIME_FIELDS = ("partition", "account", "qos", "gpus", "nodes",
                          "time_limit", "gpu_arch")

_ANVIL_PARTITIONS = {"ai", "gpu", "gpu-debug"}


def _slurm_resource_directives(runtime: RuntimeSpec) -> Tuple[str, str]:
    """The site-specific spellings Slurm needs, without site-specific paths.

    Anvil exposes GPUs through ``--gres=gpu:N`` and commonly has no QoS.  The
    Marlowe partitions use ``--gpus-per-node=N``.  Environment activation is
    deliberately not embedded here: the generated command already uses the
    driver's absolute ``sys.executable``, while hard-coding one user's checkout
    or CUDA module would make the launcher unusable for everyone else.
    """
    gpu = (f"#SBATCH --gres=gpu:{runtime.gpus}"
           if runtime.partition in _ANVIL_PARTITIONS
           else f"#SBATCH --gpus-per-node={runtime.gpus}")
    qos = f"#SBATCH --qos={runtime.qos}" if runtime.qos else ""
    return gpu, qos


#: CPUs a Marlowe H100 node has per GPU (112 cores / 8 GPUs). Used only to size
#: --cpus-per-task for the multi-node path; single-node requests are left
#: exactly as they were, since that is the topology every completed run used.
_CPUS_PER_GPU = 14


def _signal_directive(runtime: RuntimeSpec) -> str:
    """Deliver SIGUSR1 to whatever process can actually act on it.

    `B:` signals ONLY the batch shell. With srun, the trainer runs in a
    separate job step that never sees it -- and bash's default action for
    SIGUSR1 is to terminate, so the multi-node resume smoke died with
    ExitCode 0:10 at 4:48 of a 10-minute limit and wrote no resume.pt at all.
    Dropping `B:` sends the signal to the job STEPS, i.e. srun's tasks, which
    are the python processes carrying install_sigusr1_handler.

    Single-node keeps `B:`. There the trainer is a direct child of the batch
    shell and that form is what every completed run used, including a verified
    exact-resume comparison (job 440211); this is not the moment to re-derive
    it from the man page.
    """
    return ("#SBATCH --signal=USR1@300\n" if runtime.nodes > 1
            else "#SBATCH --signal=B:USR1@300\n")


def _ntasks_directive(runtime: RuntimeSpec) -> str:
    """One Slurm task per NODE when multi-node, plus the CPUs that task needs.

    --ntasks-per-node=1 is what makes srun start exactly one torchrun agent per
    node, so c10d sees the node count it was promised.

    --cpus-per-task MUST accompany it. Slurm defaults cpus-per-task to 1, and
    this partition sets DefMemPerCPU=13000, so a 2-node job was allocated
    NumCPUs=2 and 13 GB of host RAM PER NODE -- for 8 GPU processes plus their
    dataloader workers. Single-node runs never hit this because they leave
    ntasks unset and are not squeezed into one task's cgroup; the grid's
    4-GPU array sbatch carried no CPU request at all and ran fine.

    Empty for single-node runs, which keep the proven --standalone path and
    must not gain directives that have never been exercised there.
    """
    # CPUs are needed whenever a job holds MORE THAN ONE GPU, single-node
    # included. Slurm defaults cpus-per-task to 1 and this partition sets
    # DefMemPerCPU=13000, so a 1-node 4-GPU pilot was granted AllocCPUS=1 and
    # 13 GB of HOST ram for four GPU processes plus their dataloader workers,
    # memmapping a 400 GB shard -- and was oom_killed at MaxRSS 12.6 GB with
    # zero CUDA OOMs. An earlier version of this function gated the CPU request
    # on nodes>1 to "leave the proven single-node path untouched"; the proven
    # path was small models on a 20 GB shard, which is not this.
    directive = ""
    if runtime.gpus > 1:
        directive += f"#SBATCH --cpus-per-task={runtime.gpus * _CPUS_PER_GPU}\n"
    # One TASK per node only when multi-node, so srun starts exactly one
    # torchrun agent per node and c10d sees the node count it was promised.
    if runtime.nodes > 1:
        directive = "#SBATCH --ntasks-per-node=1\n" + directive
    return directive


def _submit_sbatch(path: Path) -> str:
    """Submit one generated script and retain Slurm's actionable error text."""
    try:
        result = subprocess.run(["sbatch", str(path)], check=True,
                                capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "no Slurm error output").strip()
        raise RuntimeError(
            f"Slurm rejected generated script {path}: {detail}") from exc
    return result.stdout.strip()


class Launcher(abc.ABC):
    """The spec says *what* to run; the launcher says *where* (§3.4)."""

    @abc.abstractmethod
    def build_command(self, spec: RunSpec, run_dir, *, resume: bool = False) -> List[str]:
        ...

    @abc.abstractmethod
    def submit(self, spec: RunSpec, run_dir, dry_run: bool = False, *,
               resume: bool = False, wait: bool = True):
        """Start this run. ``wait=False`` means "return before it finishes".

        On the ABC because a caller has to be able to say it without knowing
        which launcher it holds. The adaptive search needs it: pruning tails a
        RUNNING job's log, so a submit that blocks leaves nothing to prune.

        Slurm is already asynchronous -- sbatch returns a job id immediately --
        so SlurmLauncher accepts the flag and ignores it. Local execution is the
        one that has to change behaviour.
        """

    def wait_for_exit(self, handle, run_dir):
        """Finish a non-blocking local hand-off before a worker reuses a GPU.

        Slurm owns process lifetime and resource release, so its default is a
        no-op.  LocalLauncher overrides this to reap the child whose handle was
        returned by ``submit(wait=False)``.
        """
        del handle, run_dir
        return None


class _ScoresRuns:
    """Mixin: should a launcher ask its runs to score themselves?

    Set at CONSTRUCTION, not per submit, and deliberately not on the RunSpec.
    Whether a run writes quick_eval.json is a property of who launched it, not of
    the experiment -- the same spec launched by hand and by a study has to keep
    the same run_id, and anything on the spec would be hashed into it. A scored
    run and an unscored run of the same config are the same run.
    """

    eval_on_final: bool = False
    eval_data_dir = None
    #: How often the trainer prints a progress line, or None for its own default.
    #:
    #: Here for the same reason as `eval_on_final`: how often a run PRINTS is a
    #: property of who launched it, and anything on the spec would be hashed into
    #: run_id -- two runs differing only in log verbosity are the same experiment.
    #:
    #: A study MUST set it. `metrics.read_progress` parses those lines and the
    #: pruner's `interval_steps` is consulted on the same cadence, so the two have
    #: to be ONE number; they were two, and the pruner was being asked about steps
    #: no trial had reported.
    logging_steps = None

    def _scoring_kwargs(self):
        return {"eval_on_final": self.eval_on_final,
                "eval_data_dir": self.eval_data_dir,
                "logging_steps": self.logging_steps}


class LocalLauncher(Launcher, _ScoresRuns):
    """Runs training in-process via subprocess: `python -m ...` for a single
    GPU, `torchrun --standalone` when runtime.ddp and runtime.gpus > 1."""

    def __init__(self, *, eval_on_final: bool = False, eval_data_dir=None,
                 logging_steps=None):
        self.eval_on_final = eval_on_final
        self.eval_data_dir = eval_data_dir
        self.logging_steps = logging_steps

    def build_command(self, spec: RunSpec, run_dir, *, resume: bool = False) -> List[str]:
        world_size = spec.runtime.gpus if (spec.runtime.ddp and spec.runtime.gpus > 1) else 1
        train_args = build_train_argv(spec, run_dir, world_size=world_size, resume=resume,
                              **self._scoring_kwargs())
        if world_size > 1:
            # `sys.executable -m torch.distributed.run`, not the bare `torchrun`
            # console script. The two are identical -- that script's whole body
            # is `from torch.distributed.run import main; main()` -- but the
            # binary lives in the venv's bin/ and is NOT on PATH inside a Slurm
            # worker, so `torchrun` raised FileNotFoundError and failed 25 of 25
            # trials in seconds. The single-GPU branch below already uses the
            # absolute sys.executable; this makes the DDP branch agree.
            return [sys.executable, "-m", "torch.distributed.run",
                    "--standalone", f"--nproc_per_node={world_size}",
                     "-m", "experimentation.training.train", *train_args]
        return [sys.executable, "-m", "experimentation.training.train", *train_args]

    #: Written next to the run's artifacts when a run is launched with
    #: ``wait=False``, so a later poller can cancel it without having been handed
    #: the process object. Mirrors how the Slurm path is cancellable from the
    #: run directory alone (via its ``slurm-*.out`` filenames).
    PID_FILE = ".local_pid"

    #: Where a non-blocking local run's stdout goes, so a poller can tail it.
    #: Must match one of metrics.py's _LOG_GLOBS ("slurm-*.out", "*.log") or the
    #: pruner cannot see it.
    LOG_FILE = "train.log"

    # quick_eval.json is written just before the trainer exits.  Waiting a few
    # seconds is normal; five minutes indicates cleanup is stuck and the process
    # must not be left holding a CUDA context while the next trial starts.
    EXIT_TIMEOUT_SECONDS = 300

    def submit(self, spec: RunSpec, run_dir, dry_run: bool = False, *,
               resume: bool = False, wait: bool = True):
        """Run training here. Blocking by default, and that default is load-bearing.

        ``run/__main__.py`` documents relying on it (its claim is released before
        hand-off precisely *because* this blocks for the whole run), and
        ``sweep/__main__.py`` submits one cell at a time expecting each to finish
        before the next starts. Flipping the default would launch every cell of a
        sweep simultaneously.

        ``wait=False`` returns a live ``Popen`` instead, for the one caller that
        needs to watch a run rather than await it: the adaptive search prunes by
        tailing a running job's log, and with a blocking submit the objective
        reader only ever starts *after* training finished, so there was nothing
        left to prune. That made pruning silently dead for any locally-launched
        trial -- which is the mode a held GPU allocation uses to avoid paying a
        queue wait per trial.

        ``start_new_session=True`` puts the child in its own process group so the
        whole tree can be signalled. A ``torchrun`` launch spawns one worker per
        GPU, and killing only the parent would orphan them still holding the GPUs
        -- pruning that does not release the hardware saves nothing.
        """
        write_model_config(spec, run_dir)
        cmd = self.build_command(spec, run_dir, resume=resume)
        if dry_run:
            return cmd
        if wait:
            # Inherit stdout: a human running this watches the terminal.
            return subprocess.run(cmd, check=True)

        # Not waiting means somebody intends to WATCH this run, and the only
        # channel for that is a file in the run directory (metrics.py's module
        # docstring: nothing there holds a subprocess or a pipe). Slurm gets this
        # for free -- its sbatch template routes stdout to run_dir/slurm-%j.out --
        # and local runs had no log at all, so read_progress's globs matched
        # nothing and PRUNING COULD NEVER FIRE. Measured: job 439827 completed 4
        # trials with real objectives and "trials with reported intermediate
        # steps: 0", because there was nothing to parse.
        #
        # Asymmetric with the wait=True branch on purpose: an interactive run
        # wants its output on the terminal, a watched run wants it on disk.
        log_path = Path(run_dir) / self.LOG_FILE
        log = open(log_path, "w", buffering=1)          # line-buffered
        # PYTHONUNBUFFERED because `buffering=1` above only affects THIS
        # process's file object -- the child inherits a raw fd and does its own
        # buffering, and CPython block-buffers at 8 KB when stdout is a file. A
        # 200-step trial emits ~1.4 KB, so without this the log is empty until the
        # child exits and pruning has nothing to tail. The trainers also pass
        # flush=True on their progress line; this is the belt to that braces, so a
        # new trainer that forgets does not silently disable pruning again.
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        try:
            proc = subprocess.Popen(cmd, start_new_session=True, env=env,
                                    stdout=log, stderr=subprocess.STDOUT)
        finally:
            # The child inherited its own fd.  Keeping the parent's file object
            # open for every trial leaks descriptors across a long study.
            log.close()
        atomic_write_text(Path(run_dir) / self.PID_FILE, f"{proc.pid}\n")
        return proc

    def wait_for_exit(self, handle, run_dir):
        """Reap a watched trainer so its CUDA context is gone before reuse."""
        if not isinstance(handle, subprocess.Popen):
            return None
        try:
            return handle.wait(timeout=self.EXIT_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired as exc:
            # ``submit(wait=False)`` starts a new session specifically so the
            # whole torchrun tree can be stopped together.
            for sig, timeout in ((signal.SIGTERM, 30), (signal.SIGKILL, 10)):
                try:
                    os.killpg(handle.pid, sig)
                except ProcessLookupError:
                    break
                try:
                    handle.wait(timeout=timeout)
                    break
                except subprocess.TimeoutExpired:
                    continue
            raise RuntimeError(
                f"local trainer pid {handle.pid} produced an objective but did "
                f"not exit within {self.EXIT_TIMEOUT_SECONDS}s; terminated it "
                f"instead of starting another trial on the same GPU") from exc
        finally:
            if handle.poll() is not None:
                try:
                    (Path(run_dir) / self.PID_FILE).unlink()
                except FileNotFoundError:
                    pass


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
    (experimentation.sweep). `concurrency` is the array's `%K` cap (a real
    RuntimeSpec-adjacent sweep-spec field, since Marlowe's `batch` partition
    caps at 16 nodes -- an uncapped array can starve every other job on the
    account)."""
    runtime = _require_uniform_array_runtime(cells)
    if concurrency is not None and concurrency < 1:
        raise ValueError("concurrency (the array's %K cap) must be >= 1")
    sweep_dir = Path(sweep_dir)
    concurrency_suffix = f"%{concurrency}" if concurrency else ""
    gpu_directive, qos_directive = _slurm_resource_directives(runtime)
    return _ARRAY_SBATCH_TEMPLATE.format(
        gpu_directive=gpu_directive,
        qos_directive=qos_directive,
        ntasks_directive=_ntasks_directive(runtime),
        signal_directive=_signal_directive(runtime),
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


class SlurmLauncher(Launcher, _ScoresRuns):
    """Generates run_dir/launch.sbatch and submits it with `sbatch`."""

    def __init__(self, repo_root: str = ".", *, eval_on_final: bool = False,
                 eval_data_dir=None, logging_steps=None):
        self.repo_root = repo_root
        self.eval_on_final = eval_on_final
        self.eval_data_dir = eval_data_dir
        self.logging_steps = logging_steps

    def build_command(self, spec: RunSpec, run_dir, *, resume: bool = False) -> List[str]:
        world_size = spec.runtime.gpus * spec.runtime.nodes
        train_args = build_train_argv(spec, run_dir, world_size=world_size, resume=resume,
                              **self._scoring_kwargs())
        if spec.runtime.nodes > 1:
            # MULTI-NODE. The previous form -- `torch.distributed.run
            # --nnodes=N` with no srun and no rendezvous -- could never work,
            # for two independent reasons. An sbatch body runs on the FIRST
            # allocated node only, so exactly one agent was ever started while
            # --nnodes=N told it to expect N; and the default rendezvous is
            # static, needing --node_rank/--master_addr that nothing supplied.
            # The job would sit at 0% until the wall clock killed it. srun
            # starts one agent per node (hence --ntasks-per-node=1, added to
            # the sbatch template below) and c10d gives them a real meeting
            # point, elected on the first host of the allocation.
            #
            # The port is derived from the job id because c10d binds it on the
            # rank-0 HOST: two of our jobs sharing that host would otherwise
            # both try to listen on one fixed port and the loser would hang.
            # ABSOLUTE srun, resolved now. srun lives in
            # /cm/shared/apps/slurm/current/bin, and an sbatch body runs under
            # `#!/bin/bash` WITHOUT sourcing /etc/profile.d -- the same reason
            # the bare `torchrun` console script raised FileNotFoundError and
            # failed 25 of 25 trials in 22 seconds while Slurm cheerfully
            # reported COMPLETED 0:0. Do not trust PATH for an interpreter or a
            # launcher; `which` runs on the login node, which shares
            # /cm/shared with the compute nodes.
            srun = shutil.which("srun") or "srun"
            return [srun, f"--ntasks={spec.runtime.nodes}",
                    "--ntasks-per-node=1", "--kill-on-bad-exit=1",
                    sys.executable, "-m", "torch.distributed.run",
                    f"--nnodes={spec.runtime.nodes}",
                    f"--nproc_per_node={spec.runtime.gpus}",
                    # Forward SIGUSR1 to the WORKERS. srun delivers the
                    # signal to its task, which is the torchrun agent, not the
                    # trainer -- and the agent's default handled set is
                    # SIGTERM,SIGINT,SIGHUP,SIGQUIT, so USR1 killed the agent
                    # outright (the resume smoke died ExitCode 10:0 at 6:59 of
                    # a 12-minute limit with no resume.pt). Listing it makes
                    # _terminate_process_handler raise SignalException carrying
                    # sigval, which the agent passes to close(death_sig=...) --
                    # i.e. the workers get SIGUSR1 and train.py's handler runs.
                    #
                    # BEST EFFORT, NOT A GUARANTEE: close() SIGKILLs after a
                    # 30s timeout, and a 1.5B resume.pt is ~20 GB, which may not
                    # finish inside that window. atomic_torch_save writes to a
                    # temp file and renames, so a killed write leaves the
                    # PREVIOUS resume.pt intact rather than a truncated one.
                    # The thing we actually rely on is the periodic save every
                    # `save_steps` (2,000 -> ~33 min at 440M, ~55 min at 1.5B).
                    "--signals-to-handle=SIGTERM,SIGINT,SIGHUP,SIGQUIT,SIGUSR1",
                    "--rdzv_backend=c10d",
                    '--rdzv_id="$SLURM_JOB_ID"',
                    '--rdzv_endpoint="$(scontrol show hostnames'
                    ' "$SLURM_JOB_NODELIST" | head -n1)":'
                    '"$((29500 + SLURM_JOB_ID % 20000))"',
                    "-m", "experimentation.training.train", *train_args]
        if world_size > 1:
            # Single node, several GPUs: --standalone, which is proven on this
            # cluster and pins the rendezvous to localhost. Absolute
            # interpreter, for the reason given in LocalLauncher above:
            # `torchrun` is not on PATH in a Slurm step.
            return [sys.executable, "-m", "torch.distributed.run",
                    "--standalone", f"--nproc_per_node={world_size}",
                     "-m", "experimentation.training.train", *train_args]
        return [sys.executable, "-m", "experimentation.training.train", *train_args]

    def render_sbatch(self, spec: RunSpec, run_dir, *, resume: bool = False) -> str:
        run_dir = Path(run_dir)
        launch_line = " ".join(self.build_command(spec, run_dir, resume=resume))
        gpu_directive, qos_directive = _slurm_resource_directives(spec.runtime)
        return _SBATCH_TEMPLATE.format(
            gpu_directive=gpu_directive,
            qos_directive=qos_directive,
            ntasks_directive=_ntasks_directive(spec.runtime),
            signal_directive=_signal_directive(spec.runtime),
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

    def submit(self, spec: RunSpec, run_dir, dry_run: bool = False, *,
               resume: bool = False, wait: bool = True):
        # `wait` is accepted and ignored: sbatch returns a job id as soon as the
        # job is QUEUED, so this launcher is asynchronous whether or not anyone
        # asks. Accepting it lets a caller pass wait=False to any launcher
        # without first asking which one it holds.
        del wait
        run_dir = Path(run_dir)
        write_model_config(spec, run_dir)
        sbatch_path = run_dir / "launch.sbatch"
        atomic_write_text(sbatch_path, self.render_sbatch(spec, run_dir, resume=resume))
        if dry_run:
            return sbatch_path
        return _submit_sbatch(sbatch_path)

    def submit_array(self, sweep_name: str, cells: List[Tuple[RunSpec, "Path"]],
                      sweep_dir, *, concurrency: Optional[int] = None,
                      dry_run: bool = False):
        """One array job for a whole sweep (§4.2/§4.3). `cells` is a list of
        (RunSpec, run_dir) pairs -- one per surviving cell, already
        materialized (spec.yaml, attempts.jsonl) by the caller
        (experimentation.sweep). Writes `sweep_dir/cells.txt` (one run_dir per
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
        return _submit_sbatch(array_path)
