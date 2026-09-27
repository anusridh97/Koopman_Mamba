"""Regression tests for the two defects that gate any multi-day, multi-node run.

Each test reconstructs one real failure mode.

1. MULTI-NODE DDP NEVER WORKED. SlurmLauncher emitted
   `torch.distributed.run --nnodes=N --nproc_per_node=G` with no srun, no
   rendezvous backend and no node rank. An sbatch body runs on the FIRST
   allocated node only, so exactly one agent started while --nnodes=N told it to
   expect N, and the default rendezvous is static and needs a --master_addr
   nothing supplied. The job would sit at 0% until the wall clock killed it.

2. A REQUEUED JOB RESTARTED AT STEP 0. Nothing passed --resume, and train.py
   raises SystemExit when --resume is given without a resume.pt -- so the
   launch line could neither carry --resume nor omit it. Slurm re-runs the
   original sbatch verbatim on requeue, so a preemption, a node failure or a
   --dependency chain across `batch`'s 2-day limit silently discarded every
   completed step.

These assert argv[0] is ABSOLUTE AND EXISTS rather than merely checking argv
shape. Shape is what let the bare-`torchrun` bug through: the command list
looked perfectly well-formed, `torchrun` simply was not on PATH inside a Slurm
step, and 25 of 25 trials died in 22 seconds while Slurm reported COMPLETED 0:0.
"""
from __future__ import annotations

import dataclasses
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from experimentation.run import resolve
from experimentation.run.launchers import SlurmLauncher, render_array_sbatch
from experimentation.run.train_argv import build_train_argv
from experimentation.training.train import prune_step_checkpoints

BASE = "configs/runs/scaling/ska-base.yaml"


def _spec(nodes, gpus, *, eb=256, pdbs=4):
    spec = resolve.resolve_run_spec(BASE)
    rt = dataclasses.replace(spec.runtime, nodes=nodes, gpus=gpus,
                             ddp=(nodes * gpus > 1), per_device_batch_size=pdbs)
    op = dataclasses.replace(spec.optim, effective_batch=eb)
    return dataclasses.replace(spec, runtime=rt, optim=op)


# ---------------------------------------------------------------- multi-node

@pytest.mark.parametrize("nodes,gpus", [(2, 8), (4, 8), (8, 8)])
def test_multinode_uses_srun_and_c10d(nodes, gpus):
    cmd = SlurmLauncher(repo_root=".").build_command(_spec(nodes, gpus), "/tmp/rd")
    joined = " ".join(cmd)
    assert os.path.basename(cmd[0]) == "srun", (
        "multi-node must launch through srun: an sbatch body runs on node 0 "
        f"only, so one agent would wait forever for {nodes - 1} peers")
    assert f"--nnodes={nodes}" in cmd
    assert f"--nproc_per_node={gpus}" in cmd
    assert "--rdzv_backend=c10d" in cmd, (
        "the default rendezvous is static and needs --master_addr/--node_rank")
    assert "--standalone" not in cmd, "--standalone pins rendezvous to localhost"
    assert "SLURM_JOB_NODELIST" in joined, "endpoint must be elected from the allocation"


@pytest.mark.parametrize("nodes,gpus", [(1, 1), (1, 4), (1, 8), (2, 8), (8, 8)])
def test_argv0_is_absolute_and_executable(nodes, gpus):
    """The actual bug: a resolvable-looking name that PATH could not resolve.

    Applies to srun too -- it lives in /cm/shared/apps/slurm/current/bin and an
    sbatch body runs under `#!/bin/bash`, which does not source /etc/profile.d.
    """
    cmd = SlurmLauncher(repo_root=".").build_command(_spec(nodes, gpus), "/tmp/rd")
    assert os.path.isabs(cmd[0]), f"argv[0]={cmd[0]!r} is not absolute"
    assert os.path.exists(cmd[0]), f"argv[0]={cmd[0]!r} does not exist"
    assert os.access(cmd[0], os.X_OK), f"argv[0]={cmd[0]!r} is not executable"


def test_single_node_keeps_the_proven_standalone_path():
    cmd = SlurmLauncher(repo_root=".").build_command(_spec(1, 4), "/tmp/rd")
    assert "--standalone" in cmd
    assert os.path.basename(cmd[0]) != "srun"
    assert "--rdzv_backend=c10d" not in cmd


@pytest.mark.parametrize("nodes,expect", [(1, False), (2, True), (8, True)])
def test_ntasks_per_node_directive_only_when_multinode(nodes, expect):
    sb = SlurmLauncher(repo_root=".").render_sbatch(_spec(nodes, 8), "/tmp/rd")
    assert ("--ntasks-per-node=1" in sb) is expect
    arr = render_array_sbatch(
        "sw", [(_spec(nodes, 8), Path("/tmp/a")), (_spec(nodes, 8), Path("/tmp/b"))],
        "/tmp/sw", repo_root=".")
    assert ("--ntasks-per-node=1" in arr) is expect


@pytest.mark.parametrize("nodes,gpus", [(2, 8), (8, 8), (4, 4)])
def test_multinode_requests_cpus_for_the_whole_node(nodes, gpus):
    """--ntasks-per-node=1 without --cpus-per-task is a 1-CPU allocation.

    This partition sets DefMemPerCPU=13000, so the first 2-node submission was
    granted NumCPUs=2 and 13 GB of host RAM per node -- for 8 GPU processes and
    their dataloader workers. Nodes have 112 CPUs and 1950 GB.
    """
    sb = SlurmLauncher(repo_root=".").render_sbatch(_spec(nodes, gpus), "/tmp/rd")
    assert "--ntasks-per-node=1" in sb
    line = [l for l in sb.splitlines() if "--cpus-per-task" in l]
    assert line, "multi-node must request CPUs explicitly"
    assert int(line[0].split("=")[1]) == gpus * 14


@pytest.mark.parametrize("nodes,gpus,cpus", [(1, 4, 56), (1, 8, 112), (2, 8, 112)])
def test_any_multi_gpu_job_requests_cpus(nodes, gpus, cpus):
    """Host RAM is derived from CPUs, and 1 CPU is 13 GB on this partition.

    A 1-node 4-GPU pilot was granted AllocCPUS=1 / 13000M and oom_killed at
    MaxRSS 12.6 GB with ZERO CUDA OOMs -- four GPU processes plus dataloader
    workers cannot share 13 GB while memmapping a 400 GB shard. Single-node is
    not exempt.
    """
    sb = SlurmLauncher(repo_root=".").render_sbatch(_spec(nodes, gpus), "/tmp/rd")
    line = [l for l in sb.splitlines() if l.startswith("#SBATCH --cpus-per-task")]
    assert line, f"nodes={nodes} gpus={gpus} must request CPUs"
    assert int(line[0].split("=")[1]) == cpus
    # ntasks-per-node stays multi-node-only: it is what makes srun start one
    # agent per node, and a 1-task single-node job must not gain it.
    assert ("--ntasks-per-node=1" in sb) is (nodes > 1)


def test_single_gpu_job_requests_no_cpus():
    """One GPU, one process: the default allocation is what every completed
    single-GPU run used, and there is no reason to perturb it."""
    sb = SlurmLauncher(repo_root=".").render_sbatch(_spec(1, 1, eb=96, pdbs=12),
                                                    "/tmp/rd")
    assert "--cpus-per-task" not in sb
    assert "--ntasks-per-node" not in sb


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
@pytest.mark.parametrize("nodes,gpus", [(1, 1), (1, 4), (2, 8), (8, 8)])
def test_rendered_sbatch_is_valid_bash(nodes, gpus):
    """Syntax-check the rendered script with bash itself.

    The multi-node launch line nests a command substitution and arithmetic
    expansion inside a quoted --rdzv_endpoint. Asserting the Python list looks
    right cannot catch a quoting error there; only a shell parser can, and this
    is the boundary the earlier 'double check' did not test -- it graded
    artifacts it had produced instead of handing them to code it did not write.
    """
    sb = SlurmLauncher(repo_root=".").render_sbatch(_spec(nodes, gpus), "/tmp/rd")
    with tempfile.NamedTemporaryFile("w", suffix=".sbatch", delete=False) as f:
        f.write(sb)
        path = f.name
    try:
        r = subprocess.run(["bash", "-n", path], capture_output=True, text=True)
        assert r.returncode == 0, f"bash rejected the rendered sbatch:\n{r.stderr}"
    finally:
        os.unlink(path)


# -------------------------------------------------------------------- resume

def test_fresh_launch_emits_resume_if_available():
    """Neither --resume nor nothing: the first form hard-exits on attempt 1,
    the second silently restarts a multi-day run on attempt 2."""
    with tempfile.TemporaryDirectory() as d:
        argv = build_train_argv(_spec(1, 1, eb=96, pdbs=12), d, world_size=1,
                                resume=False)
        assert "--resume_if_available" in argv
        assert "--resume" not in argv


def test_explicit_resume_is_unambiguous():
    with tempfile.TemporaryDirectory() as d:
        argv = build_train_argv(_spec(1, 1, eb=96, pdbs=12), d, world_size=1,
                                resume=True)
        assert "--resume" in argv
        assert "--resume_if_available" not in argv, (
            "both flags at once leaves which one wins up to argparse order")


def test_resume_if_available_flag_exists_in_trainer_cli():
    """The launcher and the trainer must agree on the spelling. A flag the
    launcher emits and the parser rejects is `unrecognized arguments` at
    runtime -- exactly how --eval_data_dir killed 39 evals."""
    from experimentation.training import train as T
    p = T.build_arg_parser() if hasattr(T, "build_arg_parser") else None
    if p is None:
        src = Path("experimentation/training/train.py").read_text()
        assert '"--resume_if_available"' in src
    else:
        ns = p.parse_args(["--resume_if_available"])
        assert ns.resume_if_available is True


@pytest.mark.parametrize("max_steps,expect", [
    (3, 1),            # tiny run: a third of it, at least one
    (300, 100),        # short run: still a third
    (190_735, 2000),   # 100B tokens: the cap binds, not max_steps//3
])
def test_save_steps_is_a_cadence_not_a_fraction(max_steps, expect):
    """max_steps//3 is a fraction of the RUN; what interrupts a run is a
    fraction of the CLOCK. At 190,735 steps a third is one save per ~29 h --
    longer than `batch`'s 2-day limit and 7x `preempt`'s 4-hour cap."""
    spec = _spec(1, 1, eb=96, pdbs=12)
    spec = dataclasses.replace(spec, optim=dataclasses.replace(
        spec.optim, max_steps=max_steps, warmup_steps=1))
    with tempfile.TemporaryDirectory() as d:
        argv = build_train_argv(spec, d, world_size=1)
        got = int(argv[argv.index("--save_steps") + 1])
    assert got == expect


# --------------------------------------------------------------------- prune

def test_prune_keeps_newest_two_and_spares_everything_else():
    with tempfile.TemporaryDirectory() as d:
        for n in (100, 2000, 30000, 4000):
            os.makedirs(os.path.join(d, f"step_{n}"))
            Path(d, f"step_{n}", "model.pt").write_text("x")
        for keep in ("final", "eval", "step_abc", "resume.pt"):
            os.makedirs(os.path.join(d, keep), exist_ok=True)
        removed = prune_step_checkpoints(d)
        left = sorted(os.listdir(d))
        assert "step_30000" in left and "step_4000" in left, "newest two must survive"
        assert "step_100" not in left and "step_2000" not in left
        assert sorted(removed) == ["step_100", "step_2000"]
        for keep in ("final", "eval", "step_abc", "resume.pt"):
            assert keep in left, f"{keep} is not a step_<N>/ dir and must be spared"


def test_prune_is_a_noop_below_the_keep_threshold():
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "step_7"))
        assert prune_step_checkpoints(d) == []
        assert os.path.isdir(os.path.join(d, "step_7"))


# ------------------------------------------------------------- dependency chain

def _materialized_run_dir(tmp: Path, nodes=2, gpus=2) -> Path:
    from experimentation.run import resolve as _r
    rd = tmp / "rd"
    rd.mkdir()
    spec = _spec(nodes, gpus)
    L = SlurmLauncher(repo_root=".")
    (rd / "launch.sbatch").write_text(L.render_sbatch(spec, rd))
    return rd


def test_chain_guard_goes_before_the_launch_line_and_after_sbatch():
    """Placement is the whole correctness of the guard.

    Inside the #SBATCH block Slurm stops parsing directives at the first
    non-comment line, so the resource request would be silently truncated.
    After the launch line it never runs, and a leftover chunk resumes a finished
    run and re-enters finalisation.
    """
    from scripts.chain_run import build_chain_sbatch
    with tempfile.TemporaryDirectory() as d:
        rd = _materialized_run_dir(Path(d))
        text = build_chain_sbatch(rd).read_text()
        lines = text.splitlines()
        guard = next(i for i, l in enumerate(lines) if "final/model.pt" in l)
        launch = next(i for i, l in enumerate(lines)
                      if "experimentation.training.train" in l)
        last_sbatch = max(i for i, l in enumerate(lines) if l.startswith("#SBATCH"))
        assert last_sbatch < guard < launch, (
            f"guard at {guard} must sit after the last #SBATCH ({last_sbatch}) "
            f"and before the launch line ({launch})")
        assert "exit 0" in text


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_chained_sbatch_is_valid_bash_and_keeps_resume_flag():
    from scripts.chain_run import build_chain_sbatch
    with tempfile.TemporaryDirectory() as d:
        rd = _materialized_run_dir(Path(d))
        dest = build_chain_sbatch(rd)
        r = subprocess.run(["bash", "-n", str(dest)], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        text = dest.read_text()
        assert "--resume_if_available" in text, (
            "a chained chunk that cannot resume restarts the run at step 0")
        # every #SBATCH directive of the original must survive
        orig = {l for l in (rd / "launch.sbatch").read_text().splitlines()
                if l.startswith("#SBATCH")}
        assert orig <= set(text.splitlines())


def test_chain_refuses_an_unmaterialized_run_dir():
    from scripts.chain_run import build_chain_sbatch
    with tempfile.TemporaryDirectory() as d:
        with pytest.raises(SystemExit):
            build_chain_sbatch(Path(d))


# ------------------------------------------------- global vs local rank (multi-node)

def test_distributed_sampler_partitions_by_global_rank():
    """The bug the 2-node smoke exposed, stated as data coverage.

    DistributedSampler(num_replicas=W, rank=r) gives replica r the indices
    r, r+W, r+2W, ... So the union over all replicas covers the dataset exactly
    once ONLY if the ranks are 0..W-1 distinct. torchrun numbers LOCAL_RANK
    within a node, so on 8 nodes x 8 GPUs the local ranks are {0..7} repeated
    eight times: every node reads the same eighth, 7/8 of the shard is never
    trained on, and each sample is seen 8 times in a nominal single epoch.
    """
    from torch.utils.data.distributed import DistributedSampler

    n_nodes, per_node = 8, 8
    world = n_nodes * per_node
    ds = list(range(4096))

    def covered(ranks):
        seen = []
        for r in ranks:
            s = DistributedSampler(ds, num_replicas=world, rank=r,
                                   shuffle=False)
            seen.extend(list(s))
        return seen

    global_ranks = list(range(world))
    local_ranks = [r % per_node for _ in range(n_nodes) for r in range(per_node)]

    g, l = covered(global_ranks), covered(local_ranks)
    assert len(set(g)) == len(ds), "global ranks must cover the dataset exactly once"
    assert len(g) == len(set(g)), "no sample may be repeated within an epoch"
    # and the broken numbering demonstrably does not
    assert len(set(l)) == len(ds) // per_node, (
        f"local-rank numbering covers {len(set(l))} of {len(ds)} samples")
    assert max(l.count(i) for i in set(l)) == n_nodes


def test_train_uses_global_rank_for_replica_identity_and_local_for_devices():
    """Device placement wants LOCAL rank; replica identity wants GLOBAL.

    Asserted against the source because the alternative is standing up a real
    multi-node process group in a unit test. The three replica-identity uses
    are: who writes checkpoints, which data shard, which seed offset.
    """
    src = Path("experimentation/training/train.py").read_text()
    assert 'global_rank = int(os.environ.get("RANK", local_rank))' in src
    assert "is_main = global_rank == 0" in src, (
        "with local_rank, one process PER NODE writes resume.pt")
    assert "seed_everything(args.seed + global_rank)" in src
    assert "rank=global_rank, shuffle=True" in src
    # device placement must NOT have been switched to the global rank
    assert "torch.cuda.set_device(local_rank)" in src
    assert "device_ids=[local_rank]" in src


def test_loop_sampler_rebuild_defaults_to_local_rank_when_unset():
    """The seq_len curriculum rebuilds the sampler inside the loop. Its
    global_rank defaults to None -> local_rank, so single-node callers are
    bit-identical to before; the two numbers are equal there anyway."""
    src = Path("experimentation/training/loop.py").read_text()
    assert "global_rank: Optional[int] = None," in src
    assert "rank=(local_rank if global_rank is None else global_rank)," in src


# ---------------------------------------------------------------- SIGUSR1 delivery

@pytest.mark.parametrize("nodes,expect", [
    (1, "#SBATCH --signal=B:USR1@300"),
    (2, "#SBATCH --signal=USR1@300"),
    (8, "#SBATCH --signal=USR1@300"),
])
def test_sigusr1_reaches_the_process_that_handles_it(nodes, expect):
    """`B:` signals ONLY the batch shell, which under srun is not the trainer.

    The multi-node resume smoke died ExitCode 0:10 at 4:48 of a 10-minute
    limit having written no resume.pt: bash received SIGUSR1, whose default
    action is to terminate, while the python processes in srun's job step --
    the ones carrying install_sigusr1_handler -- never saw it.

    Single-node must keep `B:`; that is the form every completed run used,
    including a verified exact-resume comparison.
    """
    sb = SlurmLauncher(repo_root=".").render_sbatch(_spec(nodes, 2), "/tmp/rd")
    # Anchored on "#SBATCH --signal=" because the multi-node launch line also
    # carries --signals-to-handle, which contains "--signal" as a substring.
    sig = [l for l in sb.splitlines() if l.startswith("#SBATCH --signal=")]
    assert sig == [expect], f"got {sig}"


def test_only_one_signal_directive_is_emitted():
    """Two --signal lines would leave which one wins to Slurm's parser."""
    for nodes in (1, 2, 8):
        sb = SlurmLauncher(repo_root=".").render_sbatch(_spec(nodes, 2), "/tmp/rd")
        assert len([l for l in sb.splitlines()
                    if l.startswith("#SBATCH --signal=")]) == 1


def test_multinode_forwards_sigusr1_to_workers():
    """srun's task is the torchrun AGENT, not the trainer.

    The agent's default handled set is SIGTERM,SIGINT,SIGHUP,SIGQUIT, so a
    forwarded SIGUSR1 killed it outright and no resume.pt was written. Listing
    SIGUSR1 makes the agent pass it through to the workers as the death signal,
    where train.py's handler can act on it.
    """
    cmd = SlurmLauncher(repo_root=".").build_command(_spec(2, 2), "/tmp/rd")
    flag = [a for a in cmd if a.startswith("--signals-to-handle")]
    assert flag, "multi-node must forward SIGUSR1 past the torchrun agent"
    assert "SIGUSR1" in flag[0]
    # the defaults must be preserved, not replaced
    for sig in ("SIGTERM", "SIGINT", "SIGHUP", "SIGQUIT"):
        assert sig in flag[0], f"{sig} dropped from the handled set"


def test_single_node_does_not_gain_the_signal_flag():
    """--standalone is the proven path; it must not acquire an untested flag."""
    cmd = SlurmLauncher(repo_root=".").build_command(_spec(1, 4), "/tmp/rd")
    assert not [a for a in cmd if a.startswith("--signals-to-handle")]


def test_all_ranks_leave_the_loop_on_preemption_not_just_main():
    """Gating the break on is_main deadlocks the other ranks.

    The torchrun agent signals every worker, so all ranks set the flag -- but
    with `if is_main and ...: break`, ranks 1..N-1 continued into the next
    allreduce and blocked on a rank 0 that had already exited. torchrun
    SIGKILLed them after its 30s timeout and the job was recorded FAILED (15:0)
    despite resume.pt being written correctly (observed at step 61 of the
    2-node smoke).
    """
    src = Path("experimentation/training/loop.py").read_text()
    assert "if preempt_flag is not None and preempt_flag.is_set():" in src, (
        "the preemption check must not be gated on is_main")
    i = src.index("if preempt_flag is not None and preempt_flag.is_set():")
    block = src[i:i + 1400]
    # rank 0 still does the writing, inside the block
    assert "if is_main:" in block and "_save_all(" in block
    # and the break is reached by every rank
    assert "preempted = True" in block and "break" in block
    gated = block.index("if is_main:")
    brk = block.index("preempted = True")
    assert gated < brk, "the save must be inside, the break outside, the is_main guard"
