"""Run an adaptive search:  python -m experimentation.sweep.search <study.yaml>

The invocation `search/__init__.py` has been advertising all along, and which did
not exist. Everything below it was written and tested; nothing called it.

Shaped after `sweep/__main__.py` deliberately, because the two commands are the
same command with a different way of choosing cells:

    python -m experimentation.sweep <sweep.yaml>          a static grid
    python -m experimentation.sweep.search <study.yaml>   sampled adaptively

Both print their whole plan before touching anything, so `--dry_run` is a complete
GPU-free view; both refuse a dirty tree unless told otherwise; both drive the
unmodified run system, so a trial gets the same content-addressed run_id, shard
verification, atomic materialization and attempts.jsonl that a hand-launched run
gets.

**What this does NOT do yet, and must say so.** Nothing in the launch path writes
`run_dir/eval/<ckpt>/quick_eval.json`, which is where the objective is read from.
`train.py` does not call it and the sbatch template does not either -- the only
caller anywhere is `scripts/verify_search_and_provenance.sbatch`, by hand. So
every trial will train successfully and then be recorded FAIL. That is loud rather
than silent: the CLI checks for it up front and refuses without --force-no-eval,
because burning N GPU jobs to learn nothing is worse than a startup error. The
objective hook lands with the §6.2 TrainTask work.

**Parallelism is `workers:` in the study file, and it launches itself.** Set it
and this command spawns that many processes, pins each to a visible GPU
round-robin, waits, and writes one report. `constant_liar` follows from the same
number, so the sampler always knows the fleet size that actually exists.

That replaces `n_jobs`, which flipped `constant_liar` and launched nothing -- so
a study could declare `n_jobs: 1` while an operator ran 8 workers by hand, and
all 8 proposed from an identical history. The fleet size and the sampler's belief
about it were two numbers that could disagree; now they are one.

**Nothing on the command line redefines the study.** `--run_root`, `--launcher`
and `--n_trials` are gone. They resolved BEFORE `study_id` was computed but were
not part of it, so `--n_trials 100` changed the budget while leaving `study_id`
alone: two workers, one with the flag and one without, shared a journal and
disagreed about how many trials the study was. That is exactly the drift
`studyspec.py` says a committed spec exists to prevent, reintroduced one
argument at a time. What remains are flags that describe this invocation's
intent -- `--dry_run`, `--allow-dirty`, `--force`, `--force-no-eval` -- and none
of them changes what the study IS.

Two consequences of a fleet worth knowing before you raise `workers`:

  * MedianPruner needs COMPLETED trials, and `n_startup_trials` is
    `min(6, max(3, n_trials // 3))`. Launch 8 workers on a 12-trial study and
    nothing prunes for the entire first wave. Keep `workers` well under
    `n_trials` or the study is closer to random search than to TPE.
  * Two workers proposing the same params get the same run_id and therefore the
    same run_dir. `constant_liar` makes that unlikely, not impossible;
    `write_policy.claim_run_dir` is what actually makes it safe.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from experimentation.run.launchers import LocalLauncher, SlurmLauncher
from experimentation.run.provenance import check_git_clean
from experimentation.run.resolve import resolve_model_config
from experimentation.sweep.search.anchors import load_designs
from experimentation.sweep.search.space import search_space
from experimentation.sweep.search.studyspec import (
    load_study_spec, study_id as compute_study_id)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="python -m experimentation.sweep.search",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("study", help="path to configs/search/<name>.yaml")
    # No --run_root / --launcher / --n_trials. Every flag here describes THIS
    # INVOCATION's intent; none of them redefines the study. See the module
    # docstring for why that line exists and what removing those three fixed.
    p.add_argument("--dry_run", action="store_true",
                   help="print the plan and materialize nothing -- the GPU-free "
                        "surface, and the way to check a study before spending on it")
    p.add_argument("--allow-dirty", dest="allow_dirty", action="store_true",
                   help="launch despite uncommitted changes (recorded as dirty)")
    p.add_argument("--force", action="store_true",
                   help="reuse a run directory that is already claimed")
    p.add_argument("--force-no-eval", dest="force_no_eval", action="store_true",
                   help="proceed even though nothing writes quick_eval.json, so "
                        "every trial will be recorded FAIL. Only useful for "
                        "exercising the machinery")
    return p.parse_args(argv)


#: Set by the parent on each child so a worker does not itself fan out. Internal
#: plumbing, not configuration -- it carries no study content, so it cannot be
#: the vector for the worker-disagreement the spec-only rule exists to prevent.
WORKER_ENV = "KOOPMAN_SEARCH_WORKER_INDEX"


def visible_gpus():
    """The GPUs this process may use, as CUDA_VISIBLE_DEVICES-style strings.

    Prefers an inherited `CUDA_VISIBLE_DEVICES` because that is what Slurm sets
    for an allocation: asking the driver instead would return every GPU on the
    node, and workers would be pinned to devices the allocation does not own.
    Falls back to `nvidia-smi -L`, then to nothing.
    """
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env is not None:
        return [d.strip() for d in env.split(",") if d.strip()]
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True,
                             text=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return []
    if out.returncode != 0:
        return []
    return [str(i) for i, line in enumerate(out.stdout.splitlines())
            if line.startswith("GPU ")]


def should_fanout(workers: int, worker_index) -> bool:
    """Does THIS process launch the fleet, or is it one of it?

    A function rather than an inline condition so the worker-0 case is testable
    without running a study. `worker_index` is the raw env value: absent means
    this is the supervisor, and *any* string -- including "0" -- means this is
    already a worker. Deciding on truthiness instead would make worker 0 fan out
    again, giving one extra generation of processes on every launch.
    """
    return workers > 1 and worker_index is None


def _fanout(args, study_spec, *, gpus):
    """Run `study_spec.workers` copies of this command, one per worker.

    Returns (exit_code, per_worker_exit_codes). The parent does NOT run a trial
    itself: a supervisor that is also a worker cannot report on the worker it is,
    and the asymmetry showed up as a report written while one trial was still
    running.
    """
    procs = []
    for index in range(study_spec.workers):
        env = {**os.environ, WORKER_ENV: str(index)}
        if gpus:
            env["CUDA_VISIBLE_DEVICES"] = gpus[index % len(gpus)]
        cmd = [sys.executable, "-m", "experimentation.sweep.search", args.study]
        # Invocation-intent flags propagate; there is nothing else to pass,
        # which is the point of having removed the study-redefining ones.
        if args.allow_dirty:
            cmd.append("--allow-dirty")
        if args.force:
            cmd.append("--force")
        if args.force_no_eval:
            cmd.append("--force-no-eval")
        proc = subprocess.Popen(cmd, env=env)
        procs.append(proc)
        gpu = env.get("CUDA_VISIBLE_DEVICES", "(none)")
        print(f"[search]   worker {index}  GPU {gpu}  pid {proc.pid}")

    codes = [p.wait() for p in procs]
    for index, code in enumerate(codes):
        if code != 0:
            print(f"[search] worker {index} exited {code}")
    return (0 if all(c == 0 for c in codes) else 1), codes


def _objective_producer_exists(repo_root: Path) -> bool:
    """Does anything in the LAUNCH path write the file the objective is read from?

    Checked by looking for a caller rather than by trusting a flag, because the
    failure it prevents is expensive and entirely invisible until the study
    finishes: N trials that trained fine and scored nothing.
    """
    for rel in ("experimentation/training/train.py",
                "experimentation/run/launchers.py"):
        try:
            text = (repo_root / rel).read_text()
        except OSError:
            continue
        if "write_quick_eval" in text or "eval_on_final" in text:
            return True
    return False


def _print_plan(study_spec, args, *, n_anchors, study_dir, run_root, launcher,
                gpus=()):
    sid = compute_study_id(study_spec)
    print(f"[search] study        {study_spec.name}  (study_id {sid})")
    print(f"[search] base spec    {study_spec.base}")
    print(f"[search] budget       {study_spec.n_trials} trials "
          f"x {study_spec.max_steps} steps")
    print(f"[search] launcher     {launcher}  ->  run_root {run_root}")
    # study.py:160 writes optuna_journal.log. Printing a different name here sent
    # me looking for a file that was never going to exist -- and a second worker
    # told to "point at the journal in the plan output" would have opened an empty
    # one and believed it was collaborating.
    print(f"[search] journal      {study_dir / 'optuna_journal.log'}")
    print(f"[search] pruning      after step {study_spec.prune_after_step}, "
          f"reporting every {study_spec.logging_steps}")
    if study_spec.design_file:
        print(f"[search] anchors      {n_anchors} from {study_spec.design_file}")
    else:
        print("[search] anchors      none -- the first trials are random AND "
              "unprunable (nothing prunes until trials COMPLETE)")
    if study_spec.workers > 1:
        placement = (", ".join(f"w{i}->gpu{gpus[i % len(gpus)]}"
                               for i in range(min(study_spec.workers, 4)))
                     + (" ..." if study_spec.workers > 4 else "")) if gpus else \
                    "NO GPU DETECTED -- workers will not be pinned"
        print(f"[search] workers      {study_spec.workers} concurrent "
              f"(constant_liar ON)")
        print(f"[search] placement    {placement}")
        if gpus and study_spec.workers > len(gpus):
            per = study_spec.workers / len(gpus)
            print(f"[search]              {study_spec.workers} workers over "
                  f"{len(gpus)} GPU(s) = {per:.1f} per GPU. Intentional at small "
                  f"model sizes; enable CUDA MPS or they only time-slice.")
        # How many sequential waves the budget buys. This is the number that
        # matters, NOT workers-vs-startup: `n_startup_trials` caps at 6, so any
        # fleet of 6+ would compare unfavourably to it no matter how large the
        # study -- a warning that always fires is one nobody reads.
        waves = study_spec.n_trials / study_spec.workers
        startup = min(6, max(3, study_spec.n_trials // 3))
        print(f"[search]              first wave of {study_spec.workers} is "
              f"proposed with no completed history and cannot be pruned "
              f"(pruner needs {startup} completed)")
        if waves < 5:
            print(f"[search]              WARNING: {study_spec.n_trials} trials / "
                  f"{study_spec.workers} workers = {waves:.1f} sequential wave(s). "
                  f"Most trials are proposed before earlier ones finish, so this "
                  f"is closer to random search than to TPE. Raise n_trials or "
                  f"lower workers.")
    if study_spec.objective:
        print(f"[search] objective    loss + {study_spec.objective}")
    else:
        print("[search] objective    measured loss, nothing else")


def main(argv=None):
    args = parse_args(argv)
    study_spec = load_study_spec(args.study)

    repo_root = Path(__file__).resolve().parents[3]
    # Straight from the spec, with no command-line layer. These three used to be
    # overridable, which let a flag change the budget or the launcher WITHOUT
    # changing study_id -- so two workers could share a journal and disagree.
    run_root = study_spec.run_root
    launcher_name = study_spec.launcher
    n_trials = study_spec.n_trials

    worker_index = os.environ.get(WORKER_ENV)
    is_worker = worker_index is not None
    fanning_out = should_fanout(study_spec.workers, worker_index)

    study_dir = Path(run_root) / "_studies" / f"{study_spec.name}.{compute_study_id(study_spec)}"

    base_sections = _base_sections(study_spec.base)
    base_model = resolve_model_config(base_sections["model"])
    space = search_space(base_model, base_name=study_spec.base)

    designs = []
    if study_spec.design_file:
        designs = load_designs(study_spec.design_file, minimum=1)

    gpus = visible_gpus()
    if not is_worker:
        _print_plan(study_spec, args, n_anchors=len(designs), study_dir=study_dir,
                    run_root=run_root, launcher=launcher_name, gpus=gpus)

    no_objective = not _objective_producer_exists(repo_root)

    # A dry run spends nothing, so it gets the warning rather than the refusal --
    # seeing the plan is exactly what someone checks before committing GPU time,
    # and refusing to print it would make the missing objective harder to
    # discover rather than easier.
    if no_objective:
        # "REFUSING" only when it is actually refusing. Saying it and then
        # launching anyway is the kind of contradiction that teaches people to
        # stop reading the output.
        overridden = args.dry_run or args.force_no_eval
        verb = "WARNING" if overridden else "REFUSING TO LAUNCH"
        print()
        print(f"[search] {verb}: nothing in the launch path writes")
        print("         run_dir/eval/<ckpt>/quick_eval.json, so every trial would")
        print("         train successfully and then be recorded FAIL.")
        print("         The objective hook lands with the TrainTask work; see")
        print("         docs/superpowers/specs/2026-08-21-traintask-design.md §8.")
        if not args.dry_run:
            print("         Pass --force-no-eval to exercise the machinery anyway.")

    if args.dry_run:
        print()
        print("[search] --dry_run: nothing materialized, nothing submitted.")
        return 0

    if no_objective and not args.force_no_eval:
        return 2

    dirty = check_git_clean(allow_dirty=args.allow_dirty)

    # Imported here, not at module scope: everything above this line works with
    # optuna absent, which is what lets a study be authored and inspected on a
    # machine that has never installed it.
    from experimentation.sweep.search.driver import drive
    from experimentation.sweep.search.metrics import wait_for_objective
    from experimentation.sweep.search.report import write_report
    from experimentation.sweep.search.study import create_study, enqueue_anchors

    study = create_study(
        study_name=study_spec.name, study_dir=study_dir, seed=study_spec.seed,
        workers=study_spec.workers, prune_after_step=study_spec.prune_after_step,
        n_trials=n_trials, logging_steps=study_spec.logging_steps,
        storage_url=study_spec.storage)

    if designs:
        added = enqueue_anchors(study, designs, base_model, space,
                                base_lr=base_sections["optim"].get("lr", 4e-4))
        print(f"[search] enqueued {added} anchor(s) "
              f"({len(designs) - added} already present)")

    # eval_on_final=True is what makes a trial scoreable at all: it puts
    # --eval_on_final on the training command, so the run writes
    # run_dir/eval/final/quick_eval.json before exiting and the objective reader
    # finds something. Set on the LAUNCHER rather than the spec, since whether a
    # run scores itself is a property of who launched it -- anything on the spec
    # would be hashed into run_id, and a scored run is not a different experiment.
    scoring = {"eval_on_final": True, "eval_data_dir": study_spec.eval_data_dir}
    launcher = (LocalLauncher(**scoring) if launcher_name == "local"
                else SlurmLauncher(repo_root=str(repo_root), **scoring))

    # A FACTORY, not a reader: pruning is trial.report + trial.should_prune, so
    # the reader has to see the trial it is scoring. drive() fixes its kwargs
    # before trial 0 exists, which is why a plain reader can never prune.
    def objective_reader_for(study_, trial):
        def read(run_dir):
            # timeout_seconds is not optional now that the driver submits with
            # wait=False: a crashed trial no longer raises from submit, so its
            # only remaining symptom is an objective that never arrives.
            return wait_for_objective(
                study_, trial, run_dir,
                timeout_seconds=study_spec.trial_timeout_seconds,
                **study_spec.objective)
        return read

    # The fleet. Anchors are enqueued above by THIS process before any worker
    # starts, so N workers do not race to enqueue the same designs; from here a
    # worker only ever pulls.
    if fanning_out:
        print(f"[search] launching {study_spec.workers} worker(s)")
        code, _codes = _fanout(args, study_spec, gpus=gpus)
        # Reattach rather than reuse: this process built its Study before the
        # workers ran, and its trial list is that stale snapshot. The report has
        # to read the journal again or it describes an empty study.
        study = create_study(
            study_name=study_spec.name, study_dir=study_dir, seed=study_spec.seed,
            workers=study_spec.workers, prune_after_step=study_spec.prune_after_step,
            n_trials=n_trials, logging_steps=study_spec.logging_steps,
            storage_url=study_spec.storage)
        _print_trial_summary(study)
        written = write_report(study, study_dir, base_spec=study_spec.base,
                               base_model=base_model)
        for name, path in sorted(written.items()):
            if path is not None:
                print(f"[search] wrote {name:16} {path}")
        return code

    outcomes = drive(
        study, n_trials=n_trials,
        base_sections=base_sections, base_model=base_model, space=space,
        max_steps=study_spec.max_steps, run_root=run_root, study_name=study_spec.name,
        launcher=launcher, objective_reader_for=objective_reader_for,
        base_lr=base_sections["optim"].get("lr", 4e-4),
        backend_policy=study_spec.backend_policy, seq_len=study_spec.seq_len,
        force=args.force, dirty=dirty, batch_ladder=study_spec.batch_ladder)

    states = {}
    for outcome in outcomes:
        states[outcome.state] = states.get(outcome.state, 0) + 1
    print(f"[search] {len(outcomes)} trial(s): "
          + ", ".join(f"{n} {s}" for s, n in sorted(states.items())))

    _print_failures(study)

    written = write_report(study, study_dir, base_spec=study_spec.base,
                           base_model=base_model)
    for name, path in sorted(written.items()):
        if path is not None:
            print(f"[search] wrote {name:16} {path}")
    return 0


def _print_trial_summary(study):
    """States and grouped failure reasons, read from the journal.

    Split out of the single-worker path because the fleet path has no `outcomes`
    to count -- the trials happened in child processes. Reading the journal is
    the only account of them either way, and it is also the more honest one: it
    reports what was recorded, not what this process believes it launched.
    """
    states = {}
    for trial in study.trials:
        name = getattr(trial.state, "name", str(trial.state))
        states[name] = states.get(name, 0) + 1
    print(f"[search] {len(study.trials)} trial(s): "
          + ", ".join(f"{n} {s}" for s, n in sorted(states.items())))
    _print_failures(study)


def _print_failures(study):
    """The reason, not just the count.

    48 trials failing for 48 different reasons is a rough study; 48 failing for
    ONE reason is a bug in the harness, and the count alone cannot tell them
    apart. `t.state.name` rather than the optuna enum keeps this module
    optuna-free at import time, which is what lets --dry_run work without it.
    """
    failures = [t.user_attrs.get("failure") for t in study.trials
                if getattr(t.state, "name", "") == "FAIL"]
    if not failures:
        return
    tally = {}
    for reason in failures:
        key = reason or "(no reason recorded)"
        tally[key] = tally.get(key, 0) + 1
    for reason, count in sorted(tally.items(), key=lambda kv: -kv[1])[:3]:
        print(f"[search]   failed {count}x: {reason}")


def _base_sections(base_path):
    """The base run spec's sections as plain dicts, ready for field overrides.

    Reuses sweep/spec.py's loader rather than repeating it: both halves of
    sweep/ have to read a base spec the same way, or a study and a static sweep
    over the same base would disagree about what the base said.
    """
    from experimentation.sweep.spec import _base_sections as impl
    return impl(base_path)


if __name__ == "__main__":
    sys.exit(main())
