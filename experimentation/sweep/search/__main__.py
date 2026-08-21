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

**Parallelism is N of these processes, not a flag.** `--n_jobs` only flips the
sampler's `constant_liar`, which stops N workers proposing identical points; it
spawns nothing. To run a fleet, run this command N times against the same
`storage` journal. Two consequences worth knowing before you do:

  * MedianPruner needs COMPLETED trials, and `n_startup_trials` is
    `min(6, max(3, n_trials // 3))`. Launch 8 workers on a 12-trial study and
    nothing prunes for the entire first wave.
  * Two workers proposing the same params get the same run_id and therefore the
    same run_dir. `constant_liar` makes that unlikely, not impossible;
    `write_policy.claim_run_dir` is what actually makes it safe.
"""
from __future__ import annotations

import argparse
import os
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
    p.add_argument("--run_root", default=None,
                   help="override the spec's run_root (default: $RUN_ROOT or ./runs)")
    p.add_argument("--launcher", choices=("local", "slurm"), default=None,
                   help="override the spec's launcher. 'local' runs trials in "
                        "this process's allocation, which avoids a queue wait "
                        "per trial; 'slurm' submits one job per trial")
    p.add_argument("--dry_run", action="store_true",
                   help="print the plan and materialize nothing -- the GPU-free "
                        "surface, and the way to check a study before spending on it")
    p.add_argument("--n_trials", type=int, default=None,
                   help="override the spec's target study size. This is a TARGET, "
                        "not 'this many more': resuming a 15-trial study that "
                        "finished 10 runs 5")
    p.add_argument("--allow-dirty", dest="allow_dirty", action="store_true",
                   help="launch despite uncommitted changes (recorded as dirty)")
    p.add_argument("--force", action="store_true",
                   help="reuse a run directory that is already claimed")
    p.add_argument("--force-no-eval", dest="force_no_eval", action="store_true",
                   help="proceed even though nothing writes quick_eval.json, so "
                        "every trial will be recorded FAIL. Only useful for "
                        "exercising the machinery")
    return p.parse_args(argv)


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


def _print_plan(spec, args, *, n_anchors, study_dir, run_root, launcher):
    sid = compute_study_id(spec)
    print(f"[search] study        {spec.name}  (study_id {sid})")
    print(f"[search] base spec    {spec.base}")
    print(f"[search] budget       {args.n_trials or spec.n_trials} trials "
          f"x {spec.max_steps} steps")
    print(f"[search] launcher     {launcher}  ->  run_root {run_root}")
    print(f"[search] journal      {study_dir / 'journal.log'}")
    print(f"[search] pruning      after step {spec.prune_after_step}, "
          f"reporting every {spec.logging_steps}")
    if spec.design_file:
        print(f"[search] anchors      {n_anchors} from {spec.design_file}")
    else:
        print("[search] anchors      none -- the first trials are random AND "
              "unprunable (nothing prunes until trials COMPLETE)")
    if spec.n_jobs > 1:
        print(f"[search] n_jobs {spec.n_jobs}: constant_liar ON. This spawns "
              f"nothing -- run this command {spec.n_jobs}x on one journal")
    if spec.objective:
        print(f"[search] objective    loss + {spec.objective}")
    else:
        print("[search] objective    measured loss, nothing else")


def main(argv=None):
    args = parse_args(argv)
    spec = load_study_spec(args.study)

    repo_root = Path(__file__).resolve().parents[3]
    run_root = args.run_root or os.environ.get("RUN_ROOT") or spec.run_root
    launcher_name = args.launcher or spec.launcher
    n_trials = args.n_trials or spec.n_trials

    study_dir = Path(run_root) / "_studies" / f"{spec.name}.{compute_study_id(spec)}"

    base_sections = _base_sections(spec.base)
    base_model = resolve_model_config(base_sections["model"])
    space = search_space(base_model, base_name=spec.base)

    designs = []
    if spec.design_file:
        designs = load_designs(spec.design_file, minimum=1)

    _print_plan(spec, args, n_anchors=len(designs), study_dir=study_dir,
                run_root=run_root, launcher=launcher_name)

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
        study_name=spec.name, study_dir=study_dir, seed=spec.seed,
        n_jobs=spec.n_jobs, prune_after_step=spec.prune_after_step,
        n_trials=n_trials, logging_steps=spec.logging_steps,
        storage_url=spec.storage)

    if designs:
        added = enqueue_anchors(study, designs, base_model, space,
                                base_lr=base_sections["optim"].get("lr", 4e-4))
        print(f"[search] enqueued {added} anchor(s) "
              f"({len(designs) - added} already present)")

    launcher = LocalLauncher() if launcher_name == "local" else SlurmLauncher(
        repo_root=str(repo_root))

    # A FACTORY, not a reader: pruning is trial.report + trial.should_prune, so
    # the reader has to see the trial it is scoring. drive() fixes its kwargs
    # before trial 0 exists, which is why a plain reader can never prune.
    def objective_reader_for(study_, trial):
        def read(run_dir):
            return wait_for_objective(study_, trial, run_dir,
                                      **spec.objective)
        return read

    outcomes = drive(
        study, n_trials=n_trials,
        base_sections=base_sections, base_model=base_model, space=space,
        max_steps=spec.max_steps, run_root=run_root, study_name=spec.name,
        launcher=launcher, read_objective_factory=objective_reader_for,
        base_lr=base_sections["optim"].get("lr", 4e-4),
        backend_policy=spec.backend_policy, seq_len=spec.seq_len,
        force=args.force, dirty=dirty, batch_ladder=spec.batch_ladder)

    states = {}
    for outcome in outcomes:
        states[outcome.state] = states.get(outcome.state, 0) + 1
    print(f"[search] {len(outcomes)} trial(s): "
          + ", ".join(f"{n} {s}" for s, n in sorted(states.items())))

    written = write_report(study, study_dir, base_spec=spec.base,
                           base_model=base_model)
    for name, path in sorted(written.items()):
        if path is not None:
            print(f"[search] wrote {name:16} {path}")
    return 0


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
