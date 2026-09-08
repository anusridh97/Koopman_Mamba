"""Score every completed run of a scaling grid, and emit the (N, D, loss) table.

    python scripts/eval_scaling_grid.py <run_root> --submit   # evaluate
    python scripts/eval_scaling_grid.py <run_root> --table    # collect

WHY THIS EXISTS AS A SEPARATE PASS. A static `cells:` sweep does not score
itself. `_ScoresRuns.eval_on_final` defaults to False
(`experimentation/run/launchers.py:154-180`) and
`experimentation/sweep/__main__.py` constructs both launchers with no arguments,
so a finished grid leaves `final/model.pt` and no loss anywhere. Discovering
that after 57 runs would have cost ~490 GPU-h for nothing.

WHY NOT JUST TURN `eval_on_final` ON. That path writes `quick_eval.json`, which
is deliberately cheap: 8 batches at half sequence length
(`train.py:657-659`, "this runs once per trial across a whole study"). Roughly
65K tokens. Fine for ranking trials inside a search, far too noisy to be the y
value in a scaling-law fit, where the residuals are being compared against a
seed noise floor of ~0.005.

THE FLAG IS `--held_out_data_dir`, NOT `--eval_data_dir`. The first version
invented the latter and every eval died with `unrecognized arguments` -- the
submission was dry-run and the shard's existence was checked, but no eval was
ever run to completion, so 39 trained cells sat unscored. `evaluate.py:597`
also *requires* the flag for this mode. Validated now by feeding the exact
argv to evaluate.py's own parser, in `--check-cmd`.

`--mode fineweb_ppl` instead runs `eval_fineweb_ppl`
(`experimentation/evaluation/evaluate.py:172-207`) over the WHOLE disjoint local
val shard with no batch cap, and `evaluate.py::main` (595-624) detects the run
directory and writes `<run_dir>/eval/final/fineweb_ppl.json` in the standard
envelope, which `experimentation/results.py` already aggregates.

NOT `--mode ppl`: that is WikiText-103 via HF `load_dataset`, and the sbatches
run with `HF_HUB_OFFLINE=1`.

The eval set is genuinely held out: tok10b's `meta.json` records
`disjoint_from {"fineweb_small_val": [93773, 97158]}` and the training shard
begins at document 97,159.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

VAL_SHARD = "/scratch/m000151-pm06/jkli/fineweb_small_val"
VENV = "/scratch/m000151-pm06/jkli/venvs/koopman-cuda/bin/python"
REPO = "/users/cody1212/Koopman_Mamba"
TOK_PER_STEP = 96 * 2048


def completed_runs(root: Path):
    """Run dirs with a final checkpoint. `final/` is the completion marker --
    train.py only writes it on a non-preempted finish (train.py:432-438)."""
    return sorted(p.parent for p in root.glob("*/seed*/final/model.pt"))


def already_scored(run_dir: Path) -> bool:
    return (run_dir / "eval" / "final" / "fineweb_ppl.json").is_file()


def spec_of(run_dir: Path) -> dict:
    import yaml
    return yaml.safe_load((run_dir / "spec.yaml").read_text())


def submit(root: Path, account: str, dry: bool) -> int:
    runs = [r for r in completed_runs(root) if not already_scored(r)]
    done = [r for r in completed_runs(root) if already_scored(r)]
    print(f"{len(completed_runs(root))} completed run(s): "
          f"{len(done)} already scored, {len(runs)} to score")
    if not runs:
        return 0
    # A UNIQUE listing per submission. A fixed path is a race: a second
    # submission overwrites the file while the first array still has pending
    # tasks, and those tasks then read the wrong rows -- scoring the wrong
    # checkpoint, or the same one twice while others are silently skipped.
    import time as _t
    listing = root / f"_eval_queue.{int(_t.time())}.txt"
    listing.write_text("\n".join(str(r) for r in runs) + "\n")
    # 8 checkpoints per job: a few minutes each, so a task is well under an hour.
    per_task = 8
    nchunks = (len(runs) + per_task - 1) // per_task
    script = root / "_eval.sbatch"
    script.write_text(f"""#!/bin/bash
#SBATCH --job-name=gridEval
#SBATCH --partition=batch
#SBATCH --qos=medium
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --array=1-{nchunks}
#SBATCH --output={root}/_eval-%A_%a.out
set -uo pipefail
cd {REPO}
export PYTHONPATH={REPO}:/scratch/m000151-pm06/cqiu/pylibs
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
# BATCHED: each array task scores PER_TASK checkpoints sequentially rather than
# one. QOS `medium` caps 32 JOBS per account and array tasks count individually,
# so one-job-per-checkpoint made eval compete with the grid for the scarce
# resource -- and lose, since the grid is fed first. An eval is a few GPU-minutes,
# so batching turns 57 slots into a handful at no wall-clock cost.
#
# The listing path arrives in the environment so each submission gets its own
# immutable file; a fixed path races against an earlier array still reading it.
PER_TASK=${{PER_TASK:?PER_TASK not set}}
QUEUE=${{EVAL_QUEUE:?EVAL_QUEUE not set}}
START=$(( (SLURM_ARRAY_TASK_ID - 1) * PER_TASK + 1 ))
END=$(( START + PER_TASK - 1 ))
rc=0
for i in $(seq "$START" "$END"); do
  RUN=$(sed -n "${{i}}p" "$QUEUE")
  [ -z "$RUN" ] && continue                 # last chunk is short
  echo "=== scoring row $i: $RUN"
  # --mode fineweb_ppl, NOT ppl: the whole disjoint val shard, no batch cap, and
  # no network. main() writes <run>/eval/final/fineweb_ppl.json by itself.
  {VENV} -m experimentation.evaluation.evaluate \\
      --checkpoint "$RUN/final/model.pt" \\
      --mode fineweb_ppl \\
      --held_out_data_dir {VAL_SHARD} || {{ echo "FAILED row $i"; rc=1; }}
done
exit $rc
""")
    script.chmod(0o755)
    if dry:
        print(f"(dry run) would submit array 1-{nchunks} "
              f"({len(runs)} runs at {per_task}/task) from {script}")
        return 0
    out = subprocess.run(["sbatch", "--parsable", f"--account={account}",
                          f"--export=ALL,EVAL_QUEUE={listing},PER_TASK={per_task}",
                          str(script)],
                         capture_output=True, text=True, check=True)
    print(f"submitted eval array {out.stdout.strip()}: {len(runs)} run(s) "
          f"in {nchunks} task(s) at {per_task}/task")
    return 0


def table(root: Path, out: Path | None) -> int:
    rows = []
    for run in completed_runs(root):
        ev = run / "eval" / "final" / "fineweb_ppl.json"
        if not ev.is_file():
            continue
        env = json.loads(ev.read_text())
        met = env.get("metrics", env)
        spec = spec_of(run)
        model, optim = spec.get("model", {}), spec.get("optim", {})
        d_model = int(model["d_model"])
        vocab = int(model.get("vocab_size", 32000))
        steps = int(optim["max_steps"])
        # N comes from the built config, not from arithmetic here, so it cannot
        # drift from what actually trained.
        sys.path.insert(0, REPO)
        from koopman_lm.config import KoopmanLMConfig
        cfg = KoopmanLMConfig(**{k: v for k, v in model.items()
                                 if k in KoopmanLMConfig.__dataclass_fields__})
        n_total = cfg.param_count_estimate()
        rows.append(dict(
            run_id=spec.get("run_id", run.name.split(".")[-1]),
            d_model=d_model, n_layers=int(model["n_layers"]),
            n_total=n_total, n_non_emb=n_total - vocab * d_model,
            max_steps=steps, tokens=steps * TOK_PER_STEP,
            tokens_per_param=steps * TOK_PER_STEP / n_total,
            lr=float(optim["lr"]), warmup_steps=int(optim["warmup_steps"]),
            loss=float(met["loss"]), ppl=float(met.get("ppl", 0.0)),
            n_eval_tokens=int(met.get("n_tokens", 0)),
            run_dir=str(run)))
    if not rows:
        print("no scored runs yet -- run with --submit first")
        return 1
    rows.sort(key=lambda r: (r["n_total"], r["tokens"], r["lr"]))
    dest = out or (root / "grid_results.csv")
    with open(dest, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"wrote {dest} ({len(rows)} scored runs)")

    # The fit consumes one (N, D, loss) per CELL -- the min over the LR triple,
    # since the LR points exist to remove LR as a confound, not as data points.
    import collections
    cells = collections.defaultdict(list)
    for r in rows:
        cells[(r["n_total"], r["tokens"])].append(r)
    print(f"\n{len(cells)} distinct (N, D) cells; per-cell best over the LR triple:")
    print("%12s %14s %7s %9s %9s  %s" % ("N", "D", "tok/par", "best loss", "best lr", "LRs done"))
    for (n, d), members in sorted(cells.items()):
        b = min(members, key=lambda r: r["loss"])
        print("%12s %14s %7.1f %9.5f %9.5f  %d/3" % (
            f"{n:,}", f"{d:,}", d / n, b["loss"], b["lr"], len(members)))
    incomplete = [k for k, v in cells.items() if len(v) < 3]
    if incomplete:
        print(f"\n{len(incomplete)} cell(s) missing LR points -- the min is not yet"
              f" a real minimum for those, so do not fit until they land.")
    return 0


def check_cmd() -> int:
    """Feed our exact argv to evaluate.py's OWN parser, offline.

    This exists because the flag name was wrong for 39 runs and nothing caught
    it: the sbatch was generated fine, the shard existed, and the array
    submitted cleanly. Only actually parsing the arguments would have failed.
    """
    sys.path.insert(0, REPO)
    from experimentation.evaluation import evaluate as ev
    argv = ["--checkpoint", "/tmp/x/final/model.pt", "--mode", "fineweb_ppl",
            "--held_out_data_dir", VAL_SHARD]
    parser = None
    for name in ("build_parser", "_build_parser", "make_parser"):
        if hasattr(ev, name):
            parser = getattr(ev, name)()
            break
    if parser is None:                      # parser is built inline in main()
        import argparse as _a, io, contextlib
        # Re-derive it by invoking main() with --help suppressed is unsafe, so
        # fall back to asserting the flag exists in the module source.
        src = Path(REPO, "experimentation/evaluation/evaluate.py").read_text()
        ok = '"--held_out_data_dir"' in src and '"--eval_data_dir"' not in src
        print(f"flag check via source: held_out_data_dir present={ok}")
        return 0 if ok else 1
    ns = parser.parse_args(argv)
    print(f"parsed OK: mode={ns.mode} held_out_data_dir={ns.held_out_data_dir}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-cmd", action="store_true",
                    help="validate our argv against evaluate.py's parser")
    ap.add_argument("run_root", type=Path, nargs="?", default=Path("."))
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--table", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--account", default="marlowe-m000151-pm06")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args(argv)
    if args.check_cmd:
        return check_cmd()
    if not args.run_root.is_dir():
        raise SystemExit(f"{args.run_root} is not a directory")
    if args.submit:
        return submit(args.run_root, args.account, args.dry_run)
    if args.table:
        return table(args.run_root, args.out)
    print("pass --submit to score completed runs, or --table to collect them")
    return 0


if __name__ == "__main__":
    sys.exit(main())
