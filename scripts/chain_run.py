"""Submit one run as a chain of dependent Slurm jobs, to cross a wall-clock cap.

    python scripts/chain_run.py <run_dir> --chunks 3          # after the first submit
    python scripts/chain_run.py <run_dir> --chunks 3 --dry_run

WHY. `batch` caps at 2 days (`MaxTime=2-00:00:00`) and the 1.5B run is ~87 h at
64 GPUs, so it cannot be one job. `#SBATCH --requeue` does NOT cover this:
Slurm requeues on preemption and node failure, but a job that hits its TIME
LIMIT goes to TIMEOUT and stays there. A dependency chain submitted up front is
unattended and needs no polling.

HOW IT IS SAFE. The run's launch line already carries `--resume_if_available`
(`run/train_argv.py`), so chunk 2 picks up `resume.pt` -- written every
`save_steps` (capped at 2,000, ~16 min at 440M) and again on SIGUSR1, which
`--signal=B:USR1@300` delivers 300 s before the kill. `afterany` rather than
`afterok` because the predecessor exits non-zero on a timeout kill, and that is
exactly the case the successor exists to continue.

THE GUARD MATTERS. A chain sized for the worst case has leftover chunks when the
run finishes early. Without a guard each one would start, load a resume.pt
already at max_steps, fall straight out of `while step < args.max_steps`, and
then re-enter the finalisation path -- re-writing `final/` and, if
eval-on-final were ever enabled, re-scoring it. The guard makes a leftover chunk
exit 0 in under a second instead.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

GUARD = '''
# Inserted by scripts/chain_run.py. A chain is sized for the worst case, so the
# last chunks are usually unnecessary; exit before doing any work rather than
# resuming a finished run and re-writing final/.
if [ -f "{run_dir}/final/model.pt" ]; then
  echo "final/model.pt already present -- run complete, nothing to do"
  exit 0
fi
'''


def build_chain_sbatch(run_dir: Path) -> Path:
    """launch.sbatch + the completion guard, written beside it."""
    src = run_dir / "launch.sbatch"
    if not src.is_file():
        raise SystemExit(f"no launch.sbatch in {run_dir} -- materialize the run first "
                         f"(python -m experimentation.run <spec> --launcher slurm)")
    text = src.read_text()
    # Insert after the last #SBATCH/export block but before the launch line.
    # The launch line is the one naming the trainer module.
    lines = text.splitlines(keepends=True)
    for i, line in enumerate(lines):
        if "experimentation.training.train" in line:
            break
    else:
        raise SystemExit("could not find the launch line in launch.sbatch")
    out = "".join(lines[:i]) + GUARD.format(run_dir=run_dir) + "".join(lines[i:])
    dest = run_dir / "chain.sbatch"
    dest.write_text(out)
    dest.chmod(0o755)
    return dest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--chunks", type=int, required=True,
                    help="TOTAL jobs in the chain, including one already submitted "
                         "if --after is given")
    ap.add_argument("--after", type=str, default=None,
                    help="job id of an already-running first chunk to chain behind")
    ap.add_argument("--account", default="marlowe-m000151-pm06")
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args(argv)

    run_dir = a.run_dir.resolve()
    sbatch = build_chain_sbatch(run_dir)
    print(f"chain sbatch: {sbatch}")
    n_new = a.chunks - (1 if a.after else 0)
    if n_new < 1:
        print("nothing to submit")
        return 0

    prev, ids = a.after, []
    for k in range(n_new):
        cmd = ["sbatch", "--parsable", f"--account={a.account}"]
        if prev:
            cmd.append(f"--dependency=afterany:{prev}")
        cmd.append(str(sbatch))
        if a.dry_run:
            print(f"  (dry run) chunk {k + 1 + (1 if a.after else 0)}: {' '.join(cmd)}")
            prev = f"<chunk{k}>"
            continue
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  chunk {k}: REFUSED {r.stderr.strip()[:160]}")
            break
        prev = r.stdout.strip()
        ids.append(prev)
        print(f"  chunk {k + 1 + (1 if a.after else 0)}: {prev}"
              f"{' after ' + ids[-2] if len(ids) > 1 else (' after ' + a.after if a.after else '')}")
    if ids:
        print(f"\nsubmitted {len(ids)} dependent chunk(s). Each resumes from "
              f"resume.pt; leftovers exit 0 once final/model.pt exists.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
