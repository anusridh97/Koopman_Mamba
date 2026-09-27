"""Read an LR bracket's runs, pick the winner, and report or act.

    python scripts/pick_lr_winner.py <run_dir> <run_dir> ...            # report
    python scripts/pick_lr_winner.py <run_dir> ... --json out.json      # machine-readable

Exists so the campaign does not need a human (or an agent) awake at the moment
the bracket finishes. A Slurm job runs this, and whatever it decides is what
continues -- see scripts/campaign.sh.

DECISION RULE, and why it is not simply argmin:

1. NaN loses, always. A diverged run's loss is NaN and NaN compares false
   against everything, so a plain min() can return it depending on iteration
   order. 4 of 39 scaling-grid runs were NaN, all at the top of their LR range.

2. Ties go to the LOWER learning rate. The comparison happens at ~8% of a
   cosine-to-zero schedule, where the LR is still 98.4% of peak, so it ranks by
   early-phase progress -- which is biased toward higher LRs, because a longer
   horizon favours a lower one. That bias is measured, not assumed: across four
   rungs the optimum fell from >=0.00752 at 10M to 0.00256 at 180M as the
   horizon grew. "Within the noise floor" is sigma = 0.00582, the largest
   per-rung seed sigma measured on this architecture; a gap smaller than 2*sigma
   is not evidence.

3. An unreadable run loses rather than crashing the decision. A bracket arm
   that produced no parseable step line has failed at something, and the other
   arm is still a valid choice.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

#: Largest per-rung seed sigma measured across the four completed sweeps
#: (3M 0.00142, 10M 0.00448, 50M 0.00582, 180M 0.00144). Using the largest is
#: the conservative choice: it makes "indistinguishable" easier to declare, and
#: the tie-break then prefers the safer (lower) learning rate.
SIGMA = 0.00582
STEP_RE = re.compile(
    r"^step\s+(\d+)/(\d+)\s*\|\s*loss\s+(nan|-?[\d.]+)", re.MULTILINE)


def last_loss(run_dir: Path) -> dict:
    """Latest (step, loss) from the run's Slurm log, plus its lr from spec.yaml."""
    out = {"run_dir": str(run_dir), "step": None, "loss": None, "lr": None,
           "note": ""}
    try:
        import yaml
        spec = yaml.safe_load((run_dir / "spec.yaml").read_text())
        out["lr"] = float(spec["optim"]["lr"])
        out["max_steps"] = int(spec["optim"]["max_steps"])
    except Exception as e:
        out["note"] = f"spec unreadable: {type(e).__name__}"
        return out
    best = None
    for log in sorted(run_dir.glob("slurm-*.out")):
        try:
            text = log.read_text(errors="replace")
        except OSError:
            continue
        for m in STEP_RE.finditer(text):
            step, loss = int(m.group(1)), m.group(3)
            val = float("nan") if loss == "nan" else float(loss)
            if best is None or step >= best[0]:
                best = (step, val)
    if best is None:
        out["note"] = "no parseable step line -- run produced no progress"
        return out
    out["step"], out["loss"] = best
    if out["loss"] != out["loss"]:
        out["note"] = "DIVERGED (nan)"
    return out


def pick(rows: list) -> tuple:
    """(winner, reason). Finite losses only; ties within 2*SIGMA go lower-lr."""
    live = [r for r in rows
            if r["loss"] is not None and r["loss"] == r["loss"]]
    if not live:
        return None, "every arm diverged or produced no progress"
    live.sort(key=lambda r: r["loss"])
    best = live[0]
    close = [r for r in live if r["loss"] - best["loss"] < 2 * SIGMA]
    if len(close) > 1:
        winner = min(close, key=lambda r: r["lr"])
        gaps = ", ".join(f"lr={r['lr']:g}:{r['loss']:.5f}" for r in close)
        return winner, (f"{len(close)} arms within 2*sigma={2*SIGMA:.5f} ({gaps}); "
                        f"took the LOWER lr={winner['lr']:g} -- an 8% ranking is "
                        f"biased toward higher lr, and the ladder measured the "
                        f"optimum FALLING as the horizon grew")
    return best, (f"lr={best['lr']:g} wins by "
                  f"{live[1]['loss'] - best['loss']:.5f} over the next arm "
                  f"(> 2*sigma={2*SIGMA:.5f}), so the gap is resolvable")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+", type=Path)
    ap.add_argument("--json", type=Path, default=None)
    a = ap.parse_args(argv)

    rows = [last_loss(d) for d in a.run_dirs]
    print("%-10s %8s %12s  %s" % ("lr", "step", "loss", "note"))
    for r in sorted(rows, key=lambda x: (x["lr"] is None, x["lr"] or 0)):
        lr = f"{r['lr']:g}" if r["lr"] is not None else "?"
        st = str(r["step"]) if r["step"] is not None else "-"
        ls = ("nan" if r["loss"] is not None and r["loss"] != r["loss"]
              else (f"{r['loss']:.5f}" if r["loss"] is not None else "-"))
        print("%-10s %8s %12s  %s" % (lr, st, ls, r["note"]))
    winner, reason = pick(rows)
    print()
    if winner is None:
        print(f"NO WINNER: {reason}")
        if a.json:
            a.json.write_text(json.dumps({"winner": None, "reason": reason,
                                          "rows": rows}, indent=2))
        return 1
    print(f"WINNER lr={winner['lr']:g}  (loss {winner['loss']:.5f} at step {winner['step']})")
    print(f"  because: {reason}")
    if a.json:
        a.json.write_text(json.dumps({"winner": winner, "reason": reason,
                                      "rows": rows}, indent=2))
        print(f"  wrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
