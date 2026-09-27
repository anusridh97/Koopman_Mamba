"""Stage 2: read the two-width LR sweep, fit the trend, launch everything else.

Run by campaign_decide.sh as a Slurm job with a dependency on all ten sweep
points, so no human has to be awake when the sweep ends.

WHY A SWEEP AT TWO WIDTHS AND NOT A BRACKET AT THE TARGET. An earlier plan
bracketed the 1.5B directly with two points centred on our 180M optimum
(0.00256). That optimum is the ONLY uncensored LR measurement we have -- the 3M,
10M and 50M optima were all at their search ceilings (>=0.00541, >=0.00752,
>=0.00483) -- and it was measured at effective_batch 96 and 65 tokens/parameter,
whereas this protocol is eb 256 and 545 tok/param. Two points on a one-point
prior from a different protocol picks an endpoint rather than locating an
optimum.

Sweeping 180M (6 points) and 440M (4 points) instead costs ~744 GPU-h, 9% of the
campaign, and yields an INTERIOR optimum at two widths on the protocol we are
actually running. The 1.5B is then extrapolated one step (576 -> 832 measured,
1280 predicted) AND still verified with a 2-point bracket at its own width.

THE FIT IS DELIBERATELY HUMBLE. Two points define a single exponent p in
lr ~ width^p, with no residual and therefore no uncertainty estimate. It is
reported, not trusted: the 1.5B bracket straddles it, and if an endpoint wins
that is the signal the exponent is wrong in that direction.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from pick_lr_winner import last_loss, pick  # noqa: E402  (same scripts/ dir)

WIDTH = {"180m": 576, "440m": 832, "1p5b": 1280}


def sweep_winner(run_root: Path, size: str) -> dict:
    dirs = sorted(run_root.glob(f"mamba3-{size}-lr*/seed42.*"))
    if not dirs:
        raise SystemExit(f"no sweep points for {size} under {run_root}")
    rows = [last_loss(d) for d in dirs]
    win, reason = pick(rows)
    if win is None:
        raise SystemExit(f"{size}: {reason}")
    lrs = sorted(r["lr"] for r in rows if r["loss"] is not None
                 and r["loss"] == r["loss"])
    # An optimum at either end of the swept range is a LOWER/UPPER BOUND, not a
    # location -- exactly the censoring that made three of four ladder rungs
    # unusable for a transfer rule. Flag it loudly rather than fitting through it.
    interior = len(lrs) > 2 and lrs[0] < win["lr"] < lrs[-1]
    return {"size": size, "width": WIDTH[size], "lr": win["lr"],
            "loss": win["loss"], "step": win["step"], "reason": reason,
            "interior": interior, "swept": lrs,
            "n_finite": len(lrs), "n_total": len(rows)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_root", type=Path,
                    default=Path("/scratch/m000151-pm06/cqiu/mamba3/prod"))
    ap.add_argument("--shard", type=Path,
                    default=Path("/scratch/m000151-pm06/cqiu/tok100b/shard"))
    ap.add_argument("--acct", default="marlowe-m000151-pm06")
    ap.add_argument("--straddle", type=float, default=1.33,
                    help="1.5B bracket = extrapolated lr / f and * f")
    ap.add_argument("--submit", action="store_true")
    a = ap.parse_args(argv)

    small = sweep_winner(a.run_root, "180m")
    mid = sweep_winner(a.run_root, "440m")
    for w in (small, mid):
        flag = "" if w["interior"] else "   <-- AT RANGE EDGE, a bound not a location"
        print(f"  {w['size']:5s} width {w['width']:4d}: lr={w['lr']:<8g} "
              f"loss={w['loss']:.5f} step={w['step']} "
              f"({w['n_finite']}/{w['n_total']} finite){flag}")
        print(f"          {w['reason']}")

    p = math.log(mid["lr"] / small["lr"]) / math.log(mid["width"] / small["width"])
    lr15 = small["lr"] * (WIDTH["1p5b"] / small["width"]) ** p
    print(f"\n  fitted lr ~ width^p with p = {p:+.3f}")
    print(f"  (muP would predict p = -1; a flat optimum would be p = 0)")
    print(f"  extrapolated 1.5B lr = {lr15:.5g}")
    lo, hi = lr15 / a.straddle, lr15 * a.straddle
    print(f"  1.5B bracket straddles it: {lo:.5g} and {hi:.5g}")
    if not (small["interior"] and mid["interior"]):
        print("  WARNING: at least one width's optimum sat at a range edge, so p"
              " is a bound-derived slope. The straddle is the safeguard.")

    out = {"small": small, "mid": mid, "p": p, "lr_1p5b": lr15,
           "bracket": [lo, hi]}
    (a.run_root / "_campaign" / "stage2.json").write_text(json.dumps(out, indent=2))

    if not a.submit:
        print("\n  (report only; pass --submit to launch stage 3)")
        return 0

    V = "/scratch/m000151-pm06/jkli/venvs/koopman-cuda/bin/python"
    pts = f"{lo:.6g},{hi:.6g}"
    # 1.5B bracket at PRODUCTION width (64 GPUs), because resume is invalid
    # across a world-size change and the winner must continue, not restart.
    subprocess.run([V, str(REPO / "scripts/make_mamba3_runs.py"),
                    "--shard", str(a.shard), "--lr_bracket", "1p5b",
                    "--lr_points", pts, "--write"], check=True, cwd=REPO)
    # Full-length 180M and 440M at their OWN measured optima -- no extrapolation
    # needed for these two, which is the whole point of sweeping them.
    for w in (small, mid):
        subprocess.run([V, str(REPO / "scripts/make_mamba3_runs.py"),
                        "--shard", str(a.shard), "--sizes", w["size"],
                        "--lr", str(w["lr"]), "--write"], check=True, cwd=REPO)
    print(f"\n  wrote configs: 1p5b bracket at {pts}, "
          f"180m at {small['lr']:g}, 440m at {mid['lr']:g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
