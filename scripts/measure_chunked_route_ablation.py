"""Does the chunked route's SKA earn anything? The paired measurement.

## Why this and not a loss comparison between two cells

`REVIEW.md` records the between-cell attempt: job 439883 (chunked) against 440183
(exact) moved every objective by at most 2.3e-4 against a 0.458 spread across
anchors, i.e. unresolvable, and REVIEW.md:291-296 lists the proper version as
"a now-possible measurement that has NOT been run".

It is worth saying why a between-cell loss comparison is nearly guaranteed to be
unresolvable here, because that is a fact about the instrument and not about the
route. On `proxy-256x17` at 1500 steps the SKA ablation delta is 0.0251 -- that is
the WHOLE contribution of SKA to held-out loss. So no route change can move
held-out loss by more than about 0.0251, whatever it does to the mechanism. The
trial-vs-trial resolvable effect at the same horizon is 0.0213 (sigma 7.54e-3, job
445994). A between-cell chunked-vs-exact contrast therefore has a ceiling only
1.18x its own noise floor, and would need dozens of seeds per cell to separate
"the route costs everything SKA earns" from "the route costs nothing".

The PAIRED quantity does not have that problem. `ska_ablation.loss_delta` is
measured within one run -- the same weights, the same held-out batches, SKA on
versus SKA zeroed -- and at 1500 steps it is 0.025139 +- 0.002976 SEM over five
seeds, t = 8.4. So the question "does SKA contribute on this route?" IS resolvable
at n = 3, while "which route reaches a lower loss?" is not resolvable at n = 20.

That reframes the whole question. If the chunked route's ablation delta is
statistically zero while the exact route's is 0.025 at t = 8.4, the finding is not
"the chunked route is slightly worse". It is "on the chunked route SKA is not
contributing at all, so a chunked run is a Mamba baseline wearing SKA's parameter
count" -- and that is a statement about what the config MEASURES, which no loss
comparison could have delivered.

## What this runs

Cells at the `proxy-256x17` geometry (`ska_inverse_cholesky: true`, rank 24,
value width 64, seq 1024, 1500 steps, the horizon sigma was measured at):

  exact-invchol      x3 seeds   the control, and a re-measurement of the 0.0251
  chunked-cs64       x3 seeds   the proxy's own ska_chunk_size
  chunked-cs16       x3 seeds   `1m`'s ska_chunk_size -- the Echo Table 2 config

Then, on EVERY finished checkpoint, all four of {route as trained, route swapped}
x {SKA on, SKA zeroed}. The route flag changes no parameter shape, so the swap is
a pure evaluation of the same weights under the other operator: another paired
contrast, and the cheapest one in the file.

## MEASURED -- jobs 446363-446371 and 446412-446420, 1500 steps, H100.
## Analysis: scripts/analyze_chunked_route_ablation.py

    cell           n   held-out    ska_delta (64b)     t     swap penalty
    ------------  --   ---------   ----------------   -----  ------------
    exact-invchol  6    4.39369    +0.016402+-.00157  10.45     +0.001448
    chunked-cs16   5    4.39579    +0.013781+-.00138   9.98     -0.000072
    chunked-cs64   6    4.39284    +0.009460+-.00088  10.80     +0.000755

    vs exact (Welch):  cs16 keeps 84.0% (t 1.25, NOT resolvable)
                       cs64 keeps 57.7% (t 3.86, resolvable)

(A sixth `chunked-cs16` seed was still queued when this was written; rerun the
analysis script to fold it in. It cannot move t=1.25 to significance.)

Harness validated against a number this branch did not produce: on the exact cell
train.py's own 8-batch `--eval_on_final` delta reads +0.024418 against the
independently known +0.025139 +- 0.002976 (job 445994), which used that same
8-batch path.

**The hypothesis this was built to test is FALSE.** "If the chunked route's SKA
contributes nothing measurable, that is the cleanest possible statement of the
problem" -- it contributes plenty. At `1m`'s ska_chunk_size of 16 it retains 84%
of the exact route's entire SKA contribution and the shortfall is NOT resolvable
(t = 1.25). At chunk 64 it retains 58% and that IS resolvable (t = 3.86). So
there is a real dose-response in chunk size, and no chunk size kills SKA.

**And held-out loss cannot see the route at all.** The three cells span 0.00295 in
held-out loss against a trial-to-trial resolvable effect of 0.0213 -- 7x below the
floor, with the cell ORDER not even matching the mechanism (cs64 has the lowest
loss). The paired swap penalty -- same weights, same batches, other operator -- is
+0.0014 for the exact-trained model and -0.00007 at chunk 16, i.e. zero. So this
was never a seed-count problem: REVIEW.md's earlier 2.3e-4 attempt could not have
resolved at any n, and neither could a bigger one.

**Which makes the dissociation the finding.** The same configuration that loses
short-range recall for 93.8% of tokens at chunk 16, and whose gradient is ~100%
wrong at cosine 0.14, pays essentially nothing in aggregate LM loss. Two
consequences, and they point opposite ways:

  * goldens, LM-loss studies and loss-ranked searches on chunked configs are SAFE,
    and no amount of extra seeds would make them sensitive to the route;
  * any claim about SKA's MECHANISM measured on a chunked config -- recall,
    induction, `ska_beta_policy` -- is measuring an operator that is ~100% wrong,
    and aggregate loss will never reveal it.

The warning's "upper bound on SPEED and not a result" is therefore too broad as
stated: a chunked LM-loss number is a fine LM-loss number. What it is not is
evidence about SKA.

Usage:
    python scripts/measure_chunked_route_ablation.py \
        --run_root $SCRATCH/chunked-route-ablation --json out.json
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import subprocess
import sys
import time
import warnings
from pathlib import Path

import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

PROXY = REPO / "configs/runs/proxy-256x17.yaml"

#: (cell name, ska_inverse_cholesky, ska_chunk_size or None to keep the base's)
CELLS = (
    ("exact-invchol", True, None),
    ("chunked-cs64", False, 64),
    ("chunked-cs16", False, 16),
)
SEEDS = (42, 43, 44)


# --------------------------------------------------------------- spec build ----

def variant_spec(route_exact: bool, chunk_size, seed: int, max_steps: int,
                 out_path: Path) -> Path:
    """Write a proxy-256x17 variant differing ONLY in route / chunk / seed / steps."""
    raw = yaml.safe_load(PROXY.read_text())
    raw["model"]["ska_inverse_cholesky"] = bool(route_exact)
    if chunk_size is not None:
        raw["model"]["ska_chunk_size"] = int(chunk_size)
    raw["runtime"]["seed"] = int(seed)
    raw["optim"]["max_steps"] = int(max_steps)
    # 0.04 of the new horizon -- proxy-256x17.yaml is warmup_steps 24 at
    # max_steps 600, and its own comment names 0.04. (This comment said 0.02
    # while the code said 0.04; the code was right.)
    raw["optim"]["warmup_steps"] = max(1, int(round(0.04 * max_steps)))
    raw["name"] = out_path.stem
    out_path.write_text(yaml.safe_dump(raw, sort_keys=False))
    return out_path


# ------------------------------------------------------------------- train ----

def train_cell(spec_path: Path, run_dir: Path, eval_dir: str,
               logging_steps: int) -> Path:
    """Train one cell as a subprocess. Returns the final checkpoint path.

    `build_train_argv` EMITS `--spec run_dir/spec.yaml` and
    `--model_size run_dir/model_config.json` but writes NEITHER -- its own comment
    says "`__main__` writes it before handing off to a Launcher, so it always
    exists by now". Calling it without doing what a Launcher does gives every cell
    `SystemExit: --spec given but no spec.yaml at ...`, and the failure is then
    swallowed by the per-cell `except`, so the run completes and reports a JSON of
    N errors having paid the queue wait for all of them. Worse, the
    `_read_quick_eval` fallback cannot rescue it either: `write_quick_eval` finds
    a run directory BY the presence of spec.yaml, so with no spec.yaml no
    quick_eval.json is ever written.

    So do what `run/__main__.py:62` and `run/launchers.py:196` do, in that order.
    """
    from experimentation.run.resolve import materialize, resolve_run_spec
    from experimentation.run.train_argv import build_train_argv, write_model_config

    spec = resolve_run_spec(spec_path)
    run_dir.mkdir(parents=True, exist_ok=True)
    materialize(spec, run_dir)
    write_model_config(spec, run_dir)
    for needed in ("spec.yaml", "model_config.json"):
        if not (run_dir / needed).exists():
            raise RuntimeError(
                f"{needed} was not written to {run_dir}; train.py would exit on "
                f"it and the error would be swallowed as a per-cell failure")
    argv = build_train_argv(spec, run_dir, eval_on_final=True,
                            eval_data_dir=eval_dir, logging_steps=logging_steps)
    cmd = [sys.executable, "-m", "experimentation.training.train"] + argv
    print(f"    $ {' '.join(cmd[:6])} ... ({len(cmd)} args)", flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(REPO))
    print(f"    train exited {proc.returncode} in {time.time()-t0:.0f}s", flush=True)
    if proc.returncode != 0:
        raise RuntimeError(f"training failed for {spec_path.name}")
    return run_dir / "final" / "model.pt"


def _read_quick_eval(run_dir: Path) -> dict:
    """train.py's own --eval_on_final score, at run_dir/eval/<ckpt>/quick_eval.json.

    Read INDEPENDENTLY of the cross-route pass below, and reported alongside it.
    Two reasons: it is the number the search machinery itself would have used, so
    agreeing with it validates that the cross-route harness loads the same model;
    and it survives a failure in the cross-route step, which is the one part of
    this script that has never run.
    """
    hits = sorted((run_dir / "eval").rglob("quick_eval.json"))
    if not hits:
        return {"error": f"no quick_eval.json under {run_dir / 'eval'}"}
    doc = json.loads(hits[-1].read_text())
    m = doc.get("metrics", doc)
    return {"path": str(hits[-1]),
            "loss": m.get("full", {}).get("loss"),
            "ska_loss_delta": m.get("ska_ablation", {}).get("loss_delta")}


# ------------------------------------------------- cross-route evaluation ----

def eval_both_routes(ckpt: Path, eval_dir: str, batch_size: int,
                     max_batches: int) -> dict:
    """{route: {full/ablated loss}} for the trained route AND the other one.

    The four SKA routes share every parameter -- they differ only in how the
    sufficient statistics are formed -- so the same state_dict loads under either
    flag and the difference is purely the operator.
    """
    from experimentation.evaluation.quick_eval import run_quick_eval
    from koopman_lm.models.koopman_lm import KoopmanLM

    meta = torch.load(str(ckpt).replace("model.pt", "meta.pt"),
                      map_location="cpu", weights_only=False)
    cfg = meta["cfg"]
    # train.py::_save_checkpoint writes `model.state_dict()` RAW -- not wrapped
    # in a {"model": ...} envelope -- so this is already the tensor dict. Checked
    # against that writer rather than guessed, because `state.get("model", state)`
    # would have "worked" either way and quietly loaded nothing if it were
    # wrapped differently.
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    if "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    # A DDP-wrapped save would carry `module.` prefixes; the proxy spec is
    # ddp: false / gpus: 1, but strip them rather than silently load zero keys.
    if any(k.startswith("module.") for k in state):
        state = {k[len("module."):]: v for k, v in state.items()}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out = {}
    for label, use_exact in (("as_trained", cfg.ska_inverse_cholesky),
                             ("route_swapped", not cfg.ska_inverse_cholesky)):
        vcfg = dataclasses.replace(cfg, ska_inverse_cholesky=bool(use_exact))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = KoopmanLM(vcfg)
        # STRICT. The four routes share every parameter -- that is the premise
        # this whole comparison rests on -- so a missing key means the premise is
        # wrong, and a silently zero-loaded model would report a random-init loss
        # that looks like a plausible number. Fail here instead.
        missing, unexpected = model.load_state_dict(state, strict=True)
        model = model.to(device).eval()
        m = run_quick_eval(model, device, data_dir=eval_dir,
                           max_seq_len=min(vcfg.max_seq_len, 1024),
                           batch_size=batch_size, max_batches=max_batches,
                           ska_ablation=True)
        out[label] = {
            "route": "inverse_cholesky" if use_exact else "chunked",
            "loss": m["full"]["loss"],
            "ablated_loss": m["ska_ablation"].get("loss"),
            "ska_loss_delta": m["ska_ablation"].get("loss_delta"),
            "missing_keys": len(missing), "unexpected_keys": len(unexpected),
        }
        print(f"      {label:14s} route={out[label]['route']:17s} "
              f"loss={out[label]['loss']:.5f} "
              f"ska_delta={out[label]['ska_loss_delta']:+.6f}", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return out


# -------------------------------------------------------------------- main ----

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_root", required=True)
    ap.add_argument("--eval_data_dir",
                    default="/scratch/m000151-pm06/jkli/fineweb_small_val")
    ap.add_argument("--json", default=None)
    ap.add_argument("--max_steps", type=int, default=1500)
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    ap.add_argument("--cells", default=",".join(c[0] for c in CELLS))
    ap.add_argument("--logging_steps", type=int, default=100)
    ap.add_argument("--eval_batches", type=int, default=64)
    ap.add_argument("--eval_batch_size", type=int, default=4)
    args = ap.parse_args(argv)

    root = Path(args.run_root)
    root.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",")]
    want = set(args.cells.split(","))
    cells = [c for c in CELLS if c[0] in want]

    results = {"max_steps": args.max_steps, "seeds": seeds,
               "eval_data_dir": args.eval_data_dir, "cells": {}}
    for name, exact, cs in cells:
        results["cells"][name] = {"ska_inverse_cholesky": exact,
                                  "ska_chunk_size": cs, "trials": {}}
    # SEED-MAJOR, not cell-major. If the walltime runs out mid-run, seed-major
    # leaves every cell with the same number of completed trials -- a smaller but
    # still BALANCED comparison. Cell-major would lose the last cell entirely,
    # which here is `chunked-cs16`: the 1m/Table-2 chunk size, i.e. the one the
    # whole exercise is about.
    for seed in seeds:
        for name, exact, cs in cells:
            tag = f"{name}-seed{seed}"
            print(f"\n=== {tag} ===", flush=True)
            spec_p = variant_spec(exact, cs, seed, args.max_steps,
                                  root / f"{tag}.yaml")
            run_dir = root / tag
            rec = {}
            try:
                ckpt = train_cell(spec_p, run_dir, args.eval_data_dir,
                                  args.logging_steps)
                # train.py's OWN --eval_on_final score, read first. It is the
                # primary measurement and it is already on disk, so a failure in
                # the cross-route step below must not cost us the trial.
                rec["train_side_quick_eval"] = _read_quick_eval(run_dir)
                rec.update(eval_both_routes(ckpt, args.eval_data_dir,
                                            args.eval_batch_size,
                                            args.eval_batches))
            except Exception as exc:                        # noqa: BLE001
                print(f"    FAILED: {type(exc).__name__}: {exc}", flush=True)
                rec["error"] = f"{type(exc).__name__}: {exc}"
            results["cells"][name]["trials"][str(seed)] = rec
            if args.json:
                Path(args.json).write_text(json.dumps(results, indent=2) + "\n")

    # ---- summary ----
    print("\n" + "=" * 78)
    print(f"{'cell':16s} {'n':>2s} {'loss mean':>11s} {'ska_delta mean':>15s} "
          f"{'SEM':>10s} {'t':>7s}")
    for name, rec in results["cells"].items():
        ds, ls, srcs = [], [], []
        for t in rec["trials"].values():
            # Prefer the cross-route harness's own number; fall back to
            # train.py's --eval_on_final score so a cross-route failure still
            # yields the ablation delta, which is the primary quantity here.
            src = t.get("as_trained") or t.get("train_side_quick_eval") or {}
            if src.get("ska_loss_delta") is None or src.get("loss") is None:
                continue
            # The two sources use DIFFERENT eval budgets (this script's
            # --eval_batches, vs train.py's --eval_on_final_batches default), so
            # averaging them silently would mix noise levels. Record which.
            srcs.append("cross-route" if "as_trained" in t else "train-side")
            ds.append(src["ska_loss_delta"])
            ls.append(src["loss"])
        if not ds:
            print(f"{name:16s}  0   (all trials failed)")
            continue
        n = len(ds)
        mean = sum(ds) / n
        if n > 1:
            var = sum((d - mean) ** 2 for d in ds) / (n - 1)
            sem = (var / n) ** 0.5
            tstat = mean / sem if sem else float("inf")
        else:
            sem, tstat = float("nan"), float("nan")
        stag = "/".join(sorted(set(srcs)))
        print(f"{name:16s} {n:2d} {sum(ls)/n:11.5f} {mean:15.6f} "
              f"{sem:10.6f} {tstat:7.2f}   [{stag}]")
    print("=" * 78)
    print("ska_delta = held-out loss with SKA zeroed MINUS with SKA on.")
    print("Positive and large => SKA is load-bearing. ~0 => it earns nothing.")

    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2) + "\n")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
