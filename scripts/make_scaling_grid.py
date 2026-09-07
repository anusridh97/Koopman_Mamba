"""Emit the `cells:` sweep YAMLs for the Chinchilla-style scaling-law grid.

    python scripts/make_scaling_grid.py --out configs/sweeps/scaling-law-v1

WHY A GRID AT ALL, given four completed rungs. Every rung sits on the ray
D ~= 65N by design, so in (log N, log D) they are 94% collinear -- total
off-ray leverage is 0.248 in log(D/N) against a 4.11 range in log N. Five free
parameters against four points is under-determined, and profiling the residual
confirms it: 198 distinct (alpha, beta) pairs fit those four points equally
well, alpha anywhere in 0.05-1.00. Only runs OFF the ray separate alpha from
beta.

DESIGN NOTES, each of which cost something to learn.

* ONE architecture family, constant aspect ratio. `d_model / n_layers ~= 20`
  throughout, so only scale moves. An earlier version solved (d_model,
  n_layers) to hit round parameter targets and produced aspect ratios of
  4.4 / 5.8 / 6.1 / 23.3 / 46.2 -- which would have conflated shape change with
  scale in the one measurement that must not conflate them. N does not need to
  be a round number for a scaling law; it needs to be known and well spread.

* FIT ON NON-EMBEDDING N. With a fixed 32k vocab the embedding is 47% of the
  smallest model here and 9% of the largest, and Kaplan et al. section 2.1 is
  explicit that including it distorts the N exponent. Both counts are emitted;
  the fit uses non-embedding. The d_model=192 rung was dropped for this reason
  -- at 61% embedding it is a different kind of object.

* EVERY CELL UNDER ONE EPOCH. The shard is 10,401,294,869 tokens and cannot
  easily grow: `pretokenize.py` streams `sample-10BT`, docs 97,159-9,396,575 are
  already consumed, and `configs/runs/180m-joint-base.yaml:191-193` records that
  switching to `sample-100BT` would void the disjointness guarantee against
  `fineweb_small_val`. So the token multiple is capped per size, which drops the
  768x64 cell and makes the grid ragged (19 cells, not 20). Better a missing
  cell than one cell silently at 1.7 epochs at the most influential corner.

* THREE LR POINTS PER CELL, from a FIXED triple rather than an extrapolated
  rule. LR error is worth 0.03-0.06 loss (measured: `lr-0.3x` = +0.0598 at 50M)
  and is correlated with N, so it does not average out and goes straight into
  alpha -- this is the "optimizer convergence" defect Epoch AI found in
  Hoffmann's Approach 3. A rule was rejected because both rules we have tried
  failed: 1/width predicted 0.0027 at 50M against >=0.0088 measured, and the
  assumption that the optimum keeps rising was refuted at 180M where it fell to
  0.00256. The triple spans every optimum measured across four rungs
  (3M 0.0054, 10M 0.0075, 50M >=0.0088, 180M 0.00256).

* WARMUP IS COMPUTED PER CELL. `warmup_ratio` does not exist in the run system;
  `OptimSpec` carries only absolute `warmup_steps`, and the ratio -> steps
  conversion lives in the search at `sweep/search/space.py:701`. 2% is the repo
  convention (21/1068, 68/3400, 1017/50862).

* GROUPED BY RUNTIME. `run/launchers.py:320-333` refuses a Slurm array whose
  cells differ in partition/account/qos/gpus/nodes/time_limit/gpu_arch, so cells
  are emitted into separate sweeps per (gpus, time_limit) class.

* IDENTITY IS SAFE. `run/spec.py:299` hashes the whole `OptimSpec` dataclass, so
  two cells differing only in `max_steps` get different group_id AND run_id.
  This was checked rather than assumed -- a collision would have silently made
  two token budgets overwrite each other.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from koopman_lm.config import KoopmanLMConfig
from experimentation.sweep.search.geometry import make_layer_indices

TOK_PER_STEP = 96 * 2048          # effective_batch x max_seq_len
SHARD_TOKENS = 10_401_294_869     # /scratch/.../tok10b/shard meta.json
WARMUP_RATIO = 0.02
ASPECT = 20                       # d_model / n_layers, held constant

#: d_model values; n_layers follows from ASPECT. Not round parameter counts --
#: see the module docstring.
WIDTHS = (256, 320, 448, 576, 768)

#: Tokens per TOTAL parameter. Brackets Chinchilla's ~20 and our own ray's ~65.
MULTIPLES = (8, 16, 32, 64)

#: The LR triple, applied to every cell. See the docstring for why this is not
#: derived from a rule.
LR_POINTS = (0.002, 0.0045, 0.010)

#: Fixed across every size so only scale moves. `ska_rank: 48` resolved better
#: at both 10M and 50M; `d_state: 24` never resolved; `beta_policy: learned` and
#: `placement: late` were the winners' choice at 10M and 50M.
FAMILY = dict(vocab_size=32000, max_seq_len=2048, mamba_expand=2, d_state=24,
              ska_rank=48, ska_n_heads=4, ska_prefix_scan=False,
              ska_inverse_cholesky=True, ska_exact_intrachunk=False,
              ska_beta_policy="learned", ska_layerscale=True,
              ska_layerscale_init=0.01, ska_power_K=1)

# Measured single-GPU throughput, real trainer (not the synthetic probe, which
# was 30% optimistic): 50M -> 133,937 tok/s median over 24 study trials;
# 180M -> 63,000 tok/s from job w1b at world 1.
_N1, _T1 = 51_698_320, 133_937.0
_N2, _T2 = 182_665_776, 63_000.0
_P = math.log(_T1 / _T2) / math.log(_N2 / _N1)
_K = _T1 * _N1 ** _P


def tokens_per_sec(n_params: int) -> float:
    return _K * n_params ** -_P


def gpu_hours(n_params: int, tokens: int) -> float:
    return tokens / tokens_per_sec(n_params) / 3600


def family_member(d_model: int, n_ska: int = 4):
    n_layers = max(4, round(d_model / ASPECT))
    idx = tuple(make_layer_indices(n_layers, n_ska, "even", None))
    cfg = KoopmanLMConfig(d_model=d_model, n_layers=n_layers,
                          ska_layer_indices=idx, **FAMILY)
    total = cfg.param_count_estimate()
    return dict(d_model=d_model, n_layers=n_layers, ska_layer_indices=list(idx),
                total=total, non_emb=total - FAMILY["vocab_size"] * d_model)


def build_cells():
    """Every (size, budget, lr) cell, with steps and warmup resolved."""
    cells, dropped = [], []
    for d_model in WIDTHS:
        m = family_member(d_model)
        for mult in MULTIPLES:
            target = mult * m["total"]
            if target > SHARD_TOKENS:
                dropped.append((d_model, mult, target))
                continue
            steps = max(1, round(target / TOK_PER_STEP))
            tokens = steps * TOK_PER_STEP
            warmup = max(1, round(WARMUP_RATIO * steps))
            for lr in LR_POINTS:
                cells.append(dict(
                    d_model=d_model, n_layers=m["n_layers"],
                    ska_layer_indices=m["ska_layer_indices"],
                    total=m["total"], non_emb=m["non_emb"],
                    mult=mult, steps=steps, tokens=tokens, warmup=warmup,
                    lr=lr, gpu_h=gpu_hours(m["total"], tokens)))
    return cells, dropped


def runtime_group(cell) -> tuple[int, str]:
    """(gpus, time_limit) for a cell. Cells must be uniform within a sweep."""
    h = cell["gpu_h"]
    if h <= 4.0:
        return 1, "12:00:00"
    return 4, "1-00:00:00"          # h/4 hours of work; 24 h is >=2x the worst


def emit(cells, base: str, name: str, out_dir: Path) -> list[Path]:
    import collections
    groups = collections.defaultdict(list)
    for c in cells:
        groups[runtime_group(c)].append(c)
    written = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for (gpus, tl), members in sorted(groups.items()):
        lines = [
            f"# {name}, runtime group: {gpus} GPU(s), time_limit {tl}.",
            "#",
            "# Generated by scripts/make_scaling_grid.py -- do not hand-edit.",
            "# Split by runtime because run/launchers.py:320-333 refuses a Slurm",
            "# array whose cells differ in gpus/nodes/time_limit.",
            "#",
            f"# {len(members)} cells, {sum(c['gpu_h'] for c in members):.0f} GPU-h,",
            f"# worst cell {max(c['gpu_h'] for c in members):.1f} GPU-h"
            f" = {max(c['gpu_h'] for c in members)/gpus:.1f} h on {gpus} GPU(s).",
            "",
            f"name: {name}-g{gpus}",
            f"base: {base}",
            "cells:",
        ]
        for c in sorted(members, key=lambda c: (c["d_model"], c["mult"], c["lr"])):
            lines += [
                f"# d_model {c['d_model']} x{c['mult']} tok/param"
                f"  N={c['total']:,} (non-emb {c['non_emb']:,})"
                f"  D={c['tokens']:,}  {c['gpu_h']:.2f} GPU-h",
                f"- model.d_model: {c['d_model']}",
                f"  model.n_layers: {c['n_layers']}",
                f"  model.ska_layer_indices: {c['ska_layer_indices']}",
                f"  optim.lr: {c['lr']}",
                f"  optim.max_steps: {c['steps']}",
                f"  optim.warmup_steps: {c['warmup']}",
                f"  runtime.gpus: {gpus}",
                f"  runtime.ddp: {str(gpus > 1).lower()}",
                f"  runtime.time_limit: \"{tl}\"",
            ]
        p = out_dir / f"{name}-g{gpus}.yaml"
        p.write_text("\n".join(lines) + "\n")
        written.append(p)
    return written


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("configs/sweeps"))
    ap.add_argument("--base", default="configs/runs/scaling/ska-base.yaml")
    ap.add_argument("--name", default="scaling-law-v1-ska")
    ap.add_argument("--emit", action="store_true", help="write the YAMLs")
    args = ap.parse_args(argv)

    print("family (constant aspect ratio d/L = %d):" % ASPECT)
    print("  %7s %4s %6s %14s %14s %6s" % ("d_model", "L", "d/L", "non-emb", "total", "emb%"))
    for d in WIDTHS:
        m = family_member(d)
        print("  %7d %4d %6.1f %14s %14s %5.1f%%" % (
            d, m["n_layers"], d / m["n_layers"], f"{m['non_emb']:,}",
            f"{m['total']:,}", FAMILY["vocab_size"] * d / m["total"] * 100))
    ne = [family_member(d)["non_emb"] for d in WIDTHS]
    tt = [family_member(d)["total"] for d in WIDTHS]
    print("  span: non-emb %.0fx   total %.0fx" % (max(ne) / min(ne), max(tt) / min(tt)))

    cells, dropped = build_cells()
    n_cfg = len({(c["d_model"], c["mult"]) for c in cells})
    print("\ncells: %d configs x %d LR = %d runs" % (n_cfg, len(LR_POINTS), len(cells)))
    print("  total %.0f GPU-h | worst cell %.2f GPU-h" % (
        sum(c["gpu_h"] for c in cells), max(c["gpu_h"] for c in cells)))
    for d, mult, target in dropped:
        print("  DROPPED d_model %d x%d: D=%.2fB exceeds the %.2fB shard (%.2f epochs)"
              % (d, mult, target / 1e9, SHARD_TOKENS / 1e9, target / SHARD_TOKENS))

    import collections
    for (g, tl), members in sorted(collections.defaultdict(
            list, {k: v for k, v in _group(cells).items()}).items()):
        print("  group %d GPU(s) %s: %d runs, %.0f GPU-h, worst %.1f h wall"
              % (g, tl, len(members), sum(c["gpu_h"] for c in members),
                 max(c["gpu_h"] for c in members) / g))

    # Assertions that must hold before anything launches.
    assert all(c["tokens"] <= SHARD_TOKENS for c in cells), "a cell exceeds one epoch"
    assert all(c["warmup"] >= 1 for c in cells), "a cell has zero warmup"
    assert all(c["steps"] >= 1 for c in cells), "a cell has zero steps"
    print("\nassertions: every cell <= 1 epoch, warmup >= 1, steps >= 1  OK")

    if args.emit:
        for p in emit(cells, args.base, args.name, args.out):
            print("wrote %s" % p)
    else:
        print("\n(dry run -- pass --emit to write the YAMLs)")
    return 0


def _group(cells):
    import collections
    g = collections.defaultdict(list)
    for c in cells:
        g[runtime_group(c)].append(c)
    return g


if __name__ == "__main__":
    sys.exit(main())
