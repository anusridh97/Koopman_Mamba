"""Koopman-MLP utilization report -- the diagnostic that makes it publishable.

Measures the two pathologies the Aurora-style redesign targets, on a fresh model
or a trained checkpoint:

  * g_i histogram (lift row gains) and its coefficient of variation -- how uniform
    is neuron utilization? Lower CV == tighter, more uniform use.
  * per-pair activation-variance and the dead-pair fraction -- how many rotation
    pairs never move?

Because both metrics are defined identically for v1 and v2 configs, this is the
apples-to-apples comparison that turns "we redesigned the MLP" into "we measured
an under-utilization pathology and show the distribution tightening."

Examples
--------
  # one config, fresh init
  python scripts/koopman_utilization_report.py --config configs/180m.yaml

  # side-by-side: baseline (v1) vs the Aurora-inspired variant (v2)
  python scripts/koopman_utilization_report.py \
      --config configs/180m.yaml --compare configs/180m_v2.yaml

  # from a trained checkpoint directory (expects config.json + a *.pt state dict)
  python scripts/koopman_utilization_report.py --ckpt runs/180m_v2/final
"""
import argparse
import dataclasses
import json
from pathlib import Path

import torch

from koopman_lm.globals.config import build_config, load_config
from koopman_lm.models.koopman_lm import KoopmanLM
from koopman_lm.globals.modules.koopman_mlp_diag import (
    utilization_report, format_report)


def _load_from_ckpt(ckpt_dir):
    ckpt_dir = Path(ckpt_dir)
    cfg = load_config(ckpt_dir / "config.json")
    model = KoopmanLM(cfg)
    states = sorted(ckpt_dir.glob("*.pt")) + sorted(ckpt_dir.glob("*.bin"))
    if states:
        sd = torch.load(states[0], map_location="cpu")
        sd = sd.get("model", sd.get("state_dict", sd))
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing or unexpected:
            print(f"  [load] missing={len(missing)} unexpected={len(unexpected)}")
    else:
        print(f"  [load] no state dict in {ckpt_dir}; reporting on fresh init")
    return cfg, model


def _build(config_path=None, ckpt=None, seq_len=512, vocab_cap=None):
    if ckpt is not None:
        cfg, model = _load_from_ckpt(ckpt)
    else:
        cfg = load_config(config_path) if Path(config_path).suffix in (
            ".yaml", ".yml", ".json") else build_config(config_path)
        model = KoopmanLM(cfg)
    model.eval()
    return cfg, model


def run_one(label, cfg, model, input_ids, out_dir=None):
    rep = utilization_report(model, input_ids)
    rep["label"] = label
    rep["config"] = {k: v for k, v in dataclasses.asdict(cfg).items()
                     if k.startswith("mlp_")}
    print(f"\n=== {label} ===")
    for k in ("mlp_row_norm_lift", "mlp_rotation_param", "mlp_decay_depth_grade",
              "mlp_pair_mixer", "mlp_gated"):
        print(f"  {k}: {getattr(cfg, k, None)}")
    print(format_report(rep))
    if out_dir is not None:
        out = Path(out_dir) / f"utilization_{label}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rep, indent=2))
        print(f"  saved {out}")
    return rep


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="config name or YAML path (v1 / baseline)")
    ap.add_argument("--compare", help="second config to report side-by-side (v2)")
    ap.add_argument("--ckpt", help="checkpoint dir (config.json + state dict)")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default=None, help="write per-run JSON here")
    args = ap.parse_args()
    if not (args.config or args.ckpt):
        ap.error("pass --config and/or --ckpt")

    torch.manual_seed(args.seed)
    reports = []
    specs = []
    if args.ckpt:
        specs.append(("ckpt", dict(ckpt=args.ckpt)))
    if args.config:
        specs.append((Path(args.config).stem, dict(config_path=args.config)))
    if args.compare:
        specs.append((Path(args.compare).stem, dict(config_path=args.compare)))

    for label, kw in specs:
        cfg, model = _build(**kw)
        # random token ids within vocab -- utilization stats are input-distribution
        # robust (they read activation dispersion, not correctness).
        ids = torch.randint(0, cfg.vocab_size, (args.batch, args.seq_len))
        reports.append(run_one(label, cfg, model, ids, out_dir=args.out_dir))

    if len(reports) >= 2:
        print("\n=== comparison (pooled) ===")
        hdr = f"{'metric':<34}" + "".join(f"{r['label']:>16}" for r in reports)
        print(hdr)
        def row(name, fn):
            print(f"{name:<34}" + "".join(f"{fn(r):>16}" for r in reports))
        row("gain CV (lower=more uniform)",
            lambda r: f"{r['gains']['pooled']['cv']:.3f}")
        row("dead-neuron frac %",
            lambda r: f"{r['gains']['pooled']['dead_frac']*100:.2f}")
        row("dead-pair frac %",
            lambda r: f"{r['dead_pairs']['pooled']['dead_frac']*100:.2f}")
        row("pair-var CV",
            lambda r: f"{r['dead_pairs']['pooled']['cv']:.3f}")


if __name__ == "__main__":
    main()
