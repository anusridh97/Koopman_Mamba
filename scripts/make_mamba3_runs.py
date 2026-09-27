"""Emit the 180M / 440M / 1.5B run specs for the FineWeb-Edu 100B protocol.

    python scripts/make_mamba3_runs.py --shard /scratch/.../tok100b/shard
    python scripts/make_mamba3_runs.py --shard ... --lr_bracket 440m --write

A generator rather than three hand-written YAMLs because three numbers per file
(`data.n_tokens`, `optim.max_steps`, `optim.warmup_steps`) are derived from the
shard that actually exists, and `data_verify.py:100` raises if n_tokens differs
from the shard's meta.json by even one token.

## Sizing: the labels are TOTAL parameters

GPT-3/Pythia/Mamba size labels count total params INCLUDING the embedding.
Mamba-1/2's ladder is 130M/370M/1.4B at GPT-NeoX's 50,280 vocab; the same shapes
at Llama-3.1's 128,256 vocab are 183.4M/433.3M/1.471B -- which is where Mamba-3's
180M/440M/1.5B labels come from, to within 0.3% on all three rows. So the target
is a total-parameter count, not a body count.

We cannot reuse their (d_model, n_layers): our block carries a Mamba-2 sequence
mixer AND a separate SwiGLU channel mixer, roughly twice their per-layer cost,
so d768/L24 would be 303M for us. Solved for our own shapes at aspect ratio ~25,
which is the `*-joint-base` lineage's own ratio.

## Architecture is frozen, from the ladder's verdict

Four completed searches agree that the learning rate dominates at every scale
(variance share 0.58-0.75) and that the searchable architecture axes stop paying
by 50M-180M (`ska_rank` share 0.340 -> 0.292 -> 0.187 -> 0.075; `depth_tier`
exactly 0.000 at 50M). So nothing here is searched.

`ska_rank 48` because it resolved BETTER than 24 at both 10M and 50M and is
statistically indistinguishable at 180M (delta -0.00102 against a 0.00316
threshold). Rank 8 is excluded on evidence, not preference: it resolved WORSE at
10M by +0.0595.

`placement "even"` because placement never resolved (share 0.000 at 10M, 50M and
180M), so it is a free choice; indices come from the existing
`make_layer_indices` rather than being typed in.

## effective_batch 256, which the code requires anyway

`ddp_grad_accum` RAISES on effective_batch=96 at world 16 -- 96 is not a
multiple of per_device_batch_size * world_size. 256 sequences is 524,288
tokens/step, matches Mamba's 0.5M batch, and gives 8 seq/GPU at 32 GPUs.

## The LR bracket is launched as REAL runs, not as short proxies

`--lr_bracket <size>` emits three full-length runs differing only in `optim.lr`.
Two get cancelled at ~8% once the ranking is clear; the winner KEEPS RUNNING to
100B with no restart, so only the losers' 8% is spent (~267 GPU-h of the 401).

A short-horizon proxy was rejected: it measures the best LR for a SHORT run,
which is biased high, and our own ladder is the evidence -- the 180M optimum fell
2.9x below 10M's precisely because the horizon grew. Comparing three runs at the
same point of the same 100B schedule has no horizon bias.

Known limitation: an 8% ranking is not a 100% ranking, and a too-high LR can
look good early and diverge late (`lr-3.85x` went NaN at 35% and then ran 33,000
more steps). So the decision rule prefers the LOWER LR when two are within the
seed-noise floor, and the loss is still watched after the cut.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

SEQ = 2048
#: Set by --partition/--account/--qos/--time_limit; applied in emit().
RUNTIME_OVERRIDE: dict = {}
EFFECTIVE_BATCH = 256
WARMUP_FRAC = 0.02          # repo convention: 21/1068, 68/3400, 1017/50862
ASPECT_TARGET = 25

#: (d_model, n_layers) solved against the total-parameter targets above, and the
#: launch topology.
#:
#: per_device_batch_size comes from scripts/measure_train_memory.py -- a real
#: fwd+bwd+step peak at rank 48, seq 2048, vocab 128,256, with the trainer's own
#: fp32 params + bf16 autocast + fused AdamW. NOT from gpu_capacity.py, whose
#: fitted model has no d_model term and understated the d768 grid cells by >3x,
#: OOMing all nine. Measured peak, plus DDP's gradient bucket (another fp32 grad
#: copy, 4 bytes/param), against 79.2 GiB:
#:
#:   180m pdbs 8 -> 47.8 + 0.7 = 48.5 GiB, 30.7 free, grad_accum 1
#:   440m pdbs 4 -> 35.4 + 1.6 = 37.0 GiB, 42.2 free, grad_accum 2
#:   1p5b pdbs 2 -> 42.8 + 5.4 = 48.2 GiB, 31.0 free, grad_accum 2
#:
#: 440m at pdbs 8 fits (66.4 GiB) and would reach grad_accum 1, but 12.8 GiB of
#: headroom for a 52-hour run is not worth the few percent that removing one
#: accumulation step buys. 1p5b at pdbs 4 leaves 7.3 GiB and is rejected.
#: Rank 32 would save only 4-6 GiB, so rank 48 -- the value the ladder actually
#: resolved in favour of -- costs us nothing here.
SIZES = {
    "180m": dict(d_model=576,  n_layers=23, target=183_435_264, nodes=4, gpus=8, pdbs=8),
    "440m": dict(d_model=832,  n_layers=33, target=433_324_032, nodes=4, gpus=8, pdbs=4),
    "1p5b": dict(d_model=1280, n_layers=54, target=1_470_627_840, nodes=8, gpus=8, pdbs=2),
}

#: The spec the ladder ACTUALLY RAN, and the base every emitted config extends.
#:
#: Not `CONFIG_REGISTRY["180m"]`. The registry config differs from the measured
#: one in ten fields, and two of them matter a lot: `ska_mode` is `parallel`
#: here (SKA runs ALONGSIDE the Mamba mixer, so it carries parameters) against a
#: bare YAML's `replace` default -- a 28.5M swing at d576/L23, i.e. a model 15%
#: under target while the arithmetic claimed otherwise -- and `ska_backend`
#: `cuda_prefix` + `ska_prefix_scan: true`, a route needing ninja at run time
#: that the joint-base deliberately replaces with exact_invchol.
#: Resolving this file reproduces the ladder's 180M reference N=182,665,776
#: exactly, which is the check that it is the right base.
BASE_SPEC_ABS = REPO / "configs/runs/180m-joint-base.yaml"
#: Computed per --out, not a fixed "../180m-joint-base.yaml". `extends` resolves
#: relative to the emitted file's OWN directory (run/resolve.py:44), so a fixed
#: relative path is only correct at one depth -- it broke once for a copy in
#: /scratch and again for configs/runs/mamba3/hero/. Deriving it with relpath
#: makes --out free.
BASE_SPEC = "../180m-joint-base.yaml"

#: Brackets the plausible optimum. The 180M measurement at 65 tok/param was
#: 0.00256; this protocol changes two things in OPPOSITE directions and neither
#: is calibrated -- batch 96->256 (2.67x) pushes the optimum UP, while the
#: horizon going from 65 to ~545 tok/param pushes it DOWN. Hence a wide 4x span
#: rather than a point estimate. 0.0179 is known to diverge, so the top is well
#: below it.
LR_BRACKET = (0.0015, 0.003, 0.006)


def base_model():
    from experimentation.run import resolve
    return resolve.resolve_run_spec(REPO / "configs/runs/180m-joint-base.yaml").model


def build(size: str, lr: float, n_tokens: int, max_steps: int):
    from experimentation.sweep.search.geometry import make_layer_indices

    s = SIZES[size]
    base = base_model()
    idx = make_layer_indices(s["n_layers"], 4, "even", list(base.ska_layer_indices))
    cfg = dataclasses.replace(
        base, d_model=s["d_model"], n_layers=s["n_layers"], vocab_size=128256,
        max_seq_len=SEQ, ska_rank=48, ska_layer_indices=tuple(idx))
    total = cfg.param_count_estimate()
    warmup = max(1, round(WARMUP_FRAC * max_steps))
    return cfg, idx, total, warmup


def emit(size: str, lr: float, shard: Path, n_tokens: int, max_steps: int,
         tag: str | None, out_dir: Path | None = None) -> tuple[str, str]:
    import os
    base_rel = (os.path.relpath(BASE_SPEC_ABS, out_dir) if out_dir
                else BASE_SPEC)
    cfg, idx, total, warmup = build(size, lr, n_tokens, max_steps)
    s = SIZES[size]
    name = f"mamba3-{size}" + (f"-{tag}" if tag else "")
    band = 100 * (total - s["target"]) / s["target"]
    text = f"""# {name}: the {size} rung of the FineWeb-Edu 100B / Llama-3.1 protocol.
#
# GENERATED by scripts/make_mamba3_runs.py -- edit that, not this.
#
# EXTENDS the spec the ladder actually ran, rather than restating 40 fields.
# `extends` deep-merges base -> leaf (run/resolve.py:36-47), so everything not
# overridden below is bit-identical to what was measured: ska_mode parallel,
# exact_invchol (prefix_scan false + inverse_cholesky true, NOT the registry's
# cuda_prefix which needs ninja at run time), mlp_type swiglu, norm_type
# rmsnorm, init_policy mamba_safe, ska_norm_clip_c 4.0, tie_embeddings true.
# Basing this on CONFIG_REGISTRY["180m"] instead would silently flip ska_mode to
# `replace` and drop 28.5M parameters -- a model 15% under target.
#
# Total parameters {total:,} against a target of {s['target']:,} ({band:+.1f}%).
# The target is Mamba's own shape at Llama-3.1's vocab: d768/L24, d1024/L48 and
# d2048/L48 give 183.4M / 433.3M / 1.471B at vocab 128,256, which is where
# Mamba-3's 180M/440M/1.5B labels come from (agreement within 0.3%), because
# GPT-3/Pythia/Mamba labels count TOTAL parameters including the embedding.
# Our block carries a SwiGLU channel mixer that Mamba-2's does not -- about
# twice the per-layer cost -- so the shape is solved for OUR architecture at
# aspect ratio ~{s['d_model']/s['n_layers']:.0f}, the joint-base lineage's own ratio.
extends: {base_rel}
name: {name}

model:
  d_model: {s['d_model']}
  n_layers: {s['n_layers']}
  # 128,256 overflows uint16, so the shard is uint32 (pretokenize.token_dtype_for).
  vocab_size: 128256
  max_seq_len: {SEQ}
  # 48, not the base's 24: rank 48 resolved BETTER than 24 at both 10M and 50M,
  # and at 180M the two are statistically indistinguishable (delta -0.00102
  # against a 0.00316 threshold). Rank 8 is excluded on evidence rather than
  # preference -- it resolved WORSE at 10M by +0.0595.
  ska_rank: 48
  # make_layer_indices({s['n_layers']}, 4, "even", ...). `placement` never resolved at any
  # rung (variance share 0.000 at 10M, 50M and 180M), so "even" is a free choice.
  ska_layer_indices: {idx}

data:
  kind: shard
  shard_dir: {shard}
  # Llama-3.1 via the ungated NousResearch mirror: vocab 128,256, eos
  # <|end_of_text|> = 128001 -- the same token the lead's gigatoken script picks.
  tokenizer: NousResearch/Meta-Llama-3.1-8B
  mix: {{fineweb: 1.0}}
  # Must equal the shard's meta.json exactly; data_verify.py:100 raises otherwise.
  n_tokens: {n_tokens}

optim:
  lr: {lr}
  warmup_steps: {warmup}
  max_steps: {max_steps}
  schedule: cosine
  # 256 sequences = {EFFECTIVE_BATCH * SEQ:,} tokens/step. Not the base's 96:
  # ddp_grad_accum REFUSES 96 at world 16 (not a multiple of pdbs*world), and
  # 256 matches Mamba's 0.5M-token batch, making the comparison cleaner.
  effective_batch: {EFFECTIVE_BATCH}
  weight_decay: 0.1
  grad_clip: 1.0

runtime:
  # NOT hashed into run_id, so free to tune -- and it must come from a measured
  # fwd+bwd+step peak, not scripts/gpu_capacity.py, whose fitted model has no
  # d_model term and understated the d768 grid cells by >3x, OOMing all nine.
  per_device_batch_size: {s['pdbs']}
  seed: 42
  deterministic: true
  precision: bf16
  ddp: true
  gpus: {s['gpus']}
  nodes: {s['nodes']}
  partition: {RUNTIME_OVERRIDE.get("partition", "batch")}
  account: {RUNTIME_OVERRIDE.get("account", "marlowe-m000151-pm06")}
  qos: {RUNTIME_OVERRIDE.get("qos", "medium")}
  workers: 8
  # `batch` caps at 2 days. The launch line carries --resume_if_available, so a
  # requeue or a --dependency chain continues rather than restarting at step 0.
  time_limit: "{RUNTIME_OVERRIDE.get("time_limit", "2-00:00:00")}"
  gpu_arch: "9.0"
"""
    return name, text


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=Path, required=True)
    ap.add_argument("--sizes", default="180m,440m,1p5b")
    ap.add_argument("--lr", type=float, default=None,
                    help="single LR; default is the middle bracket point")
    ap.add_argument("--lr_bracket", default=None,
                    help="size to emit an LR bracket for")
    ap.add_argument("--lr_points", default=None,
                    help="comma-separated LRs to use instead of LR_BRACKET. The "
                         "1.5B bracket needs TWO points, not three: three runs "
                         "at its production width of 64 GPUs is 192, over the "
                         "medium QOS cap of gres/gpu=128 per account. It must "
                         "run at production width because resume is not valid "
                         "across a world-size change -- samples_consumed is a "
                         "scalar index into a global permutation and "
                         "DistributedSampler partitions by num_replicas, so a "
                         "bracket at half width could not hand off to a "
                         "full-width continuation.")
    ap.add_argument("--out", type=Path, default=REPO / "configs/runs/mamba3")
    ap.add_argument("--nodes", type=int, default=None,
                    help="override the launch topology. An LR SWEEP runs many "
                         "points at reduced width to fit the 128-GPU account "
                         "cap -- 10 points at the production 32 GPUs would be "
                         "320. Sweep points are throwaway probes by design: "
                         "resume is invalid across a world-size change, so a "
                         "narrow point cannot hand off to a full-width run. "
                         "That is affordable because 8% of a narrow run is "
                         "cheap (48 GPUs x 7.1h for six 180M points).")
    ap.add_argument("--gpus", type=int, default=None)
    ap.add_argument("--partition", default=None,
                    help="override the partition/account/qos/time_limit. The LR "
                         "SWEEP runs on `preempt`, not `batch`: on batch we rank "
                         "32-41 of 56 pending behind 34 nodes of demand in a "
                         "31-node partition, while on preempt we are TOP "
                         "priority with zero jobs ahead, MaxNodes unlimited, and "
                         "39 jobs turning over on 1-4h limits. preempt's 4h cap "
                         "is fine for short sweep points given chaining plus "
                         "--resume_if_available, both proven on GPU.")
    ap.add_argument("--account", default=None)
    ap.add_argument("--qos", default=None)
    ap.add_argument("--time_limit", default=None)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args(argv)

    meta = json.loads((a.shard / "meta.json").read_text())
    n_tokens = int(meta["n_tokens"])
    max_steps = n_tokens // (EFFECTIVE_BATCH * SEQ)
    print(f"shard {a.shard}")
    print(f"  n_tokens {n_tokens:,}  dtype {meta['dtype']}  vocab {meta['vocab_size']}")
    print(f"  -> max_steps {max_steps:,} at {EFFECTIVE_BATCH}x{SEQ} "
          f"= {max_steps * EFFECTIVE_BATCH * SEQ:,} tokens consumed")
    if meta["dtype"] != "uint32":
        raise SystemExit(f"expected uint32 for vocab 128,256, got {meta['dtype']}")

    jobs = []
    if a.lr_bracket:
        points = ([float(x) for x in a.lr_points.split(",")]
                  if a.lr_points else LR_BRACKET)
        for lr in points:
            jobs.append((a.lr_bracket, lr, f"lr{lr:g}".replace(".", "p")))
    else:
        for size in a.sizes.split(","):
            jobs.append((size.strip(), a.lr or LR_BRACKET[1], None))

    if a.partition or a.account or a.qos or a.time_limit:
        global RUNTIME_OVERRIDE
        RUNTIME_OVERRIDE = {k: v for k, v in
                            (("partition", a.partition), ("account", a.account),
                             ("qos", a.qos), ("time_limit", a.time_limit))
                            if v is not None}
    if a.nodes or a.gpus:
        for k in SIZES:
            if a.nodes:
                SIZES[k] = dict(SIZES[k], nodes=a.nodes)
            if a.gpus:
                SIZES[k] = dict(SIZES[k], gpus=a.gpus)
    a.out.mkdir(parents=True, exist_ok=True)
    for size, lr, tag in jobs:
        name, text = emit(size, lr, a.shard, n_tokens, max_steps, tag, a.out)
        cfg, idx, total, warmup = build(size, lr, n_tokens, max_steps)
        dest = a.out / f"{name}.yaml"
        print(f"\n{name}: d{SIZES[size]['d_model']}/L{SIZES[size]['n_layers']} "
              f"total {total:,} lr {lr:g} warmup {warmup:,} "
              f"{SIZES[size]['nodes']}x{SIZES[size]['gpus']} GPUs")
        if a.write:
            dest.write_text(text)
            print(f"  wrote {dest}")
        else:
            print(f"  (would write {dest}; pass --write)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
