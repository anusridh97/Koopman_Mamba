#!/usr/bin/env python
"""How many trials of a given config fit on one GPU, and what that buys.

    python scripts/gpu_capacity.py configs/runs/10m-joint-base.yaml
    python scripts/gpu_capacity.py configs/runs/10m-joint-base.yaml --microbatch 12
    python scripts/gpu_capacity.py configs/runs/10m-joint-base.yaml --worst-corner

Two numbers, and they answer different questions. `fits` is how many trials the
MEMORY allows, which is what a packing scheduler needs. `aggregate tok/s` is what
those trials actually deliver together, which is what the wall-clock needs. They
are not the same question and at these model sizes they disagree sharply -- see
the throughput note below.

## The memory model

Fitted to eight measured training-time peaks (three geometries x three
microbatches), each taken with `torch.cuda.max_memory_allocated()` around a real
fwd + bwd + AdamW step. Max residual 1.4%.

    GiB = microbatch * (A * T*V  +  B * T * n_ska_layers * n_heads * rank^2)

Two families and no third. A fitted backbone term came out NEGATIVE and is
dropped: activations over `depth * d_model` are collinear with the other two at
these sizes and contribute nothing measurable. What is left is

  * the LM head -- logits are materialized for ALL positions over a 32K vocab,
    and autocast upcasts cross-entropy to fp32, so a bf16 (B,T,V) logits tensor
    is joined by an fp32 copy, an fp32 saved log_softmax, and an fp32 backward
    gradient. 88% of the base config's peak.
  * SKA's exact_invchol stats -- `(B,T,H,r,r)` tensors held in fp32 to backward,
    per SKA layer. 70% of the expensive corner's peak, and the reason throughput
    varies 6x across this study's space.

Peak is LINEAR in microbatch with ~zero intercept, which is why halving the
microbatch halves the memory and a per-GPU queue is a real option.

Deliberately NOT read from `peak_memory_gib` in trials.csv: that column is
measured inside `quick_eval` under `no_grad`, at half the microbatch and half the
seq_len. Every one of the 1,041 completed 3M trials recorded ~4.49 GiB against a
~42 GiB training peak. Sizing a scheduler off it would be wrong by ~9x.

## The throughput note, which is the part that surprises

Fitting more trials onto a GPU does not make the GPU faster. Measured on the
10M base geometry, one H100:

    1 process  @ microbatch 24            192,104 tok/s
    6 processes @ microbatch 6, CUDA MPS  206,956 tok/s aggregate
    6 processes @ microbatch 3, no MPS     25,742 tok/s aggregate

So packing is worth ~7% when MPS is on, and is strictly WORSE than not packing
when it is off -- without MPS the CUDA contexts time-slice rather than overlap,
and six tenants do the same total work as one plus switching overhead. The GPU is
already compute-saturated at microbatch 24; memory is not the binding constraint
there (28 of 79 GiB).

That 7% is a pure-compute floor. A real trial also loads data, writes three
checkpoints and runs a final eval, and those ARE idle-GPU windows a second tenant
can fill, so 2 tenants is worth taking. It is not worth expecting 4x from.

The lever that does scale is GPUs: throughput per GPU is capped near 200K tok/s,
so N GPUs give N x 200K. `--gpus` prices that out.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from koopman_lm.config import KoopmanLMConfig                      # noqa: E402
from experimentation.run.resolve import resolve_run_spec           # noqa: E402

#: Fitted to the eight measured peaks; see the module docstring.
HEAD_COEF = 1.504e-8      # GiB per (microbatch * seq_len * vocab)
SKA_COEF = 1.018e-8       # GiB per (microbatch * seq_len * n_ska * heads * rank^2)

#: Measured on the 10M base geometry at microbatch 24, one H100, no MPS.
#: Throughput is NOT a function of tenancy -- see the module docstring.
MEASURED_BASE_TOK_S = 192_104
#: Same measurement at the expensive corner (rank 48, 8 heads, 6 SKA layers).
MEASURED_CORNER_TOK_S = 31_409


def training_peak_gib(cfg: KoopmanLMConfig, microbatch: int,
                      seq_len: int | None = None) -> float:
    """Predicted TRAINING-time peak GiB for one trial of `cfg`."""
    T = int(seq_len or cfg.max_seq_len)
    head = HEAD_COEF * T * cfg.vocab_size
    ska = SKA_COEF * T * len(cfg.ska_layer_indices) * cfg.ska_n_heads * cfg.ska_rank ** 2
    return microbatch * (head + ska)


def trials_per_gpu(cfg: KoopmanLMConfig, microbatch: int, *,
                   gpu_gib: float = 79.0, safety: float = 1.5,
                   seq_len: int | None = None) -> int:
    """How many trials fit, at `safety` x the predicted peak. At least 1.

    The safety factor is not decoration. The 3M study lost 102 trials to OOM
    because a finished trainer's CUDA context had not been torn down when the
    next trial started on the same worker -- a transient second tenant nobody
    scheduled. Headroom is what absorbs that.
    """
    need = training_peak_gib(cfg, microbatch, seq_len) * safety
    return max(1, int(gpu_gib // need))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("spec", help="path to a configs/runs/*.yaml RunSpec")
    p.add_argument("--microbatch", type=int, default=None,
                   help="default: the spec's runtime.per_device_batch_size")
    p.add_argument("--gpu-gib", type=float, default=79.0)
    p.add_argument("--safety", type=float, default=1.5)
    p.add_argument("--gpus", type=int, default=8)
    p.add_argument("--trials", type=int, default=256)
    p.add_argument("--pruned-fraction", type=float, default=0.45,
                   help="fraction pruned at prune_after_step; scales total work")
    p.add_argument("--worst-corner", action="store_true",
                   help="price the most expensive cell instead of the base config")
    a = p.parse_args(argv)

    spec = resolve_run_spec(a.spec)
    cfg = spec.model
    if a.worst_corner:
        from dataclasses import replace
        cfg = replace(cfg, ska_rank=48, ska_n_heads=8,
                      ska_layer_indices=tuple(range(1, 7)))
    mb = a.microbatch or spec.runtime.per_device_batch_size
    eb = spec.optim.effective_batch
    if eb % mb:
        print(f"WARNING: effective_batch {eb} is not divisible by microbatch {mb}; "
              f"the trainer requires an exact accumulation factor.")

    tok_per_trial = eb * cfg.max_seq_len * spec.optim.max_steps
    label = "worst corner" if a.worst_corner else "base config"
    print(f"{spec.name}  ({label})")
    print(f"  params            {cfg.param_count_estimate():,}")
    print(f"  microbatch {mb}, accum {eb // mb}, seq {cfg.max_seq_len}, "
          f"{spec.optim.max_steps} steps")
    print(f"  tokens per trial  {tok_per_trial:,} "
          f"({tok_per_trial / cfg.param_count_estimate():.1f} per param)")
    print()
    for m in sorted({mb, 24, 12, 6, 3}, reverse=True):
        if eb % m:
            continue
        peak = training_peak_gib(cfg, m)
        n = trials_per_gpu(cfg, m, gpu_gib=a.gpu_gib, safety=a.safety)
        mark = "  <- spec" if m == mb else ""
        print(f"  microbatch {m:>3}: peak {peak:>6.2f} GiB  x{a.safety} = "
              f"{peak * a.safety:>6.2f}  ->  {n} trial(s) per {a.gpu_gib:.0f} GiB GPU{mark}")

    tps = MEASURED_CORNER_TOK_S if a.worst_corner else MEASURED_BASE_TOK_S
    print(f"\n  MEASURED throughput per GPU: {tps:,} tok/s, and it does not rise with")
    print(f"  tenancy (6 MPS tenants measured 206,956 on the base geometry).")
    work = a.trials * tok_per_trial * (1 - a.pruned_fraction * 0.65)
    print(f"\n  {a.trials} trials, {a.pruned_fraction:.0%} pruned at ~35% of the horizon:")
    for g in sorted({a.gpus, 8, 16, 24}):
        print(f"    {g:>3} GPUs: {work / (g * tps) / 3600:>6.1f} h")
    return 0


if __name__ == "__main__":
    sys.exit(main())
