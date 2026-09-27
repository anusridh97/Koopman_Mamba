"""Measure the real training-time memory peak per (size, microbatch).

    python scripts/measure_train_memory.py --sizes 180m,440m,1p5b --pdbs 1,2,4,8

Prints max_memory_allocated() around a REAL fwd+bwd+optimizer step, which is
the only number CLAUDE.md accepts for choosing per_device_batch_size:

  "Choose the count from a MEASURED training-time peak
   (torch.cuda.max_memory_allocated()) around a real fwd+bwd+step, not from
   peak_memory_gib in trials.csv -- that is an eval-time number at half the
   microbatch and half the seq_len, and understates the training peak ~9x."

scripts/gpu_capacity.py is not usable here: its fitted model has no d_model
term, it understated the d768 scaling-grid cells by >3x, and all nine OOM'd.

WHY THIS IS THE BINDING CONSTRAINT AT RANK 48. ska.py:229 warns that
`ska_inverse_cholesky` keeps per-token statistics shaped (B, T, H, r, r), so
memory grows with microbatch * rank^2. At rank 48 that is 2.25x the rank-32
footprint for the same batch. The 180M pilot at per_device_batch_size=16 died
trying to allocate a single 15.66 GiB tensor with 71.11 GiB already resident.

DDP adds a gradient bucket roughly the size of the parameters on top of what a
single process measures, so the reported peak is a floor for the real per-rank
figure, not a ceiling. Headroom below 80 GiB should be taken seriously.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

SHAPES = {
    "180m": (576, 23),
    "440m": (832, 33),
    "1p5b": (1280, 54),
}


def measure(d_model: int, n_layers: int, pdbs: int, seq: int, rank: int,
            vocab: int) -> dict:
    import torch
    from experimentation.run import resolve
    from experimentation.sweep.search.geometry import make_layer_indices
    from koopman_lm.models.koopman_lm import KoopmanLM

    base = resolve.resolve_run_spec(REPO / "configs/runs/180m-joint-base.yaml").model
    idx = make_layer_indices(n_layers, 4, "even", list(base.ska_layer_indices))
    cfg = dataclasses.replace(base, d_model=d_model, n_layers=n_layers,
                              vocab_size=vocab, max_seq_len=seq, ska_rank=rank,
                              ska_layer_indices=tuple(idx))
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    # MIRROR THE TRAINER, or the number is meaningless. train.py:376-387 keeps
    # the model in its config dtype (fp32) and uses bf16 AUTOCAST plus FUSED
    # AdamW -- so the persistent cost is 16 bytes/param (fp32 params, fp32
    # grads, fp32 exp_avg + exp_avg_sq), not the 8 a bf16 model would show.
    # An earlier version of this probe cast the model with .to(bfloat16) and
    # understated 1.5B by ~11.5 GB, which is the whole margin at stake.
    model = KoopmanLM(cfg).cuda()
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.1,
                            fused=True)
    ids = torch.randint(0, vocab, (pdbs, seq), device="cuda")
    # Two steps: step 1 allocates the optimizer state lazily, so a single step
    # measures a peak that the second step would exceed.
    for _ in range(2):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(input_ids=ids, labels=ids)
        out["loss"].backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / 2**30
    del model, opt, ids, out
    torch.cuda.empty_cache()
    return {"n_params": n_params, "peak_gib": peak}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="180m,440m,1p5b")
    ap.add_argument("--pdbs", default="1,2,4,8")
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--rank", type=int, default=48)
    ap.add_argument("--vocab", type=int, default=128256)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)

    import torch
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    print(f"device: {torch.cuda.get_device_name(0)}, {total:.1f} GiB total")
    print(f"seq={a.seq} ska_rank={a.rank} vocab={a.vocab}\n")
    print("%-6s %6s %14s %10s %9s %s"
          % ("size", "pdbs", "params", "peak GiB", "headroom", "verdict"))
    results = []
    for size in a.sizes.split(","):
        d, L = SHAPES[size.strip()]
        for pdbs in (int(x) for x in a.pdbs.split(",")):
            try:
                r = measure(d, L, pdbs, a.seq, a.rank, a.vocab)
                head = total - r["peak_gib"]
                            # DDP allocates gradient buckets for the allreduce: another
                # fp32 copy of the gradients, 4 bytes/param.
                ddp_extra = r["n_params"] * 4 / 2**30
                ok = "OK" if head > ddp_extra + 8 else (
                    "TIGHT" if head > ddp_extra else "NO (DDP bucket won't fit)")
                print("%-6s %6d %14s %10.1f %9.1f %s"
                      % (size, pdbs, f"{r['n_params']:,}", r["peak_gib"], head, ok))
                results.append(dict(size=size, pdbs=pdbs, **r,
                                    headroom_gib=head, verdict=ok))
            except Exception as e:
                msg = type(e).__name__
                print("%-6s %6d %14s %10s %9s %s"
                      % (size, pdbs, "-", "-", "-", f"FAILED {msg}"))
                results.append(dict(size=size, pdbs=pdbs, error=msg))
                import torch as t
                t.cuda.empty_cache()
    if a.out:
        a.out.write_text(json.dumps(results, indent=2))
        print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
