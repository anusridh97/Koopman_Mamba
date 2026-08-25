#!/usr/bin/env python
"""What the trained write gate actually does, per SKA layer, per checkpoint.

Reports `beta_mean`, `beta_sd_token`, `beta_sd_head`, `beta_min` and `beta_max`
from `diagnostics.ska.ska_health` for every SKA layer of every checkpoint given,
on one batch of real held-out text.

## THIS DOES NOT MEASURE WHETHER THE GATE HELPS

Stated first because it is the way this number gets misused. "Beta develops
meaningful token and head variation" is a NECESSARY CONDITION for the gate to be
doing anything -- a gate stuck at its initial 0.5 cannot be earning its
parameters -- and it is not evidence of benefit. A gate can vary enormously and
help nothing. The loss effect is measured by
`scripts/analyze_beta_policy.py` against a pooled noise floor, and these two
outputs are reported side by side precisely so that neither substitutes for the
other.

What this output CAN establish, and it is worth having:

  * A null loss result is interpretable. If `beta_policy=one` is
    indistinguishable from `learned` AND the learned gate never moved off 0.5,
    the null says "this gate did not train" -- a fixable problem. If the gate
    developed strong variation and STILL bought nothing, the null says "the
    mechanism is not useful here" -- a much stronger claim, and the one worth
    reporting.
  * The two variances separate content dependence from a per-head write scale.
    A gate showing only head variation is supplying a scale `out_proj` could
    absorb, so `head_scalar` (H parameters) would be the honest model of it
    rather than a full projection (d*H + H).

## Why real text and not random hidden states

`ska_health` reads a batch of hidden states through the model's own projections.
Random Gaussians would exercise the parameterisation but not the DISTRIBUTION a
trained gate produces -- which is the entire question. So the batch comes from
the same held-out shard the study scores on.

Usage:
    python scripts/report_beta_distribution.py \\
        --eval_data_dir /scratch/.../fineweb_small_val \\
        RUN_DIR [RUN_DIR ...]

Each RUN_DIR is a trial directory containing `final/model.pt` and `spec.yaml`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


def _hidden_states(model, input_ids, layer_index):
    """The hidden states entering the SKA block at `layer_index`.

    Captured with a forward hook rather than recomputed: the embedding, every
    preceding block and the pre-norm all sit between the token ids and the
    tensor `SKAModule.forward` sees, and reimplementing that path is how a
    diagnostic ends up describing something the model does not compute.
    """
    captured = {}

    def hook(_module, args, _output):
        captured["h"] = args[0].detach()

    from koopman_lm.modules.seq.ska import SKAModule

    # Selected by ORDER in `named_modules`, not by matching the layer index into
    # a module name: `ska_mode` decides whether SKAModule sits inside an
    # SKABlock or inside a MambaSKAParallelBlock, so the name's depth differs
    # between the two and a name-matching version would silently pick nothing on
    # one of them. `named_modules` walks in registration order, which is layer
    # order, so the nth SKAModule is the nth SKA layer.
    skas = [m for _n, m in model.named_modules() if isinstance(m, SKAModule)]
    if layer_index >= len(skas):
        raise IndexError(
            f"asked for SKA layer {layer_index} but the model has {len(skas)}. "
            f"The caller derives the count from spec.ska_layer_indices, so this "
            f"means the checkpoint's architecture disagrees with its spec.yaml "
            f"-- a bare IndexError here would be cryptic.")
    target = skas[layer_index]
    handle = target.register_forward_hook(hook, with_kwargs=False)
    try:
        with torch.no_grad():
            model(input_ids=input_ids)
    finally:
        handle.remove()
    return captured.get("h"), target


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--eval_data_dir", required=True)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--json_out", type=Path, default=None)
    args = parser.parse_args(argv)

    from experimentation.run.resolve import load_materialized_spec
    from experimentation.training.data.dataset import MemmapPackedDataset
    from koopman_lm.diagnostics.ska import ska_health
    from koopman_lm.models.koopman_lm import KoopmanLM

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []

    for run_dir in args.run_dirs:
        spec_path = run_dir / "spec.yaml"
        ckpt = run_dir / "final" / "model.pt"
        if not spec_path.exists() or not ckpt.exists():
            print(f"SKIP {run_dir}: missing spec.yaml or final/model.pt",
                  file=sys.stderr)
            continue
        spec = load_materialized_spec(spec_path)
        model = KoopmanLM(spec.model)
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        state = state.get("model", state) if isinstance(state, dict) else state
        # strict=True, and this is not pedantry. Under strict=False a key
        # mismatch (renamed module, a spec whose beta_policy disagrees with the
        # checkpoint, a partial save) leaves `beta_proj` at its ZERO
        # initialisation -- so `_resolve_beta` returns a constant 0.5 and this
        # script prints "CONSTANT -- gate carries no information". That is
        # exactly this script's headline finding, produced by a failed load. The
        # one reading it most needs to be trusted on is the one strict=False
        # would fabricate.
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"SKIP {run_dir}: checkpoint does not match the spec's "
                  f"architecture -- missing={sorted(missing)[:5]} "
                  f"unexpected={sorted(unexpected)[:5]}. Loading it anyway "
                  f"would leave the write gate at its zero init and this "
                  f"script would report a CONSTANT gate, which is its own "
                  f"headline false negative.", file=sys.stderr)
            del model
            continue
        model = model.to(device).eval()

        ds = MemmapPackedDataset(args.eval_data_dir, args.seq_len)
        ids = torch.stack([ds[i]["input_ids"]
                           for i in range(args.batch)]).to(device)

        n_ska = len(spec.model.ska_layer_indices)
        for layer in range(n_ska):
            h, ska = _hidden_states(model, ids, layer)
            if h is None:
                print(f"SKIP {run_dir} layer {layer}: hook captured nothing",
                      file=sys.stderr)
                continue
            # max_batch=args.batch: ska_health defaults to 2 and TRUNCATES, so
            # `--batch 4` silently reported statistics over two sequences.
            health = ska_health(ska, h, max_batch=args.batch)
            rows.append({
                "run_dir": str(run_dir),
                "beta_policy": spec.model.ska_beta_policy,
                "seed": spec.runtime.seed,
                "ska_layer": spec.model.ska_layer_indices[layer],
                **{k: float(health[k]) for k in
                   ("beta_mean", "beta_sd_token", "beta_sd_head",
                    "beta_min", "beta_max")},
            })
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    if not rows:
        print("FATAL: nothing measured", file=sys.stderr)
        return 2

    print(f"{'policy':>12} {'seed':>4} {'layer':>5} {'mean':>8} "
          f"{'sd_token':>9} {'sd_head':>9} {'min':>8} {'max':>8}")
    for row in rows:
        print(f"{row['beta_policy']:>12} {row['seed']:>4} {row['ska_layer']:>5} "
              f"{row['beta_mean']:>8.4f} {row['beta_sd_token']:>9.5f} "
              f"{row['beta_sd_head']:>9.5f} {row['beta_min']:>8.4f} "
              f"{row['beta_max']:>8.4f}")

    # Per-policy summary, since the per-layer rows are what a reader scans and
    # the per-policy means are what gets quoted.
    print()
    by_policy: dict[str, list[dict]] = {}
    for row in rows:
        by_policy.setdefault(row["beta_policy"], []).append(row)
    print(f"{'policy':>12} {'n':>3} {'mean':>8} {'sd_token':>9} {'sd_head':>9}"
          f"  reading")
    for policy, group in sorted(by_policy.items()):
        mean = sum(r["beta_mean"] for r in group) / len(group)
        sd_t = sum(r["beta_sd_token"] for r in group) / len(group)
        sd_h = sum(r["beta_sd_head"] for r in group) / len(group)
        # 0.5 is `learned`'s initialisation, so a gate that never left it did
        # not train. Threshold on the VARIANCES rather than on the mean: a gate
        # can move its mean off 0.5 without becoming content-dependent, and it
        # is the content dependence that would justify the projection.
        if sd_t < 1e-4 and sd_h < 1e-4:
            reading = "CONSTANT -- gate carries no information"
        elif sd_t < 1e-4:
            reading = "per-head scale only (out_proj could absorb this)"
        else:
            reading = "token-dependent"
        print(f"{policy:>12} {len(group):>3} {mean:>8.4f} {sd_t:>9.5f} "
              f"{sd_h:>9.5f}  {reading}")
    print()
    print("NOTE: variation is a NECESSARY CONDITION, not evidence of benefit.")
    print("      Read alongside scripts/analyze_beta_policy.py, never instead.")

    if args.json_out:
        args.json_out.write_text(json.dumps(rows, indent=2))
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
