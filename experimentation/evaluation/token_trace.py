"""Per-token log probabilities, saved as a result rather than rendered.

Every number this repo reports about a trained model is an aggregate.
`quick_eval` says loss 7.72; `ska_ablation` says the SKA branch is worth 1.45e-05.
Both are true and neither says *where* -- which tokens the model is confident on,
which surprise it, and which ones the SKA adapters actually contribute to.

This computes that per token and **writes it through the same result envelope
`quick_eval` uses** (`evaluation/result.py::write_result`), landing at
`run_dir/eval/<checkpoint>/token_trace.json` with `run_id`, `git_commit` and
`created_at` beside the numbers.

## Why an artifact and not a renderer

The first version of this did the forward passes AND emitted HTML in one script,
which welds the expensive half to the cheap half. Splitting them buys four things:

  * **Compute once, view many.** Re-render with different colours, thresholds or
    top-k without touching a GPU.
  * **The renderer becomes a pure function** of a JSON file, so it is testable
    from a committed fixture instead of a stubbed model.
  * **Traces become comparable.** Two checkpoints' traces answer "which tokens got
    better between step 200 and step 400?", which is impossible when the numbers
    only exist inside a rendered page.
  * **A search trial can emit one**, exactly as it now emits quick_eval.json, so
    the aggregate and the per-token detail carry the same provenance.

It is also the convention this repo already commits to: `metrics.py`'s docstring
says the searcher holds "no subprocess handle or pipe" precisely so a result can
be read hours later on another machine. `spec.yaml` is the recipe,
`attempts.jsonl` the ledger, `quick_eval.json` the score. This is the detail.

## Size

A trace is bounded on purpose. `--sequences` and `--top_k` default low because
these land in run directories that already hold 200MB checkpoints, and a
2048-token sequence at top-20 is a few MB per sequence. `summary.bytes_estimate`
is reported so a caller can see what it is about to write.
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

__all__ = ["TASK", "token_report", "assemble_trace", "write_trace", "main"]

TASK = "token_trace"


def _logits(out):
    return out["logits"] if isinstance(out, dict) else out


def _decode(tokenizer, token_id: int) -> str:
    if tokenizer is None:
        return f"<{token_id}>"
    try:
        return tokenizer.decode([token_id])
    except Exception:                                   # noqa: BLE001
        return f"<{token_id}>"


def token_report(model, input_ids: torch.Tensor, labels: torch.Tensor,
                 tokenizer=None, *, top_k: int = 5,
                 ablate: bool = True) -> Dict[str, Any]:
    """One sequence -> per-token log prob, top-k, rank, and SKA-ablation delta.

    `input_ids`/`labels` must arrive offset the way the shard dataset offsets them
    (labels[i] is the target FOR position i), so this reads the model exactly as
    ShardTask's loss does. Handing it aligned synthetic data would silently report
    the off-by-one that code-tests/test_loss_alignment.py exists to prevent --
    every token would look trivially predictable, because at each supervised
    position the label IS the input there.
    """
    model.eval()
    with torch.no_grad():
        logprobs = torch.log_softmax(
            _logits(model(input_ids=input_ids.unsqueeze(0)))[0].float(), dim=-1)
        ablated = None
        if ablate and hasattr(model, "ablate"):
            with model.ablate(zero_ska=True):
                ablated = torch.log_softmax(
                    _logits(model(input_ids=input_ids.unsqueeze(0)))[0].float(),
                    dim=-1)

    k = min(top_k, logprobs.size(-1))
    tokens: List[Dict[str, Any]] = []
    for i, target in enumerate(labels.tolist()):
        if target < 0:                 # ignore_index: nothing supervised here
            continue
        lp = float(logprobs[i, target])
        top = torch.topk(logprobs[i], k=k)
        entry: Dict[str, Any] = {
            "pos": i,
            "token": _decode(tokenizer, target),
            "logprob": round(lp, 6),
            # Rank of the true token: 1 means the model would have emitted it.
            # More legible than a probability when the distribution is flat.
            "rank": int((logprobs[i] > logprobs[i, target]).sum()) + 1,
            "top": [[_decode(tokenizer, int(t)), round(math.exp(float(v)), 6)]
                    for v, t in zip(top.values, top.indices)],
        }
        if ablated is not None:
            # POSITIVE means SKA helped this token: zeroing the branch made its
            # log prob worse. Sign matters and is easy to invert by accident.
            entry["ska_delta"] = round(lp - float(ablated[i, target]), 8)
        tokens.append(entry)
    return {"tokens": tokens}


def assemble_trace(reports: List[Dict[str, Any]], *,
                   top_k: int, source: str) -> Dict[str, Any]:
    """Reports -> the `metrics` payload, with a summary the renderer can trust.

    The summary carries the percentile range and the symmetric ablation bound so
    the viewer does not have to recompute them -- and, more importantly, so two
    traces rendered separately can be put on the SAME scale by reading each
    other's bounds instead of each normalising to itself.
    """
    all_lp = sorted(t["logprob"] for r in reports for t in r["tokens"])
    deltas = [t["ska_delta"] for r in reports for t in r["tokens"]
              if t.get("ska_delta") is not None]
    n = len(all_lp)
    if not n:
        raise ValueError("no supervised tokens in any sequence")

    return {
        "sequences": reports,
        "summary": {
            "n_sequences": len(reports),
            "n_tokens": n,
            "top_k": top_k,
            "source": source,
            "mean_logprob": round(sum(all_lp) / n, 6),
            # 5th/95th, not min/max: one pathological token otherwise flattens a
            # colour ramp into a single shade.
            "logprob_p05": all_lp[int(0.05 * (n - 1))],
            "logprob_p95": all_lp[int(0.95 * (n - 1))],
            "has_ablation": bool(deltas),
            # Symmetric about zero so a diverging scale cannot misreport sign.
            "ska_delta_absmax": round(max((abs(d) for d in deltas), default=0.0), 8),
            "ska_delta_mean": (round(sum(deltas) / len(deltas), 8) if deltas else None),
            # Rough, but enough to notice before writing something enormous next
            # to a 200MB checkpoint.
            "bytes_estimate": n * (90 + 26 * top_k),
        },
    }


def write_trace(checkpoint, payload: Dict[str, Any]) -> Optional[Path]:
    """Write into the checkpoint's run directory, or return None if it has none.

    Reuses quick_eval's exact path: find_run_dir + write_result, so a trace sits
    beside the score it explains and `python -m experimentation.results` can find
    both by the same walk.
    """
    import yaml

    from experimentation.evaluation.evaluate import find_run_dir
    from experimentation.evaluation.result import write_result
    from experimentation.run.provenance import git_commit

    checkpoint = Path(checkpoint).resolve()
    run_dir = find_run_dir(checkpoint)
    if run_dir is None:
        return None
    raw_spec = yaml.safe_load((run_dir / "spec.yaml").read_text()) or {}
    return write_result(run_dir, checkpoint=checkpoint.parent.name, task=TASK,
                        metrics=payload, run_id=raw_spec.get("run_id", ""),
                        git_commit=git_commit())


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m experimentation.evaluation.token_trace",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_dir", required=True, help="a pretokenized shard")
    p.add_argument("--sequences", type=int, default=3)
    p.add_argument("--max_seq_len", type=int, default=128,
                   help="short on purpose: this is for reading, not measuring")
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--device", default=None)
    p.add_argument("--no_ablation", action="store_true",
                   help="skip the SKA-zeroed pass (halves the forward work)")
    p.add_argument("--out", default=None,
                   help="write here instead of the run directory (for a "
                        "checkpoint that has no run dir)")
    args = p.parse_args(argv)

    from experimentation.evaluation.evaluate import load_model
    from experimentation.training.data.dataset import MemmapPackedDataset

    device = torch.device(args.device or
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    model, cfg, tokenizer, model_type = load_model(args.checkpoint)
    model = model.to(device).eval()

    dataset = MemmapPackedDataset(args.data_dir, args.max_seq_len, seed=0)
    reports = []
    for i in range(min(args.sequences, len(dataset))):
        item = dataset[i]
        reports.append(token_report(
            model, item["input_ids"].to(device), item["labels"].to(device),
            tokenizer, top_k=args.top_k, ablate=not args.no_ablation))

    payload = assemble_trace(reports, top_k=args.top_k, source=args.data_dir)
    payload["summary"]["model_type"] = model_type

    if args.out:
        import json
        Path(args.out).write_text(json.dumps(payload, indent=2))
        written = Path(args.out)
    else:
        written = write_trace(args.checkpoint, payload)
        if written is None:
            print("no run directory above this checkpoint; pass --out")
            return 1

    s = payload["summary"]
    print(f"wrote {written}")
    print(f"  {s['n_tokens']} tokens over {s['n_sequences']} sequence(s), "
          f"mean logprob {s['mean_logprob']}")
    if s["has_ablation"]:
        print(f"  SKA delta: mean {s['ska_delta_mean']}, "
              f"|max| {s['ska_delta_absmax']}")
    else:
        print("  no SKA ablation -- this model exposes no .ablate()")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
