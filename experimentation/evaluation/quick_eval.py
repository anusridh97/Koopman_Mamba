"""Cheap per-trial screening: held-out loss, and how much of it SKA earns.

A search needs one scalar per trial, fast. It also needs to know whether the
config it likes actually uses the branch under study -- a trial can improve loss
while making SKA irrelevant, which for a paper about SKA is a worse result than
it looks. So both numbers come out together: the full held-out loss, and the loss
with the SKA branch zeroed.

Assembled from parts that already existed. `evaluate.py::eval_fineweb_ppl` does
the held-out pass, `KoopmanLM.ablate(zero_ska=True)` does the ablation, and
`harness.py::_ska_delta` already combines them -- but the slow way, recomputing
the full pass to get its baseline, and reporting perplexity where a search wants
loss. This does one pass each and reports both.

Two deliberate choices.

**Results go through `evaluation/result.py::write_result`.** A trial's numbers
land at `run_dir/eval/<checkpoint>/quick_eval.json` inside the standard envelope,
so `python -m experimentation.results` aggregates them with no extra wiring and a
search's output is queryable the same way a hand-launched run's is. A standalone
script emitting its own JSON shape would have been quicker to write and invisible
to everything.

**`max_batches` is a cap, not a suggestion.** Every trial must see the same number
of tokens, or part of the loss difference between two trials measures how much
data each happened to read.

`ska_ablation.supported: False` is distinct from `loss_delta: 0.0`. The first
means there was no SKA branch to zero (a mamba-only baseline); the second means
there was one and it earned nothing. Collapsing them would hide the more
interesting result.
"""
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional

import torch

from experimentation.evaluation.result import write_result
from experimentation.run.provenance import git_commit

__all__ = ["TASK", "evaluate_loss", "assemble_quick_eval", "run_quick_eval",
           "write_quick_eval"]

TASK = "quick_eval"

# eval_fineweb_ppl clamps the exponent at 20 before exponentiating; matching it
# keeps a diverged trial's perplexity finite and comparable instead of inf.
_PPL_CLAMP = 20.0


def evaluate_loss(model, device, loader: Iterable[Mapping[str, Any]], *,
                  max_batches: Optional[int] = None) -> Dict[str, Any]:
    """Token-weighted mean loss over `loader`, with throughput and peak memory.

    Token-weighted rather than batch-averaged because a packed dataset's final
    batch can be short, and averaging losses instead of tokens would silently
    over-weight it.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    n_batches = 0
    if device is not None and str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    started = time.time()
    with torch.no_grad():
        for batch in loader:
            if max_batches is not None and n_batches >= max_batches:
                break
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs["loss"]
            n_tokens = labels.numel()
            total_loss += float(loss.item()) * n_tokens
            total_tokens += n_tokens
            n_batches += 1
    elapsed = max(time.time() - started, 1e-9)

    avg_loss = total_loss / total_tokens if total_tokens else 0.0
    peak_gib = None
    if device is not None and str(device).startswith("cuda") and torch.cuda.is_available():
        peak_gib = torch.cuda.max_memory_allocated() / (1024 ** 3)

    return {
        "loss": avg_loss,
        "ppl": math.exp(min(avg_loss, _PPL_CLAMP)),
        "n_tokens": total_tokens,
        "n_batches": n_batches,
        "tokens_per_sec": total_tokens / elapsed if total_tokens else 0.0,
        "peak_memory_gib": peak_gib,
    }


_FULL_KEYS = ("loss", "ppl", "n_tokens", "n_batches", "tokens_per_sec",
              "peak_memory_gib")


def assemble_quick_eval(full: Mapping[str, Any],
                        zeroed: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """The metrics payload: the full pass, plus the SKA-zeroed delta if measured.

    Deltas are zeroed-minus-full, so positive means the model got worse without
    SKA -- i.e. SKA is load-bearing. That sign convention matters because the
    search *rewards* the delta.
    """
    metrics: Dict[str, Any] = {"full": {k: full.get(k) for k in _FULL_KEYS}}
    if zeroed is None:
        metrics["ska_ablation"] = {"supported": False}
        return metrics
    metrics["ska_ablation"] = {
        "supported": True,
        "loss": zeroed["loss"],
        "ppl": zeroed["ppl"],
        "loss_delta": zeroed["loss"] - full["loss"],
        "ppl_delta": zeroed["ppl"] - full["ppl"],
    }
    return metrics


def run_quick_eval(model, device, *, data_dir, max_seq_len: int,
                   batch_size: int, max_batches: Optional[int] = None,
                   ska_ablation: bool = True) -> Dict[str, Any]:
    """Full pass, then (optionally) the SKA-zeroed pass, over a held-out shard.

    Needs a real model and a real shard, so it is exercised on a GPU node rather
    than in the CPU suite; its arithmetic lives in evaluate_loss and
    assemble_quick_eval, which are.
    """
    from torch.utils.data import DataLoader

    from experimentation.training.data.dataset import MemmapPackedDataset

    def _loader():
        # Rebuilt per pass so the second pass reads the same batches in the same
        # order -- a delta between two different token sets is not an ablation.
        dataset = MemmapPackedDataset(str(data_dir), max_seq_len, seed=0)
        return DataLoader(dataset, batch_size=batch_size, shuffle=False)

    full = evaluate_loss(model, device, _loader(), max_batches=max_batches)

    zeroed = None
    if ska_ablation and hasattr(model, "ablate"):
        with model.ablate(zero_ska=True):
            zeroed = evaluate_loss(model, device, _loader(), max_batches=max_batches)
    return assemble_quick_eval(full, zeroed)


def write_quick_eval(checkpoint, metrics: Mapping[str, Any]) -> Optional[Path]:
    """Write `metrics` into the checkpoint's run directory, or return None.

    Reuses evaluate.py::find_run_dir rather than repeating the walk. "A
    spec.yaml above it is what makes a directory a run directory" is a contract,
    and evaluation/result.py's own docstring makes the case against encoding a
    contract twice: the copy that drifts is the reader, and it fails silently.
    The import is deferred because evaluate.py pulls in transformers and the
    model package, which callers who only want evaluate_loss should not pay for.

    `run_id` is read back from spec.yaml rather than recomputed -- that hash is a
    function of a RunSpec this never reconstructs -- and the stamped commit is
    the *eval-time* one, deliberately not the training run's.
    """
    import yaml

    from experimentation.evaluation.evaluate import find_run_dir

    checkpoint = Path(checkpoint).resolve()
    run_dir = find_run_dir(checkpoint)
    if run_dir is None:
        return None

    raw_spec = yaml.safe_load((run_dir / "spec.yaml").read_text()) or {}
    return write_result(run_dir, checkpoint=checkpoint.parent.name, task=TASK,
                        metrics=dict(metrics), run_id=raw_spec.get("run_id", ""),
                        git_commit=git_commit())
