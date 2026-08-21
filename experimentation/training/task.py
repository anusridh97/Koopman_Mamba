"""`TrainTask`: the three things that genuinely differ between training loops.

`train.py` (600 lines), `experiments/mqar_finetune.py` (534) and
`experiments/table2.py` (390) implement one algorithm three times. The loop,
resume, DDP, gradient accumulation, parameter grouping, amp, logging and
checkpointing are the same everywhere. What differs is the **data source**, the
**loss**, and an optional **in-loop eval**.

Design: `docs/superpowers/specs/2026-08-21-traintask-design.md`.

## The loss must NOT be unified, and that is the whole point

Causal LM training offsets inputs from targets by one, and this repo does that
offset in two different places:

    shard path      the DATASET offsets. training/data/dataset.py:57 emits
                    input_ids=chunk[:-1], labels=chunk[1:], so KoopmanLM.forward
                    scores logits[i] against labels[i] -- positional, no shift.
    synthetic path  the LOSS offsets. curricula.make_mqar emits ALIGNED pairs,
                    and the loop scores logits[:, :-1] against labels[:, 1:].

Both are correct next-token prediction. From outside they are indistinguishable:
same function, same `ignore_index`, both descend. Which is exactly why "these two
loss computations are duplicates, route them through the model's built-in loss"
looks like obvious cleanup and is an off-by-one.

Measured: in aligned MQAR data the label at each supervised position IS the input
there, so the positional loss scores a model on **copying a token it was just
handed**. A one-hot input-copier scores <1e-4 under the positional loss and >1.0
under the shifted one. See `code-tests/test_loss_alignment.py`.

So the convention is one invariant with three surfaces -- data prep, loss, in-loop
eval -- and a task owns all three or none.

## Why `step_loss` performs the forward call

It receives the model, not logits. That is what lets `ShardTask` call
`model(input_ids, labels, loss_weights)["loss"]` (loss computed inside the model,
on pre-offset data) while `SyntheticTask` calls `model(input_ids)` and offsets
itself -- each preserving its own convention exactly, with no flag and no branch
in the loop. The loop enters autocast around this call; a task never manages
precision.

## `build_model` is a fourth responsibility, deliberately

Mildly against the design's "a task owns nothing else". It exists so the
extension-mechanisms design's §4 (construction-time variation) and §5
(post-construction mutation) have somewhere to attach later. Nothing uses it for
that yet; it is cheap now and expensive to retrofit once call sites exist.

## `dataset`, not an iterator

`train.py` does not consume a `DataLoader`, it CONSTRUCTS one per epoch from an
explicit index list, because exact resume must skip already-consumed samples
without re-reading them (`train.py:376-393`, `resume.py:epoch_permutation`). It
also passes `generator=torch.Generator()` -- required, not an optimisation, since
`DataLoader.__iter__` draws from the global RNG on every fresh iteration and a
resumed run would otherwise desync its dropout masks. So a task hands over an
indexable `Dataset` and the loop keeps sampling and resume.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["TrainTask", "ShardTask", "SyntheticTask", "IGNORE_INDEX"]

#: Both conventions agree on this; only *where* the offset happens differs.
IGNORE_INDEX = -100


class TrainTask:
    """Base class. Subclasses supply data, loss, and optionally an in-loop eval.

    Not a `typing.Protocol`, because the two optional hooks want real defaults --
    a Protocol would make every task restate them.
    """

    #: Does this task's data arrive already offset (shard style), or aligned with
    #: the offset applied at loss time (synthetic style)? Declared rather than
    #: inferred: it cannot be recovered from a batch by inspection, and stating
    #: it is what lets a conformance test check that `step_loss` actually reads
    #: the position the convention claims.
    shift_in_data: bool = True

    name: str = "task"

    # ---------------------------------------------------------------- required

    def build_model(self, cfg) -> nn.Module:
        raise NotImplementedError

    def dataset(self, cfg, args) -> torch.utils.data.Dataset:
        raise NotImplementedError

    def step_loss(self, model, batch) -> torch.Tensor:
        raise NotImplementedError

    # ---------------------------------------------------------------- optional

    def in_loop_eval(self, model, step: int) -> Optional[Mapping[str, Any]]:
        """Periodic eval during training, or None. MQAR accuracy lives here."""
        return None

    def on_final(self, model, run_dir, cfg) -> None:
        """After the last checkpoint. Where the search's objective gets written."""
        return None


class ShardTask(TrainTask):
    """Pretraining on a pretokenized shard. `train.py`'s behaviour, unchanged.

    The dataset already offset the pair, so the loss is positional and computed
    INSIDE the model -- which is also what carries `loss_weights` through to the
    recall-weighted CE that only this path uses.
    """

    #: The dataset offsets (dataset.py:57), so the model's positional CE is
    #: correct and no shift happens here.
    shift_in_data = True
    name = "shard"

    def build_model(self, cfg) -> nn.Module:
        from koopman_lm.models.koopman_lm import KoopmanLM
        return KoopmanLM(cfg)

    def dataset(self, cfg, args) -> torch.utils.data.Dataset:
        from experimentation.training.data.dataset import MemmapPackedDataset
        return MemmapPackedDataset(args.data_dir, args.max_seq_len, seed=args.seed)

    def step_loss(self, model, batch) -> torch.Tensor:
        """Exactly `train.py:404-405`, including the loss_weights passthrough.

        `loss_weights` is not optional decoration: the weighted branch of
        KoopmanLM.forward normalises by the weight sum rather than the token
        count, so dropping it would silently change the loss even when every
        weight is 1.0 -- a different reduction, not a different scale.
        """
        loss_weights = batch.get("loss_weights")
        out = model(input_ids=batch["input_ids"], labels=batch["labels"],
                    loss_weights=loss_weights)
        return out["loss"]


class SyntheticTask(TrainTask):
    """A curricula.py generator. `mqar_finetune.py`'s behaviour, unchanged.

    The generator emits ALIGNED pairs -- `labels[p]` is the answer belonging at
    position p -- so the offset happens here, at loss time. Feeding this data to
    a positional loss would score the model on echoing its own input.
    """

    shift_in_data = False
    name = "synthetic"

    def __init__(self, *, model_type: str = "mamba_ska_swiglu",
                 eval_fn=None, eval_every: int = 0):
        self.model_type = model_type
        self._eval_fn = eval_fn
        self.eval_every = eval_every

    def build_model(self, cfg) -> nn.Module:
        from koopman_lm.models.baselines import (
            build_mamba_attention, build_mamba_only, build_mamba_ska_koopman,
            build_mamba_ska_swiglu)
        builders = {
            "mamba_only": build_mamba_only,
            "mamba_attn": build_mamba_attention,
            "mamba_ska_swiglu": build_mamba_ska_swiglu,
            "mamba_ska_koopman": build_mamba_ska_koopman,
        }
        try:
            return builders[self.model_type](cfg)
        except KeyError as exc:
            raise ValueError(
                f"unsupported model_type {self.model_type!r}; expected one of "
                f"{sorted(builders)}") from exc

    def dataset(self, cfg, args) -> torch.utils.data.Dataset:
        from experimentation.experiments.mqar_finetune import MQARDataset
        seq_len = 4 * args.num_kv_pairs + args.distractor_gap
        return MQARDataset(seq_len=seq_len, num_kv_pairs=args.num_kv_pairs,
                           vocab_size=args.task_vocab_size,
                           epoch_size=args.epoch_size, base_seed=args.seed)

    def step_loss(self, model, batch) -> torch.Tensor:
        """Exactly `mqar_finetune.py:296-301`.

        `logits[:, :-1]` against `labels[:, 1:]`: logits at position t are scored
        against the label at t+1, so the model is asked to PREDICT the answer
        rather than repeat it. `eval_mqar` (curricula.py:99) offsets identically,
        and it has to -- an accuracy computed at the other offset would report
        plausible numbers forever.
        """
        input_ids, labels = _unpack(batch)
        logits = _logits(model(input_ids=input_ids))
        return F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            labels[:, 1:].reshape(-1),
            ignore_index=IGNORE_INDEX,
        )

    def in_loop_eval(self, model, step: int) -> Optional[Mapping[str, Any]]:
        if not self._eval_fn or not self.eval_every:
            return None
        if step % self.eval_every:
            return None
        return self._eval_fn(model, step)


def _unpack(batch):
    """A synthetic generator yields a bare (inputs, labels) tuple; the shard
    dataset yields a dict. Accept both so a task is not coupled to which."""
    if isinstance(batch, Mapping):
        return batch["input_ids"], batch["labels"]
    input_ids, labels = batch
    return input_ids, labels


def _logits(out):
    """KoopmanLM.forward returns a dict; the baselines may return a tensor."""
    return out["logits"] if isinstance(out, Mapping) else out
