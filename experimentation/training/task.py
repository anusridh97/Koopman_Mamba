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

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["TrainTask", "ShardTask", "SyntheticTask", "Table2Task",
           "IterContext", "synthetic_loss", "IGNORE_INDEX"]

#: Both conventions agree on this; only *where* the offset happens differs.
IGNORE_INDEX = -100


def synthetic_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """The synthetic-convention loss, as ONE definition.

    `logits[:, :-1]` against `labels[:, 1:]`: the logits at position t are scored
    against the label at t+1, so the model must PREDICT the answer rather than
    repeat it. `eval_mqar` (curricula.py:99) offsets identically, and it has to --
    an accuracy computed at the other offset would report plausible numbers
    forever.

    Exists as a function, not only as `SyntheticTask.step_loss`, because
    `step_loss` takes the MODEL and runs the forward -- which is right for a
    training step and wrong for table2's per-task eval, which deliberately
    reuses the training forward's logits rather than paying for a second pass.
    table2.py spelled this expression out three times (the step loss and both
    curriculum halves); it now has one definition and three call sites.

    The design doc's §6.3 anticipated table2's dataset and its fp16 scaler, but
    not that its in-loop eval shares the training forward. This is that
    resolution: share the expression, leave the interface alone.
    """
    return F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.size(-1)),
        labels[:, 1:].reshape(-1),
        ignore_index=IGNORE_INDEX,
    )


@dataclass
class IterContext:
    """Everything `iter_batches` needs about where the loop is resuming from.

    A context object rather than five keyword arguments, so adding a sixth does
    not have to be threaded through every override.
    """

    #: Which pass over the dataset this is. Meaningless for step-keyed data.
    epoch: int = 0
    #: The last COMPLETED optimizer step. Iteration resumes after it.
    start_step: int = 0
    #: Samples already consumed within `epoch`, to be skipped. Epoch paths only.
    skip_samples: int = 0
    #: DDP sampler, when the loop is running distributed.
    sampler: Any = None
    is_ddp: bool = False
    local_rank: int = 0


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

    #: Which precision this loop has always trained in, so a unified loop can
    #: honour it instead of imposing one. Only `Table2Task` sets fp16, and it is
    #: the only loop that runs a GradScaler -- so this field is what keeps a
    #: shared loop from silently retraining table2 in bf16, which would change
    #: training while still producing a descending curve. `None` means "whatever
    #: the config/runtime says", which is every other task's existing behaviour.
    compute_precision: Optional[str] = None

    # ---------------------------------------------------------------- required

    def build_model(self, cfg) -> nn.Module:
        raise NotImplementedError

    def dataset(self, cfg, args) -> torch.utils.data.Dataset:
        raise NotImplementedError

    def step_loss(self, model, batch) -> torch.Tensor:
        raise NotImplementedError

    # ---------------------------------------------------------------- optional

    def iter_batches(self, dataset, args, ctx: "IterContext"):
        """One epoch's batches, in the order this task's resume arithmetic needs.

        Defaulted rather than abstract because two of the three tasks want the
        same thing: an explicit epoch permutation over a map dataset, resumed by
        skipping the consumed prefix. Only `Table2Task` differs, and it differs
        completely -- its data is a function of the step counter, with no epochs
        and no shuffle.

        This is the seam the design doc left unresolved. Its ownership table puts
        `dataset` on the task and `resume` on the loop, while §3 says the two are
        entangled -- and they are, because "where do I resume in the data" is
        answered by index arithmetic for a shard and by a step number for
        table2. Defaulting here keeps ONE loop rather than an `if` inside it.

        Lifted from train.py, not rewritten. Two details are load-bearing:

        * the order comes from `epoch_permutation`, NOT DataLoader's implicit
          `shuffle=True`, so a mid-epoch resume can skip forward over consumed
          indices without re-reading them;
        * `generator=` is REQUIRED, not an optimisation. `DataLoader.__iter__`
          draws one int64 from the GLOBAL torch RNG on every fresh iteration to
          seed `_base_seed`, even at `num_workers=0` with an explicit sampler. A
          resumed run builds a new DataLoader mid-epoch and would pay a draw its
          uninterrupted twin never pays there, desyncing dropout from the first
          resumed step.
        """
        from torch.utils.data import DataLoader

        from experimentation.training.repro import seed_worker
        from experimentation.training.resume import (epoch_permutation,
                                                     resume_indices)

        if ctx.is_ddp and ctx.sampler is not None:
            ctx.sampler.set_epoch(ctx.epoch)
            indices = list(ctx.sampler)
            if ctx.skip_samples:
                indices = indices[ctx.skip_samples:]
        else:
            seed = args.seed + ctx.local_rank
            indices = (resume_indices(len(dataset), seed, ctx.epoch, ctx.skip_samples)
                       if ctx.skip_samples
                       else epoch_permutation(len(dataset), seed, ctx.epoch))

        return DataLoader(
            dataset, batch_size=args.per_device_train_batch_size,
            sampler=indices, num_workers=args.num_workers,
            pin_memory=True, drop_last=True, worker_init_fn=seed_worker,
            generator=torch.Generator())

    def progress_fields(self, *, window_avg: float, last_loss: float):
        """`(loss to report, extra text)` for the progress line.

        The loop owns the FORMAT -- `loss %.4f | ppl | lr | K tok/s` is parsed by
        both `scripts/compare_golden_curve.py` and the search pruner -- but the
        three trainers genuinely disagree about WHICH loss belongs in it:

            train.py   running_loss / loss_count      window average
            mqar       loss.item()                    the instantaneous batch
            table2     mean of PER-TASK window averages, plus a per-task tail

        Unifying without this seam would change at least one trainer's every
        logged value. The logged value IS the golden curve, so that destroys the
        one instrument able to distinguish "reporting changed" from "training
        broke" at exactly the moment it is needed.

        Standardising on the window average may well be right. It is a deliberate
        change with a re-captured golden, not something a refactor smuggles in.
        """
        return window_avg, ""

    def in_loop_eval(self, model, step: int) -> Optional[Mapping[str, Any]]:
        """Periodic eval during training, or None. MQAR accuracy lives here.

        Offered EVERY optimizer step; the task owns its own cadence, because
        only it knows how expensive its eval is."""
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
        #: The most recent in-loop eval result. mqar passes its latest accuracy
        #: into save_checkpoint, and with the loop owning checkpointing the task
        #: is what carries that value across.
        self.last_eval = None

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
        return synthetic_loss(_logits(model(input_ids=input_ids)), labels)

    def iter_batches(self, dataset, args, ctx: "IterContext"):
        """`DataLoader(..., shuffle=True)`, exactly as mqar_finetune builds it.

        NOT the base class's epoch permutation. That is a different batch order,
        and this trainer's golden curve encodes the order it has always used --
        so inheriting the default would move every value in it for a reason
        having nothing to do with the refactor.

        Also reads `args.batch_size`, not `args.per_device_train_batch_size`:
        mqar's argparse spells it the first way and the shard trainer the second.
        Reading the wrong one silently changes the effective batch.

        No `generator=` and no `worker_init_fn`, again matching the original. The
        shard path needs both for exact mid-epoch resume; this one re-iterates a
        loader from the top each epoch, so there is no position to preserve.
        """
        from torch.utils.data import DataLoader

        return DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=True,
                          drop_last=True)

    def progress_fields(self, *, window_avg: float, last_loss: float):
        """mqar prints `loss.item()` -- the instantaneous batch, not a window
        average. See TrainTask.progress_fields for why this is a seam rather
        than something the unification standardises away."""
        return last_loss, ""

    def in_loop_eval(self, model, step: int) -> Optional[Mapping[str, Any]]:
        if not self._eval_fn or not self.eval_every:
            return None
        if step % self.eval_every:
            return None
        self.last_eval = self._eval_fn(model, step)
        return self.last_eval


class _StepKeyedBatches(torch.utils.data.Dataset):
    """`make_train_batch(step, args)` as a Dataset, indexed by the STEP number.

    Two things this exists to get right, both from the design doc §6.3.

    **The index is the step, not a zero offset.** table2's loop is
    `range(start_step + 1, max_steps + 1)`, so its first step is 1 while a
    Dataset is conventionally indexed from 0. That is not cosmetic here:
    `--curriculum mixed` alternates whole batches on `step % 2`, so a one-place
    shift swaps the two tasks for every batch of the run and still produces a
    loss that descends. Index 0 is therefore rejected rather than quietly mapped.

    **One index is one whole BATCH.** `__getitem__` conventionally returns a
    single sample for `DataLoader` to stack, but `make_train_batch` returns the
    batch. Handed to `DataLoader(batch_size=8)` this yields `[8, B, T]` *and*
    burns 8 steps of curriculum per iteration -- the shape is catchable, the
    curriculum burn shows up only in a curve. So the requirement travels with
    the object as `dataloader_batch_size = None` rather than living in a comment
    at the one call site that currently knows about it.

    Being keyed on `step` also makes resume exact: step 400 after a restart
    draws exactly what step 400 would have drawn, with no epoch position to
    recover. That is stronger than the shard path, which has to fast-forward.
    """

    #: For `DataLoader(dataset, batch_size=dataset.dataloader_batch_size)`.
    #: MUST be None -- DataLoader's default is 1, which would add a phantom
    #: leading axis instead of raising.
    dataloader_batch_size = None

    def __init__(self, make_batch, args, max_steps: int):
        self._make_batch = make_batch
        self._args = args
        self._max_steps = int(max_steps)

    def __len__(self) -> int:
        # +1 because step numbering is 1-based and `__getitem__(max_steps)` must
        # be in range.
        return self._max_steps + 1

    def __getitem__(self, step: int):
        if step <= 0:
            raise IndexError(
                f"step index {step} is out of range: table2 steps are 1-based "
                f"(`range(start_step + 1, max_steps + 1)`), and treating index 0 "
                f"as the first step shifts the whole run's curriculum parity")
        return self._make_batch(int(step), self._args)


class Table2Task(SyntheticTask):
    """table2's curriculum training, the third loop.

    Subclasses `SyntheticTask` because its loss is not merely equivalent to it,
    it is character-identical -- `logits[:, :-1]` against `labels[:, 1:]` with
    `ignore_index` -100, spelled as a literal in table2.py and as
    `IGNORE_INDEX` here. Re-deriving it would create a second copy of the one
    expression the alignment suite exists to pin.

    What is genuinely different is everything around it: no dataset (see
    `_StepKeyedBatches`), fp16 with a GradScaler, and a two-task eval split.
    """

    name = "table2"
    shift_in_data = False
    #: The only task that sets this. table2 has always trained fp16 + GradScaler
    #: and nothing else exercises the loop's scaler path.
    compute_precision = "fp16"

    def __init__(self, *, max_steps: int = 400, **kwargs):
        super().__init__(**kwargs)
        self._max_steps = max_steps

    def dataset(self, cfg, args) -> torch.utils.data.Dataset:
        from experimentation.experiments.table2 import make_train_batch

        return _StepKeyedBatches(make_train_batch, args,
                                 getattr(args, "max_steps", self._max_steps))

    def iter_batches(self, dataset, args, ctx: "IterContext"):
        """One batch per step, in step order, starting after `ctx.start_step`.

        No DataLoader and no permutation, deliberately. table2's batches are a
        pure function of the step counter, so an epoch permutation would not be
        merely unnecessary -- it would hand this trainer different batches than
        it has ever trained on. Matching `range(start_step + 1, max_steps + 1)`
        is also what makes its resume exact with no position to recover.
        """
        last = getattr(args, "max_steps", self._max_steps)
        return (dataset[step] for step in range(ctx.start_step + 1, last + 1))


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
