"""TrainTask conformance: equivalence to the code it replaces, and alignment.

Two obligations, and the first is what makes the loop migration safe.

**Equivalence.** Each task's `step_loss` must be bit-identical to the inline
expression it replaces -- `train.py:404-405` for ShardTask,
`mqar_finetune.py:296-301` for SyntheticTask. Proving that BEFORE the loop is
touched means the loop extraction only has to call `task.step_loss` instead of
inlining, with equivalence already pinned. A refactor whose two halves are each
verified separately is a much smaller risk than one verified only at the end.

**Alignment.** Each task declares `shift_in_data`, and the declaration has to
match where its loss actually reads. That is checked by perturbation rather than
by inspection: hand the loss an arbitrary [B,T,V] tensor, bump ONE vocabulary
entry at ONE time index, and see whether the number moves. A loss that reads
position k responds; one that ignores k cannot.

Perturbing a whole vocabulary row would be a no-op -- cross-entropy is
shift-invariant in its logits -- which is a mistake an earlier version of
test_loss_alignment.py actually made, and which made every negative assertion pass
vacuously.

No GPU, no training, no real model: a stub whose forward returns fixed logits is
enough, because what is under test is the arithmetic, not the network.
"""

import pathlib
import sys

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from experimentation.experiments.curricula import make_mqar  # noqa: E402
from experimentation.training.task import (  # noqa: E402
    IGNORE_INDEX, ShardTask, SyntheticTask, TrainTask)

VOCAB, SEQ, PAIRS = 64, 32, 4


class _FixedLogits(nn.Module):
    """Returns a preset tensor, so a loss is testable without a network."""

    def __init__(self, logits, as_dict=True):
        super().__init__()
        self.logits = logits
        self.as_dict = as_dict
        self.seen = []

    def forward(self, input_ids=None, labels=None, loss_weights=None):
        self.seen.append({"labels": labels, "loss_weights": loss_weights})
        if labels is None:
            return {"logits": self.logits} if self.as_dict else self.logits
        # Mirrors KoopmanLM.forward: positional CE, weighted when asked.
        if loss_weights is None:
            loss = F.cross_entropy(self.logits.reshape(-1, self.logits.size(-1)),
                                   labels.reshape(-1), ignore_index=IGNORE_INDEX)
        else:
            ce = F.cross_entropy(self.logits.reshape(-1, self.logits.size(-1)),
                                 labels.reshape(-1), ignore_index=IGNORE_INDEX,
                                 reduction="none").view_as(labels)
            w = loss_weights.to(ce.dtype) * (labels != IGNORE_INDEX).to(ce.dtype)
            loss = (ce * w).sum() / w.sum().clamp(min=1.0)
        return {"loss": loss, "logits": self.logits}


def _aligned():
    """Synthetic-style: labels sit AT the answer position."""
    inputs, labels = make_mqar(batch=1, seq_len=SEQ, num_kv_pairs=PAIRS,
                               vocab_size=VOCAB, seed=0)
    torch.manual_seed(7)
    return inputs, labels, torch.randn(1, SEQ, VOCAB)


def _preshifted():
    """Shard-style: the dataset already offset the pair."""
    torch.manual_seed(11)
    chunk = torch.randint(0, VOCAB, (SEQ + 1,))
    return (chunk[:-1].unsqueeze(0), chunk[1:].unsqueeze(0),
            torch.randn(1, SEQ, VOCAB))


def _reads(loss_fn, logits, index):
    """Does the loss read logits at time `index`? Bump ONE vocab entry there."""
    before = loss_fn(logits).item()
    poked = logits.clone()
    poked[:, index, 0] += 25.0
    return abs(loss_fn(poked).item() - before) > 1e-6


# ------------------------------------------------------------- equivalence ----

def test_shard_step_loss_equals_the_inline_expression():
    """train.py:404-405 verbatim."""
    input_ids, labels, logits = _preshifted()
    model = _FixedLogits(logits)
    weights = torch.rand(1, SEQ)
    batch = {"input_ids": input_ids, "labels": labels, "loss_weights": weights}

    got = ShardTask().step_loss(model, batch)
    want = model(input_ids=input_ids, labels=labels, loss_weights=weights)["loss"]
    assert torch.equal(got, want)


def test_shard_step_loss_passes_loss_weights_through():
    """Not decoration. The weighted branch normalises by the WEIGHT SUM rather
    than the token count, so dropping loss_weights changes the reduction even
    when every weight is 1.0."""
    input_ids, labels, logits = _preshifted()
    model = _FixedLogits(logits)
    weights = torch.rand(1, SEQ)
    ShardTask().step_loss(model, {"input_ids": input_ids, "labels": labels,
                                  "loss_weights": weights})
    assert model.seen[-1]["loss_weights"] is weights


def test_shard_step_loss_tolerates_a_batch_without_weights():
    """MemmapPackedDataset omits them when weights.bin is absent."""
    input_ids, labels, logits = _preshifted()
    model = _FixedLogits(logits)
    got = ShardTask().step_loss(model, {"input_ids": input_ids, "labels": labels})
    assert model.seen[-1]["loss_weights"] is None
    assert torch.isfinite(got)


def test_synthetic_step_loss_equals_the_inline_expression():
    """mqar_finetune.py:296-301 verbatim."""
    input_ids, labels, logits = _aligned()
    model = _FixedLogits(logits)

    got = SyntheticTask().step_loss(model, (input_ids, labels))
    want = F.cross_entropy(logits[:, :-1].reshape(-1, VOCAB),
                           labels[:, 1:].reshape(-1), ignore_index=IGNORE_INDEX)
    assert torch.equal(got, want)


def test_synthetic_step_loss_does_not_pass_labels_to_the_model():
    """It must compute the loss ITSELF. Handing labels to a model that applies a
    positional CE is precisely the off-by-one this design exists to prevent."""
    input_ids, labels, logits = _aligned()
    model = _FixedLogits(logits)
    SyntheticTask().step_loss(model, (input_ids, labels))
    assert model.seen[-1]["labels"] is None


def test_synthetic_step_loss_accepts_a_dict_batch_too():
    input_ids, labels, logits = _aligned()
    a = SyntheticTask().step_loss(_FixedLogits(logits), (input_ids, labels))
    b = SyntheticTask().step_loss(
        _FixedLogits(logits), {"input_ids": input_ids, "labels": labels})
    assert torch.equal(a, b)


def test_synthetic_step_loss_accepts_a_bare_tensor_output():
    """The baselines may return a tensor rather than KoopmanLM's dict."""
    input_ids, labels, logits = _aligned()
    got = SyntheticTask().step_loss(_FixedLogits(logits, as_dict=False),
                                    (input_ids, labels))
    assert torch.isfinite(got)


# --------------------------------------------------------------- alignment ----

def test_the_declared_conventions_are_opposite():
    assert ShardTask.shift_in_data is True
    assert SyntheticTask.shift_in_data is False


def test_synthetic_reads_the_position_before_each_answer():
    input_ids, labels, logits = _aligned()
    task, model = SyntheticTask(), _FixedLogits(logits)

    def loss(lg):
        model.logits = lg
        return task.step_loss(model, (input_ids, labels))

    supervised = (labels[0] != IGNORE_INDEX).nonzero().flatten().tolist()
    assert supervised
    for p in supervised:
        assert _reads(loss, logits, p - 1), (
            f"synthetic loss must read logits[{p-1}] -- the prediction FOR "
            f"supervised position {p}")


def test_synthetic_ignores_the_answer_position_itself():
    """The discriminating probe. logits[p] is scored against labels[p+1]; when
    that is -100 the shifted loss cannot see it, and a positional loss can."""
    input_ids, labels, logits = _aligned()
    task, model = SyntheticTask(), _FixedLogits(logits)

    def loss(lg):
        model.logits = lg
        return task.step_loss(model, (input_ids, labels))

    supervised = set((labels[0] != IGNORE_INDEX).nonzero().flatten().tolist())
    candidates = [p for p in supervised if (p + 1) not in supervised and p + 1 < SEQ]
    assert candidates
    for p in candidates:
        assert not _reads(loss, logits, p), (
            f"synthetic loss must NOT read logits[{p}]")


def test_shard_reads_the_supervised_position_itself():
    """The same probe, opposite answer, because the data arrived pre-offset."""
    input_ids, labels, logits = _preshifted()
    task, model = ShardTask(), _FixedLogits(logits)

    def loss(lg):
        model.logits = lg
        return task.step_loss(model, {"input_ids": input_ids, "labels": labels})

    for p in (0, SEQ // 2, SEQ - 1):
        assert _reads(loss, logits, p), f"shard loss must read logits[{p}]"


def test_routing_aligned_data_through_the_shard_task_is_catastrophic():
    """Why the loss is not unified. A one-hot input-copier -- a model that learnt
    nothing but identity -- scores near-zero under the shard task on aligned data
    and badly under the synthetic one."""
    input_ids, labels, _ = _aligned()
    copier = _FixedLogits(F.one_hot(input_ids, num_classes=VOCAB).float() * 30.0)

    wrong = ShardTask().step_loss(
        copier, {"input_ids": input_ids, "labels": labels}).item()
    right = SyntheticTask().step_loss(copier, (input_ids, labels)).item()

    assert wrong < 1e-4, f"expected the copier to win under the wrong loss, got {wrong}"
    assert right > 1.0, f"expected the copier to lose under the right loss, got {right}"


# ------------------------------------------------------------ optional hooks ----

def test_the_optional_hooks_default_to_nothing():
    """So a task that only supplies data and loss is complete."""
    class Minimal(TrainTask):
        pass

    assert Minimal().in_loop_eval(object(), 10) is None
    assert Minimal().on_final(object(), "/tmp", object()) is None


def test_in_loop_eval_respects_its_cadence():
    calls = []
    task = SyntheticTask(eval_fn=lambda m, s: calls.append(s) or {"acc": 1.0},
                         eval_every=100)
    assert task.in_loop_eval(object(), 50) is None
    assert task.in_loop_eval(object(), 100) == {"acc": 1.0}
    assert calls == [100]


def test_in_loop_eval_is_off_without_a_function():
    assert SyntheticTask(eval_every=10).in_loop_eval(object(), 10) is None


def test_an_unknown_model_type_is_rejected_by_name():
    with pytest.raises(ValueError, match="mamba_only"):
        SyntheticTask(model_type="nope").build_model(object())


# ------------------------------------------------------------- Table2Task ----
#
# The third loop, and the one the design doc (§6.3) calls the real migration
# cost. `table2.py` has NO dataset: `make_train_batch(step, args)` is a function
# of the STEP COUNTER returning a whole batch. Wrapping that as a Dataset has two
# traps, and the second is the dangerous one.


class _Table2Args:
    """The subset of table2's argparse namespace make_train_batch reads."""
    def __init__(self, curriculum="batch_mixed", batch_size=8, seed=42):
        self.curriculum = curriculum
        self.batch_size = batch_size
        self.seed = seed
        self.task_vocab_size = 64
        self.toolcall_keys = 4
        self.toolcall_queries = 2
        self.overwrite_prob = 0.3
        self.sysprompt_vars = 4
        self.sysprompt_decoys = 2


def test_table2_step_loss_equals_the_inline_expression():
    """The seam guarantee, same as the other two loops. table2's inline loss is
    character-identical to SyntheticTask's, including ignore_index=-100 spelled
    as a literal there and as IGNORE_INDEX here."""
    from experimentation.training.task import Table2Task

    torch.manual_seed(0)
    logits = torch.randn(2, 6, 11)
    labels = torch.randint(0, 11, (2, 6))
    model = _FixedLogits(logits)

    expected = F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.size(-1)),
        labels[:, 1:].reshape(-1), ignore_index=-100)
    got = Table2Task().step_loss(model, (torch.zeros_like(labels), labels))
    assert torch.equal(got, expected)


def test_table2_declares_the_synthetic_shift_convention():
    """It shifts in the LOSS, like MQAR and unlike the shard path. Getting this
    backwards makes every token trivially predictable -- see
    test_routing_aligned_data_through_the_shard_task_is_catastrophic."""
    from experimentation.training.task import Table2Task

    assert Table2Task().shift_in_data is False


def test_table2_declares_fp16_because_only_it_runs_a_scaler():
    """The one loop running fp16 with a GradScaler. Nothing else exercises the
    loop's scaler path, so a unified loop that silently ran this in bf16 would
    change training while still descending."""
    from experimentation.training.task import Table2Task

    assert Table2Task().compute_precision == "fp16"


# ------------------------------------------------------- the step indexing ----

def test_the_dataset_index_is_the_step_number_not_a_zero_offset():
    """THE trap. table2's loop is `range(start_step + 1, max_steps + 1)`, so the
    first step is 1, while a Dataset is indexed from 0. Off by one is not a
    cosmetic error here: `--curriculum mixed` alternates whole batches on
    `step % 2`, so shifting the index by one INVERTS the entire curriculum while
    still producing a descending loss.
    """
    from experimentation.experiments.table2 import make_train_batch
    from experimentation.training.task import Table2Task

    args = _Table2Args(curriculum="mixed")
    ds = Table2Task().dataset(None, args)

    for step in (1, 2, 3, 17):
        want_x, want_y = make_train_batch(step, args)
        got_x, got_y = ds[step]
        assert torch.equal(got_x, want_x), f"step {step}: inputs differ"
        assert torch.equal(got_y, want_y), f"step {step}: labels differ"


def test_alternating_curriculum_parity_survives_the_wrapper():
    """The observable consequence of the trap above, asserted directly: under
    `mixed`, even steps are toolcall and odd are sysprompt. If the wrapper were
    zero-offset, every batch would come from the other task."""
    from experimentation.experiments.table2 import (_make_sysprompt_batch,
                                                    _make_toolcall_batch)
    from experimentation.training.task import Table2Task

    args = _Table2Args(curriculum="mixed")
    ds = Table2Task().dataset(None, args)

    even_x, _ = _make_toolcall_batch(args.batch_size, 2, args)
    odd_x, _ = _make_sysprompt_batch(args.batch_size, 3, args)
    assert torch.equal(ds[2][0], even_x), "even step must be the toolcall task"
    assert torch.equal(ds[3][0], odd_x), "odd step must be the sysprompt task"


def test_one_index_yields_one_whole_batch_not_one_sample():
    """The collate wrinkle. `Dataset.__getitem__` conventionally returns ONE
    SAMPLE for DataLoader to stack, but make_train_batch returns a FULL BATCH.
    A naive wrapper handed to DataLoader(batch_size=B) yields [B, B, T] AND
    silently consumes B steps of curriculum per iteration -- the shape is
    catchable, the curriculum burn is only visible in a loss curve.

    So the wrapper declares batch_size=None, and this pins the shape it relies
    on: index -> a batch already.
    """
    from experimentation.training.task import Table2Task

    args = _Table2Args(batch_size=8)
    x, y = Table2Task().dataset(None, args)[1]
    assert x.shape[0] == args.batch_size, (
        f"index gave leading dim {x.shape[0]}, expected the batch size "
        f"{args.batch_size}; DataLoader must be given batch_size=None")
    assert y.shape[0] == args.batch_size


def test_the_wrapper_says_batch_size_none_out_loud():
    """Not a style assertion: DataLoader's default batch_size is 1, so wrapping
    this without saying so produces [1, B, T] and trains on a wrong shape rather
    than raising. The requirement has to travel with the dataset."""
    from experimentation.training.task import Table2Task

    ds = Table2Task().dataset(None, _Table2Args())
    assert getattr(ds, "dataloader_batch_size", "missing") is None, (
        "the dataset must advertise batch_size=None for a DataLoader")


def test_its_length_covers_the_requested_steps():
    """Deterministic in `step`, so length is the step budget -- and because it is
    keyed on step rather than on an epoch position, resume is exact: step 400
    after a restart draws what step 400 would have drawn."""
    from experimentation.training.task import Table2Task

    ds = Table2Task(max_steps=400).dataset(None, _Table2Args())
    assert len(ds) >= 400
    a = ds[400]
    b = Table2Task(max_steps=400).dataset(None, _Table2Args())[400]
    assert torch.equal(a[0], b[0]), "step 400 must be reproducible across builds"


def test_the_synthetic_loss_expression_is_callable_on_logits_directly():
    """table2 reuses its TRAINING forward's logits for the per-task eval split
    ("slice the already-computed logits, no extra forward pass"), so it cannot
    call `step_loss(model, batch)` -- that does its own forward and would double
    the cost of every step for an identical number.

    The design doc did not anticipate this. The resolution is that what must be
    shared is the EXPRESSION, not the entry point: `synthetic_loss` is the single
    definition, `step_loss` calls it after a forward, and table2 calls it three
    times on logits it already has. Before this, table2.py spelled that
    expression out three times.
    """
    from experimentation.training.task import SyntheticTask, synthetic_loss

    torch.manual_seed(0)
    logits = torch.randn(4, 7, 13)
    labels = torch.randint(0, 13, (4, 7))

    direct = synthetic_loss(logits, labels)
    via_model = SyntheticTask().step_loss(_FixedLogits(logits),
                                          (torch.zeros_like(labels), labels))
    assert torch.equal(direct, via_model), \
        "the two paths must be the SAME expression, not two equivalent ones"


def test_the_shared_expression_honours_ignore_index():
    """It is the property the alignment suite depends on, and it is one keyword
    away from being dropped in an extraction."""
    from experimentation.training.task import IGNORE_INDEX, synthetic_loss

    torch.manual_seed(1)
    logits = torch.randn(1, 5, 9)
    labels = torch.randint(0, 9, (1, 5))
    masked = labels.clone()
    masked[0, 3] = IGNORE_INDEX

    assert not torch.equal(synthetic_loss(logits, labels),
                           synthetic_loss(logits, masked)), \
        "masking a supervised position must change the loss"


def test_slicing_logits_by_half_matches_a_separate_call():
    """table2's batch_mixed eval splits one batch into toolcall/sysprompt halves
    and costs each separately off the shared logits. Pinning that the halves are
    read the way the whole is."""
    from experimentation.training.task import synthetic_loss

    torch.manual_seed(2)
    logits = torch.randn(8, 6, 11)
    labels = torch.randint(0, 11, (8, 6))
    half = 4

    assert torch.equal(synthetic_loss(logits[:half], labels[:half]),
                       synthetic_loss(logits[:half].clone(), labels[:half].clone()))

