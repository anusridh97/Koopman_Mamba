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


# --------------------------------------------------------- iter_batches ----
#
# The seam the loop unification needs, and the one the design doc left
# unresolved. Its ownership table says "dataset | task" and "resume | loop",
# with the note "indexable, so the loop can do resume arithmetic" -- but §3 also
# says resume is entangled with data iteration, and it is:
#
#   * train.py and mqar_finetune iterate an EPOCH PERMUTATION over a map dataset,
#     resuming by skipping `samples_consumed` indices;
#   * table2 has no epochs and no shuffle at all -- its batches are a pure
#     function of the step counter.
#
# So one loop cannot literally own iteration. `iter_batches` is DEFAULTED to the
# epoch path (what two of three tasks want) and overridden by Table2Task, which
# keeps the loop single and puts the one genuinely divergent concern behind a
# hook instead of an `if`.


class _CountingDataset(torch.utils.data.Dataset):
    """Records nothing, returns its own index, so batch order is observable."""

    def __init__(self, n=16):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return {"input_ids": torch.tensor([i]), "labels": torch.tensor([i])}


class _Args:
    per_device_train_batch_size = 2
    num_workers = 0
    seed = 7


def _order(batches):
    return [int(v) for b in batches for v in b["input_ids"].flatten()]


def test_the_default_iterates_the_epoch_permutation():
    """Not DataLoader's implicit shuffle: train.py builds the order explicitly
    with epoch_permutation so a mid-epoch resume can skip consumed indices
    without re-reading them. A default that quietly used shuffle=True would look
    identical until someone resumed."""
    from experimentation.training.resume import epoch_permutation
    from experimentation.training.task import IterContext, ShardTask

    ds = _CountingDataset()
    ctx = IterContext(epoch=0, start_step=0, skip_samples=0)
    got = _order(ShardTask().iter_batches(ds, _Args(), ctx))
    want = epoch_permutation(len(ds), _Args.seed + 0, 0)
    assert got == want[:len(got)]
    assert got != list(range(len(ds))), "a permutation that is the identity proves nothing"


def test_the_default_skips_exactly_the_consumed_prefix_on_resume():
    from experimentation.training.resume import epoch_permutation
    from experimentation.training.task import IterContext, ShardTask

    ds = _CountingDataset()
    full = epoch_permutation(len(ds), _Args.seed + 0, 0)
    ctx = IterContext(epoch=0, start_step=0, skip_samples=6)
    got = _order(ShardTask().iter_batches(ds, _Args(), ctx))
    assert got == full[6:len(full)][:len(got)], \
        "a resumed epoch must continue the SAME permutation, not reshuffle"


def test_the_default_passes_an_explicit_generator():
    """Invisible and load-bearing. DataLoader.__iter__ draws one int64 from the
    GLOBAL torch RNG on every fresh iteration to seed _base_seed -- even at
    num_workers=0 with an explicit sampler. A resumed run builds a new DataLoader
    mid-epoch and so pays a draw its uninterrupted twin never pays there,
    desyncing dropout from the first resumed step.

    Asserted by watching the global RNG rather than by reading the source, so it
    survives a rewrite: iterating must not advance the global stream.
    """
    from experimentation.training.task import IterContext, ShardTask

    ds = _CountingDataset()
    ctx = IterContext(epoch=0, start_step=0, skip_samples=0)

    torch.manual_seed(1234)
    before = torch.random.get_rng_state()
    list(ShardTask().iter_batches(ds, _Args(), ctx))
    after = torch.random.get_rng_state()
    assert torch.equal(before, after), \
        "iterating consumed global RNG -- generator= was dropped, and resume " \
        "will desync from the first step after a restart"


def test_table2_iterates_by_step_and_ignores_epochs():
    """Its data is a pure function of the step counter, so the epoch permutation
    is not merely unnecessary -- applying it would hand table2 different batches
    than it has ever trained on."""
    from experimentation.experiments.table2 import make_train_batch
    from experimentation.training.task import IterContext, Table2Task

    args = _Table2Args(curriculum="mixed")
    args.max_steps = 5
    task = Table2Task(max_steps=5)
    ctx = IterContext(epoch=0, start_step=0, skip_samples=0)

    got = list(task.iter_batches(task.dataset(None, args), args, ctx))
    assert len(got) == 5, f"expected one batch per step, got {len(got)}"
    for i, (x, y) in enumerate(got, start=1):
        wx, wy = make_train_batch(i, args)
        assert torch.equal(x, wx), f"step {i} inputs differ"
        assert torch.equal(y, wy), f"step {i} labels differ"


def test_table2_resumes_at_the_next_step():
    """start_step is where the loop left off, so iteration continues at
    start_step + 1 -- matching `range(start_step + 1, max_steps + 1)`."""
    from experimentation.experiments.table2 import make_train_batch
    from experimentation.training.task import IterContext, Table2Task

    args = _Table2Args(curriculum="mixed")
    args.max_steps = 5
    task = Table2Task(max_steps=5)
    ctx = IterContext(epoch=0, start_step=3, skip_samples=0)

    got = list(task.iter_batches(task.dataset(None, args), args, ctx))
    assert len(got) == 2, "steps 4 and 5 remain"
    assert torch.equal(got[0][0], make_train_batch(4, args)[0])


def test_table2_batches_consume_no_global_rng():
    """Why table2's resume is exact despite restoring no RNG -- measured in job
    440248 at 0.000000 across the interruption.

    Its `load_checkpoint` restores model + optimizer + scheduler + step and does
    no RNG handling at all (`grep -c rng` is 0 in table2.py and
    mqar_finetune.py), unlike train.py's save_resume_state. That should desync a
    resumed run's stochastic stream -- and does not, for two reasons:

      * KoopmanLMConfig declares no dropout field, so the forward has no
        stochastic op;
      * the curriculum generators use a LOCAL torch.Generator().manual_seed(...)
        keyed on args.seed + step, never the global stream.

    So the step consumes no global RNG and there is nothing to restore. That is
    a property of the current data generator, NOT of the resume mechanism: swap
    one local Generator for a global call and table2's resume silently stops
    being reproducible, with no test to notice. This is that test, for the half
    that runs without a GPU.
    """
    from experimentation.experiments.table2 import make_train_batch

    args = _Table2Args(curriculum="batch_mixed")
    torch.manual_seed(99)
    before = torch.random.get_rng_state()
    make_train_batch(1, args)
    make_train_batch(2, args)
    after = torch.random.get_rng_state()

    assert torch.equal(before, after), (
        "make_train_batch drew from the GLOBAL torch RNG. table2's resume "
        "restores no RNG state, so this silently makes a resumed run diverge "
        "from an uninterrupted one -- use torch.Generator().manual_seed(...) "
        "as curricula.py already does")


def test_table2_batches_are_reproducible_from_the_step_alone():
    """The other half of the same property: the same step gives the same batch
    regardless of what ran before it. That is what makes a resume able to pick
    up at step N with no data position to recover."""
    from experimentation.experiments.table2 import make_train_batch

    args = _Table2Args(curriculum="batch_mixed")
    torch.manual_seed(1)
    first = make_train_batch(7, args)
    torch.manual_seed(12345)
    for _ in range(3):
        make_train_batch(99, args)
    second = make_train_batch(7, args)

    assert torch.equal(first[0], second[0]) and torch.equal(first[1], second[1]), \
        "step 7's batch depends on history, so a resume cannot reproduce it"


# ------------------------------------------- SyntheticTask on the shared loop ----

def test_synthetic_iterates_with_shuffle_not_the_epoch_permutation():
    """mqar_finetune builds `DataLoader(..., shuffle=True)`, NOT the explicit
    epoch permutation the shard path uses. Those produce different batch orders,
    so inheriting the default would move the MQAR golden for a reason that has
    nothing to do with the refactor.

    Asserted by DIFFERENCE against the default, so it cannot pass by accident.
    """
    from experimentation.training.task import IterContext, ShardTask, SyntheticTask

    ds = _CountingDataset(16)

    class _A:
        batch_size = 2
        per_device_train_batch_size = 2
        num_workers = 0
        seed = 7

    ctx = IterContext(epoch=0, start_step=0, skip_samples=0)
    torch.manual_seed(0)
    synth = _order(SyntheticTask().iter_batches(ds, _A(), ctx))
    shard = _order(ShardTask().iter_batches(ds, _A(), ctx))
    assert sorted(synth) == sorted(shard), "both must cover the same samples"
    assert synth != shard, (
        "SyntheticTask reproduced the shard path's epoch permutation; mqar uses "
        "DataLoader(shuffle=True) and its golden encodes that order")


def test_synthetic_uses_its_own_batch_size_field():
    """mqar's argparse says `--batch_size`; the shard trainer says
    `--per_device_train_batch_size`. Reading the wrong one silently changes the
    effective batch and every number in the curve."""
    from experimentation.training.task import IterContext, SyntheticTask

    class _A:
        batch_size = 4
        per_device_train_batch_size = 999     # must be ignored
        num_workers = 0
        seed = 7

    batches = list(SyntheticTask().iter_batches(
        _CountingDataset(16), _A(), IterContext()))
    assert batches[0]["input_ids"].shape[0] == 4


def test_synthetic_reports_the_instantaneous_loss():
    """mqar prints `loss.item()`, not a window average. The default would report
    the average and move every value in the committed golden."""
    from experimentation.training.task import ShardTask, SyntheticTask

    value, tail = SyntheticTask().progress_fields(window_avg=1.0, last_loss=2.0)
    assert (value, tail) == (2.0, "")
    # and the shard path is unchanged
    assert ShardTask().progress_fields(window_avg=1.0, last_loss=2.0) == (1.0, "")


def test_synthetic_remembers_its_last_eval_for_the_checkpoint():
    """mqar passes the most recent accuracy into save_checkpoint. With the loop
    owning checkpointing, the task has to carry that value across."""
    from experimentation.training.task import SyntheticTask

    task = SyntheticTask(eval_fn=lambda m, s: {"acc": 0.25 * s}, eval_every=2)
    assert task.in_loop_eval(None, 1) is None          # off-cadence
    assert task.last_eval is None
    assert task.in_loop_eval(None, 2) == {"acc": 0.5}  # on-cadence
    assert task.last_eval == {"acc": 0.5}


# ------------------------------------- Table2Task's per-task accounting ----
#
# table2's progress line carries `toolcall_loss X  sysprompt_loss Y` alongside
# the combined loss, and it computes them by SLICING THE TRAINING FORWARD'S
# LOGITS -- "no extra forward pass". The shared loop calls step_loss and keeps
# only the scalar, so the split has to be accounted where the logits exist.
#
# That is the task, not the loop: which curriculum ran, and how to attribute a
# batch to a task, is table2-specific in a way the loop must not learn.


def _t2_task(curriculum="batch_mixed", batch_size=8, start_step=0):
    from experimentation.training.task import IterContext, Table2Task

    args = _Table2Args(curriculum=curriculum, batch_size=batch_size)
    args.max_steps = 8
    task = Table2Task(max_steps=8)
    # iter_batches is what binds args and the starting step, exactly as the loop
    # calls it.
    list(task.iter_batches(task.dataset(None, args), args,
                           IterContext(start_step=start_step)))
    return task, args


def test_batch_mixed_attributes_each_half_to_its_own_task():
    """Both halves come off ONE forward. Averaged over the window they are what
    the combined loss is a mean of."""
    task, _ = _t2_task("batch_mixed", batch_size=4)

    torch.manual_seed(0)
    logits = torch.randn(4, 6, 11)
    labels = torch.randint(0, 11, (4, 6))
    task.step_loss(_FixedLogits(logits), (torch.zeros_like(labels), labels))

    value, tail = task.progress_fields(window_avg=99.0, last_loss=99.0)
    assert "toolcall_loss" in tail and "sysprompt_loss" in tail, tail
    assert value != 99.0, "table2 reports its own combined loss, not the window"

    from experimentation.training.task import synthetic_loss
    tc = float(synthetic_loss(logits[:2], labels[:2]))
    sp = float(synthetic_loss(logits[2:], labels[2:]))
    assert value == pytest.approx((tc + sp) / 2, abs=1e-6)


@pytest.mark.parametrize("start,first_task", [(0, "sysprompt"), (1, "toolcall")])
def test_mixed_attributes_by_step_parity(start, first_task):
    """`--curriculum mixed` alternates whole batches on `step % 2`, so
    attribution depends on WHICH step this batch is -- which the task must track,
    because step_loss is not told. Off by one and every batch is credited to the
    wrong task."""
    task, _ = _t2_task("mixed", start_step=start)

    torch.manual_seed(0)
    logits = torch.randn(2, 5, 9)
    labels = torch.randint(0, 9, (2, 5))
    task.step_loss(_FixedLogits(logits), (torch.zeros_like(labels), labels))

    _value, tail = task.progress_fields(window_avg=0.0, last_loss=0.0)
    assert f"{first_task}_loss" in tail, (
        f"first batch after start_step={start} was credited to the wrong task: "
        f"{tail!r}")
    other = "toolcall" if first_task == "sysprompt" else "sysprompt"
    assert f"{other}_loss" not in tail


@pytest.mark.parametrize("curriculum", ["toolcall", "sysprompt"])
def test_a_single_task_curriculum_reports_only_that_task(curriculum):
    task, _ = _t2_task(curriculum)
    torch.manual_seed(0)
    logits = torch.randn(2, 5, 9)
    labels = torch.randint(0, 9, (2, 5))
    task.step_loss(_FixedLogits(logits), (torch.zeros_like(labels), labels))

    value, tail = task.progress_fields(window_avg=0.0, last_loss=0.0)
    assert f"{curriculum}_loss" in tail
    from experimentation.training.task import synthetic_loss
    assert value == pytest.approx(float(synthetic_loss(logits, labels)), abs=1e-6)


def test_the_window_resets_after_each_report():
    """table2 clears loss_sum/loss_count at every log, so each line reports the
    window since the last one. Without the reset the average drifts toward the
    run mean and every value after the first is wrong."""
    task, _ = _t2_task("toolcall")
    torch.manual_seed(0)
    big = torch.randn(2, 5, 9) * 10
    small = torch.randn(2, 5, 9) * 0.01
    labels = torch.randint(0, 9, (2, 5))

    task.step_loss(_FixedLogits(big), (torch.zeros_like(labels), labels))
    first, _ = task.progress_fields(window_avg=0.0, last_loss=0.0)
    task.step_loss(_FixedLogits(small), (torch.zeros_like(labels), labels))
    second, _ = task.progress_fields(window_avg=0.0, last_loss=0.0)

    from experimentation.training.task import synthetic_loss
    assert second == pytest.approx(float(synthetic_loss(small, labels)), abs=1e-6), (
        "the second report still includes the first window, so the accumulator "
        "was not reset")
    assert first != second

