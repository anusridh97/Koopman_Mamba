"""Where does a loss function READ its logits? Answered mechanically.

Causal LM training must offset inputs from targets by one. This repo does that
offset in two different places:

  shard path      the DATASET offsets. training/data/dataset.py:57 emits
                  input_ids=chunk[:-1], labels=chunk[1:], so KoopmanLM.forward
                  scores logits[i] against labels[i] -- positional, no shift.
  synthetic path  the LOSS offsets. curricula.make_mqar emits ALIGNED pairs
                  (labels[p] is the answer belonging at position p), and
                  mqar_finetune.py:299 / table2.py:287 score logits[:, :-1]
                  against labels[:, 1:].

Both are correct next-token prediction. They disagree only about whose job the
offset is -- and from the outside they are indistinguishable: same function, same
ignore_index, both produce a descending loss. Which makes "these two loss
computations are duplicates, route everything through the model's built-in loss"
look like obvious deduplication while being an off-by-one.

Measured: feeding ALIGNED mqar data to the positional loss scores the model on
reproducing a token it was just handed, because at every supervised position the
aligned label IS the input at that position. Trivially learnable, loss finite and
falling, MQAR accuracy would look excellent, and the recall ability the benchmark
exists to measure would never be tested.

So: pin the alignment by perturbation. No model, no GPU, no training. Logits are
just a [B, T, V] tensor, so hand the loss one, change a single time index, and see
whether the number moves. A loss that reads position k responds to logits[k]; one
that ignores position k cannot. That gives a positional dependency map, which is
exactly the property the two conventions differ on.

When §6.2's TrainTask lands, every task's loss should get this treatment -- it is
the guard that stops the next author from unifying two conventions by accident.
"""

import pathlib
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from experimentation.experiments.curricula import make_mqar  # noqa: E402

VOCAB = 64
SEQ = 32
PAIRS = 4


# ---------------------------------------------------------------- the two styles

def shifted_loss(logits, labels):
    """The synthetic path: mqar_finetune.py:299, table2.py:287, eval_mqar."""
    return F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)),
                           labels[:, 1:].reshape(-1), ignore_index=-100)


def positional_loss(logits, labels):
    """The shard path: KoopmanLM.forward, on data the dataset already offset."""
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                           labels.reshape(-1), ignore_index=-100)


# ---------------------------------------------------------------------- helpers

def _fixture():
    """Aligned MQAR data plus arbitrary-but-fixed logits."""
    inputs, labels = make_mqar(batch=1, seq_len=SEQ, num_kv_pairs=PAIRS,
                               vocab_size=VOCAB, seed=0)
    torch.manual_seed(1234)
    logits = torch.randn(1, SEQ, VOCAB)
    return inputs, labels, logits


def _reads(loss_fn, logits, labels, index):
    """Does `loss_fn` read logits at time `index`? Perturb only there.

    Bumps ONE vocabulary entry, not the whole row. Cross-entropy is
    shift-invariant in its logits -- adding a constant to every entry at a
    position leaves the softmax identical -- so a uniform bump is a no-op and
    would make every probe report "does not read", passing all the negative
    assertions vacuously. The first draft of this helper did exactly that.
    """
    before = loss_fn(logits, labels).item()
    poked = logits.clone()
    poked[:, index, 0] += 25.0
    after = loss_fn(poked, labels).item()
    return abs(after - before) > 1e-6


def _supervised(labels):
    return (labels[0] != -100).nonzero().flatten().tolist()


# ------------------------------------------------------------------------ tests

def test_the_fixture_is_aligned_not_preshifted():
    """The premise. make_mqar's labels sit AT the answer position, so at each
    supervised position the label equals the input there -- which is what makes
    the positional loss trivially satisfiable on this data."""
    inputs, labels, _ = _fixture()
    sup = _supervised(labels)
    assert sup, "no supervised positions"
    for p in sup:
        assert labels[0, p] == inputs[0, p], (
            f"position {p}: aligned data should have label == input")


def test_shifted_loss_reads_the_position_before_each_answer():
    inputs, labels, logits = _fixture()
    for p in _supervised(labels):
        assert _reads(shifted_loss, logits, labels, p - 1), (
            f"shifted loss must read logits[{p - 1}] -- the prediction FOR "
            f"supervised position {p}")


def test_shifted_loss_ignores_the_answer_position_itself():
    """The discriminating case. logits[p] is scored against labels[p+1]; when
    that is -100 the position is ignored entirely, so a shifted loss cannot see
    it. A positional loss can. One perturbation separates the conventions."""
    inputs, labels, logits = _fixture()
    sup = set(_supervised(labels))
    candidates = [p for p in sup if (p + 1) not in sup and p + 1 < SEQ]
    assert candidates, "need a supervised position whose successor is unsupervised"
    for p in candidates:
        assert not _reads(shifted_loss, logits, labels, p), (
            f"shifted loss must NOT read logits[{p}]: it is scored against "
            f"labels[{p + 1}], which is ignored")


def test_positional_loss_reads_the_answer_position_itself():
    """The same perturbation, opposite answer. This is the whole test suite's
    point: one probe, two conventions, different results."""
    inputs, labels, logits = _fixture()
    for p in _supervised(labels):
        assert _reads(positional_loss, logits, labels, p), (
            f"positional loss must read logits[{p}]")


def test_positional_loss_ignores_unsupervised_positions():
    inputs, labels, logits = _fixture()
    sup = set(_supervised(labels))
    unsup = [i for i in range(SEQ) if i not in sup]
    assert unsup
    for i in unsup[:6]:
        assert not _reads(positional_loss, logits, labels, i), (
            f"positional loss must NOT read logits[{i}] -- labels[{i}] is -100")


def test_the_two_conventions_disagree_on_this_data():
    """If this ever fails, the probe has stopped discriminating and every test
    above is passing for the wrong reason."""
    inputs, labels, logits = _fixture()
    a = shifted_loss(logits, labels).item()
    b = positional_loss(logits, labels).item()
    assert abs(a - b) > 1e-4, (
        f"the conventions must be distinguishable on this fixture: {a} vs {b}")


def test_positional_loss_is_trivially_satisfiable_on_aligned_data():
    """Why the off-by-one is dangerous rather than merely wrong: an "oracle" that
    just copies its input scores ~0 under the positional loss on aligned data. A
    model can reach that by learning identity, and nothing in the number reveals
    it never learned recall."""
    inputs, labels, _ = _fixture()
    copier = F.one_hot(inputs, num_classes=VOCAB).float() * 30.0
    assert positional_loss(copier, labels).item() < 1e-4, "copying should win"
    # Under the correct convention, copying is no help at all.
    assert shifted_loss(copier, labels).item() > 1.0, (
        "under the shifted convention an input-copier must NOT score well")


@pytest.mark.parametrize("loss_fn,name", [(shifted_loss, "shifted"),
                                          (positional_loss, "positional")])
def test_perturbing_nothing_changes_nothing(loss_fn, name):
    """Control: the probe reports movement only when logits actually move."""
    inputs, labels, logits = _fixture()
    a = loss_fn(logits, labels).item()
    b = loss_fn(logits.clone(), labels).item()
    assert a == b, f"{name} loss is not deterministic on identical input"
