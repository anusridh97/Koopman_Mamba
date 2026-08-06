"""MQAR generator correctness (CPU, pure).

The metric *numbers* need a GPU/model; the generator's shape/label
invariants are pure and tested here.
"""
import pytest

from koopman_lm.evaluation.mqar import make_mqar

pytestmark = pytest.mark.correctness


def test_make_mqar_shapes_and_labels():
    B, T, P, V = 4, 64, 4, 128
    half = V // 2
    inputs, labels = make_mqar(B, T, P, V, seed=0)
    assert inputs.shape == (B, T) and labels.shape == (B, T)
    assert inputs.min() >= 0 and inputs.max() < V
    for b in range(B):
        sup = (labels[b] != -100).nonzero().flatten().tolist()
        assert len(sup) == P                      # one answer per query
        front = inputs[b, : 2 * P]                # the key-value pair region
        for pos in sup:
            value = labels[b, pos].item()
            key = inputs[b, pos - 1].item()
            assert value >= half                  # values live in the top half
            assert key < half                     # keys in the bottom half
            # the (key, value) pair was presented earlier in the front pairs
            seen = {(front[2 * i].item(), front[2 * i + 1].item()) for i in range(P)}
            assert (key, value) in seen


def test_make_mqar_rejects_too_short():
    with pytest.raises(AssertionError):
        make_mqar(2, 8, 16, 128)                  # 16 pairs can't fit in len 8


def test_make_mqar_kv1_is_niah():
    # Table 2's "needle-in-a-haystack (KV=1)" is this generator special-cased
    # to a single key-value pair -- verify the KV=1 cell works at the lengths
    # table2.py trains/evaluates at (64..4096).
    for T in (64, 128, 4096):
        inputs, labels = make_mqar(4, T, num_kv_pairs=1, vocab_size=128, seed=0)
        assert inputs.shape == (4, T)
        for b in range(4):
            sup = (labels[b] != -100).nonzero().flatten().tolist()
            assert len(sup) == 1                  # exactly one needle/answer
