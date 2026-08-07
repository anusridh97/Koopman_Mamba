"""Pure-Python tests for contrastive extraction (koopman_lm/retrieval/data.py).

Torch-free: runs anywhere. Pins the (query, positive, hard-negative) extraction
each retrieval source depends on, and the weighted source sampler.
"""
import random

import pytest

from koopman_lm.retrieval.data import (
    hotpot_to_pair, musique_to_pair, wiki_to_pairs, extract_pairs, sample_source,
)

pytestmark = pytest.mark.correctness


def test_hotpot_positive_and_negatives():
    ex = {"question": "Who wrote X?",
          "context": {"title": ["A", "B", "C"],
                      "sentences": [["A1.", "A2."], ["B1."], ["C1."]]},
          "supporting_facts": {"title": ["B"], "sent_id": [0]}}
    p = hotpot_to_pair(ex)
    assert p.positive == "B1."
    assert set(p.hard_negatives) == {"A1. A2.", "C1."}
    assert p.positive not in p.hard_negatives


def test_hotpot_no_support_returns_none():
    ex = {"question": "q", "context": {"title": ["A"], "sentences": [["a"]]},
          "supporting_facts": {"title": ["Z"], "sent_id": [0]}}
    assert hotpot_to_pair(ex) is None


def test_musique_positive_is_supporting():
    ex = {"question": "Q?", "paragraphs": [
        {"title": "t1", "paragraph_text": "gold", "is_supporting": True},
        {"title": "t2", "paragraph_text": "d1", "is_supporting": False},
        {"title": "t3", "paragraph_text": "d2", "is_supporting": False}]}
    p = musique_to_pair(ex)
    assert p.positive == "gold" and p.hard_negatives == ["d1", "d2"]


def test_wiki_pairs_have_distinct_neg():
    ex = {"title": "Photosynthesis",
          "text": ("Intro paragraph that is comfortably long enough to survive the stub filter here.\n\n"
                   "== History ==\n\nHistory paragraph that is also comfortably long enough to pass the filter.\n\n"
                   "== Process ==\n\nProcess paragraph that is likewise comfortably long enough to be usable.")}
    pairs = wiki_to_pairs(ex, random.Random(0), max_pairs=3)
    assert pairs, "expected at least one self-supervised wiki pair"
    for p in pairs:
        assert p.query and p.positive and p.hard_negatives
        assert p.positive not in p.hard_negatives


def test_wiki_too_short_returns_empty():
    assert wiki_to_pairs({"title": "T", "text": "one short line"}, random.Random(0)) == []


def test_extract_pairs_dispatch():
    assert extract_pairs("hotpotqa",
                         {"question": "q",
                          "context": {"title": ["A"], "sentences": [["a."]]},
                          "supporting_facts": {"title": ["A"], "sent_id": [0]}},
                         random.Random(0))[0].positive == "a."
    with pytest.raises(ValueError):
        extract_pairs("unknown", {}, random.Random(0))


def test_sample_source_matches_fractions():
    rng = random.Random(2)
    n = 20000
    c = {"a": 0, "b": 0}
    for _ in range(n):
        c[sample_source({"a": 0.7, "b": 0.3}, rng)] += 1
    assert abs(c["a"] / n - 0.7) < 0.02
