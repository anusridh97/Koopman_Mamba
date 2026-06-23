"""Eval-harness plumbing + new task generators + SKA-zeroing (CPU).

The metric *numbers* need a GPU/data, but the scale detection, eval-plan
selection, JSON assembly, MQAR/RULER generators, and the SKA-zeroing ablation
are pure and tested here.
"""
import contextlib

import pytest
import torch

from koopman_lm.globals.config import build_config, config_hash
from koopman_lm.models.koopman_lm import SKABlock, KoopmanLM
from koopman_lm.evaluation.mqar.mqar import make_mqar
from koopman_lm.evaluation.ruler import (
    build_multikey_niah,
    build_variable_tracking,
    build_common_word_extraction,
)
import koopman_lm.evaluation.harness as H

pytestmark = pytest.mark.correctness


# ---- MQAR generator ----

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


# ---- RULER builders ----

@pytest.mark.parametrize("builder", [
    build_multikey_niah, build_variable_tracking, build_common_word_extraction,
])
def test_ruler_builders_embed_answer(builder):
    prompt, answer = builder(target_words=80, seed=3)
    assert isinstance(prompt, str) and isinstance(answer, str)
    assert answer in prompt                       # answer is recoverable from context
    assert len(prompt.split()) >= 70              # padded toward the budget


# ---- harness pure helpers ----

def test_detect_scale_matches_factory():
    cfg = build_config("440m")
    meta = {"cfg": cfg, "model_type": "koopman", "model_size": "440m",
            "cfg_hash": config_hash(cfg)}
    info = H.detect_scale(meta)
    assert info["matched_factory"] == "440m"
    assert info["cfg_hash"] == config_hash(cfg)
    assert info["param_count"] == int(cfg.param_count_estimate())


def test_default_eval_plan_caps_context_and_scales_batch():
    plan = H.default_eval_plan(build_config("180m"), max_seq_len=2048)
    assert all(c <= 2048 for c in plan["niah_context_lens"])
    assert all(c <= 2048 for c in plan["mqar_seq_lens"])
    # batch size shrinks as width grows
    small = H.default_eval_plan(build_config("180m"))["ppl_batch_size"]
    big = H.default_eval_plan(build_config("3b"))["ppl_batch_size"]
    assert big <= small


def test_assemble_results_structure():
    res = H.assemble_results("ck/model.pt", {"model_size": "180m"},
                             {"ppl_batch_size": 8}, {"perplexity": {"ppl": 12.3}})
    assert set(res) == {"checkpoint", "scale", "eval_plan", "metrics"}
    assert res["metrics"]["perplexity"]["ppl"] == 12.3


def test_harness_main_importable():
    assert callable(H.main) and "ppl" in H.ALL_TASKS


# ---- SKA-zeroing ablation ----

def test_ska_block_ablation_is_passthrough():
    torch.manual_seed(0)
    cfg = build_config("440m")          # layerscale + short_conv => nonzero contribution
    blk = SKABlock(cfg).eval()
    x = torch.randn(2, 16, cfg.d_model)
    with torch.no_grad():
        y = blk(x)
        assert not torch.equal(y, x)    # normally contributes
        blk._ablate = True
        assert torch.equal(blk(x), x)   # zeroed => pure residual passthrough


def test_ablate_context_manager_sets_and_restores():
    cfg = build_config("440m")
    blk = SKABlock(cfg)

    class _FakeModel:
        seq_layers = [blk]

    fake = _FakeModel()
    cm = KoopmanLM.ablate(fake, zero_ska=True)   # contextmanager bound to fake self
    assert blk._ablate is False
    with cm:
        assert blk._ablate is True
    assert blk._ablate is False

