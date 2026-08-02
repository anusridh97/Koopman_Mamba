"""The standard baselines share training semantics but not Echo components."""

from __future__ import annotations

import dataclasses

import pytest
import torch
import torch.nn.functional as F

from koopman_lm.globals.config import build_config
from koopman_lm.globals.modules.attention import CausalAttentionBlock
from koopman_lm.globals.modules.mamba import Mamba2Block
from koopman_lm.models.baselines import (
    SwiGLUMLP,
    build_mamba_only,
    build_transformer,
)

pytestmark = pytest.mark.correctness


def _tiny_config():
    return dataclasses.replace(
        build_config("1m"),
        d_model=32,
        n_layers=2,
        vocab_size=64,
        ska_n_heads=4,
        ska_rank=8,
        ska_layer_indices=(0,),
        max_seq_len=16,
    )


def test_transformer_uses_only_attention_and_swiglu() -> None:
    model = build_transformer(_tiny_config())

    assert all(isinstance(layer, CausalAttentionBlock) for layer in model.seq_layers)
    assert all(isinstance(layer, SwiGLUMLP) for layer in model.mlp_layers)
    assert not any("ska" in type(module).__name__.lower() for module in model.modules())
    assert not any("koopman" in type(module).__name__.lower() for module in model.modules())


def test_mamba_baseline_uses_only_mamba_and_swiglu() -> None:
    pytest.importorskip("mamba_ssm")
    model = build_mamba_only(_tiny_config())

    assert all(isinstance(layer, Mamba2Block) for layer in model.seq_layers)
    assert all(isinstance(layer, SwiGLUMLP) for layer in model.mlp_layers)
    assert not any("ska" in type(module).__name__.lower() for module in model.modules())
    assert not any("koopman" in type(module).__name__.lower() for module in model.modules())


def test_baseline_weighted_loss_matches_explicit_weighted_ce() -> None:
    torch.manual_seed(0)
    model = build_transformer(_tiny_config()).eval()
    input_ids = torch.randint(0, 64, (2, 8))
    labels = torch.randint(0, 64, (2, 8))
    labels[0, 0] = -100
    weights = torch.ones(2, 8)
    weights[:, 4:] = 3.0

    output = model(input_ids=input_ids, labels=labels, loss_weights=weights)
    ce = F.cross_entropy(
        output["logits"].reshape(-1, 64),
        labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).view_as(labels)
    effective = weights * (labels != -100)
    expected = (ce * effective).sum() / effective.sum()

    torch.testing.assert_close(output["loss"], expected)
