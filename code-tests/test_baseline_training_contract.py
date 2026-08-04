"""The standard baselines share training semantics but not Echo components."""

from __future__ import annotations

import dataclasses
import math

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


# ---------------------------------------------------------------------------
# Generic initialization parity with KoopmanLM.
#
# The controls must start from the SAME generic Normal(0, 0.02) distribution
# that KoopmanLM._init_weights applies. Before this was wired up, the baselines
# kept PyTorch's default N(0, 1) embedding; with a tied head that makes the
# initial logits scale as sqrt(d_model) * std(E) = 8, so the initial loss sat
# far ABOVE ln(V) (measured 34-62 vs ln(32000) = 10.37) and the first steps were
# spent shrinking logits rather than learning.
# ---------------------------------------------------------------------------

INIT_STD = 0.02
PROBE_VOCAB = 32_000
LN_PROBE_VOCAB = math.log(PROBE_VOCAB)


def _init_probe_config():
    """Real d_model/vocab (so logit scale is meaningful), few layers (so it is
    cheap enough to run on CPU in the correctness gate)."""
    return dataclasses.replace(
        build_config("1m"),
        d_model=64,
        n_layers=2,
        vocab_size=PROBE_VOCAB,
        ska_n_heads=4,
        ska_layer_indices=(),
        mamba_headdim=16,   # required for the causal_conv1d kernel at d_model=64
        max_seq_len=128,
        tie_embeddings=True,
    )


@pytest.mark.parametrize("builder", [build_transformer, build_mamba_only])
def test_baseline_embedding_std_and_tied_weight_identity(builder) -> None:
    if builder is build_mamba_only:
        pytest.importorskip("mamba_ssm")
    torch.manual_seed(0)
    model = builder(_init_probe_config())

    # embedding standard deviation must be the generic 0.02, not N(0, 1)
    assert model.embed.weight.std().item() == pytest.approx(INIT_STD, rel=0.10)

    # tying must survive the in-place init pass: the SAME tensor, not a copy
    assert model.lm_head.weight is model.embed.weight
    assert model.lm_head.weight.data_ptr() == model.embed.weight.data_ptr()

    # every Linear picked up the generic std too (0.02 is far from the default
    # Kaiming-uniform std for these shapes)
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and module.weight is not model.embed.weight:
            # default Kaiming-uniform would be 0.042-0.072 for these fan_ins,
            # so this band genuinely discriminates.
            assert module.weight.std().item() == pytest.approx(INIT_STD, rel=0.20), name


def test_transformer_initial_logit_scale_and_loss_near_uniform() -> None:
    cfg = _init_probe_config()
    torch.manual_seed(0)
    model = build_transformer(cfg).eval()

    torch.manual_seed(1234)
    chunk = torch.randint(0, cfg.vocab_size, (4, 65))
    ids, labels = chunk[:, :-1].contiguous(), chunk[:, 1:].contiguous()
    with torch.no_grad():
        out = model(input_ids=ids, labels=labels)

    # logits must be small: sqrt(d_model) * 0.02 = 0.16, NOT sqrt(d_model) * 1 = 8
    logit_std = out["logits"].float().std().item()
    assert logit_std < 0.5, f"initial logit std {logit_std:.3f} too large (regressed init?)"

    # a near-uniform softmax over V classes means loss ~= ln(V)
    loss = out["loss"].item()
    assert math.isfinite(loss)
    assert abs(loss - LN_PROBE_VOCAB) < 0.25, (
        f"initial loss {loss:.4f} is not near ln({PROBE_VOCAB})={LN_PROBE_VOCAB:.4f}"
    )


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Mamba-2 forward needs CUDA")
def test_mamba_only_initial_logit_scale_and_loss_near_uniform() -> None:
    pytest.importorskip("mamba_ssm")
    cfg = _init_probe_config()
    torch.manual_seed(0)
    model = build_mamba_only(cfg).cuda().eval()

    torch.manual_seed(1234)
    chunk = torch.randint(0, cfg.vocab_size, (4, 65), device="cuda")
    ids, labels = chunk[:, :-1].contiguous(), chunk[:, 1:].contiguous()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = model(input_ids=ids, labels=labels)

    logit_std = out["logits"].float().std().item()
    assert logit_std < 0.5, f"initial logit std {logit_std:.3f} too large (regressed init?)"
    loss = out["loss"].item()
    assert math.isfinite(loss)
    assert abs(loss - LN_PROBE_VOCAB) < 0.25, (
        f"initial loss {loss:.4f} is not near ln({PROBE_VOCAB})={LN_PROBE_VOCAB:.4f}"
    )


def test_generic_init_leaves_mamba_ssm_internals_alone() -> None:
    """The generic pass must not change ARCHITECTURE-bearing Mamba-2 state:
    A_log/D/dt_bias are plain Parameters and conv1d is a Conv1d, so only
    in_proj/out_proj (nn.Linear) may be re-initialised."""
    pytest.importorskip("mamba_ssm")
    torch.manual_seed(0)
    mamba = build_mamba_only(_init_probe_config()).seq_layers[0].mamba

    assert torch.all(mamba.D == 1.0)                       # mamba_ssm default
    assert (mamba.A_log >= 0).all()                        # log of A_init_range (1,16)
    assert (mamba.dt_bias < 0).all()                       # inverse-softplus of small dt
    assert mamba.conv1d.weight.std().item() > 0.05          # untouched, not 0.02
    assert mamba.in_proj.weight.std().item() == pytest.approx(INIT_STD, rel=0.20)


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
