"""Regression tests for the Table 2 reproduction contract (koopman_lm.experiments.table2).

table2.py builds models through the project's real stack
(koopman_lm.models.baselines), which requires mamba_ssm (CUDA/Linux only).
Tests that need an actual model are marked `gpu` and auto-skip on a CPU-only
box (see code-tests/conftest.py) -- everything else here (config, the three
curriculum generators) needs no GPU and always runs.
"""

import pytest
import torch

from koopman_lm.experiments import curricula, table2


def test_1m_config_matches_paper_sec_4_1():
    cfg = table2.build_config("1m")
    assert cfg.d_model == 128
    assert cfg.n_layers == 4
    assert cfg.vocab_size == 128
    assert cfg.d_state == 16
    assert cfg.ska_rank == 24
    assert cfg.ska_n_heads == 4


def test_model_builders_cover_all_three_table2_variants():
    # The paper's three Table 2 models (Sec 4.1, all SwiGLU) must be present.
    assert set(table2.TABLE2_MODEL_TYPES) == {"mamba_only", "mamba_attn", "mamba_ska_swiglu"}
    assert set(table2.TABLE2_MODEL_TYPES) <= set(table2.MODEL_BUILDERS)


def test_koopman_mlp_variant_is_available_but_not_a_table2_variant():
    # mamba_ska_koopman is an exploratory addition (Koopman MLP instead of
    # SwiGLU) -- available to build, but not part of Table 2's literal protocol.
    assert "mamba_ska_koopman" in table2.MODEL_BUILDERS
    assert "mamba_ska_koopman" not in table2.TABLE2_MODEL_TYPES


def test_koopman_mlp_expand_default_targets_matched_param_budget():
    # SwiGLUMLP/SpectralKoopmanMLP are pure PyTorch (only Mamba2Block needs
    # mamba_ssm), so the MLP-only math is checkable without a GPU. At the
    # shared cfg.mlp_expand default (2.667), SpectralKoopmanMLP (2 weight
    # matrices) is structurally smaller than SwiGLUMLP (3), landing the whole
    # model at ~0.74M vs build_mamba_ska_swiglu's 933,732 (confirmed on GPU)
    # -- not a matched-budget ablation. --koopman_mlp_expand defaults to 5.0
    # to compensate, holding the shared backbone (embeddings/Mamba-2/SKA)
    # fixed at its real, GPU-confirmed size.
    from koopman_lm.modules.mlp.koopman import SpectralKoopmanMLP
    from koopman_lm.models.baselines import SwiGLUMLP

    def count(m):
        return sum(p.numel() for p in m.parameters())

    n_layers = 4
    swiglu_total = 933_732  # confirmed on GPU: mamba_ska_swiglu, configs/1m.yaml
    shared_base = swiglu_total - n_layers * count(SwiGLUMLP(128, expand=2.667))

    default_expand = table2.build_arg_parser().get_default("koopman_mlp_expand")
    predicted_total = shared_base + n_layers * count(SpectralKoopmanMLP(128, expand=default_expand))
    assert 950_000 <= predicted_total <= 1_050_000, predicted_total


class _Args:
    batch_size = 8
    task_vocab_size = 128
    curriculum = "mixed"
    toolcall_keys = 8
    toolcall_queries = 4
    overwrite_prob = 0.3
    sysprompt_vars = 4
    sysprompt_decoys = 0
    seed = 0


def test_train_batch_alternates_toolcall_and_sysprompt():
    # Even steps: toolcall (4 supervised positions, GET opcode present).
    inp0, lab0 = table2.make_train_batch(0, _Args)
    assert inp0.shape == (8, table2.TRAIN_SEQ_LEN)
    assert (lab0 != -100).sum(dim=1).eq(4).all()
    assert (inp0 == curricula.TOOLCALL_GET_ID).any()
    # Odd steps: sysprompt (also 4 supervised positions, no GET opcode).
    inp1, lab1 = table2.make_train_batch(1, _Args)
    assert (lab1 != -100).sum(dim=1).eq(4).all()


def test_batch_mixed_is_the_default_and_blends_within_one_batch():
    assert table2.build_arg_parser().get_default("curriculum") == "batch_mixed"

    class BatchMixed(_Args):
        curriculum = "batch_mixed"

    inp, lab = table2.make_train_batch(5, BatchMixed)
    half = BatchMixed.batch_size // 2
    assert inp.shape == (BatchMixed.batch_size, table2.TRAIN_SEQ_LEN)
    # first half toolcall (op[0] is always TOOLCALL_SET_ID), second half sysprompt (never is)
    assert inp[:half, 0].eq(curricula.TOOLCALL_SET_ID).all()
    assert not inp[half:, 0].eq(curricula.TOOLCALL_SET_ID).any()
    assert (lab != -100).sum(dim=1).eq(4).all()


def test_curriculum_flag_isolates_a_single_task():
    class ToolcallOnly(_Args):
        curriculum = "toolcall"

    class SyspromptOnly(_Args):
        curriculum = "sysprompt"

    for step in range(4):
        inp, _ = table2.make_train_batch(step, ToolcallOnly)
        assert inp[:, 0].eq(curricula.TOOLCALL_SET_ID).all(), "toolcall-only mode must never emit sysprompt"
    for step in range(4):
        inp, _ = table2.make_train_batch(step, SyspromptOnly)
        assert not inp[:, 0].eq(curricula.TOOLCALL_SET_ID).any(), "sysprompt-only mode must never emit toolcall"


def test_toolcall_recency_is_correct():
    """The supervised label must be the LATEST SET for that key, not just any occurrence."""
    for trial in range(20):
        inputs, labels = curricula.make_toolcall(
            batch=8, seq_len=64, num_keys=8, num_queries=4, vocab_size=128,
            overwrite_prob=0.6, seed=trial,
        )
        n_ops = (64 - 3 * 4) // 3
        for b in range(8):
            row, lrow = inputs[b].tolist(), labels[b].tolist()
            latest = {}
            for i in range(0, 3 * n_ops, 3):
                if row[i] == curricula.TOOLCALL_SET_ID:
                    latest[row[i + 1]] = row[i + 2]
            for pos, val in enumerate(lrow):
                if val != -100:
                    assert latest[row[pos - 1]] == val


def test_sysprompt_decoys_never_supervised():
    """Decoy bindings in the gap must never be the queried answer."""
    for trial in range(20):
        inputs, labels = curricula.make_sysprompt(
            batch=8, seq_len=64, num_vars=4, vocab_size=128, num_decoys=4, seed=trial,
        )
        for b in range(8):
            row, lrow = inputs[b].tolist(), labels[b].tolist()
            header_vars = set(row[0:8:2])
            for pos, val in enumerate(lrow):
                if val != -100:
                    assert row[pos - 1] in header_vars


def test_niah_needle_position_is_random_not_fixed():
    inputs, labels = curricula.make_niah(batch=32, seq_len=128, vocab_size=128, seed=0)
    assert (labels != -100).sum(dim=1).eq(1).all()
    positions = []
    for b in range(32):
        key = inputs[b, -2].item()
        match = (inputs[b, :-2] == key).nonzero()
        positions.append(match[0].item() if len(match) else -1)
    assert len(set(positions)) > 1, "needle should not sit at a fixed offset"


def test_niah_never_produced_by_training_generators():
    # Structural check: make_niah's single-needle format is disjoint from
    # both training curricula's fixed supervised-position counts.
    inp, lab = curricula.make_niah(batch=8, seq_len=table2.TRAIN_SEQ_LEN, vocab_size=128, seed=0)
    assert (lab != -100).sum(dim=1).eq(1).all()
    inp_tc, lab_tc = table2.make_train_batch(0, _Args)
    inp_sp, lab_sp = table2.make_train_batch(1, _Args)
    assert (lab_tc != -100).sum(dim=1).eq(1).any().item() is False
    assert (lab_sp != -100).sum(dim=1).eq(1).any().item() is False


@pytest.mark.gpu
@pytest.mark.parametrize("model_type", sorted(table2.MODEL_BUILDERS))
def test_variants_build_and_stay_submillion_scale(model_type):
    cfg = table2.build_config("1m")
    model = table2.build_model(model_type, cfg)
    n_params = table2.count_params(model)
    assert n_params < 1_200_000


@pytest.mark.gpu
def test_table2_forward_and_shifted_loss_smoke():
    # mamba_ssm's Mamba2Block dispatches to CUDA-only kernels (causal_conv1d,
    # selective scan) -- both model and inputs must be on the GPU device, like
    # the neighboring test_niah_eval_is_zero_shot_smoke does.
    device = torch.device("cuda")
    cfg = table2.build_config("1m")
    model = table2.build_model("mamba_ska_swiglu", cfg).to(device)
    inputs, labels = table2.make_train_batch(0, _Args)
    inputs, labels = inputs.to(device), labels.to(device)
    out = model(input_ids=inputs)
    logits = out["logits"]
    loss = torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1), ignore_index=-100,
    )
    assert logits.shape == (8, table2.TRAIN_SEQ_LEN, 128)
    assert torch.isfinite(loss)


@pytest.mark.gpu
def test_niah_eval_is_zero_shot_smoke():
    cfg = table2.build_config("1m")
    model = table2.build_model("mamba_ska_swiglu", cfg)
    device = torch.device("cuda")
    model.to(device)
    acc = curricula.eval_niah(
        model, batch=8, seq_len=table2.TRAIN_SEQ_LEN, vocab_size=128, device=device, seed=9999,
    )
    assert 0.0 <= acc <= 1.0
