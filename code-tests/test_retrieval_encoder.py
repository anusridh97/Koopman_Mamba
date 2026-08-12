"""Torch tests for the retrieval encoder + InfoNCE + pooling.

CPU-only (no Mamba/CUDA): a stub backbone stands in for KoopmanLM so the encoder
wiring, InfoNCE, and pooling math are all testable without GPU kernels.
"""
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from experimentation.retrieval.encoder import RetrievalEncoder, info_nce, pool_sequence

pytestmark = pytest.mark.correctness


class _StubBackbone(nn.Module):
    """Minimal backbone exposing encode()/no_weight_decay_param_names()."""
    def __init__(self, vocab=64, d=16):
        super().__init__()
        self.embed = nn.Embedding(vocab, d)
        self.d = d

    def encode(self, input_ids, attention_mask=None, pool="mean"):
        h = self.embed(input_ids)                          # (B, T, d)
        return pool_sequence(h, attention_mask, pool)

    def no_weight_decay_param_names(self):
        return set()


def test_pool_mean_ignores_padding():
    h = torch.tensor([[[1.0, 1.0], [3.0, 3.0], [9.0, 9.0]]])   # (1, 3, 2)
    mask = torch.tensor([[1.0, 1.0, 0.0]])                     # last is pad
    out = pool_sequence(h, mask, "mean")
    assert torch.allclose(out, torch.tensor([[2.0, 2.0]]))     # mean of first two
    last = pool_sequence(h, mask, "last")
    assert torch.allclose(last, torch.tensor([[3.0, 3.0]]))    # last REAL token


def test_encoder_embeds_are_unit_norm():
    enc = RetrievalEncoder(_StubBackbone(), d_model=16, proj_dim=32, pool="mean")
    ids = torch.randint(0, 64, (4, 10))
    mask = torch.ones(4, 10)
    z = enc.embed(ids, mask)
    assert z.shape == (4, 32)
    assert torch.allclose(z.norm(dim=-1), torch.ones(4), atol=1e-5)


def test_param_groups_split_backbone_and_proj():
    enc = RetrievalEncoder(_StubBackbone(), d_model=16, proj_dim=32)
    groups = enc.param_groups(backbone_lr=8e-6, proj_lr=1e-4)
    lrs = sorted({g["lr"] for g in groups})
    assert lrs == [8e-6, 1e-4]
    # the proj group's lr must be the fast one
    proj_ids = {id(p) for p in enc.proj.parameters()}
    proj_group = [g for g in groups if any(id(p) in proj_ids for p in g["params"])]
    assert len(proj_group) == 1 and proj_group[0]["lr"] == 1e-4


def test_info_nce_minimized_when_aligned():
    torch.manual_seed(0)
    B, D = 8, 16
    q = torch.randn(B, D); q = q / q.norm(dim=-1, keepdim=True)
    # perfect: positive == query (cosine 1 on diagonal)
    aligned = info_nce(q, q.clone(), None, temperature=0.05)
    # scrambled positives -> higher loss
    perm = torch.randperm(B)
    scrambled = info_nce(q, q[perm].clone(), None, temperature=0.05)
    assert aligned < scrambled


def test_info_nce_hard_negatives_shape():
    torch.manual_seed(0)
    B, D, K = 4, 16, 2
    q = torch.randn(B, D); q = q / q.norm(dim=-1, keepdim=True)
    p = torch.randn(B, D); p = p / p.norm(dim=-1, keepdim=True)
    neg = torch.randn(B, K, D); neg = neg / neg.norm(dim=-1, keepdim=True)
    loss = info_nce(q, p, neg, temperature=0.05)
    assert loss.ndim == 0 and torch.isfinite(loss)
