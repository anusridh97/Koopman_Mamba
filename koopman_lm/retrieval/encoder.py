"""encoder.py -- dual-encoder wrapper + InfoNCE for Phase-2 retrieval adaptation.

RetrievalEncoder wraps a pretrained KoopmanLM (or any backbone exposing
``encode(input_ids, attention_mask, pool)``) with:
  * a linear projection head to ``proj_dim`` (default 768), and
  * L2 normalization,
producing unit embeddings for query and passage. Query and passage share the
SAME encoder (a symmetric/tied dual-encoder), which is standard for
question<->passage retrieval and halves the parameters vs two towers.

The projection head is NEW (random init) and trained at a higher LR than the
backbone -- see ``param_groups``. The backbone weights come from the Phase-1
continued-pretraining checkpoint.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def pool_sequence(h, attention_mask=None, pool="mean"):
    """Pool per-token hidden states (B, T, d) -> (B, d).

    Shared by KoopmanLM.encode and any encoder. RIGHT-padding assumed (the
    backbone is causal, so pads sit after real tokens). 'mean' is a mask-weighted
    mean over real tokens; 'last' is the hidden at the last real token.
    """
    if attention_mask is None:
        attention_mask = torch.ones(h.shape[:2], device=h.device, dtype=h.dtype)
    m = attention_mask.to(h.dtype)
    if pool == "mean":
        denom = m.sum(dim=1, keepdim=True).clamp(min=1.0)
        return (h * m.unsqueeze(-1)).sum(dim=1) / denom
    if pool == "last":
        idx = (m.sum(dim=1).long().clamp(min=1) - 1)
        return h[torch.arange(h.shape[0], device=h.device), idx]
    raise ValueError(f"unknown pool {pool!r} (expected 'mean' or 'last')")


class RetrievalEncoder(nn.Module):
    def __init__(self, backbone, d_model, proj_dim=768, pool="mean"):
        super().__init__()
        self.backbone = backbone
        self.pool = pool
        # Projection head: d_model -> proj_dim. bias=False keeps it a pure linear
        # map into the retrieval space; the L2 norm afterwards removes any scale.
        self.proj = nn.Linear(d_model, proj_dim, bias=False)
        nn.init.normal_(self.proj.weight, std=0.02)

    def embed(self, input_ids, attention_mask=None):
        """(B, T) ids -> (B, proj_dim) L2-normalized embeddings."""
        h = self.backbone.encode(input_ids, attention_mask=attention_mask,
                                 pool=self.pool)      # (B, d_model)
        z = self.proj(h)                              # (B, proj_dim)
        return F.normalize(z, p=2, dim=-1)

    def forward(self, input_ids, attention_mask=None):
        return self.embed(input_ids, attention_mask)

    def param_groups(self, backbone_lr, proj_lr, weight_decay=0.0):
        """Two AdamW groups: slow backbone, fast projection head.

        The backbone is already a capable LM, so it is nudged gently
        (``backbone_lr`` ~ 8e-6); the fresh projection head needs a much larger
        LR (``proj_lr`` ~ 1e-4) to learn the retrieval space. Names that the
        backbone marks decay-exempt (e.g. Koopman-MLP v2 params) are honored.
        """
        skip = set()
        fn = getattr(self.backbone, "no_weight_decay_param_names", None)
        if fn is not None:
            skip = {f"backbone.{n}" for n in fn()}
        bb_decay, bb_nodecay, proj = [], [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith("proj."):
                proj.append(p)
            elif name in skip:
                bb_nodecay.append(p)
            else:
                bb_decay.append(p)
        groups = [
            {"params": bb_decay, "lr": backbone_lr, "weight_decay": weight_decay},
            {"params": bb_nodecay, "lr": backbone_lr, "weight_decay": 0.0},
            {"params": proj, "lr": proj_lr, "weight_decay": 0.0},
        ]
        return [g for g in groups if g["params"]]


def info_nce(q, d_pos, d_neg=None, temperature=0.05):
    """InfoNCE over unit embeddings, in-batch + optional hard negatives.

    Args:
      q      : (B, D) query embeddings (L2-normalized).
      d_pos  : (B, D) positive passage embeddings (L2-normalized), aligned to q.
      d_neg  : optional (B, K, D) per-query hard negatives (L2-normalized), or
               None to use only in-batch negatives.
      temperature: softmax temperature tau (default 0.05).

    Every other query's positive in the batch is an in-batch negative for q_i
    (the off-diagonal of q @ d_posᵀ), plus q_i's own K hard negatives. Returns a
    scalar cross-entropy where the target for row i is its own positive (column i).

    Assumes embeddings are already normalized (cosine == dot product), matching
    RetrievalEncoder.embed.
    """
    B = q.shape[0]
    # in-batch: logits[i, j] = q_i · d_pos_j ; diagonal is the true pair.
    logits = q @ d_pos.t()                            # (B, B)
    if d_neg is not None and d_neg.numel() > 0:
        # per-query hard negatives: hard[i, k] = q_i · d_neg[i, k]
        hard = torch.einsum("bd,bkd->bk", q, d_neg)   # (B, K)
        logits = torch.cat([logits, hard], dim=1)     # (B, B+K)
    logits = logits / temperature
    target = torch.arange(B, device=q.device)
    return F.cross_entropy(logits, target)
