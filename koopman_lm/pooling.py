"""Sequence pooling: per-token hidden states -> one embedding per sequence.

Lives in the model core because ``KoopmanLM.encode`` needs it. It previously
lived in the since-moved ``koopman_lm/retrieval/encoder.py`` (now
``experimentation/retrieval/encoder.py``), which made the model core import from
a downstream research package -- the only edge in the repo pointing that way,
and the one thing standing between koopman_lm and a clean package boundary
(see docs/superpowers/specs/2026-08-08-package-restructure-design.md §3).

Nothing about the function changed in the move; ``retrieval.encoder`` re-exports
it so existing imports keep working.
"""
import torch


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
