r"""
csp.py -- Causal Structured Prefixing (CSP).

One framework, two concrete modes. The shared legality rule is:

    A token may use any structured object that was computable before that token.

Formally, each item alpha (a raw token, or a structured record: a turn, a chunk
summary, an entity table, a tool result, an SKA sufficient-statistic block, a
learned memory slot) has a *release time* rho_alpha, and it may condition target
token x_t only if

    rho_alpha < t.

Equivalently the attention mask over the combined [raw tokens (+) records] stream is

    A_{t,j} = 1[ rho_j < t ].

Raw tokens have rho(x_i) = i. A record produced after chunk c (ending at position
b_c) has rho(r) = b_c.

This module provides the group/release-time plumbing shared by both modes:

  Method 1 (chat / stopped-prefix): records are released at turn boundaries; loss is
    on assistant spans; SKA accumulates statistics over the prefix once, then decodes.

  Method 2 (timestamped structured-prefix LM): records are released at chunk
    boundaries during an ordinary document; loss on all tokens.

Both reduce to the same thing here: a per-token *group id* grp[t] (non-decreasing
along t), where all items in group g are released at the end of group g. A token in
group g may use groups strictly earlier than g (fully released), plus -- for
attention -- causally within its own group. The existing fixed-size chunk-causal SKA
mask is the special case grp = floor(arange(T)/chunk_size).
"""

from typing import List, Optional, Sequence
import torch

# Extra mode strings understood by train_eval.build_conditioning.
SKA_RELEASE_MODE = "release"     # SKA uses per-token release groups (grp) instead of fixed chunks
ATTN_SEGMENT_MODE = "segment"    # attention uses the segment/turn causal mask below


def groups_from_chunk_size(T: int, chunk_size: int, device="cpu") -> torch.Tensor:
    """grp[t] = t // chunk_size. Recovers the current fixed chunk-causal mask."""
    idx = torch.arange(T, device=device)
    return idx // chunk_size


def groups_from_boundaries(T: int, boundaries: Sequence[int], device="cpu") -> torch.Tensor:
    """
    grp increments after each boundary position. `boundaries` are the (exclusive)
    end positions of groups, e.g. [b_1, b_2, ...]; positions in [0, b_1) are group 0,
    [b_1, b_2) group 1, etc. This is the release-time-aligned (semantic) chunking.
    """
    idx = torch.arange(T, device=device)
    grp = torch.zeros(T, dtype=torch.long, device=device)
    for b in boundaries:
        grp = grp + (idx >= b).long()
    return grp


def group_release_positions(grp: torch.Tensor) -> torch.Tensor:
    """
    For each token, the release position of its group = last index of that group.
    grp: (B, T) non-decreasing per row. Returns (B, T) long.
    """
    B, T = grp.shape
    device = grp.device
    idx = torch.arange(T, device=device).expand(B, T)
    G = int(grp.max().item()) + 1
    # last index per (b, group): scatter max of idx by group id
    last = torch.full((B, G), -1, device=device, dtype=torch.long)
    last.scatter_reduce_(1, grp, idx, reduce="amax", include_self=True)
    return torch.gather(last, 1, grp)


def segment_causal_bias(grp: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Additive attention bias for the CSP release rule at group granularity:

        allowed[b, i, j] = (grp[b, j] <  grp[b, i])                 (earlier, closed group)
                        or (grp[b, j] == grp[b, i] and j <= i)      (causal within group)

    Recovers standard causal masking when grp = arange(T) (each token its own group),
    and multi-span prefix-LM masking when grp indexes turns.

    Returns (B, 1, T, T) with 0 where allowed and -inf where blocked.
    """
    B, T = grp.shape
    device = grp.device
    idx = torch.arange(T, device=device)
    gi = grp[:, :, None]                 # (B, T, 1)  query group
    gj = grp[:, None, :]                 # (B, 1, T)  key group
    within_causal = idx[None, None, :] <= idx[None, :, None]   # (1, T, T) j <= i
    allowed = (gj < gi) | ((gj == gi) & within_causal)         # (B, T, T)
    neg_inf = torch.finfo(dtype).min
    bias = torch.zeros(B, T, T, device=device, dtype=dtype)
    bias = bias.masked_fill(~allowed, neg_inf)
    return bias.unsqueeze(1)


def release_attn_bias(release_times: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Most general form: given a release time rho_j per item (B, T), build the mask
    A_{t,j} = 1[ rho_j < t ] for target positions t = 0..T-1. Use this when the item
    stream mixes raw tokens (rho = position) with records that were released earlier
    than their slot in the sequence.

    Returns (B, 1, T, T) additive bias.
    """
    B, T = release_times.shape
    device = release_times.device
    t_pos = torch.arange(T, device=device)[None, :, None]      # (1, T, 1) target t
    rho = release_times[:, None, :]                            # (B, 1, T) key release
    allowed = rho < t_pos                                      # (B, T, T)
    neg_inf = torch.finfo(dtype).min
    bias = torch.zeros(B, T, T, device=device, dtype=dtype)
    bias = bias.masked_fill(~allowed, neg_inf)
    return bias.unsqueeze(1)
