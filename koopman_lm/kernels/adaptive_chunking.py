"""
adaptive_chunking.py -- Alternative chunk-causal statistics for SKA.

Each function here has the SAME signature and return type as
ska._compute_chunk_stats_and_cholesky:

    Args:
        z_f:        [B, T, H, r]   float32 normalized keys
        zq_f:       [B, T, H, r]   float32 normalized queries
        v_f:        [B, T, H, P]   float32 values
        r:          int             rank
        H:          int             number of heads
        P:          int             per-head value dim
        CS:         int             base chunk size
        ridge_eps:  float           ridge regularization

    Returns:
        L_flat:     [BCH, r, r]    lower-triangular Cholesky of G_excl
        M_flat:     [BCH, r, r]    exclusive-prefix M (transition covariance)
        C_flat:     [BCH, P, r]    exclusive-prefix C_v (value readout)
        zq_flat:    [BCH, r, CS]   queries reshaped for post-Cholesky chain
        shapes:     tuple          (B, C, H, P, CS, T, T_padded, pad_len)

This means SKAModule.forward can swap between them with a single
if/elif without changing anything else in the forward path.  The
post-Cholesky matmul chain (_post_cholesky_pytorch / _post_cholesky_triton)
is completely unchanged.


Strategy overview
-----------------

standard (in ska.py):
    Non-overlapping chunks, uniform exclusive prefix-sum.
    Fast, simple, but chunk 0 has zero history and information
    at the end of chunk c isn't available until chunk c+1.

overlap:
    Chunks overlap by `overlap_fraction * CS` tokens.  Tokens near
    boundaries contribute to both neighboring chunks' operators,
    reducing the "boundary blindness" latency from 1 full chunk to
    ~half a chunk.  Cost: ~1/(1-overlap_fraction) more chunks, but
    each chunk is the same size, so the per-chunk work is identical.

decay:
    Exponential decay in the prefix-sum: chunk c's accumulated stats
    weight chunk i by alpha^(c-1-i).  Recent chunks dominate, distant
    chunks still contribute.  No hard boundaries.  Cost: replaces the
    O(C) cumsum with an O(C) serial loop (C is small, 16-64 typically).
    Best for distributed-information tasks (MQAR-Shuffle).


How to enable
-------------

In KoopmanLMConfig:
    ska_chunk_strategy:      'standard', 'overlap', or 'decay'
    ska_overlap_fraction:    0.5  (only used when strategy='overlap')
    ska_decay_alpha:         0.95 (only used when strategy='decay')

These flow through model.py -> SKABlock -> SKAModule.__init__(), where
SKAModule stores them as attributes and uses them in forward() to pick
which function to call from this file.
"""

import torch
import torch.nn.functional as F


def compute_chunk_stats_overlap(z_f, zq_f, v_f, r, H, P, CS, ridge_eps,
                                overlap_fraction=0.5):
    """
    Overlapping-chunk statistics with deduplication weighting.

    Chunks are placed with stride = CS * (1 - overlap_fraction), so
    adjacent chunks share `overlap_fraction * CS` tokens.  The
    accumulated G/M/C statistics are scaled down by the ratio of
    unique tokens to total tokens per chunk to avoid double-counting.

    The output contract is identical to _compute_chunk_stats_and_cholesky:
    the shapes tuple encodes the chunk geometry so that post-Cholesky
    code can unpack y_hat correctly.
    """
    B, T = z_f.shape[:2]
    device = z_f.device
    dtype = z_f.dtype

    stride = max(1, int(CS * (1 - overlap_fraction)))
    n_chunks = max(1, (T - CS + stride) // stride + 1)

    # Pad so every chunk has exactly CS tokens
    T_needed = (n_chunks - 1) * stride + CS
    pad_len = max(0, T_needed - T)
    if pad_len > 0:
        z_f  = F.pad(z_f,  (0, 0, 0, 0, 0, pad_len))
        zq_f = F.pad(zq_f, (0, 0, 0, 0, 0, pad_len))
        v_f  = F.pad(v_f,  (0, 0, 0, 0, 0, pad_len))

    T_padded = z_f.shape[1]
    C = n_chunks

    # Gather overlapping chunks  [B, C, CS, H, dim]
    z_c  = torch.stack([z_f[:,  i*stride : i*stride+CS] for i in range(C)], dim=1)
    zq_c = torch.stack([zq_f[:, i*stride : i*stride+CS] for i in range(C)], dim=1)
    v_c  = torch.stack([v_f[:,  i*stride : i*stride+CS] for i in range(C)], dim=1)

    # ---- Per-chunk statistics ----
    G_chunks = torch.einsum('bcthr,bcths->bchrs', z_c, z_c)
    M_chunks = torch.einsum('bcthr,bcths->bchrs', z_c[:,:,1:], z_c[:,:,:-1])
    C_chunks = torch.einsum('bcthp,bcthr->bchpr', v_c, z_c)

    # ---- Cross-chunk boundary transitions ----
    if C > 1:
        M_boundary = torch.einsum('bchr,bchs->bchrs',
                                  z_c[:, 1:, 0], z_c[:, :-1, -1])

    # ---- Deduplication scale ----
    overlap_tokens = CS - stride
    if overlap_tokens > 0 and CS > 0:
        dedup_scale = (stride + 0.5 * overlap_tokens) / CS
    else:
        dedup_scale = 1.0

    # ---- Exclusive prefix sum ----
    eye_r = torch.eye(r, device=device, dtype=dtype)

    G_scaled = G_chunks * dedup_scale
    G_cumsum = torch.cumsum(G_scaled, dim=1)
    G_excl = torch.zeros_like(G_cumsum)
    G_excl[:, 1:] = G_cumsum[:, :-1]
    G_excl = G_excl + ridge_eps * eye_r

    M_scaled = M_chunks * dedup_scale
    M_cumsum = torch.cumsum(M_scaled, dim=1)
    M_excl = torch.zeros_like(M_cumsum)
    M_excl[:, 1:] = M_cumsum[:, :-1]

    if C > 1:
        M_bnd_full = torch.zeros(B, C, H, r, r, device=device, dtype=dtype)
        M_bnd_full[:, 1:] = M_boundary
        M_excl = M_excl + torch.cumsum(M_bnd_full, dim=1)

    C_scaled = C_chunks * dedup_scale
    C_cumsum = torch.cumsum(C_scaled, dim=1)
    C_excl = torch.zeros_like(C_cumsum)
    C_excl[:, 1:] = C_cumsum[:, :-1]

    # ---- Batched Cholesky ----
    BCH = B * C * H
    G_flat = G_excl.reshape(BCH, r, r)
    G_flat = 0.5 * (G_flat + G_flat.transpose(-1, -2))

    L_flat, info = torch.linalg.cholesky_ex(G_flat)
    G_jittered = G_flat + 1e-4 * eye_r.unsqueeze(0)
    L_jittered, _ = torch.linalg.cholesky_ex(G_jittered)
    needs_fix = (info > 0).unsqueeze(-1).unsqueeze(-1)
    L_flat = torch.where(needs_fix, L_jittered, L_flat)

    M_flat = M_excl.reshape(BCH, r, r)
    C_flat = C_excl.reshape(BCH, P, r)
    zq_flat = zq_c.permute(0, 1, 3, 4, 2).reshape(BCH, r, CS)

    # Overlap produces C * CS output positions; we only want T.
    T_virtual = C * CS
    pad_len_virtual = T_virtual - T

    shapes = (B, C, H, P, CS, T, T_virtual, pad_len_virtual)
    return L_flat, M_flat, C_flat, zq_flat, shapes


def compute_chunk_stats_decay(z_f, zq_f, v_f, r, H, P, CS, ridge_eps,
                              decay_alpha=0.95):
    """
    Exponentially-decayed exclusive prefix sum.

    G_excl[c] = ridge*I + sum_{i<c} alpha^(c-1-i) * G_chunks[i]

    With alpha=1.0 this is identical to the standard uniform accumulation.
    With alpha<1.0, distant chunks are down-weighted, creating smooth
    forgetting instead of hard boundaries.

    The loop over C chunks is serial but C is small (16-64 for typical
    sequence lengths), so this adds negligible wall-clock cost compared
    to the O(BCH * r^2) Cholesky that follows.
    """
    B, T = z_f.shape[:2]
    device = z_f.device
    dtype = z_f.dtype

    n_chunks = (T + CS - 1) // CS
    T_padded = n_chunks * CS
    pad_len = T_padded - T

    if pad_len > 0:
        z_f  = F.pad(z_f,  (0, 0, 0, 0, 0, pad_len))
        zq_f = F.pad(zq_f, (0, 0, 0, 0, 0, pad_len))
        v_f  = F.pad(v_f,  (0, 0, 0, 0, 0, pad_len))

    C = n_chunks
    z_c  = z_f.reshape(B, C, CS, H, r)
    zq_c = zq_f.reshape(B, C, CS, H, r)
    v_c  = v_f.reshape(B, C, CS, H, P)

    # ---- Per-chunk statistics ----
    G_chunks = torch.einsum('bcthr,bcths->bchrs', z_c, z_c)
    M_chunks = torch.einsum('bcthr,bcths->bchrs', z_c[:,:,1:], z_c[:,:,:-1])
    C_chunks = torch.einsum('bcthp,bcthr->bchpr', v_c, z_c)

    if C > 1:
        M_boundary = torch.einsum('bchr,bchs->bchrs',
                                  z_c[:, 1:, 0], z_c[:, :-1, -1])

    # ---- Decayed exclusive prefix sum (serial over C) ----
    eye_r = torch.eye(r, device=device, dtype=dtype)
    alpha = decay_alpha

    G_excl = torch.zeros(B, C, H, r, r, device=device, dtype=dtype)
    M_excl = torch.zeros(B, C, H, r, r, device=device, dtype=dtype)
    C_excl = torch.zeros(B, C, H, P, r, device=device, dtype=dtype)

    G_acc = torch.zeros(B, H, r, r, device=device, dtype=dtype)
    M_acc = torch.zeros(B, H, r, r, device=device, dtype=dtype)
    C_acc = torch.zeros(B, H, P, r, device=device, dtype=dtype)

    for c in range(C):
        # Store exclusive (before adding chunk c)
        G_excl[:, c] = G_acc + ridge_eps * eye_r
        M_excl[:, c] = M_acc
        C_excl[:, c] = C_acc

        # Decay then add chunk c
        G_acc = alpha * G_acc + G_chunks[:, c]
        M_acc = alpha * M_acc + M_chunks[:, c]
        C_acc = alpha * C_acc + C_chunks[:, c]

        # Add boundary transition for next chunk
        if c < C - 1 and C > 1:
            M_acc = M_acc + M_boundary[:, c]

    # ---- Batched Cholesky ----
    BCH = B * C * H
    G_flat = G_excl.reshape(BCH, r, r)
    G_flat = 0.5 * (G_flat + G_flat.transpose(-1, -2))

    L_flat, info = torch.linalg.cholesky_ex(G_flat)
    G_jittered = G_flat + 1e-4 * eye_r.unsqueeze(0)
    L_jittered, _ = torch.linalg.cholesky_ex(G_jittered)
    needs_fix = (info > 0).unsqueeze(-1).unsqueeze(-1)
    L_flat = torch.where(needs_fix, L_jittered, L_flat)

    M_flat = M_excl.reshape(BCH, r, r)
    C_flat = C_excl.reshape(BCH, P, r)
    zq_flat = zq_c.permute(0, 1, 3, 4, 2).reshape(BCH, r, CS)

    shapes = (B, C, H, P, CS, T, T_padded, pad_len)
    return L_flat, M_flat, C_flat, zq_flat, shapes
