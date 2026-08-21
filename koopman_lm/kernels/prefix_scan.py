"""Exact SKA as a two-level prefix scan with O(r^2) rank-1 Cholesky writes.

This module is the correctness-first implementation of the scan design in
``PREFIX_SCAN_MATH.md``.  It separates two objects that should not be conflated:

1. The *associative* scan state is the raw sufficient-statistic segment summary

       (dG, dM, dC, first_x, last_x, nonempty),

   where dG and dC are additive and dM receives exactly one boundary outer
   product when adjacent segments are concatenated.

2. The *local recurrent* state is the whitened state

       L L^T = G,  A = L^{-1} M L^{-T},  R = C L^{-T},

   plus the previous key in the current whitened coordinates.  It is evaluated
   from the exclusive block-prefix summary, then advanced token-by-token with a
   rank-1 Givens Cholesky update and rotation replay.  Each local write costs
   O(r^2 + p r), not O(r^3).

The Python/Torch implementation is the portable correctness backend: its
forward has the exact blocked scan structure and its custom backward uses an
analytic reverse prefix scan.  The production rank-24/value-64 CUDA backend in
``csrc/prefix_scan_ext.cu`` fuses these same phases without changing the math.

Shapes used by the public entry point:
    x     : (B, T, H, r)  symmetric key sqrt(beta) * normalized_key
    q     : (B, T, H, r)  normalized query
    vbar  : (B, T, H, p)  symmetric value sqrt(beta) * value
    output: (B, T, H, p)

The read is exclusive and strictly causal: token t reads the state built from
indices i < t, then token t is written.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Tuple

import torch
import torch.nn.functional as F

from koopman_lm.kernels.lin_alg import exclusive_cumsum


# ---------------------------------------------------------------------------
# Associative raw segment summaries
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SegmentSummary:
    """Raw sufficient statistics for one contiguous, possibly empty segment.

    Tensor leading dimensions are arbitrary and shared.  Final dimensions are:
      gram       (..., r, r)
      transition (..., r, r)
      value      (..., p, r)
      first/last (..., r)
      nonempty   (...) bool

    ``transition`` contains only lag-one pairs internal to the segment.  The
    cross-segment lag pair is inserted by :func:`compose_summaries`.
    """

    gram: torch.Tensor
    transition: torch.Tensor
    value: torch.Tensor
    first: torch.Tensor
    last: torch.Tensor
    nonempty: torch.Tensor


def singleton_summary(x: torch.Tensor, vbar: torch.Tensor) -> SegmentSummary:
    """Summary of a single token."""
    if x.shape[:-1] != vbar.shape[:-1]:
        raise ValueError("x and vbar must share leading dimensions")
    gram = x.unsqueeze(-1) @ x.unsqueeze(-2)
    transition = torch.zeros_like(gram)
    value = vbar.unsqueeze(-1) @ x.unsqueeze(-2)
    nonempty = torch.ones(x.shape[:-1], dtype=torch.bool, device=x.device)
    return SegmentSummary(gram, transition, value, x, x, nonempty)


def empty_summary(
    *batch_shape: int,
    rank: int,
    value_dim: int,
    device: torch.device | str,
    dtype: torch.dtype,
) -> SegmentSummary:
    """Identity element for :func:`compose_summaries`."""
    gram = torch.zeros(*batch_shape, rank, rank, device=device, dtype=dtype)
    transition = torch.zeros_like(gram)
    value = torch.zeros(*batch_shape, value_dim, rank, device=device, dtype=dtype)
    key = torch.zeros(*batch_shape, rank, device=device, dtype=dtype)
    nonempty = torch.zeros(*batch_shape, device=device, dtype=torch.bool)
    return SegmentSummary(gram, transition, value, key, key, nonempty)


def compose_summaries(left: SegmentSummary, right: SegmentSummary) -> SegmentSummary:
    """Return the summary of ``left`` followed by ``right``.

    This is an associative operation.  For two nonempty adjacent segments the
    only new lag-one pair is ``right.first @ left.last.T``.
    """
    if left.gram.shape != right.gram.shape:
        raise ValueError("left/right gram shapes differ")
    if left.value.shape != right.value.shape:
        raise ValueError("left/right value shapes differ")

    both = (left.nonempty & right.nonempty).to(left.gram.dtype)
    while both.ndim < left.gram.ndim:
        both = both.unsqueeze(-1)
    boundary = right.first.unsqueeze(-1) @ left.last.unsqueeze(-2)

    gram = left.gram + right.gram
    transition = left.transition + right.transition + both * boundary
    value = left.value + right.value

    choose_left = left.nonempty.unsqueeze(-1)
    first = torch.where(choose_left, left.first, right.first)
    choose_right = right.nonempty.unsqueeze(-1)
    last = torch.where(choose_right, right.last, left.last)
    nonempty = left.nonempty | right.nonempty
    return SegmentSummary(gram, transition, value, first, last, nonempty)


# ---------------------------------------------------------------------------
# Basic scans and exact block-boundary states
# ---------------------------------------------------------------------------




def _future_exclusive_sum(x: torch.Tensor, dim: int) -> torch.Tensor:
    """``out[i] = sum_{j>i} x[j]`` along ``dim``."""
    if x.shape[dim] == 0:
        return x.clone()
    rev = torch.flip(x, dims=(dim,))
    inclusive = torch.cumsum(rev, dim=dim)
    future_in_rev_order = inclusive - rev
    return torch.flip(future_in_rev_order, dims=(dim,))


def _flatten_heads(t: torch.Tensor) -> torch.Tensor:
    """(B,T,H,D) -> (B*H,T,D)."""
    if t.ndim != 4:
        raise ValueError(f"expected a rank-4 tensor, got shape={tuple(t.shape)}")
    B, T, H, D = t.shape
    return t.permute(0, 2, 1, 3).contiguous().reshape(B * H, T, D)


def _unflatten_heads(t: torch.Tensor, B: int, H: int) -> torch.Tensor:
    """(B*H,T,D) -> (B,T,H,D)."""
    N, T, D = t.shape
    if N != B * H:
        raise ValueError("flattened head count does not match B*H")
    return t.reshape(B, H, T, D).permute(0, 2, 1, 3).contiguous()


def block_prefix_statistics(
    x: torch.Tensor,
    vbar: torch.Tensor,
    block_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Exact exclusive raw prefixes at block boundaries.

    Args:
      x:    (N,T,r)
      vbar: (N,T,p)

    Returns:
      G0:       (N,nb,r,r), sum x_i x_i^T before each block
      M0:       (N,nb,r,r), sum x_i x_{i-1}^T before each block
      C0:       (N,nb,p,r), sum vbar_i x_i^T before each block
      prev_x:   (N,nb,r), last raw key before each block (zero for block zero)
      has_prev: (N,nb) bool
      padded_T

    The implementation uses block GEMMs plus an ordinary exclusive scan.  It is
    algebraically the same as scanning :class:`SegmentSummary` objects.
    """
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if x.ndim != 3 or vbar.ndim != 3:
        raise ValueError("x and vbar must have shapes (N,T,r)/(N,T,p)")
    if x.shape[:2] != vbar.shape[:2]:
        raise ValueError("x and vbar must share N,T")

    N, T, r = x.shape
    p = vbar.shape[-1]
    nb = max(1, math.ceil(T / block_size))
    padded_T = nb * block_size
    pad = padded_T - T
    if pad:
        xp = F.pad(x, (0, 0, 0, pad))
        vp = F.pad(vbar, (0, 0, 0, pad))
    else:
        xp, vp = x, vbar

    xb = xp.reshape(N, nb, block_size, r)
    vb = vp.reshape(N, nb, block_size, p)

    dG = torch.einsum("nbsr,nbsk->nbrk", xb, xb)
    dC = torch.einsum("nbsp,nbsr->nbpr", vb, xb)

    dM = torch.zeros_like(dG)
    if block_size > 1:
        dM = dM + torch.einsum(
            "nbsr,nbsk->nbrk", xb[:, :, 1:], xb[:, :, :-1]
        )
    if nb > 1:
        boundary = xb[:, 1:, 0].unsqueeze(-1) @ xb[:, :-1, -1].unsqueeze(-2)
        dM[:, 1:] = dM[:, 1:] + boundary

    G0 = exclusive_cumsum(dG, dim=1)
    M0 = exclusive_cumsum(dM, dim=1)
    C0 = exclusive_cumsum(dC, dim=1)

    prev_x = torch.zeros(N, nb, r, device=x.device, dtype=x.dtype)
    has_prev = torch.zeros(N, nb, device=x.device, dtype=torch.bool)
    if nb > 1:
        boundary_indices = torch.arange(
            block_size - 1, (nb - 1) * block_size, block_size, device=x.device
        )
        prev_x[:, 1:] = x.index_select(1, boundary_indices)
        has_prev[:, 1:] = True
    return G0, M0, C0, prev_x, has_prev, padded_T


@torch.no_grad()
def _canonical_psd_factor(W: torch.Tensor, out_width: int) -> torch.Tensor:
    """Canonical factor F with F F^T = W W^T, padded to ``out_width``.

    ``W`` has shape (..., r, k).  A thin QR of W^T gives the exact factor
    R^T.  A deterministic diagonal sign convention removes the QR gauge.
    """
    _, R = torch.linalg.qr(W.transpose(-1, -2).contiguous(), mode="reduced")
    d = torch.diagonal(R, dim1=-2, dim2=-1).sign()
    d = torch.where(d == 0, torch.ones_like(d), d)
    Fthin = (R * d.unsqueeze(-1)).transpose(-1, -2).contiguous()
    if Fthin.shape[-1] == out_width:
        return Fthin
    if Fthin.shape[-1] > out_width:
        raise ValueError("canonical factor is wider than requested output")
    return F.pad(Fthin, (0, out_width - Fthin.shape[-1]))


@torch.no_grad()
def block_prefix_cholesky_factor_scan(
    x: torch.Tensor,
    block_size: int,
    ridge: float,
) -> torch.Tensor:
    """Exact block-boundary Cholesky factors via a PSD factor scan.

    Leaves are block key matrices ``X_b`` with ``X_b X_b^T = dG_b``.  The
    upsweep merges square-root factors by thin QR.  The downsweep is seeded by
    ``sqrt(ridge) I`` and forms every right-child prefix solely through rank-1
    Givens Cholesky updates.

    Returns ``L0`` with shape (N, nb, r, r), where

        L0[:,b] L0[:,b]^T = ridge I + sum_{j<b} dG_j.

    For block_size = Theta(r), total work is O(T r^2) and storage is
    O((T/block_size) r^2).
    """
    if x.ndim != 3:
        raise ValueError("x must have shape (N,T,r)")
    if block_size <= 0 or ridge <= 0:
        raise ValueError("block_size and ridge must be positive")
    N, T, r = x.shape
    nb = max(1, math.ceil(T / block_size))
    padded_T = nb * block_size
    if padded_T != T:
        xp = F.pad(x, (0, 0, 0, padded_T - T))
    else:
        xp = x

    # One fixed-width r-column factor per block.  Zero padding is harmless.
    leaves = xp.reshape(N, nb, block_size, r).transpose(-1, -2).contiguous()
    leaves = _canonical_psd_factor(leaves, r)

    nb_tree = 1 if nb <= 1 else 1 << (nb - 1).bit_length()
    if nb_tree != nb:
        leaves = F.pad(leaves, (0, 0, 0, 0, 0, nb_tree - nb))

    cur = leaves
    lefts = []
    while cur.shape[1] > 1:
        left = cur[:, 0::2]
        right = cur[:, 1::2]
        lefts.append(left)
        cur = _canonical_psd_factor(torch.cat([left, right], dim=-1), r)

    eye = torch.eye(r, device=x.device, dtype=x.dtype)
    pref = (math.sqrt(ridge) * eye).expand(N, 1, r, r).contiguous()
    for left in reversed(lefts):
        n_nodes = left.shape[1]
        right_pref = pref.reshape(N * n_nodes, r, r).clone()
        updates = left.reshape(N * n_nodes, r, r)
        for j in range(r):
            right_pref, _, _, _ = _rank1_cholesky_update(
                right_pref, updates[:, :, j]
            )
        right_pref = right_pref.reshape(N, n_nodes, r, r)
        nxt = torch.empty(
            N, 2 * n_nodes, r, r, device=x.device, dtype=x.dtype
        )
        nxt[:, 0::2] = pref
        nxt[:, 1::2] = right_pref
        pref = nxt
    return pref[:, :nb]


def _stable_cholesky(G: torch.Tensor, jitter: float = 0.0) -> torch.Tensor:
    """Batched lower Cholesky with a deterministic fallback jitter."""
    L, info = torch.linalg.cholesky_ex(G)
    if bool((info > 0).any()):
        r = G.shape[-1]
        eye = torch.eye(r, device=G.device, dtype=G.dtype)
        fallback = max(float(jitter), 1e-6)
        Lj, info_j = torch.linalg.cholesky_ex(G + fallback * eye)
        if bool((info_j > 0).any()):
            bad = int((info_j > 0).sum().item())
            raise RuntimeError(f"Cholesky failed for {bad} prefix states")
        mask = (info > 0).unsqueeze(-1).unsqueeze(-1)
        L = torch.where(mask, Lj, L)
    return L


def boundary_whitened_states(
    G0: torch.Tensor,
    M0: torch.Tensor,
    C0: torch.Tensor,
    prev_x: torch.Tensor,
    has_prev: torch.Tensor,
    ridge: float,
    jitter: float = 0.0,
    L_precomputed: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert raw exclusive block-prefix summaries to local recurrent states."""
    if ridge <= 0:
        raise ValueError("ridge must be strictly positive")
    r = G0.shape[-1]
    eye = torch.eye(r, device=G0.device, dtype=G0.dtype)
    G = 0.5 * (G0 + G0.transpose(-1, -2)) + ridge * eye
    if L_precomputed is None:
        L = _stable_cholesky(G, jitter=jitter)
    else:
        if L_precomputed.shape != G.shape:
            raise ValueError("L_precomputed shape does not match block prefixes")
        L = L_precomputed

    LiM = torch.linalg.solve_triangular(L, M0, upper=False)
    A = torch.linalg.solve_triangular(
        L, LiM.transpose(-1, -2), upper=False
    ).transpose(-1, -2)
    R = torch.linalg.solve_triangular(
        L, C0.transpose(-1, -2), upper=False
    ).transpose(-1, -2)
    h_prev = torch.linalg.solve_triangular(
        L, prev_x.unsqueeze(-1), upper=False
    ).squeeze(-1)
    h_prev = torch.where(has_prev.unsqueeze(-1), h_prev, torch.zeros_like(h_prev))
    return L, A, R, h_prev, has_prev


# ---------------------------------------------------------------------------
# O(r^2) local recurrent update via rank-1 Cholesky rotations
# ---------------------------------------------------------------------------


def _rank1_cholesky_update(
    L_prev: torch.Tensor,
    x: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact batched rank-1 update using an ordered Givens sweep.

    Returns ``(L_new, c, s, u)`` with ``u = L_new^{-1} x``.  Work is O(r^2)
    and the only sequential dependence is the r-long rotation chain.
    """
    N, r, _ = L_prev.shape
    L = L_prev.clone()
    z = x.clone()
    c_all = torch.empty(N, r, device=L.device, dtype=L.dtype)
    s_all = torch.empty_like(c_all)
    u = torch.zeros_like(c_all)
    e = torch.ones(N, device=L.device, dtype=L.dtype)

    for k in range(r):
        a = L[:, k, k]
        b = z[:, k]
        rho = torch.sqrt(a * a + b * b).clamp_min(1e-30)
        c = a / rho
        s = b / rho
        c_all[:, k] = c
        s_all[:, k] = s

        col = L[:, k:, k].clone()
        tail = z[:, k:].clone()
        L[:, k:, k] = c.unsqueeze(-1) * col + s.unsqueeze(-1) * tail
        z[:, k:] = -s.unsqueeze(-1) * col + c.unsqueeze(-1) * tail

        u[:, k] = s * e
        e = c * e
    return L, c_all, s_all, u


def _transport_vector(h: torch.Tensor, c: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """Apply ``T=L_new^{-1}L_old`` to a whitened vector using rotation replay."""
    out = h.clone()
    pad = torch.zeros(h.shape[0], device=h.device, dtype=h.dtype)
    for k in range(h.shape[-1]):
        a = out[:, k].clone()
        b = pad.clone()
        out[:, k] = c[:, k] * a + s[:, k] * b
        pad = -s[:, k] * a + c[:, k] * b
    return out


def _transport_operator(A: torch.Tensor, c: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """Compute ``T A T^T`` by replaying the Cholesky rotations."""
    N, r, _ = A.shape
    aug = torch.zeros(N, r + 1, r + 1, device=A.device, dtype=A.dtype)
    aug[:, :r, :r] = A
    pad = r
    for k in range(r):
        ck = c[:, k].unsqueeze(-1)
        sk = s[:, k].unsqueeze(-1)
        row_k = aug[:, k, :].clone()
        row_p = aug[:, pad, :].clone()
        aug[:, k, :] = ck * row_k + sk * row_p
        aug[:, pad, :] = -sk * row_k + ck * row_p
    for k in range(r):
        ck = c[:, k].unsqueeze(-1)
        sk = s[:, k].unsqueeze(-1)
        col_k = aug[:, :, k].clone()
        col_p = aug[:, :, pad].clone()
        aug[:, :, k] = ck * col_k + sk * col_p
        aug[:, :, pad] = -sk * col_k + ck * col_p
    return aug[:, :r, :r].contiguous()


def _transport_readout(R: torch.Tensor, c: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """Compute ``R T^T`` by replaying the right-side rotations."""
    N, p, r = R.shape
    aug = torch.zeros(N, p, r + 1, device=R.device, dtype=R.dtype)
    aug[:, :, :r] = R
    pad = r
    for k in range(r):
        ck = c[:, k].unsqueeze(-1)
        sk = s[:, k].unsqueeze(-1)
        col_k = aug[:, :, k].clone()
        col_p = aug[:, :, pad].clone()
        aug[:, :, k] = ck * col_k + sk * col_p
        aug[:, :, pad] = -sk * col_k + ck * col_p
    return aug[:, :, :r].contiguous()


def _read_state(
    L: torch.Tensor,
    A: torch.Tensor,
    R: torch.Tensor,
    q: torch.Tensor,
    power_k: int,
) -> torch.Tensor:
    u = torch.linalg.solve_triangular(L, q.unsqueeze(-1), upper=False)
    for _ in range(power_k):
        u = A @ u
    return (R @ u).squeeze(-1)


@torch.no_grad()
def local_prefix_scan_reference(
    x_blocks: torch.Tensor,
    q_blocks: torch.Tensor,
    v_blocks: torch.Tensor,
    valid_blocks: torch.Tensor,
    L0: torch.Tensor,
    A0: torch.Tensor,
    R0: torch.Tensor,
    hprev0: torch.Tensor,
    hasprev0: torch.Tensor,
    power_k: int,
) -> torch.Tensor:
    """Process all blocks in parallel; scan exactly within each block.

    Inputs use shapes ``(N,nb,S,*)`` and boundary states ``(N,nb,...)``.
    Returns ``(N,nb,S,p)``.
    """
    N, nb, S, r = x_blocks.shape
    p = v_blocks.shape[-1]
    states = N * nb

    xb = x_blocks.reshape(states, S, r)
    qb = q_blocks.reshape(states, S, r)
    vb = v_blocks.reshape(states, S, p)
    mb = valid_blocks.reshape(states, S)

    L = L0.reshape(states, r, r).clone()
    A = A0.reshape(states, r, r).clone()
    R = R0.reshape(states, p, r).clone()
    hprev = hprev0.reshape(states, r).clone()
    hasprev = hasprev0.reshape(states).clone()

    out = torch.zeros(states, S, p, device=x_blocks.device, dtype=x_blocks.dtype)
    for j in range(S):
        valid = mb[:, j]
        y = _read_state(L, A, R, qb[:, j], power_k)
        out[:, j] = torch.where(valid.unsqueeze(-1), y, torch.zeros_like(y))

        L_new, c, s, u = _rank1_cholesky_update(L, xb[:, j])
        v_prev = _transport_vector(hprev, c, s)
        A_new = _transport_operator(A, c, s)
        A_new = A_new + (
            hasprev.to(A.dtype).unsqueeze(-1).unsqueeze(-1)
            * (u.unsqueeze(-1) @ v_prev.unsqueeze(-2))
        )
        R_new = _transport_readout(R, c, s) + vb[:, j].unsqueeze(-1) @ u.unsqueeze(-2)

        mat_mask = valid.unsqueeze(-1).unsqueeze(-1)
        vec_mask = valid.unsqueeze(-1)
        L = torch.where(mat_mask, L_new, L)
        A = torch.where(mat_mask, A_new, A)
        R = torch.where(mat_mask, R_new, R)
        hprev = torch.where(vec_mask, u, hprev)
        hasprev = hasprev | valid

    return out.reshape(N, nb, S, p)


@torch.no_grad()
def _prefix_scan_forward_reference(
    x: torch.Tensor,
    q: torch.Tensor,
    vbar: torch.Tensor,
    ridge: float,
    power_k: int,
    block_size: int,
    jitter: float,
) -> torch.Tensor:
    """Blocked exact forward used by the custom autograd function."""
    if x.shape != q.shape:
        raise ValueError("x and q must have identical shapes")
    if x.shape[:-1] != vbar.shape[:-1]:
        raise ValueError("x/q/vbar must share B,T,H")
    if power_k < 0:
        raise ValueError("power_k must be non-negative")

    B, T, H, r = x.shape
    p = vbar.shape[-1]
    xf = _flatten_heads(x)
    qf = _flatten_heads(q)
    vf = _flatten_heads(vbar)
    N = xf.shape[0]

    G0, M0, C0, prev_x, has_prev, padded_T = block_prefix_statistics(
        xf, vf, block_size
    )
    L_factor = block_prefix_cholesky_factor_scan(xf, block_size, ridge)
    L0, A0, R0, h0, hp0 = boundary_whitened_states(
        G0, M0, C0, prev_x, has_prev, ridge=ridge, jitter=jitter,
        L_precomputed=L_factor,
    )

    nb = padded_T // block_size
    pad = padded_T - T
    if pad:
        xp = F.pad(xf, (0, 0, 0, pad))
        qp = F.pad(qf, (0, 0, 0, pad))
        vp = F.pad(vf, (0, 0, 0, pad))
    else:
        xp, qp, vp = xf, qf, vf
    x_blocks = xp.reshape(N, nb, block_size, r)
    q_blocks = qp.reshape(N, nb, block_size, r)
    v_blocks = vp.reshape(N, nb, block_size, p)

    valid = torch.arange(padded_T, device=x.device) < T
    valid = valid.reshape(1, nb, block_size).expand(N, -1, -1)
    yb = local_prefix_scan_reference(
        x_blocks,
        q_blocks,
        v_blocks,
        valid,
        L0,
        A0,
        R0,
        h0,
        hp0,
        power_k,
    )
    yf = yb.reshape(N, padded_T, p)[:, :T]
    return _unflatten_heads(yf, B, H)


# ---------------------------------------------------------------------------
# Analytic reverse scan (custom backward)
# ---------------------------------------------------------------------------


@torch.no_grad()
def _exact_prefix_tensors(
    x: torch.Tensor,
    q: torch.Tensor,
    vbar: torch.Tensor,
    ridge: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Raw exact per-token exclusive prefixes for the correctness backward."""
    N, T, r = x.shape
    p = vbar.shape[-1]
    g = x.unsqueeze(-1) @ x.unsqueeze(-2)
    m = torch.zeros(N, T, r, r, device=x.device, dtype=x.dtype)
    if T > 1:
        m[:, 1:] = x[:, 1:].unsqueeze(-1) @ x[:, :-1].unsqueeze(-2)
    c = vbar.unsqueeze(-1) @ x.unsqueeze(-2)

    eye = torch.eye(r, device=x.device, dtype=x.dtype)
    G = exclusive_cumsum(g, dim=1) + ridge * eye
    M = exclusive_cumsum(m, dim=1)
    C = exclusive_cumsum(c, dim=1)
    return G, M, C, q.unsqueeze(-1)


@torch.no_grad()
def _prefix_adjoint_from_state(
    L: torch.Tensor,
    A: torch.Tensor,
    R: torch.Tensor,
    q: torch.Tensor,
    dy: torch.Tensor,
    power_k: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Adjoint of one strictly-causal read in raw-statistic coordinates.

    The implementation never forms a dense inverse and never applies an
    ``r x r`` triangular solve with ``r`` right-hand sides.  Every transformed
    outer product is represented by two vector triangular solves, so the work
    is ``O((K+1) r^2 + p r)`` rather than ``O(r^3)``.

    Returns prefix adjoints ``(dG, dM, dC, dq)`` for the state *read* by the
    current token.  These are subsequently converted to write-contribution
    gradients by a reverse exclusive scan.
    """
    if power_k < 0:
        raise ValueError("power_k must be non-negative")
    if L.ndim != 3 or A.shape != L.shape:
        raise ValueError("L and A must have shape (N,r,r)")
    if R.ndim != 3 or q.ndim != 2 or dy.ndim != 2:
        raise ValueError("R/q/dy must have shapes (N,p,r)/(N,r)/(N,p)")

    # Whitened query trajectory U_j = A^j L^{-1} q.
    u = [torch.linalg.solve_triangular(L, q.unsqueeze(-1), upper=False)]
    for _ in range(power_k):
        u.append(A @ u[-1])

    # Convert a whitened vector a to raw-coordinate covector L^{-T} a.
    def raw_covector(a: torch.Tensor) -> torch.Tensor:
        return torch.linalg.solve_triangular(
            L.transpose(-1, -2), a, upper=True
        )

    # Raw-coordinate query trajectory.  Vector solves keep this quadratic.
    xu = [raw_covector(ui) for ui in u]
    dy_col = dy.unsqueeze(-1)
    adj = R.transpose(-1, -2) @ dy_col

    dM = torch.zeros_like(A)
    dG = torch.zeros_like(A)
    for j in range(power_k, 0, -1):
        a = raw_covector(adj)
        dM = dM + a @ xu[j - 1].transpose(-1, -2)
        dG = dG - a @ xu[j].transpose(-1, -2)
        adj = A.transpose(-1, -2) @ adj

    dq_col = raw_covector(adj)
    dG = dG - dq_col @ xu[0].transpose(-1, -2)
    dG = 0.5 * (dG + dG.transpose(-1, -2))
    dC = dy_col @ xu[power_k].transpose(-1, -2)
    return dG, dM, dC, dq_col.squeeze(-1)


@torch.no_grad()
def _advance_whitened_state(
    L: torch.Tensor,
    A: torch.Tensor,
    R: torch.Tensor,
    hprev: torch.Tensor,
    hasprev: torch.Tensor,
    x: torch.Tensor,
    vbar: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One exact write to the compact state, with quadratic work."""
    L_new, c, s, u = _rank1_cholesky_update(L, x)
    v_prev = _transport_vector(hprev, c, s)
    A_new = _transport_operator(A, c, s)
    A_new = A_new + (
        hasprev.to(A.dtype).unsqueeze(-1).unsqueeze(-1)
        * (u.unsqueeze(-1) @ v_prev.unsqueeze(-2))
    )
    R_new = _transport_readout(R, c, s) + vbar.unsqueeze(-1) @ u.unsqueeze(-2)
    return L_new, A_new, R_new, u, torch.ones_like(hasprev)


@torch.no_grad()
def _rank1_cholesky_downdate(
    L_plus: torch.Tensor,
    x: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Undo ``L_plus L_plus^T = L_prev L_prev^T + x x^T`` in O(r^2).

    The returned ``c,s`` are the *same forward Givens coefficients* that would
    update ``L_prev`` by ``x``.  ``u`` is therefore ``L_plus^{-1} x`` and also
    the final column of the rotation product.  Ridge regularization guarantees
    that every downdated matrix remains positive definite for a valid stream.
    """
    if L_plus.ndim != 3 or x.ndim != 2:
        raise ValueError("L_plus/x must have shapes (N,r,r)/(N,r)")
    N, r, _ = L_plus.shape
    L = L_plus.clone()
    z = x.clone()
    c_all = torch.empty(N, r, device=L.device, dtype=L.dtype)
    s_all = torch.empty_like(c_all)
    u = torch.zeros_like(c_all)
    e = torch.ones(N, device=L.device, dtype=L.dtype)

    for k in range(r):
        a = L[:, k, k]
        b = z[:, k]
        radicand = a * a - b * b
        # A materially negative radicand means the supplied state/key pair is
        # inconsistent, not merely roundoff.  Keep the reference fail-fast.
        tolerance = 64.0 * torch.finfo(L.dtype).eps * (a * a).clamp_min(1.0)
        if bool((radicand < -tolerance).any()):
            bad = int((radicand < -tolerance).sum().item())
            raise RuntimeError(f"invalid Cholesky downdate in {bad} states")
        rho = torch.sqrt(radicand.clamp_min(0.0))
        c = rho / a.clamp_min(torch.finfo(L.dtype).tiny)
        s = b / a.clamp_min(torch.finfo(L.dtype).tiny)
        c_all[:, k] = c
        s_all[:, k] = s
        u[:, k] = s * e
        e = c * e
        L[:, k, k] = rho

        if k + 1 < r:
            old_tail = L[:, k + 1 :, k].clone()
            z_tail = z[:, k + 1 :].clone()
            c_safe = c.clamp_min(torch.finfo(L.dtype).tiny).unsqueeze(-1)
            new_tail = (old_tail - s.unsqueeze(-1) * z_tail) / c_safe
            L[:, k + 1 :, k] = new_tail
            z[:, k + 1 :] = c.unsqueeze(-1) * z_tail - s.unsqueeze(-1) * new_tail
    return L, c_all, s_all, u


def _inverse_transport_vector(
    y: torch.Tensor,
    c: torch.Tensor,
    s: torch.Tensor,
    u: torch.Tensor,
) -> torch.Tensor:
    """Apply ``T^{-1}`` to vectors when ``y=T x`` using inverse replay.

    Forward replay maps ``[x;0]`` to ``[T x; p]``.  The discarded coordinate
    can be reconstructed from orthogonality as

        p = -u^T y / prod_k c_k,

    where ``u`` is the top-right column of the rotation product.  Replaying the
    rotations in reverse then recovers ``x`` in O(r).
    """
    e = c.prod(dim=-1)
    tiny = torch.finfo(y.dtype).tiny
    pad = -(u * y).sum(dim=-1) / e.clamp_min(tiny)
    out = y.clone()
    for k in range(y.shape[-1] - 1, -1, -1):
        a = out[:, k].clone()
        b = pad.clone()
        out[:, k] = c[:, k] * a - s[:, k] * b
        pad = s[:, k] * a + c[:, k] * b
    return out


def _inverse_transport_left(
    Y: torch.Tensor,
    c: torch.Tensor,
    s: torch.Tensor,
    u: torch.Tensor,
) -> torch.Tensor:
    """Apply ``T^{-1}`` to every column of ``Y`` in O(r * columns)."""
    e = c.prod(dim=-1)
    tiny = torch.finfo(Y.dtype).tiny
    pad = -(u.unsqueeze(-1) * Y).sum(dim=1) / e.clamp_min(tiny).unsqueeze(-1)
    out = Y.clone()
    for k in range(Y.shape[-2] - 1, -1, -1):
        a = out[:, k, :].clone()
        b = pad.clone()
        out[:, k, :] = c[:, k].unsqueeze(-1) * a - s[:, k].unsqueeze(-1) * b
        pad = s[:, k].unsqueeze(-1) * a + c[:, k].unsqueeze(-1) * b
    return out


def _inverse_transport_right_transpose(
    Y: torch.Tensor,
    c: torch.Tensor,
    s: torch.Tensor,
    u: torch.Tensor,
) -> torch.Tensor:
    """Apply ``T^{-T}`` on the right of a batched matrix in O(rows * r)."""
    e = c.prod(dim=-1)
    tiny = torch.finfo(Y.dtype).tiny
    pad = -(Y * u.unsqueeze(-2)).sum(dim=-1) / e.clamp_min(tiny).unsqueeze(-1)
    out = Y.clone()
    for k in range(Y.shape[-1] - 1, -1, -1):
        a = out[:, :, k].clone()
        b = pad.clone()
        out[:, :, k] = c[:, k].unsqueeze(-1) * a - s[:, k].unsqueeze(-1) * b
        pad = s[:, k].unsqueeze(-1) * a + c[:, k].unsqueeze(-1) * b
    return out


def _inverse_transport_operator(
    A_core: torch.Tensor,
    c: torch.Tensor,
    s: torch.Tensor,
    u: torch.Tensor,
) -> torch.Tensor:
    """Recover ``A`` from ``A_core=T A T^T`` with quadratic work."""
    left = _inverse_transport_left(A_core, c, s, u)
    return _inverse_transport_right_transpose(left, c, s, u)


def _inverse_transport_readout(
    R_core: torch.Tensor,
    c: torch.Tensor,
    s: torch.Tensor,
    u: torch.Tensor,
) -> torch.Tensor:
    """Recover ``R`` from ``R_core=R T^T`` with O(p r) work."""
    return _inverse_transport_right_transpose(R_core, c, s, u)


@torch.no_grad()
def _retreat_whitened_state(
    L_plus: torch.Tensor,
    A_plus: torch.Tensor,
    R_plus: torch.Tensor,
    h_after: torch.Tensor,
    x: torch.Tensor,
    vbar: torch.Tensor,
    x_prev: torch.Tensor,
    has_prev: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Undo one compact write exactly, using a Cholesky downdate.

    This reversibility is what makes the memory-efficient CUDA backward clean:
    a block can be traversed backward from its final compact state without
    storing every intermediate matrix.
    """
    L_prev, c, s, u_rot = _rank1_cholesky_downdate(L_plus, x)
    w = torch.linalg.solve_triangular(
        L_plus, x_prev.unsqueeze(-1), upper=False
    ).squeeze(-1)
    mask = has_prev.to(A_plus.dtype).unsqueeze(-1).unsqueeze(-1)
    A_core = A_plus - mask * (h_after.unsqueeze(-1) @ w.unsqueeze(-2))
    R_core = R_plus - vbar.unsqueeze(-1) @ h_after.unsqueeze(-2)
    A_prev = _inverse_transport_operator(A_core, c, s, u_rot)
    R_prev = _inverse_transport_readout(R_core, c, s, u_rot)
    h_prev = _inverse_transport_vector(w, c, s, u_rot)
    h_prev = torch.where(has_prev.unsqueeze(-1), h_prev, torch.zeros_like(h_prev))
    return L_prev, A_prev, R_prev, h_prev, has_prev


@torch.no_grad()
def _blocked_backward_boundary_states(
    x: torch.Tensor,
    vbar: torch.Tensor,
    ridge: float,
    block_size: int,
    jitter: float,
):
    """Build exact compact checkpoints at every block boundary."""
    G0, M0, C0, prev_x, has_prev, padded_T = block_prefix_statistics(
        x, vbar, block_size
    )
    L_factor = block_prefix_cholesky_factor_scan(x, block_size, ridge)
    L0, A0, R0, h0, hp0 = boundary_whitened_states(
        G0, M0, C0, prev_x, has_prev, ridge=ridge, jitter=jitter,
        L_precomputed=L_factor,
    )
    return L0, A0, R0, h0, hp0, padded_T


@torch.no_grad()
def _analytic_backward_blocked(
    x4: torch.Tensor,
    q4: torch.Tensor,
    v4: torch.Tensor,
    dy4: torch.Tensor,
    ridge: float,
    power_k: int,
    block_size: int,
    jitter: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact reversible two-level reverse scan with quadratic work.

    The algorithm stores one compact checkpoint per block.  It then uses the
    rank-one Cholesky downdate and inverse Givens replay to traverse each block
    backward from its terminal state; no per-token matrix trajectory is saved.

    Local work consists of:

    * one forward pass to obtain terminal block states;
    * one reverse pass to aggregate prefix adjoints per block;
    * an exclusive reverse scan of those aggregates;
    * one reverse pass to emit token gradients.

    Persistent matrix memory is ``O((T/S)(2r^2+pr))`` and total work is
    ``O(T(r^2+pr))`` for fixed ``power_k``.
    """
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    B, T, H, r = x4.shape
    p = v4.shape[-1]
    x = _flatten_heads(x4)
    q = _flatten_heads(q4)
    v = _flatten_heads(v4)
    dy = _flatten_heads(dy4)
    N = x.shape[0]

    L0, A0, R0, h0, hp0, padded_T = _blocked_backward_boundary_states(
        x, v, ridge, block_size, jitter
    )
    nb = padded_T // block_size
    states = N * nb
    pad = padded_T - T
    if pad:
        xp = F.pad(x, (0, 0, 0, pad))
        qp = F.pad(q, (0, 0, 0, pad))
        vp = F.pad(v, (0, 0, 0, pad))
        dyp = F.pad(dy, (0, 0, 0, pad))
    else:
        xp, qp, vp, dyp = x, q, v, dy

    prevp = torch.zeros_like(xp)
    if padded_T > 1:
        prevp[:, 1:] = xp[:, :-1]
    token_index = torch.arange(padded_T, device=x.device)
    valid_seq = token_index < T
    has_prev_seq = valid_seq & (token_index > 0)

    xb = xp.reshape(N, nb, block_size, r).reshape(states, block_size, r)
    qb = qp.reshape(N, nb, block_size, r).reshape(states, block_size, r)
    vb = vp.reshape(N, nb, block_size, p).reshape(states, block_size, p)
    dyb = dyp.reshape(N, nb, block_size, p).reshape(states, block_size, p)
    pb = prevp.reshape(N, nb, block_size, r).reshape(states, block_size, r)
    valid = valid_seq.reshape(1, nb, block_size).expand(N, -1, -1)
    valid = valid.reshape(states, block_size)
    has_prev = has_prev_seq.reshape(1, nb, block_size).expand(N, -1, -1)
    has_prev = has_prev.reshape(states, block_size)

    # ------------------------------------------------------------------
    # Forward checkpoint pass: exact final compact state of every block.
    # ------------------------------------------------------------------
    L_end = L0.reshape(states, r, r).clone()
    A_end = A0.reshape(states, r, r).clone()
    R_end = R0.reshape(states, p, r).clone()
    h_end = h0.reshape(states, r).clone()
    hp_end = hp0.reshape(states).clone()
    for j in range(block_size):
        mask = valid[:, j]
        mm = mask.unsqueeze(-1).unsqueeze(-1)
        vm = mask.unsqueeze(-1)
        Ln, An, Rn, hn, hpn = _advance_whitened_state(
            L_end, A_end, R_end, h_end, hp_end, xb[:, j], vb[:, j]
        )
        L_end = torch.where(mm, Ln, L_end)
        A_end = torch.where(mm, An, A_end)
        R_end = torch.where(mm, Rn, R_end)
        h_end = torch.where(vm, hn, h_end)
        hp_end = torch.where(mask, hpn, hp_end)

    # ------------------------------------------------------------------
    # Reverse aggregate pass: one prefix-adjoint sum per block.
    # ------------------------------------------------------------------
    Lr = L_end.clone()
    Ar = A_end.clone()
    Rr = R_end.clone()
    hr = h_end.clone()
    hpr = hp_end.clone()
    sum_g = torch.zeros(states, r, r, device=x.device, dtype=x.dtype)
    sum_m = torch.zeros_like(sum_g)
    sum_c = torch.zeros(states, p, r, device=x.device, dtype=x.dtype)

    for j in range(block_size - 1, -1, -1):
        mask = valid[:, j]
        mm = mask.unsqueeze(-1).unsqueeze(-1)
        vm = mask.unsqueeze(-1)
        Lb, Ab, Rb, hb, hpb = _retreat_whitened_state(
            Lr, Ar, Rr, hr, xb[:, j], vb[:, j], pb[:, j], has_prev[:, j]
        )
        dg, dm, dc, _ = _prefix_adjoint_from_state(
            Lb, Ab, Rb, qb[:, j], dyb[:, j], power_k
        )
        mf = mask.to(x.dtype).unsqueeze(-1).unsqueeze(-1)
        sum_g = sum_g + mf * dg
        sum_m = sum_m + mf * dm
        sum_c = sum_c + mf * dc
        Lr = torch.where(mm, Lb, Lr)
        Ar = torch.where(mm, Ab, Ar)
        Rr = torch.where(mm, Rb, Rr)
        hr = torch.where(vm, hb, hr)
        hpr = torch.where(mask, hpb, hpr)

    sum_g = sum_g.reshape(N, nb, r, r)
    sum_m = sum_m.reshape(N, nb, r, r)
    sum_c = sum_c.reshape(N, nb, p, r)
    future_g = _future_exclusive_sum(sum_g, dim=1).reshape(states, r, r)
    future_m = _future_exclusive_sum(sum_m, dim=1).reshape(states, r, r)
    future_c = _future_exclusive_sum(sum_c, dim=1).reshape(states, p, r)

    # ------------------------------------------------------------------
    # Reverse emission pass.  The current suffix accumulator is exactly the
    # adjoint of the contribution written at token i because the state is
    # exclusive: write i is visible only to reads t>i.
    # ------------------------------------------------------------------
    Lr = L_end.clone()
    Ar = A_end.clone()
    Rr = R_end.clone()
    hr = h_end.clone()
    hpr = hp_end.clone()
    acc_g = future_g.clone()
    acc_m = future_m.clone()
    acc_c = future_c.clone()

    dx_current = torch.zeros(states, block_size, r, device=x.device, dtype=x.dtype)
    dx_previous = torch.zeros_like(dx_current)
    dq_out = torch.zeros_like(dx_current)
    dv_out = torch.zeros(states, block_size, p, device=x.device, dtype=x.dtype)

    for j in range(block_size - 1, -1, -1):
        mask = valid[:, j]
        mm = mask.unsqueeze(-1).unsqueeze(-1)
        vm = mask.unsqueeze(-1)
        mf = mask.to(x.dtype)

        Lb, Ab, Rb, hb, hpb = _retreat_whitened_state(
            Lr, Ar, Rr, hr, xb[:, j], vb[:, j], pb[:, j], has_prev[:, j]
        )
        dg, dm, dc, dqi = _prefix_adjoint_from_state(
            Lb, Ab, Rb, qb[:, j], dyb[:, j], power_k
        )

        xi = xb[:, j]
        vi = vb[:, j]
        grad_x = (
            (acc_g + acc_g.transpose(-1, -2)) @ xi.unsqueeze(-1)
        ).squeeze(-1)
        grad_x = grad_x + (
            acc_c.transpose(-1, -2) @ vi.unsqueeze(-1)
        ).squeeze(-1)
        grad_x = grad_x + (
            acc_m @ pb[:, j].unsqueeze(-1)
        ).squeeze(-1) * has_prev[:, j].to(x.dtype).unsqueeze(-1)
        grad_prev = (
            acc_m.transpose(-1, -2) @ xi.unsqueeze(-1)
        ).squeeze(-1) * has_prev[:, j].to(x.dtype).unsqueeze(-1)
        grad_v = (acc_c @ xi.unsqueeze(-1)).squeeze(-1)

        dx_current[:, j] = mf.unsqueeze(-1) * grad_x
        dx_previous[:, j] = mf.unsqueeze(-1) * grad_prev
        dq_out[:, j] = mf.unsqueeze(-1) * dqi
        dv_out[:, j] = mf.unsqueeze(-1) * grad_v

        acc_g = acc_g + mf.unsqueeze(-1).unsqueeze(-1) * dg
        acc_m = acc_m + mf.unsqueeze(-1).unsqueeze(-1) * dm
        acc_c = acc_c + mf.unsqueeze(-1).unsqueeze(-1) * dc
        Lr = torch.where(mm, Lb, Lr)
        Ar = torch.where(mm, Ab, Ar)
        Rr = torch.where(mm, Rb, Rr)
        hr = torch.where(vm, hb, hr)
        hpr = torch.where(mask, hpb, hpr)

    dx = dx_current.reshape(N, nb, block_size, r).reshape(N, padded_T, r)
    right = dx_previous.reshape(N, nb, block_size, r).reshape(N, padded_T, r)
    if padded_T > 1:
        dx[:, :-1] = dx[:, :-1] + right[:, 1:]
    dq = dq_out.reshape(N, nb, block_size, r).reshape(N, padded_T, r)
    dv = dv_out.reshape(N, nb, block_size, p).reshape(N, padded_T, p)

    return (
        _unflatten_heads(dx[:, :T], B, H),
        _unflatten_heads(dq[:, :T], B, H),
        _unflatten_heads(dv[:, :T], B, H),
    )


@torch.no_grad()
def _analytic_backward_dense_reference(
    x4: torch.Tensor,
    q4: torch.Tensor,
    v4: torch.Tensor,
    dy4: torch.Tensor,
    ridge: float,
    power_k: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Old per-prefix oracle retained only for tests and diagnostics."""
    B, T, H, r = x4.shape
    x = _flatten_heads(x4)
    q = _flatten_heads(q4)
    v = _flatten_heads(v4)
    dy = _flatten_heads(dy4).unsqueeze(-1)

    G, M, C, qcol = _exact_prefix_tensors(x, q, v, ridge)
    L = _stable_cholesky(0.5 * (G + G.transpose(-1, -2)))
    eye = torch.eye(r, device=x.device, dtype=x.dtype)
    Pinv = torch.linalg.solve_triangular(
        L, eye.expand(*L.shape[:-2], r, r), upper=False
    )
    Pt = Pinv.transpose(-1, -2)
    W = Pinv @ M @ Pt

    U = [Pinv @ qcol]
    for _ in range(power_k):
        U.append(W @ U[-1])

    XK = Pt @ U[power_k]
    dC_prefix = dy @ XK.transpose(-1, -2)
    adj = Pinv @ (C.transpose(-1, -2) @ dy)
    dMw = torch.zeros_like(W)
    dGw = torch.zeros_like(W)
    for i in range(power_k, 0, -1):
        dMw = dMw + adj @ U[i - 1].transpose(-1, -2)
        dGw = dGw - adj @ U[i].transpose(-1, -2)
        adj = W.transpose(-1, -2) @ adj
    dGw = dGw - adj @ U[0].transpose(-1, -2)

    dq = (Pt @ adj).squeeze(-1)
    dM_prefix = Pt @ dMw @ Pinv
    dG_prefix = Pt @ dGw @ Pinv
    dG_prefix = 0.5 * (dG_prefix + dG_prefix.transpose(-1, -2))
    dG_contrib = _future_exclusive_sum(dG_prefix, dim=1)
    dM_contrib = _future_exclusive_sum(dM_prefix, dim=1)
    dC_contrib = _future_exclusive_sum(dC_prefix, dim=1)

    dx = ((dG_contrib + dG_contrib.transpose(-1, -2)) @ x.unsqueeze(-1)).squeeze(-1)
    dv = (dC_contrib @ x.unsqueeze(-1)).squeeze(-1)
    dx = dx + (dC_contrib.transpose(-1, -2) @ v.unsqueeze(-1)).squeeze(-1)
    if T > 1:
        dm = dM_contrib[:, 1:]
        dx[:, 1:] = dx[:, 1:] + (dm @ x[:, :-1].unsqueeze(-1)).squeeze(-1)
        dx[:, :-1] = dx[:, :-1] + (
            dm.transpose(-1, -2) @ x[:, 1:].unsqueeze(-1)
        ).squeeze(-1)
    return (
        _unflatten_heads(dx, B, H),
        _unflatten_heads(dq, B, H),
        _unflatten_heads(dv, B, H),
    )


class _SKAPrefixScanFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        q: torch.Tensor,
        vbar: torch.Tensor,
        ridge: float,
        power_k: int,
        block_size: int,
        jitter: float,
    ) -> torch.Tensor:
        if x.dtype not in (torch.float32, torch.float64):
            raise TypeError("prefix-scan core expects FP32 or FP64 tensors")
        y = _prefix_scan_forward_reference(
            x, q, vbar, float(ridge), int(power_k), int(block_size), float(jitter)
        )
        ctx.save_for_backward(x, q, vbar)
        ctx.ridge = float(ridge)
        ctx.power_k = int(power_k)
        ctx.block_size = int(block_size)
        ctx.jitter = float(jitter)
        return y

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        x, q, vbar = ctx.saved_tensors
        dx, dq, dv = _analytic_backward_blocked(
            x, q, vbar, dy, ctx.ridge, ctx.power_k,
            ctx.block_size, ctx.jitter,
        )
        return dx, dq, dv, None, None, None, None


_WARNED_GEOMETRIES: set = set()


def _geometry_note(x, q, vbar, power_k, block_size, jitter) -> str:
    """Which predicate actually failed, named. A warning that says only "falling
    back" sends the reader to the source; this sends them to the fix."""
    return (f"dtype={x.dtype}, cuda={x.is_cuda}, rank={x.shape[-1]}, "
            f"value_width={vbar.shape[-1]}, power_k={int(power_k)}, "
            f"block_size={int(block_size)}, jitter={float(jitter)}")


def _warn_reference_fallback(x, q, vbar, power_k, block_size, jitter) -> None:
    import warnings

    # Only a DEGRADATION is worth a warning. On CPU the reference scan is not a
    # fallback, it is the only implementation -- the fused kernel could never
    # have run. Warning there was pure noise: it fired 19 times across the unit
    # suite on 4x3 toy tensors, which is exactly how a useful warning gets
    # filtered out and then ignored when it matters.
    if not x.is_cuda:
        return

    key = (str(x.dtype), bool(x.is_cuda), int(x.shape[-1]), int(vbar.shape[-1]),
           int(power_k), int(block_size), float(jitter))
    if key in _WARNED_GEOMETRIES:
        return
    _WARNED_GEOMETRIES.add(key)
    warnings.warn(
        "ska_prefix_scan(backend='auto'): the fused CUDA kernel does not accept "
        "this geometry, so the PURE-PYTHON REFERENCE scan is running. It is "
        "exact but roughly two orders of magnitude slower (measured 137x on an "
        "H100: 4.49s vs 0.033s per fwd+bwd micro-step), because it issues ~1e5 "
        "tiny kernel launches from rank-long Python loops -- the symptom is 99% "
        "CPU with an idle GPU. The fused kernel requires FP32 CUDA, rank=24, "
        f"value width=64, power_k=1, block_size=32, jitter=0; got "
        f"{_geometry_note(x, q, vbar, power_k, block_size, jitter)}. Value width "
        "is d_model/ska_n_heads, so it cannot be fixed by changing ska_rank. "
        "Pass backend='reference' to accept this deliberately, or see "
        "ska_inverse_cholesky for a different exact route.",
        RuntimeWarning, stacklevel=3)


def ska_prefix_scan(
    x: torch.Tensor,
    q: torch.Tensor,
    vbar: torch.Tensor,
    ridge: float,
    power_k: int = 1,
    block_size: int = 32,
    jitter: float = 0.0,
    backend: str = "auto",
) -> torch.Tensor:
    """Exact, trainable SKA prefix scan.

    ``backend="cuda_prefix"`` selects the rank-24/value-64 fused CUDA
    forward and analytic backward. It is strict and never silently falls back.
    ``backend="auto"`` uses CUDA when the production geometry matches and
    otherwise uses this module's correctness implementation.
    """
    backend = str(backend).lower()
    allowed = {"auto", "cuda", "cuda_prefix", "reference", "pytorch"}
    if backend not in allowed:
        raise ValueError(f"unknown prefix-scan backend: {backend!r}")

    if backend in {"auto", "cuda", "cuda_prefix"}:
        from koopman_lm.kernels.cuda_prefix_scan import (
            fused_ska_prefix_scan,
            is_supported as cuda_prefix_supported,
        )
        supported = (
            float(jitter) == 0.0
            and cuda_prefix_supported(
                x, q, vbar, power_k=int(power_k), block_size=int(block_size)
            )
        )
        if supported:
            return fused_ska_prefix_scan(
                x, q, vbar, float(ridge), int(power_k), int(block_size)
            )
        if backend in {"cuda", "cuda_prefix"}:
            raise RuntimeError(
                "cuda_prefix requires FP32 CUDA tensors, rank=24, value "
                f"width=64, power_k=1, block_size=32, and jitter=0. Got "
                f"{_geometry_note(x, q, vbar, power_k, block_size, jitter)}"
            )
        # backend="auto" reaches here, and used to fall through in SILENCE.
        #
        # That silence cost two hours. A study on a base spec whose value width
        # is 32 (d_model 128 / 4 heads) can never satisfy the fused kernel, so
        # every trial took the reference path at ~137x -- 4.49 s versus 0.033 s
        # per micro-step on an H100, measured. Nothing said so: `cuda` and
        # `cuda_prefix` get a precise RuntimeError, `auto` got nothing, and the
        # reverse substitution is the one that is expensive.
        #
        # Warned once per distinct geometry rather than per call: this is on the
        # forward path, and a per-step warning would be its own denial of
        # service. warnings' default "once per location" dedup is not enough,
        # since one call site serves every geometry.
        _warn_reference_fallback(x, q, vbar, power_k, block_size, jitter)

    return _SKAPrefixScanFn.apply(
        x, q, vbar, float(ridge), int(power_k), int(block_size), float(jitter)
    )


# ---------------------------------------------------------------------------
# Dense mathematical oracle used by tests
# ---------------------------------------------------------------------------


def dense_exact_oracle(
    x: torch.Tensor,
    q: torch.Tensor,
    vbar: torch.Tensor,
    ridge: float,
    power_k: int = 1,
) -> torch.Tensor:
    """Simple differentiable per-prefix formula; never use as a production path."""
    B, T, H, r = x.shape
    p = vbar.shape[-1]
    xf = _flatten_heads(x)
    qf = _flatten_heads(q)
    vf = _flatten_heads(vbar)
    G, M, C, qcol = _exact_prefix_tensors_autograd(xf, qf, vf, ridge)
    L = torch.linalg.cholesky(0.5 * (G + G.transpose(-1, -2)))
    LiM = torch.linalg.solve_triangular(L, M, upper=False)
    A = torch.linalg.solve_triangular(
        L, LiM.transpose(-1, -2), upper=False
    ).transpose(-1, -2)
    u = torch.linalg.solve_triangular(L, qcol, upper=False)
    for _ in range(power_k):
        u = A @ u
    xk = torch.linalg.solve_triangular(
        L.transpose(-1, -2), u, upper=True
    )
    y = (C @ xk).squeeze(-1)
    return _unflatten_heads(y, B, H)


def _exact_prefix_tensors_autograd(
    x: torch.Tensor,
    q: torch.Tensor,
    vbar: torch.Tensor,
    ridge: float,
):
    """Autograd-enabled counterpart of :func:`_exact_prefix_tensors`."""
    N, T, r = x.shape
    g = x.unsqueeze(-1) @ x.unsqueeze(-2)
    zero_m = torch.zeros(N, 1, r, r, device=x.device, dtype=x.dtype)
    if T > 1:
        tail_m = x[:, 1:].unsqueeze(-1) @ x[:, :-1].unsqueeze(-2)
        m = torch.cat([zero_m, tail_m], dim=1)
    else:
        m = zero_m
    c = vbar.unsqueeze(-1) @ x.unsqueeze(-2)
    eye = torch.eye(r, device=x.device, dtype=x.dtype)
    G = exclusive_cumsum(g, dim=1) + ridge * eye
    M = exclusive_cumsum(m, dim=1)
    C = exclusive_cumsum(c, dim=1)
    return G, M, C, q.unsqueeze(-1)
