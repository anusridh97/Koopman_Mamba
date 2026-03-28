"""
ska.py -- Structured Kernel Attention with chunk-causal masking.

Two backends, selected automatically:
  - 'triton':  Fused Triton kernel for post-Cholesky matmul chain
  - 'pytorch': Batched PyTorch (always available)

Both share the same batched statistics + cumsum + cholesky_ex path.
The Triton backend fuses the post-Cholesky matmul chain (steps 5-7)
into a single kernel to eliminate HBM round-trips.

Three chunk strategies (selected via chunk_strategy):
  - 'standard': non-overlapping chunks, uniform prefix-sum (original)
  - 'overlap':  overlapping chunks with dedup weighting
  - 'decay':    exponentially-decayed prefix-sum
See koopman_lm/adaptive_chunking.py for details.
"""

import math
import torch
import torch.nn as nn
from contextlib import nullcontext

# ============================================================================
# Backend detection
# ============================================================================

_TRITON_AVAILABLE = False
try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except ImportError:
    pass


# ============================================================================
# Shared utilities
# ============================================================================

def _spectral_normalize_power_iter(A, n_iters=6):
    """
    Spectral normalization via power iteration. Batched over leading dims.

    Uses straight-through estimation: the scale factor is treated as a
    constant during backprop (power iteration is inside no_grad).
    This is standard practice (cf. spectral norm for GANs).
    """
    v = torch.ones(*A.shape[:-1], 1, device=A.device, dtype=A.dtype) / math.sqrt(A.shape[-1])
    with torch.no_grad():
        for _ in range(n_iters):
            Av = A @ v
            u = Av / Av.norm(dim=-2, keepdim=True).clamp(min=1e-8)
            Atu = A.transpose(-1, -2) @ u
            v = Atu / Atu.norm(dim=-2, keepdim=True).clamp(min=1e-8)
        sigma_max = (A @ v).norm(dim=-2, keepdim=False).squeeze(-1)
    scale = torch.clamp(sigma_max, min=1.0).unsqueeze(-1).unsqueeze(-1)
    return A / scale, sigma_max


def _power_spectral_filter(A_w, w_q, power_K=2):
    """Apply A_w^power_K @ w_q by iterating on w_q (cheaper than building A_w^K)."""
    result = w_q
    for _ in range(power_K):
        result = A_w @ result
    return result


# ============================================================================
# Shared: batched statistics + cumsum + cholesky (STANDARD strategy)
# ============================================================================

def _compute_chunk_stats_and_cholesky(z_f, zq_f, v_f, r, H, P, CS, ridge_eps):
    """
    Shared prefix for both backends:
      1. Pad + reshape into chunks
      2. Per-chunk statistics (parallel einsums)
      3. Exclusive prefix sum (causal)
      4. Batched cholesky_ex

    Returns: L_flat, M_flat, C_flat, zq_flat, shapes
    """
    B, T = z_f.shape[:2]
    device = z_f.device
    dtype = z_f.dtype

    n_chunks = (T + CS - 1) // CS
    T_padded = n_chunks * CS
    pad_len = T_padded - T

    if pad_len > 0:
        z_f = torch.nn.functional.pad(z_f, (0, 0, 0, 0, 0, pad_len))
        zq_f = torch.nn.functional.pad(zq_f, (0, 0, 0, 0, 0, pad_len))
        v_f = torch.nn.functional.pad(v_f, (0, 0, 0, 0, 0, pad_len))

    C = n_chunks
    z_c = z_f.reshape(B, C, CS, H, r)
    zq_c = zq_f.reshape(B, C, CS, H, r)
    v_c = v_f.reshape(B, C, CS, H, P)

    # ---- Per-chunk statistics (all parallel) ----
    G_chunks = torch.einsum('bcthr,bcths->bchrs', z_c, z_c)          # [B, C, H, r, r]
    M_chunks = torch.einsum('bcthr,bcths->bchrs',
                            z_c[:, :, 1:], z_c[:, :, :-1])           # [B, C, H, r, r]
    C_chunks = torch.einsum('bcthp,bcthr->bchpr', v_c, z_c)          # [B, C, H, P, r]

    # ---- Cross-chunk boundary transitions ----
    if C > 1:
        M_boundary = torch.einsum('bchr,bchs->bchrs',
                                  z_c[:, 1:, 0],        # first token of chunks 1..C-1
                                  z_c[:, :-1, -1])       # last token of chunks 0..C-2

    # ---- Exclusive prefix sum (causal accumulation) ----
    eye_r = torch.eye(r, device=device, dtype=dtype)

    G_cumsum = torch.cumsum(G_chunks, dim=1)
    G_excl = torch.zeros_like(G_cumsum)
    G_excl[:, 1:] = G_cumsum[:, :-1]
    G_excl = G_excl + ridge_eps * eye_r

    M_cumsum = torch.cumsum(M_chunks, dim=1)
    M_excl = torch.zeros_like(M_cumsum)
    M_excl[:, 1:] = M_cumsum[:, :-1]

    if C > 1:
        M_bnd_full = torch.zeros(B, C, H, r, r, device=device, dtype=dtype)
        M_bnd_full[:, 1:] = M_boundary
        M_bnd_inclusive = torch.cumsum(M_bnd_full, dim=1)
        M_excl = M_excl + M_bnd_inclusive

    C_cumsum = torch.cumsum(C_chunks, dim=1)
    C_excl = torch.zeros_like(C_cumsum)
    C_excl[:, 1:] = C_cumsum[:, :-1]

    # ---- Batched Cholesky ----
    BCH = B * C * H
    G_flat = G_excl.reshape(BCH, r, r)
    G_flat = 0.5 * (G_flat + G_flat.transpose(-1, -2))  # symmetrize

    L_flat, info = torch.linalg.cholesky_ex(G_flat)

    G_jittered = G_flat + 1e-4 * eye_r.unsqueeze(0)
    L_jittered, _ = torch.linalg.cholesky_ex(G_jittered)
    needs_fix = (info > 0).unsqueeze(-1).unsqueeze(-1)  # [BCH, 1, 1]
    L_flat = torch.where(needs_fix, L_jittered, L_flat)

    M_flat = M_excl.reshape(BCH, r, r)
    C_flat = C_excl.reshape(BCH, P, r)
    zq_flat = zq_c.permute(0, 1, 3, 4, 2).reshape(BCH, r, CS)

    return L_flat, M_flat, C_flat, zq_flat, (B, C, H, P, CS, T, T_padded, pad_len)


# ============================================================================
# Phase 1: Batched PyTorch post-Cholesky
# ============================================================================

def _post_cholesky_pytorch(L_flat, M_flat, C_flat, zq_flat, ssn_gamma,
                           power_K, shapes):
    """Post-Cholesky operations using batched PyTorch calls."""
    B, C, H, P, CS, T, T_padded, pad_len = shapes
    r = L_flat.shape[-1]

    # A_w = M @ G^{-1}
    Aw_T = torch.cholesky_solve(M_flat.transpose(-1, -2), L_flat)
    A_w = Aw_T.transpose(-1, -2)
    A_w, _ = _spectral_normalize_power_iter(A_w)
    gamma_safe = torch.clamp(ssn_gamma, min=1.0, max=1.5)
    A_w = A_w * gamma_safe

    # B_v = C_v @ G^{-1}
    Bv_T = torch.cholesky_solve(C_flat.transpose(-1, -2), L_flat)
    B_v = Bv_T.transpose(-1, -2)

    # w_q = L^{-1} @ zq
    w_q = torch.linalg.solve_triangular(L_flat, zq_flat, upper=False)

    # Spectral filter + project back + readout
    w_f = _power_spectral_filter(A_w, w_q, power_K)
    z_out = L_flat @ w_f
    y_flat = B_v @ z_out

    y_hat = y_flat.reshape(B, C, H, P, CS).permute(0, 1, 4, 2, 3)
    y_hat = y_hat.reshape(B, T_padded, H, P)
    if pad_len > 0:
        y_hat = y_hat[:, :T]
    return y_hat


# ============================================================================
# Phase 2: Triton fused post-Cholesky kernel
# ============================================================================

if _TRITON_AVAILABLE:

    @triton.jit
    def _fused_filter_readout_kernel(
        # Inputs (all [BCH, ...], row-major contiguous, padded to po2)
        Aw_ptr,     # [BCH, R_PAD, R_PAD]
        Bv_ptr,     # [BCH, P_PAD, R_PAD]
        Wq_ptr,     # [BCH, R_PAD, CS_PAD]
        L_ptr,      # [BCH, R_PAD, R_PAD]
        # Output
        Y_ptr,      # [BCH, P_PAD, CS_PAD]
        # Actual (unpadded) dimensions for masking stores
        actual_P: tl.constexpr,
        actual_CS: tl.constexpr,
        # Padded dimensions (power-of-2, used for tl.dot)
        R_PAD: tl.constexpr,
        P_PAD: tl.constexpr,
        CS_PAD: tl.constexpr,
        # Algorithm params
        power_K: tl.constexpr,
    ):
        """
        Fused kernel for steps 5-7 of SKA post-Cholesky chain.
        One program per (batch, chunk, head).
        """
        pid = tl.program_id(0)

        # Compute base offsets
        aw_base = pid * R_PAD * R_PAD
        l_base = pid * R_PAD * R_PAD
        bv_base = pid * P_PAD * R_PAD
        wq_base = pid * R_PAD * CS_PAD
        y_base = pid * P_PAD * CS_PAD

        ri = tl.arange(0, R_PAD)
        rj = tl.arange(0, R_PAD)
        pi = tl.arange(0, P_PAD)
        ci = tl.arange(0, CS_PAD)

        # Load A_w [R_PAD, R_PAD]
        Aw = tl.load(Aw_ptr + aw_base + ri[:, None] * R_PAD + rj[None, :])

        # Load w_q [R_PAD, CS_PAD]
        wq = tl.load(Wq_ptr + wq_base + ri[:, None] * CS_PAD + ci[None, :])

        # Step 5: w_f = A_w^K @ w_q (iterate on w_q)
        w_f = wq
        for _k in range(power_K):
            w_f = tl.dot(Aw, w_f)

        # Load L [R_PAD, R_PAD]
        L = tl.load(L_ptr + l_base + ri[:, None] * R_PAD + rj[None, :])

        # Step 6: z_out = L @ w_f
        z_out = tl.dot(L, w_f)

        # Load B_v [P_PAD, R_PAD]
        Bv = tl.load(Bv_ptr + bv_base + pi[:, None] * R_PAD + rj[None, :])

        # Step 7: y = B_v @ z_out
        y = tl.dot(Bv, z_out)

        # Store y with mask for actual (unpadded) dimensions
        p_mask = pi < actual_P
        c_mask = ci < actual_CS
        mask = p_mask[:, None] & c_mask[None, :]
        tl.store(Y_ptr + y_base + pi[:, None] * CS_PAD + ci[None, :], y, mask=mask)


    def _post_cholesky_triton(L_flat, M_flat, C_flat, zq_flat, ssn_gamma,
                              power_K, shapes):
        """
        Triton-accelerated post-Cholesky path.
        """
        B, C, H, P, CS, T, T_padded, pad_len = shapes
        r = L_flat.shape[-1]
        BCH = L_flat.shape[0]
        device = L_flat.device
        dtype = L_flat.dtype

        # Steps 1-2: A_w via batched cholesky_solve + spectral normalize
        Aw_T = torch.cholesky_solve(M_flat.transpose(-1, -2), L_flat)
        A_w = Aw_T.transpose(-1, -2).contiguous()
        A_w, _ = _spectral_normalize_power_iter(A_w)
        gamma_safe = torch.clamp(ssn_gamma, min=1.0, max=1.5)
        A_w = A_w * gamma_safe

        # Step 3: B_v
        Bv_T = torch.cholesky_solve(C_flat.transpose(-1, -2), L_flat)
        B_v = Bv_T.transpose(-1, -2).contiguous()

        # Step 4: w_q
        w_q = torch.linalg.solve_triangular(L_flat, zq_flat, upper=False)
        w_q = w_q.contiguous()
        L_flat = L_flat.contiguous()

        # Pad to power-of-2 for tl.dot
        R_PAD = triton.next_power_of_2(r)
        P_PAD = triton.next_power_of_2(P)
        CS_PAD = triton.next_power_of_2(CS)

        r_pad = R_PAD - r
        p_pad = P_PAD - P
        cs_pad = CS_PAD - CS

        needs_padding = (r_pad > 0) or (p_pad > 0) or (cs_pad > 0)

        if needs_padding:
            A_w_pad = torch.nn.functional.pad(A_w, (0, r_pad, 0, r_pad))
            L_pad = torch.nn.functional.pad(L_flat, (0, r_pad, 0, r_pad))
            Bv_pad = torch.nn.functional.pad(B_v, (0, r_pad, 0, p_pad))
            wq_pad = torch.nn.functional.pad(w_q, (0, cs_pad, 0, r_pad))
        else:
            A_w_pad = A_w
            L_pad = L_flat
            Bv_pad = B_v
            wq_pad = w_q

        y_pad = torch.empty(BCH, P_PAD, CS_PAD, device=device, dtype=dtype)

        _fused_filter_readout_kernel[(BCH,)](
            A_w_pad, Bv_pad, wq_pad, L_pad, y_pad,
            actual_P=P, actual_CS=CS,
            R_PAD=R_PAD, P_PAD=P_PAD, CS_PAD=CS_PAD,
            power_K=power_K,
        )

        if needs_padding:
            y_flat = y_pad[:, :P, :CS].contiguous()
        else:
            y_flat = y_pad

        y_hat = y_flat.reshape(B, C, H, P, CS).permute(0, 1, 4, 2, 3)
        y_hat = y_hat.reshape(B, T_padded, H, P)
        if pad_len > 0:
            y_hat = y_hat[:, :T]
        return y_hat


# ============================================================================
# SKA Module
# ============================================================================

class SKAModule(nn.Module):
    """
    Structured Kernel Attention with chunk-causal masking.

    The sequence is split into C chunks. For each chunk c, the Koopman
    operator A_w and value readout B_v are estimated using statistics
    from chunks 0..c-1 only (strictly causal).

    Automatically selects the best available backend:
      - 'triton':  Fused Triton kernel for post-Cholesky matmul chain
      - 'pytorch': Batched PyTorch (always available)

    Chunk strategy controls how statistics are accumulated:
      - 'standard':  non-overlapping chunks, uniform prefix-sum (original)
      - 'overlap':   overlapping chunks, dedup-weighted accumulation
      - 'decay':     exponentially-decayed prefix-sum

    Args:
        d_model:            model dimension
        n_heads:            number of attention heads
        rank:               rank of the Koopman operator (default 48)
        head_dim:           per-head dimension (default d_model // n_heads)
        ridge_eps:          ridge regularization for Gram matrix
        scale:              initial eta scaling parameter
        power_K:            number of spectral power iterations for the filter
        chunk_size:         tokens per causal chunk (default 64)
        backend:            'auto', 'pytorch', or 'triton'
        chunk_strategy:     'standard', 'overlap', or 'decay'
        overlap_fraction:   fraction of chunk to overlap (only for 'overlap')
        decay_alpha:        decay factor per chunk (only for 'decay')
    """
    def __init__(self, d_model, n_heads, rank=48, head_dim=None,
                 ridge_eps=1e-3, scale=1.5, power_K=2, chunk_size=64,
                 backend='auto', chunk_strategy='standard',
                 overlap_fraction=0.5, decay_alpha=0.95):
        super().__init__()
        self.rank = rank
        self.ridge_eps = ridge_eps
        self.power_K = power_K
        self.H = n_heads
        self.P = head_dim or (d_model // n_heads)
        self.d_model = d_model
        self.chunk_size = chunk_size
        self.chunk_strategy = chunk_strategy
        self.overlap_fraction = overlap_fraction
        self.decay_alpha = decay_alpha

        if backend == 'auto':
            self.backend = 'triton' if _TRITON_AVAILABLE else 'pytorch'
        else:
            self.backend = backend

        self.key_proj = nn.Linear(d_model, n_heads * rank, bias=False)
        self.query_proj = nn.Linear(d_model, n_heads * rank, bias=False)
        self.value_proj = nn.Linear(d_model, n_heads * self.P, bias=False)
        self.out_proj = nn.Linear(n_heads * self.P, d_model, bias=False)

        nn.init.orthogonal_(self.key_proj.weight)
        nn.init.orthogonal_(self.query_proj.weight)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.zeros_(self.out_proj.weight)

        self.eta = nn.Parameter(torch.tensor(scale))
        self.ssn_gamma = nn.Parameter(torch.tensor(1.0))

    def _get_chunk_stats(self, z_f, zq_f, v_f):
        """Dispatch to the right chunk-stats function based on strategy."""
        r = self.rank
        H, P = self.H, self.P
        CS = self.chunk_size

        if self.chunk_strategy == 'overlap':
            from koopman_lm.adaptive_chunking import compute_chunk_stats_overlap
            return compute_chunk_stats_overlap(
                z_f, zq_f, v_f, r, H, P, CS, self.ridge_eps,
                overlap_fraction=self.overlap_fraction)
        elif self.chunk_strategy == 'decay':
            from koopman_lm.adaptive_chunking import compute_chunk_stats_decay
            return compute_chunk_stats_decay(
                z_f, zq_f, v_f, r, H, P, CS, self.ridge_eps,
                decay_alpha=self.decay_alpha)
        else:
            # 'standard' — original implementation
            return _compute_chunk_stats_and_cholesky(
                z_f, zq_f, v_f, r, H, P, CS, self.ridge_eps)

    def forward(self, hidden_states):
        B, T, _ = hidden_states.shape
        r = self.rank
        H, P = self.H, self.P

        z = self.key_proj(hidden_states).reshape(B, T, H, r)
        zq = self.query_proj(hidden_states).reshape(B, T, H, r)
        v = self.value_proj(hidden_states).reshape(B, T, H, P)

        ctx = torch.amp.autocast('cuda', enabled=False) if hidden_states.is_cuda \
              else nullcontext()

        with ctx:
            z_f = z.float()
            zq_f = zq.float()
            v_f = v.float()

            # Normalize keys and queries by max key norm (shared scale)
            max_norm = z_f.norm(dim=-1, keepdim=True).max(dim=1, keepdim=True)[0].clamp(min=1e-6)
            z_f = z_f / max_norm
            zq_f = zq_f / max_norm

            # Chunk statistics + Cholesky (strategy-dependent)
            L_flat, M_flat, C_flat, zq_flat, shapes = self._get_chunk_stats(
                z_f, zq_f, v_f)

            # Post-Cholesky: solve + filter + readout (backend-dependent)
            if self.backend == 'triton' and _TRITON_AVAILABLE:
                y_hat = _post_cholesky_triton(
                    L_flat, M_flat, C_flat, zq_flat,
                    self.ssn_gamma, self.power_K, shapes)
            else:
                y_hat = _post_cholesky_pytorch(
                    L_flat, M_flat, C_flat, zq_flat,
                    self.ssn_gamma, self.power_K, shapes)

        y_hat = self.eta * y_hat.to(hidden_states.dtype)
        output = self.out_proj(y_hat.reshape(B, T, H * P))
        return output

    def extra_repr(self):
        parts = [
            f'd_model={self.d_model}',
            f'n_heads={self.H}',
            f'rank={self.rank}',
            f'chunk_size={self.chunk_size}',
            f'backend={self.backend}',
            f'chunk_strategy={self.chunk_strategy}',
        ]
        if self.chunk_strategy == 'overlap':
            parts.append(f'overlap_fraction={self.overlap_fraction}')
        elif self.chunk_strategy == 'decay':
            parts.append(f'decay_alpha={self.decay_alpha}')
        return ', '.join(parts)
