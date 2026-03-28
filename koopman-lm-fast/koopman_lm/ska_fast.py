"""
ska_fast.py -- Drop-in performance patches for SKAModule (v4).

Changes from original ska.py:
  1. Einsums stay in BF16 — FP32 only for Cholesky/solve AND normalization
  2. Conditional Cholesky — jittered fallback only on failing slices
  3. In-place symmetrization (no extra allocation)
  4. Per-layer cached spectral norm vector (warm-start, 1 iteration)
  5. A_w^K via repeated squaring (K=2: one matmul instead of serial loop)
  6. Fused key/query/value projection (3 Linears -> 1, old ones removed)

Checkpoint compatibility:
  The patched model uses a fused projection internally but saves/loads
  state dicts using the ORIGINAL key names (key_proj, query_proj, value_proj).
  This means checkpoints are fully compatible with unpatched KoopmanLM.

Recurrent compatibility:
  After patching, ska_module.key_proj / query_proj / value_proj are
  available as read-only property accessors that slice into fused_proj.
  This ensures RecurrentKoopmanLM._ska_prefill works correctly even
  after patching.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Spectral normalization with cached warm-start vector
# ============================================================================

class CachedSpectralNorm:
    """
    Stateful spectral normalizer that warm-starts from the previous
    call's converged vector. With warm-start, 1 power iteration suffices.

    Each SKA layer gets its own instance so cached vectors correspond
    to the same operator across forward passes.
    """
    def __init__(self):
        self._v = None

    def __call__(self, A, n_iters=1):
        r = A.shape[-1]
        if self._v is None or self._v.shape != (*A.shape[:-1], 1):
            v = torch.ones(*A.shape[:-1], 1, device=A.device,
                           dtype=A.dtype) / math.sqrt(r)
        else:
            v = self._v.to(device=A.device, dtype=A.dtype)

        with torch.no_grad():
            for _ in range(n_iters):
                Av = A @ v
                u = Av / Av.norm(dim=-2, keepdim=True).clamp(min=1e-8)
                Atu = A.transpose(-1, -2) @ u
                v = Atu / Atu.norm(dim=-2, keepdim=True).clamp(min=1e-8)
            sigma_max = (A @ v).norm(dim=-2, keepdim=False).squeeze(-1)

        self._v = v.detach()
        scale = torch.clamp(sigma_max, min=1.0).unsqueeze(-1).unsqueeze(-1)
        return A / scale, sigma_max


# ============================================================================
# Chunk statistics + Cholesky (BF16 einsums, FP32 Cholesky only)
# ============================================================================

def _compute_chunk_stats_and_cholesky_fast(z, zq, v, r, H, P, CS, ridge_eps):
    """
    Chunk statistics in input dtype (BF16), Cholesky in FP32.

    Args z, zq, v should be in their native dtype (BF16 under autocast).
    Only the Cholesky decomposition and solves are upcasted to FP32.
    """
    B, T = z.shape[:2]
    device = z.device
    dtype = z.dtype

    n_chunks = (T + CS - 1) // CS
    T_padded = n_chunks * CS
    pad_len = T_padded - T

    if pad_len > 0:
        z = F.pad(z, (0, 0, 0, 0, 0, pad_len))
        zq = F.pad(zq, (0, 0, 0, 0, 0, pad_len))
        v = F.pad(v, (0, 0, 0, 0, 0, pad_len))

    C = n_chunks
    z_c = z.reshape(B, C, CS, H, r)
    zq_c = zq.reshape(B, C, CS, H, r)
    v_c = v.reshape(B, C, CS, H, P)

    # ---- Per-chunk statistics in native dtype (BF16 on B200) ----
    G_chunks = torch.einsum('bcthr,bcths->bchrs', z_c, z_c)
    M_chunks = torch.einsum('bcthr,bcths->bchrs',
                            z_c[:, :, 1:], z_c[:, :, :-1])
    C_chunks = torch.einsum('bcthp,bcthr->bchpr', v_c, z_c)

    if C > 1:
        M_boundary = torch.einsum('bchr,bchs->bchrs',
                                  z_c[:, 1:, 0], z_c[:, :-1, -1])

    # ---- Exclusive prefix sum (stays in native dtype) ----
    # Batch G and M cumsums together (same shape) to save a kernel launch
    GM_stacked = torch.stack([G_chunks, M_chunks], dim=0)
    GM_cumsum = torch.cumsum(GM_stacked, dim=2)
    GM_excl = torch.zeros_like(GM_cumsum)
    GM_excl[:, :, 1:] = GM_cumsum[:, :, :-1]

    eye_r = torch.eye(r, device=device, dtype=dtype)
    G_excl = GM_excl[0] + ridge_eps * eye_r
    M_excl = GM_excl[1]

    if C > 1:
        M_bnd_full = torch.zeros(B, C, H, r, r, device=device, dtype=dtype)
        M_bnd_full[:, 1:] = M_boundary
        M_bnd_inclusive = torch.cumsum(M_bnd_full, dim=1)
        M_excl = M_excl + M_bnd_inclusive

    C_cumsum = torch.cumsum(C_chunks, dim=1)
    C_excl = torch.zeros_like(C_cumsum)
    C_excl[:, 1:] = C_cumsum[:, :-1]

    # ---- Upcast to FP32 only for Cholesky ----
    BCH = B * C * H
    G_flat = G_excl.reshape(BCH, r, r).float()

    # In-place symmetrization (no extra allocation)
    G_flat += G_flat.transpose(-1, -2)
    G_flat *= 0.5

    L_flat, info = torch.linalg.cholesky_ex(G_flat)

    # Conditional jitter: only recompute for FAILING slices.
    # NOTE: G_flat is mutated in-place below. This is safe because G_flat
    # (and its parent G_excl) are not read again after this point.
    fail_mask = info > 0
    if fail_mask.any():
        eye_r_f32 = torch.eye(r, device=device, dtype=torch.float32)
        G_flat[fail_mask] += 1e-4 * eye_r_f32  # in-place on failing slices only
        L_fixed, _ = torch.linalg.cholesky_ex(G_flat[fail_mask])
        L_flat[fail_mask] = L_fixed

    # Upcast M, C, zq to FP32 for the solve
    M_flat = M_excl.reshape(BCH, r, r).float()
    C_flat = C_excl.reshape(BCH, P, r).float()
    zq_flat = zq_c.permute(0, 1, 3, 4, 2).reshape(BCH, r, CS).float()

    return L_flat, M_flat, C_flat, zq_flat, (B, C, H, P, CS, T, T_padded, pad_len)


# ============================================================================
# Post-Cholesky with per-layer cached spectral norm + A_w^K via squaring
# ============================================================================

def _post_cholesky_pytorch_fast(L_flat, M_flat, C_flat, zq_flat, ssn_gamma,
                                power_K, shapes, spectral_cache):
    """spectral_cache: per-layer CachedSpectralNorm instance."""
    B, C, H, P, CS, T, T_padded, pad_len = shapes

    Aw_T = torch.cholesky_solve(M_flat.transpose(-1, -2), L_flat)
    A_w = Aw_T.transpose(-1, -2)

    A_w, _ = spectral_cache(A_w, n_iters=1)
    gamma_safe = torch.clamp(ssn_gamma, min=1.0, max=1.5)
    A_w = A_w * gamma_safe

    Bv_T = torch.cholesky_solve(C_flat.transpose(-1, -2), L_flat)
    B_v = Bv_T.transpose(-1, -2)

    w_q = torch.linalg.solve_triangular(L_flat, zq_flat, upper=False)

    if power_K == 2:
        A_w2 = A_w @ A_w
        w_f = A_w2 @ w_q
    elif power_K == 1:
        w_f = A_w @ w_q
    else:
        Ak = torch.eye(A_w.shape[-1], device=A_w.device,
                        dtype=A_w.dtype).unsqueeze(0).expand_as(A_w).clone()
        base = A_w
        k = power_K
        while k > 0:
            if k % 2 == 1:
                Ak = Ak @ base
            base = base @ base
            k //= 2
        w_f = Ak @ w_q

    z_out = L_flat @ w_f
    y_flat = B_v @ z_out

    y_hat = y_flat.reshape(B, C, H, P, CS).permute(0, 1, 4, 2, 3)
    y_hat = y_hat.reshape(B, T_padded, H, P)
    if pad_len > 0:
        y_hat = y_hat[:, :T]
    return y_hat


# ============================================================================
# Checkpoint-compatible state_dict hooks
# ============================================================================

def _fused_to_original_state_dict(ska_module, state_dict, prefix, local_metadata):
    """
    state_dict hook: when saving, split fused_proj.weight back into
    key_proj.weight, query_proj.weight, value_proj.weight so the
    checkpoint is loadable by an unpatched KoopmanLM.
    """
    fused_key = prefix + 'fused_proj.weight'
    if fused_key in state_dict:
        w = state_dict.pop(fused_key)
        H = ska_module.H
        r = ska_module.rank
        P = ska_module.P
        state_dict[prefix + 'key_proj.weight'] = w[:H * r]
        state_dict[prefix + 'query_proj.weight'] = w[H * r:2 * H * r]
        state_dict[prefix + 'value_proj.weight'] = w[2 * H * r:]


def _original_to_fused_state_dict(ska_module, state_dict, prefix,
                                   local_metadata, strict, missing_keys,
                                   unexpected_keys, error_msgs):
    """
    pre-load hook: when loading an original checkpoint, fuse the three
    projection weights into fused_proj.weight.
    """
    kp = prefix + 'key_proj.weight'
    qp = prefix + 'query_proj.weight'
    vp = prefix + 'value_proj.weight'

    if kp in state_dict and qp in state_dict and vp in state_dict:
        fused_w = torch.cat([state_dict.pop(kp),
                             state_dict.pop(qp),
                             state_dict.pop(vp)], dim=0)
        state_dict[prefix + 'fused_proj.weight'] = fused_w


# ============================================================================
# Recurrent-compatible projection accessors
# ============================================================================

class _FusedProjSlice(nn.Module):
    """
    Read-only view into a slice of fused_proj, mimicking nn.Linear.
    Used so that RecurrentKoopmanLM can call ska.key_proj(x), etc.
    after fused patching without duplicating parameters.
    """
    def __init__(self, fused_proj, start, end):
        super().__init__()
        self._fused_proj = fused_proj
        self._start = start
        self._end = end

    @property
    def weight(self):
        return self._fused_proj.weight[self._start:self._end]

    @property
    def bias(self):
        return None

    def forward(self, x):
        return F.linear(x, self._fused_proj.weight[self._start:self._end])


# ============================================================================
# Patch function
# ============================================================================

def patch_ska_module(ska_module):
    """
    Apply all performance patches to an existing SKAModule instance.

    Checkpoint-compatible: saves state dicts with original key names
    (key_proj, query_proj, value_proj) and loads them back correctly.

    Recurrent-compatible: key_proj, query_proj, value_proj remain
    accessible as thin slice views into fused_proj, so
    RecurrentKoopmanLM._ska_prefill and ._ska_step work correctly.
    """
    import types

    d_model = ska_module.d_model
    H = ska_module.H
    r = ska_module.rank
    P = ska_module.P

    # ---- Assert no bias on original projections ----
    assert ska_module.key_proj.bias is None, \
        "ska_fast assumes key_proj has no bias"
    assert ska_module.query_proj.bias is None, \
        "ska_fast assumes query_proj has no bias"
    assert ska_module.value_proj.bias is None, \
        "ska_fast assumes value_proj has no bias"

    # ---- Fuse key/query/value projections (3 Linears -> 1) ----
    fused_dim = 2 * H * r + H * P
    fused_proj = nn.Linear(d_model, fused_dim, bias=False,
                           device=ska_module.key_proj.weight.device,
                           dtype=ska_module.key_proj.weight.dtype)

    with torch.no_grad():
        fused_proj.weight[:H * r] = ska_module.key_proj.weight
        fused_proj.weight[H * r:2 * H * r] = ska_module.query_proj.weight
        fused_proj.weight[2 * H * r:] = ska_module.value_proj.weight

    ska_module.fused_proj = fused_proj

    # Remove old projections from _modules to avoid duplicate params
    ska_module._modules.pop('key_proj')
    ska_module._modules.pop('query_proj')
    ska_module._modules.pop('value_proj')

    # ---- Re-register as slice views for recurrent compatibility ----
    # These are nn.Modules registered in _modules, so they appear as
    # attributes but do NOT own parameters (they slice fused_proj).
    ska_module.key_proj = _FusedProjSlice(fused_proj, 0, H * r)
    ska_module.query_proj = _FusedProjSlice(fused_proj, H * r, 2 * H * r)
    ska_module.value_proj = _FusedProjSlice(fused_proj, 2 * H * r, fused_dim)

    # ---- Register state_dict hooks for checkpoint compatibility ----
    # On save: split fused_proj -> key_proj + query_proj + value_proj
    # On load: fuse key_proj + query_proj + value_proj -> fused_proj
    try:
        ska_module.register_state_dict_post_hook(_fused_to_original_state_dict)
        ska_module.register_load_state_dict_pre_hook(_original_to_fused_state_dict)
    except AttributeError:
        ska_module._register_state_dict_hook(_fused_to_original_state_dict)
        ska_module._register_load_state_dict_pre_hook(_original_to_fused_state_dict)

    # ---- Per-layer spectral norm cache ----
    ska_module._spectral_cache = CachedSpectralNorm()

    # ---- Override chunk stats ----
    def _get_chunk_stats_fast(self, z, zq, v):
        CS = self.chunk_size
        if self.chunk_strategy == 'overlap':
            from koopman_lm.adaptive_chunking import compute_chunk_stats_overlap
            return compute_chunk_stats_overlap(
                z, zq, v, self.rank, self.H, self.P, CS, self.ridge_eps,
                overlap_fraction=self.overlap_fraction)
        elif self.chunk_strategy == 'decay':
            from koopman_lm.adaptive_chunking import compute_chunk_stats_decay
            return compute_chunk_stats_decay(
                z, zq, v, self.rank, self.H, self.P, CS, self.ridge_eps,
                decay_alpha=self.decay_alpha)
        else:
            return _compute_chunk_stats_and_cholesky_fast(
                z, zq, v, self.rank, self.H, self.P, CS, self.ridge_eps)

    # ---- Override forward (FP32 for norm, BF16 for einsums) ----
    def forward_fast(self, hidden_states):
        B, T, _ = hidden_states.shape
        r = self.rank
        H, P = self.H, self.P

        # Fused projection — one kernel launch instead of three
        combined = self.fused_proj(hidden_states)
        z = combined[:, :, :H * r].reshape(B, T, H, r)
        zq = combined[:, :, H * r:2 * H * r].reshape(B, T, H, r)
        v = combined[:, :, 2 * H * r:].reshape(B, T, H, P)

        # Normalization in FP32 — norm computation is precision-sensitive
        # (BF16 has only ~3 decimal digits; squaring r=48 elements can
        # overflow or lose precision at 370M+ scale)
        z_f32 = z.float()
        max_norm = z_f32.norm(dim=-1, keepdim=True).max(
            dim=1, keepdim=True)[0].clamp(min=1e-6)
        z = z_f32.div(max_norm).to(z.dtype)
        zq = zq.float().div(max_norm).to(zq.dtype)

        # Chunk stats (BF16 einsums) + Cholesky (FP32)
        L_flat, M_flat, C_flat, zq_flat, shapes = self._get_chunk_stats(
            z, zq, v)

        # Post-Cholesky (FP32), using this layer's spectral norm cache
        from koopman_lm.ska import _TRITON_AVAILABLE
        if self.backend == 'triton' and _TRITON_AVAILABLE:
            from koopman_lm.ska import _post_cholesky_triton
            y_hat = _post_cholesky_triton(
                L_flat, M_flat, C_flat, zq_flat,
                self.ssn_gamma, self.power_K, shapes)
        else:
            y_hat = _post_cholesky_pytorch_fast(
                L_flat, M_flat, C_flat, zq_flat,
                self.ssn_gamma, self.power_K, shapes,
                self._spectral_cache)

        y_hat = self.eta * y_hat.to(hidden_states.dtype)
        output = self.out_proj(y_hat.reshape(B, T, H * P))
        return output

    ska_module._get_chunk_stats = types.MethodType(_get_chunk_stats_fast, ska_module)
    ska_module.forward = types.MethodType(forward_fast, ska_module)
