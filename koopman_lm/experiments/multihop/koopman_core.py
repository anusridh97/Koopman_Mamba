"""
koopman_core.py -- model core extracted VERBATIM from the user's tool_call notebook
(cell 0). Contains: RoPE helpers, CGFeatureLift, ModelConfig, the chunked/sequential
recurrence, Mamba3Block, SKA helpers, SKAModule, SwiGLUMLP, Mamba3LM, Mamba3CGSKALM.

This is the user's own, working code -- unchanged -- so the multi-hop experiment
runs on the *real* Mamba3Block and SKAModule, not a re-implementation.
"""
import math
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

HAS_SCHUR = hasattr(torch.linalg, 'schur')
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# SPD / Schur helpers (verbatim)
# ---------------------------------------------------------------------------

def _cholesky_solve_spd(A, rhs):
    try:
        L = torch.linalg.cholesky(A)
        return torch.cholesky_solve(rhs, L)
    except torch.linalg.LinAlgError:
        return torch.linalg.solve(A, rhs)


def schur_solve_level_parallel(G, rhs, base_size=8):
    r = G.shape[-1]
    if r <= base_size:
        return _cholesky_solve_spd(G, rhs)
    m = r // 2
    A, B, C, D = G[...,:m,:m], G[...,:m,m:], G[...,m:,:m], G[...,m:,m:]
    B1, B2 = rhs[...,:m,:], rhs[...,m:,:]
    combined_rhs = torch.cat([B, B1], dim=-1)
    combined_sol = schur_solve_level_parallel(A, combined_rhs, base_size)
    cols_B = B.shape[-1]
    Z1, Z2 = combined_sol[...,:cols_B], combined_sol[...,cols_B:]
    S = D - C @ Z1
    X2 = schur_solve_level_parallel(S, B2 - C @ Z2, base_size)
    X1 = Z2 - Z1 @ X2
    return torch.cat([X1, X2], dim=-2)

# 2. MQAR Data (unchanged)


def rope_rotate(u, theta):
    out = torch.empty_like(u)
    ue, uo = u[..., 0::2], u[..., 1::2]
    cos_t, sin_t = torch.cos(theta), torch.sin(theta)
    out[..., 0::2] = ue * cos_t - uo * sin_t
    out[..., 1::2] = ue * sin_t + uo * cos_t
    return out


def build_cumulative_angles(delta, omega, phase_delta=None):
    inc = delta * omega.unsqueeze(0).unsqueeze(0)
    theta = torch.cumsum(inc, dim=1)
    if phase_delta is not None:
        theta = theta + phase_delta
    return theta


class CGFeatureLift(nn.Module):
    def __init__(self, N, use_harmonic=False):
        super().__init__()
        self.N_half = N // 2
        self.use_harmonic = use_harmonic
        self.D_feat = self.N_half if not use_harmonic else 3 * self.N_half

    def forward(self, k_rot):
        kr, ki = k_rot[..., 0::2], k_rot[..., 1::2]
        psi_inv = kr**2 + ki**2
        if not self.use_harmonic:
            return psi_inv
        return torch.cat([psi_inv, kr**2 - ki**2, 2*kr*ki], dim=-1)


@dataclass
class ModelConfig:
    d_model: int = 128
    n_heads: int = 4
    d_state: int = 16
    n_layers: int = 4
    vocab_size: int = 256
    delta_min: float = 1e-3
    delta_max: float = 1.0
    alpha_max: float = 0.999
    ska_layers: List[int] = field(default_factory=list)
    ska_rank: int = 32
    ska_ridge: float = 1e-3
    ska_scale: float = 0.5
    use_cg: bool = False
    use_harmonic: bool = False
    power_K: int = 2
    ska_use_schur_filter: bool = False
    ska_eigen_threshold: float = 0.5
    ska_spectral_norm: bool = True    # ON by default — SSN gamma prevents eigenvalue crushing
    ska_seq_max_norm: bool = True     # sequence-max normalization of z features
    ska_independent: bool = True      # v7: SKA gets own projections from residual stream
    ska_gated: bool = True            # v7: sigmoid gate instead of additive blend
    # Parallelization config
    scan_mode: str = 'chunked'   # 'chunked' or 'sequential'
    chunk_size: int = 64

    @property
    def d_head(self):
        return self.d_model // self.n_heads

    @property
    def N_half(self):
        return self.d_state // 2

# 5. RECURRENCE IMPLEMENTATIONS
#
# The recurrence:  H[t] = α[t] · H[t-1] + b[t]
# where b[t] = β[t]·U[t-1] + γ[t]·U[t], U[t] = k[t] ⊗ v[t]
#
# Two implementations:
#   A) Sequential (reference, O(T) depth)
#   B) Chunked SSD (Mamba-2 style, O(C² + T/C), matmul-heavy, FP32 scan)

def _run_recurrence_sequential(k_real, k_imag, v, q_real, q_imag, alpha, beta, gamma):
    """Original sequential loop — kept as reference / fallback."""
    B, T, H, N_half = k_real.shape
    P = v.shape[-1]
    dev, dt = v.device, v.dtype

    U_real_all = k_real.unsqueeze(-1) * v.unsqueeze(-2)
    U_imag_all = k_imag.unsqueeze(-1) * v.unsqueeze(-2)

    H_re = torch.zeros(B, H, N_half, P, device=dev, dtype=dt)
    H_im = torch.zeros(B, H, N_half, P, device=dev, dtype=dt)
    U_re_prev = torch.zeros(B, H, N_half, P, device=dev, dtype=dt)
    U_im_prev = torch.zeros(B, H, N_half, P, device=dev, dtype=dt)
    outputs = torch.empty(B, T, H, P, device=dev, dtype=dt)

    for t in range(T):
        U_re_t = U_real_all[:, t]
        U_im_t = U_imag_all[:, t]
        a_t = alpha[:, t, :, None, None]
        b_t = beta[:, t, :, None, None]
        g_t = gamma[:, t, :, None, None]
        H_re = a_t * H_re + b_t * U_re_prev + g_t * U_re_t
        H_im = a_t * H_im + b_t * U_im_prev + g_t * U_im_t
        y_t = (q_real[:, t].unsqueeze(-2) @ H_re).squeeze(-2) + \
              (q_imag[:, t].unsqueeze(-2) @ H_im).squeeze(-2)
        outputs[:, t] = y_t
        U_re_prev = U_re_t
        U_im_prev = U_im_t
    return outputs


# ---------- B) Chunked SSD (Mamba-2 style, FP32 scan path) ----------

def _precompute_inputs(k_real, k_imag, v, alpha, beta, gamma):
    """
    Precompute outer products and additive inputs for the scan.
    Separated so we can force FP32 cleanly.

    Returns b_real, b_imag: (B, T, H, N_half, P) — additive scan inputs
    """
    # Outer products: U[t] = k[t] ⊗ v[t]
    U_real = k_real.unsqueeze(-1) * v.unsqueeze(-2)  # (B, T, H, N_half, P)
    U_imag = k_imag.unsqueeze(-1) * v.unsqueeze(-2)

    # Shift: U[t-1] with zero-padding at t=0
    U_real_prev = F.pad(U_real[:, :-1], (0, 0, 0, 0, 0, 0, 1, 0))
    U_imag_prev = F.pad(U_imag[:, :-1], (0, 0, 0, 0, 0, 0, 1, 0))

    # b[t] = β[t]·U[t-1] + γ[t]·U[t]
    beta_e = beta.unsqueeze(-1).unsqueeze(-1)    # (B, T, H, 1, 1)
    gamma_e = gamma.unsqueeze(-1).unsqueeze(-1)

    b_real = beta_e * U_real_prev + gamma_e * U_real
    b_imag = beta_e * U_imag_prev + gamma_e * U_imag

    return b_real, b_imag


def _segsum_lower_tri(log_alpha_chunk):
    """
    Segment sum → lower-triangular decay matrix within a chunk.

    Given log(α) of shape (..., C), produces L of shape (..., C, C) where:
        L[i,j] = exp(sum_{s=j+1}^{i} log α[s])  for i >= j
        L[i,j] = 0                                 for i < j
        L[i,i] = 1                                 (empty product)

    All arithmetic in whatever dtype the input is (caller ensures FP32).
    """
    C = log_alpha_chunk.shape[-1]
    cumsum = torch.cumsum(log_alpha_chunk, dim=-1)              # (..., C)
    # L_log[i,j] = cumsum[i] - cumsum[j]  → product from j+1..i
    L_log = cumsum.unsqueeze(-1) - cumsum.unsqueeze(-2)         # (..., C, C)
    mask = torch.tril(torch.ones(C, C, device=log_alpha_chunk.device, dtype=torch.bool))
    L_log = L_log.masked_fill(~mask, -float('inf'))
    return torch.exp(L_log)


def _run_recurrence_chunked(k_real, k_imag, v, q_real, q_imag,
                             alpha, beta, gamma, chunk_size=64):
    """
    Chunked SSD recurrence (Mamba-2 algorithm) with FP32 scan path.

    Steps:
      1. Intra-chunk: matmul with decay matrix L (parallel per chunk)
      2. Chunk states: extract boundary state at end of each chunk
      3. Inter-chunk: propagate boundary states (short sequential scan on T/C)
      4. Output states: add initial-state contribution via matmul

    Critical: ALL scan math (log/exp/cumsum/matmul/boundary) runs in FP32.
    Only the final output is cast back to the original dtype.
    """
    B, T, H, N_half = k_real.shape
    P = v.shape[-1]
    dev = v.device
    orig_dtype = v.dtype
    C = chunk_size

    # ---- Precompute additive inputs (can stay in orig dtype initially) ----
    b_real, b_imag = _precompute_inputs(k_real, k_imag, v, alpha, beta, gamma)

    # FORCE FP32 for all scan-critical operations
    # This is the single biggest correctness lever.

    b_real = b_real.float()
    b_imag = b_imag.float()
    alpha_f32 = alpha.float()
    q_real_f32 = q_real.float()
    q_imag_f32 = q_imag.float()

    # ---- Pad T to multiple of chunk_size ----
    T_orig = T
    if T % C != 0:
        pad_len = C - (T % C)
        b_real = F.pad(b_real, (0, 0, 0, 0, 0, 0, 0, pad_len))
        b_imag = F.pad(b_imag, (0, 0, 0, 0, 0, 0, 0, pad_len))
        alpha_f32 = F.pad(alpha_f32, (0, 0, 0, pad_len), value=1e-8)
        q_real_f32 = F.pad(q_real_f32, (0, 0, 0, 0, 0, pad_len))
        q_imag_f32 = F.pad(q_imag_f32, (0, 0, 0, 0, 0, pad_len))
        T = b_real.shape[1]

    n_chunks = T // C
    NP = N_half * P  # flattened state dim

    # ---- Reshape into chunks ----
    # b: (B, T, H, N, P) → (B, nc, C, H, N, P)
    b_re_c = b_real.reshape(B, n_chunks, C, H, N_half, P)
    b_im_c = b_imag.reshape(B, n_chunks, C, H, N_half, P)
    alpha_c = alpha_f32.reshape(B, n_chunks, C, H)

    # Move H before C for batched matmul: (B, H, nc, C, ...)
    # log_alpha: (B, H, nc, C)
    log_alpha = torch.log(alpha_c.clamp(min=1e-8)).permute(0, 3, 1, 2)

    # b: (B, H, nc, C, NP) — flatten N,P for matmul
    b_re_bh = b_re_c.permute(0, 3, 1, 2, 4, 5).reshape(B, H, n_chunks, C, NP)
    b_im_bh = b_im_c.permute(0, 3, 1, 2, 4, 5).reshape(B, H, n_chunks, C, NP)

    # STEP 1: Intra-chunk outputs (diagonal blocks)
    # H_intra[c, t] = sum_{s=0}^{t} L[t,s] · b[c, s]

    L = _segsum_lower_tri(log_alpha)    # (B, H, nc, C, C)  — FP32

    H_re_intra = torch.matmul(L, b_re_bh)   # (B, H, nc, C, NP)
    H_im_intra = torch.matmul(L, b_im_bh)

    # STEP 2: Chunk boundary states
    # state_c = H_intra at the last position of each chunk
    # This is the state at end of chunk c, assuming chunk c started at zero.

    chunk_states_re = H_re_intra[:, :, :, -1, :]  # (B, H, nc, NP)
    chunk_states_im = H_im_intra[:, :, :, -1, :]

    # STEP 3: Inter-chunk state propagation
    # S[c] = chunk_decay[c] · S[c-1] + chunk_states[c]
    # chunk_decay[c] = exp(sum of log_alpha within chunk c)
    # Sequential scan over n_chunks (typically T/64 ≈ 5-25 steps)

    chunk_decay_log = log_alpha.sum(dim=-1)  # (B, H, nc) total decay per chunk

    # Propagated (true) state at end of each chunk
    true_states_re = torch.zeros(B, H, n_chunks, NP, device=dev, dtype=torch.float32)
    true_states_im = torch.zeros(B, H, n_chunks, NP, device=dev, dtype=torch.float32)

    prev_re = torch.zeros(B, H, NP, device=dev, dtype=torch.float32)
    prev_im = torch.zeros(B, H, NP, device=dev, dtype=torch.float32)
    for c in range(n_chunks):
        decay = torch.exp(chunk_decay_log[:, :, c]).unsqueeze(-1)  # (B, H, 1)
        prev_re = decay * prev_re + chunk_states_re[:, :, c]
        prev_im = decay * prev_im + chunk_states_im[:, :, c]
        true_states_re[:, :, c] = prev_re
        true_states_im[:, :, c] = prev_im

    # STEP 4: Output states (off-diagonal contribution)
    # For chunk c, initial state = true_states[c-1] (zero for c=0).
    # H_off[c, t] = decay_from_start_to_t · S[c-1]
    # decay_from_start_to_t = exp(cumsum(log_alpha from 0..t))

    cumsum_within = torch.cumsum(log_alpha, dim=-1)     # (B, H, nc, C)
    decay_from_start = torch.exp(cumsum_within)          # (B, H, nc, C)

    # Shift: S[c-1] for chunk c; S[-1] = 0 for chunk 0
    init_states_re = F.pad(true_states_re[:, :, :-1], (0, 0, 1, 0))  # (B, H, nc, NP)
    init_states_im = F.pad(true_states_im[:, :, :-1], (0, 0, 1, 0))

    # H_off[c, t] = decay_from_start[c, t] · init_states[c]
    # decay: (B, H, nc, C) → (B, H, nc, C, 1)
    # init:  (B, H, nc, NP) → (B, H, nc, 1, NP)
    H_re_off = decay_from_start.unsqueeze(-1) * init_states_re.unsqueeze(-2)
    H_im_off = decay_from_start.unsqueeze(-1) * init_states_im.unsqueeze(-2)
    # → (B, H, nc, C, NP)

    # Combine intra + off-diagonal

    H_re_full = H_re_intra + H_re_off   # (B, H, nc, C, NP)
    H_im_full = H_im_intra + H_im_off

    # Unflatten NP → (N_half, P) and rearrange to (B, T, H, N_half, P)
    H_re_full = H_re_full.reshape(B, H, n_chunks, C, N_half, P)
    H_im_full = H_im_full.reshape(B, H, n_chunks, C, N_half, P)
    # (B, H, nc, C, N, P) → (B, nc, C, H, N, P) → (B, T, H, N, P)
    H_re_full = H_re_full.permute(0, 2, 3, 1, 4, 5).reshape(B, T, H, N_half, P)
    H_im_full = H_im_full.permute(0, 2, 3, 1, 4, 5).reshape(B, T, H, N_half, P)

    # Readout: y[t] = q_re[t]^T H_re[t] + q_im[t]^T H_im[t]

    # q: (B, T, H, N_half),  H: (B, T, H, N_half, P)
    y = torch.einsum('bthn,bthnp->bthp', q_real_f32, H_re_full) + \
        torch.einsum('bthn,bthnp->bthp', q_imag_f32, H_im_full)

    # Trim padding and cast back
    y = y[:, :T_orig].to(orig_dtype)
    return y

# 5b. Verification: check chunked matches sequential

def verify_scan_correctness(device=DEVICE, T=200, B=2, H=3, N_half=8, P=8,
                             chunk_sizes=(16, 32, 64)):
    """Compare chunked SSD output against sequential reference."""
    print("\n" + "="*60)
    print("SCAN CORRECTNESS VERIFICATION")
    print("="*60)

    torch.manual_seed(123)
    k_re = torch.randn(B, T, H, N_half, device=device)
    k_im = torch.randn(B, T, H, N_half, device=device)
    v = torch.randn(B, T, H, P, device=device)
    q_re = torch.randn(B, T, H, N_half, device=device)
    q_im = torch.randn(B, T, H, N_half, device=device)
    alpha = torch.sigmoid(torch.randn(B, T, H, device=device)) * 0.95 + 0.01
    beta = torch.randn(B, T, H, device=device) * 0.1
    gamma = torch.randn(B, T, H, device=device) * 0.1

    # Reference
    y_seq = _run_recurrence_sequential(k_re, k_im, v, q_re, q_im, alpha, beta, gamma)

    for C in chunk_sizes:
        y_chunk = _run_recurrence_chunked(k_re, k_im, v, q_re, q_im,
                                           alpha, beta, gamma, chunk_size=C)
        rel_err = (y_chunk - y_seq).abs().max() / (y_seq.abs().max() + 1e-8)
        print(f"  chunked C={C:>3d}: max_rel_err = {rel_err.item():.2e}"
              f"  {'✓' if rel_err < 1e-4 else '⚠ HIGH'}")

    print()

# 6. Mamba-3 Block (scan_mode selects recurrence implementation)

class Mamba3Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        H, N, P, d = cfg.n_heads, cfg.d_state, cfg.d_head, cfg.d_model
        proj_size = H*N + H*N + H*P + H + H + H
        self.in_proj = nn.Linear(d, proj_size, bias=False)
        self.out_proj = nn.Linear(d, d, bias=False)
        self.norm = nn.LayerNorm(d)
        self.delta_bias = nn.Parameter(torch.zeros(H))
        self.a_bias = nn.Parameter(torch.log(torch.linspace(0.5, 2.0, H)))
        N_half = N // 2
        base_freq = 1.0 / (10000 ** (torch.arange(0, N_half).float() / N_half))
        self.register_buffer('omega_base', base_freq.unsqueeze(0).expand(H, -1))
        self.omega_scale = nn.Parameter(torch.ones(H, 1))
        self.conv1d = nn.Conv1d(d, d, kernel_size=4, padding=3, groups=d, bias=True)

    @property
    def omega(self):
        return self.omega_scale * self.omega_base

    def forward(self, x, phase_delta=None, return_aux=False):
        B, T, d = x.shape
        C = self.cfg; H, N, P = C.n_heads, C.d_state, C.d_head
        residual = x
        x = self.norm(x)
        proj = self.in_proj(x)
        idx = 0
        b_raw = proj[..., idx:idx+H*N].reshape(B, T, H, N); idx += H*N
        c_raw = proj[..., idx:idx+H*N].reshape(B, T, H, N); idx += H*N
        v_raw_flat = proj[..., idx:idx+H*P]; idx += H*P
        delta_raw = proj[..., idx:idx+H].reshape(B, T, H); idx += H
        a_raw = proj[..., idx:idx+H].reshape(B, T, H); idx += H
        lam_raw = proj[..., idx:idx+H].reshape(B, T, H); idx += H

        v_conv = self.conv1d(v_raw_flat.transpose(1, 2))[..., :T].transpose(1, 2)
        v = F.silu(v_conv).reshape(B, T, H, P)

        delta = torch.clamp(F.softplus(delta_raw + self.delta_bias), C.delta_min, C.delta_max)
        a = F.softplus(a_raw + self.a_bias)
        alpha = torch.clamp(torch.exp(-a * delta), max=C.alpha_max)
        lam = torch.sigmoid(lam_raw)
        beta = (1.0 - lam) * delta * alpha
        gamma = lam * delta

        theta = build_cumulative_angles(delta.unsqueeze(-1), self.omega, phase_delta)
        k_rot = rope_rotate(b_raw, theta)
        q_rot = rope_rotate(c_raw, theta)

        # Select recurrence implementation
        if C.scan_mode == 'chunked':
            y = _run_recurrence_chunked(
                k_rot[..., 0::2], k_rot[..., 1::2], v,
                q_rot[..., 0::2], q_rot[..., 1::2],
                alpha, beta, gamma, chunk_size=C.chunk_size)
        else:
            y = _run_recurrence_sequential(
                k_rot[..., 0::2], k_rot[..., 1::2], v,
                q_rot[..., 0::2], q_rot[..., 1::2],
                alpha, beta, gamma)

        y = self.out_proj(y.reshape(B, T, d))
        aux = None
        if return_aux:
            aux = {'k_rot': k_rot, 'q_rot': q_rot, 'v': v, 'alpha': alpha}
        return y + residual, aux

# 7. SKA Module — FULLY PATCHED + SPECTRAL NORMALIZATION

def _robust_cholesky(G_tilde, max_retries=4):
    """Cholesky with symmetrization + adaptive jitter retry."""
    G_sym = 0.5 * (G_tilde + G_tilde.transpose(-1, -2))
    for attempt in range(max_retries):
        try:
            L = torch.linalg.cholesky(G_sym)
            return L
        except torch.linalg.LinAlgError:
            jitter = 1e-4 * (10 ** attempt)
            eye = torch.eye(G_sym.shape[-1], device=G_sym.device, dtype=G_sym.dtype)
            G_sym = G_sym + jitter * eye
    eigvals, eigvecs = torch.linalg.eigh(G_sym)
    eigvals = eigvals.clamp(min=1e-6)
    G_psd = eigvecs @ torch.diag_embed(eigvals) @ eigvecs.transpose(-1, -2)
    return torch.linalg.cholesky(G_psd)


def _spectral_normalize_power_iter(A, n_iters=6):
    """
    Spectral normalization of A via power iteration.

    Estimates σ_max(A) and returns A / max(1, σ_max).
    This constrains |λ|_max ≤ 1 BEFORE the power filter runs.

    Key choices for reliability:
      - 6 iterations (not 2) — needed when σ_max is large (1000+)
      - Deterministic init (ones/√r) — avoids random-init variance
      - σ = ||Av|| after final step (more stable than u^T A v)

    Args:
        A: (..., r, r) — batched square matrices
        n_iters: number of power iteration steps

    Returns:
        A_normalized: same shape, with spectral radius ≤ 1
        sigma_max: estimated spectral norm (for diagnostics)
    """
    r = A.shape[-1]

    # Deterministic init: ones / sqrt(r)  (not random — removes variance)
    v = torch.ones(*A.shape[:-1], 1, device=A.device, dtype=A.dtype) / math.sqrt(r)

    with torch.no_grad():
        for _ in range(n_iters):
            # u = A v / ||A v||
            Av = A @ v
            u = Av / Av.norm(dim=-2, keepdim=True).clamp(min=1e-8)
            # v = A^T u / ||A^T u||
            Atu = A.transpose(-1, -2) @ u
            v = Atu / Atu.norm(dim=-2, keepdim=True).clamp(min=1e-8)

        # σ_max ≈ ||A v||  (more stable than u^T A v for non-normal matrices)
        Av_final = A @ v
        sigma_max = Av_final.norm(dim=-2, keepdim=False).squeeze(-1)  # (...,)

    # Normalize: A / max(1, σ_max)
    scale = torch.clamp(sigma_max, min=1.0).unsqueeze(-1).unsqueeze(-1)
    A_norm = A / scale

    return A_norm, sigma_max


def _power_spectral_filter(A_w, w_q, power_K=2):
    """
    Power filter: computes A_w^K @ w_q.

    A_w should already be spectrally normalized (|λ| ≤ 1) before calling this.

    With K=2 (default):
      |λ|=0.9 → 0.81 (preserved)
      |λ|=0.5 → 0.25 (suppressed)
      |λ|=0.3 → 0.09 (strongly suppressed)

    K=2 is gentler than K=4, allowing more modes to contribute.
    K=4 was crushing everything below |λ|≈0.85 to near-zero.

    No norm rescaling — that would undo the suppression.
    """
    A_filt = A_w
    for _ in range(power_K - 1):
        A_filt = A_filt @ A_w
    return A_filt @ w_q


def _schur_spectral_filter(A_w, w_q, threshold=0.5):
    """Real Schur spectral filter (only runs if torch has schur)."""
    A_det = A_w.detach()
    B, H, r, _ = A_det.shape

    T_schur, Q = torch.linalg.schur(A_det, output='real')

    diag = T_schur.diagonal(dim1=-2, dim2=-1)
    diag_mag = diag.abs()

    subdiag = torch.zeros_like(diag)
    if r > 1:
        subdiag_vals = torch.diagonal(T_schur, offset=-1, dim1=-2, dim2=-1)
        subdiag[..., :r-1] = subdiag_vals

    superdiag = torch.zeros_like(diag)
    if r > 1:
        superdiag_vals = torch.diagonal(T_schur, offset=1, dim1=-2, dim2=-1)
        superdiag[..., :r-1] = superdiag_vals

    block_correction = (subdiag.abs() * superdiag.abs().roll(-1, dims=-1)).sqrt()
    is_2x2 = subdiag.abs() > 1e-10
    eig_mag = torch.where(is_2x2, (diag_mag**2 + block_correction**2).sqrt(), diag_mag)
    eig_mag = eig_mag.clamp(max=1.0)

    tau = threshold
    mask = torch.clamp((eig_mag - tau) / (1.0 - tau + 1e-8), 0.0, 1.0)
    T_filtered = T_schur * mask.unsqueeze(-1)

    Qt_wq = Q.transpose(-1, -2) @ w_q
    filtered = T_filtered @ Qt_wq
    w_f = Q @ filtered
    return w_f


class SKAModule(nn.Module):
    """
    Spectral Koopman Attention — v7 (Memory Vault).

    Two modes:
      - Legacy (use_independent=False): uses Mamba's k_rot/q_rot/v features
      - Independent (use_independent=True): own projections from residual stream
        → decouples SKA's feature space from Mamba's rotary decay logic
        → allows SKA to learn a "discrete address space" for long-term storage

    Supports action_mask: only high-signal tokens contribute to G/M,
    preventing noise gap tokens from polluting the Koopman operator.
    """
    def __init__(self, N, P, rank=32, ridge_eps=1e-3, scale=0.5,
                 power_K=2, use_cg=True, use_harmonic=False,
                 use_schur_filter=False, eigen_threshold=0.5,
                 use_spectral_norm=True, use_seq_max_norm=True,
                 use_independent=False, d_model=None, n_heads=None):
        super().__init__()
        self.rank = rank
        self.ridge_eps = ridge_eps
        self.power_K = power_K
        self.use_schur_filter = use_schur_filter and HAS_SCHUR
        self.eigen_threshold = eigen_threshold
        self.use_spectral_norm = use_spectral_norm
        self.use_seq_max_norm = use_seq_max_norm
        self.use_independent = use_independent

        if use_independent:
            # v7: Own projections from residual stream — decoupled from Mamba
            assert d_model is not None and n_heads is not None
            H = n_heads
            self.ska_key_proj = nn.Linear(d_model, H * rank, bias=False)
            self.ska_query_proj = nn.Linear(d_model, H * rank, bias=False)
            self.ska_value_proj = nn.Linear(d_model, H * P, bias=False)
            nn.init.orthogonal_(self.ska_key_proj.weight)
            nn.init.orthogonal_(self.ska_query_proj.weight)
            self.cg_lift = None
            self.Phi = None
        else:
            # Legacy: CG lift from Mamba's rotary features
            if use_cg:
                self.cg_lift = CGFeatureLift(N, use_harmonic)
                D_feat = self.cg_lift.D_feat
            else:
                D_feat = N // 2
                self.cg_lift = None
            self.Phi = nn.Linear(D_feat, rank, bias=False)
            nn.init.orthogonal_(self.Phi.weight)

        self.eta = nn.Parameter(torch.tensor(scale))
        # SSN (Lin et al. NeurIPS 2021): learnable variance restoration
        self.ssn_gamma = nn.Parameter(torch.tensor(1.0))

    def _lift(self, rot):
        """Legacy mode: CG lift + Phi projection."""
        if self.cg_lift is not None:
            psi = self.cg_lift(rot)
        else:
            psi = rot[..., 0::2]**2 + rot[..., 1::2]**2
        return self.Phi(psi)

    def forward(self, k_rot_or_h, q_rot_or_none, v_or_none, prefix_mask,
                action_mask=None):
        """
        Args:
            k_rot_or_h: [B,T,H,N] (legacy) or [B,T,d_model] (independent)
            q_rot_or_none: [B,T,H,N] (legacy) or None (independent)
            v_or_none: [B,T,H,P] (legacy) or None (independent)
            prefix_mask: [B,T] — everything before the answer
            action_mask: [B,T] — only high-signal tokens for G/M (optional)
        """
        if self.use_independent:
            h = k_rot_or_h
            B, T, _ = h.shape
            H = self.ska_key_proj.out_features // self.rank
            P = self.ska_value_proj.out_features // H
            r = self.rank
            z = self.ska_key_proj(h).reshape(B, T, H, r)
            zq = self.ska_query_proj(h).reshape(B, T, H, r)
            v_ska = self.ska_value_proj(h).reshape(B, T, H, P)
        else:
            B, T, H, N = k_rot_or_h.shape
            P = v_or_none.shape[-1]
            r = self.rank
            z = self._lift(k_rot_or_h)
            zq = self._lift(q_rot_or_none)
            v_ska = v_or_none

        # Use action_mask for G/M if provided, else prefix_mask
        #ska_mask = action_mask if action_mask is not None else prefix_mask
        ska_mask = prefix_mask

        from contextlib import nullcontext
        ctx = torch.amp.autocast('cuda', enabled=False) if torch.cuda.is_available() \
              else nullcontext()

        with ctx:
            z_f32 = z.float()
            v_f32 = v_ska.float()
            zq_f32 = zq.float()
            m = ska_mask.float().unsqueeze(-1).unsqueeze(-1)

            # Sequence-max normalization
            if self.use_seq_max_norm:
                z_norms = z_f32.norm(dim=-1, keepdim=True)
                max_norm = z_norms.max(dim=1, keepdim=True)[0]
                z_f32 = z_f32 / (max_norm + 1e-6)
                zq_f32 = zq_f32 / (max_norm + 1e-6)

            # Gram matrix G — built ONLY from action tokens
            z_m = z_f32 * m
            G = torch.einsum('bthr,bths->bhrs', z_m, z_m)
            G_tilde = G + self.ridge_eps * torch.eye(r, device=G.device, dtype=torch.float32)

            # Lagged cross-covariance M — also only action tokens
            m_lag = (ska_mask[:, :-1] * ska_mask[:, 1:]).float().unsqueeze(-1).unsqueeze(-1)
            M_cov = torch.einsum('bthr,bths->bhrs',
                                 z_f32[:, 1:] * m_lag,
                                 z_f32[:, :-1] * m_lag)

            # Cross-covariance C_v
            C_v = torch.einsum('bthp,bthr->bhpr',
                               v_f32 * m, z_f32 * m)

            # Robust Cholesky
            L = _robust_cholesky(G_tilde)

            # Whitened Koopman: A_w = L^{-1} M L^{-T} (transpose-correct)
            Y = torch.linalg.solve_triangular(L, M_cov, upper=False)
            Aw_T = torch.linalg.solve_triangular(L, Y.transpose(-1, -2), upper=False)
            A_w = Aw_T.transpose(-1, -2)

            # SSN: spectral norm + learned gamma recovery
            if self.use_spectral_norm:
                A_w, sigma_max = _spectral_normalize_power_iter(A_w)
                gamma_safe = torch.clamp(self.ssn_gamma, min=1.0, max=1.5)
                A_w = A_w * gamma_safe
            else:
                with torch.no_grad():
                    _, sigma_max = _spectral_normalize_power_iter(A_w.detach())

            # Value map
            B_v = torch.cholesky_solve(
                C_v.transpose(-1, -2), L).transpose(-1, -2)

            # Whiten query
            zq_perm = zq_f32.permute(0, 2, 3, 1)
            w_q = torch.linalg.solve_triangular(L, zq_perm, upper=False)

            # Spectral filter
            if self.use_schur_filter:
                w_f = _schur_spectral_filter(A_w, w_q, self.eigen_threshold)
            else:
                w_f = _power_spectral_filter(A_w, w_q, self.power_K)

            # Un-whiten and retrieve
            z_f = L @ w_f
            y_hat = (B_v @ z_f).permute(0, 3, 1, 2)

        y_hat = self.eta * y_hat.to(z.dtype)

        # Diagnostics
        G_diag = G_tilde.diagonal(dim1=-2, dim2=-1)
        diag = {
            'lambda_min_G': G_diag.min().item(),
            'cond_proxy': (G_diag.max() / G_diag.min().clamp(min=1e-10)).item(),
            'sigma_max_pre': sigma_max.max().item(),
            'ssn_gamma': torch.clamp(self.ssn_gamma, min=1.0, max=1.5).item(),
        }
        with torch.no_grad():
            try:
                eigs = torch.linalg.eigvals(A_w.detach())
                eig_mags = eigs.abs()
                diag['eig_max'] = eig_mags.max().item()
                diag['eig_mean'] = eig_mags.mean().item()
                diag['n_persistent'] = (eig_mags > self.eigen_threshold).float().mean().item()
                diag['eig_var'] = eig_mags.var().item()
                effective_mags = eig_mags ** self.power_K
                diag['persist_eff'] = (effective_mags > 0.1).float().mean().item()
            except Exception:
                pass
        return y_hat, diag


class SwiGLUMLP(nn.Module):
    def __init__(self, d, expand=8/3):
        super().__init__()
        d_ff = ((int(d * expand) + 63) // 64) * 64
        self.norm = nn.LayerNorm(d)
        self.w1 = nn.Linear(d, d_ff, bias=False)
        self.w2 = nn.Linear(d, d_ff, bias=False)
        self.w3 = nn.Linear(d_ff, d, bias=False)

    def forward(self, x):
        h = self.norm(x)
        return x + self.w3(F.silu(self.w1(h)) * self.w2(h))


class Mamba3LM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList()
        self.mlps = nn.ModuleList()
        for _ in range(cfg.n_layers):
            self.blocks.append(Mamba3Block(cfg))
            self.mlps.append(SwiGLUMLP(cfg.d_model))
        self.norm_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.name = "Mamba-3"

    def forward(self, x, prefix_mask=None, phase_delta=None, action_mask=None):
        h = self.embed(x)
        for blk, mlp in zip(self.blocks, self.mlps):
            h, _ = blk(h, phase_delta=phase_delta)
            h = mlp(h)
        return self.lm_head(self.norm_f(h)), {}


class Mamba3CGSKALM(nn.Module):
    """v7: Gated SKA with independent features and action masking."""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList()
        self.mlps = nn.ModuleList()
        self.ska_modules = nn.ModuleDict()
        self.ska_norms = nn.ModuleDict()  # LayerNorm before SKA input
        for i in range(cfg.n_layers):
            self.blocks.append(Mamba3Block(cfg))
            self.mlps.append(SwiGLUMLP(cfg.d_model))
            if i in cfg.ska_layers:
                self.ska_modules[str(i)] = SKAModule(
                    N=cfg.d_state, P=cfg.d_head, rank=cfg.ska_rank,
                    ridge_eps=cfg.ska_ridge, scale=cfg.ska_scale,
                    power_K=cfg.power_K, use_cg=cfg.use_cg,
                    use_harmonic=cfg.use_harmonic,
                    use_schur_filter=cfg.ska_use_schur_filter,
                    eigen_threshold=cfg.ska_eigen_threshold,
                    use_spectral_norm=cfg.ska_spectral_norm,
                    use_seq_max_norm=cfg.ska_seq_max_norm,
                    use_independent=cfg.ska_independent,
                    d_model=cfg.d_model, n_heads=cfg.n_heads)
                if cfg.ska_independent:
                    self.ska_norms[str(i)] = nn.LayerNorm(cfg.d_model)
        self.norm_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # v7: Gated connection — "switch, don't blend"
        self.ska_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        if cfg.ska_gated:
            self.gate_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=True)
            # Init gate to sigmoid(-2) ≈ 0.12 — SKA starts with small influence,
            # learns to open the gate as it becomes useful
            nn.init.zeros_(self.gate_proj.weight)
            nn.init.constant_(self.gate_proj.bias, -2.0)
        else:
            self.gate_proj = None
        self.name = "Mamba-3 + CG + SKA"

    def forward(self, x, prefix_mask=None, phase_delta=None, action_mask=None):
        h = self.embed(x)
        diag_all = {}
        for i, (blk, mlp) in enumerate(zip(self.blocks, self.mlps)):
            has_ska = str(i) in self.ska_modules
            need_aux = has_ska and not self.cfg.ska_independent
            h, aux = blk(h, phase_delta=phase_delta, return_aux=need_aux)

            if has_ska and prefix_mask is not None:
                ska_mod = self.ska_modules[str(i)]
                if self.cfg.ska_independent:
                    # Independent: SKA gets the normalized residual stream
                    h_normed = self.ska_norms[str(i)](h)
                    ska_out, diag = ska_mod(
                        h_normed, None, None, prefix_mask,
                        action_mask=action_mask)
                else:
                    # Legacy: SKA gets Mamba's rotary features
                    if aux is not None:
                        ska_out, diag = ska_mod(
                            aux['k_rot'], aux['q_rot'], aux['v'], prefix_mask,
                            action_mask=action_mask)
                    else:
                        ska_out = None

                if ska_out is not None:
                    B, T, _H, _P = ska_out.shape
                    ska_signal = self.ska_proj(ska_out.reshape(B, T, self.cfg.d_model))

                    if self.gate_proj is not None:
                        # v7 gated: model learns when to trust SKA vs Mamba
                        gate = torch.sigmoid(self.gate_proj(h))
                        h = (1.0 - gate) * h + gate * ska_signal
                    else:
                        # Legacy additive
                        h = h + ska_signal
                    diag_all[f'ska_L{i}'] = diag

            h = mlp(h)
        return self.lm_head(self.norm_f(h)), diag_all
