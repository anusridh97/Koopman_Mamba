"""
ska.py -- Structured Kernel Attention with chunk-causal masking.

CHANGES in the 440M rewrite (see CHANGES_440M.md):
  * eta and gamma are FIXED constants (registered buffers), not nn.Parameters.
    gamma is NOT clamped to [1.0,1.5] anymore -- it is just 1.0 (operator left
    at the spectral-norm cap). Rationale: both are absorbable by out_proj, so
    learning them adds a degenerate loss direction + lets the model evade
    weight decay on the output scale.
  * out_proj is no longer zero-initialized. It is full-rank (small std) gated
    by a LayerScale per-channel diagonal initialized to a small nonzero value.
    This breaks the exact-zero gradient stall so SKA internals receive gradient
    from step 1, while staying near-zero in magnitude for early stability.

Two backends, selected automatically (unchanged):
  - 'triton':  Fused Triton kernel for post-Cholesky matmul chain
  - 'pytorch': Batched PyTorch (always available)
"""

import math
import torch
import torch.nn as nn
from contextlib import nullcontext

# Verified parity components (match echo_jax.py math):
#   ska_core  -- whitened L^{-1}M L^{-T} forward + custom O(K r^2) backward
#                ("It Cancels"; Cholesky never differentiated)
#   chunk_stats -- beta-gated, strictly-causal sufficient statistics
from koopman_lm.kernels.ska_operator import ska_core
from koopman_lm.kernels.chunk_stats import (
    chunk_stats as _causal_chunk_stats, symmetric_key_value, causal_normalize)

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


def _squash(raw, lo, hi):
    """Smooth bound to (lo, hi) via sigmoid -- matches echo_jax.py's _squash."""
    return lo + (hi - lo) * torch.sigmoid(raw)


def _raw_init(val, lo, hi):
    """Inverse of _squash: raw value s.t. _squash(raw, lo, hi) == val."""
    s = (val - lo) / (hi - lo)
    return math.log(s / (1.0 - s))


# ============================================================================
# Shared: batched statistics + cumsum + cholesky (STANDARD strategy)
# ============================================================================

def _compute_chunk_stats_and_cholesky(z_f, zq_f, v_f, r, H, P, CS, ridge_eps):
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

    G_chunks = torch.einsum('bcthr,bcths->bchrs', z_c, z_c)
    M_chunks = torch.einsum('bcthr,bcths->bchrs',
                            z_c[:, :, 1:], z_c[:, :, :-1])
    C_chunks = torch.einsum('bcthp,bcthr->bchpr', v_c, z_c)

    if C > 1:
        M_boundary = torch.einsum('bchr,bchs->bchrs',
                                  z_c[:, 1:, 0], z_c[:, :-1, -1])

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

    return L_flat, M_flat, C_flat, zq_flat, (B, C, H, P, CS, T, T_padded, pad_len)


# ============================================================================
# Phase 1: Batched PyTorch post-Cholesky
# ============================================================================

def _post_cholesky_pytorch(L_flat, M_flat, C_flat, zq_flat, gamma_value,
                           power_K, shapes):
    """Post-Cholesky operations using batched PyTorch calls.

    gamma_value is now a plain float constant (default 1.0), not a clamped
    learnable parameter. With gamma=1.0 the operator is left exactly at the
    spectral-norm cap (radius<=1).
    """
    B, C, H, P, CS, T, T_padded, pad_len = shapes
    r = L_flat.shape[-1]

    Aw_T = torch.cholesky_solve(M_flat.transpose(-1, -2), L_flat)
    A_w = Aw_T.transpose(-1, -2)
    A_w, _ = _spectral_normalize_power_iter(A_w)
    # gamma_value: python float (fixed regime; skip multiply if 1.0) OR a tensor
    # (learnable+clamped baseline; always multiply).
    if isinstance(gamma_value, float):
        if gamma_value != 1.0:
            A_w = A_w * gamma_value
    else:
        A_w = A_w * gamma_value

    Bv_T = torch.cholesky_solve(C_flat.transpose(-1, -2), L_flat)
    B_v = Bv_T.transpose(-1, -2)

    w_q = torch.linalg.solve_triangular(L_flat, zq_flat, upper=False)

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
        Aw_ptr, Bv_ptr, Wq_ptr, L_ptr, Y_ptr,
        actual_P: tl.constexpr, actual_CS: tl.constexpr,
        R_PAD: tl.constexpr, P_PAD: tl.constexpr, CS_PAD: tl.constexpr,
        power_K: tl.constexpr,
    ):
        pid = tl.program_id(0)
        aw_base = pid * R_PAD * R_PAD
        l_base = pid * R_PAD * R_PAD
        bv_base = pid * P_PAD * R_PAD
        wq_base = pid * R_PAD * CS_PAD
        y_base = pid * P_PAD * CS_PAD

        ri = tl.arange(0, R_PAD)
        rj = tl.arange(0, R_PAD)
        pi = tl.arange(0, P_PAD)
        ci = tl.arange(0, CS_PAD)

        Aw = tl.load(Aw_ptr + aw_base + ri[:, None] * R_PAD + rj[None, :])
        wq = tl.load(Wq_ptr + wq_base + ri[:, None] * CS_PAD + ci[None, :])

        w_f = wq
        for _k in range(power_K):
            w_f = tl.dot(Aw, w_f)

        L = tl.load(L_ptr + l_base + ri[:, None] * R_PAD + rj[None, :])
        z_out = tl.dot(L, w_f)

        Bv = tl.load(Bv_ptr + bv_base + pi[:, None] * R_PAD + rj[None, :])
        y = tl.dot(Bv, z_out)

        p_mask = pi < actual_P
        c_mask = ci < actual_CS
        mask = p_mask[:, None] & c_mask[None, :]
        tl.store(Y_ptr + y_base + pi[:, None] * CS_PAD + ci[None, :], y, mask=mask)


    def _post_cholesky_triton(L_flat, M_flat, C_flat, zq_flat, gamma_value,
                              power_K, shapes):
        B, C, H, P, CS, T, T_padded, pad_len = shapes
        r = L_flat.shape[-1]
        BCH = L_flat.shape[0]
        device = L_flat.device
        dtype = L_flat.dtype

        Aw_T = torch.cholesky_solve(M_flat.transpose(-1, -2), L_flat)
        A_w = Aw_T.transpose(-1, -2).contiguous()
        A_w, _ = _spectral_normalize_power_iter(A_w)
        if isinstance(gamma_value, float):
            if gamma_value != 1.0:
                A_w = A_w * gamma_value
        else:
            A_w = A_w * gamma_value

        Bv_T = torch.cholesky_solve(C_flat.transpose(-1, -2), L_flat)
        B_v = Bv_T.transpose(-1, -2).contiguous()

        w_q = torch.linalg.solve_triangular(L_flat, zq_flat, upper=False)
        w_q = w_q.contiguous()
        L_flat = L_flat.contiguous()

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
            A_w_pad = A_w; L_pad = L_flat; Bv_pad = B_v; wq_pad = w_q

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
    """Structured Kernel Attention with exact-scan and legacy chunked backends.

    The recommended quality path is ``prefix_scan=True``. It is strictly causal
    at token resolution; ``prefix_scan_block_size`` controls scheduling only.
    """
    def __init__(self, d_model, n_heads, rank=48, head_dim=None,
                 ridge_eps=1e-3, scale=1.5, power_K=2, chunk_size=64,
                 backend='auto', chunk_strategy='standard',
                 overlap_fraction=0.5, decay_alpha=0.95,
                 eta_learnable=False, eta_value=1.0, eta_bounds=None,
                 gamma_learnable=False, gamma_value=1.0, gamma_clamp=None,
                 gamma_bounds=None,
                 layerscale=True, layerscale_init=1e-4, out_proj_std=0.02,
                 exact_intrachunk=False, inverse_cholesky=False,
                 prefix_scan=False, prefix_scan_block_size=32,
                 prefix_scan_jitter=0.0, norm_clip_c=None):
        super().__init__()
        self.rank = rank
        self.exact_intrachunk = exact_intrachunk
        self.prefix_scan = bool(prefix_scan)
        self.prefix_scan_block_size = int(prefix_scan_block_size)
        self.prefix_scan_jitter = float(prefix_scan_jitter)
        if self.prefix_scan_block_size <= 0:
            raise ValueError("prefix_scan_block_size must be positive")
        # Legacy small-rank exact path that materializes per-token inverse-
        # Cholesky statistics. The prefix-scan backend supersedes it for new
        # quality runs; this branch remains for checkpoint compatibility.
        self.inverse_cholesky = inverse_cholesky
        if inverse_cholesky:
            assert rank <= 64, (
                f"inverse_cholesky path stores per-token (r x r) stats; "
                f"rank={rank} > 64 is not supported (use rank <= 32).")
            if rank > 32:
                import warnings
                warnings.warn(
                    f"inverse_cholesky with rank={rank}: per-token stats are "
                    "(B,T,H,r,r); rank <= 32 is recommended for memory.",
                    RuntimeWarning)
        self.ridge_eps = ridge_eps
        self.power_K = power_K
        # None -> per-token L2 (legacy); float -> causal norm-clip threshold c
        self.norm_clip_c = norm_clip_c
        self.H = n_heads
        self.P = head_dim or (d_model // n_heads)
        self.d_model = d_model
        self.chunk_size = chunk_size
        # NOTE: the JAX-parity forward() calls _causal_chunk_stats (strict
        # causal beta-gated stats) DIRECTLY and does NOT route through
        # _get_chunk_stats(). So chunk_strategy='overlap'/'decay' and the old
        # _post_cholesky_* backend are DEAD in the active path -- they exist
        # only for the legacy ska_fast/baseline route. Setting chunk_strategy
        # to anything but 'standard' has NO effect on the 440M forward. Warn so
        # nobody believes they are ablating overlap/decay when they are not.
        if chunk_strategy not in ('standard',):
            import warnings
            warnings.warn(
                f"ska chunk_strategy={chunk_strategy!r} is IGNORED by the "
                "JAX-parity forward (which uses strict causal stats). It has "
                "no effect unless you route through the legacy _get_chunk_stats "
                "path.", RuntimeWarning)
        self.chunk_strategy = chunk_strategy
        self.overlap_fraction = overlap_fraction
        self.decay_alpha = decay_alpha
        self.layerscale = layerscale

        # NOTE: `backend` here selects the legacy post-Cholesky matmul chain
        # ('triton' vs 'pytorch'), which is DEAD when prefix_scan=True (see
        # the chunk_strategy warning above) -- it is resolved eagerly against
        # local triton availability purely for the old chunked path and for
        # extra_repr(). It must NOT be reused to select ska_prefix_scan's
        # backend (whose allowed values are auto/cuda/cuda_prefix/reference/
        # pytorch, and which has no 'triton' implementation): doing so used
        # to make any prefix_scan=True model with the default backend='auto'
        # crash on any machine with triton installed, since 'auto' resolved
        # to 'triton' here and ska_prefix_scan(backend='triton') raises
        # ValueError. Keep the raw, unresolved string for that call so
        # prefix_scan gets its own real 'auto' (try CUDA, else fall back)
        # semantics instead of this module's triton/pytorch choice.
        self._prefix_scan_backend = backend
        if backend == 'auto':
            self.backend = 'triton' if _TRITON_AVAILABLE else 'pytorch'
        else:
            self.backend = backend

        self.key_proj = nn.Linear(d_model, n_heads * rank, bias=False)
        self.query_proj = nn.Linear(d_model, n_heads * rank, bias=False)
        self.value_proj = nn.Linear(d_model, n_heads * self.P, bias=False)
        self.out_proj = nn.Linear(n_heads * self.P, d_model, bias=False)
        # beta write-gate (GDN-style causal normalization, matches echo_jax.py).
        # bias init 0 -> sigmoid -> beta=0.5 at start.
        self.beta_proj = nn.Linear(d_model, n_heads, bias=True)

        nn.init.orthogonal_(self.key_proj.weight)
        nn.init.orthogonal_(self.query_proj.weight)
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.zeros_(self.beta_proj.weight)
        nn.init.zeros_(self.beta_proj.bias)        # beta = sigmoid(0) = 0.5 at init
        # out_proj is now full-rank (small std), NOT zeros -- LayerScale handles
        # the near-zero start so internals still receive gradient from step 1.
        # out_proj: full-rank small-std when layerscale gates it (440M),
        # exact-zero otherwise (baseline published behavior).
        if layerscale:
            nn.init.normal_(self.out_proj.weight, mean=0.0, std=out_proj_std)
        else:
            nn.init.zeros_(self.out_proj.weight)

        # LayerScale per-channel gate (replaces exact-zero residual init)
        if layerscale:
            self.layerscale_gate = nn.Parameter(
                torch.full((d_model,), float(layerscale_init)))
        else:
            self.register_parameter('layerscale_gate', None)

        # eta: three regimes.
        #   squash (echo_jax.py parity): learnable raw param, smoothly bounded to
        #     eta_bounds via sigmoid, matching the JAX reference's _squash/_raw_init
        #     (eta in [1.4, 1.7], init 1.5) instead of an unconstrained parameter
        #     that can drift arbitrarily.
        #   learnable (unconstrained): plain nn.Parameter, no bound.
        #   fixed: buffer, constant.
        self.eta_bounds = eta_bounds
        if eta_bounds is not None:
            lo, hi = eta_bounds
            self.eta_raw = nn.Parameter(torch.tensor(_raw_init(eta_value, lo, hi)))
        elif eta_learnable:
            self.eta = nn.Parameter(torch.tensor(float(eta_value)))
        else:
            self.register_buffer('eta', torch.tensor(float(eta_value)))

        # gamma: three regimes.
        #   squash (echo_jax.py parity): learnable raw param, smoothly bounded to
        #     gamma_bounds via sigmoid (gamma in [0.5, 1.5], init 0.7) -- matches
        #     the JAX reference exactly, including that gamma can start BELOW 1.0
        #     (a damped operator), not just restore variance from spectral norm.
        #   baseline: learnable nn.Parameter, hard-clamped to gamma_clamp in forward.
        #   440M:     fixed buffer (no clamp). _gamma_const set for the fast
        #             python-float branch that skips the multiply when ==1.0.
        self.gamma_bounds = gamma_bounds
        self.gamma_learnable = gamma_learnable
        self.gamma_clamp = gamma_clamp
        if gamma_bounds is not None:
            lo, hi = gamma_bounds
            self.ssn_gamma = nn.Parameter(torch.tensor(_raw_init(gamma_value, lo, hi)))
            self._gamma_const = None
        elif gamma_learnable:
            self.ssn_gamma = nn.Parameter(torch.tensor(float(gamma_value)))
            self._gamma_const = None        # resolved per-forward (clamped)
        else:
            self.register_buffer('ssn_gamma', torch.tensor(float(gamma_value)))
            self._gamma_const = float(gamma_value)

    def _resolve_eta(self):
        """Return the eta to apply this forward (squashed tensor, unconstrained
        tensor, or fixed buffer, depending on construction regime)."""
        if self.eta_bounds is not None:
            lo, hi = self.eta_bounds
            return _squash(self.eta_raw, lo, hi)
        return self.eta

    def _resolve_gamma(self):
        """Return the gamma to apply this forward.
        Squash regime -> smoothly bounded tensor (echo_jax.py parity).
        Fixed regime -> python float (enables the skip-when-1.0 fast path).
        Learnable regime -> clamped tensor (keeps grad), as in the baseline."""
        if self.gamma_bounds is not None:
            lo, hi = self.gamma_bounds
            return _squash(self.ssn_gamma, lo, hi)
        if not self.gamma_learnable:
            return self._gamma_const
        if self.gamma_clamp is not None:
            lo, hi = self.gamma_clamp
            return torch.clamp(self.ssn_gamma, min=lo, max=hi)
        return self.ssn_gamma

    def _get_chunk_stats(self, z_f, zq_f, v_f):
        r = self.rank
        H, P = self.H, self.P
        CS = self.chunk_size
        if self.chunk_strategy == 'overlap':
            from koopman_lm.kernels.adaptive_chunking import compute_chunk_stats_overlap
            return compute_chunk_stats_overlap(
                z_f, zq_f, v_f, r, H, P, CS, self.ridge_eps,
                overlap_fraction=self.overlap_fraction)
        elif self.chunk_strategy == 'decay':
            from koopman_lm.kernels.adaptive_chunking import compute_chunk_stats_decay
            return compute_chunk_stats_decay(
                z_f, zq_f, v_f, r, H, P, CS, self.ridge_eps,
                decay_alpha=self.decay_alpha)
        else:
            return _compute_chunk_stats_and_cholesky(
                z_f, zq_f, v_f, r, H, P, CS, self.ridge_eps)

    def forward(self, hidden_states):
        B, T, _ = hidden_states.shape
        r = self.rank
        H, P = self.H, self.P

        z = self.key_proj(hidden_states).reshape(B, T, H, r)
        zq = self.query_proj(hidden_states).reshape(B, T, H, r)
        v = self.value_proj(hidden_states).reshape(B, T, H, P)
        beta = torch.sigmoid(self.beta_proj(hidden_states))            # (B,T,H)

        ctx = torch.amp.autocast('cuda', enabled=False) if hidden_states.is_cuda \
              else nullcontext()

        with ctx:
            z_f = z.float()
            zq_f = zq.float()
            v_f = v.float()
            beta_f = beta.float()

            # Causal normalization (matches echo_jax.py): per-token L2 on
            # key/query. NO non-causal sequence-max.
            z_n = causal_normalize(z_f, self.norm_clip_c)
            zq_n = causal_normalize(zq_f, self.norm_clip_c)
            # v1.1 SYMMETRIC sqrt(beta) key/value: x=sqrt(beta)*z fed into BOTH
            # key slots, vbar=sqrt(beta)*v. G,C invariant; M/boundary become the
            # contractive cross-weight sqrt(beta_t beta_{t-1}). One helper, every
            # site -> no norm-based beta re-inference (the train/decode trap).
            x_n, v_w = symmetric_key_value(z_n, beta_f, v_f)

            if self.prefix_scan:
                # Exact two-level prefix scan.  Raw sufficient statistics form
                # the associative block monoid; each block is then evaluated
                # with exact O(r^2) rank-1 Cholesky writes.
                from koopman_lm.kernels.prefix_scan import ska_prefix_scan
                Y = ska_prefix_scan(
                    x_n, zq_n, v_w, self.ridge_eps, self.power_K,
                    self.prefix_scan_block_size, self.prefix_scan_jitter,
                    backend=self._prefix_scan_backend)
                gamma_apply = self._resolve_gamma()
                if isinstance(gamma_apply, float):
                    if gamma_apply != 1.0:
                        Y = Y * (gamma_apply ** self.power_K)
                else:
                    Y = Y * (gamma_apply ** self.power_K)
                y_hat = Y
            elif self.inverse_cholesky:
                # EXACT per-token processing via the inverse-Cholesky
                # representation (small rank). Every token reads the full
                # exclusive prefix -- including t-1, which the chunked path
                # never sees -- and the whitened core is pure batched matmul
                # against P = L^{-1} with NO spectral power iteration (the
                # symmetric sqrt(beta) keys make A_w contractive). This
                # replaces chunk-64 stats + the cross-chunk boundary term.
                from koopman_lm.kernels.inverse_cholesky import (
                    ska_exact_inverse_cholesky)
                Y = ska_exact_inverse_cholesky(
                    x_n, zq_n, v_w, self.ridge_eps, self.power_K)  # (B,T,H,P)
                gamma_apply = self._resolve_gamma()
                if isinstance(gamma_apply, float):
                    if gamma_apply != 1.0:
                        Y = Y * (gamma_apply ** self.power_K)
                else:
                    Y = Y * (gamma_apply ** self.power_K)
                y_hat = Y
            elif self.exact_intrachunk:
                # EXACT per-token causal stats (across + within chunk). Fixes
                # within-chunk staleness; reuses the same verified ska_core.
                # Cost: B*T*H solves instead of B*nchunks*H.
                from koopman_lm.kernels.chunk_stats_exact import exact_stats
                from koopman_lm.kernels.factor_scan import all_prefix_chol, ska_core_given_L
                Gf, Mf, Cf, qf, (Be, Te, He, Pe) = exact_stats(
                    x_n, x_n, zq_n, v_w, self.ridge_eps)
                # factor scan over per-token update vectors (numerics-only):
                # w_t = x_n = sqrt(beta_t) z_t  =>  w w^T = beta z z^T (G increment).
                # SAME symmetric key as the stats -> L stays the factor of G.
                w = x_n.permute(0, 2, 1, 3).reshape(Be * He, Te, r)
                # exact_stats builds Gf = (ridge + 1e-4)*I + prefix; factor the
                # SAME matrix so backward's L matches the saved Gf exactly.
                Lf = all_prefix_chol(w, self.ridge_eps + 1e-4, downsweep='qr')
                Lf = Lf.reshape(Be, He, Te, r, r).permute(0, 2, 1, 3, 4) \
                       .reshape(Be * Te * He, r, r)
                Y = ska_core_given_L(Gf, Mf, Cf, qf, Lf, self.power_K)  # (N,P,1)
                gamma_apply = self._resolve_gamma()
                if isinstance(gamma_apply, float):
                    if gamma_apply != 1.0:
                        Y = Y * (gamma_apply ** self.power_K)
                else:
                    Y = Y * (gamma_apply ** self.power_K)
                y_hat = Y.reshape(Be, Te, He, Pe)                    # (B,T,H,P)
            else:
                # Strictly-causal CHUNKED statistics (exclusive chunk boundary).
                Gf, Mf, Cf, qf, shp = _causal_chunk_stats(
                    x_n, x_n, zq_n, v_w, self.ridge_eps, self.chunk_size)
                Y = ska_core(Gf, Mf, Cf, qf, self.power_K)            # (N,P,CS)
                Bc, nc, Hc, Pc, CS, Tt, pad = shp
                gamma_apply = self._resolve_gamma()
                if isinstance(gamma_apply, float):
                    gscale = gamma_apply ** self.power_K
                    Y = Y * gscale if gscale != 1.0 else Y
                else:
                    Y = Y * (gamma_apply ** self.power_K)
                Y = Y.reshape(Bc, nc, Hc, Pc, CS).permute(0, 1, 4, 2, 3)
                Y = Y.reshape(Bc, nc * CS, Hc, Pc)[:, :Tt]            # (B,T,H,P)
                y_hat = Y

        y_hat = self._resolve_eta() * y_hat.to(hidden_states.dtype)
        output = self.out_proj(y_hat.reshape(B, T, H * P))
        if self.layerscale_gate is not None:
            output = output * self.layerscale_gate
        return output

    def extra_repr(self):
        with torch.no_grad():
            eta_val = float(self._resolve_eta())
            gamma_val = self._resolve_gamma()
            gamma_val = gamma_val if isinstance(gamma_val, float) else float(gamma_val)
        parts = [
            f'd_model={self.d_model}', f'n_heads={self.H}', f'rank={self.rank}',
            f'chunk_size={self.chunk_size}', f'backend={self.backend}',
            f'chunk_strategy={self.chunk_strategy}',
            f'eta={eta_val:.4f}', f'gamma={gamma_val:.4f}',
            f'eta_bounds={self.eta_bounds}', f'gamma_bounds={self.gamma_bounds}',
            f'layerscale={self.layerscale}',
            f'inverse_cholesky={self.inverse_cholesky}',
            f'prefix_scan={self.prefix_scan}',
            f'prefix_scan_block_size={self.prefix_scan_block_size}',
        ]
        if self.chunk_strategy == 'overlap':
            parts.append(f'overlap_fraction={self.overlap_fraction}')
        elif self.chunk_strategy == 'decay':
            parts.append(f'decay_alpha={self.decay_alpha}')
        return ', '.join(parts)
