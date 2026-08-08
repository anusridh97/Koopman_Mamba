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

forward() selects one of four kernels (prefix_scan / inverse_cholesky /
exact_intrachunk / chunked); all of them compute the WHITENED operator
L^-1 M L^-T (kernels/lin_alg.py::whiten_M). There is no longer a
post-Cholesky triton/pytorch matmul chain to choose between -- that route
computed M G^-1 instead and was unreachable; see the commit removing
_post_cholesky_pytorch / _post_cholesky_triton / _get_chunk_stats.
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
from koopman_lm.kernels.lin_alg import whiten_M, spec_w

# ============================================================================
# Backend detection
# ============================================================================

# Vestigial: no triton kernel lives in this file anymore, so this probe only
# decides the cosmetic backend='auto' -> 'triton'/'pytorch' string that
# extra_repr() prints. Kept so that string does not silently change meaning on
# triton-equipped machines; see extra_repr and the __init__ note on `backend`.
_TRITON_AVAILABLE = False
try:
    import triton
    _TRITON_AVAILABLE = True
except ImportError:
    pass


# ============================================================================
# Shared utilities
# ============================================================================

def _squash(raw, lo, hi):
    """Smooth bound to (lo, hi) via sigmoid -- matches echo_jax.py's _squash."""
    return lo + (hi - lo) * torch.sigmoid(raw)


def _raw_init(val, lo, hi):
    """Inverse of _squash: raw value s.t. _squash(raw, lo, hi) == val."""
    s = (val - lo) / (hi - lo)
    return math.log(s / (1.0 - s))


def _spectral_radius(A, n_iters=12):
    """Spectral radius (max |eigenvalue|) of a batch of square matrices.

    Diagnostics-only, and SPEED-CRITICAL: torch.linalg.eigvals (the general
    non-symmetric eig) is extremely slow when called on many small matrices and
    dominated the diagnostic step cost. We instead use batched power iteration
    (just matmuls -> fast on GPU): iterate v into the dominant invariant
    subspace, then return ||A v|| for unit v. This is exact for a dominant real
    eigenvalue, and for a complex-conjugate 2x2 block (which acts as rho *
    rotation, norm-preserving) it also yields rho -- accurate enough for a
    health metric. A is (..., r, r); returns (...,).
    """
    v = torch.randn(*A.shape[:-1], 1, device=A.device, dtype=A.dtype)
    v = v / v.norm(dim=-2, keepdim=True).clamp(min=1e-12)
    for _ in range(n_iters):
        v = A @ v
        v = v / v.norm(dim=-2, keepdim=True).clamp(min=1e-12)
    return (A @ v).norm(dim=-2).squeeze(-1)


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
        # chunk_strategy / overlap_fraction / decay_alpha are INERT. forward()
        # calls _causal_chunk_stats (strict causal beta-gated stats) directly;
        # the legacy route that consumed these knobs has been deleted, so there
        # is no longer any path that reads them. They survive as attributes
        # only because KoopmanLMConfig still declares the matching fields and
        # run/resolve.py validates materialized spec.yaml field-for-field --
        # dropping them would make every archived run's spec unloadable. Warn
        # so nobody believes they are ablating overlap/decay when they are not.
        # (code-tests/test_ska_dead_route_inert.py pins the inertness.)
        if chunk_strategy not in ('standard',):
            import warnings
            warnings.warn(
                f"ska chunk_strategy={chunk_strategy!r} is IGNORED: the "
                "forward uses strict causal stats unconditionally and the "
                "overlap/decay route no longer exists. Setting this has no "
                "effect on any output.", RuntimeWarning)
        self.chunk_strategy = chunk_strategy
        self.overlap_fraction = overlap_fraction
        self.decay_alpha = decay_alpha
        self.layerscale = layerscale

        # NOTE: `backend` is half-live. The RAW string is real: it is handed to
        # ska_prefix_scan below. The eagerly RESOLVED self.backend used to pick
        # the legacy post-Cholesky matmul chain ('triton' vs 'pytorch'); that
        # chain is gone, so self.backend now feeds extra_repr() and nothing
        # else. It must NOT be reused to select ska_prefix_scan's
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

    @torch.no_grad()
    def collect_diagnostics(self, hidden_states, max_batch=2):
        """Per-(batch, chunk, head) SKA health metrics, computed off the hot
        path but FAITHFUL to the operators the training forward applies.

        Fidelity: this reuses the SAME beta-gated, strictly-causal chunk
        statistics the forward uses (`chunk_stats`), the SAME v1.1 symmetric
        sqrt(beta) key/value convention (`causal_normalize` +
        `symmetric_key_value` -- x = sqrt(beta) * key fed into BOTH key slots,
        vbar = sqrt(beta) * value), and forms each chunk's operator exactly as
        `ska_core` does -- A_eff = gamma * alpha * (L^-1 M L^-T), G/M being the
        EXCLUSIVE-prefix per-chunk sufficient statistics (Eqs. 7-10). gamma is
        included because the forward scales Y by gamma^K after ska_core
        applies (alpha W)^K -- so the operator each filter step actually
        applies is gamma * alpha * W (a no-op in the fixed gamma=1.0 regime;
        material in the squash/clamp learnable regimes, resolved the same way
        forward() resolves it via `_resolve_gamma`). So every operator
        measured here is one a real query sees, not a non-causal whole-sequence
        summary.

        NOTE: nothing is reduced here. The three operator metrics are returned
        as full (B, n_chunks, H) tensors so downstream (plotting / wandb) can
        decide how to aggregate -- per head, per chunk, or the whole pool.
        Chunk 0 has an empty exclusive prefix (G=ridge I, M=0 -> zero operator);
        it is kept (index 0) and excluded by the monitor's summaries by default.

        NOTE: this always reports the CHUNKED-path operators (the same
        beta-gated statistics family `forward()`'s non-exact branch consumes)
        as the bounded-cost health view, regardless of which of the four
        forward strategies (chunked / exact_intrachunk / inverse_cholesky /
        prefix_scan) is actually selected. prefix_scan (the recommended
        quality path) and inverse_cholesky compute per-token, not per-chunk,
        operators via different kernels entirely -- this method does not
        reach into those; it reports the chunk-level approximation as a
        cheap, bounded-cost proxy for the operator family they all share.

        Detached, fp32, no value readout. `max_batch` caps the batch used for
        the per-chunk eigendecompositions so cost is bounded regardless of the
        training batch size. Returns GPU tensors; caller does the CPU sync
        (see experimentation.training.diagnostics.SKAHealthMonitor).

        Returns dict:
          spectral_radius : (B, nc, H) max|eig(A_eff)| per instance.
                            Healthy ~[0.3, 0.95]; ~0 = unlearned; >1 = unstable.
          lambda_min      : (B, nc, H) smallest eig(G_tilde) per instance.
                            Pinned at the ridge floor => rank-deficient keys.
          gap             : (B, nc, H) ||A_eff^K - A_eff|| / ||A_eff||.
                            >0.5 => the power filter dominates the operator.
          n_chunks        : int, number of chunks (chunk 0 = no history).
          gate_mag/beta_mean/outproj_norm/eta/gamma/ridge_eps : scalars.
        """
        r, H, P = self.rank, self.H, self.P
        ridge = float(self.ridge_eps)
        K = int(self.power_K)
        CS = self.chunk_size

        x = hidden_states
        if x.shape[0] > max_batch:
            x = x[:max_batch]
        B, T, _ = x.shape

        z = self.key_proj(x).reshape(B, T, H, r).float()
        zq = self.query_proj(x).reshape(B, T, H, r).float()
        v = self.value_proj(x).reshape(B, T, H, P).float()
        beta = torch.sigmoid(self.beta_proj(x)).float()                # (B,T,H)

        # SAME normalization + symmetric sqrt(beta) key/value convention the
        # forward uses (v1.1): G/C are own-weight and beta-invariant; M
        # becomes the contractive cross-weight sqrt(beta_t beta_{t-1}).
        z_n = causal_normalize(z, self.norm_clip_c)
        zq_n = causal_normalize(zq, self.norm_clip_c)
        x_n, v_w = symmetric_key_value(z_n, beta, v)

        # SAME strictly-causal, beta-gated, exclusive-prefix chunk statistics
        # the training forward consumes. Gf already carries ridge + jitter.
        Gf, Mf, Cf, qf, shp = _causal_chunk_stats(x_n, x_n, zq_n, v_w, ridge, CS)
        Bc, nc, Hc = shp[0], shp[1], shp[2]

        # Per-instance operator (N = B*nc*H), formed exactly like ska_core.
        L, info = torch.linalg.cholesky_ex(Gf)
        if (info > 0).any():
            eye = torch.eye(r, device=Gf.device, dtype=Gf.dtype)
            L_j, _ = torch.linalg.cholesky_ex(Gf + 1e-4 * eye)
            L = torch.where((info > 0).view(-1, 1, 1), L_j, L)

        W = whiten_M(L, Mf)                        # L^-1 M L^-T  (N,r,r)
        alpha = spec_w(W)                          # (N,1) detached spectral-norm scale
        gamma = self._resolve_gamma()              # float (fixed) or tensor (learnable)
        gamma_val = gamma if isinstance(gamma, float) else float(gamma.detach())
        A_eff = alpha.unsqueeze(-1) * W            # operator applied per filter step
        if gamma_val != 1.0:
            A_eff = A_eff * gamma_val

        radius = _spectral_radius(A_eff)                               # (N,)
        lam = torch.linalg.eigvalsh(Gf).amin(dim=-1)                   # (N,)
        A_K = torch.linalg.matrix_power(A_eff, K)
        gap = (A_K - A_eff).norm(dim=(-2, -1)) / (A_eff.norm(dim=(-2, -1)) + 1e-12)

        # (N,) -> (B, nc, H). No reduction: hand back the whole distribution.
        radius = radius.view(Bc, nc, Hc)
        lam = lam.view(Bc, nc, Hc)
        gap = gap.view(Bc, nc, Hc)

        eta = self._resolve_eta().detach().float().reshape(())
        gate = self.layerscale_gate
        gate_mag = gate.abs().mean() if gate is not None else eta.abs()

        return {
            'spectral_radius': radius.detach(),
            'lambda_min': lam.detach(),
            'gap': gap.detach(),
            'n_chunks': int(nc),
            'gate_mag': gate_mag.detach(),
            'beta_mean': beta.mean().detach(),
            'outproj_norm': self.out_proj.weight.detach().float().norm(),
            'eta': eta,
            'gamma': torch.tensor(gamma_val, device=x.device),
            'ridge_eps': torch.tensor(ridge, device=x.device),
        }

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
