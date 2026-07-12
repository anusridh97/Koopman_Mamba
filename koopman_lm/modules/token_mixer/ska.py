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

The legacy chunk_strategy ('overlap'/'decay') and _post_cholesky_* backends were
removed: forward() always uses the strictly-causal beta-gated chunk statistics
(chunk_stats) with the ska_core whitened operator.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import nullcontext

# Verified parity components (match echo_jax.py math):
#   ska_core  -- whitened L^{-1}M L^{-T} forward + custom O(K r^2) backward
#                ("It Cancels"; Cholesky never differentiated)
#   chunk_stats -- beta-gated, strictly-causal sufficient statistics
from koopman_lm.config import KoopmanLMConfig
from koopman_lm.modules.kernels.ska_operator import ska_core
from koopman_lm.modules.kernels.lin_alg import _whiten_M, _spec_w
from koopman_lm.modules.kernels.chunk_stats import chunk_stats as _causal_chunk_stats


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
    """Structured Kernel Attention with chunk-causal masking.

    eta/gamma fixed (buffers); LayerScale residual gate on out_proj.
    """
    def __init__(self, d_model, n_heads, rank=48, head_dim=None,
                 ridge_eps=1e-3, scale=1.5, power_K=2, chunk_size=64,
                 eta_learnable=False, eta_value=1.0, eta_bounds=None,
                 gamma_learnable=False, gamma_value=1.0, gamma_clamp=None,
                 gamma_bounds=None,
                 layerscale=True, layerscale_init=1e-4, out_proj_std=0.02,
                 exact_intrachunk=False):
        super().__init__()
        self.rank = rank
        self.exact_intrachunk = exact_intrachunk
        self.ridge_eps = ridge_eps
        self.power_K = power_K
        self.H = n_heads
        self.P = head_dim or (d_model // n_heads)
        self.d_model = d_model
        self.chunk_size = chunk_size
        self.layerscale = layerscale

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

    @staticmethod
    def _resolve_scale(param, bounds=None, const=None, clamp=None):
        """Shared regime resolver for the eta/gamma scalar scales.

        Precedence (mirrors the construction regimes above):
          bounds -> sigmoid-squashed tensor (echo_jax.py parity)
          const  -> python float           (fixed buffer; enables the
                                            skip-multiply-when-1.0 fast path)
          clamp  -> hard-clamped tensor    (baseline learnable gamma; keeps
                                            grad inside the range, kills it
                                            at the boundary — unlike squash)
          else   -> raw tensor             (unconstrained learnable)
        """
        if bounds is not None:
            return _squash(param, *bounds)
        if const is not None:
            return const
        if clamp is not None:
            return torch.clamp(param, min=clamp[0], max=clamp[1])
        return param

    def _resolve_eta(self):
        return self._resolve_scale(
            self.eta_raw if self.eta_bounds is not None else self.eta,
            bounds=self.eta_bounds)

    def _resolve_gamma(self):
        return self._resolve_scale(
            self.ssn_gamma, bounds=self.gamma_bounds,
            const=self._gamma_const, clamp=self.gamma_clamp)


    def forward(self, hidden_states):
        B, T, _ = hidden_states.shape
        r = self.rank
        H, P = self.H, self.P

        # Fused k/q/v projection (absorbed from the retired ska_fast patch):
        # one GEMM against the row-stacked [W_k; W_q; W_v] equals the three
        # separate GEMMs by block matmul, but issues a single kernel. The
        # weights stay three separate nn.Linear parameters so checkpoints,
        # the recurrent decode path (ska.key_proj(x) etc.), and the custom
        # inits in koopman_lm.py are untouched; autograd splits the gradient
        # back through the cat. beta_proj stays separate (it has a bias).
        fused_w = torch.cat([self.key_proj.weight, self.query_proj.weight,
                             self.value_proj.weight], dim=0)
        combined = F.linear(hidden_states, fused_w)
        z = combined[..., :H * r].reshape(B, T, H, r)
        zq = combined[..., H * r:2 * H * r].reshape(B, T, H, r)
        v = combined[..., 2 * H * r:].reshape(B, T, H, P)
        beta = torch.sigmoid(self.beta_proj(hidden_states))            # (B,T,H)

        ctx = torch.amp.autocast('cuda', enabled=False) if hidden_states.is_cuda \
              else nullcontext()

        with ctx:
            z_f = z.float()
            zq_f = zq.float()
            v_f = v.float()
            beta_f = beta.float()

            # Causal normalization (matches echo_jax.py): per-token L2 on
            # key/query + learned write gate beta. NO non-causal sequence-max.
            z_n = z_f * torch.rsqrt((z_f * z_f).sum(-1, keepdim=True) + 1e-12)
            zq_n = zq_f * torch.rsqrt((zq_f * zq_f).sum(-1, keepdim=True) + 1e-12)
            zb_n = beta_f.unsqueeze(-1) * z_n                          # write-gated key

            if self.exact_intrachunk:
                # EXACT per-token causal stats (across + within chunk). Fixes
                # within-chunk staleness; reuses the same verified ska_core.
                # Cost: B*T*H solves instead of B*nchunks*H.
                from koopman_lm.modules.kernels.chunk_stats_exact import exact_stats
                from koopman_lm.modules.kernels.factor_scan import all_prefix_chol, ska_core_given_L
                Gf, Mf, Cf, qf, (Be, Te, He, Pe) = exact_stats(
                    z_n, zb_n, zq_n, v_f, self.ridge_eps)
                # factor scan over per-token update vectors (numerics-only):
                # w_t = sqrt(beta_t) z_t  =>  w w^T = beta z z^T (the G increment)
                w = beta_f.clamp_min(0).sqrt().unsqueeze(-1) * z_n     # (B,T,H,r)
                w = w.permute(0, 2, 1, 3).reshape(Be * He, Te, r)
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
                    z_n, zb_n, zq_n, v_f, self.ridge_eps, self.chunk_size)
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
        statistics the forward uses (`chunk_stats`) and forms each chunk's
        operator exactly as `ska_core` does -- A_eff = gamma * alpha *
        (L^-1 M L^-T), G/M being the EXCLUSIVE-prefix per-chunk sufficient
        statistics (Eqs. 7-10). gamma is included because the forward scales Y
        by gamma^K after ska_core applies (alpha W)^K -- so the operator each
        filter step actually applies is gamma * alpha * W (a no-op in the fixed
        gamma=1.0 regime; material in the squash/clamp learnable regimes). So
        every operator measured here is one a real query sees, not a non-causal
        whole-sequence summary.

        NOTE: nothing is reduced here. The three operator metrics are returned
        as full (B, n_chunks, H) tensors so downstream (plotting / wandb) can
        decide how to aggregate -- per head, per chunk, or the whole pool.
        Chunk 0 has an empty exclusive prefix (G=ridge I, M=0 -> zero operator);
        it is kept (index 0) and excluded by the monitor's summaries by default.

        NOTE: when exact_intrachunk=True the training forward uses per-token
        exact stats; this method still reports the CHUNK-level operators (same
        beta-gated statistics family) as the bounded-cost health view.

        Detached, fp32, no value readout. `max_batch` caps the batch used for
        the per-chunk eigendecompositions so cost is bounded regardless of the
        training batch size. Returns GPU tensors; caller does the CPU sync
        (see koopman_lm.training.diagnostics.SKAHealthMonitor).

        Returns dict:
          spectral_radius : (B, nc, H) max|eig(A_eff)| per instance -- the
                            APPLIED (alpha-clamped) operator, so always <= 1.
                            Healthy ~[0.3, 0.95]; ~0 = unlearned.
          raw_spectral_radius : (B, nc, H) max|eig(gamma * W)|, the pre-clamp
                            operator radius. >1 => unstable (alpha clamp active).
          alpha           : (B, nc, H) spectral-norm clamp factor in (0, 1];
                            <1 means the clamp fired on that (chunk, head).
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
        beta = torch.sigmoid(self.beta_proj(x).float())                # (B,T,H)

        # Exact same normalization as forward(): per-token L2 on key/query,
        # beta-gated write key.
        z_n = z * torch.rsqrt((z * z).sum(-1, keepdim=True) + 1e-12)
        zq_n = zq * torch.rsqrt((zq * zq).sum(-1, keepdim=True) + 1e-12)
        zb_n = beta.unsqueeze(-1) * z_n

        # SAME strictly-causal, beta-gated, exclusive-prefix chunk statistics
        # the training forward consumes. Gf already carries ridge + jitter.
        Gf, Mf, Cf, qf, shp = _causal_chunk_stats(z_n, zb_n, zq_n, v, ridge, CS)
        Bc, nc, Hc = shp[0], shp[1], shp[2]

        # Per-instance operator (N = B*nc*H), formed exactly like ska_core.
        L, info = torch.linalg.cholesky_ex(Gf)
        if (info > 0).any():
            eye = torch.eye(r, device=Gf.device, dtype=Gf.dtype)
            L_j, _ = torch.linalg.cholesky_ex(Gf + 1e-4 * eye)
            L = torch.where((info > 0).view(-1, 1, 1), L_j, L)

        W = _whiten_M(L, Mf)                      # L^-1 M L^-T  (N,r,r)
        alpha = _spec_w(W)                        # (N,1) detached spectral-norm clamp
        gamma = self._resolve_gamma()             # float (fixed) or tensor (learnable)
        gamma_val = gamma if isinstance(gamma, float) else float(gamma.detach())
        A_eff = alpha.unsqueeze(-1) * W           # operator applied per filter step
        if gamma_val != 1.0:
            A_eff = A_eff * gamma_val

        radius = _spectral_radius(A_eff)                               # (N,)
        # RAW operator radius, BEFORE the alpha spectral-norm clamp. `alpha`
        # forces sigma_max(A_eff) <= 1, so radius(A_eff) can NEVER exceed 1 --
        # the ">1 = unstable" alarm is dead if read off A_eff. Since
        # A_eff = alpha * (gamma * W), radius(A_eff) = alpha * raw_radius, so we
        # recover raw_radius = radius / alpha from the SAME estimate (not a
        # second, independently-seeded power iteration -- that would break the
        # exact rad <= raw relationship). raw > 1 => the clamp is load-bearing /
        # the pre-clamp operator (gamma * W) is trying to blow up.
        alpha_flat = alpha.reshape(-1)                                 # (N,)
        raw_radius = radius / alpha_flat.clamp(min=1e-8)               # (N,)
        lam = torch.linalg.eigvalsh(Gf).amin(dim=-1)                   # (N,)
        A_K = torch.linalg.matrix_power(A_eff, K)
        gap = (A_K - A_eff).norm(dim=(-2, -1)) / (A_eff.norm(dim=(-2, -1)) + 1e-12)

        # (N,) -> (B, nc, H). No reduction: hand back the whole distribution.
        radius = radius.view(Bc, nc, Hc)
        raw_radius = raw_radius.view(Bc, nc, Hc)
        alpha_bnh = alpha.reshape(Bc, nc, Hc)     # clamp factor in (0, 1]
        lam = lam.view(Bc, nc, Hc)
        gap = gap.view(Bc, nc, Hc)

        eta = self._resolve_eta().detach().float().reshape(())
        gate = self.layerscale_gate
        gate_mag = gate.abs().mean() if gate is not None else eta.abs()

        return {
            'spectral_radius': radius.detach(),
            'raw_spectral_radius': raw_radius.detach(),
            'alpha': alpha_bnh.detach(),
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
            f'chunk_size={self.chunk_size}',
            f'eta={eta_val:.4f}', f'gamma={gamma_val:.4f}',
            f'eta_bounds={self.eta_bounds}', f'gamma_bounds={self.gamma_bounds}',
            f'layerscale={self.layerscale}',
        ]
        return ', '.join(parts)


class SKABlock(nn.Module):
    """SKA layer with pre-norm, matching Nemotron-H attention block interface.

    Optional parallel short-range path: a depthwise CAUSAL conv on the normed
    input, summed into the residual alongside SKA. Covers the short-range band
    that chunked SKA stats discard (within-chunk cross-covariance), so SKA's
    gradient isn't poisoned by short-range failures. See config.ska_short_conv.

    Lives here next to SKAModule (both are the token-mixer's SKA occupant);
    models/koopman_lm.py and models/baselines.py compose it.
    """
    def __init__(self, cfg: KoopmanLMConfig):
        super().__init__()
        self.norm = nn.LayerNorm(cfg.d_model)
        self.ska = SKAModule(
            d_model=cfg.d_model,
            n_heads=cfg.ska_n_heads,
            rank=cfg.ska_rank,
            head_dim=cfg.head_dim,
            ridge_eps=cfg.ska_ridge,
            scale=cfg.ska_scale,
            power_K=cfg.ska_power_K,
            chunk_size=cfg.ska_chunk_size,
            # --- new scale-parameter + residual policy ---
            eta_learnable=cfg.ska_eta_learnable,
            eta_value=cfg.ska_eta_value,
            eta_bounds=getattr(cfg, 'ska_eta_bounds', None),
            gamma_learnable=cfg.ska_gamma_learnable,
            gamma_value=cfg.ska_gamma_value,
            gamma_clamp=cfg.ska_gamma_clamp,
            gamma_bounds=getattr(cfg, 'ska_gamma_bounds', None),
            layerscale=cfg.ska_layerscale,
            layerscale_init=cfg.ska_layerscale_init,
            out_proj_std=cfg.ska_out_proj_std,
            exact_intrachunk=getattr(cfg, 'ska_exact_intrachunk', False),
        )
        # Parallel short-range causal depthwise conv (covers within-chunk band).
        self.short_conv = None
        if getattr(cfg, 'ska_short_conv', False):
            k = cfg.ska_short_conv_kernel
            self.short_conv_pad = k - 1                     # left-pad => causal
            self.short_conv = nn.Conv1d(
                cfg.d_model, cfg.d_model, kernel_size=k,
                groups=cfg.d_model, bias=True)              # depthwise
            # Lag-biased init (current + lag-1 + lag-2), gated small. Exposes
            # local HISTORY from step 0 -- the band chunked SKA discards --
            # rather than mostly the current token. Weights sum ~1 so the gated
            # path is a gentle local average at init.
            nn.init.zeros_(self.short_conv.weight)
            with torch.no_grad():
                if k >= 3:
                    self.short_conv.weight[:, 0, -1] = 0.50   # current token
                    self.short_conv.weight[:, 0, -2] = 0.35   # lag-1
                    self.short_conv.weight[:, 0, -3] = 0.15   # lag-2
                else:
                    self.short_conv.weight[:, 0, -1] = 1.0
            nn.init.zeros_(self.short_conv.bias)
            # ...BUT gate the whole path by a small learnable per-channel scale,
            # so at init the conv contributes ~gate_init * LayerNorm(x), NOT a
            # full-strength extra residual. Otherwise the block would start as
            # x + h + tiny_ska (a free normalized residual injected at every SKA
            # layer), which confounds the "did restoring local evidence help?"
            # ablation. Gate is learnable so the local path grows as needed.
            self.short_conv_gate = nn.Parameter(
                torch.full((cfg.d_model,),
                           float(getattr(cfg, 'ska_short_conv_gate_init', 1e-2))))
        self._ablate = False   # see KoopmanLM.ablate(): zero this layer's contribution

    def forward(self, x):
        if self._ablate:
            return x            # SKA-zeroed: pure residual passthrough (no SKA, no conv)
        h = self.norm(x)
        out = x + self.ska(h)
        if self.short_conv is not None:
            # (B,T,d) -> (B,d,T), left-pad for causality, conv, trim, back, gate
            c = h.transpose(1, 2)
            c = torch.nn.functional.pad(c, (self.short_conv_pad, 0))
            c = self.short_conv(c)[..., :h.shape[1]]
            out = out + c.transpose(1, 2) * self.short_conv_gate
        return out
