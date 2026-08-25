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
import torch.nn.functional as F
from contextlib import nullcontext

# Verified parity components (match echo_jax.py math):
#   ska_core  -- whitened L^{-1}M L^{-T} forward + custom O(K r^2) backward
#                ("It Cancels"; Cholesky never differentiated)
#   chunk_stats -- beta-gated, strictly-causal sufficient statistics
from koopman_lm.kernels.ska_operator import ska_core
from koopman_lm.kernels.chunk_stats import (
    chunk_stats as _causal_chunk_stats, symmetric_key_value, causal_normalize)

#: The write-gate parameterisations `beta_policy` may name. Duplicated from
#: `koopman_lm/config.py` rather than imported, for the reason the `precision`
#: kwarg's comment gives at length: this module deliberately never imports
#: KoopmanLMConfig -- it is config-free numerics, and ska_block.py threads every
#: scalar in. `code-tests/test_ska_beta_policy.py` pins the two lists together
#: so the duplicate cannot drift silently.
BETA_POLICIES = frozenset({
    "learned", "one", "head_scalar", "linear",
    "key_linear_value_sqrt", "key_sqrt_value_linear",
})

#: (key exponent, value exponent) for every policy that applies a beta power to
#: the key and value streams. `beta ** exponent`, so 0.5 is sqrt(beta) and 1.0 is
#: beta. `one` is absent because it constructs no gate at all and short-circuits
#: in `_resolve_beta`; `head_scalar` shares `learned`'s exponents.
#:
#: A table rather than a chain of `if`s because the exponents ARE the design: the
#: 2x2 is legible here, and a new cell is a row rather than a branch.
BETA_EXPONENTS = {
    "learned":               (0.5, 0.5),
    "head_scalar":           (0.5, 0.5),
    "one":                   (0.5, 0.5),   # beta == 1, so any exponent agrees
    "linear":                (1.0, 1.0),
    "key_linear_value_sqrt": (1.0, 0.5),
    "key_sqrt_value_linear": (0.5, 1.0),
}

#: Policies that build the full `Linear(d_model, n_heads)` write gate. `one`
#: builds nothing and `head_scalar` builds H scalars; both are deliberately
#: SIMPLER MODELS, so a disabled-but-present projection would still be seen by
#: weight decay and still show in the parameter count.
_PROJECTION_POLICIES = frozenset({
    "learned", "linear", "key_linear_value_sqrt", "key_sqrt_value_linear",
})

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


# ============================================================================
# SKA Module
# ============================================================================

class SKAModule(nn.Module):
    """Structured Kernel Attention with exact-scan and legacy chunked backends.

    ``prefix_scan=True`` is strictly causal at token resolution;
    ``prefix_scan_block_size`` controls scheduling only.

    This said "the recommended quality path is ``prefix_scan=True``" until
    2026-08-24, which stopped being true when the routes were actually timed.
    Per `space.py`'s table (jobs 440122 correctness / 440135 cost) the
    recommendation is geometry-dependent, and stating one winner hid that:

      * rank 24 AND value width 64 -> ``prefix_scan=True``, which reaches the
        fused CUDA kernel at 1.01x the chunked cost. Only ``50m`` and ``180m``
        qualify, and both already set it.
      * anything else with rank <= 64 -> ``inverse_cholesky=True``. Exact to
        5.1e-13 in fp64 and 0.92x-1.41x chunked at the measured geometries. Off
        the fused kernel's geometry ``prefix_scan`` silently falls back to the
        Python reference scan at 137x-160x, so it is the WRONG default there.
      * rank > 64 -> no exact route trains. ``inverse_cholesky`` asserts
        rank <= 64; ``exact_intrachunk`` is 57x-87x. See
        `code-tests/test_exact_route_reachability.py`.
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
                 prefix_scan_jitter=0.0, norm_clip_c=None,
                 precision='fp32', beta_policy='learned'):
        super().__init__()
        if beta_policy not in BETA_POLICIES:
            raise ValueError(
                f"SKAModule ska_beta_policy={beta_policy!r}; expected one of "
                f"{sorted(BETA_POLICIES)}")
        self.beta_policy = beta_policy
        # The dtype the whitened core computes in. Arg 29 of 29 scalars: this
        # module deliberately never imports KoopmanLMConfig -- it is config-free
        # numerics -- so ska_block.py threads cfg.ska_precision in, exactly as it
        # threads the other 28. Validated here so a bad value fails at
        # construction on a login node rather than mid-forward on a GPU.
        from koopman_lm.precision import COMPONENT_PRECISIONS, dtype_of

        if precision not in COMPONENT_PRECISIONS:
            raise ValueError(
                f"SKAModule precision={precision!r}; expected one of "
                f"{sorted(COMPONENT_PRECISIONS)}. The core takes a cholesky of a "
                f"Gram matrix, so this field may only RAISE precision.")
        self.precision = precision
        self._core_dtype = dtype_of(precision)
        self.rank = rank
        self.exact_intrachunk = exact_intrachunk
        self.prefix_scan = bool(prefix_scan)
        self.prefix_scan_block_size = int(prefix_scan_block_size)
        self.prefix_scan_jitter = float(prefix_scan_jitter)
        if self.prefix_scan_block_size <= 0:
            raise ValueError("prefix_scan_block_size must be positive")
        # Small-rank exact path that materializes per-token inverse-Cholesky
        # statistics.
        #
        # NOT "legacy", and NOT "superseded by prefix_scan for new quality runs"
        # -- which is what this comment said until 2026-08-24. It is the DEFAULT
        # exact route (`space.py`'s `backend_policy` defaults to
        # `exact_invchol`), because off the fused kernel's rank-24/width-64
        # geometry prefix_scan falls back to the Python reference scan at
        # 137x-160x while this path stays at 0.92x-1.41x chunked. Calling it
        # legacy pointed a reader at the expensive route on every geometry but
        # two.
        #
        # Its real limit is the assert below, not obsolescence: per-token stats
        # are (B,T,H,r,r), so rank > 64 is refused outright and rank 64 at
        # seq 8192 is tens of GiB per SKA layer at batch 1.
        self.inverse_cholesky = inverse_cholesky

        # THE CHUNKED PATH ANNOUNCES ITSELF, because it is not a slower-but-fine
        # alternative -- it computes something different.
        #
        # With none of prefix_scan / inverse_cholesky / exact_intrachunk set,
        # chunk_stats uses EXCLUSIVE-CHUNK-PREFIX boundaries: a token in chunk c
        # sees only completed chunks < c, so every within-chunk lag-1..lag-(S-1)
        # cross-covariance term is missing -- which is the capability SKA exists
        # to provide. A number produced this way is a speed upper bound, not a
        # result.
        #
        # THE PROVENANCE OF "~100%", because this comment used to get it wrong.
        # It said "chunk_stats_exact.py's own header MEASURES that at ...". That
        # header does not measure anything: it ASSERTS the figure, and it has
        # asserted it since commit 40f6653 ("Add files via upload", 2026-05-21),
        # a bulk import of an external echo-ska-440m tree that arrived with no
        # harness, no geometry and no job id. Citing it as a measurement gave an
        # inherited sentence the standing of an artefact.
        #
        # There are now two real artefacts, and they agree with the figure:
        #   * job 440122 (H100, fp64, vs prefix_scan.dense_exact_oracle) --
        #     0.92-1.52 forward error, 93%-101% in the gradients, at both the 4m
        #     and 50m geometries. Recorded in experimentation/sweep/search/
        #     space.py; the harness itself is not committed.
        #   * code-tests/test_chunked_route_staleness.py -- in-suite, fp64, and
        #     it characterises the STRUCTURE rather than restating the scalar:
        #     exact at every chunk's first token, worst at its last; lag-1 recall
        #     destroyed for query-to-key distance 2 <= d <= j+1 where j is the
        #     token's offset into its chunk, and inflated ~20% beyond; and
        #     controlled by ska_chunk_size ALONE, with no sequence-length
        #     dependence. That last point is why max_seq_len 8192 does not help.
        #
        # Warned at CONSTRUCTION, not per forward: once per model, before any
        # time is spent, and it names all three exact routes so the reader does
        # not have to go find them.
        if not (self.prefix_scan or self.inverse_cholesky or self.exact_intrachunk):
            import warnings
            warnings.warn(
                "SKAModule is running the CHUNKED approximation: no exact path "
                "is enabled (ska_prefix_scan / ska_inverse_cholesky / "
                "ska_exact_intrachunk all off). This drops the within-chunk "
                "lag-1..lag-(S-1) cross-covariance terms entirely -- measured at "
                "~100% RELATIVE ERROR against a per-token-causal reference on "
                "short-range recall, i.e. on exactly what SKA is for. Treat any "
                "number from this configuration as an upper bound on SPEED and "
                "not as a result. Measured by job 440122 (see space.py) and by "
                "code-tests/test_chunked_route_staleness.py, which also shows "
                "the shape: the route is EXACT at each chunk's first token and "
                f"worst at its last, so with ska_chunk_size={chunk_size} a "
                "token is stale by up to that many tokens; lag-1 recall is "
                "destroyed for query-to-key distances up to the token's offset "
                "into its chunk and INFLATED ~20% beyond it; and the error is "
                "set by ska_chunk_size alone -- a longer max_seq_len does not "
                "dilute it. Exact alternatives, cheapest first to try: "
                + ("ska_inverse_cholesky=True (batched matmul, no power "
                   "iteration), "
                   if rank <= 64 else
                   f"NOT ska_inverse_cholesky -- it asserts rank <= 64 and this "
                   f"config is rank {rank}, so that route is UNAVAILABLE here "
                   f"(see code-tests/test_exact_route_reachability.py; the same "
                   f"is true of the fused prefix-scan kernel, which needs rank "
                   f"== 24 and value width == 64). ") +
                "ska_exact_intrachunk=True (per-token stats, same verified core), "
                "ska_prefix_scan=True (exact two-level scan; ~137x slower than "
                "chunked when the fused CUDA kernel's geometry does not match).",
                RuntimeWarning, stacklevel=2)
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
        #
        # `one` constructs NOTHING, and `head_scalar` constructs H scalars
        # rather than a projection. Both are the point: these policies exist to
        # be SIMPLER MODELS, and a disabled-but-present beta_proj would still be
        # seen by weight decay, still appear in the parameter count, and still
        # show up in an optimizer param-group regex -- so the comparison would
        # not be measuring what it claims.
        self.beta_proj = None
        self.beta_logit = None
        if beta_policy in _PROJECTION_POLICIES:
            self.beta_proj = nn.Linear(d_model, n_heads, bias=True)
            nn.init.zeros_(self.beta_proj.weight)
            nn.init.zeros_(self.beta_proj.bias)   # beta = sigmoid(0) = 0.5 at init
        elif beta_policy == "head_scalar":
            # Same initialisation SEMANTICS as the learned gate: logit 0 ->
            # beta = 0.5 at step 0, so `learned` and `head_scalar` start from
            # the identical operator and the contrast is about the
            # parameterisation rather than about where each one starts.
            self.beta_logit = nn.Parameter(torch.zeros(n_heads))

        nn.init.orthogonal_(self.key_proj.weight)
        nn.init.orthogonal_(self.query_proj.weight)
        nn.init.xavier_uniform_(self.value_proj.weight)
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

    def _resolve_beta(self, hidden_states):
        """The per-token, per-head write weight this forward will apply: (B,T,H).

        One method, called by the training forward, by the decode path
        (`models/recurrent.py`) and by the diagnostics (`diagnostics/ska.py`),
        for the same reason `symmetric_key_value` is one helper: a second site
        that recomputed beta from `beta_proj` directly would silently ignore the
        policy, and the train/decode divergence would show up as a quality
        regression rather than as an error.

        `one` returns an ones tensor rather than the scalar 1.0. Slightly
        wasteful and worth it: every caller then handles one shape, and the
        alternative is a `if isinstance(beta, float)` branch at three call sites,
        which is exactly the shape of the gamma fast path that has already
        caused one drift.
        """
        B, T, _ = hidden_states.shape
        if self.beta_policy == "one":
            return hidden_states.new_ones(B, T, self.H)
        if self.beta_policy == "head_scalar":
            return torch.sigmoid(self.beta_logit).expand(B, T, self.H)
        return torch.sigmoid(self.beta_proj(hidden_states))            # (B,T,H)

    def _weight_key_value(self, z_n, beta, v):
        """Apply the policy's key/value weighting. Returns (x, vbar).

        The policy selects a pair of EXPONENTS on beta, one for the key stream
        and one for the value stream (`BETA_EXPONENTS`):

            learned / head_scalar / one   (sqrt, sqrt)   the v1.1 default
            linear                        (beta, beta)
            key_linear_value_sqrt         (beta, sqrt)   the mixed cell C
            key_sqrt_value_linear         (sqrt, beta)   the mixed cell D

        THE KEY EXPONENT IS THE ONE WITH SPECTRAL CONSEQUENCES, and the value
        exponent has none. `G = ridge*I + sum x x^T` and `M = sum x_t x_{t-1}^T`
        are built entirely from the key stream, and contractivity of
        `W = L^-1 M L^-T` is a statement about exactly those two matrices. The
        value weighting enters only `C = sum vbar x^T`, which `ska_core` applies
        after the whitened operator. So:

          * every policy here is contractive, because every one of them puts a
            SINGLE key stream into BOTH slots of M -- which is all the
            Cauchy-Schwarz argument in `symmetric_key_value`'s docstring needs.
            Nothing in that argument mentions beta.
          * the value exponent cannot move sigma_max at all, pinned bit-for-bit
            by `test_the_value_exponent_cannot_change_sigma_max`.

        What the square root buys is therefore not safety but MEANING: at
        exponent 0.5 the Gram is `G = sum beta z z^T`, so beta is literally the
        per-token write weight. At exponent 1.0 it is `sum beta^2 z z^T`, which
        silently redefines the write weight as its square -- and at the shared
        `beta = 0.5` initialisation halves the own-weight against a fixed ridge,
        making a linear-key cell twice as ridge-regularised at step 0. That is a
        confound rather than a feature; `test_the_key_exponent_changes_the_
        effective_ridge_at_init` measures the factor the ridge-matched control
        cell is built from.

        `clamp_min(0)` before the power, as `symmetric_key_value` does: it keeps
        a negative beta out of a fractional power. `_resolve_beta` returns a
        sigmoid or ones, so beta is in (0,1] regardless -- the clamp costs
        nothing and means the helper is safe if a future policy widens the range.
        """
        key_exp, val_exp = BETA_EXPONENTS[self.beta_policy]
        if (key_exp, val_exp) == (0.5, 0.5):
            # The default convention, kept on its own dedicated helper so the
            # shared accumulation sites (chunk_stats, chunk_stats_exact,
            # decode) all continue to name ONE function for it, and so this
            # branch is bit-identical to what it was before the exponent table
            # existed.
            return symmetric_key_value(z_n, beta, v)
        b = beta.clamp_min(0).unsqueeze(-1)
        w_key = b if key_exp == 1.0 else b.sqrt()
        w_val = b if val_exp == 1.0 else b.sqrt()
        return w_key * z_n, w_val * v

    def forward(self, hidden_states):
        B, T, _ = hidden_states.shape
        r = self.rank
        H, P = self.H, self.P

        # Fused k/q/v projection (absorbed from the retired ska_fast patch):
        # one GEMM against the row-stacked [W_k; W_q; W_v] equals the three
        # separate GEMMs by block matmul, but issues a single kernel. The
        # weights stay three separate nn.Linear parameters so checkpoints, the
        # recurrent decode path (models/recurrent.py calls ska.key_proj(h)
        # directly), and koopman_lm.py's custom inits are all untouched;
        # autograd splits the gradient back through the cat. beta_proj stays
        # separate -- it has a bias, and is (B,T,H) not (B,T,H,r).
        #
        # This lives in forward() rather than behind a flag on purpose. The old
        # fast.py monkey-patched a SECOND forward over the module, which drifted
        # twice before anyone noticed: it read self.eta directly (AttributeError
        # under the eta_bounds squash regime) and silently skipped the
        # prefix_scan / inverse_cholesky / exact_intrachunk branches, so
        # --ska_fast on either production config quietly swapped the exact
        # operator for the chunked approximation. One forward cannot drift.
        fused_w = torch.cat([self.key_proj.weight, self.query_proj.weight,
                             self.value_proj.weight], dim=0)
        combined = F.linear(hidden_states, fused_w)
        z = combined[..., :H * r].reshape(B, T, H, r)
        zq = combined[..., H * r:2 * H * r].reshape(B, T, H, r)
        v = combined[..., 2 * H * r:].reshape(B, T, H, P)
        beta = self._resolve_beta(hidden_states)                       # (B,T,H)

        ctx = torch.amp.autocast('cuda', enabled=False) if hidden_states.is_cuda \
              else nullcontext()

        with ctx:
            # Was a hardcoded .float(). Identical at the default
            # (precision='fp32' -> torch.float32), and now stated by config
            # rather than implied -- which is also what makes an fp64 core
            # reachable on the PyTorch path, where the exact prefix scan already
            # accepts float64. Pinned bit-for-bit by
            # code-tests/test_ska_precision_wiring.py against a golden captured
            # before this change.
            core = self._core_dtype
            z_f = z.to(core)
            zq_f = zq.to(core)
            v_f = v.to(core)
            beta_f = beta.to(core)

            # Causal normalization (matches echo_jax.py): per-token L2 on
            # key/query. NO non-causal sequence-max.
            z_n = causal_normalize(z_f, self.norm_clip_c)
            zq_n = causal_normalize(zq_f, self.norm_clip_c)
            # v1.1 SYMMETRIC sqrt(beta) key/value: x=sqrt(beta)*z fed into BOTH
            # key slots, vbar=sqrt(beta)*v. G,C invariant; M/boundary become the
            # contractive cross-weight sqrt(beta_t beta_{t-1}). One helper, every
            # site -> no norm-based beta re-inference (the train/decode trap).
            # Routed through `_weight_key_value` so `ska_beta_policy` selects the
            # exponent; every policy is single-stream, hence contractive.
            x_n, v_w = self._weight_key_value(z_n, beta_f, v_f)

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
            f'beta_policy={self.beta_policy}',
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
