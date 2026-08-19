"""Spectral Koopman MLP -- lift, block-rotate, read out.

v1 (paper S3.3): x -> LN -> lift -> SiLU -> per-pair 2x2 rotation -> readout.

v2 (Aurora-inspired utilization fixes, all opt-in via config; defaults reproduce
v1 exactly, param names/shapes unchanged so existing checkpoints still load):

  1. Row-normalized lift with explicit per-row gains (``row_norm_lift``). The
     lift weight is parameterized as ``W[i,:] = g_i * v_i / ||v_i||`` (WeightNorm
     over output rows). Every lifted dimension then gets equal geometric leverage
     from the input, and "how much does this neuron matter" collapses to one
     interpretable scalar ``g_i`` per row -- readable, histogrammable,
     regularizable. Dead neurons show up as ``g_i -> 0`` instead of hiding in
     row-norm drift. NOTE: ``v`` is scale-invariant, so it MUST be excluded from
     weight decay (see KoopmanLM.no_weight_decay_param_names); otherwise decay
     drives ``||v|| -> 0`` and the 1/||v|| gradient blows up.

  2. ``rotation_param='logrho_theta'`` -- decay rate and rotation angle become
     separate parameters with comparable gradient scales:
        gamma = e^{-softplus(s)} cos(theta),  omega = e^{-softplus(s)} sin(theta)
     rho = e^{-softplus(s)} in (0, 1] builds the |lambda| <= 1 constraint in
     smoothly (no clamp, no kink). theta's gradient is the full chain rule at
     whatever rho is, rather than the small antisymmetric wedge component it gets
     through raw (gamma, omega). With ``depth_grade`` the s init is graded by
     depth (aggressive decay early, gentle late) -- the profile the model tends
     to learn anyway.

  3. ``pair_mixer`` -- an orthogonal remix of the lifted activations BEFORE the
     2x2 pairing, so the network chooses which coordinates get paired for
     rotation instead of being yoked to reshape order (dims 2i, 2i+1) forever.

  4. Diagnostics hooks (``effective_lift_row_norms`` / ``lifted``) that let the
     utilization report measure the dead-pair fraction and the g_i histogram over
     training, v1 vs v2 (see koopman_mlp_diag.py).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from koopman_lm.modules.norm import make_norm


# ---------------------------------------------------------------------------
# Rotation parameterization
# ---------------------------------------------------------------------------

def _disk_clamp(gamma, omega):
    """Legacy unit-DISK clamp of a (gamma, omega) eigenvalue pair.

    The 2x2 block [[g, o], [-o, g]] = radius * R(theta), radius = sqrt(g^2+o^2),
    so both singular values equal radius. Scaling by min(radius,1)/radius caps
    radius at 1 (|lambda| <= 1, non-expansive) but leaves radius < 1 untouched,
    so it is NOT norm-preserving once radius < 1. Kept for the legacy
    (norm_preserving=False) path and existing checkpoints.
    """
    radius = torch.sqrt(gamma * gamma + omega * omega).clamp(min=1e-8)
    scale = torch.clamp(radius, max=1.0) / radius
    return gamma * scale, omega * scale


def _resolve_rotation_param(m):
    """Resolve the rotation parameterization name for module m.

    Back-compat: rotation_param=None derives from norm_preserving -- True ->
    'angle' (cos/sin, exact rotation), False -> 'legacy' (learned gamma, omega
    disk-clamped). Explicit values 'legacy' | 'angle' | 'logrho_theta' win.
    """
    rp = getattr(m, 'rotation_param', None)
    if rp is not None:
        return rp
    return 'angle' if m.norm_preserving else 'legacy'


def _init_rotation_params(m):
    """Register the rotation parameters for module m per its parameterization."""
    mode = _resolve_rotation_param(m)
    P = m.d_k // 2
    if mode == 'angle':
        # theta ~ N(0, 0.1): cos(theta) ~ 1, sin(theta) ~ small -> matches the
        # legacy (gamma=1, omega~N(0,0.1)) rotation direction, radius == 1 exactly.
        m.theta = nn.Parameter(torch.empty(P).normal_(0, 0.1))
    elif mode == 'logrho_theta':
        # Separate decay (s) and angle (theta). rho = e^{-softplus(s)} in (0,1].
        m.theta = nn.Parameter(torch.empty(P).normal_(0, 0.1))
        m.s = nn.Parameter(_init_decay_s(m, P))
    elif mode == 'legacy':
        m.gamma = nn.Parameter(torch.ones(P))
        m.omega = nn.Parameter(torch.empty(P).normal_(0, 0.1))
    else:
        raise ValueError(f"unknown rotation_param={mode!r}")


def _softplus_inv(y):
    """s such that softplus(s) == y (y > 0). softplus(s)=log(1+e^s)."""
    return math.log(math.expm1(y))


def _init_decay_s(m, P):
    """Init tensor for the logrho_theta decay logit s (shape (P,)).

    Without depth grading: rho ~ 0.99 everywhere (near norm-preserving; the model
    learns the decay it needs). With depth grading (needs layer_idx / n_layers):
    interpolate rho from ~0.5 (aggressive decay, early layers) to ~0.97 (gentle,
    late layers) -- the depth profile these models tend to rediscover, handed to
    them at init.
    """
    rho_gentle = 0.99
    if getattr(m, 'depth_grade', False) and m.layer_idx is not None and m.n_layers:
        frac = m.layer_idx / max(m.n_layers - 1, 1)      # 0 (first) .. 1 (last)
        rho = 0.5 + frac * (0.97 - 0.5)                  # aggressive -> gentle
    else:
        rho = rho_gentle
    # rho = e^{-softplus(s)}  =>  softplus(s) = -ln(rho)  =>  s = softplus_inv(...)
    s0 = _softplus_inv(-math.log(rho))
    return torch.full((P,), float(s0))


def _rotation_coeffs(m):
    """(gamma, omega) coefficients of the block [[g, o], [-o, g]] for module m.

    'angle'        -> gamma = cos(theta), omega = sin(theta): exact rotation
        (sigma_min = sigma_max = 1), smooth everywhere -- no 1/radius origin
        singularity (paper S3.3 "Gradient preservation").
    'logrho_theta' -> gamma = rho cos(theta), omega = rho sin(theta), with
        rho = e^{-softplus(s)} in (0, 1]. Decay and angle decouple; the modulus
        constraint is built in smoothly.
    'legacy'       -> learned (gamma, omega), unit-disk clamped when
        spectral_norm_gamma.
    """
    mode = _resolve_rotation_param(m)

    # mlp_precision, when set, RAISES the width these coefficients are computed
    # at. The cast has to happen BEFORE the trig, not after: cos(theta.double())
    # is more accurate than cos(theta).double(), and only the former is worth a
    # config field. `precision=None` leaves `_up` as the identity, so the default
    # path is untouched -- bit-identical, per the design's "strict no-op".
    precision = getattr(m, 'precision', None)
    if precision is None:
        def _up(tensor):
            return tensor
    else:
        from koopman_lm.precision import dtype_of
        _raised = dtype_of(precision)

        def _up(tensor):
            return tensor.to(_raised)

    if mode == 'angle':
        theta = _up(m.theta)
        return torch.cos(theta), torch.sin(theta)
    if mode == 'logrho_theta':
        theta, s = _up(m.theta), _up(m.s)
        rho = torch.exp(-F.softplus(s))
        return rho * torch.cos(theta), rho * torch.sin(theta)
    gamma, omega = _up(m.gamma), _up(m.omega)
    if m.spectral_norm_gamma:
        gamma, omega = _disk_clamp(gamma, omega)
    return gamma, omega


# ---------------------------------------------------------------------------
# Orthogonal pair mixer (change 3)
# ---------------------------------------------------------------------------

class PairMixer(nn.Module):
    """Orthogonal remix of the lifted activations before the 2x2 pairing.

    The 2x2 blocks otherwise yoke dimensions (2i, 2i+1) by reshape order forever;
    an orthogonal mix lets the network choose which coordinates get paired.
    Block-diagonal (block size b) keeps params/compute at O(d_k * b) rather than a
    dense d_k x d_k map (which at d_k~2048 x 24 layers would add ~100M
    params/buffer). Any pairing can form WITHIN a block; b is even so each rotated
    pair (2i, 2i+1) stays inside one block. Every mode is exactly orthogonal, so
    the mix preserves the L2 norm of the lifted vector -- it composes with the
    norm-preserving rotation without touching that guarantee.

      mode='perm'    : fixed random permutation (0 params; a d_k index buffer).
      mode='ortho'   : fixed block-diagonal random orthogonal (buffer; 0 trained).
      mode='learned' : learnable block-diagonal orthogonal via the Cayley map
                       Q = (I - A)(I + A)^-1, A skew-symmetric (trained).
    """

    def __init__(self, d_k, mode, block=64, seed=0):
        super().__init__()
        self.d_k = d_k
        self.mode = mode
        b = min(block, d_k)
        if d_k % b != 0:
            raise ValueError(
                f"pair_mixer block={b} must divide d_k={d_k}")
        self.block = b
        self.n_blocks = d_k // b
        gen = torch.Generator().manual_seed(seed)

        if mode == 'perm':
            self.register_buffer('perm', torch.randperm(d_k, generator=gen))
        elif mode == 'ortho':
            self.register_buffer('Q', self._random_block_orthogonal(gen))
        elif mode == 'learned':
            # Strictly-upper-triangular free params; skew-symmetrized in forward.
            A = torch.empty(self.n_blocks, b, b).normal_(0, 1e-3, generator=gen)
            self.A_raw = nn.Parameter(torch.triu(A, diagonal=1))
        else:
            raise ValueError(f"unknown pair_mixer={mode!r}")

    def _random_block_orthogonal(self, gen):
        """(n_blocks, b, b) stack of random orthogonal matrices via QR."""
        M = torch.empty(self.n_blocks, self.block, self.block).normal_(0, 1, generator=gen)
        Q, R = torch.linalg.qr(M)
        # Fix the sign ambiguity so Q is a proper, reproducible orthogonal matrix.
        sign = torch.sign(torch.diagonal(R, dim1=-2, dim2=-1))
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        return Q * sign.unsqueeze(-2)

    def _cayley(self):
        """Q = (I - A)(I + A)^-1 per block, A skew-symmetric. Orthogonal."""
        A = self.A_raw - self.A_raw.transpose(-1, -2)        # skew-symmetric
        I = torch.eye(self.block, device=A.device, dtype=A.dtype).expand_as(A)
        return torch.linalg.solve(I + A, I - A)              # (n_blocks, b, b)

    def matrix(self):
        """Dense (d_k, d_k) form of the mix -- for tests / inspection only."""
        if self.mode == 'perm':
            M = torch.zeros(self.d_k, self.d_k, device=self.perm.device)
            M[torch.arange(self.d_k), self.perm] = 1.0
            return M
        Q = self.Q if self.mode == 'ortho' else self._cayley()
        return torch.block_diag(*Q)

    def forward(self, x):
        if self.mode == 'perm':
            return x[..., self.perm]
        Q = self.Q if self.mode == 'ortho' else self._cayley()
        xb = x.view(*x.shape[:-1], self.n_blocks, self.block)
        # out[..., n, o] = sum_i x[..., n, i] * Q[n, o, i]
        out = torch.einsum('...ni,noi->...no', xb, Q.to(x.dtype))
        return out.reshape(*x.shape)


# ---------------------------------------------------------------------------
# Lift (change 1)
# ---------------------------------------------------------------------------

def _init_lift(m, d):
    """Register the lift projection for module m (d -> d_k).

    row_norm_lift=True  -> WeightNorm rows: params lift_v (d_k, d) direction and
        lift_g (d_k,) per-row gain, with W = g * v / ||v||. Gains init to a
        constant = the expected xavier row norm, so utilization starts uniform
        AND the forward-pass scale matches a xavier lift.
    row_norm_lift=False -> plain nn.Linear (legacy; identical param name 'lift').
    """
    if m.row_norm_lift:
        m.lift_v = nn.Parameter(torch.empty(m.d_k, d))
        nn.init.xavier_uniform_(m.lift_v)
        g0 = math.sqrt(2.0 * d / (m.d_k + d))            # expected xavier row norm
        m.lift_g = nn.Parameter(torch.full((m.d_k,), g0))
    else:
        m.lift = nn.Linear(d, m.d_k, bias=False)
        nn.init.xavier_uniform_(m.lift.weight)


def _effective_lift_weight(m):
    """Effective (d_k, d) lift weight, honoring the row-norm parameterization."""
    if m.row_norm_lift:
        v = m.lift_v
        return m.lift_g.unsqueeze(1) * v / v.norm(dim=1, keepdim=True).clamp_min(1e-8)
    return m.lift.weight


def _apply_lift(m, h):
    if m.row_norm_lift:
        return F.linear(h, _effective_lift_weight(m))
    return m.lift(h)


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------

class SpectralKoopmanMLP(nn.Module):
    """Ungated spectral Koopman MLP. See module docstring for v2 options."""

    gated = False

    def __init__(self, d, expand=2.667, spectral_norm_gamma=True,
                 norm_preserving=False, rotation_param=None, row_norm_lift=False,
                 pair_mixer=None, mixer_block=64, depth_grade=False,
                 layer_idx=None, n_layers=None, norm_type='layernorm',
                 norm_eps=1e-5, precision=None):
        super().__init__()
        # None (the default) is a strict no-op: the rotation runs exactly as
        # before. fp32/fp64 raise the width its coefficients are computed at.
        # Validated here so a bad value fails at construction rather than
        # mid-forward.
        if precision is not None:
            from koopman_lm.precision import COMPONENT_PRECISIONS

            if precision not in COMPONENT_PRECISIONS:
                raise ValueError(
                    f"mlp precision={precision!r}; expected None or one of "
                    f"{sorted(COMPONENT_PRECISIONS)}. This field may only RAISE "
                    f"precision.")
        self.precision = precision
        self.d_k = ((int(d * expand) + 63) // 64) * 64
        self.spectral_norm_gamma = spectral_norm_gamma
        self.norm_preserving = norm_preserving
        self.rotation_param = rotation_param
        self.row_norm_lift = row_norm_lift
        self.depth_grade = depth_grade
        self.layer_idx = layer_idx
        self.n_layers = n_layers

        self.norm = make_norm(d, norm_type, norm_eps)
        _init_lift(self, d)
        if self.gated:
            self.gate = nn.Linear(d, self.d_k, bias=False)
            nn.init.xavier_uniform_(self.gate.weight)
        self.mixer = None
        if pair_mixer not in (None, 'none'):
            # Seed by depth so different layers get distinct fixed mixes.
            self.mixer = PairMixer(self.d_k, pair_mixer, block=mixer_block,
                                   seed=(layer_idx or 0))
        _init_rotation_params(self)
        self.readout = nn.Linear(self.d_k, d, bias=False)
        nn.init.xavier_uniform_(self.readout.weight)

    # -- init re-application (called by KoopmanLM after generic _init_weights) --
    def reset_projection_params(self):
        """Re-apply the projection inits that KoopmanLM._init_weights tramples.

        _init_weights only touches nn.Linear (readout, and lift/gate in the plain
        path), setting them to normal(0, 0.02); we restore xavier. The row-norm
        lift params, rotation params, and mixer params are not nn.Linear, so they
        are already correct from __init__ and left untouched.
        """
        nn.init.xavier_uniform_(self.readout.weight)
        if not self.row_norm_lift:
            nn.init.xavier_uniform_(self.lift.weight)
        if self.gated:
            nn.init.xavier_uniform_(self.gate.weight)

    # -- rotation --------------------------------------------------------------
    def _rotate(self, g_x):
        g_pair = g_x.view(*g_x.shape[:-1], self.d_k // 2, 2)
        g1, g2 = g_pair[..., 0], g_pair[..., 1]
        gamma, omega = _rotation_coeffs(self)
        z1 = gamma * g1 + omega * g2
        z2 = -omega * g1 + gamma * g2
        rotated = torch.stack([z1, z2], dim=-1).reshape_as(g_x)
        # Raising the rotation's internal width must not widen what the block
        # returns: torch promotes fp32 activations against fp64 coefficients, and
        # letting that escape would silently change the dtype of every layer
        # downstream. The precision is raised for this computation, not for the
        # residual stream.
        return rotated if rotated.dtype == g_x.dtype else rotated.to(g_x.dtype)

    def _postrotate_gate(self, h, z):
        return z                                             # ungated

    def forward(self, x):
        h = self.norm(x)
        g_x = F.silu(_apply_lift(self, h))
        if self.mixer is not None:
            g_x = self.mixer(g_x)
        z = self._rotate(g_x)
        z = self._postrotate_gate(h, z)
        return x + self.readout(z)

    # -- diagnostics (change 4) ------------------------------------------------
    @torch.no_grad()
    def effective_lift_row_norms(self):
        """Per-lifted-dimension gain = L2 norm of the effective lift row.

        For row_norm_lift this is exactly |lift_g| (the rows are unit-norm), so
        one interpretable scalar per neuron; for the plain lift it is the raw
        row norm. Feeds the g_i histogram / dead-neuron count.
        """
        return _effective_lift_weight(self).norm(dim=1)

    @torch.no_grad()
    def lifted(self, x):
        """SiLU-lifted, mixed activations (exactly what gets paired+rotated).

        Given the residual-stream input x to this layer, returns the (..., d_k)
        activations whose per-pair variance measures utilization.
        """
        g_x = F.silu(_apply_lift(self, self.norm(x)))
        if self.mixer is not None:
            g_x = self.mixer(g_x)
        return g_x


class SpectralKoopmanMLPGated(SpectralKoopmanMLP):
    """Gated variant: a sigmoid gate modulates the rotated activations.

    The gate is a learned d -> d_k projection applied AFTER rotation, so it adapts
    to whatever (mixed, rotated) coordinate frame the readout reads.
    """

    gated = True

    def _postrotate_gate(self, h, z):
        return z * torch.sigmoid(self.gate(h))
