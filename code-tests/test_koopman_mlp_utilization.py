"""Koopman-MLP v2 utilization/structure options (Aurora-inspired).

Covers the four changes and their invariants:
  1. Row-normalized lift with explicit per-row gains (WeightNorm).
  2. (log rho, theta) rotation reparameterization + depth-graded decay init.
  3. Orthogonal pair mixer (perm / fixed-ortho / learned-Cayley).
  4. Utilization diagnostics (gain histogram + dead-pair fraction).

Plus back-compat: defaults reproduce v1 (same param names, legacy rotation) and
the KoopmanLM weight-decay-skip set is empty unless a v2 option is enabled.
"""
import dataclasses
import itertools

import pytest
import torch

from koopman_lm.config import build_config
from koopman_lm.modules.channel_mixer.koopman import (
    SpectralKoopmanMLP, SpectralKoopmanMLPGated, PairMixer,
    _rotation_coeffs, _resolve_rotation_param, _effective_lift_weight)

pytestmark = pytest.mark.correctness

D = 128
CLASSES = [SpectralKoopmanMLP, SpectralKoopmanMLPGated]
ROTATIONS = [None, "legacy", "angle", "logrho_theta"]
MIXERS = [None, "perm", "ortho", "learned"]


# --------------------------------------------------------------------------- #
# Change 1: row-normalized lift with explicit gains
# --------------------------------------------------------------------------- #

def test_row_norm_lift_weight_is_gain_times_unit_row():
    m = SpectralKoopmanMLP(D, row_norm_lift=True)
    W = _effective_lift_weight(m)
    row_norms = W.norm(dim=1)
    # each effective row has L2 norm == |g_i| (rows are unit-norm * gain)
    assert torch.allclose(row_norms, m.lift_g.abs(), atol=1e-5)
    # and the reported per-neuron gain IS that row norm
    assert torch.allclose(m.effective_lift_row_norms(), row_norms, atol=1e-6)


def test_row_norm_lift_gain_uniform_at_init():
    """Utilization starts uniform by construction: all gains equal at init."""
    m = SpectralKoopmanMLP(D, row_norm_lift=True)
    g = m.effective_lift_row_norms()
    assert float(g.std() / g.mean()) < 1e-5           # coefficient of variation ~ 0


def test_row_norm_lift_scale_invariant_in_direction():
    """W is invariant to the scale of v (the WeightNorm direction)."""
    m = SpectralKoopmanMLP(D, row_norm_lift=True)
    W0 = _effective_lift_weight(m).clone()
    with torch.no_grad():
        m.lift_v.mul_(7.3)                             # rescale directions
    assert torch.allclose(_effective_lift_weight(m), W0, atol=1e-5)


def test_row_norm_lift_gradient_finite_near_zero_direction():
    """1/||v|| is guarded: gradients stay finite even for a tiny direction row."""
    m = SpectralKoopmanMLP(D, row_norm_lift=True)
    with torch.no_grad():
        m.lift_v[0].mul_(1e-9)                         # nearly-zero row
    x = torch.randn(2, 4, D, requires_grad=True)
    m(x).pow(2).mean().backward()
    for n, p in m.named_parameters():
        assert p.grad is None or torch.isfinite(p.grad).all(), n
    assert torch.isfinite(x.grad).all()


# --------------------------------------------------------------------------- #
# Change 2: (log rho, theta) rotation
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("s", [-50.0, -1.0, 0.0, 5.0, 50.0])
def test_logrho_theta_modulus_at_most_one(s):
    """rho = e^{-softplus(s)} in (0, 1] for ANY s -> |lambda| <= 1 by construction."""
    m = SpectralKoopmanMLP(D, rotation_param="logrho_theta")
    with torch.no_grad():
        m.s.fill_(float(s))
        m.theta.copy_(torch.linspace(-3, 3, m.theta.numel()))
    gamma, omega = _rotation_coeffs(m)
    rho = (gamma ** 2 + omega ** 2).sqrt()
    assert torch.all(rho <= 1.0 + 1e-6)
    assert torch.all(rho > 0.0)


def test_logrho_theta_depth_grade_is_monotone():
    """Depth-graded init: aggressive decay early (small rho), gentle late."""
    @torch.no_grad()
    def rho_of(layer_idx):
        m = SpectralKoopmanMLP(D, rotation_param="logrho_theta",
                               depth_grade=True, layer_idx=layer_idx, n_layers=8)
        g, o = _rotation_coeffs(m)
        return float((g ** 2 + o ** 2).sqrt().mean())
    rhos = [rho_of(i) for i in range(8)]
    assert all(a < b for a, b in zip(rhos, rhos[1:]))   # strictly increasing
    assert rhos[0] < 0.6 and rhos[-1] > 0.9 and rhos[-1] <= 1.0


def test_logrho_theta_grad_hits_theta_and_s_independently():
    """theta and s receive independent, finite gradients (decoupled coords)."""
    m = SpectralKoopmanMLP(D, rotation_param="logrho_theta")
    x = torch.randn(3, 5, D, requires_grad=True)
    m(x).pow(2).mean().backward()
    assert m.theta.grad is not None and torch.isfinite(m.theta.grad).all()
    assert m.s.grad is not None and torch.isfinite(m.s.grad).all()
    assert float(m.theta.grad.abs().sum()) > 0
    assert float(m.s.grad.abs().sum()) > 0


# --------------------------------------------------------------------------- #
# Change 3: orthogonal pair mixer
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("mode", ["perm", "ortho", "learned"])
def test_pair_mixer_is_orthogonal(mode):
    mx = PairMixer(D, mode, block=64)
    M = mx.matrix()
    assert torch.allclose(M @ M.t(), torch.eye(D), atol=1e-4), mode


@pytest.mark.parametrize("mode", ["perm", "ortho", "learned"])
def test_pair_mixer_preserves_norm(mode):
    mx = PairMixer(D, mode, block=64)
    v = torch.randn(3, 7, D)
    assert torch.allclose(mx(v).norm(dim=-1), v.norm(dim=-1), atol=1e-4), mode


def test_pair_mixer_perm_has_no_trained_params():
    assert sum(p.numel() for p in PairMixer(D, "perm").parameters()) == 0
    assert sum(p.numel() for p in PairMixer(D, "ortho").parameters()) == 0
    assert sum(p.numel() for p in PairMixer(D, "learned", block=64).parameters()) > 0


def test_pair_mixer_block_must_divide_dk():
    with pytest.raises(ValueError):
        PairMixer(D, "ortho", block=48)                # 128 % 48 != 0


def test_mixer_plus_angle_rotation_preserves_lifted_norm():
    """Orthogonal mix then exact (angle) rotation preserves the lifted L2 norm."""
    torch.manual_seed(0)
    m = SpectralKoopmanMLP(D, rotation_param="angle", pair_mixer="ortho").eval()
    with torch.no_grad():
        m.theta.copy_(torch.empty_like(m.theta).uniform_(-3, 3))
        x = torch.randn(2, 6, D)
        g = torch.nn.functional.silu(m.lift(m.norm(x)))
        z = m._rotate(m.mixer(g))
    assert torch.allclose(z.norm(dim=-1), g.norm(dim=-1), atol=1e-4)


def test_mixer_breaks_adjacency():
    """The mix genuinely reorders/combines coords (output != input)."""
    for mode in ("perm", "ortho", "learned"):
        mx = PairMixer(D, mode, block=64)
        v = torch.randn(4, D)
        assert not torch.allclose(mx(v), v)


# --------------------------------------------------------------------------- #
# All option combinations: build + forward/backward finite
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("Klass,rot,rn,mix",
    list(itertools.product(CLASSES, ROTATIONS, [False, True], MIXERS)))
def test_option_combo_forward_backward_finite(Klass, rot, rn, mix):
    m = Klass(D, rotation_param=rot, row_norm_lift=rn, pair_mixer=mix,
              mixer_block=64, depth_grade=True, layer_idx=2, n_layers=8)
    x = torch.randn(2, 5, D, requires_grad=True)
    y = m(x)
    assert y.shape == x.shape
    y.pow(2).mean().backward()
    assert torch.isfinite(y).all()
    assert torch.isfinite(x.grad).all()
    for n, p in m.named_parameters():
        assert p.grad is None or torch.isfinite(p.grad).all(), n


# --------------------------------------------------------------------------- #
# Back-compat: defaults reproduce v1
# --------------------------------------------------------------------------- #

def test_default_mlp_is_legacy_and_names_unchanged():
    m = SpectralKoopmanMLP(D)
    assert _resolve_rotation_param(m) == "legacy"
    names = dict(m.named_parameters())
    assert "lift.weight" in names and "readout.weight" in names
    assert "gamma" in names and "omega" in names
    assert "lift_v" not in names and "lift_g" not in names and "s" not in names
    assert m.mixer is None


def test_norm_preserving_still_resolves_to_angle():
    m = SpectralKoopmanMLP(D, norm_preserving=True)
    assert _resolve_rotation_param(m) == "angle"
    assert "theta" in dict(m.named_parameters())


def test_default_forward_matches_prechange_math():
    """Default path == the exact v1 lift/rotate/readout composition."""
    torch.manual_seed(0)
    m = SpectralKoopmanMLP(D).eval()
    x = torch.randn(2, 4, D)
    with torch.no_grad():
        h = m.norm(x)
        g = torch.nn.functional.silu(m.lift(h))
        gp = g.view(*g.shape[:-1], m.d_k // 2, 2)
        gamma, omega = _rotation_coeffs(m)
        z1 = gamma * gp[..., 0] + omega * gp[..., 1]
        z2 = -omega * gp[..., 0] + gamma * gp[..., 1]
        z = torch.stack([z1, z2], -1).reshape_as(g)
        expected = x + m.readout(z)
    assert torch.allclose(m(x), expected, atol=1e-6)


# --------------------------------------------------------------------------- #
# Model-level wiring + diagnostics (all-SKA tiny KoopmanLM -> no mamba_ssm)
# --------------------------------------------------------------------------- #

def _tiny_koopman(**overrides):
    from koopman_lm.models.koopman_lm import KoopmanLM
    n = overrides.pop("n_layers", 3)
    cfg = build_config("180m_v2")
    cfg = dataclasses.replace(
        cfg, vocab_size=128, n_layers=n, d_model=128, ska_n_heads=4,
        ska_rank=16, d_state=64, ska_layer_indices=tuple(range(n)), **overrides)
    return KoopmanLM(cfg), cfg


def test_layer_idx_threaded_and_depth_grade_monotone_across_model():
    model, _ = _tiny_koopman(n_layers=5)
    rhos = []
    with torch.no_grad():
        for i, mlp in enumerate(model.mlp_layers):
            assert mlp.layer_idx == i and mlp.n_layers == 5
            g, o = _rotation_coeffs(mlp)
            rhos.append(float((g ** 2 + o ** 2).sqrt().mean()))
    assert all(a < b for a, b in zip(rhos, rhos[1:]))


def test_no_weight_decay_names_are_geometric_and_wn_params():
    model, _ = _tiny_koopman()
    skip = model.no_weight_decay_param_names()
    leaves = {n.rsplit(".", 1)[-1] for n in skip}
    assert leaves == {"lift_v", "lift_g", "s", "A_raw"}
    # every skipped name is a real parameter; big linears are NOT skipped
    allp = dict(model.named_parameters())
    assert skip <= set(allp)
    assert not any(n.endswith("readout.weight") for n in skip)


def test_v1_config_has_empty_weight_decay_skip():
    model, _ = _tiny_koopman(mlp_row_norm_lift=False, mlp_rotation_param=None,
                             mlp_pair_mixer=None, mlp_decay_depth_grade=False)
    assert model.no_weight_decay_param_names() == set()


def test_reset_projection_params_keeps_v2_params_after_reinit():
    """KoopmanLM._init_weights + reset_projection_params leaves v2 params intact
    (row-norm gains uniform, mixer present, rotation s/theta registered)."""
    model, _ = _tiny_koopman()
    for mlp in model.mlp_layers:
        assert mlp.row_norm_lift and mlp.mixer is not None
        g = mlp.effective_lift_row_norms()
        assert float(g.std() / g.mean()) < 1e-4        # gains still uniform
        assert torch.isfinite(mlp.readout.weight).all()


def test_utilization_report_runs_and_detects_planted_pathologies():
    from koopman_lm.modules.channel_mixer.koopman_diag import (
        utilization_report, gain_stats, dead_pair_stats, format_report)
    model, cfg = _tiny_koopman(n_layers=4)
    model.eval()
    ids = torch.randint(0, cfg.vocab_size, (2, 32))

    rep = utilization_report(model, ids)
    assert len(rep["gains"]["per_layer"]) == 4
    assert len(rep["dead_pairs"]["per_layer"]) == 4
    for key in ("mean", "std", "cv", "dead_frac"):
        assert key in rep["gains"]["pooled"]
    assert isinstance(format_report(rep), str)

    # plant dead neurons: zero the gains of layer 0 -> its dead fraction jumps
    with torch.no_grad():
        model.mlp_layers[0].lift_g.zero_()
    gs = gain_stats(model)
    assert gs["per_layer"][0]["dead_frac"] > 0.99      # whole layer reads dead


def test_param_count_estimate_reflects_v2_additions():
    base = build_config("180m")
    v2 = build_config("180m_v2")
    # v2 only adds the learned mixer + per-row gains on top of the same backbone;
    # a few M more, not a doubling.
    d = v2.param_count_estimate() - base.param_count_estimate()
    assert 0 < d < 10_000_000
