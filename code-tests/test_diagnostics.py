"""SKA health diagnostics (Phase 1): CPU-only tests, no mamba_ssm / wandb.

Covers:
  1. SKAModule.collect_diagnostics math invariants (radius<=1, lambda_min>=ridge,
     beta~0.5 and gate~init at init, all finite, correct per-head shapes/keys),
     plus the gamma-in-A_eff adaptation for the new module's learnable regimes.
  2. Behavioral check: a persistent (near-constant) key sequence yields a larger
     spectral radius than a transient (random) one -- the metric tracks what it
     claims to. The persistent case is also rank-deficient, so lambda_min is
     pinned at the ridge floor.
  3. SKAHealthMonitor: hook registration, single-forward capture, the
     wandb-ready dict schema (scalars + per-head distributions), residual
     ratio, and attachment to a real (all-SKA) KoopmanLM.
  4. profile_overhead() runs and returns finite amortized cost.
  5. Load-bearing four-mode decomposition via the _ablate passthrough flags
     (the new package's mechanism behind KoopmanLM.ablate()).
  6. GradFlowMonitor: schema, no-backward noop, frozen-SKA, jacobian rank.

Mamba2Block needs mamba_ssm (GPU box only), so the stacked-model tests use a
fake-Mamba stand-in; the real-model test uses an all-SKA KoopmanLM config.
"""
import math
from contextlib import contextmanager

import pytest
import torch
import torch.nn as nn

from koopman_lm.globals.config import KoopmanLMConfig
from koopman_lm.globals.modules.token_mixer import SKAModule
from koopman_lm.models.koopman_lm import SKABlock, KoopmanLM
from koopman_lm.training.diagnostics import (
    SKAHealthMonitor, GradFlowMonitor, profile_overhead,
)

pytestmark = pytest.mark.correctness

D, H, R, CS = 64, 4, 16, 16
RIDGE = 1e-3


def _make_ska(layerscale=True, **kw):
    return SKAModule(
        d_model=D, n_heads=H, rank=R, head_dim=D // H,
        ridge_eps=RIDGE, power_K=2, chunk_size=CS,
        layerscale=layerscale, layerscale_init=1e-4, **kw)


def _finite(t):
    return bool(torch.isfinite(t).all())


# ---------------------------------------------------------------------------
# 1. metric invariants
# ---------------------------------------------------------------------------

def test_invariants():
    torch.manual_seed(0)
    ska = _make_ska().eval()
    B, T = 2, 40
    m = ska.collect_diagnostics(torch.randn(B, T, D))

    expected = {"spectral_radius", "lambda_min", "gap", "n_chunks", "gate_mag",
                "beta_mean", "outproj_norm", "eta", "gamma", "ridge_eps"}
    assert expected <= set(m), f"missing keys: {expected - set(m)}"

    rad, lmin, gap = m["spectral_radius"], m["lambda_min"], m["gap"]
    nc = m["n_chunks"]
    # full per-(batch, chunk, head) distributions -- nothing pre-averaged
    assert rad.shape == (B, nc, H), f"expected (B,nc,H), got {tuple(rad.shape)}"
    assert lmin.shape == (B, nc, H) and gap.shape == (B, nc, H)

    for name, t in m.items():
        if torch.is_tensor(t):
            assert _finite(t), f"{name} has non-finite values"

    # A_eff = gamma * alpha * W with gamma=1 (fixed default) has sigma_max <= 1
    # by construction => spectral radius <= 1
    assert float(rad.max()) <= 1.0 + 1e-4, f"radius>1: {float(rad.max())}"
    assert float(rad.min()) >= 0.0
    # Lemma A.4: lambda_min(G + ridge I) >= ridge
    assert float(lmin.min()) >= RIDGE - 1e-5, \
        f"lambda_min {float(lmin.min())} < ridge {RIDGE}"
    assert float(gap.min()) >= 0.0
    # at init: beta = sigmoid(0) = 0.5, gate = layerscale_init
    assert abs(float(m["beta_mean"]) - 0.5) < 0.05, float(m["beta_mean"])
    assert abs(float(m["gate_mag"]) - 1e-4) < 1e-5, float(m["gate_mag"])
    # fixed defaults: eta buffer 1.0, gamma const 1.0
    assert abs(float(m["eta"]) - 1.0) < 1e-6
    assert abs(float(m["gamma"]) - 1.0) < 1e-6


def test_max_batch_caps_diagnostic_batch():
    torch.manual_seed(0)
    ska = _make_ska().eval()
    m = ska.collect_diagnostics(torch.randn(5, 40, D), max_batch=2)
    assert m["spectral_radius"].shape[0] == 2, \
        "max_batch should cap the diagnosed batch"


def test_gate_fallback_without_layerscale():
    # when layerscale is off, gate_mag falls back to |resolved eta| (not None)
    torch.manual_seed(0)
    ska = _make_ska(layerscale=False).eval()
    m = ska.collect_diagnostics(torch.randn(2, 24, D))
    assert ska.layerscale_gate is None
    assert _finite(m["gate_mag"]) and float(m["gate_mag"]) > 0
    assert abs(float(m["gate_mag"]) - float(m["eta"])) < 1e-6


def test_gate_fallback_with_squashed_eta():
    # the eta_bounds (squash) regime has no .eta attribute, only eta_raw;
    # gate_mag must resolve through _resolve_eta()
    torch.manual_seed(0)
    ska = _make_ska(layerscale=False, eta_value=1.5, eta_bounds=(1.4, 1.7)).eval()
    m = ska.collect_diagnostics(torch.randn(2, 24, D))
    assert abs(float(m["eta"]) - 1.5) < 1e-5
    assert abs(float(m["gate_mag"]) - 1.5) < 1e-5


def test_gamma_scales_effective_operator():
    """The forward applies Y * gamma^K, i.e. gamma * alpha * W per filter step.
    The diagnosed spectral radius must include that gamma (it is what the
    module actually applies), so halving gamma halves the radius exactly."""
    torch.manual_seed(0)
    ska_1 = _make_ska(gamma_learnable=False, gamma_value=1.0).eval()
    torch.manual_seed(0)
    ska_h = _make_ska(gamma_learnable=False, gamma_value=0.5).eval()
    x = torch.randn(2, 48, D)

    torch.manual_seed(123)   # _spectral_radius power iteration uses randn
    rad_1 = ska_1.collect_diagnostics(x)["spectral_radius"]
    torch.manual_seed(123)
    rad_h = ska_h.collect_diagnostics(x)["spectral_radius"]

    assert torch.allclose(rad_h, 0.5 * rad_1, rtol=1e-4, atol=1e-6), \
        "gamma is not folded into the diagnosed operator"


# ---------------------------------------------------------------------------
# 2. behavioral: persistent vs transient
# ---------------------------------------------------------------------------

def _valid_mean(t):
    """Mean over history-bearing chunks (drop chunk 0), matching the monitor."""
    return t[:, 1:].mean().item() if t.shape[1] > 1 else t.mean().item()


def test_persistence_tracks_radius():
    torch.manual_seed(0)
    ska = _make_ska().eval()
    B, T = 2, 48

    # transient: independent keys each step
    x_rand = torch.randn(B, T, D)
    rad_rand = _valid_mean(ska.collect_diagnostics(x_rand)["spectral_radius"])

    # persistent: (almost) the same token repeated -> lag-1 operator ~ I
    base = torch.randn(B, 1, D)
    x_const = base.expand(B, T, D) + 0.02 * torch.randn(B, T, D)
    rad_const = _valid_mean(ska.collect_diagnostics(x_const)["spectral_radius"])

    assert rad_const > rad_rand, \
        f"persistent radius {rad_const:.3f} should exceed transient {rad_rand:.3f}"
    assert rad_const > 0.7, f"persistent radius {rad_const:.3f} should be near 1"


def test_rank_deficiency_pins_lambda_min():
    torch.manual_seed(0)
    ska = _make_ska().eval()
    B, T = 2, 48
    base = torch.randn(B, 1, D)
    x_const = base.expand(B, T, D).contiguous()   # exactly rank-1 keys
    lam = ska.collect_diagnostics(x_const)["lambda_min"]
    lmin = lam[:, 1:].min().item()   # history-bearing chunks only
    # rank-deficient keys => smallest Gram eig sits on the ridge floor.
    # chunk_stats adds ridge + a 1e-4 jitter, so the floor is ridge + 1e-4.
    floor = RIDGE + 1e-4
    assert abs(lmin - floor) < 3e-5, \
        f"lambda_min {lmin} not pinned at ridge floor {floor}"


# ---------------------------------------------------------------------------
# 3. monitor aggregation + schema
# ---------------------------------------------------------------------------

class _FakeMamba(nn.Module):
    """Stand-in for Mamba2Block (no mamba_ssm dependency). Supports the same
    _ablate passthrough flag as the real sequence blocks."""
    def __init__(self, d):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.proj = nn.Linear(d, d)
        self._ablate = False

    def forward(self, x):
        if self._ablate:
            return x
        return x + 0.1 * self.proj(self.norm(x))


class _StandInModel(nn.Module):
    """Minimal Nemotron-H-style stack: alternating fake-Mamba and SKA blocks."""
    def __init__(self, cfg):
        super().__init__()
        self.seq_layers = nn.ModuleList([
            _FakeMamba(cfg.d_model),
            SKABlock(cfg),
            _FakeMamba(cfg.d_model),
            SKABlock(cfg),
        ])

    def forward(self, x):
        for layer in self.seq_layers:
            x = layer(x)
        return {"loss": x.pow(2).mean(), "logits": x}


def _tiny_cfg(**kw):
    base = dict(
        d_model=D, n_layers=4, vocab_size=128,
        ska_n_heads=H, ska_rank=R, ska_chunk_size=CS,
        ska_ridge=RIDGE, ska_layerscale=True, ska_layerscale_init=1e-4,
        ska_short_conv=False, ska_layer_indices=(1, 3),
    )
    base.update(kw)
    return KoopmanLMConfig(**base)


def test_monitor_schema():
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    model = _StandInModel(cfg).eval()
    monitor = SKAHealthMonitor(model, ska_cls=SKABlock)
    assert monitor.n_ska == 2

    B, T = 2, 40
    with monitor.capture():
        model(torch.randn(B, T, D))
    metrics = monitor.collect(wrap_histograms=False)

    # per-SKA-layer scalar keys present for layers 1 and 3
    for idx in (1, 3):
        for suffix in ("spectral_radius_mean", "spectral_radius_max",
                       "lambda_min_over_ridge", "gap_mean", "gate_mag",
                       "beta_mean", "eta", "gamma", "frac_healthy",
                       "frac_unstable", "residual_delta"):
            k = f"ska/L{idx}/{suffix}"
            assert k in metrics, f"missing scalar {k}"
            assert isinstance(metrics[k], float) and math.isfinite(metrics[k]), k
        # full-pool distribution (list when wrap_histograms=False): B*(nc-1)*H
        full = metrics[f"ska/L{idx}/spectral_radius"]
        assert isinstance(full, list) and len(full) > 0, f"bad full hist L{idx}"
        # per-head breakdown has length n_heads
        by_head = metrics[f"ska/L{idx}/spectral_radius_by_head"]
        assert len(by_head) == H, f"by_head len {len(by_head)} != {H}"
        # per-chunk breakdown has one entry per history-bearing chunk
        by_chunk = metrics[f"ska/L{idx}/spectral_radius_by_chunk"]
        assert len(by_chunk) >= 1, f"by_chunk empty for L{idx}"
        # full pool = B * (nc-1) * H
        assert len(full) == B * len(by_chunk) * H, \
            f"full pool {len(full)} != B*(nc-1)*H"

    rr = metrics["ska/residual_ratio"]
    assert math.isfinite(rr) and rr > 0, f"residual_ratio invalid: {rr}"
    # no metrics for the fake-mamba layers (0, 2) -- they are the baseline bucket
    assert "ska/L0/spectral_radius_mean" not in metrics


def test_monitor_inactive_is_noop():
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    model = _StandInModel(cfg).eval()
    monitor = SKAHealthMonitor(model, ska_cls=SKABlock)
    # forward WITHOUT capture(): hooks must not buffer anything
    model(torch.randn(2, 16, D))
    assert monitor.collect() == {}, "inactive monitor should produce no metrics"


def test_monitor_on_all_ska_koopman_lm():
    """The monitor attaches to a REAL KoopmanLM (all-SKA layout so no
    mamba_ssm is needed on CPU) with the default block-class resolution."""
    torch.manual_seed(0)
    cfg = _tiny_cfg(ska_layer_indices=(0, 1, 2, 3))
    model = KoopmanLM(cfg).eval()
    monitor = SKAHealthMonitor(model)      # default ska_cls/mamba_cls resolution
    assert monitor.n_ska == 4

    ids = torch.randint(0, cfg.vocab_size, (2, 40))
    with torch.no_grad(), monitor.capture():
        model(input_ids=ids)
    metrics = monitor.collect(wrap_histograms=False)

    for idx in range(4):
        assert f"ska/L{idx}/spectral_radius_mean" in metrics
    # no non-SKA layers -> no baseline bucket -> no residual ratio
    assert "ska/residual_ratio" not in metrics


# ---------------------------------------------------------------------------
# 4. overhead profiler
# ---------------------------------------------------------------------------

def test_profile_overhead_runs():
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    model = _StandInModel(cfg)
    monitor = SKAHealthMonitor(model, ska_cls=SKABlock)

    def batch_fn():
        return {"x": torch.randn(2, 40, D)}

    stats = profile_overhead(model, batch_fn, monitor, n_warmup=1, n_iter=3)
    for k, v in stats.items():
        assert math.isfinite(v), f"{k} not finite"


# ---------------------------------------------------------------------------
# 5. Load-bearing eval -- four-mode zeroing via the _ablate flags
# ---------------------------------------------------------------------------
#
# In the old layout, zeroing was done with forward hooks in
# koopman-lm-fast/evaluate.py. The new package bakes the mechanism into the
# blocks themselves: every sequence block carries an ``_ablate`` flag that
# turns it into a pure residual passthrough (see KoopmanLM.ablate()). These
# tests validate the four-mode decomposition through that mechanism, on a
# stand-in stack whose fake-Mamba honors the same flag.


@contextmanager
def _zero_blocks(model, pred):
    """Turn every seq block matching pred into a residual passthrough."""
    blocks = [l for l in model.seq_layers if pred(l)]
    try:
        for b in blocks:
            b._ablate = True
        yield
    finally:
        for b in blocks:
            b._ablate = False


def _zero_ska_ctx(model):
    return _zero_blocks(model, lambda l: isinstance(l, SKABlock))


def _zero_mamba_ctx(model):
    return _zero_blocks(model, lambda l: not isinstance(l, SKABlock))


def _zero_both_ctx(model):
    return _zero_blocks(model, lambda l: True)


class _StandInLM(nn.Module):
    """_StandInModel + embedding + lm_head for CE-loss-based tests.

    Exposes .seq_layers at the top level so the zeroing helpers can find it.
    """
    def __init__(self, cfg):
        super().__init__()
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.backbone = _StandInModel(cfg)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        # zeroing helpers scan model.seq_layers, so expose it here
        self.seq_layers = self.backbone.seq_layers

    def forward(self, input_ids, labels=None):
        h = self.embed(input_ids)                      # (B, T, d)
        h_out = self.backbone(h)["logits"]             # (B, T, d)
        logits = self.lm_head(h_out)                   # (B, T, V)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
            )
        return {"loss": loss, "logits": logits}


def _capture_layer_io(model, layer_pred):
    """Return a list of (input_tensor, output_tensor) for every layer
    matching layer_pred, captured during the next forward pass."""
    records = []

    def _hook(m, inp, out):
        records.append((inp[0].detach().clone(), out.detach().clone()))

    hooks = [
        layer.register_forward_hook(_hook)
        for layer in model.seq_layers
        if layer_pred(layer)
    ]
    return records, hooks


def test_zeroing_ska_is_isolated():
    """Zeroing SKA: every SKABlock output == input, while the fake-Mamba
    blocks still compute (output != input)."""
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    model = _StandInModel(cfg).eval()
    x = torch.randn(2, 32, D)

    ska_rec, ska_hooks = _capture_layer_io(model, lambda l: isinstance(l, SKABlock))
    mam_rec, mam_hooks = _capture_layer_io(model, lambda l: not isinstance(l, SKABlock))
    with torch.no_grad(), _zero_ska_ctx(model):
        model(x)
    for h in ska_hooks + mam_hooks:
        h.remove()

    assert len(ska_rec) == 2 and len(mam_rec) == 2
    for inp, out in ska_rec:
        assert torch.equal(inp, out), "SKA output != input after zeroing"
    for i, (inp, out) in enumerate(mam_rec):
        assert not torch.allclose(inp, out), \
            f"Mamba layer {i} was accidentally zeroed too"


def test_zeroing_mamba_is_isolated():
    """Zeroing Mamba: every non-SKA output == input, SKA blocks still compute."""
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    model = _StandInModel(cfg).eval()
    x = torch.randn(2, 32, D)

    ska_rec, ska_hooks = _capture_layer_io(model, lambda l: isinstance(l, SKABlock))
    mam_rec, mam_hooks = _capture_layer_io(model, lambda l: not isinstance(l, SKABlock))
    with torch.no_grad(), _zero_mamba_ctx(model):
        model(x)
    for h in ska_hooks + mam_hooks:
        h.remove()

    assert len(ska_rec) == 2 and len(mam_rec) == 2
    for inp, out in mam_rec:
        assert torch.equal(inp, out), "Mamba output != input after zeroing"
    for i, (inp, out) in enumerate(ska_rec):
        assert not torch.allclose(inp, out), \
            f"SKA layer {i} was accidentally zeroed too"


def test_full_mode_matches_baseline():
    """Running inside _zero_ska_ctx then exiting and running again must give
    the same result as a plain forward -- i.e. flags are fully restored."""
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    model = _StandInModel(cfg).eval()
    x = torch.randn(2, 32, D)

    with torch.no_grad():
        with _zero_ska_ctx(model):
            out_zeroed = model(x)["logits"].detach().clone()
        out_full_1 = model(x)["logits"].detach().clone()
        out_full_2 = model(x)["logits"].detach().clone()

    # two plain forwards must be identical (deterministic)
    assert torch.allclose(out_full_1, out_full_2), \
        "baseline forwards are not deterministic"
    # zeroed and full must differ (SKA actually contributes something)
    assert not torch.allclose(out_zeroed, out_full_1), \
        "zeroed forward == full forward -- SKA contributes nothing at init " \
        "(layerscale_init may be too small; increase for this test)"
    # after context exits every _ablate flag must be back to False
    assert all(not layer._ablate for layer in model.seq_layers), \
        "_ablate flags were not restored after context exit"


def test_ppl_ordering_both_zeroed_worst():
    """Four-mode loss sanity: all values finite/positive, each zeroing mode
    changes the loss relative to the full model (zeroing actually does
    something).

    Note: the stronger ordering both_zeroed >= ska_zeroed does NOT hold for
    random weights -- a random Mamba branch can accidentally reduce loss. The
    meaningful invariant is that each mode produces a distinct result.
    """
    cfg = _tiny_cfg()
    torch.manual_seed(1)
    model = _StandInLM(cfg).eval()

    B, T = 4, 48
    input_ids = torch.randint(0, cfg.vocab_size, (B, T))
    labels = torch.randint(0, cfg.vocab_size, (B, T))

    def _loss(ctx=None):
        with torch.no_grad():
            if ctx is not None:
                with ctx(model):
                    return model(input_ids, labels=labels)["loss"].item()
            return model(input_ids, labels=labels)["loss"].item()

    loss_full = _loss()
    loss_ska_zeroed = _loss(_zero_ska_ctx)
    loss_mamba_zeroed = _loss(_zero_mamba_ctx)
    loss_both_zeroed = _loss(_zero_both_ctx)

    for name, val in [("full", loss_full), ("ska_zeroed", loss_ska_zeroed),
                      ("mamba_zeroed", loss_mamba_zeroed),
                      ("both_zeroed", loss_both_zeroed)]:
        assert math.isfinite(val) and val > 0, f"{name} loss not finite/positive: {val}"

    # Each zeroing mode must change the loss -- the flags are actually doing something
    assert loss_ska_zeroed != loss_full, "zeroing SKA had no effect on loss"
    assert loss_mamba_zeroed != loss_full, "zeroing Mamba had no effect on loss"
    assert loss_both_zeroed != loss_full, "zeroing both had no effect on loss"


def test_known_delta_synthetic():
    """With Mamba nearly silent and SKA gate at 1.0, zeroing SKA perturbs the
    output more than zeroing Mamba.

    We use output L2 distance rather than loss change, because loss
    directionality is not guaranteed with random weights (a dominant-but-random
    SKA adds noise and its removal can improve loss). Output perturbation is
    sign-agnostic: a larger gate unconditionally means a larger shift when that
    branch is removed.
    """
    cfg = _tiny_cfg()
    torch.manual_seed(2)
    model = _StandInModel(cfg).eval()

    # Suppress Mamba to near-identity
    with torch.no_grad():
        for layer in model.seq_layers:
            if not isinstance(layer, SKABlock):
                layer.proj.weight.mul_(1e-4)

    # Make SKA residual substantial
    with torch.no_grad():
        for layer in model.seq_layers:
            if isinstance(layer, SKABlock):
                if layer.ska.layerscale_gate is not None:
                    layer.ska.layerscale_gate.fill_(1.0)

    x = torch.randn(4, 48, D)
    with torch.no_grad():
        out_full = model(x)["logits"].detach()
        with _zero_ska_ctx(model):
            out_ska_z = model(x)["logits"].detach()
        with _zero_mamba_ctx(model):
            out_mamba_z = model(x)["logits"].detach()

    ska_perturb = (out_ska_z - out_full).norm().item()
    mamba_perturb = (out_mamba_z - out_full).norm().item()

    assert ska_perturb > mamba_perturb, (
        f"SKA perturbation ({ska_perturb:.4f}) should exceed Mamba "
        f"perturbation ({mamba_perturb:.4f}) when SKA gate=1.0 and Mamba~0"
    )


# ---------------------------------------------------------------------------
# 6. Gradient-flow tracking (Phase 1, Task 3)
# ---------------------------------------------------------------------------
#
# GradFlowMonitor registers param.register_hook on SKA projection weights and
# non-SKA (Mamba) Linear weights. Tests verify:
#   a. Schema -- keys present and finite after a real backward pass.
#   b. No-backward noop -- collect() is empty without calling .backward().
#   c. Frozen SKA -- ratio absent when SKA params have requires_grad=False.
#   d. Both branches active -- ratio is finite and positive.
#   e. Jacobian rank (unit) -- SVD rank logic on synthetic matrices.
#   f. Jacobian rank (integration) -- rank from a real backward is in [1, max].


def test_grad_flow_schema():
    """After a backward pass, GradFlowMonitor emits the expected keys, all finite."""
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    model = _StandInModel(cfg)
    monitor = GradFlowMonitor(model, ska_cls=SKABlock)
    assert monitor._buf  # hooks registered

    B, T = 2, 40
    with monitor.capture():
        out = model(torch.randn(B, T, D))
        out["loss"].backward()
    metrics = monitor.collect()

    assert "ska/grad_norm_ratio" in metrics, "missing grad_norm_ratio"
    assert "ska/grad_norm_ska_mean" in metrics
    assert "ska/grad_norm_mamba_mean" in metrics
    for idx in (1, 3):   # SKA layers in _tiny_cfg
        k_norm = f"ska/L{idx}/grad_norm"
        k_rank = f"ska/L{idx}/jacobian_rank"
        k_frac = f"ska/L{idx}/jacobian_rank_frac"
        assert k_norm in metrics, f"missing {k_norm}"
        assert k_rank in metrics, f"missing {k_rank}"
        assert k_frac in metrics, f"missing {k_frac}"
        assert math.isfinite(metrics[k_norm]) and metrics[k_norm] > 0, k_norm
        assert isinstance(metrics[k_rank], int) and metrics[k_rank] >= 1, k_rank
        assert 0.0 < metrics[k_frac] <= 1.0, k_frac


def test_grad_flow_no_backward():
    """Without calling .backward(), collect() returns an empty dict."""
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    model = _StandInModel(cfg)
    monitor = GradFlowMonitor(model, ska_cls=SKABlock)

    with monitor.capture():
        model(torch.randn(2, 16, D))   # forward only, no backward
    assert monitor.collect() == {}, "no-backward should produce empty dict"


def test_grad_flow_frozen_ska():
    """Freezing SKA weights means no gradient reaches them.
    The ratio key must be absent (SKA has nothing to report), while Mamba
    still accumulates gradient normally.
    """
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    model = _StandInModel(cfg)
    monitor = GradFlowMonitor(model, ska_cls=SKABlock)

    # Freeze all four SKA projection weights AFTER hook registration.
    # requires_grad=False at backward time means autograd never calls the hook.
    for layer in model.seq_layers:
        if isinstance(layer, SKABlock):
            for attr in ("key_proj", "query_proj", "value_proj", "out_proj"):
                getattr(layer.ska, attr).weight.requires_grad_(False)

    with monitor.capture():
        out = model(torch.randn(2, 40, D))
        out["loss"].backward()
    metrics = monitor.collect()

    assert "ska/grad_norm_ratio" not in metrics, (
        "ratio should be absent when SKA is frozen (no SKA gradients)")
    assert "ska/grad_norm_mamba_mean" in metrics, "Mamba should still accumulate gradient"
    assert metrics["ska/grad_norm_mamba_mean"] > 0, "Mamba gradient norm should be positive"


def test_grad_flow_active_both_branches():
    """With both branches active (normal init), ratio is finite and positive."""
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    model = _StandInModel(cfg)
    monitor = GradFlowMonitor(model, ska_cls=SKABlock)

    with monitor.capture():
        out = model(torch.randn(2, 40, D))
        out["loss"].backward()
    metrics = monitor.collect()

    ratio = metrics["ska/grad_norm_ratio"]
    assert math.isfinite(ratio) and ratio > 0, (
        f"ratio should be finite and positive; got {ratio}")


def test_jacobian_rank_unit():
    """SVD rank logic in isolation: rank-1 outer product -> rank 1;
    random full-rank matrix -> rank min(m, n).
    """
    torch.manual_seed(0)
    threshold = 0.01   # matches GradFlowMonitor default

    # Rank-1: outer product has exactly one non-zero singular value
    u = torch.randn(H * R, 1)   # (64, 1)
    v = torch.randn(1, D)       # (1, 64)
    G_r1 = (u @ v).float()
    sv = torch.linalg.svdvals(G_r1)
    approx_rank = int((sv > sv.max() * threshold).sum().item())
    assert approx_rank == 1, f"outer-product rank should be 1, got {approx_rank}"

    # Full-rank random matrix: nearly all singular values are O(1).
    # Allow up to 2 SVs below the 1%-of-max threshold due to statistical
    # fluctuation at the tail of a square Gaussian random matrix.
    G_full = torch.randn(H * R, D).float()
    sv_full = torch.linalg.svdvals(G_full)
    approx_rank_full = int((sv_full > sv_full.max() * threshold).sum().item())
    expected = min(H * R, D)
    assert approx_rank_full >= expected - 2, (
        f"random matrix rank should be ~{expected}, got {approx_rank_full}")


def test_jacobian_rank_integration():
    """Rank captured from a real backward pass is in the valid range [1, min(rows, cols)]."""
    torch.manual_seed(0)
    cfg = _tiny_cfg()
    model = _StandInModel(cfg)
    monitor = GradFlowMonitor(model, ska_cls=SKABlock)

    with monitor.capture():
        out = model(torch.randn(2, 40, D))
        out["loss"].backward()
    metrics = monitor.collect()

    max_rank = min(H * R, D)   # min(4*16, 64) = 64
    for idx in (1, 3):
        rank = metrics.get(f"ska/L{idx}/jacobian_rank")
        assert rank is not None, f"missing jacobian_rank for L{idx}"
        assert 1 <= rank <= max_rank, (
            f"L{idx} rank {rank} outside [1, {max_rank}]")
        frac = metrics[f"ska/L{idx}/jacobian_rank_frac"]
        assert 0.0 < frac <= 1.0, f"rank_frac {frac} out of (0, 1]"
