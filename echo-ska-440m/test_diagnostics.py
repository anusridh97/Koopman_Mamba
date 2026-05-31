"""
test_diagnostics.py -- CPU-only tests for SKA health instrumentation.

Run from the echo-ska-440m/ directory:
    python test_diagnostics.py

Covers:
  1. SKAModule.collect_diagnostics math invariants (radius<=1, lambda_min>=ridge,
     beta~0.5 and gate~init at init, all finite, correct per-head shapes/keys).
  2. Behavioral check: a persistent (near-constant) key sequence yields a larger
     spectral radius than a transient (random) one -- the metric tracks what it
     claims to. The persistent case is also rank-deficient, so lambda_min is
     pinned at the ridge floor.
  3. SKAHealthMonitor: hook registration, single-forward capture, and the
     wandb-ready dict schema (scalars + per-head distributions), residual ratio.
  4. profile_overhead() runs and returns finite amortized cost.

No GPU, no mamba_ssm, no wandb required.
"""

import math
import torch
import torch.nn as nn

from koopman_lm.config import KoopmanLMConfig
from koopman_lm.ska import SKAModule
from koopman_lm.model import SKABlock
from koopman_lm.diagnostics import SKAHealthMonitor, profile_overhead


torch.manual_seed(0)

D, H, R, CS = 64, 4, 16, 16
RIDGE = 1e-3


def _make_ska(layerscale=True):
    return SKAModule(
        d_model=D, n_heads=H, rank=R, head_dim=D // H,
        ridge_eps=RIDGE, power_K=2, chunk_size=CS,
        backend="pytorch", layerscale=layerscale, layerscale_init=1e-4,
    )


def _finite(t):
    return bool(torch.isfinite(t).all())


# ---------------------------------------------------------------------------
# 1. metric invariants
# ---------------------------------------------------------------------------

def test_invariants():
    ska = _make_ska().eval()
    B, T = 3, 40
    m = ska.collect_diagnostics(torch.randn(B, T, D))

    expected = {"spectral_radius", "lambda_min", "gap", "n_chunks", "gate_mag",
                "beta_mean", "outproj_norm", "ridge_eps"}
    assert expected <= set(m), f"missing keys: {expected - set(m)}"

    rad, lmin, gap = m["spectral_radius"], m["lambda_min"], m["gap"]
    nc = m["n_chunks"]
    # full per-(batch, chunk, head) distributions -- nothing pre-averaged
    assert rad.shape == (B, nc, H), f"expected (B,nc,H), got {tuple(rad.shape)}"
    assert lmin.shape == (B, nc, H) and gap.shape == (B, nc, H)

    for name, t in m.items():
        if torch.is_tensor(t):
            assert _finite(t), f"{name} has non-finite values"

    # A_eff = alpha * W has sigma_max <= 1 by construction => spectral radius <= 1
    assert float(rad.max()) <= 1.0 + 1e-4, f"radius>1: {float(rad.max())}"
    assert float(rad.min()) >= 0.0
    # Lemma A.4: lambda_min(G + ridge I) >= ridge
    assert float(lmin.min()) >= RIDGE - 1e-5, \
        f"lambda_min {float(lmin.min())} < ridge {RIDGE}"
    assert float(gap.min()) >= 0.0
    # at init: beta = sigmoid(0) = 0.5, gate = layerscale_init
    assert abs(float(m["beta_mean"]) - 0.5) < 0.05, float(m["beta_mean"])
    assert abs(float(m["gate_mag"]) - 1e-4) < 1e-5, float(m["gate_mag"])
    print("  [ok] invariants: shape (B,nc,H), radius<=1, lambda_min>=ridge, "
          "beta~0.5, gate~init")


def test_gate_fallback_without_layerscale():
    # when layerscale is off, gate_mag falls back to |eta| (not None)
    ska = _make_ska(layerscale=False).eval()
    m = ska.collect_diagnostics(torch.randn(2, 24, D))
    assert ska.layerscale_gate is None
    assert _finite(m["gate_mag"]) and float(m["gate_mag"]) > 0
    print("  [ok] gate_mag falls back to |eta| when layerscale disabled")


# ---------------------------------------------------------------------------
# 2. behavioral: persistent vs transient
# ---------------------------------------------------------------------------

def _valid_mean(t):
    """Mean over history-bearing chunks (drop chunk 0), matching the monitor."""
    return t[:, 1:].mean().item() if t.shape[1] > 1 else t.mean().item()


def test_persistence_tracks_radius():
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
    print(f"  [ok] persistence -> higher radius "
          f"(const={rad_const:.3f} > rand={rad_rand:.3f})")


def test_rank_deficiency_pins_lambda_min():
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
    print(f"  [ok] rank-deficient keys pin lambda_min at ridge floor ({lmin:.2e})")


# ---------------------------------------------------------------------------
# 3. monitor aggregation + schema
# ---------------------------------------------------------------------------

class _FakeMamba(nn.Module):
    """Stand-in for Mamba2Block (no mamba_ssm dependency)."""
    def __init__(self, d):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.proj = nn.Linear(d, d)

    def forward(self, x):
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


def _tiny_cfg():
    return KoopmanLMConfig(
        d_model=D, n_layers=4, vocab_size=128,
        ska_n_heads=H, ska_rank=R, ska_chunk_size=CS,
        ska_ridge=RIDGE, ska_layerscale=True, ska_layerscale_init=1e-4,
        ska_short_conv=False, ska_layer_indices=[1, 3],
    )


def test_monitor_schema():
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
                       "beta_mean", "frac_healthy", "frac_unstable",
                       "residual_delta"):
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
    print(f"  [ok] monitor schema: {len(metrics)} keys, n_ska={monitor.n_ska}, "
          f"full/by_head/by_chunk views present, residual_ratio={rr:.3e}")


def test_monitor_inactive_is_noop():
    cfg = _tiny_cfg()
    model = _StandInModel(cfg).eval()
    monitor = SKAHealthMonitor(model, ska_cls=SKABlock)
    # forward WITHOUT capture(): hooks must not buffer anything
    model(torch.randn(2, 16, D))
    assert monitor.collect() == {}, "inactive monitor should produce no metrics"
    print("  [ok] inactive monitor is a no-op")


# ---------------------------------------------------------------------------
# 4. overhead profiler
# ---------------------------------------------------------------------------

def test_profile_overhead_runs():
    cfg = _tiny_cfg()
    model = _StandInModel(cfg)
    monitor = SKAHealthMonitor(model, ska_cls=SKABlock)

    def batch_fn():
        return {"x": torch.randn(2, 40, D)}

    stats = profile_overhead(model, batch_fn, monitor, n_warmup=1, n_iter=3)
    for k, v in stats.items():
        assert math.isfinite(v), f"{k} not finite"
    print(f"  [ok] profile_overhead: plain={stats['plain_step_s']*1e3:.1f}ms "
          f"diag={stats['diag_step_s']*1e3:.1f}ms "
          f"every500={stats['overhead_every_500']*100:.3f}%")


if __name__ == "__main__":
    tests = [
        test_invariants,
        test_gate_fallback_without_layerscale,
        test_persistence_tracks_radius,
        test_rank_deficiency_pins_lambda_min,
        test_monitor_schema,
        test_monitor_inactive_is_noop,
        test_profile_overhead_runs,
    ]
    print("Running SKA diagnostics tests (CPU)...")
    for t in tests:
        t()
    print(f"\nAll {len(tests)} tests passed.")
