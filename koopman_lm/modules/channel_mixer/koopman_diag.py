"""Koopman-MLP utilization diagnostics (Aurora-style measurement).

Aurora's credibility came from *measuring* the pathology (25% dead neurons by
step 500), not asserting it. These are the analogous measurements for the
spectral Koopman MLP, computable on any config (v1 or v2) so the v1-vs-v2
comparison is apples-to-apples:

  * gain_stats           -- the g_i histogram: per-lifted-dimension gain
    (effective lift row-norm). Its coefficient of variation is the headline
    "is utilization uniform?" number -- lower CV == tighter, more uniform use.
    Dead neurons show up directly as g_i near 0.

  * dead_pair_stats      -- per-pair activation variance in the lifted space.
    A rotation pair that never varies across tokens contributes nothing; the
    dead-pair fraction is the direct analog of Aurora's dead-neuron count.

  * utilization_report   -- both, bundled, with per-layer and pooled summaries.

All functions run under no_grad and touch only forward activations, so they are
cheap enough to log periodically during training.
"""
import torch
import torch.nn.functional as F

from koopman_lm.globals.modules.koopman_mlp import SpectralKoopmanMLP


def iter_koopman_mlps(model):
    """Yield (index, module) for every spectral Koopman MLP in the model."""
    idx = 0
    for m in model.modules():
        if isinstance(m, SpectralKoopmanMLP):
            yield idx, m
            idx += 1


def _summ(x):
    """Summary stats of a 1-D tensor, including the coefficient of variation."""
    x = x.float()
    mean = x.mean()
    std = x.std(unbiased=False)
    return {
        "mean": float(mean),
        "std": float(std),
        "cv": float(std / mean.clamp_min(1e-12)),   # dispersion of utilization
        "min": float(x.min()),
        "max": float(x.max()),
        "median": float(x.median()),
    }


def _histogram(x, bins=30):
    x = x.float()
    counts = torch.histc(x, bins=bins, min=float(x.min()), max=float(x.max()))
    return {
        "bins": torch.linspace(float(x.min()), float(x.max()), bins + 1).tolist(),
        "counts": counts.long().tolist(),
    }


@torch.no_grad()
def gain_stats(model, dead_rel_thresh=0.05, bins=30):
    """Per-lifted-dimension gain (g_i) statistics, per layer and pooled.

    A neuron counts as dead when its gain is below ``dead_rel_thresh`` times the
    layer's median gain -- a scale-free "this row has collapsed" test.
    """
    per_layer, pooled = [], []
    for i, mlp in iter_koopman_mlps(model):
        g = mlp.effective_lift_row_norms().float().cpu()
        pooled.append(g)
        med = g.median().clamp_min(1e-12)
        dead = float((g < dead_rel_thresh * med).float().mean())
        per_layer.append({"layer": i, "dead_frac": dead,
                          "row_norm_lift": mlp.row_norm_lift, **_summ(g)})
    allg = torch.cat(pooled) if pooled else torch.zeros(1)
    med = allg.median().clamp_min(1e-12)
    return {
        "per_layer": per_layer,
        "pooled": {"dead_frac": float((allg < dead_rel_thresh * med).float().mean()),
                   "histogram": _histogram(allg, bins), **_summ(allg)},
    }


@torch.no_grad()
def _capture_layer_inputs(model, input_ids):
    """Run one forward, returning each Koopman-MLP layer's input activations.

    Uses forward-pre-hooks so we grab the true residual-stream input to each MLP
    (the output of its paired sequence layer), independent of MLP internals.
    """
    captured = {}
    handles = []
    for i, mlp in iter_koopman_mlps(model):
        def hook(mod, args, _i=i):
            captured[_i] = args[0].detach()
            return None
        handles.append(mlp.register_forward_pre_hook(hook))
    try:
        model(input_ids)
    finally:
        for h in handles:
            h.remove()
    return captured


@torch.no_grad()
def dead_pair_stats(model, input_ids, dead_rel_thresh=0.01, bins=30):
    """Per-pair activation variance in the lifted space; dead-pair fraction.

    For each rotation pair (dims 2i, 2i+1) we take the variance of its lifted
    activations across all tokens, summed over the pair. A pair is dead when that
    variance falls below ``dead_rel_thresh`` times the layer median -- it never
    moves, so the rotation has nothing to rotate. Peak memory is one layer's
    lifted activations at a time (layers processed sequentially).
    """
    inputs = _capture_layer_inputs(model, input_ids)
    per_layer, pooled = [], []
    for i, mlp in iter_koopman_mlps(model):
        x = inputs[i]
        g_x = mlp.lifted(x)                              # (..., d_k), post-mix
        flat = g_x.reshape(-1, g_x.shape[-1]).float()   # (N, d_k)
        coord_var = flat.var(dim=0, unbiased=False)      # (d_k,)
        pair_var = coord_var.view(-1, 2).sum(dim=1).cpu()  # (d_k/2,)
        pooled.append(pair_var)
        med = pair_var.median().clamp_min(1e-12)
        dead = float((pair_var < dead_rel_thresh * med).float().mean())
        per_layer.append({"layer": i, "dead_frac": dead, "n_pairs": pair_var.numel(),
                          **_summ(pair_var)})
        del g_x, flat, coord_var
    allp = torch.cat(pooled) if pooled else torch.zeros(1)
    med = allp.median().clamp_min(1e-12)
    return {
        "per_layer": per_layer,
        "pooled": {"dead_frac": float((allp < dead_rel_thresh * med).float().mean()),
                   "histogram": _histogram(allp, bins), **_summ(allp)},
    }


@torch.no_grad()
def utilization_report(model, input_ids, gain_dead_rel_thresh=0.05,
                       pair_dead_rel_thresh=0.01):
    """Bundle the gain histogram and dead-pair statistics into one report dict."""
    return {
        "gains": gain_stats(model, dead_rel_thresh=gain_dead_rel_thresh),
        "dead_pairs": dead_pair_stats(model, input_ids,
                                      dead_rel_thresh=pair_dead_rel_thresh),
    }


def format_report(report):
    """Human-readable one-block-per-metric summary of a utilization_report()."""
    g = report["gains"]["pooled"]
    p = report["dead_pairs"]["pooled"]
    lines = [
        "Koopman-MLP utilization",
        "-" * 60,
        "gains g_i (lift row-norm) -- uniform use == low CV:",
        f"  mean {g['mean']:.4f}  std {g['std']:.4f}  CV {g['cv']:.3f}"
        f"  min {g['min']:.4f}  max {g['max']:.4f}",
        f"  dead-neuron fraction (g_i < 5% of median): {g['dead_frac']*100:.2f}%",
        "",
        "per-pair activation variance -- dead pairs never move:",
        f"  mean {p['mean']:.4e}  CV {p['cv']:.3f}  median {p['median']:.4e}",
        f"  dead-pair fraction (var < 1% of median): {p['dead_frac']*100:.2f}%",
    ]
    return "\n".join(lines)
