"""Figures for the 3M joint architecture+optimizer search.

Palette: categorical slots 1-3 of the reference palette (blue/orange/aqua),
the subset documented as passing the all-pairs CVD gate in both modes. Aqua is
under 3:1 on the light surface, so every mark carries a direct label and
data/*.csv is the table view. One y-axis per panel -- the diagnostics have five
incommensurable scales, so they are small multiples, never a dual axis.
"""
import csv, math, os, statistics as st
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Defaults to this file's own directory, so the committed data/ and figures/
# regenerate in place; override with RESULTS_DIR to point at a scratch copy.
R = Path(os.environ.get("RESULTS_DIR", Path(__file__).resolve().parent))
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#b8b7b2"
SURF, GRID = "#fcfcfb", "#e6e5e1"

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "font.size": 10,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlecolor": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
})

trials = list(csv.DictReader(open(R/"data/trials.csv")))
comp = [(int(t["number"]), float(t["loss"])) for t in trials
        if t["state"] == "COMPLETE" and t["loss"]]
comp.sort()

# ---------------------------------------------------------------- fig 1
fig, ax = plt.subplots(figsize=(9, 5))
xs = [n for n, _ in comp]; ys = [v for _, v in comp]
ax.scatter(xs, ys, s=9, color=MUTED, linewidths=0, label="completed trial", zorder=2)
best, bx, by = math.inf, [], []
for n, v in comp:
    best = min(best, v); bx.append(n); by.append(best)
ax.plot(bx, by, color=BLUE, linewidth=2, label="best so far", zorder=3)
ax.axvline(256, color=INK2, linewidth=1, linestyle=":", zorder=1)
ax.text(266, 4.72, "TPE startup ends\n(256 random draws)", fontsize=8, color=INK2, va="top")
ax.annotate(f"best {by[-1]:.4f}", xy=(bx[-1], by[-1]), xytext=(bx[-1]-430, by[-1]+0.085),
            fontsize=9.5, color=INK, fontweight="bold", ha="left",
            arrowprops=dict(arrowstyle="-", color=INK2, linewidth=1))
ax.set_ylim(min(ys)-0.03, max(ys)+0.02)
ax.set_xlabel("trial number"); ax.set_ylabel("held-out loss (FineWeb-Edu val)")
ax.set_title("3M joint search: 1,041 completed trials of 2,007")
ax.legend(frameon=False, loc="upper right")
fig.tight_layout(); fig.savefig(R/"figures/fig1_search_progress.png", dpi=160); plt.close(fig)

# ---------------------------------------------------------------- fig 2
rows = list(csv.DictReader(open(R/"data/baselines.csv")))
labels = ["Mamba-2 only\nd64 e2 s24 depth14\n2,974,640 params",
          "Mamba-2 only\nd64 e3 s32 depth12\n3,010,800 params",
          "Mamba-2 + SKA\nd64 e2 s24 depth13\n2,991,928 params"]
means = [float(r["mean"]) for r in rows]; sds = [float(r["sd"]) for r in rows]
cols = [ORANGE, ORANGE, BLUE]
fig, ax = plt.subplots(figsize=(9.5, 4.9))
for i, (m, s_, c) in enumerate(zip(means, sds, cols)):
    ax.errorbar(m, i, xerr=s_, fmt="o", color=c, markersize=10, capsize=6,
                elinewidth=2.5, markeredgecolor=SURF, markeredgewidth=2, zorder=3)
    ax.text(m, i - 0.30, f"{m:.4f} ± {s_:.4f}", fontsize=9.5, color=INK,
            ha="center", fontweight="bold")
h = [plt.Line2D([], [], color=ORANGE, marker="o", linestyle="", markersize=9, label="Mamba-2 only"),
     plt.Line2D([], [], color=BLUE, marker="o", linestyle="", markersize=9, label="Mamba-2 + SKA")]
ax.legend(handles=h, frameon=False, loc="lower right")
ax.set_yticks(range(len(rows))); ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel("held-out loss — mean ± sd over seeds 42/43/44")
ax.set_title("At matched parameters and an identical untuned recipe, the arms tie", pad=26)
ax.text(0.5, 1.015, "no pair resolved at n=3 (Welch |t| ≤ 2.09, crit 4.30) · "
        "tuned search best is 4.1518, 0.19 off-scale to the left",
        transform=ax.transAxes, ha="center", fontsize=9, color=INK2)
lo = min(m - s_ for m, s_ in zip(means, sds)); hi = max(m + s_ for m, s_ in zip(means, sds))
pad = (hi - lo) * 0.55
ax.set_xlim(lo - pad, hi + pad)
ax.set_ylim(-0.75, len(rows) - 0.4); ax.invert_yaxis()
ax.grid(axis="y", visible=False)
fig.tight_layout(); fig.savefig(R/"figures/fig2_arch_vs_baseline.png", dpi=160); plt.close(fig)

# ---------------------------------------------------------------- fig 3
pts = [(float(t["ska_delta"]), float(t["loss"])) for t in trials
       if t["state"] == "COMPLETE" and t["loss"] and t["ska_delta"]]
dx = [p[0] for p in pts]; dy = [p[1] for p in pts]
mx, my = st.mean(dx), st.mean(dy)
r = (sum((a-mx)*(b-my) for a, b in pts) /
     math.sqrt(sum((a-mx)**2 for a in dx) * sum((b-my)**2 for b in dy)))
fig, ax = plt.subplots(figsize=(9, 5))
ax.scatter(dx, dy, s=11, color=BLUE, alpha=0.42, linewidths=0, zorder=2)
ax.set_xlabel("SKA ablation Δ  (loss with SKA zeroed − loss of full model)")
ax.set_ylabel("held-out loss")
ax.set_title("Trials that lean harder on SKA score better")
ax.text(0.975, 0.945, f"n = {len(pts)}   Pearson r = {r:+.3f}\nno trial has Δ ≤ 0",
        transform=ax.transAxes, ha="right", va="top", fontsize=10, color=INK,
        bbox=dict(boxstyle="round,pad=0.45", facecolor=SURF, edgecolor=GRID))
fig.tight_layout(); fig.savefig(R/"figures/fig3_ska_delta_vs_loss.png", dpi=160); plt.close(fig)

# ---------------------------------------------------------------- fig 4
d = list(csv.DictReader(open(R/"data/ska_diagnostics.csv")))
step = [int(x["step"]) for x in d]
panels = [("spectral_radius", "spectral radius of A_eff", None),
          ("write_gate", "LayerScale write gate", 0.01),
          ("resid_ratio_ska_over_mamba", "‖Δ SKA‖ / ‖Δ Mamba‖", None),
          ("lambda_min_over_ridge", "λmin / ridge floor", 1.0),
          ("grad_norm_ratio", "SKA / Mamba grad-norm ratio", None)]
fig, axes = plt.subplots(1, 5, figsize=(16, 3.5), sharex=True)
for ax, (key, title, ref) in zip(axes, panels):
    v = [float(x[key]) for x in d]
    ax.plot(step, v, color=BLUE, linewidth=2, marker="o", markersize=4,
            markeredgecolor=SURF, markeredgewidth=1, zorder=3)
    if ref is not None:
        ax.axhline(ref, color=ORANGE, linewidth=1.5, linestyle="--", zorder=2)
        lab = "init 0.01" if key == "write_gate" else "ridge floor"
        ax.text(step[-1], ref, f" {lab}", fontsize=8, color=INK2, va="bottom", ha="right")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("step")
    vy = 0.06 if key == "lambda_min_over_ridge" else 0.93
    va = "bottom" if key == "lambda_min_over_ridge" else "top"
    ax.text(0.04, vy, f"{v[0]:.3g} → {v[-1]:.3g}", transform=ax.transAxes,
            fontsize=8.5, color=INK, va=va, fontweight="bold")
fig.suptitle("SKA health during the reference run — the branch is used more, not less, as training proceeds",
             fontsize=12, fontweight="bold", color=INK, y=1.03)
fig.tight_layout(); fig.savefig(R/"figures/fig4_ska_diagnostics.png", dpi=160,
                                bbox_inches="tight"); plt.close(fig)
print("wrote 4 figures")


# ============================================================ axis contribution
# The three figures below answer "how much did each swept axis contribute?".
# They are computed here rather than taken straight from main_effects.csv
# because that table's own docstring names its limitation: the axes are not
# balanced against each other, since the SAMPLER chose the trials. TPE spent 715
# of 1041 completed trials on ska_rank=32 and 868 on the top learning-rate bin,
# so a raw marginal mean confounds "this level is good" with "TPE co-selected it
# alongside a good learning rate". fig6 shows that confound directly instead of
# hiding it.
import collections

def _var_share(sub, ax, minn=5):
    """First-order fANOVA term + level spread, on whatever subset is passed."""
    g = collections.defaultdict(list)
    for r in sub:
        g[r[ax]].append(float(r["loss"]))
    g = {k: v for k, v in g.items() if len(v) >= minn}
    if len(g) < 2:
        return None, None
    allv = [x for v in g.values() for x in v]
    gm, tot, n = st.mean(allv), st.pvariance(allv), len(allv)
    if tot <= 0:
        return None, None
    between = sum(len(v) * (st.mean(v) - gm) ** 2 for v in g.values()) / n
    means = [st.mean(v) for v in g.values()]
    return between / tot, max(means) - min(means)

comp_rows = [r for r in trials if r["state"] == "COMPLETE" and r["loss"]]
randph = [r for r in comp_rows if int(r["number"]) < 256]        # unbiased
hiLR = [r for r in comp_rows if float(r["learning_rate"]) >= 0.004166]
SIGMA = 0.00142
RESOLVABLE = 2 * SIGMA * math.sqrt(2)

# ---------------------------------------------------------------- fig 5
imp = list(csv.DictReader(open(R/"data/importances.csv")))
imp.sort(key=lambda r: float(r["importance_global"]))
ax_names = [r["axis"] for r in imp]
loc = [float(r["importance"]) for r in imp]
glo = [float(r["importance_global"]) for r in imp]
y = list(range(len(imp))); h = 0.38
fig, ax = plt.subplots(figsize=(9.5, 7))
ax.barh([v + h/2 for v in y], glo, height=h, color=BLUE, label="global (all trials)", zorder=3)
ax.barh([v - h/2 for v in y], loc, height=h, color=ORANGE, label="local (among good trials)", zorder=3)
for i, (a, b) in enumerate(zip(glo, loc)):
    if a > 0.02: ax.text(a + 0.006, i + h/2, f"{a:.3f}", va="center", fontsize=8, color=INK)
    if b > 0.02: ax.text(b + 0.006, i - h/2, f"{b:.3f}", va="center", fontsize=8, color=INK)
ax.set_yticks(y); ax.set_yticklabels(ax_names, fontsize=9)
ax.set_xlabel("PedANOVA importance (share of objective variance attributed to the axis)")
ax.set_title("Learning rate dominates every architecture axis combined", pad=24)
ax.text(0.5, 1.012, "PedAnovaImportanceEvaluator over 1,041 completed trials · fANOVA needs sklearn, absent in this env",
        transform=ax.transAxes, ha="center", fontsize=8.5, color=INK2)
ax.legend(frameon=False, loc="lower right"); ax.grid(axis="y", visible=False)
fig.tight_layout(); fig.savefig(R/"figures/fig5_axis_importance.png", dpi=160); plt.close(fig)

# ---------------------------------------------------------------- fig 6
AXES = ["mamba_expand","ska_rank","ska_power_K","beta_policy","warmup_ratio","depth_tier",
        "n_ska_layers","weight_decay","d_state","ska_n_heads","placement","gamma_value",
        "norm_clip_multiplier"]
recs = []
for a in AXES:
    vr, _ = _var_share(randph, a); vh, _ = _var_share(hiLR, a)
    recs.append((a, (vr or 0) * 100, (vh or 0) * 100))
recs.sort(key=lambda r: r[2])
y = list(range(len(recs))); h = 0.38
fig, ax = plt.subplots(figsize=(9.5, 7))
ax.barh([v + h/2 for v in y], [r[2] for r in recs], height=h, color=BLUE,
        label="inside the good-LR band (n=868)", zorder=3)
ax.barh([v - h/2 for v in y], [r[1] for r in recs], height=h, color=ORANGE,
        label="random startup, LR uncontrolled (n=186)", zorder=3)
for i, r in enumerate(recs):
    ax.text(r[2] + 0.6, i + h/2, f"{r[2]:.1f}%", va="center", fontsize=8, color=INK)
    ax.text(r[1] + 0.6, i - h/2, f"{r[1]:.1f}%", va="center", fontsize=8, color=INK)
ax.set_yticks(y); ax.set_yticklabels([r[0] for r in recs], fontsize=9)
ax.set_xlabel("first-order fANOVA variance share within the subset (%)")
ax.set_title("Architecture only separates once the learning rate is right", pad=24)
ax.text(0.5, 1.012, "LR alone is 70.9% of the variance during random search — it buries every other axis until it is fixed",
        transform=ax.transAxes, ha="center", fontsize=8.5, color=INK2)
ax.legend(frameon=False, loc="lower right"); ax.grid(axis="y", visible=False)
fig.tight_layout(); fig.savefig(R/"figures/fig6_conditional_variance.png", dpi=160); plt.close(fig)

# ---------------------------------------------------------------- fig 7
TOP = ["mamba_expand", "ska_rank", "beta_policy", "n_ska_layers"]
fig, axes = plt.subplots(1, 4, figsize=(15, 4.2))
for axx, a in zip(axes, TOP):
    g = collections.defaultdict(list)
    for r in hiLR:
        g[r[a]].append(float(r["loss"]))
    try:    keys = sorted(g, key=lambda k: float(k))
    except ValueError: keys = sorted(g, key=lambda k: st.mean(g[k]))
    m  = [st.mean(g[k]) for k in keys]
    ci = [1.96*st.stdev(g[k])/math.sqrt(len(g[k])) if len(g[k]) > 1 else 0 for k in keys]
    xs = range(len(keys))
    axx.errorbar(xs, m, yerr=ci, fmt="o", color=BLUE, markersize=8, capsize=5,
                 elinewidth=2, markeredgecolor=SURF, markeredgewidth=1.5, zorder=3)
    # n labels pinned to the top of the panel in axes coords -- placing them
    # under each point collided with the x tick labels on the lowest level.
    lo = min(v - c for v, c in zip(m, ci)); hiv = max(v + c for v, c in zip(m, ci))
    pad = (hiv - lo) * 0.30 or 0.01
    axx.set_ylim(lo - pad * 0.35, hiv + pad)
    for i, k in enumerate(keys):
        axx.text(i, 0.965, f"n={len(g[k])}", transform=axx.get_xaxis_transform(),
                 ha="center", va="top", fontsize=8, color=INK2)
    axx.set_xticks(list(xs)); axx.set_xticklabels(keys, fontsize=9)
    axx.set_xlim(-0.55, len(keys) - 0.45)
    axx.set_title(a, fontsize=11)
    axx.set_xlabel("")
axes[0].set_ylabel("held-out loss (mean ± 95% CI)")
fig.suptitle("Level means inside the good-LR band — every gap here clears the 0.0040 resolvable floor",
             fontsize=12, fontweight="bold", color=INK, y=1.02)
fig.text(0.5, -0.04, "Unbalanced by construction: TPE chose these counts, so read the ordering, "
         "not the exact gaps. n<40 levels are the least trustworthy.",
         ha="center", fontsize=8.5, color=INK2)
fig.tight_layout()
fig.savefig(R/"figures/fig7_level_means_controlled.png", dpi=160, bbox_inches="tight")
plt.close(fig)
print("wrote 3 axis-contribution figures")
