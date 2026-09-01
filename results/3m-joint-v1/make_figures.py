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
