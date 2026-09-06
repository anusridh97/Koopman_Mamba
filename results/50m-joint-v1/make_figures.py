"""Figures for the 50M joint architecture+optimizer search.

Palette: categorical slots 1-3 of the reference palette (blue/orange/aqua) in
fixed slot order, and the documented blue<->red diverging pair with a gray
midpoint for fig3, whose data is SIGNED (better/worse than the reference) and so
carries polarity rather than identity. Slots are used as published rather than
re-derived, because a categorical palette must be validated by the checker and
not by eye.

One y-axis per panel throughout -- never a dual axis. fig2 is the "how much did
each swept axis contribute" figure, and it deliberately plots THREE estimators
side by side because they disagree; see its note.
"""
import csv, json, math, os, statistics
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path(os.environ.get("RESULTS_DIR", Path(__file__).resolve().parent))
D, F = R / "data", R / "figures"
F.mkdir(exist_ok=True)

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GOOD, BAD, MID = "#2a78d6", "#d03b3b", "#f0efec"      # diverging: blue<->red, gray
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


def rows(name):
    with open(D / name) as f:
        return list(csv.DictReader(f))


def num(x, default=None):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


trials = rows("trials.csv")
nf = rows("noise_floor.csv")[0]
SIGMA = float(nf["sigma"])
REF_MEAN = float(nf["mean"])
RESOLVABLE = float(nf["min_resolvable_effect"])      # trial-vs-trial, 2*sigma*sqrt2

comp = sorted((int(t["number"]), num(t["loss"]))
              for t in trials if t["state"] == "COMPLETE" and num(t["loss"]))

# ------------------------------------------------------------------- fig 1
# Search progress. The reference band is the scale for everything else, so it is
# drawn first and labelled directly.
fig, ax = plt.subplots(figsize=(9, 5))
ax.axhspan(REF_MEAN - SIGMA, REF_MEAN + SIGMA, color=AQUA, alpha=0.16, zorder=1)
ax.axhline(REF_MEAN, color=AQUA, lw=2, zorder=2,
           label=f"untuned reference  {REF_MEAN:.4f} $\\pm\\sigma$ ({SIGMA:.5f}, n=5)")
ax.scatter([n for n, _ in comp], [v for _, v in comp], s=26, color=MUTED,
           linewidths=0, zorder=3, label=f"completed trial (n={len(comp)})")
best, bx, by = math.inf, [], []
for n, v in comp:
    best = min(best, v); bx.append(n); by.append(best)
ax.plot(bx, by, color=BLUE, lw=2, zorder=4, label="best so far")
bn, bv = min(comp, key=lambda p: p[1])
ax.scatter([bn], [bv], s=90, facecolor=BLUE, edgecolor=SURF, linewidths=2, zorder=5)
# Annotate INSIDE the axes and above the point, so it cannot collide with the
# x-axis the way the 3M fig1 annotation did.
ax.annotate(f"best  trial {bn}\n{bv:.5f}", xy=(bn, bv), xytext=(8, 26),
            textcoords="offset points", color=INK, fontsize=9, fontweight="bold",
            ha="left", va="bottom")
ax.set_xlabel("trial number"); ax.set_ylabel("validation loss (NTP)")
ax.set_title("50M joint search: 30 completed of 63 trials, 0 failed")
ax.legend(loc="upper right", frameon=False, fontsize=9)
fig.tight_layout(); fig.savefig(F / "fig1_search_progress.png", dpi=170); plt.close(fig)

# ------------------------------------------------------------------- fig 2
# Per-axis contribution -- three estimators, because they disagree and the
# disagreement is the point. Sorted by the partial-dependence share, which is the
# one computed against the study's own level means rather than against a
# reference distribution.
me = {r["axis"]: r for r in rows("main_effects.csv")}
imp = {r["axis"]: r for r in rows("importances.csv")}
axes_ = [a for a in me if me[a].get("varied") == "True"]
axes_.sort(key=lambda a: -num(me[a]["variance_share"], 0))
pd_ = [num(me[a]["variance_share"], 0) for a in axes_]
pl_ = [num(imp.get(a, {}).get("importance"), 0) or 0 for a in axes_]
pg_ = [num(imp.get(a, {}).get("importance_global"), 0) or 0 for a in axes_]

fig, ax = plt.subplots(figsize=(10, 6.4))
y = range(len(axes_)); h = 0.26
# 2px surface gap between adjacent bars is the spacer rule; at this figure size
# that is the 0.02 inset between the three offsets below.
ax.barh([i + h for i in y], pd_, height=h - 0.02, color=BLUE, zorder=3,
        label="partial dependence (level means / total variance)")
ax.barh([i for i in y], pl_, height=h - 0.02, color=ORANGE, zorder=3,
        label="PED-ANOVA, local reference")
ax.barh([i - h for i in y], pg_, height=h - 0.02, color=AQUA, zorder=3,
        label="PED-ANOVA, declared prior")
for i, a in enumerate(axes_):
    if me[a].get("resolved") == "True":
        ax.text(pd_[i] + 0.012, i + h, "resolved", va="center", ha="left",
                fontsize=8, color=INK2)
ax.set_yticks(list(y)); ax.set_yticklabels(axes_, fontsize=9)
ax.invert_yaxis()
ax.set_xlabel("share of variance attributed to the axis")
ax.set_title("Which swept axis actually moved the loss at 50M")
ax.legend(loc="lower right", frameon=False, fontsize=9)
ax.set_xlim(0, max(pd_ + pl_ + pg_) * 1.22)
fig.text(0.012, 0.012,
         "The three bars are not interchangeable. PED-ANOVA/local compares the "
         "top-quantile trials against the study's own remaining trials, so an "
         "adaptive sampler's\npath inflates it -- that is why ska_layerscale_init "
         "scores 0.34/0.44 here while its partial-dependence spread is 0.0005 and "
         "all three of its\ncontrolled anchors are flat. For claims about the "
         "model, read fig3.",
         fontsize=7.6, color=INK2, va="bottom")
fig.tight_layout(rect=(0, 0.085, 1, 1))
fig.savefig(F / "fig2_axis_contribution.png", dpi=170); plt.close(fig)

# ------------------------------------------------------------------- fig 3
# Anchor contrasts: the controlled read. One factor moved from one reference
# point, against the measured noise floor. Signed -> diverging palette.
ac = [r for r in rows("anchor_contrasts.csv") if num(r.get("delta")) is not None]
ac.sort(key=lambda r: num(r["delta"]))
names = [r["anchor"] for r in ac]
deltas = [num(r["delta"]) for r in ac]
res = [r.get("resolved") == "True" for r in ac]
thr = num(ac[0].get("threshold")) or RESOLVABLE

fig, ax = plt.subplots(figsize=(9.6, 7))
ax.axvspan(-thr, thr, color=MID, zorder=1)
ax.axvline(0, color=INK2, lw=1.2, zorder=2)
for i, (d, r) in enumerate(zip(deltas, res)):
    c = (GOOD if d < 0 else BAD) if r else MUTED
    ax.plot([0, d], [i, i], color=c, lw=2, zorder=3, solid_capstyle="round")
    ax.scatter([d], [i], s=64, facecolor=c, edgecolor=SURF, linewidths=2, zorder=4)
ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=9)
ax.invert_yaxis()
ax.set_xlabel("$\\Delta$ loss vs the untuned reference  (negative = better)")
ax.set_title("50M one-factor anchors against the measured noise floor")
# Legend by proxy handles, so identity is never colour-alone.
from matplotlib.lines import Line2D
ax.legend(handles=[
    Line2D([], [], color=GOOD, lw=2, marker="o", label="resolved better"),
    Line2D([], [], color=BAD, lw=2, marker="o", label="resolved worse"),
    Line2D([], [], color=MUTED, lw=2, marker="o", label="inside the floor (flat)"),
    Line2D([], [], color=MID, lw=8, label=f"not resolvable ($\\pm${thr:.5f})"),
], loc="upper right", frameon=False, fontsize=9)
# Upper right, NOT lower right: the two lr- anchors at the bottom are the largest
# positive deltas in the study, so their lines and markers ran straight through a
# lower-right legend. The top rows are all negative, leaving that corner empty.
# The right margin keeps the +0.0598 marker off the spine.
ax.set_xlim(min(deltas) - 0.006, max(deltas) + 0.010)
fig.tight_layout(); fig.savefig(F / "fig3_anchor_contrasts.png", dpi=170); plt.close(fig)

# ------------------------------------------------------------------- fig 4
# ska_delta: the load-bearing result. Single series -> no legend box, the title
# names it.
pts = [(num(t["ska_delta"]), num(t["loss"])) for t in trials
       if t["state"] == "COMPLETE" and num(t["ska_delta"]) is not None
       and num(t["loss"]) is not None]
xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
sx = math.sqrt(sum((v - mx) ** 2 for v in xs)); sy = math.sqrt(sum((v - my) ** 2 for v in ys))
r_ = sum((a - mx) * (b - my) for a, b in pts) / (sx * sy) if sx and sy else float("nan")

fig, ax = plt.subplots(figsize=(8.6, 5.2))
ax.axvline(0, color=BAD, lw=1.6, ls="--", zorder=2)
ax.text(0.004, max(ys), "  SKA removal would HELP\n  (0 of %d trials here)" % len(pts),
        fontsize=8.5, color=BAD, va="top", ha="left")
ax.scatter(xs, ys, s=34, color=BLUE, alpha=0.85, linewidths=0, zorder=3)
ax.set_xlabel("ska_delta  =  loss(SKA ablated) $-$ loss(as trained)")
ax.set_ylabel("validation loss (NTP)")
# statistics.median, not sorted(xs)[n//2] -- n is even here, so the naive
# index returns the upper middle value (0.1477) and disagrees with the README.
ax.set_title("SKA is load-bearing at 50M: min %.4f, median %.4f, r(loss, $\\Delta$)=%.3f"
             % (min(xs), statistics.median(xs), r_))
ax.set_xlim(left=min(-0.012, min(xs) - 0.01))
fig.tight_layout(); fig.savefig(F / "fig4_ska_delta.png", dpi=170); plt.close(fig)

# ------------------------------------------------------------------- fig 5
# The LR ladder, which is the rung's headline: still right-censored.
lr_of = {t["anchor_name"]: num(t["learning_rate"])
         for t in trials if t.get("anchor_name")}
lad = sorted((lr_of[r["anchor"]], num(r["objective"]), r["anchor"])
             for r in ac if r["anchor"].startswith("lr-") and lr_of.get(r["anchor"]))
ref_lr = next((lr_of[a] for a in lr_of if a == "reference-k1"), None)
if ref_lr:
    lad.append((ref_lr, REF_MEAN, "reference"))
    lad.sort()
fig, ax = plt.subplots(figsize=(8.6, 5.2))
ax.axhspan(REF_MEAN - SIGMA, REF_MEAN + SIGMA, color=AQUA, alpha=0.16, zorder=1)
ax.plot([p[0] for p in lad], [p[1] for p in lad], color=BLUE, lw=2, marker="o",
        markersize=8, markerfacecolor=BLUE, markeredgecolor=SURF,
        markeredgewidth=2, zorder=3)
for i, (x, yv, nm) in enumerate(lad):
    # The last point sits ON the ceiling line, so its label goes left of the
    # marker rather than under it, where the dashed rule struck through the text.
    last = (i == len(lad) - 1)
    ax.annotate(nm, xy=(x, yv), xytext=(-10 if last else 0, -16),
                textcoords="offset points", fontsize=8, color=INK2,
                ha="right" if last else "center", va="top")
ax.set_xscale("log")
# The searched range's ceiling, which is the whole point of the figure: the
# ladder is monotone all the way into it, so the optimum is a LOWER BOUND.
CEIL = 0.009
ax.axvline(CEIL, color=ORANGE, lw=2, ls="--", zorder=2)
ax.annotate("searched ceiling\n0.009", xy=(CEIL, min(p[1] for p in lad)),
            xytext=(-8, 30), textcoords="offset points", fontsize=8.5,
            color=ORANGE, ha="right", va="bottom", fontweight="bold")
# Explicit ticks at the anchors themselves -- a default log axis labelled only
# 10^-3 here, which hides the 10x span the ladder actually covers.
ticks = [p[0] for p in lad]
ax.set_xticks(ticks, minor=False)
ax.set_xticklabels([f"{t*1000:.2f}" for t in ticks], fontsize=8.5)
ax.set_xticks([], minor=True)
ax.set_xlabel("learning rate  ($\\times10^{-3}$, log scale)")
ax.set_ylabel("validation loss (NTP)")
ax.set_title("The 50M LR optimum sits at the ceiling -- third rung in a row")
ax.margins(y=0.20); ax.set_xlim(min(ticks) * 0.82, CEIL * 1.16)
fig.tight_layout(); fig.savefig(F / "fig5_lr_ladder.png", dpi=170); plt.close(fig)

# ------------------------------------------------------------------- fig 6
# Loss vs parameter count, with the Pareto front. The band is 45.5-54.5M, so the
# x-range is narrow by construction -- that is the point of a fixed-size search.
ovp = [(num(r["param_count"]), num(r["objective"]))
       for r in rows("objective_vs_params.csv")
       if num(r.get("param_count")) and num(r.get("objective"))]
par = [(num(r["param_count"]), num(r["objective"]))
       for r in rows("pareto.csv")
       if num(r.get("param_count")) and num(r.get("objective"))]
fig, ax = plt.subplots(figsize=(8.6, 5.2))
ax.scatter([p[0] / 1e6 for p in ovp], [p[1] for p in ovp], s=30, color=MUTED,
           linewidths=0, zorder=2, label=f"completed trial (n={len(ovp)})")
par.sort()
ax.plot([p[0] / 1e6 for p in par], [p[1] for p in par], color=ORANGE, lw=2,
        marker="o", markersize=7, markeredgecolor=SURF, markeredgewidth=1.6,
        zorder=3, label="Pareto front (loss vs params)")
ax.set_xlabel("total parameters (millions, band 45.5-54.5M)")
ax.set_ylabel("validation loss (NTP)")
ax.set_title("Loss vs capacity inside the fixed parameter band")
ax.legend(loc="upper right", frameon=False, fontsize=9)
fig.tight_layout(); fig.savefig(F / "fig6_loss_vs_params.png", dpi=170); plt.close(fig)

print("wrote:")
for p in sorted(F.glob("*.png")):
    print("  %-34s %6.1f KB" % (p.name, p.stat().st_size / 1024))
