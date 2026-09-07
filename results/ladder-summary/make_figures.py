"""Cross-rung summary of the 3M -> 10M -> 50M -> 180M architecture+optimizer ladder.

Palette: categorical slots 1-4 of the reference palette in fixed slot order
(blue/orange/aqua/yellow), as used by results/3m-joint-v1 and
results/50m-joint-v1 so all three directories read as one system. One y-axis per
panel; never a dual axis.

The load-bearing figure is fig2. The question this directory answers is whether
the four rungs "tell the same story", and fig2 is where the answer lives: they
agree that the LEARNING RATE dominates at every scale, and they agree that most
architecture axes do not matter -- but they do not agree on any specific
architecture value, and `ska_rank`'s importance decays monotonically with scale.
"""
import csv, math, os
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path(os.environ.get("RESULTS_DIR", Path(__file__).resolve().parent))
D, F = R / "data", R / "figures"
F.mkdir(parents=True, exist_ok=True)

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
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

rungs = list(csv.DictReader(open(D / "rungs.csv")))
axes_rows = list(csv.DictReader(open(D / "axis_importance.csv")))
NAMES = [r["rung"] for r in rungs]
Ns = [float(r["n_params"]) for r in rungs]

# ------------------------------------------------------------------- fig 1
# The ladder. Both series on one axis; the tuned line is what the ladder buys.
fig, ax = plt.subplots(figsize=(8.8, 5.4))
ref = [float(r["reference_loss"]) for r in rungs]
best = [float(r["best_loss"]) for r in rungs]
sig = [float(r["sigma"]) for r in rungs]
ax.errorbar(Ns, ref, yerr=[2 * s for s in sig], color=AQUA, lw=2, marker="o",
            markersize=8, markeredgecolor=SURF, markeredgewidth=1.8, capsize=4,
            zorder=3, label="untuned reference ($\\pm 2\\sigma$)")
ax.plot(Ns, best, color=BLUE, lw=2, marker="s", markersize=8,
        markeredgecolor=SURF, markeredgewidth=1.8, zorder=4,
        label="best config found")
for x, y, n in zip(Ns, best, NAMES):
    ax.annotate(f"{n}\n{y:.4f}", xy=(x, y), xytext=(0, -30),
                textcoords="offset points", fontsize=8.5, color=INK2, ha="center")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("total parameters N (log)")
ax.set_ylabel("validation loss, NTP (log)")
ax.set_title("The ladder: four rungs on the ray D $\\approx$ 65N")
ax.legend(loc="upper right", frameon=False, fontsize=9)
ax.margins(y=0.22)
fig.tight_layout(); fig.savefig(F / "fig1_ladder.png", dpi=170); plt.close(fig)

# ------------------------------------------------------------------- fig 2
# THE figure. Variance share per axis per rung. Only axes varied at >=3 rungs,
# so the lines are comparable; the rest are in data/axis_importance.csv.
by_axis = {}
for r in axes_rows:
    by_axis.setdefault(r["axis"], {})[r["rung"]] = r
keep = [a for a, d in by_axis.items()
        if sum(1 for k in NAMES if d.get(k, {}).get("varied") == "True") >= 3]
keep.sort(key=lambda a: -max(float(by_axis[a][k]["variance_share"])
                             for k in NAMES if k in by_axis[a]
                             and by_axis[a][k].get("varied") == "True"))
fig, ax = plt.subplots(figsize=(9.6, 6))
labels = []
slots = [BLUE, ORANGE, AQUA, YELLOW, "#e87ba4", "#4a3aa7", "#008300", "#e34948"]
x = range(len(NAMES))
for i, a in enumerate(keep[:8]):
    ys, xs, marks = [], [], []
    for j, k in enumerate(NAMES):
        r = by_axis[a].get(k)
        if not r or r.get("varied") != "True":
            continue
        xs.append(j); ys.append(float(r["variance_share"]))
        marks.append(r["resolved"] == "True")
    ax.plot(xs, ys, color=slots[i % len(slots)], lw=2, marker="o", markersize=7,
            markeredgecolor=SURF, markeredgewidth=1.6, zorder=3, label=a)
    # A filled marker means the axis RESOLVED against that rung's noise floor;
    # hollow means it did not. Identity is never colour-alone: every line is
    # also direct-labelled at its right end.
    for xx, yy, m in zip(xs, ys, marks):
        if not m:
            ax.scatter([xx], [yy], s=30, facecolor=SURF,
                       edgecolor=slots[i % len(slots)], linewidths=1.6, zorder=4)
    labels.append((xs[-1], ys[-1], a, slots[i % len(slots)]))
import collections as _c
_by_x = _c.defaultdict(list)
for lx, ly, la, lc in labels:
    _by_x[lx].append((ly, la, lc))
_span = max(ys for _, ys, _, _ in labels) if labels else 1.0
for lx, group in _by_x.items():
    group.sort()                       # ascending y
    gap = 0.045 * max(_span, 1e-9)     # minimum vertical separation
    placed = []
    for ly, la, lc in group:
        y = ly if not placed else max(ly, placed[-1] + gap)
        placed.append(y)
        # A leader line when the label had to move, so it still reads as
        # belonging to its endpoint rather than floating.
        if abs(y - ly) > gap * 0.35:
            ax.plot([lx, lx + 0.10], [ly, y], color=lc, lw=0.9, alpha=0.75,
                    zorder=2, clip_on=False)
        ax.annotate(la, xy=(lx + 0.12, y), xytext=(0, 0),
                    textcoords="offset points", fontsize=8.5, color=lc,
                    va="center", fontweight="bold", annotation_clip=False)

ax.set_xticks(list(x)); ax.set_xticklabels(
    [f"{n}\nN={float(r['n_params'])/1e6:.0f}M" for n, r in zip(NAMES, rungs)])
ax.set_ylabel("share of loss variance attributed to the axis")
ax.set_title("Do the rungs tell the same story? The optimizer does; architecture fades")
ax.set_xlim(-0.25, len(NAMES) - 1 + 1.05)
fig.text(0.012, 0.012,
         "Filled marker = resolved against that rung's noise floor; hollow = not resolved. "
         "The 3M study shipped without a reference group, so nothing there is\nmarked resolved "
         "(its sigma of 0.00142 was measured afterwards). Only axes varied at three or more "
         "rungs are drawn; see data/axis_importance.csv for all 17.",
         fontsize=7.6, color=INK2, va="bottom")
fig.tight_layout(rect=(0, 0.075, 1, 1))
fig.savefig(F / "fig2_axis_importance_across_scale.png", dpi=170); plt.close(fig)

# ------------------------------------------------------------------- fig 3
# The one thing that does NOT transfer.
fig, ax = plt.subplots(figsize=(8.6, 5.2))
lrs = [float(r["best_lr"]) for r in rungs]
cens = [r["lr_censored"] == "True" for r in rungs]
ax.plot(Ns, lrs, color=ORANGE, lw=2, marker="o", markersize=9,
        markeredgecolor=SURF, markeredgewidth=2, zorder=3)
# The censoring marker goes INSIDE the label text. A separate glyph offset from
# the point rendered as a stray stroke beside the marker and, at the leftmost
# rung, collided with the y-axis.
for xx, yy, c, n in zip(Ns, lrs, cens, NAMES):
    txt = f"{n}\n" + ("$\\geq$ " if c else "") + f"{yy:.5f}"
    if c:
        txt += "\n(ceiling)"
    ax.annotate(txt, xy=(xx, yy), xytext=(0, 18), textcoords="offset points",
                fontsize=8.5, color=INK2, ha="center", fontweight="bold" if not c else "normal")
ax.set_xlim(min(Ns) * 0.45, max(Ns) * 2.2)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("total parameters N (log)"); ax.set_ylabel("best learning rate (log)")
ax.set_title("Optimal LR does not follow a rule: it rose, then fell 2.9$\\times$")
ax.margins(y=0.30)
fig.text(0.012, 0.012,
         "3M / 10M / 50M all peaked at ~98% of their searched ceiling, so those three are LOWER BOUNDS, not locations. "
         "180M is the first rung whose\noptimum is interior and bracketed on both sides -- and it is 2.9x BELOW the 10M "
         "value. Width scaling predicted 0.0027 at 50M against >=0.0088.",
         fontsize=7.6, color=INK2, va="bottom")
fig.tight_layout(rect=(0, 0.085, 1, 1))
fig.savefig(F / "fig3_lr_does_not_transfer.png", dpi=170); plt.close(fig)

# ------------------------------------------------------------------- fig 4
# SKA's measured contribution, and the tuning gain, as small multiples --
# two incommensurable quantities, so never one dual axis.
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6))
sd = [float(r["ska_delta_median"]) for r in rungs]
nd = [int(r["ska_delta_n"]) for r in rungs]
a1.bar(range(len(NAMES)), sd, color=BLUE, width=0.62, zorder=3)
for i, (v, n) in enumerate(zip(sd, nd)):
    a1.text(i, v + 0.004, f"{v:.4f}\nn={n}", ha="center", fontsize=8.5, color=INK2)
a1.axhline(0, color="#d03b3b", lw=1.6, ls="--", zorder=2)
a1.set_xticks(range(len(NAMES))); a1.set_xticklabels(NAMES)
a1.set_ylabel("median ska_delta")
a1.set_title("SKA is load-bearing at every scale\n(0 of 1,226 trials $\\leq$ 0)", fontsize=11)
a1.margins(y=0.25)

gain = [float(r["reference_loss"]) - float(r["best_loss"]) for r in rungs]
a2.bar(range(len(NAMES)), gain, color=ORANGE, width=0.62, zorder=3)
for i, (g, s) in enumerate(zip(gain, sig)):
    a2.text(i, g + 0.004, f"{g:.4f}\n{g/s:.0f}$\\sigma$", ha="center",
            fontsize=8.5, color=INK2)
a2.set_xticks(range(len(NAMES))); a2.set_xticklabels(NAMES)
a2.set_ylabel("loss improvement over the untuned reference")
a2.set_title("What tuning bought, in units of that\nrung's own seed noise", fontsize=11)
a2.margins(y=0.25)
fig.tight_layout(); fig.savefig(F / "fig4_ska_delta_and_tuning_gain.png", dpi=170)
plt.close(fig)

print("wrote:")
for p in sorted(F.glob("*.png")):
    print("  %-44s %6.1f KB" % (p.name, p.stat().st_size / 1024))
