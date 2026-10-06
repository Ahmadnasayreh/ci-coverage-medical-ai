"""
make_fig_coverage.py — Fig. 1 of the paper.

Reads results/table9_coverage_complex.csv (notebook Section 16) and draws
empirical coverage against test-set size for F1, MCC and AUROC in the two
binormal regimes, with:
  * labelled x and y axes,
  * one legend naming all four interval constructions,
  * 95 % Monte-Carlo error bars, +/- 1.96 * sqrt(C(1-C)/M),
  * the 95 % Monte-Carlo acceptance band around the nominal 0.95.
Writes figures/fig_coverage.pdf (Type-42 fonts, IEEE PDF eXpress safe) and .png.

Run from the repository root:   python scripts/make_fig_coverage.py
"""
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullLocator, FixedFormatter

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Liberation Serif", "TeX Gyre Termes", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7.5,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.8,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300,
})

ROOT = Path(__file__).resolve().parents[1]          # repository root
df = pd.read_csv(ROOT / "results" / "table9_coverage_complex.csv")
M = 1000
band = 1.96 * np.sqrt(0.95 * 0.05 / M)          # +/- 0.0135

# colour-blind-safe categorical palette; line style and marker also encode the method
STYLE = {
    "bca":        dict(label="BCa bootstrap",        color="#2a78d6", marker="o", ls="-",  dx=0.97),
    "percentile": dict(label="Percentile bootstrap", color="#eb6834", marker="s", ls="--", dx=1.00),
    "normal":     dict(label="Normal approx. (boot. SE)", color="#1baf7a", marker="^", ls="-.", dx=1.03),
    "delong":     dict(label="DeLong (AUROC only)",  color="#eda100", marker="D", ls=":",  dx=1.06),
}
SETTINGS = [("moderate (AUROC~0.85)", "Moderate"),
            ("high (AUROC~0.97)",     "High")]
METRICS = [("f1", "F1"), ("mcc", "MCC"), ("auroc", "AUROC")]
NS = [50, 100, 200, 500]

fig, axes = plt.subplots(2, 3, figsize=(3.5, 2.5), sharex=True, sharey=True)
for i, (skey, slabel) in enumerate(SETTINGS):
    for j, (mkey, mlabel) in enumerate(METRICS):
        ax = axes[i, j]
        ax.axhspan(0.95 - band, 0.95 + band, color="#d9d9d6", alpha=0.55, lw=0, zorder=0)
        ax.axhline(0.95, color="#52514e", lw=0.7, ls=(0, (4, 2)), zorder=1)
        sub = df[(df.setting == skey) & (df.metric == mkey)]
        for meth, st in STYLE.items():
            d = sub[sub.method == meth].sort_values("n")
            if d.empty:
                continue
            x = d.n.values * st["dx"]
            err = 1.96 * d.mc_se.values
            ax.errorbar(x, d.coverage.values, yerr=err, color=st["color"], marker=st["marker"],
                        ms=3.0, lw=1.0, ls=st["ls"], elinewidth=0.7, capsize=1.5,
                        mec="white", mew=0.4, zorder=3 if meth == "bca" else 2)
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(FixedLocator(NS))
        ax.xaxis.set_major_formatter(FixedFormatter([str(n) for n in NS]))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xlim(40, 640)
        ax.set_ylim(0.765, 0.99)
        ax.set_yticks([0.80, 0.85, 0.90, 0.95])
        ax.grid(axis="y", color="#e6e6e3", lw=0.4, zorder=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        if i == 0:
            ax.set_title(mlabel, pad=3)
        if j == 2:
            ax.text(1.05, 0.5, slabel, transform=ax.transAxes, rotation=270,
                    va="center", ha="left", fontsize=7)

fig.supxlabel("Test-set size $n$ (log scale)", fontsize=7, y=0.005)
fig.supylabel("Empirical coverage of nominal 95% CI", fontsize=7, x=0.02)
handles = [plt.Line2D([], [], color=s["color"], marker=s["marker"], ls=s["ls"], lw=1.0, ms=3.2,
                      mec="white", mew=0.4) for s in STYLE.values()]
handles.append(plt.Rectangle((0, 0), 1, 1, color="#d9d9d6", alpha=0.8))
labels = [s["label"] for s in STYLE.values()] + ["Nominal 0.95 ± MC band"]
fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
           bbox_to_anchor=(0.52, 1.02), handlelength=2.4, columnspacing=1.0, labelspacing=0.3)
fig.subplots_adjust(left=0.13, right=0.93, bottom=0.145, top=0.77, wspace=0.12, hspace=0.16)
fig.savefig(ROOT / "figures" / "fig_coverage.pdf")
fig.savefig(ROOT / "figures" / "fig_coverage.png")
print("band half-width:", round(band, 4))
print("min/max plotted coverage:", df[df.metric.isin(['f1','mcc','auroc'])].coverage.min(),
      df[df.metric.isin(['f1','mcc','auroc'])].coverage.max())
