"""Appendix Figure S3: share of the SynLethDB synthetic lethal pairs captured among the edges of the
population level network, ranked from the highest to the lowest weight (x axis zoomed to the top 20%)."""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter, MultipleLocator

results_dir, out_dir = "results", "figures"
os.makedirs(out_dir, exist_ok=True)

curve = pd.read_csv(os.path.join(results_dir, "gid_enrichment_data_with_null_SynLethDB_population.csv"))
stats = pd.read_csv(os.path.join(results_dir, "gid_enrichment_stats_SynLethDB_population.csv"))
top5 = stats[stats.top_pct == 5].iloc[0]

RED, GREY, DARK = "#B22222", "#7F7F7F", "#4D4D4D"
plt.rcParams.update({"font.sans-serif": ["Inter", "Arial", "DejaVu Sans"], "font.size": 8, "axes.linewidth": 1.1,
                     "xtick.major.width": 1.1, "ytick.major.width": 1.1, "xtick.major.size": 3.5,
                     "ytick.major.size": 3.5, "pdf.fonttype": 42})

x = curve.Proportion_Edges_Ranked.values
y5 = top5.capture_pct / 100
note = f"Top {top5.top_pct:g}% of edges\n{top5.fold:.1f} fold enrichment\n{top5.p_label}\n{int(top5.gold_pairs):,} pairs"

fig, ax = plt.subplots(figsize=(5.6, 4.2))
ax.grid(True, color="#EBEBEB", lw=0.6, zorder=0)
ax.set_axisbelow(True)
ax.fill_between(x, curve.Null_CI_Lower, curve.Null_CI_Upper, step="post", color=GREY, alpha=0.30, lw=0, zorder=1)
ax.fill_between(x, curve.Model_CI_Lower, curve.Model_CI_Upper, step="post", color=RED, alpha=0.25, lw=0, zorder=2)
ax.plot(x, curve.Random_Expectation, color=DARK, ls=(0, (5, 3)), lw=1.4, zorder=3)
ax.step(x, curve.Model_Capture_Rate, where="post", color=RED, lw=2.2, zorder=4)

# the point at 5% of the edges that the note refers to
ax.plot([0.05], [y5], "o", ms=5.5, mfc=RED, mec="white", mew=0.8, zorder=5)
ax.plot([0.05, 0.05], [0, y5], color=RED, lw=0.8, ls=":", zorder=3)
ax.annotate(note, xy=(0.05, y5), xytext=(0.062, y5 + 0.17), ha="left", va="bottom", fontsize=8, color="#222222",
            linespacing=1.35, arrowprops=dict(arrowstyle="-", color="#555555", lw=0.7, shrinkA=0, shrinkB=3))

ax.set_xlim(0, 0.2)
ax.set_ylim(0, max(curve.Model_CI_Upper.max() * 1.04, 0.5))
ax.xaxis.set_major_locator(MultipleLocator(0.05))
ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
ax.set_xlabel("Proportion of network edges examined (highest to lowest weight)", fontsize=8.5, fontweight="bold", labelpad=5)
ax.set_ylabel("Proportion of SynLethDB synthetic lethal pairs captured", fontsize=8.5, fontweight="bold", labelpad=5)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
fig.tight_layout()

for ext in ("pdf", "png"):
    fig.savefig(os.path.join(out_dir, f"supp3_synlethdb_curve.{ext}"), dpi=600, facecolor="white")
