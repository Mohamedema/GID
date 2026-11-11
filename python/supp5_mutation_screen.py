"""Appendix Figure S5: altered associations of the genes of the mutation screen, with and without the tissue covariate.
Input in results/mutations/: all_genes_summary.csv, written by python -m gid.mutations.
Output in figures/: supp5_mutation_screen.pdf and supp5_mutation_screen.png (600 dpi)"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

results_dir, out_dir = "results", "figures"
os.makedirs(out_dir, exist_ok=True)

PURPLE, DOT, BAR, ORANGE = "#5E0847", "#7A7A7A", "#DCDCDC", "#D55E00"
plt.rcParams.update({"font.sans-serif": ["Inter", "Arial", "DejaVu Sans"], "font.size": 7.5, "axes.linewidth": 0.7,
                     "xtick.major.width": 0.7, "ytick.major.width": 0.7, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42})

screen = pd.read_csv(os.path.join(results_dir, "mutations", "all_genes_summary.csv"))
screen = screen[screen.status == "ok"].sort_values(["n_mutated", "gene"], ascending=[False, True]).reset_index(drop=True)
n_shuffles = int(screen.n_permutations.iloc[0])

above = screen.n_altered > screen.perm_max_altered
changed = screen.n_altered_unadjusted != screen.n_altered
x = screen.index.to_numpy()

fig, ax = plt.subplots(figsize=(7.8, 3.6))
ax.set_yscale("symlog", linthresh=1, linscale=0.55)
ax.bar(x, screen.perm_max_altered, width=0.8, color=BAR, zorder=1)
ax.vlines(x[changed], screen.n_altered[changed], screen.n_altered_unadjusted[changed], color="#B0B0B0",
          linestyles=":", linewidth=0.8, zorder=2)
ax.scatter(x[~above], screen.n_altered[~above], s=16, color=DOT, zorder=3, linewidths=0)
ax.scatter(x[above], screen.n_altered[above], s=22, color=PURPLE, zorder=3, linewidths=0)
ax.scatter(x[changed], screen.n_altered_unadjusted[changed], s=34, facecolors="white", edgecolors=ORANGE,
           linewidths=1.2, zorder=4)
for i in x[above]:
    ax.text(i + 0.55, screen.n_altered[i], f"{int(screen.n_altered[i])}", color=PURPLE, fontsize=7.5,
            fontweight="bold", va="center", ha="left")

ticks = [0, 1, 3, 10, 30, 100, 300, 1000]
ax.yaxis.set_major_locator(FixedLocator(ticks))
ax.yaxis.set_minor_locator(NullLocator())
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}"))
ax.set_ylim(-0.45, 2200)
ax.set_xlim(-0.8, len(screen) - 0.2)
ax.set_xticks(x)
ax.set_xticklabels([f"{g} ({int(n)})" for g, n in zip(screen.gene, screen.n_mutated)], rotation=90, fontsize=6.5)
ax.set_xlabel("Mutated gene (number of mutated cell lines)", fontsize=8)
ax.set_ylabel("Altered associations (FDR < 0.05)", fontsize=8)

handles = [Line2D([], [], linestyle="none", label="Tissue covariate"),
           Line2D([], [], marker="o", linestyle="none", color=PURPLE, markersize=4.5, label="Above shuffled range"),
           Line2D([], [], marker="o", linestyle="none", color=DOT, markersize=4.5, label="Within shuffled range"),
           Patch(facecolor=BAR, label=f"Maximum of {n_shuffles} shuffles"),
           Line2D([], [], marker="o", linestyle="none", markerfacecolor="white", markeredgecolor=ORANGE,
                  markeredgewidth=1.2, markersize=6, label="No tissue covariate")]
fig.legend(handles=handles, loc="upper left", ncol=5, frameon=False, fontsize=7.5, handletextpad=0.4,
           columnspacing=1.6, bbox_to_anchor=(0.06, 1.0))
fig.tight_layout(rect=(0, 0, 1, 0.95))

for ext in ("pdf", "png"):
    fig.savefig(os.path.join(out_dir, f"supp5_mutation_screen.{ext}"), dpi=600, facecolor="white")
plt.close(fig)
