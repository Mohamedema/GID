"""Fig. 3: tissue of origin in the networks of GID, LIONESS, SSN and BONOBO.
a, b  Calinski-Harabasz index, 1 / Davies-Bouldin index and 10-NN accuracy with the 28 tissues (a) or the 11 organ
      systems (b) as labels, for all cell lines, without the haematopoietic and lymphoid lines, and for the cell
      lines with all five omic layers
c     UMAP of the 50 principal components of each method (all cell lines), coloured by organ system
Input in results/benchmark/, written by python -m gid.benchmark:
  clustering_all_results_long.csv
  umap_GID_PCA50.csv, umap_LIONESS_PCA50.csv, umap_SSN_PCA50.csv, umap_BONOBO_PCA50.csv
Output in figures/: fig3_benchmark.pdf and fig3_benchmark.png (600 dpi)"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import MultipleLocator

bench_dir, out_dir = "results/benchmark", "figures"
os.makedirs(out_dir, exist_ok=True)

METHODS = ["GID", "LIONESS", "SSN", "BONOBO"]
COLORS = {"GID": "#0072B2", "LIONESS": "#E69F00", "SSN": "#009E73", "BONOBO": "#CC79A7"}
MAIN_PCT = 5                      # % of the most variable gene pairs
N_PCS = 50
KNN_K = 10
SUB_NAME = "without_subset"       # without the haematopoietic and lymphoid lines
C5_NAME = "five_layers"           # cell lines with all five omic layers
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Inter", "Arial", "DejaVu Sans"], "font.size": 7,
                     "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "xtick.major.size": 2.3, "ytick.major.size": 2.3, "ytick.alignment": "center",
                     "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})

res = pd.read_csv(os.path.join(bench_dir, "clustering_all_results_long.csv"))
main = res[(res.protocol == "harmonised") & (res.n_pcs == N_PCS) & (res.top_pct == MAIN_PCT)].copy()
main["invDB"] = 1.0 / main.DB
coords = {m: pd.read_csv(os.path.join(bench_dir, f"umap_{m}_PCA50.csv")) for m in METHODS}

subsets = ["all", SUB_NAME, C5_NAME]
n_lines = main.groupby("subset").n_samples.first()
subset_labels = {"all": f"All cell lines (n = {n_lines['all']:,})",
                 SUB_NAME: f"Without haematopoietic and lymphoid lines (n = {n_lines[SUB_NAME]:,})",
                 C5_NAME: f"Cell lines with all five layers (n = {n_lines[C5_NAME]:,})"}
hatches = [None, "////", "xxxx"]
alphas = [1.0, 0.45, 0.45]               # the lighter bars are the same colour at 45% opacity, hatch lines stay white
baseline_ls = ["--", ":", "-."]          # largest class baseline of the three analyses
panels = [("CH", "Calinski–Harabasz index"), ("invDB", "1 / Davies–Bouldin index"),
          ("kNN_acc", f"{KNN_K}-NN accuracy")]
levels = [("Tissue", "Tissue level ({} tissues)", "a"), ("Anatomy", "Organ-system level ({} organ systems)", "b")]

FIG_W, FIG_H = 7.2, 7.52
# areas of the plots in a and b, as fractions of the figure: left edge of the three columns, width, bottom edge and height of the two rows
COL_LEFT, COL_W = [0.1192, 0.4367, 0.7531], 0.2215
ROW_BOTTOM, ROW_H = [0.6856, 0.3907], [0.1998, 0.1998]
Y_MAX = {"invDB": 0.38, "kNN_acc": 0.8}  # CH: 12.5% above the highest bar of the row
Y_STEP = {"invDB": 0.1, "kNN_acc": 0.2}  # distance between the y ticks (CH uses the automatic ticks)
Y_TITLE, Y_LETTER = [0.9205, 0.6266], [0.9207, 0.6252]   # baselines of the titles and of the panel letters

fig = plt.figure(figsize=(FIG_W, FIG_H))

# legend of the three analyses
handles = [Rectangle((0, 0), 1, 1, fc=to_rgba("#555555", a), ec="white", hatch=h, lw=0)
           for a, h in zip(alphas, hatches)]
fig.legend(handles, [subset_labels[s] for s in subsets], loc="upper center", bbox_to_anchor=(0.5, 0.9886),
           ncol=3, frameon=False, fontsize=6.2, handlelength=2.0, handleheight=0.9, handletextpad=0.5,
           columnspacing=1.35)

# a, b
w = 0.8 / len(subsets)
for r, (level, title, letter) in enumerate(levels):
    n_classes = int(main[(main.level == level) & (main.subset == "all")].n_classes.iloc[0])
    ch_max = main[main.level == level].CH.max()
    for c, (metric, ylab) in enumerate(panels):
        ax = fig.add_axes([COL_LEFT[c], ROW_BOTTOM[r], COL_W, ROW_H[r]])
        for si, sub in enumerate(subsets):
            for mi, m in enumerate(METHODS):
                row = main[(main.method == m) & (main.subset == sub) & (main.level == level)].iloc[0]
                x = mi + (si - (len(subsets) - 1) / 2) * w
                ax.bar(x, row[metric], w * 0.92, color=to_rgba(COLORS[m], alphas[si]), hatch=hatches[si],
                       edgecolor="white", linewidth=0)
                if metric == "kNN_acc":
                    ax.errorbar(x, row.kNN_acc, yerr=[[max(0.0, row.kNN_acc - row.kNN_lo)],
                                                      [max(0.0, row.kNN_hi - row.kNN_acc)]],
                                color="k", elinewidth=0.6, capsize=1.2, capthick=0.6)
        if metric == "CH":
            ax.axhline(1, color="grey", ls="--", lw=0.8)
            ax.set_ylim(0, ch_max * 1.125)
        else:
            ax.set_ylim(0, Y_MAX[metric])
            ax.yaxis.set_major_locator(MultipleLocator(Y_STEP[metric]))
        if metric == "kNN_acc":
            for sub, ls in zip(subsets, baseline_ls):
                base = main[(main.level == level) & (main.subset == sub)].majority_baseline.iloc[0]
                ax.axhline(base, color="grey", ls=ls, lw=0.8)
        ax.set_xticks(range(len(METHODS)))
        ax.set_xticklabels(METHODS, fontsize=6.7)
        ax.tick_params(axis="x", pad=0.5)
        ax.tick_params(axis="y", labelsize=6.5, pad=1.9)
        ax.set_ylabel(ylab, fontsize=6.7)
        ax.yaxis.set_label_coords(-0.273, 0.5)
    fig.text(0.0073, Y_LETTER[r], letter, fontsize=10.5, fontweight="bold", ha="left", va="baseline")
    fig.text(0.0412, Y_TITLE[r], title.format(n_classes), fontsize=8.3, fontweight="bold", ha="left", va="baseline")

# c: UMAP of each method, coloured by organ system
organs = sorted({o for df in coords.values() for o in df.Anatomy})
cmap = plt.get_cmap("tab20")
col = {o: cmap(i % 20) for i, o in enumerate(organs)}
UMAP_CX, UMAP_CY, UMAP_W, UMAP_H = [0.1283, 0.3701, 0.6119, 0.8537], 0.1596, 0.2155, 0.1907
fig.text(0.0073, 0.3029, "c", fontsize=10.5, fontweight="bold", ha="left", va="baseline")
fig.text(0.0412, 0.3043, "UMAP of the first {} principal components (all cell lines), coloured by organ system"
         .format(N_PCS), fontsize=8.3, fontweight="bold", ha="left", va="baseline")
for k, (m, df) in enumerate(coords.items()):
    ax = fig.add_axes([UMAP_CX[k] - UMAP_W / 2, UMAP_CY - UMAP_H / 2, UMAP_W, UMAP_H])
    ax.scatter(df.UMAP1, df.UMAP2, s=2.9, c=[col[o] for o in df.Anatomy], linewidths=0)   # one pass, in the order of the file
    ax.set_aspect("equal", adjustable="box")
    ax.margins(0.01)
    ax.axis("off")
    fig.text(UMAP_CX[k] + 0.0011, 0.2658, m, fontsize=8.5, fontweight="bold", color=COLORS[m], ha="center",
             va="baseline")
handles = [Line2D([0], [0], marker="o", ls="", markerfacecolor=col[o], markeredgecolor="none", markersize=4.5,
                  label=o) for o in organs]
fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5011, 0.0610), ncol=6, frameon=False,
           fontsize=6.3, handletextpad=0.2, columnspacing=1.15, labelspacing=0.1)

for ext in ("pdf", "png"):
    fig.savefig(os.path.join(out_dir, f"fig3_benchmark.{ext}"), dpi=600, facecolor="white")
