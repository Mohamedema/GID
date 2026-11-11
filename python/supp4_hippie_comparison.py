"""Appendix Figure S4: GID with the STRING prior against GID with the HIPPIE prior.
a  organ system benchmark (Calinski-Harabasz, 1 / Davies-Bouldin, 10-NN accuracy)
b  UMAP of the GID networks with each prior, coloured by organ system
c  complexes assigned to a tissue with each prior
d  altered associations per gene of the mutation screen with each prior
e  genes with at least one altered association with each prior"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
from scipy.optimize import brentq

string_dir, hippie_dir, out_dir = "results", "results/hippie", "figures"
os.makedirs(out_dir, exist_ok=True)

BLUE, ORANGE, GREY = "#0072B2", "#E69F00", "#8C8C8C"
plt.rcParams.update({"font.sans-serif": ["Inter", "Arial", "DejaVu Sans"], "font.size": 7.5, "axes.linewidth": 0.7,
                     "xtick.major.width": 0.7, "ytick.major.width": 0.7, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42})

def read(folder, *parts, **kw):
    return pd.read_csv(os.path.join(folder, *parts), **kw)

# a: clustering benchmark, all 1,007 lines, 5% of the gene pairs, 50 principal components, organ system labels
string_bench = read(string_dir, "benchmark", "clustering_all_results_long.csv")
hippie_bench = read(hippie_dir, "benchmark", "clustering_all_results_long.csv")

def benchmark_row(d, method):
    r = d[(d.method == method) & (d.protocol == "harmonised") & (d.subset == "all") & (d.top_pct == 5) &
          (d.n_pcs == 50) & (d.level == "Anatomy")]
    assert len(r) == 1, (method, len(r))
    return r.iloc[0]

bars = [("GID, STRING", benchmark_row(string_bench, "GID"), BLUE), ("GID, HIPPIE", benchmark_row(hippie_bench, "GID"), ORANGE),
        ("LIONESS", benchmark_row(hippie_bench, "LIONESS"), GREY), ("SSN", benchmark_row(hippie_bench, "SSN"), GREY),
        ("BONOBO", benchmark_row(hippie_bench, "BONOBO"), GREY)]
majority = float(benchmark_row(hippie_bench, "GID")["majority_baseline"])

# b: UMAP coordinates
umap_string = read(string_dir, "benchmark", "umap_GID_PCA50.csv")
umap_hippie = read(hippie_dir, "benchmark", "umap_GID_PCA50.csv")

# c: complex and tissue pairs
cx_string = read(string_dir, "Final_Rigorously_Tissue_Specific_Complexes.csv", low_memory=False)
cx_hippie = read(hippie_dir, "Final_Rigorously_Tissue_Specific_Complexes.csv", low_memory=False)
pairs_string = set(zip(cx_string.Tissue, cx_string.complex_id.astype(int)))
pairs_hippie = set(zip(cx_hippie.Tissue, cx_hippie.complex_id.astype(int)))
n_string, n_hippie, n_both = len(pairs_string), len(pairs_hippie), len(pairs_string & pairs_hippie)

# d, e: mutation screen
screen_string = read(string_dir, "mutations", "all_genes_summary.csv")
screen_hippie = read(hippie_dir, "mutations", "all_genes_summary.csv")
screen_string = screen_string[screen_string.status == "ok"].set_index("gene")
screen_hippie = screen_hippie[screen_hippie.status == "ok"].set_index("gene")
genes = sorted(set(screen_string.index) & set(screen_hippie.index))
alt_string = screen_string.loc[genes, "n_altered"].astype(int)
alt_hippie = screen_hippie.loc[genes, "n_altered"].astype(int)
genes_both = int(((alt_string > 0) & (alt_hippie > 0)).sum())
genes_string_only = int(((alt_string > 0) & (alt_hippie == 0)).sum())
genes_hippie_only = int(((alt_string == 0) & (alt_hippie > 0)).sum())

def lens_area(r1, r2, d):
    if d >= r1 + r2:
        return 0.0
    if d <= abs(r1 - r2):
        return np.pi * min(r1, r2) ** 2
    a1 = r1 ** 2 * np.arccos((d * d + r1 * r1 - r2 * r2) / (2 * d * r1))
    a2 = r2 ** 2 * np.arccos((d * d + r2 * r2 - r1 * r1) / (2 * d * r2))
    a3 = 0.5 * np.sqrt((-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2))
    return a1 + a2 - a3

def venn2(ax, n1, n2, nb, c1, c2, l1, l2, fs=8):
    """Two circles with areas proportional to n1 and n2 and an overlap area proportional to nb."""
    r1, r2 = np.sqrt(n1 / np.pi), np.sqrt(n2 / np.pi)
    d = brentq(lambda x: lens_area(r1, r2, x) - nb, abs(r1 - r2) + 1e-9, r1 + r2 - 1e-9)
    x1, x2 = -d / 2, d / 2
    ax.add_patch(Circle((x1, 0), r1, fc=c1, ec=c1, alpha=0.45, lw=1.0))
    ax.add_patch(Circle((x2, 0), r2, fc=c2, ec=c2, alpha=0.45, lw=1.0))
    left, lens, right = (x1 - r1 + (x2 - r2)) / 2, ((x2 - r2) + (x1 + r1)) / 2, ((x1 + r1) + (x2 + r2)) / 2
    for xpos, n in ((left, n1 - nb), (lens, nb), (right, n2 - nb)):
        ax.text(xpos, 0, str(n), ha="center", va="center", fontsize=fs, fontweight="bold")
    top = max(r1, r2)
    ax.text(x1 - 0.30 * r1, top + 0.25, l1, ha="center", va="bottom", fontsize=fs - 0.5, color=c1, fontweight="bold")
    ax.text(x2 + 0.30 * r2, top + 0.25, l2, ha="center", va="bottom", fontsize=fs - 0.5, color=c2, fontweight="bold")
    ax.set_xlim(x1 - r1 - 0.4, x2 + r2 + 0.4)
    ax.set_ylim(-top - 0.4, top + 1.9)
    ax.set_aspect("equal")
    ax.axis("off")

def panel_letter(ax, letter, x=-0.02, y=1.02):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=11, fontweight="bold", va="bottom", ha="right")

fig = plt.figure(figsize=(7.2, 8.4))
grid = fig.add_gridspec(7, 1, height_ratios=[2.05, 0.80, 2.05, 0.20, 0.30, 0.10, 2.55], hspace=0.0,
                        left=0.105, right=0.985, top=0.955, bottom=0.04)

# a
sub = grid[0].subgridspec(1, 3, wspace=0.50)
metrics = [("CH", "Calinski–Harabasz\nindex"), ("invDB", "1 / Davies–Bouldin\nindex"), ("kNN_acc", "10-NN accuracy")]
for j, (col, ylab) in enumerate(metrics):
    ax = fig.add_subplot(sub[0, j])
    xs = np.arange(len(bars))
    vals = [b[1][col] for b in bars]
    ax.bar(xs, vals, width=0.74, color=[b[2] for b in bars], edgecolor="none")
    if col == "kNN_acc":
        lo, hi = [b[1]["kNN_lo"] for b in bars], [b[1]["kNN_hi"] for b in bars]
        ax.errorbar(xs, vals, yerr=[np.array(vals) - np.array(lo), np.array(hi) - np.array(vals)], fmt="none",
                    ecolor="black", elinewidth=0.8, capsize=1.8)
        ax.axhline(majority, color="#555555", ls="--", lw=0.8)
    if col == "CH":
        ax.axhline(1.0, color="#555555", ls="--", lw=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels([b[0] for b in bars], fontsize=6.5, rotation=40, ha="right", rotation_mode="anchor")
    ax.set_ylabel(ylab, fontsize=7)
    ax.set_ylim(0, {"CH": 16, "invDB": 0.38, "kNN_acc": 0.8}[col])
    if j == 0:
        panel_letter(ax, "a", x=-0.42, y=1.06)

# b
sub = grid[2].subgridspec(1, 2, wspace=0.05)
systems = ["Digestive", "Endocrine", "Head and Neck", "Immune", "Neural", "Reproductive", "Respiratory", "Skeletal",
           "Skin", "Soft Tissue", "Urinary"]
system_colours = dict(zip(systems, ["#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c", "#98df8a", "#d62728",
                                    "#ff9896", "#9467bd", "#c5b0d5", "#8c564b"]))
assert set(umap_string.Anatomy.unique()) == set(systems)
for j, (u, title) in enumerate(((umap_string, "GID with STRING"), (umap_hippie, "GID with HIPPIE"))):
    ax = fig.add_subplot(sub[0, j])
    for s in systems:
        m = u.Anatomy == s
        ax.scatter(u.UMAP1[m], u.UMAP2[m], s=7, c=system_colours[s], linewidths=0, alpha=0.95)
    ax.set_title(title, fontsize=8, fontweight="bold", color=BLUE if j == 0 else ORANGE, pad=3)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("UMAP1", fontsize=6.5, labelpad=2)
    if j == 0:
        ax.set_ylabel("UMAP2", fontsize=6.5, labelpad=2)
        panel_letter(ax, "b", x=-0.06, y=1.0)
ax = fig.add_subplot(grid[4])
ax.axis("off")
handles = [Line2D([0], [0], marker="o", ls="", markerfacecolor=system_colours[s], markeredgecolor="none",
                  markersize=4.5, label=s) for s in systems]
ax.legend(handles=handles, ncol=6, loc="center", frameon=False, fontsize=6.6, handletextpad=0.2, columnspacing=1.0)

# c, d, e
sub = grid[6].subgridspec(1, 3, wspace=0.30, width_ratios=[1, 1.15, 1])
ax = fig.add_subplot(sub[0, 0])
venn2(ax, n_string, n_hippie, n_both, BLUE, ORANGE, f"STRING\n{n_string}", f"HIPPIE\n{n_hippie}")
panel_letter(ax, "c", x=0.0, y=0.96)
ax.text(0.5, -0.02, "complex and tissue pairs", transform=ax.transAxes, ha="center", va="top", fontsize=6.5)

ax = fig.add_subplot(sub[0, 1])
x, y = np.log10(alt_string.values + 1), np.log10(alt_hippie.values + 1)
lim = 3.2
ax.plot([0, lim], [0, lim], ls="--", color="#555555", lw=0.8, zorder=1)
altered = (alt_string.values > 0) | (alt_hippie.values > 0)
ax.scatter([0], [0], s=125, c="#bdbdbd", edgecolors="white", linewidths=0.4, zorder=2)
ax.text(0, 0, str(int((~altered).sum())), ha="center", va="center", fontsize=6.4, color="#222222", zorder=5, fontweight="bold")
ax.scatter(x[altered], y[altered], s=22, c="#444444", edgecolors="white", linewidths=0.4, zorder=3)
rb1 = genes.index("RB1")
ax.scatter([x[rb1]], [y[rb1]], s=30, c="#B22222", edgecolors="white", linewidths=0.4, zorder=4)
ax.annotate("RB1", (x[rb1], y[rb1]), xytext=(-4, 7), textcoords="offset points", ha="right", fontsize=7,
            fontweight="bold", color="#B22222")
ticks = [0, 1, 10, 100, 1000]
ax.set_xticks(np.log10(np.array(ticks) + 1))
ax.set_xticklabels([str(t) for t in ticks], fontsize=6.5)
ax.set_yticks(np.log10(np.array(ticks) + 1))
ax.set_yticklabels([str(t) for t in ticks], fontsize=6.5)
ax.set_xlim(-0.12, lim)
ax.set_ylim(-0.12, lim)
ax.set_xlabel("Altered associations with STRING", fontsize=7)
ax.set_ylabel("Altered associations with HIPPIE", fontsize=7)
ax.set_aspect("equal", adjustable="box")
panel_letter(ax, "d", x=-0.2, y=1.0)

ax = fig.add_subplot(sub[0, 2])
venn2(ax, genes_string_only + genes_both, genes_hippie_only + genes_both, genes_both, BLUE, ORANGE,
      f"STRING\n{genes_string_only + genes_both}", f"HIPPIE\n{genes_hippie_only + genes_both}")
panel_letter(ax, "e", x=0.0, y=0.96)
ax.text(0.5, -0.02, "genes with at least one\naltered association", transform=ax.transAxes, ha="center", va="top",
        fontsize=6.5)

for ext in ("pdf", "png"):
    fig.savefig(os.path.join(out_dir, f"supp4_hippie_comparison.{ext}"), dpi=600, facecolor="white")
