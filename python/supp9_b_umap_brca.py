"""Appendix Figure S9b: UMAP of the pooled sample embeddings of the TCGA-BRCA tumours, coloured by PAM50 subtype.
Input in results/tcga/: umap_samples.csv, written by python -m gid.pooling for the TCGA-BRCA model.
Output in figures/: supp9_b_umap_brca.pdf and supp9_b_umap_brca.png (600 dpi)"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

results_dir, out_dir = "results", "figures"
os.makedirs(out_dir, exist_ok=True)

umap = pd.read_csv(os.path.join(results_dir, "tcga", "umap_samples.csv"))

subtype_order = ["LumA", "LumB", "Her2", "Basal"]
palette = {"Her2": "#00A087", "LumA": "#7570B3", "LumB": "#D55E00", "Basal": "#E7298A"}

sns.set_theme(style="ticks", rc={"axes.facecolor": (0, 0, 0, 0)})
plt.figure(figsize=(9, 7))
ax = sns.scatterplot(data=umap, x="UMAP1", y="UMAP2", hue="Tissue", hue_order=subtype_order, palette=palette,
                     s=70, alpha=0.85, edgecolor="white", linewidth=0.5)
ax.set_xlabel("UMAP 1", fontsize=14, fontweight="bold", labelpad=10)
ax.set_ylabel("UMAP 2", fontsize=14, fontweight="bold", labelpad=10)
ax.set_xticks([])
ax.set_yticks([])
sns.despine()
plt.legend(bbox_to_anchor=(1.05, 0.95), loc="upper left", frameon=False, title="", fontsize=12, markerscale=1.5)
plt.tight_layout()

for ext in ("pdf", "png"):
    plt.savefig(os.path.join(out_dir, f"supp9_b_umap_brca.{ext}"), dpi=600, bbox_inches="tight")
plt.close()
