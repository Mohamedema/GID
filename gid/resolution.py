import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score

from .benchmark import ANATOMY_MAP, organ_labels
from .data import EdgeMatrix, load_omics
from .embedding import edge_variance

MIN_VARIANCE = 2.0
COMPONENTS = 50
BINARY_CUTOFF = 0.7


def retained_edges(store, min_variance=MIN_VARIANCE):
    """Columns of the samples x edges matrix whose variance across the samples is at least min_variance."""
    return np.flatnonzero(edge_variance(store.x) >= min_variance)


def clustering_metrics(x, tissues, organs, components=COMPONENTS, binary_cutoff=BINARY_CUTOFF, seed=42):
    """Calinski-Harabasz and Davies-Bouldin indices of the samples, with tissues and organ systems as labels.

    The continuous values are projected on their first principal components, the binary version
    (edge score >= binary_cutoff) is used as it is.
    """
    scores = PCA(n_components=components, random_state=seed).fit_transform(x)
    binary = (x >= binary_cutoff).astype(float)
    row = {}
    for kind, matrix in (("Continuous", scores), ("Binary", binary)):
        for level, labels in (("Tissue", tissues), ("Organ", organs)):
            row[f"CH_{level}_{kind}"] = calinski_harabasz_score(matrix, labels)
            row[f"DB_{level}_{kind}"] = davies_bouldin_score(matrix, labels)
    return row


def tissue_means(x, tissues):
    names = sorted(set(tissues))
    return names, np.vstack([x[tissues == name].mean(axis=0) for name in names])


def zscore(m):
    return (m - m.mean(axis=0)) / (m.std(axis=0) + 1e-8)


def spearman_table(rows, names):
    corr, _ = spearmanr(rows, axis=1)
    return pd.DataFrame(corr, index=names, columns=names)


def tissue_similarity(means, names):
    """Spearman correlation between the tissues of the mean edge vectors, standardised for every edge."""
    return spearman_table(zscore(means), names)


def organ_similarity(means, tissue_names, anatomy):
    """Spearman correlation between organ systems: tissue means averaged per system, then standardised."""
    organs = np.array([anatomy.get(t, "Other") for t in tissue_names])
    order = sorted(set(organs))
    pooled = np.vstack([means[organs == o].mean(axis=0) for o in order])
    return spearman_table(zscore(pooled), order)


def main():
    ap = argparse.ArgumentParser(description="Tissue of origin resolution of the sample networks and tissue similarity")
    ap.add_argument("--data", required=True, help="folder written by gid.preprocess")
    ap.add_argument("--networks", required=True, help="sample_networks.npy written by gid.infer")
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="all_omics", help="run label, the prefix of the metrics file")
    ap.add_argument("--min-variance", type=float, default=MIN_VARIANCE)
    ap.add_argument("--components", type=int, default=COMPONENTS)
    ap.add_argument("--binary-cutoff", type=float, default=BINARY_CUTOFF)
    ap.add_argument("--anatomy", default=None, help="csv with the columns Tissue and Anatomy")
    ap.add_argument("--similarity", action="store_true", help="also write the tissue and organ system correlations")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data = load_omics(args.data)
    tissues = data.meta["tissue"].astype(str).to_numpy()
    anatomy = ANATOMY_MAP
    if args.anatomy:
        table = pd.read_csv(args.anatomy)
        anatomy = dict(zip(table["Tissue"], table["Anatomy"]))

    store = EdgeMatrix(args.networks, data.n_genes)
    kept = retained_edges(store, args.min_variance)
    print(f"{len(kept):,} of {store.n_edges:,} edges have a variance of at least {args.min_variance:g}")
    x = np.asarray(store.x[:, kept], dtype=np.float64)

    row = {"Tag": f"var{args.min_variance:g}"}
    row.update(clustering_metrics(x, tissues, organ_labels(tissues, anatomy), args.components,
                                  args.binary_cutoff, args.seed))
    os.makedirs(args.out, exist_ok=True)
    pd.DataFrame([row]).to_csv(os.path.join(args.out, f"{args.name}_clustering_metrics.csv"), index=False)
    print({k: round(v, 3) for k, v in row.items() if k != "Tag"})

    if args.similarity:
        names, means = tissue_means(x, tissues)
        tissue_similarity(means, names).to_csv(os.path.join(args.out, "tissue_similarity.csv"))
        organ_similarity(means, names, anatomy).to_csv(os.path.join(args.out, "anatomy_similarity.csv"))


if __name__ == "__main__":
    main()
