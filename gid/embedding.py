import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler

from .data import EdgeMatrix, load_omics


def edge_variance(x, block=20000):
    out = np.empty(x.shape[1])
    for start in range(0, x.shape[1], block):
        out[start:start + block] = np.asarray(x[:, start:start + block], dtype=np.float64).var(axis=0)
    return out


def variance_filter(x, top_pct=5.0):
    """Columns of the samples x edges matrix in the top_pct percent with the highest variance."""
    var = edge_variance(x)
    return np.flatnonzero(var >= np.percentile(var, 100 - top_pct))


def filtered_scaled(x, top_pct=5.0):
    kept = variance_filter(x, top_pct)
    return StandardScaler().fit_transform(np.asarray(x[:, kept], dtype=np.float64))


def umap_model(**params):
    """UMAP as in the 0.5 releases; later releases keep that pipeline only when asked to."""
    import inspect
    import umap
    if "compatibility_layout" in inspect.signature(umap.UMAP).parameters:
        params["compatibility_layout"] = True
    return umap.UMAP(**params)


def sample_umap(scaled, seed=42):
    reducer = umap_model(n_neighbors=15, min_dist=0.1, metric="euclidean", n_components=2, random_state=seed)
    return reducer.fit_transform(scaled)


def clustering_index(scaled, labels):
    return calinski_harabasz_score(scaled, labels), davies_bouldin_score(scaled, labels)


def main():
    ap = argparse.ArgumentParser(description="UMAP and CH/DB indices of the sample networks after a variance filter")
    ap.add_argument("--data", required=True, help="folder written by gid.preprocess")
    ap.add_argument("--networks", required=True, help="sample_networks.npy written by gid.infer")
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="GID", help="label used in the output file names")
    ap.add_argument("--top-pct", type=float, default=5.0)
    ap.add_argument("--anatomy", default=None, help="csv with the columns Tissue and Anatomy")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data = load_omics(args.data)
    tissues = data.meta["tissue"].to_numpy()
    store = EdgeMatrix(args.networks, data.n_genes)
    scaled = filtered_scaled(store.x, args.top_pct)
    print(f"{scaled.shape[1]:,} edges kept")

    table = pd.DataFrame(sample_umap(scaled, args.seed), columns=["UMAP1", "UMAP2"])
    table["Tissue"] = tissues
    if args.anatomy:
        table = table.merge(pd.read_csv(args.anatomy), on="Tissue", how="left")
    ch, db = clustering_index(scaled, tissues)
    print(f"CH {ch:.4f}  DB {db:.4f}")

    os.makedirs(args.out, exist_ok=True)
    table.to_csv(os.path.join(args.out, f"umap_cont_tissue_{args.name}.csv"), index=False)
    pd.DataFrame([{"method": args.name, "calinski_harabasz": ch, "davies_bouldin": db, "top_variance_pct": args.top_pct}]).to_csv(
        os.path.join(args.out, f"clustering_index_{args.name}.csv"), index=False)


if __name__ == "__main__":
    main()
