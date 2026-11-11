import argparse
import os
from itertools import combinations

import numpy as np
import pandas as pd

from .data import load_omics, upper_index
from .references import reference_masks

ORDER = ["CORUM", "STRING", "Reactome", "SIGNOR", "BioPlex", "HuRI", "Quantified"]


def top_edges(network, keep=60):
    """Min-max normalise the scores of a network and keep the edges above the (100 - keep) percentile."""
    ii, jj = upper_index(network.shape[0])
    scores = np.array(network, dtype=np.float32)[ii, jj]
    lo, hi = scores.min(), scores.max()
    norm = (scores - lo) / (hi - lo) if hi > lo else scores
    return norm > np.percentile(norm, 100 - keep)


def jaccard(a, b):
    union = np.count_nonzero(a | b)
    return np.count_nonzero(a & b) / union if union else np.nan


def jaccard_pairs(names, edges, references):
    """Jaccard index of every pair of tissues on all edges and on the edges of each reference resource."""
    pairs = list(combinations(range(len(names)), 2))
    rows = []
    for db in ORDER:
        if db != "Quantified" and db not in references:
            continue
        for s, t in pairs:
            a, b = edges[s], edges[t]
            if db != "Quantified":
                a, b = a & references[db], b & references[db]
            rows.append((names[s], names[t], db, jaccard(a, b)))
    return pd.DataFrame(rows, columns=["tissue_1", "tissue_2", "database", "jaccard_index"])


def main():
    ap = argparse.ArgumentParser(description="Pairwise Jaccard index of the tissue networks")
    ap.add_argument("--data", required=True, help="folder written by gid.preprocess")
    ap.add_argument("--networks", required=True, help="tissue_networks.npy written by gid.infer")
    ap.add_argument("--names", required=True, help="tissue_names.txt written by gid.infer")
    ap.add_argument("--references", required=True, help="folder with the reference resources")
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep", type=float, default=60, help="percentage of edges kept in each tissue")
    ap.add_argument("--min-samples", type=int, default=5)
    ap.add_argument("--string-threshold", type=float, default=0.7)
    args = ap.parse_args()

    data = load_omics(args.data)
    names = [line.strip() for line in open(args.names) if line.strip()]
    networks = np.load(args.networks, mmap_mode="r")
    counts = data.meta["tissue"].value_counts()
    use = [k for k, name in enumerate(names) if counts.get(name, 0) >= args.min_samples]
    print(f"{len(use)} tissues with at least {args.min_samples} samples")

    references = reference_masks(data.genes, data.adj, args.references, args.string_threshold)
    print("reference resources:", ", ".join(references))
    edges = [top_edges(networks[k], args.keep) for k in use]
    table = jaccard_pairs([names[k] for k in use], edges, references)

    os.makedirs(args.out, exist_ok=True)
    table.to_csv(os.path.join(args.out, f"tissue_jaccard_pairs_top{args.keep:g}percent.csv"), index=False)


if __name__ == "__main__":
    main()
