import argparse
import json
import os

import numpy as np
import pandas as pd

from .data import EdgeMatrix, load_omics

PERCENTILE = 70
LIKELY = 0.5
CONFIDENT = 0.7


def global_percentile(x, q, block=16, bins=1 << 20):
    """np.percentile(x, q) for a samples x edges array that is too large to load at once.

    A first pass counts the values in `bins` equal-width bins; a second pass collects the values
    of the bins that contain the two neighbouring order statistics and sorts only those.
    """
    n_total = x.shape[0] * x.shape[1]
    pos = (n_total - 1) * q / 100
    k0 = int(np.floor(pos))
    k1 = min(k0 + 1, n_total - 1)
    frac = pos - k0

    lo, hi = np.inf, -np.inf
    for start in range(0, x.shape[0], block):
        part = np.asarray(x[start:start + block])
        lo, hi = min(lo, float(part.min())), max(hi, float(part.max()))
    if lo == hi:
        return lo
    scale = bins / (hi - lo)

    def bin_of(part):
        idx = ((part.astype(np.float64) - lo) * scale).astype(np.int64)
        return np.minimum(idx, bins - 1)

    counts = np.zeros(bins, dtype=np.int64)
    for start in range(0, x.shape[0], block):
        counts += np.bincount(bin_of(np.asarray(x[start:start + block])).ravel(), minlength=bins)
    cum = np.cumsum(counts)
    b0 = int(np.searchsorted(cum, k0 + 1))
    b1 = int(np.searchsorted(cum, k1 + 1))
    below = int(cum[b0 - 1]) if b0 > 0 else 0

    picked = []
    for start in range(0, x.shape[0], block):
        part = np.asarray(x[start:start + block])
        idx = bin_of(part)
        picked.append(part[(idx >= b0) & (idx <= b1)].astype(np.float64))
    picked = np.sort(np.concatenate(picked))
    a, b = picked[k0 - below], picked[k1 - below]
    diff = b - a
    return b - diff * (1 - frac) if frac >= 0.5 else a + diff * frac


def tissue_probability(x, tissues, threshold):
    """Fraction of the samples of each tissue in which an edge is above the threshold."""
    tissues = np.asarray(tissues, dtype=str)
    names = sorted(set(tissues))
    prob = np.zeros((len(names), x.shape[1]), dtype=np.float32)
    for t, name in enumerate(names):
        rows = np.flatnonzero(tissues == name)
        for r in rows:
            prob[t] += (x[r] > threshold).astype(np.float32)
        prob[t] /= len(rows)
    return names, prob


def summarise(names, prob, likely=LIKELY, confident=CONFIDENT):
    return pd.DataFrame({
        "tissue": names,
        "quantified": prob.shape[1],
        "likely": (prob > likely).sum(axis=1),
        "confident": (prob > confident).sum(axis=1),
    })


def sharing(prob, likely=LIKELY):
    """Number of tissues in which each edge is likely, and how many edges are shared by k tissues."""
    shared = (prob > likely).sum(axis=0)
    counts = np.bincount(shared, minlength=prob.shape[0] + 1)[1:]
    distribution = pd.DataFrame({"num_tissues": range(1, prob.shape[0] + 1), "num_associations": counts})
    per_edge = pd.DataFrame({"edge_index": range(prob.shape[1]), "tissues_present_in": shared})
    return distribution, per_edge


def main():
    ap = argparse.ArgumentParser(description="Association probabilities per tissue from the sample networks")
    ap.add_argument("--data", required=True, help="folder written by gid.preprocess")
    ap.add_argument("--networks", required=True, help="sample_networks.npy written by gid.infer")
    ap.add_argument("--out", required=True)
    ap.add_argument("--percentile", type=float, default=PERCENTILE)
    ap.add_argument("--likely", type=float, default=LIKELY)
    ap.add_argument("--confident", type=float, default=CONFIDENT)
    args = ap.parse_args()

    data = load_omics(args.data)
    store = EdgeMatrix(args.networks, data.n_genes)
    threshold = np.float32(global_percentile(store.x, args.percentile))
    print(f"global threshold (percentile {args.percentile:g}): {threshold:.6f}")

    names, prob = tissue_probability(store.x, data.meta["tissue"].to_numpy(), threshold)
    os.makedirs(args.out, exist_ok=True)
    summarise(names, prob, args.likely, args.confident).to_csv(os.path.join(args.out, "tissue_summary.csv"), index=False)
    distribution, per_edge = sharing(prob, args.likely)
    distribution.to_csv(os.path.join(args.out, "sharing_distribution.csv"), index=False)
    per_edge.to_csv(os.path.join(args.out, "edge_tissue_sharing.csv"), index=False)
    with open(os.path.join(args.out, "association_threshold.json"), "w") as fh:
        json.dump({"percentile": args.percentile, "threshold": float(threshold)}, fh)


if __name__ == "__main__":
    main()
