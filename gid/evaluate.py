import argparse
import itertools
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import auc, average_precision_score, roc_curve

from .data import load_omics


def complex_pairs(path, column="subunits_gene_name"):
    """All gene pairs that share a CORUM complex, as sorted tuples."""
    corum = pd.read_csv(path, sep="\t")
    pairs = set()
    for subunits in corum[column].dropna():
        genes = [g.strip() for g in subunits.split(";") if g.strip()]
        for a, b in itertools.combinations(genes, 2):
            pairs.add(tuple(sorted((a, b))))
    return pairs


def pair_matrix(pairs, genes):
    pos = {g: i for i, g in enumerate(genes)}
    m = np.zeros((len(genes), len(genes)), dtype=int)
    for a, b in pairs:
        if a in pos and b in pos:
            m[pos[a], pos[b]] = m[pos[b], pos[a]] = 1
    return m


def recovery(score, truth):
    """ROC and precision-recall summary of a score matrix against a 0/1 matrix, over the pairs i < j."""
    iu = np.triu_indices(score.shape[0], 1)
    y, s = truth[iu], score[iu]
    if y.min() == y.max():
        raise ValueError("only one class among the pairs")
    fpr, tpr, _ = roc_curve(y, s)
    return dict(fpr=fpr, tpr=tpr, auc=float(auc(fpr, tpr)), average_precision=float(average_precision_score(y, s)),
                n_positive=int(y.sum()), n_negative=int(len(y) - y.sum()))


def string_recovery(population, adj, threshold=0.7):
    truth = (np.asarray(adj) >= threshold).astype(int)
    out = recovery(population, truth)
    iu = np.triu_indices(truth.shape[0], 1)
    out["pearson_r"] = float(pearsonr(np.asarray(adj)[iu], population[iu])[0])
    return out


def corum_recovery(population, genes, corum_path):
    return recovery(population, pair_matrix(complex_pairs(corum_path), genes))


def main():
    ap = argparse.ArgumentParser(description="How well does the population network recover STRING and CORUM")
    ap.add_argument("--data", required=True)
    ap.add_argument("--population", required=True, help="population_network.npy from gid.infer")
    ap.add_argument("--corum", default=None, help="corum_humanComplexes.txt")
    ap.add_argument("--threshold", type=float, default=0.7)
    ap.add_argument("--name", default="all_omics", help="run label, the prefix of the output files")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    data = load_omics(args.data)
    pop = np.load(args.population)
    os.makedirs(args.out, exist_ok=True)
    summary = {}
    runs = {"string": string_recovery(pop, data.adj, args.threshold)}
    if args.corum:
        runs["corum"] = corum_recovery(pop, data.genes, args.corum)
    for name, r in runs.items():
        pd.DataFrame({"FPR": r.pop("fpr"), "TPR": r.pop("tpr")}).to_csv(
            os.path.join(args.out, f"{args.name}_roc_{name}.csv"), index=False)
        summary[name] = r
        print(name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    with open(os.path.join(args.out, f"{args.name}_recovery.json"), "w") as fh:
        json.dump(summary, fh, indent=2)


if __name__ == "__main__":
    main()
