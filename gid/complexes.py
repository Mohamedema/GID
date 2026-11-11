import argparse
import os

import numpy as np
import pandas as pd

GENE_COLUMN = "subunits_gene_name"
EXCLUSIVE = "Exclusive (inf)"
RESULT_COLUMNS = ["Tissue", "Complex_ID_in_Model", "Density_Score", "Second_Best_Tissue", "Second_Best_Score",
                  "Is_Exclusive", "Fold_Change"]


def read_corum(path):
    return pd.read_csv(path, sep="\t")


def complexes_in_network(corum, genes, min_genes=3, column=GENE_COLUMN, unique=False):
    """Complex name -> positions in `genes` of its subunits.

    Subunits listed twice in CORUM are counted twice unless unique=True.
    """
    position = {g: i for i, g in enumerate(genes)}
    found = {}
    for idx, members in corum[column].items():
        if pd.isna(members):
            continue
        names = [m.strip() for m in members.split(";") if m.strip()]
        inside = [position[m] for m in names if m in position]
        if unique:
            inside = list(dict.fromkeys(inside))
        if len(inside) >= min_genes:
            found[f"Complex_{idx}"] = inside
    return found


def binarise(network, percentile=95):
    a = np.array(network)
    np.fill_diagonal(a, 0)
    return a >= np.percentile(a, percentile)


def density(binary, members):
    n = len(members)
    pairs = n * (n - 1) / 2
    sub = binary[np.ix_(members, members)]
    return sub[np.triu_indices(n, k=1)].sum() / pairs


def density_matrix(networks, names, complexes, percentile=95):
    """Fraction of connected gene pairs per complex (rows) and tissue (columns)."""
    binary = [binarise(net, percentile) for net in networks]
    values = np.zeros((len(complexes), len(names)))
    for row, members in enumerate(complexes.values()):
        for col, b in enumerate(binary):
            values[row, col] = density(b, members)
    out = pd.DataFrame(values, index=list(complexes), columns=list(names))
    return out.sort_index().sort_index(axis=1)


def fold_change(top, second, eps):
    return top / (second + eps)


def specific_complexes(dens, quantile=0.90, min_fold=1.0):
    """Assign each complex to the tissue where its density is highest.

    The complex is kept when that density reaches the quantile of the complex densities in the
    tissue and when either every other tissue has density zero or the fold change over the second
    highest tissue is at least min_fold. The smallest non-zero density of the matrix is added to
    the denominator of the fold change.
    """
    values = dens.to_numpy()
    if not (values > 0).any():
        return pd.DataFrame(columns=RESULT_COLUMNS).astype({"Is_Exclusive": bool})
    eps = values[values > 0].min()
    cutoff = dens.quantile(quantile)
    rows = []
    for name, scores in dens.iterrows():
        ranked = scores.sort_values(ascending=False, kind="stable")
        tissue, runner_up = ranked.index[0], ranked.index[1]
        top, second = ranked.iloc[0], ranked.iloc[1]
        if top <= 0 or top < cutoff[tissue]:
            continue
        exclusive = second == 0
        fold = np.inf if exclusive else fold_change(top, second, eps)
        if fold < min_fold:
            continue
        rows.append({
            "Tissue": tissue,
            "Complex_ID_in_Model": name,
            "Density_Score": top,
            "Second_Best_Tissue": runner_up,
            "Second_Best_Score": second,
            "Is_Exclusive": exclusive,
            "Fold_Change": fold if exclusive else round(fold, 2),
        })
    table = pd.DataFrame(rows, columns=RESULT_COLUMNS).astype({"Is_Exclusive": bool})
    return table.sort_values(["Tissue", "Fold_Change"], ascending=[True, False]).reset_index(drop=True)


def with_corum_columns(table, corum):
    index = [int(name.rsplit("_", 1)[-1]) for name in table["Complex_ID_in_Model"]]
    extra = corum.loc[index].reset_index(drop=True)
    return pd.concat([table, extra], axis=1)


def write_table(table, path):
    out = table.copy()
    out["Fold_Change"] = out["Fold_Change"].astype(object)
    out.loc[out["Is_Exclusive"], "Fold_Change"] = EXCLUSIVE
    out.to_csv(path, index=False)


def main():
    ap = argparse.ArgumentParser(description="Tissue specific CORUM complexes from tissue level networks")
    ap.add_argument("--networks", required=True, help="tissue_networks.npy, shape (tissues, genes, genes)")
    ap.add_argument("--names", required=True, help="tissue names, one per line, in the order of the networks")
    ap.add_argument("--genes", required=True, help="genes.txt of the data folder")
    ap.add_argument("--corum", required=True, help="CORUM complexes, tab separated")
    ap.add_argument("--out", required=True)
    ap.add_argument("--percentile", type=float, default=95)
    ap.add_argument("--quantile", type=float, default=0.90)
    ap.add_argument("--min-fold", type=float, default=1.0)
    ap.add_argument("--unique-genes", action="store_true", help="count subunits listed twice once")
    args = ap.parse_args()

    networks = np.load(args.networks, mmap_mode="r")
    names = [line.strip() for line in open(args.names) if line.strip()]
    genes = [line.strip() for line in open(args.genes) if line.strip()]
    corum = read_corum(args.corum)

    complexes = complexes_in_network(corum, genes, unique=args.unique_genes)
    print(f"{len(complexes)} complexes with at least 3 genes in the network")
    dens = density_matrix(networks, names, complexes, args.percentile)

    os.makedirs(args.out, exist_ok=True)
    dens.to_csv(os.path.join(args.out, "complex_tissue_density_matrix.csv"))
    table = specific_complexes(dens, args.quantile, args.min_fold)
    print(f"{len(table)} tissue specific complexes")
    write_table(with_corum_columns(table, corum), os.path.join(args.out, "Final_Rigorously_Tissue_Specific_Complexes.csv"))


if __name__ == "__main__":
    main()
