import argparse
import os

import numpy as np
import pandas as pd

from .data import load_omics

NO_VALUE = ["0", "0.0", "-1", "-1.0", "nan", "none", "false", "no data", "no", ""]


def read_targets(path):
    """Open Targets associated targets of a disease, with the columns used to annotate the genes."""
    table = pd.read_csv(path, sep="\t")
    for col in ["globalScore", "gwasCredibleSets", "hasSmallMoleculeBinder", "maxClinicalStage"]:
        table[col] = pd.to_numeric(table[col], errors="coerce").fillna(0)
    if "isCancerDriverGene" in table.columns:
        table["isCancerDriver"] = table["isCancerDriverGene"].apply(
            lambda v: str(v).strip().lower() not in ["0", "0.0", "nan", "none", "false", "no data", "no", ""])
    else:
        table["isCancerDriver"] = False
    if "symbol" not in table.columns:
        table = table.rename(columns={"targetSymbol": "symbol"})
    table["symbol"] = table["symbol"].astype(str).str.strip().str.upper()
    return table


def seed_genes(targets, genes, min_gwas=0.5):
    """Genes of the network with a GWAS credible set association score above min_gwas."""
    upper = {g.strip().upper(): g for g in genes}
    confident = set(targets.loc[targets["gwasCredibleSets"] > min_gwas, "symbol"])
    return [upper[g] for g in confident if g in upper]


def normalise(network):
    lo, hi = np.min(network), np.max(network)
    return (network - lo) / (hi - lo)


def neighbour_scores(norm, seed_idx, threshold=0.7):
    """Highest normalised association to any seed, for every non-seed gene that reaches the threshold."""
    seed_set = set(seed_idx)
    best = {}
    for s in seed_idx:
        for t in np.flatnonzero(norm[s] >= threshold):
            if t not in seed_set and (t not in best or norm[s, t] > best[t]):
                best[t] = norm[s, t]
    return best


def first_degree_edges(network, genes, seeds, threshold=0.7, top_n=30):
    """Seeds, their top_n strongest neighbours, and every edge among them above the threshold."""
    norm = normalise(np.asarray(network, dtype=np.float64))
    position = {g: i for i, g in enumerate(genes)}
    seed_idx = [position[g] for g in seeds]
    seed_set = set(seed_idx)

    best = neighbour_scores(norm, seed_idx, threshold)
    neighbours = [t for t, _ in sorted(best.items(), key=lambda kv: kv[1], reverse=True)[:top_n]]
    allowed = list(dict.fromkeys(seed_idx + neighbours))

    rows = []
    for a in range(len(allowed)):
        for b in range(a + 1, len(allowed)):
            i, j = sorted((allowed[a], allowed[b]))
            weight = norm[i, j]
            if weight >= threshold:
                n_seed = (i in seed_set) + (j in seed_set)
                kind = ["Neighbor-Neighbor", "Seed-Neighbor", "Seed-Seed"][n_seed]
                rows.append({"Source": genes[i], "Target": genes[j], "Weight": weight, "Edge_Type": kind})
    edges = pd.DataFrame(rows, columns=["Source", "Target", "Weight", "Edge_Type"])
    return edges.sort_values(by="Weight", ascending=False).reset_index(drop=True), allowed, seed_idx, neighbours


def node_group(node, seeds, targets):
    clean = node.strip().upper()
    match = targets[targets["symbol"] == clean]
    if clean in {s.strip().upper() for s in seeds}:
        return "Seed Gene (GWAS > 0.5)"
    if match.empty:
        return "No Data"
    druggable = match["hasSmallMoleculeBinder"].max() == 1 or match["maxClinicalStage"].max() > 0
    gwas, overall = match["gwasCredibleSets"].max(), match["globalScore"].max()
    clingen = str(match["clingen"].values[0]).strip().lower() if "clingen" in match.columns else "no data"
    tissue = str(match["tissueSpecificity"].values[0]).strip().lower() if "tissueSpecificity" in match.columns else "no data"
    if match["isCancerDriver"].max():
        return "Cancer Driver Gene"
    if gwas > 0:
        return "Associated Gene (GWAS <= 0.5)"
    if druggable:
        return "Druggable Target"
    if overall >= 0.3:
        return "High Global Score (>= 0.3)"
    if clingen not in NO_VALUE:
        return "clingen"
    if tissue not in NO_VALUE:
        return "tissueSpecificity"
    if overall > 0:
        return "Low Global Score (< 0.3)"
    return "No Data"


def main():
    ap = argparse.ArgumentParser(description="First degree network of the Open Targets seed genes in one tissue network")
    ap.add_argument("--data", required=True, help="folder written by gid.preprocess")
    ap.add_argument("--networks", required=True, help="tissue_networks.npy")
    ap.add_argument("--names", required=True, help="tissue_names.txt")
    ap.add_argument("--targets", required=True, help="Open Targets associated targets (tsv)")
    ap.add_argument("--tissue", default="Haematopoietic and Lymphoid")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prefix", default="Luk_lioness_1st_degree")
    ap.add_argument("--threshold", type=float, default=0.7)
    ap.add_argument("--top-n", type=int, default=30)
    args = ap.parse_args()

    genes = load_omics(args.data).genes
    names = [line.strip() for line in open(args.names) if line.strip()]
    network = np.load(args.networks, mmap_mode="r")[names.index(args.tissue)]
    targets = read_targets(args.targets)
    seeds = seed_genes(targets, genes)
    print(f"{len(seeds)} seed genes")

    edges, allowed, seed_idx, neighbours = first_degree_edges(network, genes, seeds, args.threshold, args.top_n)
    print(f"{len(seed_idx)} seeds and {len(neighbours)} neighbours, {len(allowed)} nodes in the sub-network")
    everyone = neighbour_scores(normalise(np.asarray(network, dtype=np.float64)), seed_idx, args.threshold)
    first_degree = sorted(genes[t] for t in everyone)
    print(f"{len(first_degree)} genes in the complete first degree neighbourhood")
    nodes = sorted(set(edges["Source"]) | set(edges["Target"]))
    node_table = pd.DataFrame({"Node": nodes, "Group": [node_group(n, seeds, targets) for n in nodes]})
    os.makedirs(args.out, exist_ok=True)
    edges.to_csv(os.path.join(args.out, f"{args.prefix}_edges.csv"), index=False)
    node_table.to_csv(os.path.join(args.out, f"{args.prefix}_nodes.csv"), index=False)
    with open(os.path.join(args.out, f"{args.prefix}_first_degree_genes.txt"), "w") as fh:
        fh.write("\n".join(first_degree) + "\n")
    print(f"{len(edges)} edges and {len(node_table)} nodes")


if __name__ == "__main__":
    main()
