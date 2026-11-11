import argparse
import os
from itertools import combinations

import numpy as np
import pandas as pd

from .data import load_omics

DELTA = 0.1
STRING_THRESHOLD = 0.7


def minmax01(a):
    lo, hi = a.min(), a.max()
    if hi - lo < 1e-10:
        return np.zeros_like(a, dtype=np.float32)
    return ((a - lo) / (hi - lo)).astype(np.float32)


def core_genes(members, upper_to_name):
    if pd.isna(members) or str(members).strip() == "":
        return []
    names = [g.strip().upper() for g in str(members).split(";") if g.strip()]
    return [upper_to_name[g] for g in names if g in upper_to_name]


def extend(core, adj, genes, upper_to_name, n_target):
    """Add the neighbours with the highest STRING score to any core member until there are n_target genes."""
    position = {g: i for i, g in enumerate(genes)}
    inside = set(core)
    nodes = list(dict.fromkeys(core))
    candidates = {}
    for g in core:
        for j, score in enumerate(adj[position[g]]):
            neighbour = upper_to_name.get(genes[j].strip().upper(), genes[j])
            if neighbour in inside:
                continue
            candidates[neighbour] = max(candidates.get(neighbour, 0.0), float(score))
    for gene, _ in sorted(candidates.items(), key=lambda kv: kv[1], reverse=True):
        if len(nodes) >= n_target:
            break
        nodes.append(gene)
    return nodes


def colour(delta, cutoff, home=True):
    up, down = ("Red", "Blue") if home else ("Blue", "Red")
    if delta > cutoff:
        return up
    if delta < -cutoff:
        return down
    return "Gray"


class TissueMeans:
    """Pan-tissue mean network and mean of the other tissues, as float32 like the tissue networks."""

    def __init__(self, networks, names):
        self.networks = networks
        self.names = list(names)
        n = len(self.names)
        self.pan = np.zeros(networks[0].shape, dtype=np.float32)
        for k in range(n):
            self.pan += np.array(networks[k], dtype=np.float32) / n

    def home_other(self, tissue):
        home = np.array(self.networks[self.names.index(tissue)], dtype=np.float32)
        total = np.zeros_like(home)
        others = 0
        for k, name in enumerate(self.names):
            if name != tissue:
                total += np.array(self.networks[k], dtype=np.float32)
                others += 1
        return home, total / max(others, 1)


def complex_networks(complex_name, tissue, members, means, adj, genes, min_nodes=9,
                     delta=DELTA, string_threshold=STRING_THRESHOLD):
    """Edge and node tables of one complex in its home tissue and in the mean of the other tissues."""
    position = {g: i for i, g in enumerate(genes)}
    upper_to_name = {g.strip().upper(): g for g in genes}
    core = core_genes(members, upper_to_name)
    if not core:
        return None
    nodes = list(dict.fromkeys(core)) if len(set(core)) >= min_nodes else extend(core, adj, genes, upper_to_name, min_nodes)
    core_set = set(core)

    home_raw, other_raw = means.home_other(tissue)
    home_norm, other_norm = minmax01(home_raw), minmax01(other_raw)

    home_rows, other_rows = [], []
    for g1, g2 in combinations(nodes, 2):
        i, j = sorted((position[g1], position[g2]))
        h, o, pc = float(home_raw[i, j]), float(other_raw[i, j]), float(means.pan[i, j])
        string = float(adj[i, j])
        diff = h - o
        base = {
            "Source": g1,
            "Target": g2,
            "LIONESS_HomeTissue_Raw": round(h, 4),
            "LIONESS_OtherMean_Raw": round(o, 4),
            "LIONESS_PanCancer_Raw": round(pc, 4),
            "Delta_Home_vs_Other": round(diff, 4),
            "Delta_Home_vs_PanCancer": round(h - pc, 4),
            "LIONESS_HomeTissue_Norm": round(float(home_norm[i, j]), 4),
            "LIONESS_OtherMean_Norm": round(float(other_norm[i, j]), 4),
            "STRING_Score": round(string, 4),
            "Known_Interaction": int(string >= string_threshold),
            "Is_Core_Edge": int(g1 in core_set and g2 in core_set),
            "Complex_Name": complex_name,
            "Home_Tissue": tissue,
        }
        home_rows.append({**base, "Edge_Weight": round(h, 4), "Edge_Color_Cytoscape": colour(diff, delta)})
        other_rows.append({**base, "Edge_Weight": round(o, 4), "Edge_Color_Cytoscape": colour(diff, delta, home=False)})

    node_table = pd.DataFrame([{
        "Node": g,
        "Is_Core": int(g in core_set),
        "Node_Type": "Core_Complex" if g in core_set else "STRING_Extended",
        "Complex_Name": complex_name,
        "Tissue": tissue,
    } for g in nodes])
    return pd.DataFrame(home_rows), pd.DataFrame(other_rows), node_table


def file_stem(complex_name, tissue):
    safe = complex_name.replace("/", "_").replace(" ", "_").replace(",", "").replace("(", "").replace(")", "")[:50]
    return f"{safe}_{tissue.replace(' ', '_').replace('/', '_')}"


def export(table, complexes, networks, names, adj, genes, out_dir, min_nodes=9, delta=DELTA,
           string_threshold=STRING_THRESHOLD):
    os.makedirs(out_dir, exist_ok=True)
    means = TissueMeans(networks, names)
    by_lower = {n.lower(): n for n in names}
    chosen = table[table["complex_name"].isin(complexes)].reset_index(drop=True)
    if chosen.empty:
        raise ValueError("none of the complexes is in the table")
    all_home, all_other, all_nodes = [], [], []
    for _, row in chosen.iterrows():
        tissue = by_lower.get(str(row["Tissue"]).strip().lower())
        if tissue is None:
            print(f"{row['complex_name']}: tissue {row['Tissue']} has no network, skipped")
            continue
        result = complex_networks(str(row["complex_name"]).strip(), tissue, row.get("subunits_gene_name", ""),
                                  means, adj, genes, min_nodes, delta, string_threshold)
        if result is None:
            print(f"{row['complex_name']}: no subunit in the network, skipped")
            continue
        home, other, nodes = result
        stem = os.path.join(out_dir, file_stem(row["complex_name"], tissue))
        home.to_csv(f"{stem}_HOME_edges.csv", index=False)
        nodes.to_csv(f"{stem}_HOME_nodes.csv", index=False)
        other.to_csv(f"{stem}_OTHER_MEAN_edges.csv", index=False)
        nodes.to_csv(f"{stem}_OTHER_MEAN_nodes.csv", index=False)
        counts = home["Edge_Color_Cytoscape"].value_counts()
        print(f"{row['complex_name']} ({tissue}): {len(nodes)} genes, {len(home)} edges, "
              f"{counts.get('Red', 0)} stronger and {counts.get('Blue', 0)} weaker than in the other tissues")
        all_home.append(home)
        all_other.append(other)
        all_nodes.append(nodes)
    if all_home:
        pd.concat(all_home).to_csv(os.path.join(out_dir, "ALL_HOME_edges.csv"), index=False)
        pd.concat(all_other).to_csv(os.path.join(out_dir, "ALL_OTHER_MEAN_edges.csv"), index=False)
        pd.concat(all_nodes).drop_duplicates("Node").reset_index(drop=True).to_csv(
            os.path.join(out_dir, "ALL_nodes.csv"), index=False)


def main():
    ap = argparse.ArgumentParser(description="Node and edge tables of tissue specific complexes for Cytoscape")
    ap.add_argument("--data", required=True, help="folder written by gid.preprocess")
    ap.add_argument("--networks", required=True, help="tissue_networks.npy")
    ap.add_argument("--names", required=True, help="tissue_names.txt")
    ap.add_argument("--table", required=True, help="Final_Rigorously_Tissue_Specific_Complexes.csv")
    ap.add_argument("--complexes", nargs="+", required=True, help="complex names as in the table")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-nodes", type=int, default=9)
    ap.add_argument("--delta", type=float, default=DELTA)
    ap.add_argument("--string-threshold", type=float, default=STRING_THRESHOLD)
    args = ap.parse_args()

    data = load_omics(args.data)
    names = [line.strip() for line in open(args.names) if line.strip()]
    networks = np.load(args.networks, mmap_mode="r")
    table = pd.read_csv(args.table, low_memory=False)
    export(table, args.complexes, networks, names, data.adj, data.genes, args.out,
           args.min_nodes, args.delta, args.string_threshold)


if __name__ == "__main__":
    main()
