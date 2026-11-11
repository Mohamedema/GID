import argparse
import os
import re

import numpy as np
import pandas as pd
from scipy.stats import hypergeom

from .data import load_omics, upper_index
from .references import gene_lookup, pairs_mask

HAEMA_KEYWORDS = [
    "leukemia", "leukaemia", "lymphoma", "myeloma", "lymphoid", "myeloid",
    "blood", "b cell", "b-cell", "t cell", "t-cell", "k562", "jurkat",
    "thp-1", "raji", "hl-60", "hl60", "macrophage", "monocyte", "erythroid",
]
TOP_FRACTIONS = (0.01, 0.05, 0.10)
CONTEXT_WORDS = ("cell", "cancer", "tumor", "tumour", "tissue", "disease", "lineage", "context", "sample", "model")


def intact_gene(alias):
    if pd.isna(alias):
        return None
    for pattern in (r"uniprotkb:([^|]+)\(gene name\)",
                    r"uniprotkb:([^|]+)\(gene name synonym\)",
                    r"psi-mi:([^|]+)\(display_short\)"):
        match = re.search(pattern, alias, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return None


def intact_pairs(path, genes, keywords=HAEMA_KEYWORDS):
    """IntAct interactions (PSI-MITAB export) whose host organism field mentions one of the keywords."""
    table = pd.read_csv(path, sep="\t", dtype=str)
    host = "Host organism(s)" if "Host organism(s)" in table.columns else table.columns[14]
    known = {g.upper() for g in genes}
    pairs = set()
    for _, row in table.iterrows():
        if not any(k in str(row[host]).lower() for k in keywords):
            continue
        a = intact_gene(row.get("Alias(es) interactor A", ""))
        b = intact_gene(row.get("Alias(es) interactor B", ""))
        if a and b and a != b and a in known and b in known:
            pairs.add(frozenset([a, b]))
    return pairs


def squash(text):
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def gene_columns(table, genes):
    """The two columns of the table that hold gene symbols of the network."""
    known = {g.upper() for g in genes}
    overlap = {}
    for col in table.columns:
        values = table[col].dropna().astype(str).str.strip().str.upper()
        overlap[col] = float(values.isin(known).mean()) if len(values) else 0.0
    candidates = []
    for c1 in table.columns:
        for c2 in table.columns:
            n1, n2 = squash(c1), squash(c2)
            diff = [k for k in range(min(len(n1), len(n2))) if n1[k] != n2[k]]
            if c1 != c2 and len(n1) == len(n2) and len(diff) == 1 and (n1[diff[0]], n2[diff[0]]) in (("a", "b"), ("1", "2")):
                candidates.append((min(overlap[c1], overlap[c2]), c1, c2))
    if candidates and max(candidates)[0] > 0:
        _, first, second = max(candidates)
    else:
        ranked = sorted(table.columns, key=lambda c: -overlap[c])
        first, second = ranked[0], ranked[1]
    if overlap[first] == 0 or overlap[second] == 0:
        raise ValueError(f"no column holds gene symbols of the network: {list(table.columns)}")
    return first, second


def read_synlethdb(path):
    table = pd.read_csv(path, sep="\t", dtype=str, encoding="utf-8-sig", encoding_errors="replace", on_bad_lines="skip")
    if table.shape[1] < 2:
        table = pd.read_csv(path, sep=None, engine="python", dtype=str, encoding="utf-8-sig", encoding_errors="replace")
    return table


def synlethdb_pairs(path, genes, keywords=None, columns=None, context_columns=None):
    """Human synthetic lethal pairs of SynLethDB with both genes in the network.

    With keywords, only the records whose cell line or cancer type entry contains one of them are kept.
    Returns the pairs and the entries of the context columns that passed the filter.
    """
    table = read_synlethdb(path)
    col_a, col_b = columns or gene_columns(table, genes)
    known = {g.upper() for g in genes}
    a = table[col_a].fillna("").astype(str).str.strip().str.upper()
    b = table[col_b].fillna("").astype(str).str.strip().str.upper()
    keep = a.isin(known) & b.isin(known)
    passed = pd.Series(dtype=str)
    if keywords is not None:
        context_columns = context_columns or [c for c in table.columns if c not in (col_a, col_b)
                                              and any(w in squash(c) for w in CONTEXT_WORDS)]
        if not context_columns:
            raise ValueError(f"no cell line or cancer type column in {list(table.columns)}")
        text = table[context_columns].fillna("").astype(str).apply(" ".join, axis=1).str.lower().str.strip()
        flat = text.str.replace(r"[^a-z0-9]", "", regex=True)
        flat_words = [squash(k) for k in keywords if "cell" not in k and len(squash(k)) >= 4]
        passes = text.apply(lambda t: any(k in t for k in keywords)) | flat.apply(lambda t: any(k in t for k in flat_words))
        keep &= passes
        passed = text[passes]
    pairs = {frozenset([x, y]) for x, y in zip(a[keep], b[keep]) if x != y}
    return pairs, passed


def rank_edges(weights, labels):
    """Edges from the highest to the lowest weight: (order, labels in that order)."""
    order = np.argsort(weights)[::-1]
    return order, labels[order]


def enrichment_table(sorted_labels, fractions=TOP_FRACTIONS):
    total = len(sorted_labels)
    gold = int(sorted_labels.sum())
    rows = []
    for fraction in fractions:
        k = int(total * fraction)
        hits = int(sorted_labels[:k].sum())
        expected = k * gold / total
        fold = hits / expected if expected > 0 else 0
        p = hypergeom.sf(hits - 1, total, gold, k)
        rows.append({
            "top_pct": int(round(fraction * 100)),
            "K": k,
            "hits": hits,
            "gold_pairs": gold,
            "capture_pct": 100 * hits / gold,
            "expected_hits": expected,
            "fold": fold,
            "hypergeom_p": p,
            "fold_label": f"{fold:.1f}-fold enrichment",
            "p_label": "P < 0.001" if p < 0.001 else f"P = {p:.3f}",
        })
    return pd.DataFrame(rows)


def capture_bands(weights, labels, sorted_labels, n_iter=100, seed=42, upto=0.20, points=200):
    """Capture curve with the bootstrap interval of the model and the permutation interval of the null."""
    rs = np.random.RandomState(seed)
    total = len(weights)
    gold = sorted_labels.sum()
    grid = np.linspace(0, upto, points)
    x = np.arange(1, total + 1) / total
    boot = np.zeros((n_iter, points))
    null = np.zeros((n_iter, points))
    for b in range(n_iter):
        idx = rs.randint(0, total, size=total)
        drawn = labels[idx][np.argsort(weights[idx])[::-1]]
        cum = np.cumsum(drawn)
        boot[b] = np.interp(grid, x, cum / cum[-1] if cum[-1] > 0 else np.zeros(total))
        shuffled = np.cumsum(rs.permutation(sorted_labels))
        null[b] = np.interp(grid, x, shuffled / gold if gold > 0 else np.zeros(total))
    observed = np.interp(grid, x, np.cumsum(sorted_labels) / gold)
    curve = pd.DataFrame({
        "Proportion_Edges_Ranked": grid,
        "Model_Capture_Rate": observed,
        "Model_CI_Lower": np.percentile(boot, 2.5, axis=0),
        "Model_CI_Upper": np.percentile(boot, 97.5, axis=0),
        "Null_CI_Lower": np.percentile(null, 2.5, axis=0),
        "Null_CI_Upper": np.percentile(null, 97.5, axis=0),
        "Random_Expectation": grid,
    })
    return curve


def bands_separate_from(curve):
    """Percentage of ranked edges from which the model and null intervals stop overlapping (NaN: never up to 20%)."""
    overlap = np.where(~(curve["Model_CI_Lower"] > curve["Null_CI_Upper"]).to_numpy())[0]
    if len(overlap) == 0:
        return 0.0
    if overlap.max() + 1 < len(curve):
        return 100 * curve["Proportion_Edges_Ranked"].iloc[overlap.max() + 1]
    return np.nan


def gold_table(order, sorted_labels, weights, genes):
    ii, jj = upper_index(len(genes))
    position = np.flatnonzero(sorted_labels == 1)
    edge = order[position]
    total = len(weights)
    out = pd.DataFrame({
        "Gene_A": [genes[ii[e]] for e in edge],
        "Gene_B": [genes[jj[e]] for e in edge],
        "Rank": position + 1,
        "Proportion_Edges_Ranked": (position + 1) / total,
        "Edge_Weight": weights[edge],
    })
    out["In_Top_5pct"] = out["Rank"] <= int(total * 0.05)
    return out


def evaluate(network, genes, pairs, out_dir, suffix="", n_iter=100, seed=42):
    """Rank the edges of a symmetric network, test the enrichment of the gold standard pairs and write the tables."""
    ii, jj = upper_index(len(genes))
    weights = np.asarray(network, dtype=np.float64)[ii, jj]
    labels = pairs_mask([tuple(p) for p in pairs], gene_lookup(genes), len(genes)).astype(np.float64)
    order, sorted_labels = rank_edges(weights, labels)
    stats = enrichment_table(sorted_labels)
    curve = capture_bands(weights, labels, sorted_labels, n_iter, seed)
    stats["bands_separate_from_pct"] = bands_separate_from(curve)
    os.makedirs(out_dir, exist_ok=True)
    curve.to_csv(os.path.join(out_dir, f"gid_enrichment_data_with_null{suffix}.csv"), index=False)
    stats.to_csv(os.path.join(out_dir, f"gid_enrichment_stats{suffix}.csv"), index=False)
    gold_table(order, sorted_labels, weights, genes).to_csv(
        os.path.join(out_dir, f"gid_gold_pairs_ranked{suffix}.csv"), index=False)
    return stats, curve


def main():
    ap = argparse.ArgumentParser(description="Enrichment of IntAct or SynLethDB pairs among the highest scoring edges")
    ap.add_argument("resource", choices=["intact", "synlethdb"])
    ap.add_argument("--data", required=True, help="folder written by gid.preprocess")
    ap.add_argument("--file", required=True, help="Cancer.txt (IntAct) or Human.SL.detailed.tsv (SynLethDB)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--population", default=None, help="population_network.npy; ranked when --tissue is not given")
    ap.add_argument("--networks", default=None, help="tissue_networks.npy")
    ap.add_argument("--names", default=None, help="tissue_names.txt")
    ap.add_argument("--tissue", default=None, help="tissue network to rank")
    ap.add_argument("--all-cells", action="store_true", help="SynLethDB: keep the pairs of every cell line")
    ap.add_argument("--suffix", default="")
    ap.add_argument("--iterations", type=int, default=100)
    args = ap.parse_args()

    genes = load_omics(args.data).genes
    if args.tissue:
        names = [line.strip() for line in open(args.names) if line.strip()]
        network = np.load(args.networks, mmap_mode="r")[names.index(args.tissue)]
    else:
        network = np.load(args.population)

    if args.resource == "intact":
        pairs = intact_pairs(args.file, genes)
    else:
        pairs, _ = synlethdb_pairs(args.file, genes, None if args.all_cells else HAEMA_KEYWORDS)
    print(f"{len(pairs)} gold standard pairs in the network")
    stats, _ = evaluate(network, genes, pairs, args.out, args.suffix, args.iterations)
    if stats["gold_pairs"].iloc[0] != len(pairs):
        print(f"{stats['gold_pairs'].iloc[0]} of them are edges of the ranked network")
    print(stats[["top_pct", "hits", "gold_pairs", "fold", "hypergeom_p"]].to_string(index=False))


if __name__ == "__main__":
    main()
