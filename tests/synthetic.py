"""Small random data sets in the formats of the raw files, used by the tests."""
import os

import numpy as np
import pandas as pd

from gid.data import OmicsData


def gene_names(n):
    return [f"GENE{i:03d}" for i in range(1, n + 1)]


def random_omics(n_genes=24, n_samples=18, seed=0, n_tissues=3):
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(n_samples, 5, n_genes))
    values[:, 4, :] = rng.integers(-2, 3, size=(n_samples, n_genes))
    mask = rng.random(size=values.shape) < 0.1
    mask[:3, 1, :] = True
    adj = rng.random((n_genes, n_genes))
    adj = np.triu(adj, 1)
    adj = adj + adj.T
    adj[adj < 0.6] = 0
    tissues = np.array(["Lung", "Breast", "Skin"])[np.arange(n_samples) % n_tissues]
    meta = pd.DataFrame({"tissue": tissues}, index=[f"SIDM{i:05d}" for i in range(n_samples)])
    return OmicsData(values, mask, gene_names(n_genes), meta, adj)


def write_raw(folder, seed=1, n_genes=40, n_samples=30):
    """Writes every raw file with the names and layouts that gid.preprocess reads."""
    rng = np.random.default_rng(seed)
    os.makedirs(folder, exist_ok=True)
    genes = gene_names(n_genes)
    sids = [f"SIDM{i:05d}" for i in range(1, n_samples + 1)]
    scale = {g: 1.0 + 0.5 * k for k, g in enumerate(genes)}     # variance grows with the gene index

    def table(rows, cols, nan_rate=0.05):
        x = np.array([[rng.normal() * scale[c] for c in cols] for _ in rows])
        x[rng.random(x.shape) < nan_rate] = np.nan
        return pd.DataFrame(x, index=rows, columns=cols)

    layers = dict(
        protein=(sids[:22], genes[2:38]),
        mrna=(sids[:26], genes[:36]),
        crispr=(sids[8:], genes[4:40]),
        cnv=(sids[3:28], genes[1:39]),
        methylation=(sids[:5] + sids[10:], genes[:37]),
    )

    # proteomics, in the sheet layout of the Excel supplement
    rows, cols = layers["protein"]
    prot = table(rows, cols)
    prot.index = [f"{s};line" for s in prot.index]
    prot.columns = [f"{g};P{int(g[4:]):05d}" for g in prot.columns]
    with pd.ExcelWriter(os.path.join(folder, "mmc3.xlsx")) as xl:
        pd.DataFrame({"a": [1]}).to_excel(xl, sheet_name="readme")
        prot.to_excel(xl, sheet_name="data", startrow=1)
    pd.DataFrame({"From": [f"P{i:05d}" for i in range(1, n_genes + 1)], "To": genes}).to_csv(
        os.path.join(folder, "idmapping_2024_07_29.tsv"), sep="\t", header=False, index=False)

    tissues = np.array(["Lung", "Breast", "Skin", "Liver"])[np.arange(n_samples) % 4]
    pd.DataFrame({"model_id": sids, "tissue": tissues, "cancer_type": "x"}).to_csv(
        os.path.join(folder, "model_list_20230517.csv"), index=False)

    rows, cols = layers["mrna"]
    table(rows, cols).T.to_csv(os.path.join(folder, "mRNA.csv"))

    rows, cols = layers["methylation"]
    table(rows, cols).T.to_csv(os.path.join(folder, "methylation.csv"))

    rows, cols = layers["cnv"]
    alt = rng.random((len(rows), len(cols))) < np.linspace(0.05, 0.8, len(cols))    # more calls for later genes
    calls = alt * rng.choice([-2.0, -1.0, 1.0, 2.0], size=alt.shape)
    calls[rng.random(calls.shape) < 0.03] = np.nan
    cnv = pd.DataFrame(calls, index=rows, columns=cols)
    long = cnv.stack().reset_index()
    long.columns = ["model_id", "gene_symbol", "gistic_score"]
    long.to_csv(os.path.join(folder, "gene_cnv_2018-11-13_1545.csv"), index=False)

    rows, cols = layers["crispr"]
    cr = table(rows, cols)
    top = pd.DataFrame([rows, ["Sanger"] * len(rows), ["True"] * len(rows), ["x"] * len(rows)],
                       index=["model_id", "source", "qc_pass", "symbol"], columns=[f"CL{i}" for i in range(len(rows))])
    body = cr.T
    body.columns = top.columns
    pd.concat([top, body]).to_csv(
        os.path.join(folder, "Project_score_combined_Sanger_v2_Broad_21Q2_fitness_scores_fold_change_values_20240111.tsv"),
        sep="\t")

    # STRING: genes 1..n_genes except the last two are known to STRING
    known = genes[:-2]
    ids = {g: f"9606.ENSP{k:08d}" for k, g in enumerate(known)}
    pd.DataFrame({"#string_protein_id": list(ids.values()), "preferred_name": list(ids.keys())}).to_csv(
        os.path.join(folder, "9606.protein.info.v12.0.txt"), sep="\t", index=False)
    recs = []
    for a in range(len(known)):
        for b in range(a + 1, len(known)):
            if rng.random() < 0.25:
                s = int(rng.integers(150, 1000))
                recs.append((ids[known[a]], ids[known[b]], s))
                recs.append((ids[known[b]], ids[known[a]], s))
    pd.DataFrame(recs, columns=["protein1", "protein2", "combined_score"]).to_csv(
        os.path.join(folder, "9606.protein.links.v12.0.txt"), sep=" ", index=False)
    return dict(layers=layers, genes=genes, sids=sids, scale=scale, string_known=known)
