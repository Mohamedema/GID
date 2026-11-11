import argparse
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from .data import OmicsData

FILES = dict(
    protein="mmc3.xlsx",
    protein_ids="idmapping_2024_07_29.tsv",
    models="model_list_20230517.csv",
    mrna="mRNA.csv",
    crispr="Project_score_combined_Sanger_v2_Broad_21Q2_fitness_scores_fold_change_values_20240111.tsv",
    cnv="gene_cnv_2018-11-13_1545.csv",
    methylation="methylation.csv",
    string_links="9606.protein.links.v12.0.txt",
    string_info="9606.protein.info.v12.0.txt",
)

LOW_VARIANCE = 0.20


def drop_low_variance(df, q=LOW_VARIANCE):
    var = df.var()
    return df.loc[:, var >= var.quantile(q)]


def minmax(df):
    return pd.DataFrame(MinMaxScaler().fit_transform(df), index=df.index, columns=df.columns)


def read_protein(path, id_path):
    """Proteomics table: mean imputation, 20% lowest variance proteins dropped, min-max scaling.

    The table is imputed before the missing values are recorded, so the mask returned here is
    empty; cell lines that are absent from the proteomics table are masked later, in assemble().
    """
    raw = pd.read_excel(path, sheet_name=1, skiprows=1, index_col=0)
    imputed = raw.fillna(raw.mean())
    imputed = drop_low_variance(imputed)
    scaled = minmax(imputed)
    mask = imputed.isna()
    scaled.columns = [c.split(";")[1] for c in scaled.columns]
    scaled.index = [r.split(";")[0] for r in scaled.index]
    mask.columns, mask.index = scaled.columns, scaled.index

    ids = pd.read_csv(id_path, sep="\t", header=None)
    ids.columns = ["From", "To"]
    to_symbol = dict(zip(ids["From"], ids["To"]))
    keep = [c for c in scaled.columns if c in to_symbol]
    scaled = scaled[keep].rename(columns=to_symbol)
    mask = mask[keep].rename(columns=to_symbol)
    scaled = scaled.loc[:, ~scaled.columns.duplicated()]
    mask = mask.loc[:, ~mask.columns.duplicated()]
    return scaled, mask


def read_mrna(path):
    x = pd.read_csv(path, index_col=0).T
    missing = x.isna()
    x = drop_low_variance(x)
    return minmax(x), missing


def read_crispr(path):
    """Project Score fold changes in the layout of the Cell Model Passports download."""
    c = pd.read_table(path, index_col=0)
    c = c.T
    c = c.drop(["source", "qc_pass", "symbol"], axis=1)
    c = c.T
    c.columns = c.iloc[0]
    c = c.drop(c.index[0])
    c = c.T
    c = c[~c.index.duplicated(keep="first")]
    c = c.apply(pd.to_numeric, errors="coerce")
    c = c.dropna(axis=1, how="all")
    return drop_low_variance(c)


def read_cnv(path):
    c = pd.read_csv(path)
    c = c.pivot_table(index="model_id", columns="gene_symbol", values="gistic_score", aggfunc="first")
    return drop_low_variance(c)


def read_methylation(path):
    return drop_low_variance(pd.read_csv(path, index_col=0).T)


def dense_scores(first, second, score, genes):
    """Symmetric gene by gene matrix of pair scores, rescaled to [0, 1].

    The rescaling uses the lowest and the highest score among the pairs of these genes.
    """
    pos = {g: i for i, g in enumerate(genes)}
    keep = (first.isin(pos) & second.isin(pos)).to_numpy()
    first, second, score = first[keep], second[keep], score[keep]
    lo, hi = score.min(), score.max()
    scaled = (score - lo) / (hi - lo)
    adj = np.zeros((len(genes), len(genes)))
    np.maximum.at(adj, (first.map(pos).to_numpy(), second.map(pos).to_numpy()), scaled.to_numpy())
    adj = np.maximum(adj, adj.T)
    np.fill_diagonal(adj, 0)
    return adj


def string_scores(links_path, info_path, genes):
    """Dense matrix of STRING combined scores between the genes, rescaled to [0, 1]."""
    links = pd.read_csv(links_path, sep=" ")
    info = pd.read_csv(info_path, sep="\t")
    names = dict(zip(info["#string_protein_id"], info["preferred_name"]))
    return dense_scores(links["protein1"].map(names), links["protein2"].map(names), links["combined_score"], genes)


def read_prior(path):
    """Edge list of another prior network (HIPPIE in the paper): tab separated, columns gene_a, gene_b and score."""
    table = pd.read_csv(path, sep="\t")
    missing = {"gene_a", "gene_b", "score"} - set(table.columns)
    if missing:
        raise ValueError(f"{path} lacks the columns {sorted(missing)}")
    return table


def string_gene_names(info_path):
    info = pd.read_csv(info_path, sep="\t")
    return set(info["preferred_name"])


def assemble(raw_dir, files=None, min_layers=2, prior=None):
    """Five-layer input. prior is the path of an edge list that replaces STRING as the prior network."""
    files = {**FILES, **(files or {})}

    def path(key):
        return os.path.join(raw_dir, files[key])

    protein, protein_mask = read_protein(path("protein"), path("protein_ids"))
    mrna, mrna_mask = read_mrna(path("mrna"))
    crispr = read_crispr(path("crispr"))
    cnv = read_cnv(path("cnv"))
    meth = read_methylation(path("methylation"))
    models = pd.read_csv(path("models")).set_index("model_id")

    # genes measured in every layer
    shared_idx = mrna.index.intersection(protein.index)
    genes = mrna.columns.intersection(protein.columns)
    genes = genes.intersection(crispr.columns)
    genes = cnv.columns.intersection(genes)
    genes = meth.columns.intersection(genes)
    protein = protein.loc[shared_idx, genes]
    protein_mask = protein_mask.loc[shared_idx, genes]
    genes = protein.columns

    crispr = crispr.loc[:, genes]
    crispr_mask = crispr.isna()
    crispr = minmax(crispr)
    cnv = cnv.loc[:, genes]
    cnv_mask = cnv.isna()
    meth = meth.loc[:, genes]
    meth = minmax(meth)
    meth_mask = meth.isna()
    mrna = mrna.loc[:, genes]
    mrna_mask = mrna_mask.loc[:, genes]

    layers = [mrna, protein, crispr, meth, cnv]
    layer_masks = [mrna_mask, protein_mask, crispr_mask, meth_mask, cnv_mask]

    if prior:
        prior_edges = read_prior(prior)
        genes = sorted(set(genes) & set(prior_edges["gene_a"]) & set(prior_edges["gene_b"]))
    else:
        genes = sorted(set(genes) & string_gene_names(path("string_info")))
    n_present = pd.Series(0, index=sorted(set().union(*[d.index for d in layers])))
    for d in layers:
        n_present.loc[d.index] += 1
    samples = [s for s in n_present.index[n_present >= min_layers] if s in models.index]

    values, masks = [], []
    for d, m in zip(layers, layer_masks):
        d = d.reindex(index=samples, columns=genes)
        m = m.reindex(index=samples, columns=genes, fill_value=False).astype(bool)
        m = m | d.isna()
        d = d.fillna(d.mean())
        m = m | d.isna()
        values.append(d.fillna(0).to_numpy())
        masks.append(m.to_numpy())

    if prior:
        adj = dense_scores(prior_edges["gene_a"], prior_edges["gene_b"], prior_edges["score"], genes)
    else:
        adj = string_scores(path("string_links"), path("string_info"), genes)
    return OmicsData(np.stack(values, axis=1), np.stack(masks, axis=1), genes, models.reindex(samples), adj)


def main():
    ap = argparse.ArgumentParser(description="Build the five-layer input from the raw DepMap, Project Score and STRING files")
    ap.add_argument("--raw", required=True, help="folder with the raw files (names as in gid.preprocess.FILES)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prior", default=None, help="edge list (gene_a, gene_b, score) used instead of STRING")
    args = ap.parse_args()
    data = assemble(args.raw, prior=args.prior)
    data.save(args.out)
    print(f"{data.n_samples} samples, {data.n_genes} genes, layers: {', '.join(data.layers)}")
    for k, name in enumerate(data.layers):
        print(f"  {name}: {int((~data.mask[:, k, :].all(axis=1)).sum())} samples with data")


if __name__ == "__main__":
    main()
