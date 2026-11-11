import argparse
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from .data import OmicsData
from .preprocess import minmax, string_gene_names, string_scores

FILES = dict(
    mrna="fpkm_data.csv",
    cnv="gistic_data.csv",
    classes="sample_classes.csv",
    string_links="9606.protein.links.v12.0.txt",
    string_info="9606.protein.info.v12.0.txt",
)

LOW_VARIANCE = 0.50


def drop_low_variance(df, q=LOW_VARIANCE):
    var = df.var()
    return df.loc[:, var >= var.quantile(q)]


def read_mrna(path):
    raw = pd.read_csv(path, index_col=0)
    missing = raw.isna()
    imputed = raw.fillna(raw.mean())
    kept = drop_low_variance(imputed)
    return minmax(kept), missing[kept.columns]


def read_cnv(path):
    """GISTIC calls, kept as called (-1, 0, 1)."""
    raw = pd.read_csv(path, index_col=0).apply(pd.to_numeric, errors="coerce")
    missing = raw.isna()
    kept = drop_low_variance(raw)
    return kept, missing[kept.columns]


def assemble(raw_dir, string_dir=None, files=None):
    """Two-layer input (mRNA and copy number) of the TCGA-BRCA tumours, subtype in the 'tissue' column."""
    files = {**FILES, **(files or {})}
    string_dir = string_dir or raw_dir

    def path(key, folder=raw_dir):
        return os.path.join(folder, files[key])

    mrna, mrna_missing = read_mrna(path("mrna"))
    cnv, cnv_missing = read_cnv(path("cnv"))
    meta = pd.read_csv(path("classes"), index_col="Sample_ID")
    meta["class_encoded"] = LabelEncoder().fit_transform(meta["class"])

    genes = sorted(set(mrna.columns) & set(cnv.columns) & string_gene_names(path("string_info", string_dir)))
    samples = sorted(set(mrna.index) & set(cnv.index) & set(meta.index))

    values, masks = [], []
    for table, missing in [(mrna, mrna_missing), (cnv, cnv_missing)]:
        d = table.reindex(index=samples, columns=genes)
        m = missing.reindex(index=samples, columns=genes, fill_value=False).astype(bool) | d.isna()
        d = d.fillna(d.mean())
        m = m | d.isna()
        values.append(d.fillna(0).to_numpy())
        masks.append(m.to_numpy())

    adj = string_scores(path("string_links", string_dir), path("string_info", string_dir), genes)
    meta = meta.reindex(samples)
    meta["tissue"] = meta["class"]
    return OmicsData(np.stack(values, axis=1), np.stack(masks, axis=1), genes, meta, adj, ["mrna", "cnv"])


def main():
    ap = argparse.ArgumentParser(description="Two-layer input from the TCGA-BRCA mRNA and copy number tables")
    ap.add_argument("--raw", required=True, help="folder with fpkm_data.csv, gistic_data.csv and sample_classes.csv")
    ap.add_argument("--string", default=None, help="folder with the STRING files (default: --raw)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    data = assemble(args.raw, args.string)
    data.save(args.out)
    print(f"{data.n_samples} tumours, {data.n_genes} genes")
    print(data.meta["class"].value_counts().to_string())


if __name__ == "__main__":
    main()
