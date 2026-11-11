import os
import sys

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

from gid import tcga
from gid.data import load_omics
from gid.preprocess import string_scores
from tests.synthetic import gene_names, write_raw


def write_tcga(folder, seed=3, n_samples=40, n_genes=40):
    rng = np.random.default_rng(seed)
    os.makedirs(folder, exist_ok=True)
    genes = gene_names(n_genes)
    sids = [f"TCGA-{i:04d}" for i in range(n_samples)]
    scale = rng.uniform(0.2, 3.0, size=n_genes)

    fpkm = pd.DataFrame(rng.normal(size=(n_samples, n_genes)) * scale + 5, index=sids, columns=genes)
    fpkm.iloc[rng.random(fpkm.shape) < 0.04] = np.nan
    fpkm = fpkm.drop(index=sids[-2])
    gistic = pd.DataFrame(rng.choice([-1.0, 0.0, 1.0], size=(n_samples, n_genes), p=[0.1, 0.8, 0.1]),
                          index=sids, columns=genes)
    gistic.iloc[:, : n_genes // 3] = 0.0
    gistic.iloc[rng.random(gistic.shape) < 0.03] = np.nan
    gistic = gistic.drop(index=sids[3])
    classes = pd.DataFrame({"class": np.array(["Basal", "Her2", "LumA", "LumB"])[np.arange(n_samples) % 4]},
                           index=pd.Index(sids, name="Sample_ID"))
    fpkm.to_csv(os.path.join(folder, "fpkm_data.csv"))
    gistic.to_csv(os.path.join(folder, "gistic_data.csv"))
    classes.iloc[:-1].to_csv(os.path.join(folder, "sample_classes.csv"))
    return fpkm, gistic, classes


def reference_steps(fpkm, gistic, classes, links, info):
    """The preprocessing written out with SimpleImputer and MinMaxScaler."""
    mrna_mask = fpkm.isna()
    imputed = pd.DataFrame(SimpleImputer(strategy="mean").fit_transform(fpkm), columns=fpkm.columns, index=fpkm.index)
    thr = imputed.var().quantile(0.50)
    imputed = imputed.loc[:, imputed.var() >= thr]
    mrna_mask = mrna_mask[imputed.columns]
    mrna = pd.DataFrame(MinMaxScaler().fit_transform(imputed), columns=imputed.columns, index=imputed.index)

    cnv = gistic.apply(pd.to_numeric, errors="coerce")
    cnv_mask = cnv.isna()
    thr = cnv.var().quantile(0.50)
    cnv = cnv.loc[:, cnv.var() >= thr]
    cnv_mask = cnv_mask[cnv.columns]

    names = pd.read_csv(info, sep="\t")["preferred_name"]
    common = sorted(set(mrna.columns) & set(cnv.columns) & set(names))
    counts = pd.Series(dtype=int)
    for df in [mrna.reindex(columns=common), cnv.reindex(columns=common)]:
        counts = counts.add(pd.Series(1, index=df.index), fill_value=0)
    selected = counts[counts >= 2].index.intersection(classes.index)

    out, masks = [], []
    for table, mask in [(mrna, mrna_mask), (cnv, cnv_mask)]:
        m = table.reindex(index=selected, columns=common)
        mk = mask.reindex(columns=common, fill_value=False).reindex(index=selected, columns=common, fill_value=False).astype(bool)
        mk = mk | m.isna()
        m = m.apply(lambda col: col.fillna(col.mean()), axis=0)
        mk = mk | m.isna()
        out.append(m.fillna(0).values)
        masks.append(mk.values)
    return common, list(selected), np.stack(out, axis=1), np.stack(masks, axis=1), string_scores(links, info, common)


def test_assemble_follows_the_reference_steps(tmp_path):
    write_raw(str(tmp_path / "string"))
    fpkm, gistic, classes = write_tcga(str(tmp_path / "tcga"))
    data = tcga.assemble(str(tmp_path / "tcga"), str(tmp_path / "string"))
    links = str(tmp_path / "string" / "9606.protein.links.v12.0.txt")
    info = str(tmp_path / "string" / "9606.protein.info.v12.0.txt")
    genes, samples, values, mask, adj = reference_steps(fpkm, gistic, classes.iloc[:-1], links, info)

    assert data.genes == genes
    assert list(data.meta.index) == samples
    assert data.layers == ["mrna", "cnv"]
    assert np.allclose(data.values, values)
    assert (data.mask == mask).all()
    assert np.allclose(data.adj, adj)
    assert list(data.meta["tissue"]) == list(classes.loc[samples, "class"])


def test_copy_number_stays_as_called_and_mrna_is_rescaled(tmp_path):
    write_raw(str(tmp_path / "string"))
    write_tcga(str(tmp_path / "tcga"))
    data = tcga.assemble(str(tmp_path / "tcga"), str(tmp_path / "string"))
    observed = data.values[:, 1, :][~data.mask[:, 1, :]]
    assert set(np.unique(observed)) <= {-1.0, 0.0, 1.0}
    assert data.values[:, 0, :].min() >= -1e-12 and data.values[:, 0, :].max() <= 1 + 1e-12
    assert data.meta.index.is_unique
    assert not {"TCGA-0003", "TCGA-0038", "TCGA-0039"} & set(data.meta.index)


def test_command_line_run(tmp_path, monkeypatch):
    write_raw(str(tmp_path / "string"))
    write_tcga(str(tmp_path / "tcga"))
    monkeypatch.setattr(sys, "argv", ["tcga", "--raw", str(tmp_path / "tcga"), "--string", str(tmp_path / "string"),
                                      "--out", str(tmp_path / "data")])
    tcga.main()
    back = load_omics(str(tmp_path / "data"))
    assert back.layers == ["mrna", "cnv"]
    assert back.values.shape[1] == 2
