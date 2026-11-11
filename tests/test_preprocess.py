import numpy as np
import pandas as pd
import pytest

from gid.preprocess import assemble, drop_low_variance, read_crispr, read_prior
from tests.synthetic import write_raw


def test_drop_low_variance_keeps_upper_80_percent():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(50, 20)) * np.arange(1, 21))
    kept = drop_low_variance(df)
    var = df.var()
    assert set(kept.columns) == set(var[var >= var.quantile(0.2)].index)
    assert 15 <= kept.shape[1] <= 17


def test_crispr_reader_gives_samples_by_genes(tmp_path):
    info = write_raw(tmp_path)
    name = "Project_score_combined_Sanger_v2_Broad_21Q2_fitness_scores_fold_change_values_20240111.tsv"
    c = read_crispr(tmp_path / name)
    assert set(c.index) == set(info["sids"][8:])
    assert c.columns.isin(info["genes"]).all()


def test_assemble_gene_and_sample_selection(tmp_path):
    info = write_raw(tmp_path)
    data = assemble(str(tmp_path))
    assert data.values.shape == (data.n_samples, 5, data.n_genes)
    assert data.mask.shape == data.values.shape
    assert data.genes == sorted(data.genes)
    # no gene outside STRING, and every gene is present in all five layers
    assert set(data.genes) <= set(info["string_known"])
    for name, (_, layer_genes) in info["layers"].items():
        assert set(data.genes) <= set(layer_genes), name
    present = {k: set(rows) for k, (rows, _) in info["layers"].items()}
    counts = {s: sum(s in rows for rows in present.values()) for s in info["sids"]}
    expected = sorted(s for s, c in counts.items() if c >= 2)
    assert list(data.meta.index) == expected


def test_layers_missing_for_a_sample_are_masked(tmp_path):
    info = write_raw(tmp_path)
    data = assemble(str(tmp_path))
    order = ["mrna", "protein", "crispr", "methylation", "cnv"]
    for s, sid in enumerate(data.meta.index):
        for k, name in enumerate(order):
            has = sid in info["layers"][name][0]
            if not has:
                assert data.mask[s, k].all(), (sid, name)


def test_imputed_values_are_layer_means(tmp_path):
    write_raw(tmp_path)
    data = assemble(str(tmp_path))
    checked = 0
    for k in range(5):
        for g in range(data.n_genes):
            seen = ~data.mask[:, k, g]
            if seen.any() and not seen.all():
                filler = data.values[~seen, k, g]
                observed = data.values[seen, k, g]
                assert np.allclose(filler, filler[0])
                assert filler[0] == 0 or np.isclose(filler[0], observed.mean())
                checked += 1
    assert checked > 20


def test_string_adjacency(tmp_path):
    write_raw(tmp_path)
    data = assemble(str(tmp_path))
    a = data.adj
    assert a.shape == (data.n_genes, data.n_genes)
    assert np.allclose(a, a.T)
    assert np.allclose(np.diag(a), 0)
    assert a.min() >= 0 and a.max() == 1.0
    assert (a == 0).sum() > 0


def write_prior(path, genes, seed=5):
    rng = np.random.default_rng(seed)
    rows = [(a, b, rng.uniform(0.3, 0.9)) for i, a in enumerate(genes) for b in genes[i + 1:]]
    pd.DataFrame(rows, columns=["gene_a", "gene_b", "score"]).to_csv(path, sep="\t", index=False)
    return pd.DataFrame(rows, columns=["gene_a", "gene_b", "score"])


def test_another_prior_replaces_string(tmp_path):
    info = write_raw(tmp_path)
    prior = write_prior(tmp_path / "prior.tsv", info["genes"][:20])
    data = assemble(str(tmp_path), prior=str(tmp_path / "prior.tsv"))
    assert data.genes and set(data.genes) <= set(info["genes"][:20])
    inside = prior[prior["gene_a"].isin(data.genes) & prior["gene_b"].isin(data.genes)]
    lo, hi = inside["score"].min(), inside["score"].max()
    pos = {g: i for i, g in enumerate(data.genes)}
    row = inside.iloc[7]
    expected = (row["score"] - lo) / (hi - lo)
    assert np.isclose(data.adj[pos[row["gene_a"]], pos[row["gene_b"]]], expected)
    assert np.allclose(data.adj, data.adj.T) and np.allclose(np.diag(data.adj), 0)
    assert data.adj.max() == 1.0 and data.adj.min() == 0.0


def test_prior_needs_the_three_columns(tmp_path):
    pd.DataFrame({"gene_a": ["A"], "gene_b": ["B"]}).to_csv(tmp_path / "prior.tsv", sep="\t", index=False)
    with pytest.raises(ValueError, match="score"):
        read_prior(str(tmp_path / "prior.tsv"))
