import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from gid.evaluate import complex_pairs, corum_recovery, pair_matrix, recovery, string_recovery


def test_complex_pairs_are_sorted_and_unique(tmp_path):
    path = tmp_path / "corum.txt"
    pd.DataFrame({"subunits_gene_name": ["B;A;C", "A;B", None, "D"]}).to_csv(path, sep="\t", index=False)
    assert complex_pairs(path) == {("A", "B"), ("A", "C"), ("B", "C")}


def test_pair_matrix_ignores_genes_outside_the_network():
    m = pair_matrix({("A", "B"), ("A", "Z")}, ["A", "B", "C"])
    assert m.sum() == 2 and m[0, 1] == m[1, 0] == 1


def test_recovery_uses_only_the_upper_triangle():
    rng = np.random.default_rng(0)
    n = 30
    truth = np.triu((rng.random((n, n)) < 0.1).astype(int), 1)
    truth = truth + truth.T
    score = truth * 0.5 + rng.random((n, n)) * 0.6
    score = (score + score.T) / 2
    np.fill_diagonal(score, 99.0)                       # a huge diagonal must not matter
    out = recovery(score, truth)
    iu = np.triu_indices(n, 1)
    assert abs(out["auc"] - roc_auc_score(truth[iu], score[iu])) < 1e-12
    assert out["n_positive"] == int(truth[iu].sum())


def test_string_recovery_perfect_and_random():
    rng = np.random.default_rng(1)
    n = 40
    adj = np.triu(rng.random((n, n)), 1)
    adj = adj + adj.T
    assert string_recovery(adj, adj, 0.7)["auc"] == 1.0
    noise = rng.random((n, n))
    noise = (noise + noise.T) / 2
    assert 0.4 < string_recovery(noise, adj, 0.7)["auc"] < 0.6


def test_corum_recovery_end_to_end(tmp_path):
    genes = [f"G{i}" for i in range(12)]
    path = tmp_path / "corum.txt"
    pd.DataFrame({"subunits_gene_name": ["G0;G1;G2", "G5;G6", "G9;G10;G11"]}).to_csv(path, sep="\t", index=False)
    truth = pair_matrix(complex_pairs(path), genes)
    out = corum_recovery(truth * 1.0 + 0.01, genes, str(path))
    assert out["auc"] == 1.0 and out["n_positive"] == 7
