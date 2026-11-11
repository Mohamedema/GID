import numpy as np
import pytest

from gid.data import EdgeMatrix, edge_column, load_omics, upper_index
from tests.synthetic import random_omics


def test_edge_column_matches_triu_order():
    g = 11
    i, j = upper_index(g)
    assert (edge_column(i, j, g) == np.arange(i.size)).all()


def test_roundtrip(tmp_path):
    d = random_omics()
    d.save(tmp_path / "d")
    back = load_omics(tmp_path / "d")
    assert back.genes == d.genes
    assert np.array_equal(back.mask, d.mask)
    assert np.allclose(back.values, d.values)
    assert np.allclose(back.adj, d.adj)
    assert list(back.meta.index) == list(d.meta.index)
    assert back.layers == d.layers


def test_select_layers_keeps_order():
    d = random_omics()
    sub = d.select_layers(["cnv", "mrna"])
    assert sub.layers == ["cnv", "mrna"]
    assert np.array_equal(sub.values[:, 0], d.values[:, 4])
    assert np.array_equal(sub.values[:, 1], d.values[:, 0])


def test_edge_matrix_pairs_and_square(tmp_path):
    g, n = 9, 4
    store = EdgeMatrix.create(str(tmp_path / "e.npy"), n, g)
    rng = np.random.default_rng(1)
    full = rng.random((n, g, g))
    full = (full + full.transpose(0, 2, 1)) / 2
    i, j = upper_index(g)
    for s in range(n):
        store.x[s] = full[s][i, j]
    ii, jj = np.array([5, 0, 7, 3]), np.array([2, 8, 1, 3 + 1])
    got = store.pairs(ii, jj, [2, 0])
    assert np.allclose(got, full[[2, 0]][:, ii, jj], atol=1e-6)
    sq = store.square(1)
    ref = full[1].copy()
    np.fill_diagonal(ref, 1)
    assert np.allclose(sq, ref, atol=1e-6)


def test_edge_matrix_wrong_size(tmp_path):
    np.save(tmp_path / "x.npy", np.zeros((3, 10), dtype=np.float32))
    with pytest.raises(ValueError):
        EdgeMatrix(str(tmp_path / "x.npy"), 9)
