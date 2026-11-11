import numpy as np
import torch

from gid.data import EdgeMatrix, upper_index
from gid.graph import build_graph
from gid.infer import cosine_similarity, population_network, sample_networks, tissue_networks
from gid.model import GCNAutoencoder
from gid.train import node_features
from tests.synthetic import random_omics


def small_setup(seed=0):
    d = random_omics(n_genes=16, n_samples=12, seed=seed)
    feats, _ = node_features(d.values, d.mask, d.layers)
    ei, ew = build_graph(d.adj, 0.6)
    torch.manual_seed(seed)
    model = GCNAutoencoder(feats.shape[1], 12, 6, 0.1)
    model.eval()
    return d, feats, ei, ew, model


def test_cosine_similarity_diagonal_and_range():
    s = cosine_similarity(torch.randn(10, 5))
    assert torch.allclose(torch.diag(s), torch.ones(10), atol=1e-5)
    assert s.abs().max() <= 1 + 1e-5


def test_population_network_is_symmetric():
    d, feats, ei, ew, model = small_setup()
    _, pop = population_network(model, feats, ei, ew)
    assert np.allclose(pop, pop.T)
    assert pop.shape == (16, 16)


def test_sample_networks_equal_the_leave_one_out_definition(tmp_path):
    d, feats, ei, ew, model = small_setup(2)
    n, g = d.n_samples, d.n_genes
    store = EdgeMatrix.create(str(tmp_path / "s.npy"), n, g)
    sample_networks(model, feats, ei, ew, n, len(d.layers), store)
    with torch.no_grad():
        _, h = model(feats, ei, ew)
        s_all = cosine_similarity(h)
        ii, jj = upper_index(g)
        for i in (0, 5, n - 1):
            x = feats.clone()
            for k in range(5):
                x[:, i + k * n] = -feats[:, i + k * n] / (n - 1)
            _, h_i = model(x, ei, ew)
            expected = (n * s_all - (n - 1) * cosine_similarity(h_i)).numpy()[ii, jj]
            assert np.allclose(store.x[i], expected, atol=1e-5)


def test_features_are_restored_after_each_sample(tmp_path):
    d, feats, ei, ew, model = small_setup(3)
    before = feats.clone()
    store = EdgeMatrix.create(str(tmp_path / "s.npy"), d.n_samples, d.n_genes)
    sample_networks(model, feats, ei, ew, d.n_samples, 5, store)
    assert torch.equal(before, feats)


def test_tissue_networks_are_means_of_their_samples(tmp_path):
    d = random_omics(n_genes=10, n_samples=9)
    store = EdgeMatrix.create(str(tmp_path / "s.npy"), 9, 10)
    rng = np.random.default_rng(0)
    store.x[:] = rng.random(store.x.shape).astype(np.float32)
    names, arr = tissue_networks(store, d.meta["tissue"])
    assert names == ["Breast", "Lung", "Skin"]
    ii, jj = upper_index(10)
    rows = np.flatnonzero(d.meta["tissue"].to_numpy() == "Lung")
    mean = np.asarray(store.x)[rows].mean(axis=0)
    lung = arr[names.index("Lung")]
    assert np.allclose(lung[ii, jj], mean, atol=1e-6)
    assert np.allclose(lung, lung.T)
    assert np.allclose(np.diag(lung), 1)
