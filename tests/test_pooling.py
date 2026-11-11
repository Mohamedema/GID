import sys

import numpy as np
import pandas as pd
import pytest
import torch

from gid import pooling
from gid.graph import build_graph
from gid.model import GCNAutoencoder
from gid.train import Config, node_features, save_model
from tests.synthetic import random_omics


def small_setup(seed=0, n_genes=16, n_samples=12):
    d = random_omics(n_genes=n_genes, n_samples=n_samples, seed=seed)
    feats, _ = node_features(d.values, d.mask, d.layers)
    ei, ew = build_graph(d.adj, 0.6)
    torch.manual_seed(seed)
    model = GCNAutoencoder(feats.shape[1], 12, 6, 0.1)
    model.eval()
    return d, feats, ei, ew, model


def test_embedding_is_the_softmax_weighted_sum_of_the_gene_embeddings():
    d, feats, ei, ew, model = small_setup()
    out = pooling.pooled_embeddings(model, feats, ei, ew, 5)
    with torch.no_grad():
        _, z = model(feats, ei, ew)
    g, n = d.n_genes, d.n_samples
    for i in (0, 4, n - 1):
        mean_input = np.mean([feats[:, k * n + i].numpy() for k in range(5)], axis=0)
        weights = np.exp(mean_input - mean_input.max())
        weights /= weights.sum()
        assert np.allclose(out[i], weights @ z.numpy(), atol=1e-5)
    assert out.shape == (n, 6)


def test_samples_with_the_same_input_get_the_same_embedding():
    d, feats, ei, ew, model = small_setup(1)
    n = d.n_samples
    feats = feats.clone()
    for k in range(5):
        feats[:, k * n + 3] = feats[:, k * n + 7]
    out = pooling.pooled_embeddings(model, feats, ei, ew, 5)
    assert np.allclose(out[3], out[7], atol=1e-6)


def test_command_line_run(tmp_path, monkeypatch):
    pytest.importorskip("umap")
    d, feats, ei, ew, model = small_setup(2, n_genes=14, n_samples=18)
    d.save(str(tmp_path / "data"))
    cfg = Config(hidden_dim=12, latent_dim=6, graph_threshold=0.6)
    save_model(str(tmp_path / "model.pt"), model, cfg, [0.0], d.layers, d.n_samples)
    monkeypatch.setattr(sys, "argv", ["pooling", "--data", str(tmp_path / "data"), "--model", str(tmp_path / "model.pt"),
                                      "--out", str(tmp_path / "out"), "--neighbours", "5"])
    pooling.main()
    table = pd.read_csv(tmp_path / "out" / "umap_samples.csv")
    assert list(table.columns) == ["Sample", "UMAP1", "UMAP2", "Tissue"]
    assert table["Sample"].tolist() == d.meta.index.tolist()
    assert table["Tissue"].tolist() == d.meta["tissue"].tolist()
    assert np.load(tmp_path / "out" / "sample_embeddings.npy").shape == (18, 6)
