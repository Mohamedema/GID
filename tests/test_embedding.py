import inspect
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler

from gid import embedding
from gid.data import EdgeMatrix
from tests.synthetic import random_omics


def test_variance_filter_keeps_the_most_variable_columns():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, 200)) * np.linspace(0.1, 3, 200)
    kept = embedding.variance_filter(x, top_pct=5)
    assert len(kept) == 10
    assert set(kept) == set(np.argsort(x.var(axis=0))[-10:])
    assert np.allclose(embedding.edge_variance(x, block=33), x.var(axis=0))


def test_clustering_indices_match_scikit_learn():
    rng = np.random.default_rng(1)
    labels = np.repeat(["a", "b", "c"], 20)
    x = rng.normal(size=(60, 300)) + np.repeat(np.arange(3), 20)[:, None] * rng.normal(size=(1, 300))
    scaled = embedding.filtered_scaled(x, top_pct=10)
    kept = embedding.variance_filter(x, 10)
    ref = StandardScaler().fit_transform(x[:, kept])
    assert np.allclose(scaled, ref)
    ch, db = embedding.clustering_index(scaled, labels)
    assert np.isclose(ch, calinski_harabasz_score(ref, labels))
    assert np.isclose(db, davies_bouldin_score(ref, labels))
    assert ch > 1


def test_umap_is_reproducible_and_separates_planted_groups():
    pytest.importorskip("umap")
    rng = np.random.default_rng(2)
    labels = np.repeat([0, 1], 30)
    x = rng.normal(size=(60, 50)) + labels[:, None] * 6.0
    a = embedding.sample_umap(StandardScaler().fit_transform(x), seed=7)
    b = embedding.sample_umap(StandardScaler().fit_transform(x), seed=7)
    assert a.shape == (60, 2) and np.allclose(a, b)
    centres = np.array([a[labels == k].mean(axis=0) for k in (0, 1)])
    closest = np.linalg.norm(a[:, None, :] - centres[None, :, :], axis=2).argmin(axis=1)
    assert (closest == labels).all()


def test_command_line_run(tmp_path, monkeypatch):
    pytest.importorskip("umap")
    data = random_omics(n_genes=12, n_samples=18)
    data.save(str(tmp_path / "data"))
    store = EdgeMatrix.create(str(tmp_path / "sample_networks.npy"), 18, 12)
    rng = np.random.default_rng(3)
    store.x[:] = rng.normal(size=store.shape).astype(np.float32)
    store.flush()
    pd.DataFrame({"Tissue": ["Lung", "Breast", "Skin"], "Anatomy": ["Respiratory", "Reproductive", "Skin"]}).to_csv(
        tmp_path / "anatomy.csv", index=False)
    monkeypatch.setattr(sys, "argv", ["embedding", "--data", str(tmp_path / "data"), "--networks", str(tmp_path / "sample_networks.npy"),
                                      "--out", str(tmp_path / "out"), "--top-pct", "20", "--anatomy", str(tmp_path / "anatomy.csv")])
    embedding.main()
    table = pd.read_csv(tmp_path / "out" / "umap_cont_tissue_GID.csv")
    assert list(table.columns) == ["UMAP1", "UMAP2", "Tissue", "Anatomy"]
    assert len(table) == 18 and table["Anatomy"].notna().all()
    assert (tmp_path / "out" / "clustering_index_GID.csv").exists()


def test_umap_model_keeps_the_legacy_pipeline_when_the_release_has_a_switch():
    umap = pytest.importorskip("umap")
    model = embedding.umap_model(n_neighbors=5, random_state=1)
    if "compatibility_layout" in inspect.signature(umap.UMAP).parameters:
        assert model.compatibility_layout is True
    assert model.n_neighbors == 5
