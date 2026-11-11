import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score

from gid import resolution
from gid.data import EdgeMatrix
from tests.synthetic import random_omics

TISSUES = np.repeat(["Lung", "Breast", "Skin"], 8)
ANATOMY = {"Lung": "Respiratory", "Breast": "Reproductive", "Skin": "Skin"}


def planted(seed=0, n_edges=120):
    """Samples x edges values in which every tissue shifts a different set of edges."""
    rng = np.random.default_rng(seed)
    x = rng.normal(scale=0.3, size=(24, n_edges))
    for k, tissue in enumerate(["Lung", "Breast", "Skin"]):
        x[TISSUES == tissue, k * 30:(k + 1) * 30] += 2.5
    return x


def test_retained_edges_follow_the_variance_cutoff(tmp_path):
    store = EdgeMatrix.create(str(tmp_path / "s.npy"), 24, 16)
    rng = np.random.default_rng(1)
    store.x[:] = (rng.normal(size=store.shape) * np.linspace(0.2, 3, store.shape[1])).astype(np.float32)
    store.flush()
    var = np.asarray(store.x).astype(np.float64).var(axis=0)
    assert np.array_equal(resolution.retained_edges(store, 2.0), np.flatnonzero(var >= 2.0))


def test_metrics_match_scikit_learn_and_planted_structure_is_found():
    x = planted()
    organs = np.array([ANATOMY[t] for t in TISSUES])
    row = resolution.clustering_metrics(x, TISSUES, organs, components=5, binary_cutoff=1.0)
    scores = PCA(n_components=5, random_state=42).fit_transform(x)
    assert np.isclose(row["CH_Tissue_Continuous"], calinski_harabasz_score(scores, TISSUES))
    assert np.isclose(row["DB_Organ_Continuous"], davies_bouldin_score(scores, organs))
    binary = (x >= 1.0).astype(float)
    assert np.isclose(row["CH_Tissue_Binary"], calinski_harabasz_score(binary, TISSUES))
    assert row["CH_Tissue_Continuous"] > 50 and row["DB_Tissue_Continuous"] < 1
    assert list(row) == ["CH_Tissue_Continuous", "DB_Tissue_Continuous", "CH_Organ_Continuous", "DB_Organ_Continuous",
                         "CH_Tissue_Binary", "DB_Tissue_Binary", "CH_Organ_Binary", "DB_Organ_Binary"]


def test_tissue_similarity_is_spearman_of_standardised_means():
    x = planted(2)
    names, means = resolution.tissue_means(x, TISSUES)
    assert names == ["Breast", "Lung", "Skin"]
    assert np.allclose(means[1], x[TISSUES == "Lung"].mean(axis=0))
    sim = resolution.tissue_similarity(means, names)
    z = (means - means.mean(axis=0)) / (means.std(axis=0) + 1e-8)
    assert np.isclose(sim.loc["Breast", "Lung"], spearmanr(z[0], z[1])[0])
    assert np.allclose(np.diag(sim), 1) and np.allclose(sim, sim.T)
    assert list(sim.index) == list(sim.columns) == names


def test_organ_similarity_pools_the_tissues_of_a_system():
    rng = np.random.default_rng(3)
    means = rng.normal(size=(4, 60))
    names = ["Breast", "Cervix", "Lung", "Skin"]
    anatomy = {"Breast": "Reproductive", "Cervix": "Reproductive", "Lung": "Respiratory", "Skin": "Skin"}
    sim = resolution.organ_similarity(means, names, anatomy)
    assert list(sim.index) == ["Reproductive", "Respiratory", "Skin"]
    pooled = np.vstack([means[:2].mean(axis=0), means[2], means[3]])
    z = (pooled - pooled.mean(axis=0)) / (pooled.std(axis=0) + 1e-8)
    assert np.isclose(sim.loc["Reproductive", "Skin"], spearmanr(z[0], z[2])[0])


def test_command_line_run(tmp_path, monkeypatch):
    data = random_omics(n_genes=12, n_samples=18)
    data.save(str(tmp_path / "data"))
    store = EdgeMatrix.create(str(tmp_path / "sample_networks.npy"), 18, 12)
    rng = np.random.default_rng(4)
    tissue = data.meta["tissue"].to_numpy()
    values = rng.normal(size=store.shape) * 3
    for k, name in enumerate(["Breast", "Lung", "Skin"]):
        values[tissue == name, k * 20:(k + 1) * 20] += 4
    store.x[:] = values.astype(np.float32)
    store.flush()
    monkeypatch.setattr(sys, "argv", ["resolution", "--data", str(tmp_path / "data"),
                                      "--networks", str(tmp_path / "sample_networks.npy"), "--out", str(tmp_path / "out"),
                                      "--name", "mrna", "--min-variance", "2", "--components", "5", "--similarity"])
    resolution.main()
    metrics = pd.read_csv(tmp_path / "out" / "mrna_clustering_metrics.csv")
    assert metrics.shape == (1, 9) and metrics["Tag"].iloc[0] == "var2"
    assert metrics["CH_Tissue_Continuous"].iloc[0] > 1
    tissue_sim = pd.read_csv(tmp_path / "out" / "tissue_similarity.csv", index_col=0)
    organ_sim = pd.read_csv(tmp_path / "out" / "anatomy_similarity.csv", index_col=0)
    assert list(tissue_sim.index) == ["Breast", "Lung", "Skin"]
    assert list(organ_sim.index) == ["Reproductive", "Respiratory", "Skin"]
