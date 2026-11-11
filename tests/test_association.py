import json
import sys

import numpy as np
import pandas as pd
import pytest

from gid import association
from gid.data import EdgeMatrix
from tests.synthetic import random_omics


@pytest.mark.parametrize("q", [0, 1, 30, 50, 70, 99.9, 100])
def test_global_percentile_equals_numpy(q):
    rng = np.random.default_rng(1)
    x = rng.normal(size=(37, 211)).astype(np.float32)
    want = np.percentile(x.astype(np.float64), q)
    for bins, block in [(64, 5), (1 << 12, 37), (7, 1)]:
        got = association.global_percentile(x, q, block=block, bins=bins)
        assert np.isclose(got, want, rtol=1e-12, atol=1e-12)


def test_global_percentile_with_many_ties():
    rng = np.random.default_rng(2)
    x = rng.integers(0, 6, size=(25, 80)).astype(np.float32)
    for q in [10, 40, 70, 95]:
        assert np.isclose(association.global_percentile(x, q, block=4, bins=32), np.percentile(x, q))
    constant = np.full((4, 9), 0.25, dtype=np.float32)
    assert association.global_percentile(constant, 70) == 0.25


def test_global_percentile_reads_a_memmap(tmp_path):
    store = EdgeMatrix.create(str(tmp_path / "s.npy"), 12, 9)
    store.x[:] = np.random.default_rng(3).random(store.shape).astype(np.float32)
    store.flush()
    assert np.isclose(association.global_percentile(store.x, 70, block=3), np.percentile(np.asarray(store.x), 70))


def test_tissue_probability_counts_samples_above_the_threshold():
    x = np.array([[0.9, 0.1, 0.6], [0.8, 0.2, 0.4], [0.1, 0.9, 0.7], [0.2, 0.8, 0.3], [0.3, 0.7, 0.9]], dtype=np.float32)
    tissues = ["Lung", "Lung", "Skin", "Skin", "Skin"]
    names, prob = association.tissue_probability(x, tissues, np.float32(0.5))
    assert names == ["Lung", "Skin"]
    assert np.allclose(prob[0], [1, 0, 0.5])
    assert np.allclose(prob[1], [0, 1, 2 / 3])


def test_summary_and_sharing_follow_the_cut_offs():
    prob = np.array([[1.0, 0.7, 0.5, 0.2, 0.0],
                     [0.9, 0.6, 0.0, 0.1, 0.0],
                     [0.8, 0.0, 0.6, 0.0, 0.0]], dtype=np.float32)
    names = ["a", "b", "c"]
    summary = association.summarise(names, prob)
    assert list(summary["likely"]) == [2, 2, 2]
    assert list(summary["confident"]) == [1, 1, 1]
    assert (summary["quantified"] == 5).all()
    distribution, per_edge = association.sharing(prob)
    assert list(per_edge["tissues_present_in"]) == [3, 2, 1, 0, 0]
    assert list(distribution["num_associations"]) == [1, 1, 1]


def test_command_line_run(tmp_path, monkeypatch):
    data = random_omics(n_genes=10, n_samples=18)
    data.save(str(tmp_path / "data"))
    store = EdgeMatrix.create(str(tmp_path / "sample_networks.npy"), 18, 10)
    store.x[:] = np.random.default_rng(4).random(store.shape).astype(np.float32)
    store.flush()
    monkeypatch.setattr(sys, "argv", ["association", "--data", str(tmp_path / "data"),
                                      "--networks", str(tmp_path / "sample_networks.npy"), "--out", str(tmp_path / "out")])
    association.main()
    summary = pd.read_csv(tmp_path / "out" / "tissue_summary.csv")
    assert list(summary["tissue"]) == ["Breast", "Lung", "Skin"]
    assert (summary["quantified"] == 45).all()
    edges = pd.read_csv(tmp_path / "out" / "edge_tissue_sharing.csv")
    assert len(edges) == 45
    info = json.load(open(tmp_path / "out" / "association_threshold.json"))
    assert np.isclose(info["threshold"], np.percentile(np.asarray(store.x), 70), atol=1e-6)
