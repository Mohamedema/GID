"""The command line steps one after the other on a small synthetic data set, as in the README."""
import json
import sys

import numpy as np
import pandas as pd
import pytest

from gid import association, complexes, conservation, evaluate, infer, pooling, preprocess, resolution, train
from tests.synthetic import write_raw


def run(module, *args, monkeypatch):
    monkeypatch.setattr(sys, "argv", [module.__name__, *map(str, args)])
    module.main()


def write_corum(path, genes, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(14):
        members = rng.choice(genes, size=int(rng.integers(3, 6)), replace=False)
        rows.append({"complex_id": k, "complex_name": f"complex {k}", "subunits_gene_name": ";".join(members)})
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    root = tmp_path_factory.mktemp("project")
    write_raw(root / "raw", n_genes=40, n_samples=36)
    return root


def test_the_steps_hand_their_files_to_each_other(project, monkeypatch):
    data, runs, results, refs = project / "data", project / "runs" / "all_omics", project / "results", project / "refs"
    refs.mkdir()

    run(preprocess, "--raw", project / "raw", "--out", data, monkeypatch=monkeypatch)
    genes = [g for g in (data / "genes.txt").read_text().split()]
    write_corum(refs / "corum_humanComplexes.txt", genes)
    (project / "small.json").write_text(json.dumps({"hidden_dim": 16, "latent_dim": 8, "batch_size": 8,
                                                    "graph_threshold": 0.5}))

    run(train, "--data", data, "--out", runs, "--config", project / "small.json", "--epochs", 2, monkeypatch=monkeypatch)
    run(infer, "--data", data, "--model", runs / "model.pt", "--out", runs, monkeypatch=monkeypatch)
    for name in ("population_network.npy", "sample_networks.npy", "tissue_networks.npy", "tissue_names.txt"):
        assert (runs / name).exists(), name

    run(evaluate, "--data", data, "--population", runs / "population_network.npy", "--corum",
        refs / "corum_humanComplexes.txt", "--name", "all_omics", "--out", results, monkeypatch=monkeypatch)
    run(association, "--data", data, "--networks", runs / "sample_networks.npy", "--out", results, monkeypatch=monkeypatch)
    run(complexes, "--networks", runs / "tissue_networks.npy", "--names", runs / "tissue_names.txt", "--genes",
        data / "genes.txt", "--corum", refs / "corum_humanComplexes.txt", "--out", results, monkeypatch=monkeypatch)
    run(conservation, "--data", data, "--networks", runs / "tissue_networks.npy", "--names", runs / "tissue_names.txt",
        "--references", refs, "--min-samples", 3, "--out", results, monkeypatch=monkeypatch)
    run(resolution, "--data", data, "--networks", runs / "sample_networks.npy", "--name", "all_omics",
        "--min-variance", 0, "--components", 5, "--similarity", "--out", results, monkeypatch=monkeypatch)

    # the tables the figure scripts read
    wanted = ["all_omics_roc_corum.csv", "all_omics_roc_string.csv", "tissue_summary.csv", "sharing_distribution.csv",
              "complex_tissue_density_matrix.csv", "Final_Rigorously_Tissue_Specific_Complexes.csv",
              "tissue_jaccard_pairs_top60percent.csv", "all_omics_clustering_metrics.csv", "tissue_similarity.csv",
              "anatomy_similarity.csv"]
    for name in wanted:
        assert (results / name).exists(), name

    metrics = pd.read_csv(results / "all_omics_clustering_metrics.csv")
    assert {"CH_Tissue_Continuous", "DB_Organ_Continuous"} <= set(metrics.columns)
    names = (runs / "tissue_names.txt").read_text().split("\n")
    similarity = pd.read_csv(results / "tissue_similarity.csv", index_col=0)
    assert list(similarity.index) == [n for n in names if n]


def test_pooled_embeddings_from_the_trained_model(project, monkeypatch):
    pytest.importorskip("umap")
    run(pooling, "--data", project / "data", "--model", project / "runs" / "all_omics" / "model.pt", "--out",
        project / "results", "--neighbours", 5, monkeypatch=monkeypatch)
    table = pd.read_csv(project / "results" / "umap_samples.csv")
    assert list(table.columns) == ["Sample", "UMAP1", "UMAP2", "Tissue"]
    assert len(table) == len(pd.read_csv(project / "data" / "samples.csv"))
