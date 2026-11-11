import sys

import numpy as np
import pandas as pd

from gid import seed_network
from tests.synthetic import gene_names, random_omics


def targets_table(path=None):
    table = pd.DataFrame({
        "symbol": ["gene001", "GENE002", "GENE003", "GENE004", "GENE005", "GENE006", "GENE007", "GENE008", "GENE099"],
        "globalScore": [0.9, 0.8, 0.5, 0.1, 0.35, 0.05, 0.2, 0.0, 0.7],
        "gwasCredibleSets": [0.9, 0.6, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.8],
        "hasSmallMoleculeBinder": [0, 0, 0, 1, 0, 0, 0, 0, 0],
        "maxClinicalStage": [0, 0, 0, 0, 0, 0, 0, 0, 0],
        "isCancerDriverGene": ["1", "no", "yes", "0", "0", "0", "0", "0", "0"],
        "clingen": ["no data", "no data", "no data", "no data", "no data", "Definitive", "no data", "no data", "no data"],
        "tissueSpecificity": ["no data"] * 6 + ["Tissue enhanced", "no data", "no data"],
    })
    if path is not None:
        table.to_csv(path, sep="\t", index=False)
    return table


def test_seeds_are_the_network_genes_above_the_gwas_cut(tmp_path):
    path = tmp_path / "ot.tsv"
    targets_table(str(path))
    targets = seed_network.read_targets(str(path))
    genes = gene_names(10)
    assert sorted(seed_network.seed_genes(targets, genes)) == ["GENE001", "GENE002"]


def test_first_degree_edges_by_hand():
    genes = gene_names(8)
    net = np.zeros((8, 8))
    def link(a, b, w):
        net[a, b] = net[b, a] = w
    np.fill_diagonal(net, 1.0)
    link(0, 1, 0.95)     # seed - seed
    link(0, 2, 0.90)     # seed - neighbour
    link(1, 3, 0.80)     # seed - neighbour
    link(0, 4, 0.75)     # seed - neighbour, cut by top_n = 3
    link(2, 3, 0.85)     # neighbour - neighbour
    link(5, 6, 0.99)     # unrelated
    link(0, 7, 0.10)     # below threshold
    edges, allowed, seed_idx, neighbours = seed_network.first_degree_edges(net, genes, [genes[0], genes[1]], 0.7, 3)
    assert neighbours == [2, 3, 4]
    assert seed_idx == [0, 1]
    kinds = {frozenset((r.Source, r.Target)): r.Edge_Type for r in edges.itertuples()}
    assert kinds == {
        frozenset((genes[0], genes[1])): "Seed-Seed",
        frozenset((genes[0], genes[2])): "Seed-Neighbor",
        frozenset((genes[1], genes[3])): "Seed-Neighbor",
        frozenset((genes[0], genes[4])): "Seed-Neighbor",
        frozenset((genes[2], genes[3])): "Neighbor-Neighbor",
    }
    assert list(edges["Weight"]) == sorted(edges["Weight"], reverse=True)
    top = seed_network.first_degree_edges(net, genes, [genes[0], genes[1]], 0.7, 1)[3]
    assert top == [2]


def test_neighbour_scores_keep_the_best_seed_and_skip_seeds():
    norm = np.zeros((6, 6))
    for a, b, w in [(0, 1, 0.9), (0, 2, 0.75), (1, 2, 0.85), (1, 3, 0.7), (0, 4, 0.69), (3, 5, 0.99)]:
        norm[a, b] = norm[b, a] = w
    best = seed_network.neighbour_scores(norm, [0, 1], threshold=0.7)
    assert best == {2: 0.85, 3: 0.7}


def test_node_groups_follow_the_hierarchy(tmp_path):
    path = tmp_path / "ot.tsv"
    targets_table(str(path))
    targets = seed_network.read_targets(str(path))
    seeds = ["GENE001", "GENE002"]

    def group(gene):
        return seed_network.node_group(gene, seeds, targets)

    assert group("GENE001") == "Seed Gene (GWAS > 0.5)"
    assert group("GENE003") == "Cancer Driver Gene"
    assert group("GENE004") == "Druggable Target"
    assert group("GENE005") == "High Global Score (>= 0.3)"
    assert group("GENE006") == "clingen"
    assert group("GENE007") == "tissueSpecificity"
    assert group("GENE008") == "No Data"
    assert group("GENE050") == "No Data"


def test_command_line_run(tmp_path, monkeypatch):
    data = random_omics(n_genes=12, n_samples=18)
    data.save(str(tmp_path / "data"))
    rng = np.random.default_rng(0)
    nets = rng.random((2, 12, 12)).astype(np.float32)
    nets = (nets + nets.transpose(0, 2, 1)) / 2
    nets[:, np.arange(12), np.arange(12)] = 1.0
    np.save(tmp_path / "tissue_networks.npy", nets)
    (tmp_path / "tissue_names.txt").write_text("Haematopoietic and Lymphoid\nLung\n")
    targets_table(str(tmp_path / "ot.tsv"))
    monkeypatch.setattr(sys, "argv", ["seed_network", "--data", str(tmp_path / "data"), "--networks", str(tmp_path / "tissue_networks.npy"),
                                      "--names", str(tmp_path / "tissue_names.txt"), "--targets", str(tmp_path / "ot.tsv"),
                                      "--out", str(tmp_path / "out"), "--threshold", "0.8", "--top-n", "4"])
    seed_network.main()
    edges = pd.read_csv(tmp_path / "out" / "Luk_lioness_1st_degree_edges.csv")
    nodes = pd.read_csv(tmp_path / "out" / "Luk_lioness_1st_degree_nodes.csv")
    assert set(nodes["Node"]) == set(edges["Source"]) | set(edges["Target"])
    assert (edges["Weight"] >= 0.8).all()
    listed = (tmp_path / "out" / "Luk_lioness_1st_degree_first_degree_genes.txt").read_text().split()
    assert listed == sorted(listed) and not {"GENE001", "GENE002"} & set(listed)
