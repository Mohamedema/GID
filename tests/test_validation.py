import sys

import numpy as np
import pandas as pd
import pytest
from scipy.stats import hypergeom

from gid import validation
from gid.data import upper_index
from tests.synthetic import random_omics

GENES = ["TP53", "MDM2", "ATM", "BRCA1", "BRCA2", "RB1", "E2F1", "CDK4", "CSK", "GRB2"]


def alias(name):
    return f"uniprotkb:{name}(gene name)|uniprotkb:{name}_HUMAN(entry name)"


def test_gene_names_are_read_from_the_alias_field():
    assert validation.intact_gene("uniprotkb:TP53(gene name)|uniprotkb:P53(gene name synonym)") == "TP53"
    assert validation.intact_gene("uniprotkb:P53(gene name synonym)") == "P53"
    assert validation.intact_gene("psi-mi:cdk4_human(display_short)") == "CDK4_HUMAN"
    assert validation.intact_gene("something else") is None
    assert validation.intact_gene(np.nan) is None


def test_intact_pairs_use_the_host_filter(tmp_path):
    rows = [
        (alias("TP53"), alias("MDM2"), "9606 - Homo sapiens (K562 cell)"),
        (alias("ATM"), alias("BRCA1"), "9606 - Homo sapiens (HeLa)"),
        (alias("RB1"), alias("E2F1"), "Chronic myeloid Leukemia"),
        (alias("CDK4"), alias("CDK4"), "K562"),
        (alias("CSK"), alias("NOTINNETWORK"), "K562"),
    ]
    path = tmp_path / "Cancer.txt"
    pd.DataFrame(rows, columns=["Alias(es) interactor A", "Alias(es) interactor B", "Host organism(s)"]).to_csv(
        path, sep="\t", index=False)
    pairs = validation.intact_pairs(str(path), GENES)
    assert pairs == {frozenset(["TP53", "MDM2"]), frozenset(["RB1", "E2F1"])}


def write_synlethdb(path, a="n1.name", b="n2.name", context="r.cell_line"):
    pd.DataFrame({
        a: ["TP53", "CSK", "ATM", "RB1", "ATM", "xyz"],
        b: ["MDM2", "GRB2", "BRCA1", "E2F1", "ATM", "TP53"],
        "r.source": ["DAISY", "GenomeRNAi", "Text", "DAISY", "x", "x"],
        context: ["K-562", "HeLa", "THP1", "", "K562", "K562"],
    }).to_csv(path, sep="\t", index=False)


@pytest.mark.parametrize("a,b", [("n1.name", "n2.name"), ("gene_a", "gene_b"), ("geneA", "geneB")])
def test_gene_columns_are_found(tmp_path, a, b):
    path = tmp_path / "sl.tsv"
    write_synlethdb(path, a, b)
    table = validation.read_synlethdb(str(path))
    assert validation.gene_columns(table, GENES) == (a, b)


def test_gene_columns_fail_without_network_genes(tmp_path):
    table = pd.DataFrame({"x": ["foo", "bar"], "y": ["baz", "qux"]})
    with pytest.raises(ValueError):
        validation.gene_columns(table, GENES)


def test_synlethdb_pairs_with_and_without_the_cell_filter(tmp_path):
    path = tmp_path / "sl.tsv"
    write_synlethdb(path)
    everything, _ = validation.synlethdb_pairs(str(path), GENES)
    assert everything == {frozenset(p) for p in [("TP53", "MDM2"), ("CSK", "GRB2"), ("ATM", "BRCA1"), ("RB1", "E2F1")]}
    haema, passed = validation.synlethdb_pairs(str(path), GENES, validation.HAEMA_KEYWORDS)
    assert haema == {frozenset(["TP53", "MDM2"]), frozenset(["ATM", "BRCA1"])}
    assert set(passed) == {"k-562", "thp1", "k562"}


def test_enrichment_numbers_by_hand():
    labels = np.zeros(1000)
    labels[[0, 3, 7, 20, 400, 900]] = 1
    table = validation.enrichment_table(labels, fractions=(0.01, 0.05, 0.10))
    row = table[table["top_pct"] == 5].iloc[0]
    assert row["K"] == 50 and row["hits"] == 4 and row["gold_pairs"] == 6
    assert np.isclose(row["expected_hits"], 0.3)
    assert np.isclose(row["fold"], 4 / 0.3)
    assert np.isclose(row["hypergeom_p"], hypergeom.sf(3, 1000, 6, 50))
    assert row["p_label"] == "P < 0.001"
    assert row["fold_label"] == "13.3-fold enrichment"


def loop_bands(weights, labels, sorted_labels, n_iter, seed):
    """Bootstrap and permutation bands as plain loops, the reference for validation.capture_bands."""
    np.random.seed(seed)
    total = len(weights)
    eval_x = np.linspace(0, 0.20, 200)
    boot_x = np.arange(1, total + 1) / total
    boot_rates = np.zeros((n_iter, 200))
    null_rates = np.zeros((n_iter, 200))
    gold = sorted_labels.sum()
    for b in range(n_iter):
        idx = np.random.randint(0, total, size=total)
        w, y = weights[idx], labels[idx]
        y_sorted = y[np.argsort(w)[::-1]]
        cum = np.cumsum(y_sorted)
        cap = cum / cum[-1] if cum[-1] > 0 else np.zeros(total)
        boot_rates[b] = np.interp(eval_x, boot_x, cap)
        perm = np.random.permutation(sorted_labels)
        null_rates[b] = np.interp(eval_x, boot_x, np.cumsum(perm) / gold)
    return (np.percentile(boot_rates, 2.5, axis=0), np.percentile(boot_rates, 97.5, axis=0),
            np.percentile(null_rates, 2.5, axis=0), np.percentile(null_rates, 97.5, axis=0))


def test_bands_match_the_plain_loops():
    rng = np.random.default_rng(0)
    total = 3000
    weights = rng.normal(size=total)
    labels = (rng.random(total) < 0.01 + 0.05 * (weights > 1.5)).astype(float)
    order, sorted_labels = validation.rank_edges(weights, labels)
    curve = validation.capture_bands(weights, labels, sorted_labels, n_iter=20, seed=42)
    want = loop_bands(weights, labels, sorted_labels, 20, 42)
    for column, expected in zip(["Model_CI_Lower", "Model_CI_Upper", "Null_CI_Lower", "Null_CI_Upper"], want):
        assert np.allclose(curve[column], expected)
    observed = np.cumsum(sorted_labels) / sorted_labels.sum()
    x = np.arange(1, total + 1) / total
    assert np.allclose(curve["Model_Capture_Rate"], np.interp(curve["Proportion_Edges_Ranked"], x, observed))
    assert np.allclose(curve["Random_Expectation"], curve["Proportion_Edges_Ranked"])


def test_bands_overlap_point():
    grid = np.linspace(0, 0.2, 200)
    curve = pd.DataFrame({"Proportion_Edges_Ranked": grid, "Model_CI_Lower": np.where(grid > 0.05, 1.0, 0.0),
                          "Null_CI_Upper": np.full(200, 0.5)})
    assert np.isclose(validation.bands_separate_from(curve), 100 * grid[np.argmax(grid > 0.05)])
    curve["Model_CI_Lower"] = 0.0
    assert np.isnan(validation.bands_separate_from(curve))
    curve["Model_CI_Lower"] = 1.0
    assert validation.bands_separate_from(curve) == 0.0


def test_gold_table_lists_the_ranks():
    genes = GENES[:5]
    ii, jj = upper_index(5)
    weights = np.linspace(0, 1, len(ii))
    labels = np.zeros(len(ii))
    labels[[2, 9]] = 1
    order, sorted_labels = validation.rank_edges(weights, labels)
    table = validation.gold_table(order, sorted_labels, weights, genes)
    assert list(table["Rank"]) == [1, 8]
    assert (table["Gene_A"].iloc[0], table["Gene_B"].iloc[0]) == (genes[ii[9]], genes[jj[9]])
    assert list(table["In_Top_5pct"]) == [False, False]


def test_evaluate_writes_the_tables(tmp_path):
    rng = np.random.default_rng(2)
    n = len(GENES)
    network = rng.random((n, n))
    network = (network + network.T) / 2
    pairs = {frozenset(["TP53", "MDM2"]), frozenset(["ATM", "BRCA1"]), frozenset(["RB1", "E2F1"])}
    for p in pairs:
        a, b = (GENES.index(x) for x in p)
        network[a, b] = network[b, a] = 5.0
    stats, curve = validation.evaluate(network, GENES, pairs, str(tmp_path), suffix="_x", n_iter=10)
    assert (stats["gold_pairs"] == 3).all()
    assert stats["hits"].iloc[-1] == 3
    for name in ["gid_enrichment_data_with_null_x.csv", "gid_enrichment_stats_x.csv", "gid_gold_pairs_ranked_x.csv"]:
        assert (tmp_path / name).exists()
    ranked = pd.read_csv(tmp_path / "gid_gold_pairs_ranked_x.csv")
    assert sorted(ranked["Rank"]) == [1, 2, 3]
    assert len(curve) == 200


def test_command_line_run(tmp_path, monkeypatch):
    data = random_omics(n_genes=12, n_samples=18)
    data.genes = GENES + ["G11", "G12"]
    data.save(str(tmp_path / "data"))
    rng = np.random.default_rng(5)
    pop = rng.random((12, 12))
    np.save(tmp_path / "population_network.npy", ((pop + pop.T) / 2).astype(np.float32))
    write_synlethdb(tmp_path / "sl.tsv")
    monkeypatch.setattr(sys, "argv", ["validation", "synlethdb", "--data", str(tmp_path / "data"),
                                      "--file", str(tmp_path / "sl.tsv"), "--population", str(tmp_path / "population_network.npy"),
                                      "--out", str(tmp_path / "out"), "--all-cells", "--iterations", "5",
                                      "--suffix", "_SynLethDB_population"])
    validation.main()
    stats = pd.read_csv(tmp_path / "out" / "gid_enrichment_stats_SynLethDB_population.csv")
    assert stats["gold_pairs"].iloc[0] == 4
