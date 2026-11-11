import os
import sys
from itertools import combinations

import numpy as np
import pandas as pd

from gid import conservation
from gid.data import edge_column, upper_index
from gid.references import (
    bioplex_mask,
    corum_mask,
    ensembl_id,
    gene_lookup,
    huri_mask,
    pairs_mask,
    reactome_mask,
    read_hgnc,
    reference_masks,
    signor_mask,
    string_mask,
)
from tests.synthetic import random_omics

GENES = ["TP53", "MDM2", "ATM", "BRCA1", "BRCA2", "RB1", "E2F1", "CDK4"]


def pair_set(mask, n):
    ii, jj = upper_index(n)
    return {(GENES[i], GENES[j]) for i, j in zip(ii[mask], jj[mask])}


def write_resources(folder):
    os.makedirs(folder, exist_ok=True)
    pd.DataFrame({"subunits_gene_name": ["TP53;MDM2;ATM", "brca1;BRCA2;ATM;NOTINNET", "RB1;E2F1", None]}).to_csv(
        os.path.join(folder, "corum_humanComplexes.txt"), sep="\t", index=False)
    pd.DataFrame({"SymbolA": ["TP53", "RB1", "CDK4", "TP53"], "SymbolB": ["MDM2", "E2F1", "RB1", "TP53"],
                  "pInt": [0.9, 0.4, 0.8, 0.99]}).to_csv(os.path.join(folder, "BioPlex3.tsv"), sep="\t", index=False)
    pd.DataFrame({"TYPEA": ["protein", "chemical", "protein"], "TYPEB": ["protein", "protein", "protein"],
                  "ENTITYA": ["ATM", "TP53", "BRCA1"], "ENTITYB": ["TP53", "MDM2", "BRCA2"]}).to_csv(
        os.path.join(folder, "signor_human.tsv"), sep="\t", index=False)
    pd.DataFrame({"symbol": ["TP53", "MDM2", "ATM", "BRCA1"],
                  "ensembl_gene_id": ["ENSG1", "ENSG2", "ENSG3", "ENSG4"]}).to_csv(
        os.path.join(folder, "hgnc_map.txt"), sep="\t", index=False)
    with open(os.path.join(folder, "HuRI.tsv"), "w") as fh:
        fh.write("ENSG1\tENSG2\nENSG3\tENSG9\nENSG4\tENSG3\n")
    with open(os.path.join(folder, "reactome_interactions.txt"), "w") as fh:
        fh.write("# header\n")
        fh.write("x\tENSEMBL:ENSG1\ty\tz\tENSEMBL:ENSG3|UNIPROT:Q1\n")
        fh.write("x\tUNIPROT:P1|ENSEMBL:ENSG2\ty\tz\tENSEMBL:ENSG4\n")
        fh.write("short\n")


def test_resources_are_read_into_pair_masks(tmp_path):
    write_resources(str(tmp_path))
    n, lookup = len(GENES), gene_lookup(GENES)
    corum = pair_set(corum_mask(str(tmp_path / "corum_humanComplexes.txt"), lookup, n), n)
    assert corum == {("TP53", "MDM2"), ("TP53", "ATM"), ("MDM2", "ATM"), ("ATM", "BRCA1"), ("ATM", "BRCA2"),
                     ("BRCA1", "BRCA2"), ("RB1", "E2F1")}
    assert pair_set(bioplex_mask(str(tmp_path / "BioPlex3.tsv"), lookup, n), n) == {("TP53", "MDM2"), ("RB1", "CDK4")}
    assert pair_set(signor_mask(str(tmp_path / "signor_human.tsv"), lookup, n), n) == {("TP53", "ATM"), ("BRCA1", "BRCA2")}
    hgnc = read_hgnc(str(tmp_path / "hgnc_map.txt"))
    assert hgnc["ENSG3"] == "ATM"
    assert pair_set(huri_mask(str(tmp_path / "HuRI.tsv"), lookup, n, hgnc), n) == {("TP53", "MDM2"), ("ATM", "BRCA1")}
    assert pair_set(reactome_mask(str(tmp_path / "reactome_interactions.txt"), lookup, n, hgnc), n) == {
        ("TP53", "ATM"), ("MDM2", "BRCA1")}


def test_missing_resources_are_skipped(tmp_path):
    write_resources(str(tmp_path))
    os.remove(tmp_path / "BioPlex3.tsv")
    os.remove(tmp_path / "hgnc_map.txt")
    adj = np.zeros((8, 8))
    masks = reference_masks(GENES, adj, str(tmp_path))
    assert set(masks) == {"STRING", "CORUM", "SIGNOR"}


def test_ensembl_id_and_string_mask():
    assert ensembl_id("UNIPROT:P1|ENSEMBL:ENSG0000012") == "ENSG0000012"
    assert ensembl_id("UNIPROT:P1") == ""
    adj = np.zeros((4, 4))
    adj[0, 1] = adj[1, 0] = 0.7
    adj[2, 3] = adj[3, 2] = 0.69
    mask = string_mask(adj, 0.7)
    assert mask.sum() == 1 and mask[edge_column(0, 1, 4)]


def test_pairs_mask_ignores_self_and_unknown_genes():
    lookup = gene_lookup(["A", "b", "C"])
    mask = pairs_mask([("a", "B"), ("A", "A"), ("A", "Z"), ("c", "b")], lookup, 3)
    assert mask.tolist() == [True, False, True]


def reference_sets(networks, masks, keep, db_sets):
    """Overlaps computed with Python sets, the reference for conservation.jaccard_pairs."""
    ii, jj = upper_index(networks[0].shape[0])
    edges = []
    for net in networks:
        scores = np.array(net, dtype=np.float32)[ii, jj]
        norm = (scores - scores.min()) / (scores.max() - scores.min())
        edges.append(set(np.where(norm > np.percentile(norm, 100 - keep))[0]))
    out = {}
    for t1, t2 in combinations(range(len(networks)), 2):
        e1, e2 = edges[t1], edges[t2]
        out[("Quantified", t1, t2)] = len(e1 & e2) / len(e1 | e2)
        for db, d in db_sets.items():
            k1, k2 = e1 & d, e2 & d
            out[(db, t1, t2)] = len(k1 & k2) / len(k1 | k2) if k1 | k2 else np.nan
    return out


def test_jaccard_matches_the_set_implementation():
    rng = np.random.default_rng(7)
    g = 30
    nets = rng.random((5, g, g))
    nets = (nets + nets.transpose(0, 2, 1)) / 2
    n_edges = g * (g - 1) // 2
    masks = {"STRING": rng.random(n_edges) < 0.2, "CORUM": rng.random(n_edges) < 0.05}
    db_sets = {k: set(np.flatnonzero(v)) for k, v in masks.items()}
    names = ["a", "b", "c", "d", "e"]
    edges = [conservation.top_edges(net, 60) for net in nets]
    assert all(abs(e.mean() - 0.6) < 0.01 for e in edges)
    got = conservation.jaccard_pairs(names, edges, masks)
    want = reference_sets(nets, masks, 60, db_sets)
    assert len(got) == len(want)
    for row in got.itertuples():
        key = (row.database, names.index(row.tissue_1), names.index(row.tissue_2))
        assert np.isclose(row.jaccard_index, want[key], equal_nan=True)
    assert list(got["database"].unique()) == ["CORUM", "STRING", "Quantified"]


def test_command_line_run(tmp_path, monkeypatch):
    data = random_omics(n_genes=12, n_samples=18)
    data.meta["tissue"] = (["Lung"] * 6 + ["Breast"] * 6 + ["Skin"] * 5 + ["Bone"])
    data.save(str(tmp_path / "data"))
    rng = np.random.default_rng(1)
    nets = rng.random((4, 12, 12)).astype(np.float32)
    nets = (nets + nets.transpose(0, 2, 1)) / 2
    np.save(tmp_path / "tissue_networks.npy", nets)
    (tmp_path / "tissue_names.txt").write_text("Bone\nBreast\nLung\nSkin\n")
    write_resources(str(tmp_path / "ref"))
    monkeypatch.setattr(sys, "argv", ["conservation", "--data", str(tmp_path / "data"),
                                      "--networks", str(tmp_path / "tissue_networks.npy"),
                                      "--names", str(tmp_path / "tissue_names.txt"),
                                      "--references", str(tmp_path / "ref"), "--out", str(tmp_path / "out")])
    conservation.main()
    table = pd.read_csv(tmp_path / "out" / "tissue_jaccard_pairs_top60percent.csv")
    assert set(table["tissue_1"]) | set(table["tissue_2"]) == {"Breast", "Lung", "Skin"}
    assert len(table[table["database"] == "Quantified"]) == 3
