import os
import sys
from itertools import combinations

import numpy as np
import pandas as pd

from gid import cytoscape
from tests.synthetic import gene_names, random_omics


def make_networks(seed=0, n_tissues=4, g=30):
    rng = np.random.default_rng(seed)
    nets = rng.normal(size=(n_tissues, g, g)).astype(np.float32)
    nets = (nets + nets.transpose(0, 2, 1)) / 2
    adj = rng.random((g, g))
    adj = np.triu(adj, 1)
    adj = adj + adj.T
    return nets, adj, gene_names(g), ["Bone", "Lung", "Skin", "Thyroid"]


def reference_export(members, tissue, nets, names, adj, genes, min_nodes, delta, string_threshold):
    """The export of a complex as plain loops over arrays."""
    g = len(genes)
    idx = {x: i for i, x in enumerate(genes)}
    upper = {x.strip().upper(): i for i, x in enumerate(genes)}
    orig = {x.strip().upper(): x for x in genes}
    pan = np.zeros((g, g), dtype=np.float32)
    for n in nets:
        pan += np.array(n, dtype=np.float32) / len(nets)
    core = [orig[x] for x in [m.strip().upper() for m in members.split(";")] if x in upper]
    if len(core) >= min_nodes:
        nodes = list(core)
    else:
        core_set, nodes, cand = set(core), list(set(core)), {}
        for c in core:
            for k, score in enumerate(adj[idx[c]]):
                nb = genes[k]
                if nb in core_set:
                    continue
                cand[nb] = max(cand.get(nb, 0.0), float(score))
        for gene, _ in sorted(cand.items(), key=lambda x: x[1], reverse=True):
            if len(nodes) >= min_nodes:
                break
            nodes.append(gene)
    home = np.array(nets[names.index(tissue)], dtype=np.float32)
    other_sum, n_other = np.zeros((g, g), dtype=np.float32), 0
    for k, name in enumerate(names):
        if name != tissue:
            other_sum += np.array(nets[k], dtype=np.float32)
            n_other += 1
    other = other_sum / max(n_other, 1)
    rows = []
    for g1, g2 in combinations(nodes, 2):
        i, j = idx[g1], idx[g2]
        if i > j:
            i, j = j, i
        d = float(home[i, j]) - float(other[i, j])
        rows.append((frozenset([g1, g2]), round(float(home[i, j]), 4), round(float(other[i, j]), 4),
                     round(float(pan[i, j]), 4), round(d, 4),
                     "Red" if d > delta else "Blue" if d < -delta else "Gray",
                     "Red" if d < -delta else "Blue" if d > delta else "Gray",
                     int(float(adj[i, j]) >= string_threshold)))
    return sorted(rows, key=lambda r: sorted(r[0]))


def test_colours_follow_the_cut_off():
    assert [cytoscape.colour(d, 0.1) for d in (0.3, 0.1, 0.0, -0.1, -0.3)] == ["Red", "Gray", "Gray", "Gray", "Blue"]
    assert [cytoscape.colour(d, 0.1, home=False) for d in (0.3, 0.0, -0.3)] == ["Blue", "Gray", "Red"]


def test_extension_takes_the_strongest_string_neighbours():
    genes = gene_names(8)
    adj = np.zeros((8, 8))
    for (a, b), s in {(0, 3): 0.9, (1, 3): 0.2, (0, 5): 0.5, (1, 6): 0.7, (0, 7): 0.1}.items():
        adj[a, b] = adj[b, a] = s
    lookup = {g.upper(): g for g in genes}
    core = [genes[0], genes[1], genes[2]]
    assert cytoscape.extend(core, adj, genes, lookup, 5) == core + [genes[3], genes[6]]
    assert cytoscape.extend(core, adj, genes, lookup, 3) == core


def test_networks_of_a_complex_match_the_reference_loops():
    nets, adj, genes, names = make_networks()
    members = ";".join([genes[2], genes[7].lower(), genes[11], "NOTINNET"])
    means = cytoscape.TissueMeans(nets, names)
    home, other, nodes = cytoscape.complex_networks("X complex", "Lung", members, means, adj, genes, min_nodes=9)
    want = reference_export(members, "Lung", nets, names, adj, genes, 9, cytoscape.DELTA, cytoscape.STRING_THRESHOLD)
    assert len(nodes) == 9 and nodes["Is_Core"].sum() == 3
    assert len(home) == len(want) == 36
    rows = zip(map(frozenset, zip(home["Source"], home["Target"])), home["LIONESS_HomeTissue_Raw"],
               home["LIONESS_OtherMean_Raw"], home["LIONESS_PanCancer_Raw"], home["Delta_Home_vs_Other"],
               home["Edge_Color_Cytoscape"], other["Edge_Color_Cytoscape"], home["Known_Interaction"])
    got = sorted(rows, key=lambda r: sorted(r[0]))
    assert [r[0] for r in got] == [r[0] for r in want]
    for mine, theirs in zip(got, want):
        assert np.allclose(mine[1:5], theirs[1:5], atol=1e-4)
        assert mine[5:] == theirs[5:]
    assert (home["Edge_Weight"] == home["LIONESS_HomeTissue_Raw"]).all()
    assert (other["Edge_Weight"] == other["LIONESS_OtherMean_Raw"]).all()
    core = set(nodes.loc[nodes.Is_Core == 1, "Node"])
    both_core = home["Source"].isin(core) & home["Target"].isin(core)
    assert (home["Is_Core_Edge"] == both_core.astype(int)).all()


def test_complex_with_enough_subunits_is_not_extended():
    nets, adj, genes, names = make_networks()
    members = ";".join(genes[:10])
    result = cytoscape.complex_networks("big", "Bone", members, cytoscape.TissueMeans(nets, names), adj, genes, min_nodes=9)
    assert len(result[2]) == 10 and (result[2]["Node_Type"] == "Core_Complex").all()


def test_export_writes_every_table(tmp_path):
    nets, adj, genes, names = make_networks()
    table = pd.DataFrame({
        "complex_name": ["A/B complex (x), y", "no genes", "wrong tissue"],
        "Tissue": ["lung", "Skin", "Atlantis"],
        "subunits_gene_name": [";".join(genes[:4]), "NOTINNET;OTHER", ";".join(genes[5:8])],
    })
    cytoscape.export(table, ["A/B complex (x), y", "no genes", "wrong tissue"], nets, names, adj, genes, str(tmp_path), min_nodes=6)
    files = sorted(os.listdir(tmp_path))
    stem = "A_B_complex_x_y_Lung"
    assert files == sorted([f"{stem}_HOME_edges.csv", f"{stem}_HOME_nodes.csv", f"{stem}_OTHER_MEAN_edges.csv",
                            f"{stem}_OTHER_MEAN_nodes.csv", "ALL_HOME_edges.csv", "ALL_OTHER_MEAN_edges.csv", "ALL_nodes.csv"])
    assert len(pd.read_csv(tmp_path / "ALL_nodes.csv")) == 6


def test_command_line_run(tmp_path, monkeypatch):
    data = random_omics(n_genes=30, n_samples=18)
    data.save(str(tmp_path / "data"))
    nets, _, _, names = make_networks()
    np.save(tmp_path / "tissue_networks.npy", nets)
    (tmp_path / "tissue_names.txt").write_text("\n".join(names) + "\n")
    pd.DataFrame({"complex_name": ["C"], "Tissue": ["Skin"], "subunits_gene_name": [";".join(data.genes[:3])]}).to_csv(
        tmp_path / "table.csv", index=False)
    monkeypatch.setattr(sys, "argv", ["cytoscape", "--data", str(tmp_path / "data"), "--networks", str(tmp_path / "tissue_networks.npy"),
                                      "--names", str(tmp_path / "tissue_names.txt"), "--table", str(tmp_path / "table.csv"),
                                      "--complexes", "C", "--out", str(tmp_path / "out"), "--min-nodes", "5"])
    cytoscape.main()
    assert len(pd.read_csv(tmp_path / "out" / "C_Skin_HOME_nodes.csv")) == 5
