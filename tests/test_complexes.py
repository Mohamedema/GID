import itertools
import os

import numpy as np
import pandas as pd

from gid.complexes import (
    EXCLUSIVE,
    binarise,
    complexes_in_network,
    density,
    density_matrix,
    fold_change,
    specific_complexes,
    with_corum_columns,
    write_table,
)

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "tissue_specific_complexes.csv")
SMALLEST = 1 / 105


def manuscript_table():
    return pd.read_csv(FIXTURE)


def test_fold_change_reproduces_the_manuscript_table():
    table = manuscript_table()
    plain = table[~table["Is_Exclusive"]]
    got = [round(fold_change(t, s, SMALLEST), 2) for t, s in zip(plain["Density_Score"], plain["Second_Best_Score"])]
    assert got == [float(x) for x in plain["Fold_Change"]]
    assert (table.loc[table["Is_Exclusive"], "Second_Best_Score"] == 0).all()
    assert (table.loc[table["Is_Exclusive"], "Fold_Change"] == EXCLUSIVE).all()


def test_selection_and_order_reproduce_the_manuscript_table():
    table = manuscript_table()
    tissues = sorted(set(table["Tissue"]) | set(table["Second_Best_Tissue"]))
    rows = {}
    for r in table.itertuples():
        scores = pd.Series(0.0, index=tissues)
        scores[r.Tissue] = r.Density_Score
        if r.Second_Best_Score > 0:
            scores[r.Second_Best_Tissue] = r.Second_Best_Score
        rows[r.Complex_ID_in_Model] = scores
    rows["Complex_90001"] = pd.Series([SMALLEST] + [0.0] * (len(tissues) - 1), index=tissues)
    for k in range(3000):
        rows[f"Complex_{91000 + k}"] = pd.Series(0.0, index=tissues)
    dens = pd.DataFrame(rows).T.sort_index()

    got = specific_complexes(dens)
    got = got[got["Complex_ID_in_Model"] != "Complex_90001"].reset_index(drop=True)

    columns = ["Tissue", "Complex_ID_in_Model", "Second_Best_Tissue", "Is_Exclusive"]
    assert got[columns].equals(table[columns])
    assert np.allclose(got["Density_Score"], table["Density_Score"])
    assert np.allclose(got["Second_Best_Score"], table["Second_Best_Score"])
    expected = pd.to_numeric(table["Fold_Change"], errors="coerce").fillna(np.inf)
    assert np.allclose(got["Fold_Change"].astype(float), expected)


def reference_selection(dens, quantile, min_fold):
    values = dens.to_numpy()
    eps = values[values > 0].min()
    cutoff = np.quantile(values, quantile, axis=0)
    kept = []
    for i in range(values.shape[0]):
        order = np.argsort(-values[i], kind="stable")
        top, second = values[i, order[0]], values[i, order[1]]
        if top <= 0 or top < cutoff[order[0]]:
            continue
        fold = np.inf if second == 0 else top / (second + eps)
        if fold >= min_fold:
            kept.append((dens.index[i], dens.columns[order[0]], dens.columns[order[1]]))
    return kept


def test_selection_matches_a_plain_implementation():
    rng = np.random.default_rng(3)
    values = rng.integers(0, 7, size=(400, 6)) / rng.integers(1, 12, size=(400, 1))
    values[rng.random(values.shape) < 0.5] = 0
    values[:5] = 0
    dens = pd.DataFrame(values, index=[f"Complex_{i}" for i in range(400)], columns=list("abcdef")).sort_index()
    for quantile, min_fold in [(0.9, 1.0), (0.75, 1.0), (0.9, 1.5), (0.5, 2.0)]:
        got = specific_complexes(dens, quantile, min_fold)
        mine = sorted(zip(got["Complex_ID_in_Model"], got["Tissue"], got["Second_Best_Tissue"]))
        assert mine == sorted(reference_selection(dens, quantile, min_fold))


def test_ties_for_the_highest_density_are_dropped():
    dens = pd.DataFrame({"a": [0.5, 0.2], "b": [0.5, 0.0], "c": [0.1, 0.0]}, index=["Complex_1", "Complex_2"])
    got = specific_complexes(dens, quantile=0.0)
    assert list(got["Complex_ID_in_Model"]) == ["Complex_2"]
    assert bool(got["Is_Exclusive"].iloc[0])


def test_complexes_listed_with_repeated_subunits():
    corum = pd.DataFrame({"subunits_gene_name": ["MALT1;MALT1;TRAF2", "A;B;C;D", None, "A; B ;X", "A;B"]})
    genes = ["A", "B", "C", "MALT1", "TRAF2"]
    found = complexes_in_network(corum, genes)
    assert found == {"Complex_0": [3, 3, 4], "Complex_1": [0, 1, 2]}
    assert complexes_in_network(corum, genes, unique=True) == {"Complex_1": [0, 1, 2]}


def test_density_of_a_complex_with_a_repeated_subunit():
    binary = np.zeros((5, 5), dtype=bool)
    binary[3, 4] = binary[4, 3] = True
    assert np.isclose(density(binary, [3, 3, 4]), 2 / 3)
    assert np.isclose(density(binary, [3, 4, 0]), 1 / 3)


def test_binarise_zeroes_the_diagonal_and_keeps_the_top_five_percent():
    rng = np.random.default_rng(5)
    a = rng.random((60, 60))
    a = (a + a.T) / 2
    np.fill_diagonal(a, 1.0)
    b = binarise(a, 95)
    assert not b.diagonal().any()
    assert abs(b.sum() - 0.05 * a.size) <= 2
    off = ~np.eye(60, dtype=bool)
    assert (a[b] >= a[off & ~b].max()).all()


def test_density_matrix_matches_pair_counts():
    rng = np.random.default_rng(11)
    nets = rng.random((3, 14, 14))
    nets = (nets + nets.transpose(0, 2, 1)) / 2
    complexes = {"Complex_7": [0, 1, 2, 3], "Complex_10": [4, 5, 6, 6], "Complex_2": [8, 9, 10, 11, 12, 13]}
    names = ["lung", "brain", "skin"]
    got = density_matrix(nets, names, complexes, percentile=80)
    assert list(got.index) == sorted(complexes)
    assert list(got.columns) == sorted(names)
    for t, name in enumerate(names):
        b = binarise(nets[t], 80)
        for c, members in complexes.items():
            pairs = list(itertools.combinations(range(len(members)), 2))
            hits = sum(b[members[i], members[j]] for i, j in pairs)
            assert np.isclose(got.loc[c, name], hits / len(pairs))


def test_table_with_corum_columns_round_trip(tmp_path):
    dens = pd.DataFrame({"a": [1.0, 0.0, 0.4], "b": [0.0, 0.5, 0.3]}, index=["Complex_0", "Complex_1", "Complex_2"])
    table = specific_complexes(dens, quantile=0.0)
    corum = pd.DataFrame({"complex_id": [11, 12, 13], "complex_name": ["x", "y", "z"]})
    full = with_corum_columns(table, corum)
    assert list(full["complex_name"]) == [corum.loc[int(c.split("_")[1]), "complex_name"] for c in full["Complex_ID_in_Model"]]
    path = tmp_path / "out.csv"
    write_table(full, path)
    back = pd.read_csv(path)
    assert (back.loc[back["Is_Exclusive"], "Fold_Change"] == EXCLUSIVE).all()
    assert back.loc[~back["Is_Exclusive"], "Fold_Change"].astype(float).notna().all()


def test_no_density_gives_an_empty_table_that_can_be_written(tmp_path):
    dens = pd.DataFrame(np.zeros((3, 2)), index=["Complex_0", "Complex_1", "Complex_2"], columns=["a", "b"])
    table = specific_complexes(dens)
    assert table.empty and list(table.columns)[:2] == ["Tissue", "Complex_ID_in_Model"]
    corum = pd.DataFrame({"complex_id": [11, 12, 13], "complex_name": ["x", "y", "z"]})
    write_table(with_corum_columns(table, corum), tmp_path / "none.csv")
    assert len(pd.read_csv(tmp_path / "none.csv")) == 0


def test_no_complex_passing_the_screen_gives_an_empty_table():
    dens = pd.DataFrame({"a": [0.5, 0.5], "b": [0.5, 0.5]}, index=["Complex_0", "Complex_1"])
    assert specific_complexes(dens, quantile=0.9, min_fold=1.5).empty
