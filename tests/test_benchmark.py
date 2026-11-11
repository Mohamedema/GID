import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score

from gid.benchmark import (ArraySource, FairBaselines, Settings, ch_db_from_gram, imputed_like, layer_blocks,
                           pass1, pass2, pca_scores, run_benchmark, top_pct_bands, wilson)
from gid.data import OmicsData


def corr_or_zero(A):
    """Correlation matrix of the rows of A; a constant row has correlation 0 with everything."""
    A = A - A.mean(axis=1, keepdims=True)
    s = np.sqrt((A * A).sum(axis=1))
    ok = s > 1e-9
    Z = np.where(ok[:, None], A / np.where(ok, s, 1.0)[:, None], 0.0)
    return Z @ Z.T


def reference_networks(layers, obs, delta=None):
    """Plain numpy definition of the baselines: loops over layers, samples and leave-one-out."""
    G, N = layers[0].shape
    iu = np.triu_indices(G, 1)
    E = iu[0].size
    num = {m: np.zeros((N, E)) for m in ("LIONESS", "SSN", "BONOBO")}
    cnt = np.zeros((N, E))
    for l, Yfull in enumerate(layers):
        rows = np.flatnonzero(obs[:, l])
        n = rows.size
        Y = Yfull[:, rows]
        vary = Y.var(axis=1) > 1e-12
        R_all = corr_or_zero(Y)
        W = {m: np.zeros((n, E)) for m in num}
        mu = Y[vary].mean(axis=1)
        for k in range(n):
            Yk = np.delete(Y, k, axis=1)
            R_loo = corr_or_zero(Yk)
            W["SSN"][k] = (R_all - R_loo)[iu]
            W["LIONESS"][k] = (n * R_all - (n - 1) * R_loo)[iu]
            cov = np.cov(Yk[vary])
            dg = np.diag(cov)
            dl = 1 / (3 + 2 * np.sqrt(dg).mean() / dg.var()) if delta is None else delta
            d = Y[vary, k] - mu
            ss = dl * np.outer(d, d) + (1 - dl) * cov
            sdv = np.sqrt(np.diag(ss))
            Rb = np.zeros((G, G))
            Rb[np.ix_(vary, vary)] = ss / np.outer(sdv, sdv)
            W["BONOBO"][k] = Rb[iu]
        pair = (vary[iu[0]] & vary[iu[1]]).astype(float)
        for m in num:
            W[m] -= W[m].mean(axis=0)
            num[m][rows] += W[m]
        cnt[rows] += pair
    return {m: num[m] / np.maximum(cnt, 1.0) for m in num}


@pytest.fixture(scope="module")
def toy():
    rng = np.random.default_rng(0)
    G, N, L = 14, 60, 3
    layers = []
    for l in range(L):
        Y = rng.normal(size=(G, N)) + 2 * rng.normal(size=(G, 1))
        Y[:5] += 1.2 * rng.normal(size=(1, N))
        layers.append(Y)
    obs = np.ones((N, L), dtype=bool)
    obs[rng.choice(np.arange(10, N), 8, replace=False), 1] = False
    obs[rng.choice(np.arange(10, N), 12, replace=False), 2] = False
    layers[1][3] = 0.0                      # gene 3 is absent from layer 1
    layers[2][6] = 0.0
    layers[2][6, 7] = 1.0                   # a copy-number-like gene with a single non-zero sample
    layers[2][9] = 0.5                      # constant gene
    return layers, obs, G, N, L


METHODS3 = ["LIONESS", "SSN", "BONOBO"]


def test_baselines_match_the_plain_definition(toy):
    layers, obs, G, N, L = toy
    E = G * (G - 1) // 2
    ref = reference_networks(layers, obs)
    fb = FairBaselines(layers, obs, METHODS3, Settings())
    got = fb.blocks(np.arange(E))
    for m in ref:
        assert np.abs(got[m] - ref[m]).max() < 1e-8, m


def test_values_of_missing_layers_are_never_read(toy):
    layers, obs, G, N, L = toy
    E = G * (G - 1) // 2
    rng = np.random.default_rng(5)
    base = FairBaselines(layers, obs, METHODS3, Settings()).blocks(np.arange(E))
    junk = [Y.copy() for Y in layers]
    for l in range(L):
        junk[l][:, ~obs[:, l]] = 50 * rng.normal(size=(G, int((~obs[:, l]).sum())))
    other = FairBaselines(junk, obs, METHODS3, Settings()).blocks(np.arange(E))
    assert max(np.abs(other[m] - base[m]).max() for m in base) == 0.0


def test_any_subset_of_edges_gives_the_same_values(toy):
    layers, obs, G, N, L = toy
    E = G * (G - 1) // 2
    fb = FairBaselines(layers, obs, METHODS3, Settings())
    full = fb.blocks(np.arange(E))
    sub = np.sort(np.random.default_rng(1).choice(E, 17, replace=False))
    part = fb.blocks(sub)
    assert max(np.abs(part[m] - full[m][:, sub]).max() for m in full) < 1e-12


def test_complete_data_equals_the_average_of_raw_layer_networks(toy):
    layers, obs, G, N, L = toy
    rng = np.random.default_rng(2)
    plain = [Y.copy() for Y in layers]
    plain[1][3] = rng.normal(size=N)
    plain[2][6] = rng.normal(size=N)
    plain[2][9] = rng.normal(size=N)
    iu = np.triu_indices(G, 1)
    E = iu[0].size
    old = {m: np.zeros((N, E)) for m in ("LIONESS", "SSN")}
    for Y in plain:
        R_all = np.corrcoef(Y)
        for q in range(N):
            R_loo = np.corrcoef(np.delete(Y, q, axis=1))
            old["LIONESS"][q] += (N * R_all - (N - 1) * R_loo)[iu] / L
            old["SSN"][q] += (R_all - R_loo)[iu] / L
    got = FairBaselines(plain, np.ones((N, L), dtype=bool), ["LIONESS", "SSN"], Settings()).blocks(np.arange(E))
    for m in old:
        assert np.abs(got[m] - (old[m] - old[m].mean(axis=0))).max() < 1e-8


def test_coverage_classes_follow_the_layers_that_measure_both_genes(toy):
    layers, obs, G, N, L = toy
    fb = FairBaselines(layers, obs, METHODS3, Settings())
    expected = np.zeros(fb.E, dtype=np.int32)
    for l in range(L):
        vary = layers[l][:, obs[:, l]].var(axis=1) > 1e-12
        expected += (vary[fb.I] & vary[fb.J]).astype(np.int32) << l
    assert (fb.strata == expected).all()
    assert len(np.unique(expected)) > 3


def test_top_percent_bands_within_groups():
    rng = np.random.default_rng(3)
    v = rng.random(1000) * np.repeat([1.0, 5.0, 25.0, 0.0], 250)
    grp = np.repeat([1, 2, 3, 4], 250)
    b5, b20 = top_pct_bands(v, [5, 20], grp)
    for g in (1, 2, 3):
        in_g = grp == g
        assert b5[in_g].sum() == (v[in_g] >= np.percentile(v[in_g], 95)).sum()
        assert (b5 | b20)[in_g].sum() == (v[in_g] >= np.percentile(v[in_g], 80)).sum()
    assert not (b5 | b20)[grp == 4].any()
    assert not (b5 & b20).any()


def test_streamed_statistics_gram_matrices_and_cluster_indices():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(60, 500)) * rng.uniform(0.2, 3, 500) + np.repeat(rng.normal(size=(4, 500)), 15, axis=0)
    lab = np.repeat(list("abcd"), 15)
    src = ArraySource(X, "T")
    rows = np.arange(5, 60)
    st = pass1(src, {"T": {"all": None, "sub": rows}}, "test", chunk=77, quiet=True)["T"]
    assert np.allclose(st["all"][0], X.mean(0)) and np.allclose(st["all"][1], X.var(0))
    assert np.allclose(st["sub"][0], X[rows].mean(0)) and np.allclose(st["sub"][1], X[rows].var(0))
    sel = top_pct_bands(st["all"][1], [20])[0]
    sel2 = top_pct_bands(st["sub"][1], [10])[0]
    jobs = {"T": {"a": dict(rows=None, mean=st["all"][0], std=np.sqrt(st["all"][1]), sel=sel),
                  "b": dict(rows=rows, mean=st["sub"][0], std=None, sel=sel2)}}
    Ks = pass2(src, jobs, "test", chunk=33, quiet=True)["T"]
    Z = (X[:, sel] - X[:, sel].mean(0)) / X[:, sel].std(0)
    assert np.allclose(Ks["a"], Z @ Z.T)
    Zb = X[rows][:, sel2] - X[rows][:, sel2].mean(0)
    assert np.allclose(Ks["b"], Zb @ Zb.T)
    assert np.allclose(pdist(pca_scores(Ks["a"], 5)), pdist(PCA(5, svd_solver="full").fit_transform(Z)), atol=1e-8)
    ch, db = ch_db_from_gram(Ks["a"], lab)
    assert abs(ch - calinski_harabasz_score(Z, lab)) < 1e-8
    assert abs(db - davies_bouldin_score(Z, lab)) < 1e-8


def test_wilson_interval():
    lo, hi = wilson(50, 100)
    assert abs(lo - 0.4038) < 1e-3 and abs(hi - 0.5962) < 1e-3


def test_imputed_blocks_are_recognised():
    rng = np.random.default_rng(0)
    blk = rng.normal(size=(8, 12))
    blk[:, [2, 3]] = blk.mean(axis=1, keepdims=True)
    blk[:, 9] = blk[:, 10] = 7.0
    flag = imputed_like(blk)
    assert flag[[2, 3, 9, 10]].all() and not flag[[0, 1, 4, 5]].any()


def toy_omics(n=64, g=16, seed=0):
    rng = np.random.default_rng(seed)
    tissues = np.array(["Lung", "Breast", "Skin", "Liver"])[np.arange(n) % 4]
    values = np.zeros((n, 5, g))
    mask = np.zeros((n, 5, g), dtype=bool)
    for l in range(5):
        base = rng.normal(size=(4, g)) * 2
        x = base[np.arange(n) % 4] + rng.normal(size=(n, g))
        lacking = rng.choice(np.arange(8, n), 6 if l else 0, replace=False)
        mask[lacking, l, :] = True
        x[lacking] = x[~np.isin(np.arange(n), lacking)].mean(axis=0)
        values[:, l, :] = x
    meta = pd.DataFrame({"tissue": tissues}, index=[f"S{i}" for i in range(n)])
    return OmicsData(values, mask, [f"G{i}" for i in range(g)], meta, np.zeros((g, g))), tissues


def test_layer_blocks_find_the_samples_that_have_each_layer():
    data, _ = toy_omics()
    cfg = Settings(min_layer_lines=30)
    blocks, observed, use = layer_blocks(data, cfg)
    assert observed.shape == (64, 5) and use == [0, 1, 2, 3, 4]
    assert observed[:, 0].all()
    assert (observed[:, 1:].sum(axis=0) == 58).all()
    assert (observed == ~data.mask.all(axis=2)).all()


def test_mask_and_data_that_disagree_are_rejected():
    data, _ = toy_omics()
    data.mask[:, 1, :] = ~data.mask[:, 1, :]
    with pytest.raises(ValueError):
        layer_blocks(data, Settings())


def test_benchmark_runs_end_to_end_and_separates_the_tissues(tmp_path):
    data, tissues = toy_omics(n=64, g=16)
    rng = np.random.default_rng(1)
    e = 16 * 15 // 2
    centres = rng.normal(size=(4, e))
    gid = (centres[np.arange(64) % 4] + 0.3 * rng.normal(size=(64, e))).astype(np.float32)
    cfg = Settings(n_perm=5, chunk=40, pct_list=(5, 25), n_pcs=10, n_pcs_list=(10, 20), min_c5_lines=1000, knn_k=5)
    res = run_benchmark(data, gid, str(tmp_path / "bm"), cfg, quick_genes=None)
    wide = pd.read_csv(tmp_path / "bm" / "Benchmark_clustering_index_GID_LIONESS_SSN_BONOBO.csv").set_index("method")
    assert set(wide.index) == {"GID", "LIONESS", "SSN", "BONOBO"}
    assert wide.loc["GID", "kNN_acc_Tissue"] > 0.95
    assert wide.loc["GID", "CH_Tissue"] > wide.loc["LIONESS", "CH_Tissue"]
    for name in ("clustering_all_results_long.csv", "layer_observation_table.csv", "run_info.json",
                 "fig_clustering_metrics.png"):
        assert (tmp_path / "bm" / name).exists(), name
    assert set(res.subset) >= {"all", "without_subset"}
