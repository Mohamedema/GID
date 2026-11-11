import json

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from scipy.special import polygamma
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests

from gid.data import EdgeMatrix, upper_index
from gid.mutations import (MutationStudy, Settings, bh_adjust, bh_passing, exact_fit, fit_prior, limma_tests,
                           permutation_hits, permuted_sums, prepare, status_design, trigamma_inverse,
                           two_sample_welch, welch_se, welch_tests)


def make_design(seed=0, per_tissue=(12, 10, 14, 9), p_mut=(0.4, 0.3, 0.5, 0.25)):
    rng = np.random.default_rng(seed)
    codes = np.repeat(np.arange(len(per_tissue)), per_tissue)
    x = np.concatenate([(rng.random(n) < p).astype(float) for n, p in zip(per_tissue, p_mut)])
    return rng, codes, x


def design_matrix(x, codes):
    dummies = np.eye(codes.max() + 1)[codes]
    return np.column_stack([dummies, x])


def test_bh_matches_statsmodels():
    p = np.random.default_rng(0).random(500) ** 3
    assert np.allclose(bh_adjust(p), multipletests(p, method="fdr_bh")[1])


def test_two_sample_welch_matches_scipy():
    rng = np.random.default_rng(1)
    a, b = rng.normal(1, 2, 17), rng.normal(0, 1, 23)
    delta, p = two_sample_welch(a.mean(), a.var(ddof=1), 17, b.mean(), b.var(ddof=1), 23)
    ref = ttest_ind(a, b, equal_var=False)
    assert np.isclose(delta, a.mean() - b.mean()) and np.isclose(p, ref.pvalue)


def test_trigamma_inverse():
    for x in (0.05, 0.4, 1.7, 30.0):
        assert abs(float(polygamma(1, trigamma_inverse(x))) - x) < 1e-6 * max(1.0, x)


def test_limma_moderation_matches_inmoose():
    limma = pytest.importorskip("inmoose.limma")
    rng, codes, x = make_design(2)
    n, e = x.size, 400
    Y = rng.normal(size=(e, n)) * rng.uniform(0.5, 2, size=(e, 1)) + np.array([0, 1, -1, 2])[codes][None, :]
    Y[:25] += 0.7 * x[None, :]
    X = design_matrix(x, codes)
    cols = ["t0", "t1", "t2", "t3", "status"]
    fit = limma.eBayes(limma.lmFit(pd.DataFrame(Y, columns=[f"s{i}" for i in range(n)]), pd.DataFrame(X, columns=cols)))
    prep = prepare(Y.T.astype(np.float32).copy(), codes, 4)
    des = status_design(x, codes, 4)
    f = exact_fit(prep, x, codes, des)
    df_resid = n - 4 - 1
    mine = limma_tests(f["beta"], f["rss"] / df_resid, des["sxx"], df_resid)
    assert abs(mine["d0"] - float(fit.df_prior)) < 1e-3 * float(fit.df_prior)
    assert abs(mine["s02"] - float(fit.s2_prior)) < 1e-3 * float(fit.s2_prior)
    assert np.allclose(mine["t"], np.asarray(fit.t)[:, 4], rtol=2e-3, atol=2e-3)
    assert np.allclose(mine["p"], np.asarray(fit.p_value)[:, 4], rtol=5e-3, atol=1e-6)


def test_fit_prior_without_variation_between_edges():
    s2 = np.full(100, 2.0)
    s02, d0 = fit_prior(s2, 20)
    assert np.isfinite(s02) and (np.isinf(d0) or d0 > 1000 or d0 > 0)


def test_tissue_adjusted_fit_matches_ols():
    rng, codes, x = make_design(3)
    n, e = x.size, 30
    Y = rng.normal(size=(n, e)) + np.array([0, 2, -1, 1])[codes][:, None] + 0.6 * x[:, None]
    prep = prepare(Y.astype(np.float32).copy(), codes, 4)
    des = status_design(x, codes, 4)
    f = exact_fit(prep, x, codes, des)
    X = design_matrix(x, codes)
    for k in (0, 7, 29):
        ols = sm.OLS(Y[:, k], X).fit()
        assert abs(f["beta"][k] - ols.params[-1]) < 1e-4
        assert abs(f["rss"][k] - ols.ssr) < 1e-3
        mut = x == 1
        assert abs(f["rss_mut"][k] - (ols.resid[mut] ** 2).sum()) < 1e-3


def test_group_summaries_of_raw_scores():
    rng, codes, x = make_design(4)
    n, e = x.size, 12
    Y = rng.normal(size=(n, e)) + np.array([0, 2, -1, 1])[codes][:, None]
    prep = prepare(Y.astype(np.float32).copy(), codes, 4)
    des = status_design(x, codes, 4)
    f = exact_fit(prep, x, codes, des)
    m = x == 1
    assert np.allclose(f["mean_m"], Y[m].mean(axis=0), atol=1e-4)
    assert np.allclose(f["mean_w"], Y[~m].mean(axis=0), atol=1e-4)
    assert np.allclose(f["var_m"], Y[m].var(axis=0, ddof=1), atol=1e-3)
    assert np.allclose(f["var_w"], Y[~m].var(axis=0, ddof=1), atol=1e-3)


def test_unequal_variance_test_follows_its_definition():
    rng, codes, x = make_design(5)
    n = x.size
    sd = np.where(x == 1, 2.5, 0.7)
    y = (rng.normal(size=n) * sd + np.array([0, 2, -1, 1])[codes] + 0.4 * x)[:, None]
    prep = prepare(y.astype(np.float32).copy(), codes, 4)
    des = status_design(x, codes, 4)
    f = exact_fit(prep, x, codes, des)
    X = design_matrix(x, codes)
    ols = sm.OLS(y[:, 0], X).fit()
    h = ols.get_influence().hat_matrix_diag
    mut = x == 1
    nu_m, nu_w = (1 - h[mut]).sum(), (1 - h[~mut]).sum()
    xr = des["xr"]
    vm = (xr[mut] ** 2).sum() * ((ols.resid[mut] ** 2).sum() / nu_m) / des["sxx"] ** 2
    vw = (xr[~mut] ** 2).sum() * ((ols.resid[~mut] ** 2).sum() / nu_w) / des["sxx"] ** 2
    se, df = welch_se(f["rss_mut"], f["rss"] - f["rss_mut"], des)
    assert abs(se[0] - np.sqrt(vm + vw)) < 1e-4
    assert abs(df[0] - (vm + vw) ** 2 / (vm ** 2 / nu_m + vw ** 2 / nu_w)) < 1e-2


def test_leave_one_tissue_out_delta_equals_a_refit():
    rng, codes, x = make_design(6)
    n, e = x.size, 10
    Y = rng.normal(size=(n, e)) + np.array([0, 2, -1, 1])[codes][:, None] + 0.5 * x[:, None]
    prep = prepare(Y.astype(np.float32).copy(), codes, 4)
    des = status_design(x, codes, 4)
    f = exact_fit(prep, x, codes, des)
    sxx_t = np.bincount(codes, weights=des["xr"] ** 2, minlength=4)
    for g in range(4):
        keep = codes != g
        X = design_matrix(x[keep], np.unique(codes[keep], return_inverse=True)[1])
        for k in (0, 5):
            refit = sm.OLS(Y[keep, k], X).fit().params[-1]
            delta = (f["beta"][k] * des["sxx"] - f["by_tissue"][g, k]) / (des["sxx"] - sxx_t[g])
            assert abs(delta - refit) < 1e-4


def test_permutation_shortcut_gives_the_same_edges_as_testing_every_edge():
    rng, codes, x = make_design(7)
    n, e = x.size, 3000
    Y = (rng.normal(size=(n, e)) * rng.uniform(0.5, 2, size=e) + np.array([0, 2, -1, 1])[codes][:, None]).astype(np.float32)
    Y[:, :60] += 1.0 * x[:, None]
    prep = prepare(Y.copy(), codes, 4)
    des = status_design(x, codes, 4)
    f = exact_fit(prep, x, codes, des)
    df_resid = n - 4 - 1
    tau = 0.2 * float(np.median(np.sqrt(f["rss"] / df_resid)))
    for test in ("welch", "limma"):
        cfg = Settings(primary_test=test)
        XR = des["xr"][None, :].astype(np.float32)
        M = des["mut"][None, :].astype(np.float32)
        s_xy, s_xy_mut, s_yy_mut = permuted_sums(prep, XR, XR * M, M, np.arange(e))
        beta = s_xy[0].astype(np.float64) / des["sxx"]
        rss = np.maximum(prep["ssw"] - beta ** 2 * des["sxx"], 0.0)
        rss_m = np.clip(s_yy_mut[0] - 2.0 * beta * s_xy_mut[0] + beta ** 2 * des["a_mut"], 0.0, rss)
        hit, hit_t = permutation_hits(beta, rss, rss_m, des, df_resid, tau, cfg)
        if test == "welch":
            full = welch_tests(beta, rss_m, rss - rss_m, des, tau=tau)
        else:
            full = limma_tests(beta, rss / df_resid, des["sxx"], df_resid, tau=tau)
        assert np.array_equal(hit, bh_adjust(full["p"]) < cfg.fdr_cut), test
        assert np.array_equal(hit_t, bh_adjust(full["p_treat"]) < cfg.fdr_cut), test
        assert hit.sum() > 10


def test_bh_passing_equals_bh_adjust():
    rng = np.random.default_rng(8)
    p = np.concatenate([rng.random(2000) ** 4, rng.random(500)])
    keep = np.flatnonzero(p < 0.05)
    got = np.zeros(p.size, dtype=bool)
    got[keep] = bh_passing(p[keep], p.size, 0.05)
    assert np.array_equal(got, bh_adjust(p) < 0.05)


G = 30
TISSUES = ["Lung", "Breast", "Skin", "Liver"]


def build_study(tmp_path, seed=0, confounded=False, planted=(1, 2, 3, 4, 5), n_per_tissue=30):
    rng = np.random.default_rng(seed)
    n = n_per_tissue * len(TISSUES)
    tissue = np.repeat(TISSUES, n_per_tissue)
    ids = [f"S{i:03d}" for i in range(n)]
    ii, jj = upper_index(G)
    shift = rng.normal(size=(len(TISSUES), ii.size)) * 0.5
    scores = rng.normal(size=(n, ii.size)) * 0.2 + shift[np.repeat(np.arange(len(TISSUES)), n_per_tissue)]
    status = np.zeros(n)
    for t in range(len(TISSUES)):
        rows = np.arange(t * n_per_tissue, (t + 1) * n_per_tissue)
        k = 24 if (confounded and t == 0) else (3 if confounded else 10)
        status[rng.choice(rows, k, replace=False)] = 1
    if not confounded:
        for j in planted:
            col = np.flatnonzero((ii == 0) & (jj == j))[0]
            scores[:, col] += 0.9 * status * (1 if j % 2 else -1)
    else:
        # the mutated lines are mostly lung and lung scores are higher for many edges
        scores[:n_per_tissue] += 0.6
    store = EdgeMatrix.create(str(tmp_path / "scores.npy"), n, G)
    store.x[:] = scores.astype(np.float32)
    genes = [f"GENE{i + 1:03d}" for i in range(G)]
    ref = np.zeros((G, G))
    for a in range(G - 1):
        ref[a, a + 1] = ref[a + 1, a] = 0.9               # a chain 0-1-2-...
    ref[0, 7] = ref[7, 0] = 0.95
    meta = pd.DataFrame({"tissue": tissue}, index=ids)
    geno = pd.DataFrame([status, np.zeros(n)], index=["GENE001", "GENE002"], columns=ids)
    return store, genes, meta, ref, geno, status, ids


def test_planted_effects_are_found_with_the_right_sign(tmp_path):
    store, genes, meta, ref, geno, status, ids = build_study(tmp_path)
    cfg = Settings(n_perm=20, min_mutated=5)
    study = MutationStudy(store, genes, meta, ref, geno, cfg)
    out = str(tmp_path / "out")
    s = study.analyse_gene("GENE001", out, verbose=False)
    assert s["status"] == "ok" and s["n_mutated"] == int(status.sum())
    edges = pd.read_csv(f"{out}/GENE001/GENE001_edges.csv.gz")
    planted = edges[(edges.Gene_A == "GENE001") & edges.Gene_B.isin([f"GENE{j + 1:03d}" for j in (1, 2, 3, 4, 5)])]
    assert (planted.Sig == 1).all()
    assert (np.sign(planted.Delta.to_numpy()) == np.array([1 if int(g[4:]) % 2 == 0 else -1 for g in planted.Gene_B])).all()
    assert np.allclose(planted.Delta.abs(), 0.9, atol=0.15)
    assert s["n_altered_degree1"] >= 5
    assert s["perm_max_altered_degree1"] <= 2
    perm = pd.read_csv(f"{out}/GENE001/GENE001_permutation.csv")
    assert len(perm) == 20


def test_degrees_follow_the_reference_network(tmp_path):
    store, genes, meta, ref, geno, status, ids = build_study(tmp_path, seed=3)
    study = MutationStudy(store, genes, meta, ref, geno, Settings(n_perm=0))
    out = str(tmp_path / "out")
    study.analyse_gene("GENE001", out, n_perm=0, verbose=False)
    edges = pd.read_csv(f"{out}/GENE001/GENE001_edges.csv.gz")
    pos = {g: i for i, g in enumerate(genes)}
    dist = study.graph_distance(0)
    assert dist[1] == 1 and dist[2] == 2 and dist[7] == 1 and dist[8] == 2
    ref_pairs = edges[edges.Degree > 1]
    for r in ref_pairs.head(60).itertuples():
        d = 1 + min(dist[pos[r.Gene_A]], dist[pos[r.Gene_B]])
        assert r.Degree == min(d, 4)
    d1 = edges[edges.Degree == 1]
    assert len(d1) == G - 1 and (d1.Gene_A == "GENE001").all()
    assert set(d1[d1.Is_Reference == 1].Gene_B) == {"GENE002", "GENE008"}


def test_tissue_adjustment_removes_confounded_effects(tmp_path):
    store, genes, meta, ref, geno, status, ids = build_study(tmp_path, seed=5, confounded=True)
    study = MutationStudy(store, genes, meta, ref, geno, Settings(n_perm=0))
    out = str(tmp_path / "out")
    s = study.analyse_gene("GENE001", out, n_perm=0, verbose=False)
    assert s["n_altered_unadjusted"] > 30
    assert s["n_altered"] <= 3
    assert s["heaviest_tissue"] in TISSUES


def test_run_skips_finished_genes_and_lists_the_rest(tmp_path, capsys):
    store, genes, meta, ref, geno, status, ids = build_study(tmp_path, seed=4)
    study = MutationStudy(store, genes, meta, ref, geno, Settings(n_perm=5))
    out = str(tmp_path / "all")
    first = study.run(out, genes="all", verbose=False)
    table = pd.read_csv(f"{out}/all_genes_summary.csv")
    assert set(table.gene) == {"GENE001", "GENE002"} or set(table.gene) == {"GENE001"}
    assert first["GENE001"]["status"] == "ok"
    assert table.loc[table.gene == "GENE002", "status"].iloc[0].startswith("skipped")
    capsys.readouterr()
    study.run(out, genes="all", verbose=False)
    assert "already done" in capsys.readouterr().out
    with open(f"{out}/GENE001/GENE001_summary.json") as fh:
        assert json.load(fh)["n_permutations"] == 5


def test_wrong_inputs_are_reported(tmp_path):
    store, genes, meta, ref, geno, status, ids = build_study(tmp_path)
    with pytest.raises(KeyError):
        MutationStudy(store, genes, meta, ref, geno, Settings(covariate="cancer_type"))
    with pytest.raises(ValueError):
        MutationStudy(store, genes, meta.iloc[:-1], ref, geno, Settings())
    with pytest.raises(ValueError):
        MutationStudy(store, genes, meta, ref * 0, geno, Settings())
