"""Differential association analysis of mutated genes, one folder per gene.

For every association (edge) of the test set the score is modelled as

    score ~ tissue + status        status: 1 = mutated, 0 = wild type

and Delta, the coefficient of status, is the mean difference between mutated and wild-type cell
lines of the same tissue. A tissue that holds only mutated or only wild-type lines adds nothing.

Test set of a gene, fixed before any result is looked at:
    degree 1   the gene with every other network gene
    degree 2+  every reference pair (score >= reference_min_score) that does not involve the gene;
               its degree is 1 + the shortest distance, in the reference network, from the gene to
               the nearer of the two genes (4 stands for "4 or more, or not connected")

One Benjamini-Hochberg correction covers the test set of a gene. Nothing is corrected across genes,
so a gene with a handful of altered edges can be chance: compare n_altered with perm_max_altered in
all_genes_summary.csv.

Two tests of the same Delta are computed for every edge:
    welch   mutated and wild-type lines may have different residual variances (Satterthwaite d.f.)
    limma   one residual variance per edge, moderated across edges as in limma (Smyth 2004)
primary_test chooses which one fills the columns p, FDR, FDR_treat and Sig. TREAT (McCarthy and
Smyth 2009) tests |Delta| against a minimum of treat_sd median residual SDs.

Checks on every gene: status is shuffled within tissue n_perm times and the whole procedure is
repeated (the edges that pass are the chance level), and Delta is re-estimated with each tissue
left out in turn.
"""
import argparse
import json
import os
import re
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import digamma, polygamma
from scipy.stats import fisher_exact, norm
from scipy.stats import t as t_dist

from .data import EdgeMatrix, load_omics


@dataclass
class Settings:
    covariate: str = "tissue"
    primary_test: str = "welch"
    reference_min_score: float = 0.7
    fdr_cut: float = 0.05
    treat_sd: float = 0.2
    min_mutated: int = 5
    min_per_stratum: int = 3
    n_perm: int = 200
    top_n: int = 24
    n_values: int = 4
    seed: int = 1
    perm_batch: int = 50
    max_reference_gb: float = 6.0
    block: int = 20000
    edge_table: str = "full"

    def identity(self):
        """The settings a finished gene must have been run with to be skipped on a restart."""
        return dict(covariate=self.covariate, primary_test=self.primary_test,
                    reference_min_score=self.reference_min_score, fdr_cut=self.fdr_cut, treat_sd=self.treat_sd,
                    n_permutations=int(self.n_perm), edge_table=self.edge_table)


def bh_adjust(p):
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(p, dtype=np.float64)
    n = p.size
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(ranked, 1.0)
    return out


def trigamma_inverse(x):
    """Solve trigamma(y) = x by Newton iteration, as limma does."""
    if x > 1e7:
        return 1.0 / np.sqrt(x)
    if x < 1e-6:
        return 1.0 / x
    y = 0.5 + 1.0 / x
    for _ in range(50):
        tri = float(polygamma(1, y))
        dif = tri * (1.0 - tri / x) / float(polygamma(2, y))
        y += dif
        if -dif / y < 1e-8:
            break
    return y


def fit_prior(s2, df1):
    """Prior variance s0^2 and prior degrees of freedom d0 (limma's fitFDist)."""
    x = np.asarray(s2, dtype=np.float64)
    x = np.maximum(x[np.isfinite(x)], 0.0)
    m = np.median(x)
    if m == 0:
        m = 1.0
    x = np.maximum(x, 1e-5 * m)
    e = np.log(x) - digamma(df1 / 2.0) + np.log(df1 / 2.0)
    emean = e.mean()
    evar = e.var(ddof=1) - float(polygamma(1, df1 / 2.0))
    if evar > 0:
        d0 = 2.0 * trigamma_inverse(evar)
        s02 = np.exp(emean + digamma(d0 / 2.0) - np.log(d0 / 2.0))
    else:
        d0 = np.inf
        s02 = np.exp(emean)
    return float(s02), float(d0)


def _tails(beta, se, df, tau):
    """Two-sided p-value, and the TREAT p-value for |beta| > tau when tau is given."""
    t = beta / se
    p = 2.0 * t_dist.sf(np.abs(t), df)
    if tau is None:
        return t, p, None
    a = np.abs(beta)
    return t, p, np.minimum(t_dist.sf((a - tau) / se, df) + t_dist.sf((a + tau) / se, df), 1.0)


def limma_se(s2, sxx, df_resid):
    """Standard error and d.f. of Delta with one residual variance per edge, moderated across edges."""
    s02, d0 = fit_prior(s2, df_resid)
    if np.isfinite(d0):
        s2_post = (d0 * s02 + df_resid * s2) / (d0 + df_resid)
        df_total = min(df_resid + d0, float(df_resid) * s2.size)
    else:
        s2_post = np.full(s2.shape, s02)
        df_total = float(df_resid) * s2.size
    return np.sqrt(s2_post / sxx), df_total, s02, d0


def limma_tests(beta, s2, sxx, df_resid, tau=None):
    se, df_total, s02, d0 = limma_se(s2, sxx, df_resid)
    t, p, p_treat = _tails(beta, se, df_total, tau)
    return dict(t=t, p=p, p_treat=p_treat, se=se, df=np.full(s2.shape, df_total), s02=s02, d0=d0)


def welch_se(rss_mut, rss_wt, des):
    """Standard error and Satterthwaite d.f. of Delta with separate residual variances for mutated and wild-type lines."""
    sxx = des["sxx"]
    vm = des["a_mut"] * (rss_mut / des["nu_mut"]) / sxx ** 2
    vw = des["a_wt"] * (rss_wt / des["nu_wt"]) / sxx ** 2
    var = np.maximum(vm + vw, 1e-300)
    df = var ** 2 / np.maximum(vm ** 2 / des["nu_mut"] + vw ** 2 / des["nu_wt"], 1e-300)
    return np.sqrt(var), df


def welch_tests(beta, rss_mut, rss_wt, des, tau=None):
    se, df = welch_se(rss_mut, rss_wt, des)
    t, p, p_treat = _tails(beta, se, df, tau)
    return dict(t=t, p=p, p_treat=p_treat, se=se, df=df)


def two_sample_welch(mean_m, var_m, n_m, mean_w, var_w, n_w):
    """Plain Welch t-test from group summaries (the test without any covariate)."""
    vm, vw = var_m / n_m, var_w / n_w
    var = np.maximum(vm + vw, 1e-300)
    df = var ** 2 / np.maximum(vm ** 2 / (n_m - 1) + vw ** 2 / (n_w - 1), 1e-300)
    delta = mean_m - mean_w
    return delta, 2.0 * t_dist.sf(np.abs(delta) / np.sqrt(var), df)


def bh_passing(p, m, fdr_cut):
    """Which of these p-values pass Benjamini-Hochberg at fdr_cut among m tests in all.

    p must hold every p-value below fdr_cut; larger ones can never pass, so they may be missing.
    """
    hit = np.zeros(p.size, dtype=bool)
    if p.size:
        order = np.argsort(p)
        ok = np.flatnonzero(p[order] * m / np.arange(1, p.size + 1) < fdr_cut)
        if ok.size:
            hit[order[:ok[-1] + 1]] = True
    return hit


def permutation_hits(beta, rss, rss_mut, des_p, df_resid, tau, cfg):
    """Edges that pass the primary test, and its TREAT version, for one permuted status.

    The result is the same as testing every edge, but p-values are only computed where they could
    pass: a t distribution has heavier tails than the normal, so p < fdr_cut needs |t| above the
    normal quantile.
    """
    z_two = float(norm.isf(cfg.fdr_cut / 2.0))
    z_one = float(norm.isf(cfg.fdr_cut))
    if cfg.primary_test == "welch":
        se, df = welch_se(rss_mut, rss - rss_mut, des_p)
    else:
        se, df_total, _, _ = limma_se(rss / df_resid, des_p["sxx"], df_resid)
        df = np.full(beta.shape, df_total)
    a = np.abs(beta)
    hit, hit_t = np.zeros(beta.size, dtype=bool), np.zeros(beta.size, dtype=bool)
    c = np.flatnonzero(a >= z_two * se)
    hit[c] = bh_passing(2.0 * t_dist.sf(a[c] / se[c], df[c]), beta.size, cfg.fdr_cut)
    c = np.flatnonzero(a - tau >= z_one * se)
    p_treat = np.minimum(t_dist.sf((a[c] - tau) / se[c], df[c]) + t_dist.sf((a[c] + tau) / se[c], df[c]), 1.0)
    hit_t[c] = bh_passing(p_treat, beta.size, cfg.fdr_cut)
    return hit, hit_t


def wilson(k, n, z=1.959964):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    den = 1 + z * z / n
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, (p + z * z / (2 * n)) / den - half), min(1.0, (p + z * z / (2 * n)) / den + half)


def agreement(a, b):
    """Share of pairs with the same sign (%), and the Pearson correlation."""
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if a.size == 0:
        return None, None
    same = round(100.0 * float(np.mean(np.sign(a) == np.sign(b))), 1)
    r = round(float(np.corrcoef(a, b)[0, 1]), 3) if a.size >= 3 and a.std() > 0 and b.std() > 0 else None
    return same, r


def prepare(Y, codes, n_groups, block=20000):
    """Remove the group (tissue) means from Y in place and keep what the fits need."""
    E = Y.shape[1]
    counts = np.bincount(codes, minlength=n_groups).astype(np.float64)
    grand = Y.mean(axis=0, dtype=np.float64)
    means = np.empty((n_groups, E), dtype=np.float64)
    for g in range(n_groups):
        rows = np.flatnonzero(codes == g)
        mu = Y[rows].mean(axis=0, dtype=np.float64)
        means[g] = mu
        Y[rows] -= mu.astype(np.float32)
    ssw = np.empty(E)
    for a in range(0, E, block):
        yb = Y[:, a:a + block].astype(np.float64)
        ssw[a:a + block] = np.einsum("ij,ij->j", yb, yb)
    ssb = (counts[:, None] * (means - grand) ** 2).sum(axis=0)
    return dict(Yr=Y, means=means, grand=grand, ssw=ssw, ssb=ssb)


def status_design(x, codes, n_groups):
    """Status with the tissue means removed, and the pieces of the unequal-variance test."""
    counts = np.bincount(codes, minlength=n_groups).astype(np.float64)
    xbar = np.bincount(codes, weights=x, minlength=n_groups) / counts
    xr = x - xbar[codes]
    sxx = float(xr @ xr)
    mut = x == 1
    lever = 1.0 / counts[codes] + (xr ** 2 / sxx if sxx > 0 else 0.0)
    return dict(xr=xr, sxx=sxx, xbar=xbar, counts=counts, mut=mut,
                nu_mut=float((1.0 - lever[mut]).sum()), nu_wt=float((1.0 - lever[~mut]).sum()),
                a_mut=float((xr[mut] ** 2).sum()), a_wt=float((xr[~mut] ** 2).sum()))


def exact_fit(prep, x, codes, des, block=20000):
    """Delta and residual sums of squares (all lines, mutated lines); what each tissue adds to Delta;
    and the group means and variances of the raw scores for the test without covariate."""
    Yr, xr, sxx, mut = prep["Yr"], des["xr"], des["sxx"], des["mut"]
    E, T = Yr.shape[1], des["counts"].size
    n_m, n_w = int(mut.sum()), int((~mut).sum())
    n_mt = np.bincount(codes[mut], minlength=T).astype(np.float64)
    onehot_mut = np.zeros((T, x.size))
    onehot_mut[codes[mut], np.flatnonzero(mut)] = 1.0
    xr_by_tissue = np.zeros((T, x.size))
    xr_by_tissue[codes, np.arange(x.size)] = xr
    out = {k: np.empty(E) for k in ("beta", "rss", "rss_mut", "mean_m", "var_m", "mean_w", "var_w")}
    out["by_tissue"] = np.empty((T, E))
    for a in range(0, E, block):
        sl = slice(a, a + block)
        yb = Yr[:, sl].astype(np.float64)
        b = (xr @ yb) / sxx
        r2 = (yb - np.outer(xr, b)) ** 2
        out["beta"][sl], out["rss"][sl], out["rss_mut"][sl] = b, r2.sum(axis=0), r2[mut].sum(axis=0)
        out["by_tissue"][:, sl] = xr_by_tissue @ yb
        mb = prep["means"][:, sl]
        s_t = onehot_mut @ yb
        sum_m = s_t.sum(axis=0) + n_mt @ mb
        sq_m = (yb[mut] ** 2).sum(axis=0) + 2.0 * (mb * s_t).sum(axis=0) + n_mt @ mb ** 2
        sum_all = des["counts"] @ mb
        sq_all = prep["ssw"][sl] + des["counts"] @ mb ** 2
        out["mean_m"][sl] = sum_m / n_m
        out["mean_w"][sl] = (sum_all - sum_m) / n_w
        out["var_m"][sl] = np.maximum(sq_m - sum_m ** 2 / n_m, 0.0) / max(n_m - 1, 1)
        out["var_w"][sl] = np.maximum((sq_all - sq_m) - (sum_all - sum_m) ** 2 / n_w, 0.0) / max(n_w - 1, 1)
    return out


def permuted_sums(prep, XR, XM, M, columns, block=20000):
    """The three sums every permuted fit needs, for a batch of permutations and the chosen edges:
    sum of xr * score, the same over mutated lines only, and the sum of score^2 over mutated lines.
    XR: status with tissue means removed (permutations x cell lines); M: 1 for a mutated line; XM = XR * M."""
    Yr = prep["Yr"]
    E = Yr.shape[1]
    out = [np.empty((XR.shape[0], E), dtype=np.float32) for _ in range(3)]
    for a in range(0, E, block):
        yb = np.ascontiguousarray(Yr[:, a:a + block])
        out[0][:, a:a + block] = XR @ yb
        out[1][:, a:a + block] = XM @ yb
        out[2][:, a:a + block] = M @ (yb * yb)
    return [o[:, columns] for o in out]


def safe_name(name):
    return re.sub(r'[\\/:*?"<>|\s]', "_", str(name))


def direction(delta):
    return np.where(np.asarray(delta) > 0, "Gained", "Lost")


def read_genotypes(path):
    """Genes x cell lines table of 1 (mutated) and 0 (wild type)."""
    geno = pd.read_csv(path, index_col=0, low_memory=False)
    geno.index = geno.index.astype(str).str.strip().str.upper()
    geno.columns = geno.columns.astype(str).str.strip()
    if geno.index.duplicated().any():
        print(f"{int(geno.index.duplicated().sum())} duplicated gene rows in {path}: the first is used")
        geno = geno[~geno.index.duplicated(keep="first")]
    if geno.columns.duplicated().any():
        print(f"{int(geno.columns.duplicated().sum())} duplicated cell line columns in {path}: the first is used")
        geno = geno.loc[:, ~geno.columns.duplicated(keep="first")]
    return geno


class MutationStudy:
    """The association scores of all cell lines, a reference network and the mutation calls.

    store      EdgeMatrix with one row of association scores per cell line
    genes      network gene names, in the order of the matrix
    meta       DataFrame indexed by cell line, rows in the order of the store, with the covariate column
    reference  genes x genes scores of the reference network (STRING)
    genotypes  genes x cell lines table of mutation calls (read_genotypes)
    """

    def __init__(self, store, genes, meta, reference, genotypes, cfg=None, targets=None):
        self.cfg = cfg or Settings()
        if self.cfg.primary_test not in ("welch", "limma"):
            raise ValueError("primary_test must be 'welch' or 'limma'")
        if self.cfg.edge_table not in ("full", "compact"):
            raise ValueError("edge_table must be 'full' or 'compact'")
        self.store = store
        self.genes = [str(g) for g in genes]
        self.G = len(self.genes)
        self.gene_idx = {g.strip().upper(): i for i, g in enumerate(self.genes)}
        self.sample_ids = [str(s).strip() for s in meta.index]
        self.sid_pos = {s: i for i, s in enumerate(self.sample_ids)}
        if store.shape[0] != len(self.sample_ids):
            raise ValueError(f"{store.shape[0]} score rows but {len(self.sample_ids)} cell lines in the metadata")
        if self.cfg.covariate not in meta.columns:
            raise KeyError(f"no '{self.cfg.covariate}' column in the metadata; columns: {list(meta.columns)}")
        self.covariate = meta[self.cfg.covariate].to_numpy()
        self.geno = genotypes
        cols = [c for c in genotypes.columns if c in self.sid_pos]
        self.geno_cols = cols
        self.geno_pos = np.array([self.sid_pos[c] for c in cols], dtype=np.int64)
        ref = np.nan_to_num(np.asarray(reference, dtype=np.float32), nan=0.0)
        ref = np.maximum(ref, ref.T)
        np.fill_diagonal(ref, 0.0)
        self.ref_score = ref
        self.ref_adj = ref >= self.cfg.reference_min_score
        self.ref_ii, self.ref_jj = np.nonzero(np.triu(self.ref_adj, k=1))
        if self.ref_ii.size == 0:
            raise ValueError("no reference pairs at this cut-off: lower reference_min_score")
        self.targets = None
        if targets is not None:
            tab = targets.copy()
            sym = "symbol" if "symbol" in tab.columns else tab.columns[0]
            tab["_key"] = tab[sym].astype(str).str.strip().str.upper()
            self.targets = tab.drop(columns=[sym]).drop_duplicates("_key")
        self._ref_cache = {}

    def edge_values(self, ii, jj, positions):
        return self.store.pairs(ii, jj, positions)

    def graph_distance(self, source):
        """Shortest reference-network distance from one gene to every gene."""
        dist = np.full(self.G, np.inf)
        dist[source] = 0
        seen = np.zeros(self.G, dtype=bool)
        seen[source] = True
        frontier = seen.copy()
        d = 0
        while frontier.any():
            d += 1
            nxt = self.ref_adj[frontier].any(axis=0) & ~seen
            dist[nxt] = d
            seen |= nxt
            frontier = nxt
        return dist

    def reference_prepared(self, positions, codes, n_groups, say=print):
        """Tissue-adjusted reference-edge matrix, reused for every gene with the same cell lines."""
        key = (positions.tobytes(), codes.tobytes())
        if key not in self._ref_cache:
            gb = len(positions) * self.ref_ii.size * 4 / 1e9
            if gb > self.cfg.max_reference_gb:
                raise MemoryError(f"the reference-edge matrix needs {gb:.1f} GB ({self.ref_ii.size:,} pairs x "
                                  f"{len(positions)} cell lines): raise reference_min_score or max_reference_gb")
            self._ref_cache.clear()
            t0 = time.time()
            self._ref_cache[key] = prepare(self.edge_values(self.ref_ii, self.ref_jj, positions), codes, n_groups,
                                           self.cfg.block)
            say(f"  reference-edge matrix: {len(positions)} cell lines x {self.ref_ii.size:,} pairs "
                f"({gb:.2f} GB, {time.time() - t0:.0f} s)")
        return self._ref_cache[key]

    def cell_lines_of(self, gene_up):
        """Cell lines with a mutation call of 0 or 1 and a known tissue: positions, status, levels, codes."""
        status = pd.to_numeric(self.geno.loc[gene_up, self.geno_cols], errors="coerce").to_numpy(dtype=np.float64)
        ok = np.isin(status, (0.0, 1.0)) & ~pd.isna(self.covariate[self.geno_pos])
        order = np.argsort(self.geno_pos[ok])
        positions, x = self.geno_pos[ok][order], status[ok][order]
        levels, codes = np.unique(self.covariate[positions].astype(str), return_inverse=True)
        return positions, x, levels, codes.astype(np.int64)

    def mutated_lines_that_count(self, gene_up):
        """Mutated lines in tissues that also hold wild-type lines."""
        _, x, levels, codes = self.cell_lines_of(gene_up)
        n_mut_t = np.bincount(codes, weights=x, minlength=levels.size)
        n_all_t = np.bincount(codes, minlength=levels.size)
        return int(n_mut_t[(n_mut_t > 0) & (n_all_t > n_mut_t)].sum())

    def analyse_gene(self, gene, out_dir, n_perm=None, verbose=True):
        """Analysis of one gene: writes its folder and returns the summary as a dict."""
        cfg = self.cfg
        n_perm = cfg.n_perm if n_perm is None else n_perm
        say = print if verbose else (lambda *a, **k: None)
        G = self.G
        gene_up = str(gene).strip().upper()
        t_start = time.time()
        if gene_up not in self.geno.index or gene_up not in self.gene_idx:
            why = "not in the mutation table" if gene_up not in self.geno.index else "not a network gene"
            say(f"{gene_up}: {why}")
            return dict(gene=gene_up, status=why)
        target = self.gene_idx[gene_up]
        target_name = self.genes[target]

        positions, x, levels, codes = self.cell_lines_of(gene_up)
        T, N = levels.size, positions.size
        des = status_design(x, codes, T)
        n_mut_t = np.bincount(codes, weights=x, minlength=T)
        n_wt_t = des["counts"] - n_mut_t
        informative = (n_mut_t > 0) & (n_wt_t > 0)
        n_mut, n_wt = int(x.sum()), int(N - x.sum())
        n_mut_inf = int(n_mut_t[informative].sum())
        summary = dict(gene=target_name, n_cell_lines=int(N), n_mutated=n_mut, n_wild_type=n_wt, n_tissues=int(T),
                       n_informative_tissues=int(informative.sum()), n_mutated_in_informative_tissues=n_mut_inf)
        if n_mut_inf < cfg.min_mutated or des["sxx"] <= 0 or des["nu_mut"] <= 1:
            summary["status"] = (f"skipped: {n_mut_inf} mutated lines in tissues that also hold wild type "
                                 f"(need {cfg.min_mutated})")
            say(f"{target_name}: {summary['status']}")
            return summary
        df_resid = N - T - 1
        sxx = des["sxx"]
        sxx_t = np.bincount(codes, weights=des["xr"] ** 2, minlength=T)
        heavy = int(np.argmax(sxx_t))

        say("=" * 78)
        say(f"{target_name}: {n_mut} mutated, {n_wt} wild type, {T} tissues "
            f"({int(informative.sum())} hold both; {n_mut_inf} mutated lines count)")
        tissue_table = pd.DataFrame({
            "tissue": levels, "n_mutated": n_mut_t.astype(int), "n_wild_type": n_wt_t.astype(int),
            "informative": informative.astype(int), "weight_pct": np.round(100 * sxx_t / sxx, 1),
        }).sort_values(["n_mutated", "n_wild_type"], ascending=False)
        say("  share of the estimate carried by each tissue (top 6): " + ", ".join(
            f"{r.tissue} {r.weight_pct:.0f}%" for r in tissue_table.sort_values("weight_pct", ascending=False).head(6).itertuples()))

        # test set: degree 1 (the gene with every other gene) and the reference pairs elsewhere
        others = np.delete(np.arange(G), target)
        prep1 = prepare(self.edge_values(np.full(G - 1, target), others, positions), codes, T, cfg.block)
        prep2 = self.reference_prepared(positions, codes, T, say)
        keep2 = (self.ref_ii != target) & (self.ref_jj != target)
        dist = self.graph_distance(target)
        deg2 = 1 + np.minimum(dist[self.ref_ii[keep2]], dist[self.ref_jj[keep2]])
        deg2 = np.where(np.isfinite(deg2), np.minimum(deg2, 4), 4).astype(int)

        f1, f2 = exact_fit(prep1, x, codes, des, cfg.block), exact_fit(prep2, x, codes, des, cfg.block)

        def both(key):
            return np.concatenate([f1[key], f2[key][keep2]])

        beta, rss, rss_mut = both("beta"), both("rss"), both("rss_mut")
        gene_a = np.concatenate([np.full(G - 1, target), self.ref_ii[keep2]])
        gene_b = np.concatenate([others, self.ref_jj[keep2]])
        degree = np.concatenate([np.ones(G - 1, dtype=int), deg2])
        is_ref = np.concatenate([self.ref_adj[target, others], np.ones(int(keep2.sum()), dtype=bool)])
        total_var = np.concatenate([prep1["ssw"] + prep1["ssb"], (prep2["ssw"] + prep2["ssb"])[keep2]]) / (N - 1)

        # edges that hardly vary carry no information and are left out before testing
        tested = total_var > 1e-6 * np.median(total_var)
        n_flat = int((~tested).sum())
        beta, rss, rss_mut = beta[tested], rss[tested], rss_mut[tested]
        by_tissue = np.concatenate([f1["by_tissue"], f2["by_tissue"][:, keep2]], axis=1)[:, tested]
        s2 = rss / df_resid
        tau = cfg.treat_sd * float(np.median(np.sqrt(s2)))
        lim = limma_tests(beta, s2, sxx, df_resid, tau=tau)
        wel = welch_tests(beta, rss_mut, rss - rss_mut, des, tau=tau)
        main = wel if cfg.primary_test == "welch" else lim
        delta_u, p_u = two_sample_welch(both("mean_m")[tested], both("var_m")[tested], n_mut,
                                        both("mean_w")[tested], both("var_w")[tested], n_wt)
        var_ratio = (rss_mut / des["nu_mut"]) / np.maximum((rss - rss_mut) / des["nu_wt"], 1e-300)

        def without_tissue(g):
            """Delta re-estimated without tissue g (None when no other tissue holds both groups)."""
            rest = sxx - sxx_t[g]
            return (beta * sxx - by_tissue[g]) / rest if rest > 1e-9 * sxx else None
        delta_wo_heavy = without_tissue(heavy)

        res = pd.DataFrame({
            "Gene_A": np.array(self.genes, dtype=object)[gene_a[tested]],
            "Gene_B": np.array(self.genes, dtype=object)[gene_b[tested]],
            "Degree": degree[tested],
            "Is_Reference": is_ref[tested].astype(int),
            "Ref_Score": np.round(self.ref_score[gene_a[tested], gene_b[tested]], 3),
            "Delta": beta, "SD": np.sqrt(s2), "Std_Effect": beta / np.sqrt(s2),
            "p": main["p"], "FDR": bh_adjust(main["p"]),
            "p_treat": main["p_treat"], "FDR_treat": bh_adjust(main["p_treat"]),
            "t_welch": wel["t"], "df_welch": wel["df"], "p_welch": wel["p"], "FDR_welch": bh_adjust(wel["p"]),
            "FDR_treat_welch": bh_adjust(wel["p_treat"]),
            "t_limma": lim["t"], "p_limma": lim["p"], "FDR_limma": bh_adjust(lim["p"]),
            "FDR_treat_limma": bh_adjust(lim["p_treat"]),
            "Var_Ratio": var_ratio,
            "Delta_unadjusted": delta_u, "FDR_unadjusted": bh_adjust(p_u),
            "Delta_without_top_tissue": np.nan if delta_wo_heavy is None else delta_wo_heavy,
        })
        res["Sig"] = (res["FDR"] < cfg.fdr_cut).astype(int)
        sig = res["Sig"].to_numpy() == 1
        sig_treat = res["FDR_treat"].to_numpy() < cfg.fdr_cut
        deg_t = res["Degree"].to_numpy()
        d1 = deg_t == 1
        refm = res["Is_Reference"].to_numpy() == 1
        n_sig, n_sig1 = int(sig.sum()), int((sig & d1).sum())

        # one tissue left out at a time: does Delta of the altered edges hold?
        loto_rows = []
        for g in np.flatnonzero(sxx_t > 0):
            b_wo = without_tissue(g)
            if b_wo is None:
                continue
            same, r = agreement(beta[sig], b_wo[sig])
            same1, r1 = agreement(beta[sig & d1], b_wo[sig & d1])
            loto_rows.append(dict(tissue_left_out=levels[g], n_mutated=int(n_mut_t[g]), n_wild_type=int(n_wt_t[g]),
                                  weight_pct=round(100 * sxx_t[g] / sxx, 1), n_altered=n_sig, same_sign_pct=same, r=r,
                                  n_altered_degree1=n_sig1, same_sign_pct_degree1=same1, r_degree1=r1))
        loto = pd.DataFrame(loto_rows, columns=["tissue_left_out", "n_mutated", "n_wild_type", "weight_pct", "n_altered",
                                                "same_sign_pct", "r", "n_altered_degree1", "same_sign_pct_degree1",
                                                "r_degree1"])
        loto = loto.sort_values("weight_pct", ascending=False).reset_index(drop=True)

        # known partners among degree 1: are reference neighbours altered more often?
        a = int((sig & d1 & refm).sum())
        b = int((~sig & d1 & refm).sum())
        c = int((sig & d1 & ~refm).sum())
        d = int((~sig & d1 & ~refm).sum())
        odds, p_fisher = fisher_exact([[a, b], [c, d]]) if (a + b) and (c + d) else (np.nan, np.nan)

        # permutation: shuffle status within tissue and repeat everything
        perm_cols = (["permutation", "n_sig", "n_sig_degree1"] + [f"n_sig_ref_degree{g}" for g in (1, 2, 3, 4)]
                     + ["n_sig_treat", "n_sig_treat_degree1"] + [f"n_sig_treat_ref_degree{g}" for g in (1, 2, 3, 4)])
        perm_rows = []
        if n_perm > 0:
            rng = np.random.default_rng(cfg.seed)
            groups = [np.flatnonzero(codes == g) for g in range(T)]
            cols1 = np.flatnonzero(tested[:G - 1])
            cols2 = np.flatnonzero(keep2)[tested[G - 1:]]
            ssw_p = np.concatenate([prep1["ssw"][cols1], prep2["ssw"][cols2]])
            for start in range(0, n_perm, cfg.perm_batch):
                designs = []
                for _ in range(min(cfg.perm_batch, n_perm - start)):
                    xp = x.copy()
                    for rows in groups:
                        xp[rows] = x[rng.permutation(rows)]
                    designs.append(status_design(xp, codes, T))
                XR = np.array([dd["xr"] for dd in designs], dtype=np.float32)
                M = np.array([dd["mut"] for dd in designs], dtype=np.float32)
                XM = XR * M
                s_xy, s_xy_mut, s_yy_mut = (np.concatenate([a1, a2], axis=1) for a1, a2 in zip(
                    permuted_sums(prep1, XR, XM, M, cols1, cfg.block), permuted_sums(prep2, XR, XM, M, cols2, cfg.block)))
                for j, des_p in enumerate(designs):
                    bp = s_xy[j].astype(np.float64) / des_p["sxx"]
                    rp = np.maximum(ssw_p - bp ** 2 * des_p["sxx"], 0.0)
                    mp = np.clip(s_yy_mut[j] - 2.0 * bp * s_xy_mut[j] + bp ** 2 * des_p["a_mut"], 0.0, rp)
                    tau_p = cfg.treat_sd * float(np.median(np.sqrt(rp / df_resid)))
                    hit, hit_t = permutation_hits(bp, rp, mp, des_p, df_resid, tau_p, cfg)
                    perm_rows.append(dict(
                        permutation=start + j + 1, n_sig=int(hit.sum()), n_sig_degree1=int((hit & d1).sum()),
                        **{f"n_sig_ref_degree{g}": int((hit & refm & (deg_t == g)).sum()) for g in (1, 2, 3, 4)},
                        n_sig_treat=int(hit_t.sum()), n_sig_treat_degree1=int((hit_t & d1).sum()),
                        **{f"n_sig_treat_ref_degree{g}": int((hit_t & refm & (deg_t == g)).sum()) for g in (1, 2, 3, 4)}))
        perm = pd.DataFrame(perm_rows, columns=perm_cols)

        # share of altered reference edges by degree
        deg_rows = []
        for g in (1, 2, 3, 4):
            m = refm & (deg_t == g)
            n_g, k_g = int(m.sum()), int((sig & m).sum())
            lo, hi = wilson(k_g, n_g)
            rowd = dict(degree=g, label={1: "1", 2: "2", 3: "3", 4: "4+"}[g], n_tested=n_g, n_altered=k_g,
                        pct_altered=100 * k_g / n_g if n_g else np.nan, ci_low=100 * lo, ci_high=100 * hi)
            if len(perm) and n_g:
                pct = 100 * perm[f"n_sig_ref_degree{g}"].to_numpy() / n_g
                rowd.update(perm_median=float(np.median(pct)), perm_q975=float(np.quantile(pct, 0.975)),
                            perm_max=float(pct.max()))
            deg_rows.append(rowd)
        degrees = pd.DataFrame(deg_rows)

        f_w, f_l, f_u = res["FDR_welch"] < cfg.fdr_cut, res["FDR_limma"] < cfg.fdr_cut, res["FDR_unadjusted"] < cfg.fdr_cut
        heavy_row = loto[loto["tissue_left_out"] == levels[heavy]]
        worst = loto.dropna(subset=["same_sign_pct"]).sort_values("same_sign_pct").head(1)
        summary.update(
            status="ok", **cfg.identity(),
            n_tested=int(tested.sum()), n_tested_degree1=int(d1.sum()), n_reference_pairs_tested=int((~d1).sum()),
            n_left_out_no_variation=n_flat, residual_df=int(df_resid),
            limma_prior_df=None if not np.isfinite(lim["d0"]) else round(lim["d0"], 1),
            median_residual_sd=round(float(np.median(res["SD"])), 4), treat_minimum_delta=round(tau, 4),
            variance_ratio_mutated_over_wt_median=round(float(np.median(var_ratio)), 3),
            n_altered=n_sig, n_altered_degree1=n_sig1,
            n_gained_degree1=int((sig & d1 & (beta > 0)).sum()), n_lost_degree1=int((sig & d1 & (beta < 0)).sum()),
            n_altered_treat=int(sig_treat.sum()), n_altered_treat_degree1=int((sig_treat & d1).sum()),
            n_altered_welch=int(f_w.sum()), n_altered_limma=int(f_l.sum()), n_altered_welch_and_limma=int((f_w & f_l).sum()),
            n_altered_unadjusted=int(f_u.sum()), n_altered_unadjusted_degree1=int((f_u & d1).sum()),
            n_altered_with_and_without_tissue=int((f_u & sig).sum()),
            smallest_abs_delta_altered=None if not n_sig else round(float(np.abs(beta[sig]).min()), 4),
            reference_neighbours_tested=a + b, reference_neighbours_altered=a,
            other_partners_tested=c + d, other_partners_altered=c,
            known_partner_odds_ratio=None if not np.isfinite(odds) else round(float(odds), 2),
            known_partner_fisher_p=None if not np.isfinite(p_fisher) else float(p_fisher),
            heaviest_tissue=str(levels[heavy]), heaviest_tissue_weight_pct=round(100 * float(sxx_t[heavy] / sxx), 1),
            without_heaviest_same_sign_pct=None if heavy_row.empty else heavy_row["same_sign_pct"].iloc[0],
            without_heaviest_r=None if heavy_row.empty else heavy_row["r"].iloc[0],
            without_heaviest_same_sign_pct_degree1=None if heavy_row.empty else heavy_row["same_sign_pct_degree1"].iloc[0],
            lowest_same_sign_pct=None if worst.empty else worst["same_sign_pct"].iloc[0],
            lowest_same_sign_tissue=None if worst.empty else str(worst["tissue_left_out"].iloc[0]),
        )
        if len(perm):
            summary.update(perm_median_altered=float(perm["n_sig"].median()),
                           perm_q95_altered=float(perm["n_sig"].quantile(0.95)), perm_max_altered=int(perm["n_sig"].max()),
                           perm_p_altered=(1 + int((perm["n_sig"] >= n_sig).sum())) / (n_perm + 1),
                           perm_median_altered_degree1=float(perm["n_sig_degree1"].median()),
                           perm_max_altered_degree1=int(perm["n_sig_degree1"].max()),
                           perm_p_altered_degree1=(1 + int((perm["n_sig_degree1"] >= n_sig1).sum())) / (n_perm + 1),
                           perm_median_altered_treat=float(perm["n_sig_treat"].median()),
                           perm_max_altered_treat=int(perm["n_sig_treat"].max()))
        summary = {k: (None if isinstance(v, (float, np.floating)) and not np.isfinite(v) else
                       v.item() if isinstance(v, np.generic) else v) for k, v in summary.items()}

        say(f"  tested: {summary['n_tested']:,} edges ({summary['n_tested_degree1']:,} of degree 1, "
            f"{summary['n_reference_pairs_tested']:,} reference pairs); no variation: {n_flat}")
        say(f"  altered at FDR < {cfg.fdr_cut} ({cfg.primary_test}): {n_sig:,} edges; degree 1: {n_sig1} "
            f"({summary['n_lost_degree1']} lost, {summary['n_gained_degree1']} gained)")
        say(f"  TREAT (|Delta| > {tau:.3f}): {summary['n_altered_treat']:,} edges; degree 1: {summary['n_altered_treat_degree1']}")
        say(f"  welch: {summary['n_altered_welch']:,} | limma: {summary['n_altered_limma']:,} | both: "
            f"{summary['n_altered_welch_and_limma']:,}; without tissue (plain Welch): {summary['n_altered_unadjusted']:,}")
        if len(perm):
            say(f"  within-tissue permutation ({n_perm}x): median {summary['perm_median_altered']:.0f} edges pass, "
                f"maximum {summary['perm_max_altered']}; empirical P = {summary['perm_p_altered']:.3g}")
        say(f"  reference neighbours altered: {a} of {a + b}; other genes: {c} of {c + d} "
            f"(odds ratio {summary['known_partner_odds_ratio']})")

        # the strongest partners of the gene, for the tables behind the figures
        first = res[d1].sort_values(["FDR", "p"])
        top_names = set(first["Gene_B"].head(cfg.top_n)) | set(
            res[d1].sort_values(["FDR_treat", "p_treat"])["Gene_B"].head(cfg.top_n))
        top = first[first["Gene_B"].isin(top_names)].reset_index(drop=True)
        col_of = {g: i for i, g in enumerate(others)}
        top_cols = np.array([col_of[self.gene_idx[g.strip().upper()]] for g in top["Gene_B"]], dtype=int)
        raw_top = prep1["Yr"][:, top_cols].astype(np.float64) + prep1["means"][codes][:, top_cols]
        # tissue-adjusted score = overall mean + Delta * status + model residual, so that the
        # mutated-minus-wild-type difference of its means is exactly Delta
        beta_top = f1["beta"][top_cols]
        adj_top = (prep1["grand"][top_cols] + np.outer(x - x.mean(), beta_top)
                   + prep1["Yr"][:, top_cols].astype(np.float64) - np.outer(des["xr"], beta_top))

        pt_rows = []
        for g in np.flatnonzero((n_mut_t >= cfg.min_per_stratum) & (n_wt_t >= cfg.min_per_stratum)):
            rows = codes == g
            ym, yw = raw_top[rows & (x == 1)], raw_top[rows & (x == 0)]
            dlt, p_t = two_sample_welch(ym.mean(axis=0), ym.var(axis=0, ddof=1), len(ym),
                                        yw.mean(axis=0), yw.var(axis=0, ddof=1), len(yw))
            se = np.sqrt(ym.var(axis=0, ddof=1) / len(ym) + yw.var(axis=0, ddof=1) / len(yw))
            for j, partner in enumerate(top["Gene_B"]):
                pt_rows.append(dict(Gene_B=partner, tissue=levels[g], n_mutated=len(ym), n_wild_type=len(yw),
                                    Delta=dlt[j], SE=se[j], p=p_t[j]))
        per_tissue = pd.DataFrame(pt_rows, columns=["Gene_B", "tissue", "n_mutated", "n_wild_type", "Delta", "SE", "p"])

        sig_first = top[top["Sig"] == 1] if (top["Sig"] == 1).any() else top
        chosen = (list(sig_first.loc[sig_first["Delta"] < 0, "Gene_B"].head((cfg.n_values + 1) // 2))
                  + list(sig_first.loc[sig_first["Delta"] > 0, "Gene_B"].head(cfg.n_values // 2)))
        chosen += [g for g in sig_first["Gene_B"] if g not in chosen][:max(0, cfg.n_values - len(chosen))]
        val_rows = []
        for partner in chosen:
            j = int(np.flatnonzero(top["Gene_B"].to_numpy() == partner)[0])
            val_rows.append(pd.DataFrame({
                "Gene_B": partner, "cell_line": np.array(self.sample_ids, dtype=object)[positions], "tissue": levels[codes],
                "status": np.where(x == 1, "Mutated", "WT"), "value": raw_top[:, j], "value_adjusted": adj_top[:, j]}))
        values = pd.concat(val_rows, ignore_index=True) if val_rows else pd.DataFrame(
            columns=["Gene_B", "cell_line", "tissue", "status", "value", "value_adjusted"])

        net = res[(~d1) & res["Gene_A"].isin(top_names) & res["Gene_B"].isin(top_names)][
            ["Gene_A", "Gene_B", "Delta", "FDR", "FDR_treat", "Sig"]]

        # Cytoscape: the gene with all its altered partners, and the reference edges between those partners
        alt1 = first[first["Sig"] == 1]
        partners = set(alt1["Gene_B"])
        between = res[(~d1) & res["Gene_A"].isin(partners) & res["Gene_B"].isin(partners)]

        def cyto_table(e, edge_type):
            return pd.DataFrame({"source": e["Gene_A"], "target": e["Gene_B"], "Edge_Type": edge_type, "Delta": e["Delta"],
                                 "Abs_Delta": e["Delta"].abs(), "FDR": e["FDR"], "FDR_treat": e["FDR_treat"],
                                 "Direction": direction(e["Delta"]), "Altered": e["Sig"], "Ref_Score": e["Ref_Score"]})

        parts = [cyto_table(e, kind) for e, kind in ((alt1, "with_target"), (between, "between_partners")) if len(e)]
        cyto_edges = pd.concat(parts, ignore_index=True) if parts else cyto_table(alt1, "with_target")
        cyto_nodes = pd.DataFrame({
            "id": [target_name] + list(alt1["Gene_B"]), "Is_Target": [1] + [0] * len(alt1),
            "Direction": ["Target"] + list(direction(alt1["Delta"])), "Delta": [0.0] + list(alt1["Delta"]),
            "FDR": [np.nan] + list(alt1["FDR"]), "Reference_Neighbour": [0] + list(alt1["Is_Reference"]),
            "Ref_Score": [np.nan] + list(alt1["Ref_Score"])})
        if self.targets is not None:
            cyto_nodes["_key"] = cyto_nodes["id"].str.strip().str.upper()
            cyto_nodes = cyto_nodes.merge(self.targets, how="left", on="_key").drop(columns="_key")

        folder = os.path.join(out_dir, safe_name(target_name))
        os.makedirs(folder, exist_ok=True)
        stem = os.path.join(folder, safe_name(target_name))
        edges_out = res.sort_values(["FDR", "p"])
        if cfg.edge_table == "compact":
            edges_out = edges_out[(edges_out["Degree"] == 1) | (edges_out["Sig"] == 1)]
        edges_out.to_csv(stem + "_edges.csv.gz", index=False, float_format="%.6g",
                         compression={"method": "gzip", "compresslevel": 4})
        tissue_table.to_csv(stem + "_tissue_table.csv", index=False)
        per_tissue.to_csv(stem + "_per_tissue_effects.csv", index=False)
        values.to_csv(stem + "_top_edge_values.csv", index=False, float_format="%.6g")
        net.to_csv(stem + "_network_edges.csv", index=False)
        degrees.to_csv(stem + "_degrees.csv", index=False)
        perm.to_csv(stem + "_permutation.csv", index=False)
        loto.to_csv(stem + "_leave_one_tissue_out.csv", index=False)
        cyto_edges.to_csv(stem + "_cytoscape_edges.csv", index=False)
        cyto_nodes.to_csv(stem + "_cytoscape_nodes.csv", index=False)
        summary["folder_mb"] = round(sum(os.path.getsize(os.path.join(folder, f)) for f in os.listdir(folder)) / 1e6, 2)
        summary["seconds"] = round(time.time() - t_start, 1)
        pd.DataFrame({"key": list(summary), "value": list(summary.values())}).to_csv(stem + "_summary.csv", index=False)
        with open(stem + "_summary.json", "w") as fh:          # written last: its presence marks the gene as finished
            json.dump(summary, fh, indent=2)
        say(f"  written: {folder}   ({summary['seconds']} s, {summary['folder_mb']} MB)")
        return summary

    def finished_before(self, gene_up, out_dir):
        name = safe_name(self.genes[self.gene_idx[gene_up]])
        path = os.path.join(out_dir, name, name + "_summary.json")
        if not os.path.exists(path):
            return None
        try:
            with open(path) as fh:
                old = json.load(fh)
        except (OSError, ValueError):
            return None
        return old if all(old.get(k) == v for k, v in self.cfg.identity().items()) else None

    def run(self, out_dir, genes="all", skip_done=True, verbose=False):
        """Every requested gene, one folder each, and all_genes_summary.csv."""
        cfg = self.cfg
        os.makedirs(out_dir, exist_ok=True)
        if isinstance(genes, str):
            if genes.strip().lower() != "all":
                raise ValueError('genes must be "all" or a list of gene names')
            wanted = [g for g in self.geno.index if g in self.gene_idx]
            print(f"{len(self.geno):,} genes in the mutation table, {len(wanted):,} of them are network genes")
        else:
            wanted = list(dict.fromkeys(str(g).strip().upper() for g in genes))
        counts = {g: (self.mutated_lines_that_count(g) if g in self.geno.index and g in self.gene_idx else None)
                  for g in wanted}
        to_run = [g for g in wanted if counts[g] is not None and counts[g] >= cfg.min_mutated]
        print(f"genes to analyse: {len(to_run):,} (at least {cfg.min_mutated} mutated lines in tissues that also hold "
              f"wild type), left out: {len(wanted) - len(to_run):,}")

        summaries = {}
        for g in wanted:
            if g not in to_run:
                summaries[g] = dict(gene=g, n_mutated_in_informative_tissues=counts[g], status=(
                    "not in the mutation table" if g not in self.geno.index else
                    "not a network gene" if g not in self.gene_idx else
                    f"skipped: {counts[g]} mutated lines in tissues that also hold wild type (need {cfg.min_mutated})"))
        summary_file = os.path.join(out_dir, "all_genes_summary.csv")

        def save_summary():
            table = pd.DataFrame([summaries[g] for g in wanted if g in summaries])
            if len(table):
                table.to_csv(summary_file, index=False)

        n_new, seconds, t_begin = 0, [], time.time()
        try:
            for k, g in enumerate(to_run, 1):
                tag = f"[{k}/{len(to_run)}] {self.genes[self.gene_idx[g]]}"
                old = self.finished_before(g, out_dir) if skip_done else None
                if old is not None:
                    summaries[g] = old
                    print(f"{tag}: already done")
                    continue
                try:
                    s = self.analyse_gene(g, out_dir, verbose=verbose)
                except MemoryError:
                    raise
                except Exception as err:
                    if n_new == 0:
                        raise
                    s = dict(gene=self.genes[self.gene_idx[g]], status=f"error: {type(err).__name__}: {err}")
                summaries[g] = s
                if s.get("status") == "ok":
                    n_new += 1
                    seconds.append(s["seconds"])
                    chance = f"; shuffled status: at most {s['perm_max_altered']}" if "perm_max_altered" in s else ""
                    print(f"{tag}: {s['n_mutated']} mutated | altered edges {s['n_altered']:,} "
                          f"(degree 1: {s['n_altered_degree1']:,}; TREAT: {s['n_altered_treat']:,}){chance} | {s['seconds']:.0f} s")
                else:
                    print(f"{tag}: {s.get('status')}")
                if k % 10 == 0:
                    save_summary()
        finally:
            save_summary()
        ok = [g for g in to_run if summaries.get(g, {}).get("status") == "ok"]
        print(f"finished: {len(ok):,} genes analysed ({n_new:,} in this run, {(time.time() - t_begin) / 3600:.2f} h); "
              f"with altered edges: {sum(summaries[g]['n_altered'] > 0 for g in ok):,}")
        print(f"summary of all genes: {summary_file}")
        return summaries


def main():
    ap = argparse.ArgumentParser(description="Differential association analysis of mutated genes")
    ap.add_argument("--data", required=True, help="folder written by gid.preprocess")
    ap.add_argument("--networks", required=True, help="sample_networks.npy from gid.infer")
    ap.add_argument("--genomics", required=True, help="genes x cell lines table, 1 = mutated, 0 = wild type")
    ap.add_argument("--out", required=True)
    ap.add_argument("--genes", nargs="+", default=["all"])
    ap.add_argument("--covariate", default="tissue")
    ap.add_argument("--primary-test", choices=["welch", "limma"], default="welch")
    ap.add_argument("--permutations", type=int, default=200)
    ap.add_argument("--targets", default=None, help="optional Open Targets table (tsv with a 'symbol' column)")
    ap.add_argument("--edge-table", choices=["full", "compact"], default="full")
    ap.add_argument("--no-skip", action="store_true", help="redo genes that are already finished")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    data = load_omics(args.data)
    store = EdgeMatrix(args.networks, data.n_genes)
    cfg = Settings(covariate=args.covariate, primary_test=args.primary_test, n_perm=args.permutations,
                   edge_table=args.edge_table)
    targets = pd.read_csv(args.targets, sep="\t", low_memory=False) if args.targets else None
    study = MutationStudy(store, data.genes, data.meta, data.adj, read_genotypes(args.genomics), cfg, targets)
    genes = "all" if args.genes == ["all"] else args.genes
    study.run(args.out, genes, skip_done=not args.no_skip, verbose=args.verbose)


if __name__ == "__main__":
    main()
