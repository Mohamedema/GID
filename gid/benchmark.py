"""Tissue-of-origin clustering of single-sample networks: GID against LIONESS, SSN and BONOBO.

The three baselines are computed layer by layer, inside the cell lines that have the layer, and a
cell line's network is the mean over the layers it has. Nothing larger than a block of edges is ever
held in memory: edge means and variances are collected in a first pass over the edges, the Gram
matrices of the selected edges in a second.
"""
import argparse
import json
import os
import sys
import time
import warnings
from collections import Counter
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import sklearn
from sklearn.cluster import KMeans
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score,
                             normalized_mutual_info_score)
from sklearn.neighbors import NearestNeighbors

from .data import EdgeMatrix, load_omics, upper_index
from .embedding import umap_model

warnings.filterwarnings("ignore", message="n_jobs value 1 overridden")

SUB_NAME = "without_subset"
C5_NAME = "five_layers"
METHODS = ["GID", "LIONESS", "SSN", "BONOBO"]
COLORS = {"GID": "#0072B2", "LIONESS": "#E69F00", "SSN": "#009E73", "BONOBO": "#CC79A7"}

ANATOMY_MAP = {
    "Adrenal Gland": "Endocrine", "Biliary Tract": "Digestive", "Bladder": "Urinary", "Bone": "Skeletal",
    "Breast": "Reproductive", "Central Nervous System": "Neural", "Cervix": "Reproductive",
    "Endometrium": "Reproductive", "Esophagus": "Digestive", "Haematopoietic and Lymphoid": "Immune",
    "Head and Neck": "Head and Neck", "Kidney": "Urinary", "Large Intestine": "Digestive", "Liver": "Digestive",
    "Lung": "Respiratory", "Ovary": "Reproductive", "Pancreas": "Digestive",
    "Peripheral Nervous System": "Neural", "Placenta": "Reproductive", "Prostate": "Reproductive", "Skin": "Skin",
    "Small Intestine": "Digestive", "Soft Tissue": "Soft Tissue", "Stomach": "Digestive", "Testis": "Reproductive",
    "Thyroid": "Endocrine", "Uterus": "Reproductive", "Vulva": "Reproductive",
}


@dataclass
class Settings:
    seed: int = 42
    main_pct: int = 5
    pct_list: tuple = (1, 5, 10, 25)
    n_pcs: int = 50
    n_pcs_list: tuple = (20, 50, 100)
    knn_k: int = 10
    subset_exclude: tuple = ("Haematopoietic and Lymphoid",)
    stratify_by_coverage: bool = True
    legacy_gid_var: float = 2.0
    n_perm: int = 200
    chunk: int = 8000
    rel_tol: float = 1e-10
    min_layer_lines: int = 30
    obs_min_frac: float = 0.5
    min_c5_lines: int = 100
    bonobo_delta: float = None
    five_layer_control: bool = True
    umap_neighbors: int = 15
    umap_min_dist: float = 0.1


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def organ_labels(tissues, anatomy=ANATOMY_MAP):
    return pd.Series(np.asarray(tissues, dtype=str)).map(anatomy).fillna("Other").to_numpy()


class FairBaselines:
    """Single-sample networks over several omic layers, served block by block.

    layers    list of (genes, samples) arrays; the values of samples that lack a layer are never read
    observed  (samples, layers) boolean, True where the sample has the layer

    Inside a layer the population is the set of samples that have it, and a gene that does not vary
    among them is constant (correlation 0). Per layer and edge the single-sample values are centred on
    their mean over those samples; a sample's network is the mean over the layers it has and, for a
    pair of genes, over the layers in which both genes vary.
    """

    def __init__(self, layers, observed, methods, cfg):
        self.cfg = cfg
        self.G, self.N = layers[0].shape
        observed = np.asarray(observed, dtype=bool)
        keep = []
        for l in range(len(layers)):
            n = int(observed[:, l].sum())
            if n < cfg.min_layer_lines:
                log(f"layer {l} has only {n} samples (< {cfg.min_layer_lines}), left out")
            else:
                keep.append(l)
        if not keep:
            raise ValueError("no layer is observed in enough samples")
        self.methods = [m for m in methods if m in ("LIONESS", "SSN", "BONOBO")]
        self.names = self.methods
        self.O = observed[:, keep].astype(np.float64)
        self.I, self.J = np.triu_indices(self.G, 1)
        self.E = self.I.size
        self.layers = []
        for l in keep:
            rows = np.flatnonzero(observed[:, l])
            n = rows.size
            D = np.array(layers[l][:, rows], dtype=np.float64)
            D -= D.mean(axis=1, keepdims=True)
            s2 = np.einsum("ij,ij->i", D, D)
            vary = s2 > 1e-12 * max(s2.max(), 1e-300)
            D[~vary] = 0.0
            s2 = np.where(vary, s2, 0.0)
            S2 = D @ D.T
            f = n / (n - 1.0)
            if cfg.bonobo_delta is None:
                V = (s2[:, None] - f * D * D) / (n - 2.0)
                np.clip(V, 0.0, None, out=V)
                Vi = V[vary]
                with np.errstate(divide="ignore", invalid="ignore"):
                    dl = 1.0 / (3.0 + 2.0 * np.sqrt(Vi).mean(axis=0) / Vi.var(axis=0))
                dl = np.nan_to_num(dl, nan=0.0, posinf=0.0)
                del V, Vi
            else:
                dl = np.full(n, float(cfg.bonobo_delta))
            self.layers.append(dict(layer=l, rows=rows, n=n, DT=np.ascontiguousarray(D.T), s2=s2, S2=S2,
                                    sd=np.sqrt(s2), vary=vary, delta=dl, full=(n == self.N)))
            del D
        code = np.zeros(self.E, dtype=np.int32)
        for Ly in self.layers:
            code += (Ly["vary"][self.I] & Ly["vary"][self.J]).astype(np.int32) << Ly["layer"]
        self.strata = code

    def layer_report(self, names):
        for Ly in self.layers:
            log(f"  layer {Ly['layer']} ({names[Ly['layer']]}): {Ly['n']} samples, {int(Ly['vary'].sum())} of {self.G} genes vary")

    def _add(self, acc, x, Ly):
        if Ly["full"]:
            acc += x
        else:
            acc[Ly["rows"]] += x

    def layer_values(self, Ly, I, J, want_ssn, want_bon, centre=True):
        """(rho_all - rho_loo) and BONOBO for the edges (I, J) and the n samples of one layer, each (n, len(I))."""
        n = Ly["n"]
        f = n / (n - 1.0)
        s2 = Ly["s2"]
        Ei = Ly["DT"].take(I, axis=1)
        Ej = Ly["DT"].take(J, axis=1)
        Pij = Ei * Ej
        np.multiply(Ei, Ei, out=Ei)
        np.multiply(Ej, Ej, out=Ej)
        s2i, s2j = s2[I][None, :], s2[J][None, :]
        Cij = Pij * (-f)
        Cij += Ly["S2"][I, J][None, :]
        Cii = Ei * (-f)
        Cii += s2i
        Cjj = Ej * (-f)
        Cjj += s2j
        b = bon = None
        if want_ssn:
            den = Cii * Cjj
            den[(Cii <= self.cfg.rel_tol * s2i) | (Cjj <= self.cfg.rel_tol * s2j)] = np.inf
            np.sqrt(den, out=den)
            np.divide(Cij, den, out=den)
            sdd = Ly["sd"][I] * Ly["sd"][J]
            rho = np.divide(Ly["S2"][I, J], sdd, out=np.zeros(I.size), where=sdd > 0)
            np.subtract(rho[None, :], den, out=den)
            b = den
            if centre:
                b -= b.mean(axis=0, keepdims=True)
        if want_bon:
            dl = Ly["delta"][:, None]
            a = (1.0 - dl) / (n - 2.0)
            Pij *= dl
            Cij *= a
            Pij += Cij
            Ei *= dl
            Cii *= a
            Ei += Cii
            Ej *= dl
            Cjj *= a
            Ej += Cjj
            dd = Ei * Ej
            dd[(Ei <= 0) | (Ej <= 0)] = np.inf
            np.sqrt(dd, out=dd)
            np.divide(Pij, dd, out=dd)
            bon = dd
            if centre:
                bon -= bon.mean(axis=0, keepdims=True)
        return b, bon

    def blocks(self, idx):
        if isinstance(idx, slice):
            idx = np.arange(*idx.indices(self.E))
        I, J = self.I[idx], self.J[idx]
        c = I.size
        want_ssn = "SSN" in self.methods or "LIONESS" in self.methods
        want_bon = "BONOBO" in self.methods
        acc = {m: np.zeros((self.N, c)) for m in self.methods}
        P = np.zeros((len(self.layers), c))
        for k, Ly in enumerate(self.layers):
            P[k] = Ly["vary"][I] & Ly["vary"][J]
            b, bon = self.layer_values(Ly, I, J, want_ssn, want_bon)
            if "SSN" in acc:
                self._add(acc["SSN"], b, Ly)
            if "LIONESS" in acc:
                b *= (Ly["n"] - 1.0)
                self._add(acc["LIONESS"], b, Ly)
            if "BONOBO" in acc:
                self._add(acc["BONOBO"], bon, Ly)
            del b, bon
        cnt = self.O @ P
        np.maximum(cnt, 1.0, out=cnt)
        for m in acc:
            acc[m] /= cnt
        return acc


class ArraySource:
    """A samples x edges matrix, in memory or memory-mapped, served in edge blocks like FairBaselines."""

    def __init__(self, X, name):
        self.X, self.name, self.names = X, name, [name]
        self.N, self.E = X.shape

    def blocks(self, idx):
        sub = self.X[:, idx] if isinstance(idx, slice) else np.take(self.X, idx, axis=1)
        return {self.name: np.ascontiguousarray(sub, dtype=np.float64)}


def _progress(label, ci, nch, t0, every):
    if (ci + 1) % every == 0 or ci + 1 == nch:
        el = time.time() - t0
        log(f"  {label}: block {ci + 1}/{nch}  ({el / 60:.1f} min, about {el / (ci + 1) * (nch - ci - 1) / 60:.1f} min left)")


def pass1(src, row_sets, label, chunk, probe=None, quiet=False):
    """Per-edge mean and variance across samples, for each named set of rows (None = all rows).

    row_sets[name] = {set: rows or None}; returns st[name][set] = (mean, var), each of length E.
    """
    E = src.E
    st = {m: {k: (np.empty(E), np.empty(E)) for k in row_sets[m]} for m in src.names}
    nch = (E + chunk - 1) // chunk
    t0 = time.time()
    for ci, a in enumerate(range(0, E, chunk)):
        b = min(a + chunk, E)
        blk = src.blocks(slice(a, b))
        if probe is not None and ci in (0, nch // 2, nch - 1):
            probe(blk)
        for m in src.names:
            B = blk[m]
            for k, rows in row_sets[m].items():
                sub = B if rows is None else B[rows]
                st[m][k][0][a:b] = sub.mean(axis=0)
                st[m][k][1][a:b] = sub.var(axis=0)
        del blk
        if not quiet:
            _progress(label + " pass 1", ci, nch, t0, max(1, nch // 10))
    return st


def top_pct_bands(var, pcts, groups=None):
    """Nested 'top p % most variable edges' returned as disjoint bands.

    With groups (one label per edge) the threshold is taken inside each group, so every group
    contributes its own top p %.
    """
    v = np.asarray(var)
    ok = v > 0
    members = None if groups is None else [np.flatnonzero(groups == g) for g in np.unique(groups)]
    prev = np.zeros(v.size, dtype=bool)
    bands = []
    for p in sorted(pcts):
        if members is None:
            cur = (v >= np.percentile(v, 100 - p)) & ok
        else:
            cur = np.zeros(v.size, dtype=bool)
            for ix in members:
                cur[ix] = v[ix] >= np.percentile(v[ix], 100 - p)
            cur &= ok
        bands.append(cur & ~prev)
        prev = cur
    return bands


def pass2(src, jobs, label, chunk, quiet=False):
    """Gram matrices Z Z' of the centred (and optionally z-scored) selected edges.

    jobs[name][tag] = dict(rows, mean, std or None, sel); only edges selected by some job are computed.
    """
    E = src.E
    union = np.zeros(E, dtype=bool)
    for m in src.names:
        for j in jobs[m].values():
            union |= j["sel"]
    idx_all = np.flatnonzero(union)
    Ks = {m: {t: np.zeros((src.N if j["rows"] is None else len(j["rows"]),) * 2) for t, j in jobs[m].items()}
          for m in src.names}
    nch = max(1, (idx_all.size + chunk - 1) // chunk)
    t0 = time.time()
    if not quiet:
        log(f"  {label} pass 2: {idx_all.size:,} of {E:,} edges are needed")
    for ci, a in enumerate(range(0, idx_all.size, chunk)):
        idx = idx_all[a:a + chunk]
        blk = src.blocks(idx)
        for m in src.names:
            B = blk[m]
            for t, j in jobs[m].items():
                s = j["sel"][idx]
                if not s.any():
                    continue
                Z = B.take(np.flatnonzero(s), axis=1)
                if j["rows"] is not None:
                    Z = Z[j["rows"]]
                Z -= j["mean"][idx][s][None, :]
                if j["std"] is not None:
                    Z /= j["std"][idx][s][None, :]
                Ks[m][t] += Z @ Z.T
        del blk
        if not quiet:
            _progress(label + " pass 2", ci, nch, t0, max(1, nch // 10))
    return Ks


def prepare_jobs(name, st, rows, cfg, groups=None):
    m_all, v_all = st["all"]
    if not np.isfinite(v_all).all():
        raise ValueError(f"{name}: {int((~np.isfinite(v_all)).sum()):,} edges contain NaN or inf")
    sd_all = np.sqrt(v_all)
    sd_all[sd_all == 0] = 1.0
    pcts = sorted(cfg.pct_list)
    jobs = {}
    for p, band in zip(pcts, top_pct_bands(v_all, pcts, groups)):
        jobs[("all", p)] = dict(rows=None, mean=m_all, std=sd_all, sel=band)
    for key in ("sub", "c5"):
        if key in st:
            m_s, v_s = st[key]
            sd_s = np.sqrt(v_s)
            sd_s[sd_s == 0] = 1.0
            jobs[(key, cfg.main_pct)] = dict(rows=rows[key], mean=m_s, std=sd_s,
                                             sel=top_pct_bands(v_s, [cfg.main_pct], groups)[0])
    if name == "GID":
        jobs[("legacy", None)] = dict(rows=None, mean=m_all, std=None, sel=v_all >= cfg.legacy_gid_var)
    return jobs


def pca_scores(K, n_pcs):
    """Principal-component scores from a centred Gram matrix (same as PCA on the matrix, up to sign)."""
    w, V = np.linalg.eigh(K)
    o = np.argsort(w)[::-1][:n_pcs]
    return V[:, o] * np.sqrt(np.clip(w[o], 0, None))


def ch_db_from_gram(K, labels):
    """Calinski-Harabasz and Davies-Bouldin of the points behind a centred Gram matrix (full dimension)."""
    classes, y = np.unique(np.asarray(labels), return_inverse=True)
    k, n = len(classes), len(y)
    Ind = np.zeros((n, k))
    Ind[np.arange(n), y] = 1.0
    nc = Ind.sum(axis=0)
    KI = K @ Ind
    M = (Ind.T @ KI) / np.outer(nc, nc)
    between = float((nc * np.diag(M)).sum())
    within = float(np.trace(K)) - between
    ch = (between / (k - 1)) / (within / (n - k))
    d2 = np.diag(K) - 2.0 * KI[np.arange(n), y] / nc[y] + np.diag(M)[y]
    s = np.bincount(y, weights=np.sqrt(np.clip(d2, 0, None)), minlength=k) / nc
    dm = np.sqrt(np.clip(np.diag(M)[:, None] - 2.0 * M + np.diag(M)[None, :], 0, None))
    np.fill_diagonal(dm, np.inf)
    db = float(np.mean(np.max((s[:, None] + s[None, :]) / dm, axis=1)))
    return float(ch), db


def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return float(c - h), float(c + h)


def score_points(P, labels, cfg):
    classes, y = np.unique(np.asarray(labels), return_inverse=True)
    k = len(classes)
    idx = NearestNeighbors(n_neighbors=cfg.knn_k + 1).fit(P).kneighbors(P, return_distance=False)[:, 1:]
    votes = np.array([np.bincount(r, minlength=k).argmax() for r in y[idx]])
    hits = int((votes == y).sum())
    lo, hi = wilson(hits, len(y))
    km = KMeans(n_clusters=k, n_init=10, random_state=cfg.seed).fit(P)
    return dict(n_samples=len(y), n_classes=k,
                CH=float(calinski_harabasz_score(P, y)), DB=float(davies_bouldin_score(P, y)),
                kNN_acc=hits / len(y), kNN_lo=lo, kNN_hi=hi, majority_baseline=float(np.bincount(y).max() / len(y)),
                ARI=float(adjusted_rand_score(y, km.labels_)), NMI=float(normalized_mutual_info_score(y, km.labels_)))


def perm_null(P, labels, rng, cfg):
    """CH with shuffled labels (about 1 without structure) and the permutation p-value of the observed CH."""
    y = np.unique(np.asarray(labels), return_inverse=True)[1]
    ch_obs = calinski_harabasz_score(P, y)
    null = np.array([calinski_harabasz_score(P, rng.permutation(y)) for _ in range(cfg.n_perm)])
    return dict(CH_null_mean=float(null.mean()), CH_null_max=float(null.max()),
                CH_perm_p=float((1 + (null >= ch_obs).sum()) / (cfg.n_perm + 1)))


def evaluate_method(name, tissues, jobs, Ks, rng, cfg):
    """All analyses of one method from its Gram matrices: (result rows, PCA scores of the main setting)."""
    tissues = np.asarray(tissues, dtype=str)
    organs = organ_labels(tissues)
    pcts = sorted(cfg.pct_list)
    K_by_pct = dict(zip(pcts, np.cumsum(np.stack([Ks[("all", p)] for p in pcts]), axis=0)))
    n_edges = dict(zip(pcts, np.cumsum([int(jobs[("all", p)]["sel"].sum()) for p in pcts])))
    rows = []

    def add(protocol, subset, pct, n_pcs, edges, tis_lab, org_lab, P=None, K=None, unc=False):
        for level, lab in (("Tissue", tis_lab), ("Anatomy", org_lab)):
            r = dict(method=name, protocol=protocol, subset=subset, top_pct=pct, n_pcs=n_pcs, n_edges=edges, level=level)
            if P is not None:
                r.update(score_points(P, lab, cfg))
                if unc:
                    r.update(perm_null(P, lab, rng, cfg))
            else:
                r["CH"], r["DB"] = ch_db_from_gram(K, lab)
                r["n_samples"] = len(lab)
            rows.append(r)

    P_main = None
    for p in pcts:
        for npc in (cfg.n_pcs_list if p == cfg.main_pct else [cfg.n_pcs]):
            P = pca_scores(K_by_pct[p], npc)
            main = (p == cfg.main_pct and npc == cfg.n_pcs)
            add("harmonised", "all", p, npc, int(n_edges[p]), tissues, organs, P=P, unc=main)
            if main:
                P_main = P
    for key, sname in (("sub", SUB_NAME), ("c5", C5_NAME)):
        if (key, cfg.main_pct) in jobs:
            r = jobs[(key, cfg.main_pct)]["rows"]
            add("harmonised", sname, cfg.main_pct, cfg.n_pcs, int(jobs[(key, cfg.main_pct)]["sel"].sum()),
                tissues[r], organs[r], P=pca_scores(Ks[(key, cfg.main_pct)], cfg.n_pcs), unc=True)
    add("legacy_baseline_noPCA", "all", cfg.main_pct, None, int(n_edges[cfg.main_pct]), tissues, organs,
        K=K_by_pct[cfg.main_pct])
    if ("legacy", None) in jobs:
        n_leg = int(jobs[("legacy", None)]["sel"].sum())
        if n_leg < 2:
            log(f"GID: no edge has variance >= {cfg.legacy_gid_var:g}, legacy GID row skipped")
        else:
            add("legacy_GID_var_ge_%g_noscale_PCA" % cfg.legacy_gid_var, "all", None, cfg.n_pcs, n_leg, tissues,
                organs, P=pca_scores(Ks[("legacy", None)], cfg.n_pcs))
    return rows, P_main


def run_pipeline(src, tissues, row_sets, label, rng, cfg, probe=None, strata=None):
    log(f"{label}: {src.N} samples x {src.E:,} edges")
    st = pass1(src, row_sets, label, cfg.chunk, probe=probe)
    jobs = {m: prepare_jobs(m, st[m], row_sets[m], cfg, strata) for m in src.names}
    del st
    Ks = pass2(src, jobs, label, cfg.chunk)
    rows, P_main = [], {}
    for m in src.names:
        log(f"{m}: metrics")
        r, P_main[m] = evaluate_method(m, tissues[m], jobs[m], Ks[m], rng, cfg)
        rows += r
    return rows, P_main


def imputed_like(blk):
    """Samples whose block holds no measurement: it equals the gene means, or it is identical to the block of another sample."""
    dev = np.abs(blk - blk.mean(axis=1, keepdims=True)).max(axis=0)
    flat = dev < 1e-6 * max(1.0, float(np.abs(blk).max()))
    keys = [hash(r.tobytes()) for r in np.ascontiguousarray(np.round(blk.T, 8))]
    cnt = Counter(keys)
    return flat | np.array([cnt[k] >= 2 for k in keys])


def layer_blocks(data, cfg, use=None):
    """Per layer the (genes, samples) values and which samples have the layer.

    A sample has a layer when at least obs_min_frac of the layer's genes are unmasked for it, and its
    block is not just the imputed gene means.
    Returns (blocks, observed (samples, layers), indices of the layers used).
    """
    G, N = data.n_genes, data.n_samples
    use = list(range(len(data.layers))) if use is None else list(use)
    blocks, cols = [], []
    log("samples that have each layer:")
    for l in use:
        blk = np.array(data.values[:, l, :].T, dtype=np.float64)
        miss = np.asarray(data.mask[:, l, :].T, dtype=bool)
        imp = imputed_like(blk)
        has_gene = ~miss.all(axis=1)
        if not has_gene.any():
            raise ValueError(f"layer {data.layers[l]}: every entry is masked")
        frac = 1.0 - miss[has_gene].mean(axis=0)
        o = frac >= cfg.obs_min_frac
        agree = float((imp == ~o).mean())
        if imp.any() and agree < 0.2:
            raise ValueError(f"layer {data.layers[l]}: the mask and the data contradict each other "
                             f"(they agree for {agree:.1%} of the samples)")
        o = o & ~imp
        log(f"  {data.layers[l]}: {int(o.sum())} of {N} samples, {int(has_gene.sum())}/{G} genes with data")
        if o.sum() < cfg.min_layer_lines:
            raise ValueError(f"layer {data.layers[l]} is present in only {int(o.sum())} samples")
        blocks.append(blk)
        cols.append(o)
    return blocks, np.column_stack(cols), use


def plot_metrics(res, path, subset_labels, cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    main = res[(res.protocol == "harmonised") & (res.n_pcs == cfg.n_pcs) & (res.top_pct == cfg.main_pct)].copy()
    main["invDB"] = 1.0 / main.DB
    methods = [m for m in METHODS if m in set(main.method)]
    subsets = [s for s in ["all", SUB_NAME, C5_NAME] if s in set(main.subset)]
    hatches = [None, "//", "xx"]
    panels = [("CH", "Calinski-Harabasz\n(higher = better)"), ("invDB", "1 / Davies-Bouldin\n(higher = better)"),
              ("kNN_acc", f"{cfg.knn_k}-NN accuracy\n(dashed: largest class)")]
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.2))
    w = 0.8 / len(subsets)
    for r, level in enumerate(["Tissue", "Anatomy"]):
        for c, (col, title) in enumerate(panels):
            ax = axes[r, c]
            for si, sub in enumerate(subsets):
                for mi, m in enumerate(methods):
                    row = main[(main.method == m) & (main.subset == sub) & (main.level == level)]
                    if row.empty:
                        continue
                    row = row.iloc[0]
                    x = mi + (si - (len(subsets) - 1) / 2) * w
                    ax.bar(x, row[col], w * 0.92, color=COLORS.get(m, "grey"), alpha=1.0 if si == 0 else 0.5,
                           hatch=hatches[si], edgecolor="white", linewidth=0.5)
                    if col == "kNN_acc":
                        ax.errorbar(x, row.kNN_acc, yerr=[[max(0.0, row.kNN_acc - row.kNN_lo)],
                                                          [max(0.0, row.kNN_hi - row.kNN_acc)]], color="k", lw=0.8, capsize=2)
            if col == "CH":
                ax.axhline(1, color="grey", ls="--", lw=0.8)
            if col == "kNN_acc":
                ax.axhline(main[(main.level == level) & (main.subset == subsets[0])].majority_baseline.iloc[0],
                           color="grey", ls="--", lw=0.8)
            ax.set_xticks(range(len(methods)))
            ax.set_xticklabels(methods, fontsize=9)
            ax.set_title(f"{level}-level" if c == 0 else "", loc="left", fontsize=10, fontweight="bold")
            ax.set_ylabel(title, fontsize=9)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, fc="grey", alpha=1.0 if i == 0 else 0.5, hatch=hatches[i], ec="white")
               for i in range(len(subsets))]
    fig.legend(handles, [subset_labels.get(s_, s_) for s_ in subsets], loc="lower center", ncol=len(subsets),
               frameon=False, fontsize=9)
    fig.suptitle(f"Top {cfg.main_pct}% variable edges, z-score, PCA-{cfg.n_pcs} for every method", fontsize=10)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_umaps(coords, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    organs = sorted({o for df in coords.values() for o in df.Anatomy})
    cmap = plt.get_cmap("tab20")
    col = {o: cmap(i % 20) for i, o in enumerate(organs)}
    fig, axes = plt.subplots(1, len(coords), figsize=(4.2 * len(coords), 4.4), squeeze=False)
    for ax, (m, df) in zip(axes[0], coords.items()):
        for o in organs:
            s = df[df.Anatomy == o]
            ax.scatter(s.UMAP1, s.UMAP2, s=7, color=col[o], linewidths=0, label=o)
        ax.set_title(m, fontsize=11, fontweight="bold", color=COLORS.get(m, "k"))
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=min(len(l), 6), frameon=False, fontsize=8, markerscale=2)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(path, dpi=300)
    plt.close(fig)


def as_text(table):
    return table.round(3).to_string()


def print_reports(res, wide, subset_labels, cfg):
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 40)
    print(f"\nMain setting: top {cfg.main_pct}% variable edges, z-score, PCA-{cfg.n_pcs}")
    print(as_text(wide))
    base = res[(res.protocol == "harmonised") & (res.subset == "all") & (res.top_pct == cfg.main_pct) & (res.n_pcs == cfg.n_pcs)]
    mb = base.groupby("level").majority_baseline.first()
    nm = base.groupby("level").CH_null_mean.mean()
    print(f"largest class: {mb['Tissue']:.3f} (tissue), {mb['Anatomy']:.3f} (organ system); "
          f"CH with shuffled labels: {nm['Tissue']:.2f} / {nm['Anatomy']:.2f}")
    for sname in (SUB_NAME, C5_NAME):
        sub = res[(res.protocol == "harmonised") & (res.subset == sname)]
        if len(sub):
            print(f"\n{subset_labels.get(sname, sname)}")
            w2 = sub.pivot(index="method", columns="level", values=["CH", "DB", "kNN_acc"])
            w2.columns = [f"{a}_{b}" for a, b in w2.columns]
            print(as_text(w2.reindex([m for m in METHODS if m in w2.index])[
                ["CH_Tissue", "DB_Tissue", "CH_Anatomy", "DB_Anatomy", "kNN_acc_Tissue", "kNN_acc_Anatomy"]]))
    sens = res[(res.protocol == "harmonised") & (res.subset == "all") & (res.level == "Tissue")]
    print("\nCH at tissue level against the fraction of edges kept:")
    t1 = sens[sens.n_pcs == cfg.n_pcs].pivot(index="method", columns="top_pct", values="CH")
    t1.columns = [f"top{int(c)}%" for c in t1.columns]
    print(as_text(t1.reindex([m for m in METHODS if m in t1.index])))
    print("CH at tissue level against the number of PCs:")
    t2 = sens[sens.top_pct == cfg.main_pct].pivot(index="method", columns="n_pcs", values="CH")
    t2.columns = [f"{int(c)} PCs" for c in t2.columns]
    print(as_text(t2.reindex([m for m in METHODS if m in t2.index])))


def run_benchmark(data, gid_edges, out_dir, cfg=None, methods=METHODS, quick_genes=None, tissue_col="tissue"):
    """GID against the baselines on the same samples. gid_edges is the samples x edges matrix of GID."""
    cfg = cfg or Settings()
    t_start = time.time()
    out = out_dir + ("_quick" if quick_genes else "")
    os.makedirs(out, exist_ok=True)
    assert cfg.main_pct in cfg.pct_list and cfg.n_pcs in cfg.n_pcs_list
    rng = np.random.default_rng(cfg.seed)

    tissues = data.meta[tissue_col].astype(str).to_numpy()
    ids = np.asarray(data.meta.index, dtype=str)
    N = data.n_samples
    G_full = data.n_genes
    if gid_edges is not None and gid_edges.shape != (N, G_full * (G_full - 1) // 2):
        raise ValueError(f"GID matrix has shape {gid_edges.shape}, expected {(N, G_full * (G_full - 1) // 2)}")

    G, gid_X = G_full, gid_edges
    if quick_genes:
        G = int(quick_genes)
        iu = np.triu_indices(G_full, 1)
        sel = (iu[0] < G) & (iu[1] < G)
        if gid_X is not None:
            gid_X = np.ascontiguousarray(gid_X[:, sel])
        log(f"quick run on the first {G} genes: do not use these numbers")

    sub = data
    if quick_genes:
        sub = type(data)(data.values[:, :, :G], data.mask[:, :, :G], data.genes[:G], data.meta, data.adj[:G, :G], data.layers)

    need_base = [m for m in methods if m != "GID"]
    blocks, observed, use = layer_blocks(sub, cfg)
    n_have = observed.sum(axis=1)
    log("samples by number of layers: " + ", ".join(f"{k}: {int((n_have == k).sum())}" for k in range(observed.shape[1] + 1)))
    names = [data.layers[l] for l in use]
    table = pd.DataFrame(observed, index=ids, columns=names)
    table.insert(0, "tissue", tissues)
    table["n_layers"] = n_have
    table["all_layers"] = observed.all(axis=1)
    table.index.name = "sample_id"
    table.to_csv(os.path.join(out, "layer_observation_table.csv"))
    c5_flag = observed.all(axis=1)

    fb = FairBaselines(blocks, observed, need_base, cfg)
    del blocks
    fb.layer_report(data.layers)
    strata = fb.strata if cfg.stratify_by_coverage else None
    if strata is not None:
        cls = pd.Series(strata).value_counts()
        log(f"edges are ranked inside {len(cls)} classes of pairs measured in the same set of layers")

    def row_sets():
        rs = {"all": None, "sub": np.flatnonzero(~np.isin(tissues, cfg.subset_exclude))}
        if len(rs["sub"]) == len(tissues):
            log(f"none of {list(cfg.subset_exclude)} occurs in the tissue labels")
        if cfg.five_layer_control:
            c5 = np.flatnonzero(c5_flag)
            if len(c5) >= cfg.min_c5_lines:
                rs["c5"] = c5
            else:
                log(f"all-layers control skipped: only {len(c5)} samples have every layer")
        return rs

    rows, P_main, who = [], {}, {}
    if "GID" in methods:
        r, p = run_pipeline(ArraySource(gid_X, "GID"), {"GID": tissues}, {"GID": row_sets()}, "GID", rng, cfg, strata=strata)
        rows += r
        P_main.update(p)
        who["GID"] = (tissues, ids)
    if need_base:
        def probe(blk):
            if "LIONESS" in blk and "SSN" in blk:
                A, B = blk["LIONESS"], blk["SSN"]
                A = (A - A.mean(0, keepdims=True)) / (A.std(0, keepdims=True) + 1e-12)
                B = (B - B.mean(0, keepdims=True)) / (B.std(0, keepdims=True) + 1e-12)
                r = (A * B).mean(0)
                log(f"  LIONESS against SSN, per-edge correlation over samples: min {r.min():.5f}, median {np.median(r):.5f}")
        r, p = run_pipeline(fb, {m: tissues for m in fb.methods}, {m: row_sets() for m in fb.methods}, "baselines",
                            rng, cfg, probe=probe, strata=strata)
        rows += r
        P_main.update(p)
        for m in fb.methods:
            who[m] = (tissues, ids)

    res = pd.DataFrame(rows)
    res["invDB"] = 1.0 / res["DB"]
    res.to_csv(os.path.join(out, "clustering_all_results_long.csv"), index=False)
    s = (res.protocol == "harmonised") & (res.subset == "all") & (res.top_pct == cfg.main_pct) & (res.n_pcs == cfg.n_pcs)
    wide = res[s].pivot(index="method", columns="level", values=["CH", "DB", "kNN_acc", "ARI", "NMI"])
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide.reindex([m for m in METHODS if m in wide.index])
    wide = wide[["CH_Tissue", "DB_Tissue", "CH_Anatomy", "DB_Anatomy", "kNN_acc_Tissue", "kNN_acc_Anatomy",
                 "ARI_Tissue", "ARI_Anatomy", "NMI_Tissue", "NMI_Anatomy"]]
    wide.reset_index().to_csv(os.path.join(out, "Benchmark_clustering_index_GID_LIONESS_SSN_BONOBO.csv"), index=False)
    n_sub = int(res[res.subset == SUB_NAME].n_samples.max()) if (res.subset == SUB_NAME).any() else 0
    n_c5 = int(res[res.subset == C5_NAME].n_samples.max()) if (res.subset == C5_NAME).any() else 0
    subset_labels = {"all": f"all cell lines (n = {N})",
                     SUB_NAME: "without " + ", ".join(cfg.subset_exclude) + f" (n = {n_sub})",
                     C5_NAME: f"cell lines with every layer (n = {n_c5})"}
    print_reports(res, wide, subset_labels, cfg)
    plot_metrics(res, os.path.join(out, "fig_clustering_metrics.png"), subset_labels, cfg)

    try:
        coords = {}
        for m in methods:
            tis, sid = who[m]
            U = umap_model(random_state=cfg.seed, n_neighbors=cfg.umap_neighbors, min_dist=cfg.umap_min_dist,
                           metric="euclidean").fit_transform(P_main[m])
            df = pd.DataFrame({"sample_id": sid, "UMAP1": U[:, 0], "UMAP2": U[:, 1], "Tissue": tis,
                               "Anatomy": organ_labels(tis)})
            df.to_csv(os.path.join(out, f"umap_{m}_PCA{cfg.n_pcs}.csv"), index=False)
            coords[m] = df
        plot_umaps(coords, os.path.join(out, "fig_umap_panels.png"))
    except ImportError:
        log("umap-learn is not installed: UMAP coordinates skipped")

    info = asdict(cfg)
    info.update(layers_used=names, samples_per_layer=[int(x) for x in observed.sum(axis=0)],
                samples_with_every_layer=int(c5_flag.sum()),
                versions=dict(python=sys.version.split()[0], numpy=np.__version__, pandas=pd.__version__,
                              sklearn=sklearn.__version__),
                minutes=round((time.time() - t_start) / 60, 1))
    with open(os.path.join(out, "run_info.json"), "w") as fh:
        json.dump(info, fh, indent=2, default=str)
    log(f"done in {(time.time() - t_start) / 60:.1f} min, results in {os.path.abspath(out)}")
    return res


def main():
    ap = argparse.ArgumentParser(description="Clustering benchmark: GID against LIONESS, SSN and BONOBO")
    ap.add_argument("--data", required=True)
    ap.add_argument("--networks", required=True, help="sample_networks.npy from gid.infer")
    ap.add_argument("--out", required=True)
    ap.add_argument("--methods", nargs="+", default=METHODS)
    ap.add_argument("--quick-genes", type=int, default=None, help="dry run on the first n genes")
    ap.add_argument("--chunk", type=int, default=None)
    args = ap.parse_args()
    cfg = Settings()
    if args.chunk:
        cfg.chunk = args.chunk
    data = load_omics(args.data)
    store = EdgeMatrix(args.networks, data.n_genes) if "GID" in args.methods else None
    run_benchmark(data, None if store is None else store.x, args.out, cfg, args.methods, args.quick_genes)


if __name__ == "__main__":
    main()
