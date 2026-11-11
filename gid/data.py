import json
import os

import numpy as np
import pandas as pd

from . import LAYERS


class OmicsData:
    """Five aligned omic layers on a common gene set.

    values  (samples, layers, genes) float64; masked entries hold the imputed value
    mask    same shape, True where the value was not measured
    adj     (genes, genes) STRING scores scaled to [0, 1], zero diagonal
    meta    one row per sample, indexed by sample id, with a 'tissue' column
    """

    def __init__(self, values, mask, genes, meta, adj, layers=None):
        self.values = values
        self.mask = mask
        self.genes = list(genes)
        self.meta = meta
        self.adj = adj
        self.layers = list(layers) if layers is not None else list(LAYERS)

    @property
    def n_samples(self):
        return self.values.shape[0]

    @property
    def n_genes(self):
        return self.values.shape[2]

    def select_layers(self, names):
        idx = [self.layers.index(n) for n in names]
        return OmicsData(self.values[:, idx], self.mask[:, idx], self.genes, self.meta, self.adj,
                         [self.layers[i] for i in idx])

    def save(self, folder):
        os.makedirs(folder, exist_ok=True)
        np.save(os.path.join(folder, "values.npy"), self.values.astype(np.float64))
        np.save(os.path.join(folder, "mask.npy"), self.mask.astype(bool))
        np.save(os.path.join(folder, "adjacency.npy"), self.adj.astype(np.float32))
        with open(os.path.join(folder, "genes.txt"), "w") as fh:
            fh.write("\n".join(self.genes) + "\n")
        self.meta.to_csv(os.path.join(folder, "samples.csv"))
        with open(os.path.join(folder, "layers.json"), "w") as fh:
            json.dump(self.layers, fh)


def load_omics(folder):
    values = np.load(os.path.join(folder, "values.npy"))
    mask = np.load(os.path.join(folder, "mask.npy"))
    adj = np.load(os.path.join(folder, "adjacency.npy"))
    with open(os.path.join(folder, "genes.txt")) as fh:
        genes = [g.strip() for g in fh if g.strip()]
    meta = pd.read_csv(os.path.join(folder, "samples.csv"), index_col=0)
    layers = None
    path = os.path.join(folder, "layers.json")
    if os.path.exists(path):
        with open(path) as fh:
            layers = json.load(fh)
    return OmicsData(values, mask, genes, meta, adj, layers)


def upper_index(n_genes):
    """Gene pairs (i < j) in the order used for every sample-by-edge matrix."""
    return np.triu_indices(n_genes, 1)


def edge_column(i, j, n_genes):
    """Column of the pair (i, j), i < j, in the upper-triangle layout of upper_index."""
    i = np.asarray(i, dtype=np.int64)
    j = np.asarray(j, dtype=np.int64)
    return i * n_genes - i * (i + 1) // 2 + (j - i - 1)


class EdgeMatrix:
    """Samples x edges matrix of upper-triangle association scores kept on disk (np.memmap)."""

    def __init__(self, path, n_genes, mode="r"):
        self.path = path
        self.n_genes = n_genes
        self.n_edges = n_genes * (n_genes - 1) // 2
        self.x = np.load(path, mmap_mode=mode)
        if self.x.shape[1] != self.n_edges:
            raise ValueError(f"{path}: {self.x.shape[1]} columns, expected {self.n_edges} for {n_genes} genes")

    @classmethod
    def create(cls, path, n_samples, n_genes):
        n_edges = n_genes * (n_genes - 1) // 2
        mm = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(n_samples, n_edges))
        del mm
        return cls(path, n_genes, mode="r+")

    @property
    def shape(self):
        return self.x.shape

    def flush(self):
        self.x.flush()

    def pairs(self, ii, jj, rows):
        """Scores of the pairs (ii[k], jj[k]) for the given sample rows, as (len(rows), len(ii)) float32."""
        ii, jj = np.asarray(ii), np.asarray(jj)
        lo, hi = np.minimum(ii, jj), np.maximum(ii, jj)
        cols = edge_column(lo, hi, self.n_genes)
        order = np.argsort(cols)
        out = np.empty((len(rows), cols.size), dtype=np.float32)
        for k, r in enumerate(rows):
            out[k, order] = self.x[r, cols[order]]
        return out

    def square(self, row):
        """Full symmetric G x G matrix of one sample (diagonal 1)."""
        g = self.n_genes
        m = np.eye(g, dtype=np.float32)
        i, j = upper_index(g)
        m[i, j] = self.x[row]
        m[j, i] = self.x[row]
        return m
