import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .data import EdgeMatrix, load_omics, upper_index
from .graph import build_graph
from .train import load_model, node_features


def cosine_similarity(h):
    hn = F.normalize(h, p=2, dim=1)
    return hn @ hn.t()


@torch.no_grad()
def population_network(model, features, edge_index, edge_weight):
    """Gene by gene cosine similarity of the latent embeddings of all samples together."""
    model.eval()
    _, h = model(features, edge_index, edge_weight)
    sim = cosine_similarity(h)
    a = sim.cpu().numpy()
    return sim, 0.5 * (a + a.T)


@torch.no_grad()
def sample_networks(model, features, edge_index, edge_weight, n_samples, n_layers, store, every=50):
    """LIONESS in embedding space: N * S(all samples) - (N - 1) * S(all samples but i), for every sample i.

    Sample i is removed by replacing its columns in each layer with -x / (N - 1), the value that
    the leave-one-out mean takes when the layer is centred over the samples.
    """
    model.eval()
    _, h_all = model(features, edge_index, edge_weight)
    s_all = cosine_similarity(h_all)
    n = n_samples
    ii, jj = upper_index(features.shape[0])
    x = features.clone()
    for i in range(n):
        cols = [i + k * n for k in range(n_layers)]
        x[:, cols] = -features[:, cols] / (n - 1)
        _, h = model(x, edge_index, edge_weight)
        x[:, cols] = features[:, cols]
        s = n * s_all - (n - 1) * cosine_similarity(h)
        store.x[i] = s.cpu().numpy()[ii, jj]
        if (i + 1) % every == 0 or i + 1 == n:
            print(f"sample networks: {i + 1}/{n}", flush=True)
    store.flush()


def tissue_networks(store, tissues):
    """Mean network of every tissue as full symmetric matrices: returns (names, array of shape (T, G, G))."""
    tissues = np.asarray(tissues, dtype=str)
    names = sorted(set(tissues))
    g = store.n_genes
    ii, jj = upper_index(g)
    out = np.zeros((len(names), g, g), dtype=np.float32)
    for t, name in enumerate(names):
        rows = np.flatnonzero(tissues == name)
        acc = np.zeros(store.n_edges, dtype=np.float64)
        for r in rows:
            acc += store.x[r]
        acc /= len(rows)
        out[t][ii, jj] = acc
        out[t][jj, ii] = acc
        out[t][np.arange(g), np.arange(g)] = 1.0
    return names, out


def main():
    ap = argparse.ArgumentParser(description="Population, sample and tissue networks from a trained model")
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True, help="model.pt written by gid.train")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    model, cfg, ck = load_model(args.model, device=args.device)
    data = load_omics(args.data).select_layers(ck["layers"])
    feats, _ = node_features(data.values, data.mask, data.layers)
    feats = feats.to(args.device)
    edge_index, edge_weight = build_graph(data.adj, cfg.graph_threshold)
    edge_index, edge_weight = edge_index.to(args.device), edge_weight.to(args.device)

    os.makedirs(args.out, exist_ok=True)
    _, pop = population_network(model, feats, edge_index, edge_weight)
    np.save(os.path.join(args.out, "population_network.npy"), pop.astype(np.float32))

    store = EdgeMatrix.create(os.path.join(args.out, "sample_networks.npy"), data.n_samples, data.n_genes)
    sample_networks(model, feats, edge_index, edge_weight, data.n_samples, len(data.layers), store)

    names, tis = tissue_networks(store, data.meta["tissue"].to_numpy())
    np.save(os.path.join(args.out, "tissue_networks.npy"), tis)
    pd.Series(names).to_csv(os.path.join(args.out, "tissue_names.txt"), index=False, header=False)


if __name__ == "__main__":
    main()
