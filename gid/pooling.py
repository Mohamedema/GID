import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .data import load_omics
from .embedding import umap_model
from .graph import build_graph
from .train import load_model, node_features


@torch.no_grad()
def pooled_embeddings(model, features, edge_index, edge_weight, n_layers):
    """One vector per sample: the latent gene embeddings weighted by a softmax of the sample's input.

    The input of every gene is averaged over the layers, the softmax runs over the genes of a sample
    and the weighted sum of the latent embeddings Z is the embedding of that sample.
    """
    model.eval()
    _, z = model(features, edge_index, edge_weight)
    n_genes = features.shape[0]
    n_samples = features.shape[1] // n_layers
    mean_input = features.view(n_genes, n_layers, n_samples).mean(dim=1).t()
    weights = F.softmax(mean_input, dim=1)
    return (weights @ z).cpu().numpy()


def project(embeddings, neighbours=30, min_dist=0.1, seed=42):
    reducer = umap_model(n_neighbors=neighbours, min_dist=min_dist, n_components=2, metric="cosine", random_state=seed)
    return reducer.fit_transform(embeddings)


def main():
    ap = argparse.ArgumentParser(description="UMAP of sample embeddings pooled from the latent gene embeddings")
    ap.add_argument("--data", required=True, help="folder written by gid.preprocess or gid.tcga")
    ap.add_argument("--model", required=True, help="model.pt written by gid.train")
    ap.add_argument("--out", required=True)
    ap.add_argument("--neighbours", type=int, default=30)
    ap.add_argument("--min-dist", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    model, cfg, ck = load_model(args.model, device=args.device)
    data = load_omics(args.data).select_layers(ck["layers"])
    feats, _ = node_features(data.values, data.mask, data.layers)
    edge_index, edge_weight = build_graph(data.adj, cfg.graph_threshold)
    embeddings = pooled_embeddings(model, feats.to(args.device), edge_index.to(args.device),
                                   edge_weight.to(args.device), len(data.layers))
    print(f"sample embeddings: {embeddings.shape[0]} samples, {embeddings.shape[1]} dimensions")

    table = pd.DataFrame(project(embeddings, args.neighbours, args.min_dist, args.seed), columns=["UMAP1", "UMAP2"])
    table.insert(0, "Sample", data.meta.index.to_numpy())
    table["Tissue"] = data.meta["tissue"].to_numpy()
    os.makedirs(args.out, exist_ok=True)
    table.to_csv(os.path.join(args.out, "umap_samples.csv"), index=False)
    np.save(os.path.join(args.out, "sample_embeddings.npy"), embeddings.astype(np.float32))


if __name__ == "__main__":
    main()
