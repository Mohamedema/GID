import argparse
import json
import os
import random
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from torch_geometric.utils import subgraph

from .data import load_omics
from .graph import build_graph
from .model import GCNAutoencoder, reconstruction_loss


@dataclass
class Config:
    hidden_dim: int = 400
    latent_dim: int = 190
    epochs: int = 1000
    batch_size: int = 32
    lr: float = 0.0006562883581637201
    weight_decay: float = 0.0005541688853970498
    dropout: float = 0.1518412589191374
    graph_threshold: float = 0.7
    cnv_levels: tuple = (-2.0, -1.0, 0.0, 1.0, 2.0)
    seed: int = 42


class GeneDataset(Dataset):
    def __init__(self, features, mask):
        self.features = features
        self.mask = mask

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.mask[idx], idx


def scale_layers(values, layers):
    """Standardise each (layer, gene) series over the samples; the copy number layer is left as called."""
    out = np.array(values, dtype=np.float64)
    for k, name in enumerate(layers):
        if name != "cnv":
            out[:, k, :] = StandardScaler().fit_transform(out[:, k, :])
    return out


def node_features(values, mask, layers):
    """(samples, layers, genes) -> (genes, layers * samples), one block of sample columns per layer."""
    x = scale_layers(values, layers)
    n_genes = x.shape[2]
    feats = torch.from_numpy(x).float().permute(2, 1, 0).reshape(n_genes, -1)
    miss = torch.from_numpy(np.asarray(mask)).bool().permute(2, 1, 0).reshape(n_genes, -1)
    return feats, miss


def n_continuous_columns(layers, n_samples):
    return sum(1 for name in layers if name != "cnv") * n_samples


def fit(features, mask, edge_index, edge_weight, n_continuous, cfg, device="cpu", every=50):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    n_genes = features.shape[0]
    # batch norm needs more than one gene in the last batch
    loader = DataLoader(GeneDataset(features, mask), batch_size=cfg.batch_size, shuffle=True,
                        drop_last=(n_genes % cfg.batch_size == 1))
    model = GCNAutoencoder(features.shape[1], cfg.hidden_dim, cfg.latent_dim, cfg.dropout).to(device)
    opt = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    edge_index, edge_weight = edge_index.to(device), edge_weight.to(device)
    history = []
    t0 = time.time()
    for epoch in range(cfg.epochs):
        model.train()
        running = 0.0
        for x, m, idx in loader:
            x, m = x.to(device), m.to(device)
            opt.zero_grad()
            ei, ew = subgraph(idx.to(device), edge_index, edge_attr=edge_weight,
                              relabel_nodes=True, num_nodes=n_genes)
            out, _ = model(x, ei, ew)
            loss = reconstruction_loss(out, x, m, n_continuous, cfg.cnv_levels)
            loss.backward()
            opt.step()
            running += loss.item()
        history.append(running / len(loader))
        if (epoch + 1) % every == 0 or epoch == 0:
            print(f"epoch {epoch + 1}/{cfg.epochs}  loss {history[-1]:.4f}  {time.time() - t0:.0f}s", flush=True)
    return model, history


def save_model(path, model, cfg, history, layers, n_samples):
    torch.save(dict(state_dict=model.state_dict(), config=asdict(cfg), history=history,
                    layers=list(layers), n_samples=int(n_samples)), path)


def load_model(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = Config(**ck["config"])
    input_dim = len(ck["layers"]) * ck["n_samples"]
    model = GCNAutoencoder(input_dim, cfg.hidden_dim, cfg.latent_dim, cfg.dropout).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, cfg, ck


def main():
    ap = argparse.ArgumentParser(description="Train the GID graph autoencoder")
    ap.add_argument("--data", required=True, help="folder written by gid.preprocess")
    ap.add_argument("--out", required=True, help="folder for the checkpoint and the loss curve")
    ap.add_argument("--layers", nargs="+", default=None,
                    help="omic layers to use (default: all five); one layer gives the single-omic models")
    ap.add_argument("--config", default=None, help="json file with hyperparameters")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    cfg = Config()
    if args.config:
        with open(args.config) as fh:
            cfg = Config(**{**asdict(cfg), **json.load(fh)})
    if args.epochs:
        cfg.epochs = args.epochs

    data = load_omics(args.data)
    if args.layers:
        data = data.select_layers(args.layers)
    feats, miss = node_features(data.values, data.mask, data.layers)
    edge_index, edge_weight = build_graph(data.adj, cfg.graph_threshold)
    n_cont = n_continuous_columns(data.layers, data.n_samples)
    print(f"{data.n_genes} genes, {data.n_samples} samples, layers: {', '.join(data.layers)}; "
          f"{edge_index.shape[1] - data.n_genes} directed edges in the graph")

    model, history = fit(feats, miss, edge_index, edge_weight, n_cont, cfg, device=args.device)
    os.makedirs(args.out, exist_ok=True)
    save_model(os.path.join(args.out, "model.pt"), model, cfg, history, data.layers, data.n_samples)
    with open(os.path.join(args.out, "training_loss.csv"), "w") as fh:
        fh.write("epoch,loss\n")
        fh.writelines(f"{i + 1},{v:.6f}\n" for i, v in enumerate(history))


if __name__ == "__main__":
    main()
