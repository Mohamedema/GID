import numpy as np
import torch
from torch_geometric.utils import add_self_loops, dense_to_sparse, to_undirected


def build_graph(adj, threshold=0.7):
    """Weighted, undirected STRING graph with self loops, in the edge_index / edge_weight form of PyG.

    Scores below the threshold are dropped. to_undirected adds the two directions of every pair, so
    the kept weights are twice the normalised STRING score; self loops get weight 1.
    """
    a = np.array(adj, dtype=np.float32)
    a[a < threshold] = 0
    edge_index, edge_weight = dense_to_sparse(torch.from_numpy(a))
    edge_index, edge_weight = to_undirected(edge_index, edge_weight)
    edge_index, edge_weight = add_self_loops(edge_index, edge_attr=edge_weight, num_nodes=a.shape[0])
    return edge_index, edge_weight


def graph_summary(adj, threshold=0.7):
    a = np.triu(np.asarray(adj), 1)
    n = a.shape[0]
    kept = int((a >= threshold).sum())
    return dict(genes=n, pairs=n * (n - 1) // 2, pairs_kept=kept, density=kept / (n * (n - 1) / 2))
