import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

CNV_CENTRES = (-2.0, -1.0, 0.0, 1.0, 2.0)


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, dropout_rate=0.3):
        super().__init__()
        self.gcn1 = GCNConv(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, latent_dim)
        self.bn2 = nn.BatchNorm1d(latent_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.prelu = nn.PReLU()
        self.residual_transform = nn.Linear(input_dim, latent_dim)

    def forward(self, x, edge_index, edge_weight):
        skip = self.residual_transform(x)
        x = self.gcn1(x, edge_index, edge_weight)
        x = self.prelu(self.bn1(x))
        x = self.dropout(x)
        x = self.bn2(self.gcn2(x, edge_index, edge_weight))
        return x + skip


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, dropout_rate=0.3):
        super().__init__()
        self.gcn1 = GCNConv(latent_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, output_dim)
        self.bn2 = nn.BatchNorm1d(output_dim)
        self.prelu = nn.PReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.residual_transform = nn.Linear(latent_dim, output_dim)

    def forward(self, x, edge_index, edge_weight):
        skip = self.residual_transform(x)
        x = self.gcn1(x, edge_index, edge_weight)
        x = self.prelu(self.bn1(x))
        x = self.dropout(x)
        x = self.bn2(self.gcn2(x, edge_index, edge_weight))
        return x + skip


class GCNAutoencoder(nn.Module):
    """Genes are the nodes; the features of a gene are its values over all samples in every layer."""

    def __init__(self, input_dim, hidden_dim, latent_dim, dropout_rate):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim, dropout_rate)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim, dropout_rate)

    def forward(self, x, edge_index, edge_weight):
        h = self.encoder(x, edge_index, edge_weight)
        return self.decoder(h, edge_index, edge_weight), h


def reconstruction_loss(out, features, mask, n_continuous, levels=CNV_CENTRES):
    """Masked MSE on the continuous columns plus cross entropy on the copy number columns.

    The first n_continuous columns hold the continuous layers, the rest the copy number layer
    (GISTIC calls, -2..2 for the cell lines, -1..1 for TCGA). Copy number classes are scored by
    their squared distance to the class centres in `levels`. Entries with mask True are not
    measured and do not count.
    """
    valid = ~mask
    v_cont = valid[:, :n_continuous]
    if v_cont.any():
        sq = (out[:, :n_continuous] - features[:, :n_continuous]) ** 2
        mse = sq[v_cont].mean()
    else:
        mse = torch.zeros((), device=features.device)
    v_cnv = valid[:, n_continuous:]
    if v_cnv.any():
        pred = out[:, n_continuous:][v_cnv]
        truth = features[:, n_continuous:][v_cnv]
        classes = (truth - levels[0]).long().clamp(0, len(levels) - 1)
        centres = torch.tensor(levels, dtype=torch.float32, device=features.device).view(1, -1)
        logits = -((pred.unsqueeze(1) - centres) ** 2)
        ce = F.cross_entropy(logits, classes)
    else:
        ce = torch.zeros((), device=features.device)
    return mse + ce
