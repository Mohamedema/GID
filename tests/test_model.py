import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch_geometric.utils import subgraph

from gid.graph import build_graph
from gid.model import GCNAutoencoder, reconstruction_loss
from gid.train import Config, fit, load_model, n_continuous_columns, node_features, save_model
from tests.synthetic import random_omics


def features_step_by_step(values, mask):
    """Node feature construction written out step by step."""
    n, layers, g = values.shape
    combined = values.reshape(n, layers * g)          # samples x (layer * gene)
    data = combined.T
    start = 4 * g
    cont = StandardScaler().fit_transform(data[:start, :].T).T
    scaled = np.vstack([cont, data[start:, :]])
    feats = torch.FloatTensor(scaled).view(layers, g, n).permute(1, 0, 2).reshape(g, -1)
    miss = torch.BoolTensor(mask.reshape(n, layers * g).T).view(layers, g, n).permute(1, 0, 2).reshape(g, -1)
    return feats, miss


def test_node_features_match_the_step_by_step_construction():
    d = random_omics(n_genes=17, n_samples=12)
    feats, miss = node_features(d.values, d.mask, d.layers)
    f0, m0 = features_step_by_step(d.values, d.mask)
    assert feats.shape == (17, 5 * 12)
    assert torch.allclose(feats, f0, atol=1e-6)
    assert torch.equal(miss, m0)


def test_continuous_layers_are_standardised_and_cnv_is_not():
    d = random_omics()
    feats, _ = node_features(d.values, d.mask, d.layers)
    n = d.n_samples
    block = feats[:, :n].numpy()
    assert np.allclose(block.mean(axis=1), 0, atol=1e-5)
    assert np.allclose(block.std(axis=1), 1, atol=1e-4)
    assert np.array_equal(feats[:, 4 * n:].numpy(), d.values[:, 4, :].T.astype(np.float32))


def test_graph_weights_are_doubled_and_self_loops_are_one():
    d = random_omics(n_genes=12)
    ei, ew = build_graph(d.adj, 0.7)
    dense = torch.zeros(12, 12)
    dense[ei[0], ei[1]] = ew
    a = (d.adj >= 0.7) * d.adj
    off = dense - torch.diag(torch.diag(dense))
    assert np.allclose(off.numpy(), 2 * a, atol=1e-6)
    assert np.allclose(torch.diag(dense).numpy(), 1.0)
    assert np.allclose(dense.numpy(), dense.numpy().T)


def test_threshold_removes_weak_pairs():
    adj = np.array([[0, 0.69, 0.7], [0.69, 0, 0.3], [0.7, 0.3, 0]])
    ei, ew = build_graph(adj, 0.7)
    pairs = {(int(a), int(b)) for a, b in ei.T if a != b}
    assert pairs == {(0, 2), (2, 0)}


def test_subgraph_follows_the_order_of_the_batch():
    d = random_omics(n_genes=15)
    ei, ew = build_graph(d.adj, 0.6)
    batch = torch.tensor([9, 2, 14, 5, 0])
    sub_ei, sub_ew = subgraph(batch, ei, edge_attr=ew, relabel_nodes=True, num_nodes=15)
    small = torch.zeros(5, 5)
    small[sub_ei[0], sub_ei[1]] = sub_ew
    full = torch.zeros(15, 15)
    full[ei[0], ei[1]] = ew
    assert torch.allclose(small, full[batch][:, batch])


def test_loss_ignores_masked_entries():
    torch.manual_seed(0)
    n_cont = 6
    out = torch.randn(4, 9)
    feats = torch.randn(4, 9)
    feats[:, n_cont:] = torch.randint(-2, 3, (4, 3)).float()
    mask = torch.zeros(4, 9, dtype=torch.bool)
    mask[0, 1] = mask[2, 7] = True
    base = reconstruction_loss(out, feats, mask, n_cont)
    other = feats.clone()
    other[0, 1] = 100.0
    other[2, 7] = 2.0 if feats[2, 7] != 2 else -2.0
    assert torch.isclose(base, reconstruction_loss(out, other, mask, n_cont))


def test_loss_value_by_hand():
    out = torch.tensor([[1.0, 3.0, 0.5]])
    feats = torch.tensor([[0.0, 1.0, 2.0]])
    mask = torch.zeros(1, 3, dtype=torch.bool)
    loss = reconstruction_loss(out, feats, mask, 2)
    mse = ((1.0 - 0.0) ** 2 + (3.0 - 1.0) ** 2) / 2
    centres = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    logits = -((0.5 - centres) ** 2)
    ce = -torch.log_softmax(logits, dim=0)[4]
    assert torch.isclose(loss, mse + ce)


def test_loss_with_three_copy_number_levels():
    out = torch.tensor([[1.0, 3.0, 0.5, -0.2]])
    feats = torch.tensor([[0.0, 1.0, 1.0, -1.0]])
    mask = torch.zeros(1, 4, dtype=torch.bool)
    loss = reconstruction_loss(out, feats, mask, 2, levels=(-1.0, 0.0, 1.0))
    mse = ((1.0 - 0.0) ** 2 + (3.0 - 1.0) ** 2) / 2
    centres = torch.tensor([-1.0, 0.0, 1.0])
    ce = 0.0
    for pred, call in [(0.5, 2), (-0.2, 0)]:
        ce += -torch.log_softmax(-((pred - centres) ** 2), dim=0)[call].item() / 2
    assert torch.isclose(loss, torch.tensor(mse + ce, dtype=loss.dtype), atol=1e-6)


def test_loss_with_nothing_observed_is_zero():
    out = torch.randn(3, 5)
    mask = torch.ones(3, 5, dtype=torch.bool)
    assert reconstruction_loss(out, torch.randn(3, 5), mask, 3).item() == 0.0


def test_training_reduces_the_loss_and_checkpoint_roundtrip(tmp_path):
    d = random_omics(n_genes=20, n_samples=16)
    feats, miss = node_features(d.values, d.mask, d.layers)
    ei, ew = build_graph(d.adj, 0.6)
    cfg = Config(hidden_dim=16, latent_dim=8, epochs=40, batch_size=8, lr=0.005, seed=3)
    n_cont = n_continuous_columns(d.layers, d.n_samples)
    model, history = fit(feats, miss, ei, ew, n_cont, cfg, every=1000)
    assert len(history) == 40
    assert np.mean(history[-5:]) < np.mean(history[:5])
    path = str(tmp_path / "m.pt")
    save_model(path, model, cfg, history, d.layers, d.n_samples)
    again, cfg2, ck = load_model(path)
    assert cfg2 == cfg
    model.eval()
    with torch.no_grad():
        a, ha = model(feats, ei, ew)
        b, hb = again(feats, ei, ew)
    assert torch.allclose(ha, hb) and torch.allclose(a, b)


def test_single_layer_input_dimension():
    d = random_omics(n_genes=14, n_samples=10).select_layers(["protein"])
    feats, miss = node_features(d.values, d.mask, d.layers)
    assert feats.shape == (14, 10)
    assert n_continuous_columns(d.layers, 10) == 10
    only_cnv = random_omics(n_genes=14, n_samples=10).select_layers(["cnv"])
    assert n_continuous_columns(only_cnv.layers, 10) == 0


def test_same_seed_same_training():
    d = random_omics(n_genes=12, n_samples=10)
    feats, miss = node_features(d.values, d.mask, d.layers)
    ei, ew = build_graph(d.adj, 0.6)
    cfg = Config(hidden_dim=8, latent_dim=4, epochs=3, batch_size=4, seed=11)
    n_cont = n_continuous_columns(d.layers, d.n_samples)
    _, h1 = fit(feats, miss, ei, ew, n_cont, cfg, every=1000)
    _, h2 = fit(feats, miss, ei, ew, n_cont, cfg, every=1000)
    assert h1 == h2
