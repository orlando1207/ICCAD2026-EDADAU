"""
Phase 2 acceptance test (VERSION_B_RL_PLACER.md §5 Phase 2).

Checks the pure-torch GNN encoder:
  * variable block counts (5, 60, 120) -> correct output shapes
  * isolated graph (no edges) works
  * gradients flow
  * runs on a real validation sample

Run:
    cd FloorSet/iccad2026contest
    python3 DL_RL/test_gnn_encoder.py
"""

import sys
from pathlib import Path

import torch

_FLOORSET_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_FLOORSET_ROOT) not in sys.path:
    sys.path.insert(0, str(_FLOORSET_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lite_dataset_test import FloorplanDatasetLiteTest  # noqa: E402
from gnn_encoder import (  # noqa: E402
    GNNEncoder, build_node_features, build_edges, NODE_FEAT_DIM,
)


def _synthetic(n_blocks: int, n_edges: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    area = torch.rand(n_blocks, generator=g) * 400 + 100
    cons = torch.zeros(n_blocks, 5)
    # give a couple of blocks boundary codes / flags to exercise feature build
    if n_blocks > 3:
        cons[0, 0] = 1            # fixed
        cons[1, 1] = 1            # preplaced
        cons[2, 4] = 1 | 4        # boundary LEFT|TOP
    if n_edges > 0:
        i = torch.randint(0, n_blocks, (n_edges,), generator=g)
        j = torch.randint(0, n_blocks, (n_edges,), generator=g)
        w = torch.rand(n_edges, generator=g) * 0.01
        b2b = torch.stack([i.float(), j.float(), w], dim=1)
    else:
        b2b = torch.zeros((0, 3))
    return area, cons, b2b


def test_shapes():
    out_dim = 64
    enc = GNNEncoder(out_dim=out_dim, hidden=32, n_layers=3)
    for n in (5, 60, 120):
        area, cons, b2b = _synthetic(n, n_edges=n * 2)
        node_emb, graph_emb = enc.encode_problem(area, cons, b2b, block_count=n)
        assert node_emb.shape == (n, out_dim), (n, node_emb.shape)
        assert graph_emb.shape == (out_dim,), graph_emb.shape
        assert torch.isfinite(node_emb).all() and torch.isfinite(graph_emb).all()
    print("Shape checks PASS for N in {5, 60, 120}.")

    # feature matrix sanity
    area, cons, b2b = _synthetic(5, 4)
    feats = build_node_features(area, cons, 5)
    assert feats.shape == (5, NODE_FEAT_DIM), feats.shape
    assert float(feats[0, 2]) == 1.0          # block 0 fixed flag
    assert float(feats[1, 3]) == 1.0          # block 1 preplaced flag
    # feature order: [sqrt, rel, fixed, preplaced, has_mib, has_cluster,
    #                 left(6), right(7), top(8), bottom(9)]
    assert float(feats[2, 6]) == 1.0          # block 2 boundary LEFT
    assert float(feats[2, 8]) == 1.0          # block 2 boundary TOP
    print("Node-feature encoding PASS (flags/boundary bits).")


def test_no_edges():
    enc = GNNEncoder(out_dim=32, hidden=16, n_layers=2)
    area, cons, b2b = _synthetic(8, n_edges=0)
    node_emb, graph_emb = enc.encode_problem(area, cons, b2b, block_count=8)
    assert node_emb.shape == (8, 32)
    assert torch.isfinite(node_emb).all()
    print("Isolated-graph (no edges) PASS.")


def test_gradients():
    enc = GNNEncoder(out_dim=16, hidden=16, n_layers=2)
    area, cons, b2b = _synthetic(20, n_edges=40)
    node_emb, graph_emb = enc.encode_problem(area, cons, b2b, block_count=20)
    loss = node_emb.pow(2).mean() + graph_emb.pow(2).mean()
    loss.backward()
    grads = [p.grad for p in enc.parameters() if p.grad is not None]
    assert len(grads) > 0 and any(g.abs().sum() > 0 for g in grads)
    print("Gradient-flow PASS.")


def test_real_sample():
    ds = FloorplanDatasetLiteTest(str(_FLOORSET_ROOT))
    sample = ds[0]
    at, b2b, p2b, pins, cons = sample["input"]
    block_count = int((at != -1).sum().item())
    enc = GNNEncoder(out_dim=128)
    node_emb, graph_emb = enc.encode_problem(at, cons, b2b, block_count)
    edge_index, _ = build_edges(b2b, block_count)
    print(f"Real sample 0: blocks={block_count}, edges(undirected)={edge_index.shape[1]}, "
          f"node_emb={tuple(node_emb.shape)}, graph_emb={tuple(graph_emb.shape)}")
    assert node_emb.shape == (block_count, 128)


def main():
    torch.manual_seed(0)
    test_shapes()
    test_no_edges()
    test_gradients()
    test_real_sample()
    print("\nPhase 2 PASS.")


if __name__ == "__main__":
    main()
