"""
Phase 2 — GNN encoder (VERSION_B_RL_PLACER.md §5 Phase 2).

netlist -> per-block embeddings + a whole-graph embedding.

Implemented as a minimal pure-PyTorch message-passing network (GraphSAGE-style,
weighted-mean aggregation) so we depend on nothing beyond torch. PyTorch
Geometric is intentionally NOT required (see Phase 2 note in the spec). The
network handles a variable number of blocks (5..120) natively.

Node features (dim = 10), per block:
  0 sqrt(area), graph-normalised      4 has_cluster flag
  1 area / total_area                 5 boundary touches LEFT
  2 fixed-shape flag                  6 boundary touches RIGHT
  3 preplaced flag                    7 boundary touches TOP
  (has_mib flag)                      8 boundary touches BOTTOM
The flags come from constraints[N,5] = (fixed, preplaced, mib, cluster, boundary).
Boundary code bits: LEFT=1 RIGHT=2 TOP=4 BOTTOM=8.

Edges: b2b connectivity, treated as undirected (both directions added).
Edge weight is used in a *weighted mean* so the tiny raw weights (~1e-3)
don't vanish the signal — aggregation is scale-invariant.
p2b/pins are ignored in this first version (Phase note: add a virtual pin
node later if it helps).
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

NODE_FEAT_DIM = 10


def build_node_features(area_target: torch.Tensor,
                        constraints: torch.Tensor,
                        block_count: int) -> torch.Tensor:
    """[block_count, NODE_FEAT_DIM] node feature matrix. See module docstring."""
    a = area_target[:block_count].float().clamp(min=0.0)
    sqrt_a = a.sqrt()
    sqrt_a_norm = sqrt_a / (sqrt_a.mean() + 1e-6)
    rel_a = a / (a.sum() + 1e-6)

    c = constraints[:block_count].float()
    fixed = (c[:, 0] > 0).float()
    preplaced = (c[:, 1] > 0).float()
    has_mib = (c[:, 2] > 0).float()
    has_cluster = (c[:, 3] > 0).float()

    bcode = c[:, 4].long()
    left = ((bcode & 1) > 0).float()
    right = ((bcode & 2) > 0).float()
    top = ((bcode & 4) > 0).float()
    bottom = ((bcode & 8) > 0).float()

    return torch.stack(
        [sqrt_a_norm, rel_a, fixed, preplaced, has_mib,
         has_cluster, left, right, top, bottom], dim=1)


def build_edges(b2b: torch.Tensor, block_count: int,
                device: str = "cpu") -> Tuple[torch.Tensor, torch.Tensor]:
    """b2b[E,3]=(i,j,w) -> (edge_index[2,2E], edge_weight[2E]) undirected.
    Drops padding rows (<0) and any endpoint >= block_count."""
    if b2b is None or b2b.numel() == 0:
        return (torch.zeros((2, 0), dtype=torch.long, device=device),
                torch.zeros((0,), device=device))
    valid = b2b[b2b[:, 0] >= 0]
    i = valid[:, 0].long()
    j = valid[:, 1].long()
    w = valid[:, 2].float()
    mask = (i < block_count) & (j < block_count) & (i >= 0) & (j >= 0)
    i, j, w = i[mask], j[mask], w[mask]
    src = torch.cat([i, j]).to(device)
    dst = torch.cat([j, i]).to(device)
    ew = torch.cat([w, w]).to(device)
    return torch.stack([src, dst]), ew


class SAGELayer(nn.Module):
    """One message-passing layer: weighted-mean neighbour aggregation."""

    def __init__(self, dim: int):
        super().__init__()
        self.w_self = nn.Linear(dim, dim)
        self.w_neigh = nn.Linear(dim, dim)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor,
                edge_weight: torch.Tensor) -> torch.Tensor:
        N = h.shape[0]
        if edge_index.shape[1] > 0:
            src, dst = edge_index[0], edge_index[1]
            msg = h[src] * edge_weight.unsqueeze(1)          # [E, dim]
            agg = torch.zeros_like(h).index_add(0, dst, msg)
            wsum = torch.zeros(N, device=h.device).index_add(0, dst, edge_weight)
            agg = agg / wsum.clamp(min=1e-6).unsqueeze(1)    # weighted mean
        else:
            agg = torch.zeros_like(h)
        return torch.relu(self.w_self(h) + self.w_neigh(agg))


class GNNEncoder(nn.Module):
    """netlist -> (node_emb[N, out_dim], graph_emb[out_dim])."""

    def __init__(self, node_in_dim: int = NODE_FEAT_DIM, hidden: int = 128,
                 n_layers: int = 3, out_dim: int = 128):
        super().__init__()
        self.input_proj = nn.Linear(node_in_dim, hidden)
        self.layers = nn.ModuleList([SAGELayer(hidden) for _ in range(n_layers)])
        self.node_out = nn.Linear(hidden, out_dim)
        self.graph_out = nn.Linear(hidden, out_dim)

    def forward(self, node_feats: torch.Tensor, edge_index: torch.Tensor,
                edge_weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = torch.relu(self.input_proj(node_feats))
        for layer in self.layers:
            h = layer(h, edge_index, edge_weight)
        node_emb = self.node_out(h)
        graph_emb = self.graph_out(h.mean(dim=0))
        return node_emb, graph_emb

    def encode_problem(self, area_target: torch.Tensor, constraints: torch.Tensor,
                       b2b: torch.Tensor, block_count: int, device: str = "cpu"):
        """Convenience: raw per-sample tensors -> (node_emb, graph_emb)."""
        feats = build_node_features(area_target, constraints, block_count).to(device)
        edge_index, edge_weight = build_edges(b2b, block_count, device)
        return self.forward(feats, edge_index, edge_weight)
