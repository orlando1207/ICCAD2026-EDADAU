"""
JointPlacer — GNN + self-attention -> per-block continuous (x, y, aspect).

Why this exists (see ../../ARCHITECTURE.md and the chat thread):
the CNN_RL placer is grid-quantized (64x64), sequential/greedy, and trained by
behaviour cloning. This model removes all three limits at once:

  * CONTINUOUS  : positions are real numbers, no grid snapping.
  * JOINT       : a self-attention stack sees every block together and emits all
                  placements in ONE shot, so block shape and position co-adapt
                  (the "Option B" lever the slack-shaping PoC found is the real
                  driver of Area_gap).
  * COST-TRAINED: trained directly on the differentiable contest cost
                  (compute_training_loss_differentiable), not on imitation.

Area is kept EXACT by construction: the head predicts only a log-aspect, and
w,h are derived from the known target area (w=sqrt(a*ar), h=sqrt(a/ar)), so the
area-tolerance violation term is 0 by design — one fewer thing for the optimizer
to fight.

Scope of THIS PoC: it measures whether "continuous + joint + cost-trained" can
drive the soft quality terms (HPWL_gap, Area_gap, overlap) below the grid+greedy
baseline. Hard-constraint enforcement (boundary/MIB/cluster/preplaced) and the
skyline legalize tail are deliberately OUT of scope here — same philosophy as
poc_slack_shaping.py: measure the lever first, productionize second.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

# reuse the existing GNN encoder from CNN_RL/
_CNN_RL = Path(__file__).resolve().parent.parent
if str(_CNN_RL) not in sys.path:
    sys.path.insert(0, str(_CNN_RL))
from gnn_encoder import GNNEncoder  # noqa: E402


class JointPlacer(nn.Module):
    def __init__(self, gnn_out: int = 128, d_model: int = 128, n_heads: int = 4,
                 n_attn: int = 3, ar_cap: float = 4.0):
        super().__init__()
        self.gnn = GNNEncoder(out_dim=gnn_out)
        self.proj = nn.Linear(gnn_out, d_model)
        enc = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=4 * d_model,
                                         batch_first=True, dropout=0.0)
        self.attn = nn.TransformerEncoder(enc, n_attn)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(),
            nn.Linear(d_model, 3),                 # -> (x_raw, y_raw, log_aspect_raw)
        )
        self.ar_cap = ar_cap

    def forward(self, area_target, constraints, b2b, block_count):
        """Returns raw head output [block_count, 3]."""
        node_emb, _ = self.gnn.encode_problem(area_target, constraints, b2b, block_count)
        h = self.proj(node_emb).unsqueeze(0)       # [1, N, d]
        h = self.attn(h).squeeze(0)                # [N, d]
        return self.head(h)                        # [N, 3]

    def positions(self, raw, area_target, block_count, scale):
        """Map raw head output -> positions [N,4] = (x, y, w, h), area-exact.

        x,y in [0, scale] via sigmoid (lower-left corner); aspect in
        [1/ar_cap, ar_cap] via tanh, so w*h == area exactly."""
        a = area_target[:block_count].float().clamp(min=1e-6)
        log_ar = torch.tanh(raw[:, 2]) * math.log(self.ar_cap)
        ar = torch.exp(log_ar)
        w = torch.sqrt(a * ar)
        h = torch.sqrt(a / ar)
        x = torch.sigmoid(raw[:, 0]) * scale
        y = torch.sigmoid(raw[:, 1]) * scale
        return torch.stack([x, y, w, h], dim=1)

    @staticmethod
    def canvas_scale(area_target, block_count, margin: float = 1.4) -> float:
        a = area_target[:block_count].float().clamp(min=0.0)
        return float(math.sqrt(float(a.sum()) * margin))
