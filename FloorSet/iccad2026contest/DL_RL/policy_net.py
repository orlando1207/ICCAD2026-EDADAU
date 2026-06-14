"""
Phase 3 — Policy / Value network (VERSION_B_RL_PLACER.md §5 Phase 3).

Combines the two encoders into the actor-critic head:
  * CNN reads the canvas raster (canvas_raster.py, [4, G, G]).
  * The current block's GNN embedding (gnn_encoder.py, [D]) is broadcast as
    extra channels so the spatial decision is conditioned on *which* block
    is being placed and its graph context.
  * Policy head -> per-cell logits [G, G], masked by feasibility, softmaxed
    into a distribution over the G*G lower-left placement cells.
  * Value head -> scalar V(s) for the PPO baseline (Phase 4).

Single-sample (one problem at a time), matching PlacementEnv. A batch dim is
added/removed internally so the convs work.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyValueNet(nn.Module):
    def __init__(self, in_channels: int = 4, node_dim: int = 128,
                 node_channels: int = 32, hidden: int = 64, n_conv: int = 4,
                 n_aspect: int = 5):
        super().__init__()
        self.node_proj = nn.Linear(node_dim, node_channels)

        chans = in_channels + node_channels
        convs = []
        for k in range(n_conv):
            convs += [nn.Conv2d(chans if k == 0 else hidden, hidden, 3, padding=1),
                      nn.ReLU()]
        self.cnn = nn.Sequential(*convs)

        self.policy_head = nn.Conv2d(hidden, 1, kernel_size=1)
        self.value_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        # Phase 5: aspect-ratio head (picks a shape bucket for the current block)
        self.aspect_head = nn.Linear(hidden, n_aspect)

    def _trunk(self, canvas: torch.Tensor, node_emb_current: torch.Tensor):
        """Shared CNN trunk -> (position logits [G,G], pooled feature [hidden])."""
        _C, G, _ = canvas.shape
        x = canvas.unsqueeze(0)                                   # [1,C,G,G]
        node_vec = self.node_proj(node_emb_current)              # [node_channels]
        node_map = node_vec.view(1, -1, 1, 1).expand(1, -1, G, G)
        x = torch.cat([x, node_map], dim=1)                      # [1,C+nc,G,G]
        feat = self.cnn(x)                                       # [1,hidden,G,G]
        logits = self.policy_head(feat).view(G, G)              # [G,G]
        pooled = feat.mean(dim=(2, 3)).view(-1)                  # [hidden]
        return logits, pooled

    @staticmethod
    def _masked_probs(logits: torch.Tensor, feasibility_mask: torch.Tensor):
        mask = (feasibility_mask > 0.5)
        masked = logits.masked_fill(~mask, float("-inf")) if mask.any() else logits
        return F.softmax(masked.view(-1), dim=0)

    def forward(self, canvas: torch.Tensor, node_emb_current: torch.Tensor,
                feasibility_mask: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """canvas [C,G,G], node_emb_current [D], feasibility_mask [G,G] (1=legal).
        Returns (probs[G*G], value scalar, logits[G,G]). Position-only (Phase 0-4.5)."""
        logits, pooled = self._trunk(canvas, node_emb_current)
        probs = self._masked_probs(logits, feasibility_mask)
        value = self.value_head(pooled).squeeze()               # scalar
        return probs, value, logits

    def forward_aspect(self, canvas: torch.Tensor, node_emb_current: torch.Tensor,
                       feasibility_mask: torch.Tensor):
        """Phase 5: also produce an aspect-ratio distribution.
        Returns (pos_probs[G*G], aspect_probs[n_aspect], value, pos_logits[G,G])."""
        logits, pooled = self._trunk(canvas, node_emb_current)
        probs = self._masked_probs(logits, feasibility_mask)
        value = self.value_head(pooled).squeeze()
        aspect_probs = F.softmax(self.aspect_head(pooled), dim=0)
        return probs, aspect_probs, value, logits

    @torch.no_grad()
    def act(self, canvas: torch.Tensor, node_emb_current: torch.Tensor,
            feasibility_mask: torch.Tensor, greedy: bool = False):
        """Sample (or argmax) a position action; returns (action:int, logprob, value)."""
        probs, value, _ = self.forward(canvas, node_emb_current, feasibility_mask)
        action = int(torch.argmax(probs)) if greedy else int(torch.multinomial(probs, 1))
        logprob = torch.log(probs[action].clamp_min(1e-12))
        return action, float(logprob), float(value)

    @torch.no_grad()
    def act_aspect(self, canvas: torch.Tensor, node_emb_current: torch.Tensor,
                   feasibility_mask: torch.Tensor, greedy: bool = False):
        """Phase 5: sample (or argmax) both a position and an aspect bucket.
        Returns (action:int, aspect_idx:int, logprob_total, value)."""
        probs, aspect_probs, value, _ = self.forward_aspect(
            canvas, node_emb_current, feasibility_mask)
        if greedy:
            action = int(torch.argmax(probs))
            aspect_idx = int(torch.argmax(aspect_probs))
        else:
            action = int(torch.multinomial(probs, 1))
            aspect_idx = int(torch.multinomial(aspect_probs, 1))
        logp = (torch.log(probs[action].clamp_min(1e-12))
                + torch.log(aspect_probs[aspect_idx].clamp_min(1e-12)))
        return action, aspect_idx, float(logp), float(value)
