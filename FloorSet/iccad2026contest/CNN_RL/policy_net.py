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

Trunk (Phase 10 redesign): keeps full G x G resolution (no down/up-sampling),
but widens the receptive field two ways so each cell sees global structure:
  * a stack of **dilated** 3x3 convs (dilation 1,2,4,8,...) -> receptive field
    grows exponentially to cover the whole grid at the same resolution;
  * a **global-context branch**: global-average-pool the feature map -> MLP ->
    broadcast back to every cell, then fuse. Every cell gets a whole-layout
    summary (the old 4x plain-3x3 trunk only saw a ~9x9 neighbourhood, which
    correlated with the loose/spread-out placements observed in case 99).

Both the single-sample path (`forward`/`act`, used by the env rollout) and a
batched path (`forward_batch`, used by the fast trainer) share `_trunk`, which
accepts either `[C,G,G]` or `[B,C,G,G]`.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyValueNet(nn.Module):
    def __init__(self, in_channels: int = 4, node_dim: int = 128,
                 node_channels: int = 32, hidden: int = 64, n_conv: int = 4,
                 n_aspect: int = 5, dilations: Tuple[int, ...] = None):
        super().__init__()
        self.node_proj = nn.Linear(node_dim, node_channels)

        chans = in_channels + node_channels
        self.stem = nn.Conv2d(chans, hidden, 3, padding=1)

        # Dilated residual conv stack: dilation doubles each layer so the
        # receptive field grows exponentially (1,2,4,8,...) at fixed resolution.
        if dilations is None:
            dilations = tuple(2 ** k for k in range(n_conv))
        self.dil_convs = nn.ModuleList(
            [nn.Conv2d(hidden, hidden, 3, padding=d, dilation=d) for d in dilations])

        # Global-context branch: pool -> MLP -> broadcast -> fuse back.
        self.global_mlp = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.fuse = nn.Conv2d(hidden * 2, hidden, 1)

        self.policy_head = nn.Conv2d(hidden, 1, kernel_size=1)
        self.value_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        # Phase 5: aspect-ratio head (picks a shape bucket for the current block)
        self.aspect_head = nn.Linear(hidden, n_aspect)

    def _trunk(self, canvas: torch.Tensor, node_emb_current: torch.Tensor):
        """Shared CNN trunk. Accepts canvas [C,G,G] or [B,C,G,G] and
        node_emb_current [D] or [B,D]. Returns (logits, pooled) with a leading
        batch dim iff the input had one (single-sample -> [G,G],[hidden])."""
        single = (canvas.dim() == 3)
        if single:
            canvas = canvas.unsqueeze(0)                       # [1,C,G,G]
            node_emb_current = node_emb_current.unsqueeze(0)   # [1,D]
        B, _C, G, _ = canvas.shape

        node_vec = self.node_proj(node_emb_current)            # [B,nc]
        node_map = node_vec.view(B, -1, 1, 1).expand(B, -1, G, G)
        x = torch.cat([canvas, node_map], dim=1)               # [B,C+nc,G,G]

        x = F.relu(self.stem(x))                               # [B,hidden,G,G]
        for conv in self.dil_convs:
            x = F.relu(conv(x) + x)                            # residual dilated

        g = self.global_mlp(x.mean(dim=(2, 3)))                # [B,hidden]
        g_map = g.view(B, -1, 1, 1).expand(B, -1, G, G)        # [B,hidden,G,G]
        x = F.relu(self.fuse(torch.cat([x, g_map], dim=1)))    # [B,hidden,G,G]

        logits = self.policy_head(x).view(B, G, G)             # [B,G,G]
        pooled = x.mean(dim=(2, 3))                            # [B,hidden]
        if single:
            return logits[0], pooled[0]
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
        value = self.value_head(pooled).squeeze()             # scalar
        return probs, value, logits

    def forward_batch(self, canvas: torch.Tensor, node_emb_current: torch.Tensor,
                      mask: torch.Tensor = None):
        """Batched training path. canvas [B,C,G,G], node_emb_current [B,D].
        Returns (logits[B,G*G], value[B], aspect_logits[B,n_aspect]). Position
        logits are RAW (unmasked) for a stable cross-entropy target (BC trains
        on raw logits, exactly like the old single-sample path; the
        feasibility mask is applied only at action time in `act`). `mask` is
        accepted for API symmetry but unused here."""
        logits, pooled = self._trunk(canvas, node_emb_current)  # [B,G,G],[B,hidden]
        B = logits.shape[0]
        value = self.value_head(pooled).squeeze(-1)             # [B]
        aspect_logits = self.aspect_head(pooled)                 # [B,n_aspect]
        return logits.view(B, -1), value, aspect_logits

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
