"""
Phase 1 — Canvas rasterizer (VERSION_B_RL_PLACER.md §5 Phase 1).

Turns the current (partial) placement into a multi-channel grid image — the
"vision" input that the CNN in policy_net.py (Phase 3) will consume. This
replaces the occupancy placeholder in placement_env.py.

Channels (C=4), each [G, G]:
  0 occupancy   : 1 where any already-placed block covers the cell
  1 density     : how many placed blocks cover the cell (crowding / overlap)
  2 wl_pull     : for the block about to be placed, a [0,1] field that is high
                  where placing it would give LOW weighted HPWL to its already
                  -placed connected neighbours (i.e. "pull" toward good spots)
  3 feasibility : 1 where the current block (given its w,h) fits fully inside
                  the canvas if its lower-left corner is in that cell
                  (boundary/hard constraints come in Phase 6)

Coordinate convention matches placement_env.py:
  grid is indexed [row, col]; row -> y (up), col -> x (right);
  cell_w = canvas_w / G, cell_h = canvas_h / G; a cell's lower-left is
  (col*cell_w, row*cell_h). Origin (0,0) is the canvas lower-left.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch

N_CHANNELS = 4
CH_OCCUPANCY = 0
CH_DENSITY = 1
CH_WL_PULL = 2
CH_FEASIBILITY = 3


def _placed_neighbors(current_block: int,
                      positions: torch.Tensor,
                      b2b: torch.Tensor):
    """Return (centers[K,2], weights[K]) of already-placed blocks connected to
    `current_block` via b2b edges. positions rows that are NaN are unplaced.
    Fully vectorised (no Python edge loop)."""
    if b2b is None or b2b.numel() == 0:
        return None, None
    valid = b2b[b2b[:, 0] >= 0]
    if valid.numel() == 0:
        return None, None
    i = valid[:, 0].long()
    j = valid[:, 1].long()
    w = valid[:, 2]
    touch_i = i == current_block
    touch_j = j == current_block
    touch = touch_i | touch_j
    other = torch.where(touch_i, j, i)               # the "other" endpoint
    P = positions.shape[0]
    sel = touch & (other < P)
    other = other[sel]
    ww = w[sel]
    if other.numel() == 0:
        return None, None
    rows = positions[other]                          # [K,4] = x,y,w,h
    placed = ~torch.isnan(rows[:, 0])
    rows = rows[placed]
    ww = ww[placed]
    if rows.numel() == 0:
        return None, None
    centers = rows[:, :2] + rows[:, 2:] / 2          # [K,2] = cx,cy
    return centers, ww


def rasterize(positions: torch.Tensor,
              canvas: Tuple[float, float],
              grid: int,
              current_block: Optional[int] = None,
              current_dims: Optional[Tuple[float, float]] = None,
              b2b: Optional[torch.Tensor] = None,
              device: str = "cpu") -> torch.Tensor:
    """Build the [N_CHANNELS, grid, grid] state image. See module docstring."""
    canvas_w, canvas_h = canvas
    G = grid
    cell_w = canvas_w / G
    cell_h = canvas_h / G

    out = torch.zeros((N_CHANNELS, G, G), device=device)

    # --- channels 0,1: occupancy / density -------------------------------
    # Vectorised rectangle accumulation via a 2D difference (integral) image:
    # add +1/-1 at the four corners of each placed block's cell-bbox, then a
    # 2D prefix-sum recovers the per-cell coverage count in one shot (no loop).
    pos = positions.to(device)
    placed = ~torch.isnan(pos[:, 0])
    if placed.any():
        P = pos[placed]                                        # [K,4] x,y,w,h
        c0 = (P[:, 0] / cell_w).floor().clamp(min=0, max=G).long()
        r0 = (P[:, 1] / cell_h).floor().clamp(min=0, max=G).long()
        c1 = torch.ceil((P[:, 0] + P[:, 2]) / cell_w).clamp(min=0, max=G).long()
        r1 = torch.ceil((P[:, 1] + P[:, 3]) / cell_h).clamp(min=0, max=G).long()
        diff = torch.zeros((G + 1, G + 1), device=device)
        ones = torch.ones_like(c0, dtype=diff.dtype)
        diff.index_put_((r0, c0), ones, accumulate=True)
        diff.index_put_((r0, c1), -ones, accumulate=True)
        diff.index_put_((r1, c0), -ones, accumulate=True)
        diff.index_put_((r1, c1), ones, accumulate=True)
        density = diff.cumsum(0).cumsum(1)[:G, :G]
        out[CH_DENSITY] = density
        out[CH_OCCUPANCY] = (density > 0).float()

    # --- channel 2: wirelength pull for the current block ----------------
    if current_block is not None and current_dims is not None:
        cw, ch = current_dims
        centers, weights = _placed_neighbors(current_block, positions, b2b)
        if centers is not None:
            centers = centers.to(device)
            weights = weights.to(device)
            # cell-center coords of where this block's CENTER would land
            cols = torch.arange(G, device=device).float()
            rows = torch.arange(G, device=device).float()
            cx = cols * cell_w + cw / 2            # [G]  (x per col)
            cy = rows * cell_h + ch / 2            # [G]  (y per row)
            cx_grid = cx.view(1, 1, G)             # [1,1,G]
            cy_grid = cy.view(1, G, 1)             # [1,G,1]
            nx = centers[:, 0].view(-1, 1, 1)      # [K,1,1]
            ny = centers[:, 1].view(-1, 1, 1)      # [K,1,1]
            # weighted Manhattan distance summed over neighbours, vectorised
            cost = (weights.view(-1, 1, 1)
                    * ((cx_grid - nx).abs() + (cy_grid - ny).abs())).sum(0)  # [G,G]
            # invert + normalise to [0,1]: high pull == low HPWL cost
            cmin, cmax = cost.min(), cost.max()
            if (cmax - cmin) > 1e-9:
                out[CH_WL_PULL] = 1.0 - (cost - cmin) / (cmax - cmin)
            else:
                out[CH_WL_PULL] = torch.ones((G, G), device=device)

    # --- channel 3: feasibility mask -------------------------------------
    if current_dims is not None:
        cw, ch = current_dims
        cols = torch.arange(G, device=device).float()
        rows = torch.arange(G, device=device).float()
        x_ll = cols * cell_w
        y_ll = rows * cell_h
        col_ok = (x_ll + cw) <= (canvas_w + 1e-6)   # [G]
        row_ok = (y_ll + ch) <= (canvas_h + 1e-6)   # [G]
        out[CH_FEASIBILITY] = (row_ok.view(G, 1) & col_ok.view(1, G)).float()
    else:
        out[CH_FEASIBILITY] = torch.ones((G, G), device=device)

    return out


def rasterize_env(env, device: Optional[str] = None) -> torch.Tensor:
    """Adapter: build the rasterized state directly from a PlacementEnv.
    Uses the env's Phase-0 square-block dims for the current block.
    `device` overrides where the (vectorised) raster is built — pass "cuda" in
    training to keep the heavy grid ops off the CPU (the per-step rasterizer is
    the training bottleneck)."""
    state = env._build_state()
    current_block = state["current_block"]
    current_dims = None
    if current_block is not None:
        current_dims = env._block_dims(current_block)
    return rasterize(
        positions=env.positions,
        canvas=(env.canvas_w, env.canvas_h),
        grid=env.grid,
        current_block=current_block,
        current_dims=current_dims,
        b2b=env.b2b,
        device=device or env.device,
    )
