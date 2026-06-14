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
    `current_block` via b2b edges. positions rows that are NaN are unplaced."""
    if b2b is None or b2b.numel() == 0:
        return None, None
    valid = b2b[b2b[:, 0] >= 0]
    cxs, cys, ws = [], [], []
    for edge in valid:
        i, j, w = int(edge[0]), int(edge[1]), float(edge[2])
        other = None
        if i == current_block:
            other = j
        elif j == current_block:
            other = i
        if other is None or other >= positions.shape[0]:
            continue
        row = positions[other]
        if torch.isnan(row[0]):
            continue  # neighbour not placed yet -> no signal
        ox, oy, ow, oh = (float(v) for v in row)
        cxs.append(ox + ow / 2)
        cys.append(oy + oh / 2)
        ws.append(w)
    if not cxs:
        return None, None
    centers = torch.tensor([cxs, cys]).t()  # [K,2]
    weights = torch.tensor(ws)              # [K]
    return centers, weights


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
    density = out[CH_DENSITY]
    for i in range(positions.shape[0]):
        row = positions[i]
        if torch.isnan(row[0]):
            continue
        x, y, w, h = (float(v) for v in row)
        c0 = max(int(x / cell_w), 0)
        r0 = max(int(y / cell_h), 0)
        c1 = min(int(math.ceil((x + w) / cell_w)), G)
        r1 = min(int(math.ceil((y + h) / cell_h)), G)
        density[r0:r1, c0:c1] += 1.0
    out[CH_OCCUPANCY] = (density > 0).float()

    # --- channel 2: wirelength pull for the current block ----------------
    if current_block is not None and current_dims is not None:
        cw, ch = current_dims
        centers, weights = _placed_neighbors(current_block, positions, b2b)
        if centers is not None:
            # cell-center coords of where this block's CENTER would land
            cols = torch.arange(G, device=device).float()
            rows = torch.arange(G, device=device).float()
            cx = cols * cell_w + cw / 2            # [G]  (x per col)
            cy = rows * cell_h + ch / 2            # [G]  (y per row)
            cx_grid = cx.view(1, G).expand(G, G)   # [G,G]
            cy_grid = cy.view(G, 1).expand(G, G)   # [G,G]
            cost = torch.zeros((G, G), device=device)
            for k in range(centers.shape[0]):
                nx, ny = float(centers[k, 0]), float(centers[k, 1])
                cost += float(weights[k]) * (
                    (cx_grid - nx).abs() + (cy_grid - ny).abs())
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


def rasterize_env(env) -> torch.Tensor:
    """Adapter: build the rasterized state directly from a PlacementEnv.
    Uses the env's Phase-0 square-block dims for the current block."""
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
        device=env.device,
    )
