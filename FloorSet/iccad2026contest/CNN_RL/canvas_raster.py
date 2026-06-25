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

N_CHANNELS = 5
CH_OCCUPANCY = 0
CH_DENSITY = 1
CH_WL_PULL = 2
CH_FEASIBILITY = 3
CH_PIN_PULL = 4          # HPWL pull toward the current block's connected I/O pins (p2b)


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


def _connected_pins(current_block: int,
                    p2b: torch.Tensor,
                    pins: torch.Tensor):
    """Return (pin_centers[K,2], weights[K]) of I/O pins connected to
    `current_block` via p2b edges. Pins are fixed terminals (always "placed"),
    so unlike b2b neighbours there is no placed/unplaced filter. p2b rows are
    (pin_idx, block_idx, weight). Fully vectorised."""
    if p2b is None or p2b.numel() == 0 or pins is None or pins.numel() == 0:
        return None, None
    valid = p2b[p2b[:, 0] >= 0]
    if valid.numel() == 0:
        return None, None
    pin_i = valid[:, 0].long()
    blk_j = valid[:, 1].long()
    w = valid[:, 2]
    sel = (blk_j == current_block) & (pin_i < pins.shape[0])
    pin_i = pin_i[sel]
    ww = w[sel]
    if pin_i.numel() == 0:
        return None, None
    centers = pins[pin_i][:, :2].float()             # [K,2] = (px, py)
    return centers, ww


def _pull_field(centers, weights, G, cell_w, cell_h, cw, ch, device,
                anchor: str = "ll"):
    """[0,1] field over the G×G grid: high where placing the current block's
    CENTER gives LOW total weighted Manhattan distance to `centers`. Shared by
    the b2b (wl_pull) and pin (pin_pull) channels.

    The block-center implied by a cell depends on the action anchoring (must
    match placement_env._action_to_xy): for "ll" the cell is the lower-left
    corner so center = corner + (cw/2, ch/2) — size-dependent; for "center" the
    cell IS the center so center = cell center — size-invariant."""
    centers = centers.to(device)
    weights = weights.to(device)
    cols = torch.arange(G, device=device).float()
    rows = torch.arange(G, device=device).float()
    ox = cell_w / 2 if anchor == "center" else cw / 2
    oy = cell_h / 2 if anchor == "center" else ch / 2
    cx = (cols * cell_w + ox).view(1, 1, G)           # [1,1,G] block-center x per col
    cy = (rows * cell_h + oy).view(1, G, 1)           # [1,G,1] block-center y per row
    nx = centers[:, 0].view(-1, 1, 1)
    ny = centers[:, 1].view(-1, 1, 1)
    cost = (weights.view(-1, 1, 1)
            * ((cx - nx).abs() + (cy - ny).abs())).sum(0)   # [G,G]
    cmin, cmax = cost.min(), cost.max()
    if (cmax - cmin) > 1e-9:
        return 1.0 - (cost - cmin) / (cmax - cmin)
    return torch.ones((G, G), device=device)


def rasterize(positions: torch.Tensor,
              canvas: Tuple[float, float],
              grid: int,
              current_block: Optional[int] = None,
              current_dims: Optional[Tuple[float, float]] = None,
              b2b: Optional[torch.Tensor] = None,
              p2b: Optional[torch.Tensor] = None,
              pins: Optional[torch.Tensor] = None,
              device: str = "cpu",
              anchor: str = "ll") -> torch.Tensor:
    """Build the [N_CHANNELS, grid, grid] state image. See module docstring.
    `anchor` ("ll"|"center") must match placement_env._action_to_xy so the
    wl/pin pull and feasibility channels are in the same frame as the action."""
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

    # --- channel 2: b2b wirelength pull; channel 4: pin (p2b) pull -------
    if current_block is not None and current_dims is not None:
        cw, ch = current_dims
        centers, weights = _placed_neighbors(current_block, positions, b2b)
        if centers is not None:
            out[CH_WL_PULL] = _pull_field(centers, weights, G, cell_w, cell_h,
                                          cw, ch, device, anchor)
        pin_centers, pin_w = _connected_pins(current_block, p2b, pins)
        if pin_centers is not None:
            out[CH_PIN_PULL] = _pull_field(pin_centers, pin_w, G, cell_w, cell_h,
                                           cw, ch, device, anchor)

    # --- channel 3: feasibility mask -------------------------------------
    if current_dims is not None:
        cw, ch = current_dims
        cols = torch.arange(G, device=device).float()
        rows = torch.arange(G, device=device).float()
        if anchor == "center":
            # cell is the block CENTER: feasible iff the block fits centered there
            # (lower-left = center - dims/2 stays inside [0, canvas]).
            cx = (cols + 0.5) * cell_w
            cy = (rows + 0.5) * cell_h
            col_ok = ((cx - cw / 2) >= -1e-6) & ((cx + cw / 2) <= (canvas_w + 1e-6))
            row_ok = ((cy - ch / 2) >= -1e-6) & ((cy + ch / 2) <= (canvas_h + 1e-6))
        else:
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
        p2b=getattr(env, "p2b", None),
        pins=getattr(env, "pins_pos", None),
        device=device or env.device,
        anchor=getattr(env, "anchor", "ll"),
    )
