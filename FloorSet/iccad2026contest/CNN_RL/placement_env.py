"""
Phase 0 — RL placement environment skeleton.

See VERSION_B_RL_PLACER.md §5 Phase 0.

Gym-style environment that wraps ONE floorplan problem and exposes the
"place one block at a time" MDP:

    state = env.reset(problem)
    while not done:
        action = policy(state)          # an integer grid cell in [0, G*G)
        state, reward, done, info = env.step(action)

Scope of Phase 0 (deliberately minimal — do NOT over-build):
  * blocks are treated as squares (w = h = sqrt(area)); real soft-block
    aspect-ratio actions arrive in Phase 5.
  * no hard-constraint handling (fixed/preplaced/boundary/MIB) — Phase 6.
  * reward is terminal only: -(contest cost). Per-step shaping is Phase 4 (§4).
  * the "canvas grid" in the state is a simple occupancy map placeholder;
    the real multi-channel rasterizer is Phase 1 (canvas_raster.py).

The reward reuses the official differentiable contest cost so we never
re-derive the scoring formula (VERSION_B_RL_PLACER.md §3).
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import torch

# Make `iccad2026contest.iccad2026_evaluate` importable regardless of cwd.
# DL_RL/ -> iccad2026contest/ -> FloorSet/   (FloorSet must be on path)
_FLOORSET_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_FLOORSET_ROOT) not in sys.path:
    sys.path.insert(0, str(_FLOORSET_ROOT))

from iccad2026contest.iccad2026_evaluate import compute_training_loss_differentiable


def estimate_canvas(area_target: torch.Tensor,
                    pins_pos: torch.Tensor,
                    area_margin: float = 1.6) -> Tuple[float, float]:
    """Pick a canvas (W, H) that (a) has enough area for all blocks and
    (b) contains every pin. Pins are fixed terminals the placement must
    reach, so they are a hard lower bound on the canvas extent.

    NOTE (Phase 16, reverted): tried sizing the canvas to the PIN BOUNDING BOX
    (1.23× GT) instead of `sqrt(area·1.6)` (1.80× GT), expecting tighter centers
    → smaller output bbox → lower Area_gap. **It did not work** — Area_gap stayed
    0.477→0.484, output bbox 1.47×→1.48×, Total Score regressed 1.8517→1.8688.
    Reason: `skyline_legalize` chooses its own strip width and re-packs the
    *relative* arrangement (scale-invariant) independently of the canvas, so the
    canvas scale washes out of the final geometry. Area_gap is set by how skyline
    packs the arrangement (→ block reshaping), not by the canvas size. Reverted.
    """
    valid_area = area_target[area_target > 0]
    total_area = float(valid_area.sum()) if valid_area.numel() else 1.0
    base = math.sqrt(total_area * area_margin)

    pin_x_max = float(pins_pos[:, 0].max()) if pins_pos.numel() else 0.0
    pin_y_max = float(pins_pos[:, 1].max()) if pins_pos.numel() else 0.0

    w = max(base, pin_x_max)
    h = max(base, pin_y_max)
    return w, h


def placement_order(area_target: torch.Tensor,
                    constraints: Optional[torch.Tensor],
                    block_count: int) -> list[int]:
    """Fixed placement order (not learned in Phase 0):
        1. fixed / preplaced blocks first (so they anchor the canvas)
        2. remaining blocks by area, largest first

    constraints columns: (fixed, preplaced, mib_group, cluster_group, boundary).
    Phase 0 only *orders* by these flags; actually honouring the constraints
    (overriding shape/position) is Phase 6.
    """
    idx = list(range(block_count))

    def is_anchored(i: int) -> bool:
        if constraints is None:
            return False
        return bool(constraints[i, 0] > 0 or constraints[i, 1] > 0)

    anchored = [i for i in idx if is_anchored(i)]
    free = [i for i in idx if not is_anchored(i)]
    free.sort(key=lambda i: float(area_target[i]), reverse=True)
    return anchored + free


class PlacementEnv:
    """Sequential single-block placement environment for one problem."""

    def __init__(self, grid: int = 84, area_margin: float = 1.6,
                 device: str = "cpu"):
        self.grid = grid
        self.area_margin = area_margin
        self.device = device
        # Action anchoring (see `_action_to_xy`). "ll" (default) = the grid cell
        # is the block's LOWER-LEFT corner, origin (0,0) at canvas lower-left
        # (the convention every shipped checkpoint was trained under). "center" =
        # the cell is the block's CENTER (Mirhoseini-style): size-invariant, so a
        # bottom-touching block reports the same cy regardless of its height,
        # removing the height leak into the position signal the skyline sorts on.
        # Toggle without code edits via PLACE_ANCHOR=center.
        self.anchor = os.environ.get("PLACE_ANCHOR", "ll").lower()
        # Inference fast-path flags (default off = training/test behaviour). The
        # greedy rollout in RLSkylineOptimizer reads only `current_block` from the
        # state and discards the reward, so the placeholder occupancy grid and the
        # terminal contest-cost are pure waste during inference.
        self.skip_occupancy = False
        self.skip_terminal_cost = False
        self._reset_internal_state()

    # ------------------------------------------------------------------ #
    # Spaces
    # ------------------------------------------------------------------ #
    @property
    def action_space_size(self) -> int:
        """One discrete action per grid cell (the lower-left corner cell)."""
        return self.grid * self.grid

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def _reset_internal_state(self) -> None:
        self.area_target = None
        self.b2b = None
        self.p2b = None
        self.pins_pos = None
        self.constraints = None
        self.metrics = None
        self.block_count = 0
        self.order = []
        self.ptr = 0
        self.canvas_w = 0.0
        self.canvas_h = 0.0
        # positions[i] = (x, y, w, h); NaN rows = not yet placed.
        self.positions = None
        # Phase 6 hard-constraint state
        self.target_positions = None      # [N,4] spec, -1 = free
        self.fixed_dims = {}              # block_idx -> (w, h) forced
        self.mib_dims = {}                # block_idx -> (w, h) shared in MIB group
        self.boundary_codes = None        # [N] boundary code per block

    def reset(self, area_target: torch.Tensor, b2b: torch.Tensor,
              p2b: torch.Tensor, pins_pos: torch.Tensor,
              constraints: torch.Tensor, metrics: torch.Tensor,
              target_positions: Optional[torch.Tensor] = None) -> dict:
        """Start an episode for one problem. Tensors are the raw per-sample
        tensors from FloorplanDatasetLiteTest (see VERSION_B_RL_PLACER.md §3).

        target_positions[N,4]=(x,y,w,h), -1=free (Phase 6): preplaced blocks
        have all four set (placed immediately, excluded from RL order);
        fixed-shape blocks have (w,h) set (dims forced, position learned)."""
        self._reset_internal_state()

        self.area_target = area_target.to(self.device).float()
        self.b2b = b2b.to(self.device).float()
        self.p2b = p2b.to(self.device).float()
        self.pins_pos = pins_pos.to(self.device).float()
        self.constraints = constraints.to(self.device).float()
        self.metrics = metrics.to(self.device).float()

        self.block_count = int((self.area_target != -1).sum().item())
        self.canvas_w, self.canvas_h = estimate_canvas(
            self.area_target[:self.block_count], self.pins_pos, self.area_margin)

        self.order = placement_order(
            self.area_target, self.constraints, self.block_count)
        self.ptr = 0
        self.positions = torch.full((self.block_count, 4), float("nan"),
                                    device=self.device)

        # ---- Phase 6: hard-constraint setup ----
        self.boundary_codes = self.constraints[:self.block_count, 4].long()
        if target_positions is not None:
            tp = target_positions.to(self.device).float()
            self.target_positions = tp
            for i in range(self.block_count):
                if self.constraints[i, 0] > 0:           # fixed-shape dims
                    self.fixed_dims[i] = (float(tp[i, 2]), float(tp[i, 3]))
            preplaced = {i for i in range(self.block_count)
                         if self.constraints[i, 1] > 0}
            for i in preplaced:                          # place + exclude from RL
                self.positions[i] = tp[i].clone()
            self.order = [i for i in self.order if i not in preplaced]

        # MIB: members of a group share one (w,h); prefer a fixed/preplaced
        # member's spec dims, else fall back to the square of the area.
        groups = {}
        for i in range(self.block_count):
            g = int(self.constraints[i, 2])
            if g > 0:
                groups.setdefault(g, []).append(i)
        for members in groups.values():
            ref = None
            for i in members:
                if i in self.fixed_dims:
                    ref = self.fixed_dims[i]
                    break
                if (self.target_positions is not None
                        and self.constraints[i, 1] > 0):
                    ref = (float(self.target_positions[i, 2]),
                           float(self.target_positions[i, 3]))
                    break
            if ref is None:
                s = math.sqrt(max(float(self.area_target[members[0]]), 1e-6))
                ref = (s, s)
            for i in members:
                self.mib_dims[i] = ref

        return self._build_state()

    # ------------------------------------------------------------------ #
    # Step
    # ------------------------------------------------------------------ #
    def _block_dims(self, block_idx: int, aspect: float = 1.0) -> Tuple[float, float]:
        """Block (w, h) from area + aspect ratio (Phase 5). aspect = w/h.
        aspect=1.0 -> square (Phase 0 default). Area is exact for any aspect:
        w = sqrt(area*aspect), h = sqrt(area/aspect) -> w*h = area.
        Phase 6 overrides fixed-shape blocks with their given (w, h)."""
        if block_idx in self.fixed_dims:                 # fixed-shape: immutable
            return self.fixed_dims[block_idx]
        if block_idx in self.mib_dims:                   # MIB: shared shape
            return self.mib_dims[block_idx]
        area = max(float(self.area_target[block_idx]), 1e-6)
        a = max(aspect, 1e-6)
        return math.sqrt(area * a), math.sqrt(area / a)

    def _snap_boundary(self, block_idx: int, x: float, y: float,
                       w: float, h: float):
        """Phase 6: snap a boundary-constrained block to the canvas edge(s) it
        must touch (enforced, not just masked). LEFT=1 RIGHT=2 TOP=4 BOTTOM=8."""
        code = int(self.boundary_codes[block_idx]) if self.boundary_codes is not None else 0
        if code & 1:               # LEFT
            x = 0.0
        if code & 2:               # RIGHT
            x = self.canvas_w - w
        if code & 8:               # BOTTOM
            y = 0.0
        if code & 4:               # TOP
            y = self.canvas_h - h
        return x, y

    def _action_to_xy(self, action: int, w: float, h: float) -> Tuple[float, float]:
        """Map a flat grid-cell action to a clamped lower-left (x, y).

        anchor="ll"     : cell -> lower-left corner (origin (0,0) at canvas LL).
                          Size-dependent center: cy = row*cell_h + h/2.
        anchor="center" : cell -> block CENTER (cx,cy at the cell center), then
                          back out the lower-left. Size-invariant center: a
                          bottom-row block centers at cell_h/2 whatever its h.
        """
        row, col = divmod(int(action), self.grid)
        cell_w = self.canvas_w / self.grid
        cell_h = self.canvas_h / self.grid
        if self.anchor == "center":
            cx = (col + 0.5) * cell_w
            cy = (row + 0.5) * cell_h
            x = cx - w / 2.0
            y = cy - h / 2.0
        else:
            x = col * cell_w
            y = row * cell_h
        x = min(max(x, 0.0), max(self.canvas_w - w, 0.0))
        y = min(max(y, 0.0), max(self.canvas_h - h, 0.0))
        return x, y

    def step(self, action: int, aspect: float = 1.0) -> Tuple[dict, float, bool, dict]:
        if self.ptr >= len(self.order):
            raise RuntimeError("step() called on a finished episode; call reset().")

        block_idx = self.order[self.ptr]
        w, h = self._block_dims(block_idx, aspect)
        x, y = self._action_to_xy(action, w, h)
        x, y = self._snap_boundary(block_idx, x, y, w, h)   # Phase 6 boundary
        self.positions[block_idx] = torch.tensor([x, y, w, h], device=self.device)
        self.ptr += 1

        done = self.ptr >= len(self.order)
        reward = 0.0  # Phase 0: terminal reward only (shaping is Phase 4 §4)
        info = {"block_idx": block_idx}
        if done and not self.skip_terminal_cost:     # inference discards the reward
            cost = self._terminal_cost()
            reward = -cost
            info["contest_cost"] = cost

        return self._build_state(), reward, done, info

    # ------------------------------------------------------------------ #
    # Reward
    # ------------------------------------------------------------------ #
    def _terminal_cost(self) -> float:
        """Contest cost of the finished (possibly overlapping) placement.
        Phase 7 will legalize before scoring; Phase 0 scores raw."""
        positions = self.positions.detach().clone()
        with torch.no_grad():
            cost = compute_training_loss_differentiable(
                positions,
                self.b2b,
                self.p2b,
                self.pins_pos,
                self.area_target[:self.block_count],
                self.metrics,
            )
        return float(cost)

    # ------------------------------------------------------------------ #
    # State
    # ------------------------------------------------------------------ #
    def _occupancy_grid(self) -> torch.Tensor:
        """Minimal occupancy placeholder for Phase 0. Phase 1 (canvas_raster.py)
        replaces this with a proper multi-channel rasterization."""
        g = torch.zeros((self.grid, self.grid), device=self.device)
        cell_w = self.canvas_w / self.grid
        cell_h = self.canvas_h / self.grid
        for i in range(self.block_count):
            row = self.positions[i]
            if torch.isnan(row[0]):
                continue
            x, y, w, h = (float(v) for v in row)
            c0 = max(int(x / cell_w), 0)
            r0 = max(int(y / cell_h), 0)
            c1 = min(int(math.ceil((x + w) / cell_w)), self.grid)
            r1 = min(int(math.ceil((y + h) / cell_h)), self.grid)
            g[r0:r1, c0:c1] += 1.0
        return g

    def _build_state(self) -> dict:
        current_block = (self.order[self.ptr] if self.ptr < len(self.order)
                         else None)
        return {
            "occupancy": (None if self.skip_occupancy   # unused in inference
                          else self._occupancy_grid()),  # [G, G] placeholder
            "positions": self.positions.clone(),     # [N, 4], NaN = unplaced
            "current_block": current_block,           # block id to place next
            "ptr": self.ptr,
            "block_count": self.block_count,
            "canvas": (self.canvas_w, self.canvas_h),
            # static problem tensors (a GNN encoder will consume these in Phase 2)
            "area_target": self.area_target[:self.block_count],
            "b2b": self.b2b,
            "p2b": self.p2b,
            "pins_pos": self.pins_pos,
            "constraints": self.constraints[:self.block_count],
        }
