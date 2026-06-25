"""
Phase 7 — FloorplanOptimizer wrapper + legalization (VERSION_B_RL_PLACER.md §5 Phase 7).

Produces a GUARANTEED-FEASIBLE placement (overlap-free + exact areas, the only
HARD constraints per ProblemC) via a self-contained shelf legalizer — kept
independent of analytic_legalizer/ per this folder's charter.

Feasibility recipe:
  * preplaced blocks: kept at their spec location/dims (immutable);
  * every other block: shelf-packed into rows in the region ABOVE all preplaced
    blocks, so movable never overlaps preplaced and never overlaps each other;
  * each block keeps an area-exact (w,h): fixed-shape -> given dims; soft -> square.

Soft constraints (preplaced location is honoured -> 0; boundary/MIB/grouping may
incur soft penalty) only affect score, not feasibility. RL/BC model-guided
shape & ordering (better HPWL/area) is the Phase 8 quality lever; this wrapper
first establishes a feasible, scoreable pipeline.

Run:
    cd FloorSet/iccad2026contest
    python3 iccad2026_evaluate.py --evaluate DL_RL/rl_optimizer.py --test-id 0 --verbose
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import torch

# iccad2026contest must be importable for FloorplanOptimizer (matches analytic).
_CONTEST_DIR = Path(__file__).resolve().parent.parent
if str(_CONTEST_DIR) not in sys.path:
    sys.path.insert(0, str(_CONTEST_DIR))
_DL_DIR = Path(__file__).resolve().parent
if str(_DL_DIR) not in sys.path:
    sys.path.insert(0, str(_DL_DIR))

from iccad2026_evaluate import FloorplanOptimizer  # noqa: E402

_DEFAULT_CKPT = _DL_DIR / "checkpoints" / "phase8_bc.pt"


def _block_dims(i: int, area_targets, constraints, target_positions):
    """Area-exact (w,h) for block i: fixed/preplaced -> given dims, soft -> square."""
    if target_positions is not None:
        is_fixed = float(constraints[i, 0]) != 0
        is_preplaced = float(constraints[i, 1]) != 0
        if is_fixed or is_preplaced:
            tw, th = float(target_positions[i, 2]), float(target_positions[i, 3])
            if tw > 0 and th > 0:
                return tw, th
    side = math.sqrt(max(float(area_targets[i]), 1e-6))
    return side, side


def shelf_legalize(block_count, area_targets, constraints, target_positions,
                   b2b=None) -> List[Tuple[float, float, float, float]]:
    """Overlap-free, area-exact placement. See module docstring."""
    dims = [_block_dims(i, area_targets, constraints, target_positions)
            for i in range(block_count)]
    positions: List[Optional[Tuple[float, float, float, float]]] = [None] * block_count

    preplaced = []
    if target_positions is not None:
        preplaced = [i for i in range(block_count) if float(constraints[i, 1]) != 0]
    for i in preplaced:
        x, y = float(target_positions[i, 0]), float(target_positions[i, 1])
        positions[i] = (x, y, dims[i][0], dims[i][1])

    # movable region starts above every preplaced block (no vertical overlap)
    y_start = 0.0
    for i in preplaced:
        y_start = max(y_start, positions[i][1] + positions[i][3])

    movable = [i for i in range(block_count) if i not in set(preplaced)]
    total_area = sum(dims[i][0] * dims[i][1] for i in movable)
    max_w = max((dims[i][0] for i in movable), default=1.0)
    W = max(math.sqrt(max(total_area, 1e-6)) * 1.05, max_w)

    # tallest-first keeps rows compact (best bbox area; HPWL-aware ordering was
    # tried and regressed Total Score on the large, e^n-weighted cases).
    movable.sort(key=lambda i: dims[i][1], reverse=True)
    x, y, row_h = 0.0, y_start, 0.0
    for i in movable:
        w, h = dims[i]
        if x + w > W + 1e-9:          # new row
            y += row_h
            x, row_h = 0.0, 0.0
        positions[i] = (x, y, w, h)
        x += w
        row_h = max(row_h, h)

    return [positions[i] for i in range(block_count)]


def model_guided_legalize(block_count, dims, centers, preplaced_pos, W, y_start
                          ) -> List[Tuple[float, float, float, float]]:
    """Phase 8: position-preserving legalization. Movable blocks are packed into
    rows ordered bottom-to-top by their model-predicted cy, and left-to-right
    within a row by model cx. This reconstructs the model's 2D arrangement as a
    non-overlapping row layout (the model is trained toward compact GT layouts,
    so cy/cx ordering tracks GT structure). Overlap-free + area-exact -> feasible.
    """
    positions: List[Optional[Tuple[float, float, float, float]]] = [None] * block_count
    for i, p in preplaced_pos.items():
        positions[i] = p
    movable = [i for i in range(block_count) if i not in preplaced_pos]
    movable.sort(key=lambda i: (centers[i][1], centers[i][0]))  # cy then cx

    rows, cur, cur_w = [], [], 0.0
    for i in movable:
        w, _h = dims[i]
        if cur and cur_w + w > W + 1e-9:
            rows.append(cur)
            cur, cur_w = [], 0.0
        cur.append(i)
        cur_w += w
    if cur:
        rows.append(cur)

    y = y_start
    for row in rows:
        row.sort(key=lambda i: centers[i][0])      # left-to-right by model cx
        x, row_h = 0.0, 0.0
        for i in row:
            w, h = dims[i]
            positions[i] = (x, y, w, h)
            x += w
            row_h = max(row_h, h)
        y += row_h
    return [positions[i] for i in range(block_count)]


class RLPlacerOptimizer(FloorplanOptimizer):
    """Phase 7/8 wrapper. With a trained checkpoint, runs the model greedily to
    get a per-block 2D arrangement, then position-preservingly legalizes it.
    Without a checkpoint, falls back to the self-contained shelf legalizer."""

    def __init__(self, verbose: bool = False, checkpoint=None):
        super().__init__(verbose)
        self._model = None
        ckpt_path = Path(checkpoint) if checkpoint else _DEFAULT_CKPT
        if ckpt_path.exists():
            try:
                import torch as _t
                from gnn_encoder import GNNEncoder
                from policy_net import PolicyValueNet
                ck = _t.load(ckpt_path, map_location="cpu")
                gnn = GNNEncoder(out_dim=ck["gnn_out"], hidden=ck["hidden"])
                pol = PolicyValueNet(in_channels=4, node_dim=ck["gnn_out"],
                                     hidden=ck["hidden"])
                gnn.load_state_dict(ck["gnn"])
                pol.load_state_dict(ck["policy"])
                gnn.eval()
                pol.eval()
                self._model = (gnn, pol, int(ck["grid"]))
                if verbose:
                    print(f"[RLPlacer] loaded model {ckpt_path} (grid={ck['grid']})")
            except Exception as e:  # pragma: no cover
                if verbose:
                    print(f"[RLPlacer] checkpoint load failed ({e}); shelf fallback")
                self._model = None

    def _model_solve(self, block_count, area_targets, b2b, p2b, pins, constraints,
                     target_positions):
        from placement_env import PlacementEnv
        from canvas_raster import rasterize_env, CH_FEASIBILITY
        import torch as _t
        gnn, policy, grid = self._model
        env = PlacementEnv(grid=grid)
        # metrics only feeds the (ignored-at-inference) terminal reward; dummy is fine
        env.reset(area_targets, b2b, p2b, pins, constraints, _t.ones(8),
                  target_positions=target_positions)
        node_emb, _ = gnn.encode_problem(area_targets, constraints, b2b, block_count)
        done = len(env.order) == 0
        while not done:
            st = env._build_state()
            cur = st["current_block"]
            canvas = rasterize_env(env)
            mask = canvas[CH_FEASIBILITY]
            a, _lp, _v = policy.act(canvas, node_emb[cur], mask, greedy=True)
            _s, _r, done, _i = env.step(a)

        dims = [(_block_dims(i, area_targets, constraints, target_positions))
                for i in range(block_count)]
        centers = {}
        preplaced_pos = {}
        preplaced = set()
        if target_positions is not None:
            preplaced = {i for i in range(block_count)
                         if float(constraints[i, 1]) != 0}
        for i in range(block_count):
            x, y, w, h = (float(t) for t in env.positions[i])
            centers[i] = (x + w / 2, y + h / 2)
            if i in preplaced:
                preplaced_pos[i] = (x, y, w, h)
        y_start = max((preplaced_pos[i][1] + preplaced_pos[i][3]
                       for i in preplaced_pos), default=0.0)
        total_area = sum(dims[i][0] * dims[i][1]
                         for i in range(block_count) if i not in preplaced)
        max_w = max((dims[i][0] for i in range(block_count)), default=1.0)
        W = max(math.sqrt(max(total_area, 1e-6)) * 1.05, max_w)
        return model_guided_legalize(block_count, dims, centers, preplaced_pos,
                                     W, y_start)

    def solve(self, block_count, area_targets, b2b_connectivity,
              p2b_connectivity, pins_pos, constraints, target_positions=None
              ) -> List[Tuple[float, float, float, float]]:
        if self._model is not None:
            try:
                return self._model_solve(
                    block_count, area_targets, b2b_connectivity,
                    p2b_connectivity, pins_pos, constraints, target_positions)
            except Exception:
                pass  # fall back to shelf if the model path errors
        return shelf_legalize(block_count, area_targets, constraints,
                              target_positions, b2b=b2b_connectivity)


# The --evaluate loader matches by name (its issubclass check fails across
# importlib's duplicate FloorplanOptimizer copy), so expose a fallback name.
Optimizer = RLPlacerOptimizer
