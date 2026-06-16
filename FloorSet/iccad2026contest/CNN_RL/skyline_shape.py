"""
Phase 15 / B1 — contour-aware block shaping inside the skyline packer.

The PoC (`poc_slack_shaping.py`) showed that reshaping soft blocks is the
dominant lever for `Area_gap` (ceiling ≈ -0.25), but that it cannot be captured
as a post-pass on free blocks only (the constrained majority pins the envelope).
The lever has to live *inside* the packer, where every block's shape can adapt
as it lands on the contour.

This module forks ONLY the per-width packing routine, adding a per-block shape
choice: when a soft, non-boundary, non-cluster block lands on the skyline, it
tries a ladder of aspect ratios (area-constant) and keeps the (shape, x) that
minimises landing height + center pull — i.e. the shape that best fills the
current contour notch. It reuses `analytic_legalizer`'s pure helpers (`Skyline`,
`_build_units`, `_materialize`, …) and does NOT modify `analytic_legalizer/`.

`skyline_legalize_shaped` mirrors the stock `skyline_legalize` width-ladder and
keeps its two existing shape modes (square, row-assign) as candidates, then adds
the contour-shaped mode as a THIRD candidate scored by the same proxy. So the
result is never worse than stock by construction (shaped only wins when it
scores lower); it is a clean A/B against `skyline_legalize`.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from analytic_legalizer.skyline_legalizer import (  # pure helpers, not modified
    Skyline, _build_units, _materialize, _raw_hpwl, _count_boundary_unmet,
    _row_assign_reshape, _pack_one_width, AR_MAX,
)
from analytic_legalizer.constraints import (
    BlockInfo, SuperBlock,
    BOUND_LEFT, BOUND_RIGHT, BOUND_TOP, BOUND_BOTTOM,
)

# Aspect ladder (h/w) each soft block is allowed to try when fitting the contour.
# Spans the GT range without degenerate slivers (clamped by AR_MAX = 3.0).
SHAPE_ASPECTS = [0.34, 0.5, 0.7, 1.0, 1.43, 2.0, 2.9]


def _pack_one_width_shaped(movable, obstacles, W: float, lam: float) -> Optional[Dict]:
    """Like `_pack_one_width`, but a soft non-boundary non-cluster block chooses
    the aspect (area-constant) that best fits the contour at placement time:
    minimise `landing_y + lam·|center − analytic_cx|` jointly over shape and x."""
    for (ox, oy, ow, oh, _) in obstacles:
        if ox + ow > W + 1e-6:
            return None

    sky = Skyline(W)

    def _land_y(x: float, w: float, h: float) -> float:
        y = sky.max_h(x, x + w)
        for _ in range(len(obstacles) + 1):
            bumped = False
            for (ox, oy, ow, oh, _) in obstacles:
                if ox < x + w - 1e-9 and ox + ow > x + 1e-9:
                    if y < oy + oh - 1e-9 and y + h > oy + 1e-9:
                        y = oy + oh
                        bumped = True
            if not bumped:
                break
        return y

    def order_key(u):
        bottom = 1 if (u.bc & BOUND_BOTTOM) else 0
        top = 1 if (u.bc & BOUND_TOP) else 0
        return (-bottom, top, u.cy, u.cx)

    placement: Dict = {}
    for u in sorted(movable, key=order_key):
        reshapeable = u.soft and (not u.is_cluster) and (u.bc == 0)
        if reshapeable:
            shapes = []
            for a in SHAPE_ASPECTS:                       # a = h / w
                w = math.sqrt(u.area / a)
                h = math.sqrt(u.area * a)
                if w <= W + 1e-6:
                    shapes.append((w, h))
            if not shapes:                                # area too big to be narrow
                shapes = [(W, u.area / W)]
        else:
            if u.w > W + 1e-6:
                return None
            shapes = [(u.w, u.h)]

        best = None                                       # (score, x, y, w, h)
        for (w, h) in shapes:
            if u.bc & BOUND_LEFT:
                cands = [0.0]
            elif u.bc & BOUND_RIGHT:
                cands = [max(0.0, W - w)]
            else:
                cands = sky.candidate_xs(w)
            for x in cands:
                x = min(x, W - w)
                if x < -1e-9:
                    x = 0.0
                y = _land_y(x, w, h)
                score = y + lam * abs((x + w / 2.0) - u.cx)
                if best is None or score < best[0] - 1e-9:
                    best = (score, x, y, w, h)
        if best is None:
            w, h = shapes[0]
            best = (0.0, 0.0, _land_y(0.0, w, h), w, h)

        _, bx, by, bw, bh = best
        sky.raise_to(bx, bx + bw, by + bh)
        placement[u.key] = (bx, by, bw, bh)

    return placement


def skyline_legalize_shaped(
    blocks: List[BlockInfo],
    super_blocks: Dict[int, SuperBlock],
    cluster_groups: Dict[int, List[int]],
    cx, cy, area_targets,
    lam: float = 0.3, b2b=None, p2b=None, pins=None,
) -> Tuple[List[Tuple[float, float, float, float]], float]:
    """Stock `skyline_legalize` width-ladder + square/row modes, PLUS a contour-
    shaped mode. Returns (positions, score) for the best-scoring candidate."""
    movable, obstacles = _build_units(blocks, super_blocks, cluster_groups, cx, cy)
    total_area = sum(u.area for u in movable) + sum(o[2] * o[3] for o in obstacles)
    max_pre_r = max((o[0] + o[2] for o in obstacles), default=0.0)
    n = len(blocks)

    # Boundary-derived aspect prior (identical to stock).
    sh_lr, sw_tb = 1e-6, 1e-6
    for b in blocks:
        if b.boundary_code & BOUND_LEFT or b.boundary_code & BOUND_RIGHT:
            sh_lr += b.h
        if b.boundary_code & BOUND_TOP or b.boundary_code & BOUND_BOTTOM:
            sw_tb += b.w
    asp_pred = min(max(sh_lr / sw_tb, 0.3), 3.2)
    PRIOR = 0.5

    if movable:
        ax_min = min(u.cx - u.w / 2.0 for u in movable)
        ax_max = max(u.cx + u.w / 2.0 for u in movable)
        ay_min = min(u.cy - u.h / 2.0 for u in movable)
        ay_max = max(u.cy + u.h / 2.0 for u in movable)
        analytic_aspect = max(ay_max - ay_min, 1e-6) / max(ax_max - ax_min, 1e-6)
    else:
        analytic_aspect = 1.0

    aspects = [analytic_aspect, asp_pred, 0.4, 0.6, 0.8, 1.0, 1.3, 1.6, 2.0, 2.5]
    sq_side = math.sqrt(sum(u.area for u in movable if u.soft and u.bc == 0) or total_area)
    W_min_sq = max(sq_side, max_pre_r, 1e-6)
    cand_W_base: List[float] = []
    for asp in aspects:
        cand_W_base.append(round(max(math.sqrt(total_area / max(asp, 1e-3)), W_min_sq), 3))
    cand_W_base.append(round(W_min_sq, 3))
    cand_W_base = sorted(set(cand_W_base))

    best_pos = None
    best_score = float("inf")

    def _restore_squares():
        for u in movable:
            if u.soft and u.bc == 0:
                s = math.sqrt(u.area); u.w = u.h = s

    def _area_proxy(pos):
        x2 = max(p[0] + p[2] for p in pos)
        y2 = max(p[1] + p[3] for p in pos)
        bv = _count_boundary_unmet(pos, blocks, x2, y2)
        cand_aspect = y2 / max(x2, 1e-6)
        return (x2 * y2) * math.exp(2.0 * bv / max(n, 1)) * (
            1.0 + PRIOR * abs(math.log(cand_aspect / asp_pred)))

    # --- Pass 1: square shapes (identical to stock) ---
    _restore_squares()
    cache1: Dict = {}
    sq_area: List[Tuple[float, float, object]] = []
    for W in cand_W_base:
        if any(u.w > W + 1e-6 for u in movable):
            pos = None
        else:
            pl = _pack_one_width(movable, obstacles, W, lam)
            pos = None if pl is None else _materialize(pl, blocks, super_blocks, cluster_groups)
        a_s = _area_proxy(pos) if pos is not None else float("inf")
        sq_area.append((W, a_s, pos))

    for W, a_s, pos in sorted(sq_area, key=lambda t: t[1])[:2]:
        if pos is None:
            continue
        hp = _raw_hpwl(pos, b2b, p2b, pins)
        score = a_s * (hp if hp > 1e-9 else 1.0)
        if score < best_score:
            best_score = score; best_pos = pos

    top_Ws = [W for W, _, _ in sorted(sq_area, key=lambda t: t[1])[:4]]

    # --- Pass 2: row-assigned shapes (identical to stock) ---
    for W in top_Ws:
        _row_assign_reshape(movable, W)
        pl = _pack_one_width(movable, obstacles, W, lam)
        _restore_squares()
        if pl is None:
            continue
        pos = _materialize(pl, blocks, super_blocks, cluster_groups)
        a_s = _area_proxy(pos)
        hp = _raw_hpwl(pos, b2b, p2b, pins)
        score = a_s * (hp if hp > 1e-9 else 1.0)
        if score < best_score:
            best_score = score; best_pos = pos

    # --- Pass 3: contour-aware shaped (NEW) ---
    for W in top_Ws:
        pl = _pack_one_width_shaped(movable, obstacles, W, lam)
        if pl is None:
            continue
        pos = _materialize(pl, blocks, super_blocks, cluster_groups)
        a_s = _area_proxy(pos)
        hp = _raw_hpwl(pos, b2b, p2b, pins)
        score = a_s * (hp if hp > 1e-9 else 1.0)
        if score < best_score:
            best_score = score; best_pos = pos

    _restore_squares()

    if best_pos is None:                                  # square-shape fallback
        placement = _pack_one_width(movable, obstacles, W_min_sq, lam)
        best_pos = _materialize(placement, blocks, super_blocks, cluster_groups)
        x2 = max(p[0] + p[2] for p in best_pos)
        y2 = max(p[1] + p[3] for p in best_pos)
        best_score = x2 * y2

    return best_pos, best_score
