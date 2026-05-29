"""
Skyline strip-packing legalizer (replaces longest-path / compact).

Deterministic, one-pass constructive packing guided by the analytic positions.
Packs all movable units (free blocks + rigid cluster super-blocks) into a strip
of fixed width W (height grows upward); preplaced blocks are fixed obstacles.

See docs/superpowers/specs/2026-05-29-skyline-legalizer-design.md
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from .constraints import (
    BlockInfo, SuperBlock,
    BOUND_LEFT, BOUND_RIGHT, BOUND_TOP, BOUND_BOTTOM,
)


class Skyline:
    """Top contour of occupied space over [0, W], from y=0 upward.

    Stored as a list of contiguous segments [x_start, x_end, height] that
    always tile [0, W] with no gaps or overlaps.
    """

    def __init__(self, width: float):
        self.W = width
        self.segs: List[List[float]] = [[0.0, width, 0.0]]

    def max_h(self, xl: float, xr: float) -> float:
        """Max height over [xl, xr]."""
        xl = max(0.0, xl)
        xr = min(self.W, xr)
        m = 0.0
        for s, e, h in self.segs:
            if e <= xl + 1e-12 or s >= xr - 1e-12:
                continue
            if h > m:
                m = h
        return m

    def raise_to(self, xl: float, xr: float, top: float) -> None:
        """Set height over [xl, xr] to `top` (splitting segments as needed)."""
        xl = max(0.0, xl)
        xr = min(self.W, xr)
        if xr <= xl + 1e-12:
            return
        new: List[List[float]] = []
        for s, e, h in self.segs:
            if e <= xl + 1e-12 or s >= xr - 1e-12:
                new.append([s, e, h])
                continue
            if s < xl - 1e-12:
                new.append([s, xl, h])
            new.append([max(s, xl), min(e, xr), top])
            if e > xr + 1e-12:
                new.append([xr, e, h])
        new.sort(key=lambda t: t[0])
        merged: List[List[float]] = [new[0]]
        for s, e, h in new[1:]:
            if abs(merged[-1][2] - h) < 1e-9 and abs(merged[-1][1] - s) < 1e-9:
                merged[-1][1] = e
            else:
                merged.append([s, e, h])
        self.segs = merged

    def candidate_xs(self, w: float) -> List[float]:
        """Left-x positions to try: each segment start that leaves room for w,
        plus a flush-right option against any segment's end."""
        cands = []
        for s, e, h in self.segs:
            if s + w <= self.W + 1e-6:
                cands.append(s)
            xr_flush = e - w
            if xr_flush >= -1e-6 and xr_flush + w <= self.W + 1e-6:
                cands.append(max(0.0, xr_flush))
        if not cands:
            cands = [0.0]
        return cands


class _Unit:
    """A thing to place: a free block or a rigid cluster box."""
    __slots__ = ("w", "h", "cx", "cy", "bc", "is_cluster", "key")

    def __init__(self, w, h, cx, cy, bc, is_cluster, key):
        self.w = w
        self.h = h
        self.cx = cx
        self.cy = cy
        self.bc = bc
        self.is_cluster = is_cluster
        self.key = key  # block index, or cluster gid


def _build_units(
    blocks: List[BlockInfo],
    super_blocks: Dict[int, SuperBlock],
    cluster_groups: Dict[int, List[int]],
    cx: np.ndarray,
    cy: np.ndarray,
) -> Tuple[List[_Unit], List[Tuple[float, float, float, float, int]]]:
    """Return (movable_units, preplaced_obstacles).

    preplaced_obstacle = (x, y, w, h, block_idx).
    """
    movable: List[_Unit] = []
    obstacles: List[Tuple[float, float, float, float, int]] = []

    in_cluster = set()
    for gid, members in cluster_groups.items():
        sb = super_blocks[gid]
        for m in members:
            in_cluster.add(m)
        rep = members[0]
        dx0, dy0 = sb.offsets[0]
        sb_ll_x = cx[rep] - blocks[rep].w / 2.0 - dx0
        sb_ll_y = cy[rep] - blocks[rep].h / 2.0 - dy0
        movable.append(_Unit(
            w=sb.w, h=sb.h,
            cx=sb_ll_x + sb.w / 2.0, cy=sb_ll_y + sb.h / 2.0,
            bc=sb.boundary_code, is_cluster=True, key=("c", gid),
        ))

    for i, b in enumerate(blocks):
        if i in in_cluster:
            continue
        if b.is_preplaced:
            obstacles.append((b.fixed_x, b.fixed_y, b.w, b.h, i))
            continue
        movable.append(_Unit(
            w=b.w, h=b.h, cx=float(cx[i]), cy=float(cy[i]),
            bc=b.boundary_code, is_cluster=False, key=("b", i),
        ))
    return movable, obstacles


def _pack_one_width(
    movable: List[_Unit],
    obstacles: List[Tuple[float, float, float, float, int]],
    W: float,
    lam: float,
) -> Optional[Dict]:
    """One deterministic packing pass at width W. Returns a placement dict
    {unit_key: (x, y)} for movable units, or None if infeasible at this W."""
    if any(u.w > W + 1e-6 for u in movable):
        return None
    for (ox, oy, ow, oh, _) in obstacles:
        if ox + ow > W + 1e-6:
            return None

    sky = Skyline(W)

    def _land_y(x: float, w: float, h: float) -> float:
        """Lowest y at which a w×h box at left-x x rests on the skyline without
        intersecting any preplaced obstacle (bump up over obstacles it would hit;
        blocks may still sit below/beside a floating obstacle)."""
        y = sky.max_h(x, x + w)
        for _ in range(len(obstacles) + 1):
            bumped = False
            for (ox, oy, ow, oh, _) in obstacles:
                if ox < x + w - 1e-9 and ox + ow > x + 1e-9:      # x-overlap
                    if y < oy + oh - 1e-9 and y + h > oy + 1e-9:  # y-overlap
                        y = oy + oh
                        bumped = True
            if not bumped:
                break
        return y

    # Placement order: BOTTOM-forced first (so they reach y=0 while empty),
    # then by analytic (cy, cx); TOP-forced last (best-effort top).
    def order_key(u: _Unit):
        bottom = 1 if (u.bc & BOUND_BOTTOM) else 0
        top = 1 if (u.bc & BOUND_TOP) else 0
        return (-bottom, top, u.cy, u.cx)

    placement: Dict = {}
    for u in sorted(movable, key=order_key):
        if u.bc & BOUND_LEFT:
            cands = [0.0]
        elif u.bc & BOUND_RIGHT:
            cands = [max(0.0, W - u.w)]
        else:
            cands = sky.candidate_xs(u.w)

        best_x = None
        best_score = float("inf")
        best_y = 0.0
        for x in cands:
            x = min(x, W - u.w)
            if x < -1e-9:
                x = 0.0
            y = _land_y(x, u.w, u.h)
            score = y + lam * abs((x + u.w / 2.0) - u.cx)
            if score < best_score - 1e-9:
                best_score = score
                best_x = x
                best_y = y
        if best_x is None:
            best_x, best_y = 0.0, _land_y(0.0, u.w, u.h)

        sky.raise_to(best_x, best_x + u.w, best_y + u.h)
        placement[u.key] = (best_x, best_y)

    return placement


def _count_boundary_unmet(positions, blocks, x2, y2, tol=1.0) -> int:
    bv = 0
    for i, b in enumerate(blocks):
        bc = b.boundary_code
        if bc == 0:
            continue
        x, y, w, h = positions[i]
        if (bc & BOUND_LEFT) and abs(x) > tol:
            bv += 1; continue
        if (bc & BOUND_RIGHT) and abs((x + w) - x2) > tol:
            bv += 1; continue
        if (bc & BOUND_BOTTOM) and abs(y) > tol:
            bv += 1; continue
        if (bc & BOUND_TOP) and abs((y + h) - y2) > tol:
            bv += 1; continue
    return bv


def _materialize(
    placement: Dict,
    blocks: List[BlockInfo],
    super_blocks: Dict[int, SuperBlock],
    cluster_groups: Dict[int, List[int]],
) -> List[Tuple[float, float, float, float]]:
    n = len(blocks)
    pos: List[Tuple[float, float, float, float]] = [(0.0, 0.0, 0.0, 0.0)] * n
    cluster_of = {}
    for gid, members in cluster_groups.items():
        for m in members:
            cluster_of[m] = gid
    for i, b in enumerate(blocks):
        if b.is_preplaced:
            pos[i] = (b.fixed_x, b.fixed_y, b.w, b.h)
        elif i in cluster_of:
            gid = cluster_of[i]
            sb = super_blocks[gid]
            X, Y = placement[("c", gid)]
            mi = sb.members.index(i)
            dx, dy = sb.offsets[mi]
            pos[i] = (X + dx, Y + dy, b.w, b.h)
        else:
            X, Y = placement[("b", i)]
            pos[i] = (X, Y, b.w, b.h)
    return pos


def skyline_legalize(
    blocks: List[BlockInfo],
    super_blocks: Dict[int, SuperBlock],
    cluster_groups: Dict[int, List[int]],
    cx: np.ndarray,
    cy: np.ndarray,
    area_targets,
    lam: float = 0.3,
) -> Tuple[List[Tuple[float, float, float, float]], float]:
    """Legalize via skyline packing. Tries a ladder of container widths and
    returns (positions, score) for the best (lowest cost-proxy) feasible width."""
    movable, obstacles = _build_units(blocks, super_blocks, cluster_groups, cx, cy)

    total_area = sum(u.w * u.h for u in movable) + sum(o[2] * o[3] for o in obstacles)
    max_unit_w = max((u.w for u in movable), default=1.0)
    max_pre_r = max((o[0] + o[2] for o in obstacles), default=0.0)
    W_min = max(max_unit_w, max_pre_r, 1e-6)

    # Analytic-bbox aspect as a hint.
    if movable:
        ax_min = min(u.cx - u.w / 2.0 for u in movable)
        ax_max = max(u.cx + u.w / 2.0 for u in movable)
        ay_min = min(u.cy - u.h / 2.0 for u in movable)
        ay_max = max(u.cy + u.h / 2.0 for u in movable)
        aw = max(ax_max - ax_min, 1e-6)
        ah = max(ay_max - ay_min, 1e-6)
        analytic_aspect = ah / aw  # height/width
    else:
        analytic_aspect = 1.0

    aspects = [analytic_aspect, 1.0, 1.3, 1.6, 2.0, 2.5]
    cand_W = set()
    for asp in aspects:
        asp = max(asp, 1e-3)
        w = math.sqrt(total_area / asp)
        cand_W.add(round(max(w, W_min), 3))
    cand_W.add(round(W_min, 3))

    n = len(blocks)
    best_pos = None
    best_score = float("inf")
    for W in sorted(cand_W):
        placement = _pack_one_width(movable, obstacles, W, lam)
        if placement is None:
            continue
        pos = _materialize(placement, blocks, super_blocks, cluster_groups)
        x2 = max(p[0] + p[2] for p in pos)
        y2 = max(p[1] + p[3] for p in pos)
        bv = _count_boundary_unmet(pos, blocks, x2, y2)
        score = (x2 * y2) * math.exp(2.0 * bv / max(n, 1))
        if score < best_score:
            best_score = score
            best_pos = pos

    if best_pos is None:
        # Fallback: pack at W_min (always feasible width-wise).
        placement = _pack_one_width(movable, obstacles, W_min, lam)
        best_pos = _materialize(placement, blocks, super_blocks, cluster_groups)
        x2 = max(p[0] + p[2] for p in best_pos)
        y2 = max(p[1] + p[3] for p in best_pos)
        best_score = x2 * y2

    return best_pos, best_score
