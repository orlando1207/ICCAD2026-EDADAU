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
import torch

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
    __slots__ = (
        "w", "h", "cx", "cy", "bc", "is_cluster", "key",
        "members", "center_offsets", "net_weight",
    )

    def __init__(self, w, h, cx, cy, bc, is_cluster, key,
                 members, center_offsets):
        self.w = w
        self.h = h
        self.cx = cx
        self.cy = cy
        self.bc = bc
        self.is_cluster = is_cluster
        self.key = key  # block index, or cluster gid
        self.members = members
        self.center_offsets = center_offsets
        self.net_weight = 0.0


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
            members=list(members),
            center_offsets=[
                (dx + blocks[m].w / 2.0, dy + blocks[m].h / 2.0)
                for m, (dx, dy) in zip(sb.members, sb.offsets)
            ],
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
            members=[i],
            center_offsets=[(b.w / 2.0, b.h / 2.0)],
        ))
    return movable, obstacles


def _pack_one_width(
    movable: List[_Unit],
    obstacles: List[Tuple[float, float, float, float, int]],
    W: float,
    lam: float,
    order_mode: str = "analytic",
    net_weight: float = 0.0,
    net_ctx: Optional[Dict] = None,
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
    # TOP-forced last (best-effort top), then a deterministic mode-specific
    # priority.  Trying a few orders is cheap and reduces greedy failures.
    def order_key(u: _Unit):
        bottom = 1 if (u.bc & BOUND_BOTTOM) else 0
        top = 1 if (u.bc & BOUND_TOP) else 0
        if order_mode == "area":
            return (-bottom, top, -u.w * u.h, u.cy, u.cx)
        if order_mode == "net":
            return (-bottom, top, -u.net_weight, u.cy, u.cx)
        if order_mode == "cluster":
            return (-bottom, top, 0 if u.is_cluster else 1, u.cy, u.cx)
        return (-bottom, top, u.cy, u.cx)

    placement: Dict = {}
    placed_centers: Dict[int, Tuple[float, float]] = {}
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
            if net_weight > 0.0 and net_ctx is not None:
                score += net_weight * _incremental_net_cost(
                    u, x, y, placed_centers, net_ctx)
            if score < best_score - 1e-9:
                best_score = score
                best_x = x
                best_y = y
        if best_x is None:
            best_x, best_y = 0.0, _land_y(0.0, u.w, u.h)

        sky.raise_to(best_x, best_x + u.w, best_y + u.h)
        placement[u.key] = (best_x, best_y)
        for m, (ox, oy) in zip(u.members, u.center_offsets):
            placed_centers[m] = (best_x + ox, best_y + oy)

    return placement


def _build_net_context(block_count, cx, cy, b2b_connectivity, p2b_connectivity,
                       pins_pos) -> Dict:
    adj = [[] for _ in range(block_count)]
    pin_adj = [[] for _ in range(block_count)]
    incident = np.ones(block_count, dtype=np.float64) * 1e-6

    if b2b_connectivity is not None:
        b2b_iter = (b2b_connectivity.detach().cpu().numpy()
                    if isinstance(b2b_connectivity, torch.Tensor)
                    else b2b_connectivity)
        for edge in b2b_iter:
            if int(edge[0]) == -1:
                continue
            i, j, w = int(edge[0]), int(edge[1]), float(edge[2])
            if i >= block_count or j >= block_count:
                continue
            adj[i].append((j, w))
            adj[j].append((i, w))
            incident[i] += w
            incident[j] += w

    pins = None
    if pins_pos is not None:
        pins = pins_pos.detach().cpu().numpy() if isinstance(pins_pos, torch.Tensor) else pins_pos

    if p2b_connectivity is not None and pins is not None:
        p2b_iter = (p2b_connectivity.detach().cpu().numpy()
                    if isinstance(p2b_connectivity, torch.Tensor)
                    else p2b_connectivity)
        for edge in p2b_iter:
            if int(edge[0]) == -1:
                continue
            pin, block, w = int(edge[0]), int(edge[1]), float(edge[2])
            if block >= block_count or pin >= len(pins):
                continue
            pin_adj[block].append((float(pins[pin, 0]), float(pins[pin, 1]), w))
            incident[block] += w

    return {
        "adj": adj,
        "pin_adj": pin_adj,
        "incident": incident,
        "analytic": [(float(cx[i]), float(cy[i])) for i in range(block_count)],
    }


def _incremental_net_cost(unit: _Unit, x: float, y: float, placed_centers: Dict,
                          net_ctx: Dict) -> float:
    """Average estimated net distance for placing this unit at (x, y).

    Already placed neighbors use final centers. Unplaced neighbors use analytic
    centers, so the term is available during one-pass skyline placement.
    """
    adj = net_ctx["adj"]
    pin_adj = net_ctx["pin_adj"]
    analytic = net_ctx["analytic"]
    incident = net_ctx["incident"]

    total = 0.0
    weight = 0.0
    candidate_centers = {}
    for m, (ox, oy) in zip(unit.members, unit.center_offsets):
        candidate_centers[m] = (x + ox, y + oy)

    member_set = set(unit.members)
    for m, (mx, my) in candidate_centers.items():
        for nb, w in adj[m]:
            if nb in member_set:
                continue
            nx, ny = placed_centers.get(nb, analytic[nb])
            total += w * (abs(mx - nx) + abs(my - ny))
            weight += w
        for px, py, w in pin_adj[m]:
            total += w * (abs(mx - px) + abs(my - py))
            weight += w

    return total / max(weight, 1e-6)


def _assign_unit_net_weights(units: List[_Unit], net_ctx: Optional[Dict]) -> None:
    if net_ctx is None:
        return
    incident = net_ctx["incident"]
    for u in units:
        u.net_weight = float(sum(incident[m] for m in u.members))


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
    b2b_connectivity=None,
    p2b_connectivity=None,
    pins_pos=None,
    hpwl_weight: float = 0.0,
    aspect_ladder: Optional[List[float]] = None,
    net_weight: float = 0.0,
    order_modes: Optional[List[str]] = None,
    refine_widths: bool = False,
) -> Tuple[List[Tuple[float, float, float, float]], float]:
    """Legalize via skyline packing. Tries a ladder of container widths and
    returns (positions, score) for the best (lowest cost-proxy) feasible width."""
    movable, obstacles = _build_units(blocks, super_blocks, cluster_groups, cx, cy)
    order_modes = order_modes or ["analytic"]
    net_ctx = None
    if net_weight > 0.0 or any(m in ("net",) for m in order_modes):
        net_ctx = _build_net_context(
            len(blocks), cx, cy, b2b_connectivity, p2b_connectivity, pins_pos)
        _assign_unit_net_weights(movable, net_ctx)

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

    aspects = list(aspect_ladder) if aspect_ladder is not None else [
        0.85, 1.0, 1.15, 1.3, 1.45, 1.6, 1.8, 2.0, 2.25, 2.5, 2.8
    ]
    aspects.append(analytic_aspect)
    cand_W = set()
    for asp in aspects:
        asp = max(asp, 1e-3)
        w = math.sqrt(total_area / asp)
        cand_W.add(round(max(w, W_min), 3))
    cand_W.add(round(W_min, 3))
    if refine_widths:
        base = list(cand_W)
        for W in base:
            for mul in (0.94, 0.97, 1.03, 1.06):
                cand_W.add(round(max(W * mul, W_min), 3))

    n = len(blocks)
    best_pos = None
    best_score = float("inf")
    for W in sorted(cand_W):
        for order_mode in order_modes:
            placement = _pack_one_width(
                movable, obstacles, W, lam,
                order_mode=order_mode,
                net_weight=net_weight,
                net_ctx=net_ctx,
            )
            if placement is None:
                continue
            pos = _materialize(placement, blocks, super_blocks, cluster_groups)
            x2 = max(p[0] + p[2] for p in pos)
            y2 = max(p[1] + p[3] for p in pos)
            bv = _count_boundary_unmet(pos, blocks, x2, y2)
            hpwl = _raw_hpwl(pos, b2b_connectivity, p2b_connectivity, pins_pos)
            score = _candidate_score(x2 * y2, bv, n, hpwl, hpwl_weight)
            if score < best_score:
                best_score = score
                best_pos = pos

    if best_pos is None:
        # Fallback: pack at W_min (always feasible width-wise).
        placement = _pack_one_width(
            movable, obstacles, W_min, lam,
            order_mode=order_modes[0],
            net_weight=net_weight,
            net_ctx=net_ctx,
        )
        best_pos = _materialize(placement, blocks, super_blocks, cluster_groups)
        x2 = max(p[0] + p[2] for p in best_pos)
        y2 = max(p[1] + p[3] for p in best_pos)
        best_score = x2 * y2

    # Finetune: tuck bbox-defining free blocks into cluster-internal whitespace
    # to shrink the bbox (grouping unaffected — a non-member in a group's gap does
    # not change that group's connectivity).
    best_pos = _finetune_fill_gaps(best_pos, blocks, cluster_groups)
    x2 = max(p[0] + p[2] for p in best_pos)
    y2 = max(p[1] + p[3] for p in best_pos)
    bv = _count_boundary_unmet(best_pos, blocks, x2, y2)
    hpwl = _raw_hpwl(best_pos, b2b_connectivity, p2b_connectivity, pins_pos)
    best_score = _candidate_score(x2 * y2, bv, n, hpwl, hpwl_weight)

    return best_pos, best_score


def _candidate_score(area: float, boundary_unmet: int, n: int,
                     hpwl: float, hpwl_weight: float) -> float:
    """Selection proxy for candidate widths.

    The official score uses baseline-relative gaps, which the legalizer does not
    know.  Raw HPWL is still useful for comparing candidates from the same case.
    A fractional HPWL exponent keeps area/boundary from being overwhelmed.
    """
    boundary_factor = math.exp(2.0 * boundary_unmet / max(n, 1))
    if hpwl_weight <= 0.0 or hpwl <= 1e-9:
        return area * boundary_factor
    return area * boundary_factor * (hpwl ** hpwl_weight)


def _raw_hpwl(positions, b2b_connectivity, p2b_connectivity, pins_pos) -> float:
    hpwl = 0.0
    if b2b_connectivity is not None:
        if isinstance(b2b_connectivity, torch.Tensor):
            b2b_iter = b2b_connectivity.detach().cpu().numpy()
        else:
            b2b_iter = b2b_connectivity
        for edge in b2b_iter:
            if int(edge[0]) == -1:
                continue
            i, j, w = int(edge[0]), int(edge[1]), float(edge[2])
            if i >= len(positions) or j >= len(positions):
                continue
            xi, yi, wi, hi = positions[i]
            xj, yj, wj, hj = positions[j]
            hpwl += w * (abs((xi + wi / 2.0) - (xj + wj / 2.0))
                         + abs((yi + hi / 2.0) - (yj + hj / 2.0)))
    if p2b_connectivity is not None and pins_pos is not None:
        if isinstance(p2b_connectivity, torch.Tensor):
            p2b_iter = p2b_connectivity.detach().cpu().numpy()
        else:
            p2b_iter = p2b_connectivity
        pins = pins_pos.detach().cpu().numpy() if isinstance(pins_pos, torch.Tensor) else pins_pos
        for edge in p2b_iter:
            if int(edge[0]) == -1:
                continue
            pin, block, w = int(edge[0]), int(edge[1]), float(edge[2])
            if block >= len(positions) or pin >= len(pins):
                continue
            x, y, bw, bh = positions[block]
            hpwl += w * (abs(float(pins[pin, 0]) - (x + bw / 2.0))
                         + abs(float(pins[pin, 1]) - (y + bh / 2.0)))
    return hpwl


def _finetune_fill_gaps(positions, blocks, cluster_groups):
    """Relocate bbox-defining free blocks into cluster-internal gaps when doing so
    strictly shrinks the bounding box. Conservative: only frontier blocks move, and
    only if the bbox area decreases (keeps the main packing's positions/HPWL)."""
    pos = list(positions)
    n = len(pos)
    movable = [i for i, b in enumerate(blocks)
               if not b.is_preplaced and b.cluster_group == 0]
    clbb = []
    for mem in cluster_groups.values():
        xs = min(pos[m][0] for m in mem); ys = min(pos[m][1] for m in mem)
        xe = max(pos[m][0] + pos[m][2] for m in mem)
        ye = max(pos[m][1] + pos[m][3] for m in mem)
        clbb.append((xs, ys, xe, ye))
    if not clbb:
        return pos

    def overlaps_any(rx, ry, rw, rh, exclude):
        for j in range(n):
            if j == exclude:
                continue
            jx, jy, jw, jh = pos[j]
            if (min(rx + rw, jx + jw) - max(rx, jx) > 1e-6 and
                    min(ry + rh, jy + jh) - max(ry, jy) > 1e-6):
                return True
        return False

    for _ in range(6):
        x2 = max(p[0] + p[2] for p in pos)
        y2 = max(p[1] + p[3] for p in pos)
        area0 = x2 * y2
        frontier = [i for i in movable
                    if pos[i][0] + pos[i][2] > x2 - 1e-6 or pos[i][1] + pos[i][3] > y2 - 1e-6]
        # Largest frontier blocks first (biggest bbox-shrink potential).
        frontier.sort(key=lambda i: -pos[i][2] * pos[i][3])
        moved = False
        for i in frontier:
            w, h = pos[i][2], pos[i][3]
            step = max(1.0, min(w, h) * 0.5)
            for (cx0, cy0, cx1, cy1) in clbb:
                if cx1 - cx0 < w - 1e-9 or cy1 - cy0 < h - 1e-9:
                    continue
                gy = cy0
                while gy <= cy1 - h + 1e-9:
                    gx = cx0
                    while gx <= cx1 - w + 1e-9:
                        if not overlaps_any(gx, gy, w, h, i):
                            old = pos[i]
                            pos[i] = (gx, gy, w, h)
                            nx2 = max(p[0] + p[2] for p in pos)
                            ny2 = max(p[1] + p[3] for p in pos)
                            if nx2 * ny2 < area0 - 1e-6:
                                moved = True
                                break
                            pos[i] = old
                        gx += step
                    if moved:
                        break
                    gy += step
                if moved:
                    break
            if moved:
                break
        if not moved:
            break
    return pos
