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


def _raw_hpwl(positions, b2b, p2b, pins) -> float:
    """Raw half-perimeter wirelength (b2b + p2b) on block centers; 0 if no nets."""
    if b2b is None and p2b is None:
        return 0.0
    total = 0.0
    if b2b is not None and len(b2b) > 0:
        for e in b2b:
            i, j, w = int(e[0]), int(e[1]), float(e[2])
            if i < 0 or j < 0:
                continue
            xi, yi, wi, hi = positions[i]; xj, yj, wj, hj = positions[j]
            total += w * (abs((xi + wi / 2) - (xj + wj / 2)) + abs((yi + hi / 2) - (yj + hj / 2)))
    if p2b is not None and len(p2b) > 0:
        for e in p2b:
            pi, bi, w = int(e[0]), int(e[1]), float(e[2])
            if bi < 0:
                continue
            x, y, w2, h2 = positions[bi]
            total += w * (abs((x + w2 / 2) - float(pins[pi][0])) + abs((y + h2 / 2) - float(pins[pi][1])))
    return total


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
    b2b=None,
    p2b=None,
    pins=None,
) -> Tuple[List[Tuple[float, float, float, float]], float]:
    """Legalize via skyline packing. Tries a ladder of container widths and
    returns (positions, score) for the best (lowest cost-proxy) feasible width.
    Width selection uses area·HPWL·e^(2·boundary): HPWL is the dominant cost term,
    so an area-only proxy mis-picks (chooses too-narrow boxes with worse wirelength)."""
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

    # Boundary-derived aspect prior: a LEFT/RIGHT block must sit on a vertical wall
    # (needs height), a TOP/BOTTOM block on a horizontal wall (needs width), so the
    # intended outline aspect H/W ≈ Σheight(LR) / Σwidth(TB). This correlates ~0.9
    # with the GT aspect — a strong physics-based prior the area·HPWL proxy lacks.
    sh_lr, sw_tb = 1e-6, 1e-6
    for b in blocks:
        bc = b.boundary_code
        if bc & BOUND_LEFT or bc & BOUND_RIGHT:
            sh_lr += b.h
        if bc & BOUND_TOP or bc & BOUND_BOTTOM:
            sw_tb += b.w
    asp_pred = min(max(sh_lr / sw_tb, 0.3), 3.2)
    PRIOR = 0.5  # mild: prediction is informative but noisy — nudge, don't dictate

    # aspect = height/width: <1 = wide/short, >1 = tall/narrow. Include both so a
    # wide container can win when the instance prefers it (the score proxy selects).
    aspects = [analytic_aspect, asp_pred, 0.4, 0.6, 0.8, 1.0, 1.3, 1.6, 2.0, 2.5]
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
        hp = _raw_hpwl(pos, b2b, p2b, pins)
        score = (x2 * y2) * (hp if hp > 1e-9 else 1.0) * math.exp(2.0 * bv / max(n, 1))
        # Soft prior toward the boundary-predicted aspect.
        cand_aspect = y2 / max(x2, 1e-6)
        score *= 1.0 + PRIOR * abs(math.log(cand_aspect / asp_pred))
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

    # Finetune: tuck bbox-defining free blocks into cluster-internal whitespace
    # to shrink the bbox (grouping unaffected — a non-member in a group's gap does
    # not change that group's connectivity).
    best_pos = _finetune_fill_gaps(best_pos, blocks, cluster_groups)
    x2 = max(p[0] + p[2] for p in best_pos)
    y2 = max(p[1] + p[3] for p in best_pos)
    bv = _count_boundary_unmet(best_pos, blocks, x2, y2)
    best_score = (x2 * y2) * math.exp(2.0 * bv / max(n, 1))

    return best_pos, best_score


def _clip_axis(pos, i, delta, axis):
    """Largest move of block i by `delta` along `axis` (0=x,1=y) that creates no
    overlap with any perpendicular-aligned block and keeps it at coord >= 0."""
    if abs(delta) < 1e-9:
        return 0.0
    xi, yi, wi, hi = pos[i]
    lo_i = xi if axis == 0 else yi
    sz_i = wi if axis == 0 else hi
    plo_i = yi if axis == 0 else xi
    psz_i = hi if axis == 0 else wi
    for k in range(len(pos)):
        if k == i:
            continue
        xk, yk, wk, hk = pos[k]
        plo_k = yk if axis == 0 else xk
        psz_k = hk if axis == 0 else wk
        if min(plo_i + psz_i, plo_k + psz_k) - max(plo_i, plo_k) <= 1e-6:
            continue  # no perpendicular overlap → can't collide along this axis
        lo_k = xk if axis == 0 else yk
        sz_k = wk if axis == 0 else hk
        if delta > 0 and lo_k >= lo_i + sz_i - 1e-6:
            delta = min(delta, lo_k - (lo_i + sz_i))
        elif delta < 0 and lo_k + sz_k <= lo_i + 1e-6:
            delta = max(delta, (lo_k + sz_k) - lo_i)
    if delta < 0:
        delta = max(delta, -lo_i)
    return delta


def _wmedian(vals_w):
    """Weighted median of [(value, weight), ...]."""
    if not vals_w:
        return None
    vals_w = sorted(vals_w)
    tot = sum(w for _, w in vals_w)
    acc = 0.0
    for v, w in vals_w:
        acc += w
        if acc >= tot / 2.0:
            return v
    return vals_w[-1][0]


def _detailed_place(positions, blocks, b2b, p2b, pins, n_passes=5):
    """HPWL detailed placement: slide each free block toward the weighted-median
    of its connected neighbours (the HPWL-optimal point), clipped to stay legal.
    Moving toward the median is monotone non-increasing in HPWL → safe & convergent.
    Preplaced and cluster members are not moved."""
    pos = list(positions)
    n = len(pos)
    # Only move interior free blocks: a boundary block dragged toward its neighbours
    # loses its edge → V_rel (e^2·) penalty far outweighs the HPWL gain.
    movable = [i for i, b in enumerate(blocks)
               if not b.is_preplaced and b.cluster_group == 0 and b.boundary_code == 0]
    if not movable:
        return pos
    nb = [[] for _ in range(n)]  # neighbour list: (kind, data, weight)
    if b2b is not None and len(b2b) > 0:
        for e in b2b:
            i, j, w = int(e[0]), int(e[1]), float(e[2])
            if i < 0 or j < 0:
                continue
            nb[i].append(('b', j, w)); nb[j].append(('b', i, w))
    if p2b is not None and len(p2b) > 0:
        for e in p2b:
            pi, bi, w = int(e[0]), int(e[1]), float(e[2])
            if bi < 0:
                continue
            px, py = float(pins[pi][0]), float(pins[pi][1])
            nb[bi].append(('p', (px, py), w))

    for _ in range(n_passes):
        for i in movable:
            if not nb[i]:
                continue
            xi, yi, wi, hi = pos[i]
            cx_i, cy_i = xi + wi / 2.0, yi + hi / 2.0
            xs, ys = [], []
            for (t, d, w) in nb[i]:
                if t == 'b':
                    xk, yk, wk, hk = pos[d]
                    xs.append((xk + wk / 2.0, w)); ys.append((yk + hk / 2.0, w))
                else:
                    xs.append((d[0], w)); ys.append((d[1], w))
            dx = _clip_axis(pos, i, _wmedian(xs) - cx_i, 0)
            pos[i] = (pos[i][0] + dx, pos[i][1], wi, hi)
            dy = _clip_axis(pos, i, _wmedian(ys) - cy_i, 1)
            pos[i] = (pos[i][0], pos[i][1] + dy, wi, hi)
    return pos


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

    # Vectorized occupancy arrays (rebuilt each pass; pos only changes on a move,
    # which breaks the pass).
    X = np.array([p[0] for p in pos]); Y = np.array([p[1] for p in pos])
    XE = np.array([p[0] + p[2] for p in pos]); YE = np.array([p[1] + p[3] for p in pos])

    def overlaps_any(rx, ry, rw, rh, exclude):
        ox = np.minimum(rx + rw, XE) - np.maximum(rx, X)
        oy = np.minimum(ry + rh, YE) - np.maximum(ry, Y)
        hit = (ox > 1e-6) & (oy > 1e-6)
        hit[exclude] = False
        return bool(hit.any())

    for _ in range(6):
        X[:] = [p[0] for p in pos]; Y[:] = [p[1] for p in pos]
        XE[:] = [p[0] + p[2] for p in pos]; YE[:] = [p[1] + p[3] for p in pos]
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
