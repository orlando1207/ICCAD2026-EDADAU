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
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from legalizer.skyline_legalizer import (  # pure helpers, not modified
    Skyline, _build_units, _materialize, _raw_hpwl, _count_boundary_unmet,
    _row_assign_reshape, _pack_one_width, AR_MAX,
)
from legalizer.constraints import (
    BlockInfo, SuperBlock,
    BOUND_LEFT, BOUND_RIGHT, BOUND_TOP, BOUND_BOTTOM,
)

# Aspect ladder (h/w) each soft block is allowed to try when fitting the contour.
# Spans the GT range without degenerate slivers (clamped by AR_MAX = 3.0).
SHAPE_ASPECTS = [0.34, 0.5, 0.7, 1.0, 1.43, 2.0, 2.9]

# Height penalty in the per-block landing objective. With a free shape, scoring on
# landing-y alone (β=0) rewards tall narrow shapes that drop into contour notches but
# spike the skyline; adding β·h internalises the block's own contribution to the
# envelope height. β=0 reproduces the original behaviour. A 100-case sweep (B1+) put
# the optimum at β=0.3 (Total Score 1.8517→1.8088); the curve is non-monotonic
# because the proxy gate accepts/rejects the shaped layout discretely per case.
# Override via SHAPE_HEIGHT_BETA to re-sweep.
SHAPE_HEIGHT_BETA = float(os.environ.get("SHAPE_HEIGHT_BETA", "0.3"))

# Experiment: cy-pull in the landing objective (CY_PULL=μ, default 0 = off). The skyline
# packer is gravity-based — y is forced to the lowest skyline support — so this term can
# only bias the x-choice toward positions whose landing-y is near the rollout's analytic
# cy; it cannot lift a block up (that would leave a gap the skyline model can't represent).
# Tests whether nudging blocks toward their rollout vertical order helps the final cost.
CY_PULL = float(os.environ.get("CY_PULL", "0.0"))
# Horizontal-order anchor weight in the landing objective (lam·|x_center − cx|).
# Default 0.3 = weak tiebreak (landing_y dominates), so the packer freely re-orders
# blocks left-to-right vs the model's centers. Raise via SKY_LAM to test whether
# preserving the rollout's relative (left/right) order — which often matches GT —
# beats the packer's density-first re-ordering.
SKY_LAM = float(os.environ.get("SKY_LAM", "0.3"))
# Height-invariant queue order (Method C). The skyline processes blocks by ascending
# vertical key; default uses the CENTER cy = y + h/2, which for a bottom-touching block
# (y≈0) collapses to h/2 — so the sort order of blocks on the floor is decided by their
# HEIGHT, not their intended position (a tall floor block sorts after a short one and is
# placed later/unfairly). ORDER_YLL=1 sorts by the lower-left edge (cy - h/2 = y instead),
# so all floor blocks tie at 0 and fall back to left→right cx order.
ORDER_YLL = os.environ.get("ORDER_YLL", "0") == "1"
# Method B — relative-order-preserving x bound. The greedy packer drops each block at
# the lowest-landing x, which lets a block grab a far-left notch and cross to the wrong
# side of a model-left neighbour (e.g. 88 ending up left of 61). Unlike SKY_LAM (which
# penalises ABSOLUTE cx deviation and fights density everywhere), this enforces only
# RELATIVE order: a block's left edge may not go left of any already-placed block the
# model put to its left in an overlapping vertical band. Density (landing_y) is otherwise
# untouched, so it only removes left/right CROSSINGS, not tight packing. ORDER_PRESERVE_X=1.
ORDER_PRESERVE_X = os.environ.get("ORDER_PRESERVE_X", "0") == "1"
# Cluster/MIB-exempt Method B (default on when B is on): the order bound is applied
# only among FREE, non-grouped blocks. Grouped members (cohesion cluster / MIB) are
# neither constrained nor used as a constraint source, so the order constraint can't
# pull a member out of its cohesion blob (case99: B shrank bbox but doubled grouping
# violations — this keeps the area win without the violation hit).
ORDER_EXEMPT_GROUP = os.environ.get("ORDER_EXEMPT_GROUP", "1") == "1"
# Queue ordering mode (2D-aware sweep experiment). "cy" = vertical (default);
# "diag" = cx+cy diagonal sweep (lower-left → upper-right); "cx" = pure left→right.
ORDER_MODE = os.environ.get("ORDER_MODE", "cy")

# Add flush-beside-obstacle x candidates (default ON) so a unit — especially a
# rigid cluster super-block — can sit next to a preplaced block instead of being
# bumped *over* it. The skyline only tracks the top contour, so without these the
# packer never sees the gap beside a floating preplaced obstacle. All 100 cases
# have preplaced+cluster; this drops Total Score 1.8032→1.7608 at no runtime cost.
OBS_XCAND = os.environ.get("OBS_XCAND", "1") == "1"

# Ceiling-lift (TOP two-pass): after packing, lift each TOP-forced block straight up
# to touch the pack top whenever the strip above it is empty. MEASURED NET LOSS, so
# default OFF: the naive "fix all top boundary" headroom (~0.09 Total) assumed free
# fixes, but the violations are geometrically locked — only 4/218 had an empty strip
# (inter-top stacking blocks the rest), and those few lifts pulled blocks off their
# nets, raising HPWL/area more than the V drop saved (Total 1.6039→1.6011, avg cost
# 1.6847→1.6897 worse). Kept behind TOP_LIFT=1 for reference. The real top lever would
# need a placement-level rewrite (reserve top blocks, tile along a common ceiling) and
# would still pay the same HPWL relocation cost — likely the boundary lever is much
# smaller than the static estimate implied.
TOP_LIFT = os.environ.get("TOP_LIFT", "0") == "1"


# Cluster super-block aspect ladder: row-width multipliers on sqrt(cluster_area).
# Re-running the shelf-pack at each row width yields a differently-shaped (but
# area-/connectivity-preserving) super-block; the packer then picks the aspect that
# best fits the contour, exactly as a free soft block does (Item B-cluster).
CLUSTER_ROW_FACTORS = [0.6, 0.8, 1.0, 1.3, 1.7, 2.2]

# Option C — cohesion weight. When cluster members are placed as individual
# reshapeable blocks (freed from the rigid super-block), this term rewards landing
# adjacent to an already-placed member of the same group, so the cluster grows as a
# connected blob that still conforms to the contour. Too small → it scatters
# (grouping violations); too large → it ignores the contour and clumps. A 100-case
# sweep put the optimum at 8.0 (Total Score 1.7471→1.6039); case99 plateaus from 8.
COHESION_W = float(os.environ.get("COHESION_W", "8.0"))


def _make_hpwl_fn(b2b, p2b, pins):
    """Precompute the net edge index arrays ONCE and return a vectorised
    `hpwl(pos)`. Reproduces `analytic_legalizer._raw_hpwl` exactly, but the
    width-ladder scores ~60 candidate layouts per legalize and the stock function
    re-iterates the edge *tensors* (per-element `int()` casts) every call — that
    pure-Python loop is the single biggest legalize hotspot. Here the edges are
    converted to numpy once; each call is a vectorised gather + abs-diff."""
    def _np(x):
        return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)

    bi = bj = bw = None
    if b2b is not None and len(b2b) > 0:
        a = _np(b2b); m = (a[:, 0] >= 0) & (a[:, 1] >= 0); a = a[m]
        bi = a[:, 0].astype(np.int64); bj = a[:, 1].astype(np.int64)
        bw = a[:, 2].astype(np.float64)

    pblk = pw = pinx = piny = None
    if p2b is not None and len(p2b) > 0 and pins is not None:
        a = _np(p2b); m = a[:, 1] >= 0; a = a[m]
        ppin = a[:, 0].astype(np.int64); pblk = a[:, 1].astype(np.int64)
        pw = a[:, 2].astype(np.float64)
        pa = _np(pins); pinx = pa[ppin, 0].astype(np.float64)
        piny = pa[ppin, 1].astype(np.float64)

    def hpwl(pos) -> float:
        p = np.asarray(pos, dtype=np.float64)        # [N,4]
        cx = p[:, 0] + p[:, 2] / 2.0
        cy = p[:, 1] + p[:, 3] / 2.0
        total = 0.0
        if bi is not None and len(bi):
            total += float((bw * (np.abs(cx[bi] - cx[bj])
                                  + np.abs(cy[bi] - cy[bj]))).sum())
        if pblk is not None and len(pblk):
            total += float((pw * (np.abs(cx[pblk] - pinx)
                                  + np.abs(cy[pblk] - piny))).sum())
        return total

    return hpwl


def _shelf_pack_rw(widths: List[float], heights: List[float],
                   row_width: float) -> Tuple[List[Tuple[float, float]], float, float]:
    """Shelf-pack members (decreasing height, fill rows L→R) at a GIVEN row width.
    Mirrors `analytic_legalizer.constraints._shelf_pack` but exposes `row_width` so
    the same members can be re-packed into different super-block aspects. Member
    dims are untouched (area + connectivity preserved); only the arrangement varies.
    Offsets are returned in members-list order (matching SuperBlock.offsets)."""
    n = len(widths)
    order = sorted(range(n), key=lambda i: -heights[i])
    rows: List[List[int]] = [[]]
    row_x: List[float] = [0.0]
    for idx in order:
        w = widths[idx]
        if row_x[-1] + w > row_width and row_x[-1] > 0:
            rows.append([]); row_x.append(0.0)
        rows[-1].append(idx); row_x[-1] += w
    offsets_by_orig: Dict[int, Tuple[float, float]] = {}
    y = 0.0
    for row in rows:
        row_h = max(heights[i] for i in row)
        x = 0.0
        for i in row:
            offsets_by_orig[i] = (x, y); x += widths[i]
        y += row_h
    total_w = max(offsets_by_orig[i][0] + widths[i] for i in range(n))
    total_h = y
    offsets = [offsets_by_orig[i] for i in range(n)]
    return offsets, total_w, total_h


def _pack_one_width_shaped(movable, obstacles, W: float, lam: float,
                           blocks=None, mib_shape: bool = False,
                           super_blocks=None, cluster_groups=None,
                           cluster_shape: bool = False,
                           cohesion: bool = False) -> Optional[Dict]:
    """Like `_pack_one_width`, but a soft block chooses the aspect (area-constant)
    that best fits the contour at placement time: minimise
    `landing_y + lam·|center − analytic_cx|` jointly over shape and x.

    B-MIB: an MIB-group member also reshapes, but COUPLED — the first member of a
    group searches the aspect ladder and commits its aspect; the rest reuse it, so
    all members keep an identical shape (each computed from its own area, which is
    equal within an MIB group) and the MIB soft-constraint stays satisfied."""
    for (ox, oy, ow, oh, _) in obstacles:
        if ox + ow > W + 1e-6:
            return None

    # An MIB group is reshapeable only if EVERY member is a free individual block
    # (not clustered/fixed/preplaced). Otherwise a rigid member keeps its square
    # shape while free ones reshape → the group's shapes diverge → MIB violation.
    eligible_mib = set()
    if blocks is not None and mib_shape:
        gids = {b.mib_group for b in blocks if b.mib_group != 0}
        for g in gids:
            members = [b for b in blocks if b.mib_group == g]
            if all(b.cluster_group == 0 and not b.is_fixed_shape
                   and not b.is_preplaced for b in members):
                eligible_mib.add(g)

    def _mib_of(u):
        if blocks is not None and isinstance(u.key, tuple) and u.key[0] == "b":
            g = blocks[u.key[1]].mib_group
            if g in eligible_mib:
                return g
        return 0

    do_cluster = (cluster_shape and blocks is not None
                  and super_blocks is not None and cluster_groups is not None)

    def _cluster_eligible(cgid) -> bool:
        # Only plain shelf-packed clusters are safely reshapeable here. A boundary
        # or preplaced member would route `prepack_clusters` through a different
        # packer whose arrangement must be preserved, so skip those groups.
        members = cluster_groups.get(cgid, [])
        return bool(members) and all(
            blocks[m].boundary_code == 0 and not blocks[m].is_preplaced
            for m in members)

    def _cluster_shapes(cgid, Wlim: float):
        members = cluster_groups[cgid]
        ws = [blocks[m].w for m in members]
        hs = [blocks[m].h for m in members]
        area = sum(a * b for a, b in zip(ws, hs))
        base = math.sqrt(area) if area > 0 else max(ws + [1e-6])
        out = []
        seen = set()
        for f in CLUSTER_ROW_FACTORS:
            rw = max(base * f, max(ws))
            offs, cw, ch = _shelf_pack_rw(ws, hs, rw)
            if cw <= Wlim + 1e-6:
                k = (round(cw, 2), round(ch, 2))
                if k not in seen:
                    seen.add(k)
                    out.append((cw, ch, ("cl", offs)))
        return out

    group_aspect: Dict[int, float] = {}                # mib gid -> committed h/w
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

    # Option C cohesion: when packing freed cluster members, group membership comes
    # from the (intact) block attribute, and same-group members are placed
    # consecutively (anchored at the group's lowest cy) so each can abut the ones
    # already down.
    def _coh_grp(u):
        if cohesion and blocks is not None and isinstance(u.key, tuple) and u.key[0] == "b":
            return blocks[u.key[1]].cluster_group
        return 0

    def _ykey(u):                                  # primary queue-sort key
        if ORDER_MODE == "diag":
            return u.cx + u.cy                     # 2D diagonal sweep
        if ORDER_MODE == "cx":
            return u.cx                            # left → right
        return (u.cy - u.h / 2.0) if ORDER_YLL else u.cy   # vertical (Method C)

    grp_anchor: Dict[int, float] = {}
    if cohesion:
        for u in movable:
            g = _coh_grp(u)
            if g:
                grp_anchor[g] = min(grp_anchor.get(g, _ykey(u)), _ykey(u))

    def order_key(u):
        bottom = 1 if (u.bc & BOUND_BOTTOM) else 0
        top = 1 if (u.bc & BOUND_TOP) else 0
        g = _coh_grp(u)
        yk = _ykey(u)
        anchor = grp_anchor.get(g, yk) if g else yk
        return (-bottom, top, anchor, g, yk, u.cx)

    placed_by_group: Dict[int, List[Tuple[float, float, float, float]]] = {}

    def _cohesion_gap(x, y, w, h, g):
        rects = placed_by_group.get(g)
        if not rects:
            return 0.0
        best = float("inf")
        for (mx, my, mw, mh) in rects:
            gx = max(mx - (x + w), x - (mx + mw), 0.0)
            gy = max(my - (y + h), y - (my + mh), 0.0)
            best = min(best, gx + gy)
        return best

    placed_units: List = []                            # (unit, left_x) for Method B

    def _grouped(u):
        """Cohesion-cluster or MIB member (an individual 'b' unit with a group).
        Rigid cluster super-blocks (is_cluster) are NOT grouped here — they move as
        one piece, so order-constraining them can't break internal cohesion."""
        if getattr(u, "is_cluster", False):
            return False
        if blocks is not None and isinstance(u.key, tuple) and u.key[0] == "b":
            b = blocks[u.key[1]]
            return b.cluster_group != 0 or b.mib_group != 0
        return False

    def _order_xlb(u):
        """Method B: lowest legal left-x for u that preserves the model's relative
        left/right order. u may not start left of any already-placed v that the model
        put to u's left (v.cx < u.cx) within an overlapping vertical band."""
        if not ORDER_PRESERVE_X or (u.bc & (BOUND_LEFT | BOUND_RIGHT)):
            return 0.0
        if ORDER_EXEMPT_GROUP and _grouped(u):         # don't pull grouped members
            return 0.0
        # All comparisons in LOWER-LEFT corner coords (the model's raw anchor), not the
        # width/height-contaminated center: lx = cx - w/2, ly = cy - h/2.
        u_lx = u.cx - u.w / 2.0
        ut, ub = u.cy + u.h / 2.0, u.cy - u.h / 2.0    # u's y-extent [ly, ly+h]
        lb = 0.0
        for (v, vbx) in placed_units:
            if ORDER_EXEMPT_GROUP and _grouped(v):     # grouped blocks aren't sources
                continue
            if (v.cx - v.w / 2.0) < u_lx - 1e-9:       # model: v's LL corner left of u's
                vt, vb = v.cy + v.h / 2.0, v.cy - v.h / 2.0
                if vb < ut - 1e-9 and ub < vt - 1e-9:  # y-extents overlap (same band)
                    if vbx > lb:
                        lb = vbx
        return lb

    placement: Dict = {}
    for u in sorted(movable, key=order_key):
        gid = _mib_of(u)
        x_lb = _order_xlb(u)                            # Method B left bound (0 if off)
        reshapeable = u.soft and (not u.is_cluster) and (u.bc == 0)
        if gid != 0:                                      # MIB member: coupled reshape
            aspects = ([group_aspect[gid]] if gid in group_aspect else SHAPE_ASPECTS)
            shapes = []
            for a in aspects:                             # a = h / w, from this member's area
                w = math.sqrt(u.area / a)
                h = math.sqrt(u.area * a)
                if w <= W + 1e-6:
                    shapes.append((w, h, a))
            if not shapes:
                shapes = [(W, u.area / W, (u.area / W) / W)]
        elif reshapeable:
            shapes = []
            for a in SHAPE_ASPECTS:                       # a = h / w
                w = math.sqrt(u.area / a)
                h = math.sqrt(u.area * a)
                if w <= W + 1e-6:
                    shapes.append((w, h, a))
            if not shapes:                                # area too big to be narrow
                shapes = [(W, u.area / W, None)]
        elif u.is_cluster and do_cluster and _cluster_eligible(u.key[1]):
            shapes = _cluster_shapes(u.key[1], W)         # B-cluster: re-pack candidates
            if not shapes:
                if u.w > W + 1e-6:
                    return None
                shapes = [(u.w, u.h, None)]
        else:
            if u.w > W + 1e-6:
                return None
            shapes = [(u.w, u.h, None)]

        best = None                                       # (score, x, y, w, h, aspect)
        for (w, h, a) in shapes:
            if u.bc & BOUND_LEFT:
                cands = [0.0]
            elif u.bc & BOUND_RIGHT:
                cands = [max(0.0, W - w)]
            else:
                cands = sky.candidate_xs(w)
                if OBS_XCAND and obstacles:
                    # Obstacle-aware x: skyline candidate_xs only knows the top
                    # contour, so a rigid super-block wider than the gap beside a
                    # preplaced obstacle gets bumped *over* it. Offer the flush-left
                    # and flush-right-of-obstacle x's so it can slot beside instead.
                    for (ox, oy, ow, oh, _) in obstacles:
                        for xc in (ox - w, ox + ow):
                            if -1e-6 <= xc <= W - w + 1e-6:
                                cands.append(min(max(xc, 0.0), W - w))
                if ORDER_PRESERVE_X and x_lb > 1e-9:    # Method B: drop crossing x's
                    cands = [x for x in cands if x >= x_lb - 1e-6]
                    if not cands:
                        cands = [min(max(x_lb, 0.0), W - w)]
            for x in cands:
                x = min(x, W - w)
                if x < -1e-9:
                    x = 0.0
                y = _land_y(x, w, h)
                score = y + SHAPE_HEIGHT_BETA * h + lam * abs((x + w / 2.0) - u.cx)
                if CY_PULL > 0.0:                       # bias toward rollout vertical order
                    score += CY_PULL * abs((y + h / 2.0) - u.cy)
                if cohesion:
                    cg_id = _coh_grp(u)
                    if cg_id:
                        score += COHESION_W * _cohesion_gap(x, y, w, h, cg_id)
                if best is None or score < best[0] - 1e-9:
                    best = (score, x, y, w, h, a)
        if best is None:
            w, h, a = shapes[0]
            best = (0.0, 0.0, _land_y(0.0, w, h), w, h, a)

        _, bx, by, bw, bh, ba = best
        if isinstance(ba, tuple) and len(ba) == 2 and ba[0] == "cl":
            sb = super_blocks[u.key[1]]                   # commit reshaped cluster
            sb.offsets = ba[1]; sb.w = bw; sb.h = bh      # _materialize reads these
        elif gid != 0 and gid not in group_aspect and ba is not None:
            group_aspect[gid] = ba                        # commit shared aspect
        sky.raise_to(bx, bx + bw, by + bh)
        placement[u.key] = (bx, by, bw, bh)
        if ORDER_PRESERVE_X:
            placed_units.append((u, bx))               # Method B: record left edge
        if cohesion:
            cg_id = _coh_grp(u)
            if cg_id:
                placed_by_group.setdefault(cg_id, []).append((bx, by, bw, bh))

    # Ceiling-lift (TOP two-pass): TOP-forced blocks are placed last (order_key) and
    # land on the contour at whatever local height — usually short of the global pack
    # top H, so they violate their boundary. Nothing is placed above them, so lift
    # each up to touch H whenever the column strip above it is empty. Fixes the
    # violation without growing the bbox (H unchanged) and without overlap risk
    # (strip verified free). x and shape are untouched, so LEFT/RIGHT corner contact
    # (T+L, T+R) is preserved; skip T+B (can't satisfy both by lifting).
    if TOP_LIFT and placement:
        H = max(y + h for (_, y, _, h) in placement.values())
        obst = [(ox, oy, ow, oh) for (ox, oy, ow, oh, _) in obstacles]
        top_keys = [u.key for u in movable
                    if (u.bc & BOUND_TOP) and not (u.bc & BOUND_BOTTOM)]
        # lift the highest-topped first so a lower top block can't block a higher one
        top_keys.sort(key=lambda k: -(placement[k][1] + placement[k][3]))
        for k in top_keys:
            x, y, w, h = placement[k]
            if abs((y + h) - H) < 1e-6:
                continue                                   # already touches the top
            y0 = y + h
            free = True
            for kk, (xj, yj, wj, hj) in placement.items():
                if kk == k:
                    continue
                if (min(x + w, xj + wj) - max(x, xj) > 1e-6 and
                        min(H, yj + hj) - max(y0, yj) > 1e-6):
                    free = False
                    break
            if free:
                for (ox, oy, ow, oh) in obst:
                    if (min(x + w, ox + ow) - max(x, ox) > 1e-6 and
                            min(H, oy + oh) - max(y0, oy) > 1e-6):
                        free = False
                        break
            if free:
                placement[k] = (x, H - h, w, h)

    return placement


def skyline_legalize_shaped(
    blocks: List[BlockInfo],
    super_blocks: Dict[int, SuperBlock],
    cluster_groups: Dict[int, List[int]],
    cx, cy, area_targets,
    lam: float = SKY_LAM, b2b=None, p2b=None, pins=None, mib_shape: bool = False,
    cluster_shape: bool = False, cohesion: bool = False,
) -> Tuple[List[Tuple[float, float, float, float]], float]:
    """Stock `skyline_legalize` width-ladder + square/row modes, PLUS a contour-
    shaped mode. Returns (positions, score) for the best-scoring candidate.
    `mib_shape` (default off) also reshapes fully-free MIB groups in the shaped
    pass — measured as a net regression (1.8517→1.8576), kept for the record."""
    movable, obstacles = _build_units(blocks, super_blocks, cluster_groups, cx, cy)
    hpwl_fn = _make_hpwl_fn(b2b, p2b, pins)          # precompute net arrays once
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
    # Weight of the aspect-prior regularizer in the area-proxy gate. This biases width
    # selection toward asp_pred and is NOT part of the contest cost, so at 0.5 it was
    # over-constraining the area/HPWL trade. A 100-case sweep found a basin at
    # 0.25–0.35 (0.3 → Total 1.5972 vs 0.5 → 1.6039, both HPWL & area improve weighted);
    # past 0.5 it degrades (1.0 → 1.6245). Override via ASPECT_PRIOR to re-sweep.
    PRIOR = float(os.environ.get("ASPECT_PRIOR", "0.3"))

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

    # Cohesion mode packs freed cluster members into connected blobs; passes 1 & 2
    # (square / row) place them disconnected and the internal area·HPWL proxy ignores
    # grouping, so they would wrongly out-score the cohesion pass. Pass 1 still runs
    # for the width search (top_Ws); only its best_pos update is suppressed.
    if not cohesion:
        for W, a_s, pos in sorted(sq_area, key=lambda t: t[1])[:2]:
            if pos is None:
                continue
            hp = hpwl_fn(pos)
            score = a_s * (hp if hp > 1e-9 else 1.0)
            if score < best_score:
                best_score = score; best_pos = pos

    top_Ws = [W for W, _, _ in sorted(sq_area, key=lambda t: t[1])[:4]]

    # --- Pass 2: row-assigned shapes (identical to stock) ---
    if not cohesion:
        for W in top_Ws:
            _row_assign_reshape(movable, W)
            pl = _pack_one_width(movable, obstacles, W, lam)
            _restore_squares()
            if pl is None:
                continue
            pos = _materialize(pl, blocks, super_blocks, cluster_groups)
            a_s = _area_proxy(pos)
            hp = hpwl_fn(pos)
            score = a_s * (hp if hp > 1e-9 else 1.0)
            if score < best_score:
                best_score = score; best_pos = pos

    # --- Pass 3: contour-aware shaped (NEW; B-MIB / B-cluster reshape too) ---
    # B-cluster mutates super_blocks[gid] (offsets/w/h) at landing time so the
    # immediately-following _materialize is consistent. Snapshot + restore so the
    # object is pristine afterwards (downstream slide/enforce read only positions).
    sb_snapshot = {gid: (list(sb.offsets), sb.w, sb.h)
                   for gid, sb in super_blocks.items()} if cluster_shape else None
    for W in top_Ws:
        pl = _pack_one_width_shaped(movable, obstacles, W, lam, blocks=blocks,
                                    mib_shape=mib_shape, super_blocks=super_blocks,
                                    cluster_groups=cluster_groups,
                                    cluster_shape=cluster_shape, cohesion=cohesion)
        if pl is None:
            continue
        pos = _materialize(pl, blocks, super_blocks, cluster_groups)
        a_s = _area_proxy(pos)
        hp = hpwl_fn(pos)
        score = a_s * (hp if hp > 1e-9 else 1.0)
        if score < best_score:
            best_score = score; best_pos = pos
    if sb_snapshot is not None:
        for gid, (offs, w, h) in sb_snapshot.items():
            super_blocks[gid].offsets = offs
            super_blocks[gid].w = w
            super_blocks[gid].h = h

    _restore_squares()

    if best_pos is None:                                  # square-shape fallback
        placement = _pack_one_width(movable, obstacles, W_min_sq, lam)
        best_pos = _materialize(placement, blocks, super_blocks, cluster_groups)
        x2 = max(p[0] + p[2] for p in best_pos)
        y2 = max(p[1] + p[3] for p in best_pos)
        best_score = x2 * y2

    return best_pos, best_score
