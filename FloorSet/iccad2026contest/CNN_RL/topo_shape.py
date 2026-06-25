"""
Phase 15 / B2 — joint critical-path shape optimisation on the constraint graph.

Where B1 (`skyline_shape.py`) chooses each block's aspect greedily as it lands on
the skyline contour (one block at a time, can't see the global critical path,
skips boundary/cluster), B2 fixes the relative order (HCG/VCG) from an already-
legalised layout and reshapes blocks based on **global slack**:

  - W_env = longest weighted path in the HCG (widths);  H_env = same in VCG (heights).
  - A block's horizontal slack = how much its x could move without changing W_env
    (slack≈0 ⇒ it is ON the horizontal critical path). Vertical slack symmetric.
  - To shrink the binding dimension (say height): take blocks on the *vertical
    critical path* that have *horizontal slack*, make them shorter+wider
    (area-constant) into that slack. This reduces H_env without growing W_env —
    the move B1's greedy and the raw-geometry PoC structurally cannot make.

This is Yan-Chu slack-driven shaping (ISPD'08) on `analytic_legalizer`'s existing
constraint graph. It imports the pure topology helpers (`build_topology`,
`longest_path_pack`) and does NOT modify `analytic_legalizer/`. Edge weights are
refreshed from current `blocks[i].w/h` each iteration (the cached weights in the
succ lists go stale once we reshape). The caller keeps the result only if it
scores better than B1 (never-worse).
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

from legalizer.topology import build_topology, longest_path_pack, compact
from legalizer.constraints import (
    BlockInfo, SuperBlock,
    BOUND_LEFT, BOUND_RIGHT, BOUND_TOP, BOUND_BOTTOM,
)

AR_MAX = 3.0


def _rep_fn(blocks, cluster_groups):
    cluster_rep = {gid: members[0] for gid, members in cluster_groups.items()}

    def rep(i):
        g = blocks[i].cluster_group
        return cluster_rep[g] if g > 0 else i
    return rep


def _rep_dims(reps, blocks, super_blocks, rep):
    """Current (w, h) per rep from live blocks[i].w/h (clusters → super-block)."""
    rw: Dict[int, float] = {}
    rh: Dict[int, float] = {}
    member = {r: None for r in reps}
    for i in range(len(blocks)):
        r = rep(i)
        if member[r] is None:
            member[r] = i
    for r in reps:
        i = member[r]
        g = blocks[i].cluster_group
        if g > 0 and g in super_blocks:
            rw[r] = super_blocks[g].w
            rh[r] = super_blocks[g].h
        else:
            rw[r] = blocks[i].w
            rh[r] = blocks[i].h
    return rw, rh


def _refresh(succ, reps, dim):
    """Rebuild succ edge weights = current source dim (w for HCG, h for VCG)."""
    return [[(j, dim[i]) for (j, _w) in succ[i]] if i in dim else succ[i]
            for i in range(len(succ))]


def _path_lengths(reps, succ, dim):
    """Return (start, out): start[r] = longest weighted path INTO r (= earliest
    lower coord); out[r] = longest weighted path FROM r including r's own size.
    Envelope extent = max_r out[r]. slack[r] = (E - out[r]) - start[r]."""
    rset = set(reps)
    succs = {r: [(j, w) for (j, w) in succ[r] if j in rset] for r in reps}
    preds: Dict[int, List[int]] = {r: [] for r in reps}
    indeg = {r: 0 for r in reps}
    for r in reps:
        for (j, _w) in succs[r]:
            preds[j].append(r)
            indeg[j] += 1
    # Kahn topo order.
    order: List[int] = []
    q = [r for r in reps if indeg[r] == 0]
    seen = dict(indeg)
    while q:
        u = q.pop()
        order.append(u)
        for (v, _w) in succs[u]:
            seen[v] -= 1
            if seen[v] == 0:
                q.append(v)
    if len(order) < len(reps):                       # residual cycle fallback
        order = list(reps)

    start = {r: 0.0 for r in reps}
    for u in order:
        for (v, w) in succs[u]:
            if start[u] + w > start[v]:
                start[v] = start[u] + w
    out = {r: dim[r] for r in reps}
    for u in reversed(order):
        best_child = 0.0
        for (v, _w) in succs[u]:
            if out[v] > best_child:
                best_child = out[v]
        out[u] = dim[u] + best_child
    return start, out


def legalize_b2(positions, blocks, super_blocks, cluster_groups, area_targets,
                lam: float = 0.3, b2b=None, p2b=None, pins=None,
                n_iter: int = 20):
    """Critical-path slack shaping on top of an already-legalised layout
    `positions`. Fixes the relative order from the layout's centers, iteratively
    reshapes soft blocks on the binding critical path into perpendicular slack,
    then RE-PACKS with lean directional `compact` (NOT longest_path_pack, which
    over-serialises ~2×). Returns positions (caller keeps them only if better)."""
    n = len(blocks)
    rep = _rep_fn(blocks, cluster_groups)
    reps = sorted(set(rep(i) for i in range(n)))

    cx = [p[0] + p[2] / 2.0 for p in positions]
    cy = [p[1] + p[3] / 2.0 for p in positions]
    hcg, vcg = build_topology(cx, cy, blocks, super_blocks, cluster_groups)

    # Which reps may reshape: singleton, soft (area-only), not preplaced. Boundary
    # blocks are allowed (topology keeps their edge); clusters/MIB/fixed are rigid.
    reshapeable: Dict[int, bool] = {}
    for r in reps:
        b = blocks[r]
        free = (b.cluster_group == 0 and not b.is_fixed_shape
                and b.mib_group == 0 and not b.is_preplaced)
        reshapeable[r] = free
    area_of = {r: max(float(area_targets[r]), 1e-6) for r in reps if reshapeable[r]}

    rw, rh = _rep_dims(reps, blocks, super_blocks, rep)
    prev_bbox = float("inf")
    for _ in range(n_iter):
        hx = _refresh(hcg, reps, rw)
        vy = _refresh(vcg, reps, rh)
        sx, ox = _path_lengths(reps, hx, rw)        # horizontal
        sy, oy = _path_lengths(reps, vy, rh)        # vertical
        W = max((ox[r] for r in reps), default=1.0)
        H = max((oy[r] for r in reps), default=1.0)
        if W * H >= prev_bbox - 1e-6:
            break
        prev_bbox = W * H

        if H >= W:                                  # shrink height: shorter+wider
            for r in reps:
                if not reshapeable[r]:
                    continue
                slack_v = (H - oy[r]) - sy[r]        # vertical slack
                slack_h = (W - ox[r]) - sx[r]        # horizontal room to grow
                if slack_v <= 1e-6 and slack_h > 1e-6:    # on V-critical path, has H room
                    a = area_of[r]
                    new_w = min(rw[r] + slack_h, math.sqrt(a * AR_MAX))
                    if new_w > rw[r] + 1e-9:
                        rw[r] = new_w
                        rh[r] = a / new_w
        else:                                       # shrink width: narrower+taller
            for r in reps:
                if not reshapeable[r]:
                    continue
                slack_h = (W - ox[r]) - sx[r]
                slack_v = (H - oy[r]) - sy[r]
                if slack_h <= 1e-6 and slack_v > 1e-6:
                    a = area_of[r]
                    new_h = min(rh[r] + slack_v, math.sqrt(a * AR_MAX))
                    if new_h > rh[r] + 1e-9:
                        rh[r] = new_h
                        rw[r] = a / new_h

    # Commit reshaped dims to the live blocks AND into the positions tuples
    # (compact reads sizes from positions), keeping each block's lower-left, then
    # re-pack with lean directional compaction to absorb the freed whitespace.
    member = {r: None for r in reps}
    for i in range(n):
        member.setdefault(rep(i), None)
    reshaped_pos = list(positions)
    for r in reps:
        if reshapeable[r]:
            blocks[r].w = rw[r]
            blocks[r].h = rh[r]
            x, y, _w, _h = reshaped_pos[r]
            reshaped_pos[r] = (x, y, rw[r], rh[r])
    return compact(reshaped_pos, blocks, super_blocks, cluster_groups)
