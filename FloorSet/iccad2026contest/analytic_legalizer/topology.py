"""
Steps 4 + 5 — Topology extraction and longest-path legalization.

Step 4: Build CYCLE-FREE HCG and VCG using a sweep-line approach:
  - Sort by cx; add HCG edge i→j (i left of j) only when their y-projections overlap.
  - Sort by cy; add VCG edge i→j (i below j) only when their x-projections overlap.
  - For blocks that are diagonally separated (no projection overlap), add one edge
    based on cx order to prevent them both landing at x=0.
  - This construction is GUARANTEED acyclic (edges only go in increasing cx or cy
    order), so Kahn's longest-path always terminates completely.

Step 5: Longest-path on HCG gives x-coords, VCG gives y-coords.
  - Preplaced blocks enforced to exact coords with forward+backward propagation.
"""

from typing import Dict, List, Set, Tuple

import numpy as np

from .constraints import (
    BlockInfo, SuperBlock, BOUND_LEFT, BOUND_BOTTOM, BOUND_RIGHT, BOUND_TOP,
)


def build_topology(
    block_cx: np.ndarray,
    block_cy: np.ndarray,
    blocks: List[BlockInfo],
    super_blocks: Dict[int, SuperBlock],
    cluster_groups: Dict[int, List[int]],
) -> Tuple[List[List[Tuple[int, float]]], List[List[Tuple[int, float]]]]:
    """
    Step 4: Build cycle-free HCG and VCG.
    Returns hcg_succ[i], vcg_succ[i] indexed by representative block index.
    """
    n = len(blocks)

    cluster_rep: Dict[int, int] = {}
    for gid, members in cluster_groups.items():
        cluster_rep[gid] = members[0]

    def rep(i): return cluster_rep[blocks[i].cluster_group] if blocks[i].cluster_group > 0 else i

    def bw(r):
        for i in range(n):
            if rep(i) == r:
                gid = blocks[i].cluster_group
                return super_blocks[gid].w if gid > 0 and gid in super_blocks else blocks[i].w
        return 1.0

    def bh(r):
        for i in range(n):
            if rep(i) == r:
                gid = blocks[i].cluster_group
                return super_blocks[gid].h if gid > 0 and gid in super_blocks else blocks[i].h
        return 1.0

    reps = sorted(set(rep(i) for i in range(n)))
    n_reps = len(reps)

    # Precompute center and size for each rep.
    # For cluster reps, block_cx[r] is the center of the *first member*,
    # not the super-block center. Correct it using member offsets.
    rep_cx: Dict[int, float] = {}
    rep_cy: Dict[int, float] = {}
    rep_w    = {r: bw(r) for r in reps}
    rep_h    = {r: bh(r) for r in reps}

    # Boundary code: use super_blocks[gid].boundary_code (OR of all member codes)
    # for cluster reps; fall back to blocks[r].boundary_code for singletons.
    rep_bc: Dict[int, int] = {}
    for r in reps:
        gid = blocks[r].cluster_group if r < n else 0
        if gid > 0 and gid in super_blocks:
            rep_bc[r] = super_blocks[gid].boundary_code
        else:
            rep_bc[r] = blocks[r].boundary_code

    for r in reps:
        gid = blocks[r].cluster_group if r < n else 0
        if gid > 0 and gid in super_blocks:
            sb = super_blocks[gid]
            mi = sb.members.index(r)
            dx_r, dy_r = sb.offsets[mi]
            # block_cx[r] = sb_ll_x + dx_r + blocks[r].w/2
            # sb_center_x = sb_ll_x + sb.w/2
            rep_cx[r] = block_cx[r] - dx_r - blocks[r].w / 2.0 + sb.w / 2.0
            rep_cy[r] = block_cy[r] - dy_r - blocks[r].h / 2.0 + sb.h / 2.0
        else:
            rep_cx[r] = block_cx[r]
            rep_cy[r] = block_cy[r]

    # Block extents (lower-left / upper-right from center)
    def x1(r): return rep_cx[r] - rep_w[r] / 2.0
    def x2(r): return rep_cx[r] + rep_w[r] / 2.0
    def y1(r): return rep_cy[r] - rep_h[r] / 2.0
    def y2(r): return rep_cy[r] + rep_h[r] / 2.0

    hcg_adj: Set = set()   # (ri, rj) = ri is left of rj
    vcg_adj: Set = set()

    # --- Boundary hints ---
    # LEFT blocks: no in-edges in HCG (will land at x=0)
    # BOTTOM blocks: no in-edges in VCG (will land at y=0)
    left_forced:   Set[int] = set()
    bottom_forced: Set[int] = set()
    for r in reps:
        if rep_bc[r] & BOUND_LEFT:
            left_forced.add(r)
        if rep_bc[r] & BOUND_BOTTOM:
            bottom_forced.add(r)

    # --- Sweep-line topology construction ---
    # HCG: sort by cx; for each pair (i<j in cx order) decide if they need x-separation
    reps_by_cx = sorted(reps, key=lambda r: rep_cx[r])
    reps_by_cy = sorted(reps, key=lambda r: rep_cy[r])

    for k_i, ri in enumerate(reps_by_cx):
        for k_j in range(k_i + 1, n_reps):
            rj = reps_by_cx[k_j]
            if rj in left_forced:
                # rj is forced to x=0 (no in-edges).  ri has smaller analytic cx
                # but rj will physically be at x=0, so ri must come AFTER rj.
                # Add the reverse edge rj→ri so ri is pushed right of rj.
                y_overlap = min(y2(ri), y2(rj)) - max(y1(ri), y1(rj))
                if y_overlap > -1e-6 and (rj, ri) not in hcg_adj and ri not in left_forced:
                    hcg_adj.add((rj, ri))
                continue  # never add in-edges to LEFT-forced blocks

            # Check y-projection overlap → need x-separation
            y_overlap = min(y2(ri), y2(rj)) - max(y1(ri), y1(rj))
            if y_overlap > -1e-6:   # y-projections overlap or nearly touch
                if (ri, rj) not in hcg_adj:
                    hcg_adj.add((ri, rj))

    # VCG: sort by cy; same logic for x-projection overlap
    for k_i, ri in enumerate(reps_by_cy):
        for k_j in range(k_i + 1, n_reps):
            rj = reps_by_cy[k_j]
            if rj in bottom_forced:
                # rj is forced to y=0; add reverse edge rj→ri so ri is pushed above rj.
                x_overlap = min(x2(ri), x2(rj)) - max(x1(ri), x1(rj))
                if x_overlap > -1e-6 and (rj, ri) not in vcg_adj and ri not in bottom_forced:
                    vcg_adj.add((rj, ri))
                continue

            x_overlap = min(x2(ri), x2(rj)) - max(x1(ri), x1(rj))
            if x_overlap > -1e-6:
                if (ri, rj) not in vcg_adj:
                    vcg_adj.add((ri, rj))

    # --- Stack multiple forced-boundary blocks to prevent them all landing at x=0 / y=0 ---
    # Multiple LEFT-forced blocks all land at x=0; if their y-extents overlap they'd
    # overlap each other. Stack them with VCG edges (sorted by cy, bottom→top).
    # But only add VCG in-edges to blocks that are NOT also bottom_forced (those need y=0).
    left_only = [r for r in left_forced if r not in bottom_forced]
    bottom_only = [r for r in bottom_forced if r not in left_forced]
    # Corner blocks need both constraints — separate them with HCG (sorted by cx)
    corner_left_bottom = [r for r in left_forced if r in bottom_forced]

    left_only_sorted = sorted(left_only, key=lambda r: rep_cy[r])
    for k in range(len(left_only_sorted) - 1):
        ri = left_only_sorted[k]
        rj = left_only_sorted[k + 1]
        # At x=0 they both start at x=0 → x-extents always overlap; check y-extents
        y_ov = min(y2(ri), y2(rj)) - max(y1(ri), y1(rj))
        if y_ov > -1e-6 and (ri, rj) not in vcg_adj:
            vcg_adj.add((ri, rj))

    bottom_only_sorted = sorted(bottom_only, key=lambda r: rep_cx[r])
    for k in range(len(bottom_only_sorted) - 1):
        ri = bottom_only_sorted[k]
        rj = bottom_only_sorted[k + 1]
        x_ov = min(x2(ri), x2(rj)) - max(x1(ri), x1(rj))
        if x_ov > -1e-6 and (ri, rj) not in hcg_adj:
            hcg_adj.add((ri, rj))

    # Corner blocks (LEFT+BOTTOM): stack with VCG so they at least have x=0 (LEFT
    # satisfied); only the bottommost keeps y=0 (BOTTOM satisfied for one block).
    corner_sorted = sorted(corner_left_bottom, key=lambda r: rep_cy[r])
    for k in range(len(corner_sorted) - 1):
        ri = corner_sorted[k]
        rj = corner_sorted[k + 1]
        if (ri, rj) not in vcg_adj:
            vcg_adj.add((ri, rj))

    # --- RIGHT / TOP forcing (mirror of LEFT / BOTTOM) ---
    # A RIGHT block should be the right frontier of its y-band: every block whose
    # y-projection overlaps it must come BEFORE it in HCG (edge s→r), and it must
    # have no HCG out-edges.  Symmetric for TOP in VCG.  Multiple RIGHT blocks in
    # the same band are stacked in VCG so they don't all collide at the frontier.
    right_forced: Set[int] = {r for r in reps if rep_bc[r] & BOUND_RIGHT}
    top_forced:   Set[int] = {r for r in reps if rep_bc[r] & BOUND_TOP}

    for r in right_forced:
        for s in reps:
            if s == r or s in right_forced:
                continue
            y_ov = min(y2(r), y2(s)) - max(y1(r), y1(s))
            if y_ov > -1e-6:
                hcg_adj.discard((r, s))  # r has no out-edges
                hcg_adj.add((s, r))      # s is left of r
    right_sorted = sorted(right_forced, key=lambda r: rep_cy[r])
    for k in range(len(right_sorted) - 1):
        ri, rj = right_sorted[k], right_sorted[k + 1]
        if (ri, rj) not in vcg_adj and (rj, ri) not in vcg_adj:
            vcg_adj.add((ri, rj))

    for r in top_forced:
        for s in reps:
            if s == r or s in top_forced:
                continue
            x_ov = min(x2(r), x2(s)) - max(x1(r), x1(s))
            if x_ov > -1e-6:
                vcg_adj.discard((r, s))
                vcg_adj.add((s, r))
    top_sorted = sorted(top_forced, key=lambda r: rep_cx[r])
    for k in range(len(top_sorted) - 1):
        ri, rj = top_sorted[k], top_sorted[k + 1]
        if (ri, rj) not in hcg_adj and (rj, ri) not in hcg_adj:
            hcg_adj.add((ri, rj))

    # For pairs that are diagonally separated (no projection overlap in either axis),
    # force an HCG edge based on cx order to prevent both landing at x=0.
    all_pairs: Set[Tuple[int,int]] = set()
    for ri in reps:
        for rj in reps:
            if ri < rj:
                all_pairs.add((ri, rj))

    for (ri, rj) in all_pairs:
        # Make canonical order by cx
        if rep_cx[ri] > rep_cx[rj]:
            ri, rj = rj, ri

        has_hcg = (ri, rj) in hcg_adj or (rj, ri) in hcg_adj
        has_vcg = (ri, rj) in vcg_adj or (rj, ri) in vcg_adj

        if not has_hcg and not has_vcg:
            # Diagonally separated: add HCG edge (smaller cx → larger cx)
            if rj not in left_forced:
                hcg_adj.add((ri, rj))

    # --- Convert to adjacency lists ---
    hcg_succ: List[List[Tuple[int, float]]] = [[] for _ in range(n)]
    vcg_succ: List[List[Tuple[int, float]]] = [[] for _ in range(n)]

    for (ri, rj) in hcg_adj:
        hcg_succ[ri].append((rj, rep_w[ri]))

    for (ri, rj) in vcg_adj:
        vcg_succ[ri].append((rj, rep_h[ri]))

    return hcg_succ, vcg_succ


def longest_path_pack(
    hcg_succ: List[List[Tuple[int, float]]],
    vcg_succ: List[List[Tuple[int, float]]],
    blocks: List[BlockInfo],
    super_blocks: Dict[int, SuperBlock],
    cluster_groups: Dict[int, List[int]],
) -> List[Tuple[float, float, float, float]]:
    """
    Step 5: Compute overlap-free positions from HCG/VCG via longest-path.
    Preplaced blocks are fixed via forward+backward constraint propagation.
    """
    n = len(blocks)

    cluster_rep: Dict[int, int] = {}
    for gid, members in cluster_groups.items():
        cluster_rep[gid] = members[0]

    def rep(i): return cluster_rep[blocks[i].cluster_group] if blocks[i].cluster_group > 0 else i

    def bw_rep(r):
        for i in range(n):
            if rep(i) == r:
                gid = blocks[i].cluster_group
                return super_blocks[gid].w if gid > 0 and gid in super_blocks else blocks[i].w
        return 1.0

    def bh_rep(r):
        for i in range(n):
            if rep(i) == r:
                gid = blocks[i].cluster_group
                return super_blocks[gid].h if gid > 0 and gid in super_blocks else blocks[i].h
        return 1.0

    reps = sorted(set(rep(i) for i in range(n)))
    reps_set = set(reps)

    # Build preplaced info indexed by rep.
    # If a preplaced block is inside a cluster, the cluster rep's position must be
    # offset so that (rep_pos + member_offset) == preplaced block's fixed position.
    preplaced_x: Dict[int, float] = {}
    preplaced_y: Dict[int, float] = {}
    for i, b in enumerate(blocks):
        if b.is_preplaced:
            r = rep(i)
            if b.cluster_group > 0 and b.cluster_group in super_blocks:
                sb = super_blocks[b.cluster_group]
                mi = sb.members.index(i)
                dx_i, dy_i = sb.offsets[mi]
                preplaced_x[r] = b.fixed_x - dx_i
                preplaced_y[r] = b.fixed_y - dy_i
            else:
                preplaced_x[r] = b.fixed_x
                preplaced_y[r] = b.fixed_y

    # --- Longest path ---
    rep_x = _longest_path(reps, hcg_succ)
    rep_y = _longest_path(reps, vcg_succ)

    # --- Override preplaced to exact coords ---
    for r, fx in preplaced_x.items():
        rep_x[r] = fx
    for r, fy in preplaced_y.items():
        rep_y[r] = fy

    # --- Propagation: interleaved forward + backward until convergence ---
    # Forward: successor can't start before predecessor ends.
    # Backward: predecessor must end before successor starts (general, not just preplaced).
    # Interleaved because a backward push on B may require a further backward push on A (A→B→C).
    for _ in range(2 * (len(reps) + 1)):
        changed = False

        # HCG forward pass (always uses current block width via bw_rep)
        for u in reps:
            uw = bw_rep(u)
            for (v, w_edge) in hcg_succ[u]:
                if v not in reps_set or v in preplaced_x:
                    continue
                new_vx = rep_x[u] + uw
                if rep_x.get(v, 0) < new_vx - 1e-9:
                    rep_x[v] = new_vx
                    changed = True

        # HCG backward pass (general: fires for all edges, not just preplaced v)
        for u in reps:
            if u in preplaced_x:
                continue
            for (v, w_edge) in hcg_succ[u]:
                if v not in reps_set:
                    continue
                vx = preplaced_x.get(v, rep_x.get(v, 0.0))
                max_u_x = vx - w_edge
                if rep_x.get(u, 0) > max_u_x + 1e-9:
                    rep_x[u] = max_u_x
                    changed = True

        if not changed:
            break

    for _ in range(2 * (len(reps) + 1)):
        changed = False

        # VCG forward pass (always uses current block height via bh_rep)
        for u in reps:
            uh = bh_rep(u)
            for (v, h_edge) in vcg_succ[u]:
                if v not in reps_set or v in preplaced_y:
                    continue
                new_vy = rep_y[u] + uh
                if rep_y.get(v, 0) < new_vy - 1e-9:
                    rep_y[v] = new_vy
                    changed = True

        # VCG backward pass (general)
        for u in reps:
            if u in preplaced_y:
                continue
            for (v, h_edge) in vcg_succ[u]:
                if v not in reps_set:
                    continue
                vy = preplaced_y.get(v, rep_y.get(v, 0.0))
                max_u_y = vy - h_edge
                if rep_y.get(u, 0) > max_u_y + 1e-9:
                    rep_y[u] = max_u_y
                    changed = True

        if not changed:
            break

    # Clamp all rep_x/rep_y to ≥ 0 (boundary blocks forced to origin), then
    # run one final forward pass so successors properly account for the clamped
    # values.  Without the clamp, backward propagation can push a BOUND_LEFT
    # rep to rep_x = -12.5 and its successor to rep_x = 12.5.  max(0, -12.5) = 0
    # at position-build time, but the successor still sits at 12.5 — which now
    # OVERLAPS the cluster that physically spans [0, 25].
    for r in reps:
        if r not in preplaced_x:
            rep_x[r] = max(0.0, rep_x.get(r, 0.0))
        if r not in preplaced_y:
            rep_y[r] = max(0.0, rep_y.get(r, 0.0))

    # Final forward-only pass: re-establish topology constraints from clamped values.
    for _ in range(len(reps) + 1):
        changed = False
        for u in reps:
            uw = bw_rep(u)
            for (v, w_edge) in hcg_succ[u]:
                if v not in reps_set or v in preplaced_x:
                    continue
                new_vx = rep_x[u] + uw
                if rep_x.get(v, 0) < new_vx - 1e-9:
                    rep_x[v] = new_vx
                    changed = True
        if not changed:
            break

    for _ in range(len(reps) + 1):
        changed = False
        for u in reps:
            uh = bh_rep(u)
            for (v, h_edge) in vcg_succ[u]:
                if v not in reps_set or v in preplaced_y:
                    continue
                new_vy = rep_y[u] + uh
                if rep_y.get(v, 0) < new_vy - 1e-9:
                    rep_y[v] = new_vy
                    changed = True
        if not changed:
            break

    # --- Build final positions ---
    positions: List[Tuple[float, float, float, float]] = [(0.0, 0.0, 0.0, 0.0)] * n

    for i in range(n):
        r = rep(i)
        x_r = max(0.0, rep_x.get(r, 0.0))
        y_r = max(0.0, rep_y.get(r, 0.0))

        if blocks[i].cluster_group > 0 and blocks[i].cluster_group in super_blocks:
            sb = super_blocks[blocks[i].cluster_group]
            mi = sb.members.index(i)
            dx, dy = sb.offsets[mi]
            positions[i] = (x_r + dx, y_r + dy, blocks[i].w, blocks[i].h)
        else:
            positions[i] = (x_r, y_r, bw_rep(r), bh_rep(r))

    return positions


def compact(
    positions: List[Tuple[float, float, float, float]],
    blocks: List[BlockInfo],
    super_blocks: Dict[int, SuperBlock],
    cluster_groups: Dict[int, List[int]],
    rounds: int = 12,
) -> List[Tuple[float, float, float, float]]:
    """
    Directional (shadow) compaction.  Alternates x- and y-packing: each pass
    re-derives a LEAN constraint graph from the current layout — an edge only
    between reps whose PERPENDICULAR projections currently overlap — then packs
    that axis toward the origin with preplaced reps pinned.

    Because every perpendicular-overlapping pair gets an edge, the packed axis is
    guaranteed overlap-free (pairs that do not overlap perpendicularly can never
    overlap regardless of their coordinate on the packed axis).  Unlike
    build_topology this adds NO diagonal-separation edges, so it does not
    over-serialize — it removes whitespace instead of creating it.

    Preplaced blocks (always individual reps — detached from clusters upstream)
    keep their fixed coordinate.  Clusters move as rigid units; their internal
    member layout is preserved.
    """
    n = len(blocks)

    cluster_rep: Dict[int, int] = {gid: members[0] for gid, members in cluster_groups.items()}

    def rep(i: int) -> int:
        return cluster_rep[blocks[i].cluster_group] if blocks[i].cluster_group > 0 else i

    reps = sorted(set(rep(i) for i in range(n)))
    members_of: Dict[int, List[int]] = {r: [] for r in reps}
    for i in range(n):
        members_of[rep(i)].append(i)

    pos = [tuple(p) for p in positions]

    # Rep extents derived from current member positions (robust to reshaping).
    def rep_box(r):
        ms = members_of[r]
        lo_x = min(pos[m][0] for m in ms)
        lo_y = min(pos[m][1] for m in ms)
        hi_x = max(pos[m][0] + pos[m][2] for m in ms)
        hi_y = max(pos[m][1] + pos[m][3] for m in ms)
        return lo_x, lo_y, hi_x, hi_y

    pin_x: Dict[int, float] = {}
    pin_y: Dict[int, float] = {}
    for i in range(n):
        if blocks[i].is_preplaced:
            r = rep(i)
            lo_x, lo_y, _, _ = rep_box(r)
            pin_x[r] = lo_x
            pin_y[r] = lo_y

    for _ in range(rounds):
        before = (max(pos[i][0] + pos[i][2] for i in range(n)),
                  max(pos[i][1] + pos[i][3] for i in range(n)))
        _pack_axis(pos, reps, members_of, rep_box, pin_x, axis=0)
        _pack_axis(pos, reps, members_of, rep_box, pin_y, axis=1)
        after = (max(pos[i][0] + pos[i][2] for i in range(n)),
                 max(pos[i][1] + pos[i][3] for i in range(n)))
        if abs(after[0] - before[0]) < 1e-6 and abs(after[1] - before[1]) < 1e-6:
            break

    return [tuple(p) for p in pos]


def _pack_axis(pos, reps, members_of, rep_box, pin, axis: int) -> None:
    """
    Pack all reps toward the origin along `axis` (0=x, 1=y), in place.
    An edge i→j (i before j) is added for every rep pair whose extents on the
    PERPENDICULAR axis currently overlap, oriented by current low-coordinate.
    Longest-path with pinned reps (forward + backward propagation) then sets the
    packed-axis low-coordinate of each rep; member positions are shifted rigidly.
    """
    perp = 1 - axis

    boxes = {r: rep_box(r) for r in reps}
    lo = {r: boxes[r][axis] for r in reps}            # current low on packed axis
    size = {r: boxes[r][axis + 2] - boxes[r][axis] for r in reps}
    plo = {r: boxes[r][perp] for r in reps}
    phi = {r: boxes[r][perp + 2] for r in reps}

    reps_set = set(reps)
    succ: Dict[int, List[int]] = {r: [] for r in reps}
    order = sorted(reps, key=lambda r: (lo[r], r))
    for a in range(len(order)):
        ra = order[a]
        for b in range(a + 1, len(order)):
            rb = order[b]
            # perpendicular projections overlap → need separation on packed axis
            if min(phi[ra], phi[rb]) - max(plo[ra], plo[rb]) > 1e-6:
                succ[ra].append(rb)   # ra (smaller lo) before rb

    new_lo = {r: (pin[r] if r in pin else 0.0) for r in reps}

    # Forward + backward propagation until convergence.
    for _ in range(2 * (len(reps) + 1)):
        changed = False
        # forward: successor starts after predecessor ends
        for u in order:
            for v in succ[u]:
                if v in pin:
                    continue
                cand = new_lo[u] + size[u]
                if new_lo[v] < cand - 1e-9:
                    new_lo[v] = cand
                    changed = True
        # backward: a pinned/placed successor pushes predecessors back
        for u in order:
            if u in pin:
                continue
            for v in succ[u]:
                cap = new_lo[v] - size[u]
                if new_lo[u] > cap + 1e-9:
                    new_lo[u] = cap
                    changed = True
        if not changed:
            break

    # Clamp non-pinned reps to >= 0, then one forward pass to restore order.
    for r in reps:
        if r not in pin and new_lo[r] < 0.0:
            new_lo[r] = 0.0
    for _ in range(len(reps) + 1):
        changed = False
        for u in order:
            for v in succ[u]:
                if v in pin:
                    continue
                cand = new_lo[u] + size[u]
                if new_lo[v] < cand - 1e-9:
                    new_lo[v] = cand
                    changed = True
        if not changed:
            break

    # Apply: shift each rep's members rigidly by (new_lo - old_lo) on this axis.
    for r in reps:
        d = new_lo[r] - lo[r]
        if abs(d) < 1e-12:
            continue
        for m in members_of[r]:
            x, y, w, h = pos[m]
            if axis == 0:
                pos[m] = (x + d, y, w, h)
            else:
                pos[m] = (x, y + d, w, h)


def augment_topology(
    positions: List[Tuple[float, float, float, float]],
    blocks: List[BlockInfo],
    super_blocks: Dict[int, SuperBlock],
    cluster_groups: Dict[int, List[int]],
    hcg_succ: List[List[Tuple[int, float]]],
    vcg_succ: List[List[Tuple[int, float]]],
) -> bool:
    """
    Scan for overlapping pairs that have no HCG or VCG edge between their
    representatives.  For each such pair, insert a missing edge (HCG or VCG
    based on overlap geometry) and return True if any edge was added.

    Direction heuristic:
      - If a block has LEFT boundary code its rep must stay at x=0 (add HCG
        edge away from it, not into it).
      - If a block has BOTTOM boundary code its rep must stay at y=0 (add VCG
        edge away from it).
      - Otherwise use the smaller centre coordinate to pick the "upstream" node.
    """
    n = len(blocks)
    cluster_rep: Dict[int, int] = {gid: members[0] for gid, members in cluster_groups.items()}

    def rep(i: int) -> int:
        return cluster_rep[blocks[i].cluster_group] if blocks[i].cluster_group > 0 else i

    def sb_bc(r: int) -> int:
        gid = blocks[r].cluster_group
        if gid > 0 and gid in super_blocks:
            return super_blocks[gid].boundary_code
        return blocks[r].boundary_code

    def bw(r: int) -> float:
        gid = blocks[r].cluster_group
        return super_blocks[gid].w if gid > 0 and gid in super_blocks else blocks[r].w

    def bh(r: int) -> float:
        gid = blocks[r].cluster_group
        return super_blocks[gid].h if gid > 0 and gid in super_blocks else blocks[r].h

    hcg_set: Set = set()
    vcg_set: Set = set()
    for u in range(n):
        for (v, _) in hcg_succ[u]:
            hcg_set.add((u, v))
        for (v, _) in vcg_succ[u]:
            vcg_set.add((u, v))

    added = False
    for i in range(n):
        ri = rep(i)
        for j in range(i + 1, n):
            rj = rep(j)
            if ri == rj:
                continue
            xi, yi, wi, hi = positions[i]
            xj, yj, wj, hj = positions[j]
            ox = min(xi + wi, xj + wj) - max(xi, xj)
            oy = min(yi + hi, yj + hj) - max(yi, yj)
            if ox <= 1e-6 or oy <= 1e-6:
                continue
            if (ri, rj) in hcg_set or (rj, ri) in hcg_set:
                continue
            if (ri, rj) in vcg_set or (rj, ri) in vcg_set:
                continue

            bc_ri = sb_bc(ri)
            bc_rj = sb_bc(rj)
            ri_left   = bool(bc_ri & BOUND_LEFT)
            rj_left   = bool(bc_rj & BOUND_LEFT)
            ri_bottom = bool(bc_ri & BOUND_BOTTOM)
            rj_bottom = bool(bc_rj & BOUND_BOTTOM)

            if ox < oy:
                # Add HCG edge: left-forced goes first, else use x-centre
                if ri_left and not rj_left:
                    hcg_succ[ri].append((rj, bw(ri)))
                    hcg_set.add((ri, rj))
                elif rj_left and not ri_left:
                    hcg_succ[rj].append((ri, bw(rj)))
                    hcg_set.add((rj, ri))
                else:
                    cx_i = xi + wi / 2.0
                    cx_j = xj + wj / 2.0
                    if cx_i <= cx_j:
                        hcg_succ[ri].append((rj, bw(ri)))
                        hcg_set.add((ri, rj))
                    else:
                        hcg_succ[rj].append((ri, bw(rj)))
                        hcg_set.add((rj, ri))
            else:
                # Add VCG edge: bottom-forced goes first, else use y-centre
                if ri_bottom and not rj_bottom:
                    vcg_succ[ri].append((rj, bh(ri)))
                    vcg_set.add((ri, rj))
                elif rj_bottom and not ri_bottom:
                    vcg_succ[rj].append((ri, bh(rj)))
                    vcg_set.add((rj, ri))
                else:
                    cy_i = yi + hi / 2.0
                    cy_j = yj + hj / 2.0
                    if cy_i <= cy_j:
                        vcg_succ[ri].append((rj, bh(ri)))
                        vcg_set.add((ri, rj))
                    else:
                        vcg_succ[rj].append((ri, bh(rj)))
                        vcg_set.add((rj, ri))
            added = True

    return added


def _longest_path(
    nodes: List[int],
    succ: List[List[Tuple[int, float]]],
) -> Dict[int, float]:
    """
    Compute longest-path distances (Kahn's topological sort + relaxation).
    All nodes start at dist=0. Guaranteed acyclic input → complete processing.
    """
    node_set = set(nodes)
    in_deg: Dict[int, int] = {r: 0 for r in nodes}

    for r in nodes:
        for (rj, _) in succ[r]:
            if rj in node_set:
                in_deg[rj] = in_deg.get(rj, 0) + 1

    dist: Dict[int, float] = {r: 0.0 for r in nodes}
    queue = [r for r in nodes if in_deg[r] == 0]
    queue.sort()

    while queue:
        queue.sort(key=lambda r: dist[r])
        u = queue.pop(0)
        for (v, w) in succ[u]:
            if v not in node_set:
                continue
            new_d = dist[u] + w
            if new_d > dist[v]:
                dist[v] = new_d
            in_deg[v] -= 1
            if in_deg[v] == 0:
                queue.append(v)

    # Any unprocessed node (from residual cycles) gets max dist as fallback
    max_d = max(dist.values()) if dist else 0.0
    for r in nodes:
        if in_deg.get(r, 0) > 0:
            dist[r] = max_d

    return dist
