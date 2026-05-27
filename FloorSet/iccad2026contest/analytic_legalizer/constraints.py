"""
Steps 0, 1, 2, 7, 8 — constraint parsing, shape init, cluster pre-packing,
boundary slide, and hard-constraint enforcement.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# Boundary codes from the spec
BOUND_LEFT   = 1
BOUND_RIGHT  = 2
BOUND_TOP    = 4
BOUND_BOTTOM = 8
BOUND_TL     = 5   # TOP | LEFT
BOUND_TR     = 6   # TOP | RIGHT
BOUND_BL     = 9   # BOTTOM | LEFT
BOUND_BR     = 10  # BOTTOM | RIGHT


@dataclass
class BlockInfo:
    idx: int
    w: float
    h: float
    # None means free
    fixed_x: Optional[float] = None
    fixed_y: Optional[float] = None
    # flags
    is_preplaced: bool = False
    is_fixed_shape: bool = False
    mib_group: int = 0       # 0 = none
    cluster_group: int = 0   # 0 = none
    boundary_code: int = 0   # 0 = none


@dataclass
class SuperBlock:
    """A cluster group pre-packed into one rectangle."""
    members: List[int]           # original block indices
    offsets: List[Tuple[float, float]]  # (dx, dy) relative to super-block origin for each member
    w: float
    h: float
    boundary_code: int = 0       # inherited from any member


def parse_and_init(
    block_count: int,
    area_targets: torch.Tensor,
    constraints: torch.Tensor,
    target_positions: torch.Tensor,
) -> Tuple[List[BlockInfo], Dict[int, List[int]], Dict[int, List[int]]]:
    """
    Steps 0 + 1: parse constraints and assign initial shapes.
    Returns (blocks, mib_groups, cluster_groups).
    """
    blocks: List[BlockInfo] = []
    mib_groups: Dict[int, List[int]] = {}
    cluster_groups: Dict[int, List[int]] = {}

    for i in range(block_count):
        c = constraints[i]  # [fixed, preplaced, mib_id, cluster_id, boundary]
        tp = target_positions[i] if target_positions is not None else None

        is_fixed    = bool(c[0].item() > 0)
        is_preplaced = bool(c[1].item() > 0)
        mib_id      = int(c[2].item())
        cluster_id  = int(c[3].item())
        boundary    = int(c[4].item())

        has_dims = tp is not None and float(tp[2]) != -1 and float(tp[3]) != -1
        has_xy   = tp is not None and float(tp[0]) != -1 and float(tp[1]) != -1

        if has_dims:
            w = float(tp[2])
            h = float(tp[3])
        else:
            area = max(float(area_targets[i]), 1e-6)
            w = h = math.sqrt(area)

        b = BlockInfo(
            idx=i, w=w, h=h,
            is_preplaced=(is_preplaced and has_xy and has_dims),
            is_fixed_shape=(is_fixed and has_dims),
            mib_group=mib_id,
            cluster_group=cluster_id,
            boundary_code=boundary,
        )
        if b.is_preplaced:
            b.fixed_x = float(tp[0])
            b.fixed_y = float(tp[1])

        blocks.append(b)

        if mib_id > 0:
            mib_groups.setdefault(mib_id, []).append(i)
        if cluster_id > 0:
            cluster_groups.setdefault(cluster_id, []).append(i)

    # Detach preplaced members from their clusters.
    # A preplaced block has a fixed (x,y); keeping it inside a rigid super-block
    # pins the whole rectangle at that spot.  When two clusters each contain a
    # preplaced block at a similar coordinate, both rigid rectangles get forced
    # into the same narrow strip and overlap unavoidably (rigid → no DOF).
    # Grouping is a SOFT constraint (violation = connected-components − 1), so we
    # concede at most 1 component per group: the preplaced member becomes its own
    # individually-pinned node, while the remaining members stay abutted as a free
    # (un-pinned) super-block that longest-path can place anywhere.
    for gid in list(cluster_groups.keys()):
        members = cluster_groups[gid]
        pp = [m for m in members if blocks[m].is_preplaced]
        if not pp:
            continue
        for m in pp:
            blocks[m].cluster_group = 0
        remaining = [m for m in members if not blocks[m].is_preplaced]
        if len(remaining) >= 2:
            cluster_groups[gid] = remaining
        else:
            # 0 or 1 non-preplaced members left → no meaningful cluster.
            for m in remaining:
                blocks[m].cluster_group = 0
            del cluster_groups[gid]

    # Step 1: MIB shape unification
    _unify_mib_shapes(blocks, mib_groups, area_targets)

    return blocks, mib_groups, cluster_groups


def _unify_mib_shapes(
    blocks: List[BlockInfo],
    mib_groups: Dict[int, List[int]],
    area_targets: torch.Tensor,
) -> None:
    """All members of each MIB group get the same (w,h)."""
    for gid, members in mib_groups.items():
        # Check for locked anchor (preplaced or fixed-shape member)
        anchor = None
        for idx in members:
            if blocks[idx].is_preplaced or blocks[idx].is_fixed_shape:
                anchor = blocks[idx]
                break

        if anchor is not None:
            # All members adopt the anchor's shape (best we can do)
            aw, ah = anchor.w, anchor.h
            for idx in members:
                if not blocks[idx].is_preplaced and not blocks[idx].is_fixed_shape:
                    blocks[idx].w = aw
                    blocks[idx].h = ah
        else:
            # Free group: pick representative area and square shape
            areas = [float(area_targets[idx]) for idx in members]
            a_rep = sum(areas) / len(areas)
            side = math.sqrt(max(a_rep, 1e-6))
            for idx in members:
                blocks[idx].w = side
                blocks[idx].h = side


def prepack_clusters(
    blocks: List[BlockInfo],
    cluster_groups: Dict[int, List[int]],
) -> Dict[int, SuperBlock]:
    """
    Step 2: pack each cluster into a rigid super-block rectangle.
    Returns dict: cluster_id -> SuperBlock.
    Packing: sort members by decreasing height, pack in rows (shelf-packing).
    """
    super_blocks: Dict[int, SuperBlock] = {}

    for gid, members in cluster_groups.items():
        widths  = [blocks[i].w for i in members]
        heights = [blocks[i].h for i in members]

        # If a preplaced member exists, anchor it at offset (0,0) and stack
        # all other members above it.  This ensures the cluster virtual origin
        # equals the preplaced block's fixed position (always ≥ 0), so no
        # non-preplaced cluster member ever receives a negative absolute coord.
        preplaced_idx = next(
            (k for k, idx in enumerate(members) if blocks[idx].is_preplaced), None
        )
        if preplaced_idx is not None:
            offsets, sb_w, sb_h = _pack_with_anchor(widths, heights, preplaced_idx)
        else:
            offsets, sb_w, sb_h = _shelf_pack(widths, heights)

        # Inherit boundary code from any member
        bc = 0
        for idx in members:
            bc = bc | blocks[idx].boundary_code

        super_blocks[gid] = SuperBlock(
            members=list(members),
            offsets=offsets,
            w=sb_w,
            h=sb_h,
            boundary_code=bc,
        )

    return super_blocks


def _shelf_pack(
    widths: List[float], heights: List[float]
) -> Tuple[List[Tuple[float, float]], float, float]:
    """
    Shelf (row) packing: sort by decreasing height, fill rows left-to-right.
    Returns (offsets, total_w, total_h).
    """
    n = len(widths)
    order = sorted(range(n), key=lambda i: -heights[i])

    # Estimate row width as sqrt of total area (near-square target)
    total_area = sum(w * h for w, h in zip(widths, heights))
    row_width = max(math.sqrt(total_area), max(widths))

    rows: List[List[int]] = [[]]
    row_x: List[float] = [0.0]

    for idx in order:
        w = widths[idx]
        cx = row_x[-1]
        if cx + w > row_width and cx > 0:
            rows.append([])
            row_x.append(0.0)
        rows[-1].append(idx)
        row_x[-1] += w

    # Compute (x, y) offsets per original index
    offsets_by_orig = {}
    y = 0.0
    for row in rows:
        row_h = max(heights[i] for i in row)
        x = 0.0
        for i in row:
            offsets_by_orig[i] = (x, y)
            x += widths[i]
        y += row_h

    total_w = max(offsets_by_orig[i][0] + widths[i] for i in range(n))
    total_h = y

    offsets = [offsets_by_orig[i] for i in range(n)]
    return offsets, total_w, total_h


def _pack_with_anchor(
    widths: List[float], heights: List[float], anchor_idx: int
) -> Tuple[List[Tuple[float, float]], float, float]:
    """
    Pack n blocks into a super-block with the anchor (preplaced) block at (0,0).
    Other blocks are stacked vertically above the anchor so all offsets are ≥ 0.
    Returns (offsets, total_w, total_h) in original-index order.
    """
    n = len(widths)
    offsets: List[Tuple[float, float]] = [(0.0, 0.0)] * n
    offsets[anchor_idx] = (0.0, 0.0)

    # Stack others above the anchor (y ≥ h_anchor) in a single column
    y_cursor = heights[anchor_idx]
    sb_w = widths[anchor_idx]
    for k in range(n):
        if k == anchor_idx:
            continue
        offsets[k] = (0.0, y_cursor)
        y_cursor += heights[k]
        sb_w = max(sb_w, widths[k])

    sb_h = y_cursor
    return offsets, sb_w, sb_h


def slide_boundary(
    positions: List[Tuple[float, float, float, float]],
    blocks: List[BlockInfo],
    super_blocks: Dict[int, SuperBlock],
    cluster_groups: Dict[int, List[int]],
) -> List[Tuple[float, float, float, float]]:
    """
    Step 7: slide blocks/clusters with RIGHT/TOP boundary requirements to the bbox edge.
    Process corners first, then edges.
    Never moves preplaced blocks.
    Returns updated positions list.
    """
    pos = list(positions)

    bbox_x2 = max(x + w for x, y, w, h in pos)
    bbox_y2 = max(y + h for x, y, w, h in pos)

    def _needs_left(bc):   return bc & BOUND_LEFT
    def _needs_right(bc):  return bc & BOUND_RIGHT
    def _needs_top(bc):    return bc & BOUND_TOP
    def _needs_bottom(bc): return bc & BOUND_BOTTOM

    # Build set of blocks already on left/bottom by longest-path packing
    # (they don't need sliding — we only slide for RIGHT/TOP unmet cases)
    corner_codes = {BOUND_TL, BOUND_TR, BOUND_BL, BOUND_BR}
    priority_order = sorted(
        range(len(blocks)),
        key=lambda i: (0 if blocks[i].boundary_code in corner_codes else 1)
    )

    moved_clusters: set = set()

    for i in priority_order:
        b = blocks[i]
        bc = b.boundary_code
        if bc == 0 or b.is_preplaced:
            continue
        if not (_needs_right(bc) or _needs_top(bc)):
            continue  # LEFT/BOTTOM already handled by packing origin

        # Determine slide set: entire cluster if block is in one
        if b.cluster_group > 0 and b.cluster_group not in moved_clusters:
            members = cluster_groups[b.cluster_group]
            # Never slide a cluster that contains a preplaced member — 8a will restore
            # it and any slide would create a post-restore overlap.
            if any(blocks[m].is_preplaced for m in members):
                moved_clusters.add(b.cluster_group)
                continue
            slide_set = members
            moved_clusters.add(b.cluster_group)
        elif b.cluster_group > 0:
            continue  # already moved with cluster
        else:
            slide_set = [i]

        # Compute required translation
        min_x = min(pos[j][0] for j in slide_set)
        min_y = min(pos[j][1] for j in slide_set)
        max_x = max(pos[j][0] + pos[j][2] for j in slide_set)
        max_y = max(pos[j][1] + pos[j][3] for j in slide_set)

        dx, dy = 0.0, 0.0
        if _needs_right(bc):
            dx = bbox_x2 - max_x
        if _needs_top(bc):
            dy = bbox_y2 - max_y

        if abs(dx) < 1e-9 and abs(dy) < 1e-9:
            continue

        # Reduce dx/dy to avoid creating NEW overlaps with any other block.
        # We skip pairs that are already overlapping before the slide so that
        # a pre-existing overlap never makes dx/dy flip sign.
        # Iterate the clip loop until dx/dy stabilize: a later clip may expose
        # an earlier obstacle that wasn't tight at the original dx/dy.
        slide_set_idx = set(slide_set)
        if dx != 0.0 or dy != 0.0:
            for _clip_pass in range(len(blocks) + 1):
                prev_dx, prev_dy = dx, dy
                for k in range(len(blocks)):
                    if k in slide_set_idx:
                        continue
                    px, py, pw, ph = pos[k]
                    for j in slide_set:
                        sjx, sjy, sjw, sjh = pos[j]
                        # Skip if already overlapping before the slide
                        pre_ox = min(sjx + sjw, px + pw) - max(sjx, px)
                        pre_oy = min(sjy + sjh, py + ph) - max(sjy, py)
                        if pre_ox > 1e-6 and pre_oy > 1e-6:
                            continue  # pre-existing overlap; don't clip
                        new_x = sjx + dx
                        new_y = sjy + dy
                        ox = min(new_x + sjw, px + pw) - max(new_x, px)
                        oy = min(new_y + sjh, py + ph) - max(new_y, py)
                        if ox > 1e-6 and oy > 1e-6:
                            # Clip dx/dy; clamp to never flip slide direction.
                            if dx > 0:
                                dx = max(0.0, min(dx, px - (sjx + sjw)))
                            elif dx < 0:
                                dx = min(0.0, max(dx, (px + pw) - sjx))
                            if dy > 0:
                                dy = max(0.0, min(dy, py - (sjy + sjh)))
                            elif dy < 0:
                                dy = min(0.0, max(dy, (py + ph) - sjy))
                if abs(dx - prev_dx) < 1e-9 and abs(dy - prev_dy) < 1e-9:
                    break

        if abs(dx) < 1e-9 and abs(dy) < 1e-9:
            continue

        for j in slide_set:
            x, y, w, h = pos[j]
            pos[j] = (x + dx, y + dy, w, h)

    return pos


def enforce_hard(
    positions: List[Tuple[float, float, float, float]],
    blocks: List[BlockInfo],
    area_targets: torch.Tensor,
) -> List[Tuple[float, float, float, float]]:
    """
    Step 8: guarantee all hard constraints (preplaced exact, fixed-shape exact,
    soft area exact, overlap-free).
    """
    pos = list(positions)
    n = len(blocks)

    # 8a: restore preplaced and fixed-shape exact dimensions/positions
    for i, b in enumerate(blocks):
        x, y, w, h = pos[i]
        if b.is_preplaced:
            pos[i] = (b.fixed_x, b.fixed_y, b.w, b.h)
        elif b.is_fixed_shape:
            pos[i] = (x, y, b.w, b.h)

    # 8b: adjust soft-block area to exact (nudge h only, ≤1%)
    for i, b in enumerate(blocks):
        if b.is_preplaced or b.is_fixed_shape:
            continue
        x, y, w, h = pos[i]
        target_area = float(area_targets[i])
        if w > 1e-9:
            h_exact = target_area / w
            rel_err = abs(w * h_exact - target_area) / max(target_area, 1e-9)
            if rel_err <= 0.011:  # stay within 1.1% to be safe
                pos[i] = (x, y, w, h_exact)

    # 8c: resolve overlaps by minimal push (never push preplaced/fixed)
    pos = _resolve_overlaps(pos, blocks)

    return pos


def _resolve_overlaps(
    positions: List[Tuple[float, float, float, float]],
    blocks: List[BlockInfo],
) -> List[Tuple[float, float, float, float]]:
    """
    O(n²) overlap resolution: push the lighter (movable) block away.
    Cluster members are treated as rigid units — all pushed by the same delta.
    """
    pos = list(positions)
    n = len(pos)

    # Build cluster membership: cluster_id -> list of indices
    cluster_members: Dict[int, List[int]] = {}
    for idx, b in enumerate(blocks):
        if b.cluster_group > 0:
            cluster_members.setdefault(b.cluster_group, []).append(idx)

    def _push_block(k: int, dx: float, dy: float) -> None:
        """Push block k and all non-preplaced cluster-mates as a rigid unit.
        Preplaced blocks are never moved (their positions were restored in 8a).
        Clamp delta uniformly so no member is pushed past the origin."""
        gid = blocks[k].cluster_group
        all_members = cluster_members[gid] if gid > 0 else [k]
        # Never move preplaced members — 8a already restored them to exact coords.
        targets = [t for t in all_members if not blocks[t].is_preplaced]
        if not targets:
            return
        if dx < 0:
            min_xt = min(pos[t][0] for t in targets)
            dx = max(dx, -min_xt)
        if dy < 0:
            min_yt = min(pos[t][1] for t in targets)
            dy = max(dy, -min_yt)
        for t in targets:
            xt, yt, wt, ht = pos[t]
            pos[t] = (xt + dx, yt + dy, wt, ht)

    for _ in range(80):  # up to 80 passes for convergence
        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                # Use bounding box of cluster if applicable
                gi = blocks[i].cluster_group
                gj = blocks[j].cluster_group

                # Skip pairs in the same cluster (abutted by construction)
                if gi > 0 and gi == gj:
                    continue

                xi, yi, wi, hi = pos[i]
                xj, yj, wj, hj = pos[j]

                ox = min(xi + wi, xj + wj) - max(xi, xj)
                oy = min(yi + hi, yj + hj) - max(yi, yj)

                if ox <= 1e-9 or oy <= 1e-9:
                    continue  # no overlap

                changed = True
                i_locked = blocks[i].is_preplaced
                j_locked = blocks[j].is_preplaced

                if i_locked and j_locked:
                    continue  # two preplaced: unavoidable, skip

                if ox < oy:
                    push = ox + 1e-6
                    # Push the block that has room to move; avoid pushing into x=0 boundary
                    if i_locked or (not j_locked and xi <= 1e-9):
                        _push_block(j, push, 0.0)
                    elif j_locked or xj > xi:
                        _push_block(i, -push, 0.0)
                    else:
                        _push_block(j, push, 0.0)
                else:
                    push = oy + 1e-6
                    # Push the block that has room to move; avoid pushing into y=0 boundary
                    if i_locked or (not j_locked and yi <= 1e-9):
                        _push_block(j, 0.0, push)
                    elif j_locked or yj > yi:
                        # Want to push i down, but if i is at y=0 and j is fixed,
                        # fall back to pushing i sideways instead.
                        if yi <= 1e-9 and j_locked:
                            # Push i away from j in x: if i's left is ≥ j's left, push right;
                            # otherwise push left (but never past x=0).
                            x_push = ox + 1e-6
                            if xi >= xj:
                                _push_block(i, x_push, 0.0)
                            elif xi > 1e-9:
                                _push_block(i, -x_push, 0.0)
                            else:
                                _push_block(i, x_push, 0.0)  # at x=0, only option is right
                        else:
                            _push_block(i, 0.0, -push)
                    else:
                        _push_block(j, 0.0, push)

        if not changed:
            break

    # --- Final guaranteed escape pass ---
    # The greedy push above can stall or oscillate when an overlap involves an
    # immovable preplaced block.  As a last resort, relocate every movable unit
    # (a whole cluster, or an individual non-preplaced block) that still overlaps
    # something into a clean grid ABOVE the anchor bbox.  Anything placed at
    # y >= anchor_top cannot overlap an anchor (all anchors end at y <= anchor_top),
    # and grid placement keeps the escaped units from overlapping each other.
    pos = _escape_overlaps(pos, blocks, cluster_members)

    return pos


def _escape_overlaps(
    positions: List[Tuple[float, float, float, float]],
    blocks: List[BlockInfo],
    cluster_members: Dict[int, List[int]],
) -> List[Tuple[float, float, float, float]]:
    pos = list(positions)
    n = len(pos)

    def unit_key(i: int):
        gid = blocks[i].cluster_group
        return ("c", gid) if gid > 0 else ("b", i)

    def unit_blocks(key) -> List[int]:
        kind, v = key
        return cluster_members[v] if kind == "c" else [v]

    def unit_movable(key) -> bool:
        if key[0] == "b":
            return not blocks[key[1]].is_preplaced
        return True  # clusters contain no preplaced members (detached upstream)

    # Find movable units that still overlap a block from a different unit.
    floating: set = set()
    for i in range(n):
        ki = unit_key(i)
        xi, yi, wi, hi = pos[i]
        for j in range(i + 1, n):
            kj = unit_key(j)
            if ki == kj:
                continue
            xj, yj, wj, hj = pos[j]
            ox = min(xi + wi, xj + wj) - max(xi, xj)
            oy = min(yi + hi, yj + hj) - max(yi, yj)
            if ox <= 1e-6 or oy <= 1e-6:
                continue
            mi, mj = unit_movable(ki), unit_movable(kj)
            if mi:
                floating.add(ki)
            if mj:
                floating.add(kj)
            # if neither movable (two preplaced) → unavoidable, leave it

    if not floating:
        return pos

    # Anchor bbox = everything not floating.
    anchor_idxs = [i for i in range(n) if unit_key(i) not in floating]
    if anchor_idxs:
        anchor_top = max(pos[i][1] + pos[i][3] for i in anchor_idxs)
        anchor_w = max(pos[i][0] + pos[i][2] for i in anchor_idxs)
    else:
        anchor_top = 0.0
        anchor_w = 0.0

    # Grid-place floating units above the anchor bbox, left→right then up.
    GAP = 1.0
    x_cursor = 0.0
    y_cursor = anchor_top + GAP
    row_h = 0.0
    row_width = max(anchor_w, 1.0)

    for key in sorted(floating):
        idxs = unit_blocks(key)
        umin_x = min(pos[t][0] for t in idxs)
        umin_y = min(pos[t][1] for t in idxs)
        umax_x = max(pos[t][0] + pos[t][2] for t in idxs)
        umax_y = max(pos[t][1] + pos[t][3] for t in idxs)
        uw = umax_x - umin_x
        uh = umax_y - umin_y

        if x_cursor > 0.0 and x_cursor + uw > row_width:
            # new row
            x_cursor = 0.0
            y_cursor += row_h + GAP
            row_h = 0.0

        dx = x_cursor - umin_x
        dy = y_cursor - umin_y
        for t in idxs:
            xt, yt, wt, ht = pos[t]
            pos[t] = (xt + dx, yt + dy, wt, ht)

        x_cursor += uw + GAP
        row_h = max(row_h, uh)

    return pos
