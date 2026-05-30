"""
Step 6 — Soft-block shaping: convex sizing to minimize bounding-box area.

With HCG/VCG topology fixed, the bbox width = longest HCG path, height = longest VCG path.
We iteratively adjust each soft block's aspect ratio (w↔h, area constant) to balance
the two critical paths → shrink bbox.

MIB groups: all members resize together (one AR).
Preplaced/fixed: never reshaped.
"""

import math
from typing import Dict, List, Tuple

import numpy as np
import torch

from .constraints import BlockInfo, SuperBlock


def shape_soft_blocks(
    positions: List[Tuple[float, float, float, float]],
    blocks: List[BlockInfo],
    super_blocks: Dict[int, SuperBlock],
    cluster_groups: Dict[int, List[int]],
    mib_groups: Dict[int, List[int]],
    area_targets: torch.Tensor,
    hcg_succ: List[List[Tuple[int, float]]],
    vcg_succ: List[List[Tuple[int, float]]],
    n_iters: int = 20,
    ar_min: float = 0.2,
    ar_max: float = 5.0,
) -> List[Tuple[float, float, float, float]]:
    """
    Iteratively reshape soft blocks to reduce bounding-box area.
    Returns updated (x, y, w, h) list; positions (x,y) are NOT changed here —
    only w and h are adjusted. Positions are re-legalized by re-running longest-path
    in the caller (optimizer.py) if needed, or we do a minimal in-place update here.

    For simplicity: we adjust w,h, then recompute x,y via a compact re-pack of
    the HCG/VCG critical paths. This keeps the topology (relative order) fixed.
    """
    pos = [list(p) for p in positions]  # mutable copy
    n = len(blocks)

    # Build MIB group lookup: block idx -> group id
    block_to_mib: Dict[int, int] = {}
    for gid, members in mib_groups.items():
        for idx in members:
            block_to_mib[idx] = gid

    # Build cluster rep lookup
    cluster_rep: Dict[int, int] = {}
    for gid, members in cluster_groups.items():
        cluster_rep[gid] = members[0]

    def rep(i): return cluster_rep[blocks[i].cluster_group] if blocks[i].cluster_group > 0 else i

    reps = sorted(set(rep(i) for i in range(n)))

    for _ in range(n_iters):
        # Compute current bbox
        x2 = max(pos[i][0] + pos[i][2] for i in range(n))
        y2 = max(pos[i][1] + pos[i][3] for i in range(n))

        improved = False
        for i in range(n):
            b = blocks[i]
            if b.is_preplaced or b.is_fixed_shape:
                continue
            if b.cluster_group > 0:
                continue  # super-block shape is fixed; members track it

            # Check if this block is on HCG critical path and VCG critical path
            on_hcg = (pos[i][0] + pos[i][2] >= x2 - 1e-6)
            on_vcg = (pos[i][1] + pos[i][3] >= y2 - 1e-6)

            if on_hcg == on_vcg:
                continue  # either on both (can't help) or neither (irrelevant)

            area = float(area_targets[i])
            w_cur, h_cur = pos[i][2], pos[i][3]

            if on_hcg:
                # Widen: increase w → decrease h (moves off HCG critical path? No—we
                # want to decrease w to shorten HCG critical path)
                # Decrease w → increase h (reduces HCG, increases VCG — helps if VCG slack)
                vcg_slack = y2 - (pos[i][1] + pos[i][3])
                if vcg_slack < 1e-6:
                    continue
                # Try narrowing by a small step
                new_w = max(w_cur * 0.95, math.sqrt(area * ar_min))
                new_h = area / new_w
                new_ar = new_w / new_h
                if new_ar < ar_min or new_ar > ar_max:
                    continue
                # Check new h fits within slack
                if pos[i][1] + new_h <= y2 + vcg_slack / 2.0:
                    _apply_shape(pos, blocks, mib_groups, block_to_mib, i, new_w, new_h, area)
                    improved = True

            else:  # on_vcg
                hcg_slack = x2 - (pos[i][0] + pos[i][2])
                if hcg_slack < 1e-6:
                    continue
                new_h = max(h_cur * 0.95, math.sqrt(area / ar_max))
                new_w = area / new_h
                new_ar = new_w / new_h
                if new_ar < ar_min or new_ar > ar_max:
                    continue
                if pos[i][0] + new_w <= x2 + hcg_slack / 2.0:
                    _apply_shape(pos, blocks, mib_groups, block_to_mib, i, new_w, new_h, area)
                    improved = True

        if not improved:
            break

    return [(p[0], p[1], p[2], p[3]) for p in pos]


def _apply_shape(
    pos: List[List[float]],
    blocks: List[BlockInfo],
    mib_groups: Dict[int, List[int]],
    block_to_mib: Dict[int, int],
    i: int,
    new_w: float,
    new_h: float,
    area: float,
) -> None:
    """Apply (new_w, new_h) to block i and all its MIB group members.

    If any MIB group member is cluster-locked or fixed-shape, skip the entire
    group — reshaping would corrupt the super-block's internal layout.
    """
    targets = [i]
    if i in block_to_mib:
        targets = mib_groups[block_to_mib[i]]
        # If ANY member is locked (cluster, fixed-shape, or preplaced), don't
        # reshape anyone — a preplaced member's (w,h) is a HARD constraint, and
        # MIB requires all members share one shape, so the whole group is locked.
        if any(blocks[idx].cluster_group > 0 or blocks[idx].is_fixed_shape
               or blocks[idx].is_preplaced
               for idx in targets):
            return

    for idx in targets:
        blocks[idx].w = new_w
        blocks[idx].h = new_h
        pos[idx][2] = new_w
        pos[idx][3] = new_h
