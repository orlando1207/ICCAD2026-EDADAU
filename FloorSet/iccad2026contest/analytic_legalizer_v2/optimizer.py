"""
Analytic + Legalization Floorplanner — MyOptimizer implementation.
Orchestrates Steps 0–8 per the plan.
"""

import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from iccad2026_evaluate import (
    FloorplanOptimizer, calculate_hpwl_b2b, calculate_hpwl_p2b,
)

from .constraints import (
    parse_and_init, prepack_clusters,
    slide_boundary, enforce_hard,
)
from .quadratic_placer import analytic_place
from .skyline_legalizer import skyline_legalize


class MyOptimizer(FloorplanOptimizer):
    """
    Analytic (quadratic+spread) + longest-path legalization floorplanner.
    Pipeline: parse → MIB unify → cluster super-blocks → analytic place →
              topology extract → longest-path pack → soft-block shaping →
              boundary slide → hard enforcement.
    """

    def __init__(self, verbose: bool = False):
        super().__init__(verbose)

    def solve(
        self,
        block_count: int,
        area_targets: torch.Tensor,
        b2b_connectivity: torch.Tensor,
        p2b_connectivity: torch.Tensor,
        pins_pos: torch.Tensor,
        constraints: torch.Tensor,
        target_positions: torch.Tensor = None,
    ) -> List[Tuple[float, float, float, float]]:

        # ------------------------------------------------------------------
        # Step 0 + 1: parse constraints, classify blocks, unify MIB shapes
        # ------------------------------------------------------------------
        blocks, mib_groups, cluster_groups = parse_and_init(
            block_count, area_targets, constraints, target_positions
        )

        # ------------------------------------------------------------------
        # Step 2: pre-pack clusters into rigid super-blocks
        # ------------------------------------------------------------------
        super_blocks = prepack_clusters(blocks, cluster_groups)

        # ------------------------------------------------------------------
        # Steps 3–8: analytic place → skyline legalize → slide → enforce, run
        # N_STARTS times with different analytic seeds. Select the start with the
        # smallest area·HPWL proxy: HPWL is now the dominant cost term, so the old
        # min-area / area·e^(2·boundary) selectors actually picked worse layouts.
        # An area·HPWL selector matches the oracle (best actual cost) on the
        # dominant cases. HPWL is computed from connectivity (no baseline needed).
        # Seed 0 = no noise (deterministic); seeds 1+ add perturbation.
        # ------------------------------------------------------------------
        N_STARTS  = 1      # single deterministic seed (no noise); fast baseline.
        NOISE_STD = 0.12   # used only if N_STARTS > 1 (raise for multistart)
        # The official scorer now weights cases by e^(n/12), so sub-116 cases
        # are ~28% of the grade (and are empirically the worst-scoring ones).
        # They are also cheap to legalize, so run the full search at every size.
        RICH_SEARCH = block_count >= 1
        # "irls" minimizes the contest's true edge-based L1 wirelength exactly
        # (iteratively-reweighted least squares); "lse" is the older smooth-L1
        # approximation. Both are kept as candidate generators — selection by the
        # area·HPWL proxy picks the best legalized result per case.
        WL_MODELS = ("quadratic", "lse", "irls") if RICH_SEARCH else ("quadratic",)
        if RICH_SEARCH:
            SKYLINE_CONFIGS = (
                # (lambda, width-selection HPWL exponent, net weight, orders, width refine)
                (0.20, 0.00, 0.00, ("analytic",), False),
                (0.45, 0.00, 0.00, ("analytic",), False),
                (0.30, 0.00, 0.18, ("analytic",), False),
                (0.30, 0.00, 0.00, ("analytic", "net"), False),
                (0.30, 0.00, 0.12, ("analytic", "net"), False),
                (0.20, 0.00, 0.12, ("analytic",), False),
                (0.45, 0.00, 0.08, ("cluster",), False),
                (0.30, 0.00, 0.35, ("cluster",), False),
            )
        else:
            SKYLINE_CONFIGS = (
                (0.20, 0.00, 0.00, ("analytic",), False),
                (0.45, 0.00, 0.00, ("analytic",), False),
            )

        best_positions = None
        best_proxy = float('inf')

        for _start in range(N_STARTS):
            seed = _start  # seed 0 → no noise (baseline)

            for wl_model in WL_MODELS:
                _cx, _cy = analytic_place(
                    blocks, super_blocks, cluster_groups,
                    b2b_connectivity, p2b_connectivity, pins_pos,
                    seed=seed, noise_std=(NOISE_STD if seed > 0 else 0.0),
                    wl_model=wl_model,
                )
                for lam, hpwl_weight, net_weight, order_modes, refine_widths in SKYLINE_CONFIGS:
                    _pos, _ = skyline_legalize(
                        blocks, super_blocks, cluster_groups, _cx, _cy, area_targets,
                        lam=lam,
                        b2b_connectivity=b2b_connectivity,
                        p2b_connectivity=p2b_connectivity,
                        pins_pos=pins_pos,
                        hpwl_weight=hpwl_weight,
                        net_weight=net_weight,
                        order_modes=list(order_modes),
                        refine_widths=refine_widths,
                    )
                    _pos = slide_boundary(_pos, blocks, super_blocks, cluster_groups)
                    _pos = enforce_hard(_pos, blocks, area_targets)
                    _pos = _local_slide_refine(
                        _pos, constraints, b2b_connectivity, p2b_connectivity, pins_pos)

                    _x2 = max(p[0] + p[2] for p in _pos)
                    _y2 = max(p[1] + p[3] for p in _pos)
                    _hpwl = (calculate_hpwl_b2b(_pos, b2b_connectivity)
                             + calculate_hpwl_p2b(_pos, p2b_connectivity, pins_pos))
                    _soft = _soft_violation_proxy(_pos, constraints)
                    _proxy = ((_x2 * _y2)
                              * (_hpwl if _hpwl > 1e-9 else 1.0)
                              * np.exp(3.0 * _soft / max(block_count, 1)))
                    if _proxy < best_proxy:
                        best_proxy = _proxy
                        best_positions = _pos

        return best_positions


def _soft_violation_proxy(
    positions: List[Tuple[float, float, float, float]],
    constraints: torch.Tensor,
) -> int:
    """Cheap final-candidate selector for soft violations.

    This mirrors the two soft terms the optimizer can influence after hard
    enforcement: boundary misses and cluster connected components.  MIB is
    already unified upstream and is not needed for tie-breaking.
    """
    n = len(positions)
    if constraints is None or len(constraints) < n or constraints.shape[1] < 5:
        return 0

    soft = 0
    x_min = min(p[0] for p in positions)
    y_min = min(p[1] for p in positions)
    x_max = max(p[0] + p[2] for p in positions)
    y_max = max(p[1] + p[3] for p in positions)
    eps = 1e-6

    for i in range(n):
        code = int(constraints[i, 4].item())
        if code == 0:
            continue
        x, y, w, h = positions[i]
        if (code & 1) and abs(x - x_min) > eps:
            soft += 1
            continue
        if (code & 2) and abs((x + w) - x_max) > eps:
            soft += 1
            continue
        if (code & 4) and abs((y + h) - y_max) > eps:
            soft += 1
            continue
        if (code & 8) and abs(y - y_min) > eps:
            soft += 1

    cluster_ids = sorted({
        int(constraints[i, 3].item())
        for i in range(n)
        if int(constraints[i, 3].item()) > 0
    })
    for gid in cluster_ids:
        members = [i for i in range(n) if int(constraints[i, 3].item()) == gid]
        if len(members) <= 1:
            continue
        parent = {i: i for i in members}

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for ai, a in enumerate(members):
            ax, ay, aw, ah = positions[a]
            for b in members[ai + 1:]:
                bx, by, bw, bh = positions[b]
                y_ov = min(ay + ah, by + bh) - max(ay, by)
                x_ov = min(ax + aw, bx + bw) - max(ax, bx)
                x_touch = abs((ax + aw) - bx) <= eps or abs((bx + bw) - ax) <= eps
                y_touch = abs((ay + ah) - by) <= eps or abs((by + bh) - ay) <= eps
                if (x_touch and y_ov > eps) or (y_touch and x_ov > eps):
                    union(a, b)

        comps = len({find(i) for i in members})
        soft += max(0, comps - 1)

    return soft


def _local_slide_refine(
    positions: List[Tuple[float, float, float, float]],
    constraints: torch.Tensor,
    b2b_connectivity: torch.Tensor,
    p2b_connectivity: torch.Tensor,
    pins_pos: torch.Tensor,
) -> List[Tuple[float, float, float, float]]:
    """Conservative post-legalization compaction for free blocks.

    Only unconstrained, non-cluster blocks are considered, and only left/down
    slides are tested.  A move is accepted if the same final selector proxy
    improves and no overlap is introduced.  This targets residual right/top
    frontier whitespace without touching hard-constrained blocks.
    """
    pos = list(positions)
    n = len(pos)
    if n == 0 or constraints is None or len(constraints) < n:
        return pos

    def hpwl(cur):
        return (calculate_hpwl_b2b(cur, b2b_connectivity)
                + calculate_hpwl_p2b(cur, p2b_connectivity, pins_pos))

    def proxy(cur):
        x2 = max(p[0] + p[2] for p in cur)
        y2 = max(p[1] + p[3] for p in cur)
        h = hpwl(cur)
        s = _soft_violation_proxy(cur, constraints)
        return (x2 * y2) * (h if h > 1e-9 else 1.0) * np.exp(3.0 * s / max(n, 1))

    def overlaps_any(i, x, y):
        _, _, w, h = pos[i]
        for j, (jx, jy, jw, jh) in enumerate(pos):
            if i == j:
                continue
            if (min(x + w, jx + jw) - max(x, jx) > 1e-6 and
                    min(y + h, jy + jh) - max(y, jy) > 1e-6):
                return True
        return False

    movable = []
    for i in range(n):
        c = constraints[i]
        is_fixed = bool(c[0].item() > 0)
        is_preplaced = bool(c[1].item() > 0)
        in_cluster = int(c[3].item()) > 0
        has_boundary = int(c[4].item()) > 0
        if not (is_fixed or is_preplaced or in_cluster or has_boundary):
            movable.append(i)

    best_proxy = proxy(pos)
    for _ in range(3):
        x2 = max(p[0] + p[2] for p in pos)
        y2 = max(p[1] + p[3] for p in pos)
        frontier = [
            i for i in movable
            if pos[i][0] + pos[i][2] > x2 - 1e-6
            or pos[i][1] + pos[i][3] > y2 - 1e-6
        ]
        frontier.sort(key=lambda i: -pos[i][2] * pos[i][3])
        moved = False

        for i in frontier:
            x, y, w, h = pos[i]
            candidates = []

            # Slide left until blocked by an overlapping vertical span.
            new_x = 0.0
            for j, (jx, jy, jw, jh) in enumerate(pos):
                if i == j:
                    continue
                y_ov = min(y + h, jy + jh) - max(y, jy)
                if y_ov > 1e-6 and jx + jw <= x + 1e-6:
                    new_x = max(new_x, jx + jw)
            if new_x < x - 1e-6:
                candidates.append((new_x, y))

            # Slide down until blocked by an overlapping horizontal span.
            new_y = 0.0
            for j, (jx, jy, jw, jh) in enumerate(pos):
                if i == j:
                    continue
                x_ov = min(x + w, jx + jw) - max(x, jx)
                if x_ov > 1e-6 and jy + jh <= y + 1e-6:
                    new_y = max(new_y, jy + jh)
            if new_y < y - 1e-6:
                candidates.append((x, new_y))

            if new_x < x - 1e-6 and new_y < y - 1e-6:
                candidates.append((new_x, new_y))

            for nx, ny in candidates:
                if overlaps_any(i, nx, ny):
                    continue
                old = pos[i]
                pos[i] = (nx, ny, w, h)
                cand_proxy = proxy(pos)
                if cand_proxy < best_proxy - 1e-6:
                    best_proxy = cand_proxy
                    moved = True
                    break
                pos[i] = old
            if moved:
                break

        if not moved:
            break

    return pos
