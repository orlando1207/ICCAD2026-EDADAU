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
        WL_MODELS = ("quadratic", "lse") if block_count >= 116 else ("quadratic",)
        SKYLINE_CONFIGS = (
            # (lambda, width-selection HPWL exponent)
            (0.20, 0.00),
            (0.45, 0.00),
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
                for lam, hpwl_weight in SKYLINE_CONFIGS:
                    _pos, _ = skyline_legalize(
                        blocks, super_blocks, cluster_groups, _cx, _cy, area_targets,
                        lam=lam,
                        b2b_connectivity=b2b_connectivity,
                        p2b_connectivity=p2b_connectivity,
                        pins_pos=pins_pos,
                        hpwl_weight=hpwl_weight,
                    )
                    _pos = slide_boundary(_pos, blocks, super_blocks, cluster_groups)
                    _pos = enforce_hard(_pos, blocks, area_targets)

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
