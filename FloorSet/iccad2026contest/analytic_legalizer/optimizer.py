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
from .skyline_legalizer import skyline_legalize, _detailed_place


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

        best_positions = None
        best_proxy = float('inf')

        for _start in range(N_STARTS):
            seed = _start  # seed 0 → no noise (baseline)

            _cx, _cy = analytic_place(
                blocks, super_blocks, cluster_groups,
                b2b_connectivity, p2b_connectivity, pins_pos,
                seed=seed, noise_std=(NOISE_STD if seed > 0 else 0.0),
            )
            _pos, _ = skyline_legalize(
                blocks, super_blocks, cluster_groups, _cx, _cy, area_targets,
            )
            _pos = slide_boundary(_pos, blocks, super_blocks, cluster_groups)
            _pos = enforce_hard(_pos, blocks, area_targets)
            # HPWL detailed placement: slide free blocks toward their wirelength
            # optimum, clipped to stay overlap-free (legality preserved).
            _pos = _detailed_place(
                _pos, blocks, b2b_connectivity, p2b_connectivity, pins_pos,
            )

            if N_STARTS == 1:
                return _pos  # single candidate — skip the (expensive) selection proxy

            _x2 = max(p[0] + p[2] for p in _pos)
            _y2 = max(p[1] + p[3] for p in _pos)
            _hpwl = (calculate_hpwl_b2b(_pos, b2b_connectivity)
                     + calculate_hpwl_p2b(_pos, p2b_connectivity, pins_pos))
            _proxy = (_x2 * _y2) * (_hpwl if _hpwl > 1e-9 else 1.0)
            if _proxy < best_proxy:
                best_proxy = _proxy
                best_positions = _pos

        return best_positions
