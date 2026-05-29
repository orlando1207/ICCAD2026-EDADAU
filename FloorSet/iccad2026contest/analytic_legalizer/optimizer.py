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

from iccad2026_evaluate import FloorplanOptimizer

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
        # Steps 3–6c: analytic place → topology → pack → shape → compact.
        # Run N_STARTS times with different spreading seeds; keep the layout
        # with the smallest bounding-box area (proxy for area_gap).
        # Seed 0 = no noise (deterministic baseline); seeds 1+ add perturbation.
        # ------------------------------------------------------------------
        N_STARTS  = 1
        NOISE_STD = 0.12   # fraction of chip_side added as Gaussian noise

        best_positions = None
        best_bbox_area = float('inf')

        for _start in range(N_STARTS):
            seed = _start  # seed 0 → no noise (baseline)

            # Step 3: analytic global placement
            _cx, _cy = analytic_place(
                blocks, super_blocks, cluster_groups,
                b2b_connectivity, p2b_connectivity, pins_pos,
                seed=seed, noise_std=(NOISE_STD if seed > 0 else 0.0),
            )

            # Steps 4–6: skyline strip-packing legalization (replaces topology /
            # longest-path / shaping / compact). Returns a cost-proxy score used
            # to select across analytic seeds.
            _pos, _score = skyline_legalize(
                blocks, super_blocks, cluster_groups, _cx, _cy, area_targets,
            )
            if _score < best_bbox_area:
                best_bbox_area = _score
                best_positions = _pos

        positions = best_positions

        # ------------------------------------------------------------------
        # Step 7: boundary slide (RIGHT/TOP blocks to bbox edge)
        # ------------------------------------------------------------------
        positions = slide_boundary(
            positions, blocks, super_blocks, cluster_groups
        )

        # ------------------------------------------------------------------
        # Step 8: hard constraint enforcement (overlap, area, preplaced)
        # ------------------------------------------------------------------
        positions = enforce_hard(positions, blocks, area_targets)

        return positions
