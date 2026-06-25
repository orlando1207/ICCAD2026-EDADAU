"""
ICCAD 2026 FloorSet Challenge — My Optimizer entry point.
<<<<<<< HEAD
Delegates to analytic_legalizer.MyOptimizer.
"""

from analytic_legalizer import MyOptimizer
=======
Uses SP+SA (Sequence-Pair + Simulated Annealing) for placement.
Falls back to analytic_legalizer_v2 if SP+SA is unavailable.
"""
import sys
import os
from pathlib import Path

# Add sp-engine to path so SPFloorplanner imports work
_ML_ENGINE = Path(__file__).parent / 'sp-engine'
if str(_ML_ENGINE) not in sys.path:
    sys.path.insert(0, str(_ML_ENGINE))

try:
    from floorplanner import SPFloorplanner as _SPFloorplanner
    _SP_AVAILABLE = True
except Exception:
    _SP_AVAILABLE = False

if not _SP_AVAILABLE:
    from analytic_legalizer_v2 import MyOptimizer
else:
    class MyOptimizer:
        """SP+SA floorplanner with size-adaptive time budget."""

        # Budget for largest cases (n>=90); smaller cases get a fraction.
        MAX_BUDGET = 22.0
        N_STARTS = 12

        def __init__(self, verbose=False):
            self._fp = _SPFloorplanner(
                time_budget=self.MAX_BUDGET,
                n_starts=self.N_STARTS,
                enable_rotation=False,
                seed=0,
                use_macros=False,
                verbose=verbose,
            )

        def solve(self, block_count, area_targets, b2b_connectivity,
                  p2b_connectivity, pins_pos, constraints,
                  target_positions=None):
            n = int(block_count)
            # Size-adaptive budget: large cases get the full budget; small
            # cases get proportionally less so they don't stall on trivial SA.
            if n >= 90:
                self._fp.time_budget = self.MAX_BUDGET
            elif n >= 50:
                self._fp.time_budget = max(1.5, self.MAX_BUDGET * 0.45)
            else:
                self._fp.time_budget = max(1.0, self.MAX_BUDGET * 0.18)

            return self._fp.solve(
                n, area_targets, b2b_connectivity, p2b_connectivity,
                pins_pos, constraints, target_positions,
            )
>>>>>>> feature/baseline

__all__ = ["MyOptimizer"]
