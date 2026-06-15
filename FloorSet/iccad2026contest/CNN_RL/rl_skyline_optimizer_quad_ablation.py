"""Ablation harness: feed analytic quadratic-placer centers through the same
skyline_legalize -> slide_boundary -> enforce_hard -> _detailed_place chain as
rl_skyline_optimizer.py, to sanity-check the transplant reproduces
analytic_legalizer's own ~1.785 score. Not part of the final pipeline."""

import sys
from pathlib import Path

_DL_DIR = Path(__file__).resolve().parent
if str(_DL_DIR) not in sys.path:
    sys.path.insert(0, str(_DL_DIR))

from rl_skyline_optimizer import RLSkylineOptimizer  # noqa: E402


class RLSkylineOptimizerQuadAblation(RLSkylineOptimizer):
    def __init__(self, verbose: bool = False):
        super().__init__(verbose=verbose, center_source="quadratic")


Optimizer = RLSkylineOptimizerQuadAblation
