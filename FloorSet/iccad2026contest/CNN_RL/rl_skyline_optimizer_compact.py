"""Experiment: RLSkylineOptimizer + an extra whitespace-compaction pass
(analytic_legalizer.topology.compact) after skyline_legalize, to test whether
squeezing the residual gaps in the (looser) RL layout is a net win or whether
the HPWL cost of moving blocks outweighs the area gain (as it did for analytic).
"""
import sys
from pathlib import Path
_DL_DIR = Path(__file__).resolve().parent
if str(_DL_DIR) not in sys.path:
    sys.path.insert(0, str(_DL_DIR))
from rl_skyline_optimizer import RLSkylineOptimizer  # noqa: E402


class RLSkylineOptimizerCompact(RLSkylineOptimizer):
    def __init__(self, verbose: bool = False):
        super().__init__(verbose=verbose, center_source="model", compact_pass=True)


Optimizer = RLSkylineOptimizerCompact
