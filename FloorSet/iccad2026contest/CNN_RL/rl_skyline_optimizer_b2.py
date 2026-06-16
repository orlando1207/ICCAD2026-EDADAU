"""A/B wrapper: default RL skyline optimizer + Phase 15/B2 critical-path shaping
(b2_pass=True). Same checkpoint/pipeline as rl_skyline_optimizer.py."""
import sys
from pathlib import Path
_DL_DIR = Path(__file__).resolve().parent
if str(_DL_DIR) not in sys.path:
    sys.path.insert(0, str(_DL_DIR))
from rl_skyline_optimizer import RLSkylineOptimizer


class RLSkylineB2Optimizer(RLSkylineOptimizer):
    def __init__(self, verbose: bool = False, checkpoint=None):
        super().__init__(verbose=verbose, checkpoint=checkpoint, b2_pass=True)


Optimizer = RLSkylineB2Optimizer
