"""
Phase 5 — aspect-ratio action space (VERSION_B_RL_PLACER.md §5 Phase 5).

Soft blocks need a shape, not just a position. We discretise the aspect ratio
(w/h) into a small set of log-spaced buckets so the policy picks both a cell
and a shape. Area stays exact for any bucket (handled in env._block_dims).
"""

from __future__ import annotations

import math
from typing import List

# log-spaced; index 2 (==1.0) is the square fallback used by Phase 0-4.5
ASPECT_BUCKETS: List[float] = [0.25, 0.5, 1.0, 2.0, 4.0]
N_ASPECT = len(ASPECT_BUCKETS)


def gt_aspect_bucket(w: float, h: float) -> int:
    """Nearest aspect bucket (in log space) to a ground-truth (w, h)."""
    ar = max(w, 1e-6) / max(h, 1e-6)
    log_ar = math.log(ar)
    return min(range(N_ASPECT),
               key=lambda i: abs(log_ar - math.log(ASPECT_BUCKETS[i])))
