"""
Legalize Optimization — Item A: targeted TOP/RIGHT boundary reshape.

70% of our soft-violation tax (e^{2V}) is boundary: forced blocks that should
touch a bbox edge land short of it (median miss ~10% of chip width). For the
TOP/RIGHT edges this is reshape-fixable: a soft block that should touch y_max can
be grown taller (area-preserving ⇒ it gets narrower) up into the empty space
above it until its top reaches y_max — fixing the violation without growing the
bbox and without HARD-area or overlap risk. RIGHT is symmetric (grow wider to
x_max).

Overlap-safe by construction:
  - growing taller ⇒ narrower, so the horizontal span is a subset of the old one
    (no new side overlap);
  - the only new area is the rectangle between the old top and y_max within the
    (narrower) x-span — committed only if that rectangle is verified empty;
  - growth is capped at the *current* bbox edge, so the bbox never grows.

Reshapeable = soft and not fixed/preplaced. MIB/cluster members are skipped here
(they need joint reshaping — Item B). Imports nothing from analytic_legalizer
except the boundary-code constants.
"""

from __future__ import annotations

import math
from typing import List, Tuple

from legalizer.constraints import (
    BOUND_LEFT, BOUND_RIGHT, BOUND_TOP, BOUND_BOTTOM,
)

AR_MAX = 3.0
EPS = 1e-6


def _region_free(pos, i, x0, y0, x1, y1) -> bool:
    """True if rectangle [x0,x1]×[y0,y1] is free of every block except i."""
    for j, (xj, yj, wj, hj) in enumerate(pos):
        if j == i:
            continue
        if (min(x1, xj + wj) - max(x0, xj) > EPS and
                min(y1, yj + hj) - max(y0, yj) > EPS):
            return False
    return True


def boundary_reshape(positions, blocks, constraints, area_targets,
                     ar_max: float = AR_MAX) -> List[Tuple[float, float, float, float]]:
    """Grow reshapeable TOP/RIGHT-forced blocks to reach the bbox edge they must
    touch (area-preserving, overlap-safe). Returns new positions; never grows the
    bbox, never creates overlap, never touches fixed/preplaced/MIB/cluster blocks."""
    pos = [tuple(map(float, p)) for p in positions]
    n = len(pos)
    x_min = min(p[0] for p in pos)
    y_min = min(p[1] for p in pos)
    x_max = max(p[0] + p[2] for p in pos)
    y_max = max(p[1] + p[3] for p in pos)

    for i in range(n):
        code = int(constraints[i, 4]) if constraints.dim() > 1 else 0
        if code == 0:
            continue
        b = blocks[i]
        if b.is_fixed_shape or b.is_preplaced or b.mib_group != 0 or b.cluster_group != 0:
            continue                                   # rigid / joint — Item B
        x, y, w, h = pos[i]
        area = max(float(area_targets[i]), 1e-6)

        # Only single-edge TOP or RIGHT (skip corners / multi-bit for Item A).
        want_top = (code & BOUND_TOP) and not (code & (BOUND_LEFT | BOUND_RIGHT | BOUND_BOTTOM))
        want_right = (code & BOUND_RIGHT) and not (code & (BOUND_LEFT | BOUND_TOP | BOUND_BOTTOM))

        if want_top and abs((y + h) - y_max) > EPS and y_max > y + h:
            new_h = y_max - y
            new_w = area / new_h
            if new_h / new_w <= ar_max + 1e-9 and new_w <= w + EPS:
                # new area added: [x, x+new_w] × [y+h, y_max]
                if _region_free(pos, i, x, y + h, x + new_w, y_max):
                    pos[i] = (x, y, new_w, new_h)
                    b.w, b.h = new_w, new_h
                    continue

        if want_right and abs((x + w) - x_max) > EPS and x_max > x + w:
            new_w = x_max - x
            new_h = area / new_w
            if new_w / new_h <= ar_max + 1e-9 and new_h <= h + EPS:
                if _region_free(pos, i, x + w, y, x_max, y + new_h):
                    pos[i] = (x, y, new_w, new_h)
                    b.w, b.h = new_w, new_h

    return pos
