"""
Fast, vectorized cost evaluation for the SP floorplanner.

Mirrors the official `evaluate_solution` semantics (tolerances, boundary touch,
HPWL via centroids, bbox area) but uses numpy arrays and never calls shapely in
the inner loop. Grouping is guaranteed by rigid macros (so V_grouping=0 by
construction); shapely is used only for a final audit in the harness.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

EPS = 1e-6          # overlap / boundary-touch tolerance (matches evaluator)
AREA_TOL = 0.01     # soft-block area tolerance


@dataclass
class Nets:
    """Pre-extracted, numpy-vectorized connectivity for one case."""
    b2b_i: np.ndarray   # (E,) int
    b2b_j: np.ndarray   # (E,) int
    b2b_w: np.ndarray   # (E,) float
    p2b_pin: np.ndarray  # (F,) int
    p2b_blk: np.ndarray  # (F,) int
    p2b_w: np.ndarray   # (F,) float
    pins: np.ndarray    # (P,2) float


def build_nets(b2b, p2b, pins_pos) -> Nets:
    """Convert the contest tensors into flat numpy arrays (drop -1 padding)."""
    def _arr(t):
        return t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)

    b2b = _arr(b2b) if b2b is not None else np.zeros((0, 3))
    p2b = _arr(p2b) if p2b is not None else np.zeros((0, 3))
    pins = _arr(pins_pos) if pins_pos is not None else np.zeros((0, 2))

    if b2b.size:
        m = b2b[:, 0] != -1
        b2b = b2b[m]
    if p2b.size:
        m = p2b[:, 0] != -1
        p2b = p2b[m]

    return Nets(
        b2b_i=b2b[:, 0].astype(np.int64) if b2b.size else np.zeros(0, np.int64),
        b2b_j=b2b[:, 1].astype(np.int64) if b2b.size else np.zeros(0, np.int64),
        b2b_w=b2b[:, 2].astype(np.float64) if b2b.size else np.zeros(0),
        p2b_pin=p2b[:, 0].astype(np.int64) if p2b.size else np.zeros(0, np.int64),
        p2b_blk=p2b[:, 1].astype(np.int64) if p2b.size else np.zeros(0, np.int64),
        p2b_w=p2b[:, 2].astype(np.float64) if p2b.size else np.zeros(0),
        pins=pins.astype(np.float64) if pins.size else np.zeros((0, 2)),
    )


def hpwl(cx: np.ndarray, cy: np.ndarray, nets: Nets) -> float:
    """Weighted Manhattan HPWL over block centroids + pin->block edges."""
    total = 0.0
    if nets.b2b_i.size:
        dx = np.abs(cx[nets.b2b_i] - cx[nets.b2b_j])
        dy = np.abs(cy[nets.b2b_i] - cy[nets.b2b_j])
        total += float(np.sum(nets.b2b_w * (dx + dy)))
    if nets.p2b_blk.size:
        px = nets.pins[nets.p2b_pin, 0]
        py = nets.pins[nets.p2b_pin, 1]
        dx = np.abs(px - cx[nets.p2b_blk])
        dy = np.abs(py - cy[nets.p2b_blk])
        total += float(np.sum(nets.p2b_w * (dx + dy)))
    return total


def bbox_area(bx, by, bw, bh):
    x0 = float(bx.min()); y0 = float(by.min())
    x1 = float((bx + bw).max()); y1 = float((by + bh).max())
    return (x1 - x0) * (y1 - y0), (x0, y0, x1, y1)


def boundary_violations(bx, by, bw, bh, codes: np.ndarray, bbox) -> int:
    """Count blocks not touching their required bbox edge/corner (eps=1e-6)."""
    x0, y0, x1, y1 = bbox
    v = 0
    idx = np.nonzero(codes)[0]
    for i in idx:
        c = int(codes[i])
        ok = True
        if c & 1 and abs(bx[i] - x0) >= EPS:           # left
            ok = False
        if ok and c & 2 and abs(bx[i] + bw[i] - x1) >= EPS:  # right
            ok = False
        if ok and c & 4 and abs(by[i] + bh[i] - y1) >= EPS:  # top
            ok = False
        if ok and c & 8 and abs(by[i] - y0) >= EPS:    # bottom
            ok = False
        if not ok:
            v += 1
    return v


def count_overlaps(bx, by, bw, bh) -> int:
    """O(n^2) exact overlap count (intersection-area > 0 with eps), for audits."""
    n = len(bx)
    x1 = bx + bw; y1 = by + bh
    v = 0
    for i in range(n):
        ox = np.minimum(x1[i], x1) - np.maximum(bx[i], bx)
        oy = np.minimum(y1[i], y1) - np.maximum(by[i], by)
        hit = (ox > EPS) & (oy > EPS)
        hit[i] = False
        v += int(np.count_nonzero(hit[i + 1:]))
    return v


@dataclass
class Score:
    cost: float
    hpwl_gap: float
    area_gap: float
    v_rel: float
    v_boundary: int
    feasible_geom: bool   # overlap-free + preplaced-on-target (dims handled by construction)


def score_layout(bx, by, bw, bh, nets, codes, n_soft,
                 hpwl_base, area_base,
                 alpha=0.5, beta=2.0,
                 v_group=0, v_mib=0,
                 runtime_factor=1.0) -> Score:
    """Compute the exact contest cost for a fully-expanded block layout.

    Assumes feasibility of dimensions/area is guaranteed by construction (fixed
    dims). `feasible_geom` reflects only overlap (caller checks) — here we always
    compute the quality cost; the caller decides feasibility separately.
    """
    cx = bx + bw / 2.0
    cy = by + bh / 2.0
    hp = hpwl(cx, cy, nets)
    area, box = bbox_area(bx, by, bw, bh)
    hgap = max(0.0, (hp - hpwl_base) / max(hpwl_base, 1e-6))
    agap = max(0.0, (area - area_base) / max(area_base, 1e-6))
    vb = boundary_violations(bx, by, bw, bh, codes, box)
    v_rel = (vb + v_group + v_mib) / max(n_soft, 1)
    q = 1.0 + alpha * (hgap + agap)
    cost = q * np.exp(beta * v_rel) * max(0.7, runtime_factor ** 0.3)
    return Score(cost=cost, hpwl_gap=hgap, area_gap=agap, v_rel=v_rel,
                 v_boundary=vb, feasible_geom=True)
