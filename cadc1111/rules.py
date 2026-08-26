"""Authoritative restatement of the ICCAD-2026 FloorSet contest rules.

Single source of truth for "is this layout legal, and what does it score".
Everything here mirrors `iccad2026_evaluate.py` *exactly* -- same epsilons,
same strict/non-strict comparisons, same tie-breaks.  Where the PDF and the
evaluator disagree, the evaluator wins (it is what grades the submission).

------------------------------------------------------------------ input
A case is (block_count k, area targets a[k], constraints C[k,5], b2b, p2b,
pins, target_positions[k,4]).

`C` columns -- this layout is *positional*, there are no names in the tensor:

    C[:,0]  fixed      != 0  -> block's (w,h) is immutable
    C[:,1]  preplaced  != 0  -> block's (x,y,w,h) are all immutable
    C[:,2]  mib_id     0 = none, else group id 1..max   (identical shapes)
    C[:,3]  cluster_id 0 = none, else group id 1..max   (must abut)
    C[:,4]  boundary   bitmask, 0 = none
                       1 = left, 2 = right, 4 = top, 8 = bottom
                       corners are sums: 5 = top-left, 6 = top-right,
                       9 = bottom-left, 10 = bottom-right

`target_positions[i]` is (x,y,w,h) supplied by the evaluator:
  * preplaced block -> all four entries are meaningful and immutable
  * fixed block     -> only (w,h) meaningful, (x,y) are -1
  * anything else   -> all -1

------------------------------------------------------- HARD constraints
Any single violation => the whole test case scores M = 10.0.  There is no
partial credit; a hard violation is ~9x worse than the worst legal layout.

H1 OVERLAP        for every pair i<j:
                    ox = min(x_i+w_i, x_j+w_j) - max(x_i, x_j)
                    oy = min(y_i+h_i, y_j+h_j) - max(y_i, y_j)
                  violation iff  ox > 1e-6 AND oy > 1e-6.
                  Touching (ox or oy == 0) is legal.  Note this is an
                  *absolute* epsilon on a canvas whose coordinates run to
                  O(1e2..1e3) -- it does not scale with the design.

H2 AREA TOLERANCE for every block that is neither fixed nor preplaced,
                  and whose target area a_i is > 0 and != -1:
                    |w_i*h_i - a_i| / a_i  <= 0.01
                  Strict: the check is `diff > 0.01` -> violation, so
                  exactly 0.01 passes.  Aspect ratio is unconstrained.

H3 FIXED SHAPE    for every block with C[i,0] != 0:
                    |w_i - w_in| <= 1e-4  and  |h_i - h_in| <= 1e-4
                  The target area a_i is *ignored* for these blocks.

H4 PREPLACED      for every block with C[i,1] != 0: H3 on (w,h) *and*
                    |x_i - x_in| <= 1e-4  and  |y_i - y_in| <= 1e-4
                  Again a_i is ignored.  Tolerance is absolute, not relative.

A block that is both fixed and preplaced is checked as preplaced.
Fixed/preplaced blocks are exempt from H2 (they have the stricter H3/H4).

------------------------------------------------------- SOFT constraints
Never cause infeasibility; they multiply the cost by exp(2 * V_rel).

S1 BOUNDARY   per block, binary.  Bounding box is that of the *solution*
              (min/max over all blocks), not a fixed outline.  For each bit
              set in C[i,4] the corresponding block edge must coincide with
              the corresponding bbox edge within 1e-6.  All set bits must
              hold, otherwise the block contributes 1.
S2 GROUPING   per cluster group p: c_p = number of connected components
              when members are unioned as polygons; contributes c_p - 1.
              Connectivity = a shared edge of *non-zero length*.  Corner
              contact does not connect.  A gap of 1e-12 does not connect
              (shapely is exact, there is no snapping tolerance).
S3 MIB        per mib group q: s_q = number of distinct (round(w,4),
              round(h,4)) pairs; contributes s_q - 1.  Note the rounding to
              4 decimals -- shapes within 1e-4 of each other collapse.

    V_rel = (V_boundary + V_grouping + V_mib) / N_soft,  clipped by
    construction to [0,1], where
    N_soft = #{i : C[i,4] != 0} + sum_p (|G_p| - 1) + sum_q (|M_q| - 1).
    N_soft == 0 -> V_rel = 0 (denominator is max(N_soft, 1)).

Fixed and preplaced do NOT appear in N_soft: they were promoted to hard
constraints, and `fixed_violations` / `preplaced_violations` in the
evaluator's metrics are debug output only.

--------------------------------------------------------------- scoring
    quality  = 1 + 0.5 * (max(0, HPWL_gap) + max(0, Area_gap))
    penalty  = exp(2.0 * V_rel)
    runtime  = max(0.7, (your_runtime / median_runtime) ** 0.3)
    cost     = min(quality * penalty * runtime, 10 - 1e-6)   if feasible
             = 10.0                                          if infeasible

Both gaps are clamped at zero from below: beating the baseline earns
nothing.  Total score = sum_i cost_i * exp(n_i/12) / sum_j exp(n_j/12), so
the 120-block case is worth ~2000x the 21-block case: one infeasible large
case costs far more than mediocre quality everywhere.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------- constants
# These mirror iccad2026_evaluate.py; do not "round them off".
ALPHA = 0.5
BETA = 2.0
GAMMA = 0.3
M_PENALTY = 10.0
AREA_TOLERANCE = 0.01     # H2, relative, strict >
DIM_TOLERANCE = 1e-4      # H3/H4, absolute
OVERLAP_EPS = 1e-6        # H1, absolute, strict >, on BOTH axes
BOUNDARY_EPS = 1e-6       # S1, absolute
MIB_ROUND = 4             # S3, decimals

COL_FIXED, COL_PREPLACED, COL_MIB, COL_CLUSTER, COL_BOUNDARY = range(5)
BIT_LEFT, BIT_RIGHT, BIT_TOP, BIT_BOTTOM = 1, 2, 4, 8


def _as_np(x, dtype=np.float64) -> np.ndarray:
    if x is None:
        return np.zeros((0,), dtype=dtype)
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=dtype)


# ------------------------------------------------------------------- spec
@dataclass
class CaseSpec:
    """Everything the rules need about one test case, in plain numpy."""
    n: int
    area: np.ndarray                 # (n,)   target areas, -1 = ignore
    cons: np.ndarray                 # (n,5)  int constraint columns
    target: Optional[np.ndarray]     # (n,4)  x,y,w,h; -1 where meaningless

    @classmethod
    def from_evaluator(cls, block_count, area_targets, constraints,
                       target_positions=None) -> "CaseSpec":
        n = int(block_count)
        area = _as_np(area_targets)[:n]
        cons = _as_np(constraints, np.int64)[:n]
        if cons.ndim == 1:
            cons = cons.reshape(n, -1)
        if cons.shape[1] < 5:                      # pad missing columns
            cons = np.pad(cons, ((0, 0), (0, 5 - cons.shape[1])))
        tgt = None
        if target_positions is not None:
            tgt = _as_np(target_positions)[:n]
            if tgt.size == 0:
                tgt = None
        return cls(n=n, area=area, cons=cons[:, :5], target=tgt)

    # --- membership helpers -------------------------------------------
    @property
    def fixed_mask(self) -> np.ndarray:
        return self.cons[:, COL_FIXED] != 0

    @property
    def preplaced_mask(self) -> np.ndarray:
        return self.cons[:, COL_PREPLACED] != 0

    @property
    def soft_mask(self) -> np.ndarray:
        """Blocks subject to H2 (the 1% area rule)."""
        return ~(self.fixed_mask | self.preplaced_mask)

    def groups(self, col: int) -> List[np.ndarray]:
        ids = self.cons[:, col]
        top = int(ids.max()) if ids.size else 0
        return [np.nonzero(ids == g)[0] for g in range(1, top + 1)]

    @property
    def mib_groups(self) -> List[np.ndarray]:
        return self.groups(COL_MIB)

    @property
    def cluster_groups(self) -> List[np.ndarray]:
        return self.groups(COL_CLUSTER)

    @property
    def n_soft(self) -> int:
        n = int((self.cons[:, COL_BOUNDARY] != 0).sum())
        for g in self.mib_groups:
            n += max(0, len(g) - 1)
        for g in self.cluster_groups:
            n += max(0, len(g) - 1)
        return n


# ------------------------------------------------------------ hard report
@dataclass
class HardReport:
    """Per-violation detail, so a failure is actionable rather than a bool."""
    nonfinite: List[int] = field(default_factory=list)
    overlap_pairs: List[Tuple[int, int, float, float]] = field(default_factory=list)
    area_blocks: List[Tuple[int, float]] = field(default_factory=list)
    fixed_blocks: List[Tuple[int, float, float]] = field(default_factory=list)
    preplaced_blocks: List[Tuple[int, float, float]] = field(default_factory=list)
    # worst-case margins: how far from tripping each rule (>0 = slack left)
    overlap_margin: float = float("inf")   # OVERLAP_EPS - max penetration
    area_margin: float = float("inf")      # AREA_TOLERANCE - max rel error
    dim_margin: float = float("inf")       # DIM_TOLERANCE - max abs error

    @property
    def n_overlap(self) -> int:
        return len(self.overlap_pairs)

    @property
    def n_area(self) -> int:
        return len(self.area_blocks)

    @property
    def n_dimension(self) -> int:
        """Matches the evaluator's dimension_violations counter: one per
        offending block, and a (w,h) failure short-circuits the (x,y) check."""
        return len({i for i, *_ in self.fixed_blocks} |
                   {i for i, *_ in self.preplaced_blocks})

    @property
    def feasible(self) -> bool:
        return not (self.nonfinite or self.overlap_pairs or self.area_blocks
                    or self.fixed_blocks or self.preplaced_blocks)

    def summary(self) -> str:
        if self.feasible:
            return (f"FEASIBLE (slack: overlap {self.overlap_margin:.2e}, "
                    f"area {self.area_margin:.2e}, dim {self.dim_margin:.2e})")
        bits = []
        if self.nonfinite:
            bits.append(f"{len(self.nonfinite)} block(s) with NaN/Inf "
                        f"(first {self.nonfinite[0]})")
        if self.overlap_pairs:
            i, j, ox, oy = max(self.overlap_pairs, key=lambda p: min(p[2], p[3]))
            bits.append(f"{len(self.overlap_pairs)} overlap pair(s), "
                        f"worst {i}~{j} by {min(ox, oy):.3e}")
        if self.area_blocks:
            i, e = max(self.area_blocks, key=lambda p: p[1])
            bits.append(f"{len(self.area_blocks)} area violation(s), "
                        f"worst block {i} at {e:.4%}")
        if self.fixed_blocks:
            bits.append(f"{len(self.fixed_blocks)} fixed-shape mismatch(es)")
        if self.preplaced_blocks:
            bits.append(f"{len(self.preplaced_blocks)} preplaced mismatch(es)")
        return "INFEASIBLE: " + "; ".join(bits)


def check_hard(positions, spec: CaseSpec) -> HardReport:
    """H1-H4.  `positions` is (n,4) of (x, y, w, h)."""
    p = _as_np(positions)
    rep = HardReport()
    n = min(len(p), spec.n)
    if n == 0:
        return rep

    # Non-finite coordinates are a failure class of their own, and a nastier
    # one than infeasibility.  Every comparison against NaN is False, so the
    # official evaluator finds no overlap, no area error and no dimension
    # drift -- it calls the layout FEASIBLE, then computes a NaN bounding-box
    # area, a NaN cost, and a NaN Total Score for the whole submission.  We
    # deliberately diverge from the evaluator here: on finite input this check
    # is inert, and on non-finite input "silently correct" is not an option.
    bad = ~np.isfinite(p[:n]).all(axis=1)
    if bad.any():
        rep.nonfinite = [int(i) for i in np.nonzero(bad)[0]]
        return rep

    x0, y0, w, h = p[:n, 0], p[:n, 1], p[:n, 2], p[:n, 3]
    x1, y1 = x0 + w, y0 + h

    # --- H1 overlap (vectorised; identical predicate to the evaluator)
    ox = np.minimum(x1[:, None], x1[None, :]) - np.maximum(x0[:, None], x0[None, :])
    oy = np.minimum(y1[:, None], y1[None, :]) - np.maximum(y0[:, None], y0[None, :])
    pen = np.minimum(ox, oy)
    iu, ju = np.triu_indices(n, k=1)
    if iu.size:
        rep.overlap_margin = float(OVERLAP_EPS - pen[iu, ju].max())
        for k in np.nonzero((ox[iu, ju] > OVERLAP_EPS) & (oy[iu, ju] > OVERLAP_EPS))[0]:
            i, j = int(iu[k]), int(ju[k])
            rep.overlap_pairs.append((i, j, float(ox[i, j]), float(oy[i, j])))

    # --- H2 area tolerance (soft blocks only)
    soft = spec.soft_mask[:n]
    valid = soft & (spec.area[:n] > 0) & (spec.area[:n] != -1)
    if valid.any():
        err = np.abs(w * h - spec.area[:n]) / np.where(valid, spec.area[:n], 1.0)
        err = np.where(valid, err, 0.0)
        rep.area_margin = float(AREA_TOLERANCE - err.max())
        for i in np.nonzero(err > AREA_TOLERANCE)[0]:
            rep.area_blocks.append((int(i), float(err[i])))

    # --- H3/H4 immutability
    if spec.target is not None:
        t = spec.target[:n]
        dim_worst = 0.0
        for i in range(n):
            is_fixed = bool(spec.fixed_mask[i])
            is_pre = bool(spec.preplaced_mask[i])
            if not (is_fixed or is_pre):
                continue
            dw, dh = abs(w[i] - t[i, 2]), abs(h[i] - t[i, 3])
            dim_worst = max(dim_worst, dw, dh)
            if dw > DIM_TOLERANCE or dh > DIM_TOLERANCE:
                # evaluator counts this block once and skips its (x,y) check
                (rep.preplaced_blocks if is_pre else rep.fixed_blocks).append(
                    (int(i), float(dw), float(dh)))
                continue
            if is_pre:
                dx, dy = abs(x0[i] - t[i, 0]), abs(y0[i] - t[i, 1])
                dim_worst = max(dim_worst, dx, dy)
                if dx > DIM_TOLERANCE or dy > DIM_TOLERANCE:
                    rep.preplaced_blocks.append((int(i), float(dx), float(dy)))
        rep.dim_margin = float(DIM_TOLERANCE - dim_worst)
    return rep


# ------------------------------------------------------------ soft report
@dataclass
class SoftReport:
    boundary: int = 0
    grouping: int = 0
    mib: int = 0
    n_soft: int = 0
    boundary_blocks: List[int] = field(default_factory=list)
    grouping_ids: List[Tuple[int, int]] = field(default_factory=list)  # (group, comps)
    mib_ids: List[Tuple[int, int]] = field(default_factory=list)       # (group, shapes)

    @property
    def total(self) -> int:
        return self.boundary + self.grouping + self.mib

    @property
    def relative(self) -> float:
        return self.total / max(self.n_soft, 1)


def check_soft(positions, spec: CaseSpec) -> SoftReport:
    """S1-S3.  Grouping uses shapely when available, with an exact
    interval-overlap fallback that agrees with it on axis-aligned boxes."""
    p = _as_np(positions)
    n = min(len(p), spec.n)
    rep = SoftReport(n_soft=spec.n_soft)
    if n == 0:
        return rep
    x0, y0, w, h = p[:n, 0], p[:n, 1], p[:n, 2], p[:n, 3]
    x1, y1 = x0 + w, y0 + h

    # --- S1 boundary, against the solution's own bounding box
    bits = spec.cons[:n, COL_BOUNDARY]
    if (bits != 0).any():
        bx0, by0, bx1, by1 = x0.min(), y0.min(), x1.max(), y1.max()
        for i in np.nonzero(bits)[0]:
            code = int(bits[i])
            ok = True
            if code & BIT_LEFT:
                ok &= abs(x0[i] - bx0) < BOUNDARY_EPS
            if code & BIT_RIGHT:
                ok &= abs(x1[i] - bx1) < BOUNDARY_EPS
            if code & BIT_TOP:
                ok &= abs(y1[i] - by1) < BOUNDARY_EPS
            if code & BIT_BOTTOM:
                ok &= abs(y0[i] - by0) < BOUNDARY_EPS
            if not ok:
                rep.boundary += 1
                rep.boundary_blocks.append(int(i))

    # --- S2 grouping: connected components under shared-edge adjacency
    for gid, mem in enumerate(spec.cluster_groups, start=1):
        mem = mem[mem < n]
        if len(mem) < 2:
            continue
        comps = _components(x0[mem], y0[mem], x1[mem], y1[mem])
        rep.grouping += comps - 1
        if comps > 1:
            rep.grouping_ids.append((gid, comps))

    # --- S3 mib: distinct rounded shapes
    for gid, mem in enumerate(spec.mib_groups, start=1):
        mem = mem[mem < n]
        if len(mem) < 2:
            continue
        shapes = {(round(float(w[i]), MIB_ROUND), round(float(h[i]), MIB_ROUND))
                  for i in mem}
        rep.mib += len(shapes) - 1
        if len(shapes) > 1:
            rep.mib_ids.append((gid, len(shapes)))
    return rep


def _components(x0, y0, x1, y1) -> int:
    """Number of connected components where two boxes are adjacent iff they
    share a boundary segment of strictly positive length (corner contact and
    any gap, however small, do not connect).  This is exactly what shapely's
    unary_union reports for axis-aligned boxes."""
    m = len(x0)
    parent = list(range(m))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(m):
        for j in range(i + 1, m):
            ox = min(x1[i], x1[j]) - max(x0[i], x0[j])
            oy = min(y1[i], y1[j]) - max(y0[i], y0[j])
            # touching or overlapping on one axis, strictly positive on the other
            if (ox >= 0 and oy > 0) or (oy >= 0 and ox > 0):
                ra, rb = find(i), find(j)
                if ra != rb:
                    parent[ra] = rb
    return len({find(i) for i in range(m)})


# ----------------------------------------------------------------- scoring
def bbox_area(positions) -> float:
    p = _as_np(positions)
    if len(p) == 0:
        return 0.0
    return float((p[:, 0] + p[:, 2]).max() - p[:, 0].min()) * \
           float((p[:, 1] + p[:, 3]).max() - p[:, 1].min())


def hpwl(positions, b2b=None, p2b=None, pins=None) -> float:
    """Weighted Manhattan wirelength, mirroring the evaluator exactly.

    Row layouts differ between the two edge tensors and this is easy to get
    backwards:
        b2b row = (block_i, block_j, weight)
        p2b row = (PIN_idx, block_idx, weight)   <- pin first, not block
    A row is padding iff its FIRST entry is -1; the evaluator does not test
    the second entry, and a negative index there would wrap round in Python,
    so we reproduce that rather than filtering it out.
    """
    p = _as_np(positions)
    if len(p) == 0:
        return 0.0
    cx, cy = p[:, 0] + p[:, 2] / 2, p[:, 1] + p[:, 3] / 2
    n = len(p)
    total = 0.0

    e = _as_np(b2b)
    if e.size:
        e = e.reshape(-1, 3)
        e = e[e[:, 0] != -1]
        i, j, wt = e[:, 0].astype(int), e[:, 1].astype(int), e[:, 2]
        keep = (i < n) & (j < n)
        i, j, wt = i[keep], j[keep], wt[keep]
        total += float((wt * (np.abs(cx[i] - cx[j]) + np.abs(cy[i] - cy[j]))).sum())

    e = _as_np(p2b)
    pn = _as_np(pins).reshape(-1, 2)
    if e.size and len(pn):
        e = e.reshape(-1, 3)
        e = e[e[:, 0] != -1]
        t, b, wt = e[:, 0].astype(int), e[:, 1].astype(int), e[:, 2]
        keep = (b < n) & (t < len(pn))
        t, b, wt = t[keep], b[keep], wt[keep]
        total += float((wt * (np.abs(pn[t, 0] - cx[b]) +
                              np.abs(pn[t, 1] - cy[b]))).sum())
    return total


def cost(hpwl_gap: float, area_gap: float, violations_relative: float,
         runtime_factor: float = 1.0, feasible: bool = True) -> float:
    if not feasible:
        return M_PENALTY
    quality = 1.0 + ALPHA * (max(0.0, hpwl_gap) + max(0.0, area_gap))
    penalty = math.exp(BETA * violations_relative)
    runtime = max(0.7, runtime_factor ** GAMMA)
    return min(quality * penalty * runtime, M_PENALTY - 1e-6)


def total_score(costs: Sequence[float], block_counts: Sequence[int]) -> float:
    if not costs:
        return 0.0
    mx = max(block_counts)
    wts = [math.exp((n - mx) / 12) for n in block_counts]
    z = sum(wts)
    return sum(c * w for c, w in zip(costs, wts)) / z


# --------------------------------------------------------------- one-shot
@dataclass
class Verdict:
    hard: HardReport
    soft: SoftReport
    hpwl: float
    area: float
    hpwl_gap: float
    area_gap: float
    cost: float

    @property
    def feasible(self) -> bool:
        return self.hard.feasible


def evaluate(positions, spec: CaseSpec, b2b=None, p2b=None, pins=None,
             hpwl_baseline: Optional[float] = None,
             area_baseline: Optional[float] = None,
             runtime_factor: float = 1.0) -> Verdict:
    hard = check_hard(positions, spec)
    soft = check_soft(positions, spec)
    wl = hpwl(positions, b2b, p2b, pins)
    ar = bbox_area(positions)
    hg = 0.0 if not hpwl_baseline else (wl - hpwl_baseline) / max(hpwl_baseline, 1e-6)
    ag = 0.0 if not area_baseline else (ar - area_baseline) / max(area_baseline, 1e-6)
    return Verdict(hard, soft, wl, ar, hg, ag,
                   cost(hg, ag, soft.relative, runtime_factor, hard.feasible))
