"""Last-resort hard-constraint enforcement for the contest optimizer.

Why this exists
---------------
The scoring cliff is brutally asymmetric: a mediocre legal layout costs ~1-3,
an illegal one costs exactly 10, and the case weight e^(n/12) means one
illegal 120-block case outweighs the entire 21-59 block range combined.  So
the last thing a solver does must be "prove this is legal, and if it is not,
make it legal" -- never "trust the pipeline".

The existing legalizer only ever guards H1 (overlap), through `proxy_cost`
returning 10.0 when `max_penetration > 1e-6`.  H2/H3/H4 (area tolerance,
fixed dims, preplaced position) are stamped once at the start and then
assumed to survive; and when *every* rung of the assignment ladder fails,
the overlapping result is still returned, because best-of-k selects the
lowest proxy cost even when that cost is 10.

`enforce` closes both holes.  It is a no-op on a layout that is already
legal (verified bit-identical on all 100 validation solutions), and the
repairs are ordered so that the least destructive one that works wins:

    1. re-stamp preplaced (x,y,w,h) and fixed (w,h) from the input, and
       rescale only those soft blocks that actually miss the 1% window
    2. evict the minimum set of blocks involved in overlaps and re-place
       just those, keeping the rest of the layout intact
    3. repack everything from scratch -- legal by construction, poor quality
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import rules
from rules import CaseSpec, HardReport, check_hard

# Stamp soft-block areas a hair under target rather than exactly on it: the
# evaluator recomputes w*h in float64 from the numbers we emit, and landing
# exactly on the boundary of a strict `>` comparison is not a place to be.
AREA_SAFETY = 0.999
# Only rescale a soft block once it is within a whisker of the hard limit.
# The legalizer deliberately parks soft blocks at 0.009 relative error (it
# shrinks them to 0.991x target to buy packing slack), so a trigger anywhere
# below that would rewrite the dimensions of a perfectly legal layout and
# turn a one-block repair into a full repack.
AREA_TRIGGER = rules.AREA_TOLERANCE * 0.98
ASPECT_CAP = 3.0



def _mib_buckets(spec: CaseSpec, realized: np.ndarray) -> List[List[int]]:
    """Split every MIB group into equal-target-area subsets.

    Identical dimensions is a soft constraint; the 1% area window is hard, and
    blocks with different target areas cannot share a shape at all.  So the
    most a legal layout can do is give one shape to each distinct area, which
    is what these buckets are.  Frozen members are bucketed by the area their
    immutable dims actually realise, since their target area is ignored.
    """
    out: List[List[int]] = []
    frozen = spec.fixed_mask | spec.preplaced_mask
    for mem in spec.mib_groups:
        mem = mem[mem < spec.n]
        if len(mem) < 2:
            continue
        buckets: List[Tuple[float, List[int]]] = []
        for i in mem:
            i = int(i)
            v = float(realized[i]) if frozen[i] else float(spec.area[i])
            for b in buckets:
                if abs(b[0] - v) <= 1e-6 * max(abs(b[0]), 1.0):
                    b[1].append(i)
                    break
            else:
                buckets.append((v, [i]))
        out.extend(b[1] for b in buckets)
    return out


@dataclass
class GuardInfo:
    stage: str            # 'clean' | 'stamped' | 'evicted' | 'repacked'
    before: HardReport
    after: HardReport
    notes: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        head = f"[guard] {self.stage}: {self.after.summary()}"
        return head + ("\n  " + "\n  ".join(self.notes) if self.notes else "")


# ================================================================= entry point
def enforce(positions, spec: CaseSpec, hint: Optional[np.ndarray] = None
            ) -> Tuple[np.ndarray, GuardInfo]:
    """Return a layout satisfying every hard constraint, plus a report."""
    sol = rules._as_np(positions).reshape(-1, 4).copy()
    if len(sol) < spec.n:                       # solver returned too few rows
        pad = np.zeros((spec.n - len(sol), 4))
        side = np.sqrt(np.maximum(spec.area[len(sol):spec.n], 1e-9))
        pad[:, 2] = pad[:, 3] = side
        sol = np.vstack([sol, pad])
    sol = sol[:spec.n]

    before = check_hard(sol, spec)
    if before.feasible:
        return sol, GuardInfo("clean", before, before)

    notes = [f"input infeasible: {before.summary()}"]

    # NaN/Inf first: every later repair compares against these values, and a
    # comparison with NaN is False, so an unsanitised row would slip through
    # each stage looking fine.
    bad = ~np.isfinite(sol).all(axis=1)
    if bad.any():
        notes.append(f"sanitised {int(bad.sum())} non-finite block(s)")
        for i in np.nonzero(bad)[0]:
            a = float(spec.area[i]) if spec.area[i] > 0 else 1.0
            side = math.sqrt(a * AREA_SAFETY)
            sol[i] = (0.0, 0.0, side, side)

    stamped = _stamp(sol, spec, notes)
    mid = check_hard(stamped, spec)
    if mid.feasible:
        return stamped, GuardInfo("stamped", before, mid, notes)

    evicted = _repair_overlaps(stamped, spec, notes)
    if evicted is not None:
        rep = check_hard(evicted, spec)
        if rep.feasible:
            return evicted, GuardInfo("evicted", before, rep, notes)
        notes.append(f"eviction left: {rep.summary()}")

    notes.append("repacking from scratch")
    packed = shelf_pack(spec, hint=hint if hint is not None else sol)
    after = check_hard(packed, spec)
    if not after.feasible:      # should be unreachable; say so loudly if not
        notes.append(f"REPACK ITSELF INFEASIBLE: {after.summary()}")
    return packed, GuardInfo("repacked", before, after, notes)


# ================================================================== stage 1
def _stamp(sol: np.ndarray, spec: CaseSpec, notes: List[str]) -> np.ndarray:
    """Restore immutable values and pull stray areas back inside tolerance.
    Only blocks that actually violate something are touched -- rewriting a
    block that was already legal is how a repair turns into a regression."""
    sol = sol.copy()
    n = spec.n
    pre, fix = spec.preplaced_mask, spec.fixed_mask

    if spec.target is not None:
        t = spec.target
        off = np.abs(sol[:, 2:4] - t[:, 2:4]).max(axis=1) > rules.DIM_TOLERANCE
        moved = np.abs(sol[:, 0:2] - t[:, 0:2]).max(axis=1) > rules.DIM_TOLERANCE
        hit = pre & (off | moved)
        if hit.any():
            sol[hit] = t[hit]
            notes.append(f"restored {int(hit.sum())} preplaced block(s)")
        hit = fix & ~pre & off
        if hit.any():
            sol[hit, 2:4] = t[hit, 2:4]
            notes.append(f"restored {int(hit.sum())} fixed shape(s)")

    # MIB members must move as a unit, otherwise repairing one member's area
    # splits the group's shape -- but only within an equal-area bucket, since
    # blocks with different target areas can never share a shape and forcing
    # them to would trade the hard area constraint for the soft MIB one.
    in_mib = np.zeros(n, dtype=bool)
    regrouped = 0
    for sub in _mib_buckets(spec, sol[:, 2] * sol[:, 3]):
        in_mib[sub] = True
        movable = [i for i in sub if spec.soft_mask[i] and spec.area[i] > 0]
        if not movable:
            continue                              # frozen member sets the shape
        tgt = float(spec.area[movable[0]])
        rep = movable[0]
        w, h = float(sol[rep, 2]), float(sol[rep, 3])
        shapes = {(round(float(sol[i, 2]), rules.MIB_ROUND),
                   round(float(sol[i, 3]), rules.MIB_ROUND)) for i in sub}
        worst = max(abs(float(sol[i, 2] * sol[i, 3]) - float(spec.area[i]))
                    / float(spec.area[i]) for i in movable)
        if worst <= AREA_TRIGGER and len(shapes) == 1:
            continue                              # already fine, leave it be
        if w <= 0 or h <= 0:
            w = h = math.sqrt(tgt)
        s = math.sqrt(tgt * AREA_SAFETY / (w * h))
        mv = [i for i in sub if spec.soft_mask[i]]
        sol[mv, 2], sol[mv, 3] = w * s, h * s
        regrouped += 1
    if regrouped:
        notes.append(f"re-tied {regrouped} MIB bucket(s)")

    rescaled = 0
    for i in np.nonzero(spec.soft_mask & ~in_mib)[0]:
        a = float(spec.area[i])
        if a <= 0 or a == -1:
            continue
        w, h = float(sol[i, 2]), float(sol[i, 3])
        if w > 0 and h > 0 and abs(w * h - a) / a <= AREA_TRIGGER:
            continue
        if w <= 0 or h <= 0:
            w = h = math.sqrt(a)
        s = math.sqrt(a * AREA_SAFETY / (w * h))
        sol[i, 2], sol[i, 3] = w * s, h * s
        rescaled += 1
    if rescaled:
        notes.append(f"rescaled {rescaled} soft block(s) onto target area")
    return sol


# ================================================================== stage 2
def _repair_quality(sol: np.ndarray, spec: CaseSpec) -> float:
    """Rank candidate repairs the way the contest cost would.

    cost = (1 + 0.5*(hpwl_gap + area_gap)) * exp(2*V_rel), so bbox area and
    V_rel are the two terms a repair can move, and V_rel sits in an exponent.
    HPWL is unavailable here (`enforce` never sees the edge lists), so the
    placement search below uses displacement from the input layout as its
    stand-in; this function ranks only what it can measure exactly.
    """
    soft = rules.check_soft(sol, spec)
    return rules.bbox_area(sol) * math.exp(2.0 * soft.relative)


# Weight sets for the local search: (bbox growth, displacement, cluster
# cohesion, boundary).  Each produces one candidate layout and the caller
# keeps whichever scores best, so these are search directions rather than
# tuned constants -- a bad set costs one wasted candidate, never a regression.
_LOCAL_STRATEGIES = (
    ("tight", (1.00, 0.10, 0.25, 0.20)),   # guard the bbox above all
    ("near",  (0.35, 0.60, 0.25, 0.15)),   # keep blocks where the placer put them
    ("group", (0.45, 0.15, 0.90, 0.20)),   # hold cluster members together
)


def _place_local(sol: np.ndarray, kept: Sequence[int], evicted: Sequence[int],
                 spec: CaseSpec, weights: Tuple[float, float, float, float]
                 ) -> np.ndarray:
    """Slot each evicted block into free space that already exists.

    The shelf fallback stacks every evicted block above the layout's skyline.
    That is legal but disproportionate: a three-pair overlap ends up paying
    for a whole new row, the bbox grows, blocks that were touching it stop
    counting as boundary-satisfied, and cluster members land in a different
    band -- all three soft counters move the wrong way at once, and V_rel is
    in an exponent.  Most evictions do not need a new row; there is usually a
    hole inside the block's own neighbourhood.  Look there first.

    Candidates are the classic bottom-left corner points: a bottom-left
    justified placement always has its left edge on the bbox or on some
    placed block's right edge, and its bottom edge on the bbox or on some
    placed block's top edge, so that set is enough to find any hole.  Blocks
    that genuinely do not fit are collected and handed to `_skyline_place`
    together, so they share one tight row instead of one row each.
    """
    w_area, w_disp, w_clust, w_bnd = weights
    out = sol.copy()
    eps = rules.OVERLAP_EPS
    A = float(max(spec.area[:spec.n].clip(min=0).sum(), 1e-9))
    S = math.sqrt(A)

    P = out[list(kept), :4].copy() if len(kept) else np.zeros((0, 4))
    if len(P):
        bx0, by0 = float(P[:, 0].min()), float(P[:, 1].min())
        bx1 = float((P[:, 0] + P[:, 2]).max())
        by1 = float((P[:, 1] + P[:, 3]).max())
    else:
        bx0, by0, bx1, by1 = 0.0, 0.0, 0.0, 0.0
    placed = list(kept)
    leftover: List[int] = []

    cluster = spec.cons[:spec.n, rules.COL_CLUSTER]
    bits = spec.cons[:spec.n, rules.COL_BOUNDARY]
    MAX_CAND = 20000

    # big blocks first: hardest to fit, most damaging to get wrong
    for i in sorted(evicted, key=lambda i: -float(out[i, 2] * out[i, 3])):
        w, h = float(out[i, 2]), float(out[i, 3])
        ox, oy = float(out[i, 0]), float(out[i, 1])

        if not len(P):
            out[i, 0], out[i, 1] = ox, oy
            P = np.vstack([P, out[i:i + 1, :4]])
            placed.append(i)
            continue

        px = np.unique(np.concatenate([[bx0], P[:, 0] + P[:, 2]]))
        py = np.unique(np.concatenate([[by0], P[:, 1] + P[:, 3]]))
        cx = np.repeat(px, len(py))
        cy = np.tile(py, len(px))
        if len(cx) > MAX_CAND:                 # keep the nearest ones
            keep = np.argsort(np.abs(cx - ox) + np.abs(cy - oy))[:MAX_CAND]
            cx, cy = cx[keep], cy[keep]

        R0, R1 = P[:, 0], P[:, 0] + P[:, 2]
        T0, T1 = P[:, 1], P[:, 1] + P[:, 3]
        ovx = np.minimum(cx[:, None] + w, R1[None, :]) - np.maximum(cx[:, None], R0[None, :])
        ovy = np.minimum(cy[:, None] + h, T1[None, :]) - np.maximum(cy[:, None], T0[None, :])
        free = ~((ovx > eps) & (ovy > eps)).any(axis=1)
        del ovx, ovy

        if not free.any():
            leftover.append(i)
            continue
        cx, cy = cx[free], cy[free]

        gx0 = np.minimum(bx0, cx); gx1 = np.maximum(bx1, cx + w)
        gy0 = np.minimum(by0, cy); gy1 = np.maximum(by1, cy + h)
        score = w_area * ((gx1 - gx0) * (gy1 - gy0)
                          - (bx1 - bx0) * (by1 - by0)) / A
        score = score + w_disp * (np.abs(cx - ox) + np.abs(cy - oy)) / S

        gid = int(cluster[i])
        if gid and w_clust:
            mem = [j for j in placed if int(cluster[j]) == gid]
            if mem:
                M = out[mem, :4]
                mcx = float((M[:, 0] + M[:, 2] / 2).mean())
                mcy = float((M[:, 1] + M[:, 3] / 2).mean())
                score = score + w_clust * (np.abs(cx + w / 2 - mcx)
                                           + np.abs(cy + h / 2 - mcy)) / S

        code = int(bits[i])
        if code and w_bnd:
            miss = np.zeros(len(cx))
            if code & rules.BIT_LEFT:
                miss += np.abs(cx - gx0)
            if code & rules.BIT_RIGHT:
                miss += np.abs(cx + w - gx1)
            if code & rules.BIT_TOP:
                miss += np.abs(cy + h - gy1)
            if code & rules.BIT_BOTTOM:
                miss += np.abs(cy - gy0)
            score = score + w_bnd * miss / S

        b = int(np.argmin(score))
        out[i, 0], out[i, 1] = cx[b], cy[b]
        P = np.vstack([P, out[i:i + 1, :4]])
        placed.append(i)
        bx0, by0 = min(bx0, float(cx[b])), min(by0, float(cy[b]))
        bx1 = max(bx1, float(cx[b]) + w)
        by1 = max(by1, float(cy[b]) + h)

    if leftover:
        obstacles = [(float(out[j, 0]), float(out[j, 1]),
                      float(out[j, 2]), float(out[j, 3])) for j in placed]
        items = [(i, float(out[i, 2]), float(out[i, 3])) for i in leftover]
        width = max(bx1 - bx0, max(w for _, w, _ in items))
        for i, (qx, qy) in _skyline_place(obstacles, items, width, bx0, by0).items():
            out[i, 0], out[i, 1] = qx, qy

    return out


def _repair_overlaps(sol: np.ndarray, spec: CaseSpec, notes: List[str]
                     ) -> Optional[np.ndarray]:
    """Keep the largest sensible overlap-free subset in place and re-place
    only the rest.  Preplaced blocks are never evicted -- they cannot move --
    so a preplaced/preplaced overlap is unrepairable here and we bail out."""
    rep = check_hard(sol, spec)
    if not rep.overlap_pairs:
        return None
    pre = spec.preplaced_mask
    if any(pre[i] and pre[j] for i, j, _, _ in rep.overlap_pairs):
        notes.append("preplaced blocks overlap each other in the input")
        return None

    sol = sol.copy()
    x0, y0 = sol[:, 0], sol[:, 1]
    x1, y1 = x0 + sol[:, 2], y0 + sol[:, 3]

    # greedy: preplaced first (immovable), then largest area -- a big block
    # is both harder to re-place and more damaging to move
    area = sol[:, 2] * sol[:, 3]
    order = sorted(range(spec.n), key=lambda i: (0 if pre[i] else 1, -area[i], i))
    kept: List[int] = []
    evicted: List[int] = []
    for i in order:
        clash = False
        for j in kept:
            ox = min(x1[i], x1[j]) - max(x0[i], x0[j])
            oy = min(y1[i], y1[j]) - max(y0[i], y0[j])
            if ox > rules.OVERLAP_EPS and oy > rules.OVERLAP_EPS:
                clash = True
                break
        (evicted if clash else kept).append(i)
    if not evicted:
        return None
    notes.append(f"evicting {len(evicted)} of {spec.n} block(s), "
                 f"keeping {len(kept)} in place")

    # MIB groups keep one shape, so evict whole groups together
    ev = set(evicted)
    for sub in _mib_buckets(spec, sol[:, 2] * sol[:, 3]):
        if ev & set(sub) and not all(pre[i] for i in sub):
            for i in sub:
                if not pre[i]:
                    ev.add(i)
    evicted = sorted(ev)
    kept = [i for i in range(spec.n) if i not in ev]

    # Several ways to re-place the evicted blocks; keep whichever actually
    # scores best.  The shelf packing is one of the candidates, so this can
    # only ever match or beat the previous behaviour.
    cands: List[Tuple[str, np.ndarray]] = []
    for label, wts in _LOCAL_STRATEGIES:
        c = _place_local(sol, kept, evicted, spec, wts)
        if check_hard(c, spec).feasible:
            cands.append((label, c))

    shelf = sol.copy()
    obstacles = [(float(sol[i, 0]), float(sol[i, 1]),
                  float(sol[i, 2]), float(sol[i, 3])) for i in kept]
    items = [(i, float(sol[i, 2]), float(sol[i, 3])) for i in evicted]
    x_lo = float(min(x0)) if spec.n else 0.0
    width = max((max(x1[kept]) - x_lo) if kept else 0.0,
                max(w for _, w, _ in items))
    for i, (px, py) in _skyline_place(obstacles, items, width, x_lo,
                                      float(min(y0))).items():
        shelf[i, 0], shelf[i, 1] = px, py
    if check_hard(shelf, spec).feasible:
        cands.append(("shelf", shelf))

    if not cands:
        return shelf          # infeasible; caller falls through to a repack

    scored = [(_repair_quality(c, spec), label, c) for label, c in cands]
    scored.sort(key=lambda t: t[0])
    best_q, best_label, best = scored[0]
    notes.append("re-placement: " + ", ".join(
        f"{lab}{'*' if lab == best_label else ''} {q:.4g}"
        for q, lab, _ in scored))
    return best


# ================================================================== stage 3
def shelf_pack(spec: CaseSpec, hint: Optional[np.ndarray] = None) -> np.ndarray:
    """A layout that is legal by construction, at some cost in quality.

    Skyline (bottom-left) packing inside a roughly square outline, with the
    preplaced blocks pre-loaded into the profile as obstacles.  Dimensions
    come from the exact target area (or the immutable input), so H2/H3/H4
    hold by construction, and the skyline invariant gives H1.

    `hint` (n,4) donates aspect ratios for soft blocks; areas are always
    re-derived from the targets, only the w:h ratio is borrowed.
    """
    n = spec.n
    sol = np.zeros((n, 4))
    pre, fix = spec.preplaced_mask, spec.fixed_mask

    for i in range(n):
        if spec.target is not None and (pre[i] or fix[i]):
            sol[i, 2:4] = spec.target[i, 2:4]
            if pre[i]:
                sol[i, 0:2] = spec.target[i, 0:2]
            continue
        a = float(spec.area[i])
        a = a * AREA_SAFETY if a > 0 else 1.0
        ratio = 1.0
        if hint is not None and i < len(hint) and hint[i, 2] > 0 and hint[i, 3] > 0:
            ratio = min(max(float(hint[i, 2]) / float(hint[i, 3]),
                            1 / ASPECT_CAP), ASPECT_CAP)
        h = math.sqrt(a / ratio)
        sol[i, 2:4] = (a / h, h)
    for sub in _mib_buckets(spec, sol[:, 2] * sol[:, 3]):
        frozen = [i for i in sub if pre[i] or fix[i]]
        rep = frozen[0] if frozen else sub[0]
        mv = [i for i in sub if not (pre[i] or fix[i])]   # frozen dims are law
        if mv:
            sol[mv, 2], sol[mv, 3] = sol[rep, 2], sol[rep, 3]

    movable = [i for i in range(n) if not pre[i]]
    if not movable:
        return sol

    total = float((sol[:, 2] * sol[:, 3]).sum())
    pre_right = float((sol[pre, 0] + sol[pre, 2]).max()) if pre.any() else 0.0
    width = max(math.sqrt(total) * 1.05, float(sol[movable, 2].max()), pre_right)

    obstacles = [(float(sol[i, 0]), float(sol[i, 1]),
                  float(sol[i, 2]), float(sol[i, 3])) for i in np.nonzero(pre)[0]]
    x_lo = float(sol[pre, 0].min()) if pre.any() else 0.0
    y_lo = float(sol[pre, 1].min()) if pre.any() else 0.0
    for i, (px, py) in _skyline_place(obstacles, _pack_order(spec, sol, movable),
                                      width, x_lo, y_lo).items():
        sol[i, 0], sol[i, 1] = px, py
    return sol


def _pack_order(spec: CaseSpec, sol: np.ndarray, movable: Sequence[int]
                ) -> List[Tuple[int, float, float]]:
    """Cluster members consecutive (so they tend to abut and the grouping
    penalty does not saturate), groups and singletons largest-first."""
    cid = spec.cons[:, rules.COL_CLUSTER]
    area = sol[:, 2] * sol[:, 3]
    mset = set(movable)
    key: Dict[int, float] = {}
    for gid, mem in enumerate(spec.cluster_groups, start=1):
        mem = [i for i in mem if i in mset]
        if mem:
            key[gid] = -max(float(area[i]) for i in mem)
    order = sorted(movable, key=lambda i: (
        key.get(int(cid[i]), -float(area[i])), int(cid[i]), -float(area[i]), i))
    return [(int(i), float(sol[i, 2]), float(sol[i, 3])) for i in order]


# --------------------------------------------------------------- skyline core
def _skyline_place(obstacles: Sequence[Tuple[float, float, float, float]],
                   items: Sequence[Tuple[int, float, float]],
                   width: float, x_lo: float = 0.0, y_lo: float = 0.0
                   ) -> Dict[int, Tuple[float, float]]:
    """Bottom-left skyline packing of `items` above the profile induced by
    `obstacles`, inside x in [0, width].

    Legality argument: the profile is an upper envelope of everything already
    placed, and each item is placed with its bottom edge exactly on the
    profile maximum over its own x-span, then raises the profile there.  So
    no item can dip below anything it spans, and no two items can interleave.
    """
    x_hi = x_lo + max(width, 1e-9)
    xs = {x_lo, x_hi}
    for bx, _by, bw, _bh in obstacles:
        for v in (bx, bx + bw):
            if x_lo < v < x_hi:
                xs.add(v)
    bounds = sorted(xs)
    seg_x, seg_end = bounds[:-1], bounds[1:]
    if not seg_x:
        seg_x, seg_end = [x_lo], [x_hi]
    seg_h = [y_lo] * len(seg_x)
    for bx, by, bw, bh in obstacles:
        for s in range(len(seg_x)):
            if seg_end[s] > bx + 1e-12 and seg_x[s] < bx + bw - 1e-12:
                seg_h[s] = max(seg_h[s], by + bh)

    def profile(a: float, b: float) -> float:
        top = y_lo
        for s in range(len(seg_x)):
            if seg_end[s] > a + 1e-12 and seg_x[s] < b - 1e-12:
                top = max(top, seg_h[s])
        return top

    def raise_to(a: float, b: float, y: float) -> None:
        nonlocal seg_x, seg_end, seg_h
        nx, ne, nh = [], [], []
        for s in range(len(seg_x)):
            s0, s1, sh = seg_x[s], seg_end[s], seg_h[s]
            if s1 <= a + 1e-12 or s0 >= b - 1e-12:
                nx.append(s0); ne.append(s1); nh.append(sh)
                continue
            if s0 < a - 1e-12:
                nx.append(s0); ne.append(a); nh.append(sh)
            nx.append(max(s0, a)); ne.append(min(s1, b)); nh.append(max(sh, y))
            if s1 > b + 1e-12:
                nx.append(b); ne.append(s1); nh.append(sh)
        seg_x, seg_end, seg_h = nx, ne, nh

    out: Dict[int, Tuple[float, float]] = {}
    for i, w, h in items:
        limit = x_lo + max(width - w, 0.0)
        best_x, best_y = x_lo, float("inf")
        for cand in sorted(set(seg_x + [limit])):
            px = min(max(cand, x_lo), limit)
            py = profile(px, px + w)
            if py < best_y - 1e-12 or (abs(py - best_y) <= 1e-12 and px < best_x):
                best_x, best_y = px, py
        out[i] = (best_x, best_y)
        raise_to(best_x, best_x + w, best_y + h)
    return out


# ================================================================== wrapper
def guarded_solve(inner, spec: CaseSpec, *args, verbose: bool = False, **kwargs):
    """Run `inner(*args, **kwargs)` and never let a hard violation -- or an
    exception -- reach the evaluator.  An exception inside `solve` is scored
    exactly like an overlap (cost = M = 10); this turns both into a legal,
    merely-mediocre layout."""
    try:
        raw = inner(*args, **kwargs)
    except BaseException as exc:                # noqa: BLE001 - deliberate
        print(f"[guard] solver raised {type(exc).__name__}: {exc} -> repack")
        sol = shelf_pack(spec)
        rep = check_hard(sol, spec)
        if not rep.feasible:
            print(f"[guard] REPACK INFEASIBLE: {rep.summary()}")
        return [tuple(map(float, r)) for r in sol]

    sol, info = enforce(raw, spec)
    if info.stage != "clean" or verbose:
        print(info)
    return [tuple(map(float, r)) for r in sol]
