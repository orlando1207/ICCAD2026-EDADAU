"""
SP floorplanner orchestrator.

Pipeline (given fixed block dims):
  compile -> warm-start SP -> simulated annealing -> slack redistribution
  -> snap preplaced exact -> emit positions.

Guarantees: always returns a geometrically feasible layout (overlap-free,
preplaced exact). If SA never finds a feasible topology, a constructive floor
(movable units shelf-packed clear of the preplaced obstacles) is used.
"""

import multiprocessing as mp

import numpy as np

from compile_problem import compile_with_targets, expand
from init_place import initial_sp, _simpl_place
from packer import pack, build_adj, redistribute
from anneal import anneal
from fastcore import extract_arrays, energy_nb
from skyline import skyline_legalize
from spcost import score_layout, count_overlaps


def _grouping_from_pos(comp, bx, by, bw, bh):
    """V_grouping (components-1 per cluster) from a final layout (edge-adjacency)."""
    if comp.clu is None or comp.clu.size == 0:
        return 0
    eps = 1e-6
    vg = 0
    for cid in sorted({int(c) for c in comp.clu.tolist() if c > 0}):
        mem = [b for b in range(comp.n) if int(comp.clu[b]) == cid]
        m = len(mem)
        if m < 2:
            continue
        parent = list(range(m))

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]; a = parent[a]
            return a
        for ii in range(m):
            a = mem[ii]
            for jj in range(ii + 1, m):
                b = mem[jj]
                yov = min(by[a] + bh[a], by[b] + bh[b]) - max(by[a], by[b])
                xov = min(bx[a] + bw[a], bx[b] + bw[b]) - max(bx[a], bx[b])
                touch = ((abs(bx[a] + bw[a] - bx[b]) < eps or
                          abs(bx[b] + bw[b] - bx[a]) < eps) and yov > eps) or \
                        ((abs(by[a] + bh[a] - by[b]) < eps or
                          abs(by[b] + bh[b] - by[a]) < eps) and xov > eps)
                if touch:
                    ra, rb = find(ii), find(jj)
                    if ra != rb:
                        parent[rb] = ra
        vg += len({find(i) for i in range(m)}) - 1
    return vg


def _anneal_worker(args):
    """Seed (shelf-pack@W* for compact area, or spread for low HPWL), anneal under
    the outline penalty, finalize, and return the TRUE final cost. Multistart mixes
    both seed types and selects the winner per case."""
    comp, budget, seed, outline, method = args
    P0, N0 = initial_sp(comp, outline=outline, method=method)
    bP, bN, _, feas = anneal(comp, P0, N0, time_budget=budget, seed=seed,
                             outline=outline)
    ux, uy, res = finalize(comp, bP, bN)
    bx, by, bw, bh = expand(comp, ux, uy)
    ok = (res < 1e-6) and count_overlaps(bx, by, bw, bh) == 0
    vg = _grouping_from_pos(comp, bx, by, bw, bh)
    sc = score_layout(bx, by, bw, bh, comp.nets, comp.codes, comp.n_soft,
                      comp.hpwl_base, comp.area_base, v_group=vg)
    return (sc.cost, ux, uy, ok)


def _inv_posn(N, U):
    p = np.empty(U, dtype=np.int64)
    p[np.asarray(N, np.int64)] = np.arange(U, dtype=np.int64)
    return p


def _candidate_outlines(comp, area_star, k):
    """Target boxes (W*, H*) with W*.H* = area_star, using preplaced as a lower
    bound on the box and boundary+preplaced to pin a dimension when present.
    Returns up to k candidates (aspect sweep over the feasible range)."""
    A = max(area_star, 1e-9)
    # preplaced lower bounds (absolute coords) and boundary pins
    w_lo = h_lo = 0.0
    w_pin = h_pin = 0.0
    for u in range(comp.U):
        if not comp.pre[u]:
            continue
        rx = comp.px[u] + comp.uw[u]; ty = comp.py[u] + comp.uh[u]
        w_lo = max(w_lo, rx); h_lo = max(h_lo, ty)
        code = 0
        for (b, _, _) in comp.members[u]:
            code |= int(comp.codes[b])
        if code & 2:
            w_pin = max(w_pin, rx)
        if code & 4:
            h_pin = max(h_pin, ty)
    cands = []
    if w_pin > 0 and w_pin * w_pin <= A * 1.0001:
        cands.append((w_pin, A / w_pin))
    if h_pin > 0 and h_pin * h_pin <= A * 1.0001:
        cands.append((A / h_pin, h_pin))
    # aspect sweep r = W/H within the range allowed by the lower bounds
    r_lo = (w_lo * w_lo / A) if w_lo > 0 else 0.25
    r_hi = (A / (h_lo * h_lo)) if h_lo > 0 else 4.0
    r_lo = max(r_lo, 0.2); r_hi = min(max(r_hi, r_lo * 1.01), 5.0)
    import math as _m
    nsweep = max(1, k - len(cands))
    for t in range(nsweep):
        f = t / max(nsweep - 1, 1)
        r = _m.exp(_m.log(r_lo) * (1 - f) + _m.log(r_hi) * f)
        W = _m.sqrt(A * r); H = A / W
        if W >= w_lo - 1e-6 and H >= h_lo - 1e-6:
            cands.append((W, H))
    if not cands:
        s = A ** 0.5
        cands.append((max(s, w_lo), A / max(s, w_lo)))
    return cands[:k]


def boundary_repair(bx, by, bw, bh, codes, pre_mask):
    """Relocate boundary-violating blocks onto their required edge using existing
    whitespace (we have ~40% slack). For each violating block, find the lowest
    free slot in the edge column/row that fits, and move it there (overlap-safe;
    leaves a gap behind, which is fine). Preplaced are never moved. Returns new
    bx, by (copies)."""
    n = len(bx)
    bx = bx.copy(); by = by.copy()
    eps = 1e-6

    def free_at(i, x, y):
        x1i = x + bw[i]; y1i = y + bh[i]
        for j in range(n):
            if j == i:
                continue
            if (min(x1i, bx[j] + bw[j]) - max(x, bx[j]) > eps and
                    min(y1i, by[j] + bh[j]) - max(y, by[j]) > eps):
                return False
        return True

    def slots_along(lo, hi, fixed_is_x, fixed_val, size_fixed):
        """Candidate positions along [lo,hi] (edges of blocks intersecting the
        fixed band), lowest first (packs boundary blocks low -> tight area)."""
        cands = {lo}
        for j in range(n):
            if fixed_is_x:
                if bx[j] + bw[j] > fixed_val + eps and bx[j] < fixed_val + size_fixed - eps:
                    cands.add(by[j]); cands.add(by[j] + bh[j])
            else:
                if by[j] + bh[j] > fixed_val + eps and by[j] < fixed_val + size_fixed - eps:
                    cands.add(bx[j]); cands.add(bx[j] + bw[j])
        return sorted(c for c in cands if lo - eps <= c <= hi + eps)

    for _ in range(2):  # two passes (bbox may shift as blocks move)
        x0 = bx.min(); y0 = by.min(); x1 = (bx + bw).max(); y1 = (by + bh).max()
        for i in range(n):
            c = int(codes[i])
            if c == 0 or pre_mask[i]:
                continue
            tx = bx[i]; ty = by[i]
            need_x = None; need_y = None
            if c & 1:
                need_x = x0
            if c & 2:
                need_x = x1 - bw[i]
            if c & 8:
                need_y = y0
            if c & 4:
                need_y = y1 - bh[i]
            ok = ((need_x is None or abs(bx[i] - need_x) < eps) and
                  (need_y is None or abs(by[i] - need_y) < eps))
            if ok:
                continue
            placed = False
            if need_x is not None and need_y is not None:      # corner
                if free_at(i, need_x, need_y):
                    bx[i] = need_x; by[i] = need_y; placed = True
            elif need_x is not None:                            # left/right column
                for y in slots_along(y0, y1 - bh[i], True, need_x, bw[i]):
                    if free_at(i, need_x, y):
                        bx[i] = need_x; by[i] = y; placed = True; break
            elif need_y is not None:                            # bottom/top row
                for x in slots_along(x0, x1 - bw[i], False, need_y, bh[i]):
                    if free_at(i, x, need_y):
                        bx[i] = x; by[i] = need_y; placed = True; break
    return bx, by


def _constructive_floor(comp):
    """Deterministic feasible layout: preplaced at targets; movable units
    shelf-packed in the half-plane to the right of every obstacle."""
    U = comp.U
    ux = np.zeros(U); uy = np.zeros(U)
    x_clear = 0.0
    for u in range(U):
        if comp.pre[u]:
            ux[u] = comp.px[u]; uy[u] = comp.py[u]
            x_clear = max(x_clear, comp.px[u] + comp.uw[u])
    movable = [u for u in range(U) if not comp.pre[u]]
    movable.sort(key=lambda u: -comp.uh[u])
    total_w = sum(comp.uw[u] for u in movable)
    row_target = max(np.sqrt(max(total_w, 1.0) * max(comp.uh.max(), 1.0)),
                     comp.uw.max() if comp.uw.size else 1.0)
    x = x_clear; y = 0.0; row_h = 0.0
    for u in movable:
        if x > x_clear + 1e-9 and (x - x_clear) + comp.uw[u] > row_target:
            y += row_h; x = x_clear; row_h = 0.0
        ux[u] = x; uy[u] = y
        x += comp.uw[u]; row_h = max(row_h, comp.uh[u])
    return ux, uy


def finalize(comp, P, N, sweeps=12):
    posn = np.empty(comp.U, dtype=np.int64)
    for k, u in enumerate(N):
        posn[u] = k
    Pa = np.asarray(P, np.int64)
    ux, uy, res = pack(comp, Pa, posn)
    adj = build_adj(comp, Pa, posn)
    ux, uy = redistribute(comp, ux, uy, adj, sweeps=sweeps)
    # snap preplaced exact
    ux[comp.pre] = comp.px[comp.pre]
    uy[comp.pre] = comp.py[comp.pre]
    return ux, uy, res


class SPFloorplanner:
    def __init__(self, time_budget=2.0, verbose=False, seed=0, use_macros=False,
                 n_starts=8):
        self.time_budget = time_budget
        self.verbose = verbose
        self.seed = seed
        self.use_macros = use_macros
        self.n_starts = n_starts

    def solve_with_dims(self, n, b2b, p2b, pins_pos, constraints,
                        dims_wh, pre_xy, hpwl_base, area_base):
        comp = compile_with_targets(n, b2b, p2b, pins_pos, constraints,
                                    dims_wh, pre_xy, hpwl_base, area_base,
                                    use_macros=self.use_macros)
        if comp.U == 0:
            return [(0.0, 0.0, float(comp.bw[i]), float(comp.bh[i]))
                    for i in range(n)]

        # Parallel multistart: run n_starts anneals, each with an outline-shaped
        # seed + fixed-outline penalty; each worker finalizes and returns its TRUE
        # final cost, so we select the best finalized layout. Prewarm the JIT
        # in-parent so forked workers inherit compiled code (no recompile).
        k = max(1, int(self.n_starts))
        comp._fast = extract_arrays(comp)
        cl = max(float(np.sqrt(max(comp.area_base, 1e-9))), 1e-6)
        A = comp._fast
        P0, N0 = initial_sp(comp)
        energy_nb(np.asarray(P0, np.int64), _inv_posn(N0, comp.U),
                  A["uw"], A["uh"], A["pre"], A["px"], A["py"],
                  A["block_unit"], A["offx"], A["offy"], A["bw"], A["bh"],
                  A["codes"], A["b2b_i"], A["b2b_j"], A["b2b_w"], A["p2b_pin"],
                  A["p2b_blk"], A["p2b_w"], A["pin_x"], A["pin_y"],
                  A["hpwl_base"], A["area_base"], A["n_soft"], 25.0, cl,
                  A["cmem"], A["cptr"], 1e18, 1e18, 0.0)
        # fixed-outline targets (box area = area_base, aspects swept), with mixed
        # seeds: 'simpl' (analytic quadratic -> low wirelength) and 'shelf'
        # (fixed-width pack -> low area). Selection keeps the best per case.
        outlines = _candidate_outlines(comp, comp.area_base, max(1, k // 2))
        # analytic 'simpl' seed for most workers (low wirelength); a few 'shelf'
        # seeds (low area) for diversity. Selection keeps the best per case.
        tasks = [(comp, self.time_budget, self.seed + s,
                  outlines[s % len(outlines)],
                  "shelf" if s % 4 == 3 else "simpl") for s in range(k)]
        if k > 1:
            try:
                ctx = mp.get_context("fork")
                with ctx.Pool(min(k, mp.cpu_count())) as pool:
                    results = pool.map(_anneal_worker, tasks)
            except Exception:
                results = [_anneal_worker(t) for t in tasks]
        else:
            results = [_anneal_worker(tasks[0])]

        ok_res = [r for r in results if r[3]]

        # Outline-filling skyline candidates (cheap, deterministic): analytic seed
        # packed densely into each candidate width, boundary by construction.
        # Strong on area+boundary where SP-SA is weak; selection keeps the best.
        cxs, cys = _simpl_place(comp)
        for (W, Hh) in outlines:
            r = skyline_legalize(comp, cxs, cys, W, Htarget=Hh, tol=0.05)
            if r is None:
                continue
            sux, suy = r
            sbx, sby, sbw, sbh = expand(comp, sux, suy)
            if count_overlaps(sbx, sby, sbw, sbh) > 0:
                continue
            vg = _grouping_from_pos(comp, sbx, sby, sbw, sbh)
            ssc = score_layout(sbx, sby, sbw, sbh, comp.nets, comp.codes,
                               comp.n_soft, comp.hpwl_base, comp.area_base, v_group=vg)
            ok_res.append((ssc.cost, sux, suy, True))

        if ok_res:
            _, ux, uy, _ = min(ok_res, key=lambda r: r[0])
        else:
            ux, uy = _constructive_floor(comp)

        bx, by, bw, bh = expand(comp, ux, uy)
        # safety audit: if any real overlap slipped through, fall back to floor
        if count_overlaps(bx, by, bw, bh) > 0:
            ux, uy = _constructive_floor(comp)
            bx, by, bw, bh = expand(comp, ux, uy)

        # boundary repair: relocate boundary-violating blocks onto their edge
        # using whitespace; keep only if it improves the true cost.
        # don't relocate preplaced (hard) or clustered (would break grouping)
        pre_mask = np.zeros(n, dtype=bool)
        for u in np.where(comp.pre)[0]:
            for (b, _, _) in comp.members[u]:
                pre_mask[b] = True
        if comp.clu is not None:
            pre_mask |= (comp.clu > 0)
        def _cost(_bx, _by):
            vg = _grouping_from_pos(comp, _bx, _by, bw, bh)
            return score_layout(_bx, _by, bw, bh, comp.nets, comp.codes,
                                comp.n_soft, comp.hpwl_base, comp.area_base,
                                v_group=vg).cost
        rbx, rby = boundary_repair(bx, by, bw, bh, comp.codes, pre_mask)
        if count_overlaps(rbx, rby, bw, bh) == 0 and _cost(rbx, rby) < _cost(bx, by):
            bx, by = rbx, rby

        positions = [(float(bx[i]), float(by[i]), float(bw[i]), float(bh[i]))
                     for i in range(n)]
        return positions
