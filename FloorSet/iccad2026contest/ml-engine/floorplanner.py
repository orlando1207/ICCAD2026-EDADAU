"""
SP floorplanner orchestrator.

Pipeline (given fixed block dims):
  compile -> warm-start SP -> simulated annealing -> slack redistribution
  -> snap preplaced exact -> emit positions.

Guarantees: always returns a geometrically feasible layout (overlap-free,
preplaced exact). If SA never finds a feasible topology, a constructive floor
(movable units shelf-packed clear of the preplaced obstacles) is used.

Shape awareness: each SA worker independently optimises block aspect ratios
(for free singleton units) via 7-discrete-ratio moves. The final shapes are
synced back to the worker's comp before finalize(), so bx/by/bw/bh returned
by the worker are always consistent with the actual placed dimensions. The
parent NEVER calls expand() on the best worker result — it uses the worker's
already-expanded arrays directly to avoid the shape-mismatch bug.
"""

import math
import multiprocessing as mp

import numpy as np

from compile_problem import compile_with_targets, expand, dims_from_areas
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


def _apply_shape_to_comp(comp, shape_r):
    """Sync final aspect ratios from anneal() back to comp before finalize().

    shape_r[u] is the final w/h ratio for unit u.  Only free singleton units
    (rot_ok=True) may differ from the initial 1:1 ratio.  area is preserved:
      new_w = sqrt(area_u * r),  new_h = sqrt(area_u / r)
    where area_u = comp.uw[u] * comp.uh[u] (original, unmodified by SA).
    """
    area_u = comp.uw * comp.uh   # comp was NOT modified by anneal (A is a copy)
    changed = False
    for u in range(comp.U):
        if not comp.rot_ok[u] or len(comp.members[u]) != 1:
            continue
        r = float(shape_r[u])
        new_w = math.sqrt(max(area_u[u] * r, 1e-12))
        new_h = math.sqrt(max(area_u[u] / r, 1e-12))
        if abs(new_w - comp.uw[u]) > 1e-9:
            comp.uw[u] = new_w; comp.uh[u] = new_h
            b = comp.members[u][0][0]
            comp.bw[b] = new_w; comp.bh[b] = new_h
            changed = True
    if changed:
        comp.build_redistribution()


def _anneal_worker(args):
    """Seed, anneal, finalize; return (cost, bx, by, ok, bw, bh) 6-tuple.

    bw/bh reflect the worker's final block shapes (possibly reshaped vs. the
    parent's original comp).  The parent must NOT call expand(comp,...) on the
    selected result; it uses bx/by/bw/bh directly.
    """
    comp, budget, seed, outline, method, enable_rotation = args
    P0, N0 = initial_sp(comp, outline=outline, method=method)
    bP, bN, _, feas, shape_r = anneal(
        comp, P0, N0, time_budget=budget, seed=seed,
        outline=outline, enable_rotation=enable_rotation)
    _apply_shape_to_comp(comp, shape_r)   # sync final shapes to comp
    ux, uy, res = finalize(comp, bP, bN)
    bx, by, bw, bh = expand(comp, ux, uy)
    ok = (res < 1e-6) and count_overlaps(bx, by, bw, bh) == 0
    vg = _grouping_from_pos(comp, bx, by, bw, bh)
    sc = score_layout(bx, by, bw, bh, comp.nets, comp.codes, comp.n_soft,
                      comp.hpwl_base, comp.area_base, v_group=vg)
    return (sc.cost, bx.copy(), by.copy(), ok, bw.copy(), bh.copy())


def _inv_posn(N, U):
    p = np.empty(U, dtype=np.int64)
    p[np.asarray(N, np.int64)] = np.arange(U, dtype=np.int64)
    return p


def _candidate_outlines(comp, area_star, k):
    """Target boxes (W*, H*) with W*.H* ≈ area_star, swept over feasible aspects."""
    A = max(area_star, 1e-9)
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
    r_lo = (w_lo * w_lo / A) if w_lo > 0 else 0.25
    r_hi = (A / (h_lo * h_lo)) if h_lo > 0 else 4.0
    r_lo = max(r_lo, 0.2); r_hi = min(max(r_hi, r_lo * 1.01), 5.0)
    nsweep = max(1, k - len(cands))
    for t in range(nsweep):
        f = t / max(nsweep - 1, 1)
        r = math.exp(math.log(r_lo) * (1 - f) + math.log(r_hi) * f)
        W = math.sqrt(A * r); H = A / W
        if W >= w_lo - 1e-6 and H >= h_lo - 1e-6:
            cands.append((W, H))
    if not cands:
        s = A ** 0.5
        cands.append((max(s, w_lo), A / max(s, w_lo)))
    return cands[:k]


def boundary_repair(bx, by, bw, bh, codes, pre_mask):
    """Relocate boundary-violating blocks onto their required edge using whitespace."""
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
        cands = {lo}
        for j in range(n):
            if fixed_is_x:
                if bx[j] + bw[j] > fixed_val + eps and bx[j] < fixed_val + size_fixed - eps:
                    cands.add(by[j]); cands.add(by[j] + bh[j])
            else:
                if by[j] + bh[j] > fixed_val + eps and by[j] < fixed_val + size_fixed - eps:
                    cands.add(bx[j]); cands.add(bx[j] + bw[j])
        return sorted(c for c in cands if lo - eps <= c <= hi + eps)

    for _ in range(6):
        x0 = bx.min(); y0 = by.min(); x1 = (bx + bw).max(); y1 = (by + bh).max()
        for i in range(n):
            c = int(codes[i])
            if c == 0 or pre_mask[i]:
                continue
            need_x = None; need_y = None
            if c & 1:  need_x = x0
            if c & 2:  need_x = x1 - bw[i]
            if c & 8:  need_y = y0
            if c & 4:  need_y = y1 - bh[i]
            ok = ((need_x is None or abs(bx[i] - need_x) < eps) and
                  (need_y is None or abs(by[i] - need_y) < eps))
            if ok:
                continue
            if need_x is not None and need_y is not None:
                # corner block: only the exact corner satisfies both constraints
                if free_at(i, need_x, need_y):
                    bx[i] = need_x; by[i] = need_y
            elif need_x is not None:
                for y in slots_along(y0, y1 - bh[i], True, need_x, bw[i]):
                    if free_at(i, need_x, y):
                        bx[i] = need_x; by[i] = y; break
            elif need_y is not None:
                for x in slots_along(x0, x1 - bw[i], False, need_y, bh[i]):
                    if free_at(i, x, need_y):
                        bx[i] = x; by[i] = need_y; break
    return bx, by


def _constructive_floor(comp):
    """Deterministic feasible layout: preplaced at targets; movable shelf-packed."""
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
    ux[comp.pre] = comp.px[comp.pre]
    uy[comp.pre] = comp.py[comp.pre]
    return ux, uy, res


class SPFloorplanner:
    def __init__(self, time_budget=2.0, verbose=False, seed=0, use_macros=False,
                 n_starts=8, enable_rotation=True):
        self.time_budget = time_budget
        self.verbose = verbose
        self.seed = seed
        self.use_macros = use_macros
        self.n_starts = n_starts
        self.enable_rotation = enable_rotation

    def solve_with_dims(self, n, b2b, p2b, pins_pos, constraints,
                        dims_wh, pre_xy, hpwl_base, area_base):
        """Run SP+SA given explicit block dimensions."""
        comp = compile_with_targets(n, b2b, p2b, pins_pos, constraints,
                                    dims_wh, pre_xy, hpwl_base, area_base,
                                    use_macros=self.use_macros)
        if comp.U == 0:
            return [(0.0, 0.0, float(comp.bw[i]), float(comp.bh[i]))
                    for i in range(n)]

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

        outlines = _candidate_outlines(comp, comp.area_base, max(1, k // 2))
        tasks = [(comp, self.time_budget, self.seed + s,
                  outlines[s % len(outlines)],
                  "shelf" if s % 4 == 3 else "simpl",
                  self.enable_rotation) for s in range(k)]

        if k > 1:
            try:
                ctx = mp.get_context("fork")
                with ctx.Pool(min(k, mp.cpu_count())) as pool:
                    results = pool.map(_anneal_worker, tasks)
            except Exception:
                results = [_anneal_worker(t) for t in tasks]
        else:
            results = [_anneal_worker(tasks[0])]

        # Workers return (cost, bx, by, ok, bw, bh) — shapes already applied.
        ok_res = [r for r in results if r[3]]

        # Skyline candidates use original comp dims (no shape modification).
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
                               comp.n_soft, comp.hpwl_base, comp.area_base,
                               v_group=vg)
            ok_res.append((ssc.cost, sbx.copy(), sby.copy(), True,
                           sbw.copy(), sbh.copy()))

        if ok_res:
            _, bx, by, _, bw, bh = min(ok_res, key=lambda r: r[0])
        else:
            ux, uy = _constructive_floor(comp)
            bx, by, bw, bh = expand(comp, ux, uy)

        if count_overlaps(bx, by, bw, bh) > 0:
            ux, uy = _constructive_floor(comp)
            bx, by, bw, bh = expand(comp, ux, uy)

        # boundary repair — only lock truly preplaced blocks (not cluster members)
        pre_mask = np.zeros(n, dtype=bool)
        for u in np.where(comp.pre)[0]:
            for (b, _, _) in comp.members[u]:
                pre_mask[b] = True

        def _cost(_bx, _by):
            vg = _grouping_from_pos(comp, _bx, _by, bw, bh)
            return score_layout(_bx, _by, bw, bh, comp.nets, comp.codes,
                                comp.n_soft, comp.hpwl_base, comp.area_base,
                                v_group=vg).cost

        rbx, rby = boundary_repair(bx, by, bw, bh, comp.codes, pre_mask)
        if count_overlaps(rbx, rby, bw, bh) == 0 and _cost(rbx, rby) < _cost(bx, by):
            bx, by = rbx, rby

        return [(float(bx[i]), float(by[i]), float(bw[i]), float(bh[i]))
                for i in range(n)]

    def solve(self, n, area_targets, b2b, p2b, pins_pos, constraints,
              target_positions=None):
        """Contest API: no GT dims — derives shapes from area_targets.

        Free blocks → w = h = sqrt(area).  Fixed-shape / preplaced blocks use
        target_positions.  Estimates hpwl_base / area_base from a quick analytic
        placement so the SA cost function is properly normalized.
        """
        dims_wh, pre_xy = dims_from_areas(
            n, area_targets, constraints, target_positions, aspect=1.0)

        comp_q = compile_with_targets(n, b2b, p2b, pins_pos, constraints,
                                      dims_wh, pre_xy, 1.0, 1.0,
                                      use_macros=False)
        cx, cy = _simpl_place(comp_q)
        ne = comp_q.nets
        bwq = comp_q.bw; bhq = comp_q.bh; buq = comp_q.block_unit
        hp = 0.0
        for k in range(ne.b2b_i.size):
            i = int(ne.b2b_i[k]); j = int(ne.b2b_j[k]); w = float(ne.b2b_w[k])
            hp += w * (abs(float(cx[buq[i]]) + float(bwq[i]) * 0.5 -
                           float(cx[buq[j]]) - float(bwq[j]) * 0.5) +
                       abs(float(cy[buq[i]]) + float(bhq[i]) * 0.5 -
                           float(cy[buq[j]]) - float(bhq[j]) * 0.5))
        pins_arr = ne.pins
        for k in range(ne.p2b_blk.size):
            b_idx = int(ne.p2b_blk[k]); p = int(ne.p2b_pin[k]); w = float(ne.p2b_w[k])
            hp += w * (abs(float(pins_arr[p, 0]) -
                           float(cx[buq[b_idx]]) - float(bwq[b_idx]) * 0.5) +
                       abs(float(pins_arr[p, 1]) -
                           float(cy[buq[b_idx]]) - float(bhq[b_idx]) * 0.5))
        # Connectivity-based hpwl estimate: empirically GT_hpwl ≈ 3.0 × total_w ×
        # sqrt(area_per_block).  The analytic placement underestimates for
        # highly-connected cases (blocks co-locate in the unconstrained solution);
        # this formula keeps the SA energy calibrated relative to GT.
        total_w = ((float(np.sum(ne.b2b_w)) if ne.b2b_i.size > 0 else 0.0) +
                   (float(np.sum(ne.p2b_w)) if ne.p2b_blk.size > 0 else 0.0))
        avg_sqrt_area = float(np.mean(np.sqrt(dims_wh[:n, 0] * dims_wh[:n, 1])))
        hpwl_conn = total_w * 3.0 * avg_sqrt_area
        hpwl_est = max(hp, hpwl_conn, 1.0)
        # Fill fraction is ~97% so block_area_sum ≈ 0.97 × GT_area — close enough.
        area_est = float(np.sum(dims_wh[:n, 0] * dims_wh[:n, 1]))

        return self.solve_with_dims(n, b2b, p2b, pins_pos, constraints,
                                    dims_wh, pre_xy, hpwl_est, area_est)
