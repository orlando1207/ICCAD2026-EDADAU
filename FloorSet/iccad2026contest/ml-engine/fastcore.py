"""
Numba-JIT'd energy evaluation for the SP annealer.

The pure-Python pack (O(U^2) longest path) caps at ~1200 evals/s at U=96, far too
slow for SP-SA to converge on the large (heavily-weighted) cases. This module
JITs the whole evaluation -- pack -> expand -> HPWL -> bbox -> boundary -> cost --
into one nopython function so the annealer can run 10^5-10^6 evals/case.

Grouping (V_grouping) is NOT computed here: with rigid macros it is 0 by
construction; the official evaluator audits it with shapely at the end.
"""

import numpy as np
from numba import njit


def extract_arrays(comp):
    """Flatten a Compiled into numba-friendly arrays (built once per case)."""
    ne = comp.nets
    pins = ne.pins if ne.pins.size else np.zeros((1, 2))
    # cluster CSR (only groups with >=2 members contribute to V_grouping)
    cmem = []
    cptr = [0]
    if comp.clu is not None and comp.clu.size:
        for cid in sorted({int(c) for c in comp.clu.tolist() if c > 0}):
            mem = [b for b in range(comp.n) if int(comp.clu[b]) == cid]
            if len(mem) >= 2:
                cmem.extend(mem)
                cptr.append(len(cmem))
    return dict(
        uw=comp.uw.astype(np.float64),
        uh=comp.uh.astype(np.float64),
        pre=comp.pre.astype(np.bool_),
        px=comp.px.astype(np.float64),
        py=comp.py.astype(np.float64),
        block_unit=comp.block_unit.astype(np.int64),
        offx=comp.offx.astype(np.float64),
        offy=comp.offy.astype(np.float64),
        bw=comp.bw.astype(np.float64),
        bh=comp.bh.astype(np.float64),
        codes=comp.codes.astype(np.int64),
        b2b_i=ne.b2b_i.astype(np.int64),
        b2b_j=ne.b2b_j.astype(np.int64),
        b2b_w=ne.b2b_w.astype(np.float64),
        p2b_pin=ne.p2b_pin.astype(np.int64),
        p2b_blk=ne.p2b_blk.astype(np.int64),
        p2b_w=ne.p2b_w.astype(np.float64),
        pin_x=np.ascontiguousarray(pins[:, 0]).astype(np.float64),
        pin_y=np.ascontiguousarray(pins[:, 1]).astype(np.float64),
        hpwl_base=float(comp.hpwl_base),
        area_base=float(comp.area_base),
        n_soft=int(comp.n_soft),
        cmem=np.asarray(cmem, np.int64),
        cptr=np.asarray(cptr, np.int64),
    )


@njit(cache=True, fastmath=True)
def pack_nb(P, posn, uw, uh, pre, px, py):
    """Longest-path SP packing with preplaced pinning. Returns ux, uy, residual."""
    U = P.shape[0]
    ux = np.zeros(U)
    uy = np.zeros(U)
    residual = 0.0
    for k in range(U):
        j = P[k]
        xj = 0.0
        yj = 0.0
        pj = posn[j]
        for t in range(k):
            i = P[t]
            if posn[i] < pj:           # i is LEFT of j
                v = ux[i] + uw[i]
                if v > xj:
                    xj = v
            else:                      # i is BELOW j
                v = uy[i] + uh[i]
                if v > yj:
                    yj = v
        if pre[j]:
            if xj > px[j] + 1e-9:
                residual += xj - px[j]
            if yj > py[j] + 1e-9:
                residual += yj - py[j]
            xj = px[j]
            yj = py[j]
        ux[j] = xj
        uy[j] = yj
    return ux, uy, residual


@njit(cache=True, fastmath=True)
def _grouping_viol(cmem, cptr, bxl, byl, bw, bh):
    """Sum over clusters of (connected components - 1), where two members are
    connected if they share an edge of nonzero length (matches the evaluator)."""
    eps = 1e-6
    vg = 0
    nc = cptr.shape[0] - 1
    for c in range(nc):
        s = cptr[c]; e = cptr[c + 1]; m = e - s
        parent = np.arange(m)
        for ii in range(m):
            a = cmem[s + ii]
            ax0 = bxl[a]; ax1 = ax0 + bw[a]; ay0 = byl[a]; ay1 = ay0 + bh[a]
            for jj in range(ii + 1, m):
                b = cmem[s + jj]
                bx0 = bxl[b]; bx1 = bx0 + bw[b]; by0 = byl[b]; by1 = by0 + bh[b]
                yov = min(ay1, by1) - max(ay0, by0)
                xov = min(ax1, bx1) - max(ax0, bx0)
                touch = False
                if (abs(ax1 - bx0) < eps or abs(bx1 - ax0) < eps) and yov > eps:
                    touch = True
                if (abs(ay1 - by0) < eps or abs(by1 - ay0) < eps) and xov > eps:
                    touch = True
                if touch:
                    ra = ii
                    while parent[ra] != ra:
                        ra = parent[ra]
                    rb = jj
                    while parent[rb] != rb:
                        rb = parent[rb]
                    if ra != rb:
                        parent[rb] = ra
        comps = 0
        for ii in range(m):
            if parent[ii] == ii:
                comps += 1
        vg += comps - 1
    return vg


@njit(cache=True, fastmath=True)
def energy_nb(P, posn, uw, uh, pre, px, py,
              block_unit, offx, offy, bw, bh, codes,
              b2b_i, b2b_j, b2b_w, p2b_pin, p2b_blk, p2b_w, pin_x, pin_y,
              hpwl_base, area_base, n_soft, w_res, charlen, cmem, cptr,
              wstar, hstar, w_out):
    ux, uy, residual = pack_nb(P, posn, uw, uh, pre, px, py)
    n = block_unit.shape[0]
    cx = np.empty(n)
    cy = np.empty(n)
    bxl = np.empty(n)
    byl = np.empty(n)
    x0 = 1e18; y0 = 1e18; x1 = -1e18; y1 = -1e18
    for b in range(n):
        u = block_unit[b]
        lx = ux[u] + offx[b]
        ly = uy[u] + offy[b]
        bxl[b] = lx; byl[b] = ly
        cx[b] = lx + bw[b] * 0.5
        cy[b] = ly + bh[b] * 0.5
        if lx < x0:
            x0 = lx
        if ly < y0:
            y0 = ly
        if lx + bw[b] > x1:
            x1 = lx + bw[b]
        if ly + bh[b] > y1:
            y1 = ly + bh[b]
    area = (x1 - x0) * (y1 - y0)

    hp = 0.0
    for e in range(b2b_i.shape[0]):
        i = b2b_i[e]; j = b2b_j[e]
        hp += b2b_w[e] * (abs(cx[i] - cx[j]) + abs(cy[i] - cy[j]))
    for e in range(p2b_blk.shape[0]):
        b = p2b_blk[e]; p = p2b_pin[e]
        hp += p2b_w[e] * (abs(pin_x[p] - cx[b]) + abs(pin_y[p] - cy[b]))

    vb = 0
    eps = 1e-6
    for b in range(n):
        c = codes[b]
        if c == 0:
            continue
        ok = True
        if (c & 1) and abs(bxl[b] - x0) >= eps:
            ok = False
        if ok and (c & 2) and abs(bxl[b] + bw[b] - x1) >= eps:
            ok = False
        if ok and (c & 4) and abs(byl[b] + bh[b] - y1) >= eps:
            ok = False
        if ok and (c & 8) and abs(byl[b] - y0) >= eps:
            ok = False
        if not ok:
            vb += 1

    hgap = (hp - hpwl_base) / (hpwl_base if hpwl_base > 1e-6 else 1e-6)
    if hgap < 0.0:
        hgap = 0.0
    agap = (area - area_base) / (area_base if area_base > 1e-6 else 1e-6)
    if agap < 0.0:
        agap = 0.0
    vg = 0
    if cptr.shape[0] > 1:
        vg = _grouping_viol(cmem, cptr, bxl, byl, bw, bh)
    nsd = n_soft if n_soft > 0 else 1
    vrel = (vb + vg) / nsd
    cost = (1.0 + 0.5 * (hgap + agap)) * np.exp(2.0 * vrel)
    e = cost
    if residual > 1e-9:
        e += w_res * residual / charlen
    # fixed-outline penalty: discourage exceeding the target box W* x H*
    # (= area_base, derived from preplaced/boundary). Drives compaction into the
    # known optimal outline -> area_gap -> 0. Additive to the SA energy only;
    # the returned `cost` is the true contest cost (used for selection).
    if w_out > 0.0:
        W = x1 - x0; Hh = y1 - y0
        ov = 0.0
        if W > wstar:
            ov += (W - wstar) / wstar
        if Hh > hstar:
            ov += (Hh - hstar) / hstar
        e += w_out * ov
    return e, cost, residual, vb
