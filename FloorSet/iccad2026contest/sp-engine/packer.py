"""
Sequence-pair geometry realization + wirelength-driven slack redistribution.

Convention (Wong): for units a,b with Gamma+ position and Gamma- position posn:
  a before b in BOTH  -> a is LEFT of b   (x_a + w_a <= x_b)
  a before b in Gamma+, after in Gamma-   -> a is BELOW b (y_a + h_a <= y_b)
Both HCG and VCG edges go forward in Gamma+ order, so Gamma+ order is a valid
topological order for the longest-path packing.

Preplaced units are pinned: their coord is forced to target; the longest path
routes neighbours around them. If a unit cannot fit before a pinned obstacle the
residual overlap is returned and penalised by the annealer.

Redistribution moves each unit within bounds derived from the CURRENT positions
of its constraint-graph neighbours (Gauss-Seidel), which guarantees the layout
stays overlap-free while reducing HPWL. The bbox never grows.
"""

import numpy as np

from compile_problem import expand


def pack(comp, gamma_p, posn):
    """Longest-path packing. Returns ux, uy (lower-left/unit), residual."""
    U = comp.U
    uw = comp.uw; uh = comp.uh
    x = np.zeros(U); y = np.zeros(U)
    residual = 0.0
    for k in range(U):
        j = gamma_p[k]
        if k == 0:
            xj = yj = 0.0
        else:
            earlier = gamma_p[:k]
            left_mask = posn[earlier] < posn[j]
            le = earlier[left_mask]
            be = earlier[~left_mask]
            xj = float(np.max(x[le] + uw[le])) if le.size else 0.0
            yj = float(np.max(y[be] + uh[be])) if be.size else 0.0
        if comp.pre[j]:
            if xj > comp.px[j] + 1e-9:
                residual += xj - comp.px[j]
            if yj > comp.py[j] + 1e-9:
                residual += yj - comp.py[j]
            xj = comp.px[j]; yj = comp.py[j]
        x[j] = xj; y[j] = yj
    return x, y, residual


def build_adj(comp, gamma_p, posn):
    """HCG/VCG predecessor & successor lists per unit (arrays of unit ids)."""
    U = comp.U
    hpred = [[] for _ in range(U)]
    hsucc = [[] for _ in range(U)]
    vpred = [[] for _ in range(U)]
    vsucc = [[] for _ in range(U)]
    for a in range(U):
        ua = gamma_p[a]
        for b in range(a + 1, U):
            ub = gamma_p[b]
            if posn[ua] < posn[ub]:   # ua LEFT of ub
                hsucc[ua].append(ub); hpred[ub].append(ua)
            else:                     # ua BELOW ub
                vsucc[ua].append(ub); vpred[ub].append(ua)
    to_arr = lambda L: [np.asarray(a, np.int64) for a in L]
    return to_arr(hpred), to_arr(hsucc), to_arr(vpred), to_arr(vsucc)


def _wmedian(weights, vals):
    if vals.size == 0:
        return None
    order = np.argsort(vals)
    v = vals[order]; w = weights[order]
    cw = np.cumsum(w)
    k = int(np.searchsorted(cw, cw[-1] / 2.0))
    return float(v[min(k, v.size - 1)])


def redistribute(comp, ux, uy, adj, sweeps=6):
    """Gauss-Seidel weighted-median placement within dynamic no-overlap bounds."""
    ux = ux.copy(); uy = uy.copy()
    hpred, hsucc, vpred, vsucc = adj
    U = comp.U
    uw = comp.uw; uh = comp.uh
    deg = np.array([comp.red_w[u].sum() if comp.red_w[u].size else 0.0
                    for u in range(U)])
    order = np.argsort(-deg)
    pins = comp.nets.pins
    # per-axis freeze: keep boundary blocks on their edge (left/right freeze x;
    # top/bottom freeze y); preplaced freeze both. Wirelength still tunes the
    # free axis without breaking the boundary touch SA achieved.
    freeze_x = comp.pre.copy()
    freeze_y = comp.pre.copy()
    import os as _os
    _bfreeze = _os.environ.get("SPFP_BFREEZE", "1") != "0"
    if _bfreeze:
        for u in range(U):
            code = 0
            for (b, _, _) in comp.members[u]:
                code |= int(comp.codes[b])
            if code & 3:
                freeze_x[u] = True
            if code & 12:
                freeze_y[u] = True
    for _ in range(sweeps):
        bx, by, bw, bh = expand(comp, ux, uy)
        cx = bx + bw / 2.0; cy = by + bh / 2.0
        W = float(np.max(ux + uw)); H = float(np.max(uy + uh))
        for u in order:
            if comp.pre[u]:
                continue
            w = comp.red_w[u]
            if w.size == 0:
                continue
            # dynamic bounds from current neighbour positions
            hp = hpred[u]; hs = hsucc[u]; vp = vpred[u]; vs = vsucc[u]
            xlo = float(np.max(ux[hp] + uw[hp])) if hp.size else 0.0
            xhi = (float(np.min(ux[hs])) - uw[u]) if hs.size else (W - uw[u])
            ylo = float(np.max(uy[vp] + uh[vp])) if vp.size else 0.0
            yhi = (float(np.min(uy[vs])) - uh[u]) if vs.size else (H - uh[u])
            if xhi < xlo:
                xhi = xlo
            if yhi < ylo:
                yhi = ylo
            oth = comp.red_oth[u]
            blk = oth >= 0
            ocx = np.empty(oth.size); ocy = np.empty(oth.size)
            if blk.any():
                ocx[blk] = cx[oth[blk]]; ocy[blk] = cy[oth[blk]]
            if (~blk).any():
                pid = -oth[~blk] - 1
                ocx[~blk] = pins[pid, 0]; ocy[~blk] = pins[pid, 1]
            tx = _wmedian(w, ocx - comp.red_mox[u])
            ty = _wmedian(w, ocy - comp.red_moy[u])
            if tx is not None and not freeze_x[u]:
                nx = min(max(tx, xlo), xhi)
                if abs(nx - ux[u]) > 1e-12:
                    mids = [b for (b, _, _) in comp.members[u]]
                    cx[mids] += nx - ux[u]; ux[u] = nx
            if ty is not None and not freeze_y[u]:
                ny = min(max(ty, ylo), yhi)
                if abs(ny - uy[u]) > 1e-12:
                    mids = [b for (b, _, _) in comp.members[u]]
                    cy[mids] += ny - uy[u]; uy[u] = ny
    return ux, uy
