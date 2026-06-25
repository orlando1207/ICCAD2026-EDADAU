"""
Warm start: spread global placement -> edge-relation legalization -> sequence
pair extracted from the legal layout's constraint graph.

Key finding (validated): a sequence pair extracted from CENTER sorts (cx+/-cy)
produces staircase topologies and is a terrible SA seed. Extracting the SP from
a SPREAD, legalized layout's geometric left/below relations gives a near-tight
seed from which SA converges quickly. Force-directed alone clumps (bad order);
alternating area-equalization spreading fixes the order before legalization.
"""

import heapq

import numpy as np


def _unit_adj(comp):
    U = comp.U
    nbr = [[] for _ in range(U)]
    ne = comp.nets
    for k in range(ne.b2b_i.size):
        u = int(comp.block_unit[ne.b2b_i[k]]); v = int(comp.block_unit[ne.b2b_j[k]])
        if u != v:
            w = float(ne.b2b_w[k]); nbr[u].append((v, w)); nbr[v].append((u, w))
    up = [[] for _ in range(U)]
    for k in range(ne.p2b_blk.size):
        u = int(comp.block_unit[ne.p2b_blk[k]])
        up[u].append((int(ne.p2b_pin[k]), float(ne.p2b_w[k])))
    return nbr, up


def _spread(comp, iters=40, outline=None):
    """Force-directed attraction blended with ramped area-equalization spread.
    If `outline=(W*,H*)` is given, x/y are equalized over [0,W*] and [0,H*] so
    the seed already has the target aspect/extent (SA then starts in the right
    basin instead of a loose square)."""
    U = comp.U
    w = comp.uw; h = comp.uh; area = w * h
    nbr, up = _unit_adj(comp)
    pins = comp.nets.pins
    s = float(np.sqrt(area.sum())) * 1.05
    spanx, spany = (float(outline[0]), float(outline[1])) if outline else (s, s)
    pc = pins.mean(0) if pins.size else np.array([spanx / 2, spany / 2])
    cx = np.full(U, pc[0]); cy = np.full(U, pc[1])
    pre = comp.pre
    cx[pre] = comp.px[pre] + w[pre] / 2.0
    cy[pre] = comp.py[pre] + h[pre] / 2.0
    tot = area.sum()

    def equalize(c, span):
        o = np.argsort(c); tgt = np.empty_like(c); cc = 0.0
        for i in o:
            tgt[i] = (cc + area[i] / 2) / tot * span; cc += area[i]
        return tgt

    for it in range(iters):
        lam = it / max(iters - 1, 1)
        for u in range(U):
            if pre[u]:
                continue
            sw = sx = sy = 0.0
            for (v, ww) in nbr[u]:
                sw += ww; sx += ww * cx[v]; sy += ww * cy[v]
            for (p, ww) in up[u]:
                sw += ww; sx += ww * pins[p, 0]; sy += ww * pins[p, 1]
            if sw > 0:
                cx[u] = sx / sw; cy[u] = sy / sw
        tx = equalize(cx, spanx); ty = equalize(cy, spany); b = 0.5 * lam; m = ~pre
        cx[m] = (1 - b) * cx[m] + b * tx[m]
        cy[m] = (1 - b) * cy[m] + b * ty[m]
    return cx, cy


def _build_laplacian(comp):
    """Weighted Laplacian of the unit netlist + pin (b2b clique-as-edges, p2b as
    diagonal anchors). Returns L (U x U), bx, by (pin-driven RHS)."""
    U = comp.U
    ne = comp.nets
    L = np.zeros((U, U))
    bx = np.zeros(U); by = np.zeros(U)
    bu = comp.block_unit
    for k in range(ne.b2b_i.size):
        i = int(bu[ne.b2b_i[k]]); j = int(bu[ne.b2b_j[k]]); w = float(ne.b2b_w[k])
        if i == j:
            continue
        L[i, i] += w; L[j, j] += w; L[i, j] -= w; L[j, i] -= w
    for k in range(ne.p2b_blk.size):
        u = int(bu[ne.p2b_blk[k]]); p = int(ne.p2b_pin[k]); w = float(ne.p2b_w[k])
        L[u, u] += w; bx[u] += w * ne.pins[p, 0]; by[u] += w * ne.pins[p, 1]
    return L, bx, by


def _l1_solve(comp, irls_iters=6):
    """L1 (Manhattan) wirelength placement via IRLS, matching the contest HPWL
    objective (minimize sum w|ci-cj|, separable in x/y). Preplaced are fixed
    anchors. Each IRLS round reweights edges by 1/max(|dist|,delta) and re-solves
    the weighted-Laplacian least squares. The L2 solution is the first iterate."""
    U = comp.U
    ne = comp.nets
    bu = comp.block_unit
    # unit-level edges (b2b)
    ei = bu[ne.b2b_i].astype(np.int64)
    ej = bu[ne.b2b_j].astype(np.int64)
    ew = ne.b2b_w.astype(float)
    keep = ei != ej
    ei, ej, ew = ei[keep], ej[keep], ew[keep]
    pu = bu[ne.p2b_blk].astype(np.int64)
    pw = ne.p2b_w.astype(float)
    ppx = ne.pins[ne.p2b_pin, 0]; ppy = ne.pins[ne.p2b_pin, 1]

    pre = comp.pre
    free = np.where(~pre)[0]
    pinned = np.where(pre)[0]
    cx = np.zeros(U); cy = np.zeros(U)
    cx[pinned] = comp.px[pinned] + comp.uw[pinned] / 2.0
    cy[pinned] = comp.py[pinned] + comp.uh[pinned] / 2.0
    if free.size == 0:
        return cx, cy
    fidx = -np.ones(U, np.int64); fidx[free] = np.arange(free.size)
    delta = 1e-2

    def build_and_solve(wx, wpin):
        # assemble weighted Laplacian over free units for one axis pair (shared)
        L = np.zeros((free.size, free.size))
        bxv = np.zeros(free.size); byv = np.zeros(free.size)
        for k in range(ei.size):
            a, b, w = ei[k], ej[k], wx[k]
            fa, fb = fidx[a], fidx[b]
            if fa >= 0:
                L[fa, fa] += w
                if fb >= 0:
                    L[fa, fb] -= w
                else:
                    bxv[fa] += w * cx[b]; byv[fa] += w * cy[b]
            if fb >= 0:
                L[fb, fb] += w
                if fa >= 0:
                    L[fb, fa] -= w
                else:
                    bxv[fb] += w * cx[a]; byv[fb] += w * cy[a]
        for k in range(pu.size):
            u = pu[k]; fu = fidx[u]
            if fu >= 0:
                w = wpin[k]
                L[fu, fu] += w; bxv[fu] += w * ppx[k]; byv[fu] += w * ppy[k]
        L += np.eye(free.size) * 1e-6
        return np.linalg.solve(L, bxv), np.linalg.solve(L, byv)

    wx = ew.copy(); wpin = pw.copy()
    for it in range(irls_iters):
        nx, ny = build_and_solve(wx, wpin)
        cx[free] = nx; cy[free] = ny
        # reweight (L1): w / |dist| per axis -- use combined for stability
        dx = np.abs(cx[ei] - cx[ej]); dy = np.abs(cy[ei] - cy[ej])
        wx = ew / np.maximum(dx + dy, delta)
        dpx = np.abs(cx[pu] - ppx); dpy = np.abs(cy[pu] - ppy)
        wpin = pw / np.maximum(dpx + dpy, delta)
    return cx, cy


def _bisect_targets(comp, cx, cy, box, free):
    """SimPL look-ahead legalization via recursive geometric bisection: split the
    box by its longer dimension, partition cells (sorted by position along that
    dim) into the two halves balancing AREA, cut the box proportionally, recurse.
    Returns per-unit target positions that fill the box uniformly-by-area while
    preserving 2D locality (connected cells stay together)."""
    area = comp.uw * comp.uh
    tx = cx.copy(); ty = cy.copy()
    stack = [(list(free), box)]
    while stack:
        idx, (x0, y0, x1, y1) = stack.pop()
        m = len(idx)
        if m == 0:
            continue
        if m == 1:
            u = idx[0]; tx[u] = (x0 + x1) / 2.0; ty[u] = (y0 + y1) / 2.0
            continue
        w = x1 - x0; h = y1 - y0
        horiz = w >= h
        idx.sort(key=(lambda u: cx[u]) if horiz else (lambda u: cy[u]))
        tot = sum(area[u] for u in idx)
        acc = 0.0; sp = 1
        half = tot / 2.0
        for k, u in enumerate(idx):
            acc += area[u]
            if acc >= half:
                sp = k + 1; break
        sp = max(1, min(sp, m - 1))
        frac = sum(area[u] for u in idx[:sp]) / tot
        if horiz:
            xc = x0 + frac * w
            stack.append((idx[:sp], (x0, y0, xc, y1)))
            stack.append((idx[sp:], (xc, y0, x1, y1)))
        else:
            yc = y0 + frac * h
            stack.append((idx[:sp], (x0, y0, x1, yc)))
            stack.append((idx[sp:], (x0, yc, x1, y1)))
    return tx, ty


def _simpl_proper(comp, outline, iters=12):
    """SimPL: quadratic lower bound + recursive-bisection upper bound + anchored
    re-solves with growing spring. Produces a spread, low-wirelength placement
    filling the target outline -- a clean (low-overlap) basis for legalization."""
    U = comp.U
    L, bx, by = _build_laplacian(comp)
    area = comp.uw * comp.uh
    A = float(area.sum()) + 1e-9
    W, Hh = (float(outline[0]), float(outline[1])) if outline else (A ** 0.5, A ** 0.5)
    pre = comp.pre
    free = np.where(~pre)[0]; pinned = np.where(pre)[0]
    cx = np.zeros(U); cy = np.zeros(U)
    cx[pinned] = comp.px[pinned] + comp.uw[pinned] / 2.0
    cy[pinned] = comp.py[pinned] + comp.uh[pinned] / 2.0
    if free.size == 0:
        return cx, cy
    Lff = L[np.ix_(free, free)]
    Lfp = L[np.ix_(free, pinned)] if pinned.size else None
    bxf = bx[free] - (Lfp @ cx[pinned] if pinned.size else 0.0)
    byf = by[free] - (Lfp @ cy[pinned] if pinned.size else 0.0)
    Iv = np.eye(free.size)
    base = max(float(np.median(np.diag(Lff))), 1e-6)
    cx[free] = np.linalg.solve(Lff + Iv * 1e-6, bxf)
    cy[free] = np.linalg.solve(Lff + Iv * 1e-6, byf)
    for it in range(iters):
        tx, ty = _bisect_targets(comp, cx, cy, (0.0, 0.0, W, Hh), free)
        alpha = base * (0.5 + 3.5 * it / max(iters - 1, 1))
        Amat = Lff + Iv * alpha
        cx[free] = np.linalg.solve(Amat, bxf + alpha * tx[free])
        cy[free] = np.linalg.solve(Amat, byf + alpha * ty[free])
    return cx, cy


def _simpl_place(comp, outline=None, iters=0, alpha_max=4.0):
    """Analytic placement seed for SP-SA: L2 quadratic wirelength solve (preplaced
    anchors). Returns near-GT-wirelength clumped positions; the SP packer + SA +
    outline penalty spread them. (Uniform-density spreading -- equalization or
    SimPL recursive bisection -- was tried and HURTS: it destroys wirelength,
    because near-100%-density tiling needs a specific arrangement, not a uniform
    spread. Feeding the clumped quadratic straight into SP-SA is better.)"""
    return _l1_solve(comp, irls_iters=1)


def _shelf_pack(comp, W, order):
    """Skyline strip-pack units into a fixed width W (height grows up), placing
    each unit (in `order`) at the lowest feasible position. Produces a compact
    layout with x-extent <= W by construction -> the SP extracted from it has
    short horizontal chains, which is exactly what SA fails to find from a loose
    spread. Preplaced fixedness is ignored here (this is a topology seed; the SP
    packer pins preplaced to their real coords afterwards)."""
    U = comp.U
    uw = comp.uw; uh = comp.uh
    W = max(W, float(uw.max()) + 1e-6)
    segs = [[0.0, W, 0.0]]   # contiguous [xs, xe, height] tiling [0, W]
    ux = np.zeros(U); uy = np.zeros(U)

    def height_over(xs, xe):
        m = 0.0
        for s, e, h in segs:
            if e <= xs + 1e-12 or s >= xe - 1e-12:
                continue
            if h > m:
                m = h
        return m

    def place(xpos, w, top):
        # raise skyline over [xpos, xpos+w] to `top`
        xe = xpos + w
        ns = []
        for s, e, h in segs:
            if e <= xpos + 1e-12 or s >= xe - 1e-12:
                ns.append([s, e, h]); continue
            if s < xpos:
                ns.append([s, xpos, h])
            if e > xe:
                ns.append([xe, e, h])
        ns.append([xpos, xe, top])
        ns.sort()
        # merge equal-height neighbours
        merged = [ns[0]]
        for s, e, h in ns[1:]:
            if abs(merged[-1][2] - h) < 1e-9 and abs(merged[-1][1] - s) < 1e-9:
                merged[-1][1] = e
            else:
                merged.append([s, e, h])
        segs[:] = merged

    for u in order:
        w = uw[u]; h = uh[u]
        # candidate x positions = segment starts where the unit fits in [0,W]
        best_x = 0.0; best_y = float("inf")
        xs_cands = sorted({0.0} | {s for s, _, _ in segs})
        for x in xs_cands:
            if x + w > W + 1e-9:
                continue
            y = height_over(x, x + w)
            if y < best_y - 1e-12:
                best_y = y; best_x = x
        ux[u] = best_x; uy[u] = best_y
        place(best_x, w, best_y + h)
    return ux, uy


def _topo(G, indeg, key, U):
    indeg = indeg.copy()
    heap = [(key[i], i) for i in range(U) if indeg[i] == 0]
    heapq.heapify(heap)
    out = []
    while heap:
        _, u = heapq.heappop(heap)
        out.append(u)
        for v in G[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                heapq.heappush(heap, (key[v], v))
    return out if len(out) == U else list(range(U))


def initial_sp(comp, iters=40, outline=None, method="simpl"):
    """Return (P, N): Gamma+ and Gamma- as unit-id lists, extracted from a global
    placement. method:
      'simpl'  -> analytic quadratic + anchored spreading (near-GT wirelength)
      'shelf'  -> fixed-width shelf pack at W* (compact, low area)
      'spread' -> force-directed + area-equalization (legacy)
    """
    U = comp.U
    if U <= 1:
        return list(range(U)), list(range(U))
    if method == "simpl":
        cx, cy = _simpl_place(comp, outline=outline)
    else:
        cx, cy = _spread(comp, iters=iters, outline=outline)
        if method == "shelf" and outline is not None:
            order = np.lexsort((cx, cy))
            sux, suy = _shelf_pack(comp, float(outline[0]), order)
            cx = sux + comp.uw / 2.0
            cy = suy + comp.uh / 2.0
    # bias boundary units toward their edges so the seed already places them at
    # the layout extremes (the SP then naturally puts them on the boundary)
    if comp.codes is not None:
        xr = float(cx.max() - cx.min()) + 1e-6
        yr = float(cy.max() - cy.min()) + 1e-6
        for u in range(U):
            code = 0
            for (b, _, _) in comp.members[u]:
                code |= int(comp.codes[b])
            if comp.pre[u] or code == 0:
                continue
            if code & 1:
                cx[u] = cx.min() - 0.5 * xr
            if code & 2:
                cx[u] = cx.max() + 0.5 * xr
            if code & 8:
                cy[u] = cy.min() - 0.5 * yr
            if code & 4:
                cy[u] = cy.max() + 0.5 * yr
    # geometric left/below relations via dominant axis
    hp = [[] for _ in range(U)]   # hp[b] = units left of b
    vp = [[] for _ in range(U)]   # vp[b] = units below b
    for a in range(U):
        for b in range(a + 1, U):
            dx = cx[a] - cx[b]; dy = cy[a] - cy[b]
            if abs(dx) >= abs(dy):
                (hp[b] if dx < 0 else hp[a]).append(a if dx < 0 else b)
            else:
                (vp[b] if dy < 0 else vp[a]).append(a if dy < 0 else b)
    # Gamma+: a before b if a left-of b OR a below b
    Gp = [[] for _ in range(U)]; indp = np.zeros(U, int)
    # Gamma-: a before b if a left-of b OR a above b (b below a)
    Gn = [[] for _ in range(U)]; indn = np.zeros(U, int)
    for b in range(U):
        for a in hp[b]:
            Gp[a].append(b); indp[b] += 1
            Gn[a].append(b); indn[b] += 1
        for a in vp[b]:
            Gp[a].append(b); indp[b] += 1   # a below b -> a before b in Gamma+
    for a in range(U):
        for c in vp[a]:
            Gn[a].append(c); indn[c] += 1    # a above c -> a before c in Gamma-
    P = _topo(Gp, indp, cx + cy, U)
    N = _topo(Gn, indn, cx - cy, U)
    return P, N
