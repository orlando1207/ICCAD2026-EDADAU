"""
Outline-filling, wirelength-aware skyline legalizer.

Given an analytic placement (cx, cy) and the KNOWN target width W*, pack units
into the strip [0, W*] bottom-up. Each unit lands at the lowest skyline position
that fits; among near-lowest positions, the one closest to the unit's analytic x
is chosen (preserves wirelength). Boundary units are forced onto their edge by
construction; preplaced units are fixed obstacles.

This converts the GT-dimension advantage into low area (dense skyline at the
right width) while keeping wirelength (analytic guidance) -- the two things the
SP longest-path legalizer squandered.
"""

import numpy as np


class _Skyline:
    """Top contour over [0, W] as contiguous [xs, xe, h] segments tiling [0, W]."""

    def __init__(self, W):
        self.W = W
        self.segs = [[0.0, W, 0.0]]

    def height_over(self, xl, xr):
        xl = max(0.0, xl); xr = min(self.W, xr)
        m = 0.0
        for s, e, h in self.segs:
            if e <= xl + 1e-12 or s >= xr - 1e-12:
                continue
            if h > m:
                m = h
        return m

    def raise_over(self, xl, xr, top):
        xl = max(0.0, xl); xr = min(self.W, xr)
        if xr <= xl + 1e-12:
            return
        ns = []
        for s, e, h in self.segs:
            if e <= xl + 1e-12 or s >= xr - 1e-12:
                ns.append([s, e, h]); continue
            if s < xl:
                ns.append([s, xl, h])
            if e > xr:
                ns.append([xr, e, h])
        ns.append([xl, xr, top])
        ns.sort()
        merged = [ns[0]]
        for s, e, h in ns[1:]:
            if abs(merged[-1][2] - h) < 1e-9 and abs(merged[-1][1] - s) < 1e-9:
                merged[-1][1] = e
            else:
                merged.append([s, e, h])
        self.segs = merged

    def starts(self):
        return [0.0] + [s for s, _, _ in self.segs]


def skyline_legalize(comp, cx, cy, W, Htarget=None, tol=0.15):
    """Return ux, uy (unit lower-left) packing into width W, or None if a unit
    cannot fit. Boundary by construction; preplaced fixed. `tol` (fraction of
    Htarget) lets a unit take a slightly-higher slot to stay near its analytic x
    (trades a little density for wirelength)."""
    U = comp.U
    uw = comp.uw; uh = comp.uh
    W = max(float(W), float(uw.max()) + 1e-9)
    if Htarget is None:
        Htarget = float((comp.uw * comp.uh).sum()) / W
    ytol = tol * Htarget
    sky = _Skyline(W)
    ux = np.full(U, -1.0); uy = np.full(U, -1.0)

    # unit boundary codes
    code = np.zeros(U, dtype=int)
    for u in range(U):
        c = 0
        for (b, _, _) in comp.members[u]:
            c |= int(comp.codes[b])
        code[u] = c

    # 1) preplaced: fixed; reserve their footprint by raising the skyline over
    #    their x-range up to their top edge (wastes space below, but is safe).
    for u in np.where(comp.pre)[0]:
        ux[u] = comp.px[u]; uy[u] = comp.py[u]
        sky.raise_over(comp.px[u], comp.px[u] + uw[u], comp.py[u] + uh[u])

    movable = [u for u in range(U) if not comp.pre[u]]

    def place_at(u, x):
        y = sky.height_over(x, x + uw[u])
        ux[u] = x; uy[u] = y
        sky.raise_over(x, x + uw[u], y + uh[u])

    def best_x(u):
        """Among slots within ytol of the lowest fit, pick the one closest to the
        analytic x (preserve wirelength); fall back to strict lowest-fit."""
        w = uw[u]
        tgt = min(max(cx[u] - w / 2.0, 0.0), W - w)   # desired lower-left x
        cands = set(min(max(x, 0.0), W - w)
                    for x in sky.starts() if x + w <= W + 1e-9)
        cands.add(W - w); cands.add(tgt)
        ys = {x: sky.height_over(x, x + w) for x in cands}
        ymin = min(ys.values())
        ok = [x for x in cands if ys[x] <= ymin + ytol]
        return min(ok, key=lambda x: abs(x - tgt))

    # 2) ordering: BOTTOM-edge units first (reach y=0 while skyline low),
    #    then the rest bottom-to-top by analytic y, TOP-edge units last.
    def okey(u):
        c = code[u]
        tier = 0 if (c & 8) else (2 if (c & 4) else 1)
        return (tier, cy[u], cx[u])
    order = sorted(movable, key=okey)

    for u in order:
        c = code[u]
        if c & 1:                       # LEFT -> x = 0
            place_at(u, 0.0)
        elif c & 2:                     # RIGHT -> x = W - w
            place_at(u, W - uw[u])
        else:
            x = best_x(u)
            if x is None:
                return None
            place_at(u, x)

    # 3) TOP-edge units: slide up to touch the final top edge (boundary), if the
    #    move stays overlap-free (nothing above them in their x-span already).
    H = float(np.max(uy + uh)) if U else 0.0
    for u in movable:
        if code[u] & 4:
            # is there anything above this unit in its x-span?
            above = sky.height_over(ux[u], ux[u] + uw[u])
            if above <= uy[u] + uh[u] + 1e-6:   # it's the top in its span
                uy[u] = H - uh[u]
    return ux, uy
