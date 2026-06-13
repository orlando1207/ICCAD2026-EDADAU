"""
Compile raw contest tensors + fixed block dimensions into placement primitives.

Output `Compiled`:
  - SP units = free blocks + fixed-shape blocks + grouping macros + preplaced(pinned)
  - grouping macros are rigid, internally connected (V_grouping=0 by construction)
  - preplaced are SP units flagged `pre` and pinned to their target coords
  - MIB is satisfied for free (fixed dims already equal within a group)
  - per-unit redistribution edge lists for the wirelength slack pass
"""

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from spcost import Nets, build_nets

SNAP = 1e-7  # round abutment coords so shared edges are bit-identical (avoid ULP splits)


def _to_np(t):
    return t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)


@dataclass
class Compiled:
    n: int
    bw: np.ndarray
    bh: np.ndarray
    codes: np.ndarray            # boundary bitmask per block

    U: int
    uw: np.ndarray               # unit widths
    uh: np.ndarray               # unit heights
    pre: np.ndarray              # bool: pinned preplaced unit
    px: np.ndarray               # preplaced target x (lower-left)
    py: np.ndarray
    rot_ok: np.ndarray           # bool: rotation allowed

    members: List[List[Tuple[int, float, float]]]  # unit -> [(blk, dx, dy)]
    block_unit: np.ndarray       # block -> unit
    offx: np.ndarray             # block lower-left offset within its unit
    offy: np.ndarray

    nets: Nets
    n_soft: int
    hpwl_base: float
    area_base: float
    clu: np.ndarray = None       # per-block cluster id (0=none), for grouping eval

    # redistribution edge lists (per unit), built in build_redistribution()
    red_w: list = field(default_factory=list)
    red_oth: list = field(default_factory=list)   # other block id, or -(pin+1) for pins
    red_mox: list = field(default_factory=list)   # member center-offset x
    red_moy: list = field(default_factory=list)

    def build_redistribution(self):
        coffx = self.offx + self.bw / 2.0   # member center offset within unit
        coffy = self.offy + self.bh / 2.0
        rw = [[] for _ in range(self.U)]
        ro = [[] for _ in range(self.U)]
        rx = [[] for _ in range(self.U)]
        ry = [[] for _ in range(self.U)]
        ne = self.nets
        for k in range(ne.b2b_i.size):
            i = int(ne.b2b_i[k]); j = int(ne.b2b_j[k]); w = float(ne.b2b_w[k])
            ui = self.block_unit[i]; uj = self.block_unit[j]
            rw[ui].append(w); ro[ui].append(j); rx[ui].append(coffx[i]); ry[ui].append(coffy[i])
            rw[uj].append(w); ro[uj].append(i); rx[uj].append(coffx[j]); ry[uj].append(coffy[j])
        for k in range(ne.p2b_blk.size):
            b = int(ne.p2b_blk[k]); pin = int(ne.p2b_pin[k]); w = float(ne.p2b_w[k])
            u = self.block_unit[b]
            rw[u].append(w); ro[u].append(-(pin + 1)); rx[u].append(coffx[b]); ry[u].append(coffy[b])
        self.red_w = [np.asarray(a, float) for a in rw]
        self.red_oth = [np.asarray(a, np.int64) for a in ro]
        self.red_mox = [np.asarray(a, float) for a in rx]
        self.red_moy = [np.asarray(a, float) for a in ry]


def _macro_layout(member_dims, internal_edges):
    """Left-justified shelf pack of cluster members -> rigid, single connected
    component. Returns offsets [(dx,dy)] aligned with member order, and (W,H).

    Connectivity guarantee: every row starts at x=0 (shared left spine connects
    consecutive rows) and members within a row are bottom-aligned & contiguous
    (shared vertical edges). => one connected component.
    """
    m = len(member_dims)
    order = _connectivity_order(m, internal_edges)
    areas = [member_dims[i][0] * member_dims[i][1] for i in range(m)]
    total = sum(areas)
    row_w_target = max(np.sqrt(total), max(member_dims[i][0] for i in range(m)))

    offs = [None] * m
    x = 0.0; y = 0.0; row_h = 0.0; row_start_w = 0.0
    for idx in order:
        w, h = member_dims[idx]
        if x > SNAP and x + w > row_w_target + SNAP:
            # new row
            y = round(y + row_h, 7)
            x = 0.0
            row_h = 0.0
        offs[idx] = (round(x, 7), y)
        x = round(x + w, 7)
        row_h = max(row_h, h)
    W = max(offs[i][0] + member_dims[i][0] for i in range(m))
    H = max(offs[i][1] + member_dims[i][1] for i in range(m))
    return offs, round(W, 7), round(H, 7)


def _connectivity_order(m, internal_edges):
    """Greedy ordering: start at highest-degree member, append the member most
    strongly connected to the already-placed set. Falls back to index order."""
    if m <= 1:
        return list(range(m))
    wdeg = np.zeros(m)
    adj = [dict() for _ in range(m)]
    for a, b, w in internal_edges:
        wdeg[a] += w; wdeg[b] += w
        adj[a][b] = adj[a].get(b, 0) + w
        adj[b][a] = adj[b].get(a, 0) + w
    start = int(np.argmax(wdeg)) if wdeg.any() else 0
    placed = [start]
    seen = {start}
    while len(placed) < m:
        best, best_w = None, -1.0
        for c in range(m):
            if c in seen:
                continue
            cw = sum(adj[c].get(p, 0) for p in placed)
            if cw > best_w:
                best_w, best = cw, c
        placed.append(best); seen.add(best)
    return placed


def compile_with_targets(n, b2b, p2b, pins_pos, constraints,
                         dims_wh, pre_xy, hpwl_base, area_base,
                         use_macros=True) -> Compiled:
    """
    dims_wh : (n,2) fixed (w,h) per block.
    pre_xy  : (n,2) preplaced lower-left (x,y); ignored where not preplaced.
    use_macros: if False, cluster members become individual units. Grouping is
        then handled by an exact penalty in the JIT energy (union-find over
        member edge-adjacency), letting SA abut clusters while packing tightly --
        rigid macros inflate area badly. clu ids are retained either way so the
        energy / final score account for grouping.
    """
    C = _to_np(constraints)
    dims = _to_np(dims_wh).astype(float)
    pre = _to_np(pre_xy).astype(float)
    bw = dims[:, 0].copy(); bh = dims[:, 1].copy()
    ncol = C.shape[1] if C.ndim > 1 else 0
    fixed_f = C[:, 0] if ncol > 0 else np.zeros(n)
    prep_f = C[:, 1] if ncol > 1 else np.zeros(n)
    mib_id = C[:, 2] if ncol > 2 else np.zeros(n)
    clu_id = (C[:, 3] if ncol > 3 else np.zeros(n)).astype(int)
    codes = (C[:, 4] if ncol > 4 else np.zeros(n)).astype(int)

    is_pre = prep_f != 0
    is_fixed = fixed_f != 0

    nets = build_nets(b2b, p2b, pins_pos)

    # n_soft (matches evaluator): |boundary| + Σ(|G|-1) + Σ(|M|-1)
    n_boundary = int(np.count_nonzero(codes))
    n_soft = n_boundary
    for g in range(1, int(mib_id.max()) + 1 if mib_id.size else 1):
        n_soft += max(0, int(np.count_nonzero(mib_id == g)) - 1)
    for g in range(1, int(clu_id.max()) + 1 if clu_id.size else 1):
        n_soft += max(0, int(np.count_nonzero(clu_id == g)) - 1)

    # internal b2b edges (block-id keyed) for macro connectivity ordering
    b2b_pairs = list(zip(nets.b2b_i.tolist(), nets.b2b_j.tolist(), nets.b2b_w.tolist()))

    members: List[List[Tuple[int, float, float]]] = []
    uw_l, uh_l, pre_l, px_l, py_l, rot_l = [], [], [], [], [], []

    assigned = np.zeros(n, dtype=bool)

    # 1) preplaced units (each its own pinned unit; detached from any cluster)
    for i in np.nonzero(is_pre)[0]:
        members.append([(int(i), 0.0, 0.0)])
        uw_l.append(bw[i]); uh_l.append(bh[i])
        pre_l.append(True); px_l.append(pre[i, 0]); py_l.append(pre[i, 1])
        rot_l.append(False)
        assigned[i] = True

    # 2) grouping macros (exclude preplaced members -> already assigned)
    in_mib = mib_id  # for rotation lock
    cluster_ids = sorted({c for c in clu_id.tolist() if c > 0}) if use_macros else []
    for cid in cluster_ids:
        mem = [int(i) for i in np.nonzero(clu_id == cid)[0] if not assigned[i]]
        if not mem:
            continue
        if len(mem) == 1:
            i = mem[0]
            members.append([(i, 0.0, 0.0)])
            uw_l.append(bw[i]); uh_l.append(bh[i])
            pre_l.append(False); px_l.append(0.0); py_l.append(0.0)
            rot_l.append(not (is_fixed[i] or in_mib[i] != 0))
            assigned[i] = True
            continue
        local = {b: k for k, b in enumerate(mem)}
        mdims = [(bw[b], bh[b]) for b in mem]
        iedges = [(local[a], local[c], w) for (a, c, w) in b2b_pairs
                  if a in local and c in local]
        offs, W, H = _macro_layout(mdims, iedges)
        ml = []
        for k, b in enumerate(mem):
            dx, dy = offs[k]
            ml.append((b, dx, dy))
            assigned[b] = True
        members.append(ml)
        uw_l.append(W); uh_l.append(H)
        pre_l.append(False); px_l.append(0.0); py_l.append(0.0)
        rot_l.append(False)  # macros: keep orientation in v1 (MIB-safe)

    # 3) remaining free / fixed-shape singleton units
    for i in np.nonzero(~assigned)[0]:
        members.append([(int(i), 0.0, 0.0)])
        uw_l.append(bw[i]); uh_l.append(bh[i])
        pre_l.append(False); px_l.append(0.0); py_l.append(0.0)
        rot_l.append(not (is_fixed[i] or mib_id[i] != 0))
        assigned[i] = True

    U = len(members)
    block_unit = np.zeros(n, dtype=np.int64)
    offx = np.zeros(n); offy = np.zeros(n)
    for u, ml in enumerate(members):
        for (b, dx, dy) in ml:
            block_unit[b] = u
            offx[b] = dx; offy[b] = dy

    comp = Compiled(
        n=n, bw=bw, bh=bh, codes=codes,
        U=U,
        uw=np.asarray(uw_l, float), uh=np.asarray(uh_l, float),
        pre=np.asarray(pre_l, bool),
        px=np.asarray(px_l, float), py=np.asarray(py_l, float),
        rot_ok=np.asarray(rot_l, bool),
        members=members, block_unit=block_unit, offx=offx, offy=offy,
        nets=nets, n_soft=n_soft, hpwl_base=hpwl_base, area_base=area_base,
        clu=clu_id.astype(np.int64),
    )
    comp.build_redistribution()
    return comp


def expand(comp: Compiled, ux: np.ndarray, uy: np.ndarray):
    """Unit positions -> block lower-left arrays (bx,by,bw,bh)."""
    bx = ux[comp.block_unit] + comp.offx
    by = uy[comp.block_unit] + comp.offy
    return bx, by, comp.bw, comp.bh
