"""MPCG: Minimal-Perturbation Constraint-Graph legalizer (stage 2).

Implements docs/superpowers/specs/2026-07-16-legalizer-design.md:
  Stage A  H/V constraint graphs from the predicted relative order
           (separation axis = smaller violation depth, direction = predicted order)
  Stage B  two anchored LPs (x then y): exact weighted HPWL + anchor + bbox width,
           subject to separations, preplaced equalities (as fixed bounds)
  Stage C  soft-constraint attachments: boundary -> bbox-extreme equalities
           (hard, auto-demoted to penalties on infeasibility), cluster ->
           spanning-forest abutment equalities + cross-axis overlap inequalities
  Stage D  exact snapping (contacts, boundary, preplaced), verification with
           official semantics, eps-bump / soften fallback ladder

Dims are never modified (areas stay exact; MIB stays tied; fixed dims stay exact).

Run from iccad2026contest/:
  python -m floordiff.legalize --pred floordiff/out/preds.json \
      --out floordiff/out/legalized.json [--cases 100,120]
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from scipy import sparse
from scipy.optimize import linprog

from shapely.geometry import box as _sbox
from shapely.ops import unary_union as _sunion

from .data import VALIDATION_NS, gt_xywh, load_validation_case, target_xywh
from .evaluate import bbox_area as _bbox_area
from .evaluate import weighted_hpwl as _whpwl

DEFAULT_CFG = {
    # alpha / gamma / lam are calibrated per case to official-cost units (see
    # legalize_case); these are multipliers on that calibration
    'alpha_mult': 1.0,   # weighted-HPWL term (x 0.5/hpwl_baseline)
    'beta': None,        # anchor per unit move; None -> 0.1/S per case
    'gamma_mult': 1.0,   # bbox extent (x 0.5*other_extent/area_baseline)
    'lam_mult': 1.0,     # soft attach/contact penalty (x 2/(Nsoft*0.02*S))
    'eps_rel': 0.0,      # separation margin, x S (touching is legal; 97%-packed GT chains have zero slack, any margin forces stretch)
    'contact_tol_rel': 0.03,   # max gap to still create an abutment contact, x S
    'attach_tol_rel': 0.02,    # max distance to still attach to a bbox side, x S
    'delta_rel': 0.002,  # min shared-edge length for abutment, x S (official rule only needs > 0)
    'iterations': 3,     # graph re-derivation passes (pass k uses pass k-1 geometry)
    'milp_time': 2.0,    # per-axis MILP time limit, seconds (pass 1 only)
    'good_enough': 1.10,  # proxy cost below which we stop exploring (rungs/passes/seeds)
    'bbox_reshape': True,  # post-pass: shrink critical bbox chains when area gap > 4%
    'seed_stop': 1.05,    # stop trying more seeds below this cost (stricter than good_enough)
}


# --------------------------------------------------------------------------- stage A

def build_pairs(xywh):
    """For every pair (i<j): separation axis (H if x-gap >= y-gap) and direction.
    Returns (H_pairs, V_pairs) as (m, 2) int arrays with [leader, follower]."""
    x0, y0, w, h = xywh[:, 0], xywh[:, 1], xywh[:, 2], xywh[:, 3]
    x1, y1 = x0 + w, y0 + h
    gx = np.maximum(x0[:, None], x0[None, :]) - np.minimum(x1[:, None], x1[None, :])
    gy = np.maximum(y0[:, None], y0[None, :]) - np.minimum(y1[:, None], y1[None, :])
    n = len(x0)
    iu, ju = np.triu_indices(n, k=1)
    horiz = gx[iu, ju] >= gy[iu, ju]
    cx, cy = x0 + w / 2, y0 + h / 2

    def orient(ii, jj, c):
        swap = c[ii] > c[jj]
        return np.stack([np.where(swap, jj, ii), np.where(swap, ii, jj)], axis=1)

    H = orient(iu[horiz], ju[horiz], cx)
    V = orient(iu[~horiz], ju[~horiz], cy)
    return H, V


def repair_axes(H, V, geom, areas, pre_mask, shrinkable, eps, locked=(),
                fixedH=(), fixedV=(), max_rounds=300, max_shrink=0.25,
                aspect_cap=3.6):
    """Conflict-directed repair (FLOORIST-style) with soft-block reshaping.

    Greedy per-pair axis choice can create separation chains between preplaced
    anchors that exceed their fixed span. Two remedies, in order:
      1. RESHAPE: predicted dims carry ~13% shape error while GT fills anchor
         spans at ~97% utilization — chains often genuinely don't fit. Shrink the
         chain axis of the path's shrinkable soft blocks (area preserved exactly,
         aspect kept within data range) so the chain fits without restructuring.
      2. FLIP: move the path edge that creates the shortest chain on the other
         axis (each edge flips at most once, locked pairs last).
    Returns (H, V, ok, w, h) — w/h possibly reshaped.
    """
    x0, y0 = geom[:, 0].copy(), geom[:, 1].copy()
    w, h = geom[:, 2].copy(), geom[:, 3].copy()
    x1, y1 = x0 + w, y0 + h
    gx = np.maximum(x0[:, None], x0[None, :]) - np.minimum(x1[:, None], x1[None, :])
    gy = np.maximum(y0[:, None], y0[None, :]) - np.minimum(y1[:, None], y1[None, :])
    n = len(x0)
    H, V = [tuple(map(int, p)) for p in H], [tuple(map(int, p)) for p in V]
    locked = set(locked)
    flipped = set()
    reshaped = {'H': set(), 'V': set()}

    def key(i, j):
        return (min(i, j), max(i, j))

    def find_conflict(edges, pos, size, order_key):
        """Returns (path_edge_indices, path_nodes, excess) or None."""
        order = sorted(range(n), key=lambda i: (order_key[i], i))
        adj = {}
        for k, (i, j) in enumerate(edges):
            adj.setdefault(i, []).append((j, k))
        lb = np.full(n, -np.inf)
        lb[pre_mask] = pos[pre_mask]
        parent = {}
        for i in order:
            if lb[i] == -np.inf:
                continue
            if pre_mask[i] and lb[i] > pos[i] + 1e-7:
                path, nodes = [], [i]
                v = i
                while v in parent:
                    path.append(parent[v][1])
                    v = parent[v][0]
                    nodes.append(v)
                return path, nodes[::-1], float(lb[i] - pos[i])
            base = pos[i] if pre_mask[i] else lb[i]
            for j, k in adj.get(i, ()):
                cand = base + size[i] + eps
                if cand > lb[j]:
                    lb[j] = cand
                    parent[j] = (i, k)
        return None

    def chain_ends(edges, size, order_key):
        order = sorted(range(n), key=lambda i: (order_key[i], i))
        head = np.zeros(n)
        tail = np.array(size, dtype=np.float64).copy()
        adj = {}
        for i, j in edges:
            adj.setdefault(i, []).append(j)
        for i in order:
            for j in adj.get(i, ()):
                head[j] = max(head[j], head[i] + size[i])
        for i in reversed(order):
            for j in adj.get(i, ()):
                tail[i] = max(tail[i], size[i] + tail[j])
        return head, tail

    def try_reshape(axis, nodes, excess):
        """Shrink the chain axis of shrinkable middle nodes; grow the other axis
        (area exact). Absorbs as much of the excess as the shrink/aspect caps
        allow; returns True only if fully absorbed (else a flip handles the rest,
        with the chain already shorter)."""
        size, osize = (w, h) if axis == 'H' else (h, w)
        mid = [v for v in nodes[1:-1]
               if shrinkable[v] and v not in reshaped[axis]]
        need = excess * 1.0001 + 1e-9
        while mid and need > 1e-9:
            tot = sum(size[v] for v in mid)
            ratio = max(1.0 - need / tot, 1.0 - max_shrink)
            ok = []
            for v in mid:   # per-block aspect cap after growth of the other side
                new_s = size[v] * ratio
                new_o = areas[v] / new_s
                if max(new_o / new_s, new_s / new_o) <= aspect_cap:
                    ok.append(v)
            if len(ok) < len(mid):
                mid = ok
                continue
            for v in mid:
                size[v] *= ratio
                osize[v] = areas[v] / size[v]
                reshaped[axis].add(v)
            need -= (1.0 - ratio) * tot
            break
        return need <= 1e-9

    fixedH = [tuple(map(int, p)) for p in fixedH]
    fixedV = [tuple(map(int, p)) for p in fixedV]
    for _ in range(max_rounds):
        res = find_conflict(H + fixedH, x0, w, x0 + w / 2)
        axis = 'H'
        if res is None:
            res = find_conflict(V + fixedV, y0, h, y0 + h / 2)
            axis = 'V'
        if res is None:
            return (np.array(H, dtype=np.int64).reshape(-1, 2),
                    np.array(V, dtype=np.int64).reshape(-1, 2), True, w, h)
        conflict, nodes, excess = res
        if try_reshape(axis, nodes, excess):
            continue
        if axis == 'H':
            edges, other, fother, g_other, okey, osize = \
                H, V, fixedV, gy, y0 + h / 2, h
        else:
            edges, other, fother, g_other, okey, osize = \
                V, H, fixedH, gx, x0 + w / 2, w
        flippable = [k for k in conflict if k < len(edges)]
        cand = [k for k in flippable
                if key(*edges[k]) not in locked and key(*edges[k]) not in flipped]
        if not cand:   # second tier: allow flipping locked pairs (the LP ladder
            cand = [k for k in flippable if key(*edges[k]) not in flipped]
        if not cand:   # will demote the corresponding attachment to soft)
            break
        head, tail = chain_ends(other + fother, osize, okey)

        def flip_chain(k):
            i, j = edges[k]
            if (okey[i], i) > (okey[j], j):
                i, j = j, i
            return head[i] + osize[i] + tail[j]

        best = min(cand, key=lambda k: (flip_chain(k),
                                        -g_other[edges[k][0], edges[k][1]]))
        i, j = edges.pop(best)
        flipped.add(key(i, j))
        if (okey[i], i) > (okey[j], j):
            i, j = j, i
        other.append((i, j))
    return (np.array(H, dtype=np.int64).reshape(-1, 2),
            np.array(V, dtype=np.int64).reshape(-1, 2), False, w, h)


# --------------------------------------------------------------------------- stage C prep

class _UF:
    def __init__(self, items):
        self.p = {i: i for i in items}

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        self.p[ra] = rb
        return True


def cluster_contacts(xywh, cons, S, tol):
    """Spanning forest of intended abutments per cluster group.
    Returns list of (i, j, axis) with i the leader on that axis ('H': i left of j)."""
    x0, y0, w, h = xywh[:, 0], xywh[:, 1], xywh[:, 2], xywh[:, 3]
    x1, y1 = x0 + w, y0 + h
    cx, cy = x0 + w / 2, y0 + h / 2
    contacts = []
    gids = cons[:, 3].numpy()
    for g in np.unique(gids):
        if g == 0:
            continue
        mem = np.nonzero(gids == g)[0]
        if len(mem) < 2:
            continue
        cand = []
        for a in range(len(mem)):
            for b in range(a + 1, len(mem)):
                i, j = int(mem[a]), int(mem[b])
                gx = max(x0[i], x0[j]) - min(x1[i], x1[j])
                gy = max(y0[i], y0[j]) - min(y1[i], y1[j])
                cand.append((max(gx, gy), gx, gy, i, j))
        cand.sort()
        uf = _UF(list(mem))
        picked = 0
        for sep, gx, gy, i, j in cand:
            if picked == len(mem) - 1:
                break
            if sep > tol:      # too far apart to abut cheaply — accept the
                continue       # fragmentation violation instead of dragging
            if not uf.union(i, j):
                continue
            picked += 1
            if gx >= gy:   # abut horizontally
                lead, fol = (i, j) if cx[i] <= cx[j] else (j, i)
                contacts.append((lead, fol, 'H'))
            else:
                lead, fol = (i, j) if cy[i] <= cy[j] else (j, i)
                contacts.append((lead, fol, 'V'))
    return contacts


def boundary_lists(cons, pre_mask):
    """Blocks required to touch each bbox side (preplaced excluded — immovable)."""
    bits = cons[:, 4].numpy()
    free = ~pre_mask
    return {side: np.nonzero(((bits & b) > 0) & free)[0].tolist()
            for side, b in (('L', 1), ('R', 2), ('T', 4), ('B', 8))}


# --------------------------------------------------------------------------- stage B

def _abs_rows(n_var, base_r, entries):
    """Vectorized |expr| <= t rows. entries: (i_col, t_col, rhs) arrays.
    Emits two rows per entry: +expr - t <= rhs ; -expr - t <= -rhs."""
    i_col, t_col, rhs = entries
    m = len(i_col)
    r = np.repeat(np.arange(base_r, base_r + 2 * m), 2)
    c = np.empty(4 * m, dtype=np.int64)
    v = np.empty(4 * m)
    c[0::4], v[0::4] = i_col, 1.0
    c[1::4], v[1::4] = t_col, -1.0
    c[2::4], v[2::4] = i_col, -1.0
    c[3::4], v[3::4] = t_col, -1.0
    b = np.empty(2 * m)
    b[0::2], b[1::2] = rhs, -rhs
    return r, c, v, b, base_r + 2 * m


def axis_lp(n, pos_p, size, sep, eq_contacts, cross_overlap, pre_mask, pre_val,
            attach_lo, attach_hi, nets, pin_lin, cfg, S,
            hard_attach=True, hard_contact=True, eps=None):
    """One axis of the anchored LP (vectorized assembly).

    sep: (m,2) leader/follower separation inequalities
    eq_contacts: (lead, fol) abutment equalities on THIS axis
    cross_overlap: (i, j, delta) shared-interval requirements on this axis
    attach_lo/hi: block indices attached to bbox min/max
    nets: (E, 3) [i, j, weight] (pruned); pin_lin: (n,) linearized pin-pull
    coefficient dHPWL/dpos at the anchor point (exact while blocks stay on the
    same side of their pins — moves are sliver-scale).
    Returns solution positions or None if infeasible.
    """
    eps = eps if eps is not None else cfg['eps_rel'] * S
    E = len(nets)
    n_soft = 0 if hard_attach else len(attach_lo) + len(attach_hi)
    n_softc = 0 if hard_contact else len(eq_contacts)
    ib_min, ib_max = n, n + 1
    it_net = n + 2
    iu = it_net + E
    iv_a = iu + n
    iv_c = iv_a + n_soft
    nv = iv_c + n_softc

    c = np.zeros(nv)
    c[:n] = cfg['alpha'] * pin_lin
    c[ib_min], c[ib_max] = -cfg['gamma'], cfg['gamma']
    if E:
        c[it_net:it_net + E] = cfg['alpha'] * nets[:, 2]
    c[iu:iu + n] = cfg['beta']
    c[iv_a:nv] = cfg['lam']

    R, C, V, B = [], [], [], []
    r = 0
    sep = np.asarray(sep, dtype=np.int64).reshape(-1, 2)
    m = len(sep)
    if m:                                    # pos_i + size_i + eps <= pos_j
        R.append(np.repeat(np.arange(r, r + m), 2))
        C.append(sep.ravel())
        V.append(np.tile([1.0, -1.0], m))
        B.append(-(size[sep[:, 0]] + eps))
        r += m
    idx = np.arange(n)                       # bbox
    R.append(np.repeat(np.arange(r, r + n), 2))
    C.append(np.stack([np.full(n, ib_min), idx], 1).ravel())
    V.append(np.tile([1.0, -1.0], n))
    B.append(np.zeros(n))
    r += n
    R.append(np.repeat(np.arange(r, r + n), 2))
    C.append(np.stack([idx, np.full(n, ib_max)], 1).ravel())
    V.append(np.tile([1.0, -1.0], n))
    B.append(-size)
    r += n
    if E:                                    # |c_i - c_j| <= t (2 rows, 3 nnz)
        i = nets[:, 0].astype(np.int64)
        j = nets[:, 1].astype(np.int64)
        off = (size[j] - size[i]) / 2
        rr = np.repeat(np.arange(r, r + 2 * E), 3)
        cc = np.empty(6 * E, dtype=np.int64)
        vv = np.empty(6 * E)
        tcol = it_net + np.arange(E)
        cc[0::6], vv[0::6] = i, 1.0
        cc[1::6], vv[1::6] = j, -1.0
        cc[2::6], vv[2::6] = tcol, -1.0
        cc[3::6], vv[3::6] = j, 1.0
        cc[4::6], vv[4::6] = i, -1.0
        cc[5::6], vv[5::6] = tcol, -1.0
        bb = np.empty(2 * E)
        bb[0::2], bb[1::2] = off, -off
        R.append(rr); C.append(cc); V.append(vv); B.append(bb)
        r += 2 * E
    rr, cc, vv, bb, r = _abs_rows(nv, r, (idx, iu + idx, pos_p[:n]))  # anchor
    R.append(rr); C.append(cc); V.append(vv); B.append(bb)

    rows, cols, vals, b = [list(x) for x in ([], [], [], [])]

    def add(coef, rhs):
        nonlocal r
        for col_, v_ in coef:
            rows.append(r)
            cols.append(col_)
            vals.append(v_)
        b.append(rhs)
        r += 1

    for i, j, delta in cross_overlap:        # shared interval >= delta
        add([(int(j), 1.0), (int(i), -1.0)], size[int(i)] - delta)
        add([(int(i), 1.0), (int(j), -1.0)], size[int(j)] - delta)

    erows, ecols, evals, eb = [], [], [], []
    er = 0

    def add_eq(coef, rhs):
        nonlocal er
        for col_, v_ in coef:
            erows.append(er)
            ecols.append(col_)
            evals.append(v_)
        eb.append(rhs)
        er += 1

    if hard_contact:
        for i, j in eq_contacts:             # pos_i + size_i = pos_j
            add_eq([(int(i), 1.0), (int(j), -1.0)], -size[int(i)])
    else:
        for k, (i, j) in enumerate(eq_contacts):
            add([(int(i), 1.0), (int(j), -1.0)], -size[int(i)])
            add([(int(j), 1.0), (int(i), -1.0), (iv_c + k, -1.0)], size[int(i)])
    if hard_attach:
        for i in attach_lo:
            add_eq([(int(i), 1.0), (ib_min, -1.0)], 0.0)
        for i in attach_hi:
            add_eq([(int(i), 1.0), (ib_max, -1.0)], -size[int(i)])
    else:
        for k, i in enumerate(list(attach_lo) + list(attach_hi)):
            hi = k >= len(attach_lo)
            anchor_col = ib_max if hi else ib_min
            rhs = -size[int(i)] if hi else 0.0
            add([(int(i), 1.0), (anchor_col, -1.0), (iv_a + k, -1.0)], rhs)
            add([(anchor_col, 1.0), (int(i), -1.0), (iv_a + k, -1.0)], -rhs)

    R.append(np.array(rows, dtype=np.int64))
    C.append(np.array(cols, dtype=np.int64))
    V.append(np.array(vals))
    B.append(np.array(b))
    A_ub = sparse.coo_matrix(
        (np.concatenate(V), (np.concatenate(R), np.concatenate(C))),
        shape=(r, nv)).tocsr()
    A_eq = (sparse.coo_matrix((evals, (erows, ecols)), shape=(er, nv)).tocsr()
            if er else None)
    bounds = [(None, None)] * nv
    for i in range(n):
        if pre_mask[i]:
            bounds[i] = (float(pre_val[i]), float(pre_val[i]))
    for k in range(it_net, nv):
        bounds[k] = (0, None)

    res = linprog(c, A_ub=A_ub, b_ub=np.concatenate(B),
                  A_eq=A_eq, b_eq=np.array(eb) if er else None,
                  bounds=bounds, method='highs')
    return res.x[:n] if res.status == 0 else None


def axis_milp(n, pos_p, size, sep, contacts, cross_overlap, pre_mask, pre_val,
              attach_lo, attach_hi, nets, pin_lin, cfg, S,
              ben_attach_lo, ben_attach_hi, ben_contact, eps=0.0,
              time_limit=10.0):
    """MILP variant of axis_lp: per-attachment / per-contact binary selection.

    Each attachment/contact gets a binary z (1 = satisfied, big-M linked) whose
    objective reward equals its violation saving in official-cost units — the
    solver picks the profitable subset instead of the all-or-nothing rungs.
    Returns (positions, z_contact_bool_list) or (None, None).
    """
    from scipy.optimize import Bounds, LinearConstraint, milp
    E = len(nets)
    A_att = len(attach_lo) + len(attach_hi)
    C = len(contacts)
    M = 3.0 * S
    ib_min, ib_max = n, n + 1
    it_net = n + 2
    iu = it_net + E
    iz_a = iu + n
    iz_c = iz_a + A_att
    nv = iz_c + C

    c = np.zeros(nv)
    c[:n] = cfg['alpha'] * pin_lin
    c[ib_min], c[ib_max] = -cfg['gamma'], cfg['gamma']
    if E:
        c[it_net:it_net + E] = cfg['alpha'] * nets[:, 2]
    c[iu:iu + n] = cfg['beta']
    bens = list(ben_attach_lo) + list(ben_attach_hi)
    for k in range(A_att):
        c[iz_a + k] = -bens[k]
    for k in range(C):
        c[iz_c + k] = -ben_contact[k]

    rows, cols, vals, lo, up = [], [], [], [], []
    r = 0

    def add(coef, lb, ub):
        nonlocal r
        for cc, vv in coef:
            rows.append(r)
            cols.append(cc)
            vals.append(vv)
        lo.append(lb)
        up.append(ub)
        r += 1

    for i, j in sep:
        add([(int(i), 1.0), (int(j), -1.0)], -np.inf, -(size[int(i)] + eps))
    for i in range(n):
        add([(ib_min, 1.0), (i, -1.0)], -np.inf, 0.0)
        add([(i, 1.0), (ib_max, -1.0)], -np.inf, -size[i])
    for e in range(E):
        i, j = int(nets[e, 0]), int(nets[e, 1])
        off = (size[j] - size[i]) / 2
        add([(i, 1.0), (j, -1.0), (it_net + e, -1.0)], -np.inf, off)
        add([(j, 1.0), (i, -1.0), (it_net + e, -1.0)], -np.inf, -off)
    for i in range(n):
        add([(i, 1.0), (iu + i, -1.0)], -np.inf, pos_p[i])
        add([(i, -1.0), (iu + i, -1.0)], -np.inf, -pos_p[i])
    for i, j, delta in cross_overlap:
        add([(int(j), 1.0), (int(i), -1.0)], -np.inf, size[int(i)] - delta)
        add([(int(i), 1.0), (int(j), -1.0)], -np.inf, size[int(j)] - delta)
    for k, i in enumerate(list(attach_lo)):       # |pos_i - bmin| <= M(1-z)
        add([(int(i), 1.0), (ib_min, -1.0), (iz_a + k, M)], -np.inf, M)
        add([(ib_min, 1.0), (int(i), -1.0), (iz_a + k, M)], -np.inf, M)
    off0 = len(attach_lo)
    for k, i in enumerate(list(attach_hi)):       # |pos_i+size_i - bmax| <= M(1-z)
        add([(int(i), 1.0), (ib_max, -1.0), (iz_a + off0 + k, M)],
            -np.inf, M - size[int(i)])
        add([(ib_max, 1.0), (int(i), -1.0), (iz_a + off0 + k, M)],
            -np.inf, M + size[int(i)])
    for k, (i, j) in enumerate(contacts):
        # hard separation + gap <= M(1-z)
        add([(int(i), 1.0), (int(j), -1.0)], -np.inf, -size[int(i)])
        add([(int(j), 1.0), (int(i), -1.0), (iz_c + k, M)],
            -np.inf, size[int(i)] + M)

    A = sparse.coo_matrix((vals, (rows, cols)), shape=(r, nv)).tocsr()
    lb_v = np.full(nv, -np.inf)
    ub_v = np.full(nv, np.inf)
    for i in range(n):
        if pre_mask[i]:
            lb_v[i] = ub_v[i] = float(pre_val[i])
    lb_v[it_net:iz_a] = 0.0
    lb_v[iz_a:nv] = 0.0
    ub_v[iz_a:nv] = 1.0
    integrality = np.zeros(nv)
    integrality[iz_a:nv] = 1
    try:
        res = milp(c, constraints=LinearConstraint(A, np.array(lo), np.array(up)),
                   integrality=integrality, bounds=Bounds(lb_v, ub_v),
                   options={'time_limit': time_limit, 'mip_rel_gap': 0.02})
    except Exception:
        return None, None, None
    if res.x is None:
        return None, None, None
    zc = [bool(round(res.x[iz_c + k])) for k in range(C)]
    za = [bool(round(res.x[iz_a + k])) for k in range(A_att)]
    return res.x[:n], zc, za


# --------------------------------------------------------------------------- stage D

def _penetrations(xywh):
    x0, y0, w, h = xywh[:, 0], xywh[:, 1], xywh[:, 2], xywh[:, 3]
    x1, y1 = x0 + w, y0 + h
    ox = np.minimum(x1[:, None], x1) - np.maximum(x0[:, None], x0)
    oy = np.minimum(y1[:, None], y1) - np.maximum(y0[:, None], y0)
    both = np.minimum(ox, oy)
    np.fill_diagonal(both, -1)
    return max(0.0, both.max())


def _violations_official(sol, case):
    """Soft violations with exactly the official evaluator's semantics:
    boundary eps = 1e-6 absolute, grouping connectivity via shapely unary_union,
    MIB distinct shapes after round(4)."""
    cons = case['cons'].numpy()
    x0, y0 = sol[:, 0], sol[:, 1]
    x1, y1 = x0 + sol[:, 2], y0 + sol[:, 3]
    bx0, bx1, by0, by1 = x0.min(), x1.max(), y0.min(), y1.max()
    eps = 1e-6
    vb = 0
    for i in np.nonzero(cons[:, 4])[0]:
        bits = int(cons[i, 4])
        ok = True
        if bits & 1:
            ok &= abs(x0[i] - bx0) < eps
        if bits & 2:
            ok &= abs(x1[i] - bx1) < eps
        if bits & 4:
            ok &= abs(y1[i] - by1) < eps
        if bits & 8:
            ok &= abs(y0[i] - by0) < eps
        vb += int(not ok)
    vg = 0
    for g in np.unique(cons[:, 3]):
        if g == 0:
            continue
        idx = np.nonzero(cons[:, 3] == g)[0]
        u = _sunion([_sbox(x0[i], y0[i], x1[i], y1[i]) for i in idx])
        if u.geom_type == 'MultiPolygon':
            vg += len(u.geoms) - 1
    vm = 0
    for g in np.unique(cons[:, 2]):
        if g == 0:
            continue
        idx = np.nonzero(cons[:, 2] == g)[0]
        vm += len({(round(float(sol[i, 2]), 4), round(float(sol[i, 3]), 4))
                   for i in idx}) - 1
    return vb, vg, vm


def _proxy_cost(sol_np, case, S, hpwl_base, area_base, n_soft_norm):
    """In-loop selector: official-parity cost, vectorized (runtime factor neutral).
    Uses only inputs available at contest time (baselines, constraint metadata)."""
    if _penetrations(sol_np) > 1e-6:
        return 10.0
    t = torch.tensor(sol_np, dtype=torch.float64)
    hg = (_whpwl(t, case) - hpwl_base) / max(hpwl_base, 1e-6)
    ag = (_bbox_area(t) - area_base) / max(area_base, 1e-6)
    vb, vg, vm = _violations_official(sol_np, case)
    vr = (vb + vg + vm) / max(n_soft_norm, 1)
    return min((1 + 0.5 * (hg + ag)) * math.exp(2 * vr), 10 - 1e-6)


def legalize_case(pred_xywh, case, cfg=None, verbose=False):
    """(n,4) predicted (x,y,w,h) -> feasible (x,y,w,h). Dims never change.

    Iterated MPCG: pass 1 derives the constraint graph from the prediction
    (repairing axis conflicts), later passes re-derive it from the previous
    legal solution — a self-consistent graph with no forced stretch — while
    still anchoring to the original prediction. Best pass by cost proxy wins.
    """
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    t0 = time.time()
    pred = pred_xywh.numpy().astype(np.float64).copy()
    cons = case['cons']
    n = pred.shape[0]
    S = float(case['area'].sum().sqrt())
    gt = target_xywh(case).numpy().astype(np.float64)   # fixed/pre rows only

    pre_mask = (cons[:, 1] > 0).numpy()
    pred[pre_mask] = gt[pre_mask]          # stamp preplaced exactly (pos+dims)
    fixed_mask = (cons[:, 0] > 0).numpy()
    pred[fixed_mask, 2:4] = gt[fixed_mask, 2:4]
    areas = case['area'].numpy().astype(np.float64)
    # reshape candidates: soft blocks outside MIB groups (MIB dims must stay tied)
    shrinkable = (~pre_mask) & (~fixed_mask) & (cons[:, 2].numpy() == 0)

    b2b = case['b2b'].numpy()
    p2b = case['p2b'].numpy()
    pins = case['pins'].numpy()
    nets = np.stack([b2b[:, 0], b2b[:, 1], b2b[:, 2]], axis=1) if len(b2b) else \
        np.zeros((0, 3))
    if len(nets) > 64:      # keep edges covering 95% of weight mass (cap 3000)
        order = np.argsort(-nets[:, 2])
        cum = np.cumsum(nets[order, 2])
        k = min(max(int(np.searchsorted(cum, 0.95 * cum[-1])) + 1, 64), 3000)
        nets = nets[order[:k]]
    # linearized pin pull: dHPWL/dpos at the anchor (exact while blocks stay on
    # the same side of their pins — moves are sliver-scale)
    pinx = np.zeros(n)
    piny = np.zeros(n)
    if len(p2b):
        bi = p2b[:, 1].astype(int)
        pi = p2b[:, 0].astype(int)
        pw = p2b[:, 2]
        pcx = pred[bi, 0] + pred[bi, 2] / 2
        pcy = pred[bi, 1] + pred[bi, 3] / 2
        np.add.at(pinx, bi, pw * np.sign(pcx - pins[pi, 0]))
        np.add.at(piny, bi, pw * np.sign(pcy - pins[pi, 1]))

    # calibrate LP weights to official-cost units: cost ~ 0.5*(dHPWL/hpwl_base +
    # dArea/area_base) + 2*V/Nsoft. A bbox-width unit costs 0.5*height/area_base;
    # a snapped attachment/contact earns 2/Nsoft over <= attach_tol distance.
    if case.get('metrics') is not None:
        hpwl_base = float(case['metrics'][6] + case['metrics'][7])
        area_base = float(case['metrics'][0])
    else:   # solve-time: baselines unavailable -> the prediction sets the scale
        hpwl_base = float(_whpwl(torch.tensor(pred), case))
        area_base = float(_bbox_area(torch.tensor(pred)))
    bits = cons[:, 4].numpy()
    n_soft_norm = int((bits > 0).sum())
    for c in (2, 3):
        for g in torch.unique(cons[:, c]):
            if g > 0:
                n_soft_norm += int((cons[:, c] == g).sum()) - 1
    Wp = (pred[:, 0] + pred[:, 2]).max() - pred[:, 0].min()
    Hp = (pred[:, 1] + pred[:, 3]).max() - pred[:, 1].min()
    alpha = cfg['alpha_mult'] * 0.5 / max(hpwl_base, 1e-6)
    beta = cfg['beta'] if cfg['beta'] is not None else 0.1 / S
    lam = cfg['lam_mult'] * 2.0 / (max(n_soft_norm, 1) * cfg['attach_tol_rel'] * S)
    cfgx = {**cfg, 'alpha': alpha, 'beta': beta, 'lam': lam,
            'gamma': cfg['gamma_mult'] * 0.5 * Hp / max(area_base, 1e-6)}
    cfgy = {**cfg, 'alpha': alpha, 'beta': beta, 'lam': lam,
            'gamma': cfg['gamma_mult'] * 0.5 * Wp / max(area_base, 1e-6)}
    att_full = boundary_lists(cons, pre_mask)

    def one_pass(geom, use_milp=True):
        """Derive graph/contacts/attachments from `geom`, solve anchored to pred."""
        H, V = build_pairs(geom)
        contacts = cluster_contacts(geom, cons, S, cfg['contact_tol_rel'] * S)
        contacts = [(i, j, a) for i, j, a in contacts
                    if not (pre_mask[i] and pre_mask[j])]
        contact_set = {(min(i, j), max(i, j)) for i, j, _ in contacts}

        def keep(p):
            i, j = int(p[0]), int(p[1])
            if (min(i, j), max(i, j)) in contact_set:
                return False
            return not (pre_mask[i] and pre_mask[j])

        H = [tuple(map(int, p)) for p in H if keep(p)]
        V = [tuple(map(int, p)) for p in V if keep(p)]

        # attach only blocks already near their required side — snapping is cheap;
        # dragging a far block costs more quality than the violation it fixes
        gx0, gy0 = geom[:, 0].min(), geom[:, 1].min()
        gx1 = (geom[:, 0] + geom[:, 2]).max()
        gy1 = (geom[:, 1] + geom[:, 3]).max()
        att_tol = cfg['attach_tol_rel'] * S
        att = {
            'L': [i for i in att_full['L'] if geom[i, 0] - gx0 < att_tol],
            'R': [i for i in att_full['R']
                  if gx1 - (geom[i, 0] + geom[i, 2]) < att_tol],
            'B': [i for i in att_full['B'] if geom[i, 1] - gy0 < att_tol],
            'T': [i for i in att_full['T']
                  if gy1 - (geom[i, 1] + geom[i, 3]) < att_tol],
        }

        # blocks attached to the same vertical side cannot be H-separated ->
        # force perpendicular; symmetric for T/B. Lock these pairs.
        sameL, sameR = set(att['L']), set(att['R'])
        sameB, sameT = set(att['B']), set(att['T'])
        ccx = geom[:, 0] + geom[:, 2] / 2
        ccy = geom[:, 1] + geom[:, 3] / 2
        locked = set()

        def same_side(i, j, a, b):
            return (i in a and j in a) or (i in b and j in b)

        H2, V2 = [], []
        for i, j in H:
            if same_side(i, j, sameL, sameR):
                lead, fol = (i, j) if (ccy[i], i) <= (ccy[j], j) else (j, i)
                V2.append((lead, fol))
                locked.add((min(i, j), max(i, j)))
            else:
                H2.append((i, j))
        for i, j in V:
            if same_side(i, j, sameB, sameT):
                lead, fol = (i, j) if (ccx[i], i) <= (ccx[j], j) else (j, i)
                H2.append((lead, fol))
                locked.add((min(i, j), max(i, j)))
            else:
                V2.append((i, j))
        H = np.array(H2, dtype=np.int64).reshape(-1, 2)
        V = np.array(V2, dtype=np.int64).reshape(-1, 2)
        cH = [(i, j) for i, j, a in contacts if a == 'H']
        cV = [(i, j) for i, j, a in contacts if a == 'V']
        H, V, repaired, wd, hd = repair_axes(
            H, V, geom, areas, pre_mask, shrinkable, cfg['eps_rel'] * S,
            locked=locked, fixedH=cH, fixedV=cV)
        delta = cfg['delta_rel'] * S
        ovlH = [(i, j, min(delta, 0.4 * min(hd[i], hd[j]))) for i, j in cH]
        ovlV = [(i, j, min(delta, 0.4 * min(wd[i], wd[j]))) for i, j in cV]

        def solve(hard_attach, hard_contact, eps_override=None,
                  with_contacts=True, with_attach=True):
            eps = cfg['eps_rel'] * S if eps_override is None else eps_override
            _cH = cH if with_contacts else []
            _cV = cV if with_contacts else []
            _oH = ovlH if with_contacts else []
            _oV = ovlV if with_contacts else []
            _att = att if with_attach else {k: [] for k in ('L', 'R', 'T', 'B')}
            _H = H if with_contacts else \
                np.vstack([H, np.array(cH, dtype=np.int64).reshape(-1, 2)])
            _V = V if with_contacts else \
                np.vstack([V, np.array(cV, dtype=np.int64).reshape(-1, 2)])
            x = axis_lp(n, pred[:, 0], wd, _H, _cH, _oV, pre_mask,
                        pred[:, 0], _att['L'], _att['R'], nets, pinx, cfgx, S,
                        hard_attach, hard_contact, eps)
            if x is None:
                return None
            y = axis_lp(n, pred[:, 1], hd, _V, _cV, _oH, pre_mask,
                        pred[:, 1], _att['B'], _att['T'], nets, piny, cfgy, S,
                        hard_attach, hard_contact, eps)
            if y is None:
                return None
            out = pred.copy()
            out[:, 0], out[:, 1] = x, y
            out[:, 2], out[:, 3] = wd, hd
            return out

        def _snap(sol):
            # exact snapping, guarded: only correct solver-precision residue
            # (in soft modes a gap can be genuinely large and must stay a
            # violation, not become a shove)
            snap_tol = 1e-5 * S
            sol[pre_mask, 0:2] = gt[pre_mask, 0:2]
            for i, j in cH:
                gap = sol[j, 0] - (sol[i, 0] + sol[i, 2])
                if abs(gap) >= snap_tol:
                    continue
                if not pre_mask[j]:
                    sol[j, 0] = sol[i, 0] + sol[i, 2]
                elif not pre_mask[i]:
                    sol[i, 0] = sol[j, 0] - sol[i, 2]
            for i, j in cV:
                gap = sol[j, 1] - (sol[i, 1] + sol[i, 3])
                if abs(gap) >= snap_tol:
                    continue
                if not pre_mask[j]:
                    sol[j, 1] = sol[i, 1] + sol[i, 3]
                elif not pre_mask[i]:
                    sol[i, 1] = sol[j, 1] - sol[i, 3]
            bx0, by0 = sol[:, 0].min(), sol[:, 1].min()
            bx1 = (sol[:, 0] + sol[:, 2]).max()
            by1 = (sol[:, 1] + sol[:, 3]).max()
            for i in att['L']:
                if abs(sol[i, 0] - bx0) < snap_tol:
                    sol[i, 0] = bx0
            for i in att['R']:
                if abs(sol[i, 0] + sol[i, 2] - bx1) < snap_tol:
                    sol[i, 0] = bx1 - sol[i, 2]
            for i in att['B']:
                if abs(sol[i, 1] - by0) < snap_tol:
                    sol[i, 1] = by0
            for i in att['T']:
                if abs(sol[i, 1] + sol[i, 3] - by1) < snap_tol:
                    sol[i, 1] = by1 - sol[i, 3]
            return sol

        # primary (pass 1 only): MILP with per-attachment/per-contact binary
        # selection — the solver picks the profitable subset in cost units
        candidates = []
        bits_arr = cons[:, 4].numpy()
        nbits = np.array([max(bin(int(b)).count('1'), 1) for b in bits_arr])
        ben_unit = 2.2 / max(n_soft_norm, 1)   # ~cost saving of one violation
        benL = [ben_unit / nbits[i] for i in att['L']]
        benR = [ben_unit / nbits[i] for i in att['R']]
        mx = None
        if use_milp:
            mx, zH, zAx = axis_milp(n, pred[:, 0], wd, H, cH, ovlV, pre_mask,
                                    pred[:, 0], att['L'], att['R'], nets, pinx,
                                    cfgx, S, benL, benR, [ben_unit] * len(cH),
                                    time_limit=cfg['milp_time'])
        if mx is not None:
            # corner coupling: a corner block's y-side is worth the FULL benefit
            # iff its x-side was just satisfied (a half-satisfied corner earns
            # nothing officially)
            xsel = {}
            for k, i in enumerate(list(att['L'])):
                xsel[int(i)] = zAx[k]
            for k, i in enumerate(list(att['R'])):
                xsel[int(i)] = zAx[len(att['L']) + k]

            def ben_y(i):
                if not (int(bits_arr[i]) & 3):
                    return ben_unit                   # pure T/B block
                return ben_unit if xsel.get(int(i), False) else 0.15 * ben_unit

            benB = [ben_y(i) for i in att['B']]
            benT = [ben_y(i) for i in att['T']]
            ovlH_sel = [o for o, z in zip(ovlH, zH) if z]
            my, _zV, _zAy = axis_milp(n, pred[:, 1], hd, V, cV, ovlH_sel,
                                      pre_mask, pred[:, 1], att['B'], att['T'],
                                      nets, piny, cfgy, S, benB, benT,
                                      [ben_unit] * len(cV),
                                      time_limit=cfg['milp_time'])
            if my is not None:
                out = pred.copy()
                out[:, 0], out[:, 1] = mx, my
                out[:, 2], out[:, 3] = wd, hd
                for bump in range(3):
                    if _penetrations(out) <= 1e-7:
                        break
                    out[:, 0:2] += 0.0   # (penetration here is solver noise only)
                cost = _proxy_cost(_snap(out.copy()), case, S, hpwl_base,
                                   area_base, n_soft_norm)
                candidates.append((cost, out, 'milp'))
                if cost < cfg['good_enough'] + 0.05:
                    candidates.sort(key=lambda t: t[0])
                    return _snap(candidates[0][1]), candidates[0][2]

        # fallback rungs (all-or-nothing modes), skipped when the MILP already
        # produced a solution — hard attachments are sometimes feasible yet
        # ruinous (corner blocks dragged outward)
        milp_ok = any(m == 'milp' for _c, _s, m in candidates)
        for hard_a, hard_c, w_c, w_a, m in (
                (True, True, True, True, 'attach=hard,contact=hard'),
                (False, True, True, True, 'attach=soft,contact=hard'),
                (False, False, True, True, 'attach=soft,contact=soft'),
                (False, False, True, False, 'contacts-only-soft'),
                (False, False, False, True, 'attach-only-soft'),
                (False, False, False, False, 'separation-only')):
            if milp_ok and m != 'attach=hard,contact=hard':
                break        # MILP already covers the soft trade-offs
            if m in ('contacts-only-soft', 'attach-only-soft',
                     'separation-only') and candidates:
                break
            cand = solve(hard_a, hard_c, with_contacts=w_c, with_attach=w_a)
            if cand is None:
                continue
            sep_only = m == 'separation-only'
            for bump in range(3):
                if _penetrations(cand) <= 1e-7:
                    break
                retry = solve(hard_a, hard_c, eps_override=1e-7 * S * 10 ** bump,
                              with_contacts=not sep_only, with_attach=not sep_only)
                if retry is not None:
                    cand = retry
            cost = _proxy_cost(_snap(cand.copy()), case, S, hpwl_base, area_base,
                               n_soft_norm)
            candidates.append((cost, cand, m))
            if m == 'attach=hard,contact=hard' and cost < 1.12:
                break        # already excellent; softer rungs can't add much
        if not candidates:
            return None, 'FAILED'
        candidates.sort(key=lambda t: t[0])
        sol, mode = candidates[0][1], candidates[0][2]

        return _snap(sol), mode

    best, best_cost, best_info = None, float('inf'), {}
    geom = pred.copy()
    for it in range(max(1, int(cfg['iterations']))):
        sol, mode = one_pass(geom, use_milp=(it == 0))
        if sol is None:
            break
        cost = _proxy_cost(sol, case, S, hpwl_base, area_base, n_soft_norm)
        if cost < best_cost:
            best, best_cost = sol, cost
            best_info = {'mode': mode, 'pass': it + 1, 'proxy_cost': cost}
        elif it > 0:
            break            # a pass that doesn't improve won't improve later
        if best_cost < cfg['good_enough']:
            break
        geom = sol

    # ---- bbox-targeted reshape post-pass: the dominant residual on hard cases
    # is a stuck area gap; shrink the soft blocks on the critical extent chain
    # toward the predicted extent (area exact) and re-solve once.
    if best is not None and cfg['bbox_reshape']:
        for _attempt in range(2):
            ag = (_bbox_area(torch.tensor(best)) - area_base) / max(area_base, 1e-6)
            if ag < 0.04 or best_cost < cfg['good_enough']:
                break
            geom2 = best.copy()
            W_sol = (geom2[:, 0] + geom2[:, 2]).max() - geom2[:, 0].min()
            H_sol = (geom2[:, 1] + geom2[:, 3]).max() - geom2[:, 1].min()
            exc_w, exc_h = W_sol / max(Wp, 1e-9), H_sol / max(Hp, 1e-9)
            axis = 'H' if exc_w >= exc_h else 'V'
            col = 2 if axis == 'H' else 3
            Hg, Vg = build_pairs(geom2)
            edges = [tuple(map(int, e)) for e in (Hg if axis == 'H' else Vg)
                     if not (pre_mask[int(e[0])] and pre_mask[int(e[1])])]
            okey = geom2[:, 0] + geom2[:, 2] / 2 if axis == 'H' else \
                geom2[:, 1] + geom2[:, 3] / 2
            size = geom2[:, col]
            order = sorted(range(n), key=lambda i: (okey[i], i))
            head = np.zeros(n)
            parent = {}
            adj = {}
            for i, j in edges:
                adj.setdefault(i, []).append(j)
            for i in order:
                for j in adj.get(i, ()):
                    if head[i] + size[i] > head[j]:
                        head[j] = head[i] + size[i]
                        parent[j] = i
            end = int(np.argmax(head + size))
            path = [end]
            while path[-1] in parent:
                path.append(parent[path[-1]])
            target = Wp if axis == 'H' else Hp
            chain = head[end] + size[end]
            mid = [v for v in path if shrinkable[v]
                   and max(geom2[v, 2] / geom2[v, 3],
                           geom2[v, 3] / geom2[v, 2]) < 3.4]
            tot = sum(size[v] for v in mid)
            if tot <= 0 or chain <= target:
                break
            ratio = max(1.0 - (chain - target) / tot, 0.78)
            for v in mid:
                geom2[v, col] *= ratio
                geom2[v, 5 - col] = areas[v] / geom2[v, col]
            sol2, mode2 = one_pass(geom2, use_milp=False)
            if sol2 is None:
                break
            cost2 = _proxy_cost(sol2, case, S, hpwl_base, area_base, n_soft_norm)
            if cost2 < best_cost:
                best, best_cost = sol2, cost2
                best_info = {'mode': mode2 + '+bboxreshape',
                             'pass': best_info.get('pass', 0), 'proxy_cost': cost2}
            else:
                break

    if best is None:
        return pred_xywh, {'mode': 'FAILED', 'penetration': float('inf'),
                           'move_mean': 0.0, 'runtime_s': time.time() - t0}
    info = dict(best_info)
    info['penetration'] = _penetrations(best)
    info['move_mean'] = float(
        np.abs(best[:, 0:2] - pred_xywh.numpy()[:, 0:2]).mean() / S)
    info['runtime_s'] = time.time() - t0
    if verbose:
        print(f"  mode={info['mode']} pass={info['pass']} "
              f"proxy={info['proxy_cost']:.4f} pen={info['penetration']:.2e} "
              f"move={info['move_mean']:.4f} t={info['runtime_s']:.2f}s")
    # float64 all the way out: float32 rounding (~1e-5 at coord ~200) exceeds the
    # official 1e-6 overlap tolerance and would re-create overlaps
    return torch.tensor(best, dtype=torch.float64), info


def legalize_best_of(cand_list, case, cfg=None, verbose=False):
    """Legalize each prediction candidate (seeds) and keep the best by the
    official-cost proxy — hedges cases where the top seed's structure legalizes
    poorly. cand_list: list of (n,4) tensors."""
    S = float(case['area'].sum().sqrt())
    if case.get('metrics') is not None:
        hpwl_base = float(case['metrics'][6] + case['metrics'][7])
        area_base = float(case['metrics'][0])
    else:
        hpwl_base = float(_whpwl(cand_list[0].double(), case))
        area_base = float(_bbox_area(cand_list[0].double()))
    cons = case['cons']
    n_soft_norm = int((cons[:, 4] > 0).sum())
    for c in (2, 3):
        for g in torch.unique(cons[:, c]):
            if g > 0:
                n_soft_norm += int((cons[:, c] == g).sum()) - 1
    cfg_full = {**DEFAULT_CFG, **(cfg or {})}
    best, total_t = None, 0.0
    for k, pred in enumerate(cand_list):
        sol, info = legalize_case(pred, case, cfg, verbose=verbose)
        total_t += info['runtime_s']
        cost = _proxy_cost(sol.numpy(), case, S, hpwl_base, area_base, n_soft_norm)
        info['seed_rank'] = k
        info['proxy_cost'] = cost
        if best is None or cost < best[0]:
            best = (cost, sol, info)
        if cost < cfg_full['seed_stop']:
            break            # this seed is already excellent
    cost, sol, info = best
    info['runtime_s'] = total_t
    return sol, info


# --------------------------------------------------------------------------- CLI

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred', type=str, required=True)
    ap.add_argument('--out', type=str, default='floordiff/out/legalized.json')
    ap.add_argument('--cases', type=str, default='')
    ap.add_argument('--alpha-mult', type=float, default=DEFAULT_CFG['alpha_mult'])
    ap.add_argument('--gamma-mult', type=float, default=DEFAULT_CFG['gamma_mult'])
    ap.add_argument('--lam-mult', type=float, default=DEFAULT_CFG['lam_mult'])
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()

    cfg = {'alpha_mult': args.alpha_mult, 'gamma_mult': args.gamma_mult,
           'lam_mult': args.lam_mult}
    preds = json.loads(Path(args.pred).read_text())['cases']
    ns = [int(x) for x in args.cases.split(',')] if args.cases else \
        sorted(int(k) for k in preds)

    out = {'cases': {}, 'meta': {'pred': args.pred, 'cfg': cfg}}
    for nb in ns:
        case = load_validation_case(nb)
        entry = preds[str(nb)]
        if 'candidates' in entry:      # best-of-k over stored seeds
            cands = [torch.tensor(c, dtype=torch.float64)
                     for c in entry['candidates']]
            sol, info = legalize_best_of(cands, case, cfg, verbose=args.verbose)
        else:
            pred = torch.tensor(entry['positions'], dtype=torch.float64)
            sol, info = legalize_case(pred, case, cfg, verbose=args.verbose)
        out['cases'][str(nb)] = {
            'n': nb, 'positions': sol.tolist(),
            'runtime_s': preds[str(nb)].get('runtime_s', 0) + info['runtime_s'],
            'legalize': {k: v for k, v in info.items()},
        }
        print(f"case n={nb:>3}: {info['mode']:<28} pen={info['penetration']:.1e} "
              f"move={info['move_mean']:.4f} t={info['runtime_s']:.2f}s")
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out))
    print(f'wrote {p}')


if __name__ == '__main__':
    main()
