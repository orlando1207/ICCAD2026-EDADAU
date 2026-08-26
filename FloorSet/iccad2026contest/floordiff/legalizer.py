"""EGL: ePlace-Gradient + Graph legalizer (stage 2).

Implements docs/superpowers/specs/2026-07-20-egl-legalizer-design.md:
  stamp  hard constraints (preplaced pos+dims, fixed dims, compatible MIB dims)
  G      ePlace-lite gradient phase: weighted HPWL + pairwise overlap field +
         bbox / boundary / cluster springs + anchor, Nesterov with Lipschitz
         step prediction and Jacobi preconditioning (ePlace Alg. 2, Eqs 29-33)
  L      constraint-graph legalization: axis per pair from G geometry, repair
         against preplaced anchors (reshape then flip, NTUplace-style), then
         Tetris/topological minimal-movement assignment (zero overlap by
         construction)
  P      Abacus-style per-axis L1 median sweeps within fresh slack intervals
  S      profit-gated exact snapping (boundary sides, cluster abutment)
  R      bounded DAG ripple repair for close cluster components blocked by
         neighboring rectangles
  Q      bounded area-preserving reshape repair for residual group gaps and
         preplaced boundary anchors

No LP/MILP anywhere; vectorized numpy + small per-block loops. All edges are
oriented by a FIXED per-axis total order (the G-phase centers), which keeps
both graphs acyclic through repair flips and makes argsort(key) a valid
topological order everywhere.

Run from iccad2026contest/:
  python -m floordiff.legalizer --pred floordiff/out/preds_full.json \
      --out floordiff/out/legalized.json [--cases 100,120]
"""

import argparse
import heapq
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from shapely.geometry import box as _sbox
from shapely.ops import unary_union as _sunion

from .data import load_validation_case, target_xywh
from .evaluate import bbox_area as _bbox_area
from .evaluate import weighted_hpwl as _whpwl

DEFAULT_CFG = {
    'g_iters': 100,          # gradient-phase iteration cap (local cleanup)
    'g_tol_rel': 1e-4,       # G stops when max penetration < this x S
    'attach_tol_rel': 0.05,  # boundary spring engages within this x S
    'delta_rel': 1e-3,       # min shared-edge length for cluster abutment, x S
    'anchor_w': 0.02,        # anchor spring strength (cost units per S moved)
    'polish_sweeps': 4,
    'seed_stop': 1.05,       # stop trying more candidates below this proxy
    'budget_s': 4.0,         # per-case wall-clock budget for exploration
    # stage-2 detailed placement:
    'area_scale': 0.991,     # soft blocks use 99.1% of target area (contest
                             # allows 1% under; frees packing slack -> area_gap).
                             # strict diff>0.01 check, so 0.991 (diff 0.009) is safe
    'cluster_moves': 48,     # max rigid component merges per cluster_repair call
    'cluster_drag_rel': 0.30,  # skip merges needing a drag beyond this x S
    'perp_align': True,      # stage DP-1: shared-edge repair within graph slack
    'perp_moves': 40,        # max slack shifts per cluster_perp_align call
    'cluster_align': True,   # stage L: band-align cluster members at assignment
                             # time (while slack still exists) so abutting
                             # members share an edge instead of a corner
    # bounded ripple repair for close, blocked cluster components.  Each trial
    # is a pair-edge swap plus two O(n^2) DAG assignments; strict caps keep this
    # stage small relative to diffusion sampling and the two legal rounds.
    'ripple_repair': True,
    'ripple_close_rel': 0.06,
    'ripple_drag_rel': 0.08,
    'ripple_budget_s': 0.04,
    'ripple_trials': 32,
    'ripple_moves': 4,
    'ripple_pair_choices': 3,
    'ripple_max_blocks': 24,
    'ripple_total_disp_rel': 0.30,
    # Shape-aware detailed placement.  Residual group gaps are closed by
    # elongating one soft block toward contact (area exact); small bbox sides
    # beyond a preplaced boundary anchor are contracted by shrinking a critical
    # path.  Every trial is proxy-gated and stays inside the entry bbox.
    'reshape_repair': True,
    'reshape_group': True,
    'reshape_boundary': True,
    'reshape_close_rel': 0.02,
    'reshape_budget_s': 0.025,
    'reshape_trials': 16,
    'reshape_moves': 2,
    'reshape_max_growth': 0.10,
    'reshape_max_shrink': 0.10,
    'reshape_aspect_cap': 3.6,
    'reshape_max_blocks': 48,
    'reshape_total_disp_rel': 0.20,
    # Feasibility completeness (see FEASIBILITY_ANALYSIS.md 4-5).
    'evict_repair': True,    # stage E: evict blocks off anchor-conflicting
                             # critical paths until the relation set is
                             # consistent -> assignment cannot overlap
    'evict_max': 64,         # eviction budget per legal_round
    # 'above' | 'right' | 'auto'. Keep 'above'. 'auto' picks the side that looks
    # cheaper in bbox area and MEASURED WORSE (stress kit total 1.6938 -> 1.7833,
    # 23 cases better / 42 worse): a shelf above the core sits ON the bbox's top
    # edge, so evicted blocks satisfy a TOP boundary bit for free -- only 1% of
    # boundary violations land on evicted blocks. Moving the shelf sideways
    # throws that away for an area saving that does not cover it.
    'evict_shelf': 'above',
    # victim scoring: (boundary, cluster, mib, degree, area)
    'evict_weights': (4.0, 2.0, 1.0, 2.0, 1.0),
    'reclaim': True,         # pull evicted blocks back into free holes
    'reclaim_trials': 24,    # max blocks reclaimed per call
    'reclaim_probe': 8,      # free positions scored with the real proxy per block
    'reclaim_round': True,   # re-run a legal round so the soft stages see the
                             # reclaimed blocks (they were in the shelf before)
    'reclaim_budget_s': 0.05,
    'guaranteed_floor': True,   # never return an infeasible solution
    # relation-set construction (see build_graph)
    'graph_rule': 'gap',     # 'gap' | 'norm' | 'square' | 'anchor'
    'graph_square_k': 0.25,  # 'square' bias strength toward the shorter bbox side
    'graph_key': 'center',   # per-axis total order: 'center' | 'edge'
    # Graph rebuilds from legal geometry. Measured on the official 100 with the
    # DP time budgets neutralised (see tools/README.md): 2 -> 1.0492,
    # 3 -> 1.0466, 5 -> 1.0466, i.e. the gain saturates at 3. `legal_round_stop`
    # takes that gain at close to 2-round wall-clock by stopping as soon as a
    # rebuild fails to improve the proxy.
    'legal_rounds': 3,
    'legal_round_stop': True,
    # stage-G shape (were hard-coded; exposed so they can be A/B'd)
    'g_omega': 0.8,          # overlap-impulse relaxation
    'g_drift_rel': 0.002,    # quality-gradient step cap, x S per iteration
    'g_spring_rate': 0.4,    # boundary/cluster gap fraction closed per iteration
    'g_spring_cap_rel': 0.01,
}

_EPS_OVL = 1e-6              # official: pair violates only if BOTH axes > 1e-6
_EPS_DIM = 1e-4              # official immutable-coordinate tolerance
_AREA_TOL = 0.01             # official relative soft-block area tolerance
_AREA_SAFE_TOL = _AREA_TOL - 1e-9  # stay off floating-point acceptance edge


# ------------------------------------------------------------------ metrics

def max_penetration(xywh):
    x0, y0, w, h = xywh[:, 0], xywh[:, 1], xywh[:, 2], xywh[:, 3]
    x1, y1 = x0 + w, y0 + h
    ox = np.minimum(x1[:, None], x1) - np.maximum(x0[:, None], x0)
    oy = np.minimum(y1[:, None], y1) - np.maximum(y0[:, None], y0)
    both = np.minimum(ox, oy)
    np.fill_diagonal(both, -1.0)
    return max(0.0, float(both.max()))


def hard_feasibility(sol, case):
    """Return official hard-constraint status plus useful diagnostics.

    Missing area/target data is skipped so focused geometry helpers can use this
    validator too. Numeric/positive-size checks guard malformed model output.
    """
    xywh = np.asarray(sol, dtype=np.float64)
    n = len(xywh)
    cons_t = case.get('cons')
    cons = (cons_t.detach().cpu().numpy() if torch.is_tensor(cons_t)
            else np.asarray(cons_t)) if cons_t is not None else np.zeros((n, 2))
    fixed = cons[:n, 0] != 0 if cons.shape[1] > 0 else np.zeros(n, dtype=bool)
    pre = cons[:n, 1] != 0 if cons.shape[1] > 1 else np.zeros(n, dtype=bool)

    finite_rows = np.isfinite(xywh).all(axis=1)
    positive = (xywh[:, 2] > 0) & (xywh[:, 3] > 0)
    numeric_bad = np.nonzero(~finite_rows | ~positive)[0]

    # Match check_overlap(): both intersections must be strictly above 1e-6.
    safe = xywh.copy()
    safe[~finite_rows] = 0.0
    x0, y0, w, h = safe.T
    x1, y1 = x0 + w, y0 + h
    ox = np.minimum(x1[:, None], x1) - np.maximum(x0[:, None], x0)
    oy = np.minimum(y1[:, None], y1) - np.maximum(y0[:, None], y0)
    overlap_mask = np.triu((ox > _EPS_OVL) & (oy > _EPS_OVL), k=1)
    overlap_pairs = np.argwhere(overlap_mask)
    pair_pen = np.minimum(ox, oy)[overlap_mask]

    area_bad = np.empty(0, dtype=np.int64)
    max_area_error = 0.0
    area_t = case.get('area')
    if area_t is not None:
        target_area = (area_t.detach().cpu().numpy() if torch.is_tensor(area_t)
                       else np.asarray(area_t)).astype(np.float64)[:n]
        check = (~fixed) & (~pre) & (target_area > 0) & finite_rows
        rel = np.zeros(n, dtype=np.float64)
        rel[check] = np.abs(xywh[check, 2] * xywh[check, 3]
                            - target_area[check]) / target_area[check]
        area_bad = np.nonzero(check & (rel > _AREA_TOL))[0]
        if check.any():
            max_area_error = float(rel[check].max())

    dim_bad = np.empty(0, dtype=np.int64)
    max_dimension_error = 0.0
    target = None
    if case.get('gt') is not None:
        target = target_xywh(case).detach().cpu().numpy()
    elif case.get('target') is not None:
        target_t = case['target']
        target = (target_t.detach().cpu().numpy() if torch.is_tensor(target_t)
                  else np.asarray(target_t))
    if target is not None:
        target = np.asarray(target, dtype=np.float64)[:n]
        immutable = fixed | pre
        err = np.zeros(n, dtype=np.float64)
        if immutable.any():
            err[fixed] = np.max(np.abs(xywh[fixed, 2:4]
                                       - target[fixed, 2:4]), axis=1)
            err[pre] = np.max(np.abs(xywh[pre] - target[pre]), axis=1)
            dim_bad = np.nonzero(immutable & (err > _EPS_DIM))[0]
            max_dimension_error = float(err[immutable].max())

    total = (len(overlap_pairs) + len(area_bad) + len(dim_bad)
             + len(numeric_bad))
    return {
        'feasible': total == 0,
        'total_violations': int(total),
        'overlap_violations': int(len(overlap_pairs)),
        'area_violations': int(len(area_bad)),
        'dimension_violations': int(len(dim_bad)),
        'numeric_violations': int(len(numeric_bad)),
        'overlap_pairs': overlap_pairs.tolist(),
        'area_blocks': area_bad.tolist(),
        'dimension_blocks': dim_bad.tolist(),
        'numeric_blocks': numeric_bad.tolist(),
        'max_penetration': float(pair_pen.max()) if len(pair_pen) else 0.0,
        'max_area_error': max_area_error,
        'max_dimension_error': max_dimension_error,
    }


def _selection_key(info, seed_rank=0):
    """Feasible candidates always beat infeasible candidates, deterministically."""
    hard = info['hard']
    if hard['feasible']:
        return (0, float(info['proxy_cost']), int(seed_rank))
    return (1, int(hard['total_violations']),
            int(hard['overlap_violations']), int(hard['area_violations']),
            int(hard['dimension_violations']), int(hard['numeric_violations']),
            float(info['proxy_cost']), int(seed_rank))


def _tie_compatible_mib_dims(pred, case, fixed_mask, pre_mask):
    """Tie MIB shapes only when equality is compatible with every hard rule.

    MIB equality is soft. An incompatible group is left untied: frozen members
    remain exact and each ordinary member retains its legal area.
    """
    cons_t = case['cons']
    cons = (cons_t.detach().cpu().numpy() if torch.is_tensor(cons_t)
            else np.asarray(cons_t))
    area_t = case['area']
    target_area = (area_t.detach().cpu().numpy() if torch.is_tensor(area_t)
                   else np.asarray(area_t)).astype(np.float64)
    stats = {'groups': 0, 'tied': 0, 'incompatible': 0,
             'incompatible_frozen': 0, 'incompatible_area': 0}

    for g in np.unique(cons[:, 2]):
        if g == 0:
            continue
        stats['groups'] += 1
        mem = np.nonzero(cons[:, 2] == g)[0]
        frozen = mem[fixed_mask[mem] | pre_mask[mem]]
        soft = mem[~(fixed_mask[mem] | pre_mask[mem])]

        if len(frozen):
            # Frozen shapes are restored exactly; rounded equality is the MIB
            # evaluator's own comparison and requires no immutable rewrite.
            rounded = {(round(float(pred[i, 2]), 4),
                        round(float(pred[i, 3]), 4)) for i in frozen}
            if len(rounded) != 1:
                stats['incompatible'] += 1
                stats['incompatible_frozen'] += 1
                continue
            rep = int(frozen[0])
            common_area = float(pred[rep, 2] * pred[rep, 3])
            if any(abs(common_area - target_area[i]) / target_area[i]
                   > _AREA_SAFE_TOL for i in soft if target_area[i] > 0):
                stats['incompatible'] += 1
                stats['incompatible_area'] += 1
                continue
            pred[soft, 2:4] = pred[rep, 2:4]
            stats['tied'] += 1
            continue

        # All-soft groups admit a common legal shape iff their 1% target-area
        # intervals intersect. Preserve the representative when already legal.
        if not (target_area[mem] > 0).all():
            stats['incompatible'] += 1
            stats['incompatible_area'] += 1
            continue
        lo = float(np.max((1.0 - _AREA_SAFE_TOL) * target_area[mem]))
        hi = float(np.min((1.0 + _AREA_SAFE_TOL) * target_area[mem]))
        if lo > hi:
            stats['incompatible'] += 1
            stats['incompatible_area'] += 1
            continue
        rep = int(mem[0])
        rep_area = float(pred[rep, 2] * pred[rep, 3])
        common_area = min(max(rep_area, lo), hi)
        ratio = max(float(pred[rep, 2] / pred[rep, 3]), 1e-12)
        common_w = math.sqrt(common_area * ratio)
        common_h = common_area / common_w
        pred[mem, 2] = common_w
        pred[mem, 3] = common_h
        stats['tied'] += 1
    return stats


def _violations_official(sol, cons):
    """(v_boundary, v_grouping, v_mib) with the official evaluator's semantics."""
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


def _n_soft_norm(cons):
    n = int((cons[:, 4] > 0).sum())
    for c in (2, 3):
        for g in np.unique(cons[:, c]):
            if g > 0:
                n += int((cons[:, c] == g).sum()) - 1
    return n


def proxy_cost(sol_np, case, hpwl_base, area_base, n_soft):
    """Official cost formula (RuntimeFactor neutral); 10.0 if hard-invalid."""
    if not hard_feasibility(sol_np, case)['feasible']:
        return 10.0
    t = torch.tensor(sol_np, dtype=torch.float64)
    hg = (_whpwl(t, case) - hpwl_base) / max(hpwl_base, 1e-6)
    ag = (_bbox_area(t) - area_base) / max(area_base, 1e-6)
    vb, vg, vm = _violations_official(sol_np, case['cons'].numpy())
    vr = (vb + vg + vm) / max(n_soft, 1)
    return min((1 + 0.5 * (hg + ag)) * math.exp(2 * vr), 10 - 1e-6)


def _baselines(case, pred):
    if case.get('metrics') is not None:
        return (float(case['metrics'][6] + case['metrics'][7]),
                float(case['metrics'][0]))
    t = torch.tensor(pred, dtype=torch.float64)
    return float(_whpwl(t, case)), float(_bbox_area(t))


def _cluster_forest(xywh, members):
    """Proximity spanning forest inside one cluster group: [(i, j), ...]."""
    m = len(members)
    if m < 2:
        return []
    x0, y0 = xywh[members, 0], xywh[members, 1]
    x1 = x0 + xywh[members, 2]
    y1 = y0 + xywh[members, 3]
    gx = np.maximum(x0[:, None], x0) - np.minimum(x1[:, None], x1)
    gy = np.maximum(y0[:, None], y0) - np.minimum(y1[:, None], y1)
    gap = np.maximum(gx, gy)
    iu, ju = np.triu_indices(m, k=1)
    order = np.argsort(gap[iu, ju], kind='stable')
    parent = list(range(m))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    edges, picked = [], 0
    for k in order:
        a, b = int(iu[k]), int(ju[k])
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        parent[ra] = rb
        edges.append((int(members[a]), int(members[b])))
        picked += 1
        if picked == m - 1:
            break
    return edges


# ------------------------------------------------------------------ stage G

def gradient_phase(pred, case, pre_mask, cfg, S, hpwl_base, area_base,
                   iters=None):
    """ePlace-lite overlap-removal loop on block centers.

    Overlap term is applied as a penetration-proportional impulse (move =
    relaxation x depth/2, area-weighted like ePlace's charge preconditioner)
    instead of a raw force step — for a near-legal warm start this converges
    geometrically and cannot fly apart. Quality terms (weighted HPWL, bbox,
    boundary/cluster springs, anchor) drift with a small bounded step.
    Returns (n,4) xywh."""
    n = pred.shape[0]
    w, h = pred[:, 2].copy(), pred[:, 3].copy()
    c0 = np.stack([pred[:, 0] + w / 2, pred[:, 1] + h / 2], axis=1)
    areas = w * h
    iters = iters if iters is not None else cfg['g_iters']

    b2b = case['b2b'].numpy()
    p2b = case['p2b'].numpy()
    pins = case['pins'].numpy()
    ei = b2b[:, 0].astype(np.int64) if len(b2b) else np.zeros(0, np.int64)
    ej = b2b[:, 1].astype(np.int64) if len(b2b) else np.zeros(0, np.int64)
    ew = b2b[:, 2] if len(b2b) else np.zeros(0)
    pb = p2b[:, 1].astype(np.int64) if len(p2b) else np.zeros(0, np.int64)
    pp = p2b[:, 0].astype(np.int64) if len(p2b) else np.zeros(0, np.int64)
    pw = p2b[:, 2] if len(p2b) else np.zeros(0)
    gam = 1e-3 * S
    alpha = 0.5 / max(hpwl_base, 1e-6)          # cost units per HPWL unit
    garea = 0.5 / max(area_base, 1e-6)

    # per-block weighted degree (Jacobi preconditioner, ePlace Eq. 31)
    deg = np.zeros(n)
    np.add.at(deg, ei, ew)
    np.add.at(deg, ej, ew)
    np.add.at(deg, pb, pw)
    deg = alpha * np.maximum(deg, 1e-12)

    cons = case['cons'].numpy()
    bits = cons[:, 4]
    bnd_idx = np.nonzero(bits)[0]
    att_tol = cfg['attach_tol_rel'] * S

    groups = [np.nonzero(cons[:, 3] == g)[0]
              for g in np.unique(cons[:, 3]) if g > 0]
    forest = []
    idx = np.arange(n)
    tie = np.where(idx[:, None] > idx[None, :], 1.0, -1.0)

    def hpwl_grad(cc):
        g = np.zeros_like(cc)
        if len(ei):
            d = cc[ei] - cc[ej]
            s = d / np.sqrt(d * d + gam * gam)
            np.add.at(g, ei, ew[:, None] * s)
            np.add.at(g, ej, -ew[:, None] * s)
        if len(pb):
            d = cc[pb] - pins[pp]
            s = d / np.sqrt(d * d + gam * gam)
            np.add.at(g, pb, pw[:, None] * s)
        return alpha * g

    # pair mass weights: lighter block absorbs more of the separation
    # (ePlace charge preconditioning); preplaced blocks absorb none
    mass = areas.copy()
    mass[pre_mask] = np.inf
    with np.errstate(invalid='ignore'):
        wt = mass[None, :] / (mass[:, None] + mass[None, :])  # share, row i
    wt = np.nan_to_num(wt, nan=0.0, posinf=1.0)

    def overlap_impulse(cc):
        """Per-block displacement resolving each penetrating pair along its
        cheaper axis by its full depth, mass-shared. Returns (disp, max_pen)."""
        x0 = cc[:, 0] - w / 2
        y0 = cc[:, 1] - h / 2
        x1, y1 = x0 + w, y0 + h
        ox = np.minimum(x1[:, None], x1) - np.maximum(x0[:, None], x0)
        oy = np.minimum(y1[:, None], y1) - np.maximum(y0[:, None], y0)
        pen = (ox > 0) & (oy > 0)
        np.fill_diagonal(pen, False)
        d = np.zeros_like(cc)
        if not pen.any():
            return d, 0.0
        dx = cc[:, 0][:, None] - cc[:, 0][None, :]
        dy = cc[:, 1][:, None] - cc[:, 1][None, :]
        sx = np.where(dx == 0, tie, np.sign(dx))
        sy = np.where(dy == 0, tie, np.sign(dy))
        ux = pen & (ox <= oy)
        uy = pen & ~ux
        d[:, 0] = (np.where(ux, sx * ox * wt, 0.0)).sum(1)
        d[:, 1] = (np.where(uy, sy * oy * wt, 0.0)).sum(1)
        mx = float(np.minimum(ox, oy)[pen].max())
        return d, mx

    def bbox_grad(cc):
        """Subgradient of 0.5*area_gap, softmax-shared over extreme blocks."""
        x0 = cc[:, 0] - w / 2
        y0 = cc[:, 1] - h / 2
        x1, y1 = x0 + w, y0 + h
        W = x1.max() - x0.min()
        H = y1.max() - y0.min()
        tau = 0.02 * S
        g = np.zeros_like(cc)

        def soft(vals, hi):
            z = (vals - vals.max()) / tau if hi else (vals.min() - vals) / tau
            e = np.exp(z)
            return e / e.sum()

        g[:, 0] += garea * H * (soft(x1, True) - soft(x0, False))
        g[:, 1] += garea * W * (soft(y1, True) - soft(y0, False))
        return g

    spring_rate = float(cfg.get('g_spring_rate', 0.4))   # gap fraction / iter
    spring_cap = float(cfg.get('g_spring_cap_rel', 0.01)) * S

    def spring_impulse(cc):
        """Direct gap-closing moves for boundary attachment and cluster
        contacts — decoupled from the quality-drift cap so a 5%-S gap closes
        in a handful of iterations."""
        d = np.zeros_like(cc)
        x0 = cc[:, 0] - w / 2
        y0 = cc[:, 1] - h / 2
        x1, y1 = x0 + w, y0 + h
        bx0, bx1 = x0.min(), x1.max()
        by0, by1 = y0.min(), y1.max()

        def pull(v):
            return min(spring_rate * v, spring_cap)

        for i in bnd_idx:
            b = int(bits[i])
            if b & 1 and 0 < x0[i] - bx0 < att_tol:
                d[i, 0] -= pull(x0[i] - bx0)
            if b & 2 and 0 < bx1 - x1[i] < att_tol:
                d[i, 0] += pull(bx1 - x1[i])
            if b & 4 and 0 < by1 - y1[i] < att_tol:
                d[i, 1] += pull(by1 - y1[i])
            if b & 8 and 0 < y0[i] - by0 < att_tol:
                d[i, 1] -= pull(y0[i] - by0)
        for (i, j) in forest:
            gx = max(x0[i], x0[j]) - min(x1[i], x1[j])
            gy = max(y0[i], y0[j]) - min(y1[i], y1[j])
            if max(gx, gy) <= 0:
                continue               # overlapping/touching already
            if gx >= gy:               # pull together along x
                s = 1.0 if cc[i, 0] < cc[j, 0] else -1.0
                mv = pull(gx)
                d[i, 0] += 0.5 * s * mv
                d[j, 0] -= 0.5 * s * mv
            else:
                s = 1.0 if cc[i, 1] < cc[j, 1] else -1.0
                mv = pull(gy)
                d[i, 1] += 0.5 * s * mv
                d[j, 1] -= 0.5 * s * mv
        return d

    anchor_k = cfg['anchor_w'] / S
    omega = float(cfg.get('g_omega', 0.8))               # impulse relaxation
    drift_cap = float(cfg.get('g_drift_rel', 0.002)) * S  # per-block quality drift

    c = c0.copy()
    for it in range(iters):
        if it % 20 == 0:
            xy = np.stack([c[:, 0] - w / 2, c[:, 1] - h / 2, w, h], axis=1)
            forest = []
            for mem in groups:
                forest += _cluster_forest(xy, mem)
        disp, mx_pen = overlap_impulse(c)
        if mx_pen < cfg['g_tol_rel'] * S:
            break
        g = hpwl_grad(c) + bbox_grad(c) + anchor_k * (c - c0)
        g = g / np.maximum(deg + areas / (S * S), 1e-12)[:, None]
        gmax = np.abs(g).max()
        if gmax > 0:
            g *= min(1.0, drift_cap / gmax)
        move = omega * disp + spring_impulse(c) - g
        move[pre_mask] = 0.0
        c += move

    out = np.empty((n, 4))
    out[:, 0] = c[:, 0] - w / 2
    out[:, 1] = c[:, 1] - h / 2
    out[:, 2], out[:, 3] = w, h
    return out


# ------------------------------------------------------------------ stage L

def build_graph(xywh, keyx, keyy, rule='gap', pre_mask=None, square_k=0.25):
    """Axis per pair, oriented by the FIXED keys. Returns (H, V) lists of
    (leader, follower).

    `rule` selects which axis a pair is assigned to:
      'gap'    H iff x-gap >= y-gap.  For an overlapping pair both gaps are
               negative and this picks the smaller penetration, i.e. the cheaper
               axis to separate on; for a separated pair it picks the axis with
               more margin.
      'norm'   the same comparison on gaps normalised by the pair's summed
               extent, so which axis is "cheaper" does not depend on the blocks
               being wide or tall.
      'square' 'norm' plus a bias onto the shorter bbox side -- MEASURED HARMFUL,
               see the note in the body; do not enable.
      'anchor' 'gap', except that for a pair with exactly one preplaced member
               an axis on which the two are ALREADY disjoint wins.  Such a
               relation holds at the anchor's fixed coordinate by construction,
               so it can never put the block on an over-full anchor-to-anchor
               path -- it removes single-anchor conflicts before stage E has to.
    """
    x0, y0, w, h = xywh[:, 0], xywh[:, 1], xywh[:, 2], xywh[:, 3]
    x1, y1 = x0 + w, y0 + h
    gx = np.maximum(x0[:, None], x0) - np.minimum(x1[:, None], x1)
    gy = np.maximum(y0[:, None], y0) - np.minimum(y1[:, None], y1)
    n = len(x0)
    iu, ju = np.triu_indices(n, k=1)
    if rule in ('norm', 'square'):
        sx = np.maximum(w[iu] + w[ju], 1e-12)
        sy = np.maximum(h[iu] + h[ju], 1e-12)
        score = gx[iu, ju] / sx - gy[iu, ju] / sy
        if rule == 'square':
            # MEASURED HARMFUL -- kept only to document the negative result.
            # The idea was that a pair placed in H stands side by side (growing
            # width) while in V it stacks (growing height), so biasing ambiguous
            # pairs onto the shorter bbox side should square the layout up. It
            # does the opposite: piling relations onto one axis lengthens that
            # axis' critical path, so the assignment's minimum extent grows.
            # Official 100, runtime-neutral total vs 1.0496 baseline:
            #   k=0.15 1.0467 · k=0.25 1.3607 · k=0.50 1.5262 · k=1.00 1.7279
            # k=0.15's small win does not survive the cliff behind it.
            bw = float(x1.max() - x0.min())
            bh = float(y1.max() - y0.min())
            score -= square_k * math.log(max(bw, 1e-12) / max(bh, 1e-12))
        horiz = score >= 0
    else:
        horiz = gx[iu, ju] >= gy[iu, ju]
    if rule == 'anchor' and pre_mask is not None and pre_mask.any():
        one_anchor = pre_mask[iu] ^ pre_mask[ju]
        sep_x = gx[iu, ju] >= 0.0
        sep_y = gy[iu, ju] >= 0.0
        horiz = np.where(one_anchor & sep_x & ~sep_y, True,
                         np.where(one_anchor & sep_y & ~sep_x, False, horiz))

    def orient(ii, jj, key):
        swap = (key[ii] > key[jj]) | ((key[ii] == key[jj]) & (ii > jj))
        return list(zip(np.where(swap, jj, ii).tolist(),
                        np.where(swap, ii, jj).tolist()))

    H = orient(iu[horiz], ju[horiz], keyx)
    V = orient(iu[~horiz], ju[~horiz], keyy)
    return H, V


def _find_conflict(n, edges, pos, size, pre_mask, order, origin=-np.inf):
    """Anchored longest path (sources at `origin`, e.g. a boundary wall);
    first preplaced node whose lower bound exceeds its fixed position ->
    (edge_path, node_path, excess), else None."""
    adj = {}
    for k, (i, j) in enumerate(edges):
        adj.setdefault(i, []).append((j, k))
    lb = np.full(n, origin)
    lb[pre_mask] = pos[pre_mask]
    parent = {}
    for i in order:
        if lb[i] == -np.inf:
            continue
        if pre_mask[i] and lb[i] > pos[i] + 1e-7:
            path, nodes, v = [], [i], i
            while v in parent:
                path.append(parent[v][1])
                v = parent[v][0]
                nodes.append(v)
            return path, nodes[::-1], float(lb[i] - pos[i])
        base = pos[i] if pre_mask[i] else lb[i]
        for j, k in adj.get(i, ()):
            if base + size[i] > lb[j]:
                lb[j] = base + size[i]
                parent[j] = (i, k)
    return None


def repair_graph(H, V, geom, keyx, keyy, pre_mask, shrinkable, areas,
                 max_rounds=200, max_shrink=0.25, aspect_cap=3.6,
                 origin_x=-np.inf, origin_y=-np.inf):
    """Make both axis graphs consistent with preplaced anchors.
    Remedies per conflict: reshape critical-path soft blocks (area exact),
    else flip the cheapest path edge to the other axis.
    Returns (H, V, w, h, ok)."""
    n = geom.shape[0]
    w, h = geom[:, 2].copy(), geom[:, 3].copy()
    x0, y0 = geom[:, 0], geom[:, 1]
    ordx = np.lexsort((np.arange(n), keyx)).tolist()
    ordy = np.lexsort((np.arange(n), keyy)).tolist()
    gx = np.maximum(x0[:, None], x0) - np.minimum(
        (x0 + w)[:, None], x0 + w)
    gy = np.maximum(y0[:, None], y0) - np.minimum(
        (y0 + h)[:, None], y0 + h)
    flipped = set()
    reshaped = {'H': set(), 'V': set()}

    for _ in range(max_rounds):
        res, axis = _find_conflict(n, H, x0, w, pre_mask, ordx,
                                   origin_x), 'H'
        if res is None:
            res, axis = _find_conflict(n, V, y0, h, pre_mask, ordy,
                                       origin_y), 'V'
        if res is None:
            return H, V, w, h, True
        path, nodes, excess = res
        size, other = (w, h) if axis == 'H' else (h, w)
        # remedy 1: reshape shrinkable path nodes (conflict endpoint is
        # preplaced and never included; the source may be reshapable)
        mid = [v for v in nodes[:-1]
               if shrinkable[v] and v not in reshaped[axis]]
        need = excess * 1.0001 + 1e-9
        while mid and need > 1e-9:
            tot = sum(size[v] for v in mid)
            ratio = max(1.0 - need / tot, 1.0 - max_shrink)
            ok = []
            for v in mid:
                ns = size[v] * ratio
                no = areas[v] / ns
                if max(no / ns, ns / no) <= aspect_cap:
                    ok.append(v)
            if len(ok) < len(mid):
                mid = ok
                continue
            for v in mid:
                size[v] *= ratio
                other[v] = areas[v] / size[v]
                reshaped[axis].add(v)
            need -= (1.0 - ratio) * tot
            break
        if need <= 1e-9:
            continue
        # remedy 2: flip the path edge with the most other-axis gap
        edges = H if axis == 'H' else V
        oth = V if axis == 'H' else H
        okey = keyy if axis == 'H' else keyx
        gother = gy if axis == 'H' else gx
        cand = [k for k in path
                if (min(edges[k]), max(edges[k])) not in flipped
                and not (pre_mask[edges[k][0]] and pre_mask[edges[k][1]])]
        if not cand:
            return H, V, w, h, False
        best = max(cand, key=lambda k: gother[edges[k][0], edges[k][1]])
        i, j = edges.pop(best)
        flipped.add((min(i, j), max(i, j)))
        if (okey[i], i) > (okey[j], j):
            i, j = j, i
        oth.append((i, j))
    return H, V, w, h, False


def assign_axis(n, edges, target, size, pre_mask, pre_pos, order, eps=0.0,
                wall_hi=None, wall_lo=None, contact=None):
    """Tetris/topological minimal-movement assignment along one axis:
    pos_i = clip(target_i, max over preds (pos_p + size_p), U_i).
    wall_hi/wall_lo (optional) bound every block to [wall_lo, wall_hi].
    contact (optional) maps follower -> leader: the follower targets exact
    abutment with its (already-assigned) leader — cluster contacts form
    during assignment instead of being slid into place afterwards."""
    succ, pred = {}, {}
    for i, j in edges:
        succ.setdefault(i, []).append(j)
        pred.setdefault(j, []).append(i)
    U = np.full(n, np.inf)
    for i in reversed(order):
        ui = np.inf if wall_hi is None else wall_hi - size[i]
        for j in succ.get(i, ()):
            ui = min(ui, U[j] - size[i] - eps)
        if pre_mask[i]:
            ui = pre_pos[i]
        U[i] = ui
    pos = np.empty(n)
    base = -np.inf if wall_lo is None else wall_lo
    for i in order:
        if pre_mask[i]:
            pos[i] = pre_pos[i]
            continue
        lb = base
        for p in pred.get(i, ()):
            lb = max(lb, pos[p] + size[p] + eps)
        ti = target[i]
        if contact is not None and i in contact:
            ld = contact[i]
            ti = pos[ld] + size[ld]        # exact abutment with the leader
        t = ti if lb == -np.inf else max(ti, lb)
        if np.isfinite(U[i]):
            t = min(t, U[i])
        if lb != -np.inf:
            t = max(t, lb)      # overlap-freedom wins over a stale U residue
        pos[i] = t
    return pos


def min_extent(n, edges, size, pre_mask, pre_pos, order, origin):
    """(critical-path minimum wall, lb positions, critical node path) for one
    axis, with sources at `origin` and preplaced blocks pinned."""
    pred = {}
    for i, j in edges:
        pred.setdefault(j, []).append(i)
    L = np.empty(n)
    parent = {}
    for i in order:
        lb, par = origin, None
        for p in pred.get(i, ()):
            if L[p] + size[p] > lb:
                lb, par = L[p] + size[p], p
        L[i] = pre_pos[i] if pre_mask[i] else lb
        if par is not None and not pre_mask[i]:
            parent[i] = par
    end = int(np.argmax(L + size))
    path = [end]
    while path[-1] in parent:
        path.append(parent[path[-1]])
    return float(L[end] + size[end]), L, path[::-1]


def reshape_chain(path, axis, sol, areas, shrinkable, excess,
                  max_shrink=0.25, aspect_cap=3.6):
    """Shrink the chain axis of shrinkable path blocks (area exact) to absorb
    `excess` of critical-path length. Returns absorbed amount."""
    col, ocol = (2, 3) if axis == 0 else (3, 2)
    mid = [v for v in path if shrinkable[v]]
    absorbed = 0.0
    while mid and absorbed < excess - 1e-9:
        tot = sum(sol[v, col] for v in mid)
        ratio = max(1.0 - (excess - absorbed) / tot, 1.0 - max_shrink)
        ok = []
        for v in mid:
            ns = sol[v, col] * ratio
            no = areas[v] / ns
            if max(no / ns, ns / no) <= aspect_cap:
                ok.append(v)
        if len(ok) < len(mid):
            mid = ok
            continue
        for v in mid:
            sol[v, col] *= ratio
            sol[v, ocol] = areas[v] / sol[v, col]
        absorbed += (1.0 - ratio) * tot
        break
    return absorbed


def chain_ends(n, edges, size, order):
    """Pure size-based longest chain into (head) and out of (tail, inclusive)
    each node — used to price flipping an edge onto this axis."""
    head = np.zeros(n)
    tail = size.astype(np.float64).copy()
    succ = {}
    for i, j in edges:
        succ.setdefault(i, []).append(j)
    for i in order:
        for j in succ.get(i, ()):
            head[j] = max(head[j], head[i] + size[i])
    for i in reversed(order):
        for j in succ.get(i, ()):
            tail[i] = max(tail[i], size[i] + tail[j])
    return head, tail


def extent_repair(n, H, V, sol, pre_mask, shrinkable, areas, gt, tgt_ext,
                  keyx, keyy, ordx, ordy, budget=60):
    """Drive both axes' critical paths toward the target extents:
    reshape path blocks (area exact) first, else flip the path edge whose
    other-axis chain stays shortest (NTUplace/FLOORIST-style rebalancing)."""
    flipped = set()
    reshaped_stuck = {0: False, 1: False}
    for _ in range(budget):
        w_, h_ = sol[:, 2], sol[:, 3]
        x0 = float(sol[:, 0].min())
        y0 = float(sol[:, 1].min())
        wminx, _, pathx = min_extent(n, H, w_, pre_mask, gt[:, 0], ordx, x0)
        wminy, _, pathy = min_extent(n, V, h_, pre_mask, gt[:, 1], ordy, y0)
        rx = (wminx - x0) / max(tgt_ext[0], 1e-9)
        ry = (wminy - y0) / max(tgt_ext[1], 1e-9)
        if rx <= 1.02 and ry <= 1.02:
            break
        if rx - 1.02 >= ry - 1.02:
            axis, edges, oth, path = 0, H, V, pathx
            excess = (wminx - x0) - tgt_ext[0]
            size_o, okey, order_o = h_, keyy, ordy
            tgt_o, wmin_o = tgt_ext[1], wminy - y0
        else:
            axis, edges, oth, path = 1, V, H, pathy
            excess = (wminy - y0) - tgt_ext[1]
            size_o, okey, order_o = w_, keyx, ordx
            tgt_o, wmin_o = tgt_ext[0], wminx - x0
        if not reshaped_stuck[axis] and reshape_chain(
                path, axis, sol, areas, shrinkable, excess) > 1e-9:
            continue
        reshaped_stuck[axis] = True     # aspect caps reached on this axis
        edge_set = {}
        for k, (i, j) in enumerate(edges):
            edge_set[(i, j)] = k
        head_o, tail_o = chain_ends(n, oth, size_o, order_o)
        best, best_len = None, np.inf
        for (u, v) in zip(path[:-1], path[1:]):
            k = edge_set.get((u, v))
            if k is None or (min(u, v), max(u, v)) in flipped:
                continue
            if pre_mask[u] and pre_mask[v]:
                continue
            a, b = (u, v) if (okey[u], u) <= (okey[v], v) else (v, u)
            cand = head_o[a] + size_o[a] + tail_o[b]
            if cand < best_len:
                best, best_len = k, cand
        # flip only if the other axis stays within its own wall-ish range
        if best is None or best_len > max(wmin_o, tgt_o) * 1.05:
            break
        i, j = edges.pop(best)
        flipped.add((min(i, j), max(i, j)))
        if (okey[i], i) > (okey[j], j):
            i, j = j, i
        oth.append((i, j))
    return H, V


def _adj_arrays(n, edges):
    """Per-block predecessor / successor index arrays."""
    preds = [[] for _ in range(n)]
    succs = [[] for _ in range(n)]
    for i, j in edges:
        preds[j].append(i)
        succs[i].append(j)
    return ([np.asarray(p, dtype=np.int64) for p in preds],
            [np.asarray(s, dtype=np.int64) for s in succs])


def _slack(i, preds, succs, pos, size):
    """Feasible [lo, hi] for block i's position with all others fixed."""
    p, s = preds[i], succs[i]
    lo = float((pos[p] + size[p]).max()) if len(p) else -np.inf
    hi = float((pos[s]).min() - size[i]) if len(s) else np.inf
    return lo, hi


# ------------------------------------------------------------------ stage P

def _nbr_lists(case, n):
    """Per-block connectivity: list of (other_block_or_-1, weight, pin_xy)."""
    b2b = case['b2b'].numpy()
    p2b = case['p2b'].numpy()
    pins = case['pins'].numpy()
    nbr_i = [[] for _ in range(n)]
    for r in range(len(b2b)):
        i, j, wgt = int(b2b[r, 0]), int(b2b[r, 1]), float(b2b[r, 2])
        nbr_i[i].append((j, wgt, None))
        nbr_i[j].append((i, wgt, None))
    for r in range(len(p2b)):
        pi, bi, wgt = int(p2b[r, 0]), int(p2b[r, 1]), float(p2b[r, 2])
        nbr_i[bi].append((-1, wgt, (float(pins[pi, 0]), float(pins[pi, 1]))))
    return nbr_i


def _hpwl_delta(i, axis, new_pos, sol, nbrs):
    """Exact weighted-HPWL change if block i's axis position becomes new_pos."""
    half = sol[i, 2 + axis] / 2
    oldc, newc = sol[i, axis] + half, new_pos + half
    d = 0.0
    for (j, wgt, pxy) in nbrs[i]:
        t = (sol[j, axis] + sol[j, 2 + axis] / 2) if j >= 0 else pxy[axis]
        d += wgt * (abs(newc - t) - abs(oldc - t))
    return d


def polish(sol, case, adjH, adjV, pre_mask, sweeps, nbr_i=None):
    """Abacus-flavor L1 coordinate descent: each block moves to the weighted
    median of its connected coordinates, clipped to its FRESH slack interval
    (recomputed per block, so moves can never create overlap). Moves are also
    capped at the entry bbox, so polish can only improve HPWL, never area."""
    n = sol.shape[0]
    if nbr_i is None:
        nbr_i = _nbr_lists(case, n)
    walls = []
    for axis in (0, 1):
        walls.append((float(sol[:, axis].min()),
                      float((sol[:, axis] + sol[:, 2 + axis]).max())))

    for _ in range(sweeps):
        moved = 0.0
        for axis, (preds, succs) in ((0, adjH), (1, adjV)):
            pos = sol[:, axis]
            size = sol[:, 2 + axis]
            w0, w1 = walls[axis]
            for i in range(n):
                if pre_mask[i] or not nbr_i[i]:
                    continue
                lo, hi = _slack(i, preds, succs, pos, size)
                lo = max(lo, w0)
                hi = min(hi, w1 - size[i])
                if lo > hi:
                    continue
                vals, wts = [], []
                for (j, wgt, pxy) in nbr_i[i]:
                    tgt = (pos[j] + size[j] / 2) if j >= 0 else pxy[axis]
                    vals.append(tgt - size[i] / 2)
                    wts.append(wgt)
                order = np.argsort(vals)
                cum = np.cumsum(np.asarray(wts, dtype=np.float64)[order])
                med = vals[int(order[np.searchsorted(cum, cum[-1] / 2)])]
                new = min(max(med, lo), hi)
                moved += abs(new - pos[i])
                pos[i] = new
        if moved < 1e-9:
            break
    return sol


# ------------------------------------------------------------------ stage S

def snap_soft(sol, case, adjH, adjV, pre_mask, cfg, S, alpha, n_soft,
              nbr_i=None):
    """Profit-gated exact snapping with fresh slack guards: a snap happens
    iff feasible AND its exact HPWL-delta cost < the violation saved.
    Cluster abutment first (slides blocks), boundary sides last (so
    attachments are exact at exit)."""
    n = sol.shape[0]
    if nbr_i is None:
        nbr_i = _nbr_lists(case, n)
    cons = case['cons'].numpy()
    bits = cons[:, 4]
    x, y = sol[:, 0], sol[:, 1]
    w, h = sol[:, 2], sol[:, 3]
    delta = cfg['delta_rel'] * S
    benefit = 2.0 / max(n_soft, 1)

    def gain(i, axis, tgt):
        return alpha * _hpwl_delta(i, axis, tgt, sol, nbr_i)

    def comp_of(mem):
        """Touching-components (official union semantics, conservative) of a
        cluster group's members. Returns {block: frozenset(component)}."""
        mem = [int(m) for m in mem]
        parent = {m: m for m in mem}

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for a in range(len(mem)):
            for b in range(a + 1, len(mem)):
                i, j = mem[a], mem[b]
                gx = max(x[i], x[j]) - min(x[i] + w[i], x[j] + w[j])
                gy = max(y[i], y[j]) - min(y[i] + h[i], y[j] + h[j])
                touch = (abs(gx) <= 1e-9 and gy < -1e-9) or \
                        (abs(gy) <= 1e-9 and gx < -1e-9) or \
                        (gx < -1e-9 and gy < -1e-9)
                if touch:
                    parent[find(i)] = find(j)
        comps = {}
        for m in mem:
            comps.setdefault(find(m), []).append(m)
        return {m: frozenset(comps[find(m)]) for m in mem}

    def comp_slide(comp, axis, dmove, bb):
        """Rigidly slide a touching component by dmove along axis if every
        member's slack (vs non-members) and the bbox allow it. Applies and
        returns True on success."""
        pos = x if axis == 0 else y
        size = w if axis == 0 else h
        preds, succs = (adjH if axis == 0 else adjV)
        lo_d, hi_d = -np.inf, np.inf
        cost = 0.0
        for m in comp:
            if pre_mask[m]:
                return False
            lo, hi = -np.inf, np.inf
            for p in preds[m]:
                if p not in comp:
                    lo = max(lo, pos[p] + size[p])
            for s in succs[m]:
                if s not in comp:
                    hi = min(hi, pos[s] - size[m])
            lo_d = max(lo_d, lo - pos[m])
            hi_d = min(hi_d, hi - pos[m])
            lo_d = max(lo_d, bb[0] - pos[m])
            hi_d = min(hi_d, bb[1] - size[m] - pos[m])
            for (jn, wgt, pxy) in nbr_i[m]:
                if jn >= 0 and jn in comp:
                    continue            # internal net: relative move is zero
                t = (pos[jn] + size[jn] / 2) if jn >= 0 else pxy[axis]
                c_old = pos[m] + size[m] / 2
                cost += wgt * (abs(c_old + dmove - t) - abs(c_old - t))
        if not (lo_d - 1e-9 <= dmove <= hi_d + 1e-9):
            return False
        if alpha * cost >= benefit:
            return False
        for m in comp:
            pos[m] += dmove
        return True

    # ---- cluster abutment (two passes: chains abut incrementally)
    for _pass in range(2):
        bx0, bx1 = x.min(), (x + w).max()
        by0, by1 = y.min(), (y + h).max()
        for g in np.unique(cons[:, 3]):
            if g == 0:
                continue
            mem = np.nonzero(cons[:, 3] == g)[0]
            comps = comp_of(mem)
            for (i, j) in _cluster_forest(sol, mem):
                if comps.get(i) == comps.get(j):
                    continue
                gx = max(x[i], x[j]) - min(x[i] + w[i], x[j] + w[j])
                gy = max(y[i], y[j]) - min(y[i] + h[i], y[j] + h[j])
                if max(gx, gy) <= 1e-9:
                    continue                   # already touching
                done = False
                for mv, oth in ((j, i), (i, j)):
                    if pre_mask[mv]:
                        continue
                    if gx >= gy:               # close the x gap
                        tgt = x[oth] + w[oth] if x[mv] >= x[oth] else \
                            x[oth] - w[mv]
                        yov = min(y[mv] + h[mv], y[oth] + h[oth]) - max(
                            y[mv], y[oth])
                        lo, hi = _slack(mv, *adjH, x, w)
                        if yov > delta and lo - 1e-9 <= tgt <= hi + 1e-9 \
                                and bx0 <= tgt and tgt + w[mv] <= bx1 \
                                and gain(mv, 0, tgt) < benefit:
                            x[mv] = tgt
                            done = True
                    else:
                        tgt = y[oth] + h[oth] if y[mv] >= y[oth] else \
                            y[oth] - h[mv]
                        xov = min(x[mv] + w[mv], x[oth] + w[oth]) - max(
                            x[mv], x[oth])
                        lo, hi = _slack(mv, *adjV, y, h)
                        if xov > delta and lo - 1e-9 <= tgt <= hi + 1e-9 \
                                and by0 <= tgt and tgt + h[mv] <= by1 \
                                and gain(mv, 1, tgt) < benefit:
                            y[mv] = tgt
                            done = True
                    if done:
                        break
                if done:
                    comps = comp_of(mem)
                    continue
                # fallback: slide the whole touching component rigidly
                axis = 0 if gx >= gy else 1
                gap = gx if axis == 0 else gy
                perp = (min(y[i] + h[i], y[j] + h[j]) - max(y[i], y[j])) \
                    if axis == 0 else \
                    (min(x[i] + w[i], x[j] + w[j]) - max(x[i], x[j]))
                if perp <= delta:
                    continue
                pos = x if axis == 0 else y
                bb = (bx0, bx1) if axis == 0 else (by0, by1)
                for mv, oth in ((j, i), (i, j)):
                    sgn = 1.0 if pos[mv] < pos[oth] else -1.0
                    if comp_slide(comps[mv], axis, sgn * gap, bb):
                        comps = comp_of(mem)
                        break

    # ---- boundary sides (two passes: snaps can enable each other)
    for _pass in range(2):
        bx0, bx1 = x.min(), (x + w).max()
        by0, by1 = y.min(), (y + h).max()
        for i in np.nonzero(bits)[0]:
            if pre_mask[i]:
                continue
            b = int(bits[i])
            nx, ny = x[i], y[i]
            if b & 1:
                nx = bx0
            if b & 2:
                nx = bx1 - w[i]
            if b & 4:
                ny = by1 - h[i]
            if b & 8:
                ny = by0
            if abs(nx - x[i]) + abs(ny - y[i]) < 1e-12:
                continue
            cost = 0.0
            if nx != x[i]:
                lo, hi = _slack(i, *adjH, x, w)
                if not (lo - 1e-9 <= nx <= hi + 1e-9):
                    continue
                cost += gain(i, 0, nx)
            if ny != y[i]:
                lo, hi = _slack(i, *adjV, y, h)
                if not (lo - 1e-9 <= ny <= hi + 1e-9):
                    continue
                cost += gain(i, 1, ny)
            if cost >= benefit:
                continue
            x[i], y[i] = nx, ny
    return sol


# ------------------------------------------------------------------ stage DP: cluster repair

def _touch_components(sol, mem):
    """Union-find over cluster members by edge-adjacency, matching shapely
    unary_union connectivity (a shared edge of positive length or an area
    overlap connects; a bare corner touch does NOT). Returns component lists."""
    m = len(mem)
    x = sol[mem, 0]; y = sol[mem, 1]
    x1 = x + sol[mem, 2]; y1 = y + sol[mem, 3]
    parent = list(range(m))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    eps = 1e-7
    for a in range(m):
        for b in range(a + 1, m):
            ox = min(x1[a], x1[b]) - max(x[a], x[b])
            oy = min(y1[a], y1[b]) - max(y[a], y[b])
            if (ox > eps and oy > eps) or \
               (abs(ox) <= eps and oy > eps) or \
               (abs(oy) <= eps and ox > eps):
                parent[find(a)] = find(b)
    comps = {}
    for a in range(m):
        comps.setdefault(find(a), []).append(int(mem[a]))
    return list(comps.values())


def _perp_shift(pm, sm, pf, sf, delta):
    """Minimal perpendicular shift of block m so it shares an edge of length
    >= delta with block f (intervals [pm,pm+sm] and [pf,pf+sf])."""
    ov = min(pm + sm, pf + sf) - max(pm, pf)
    if ov >= delta:
        return 0.0
    if pm + sm <= pf:
        return (pf + delta) - (pm + sm)          # m entirely below -> up
    if pm >= pf + sf:
        return (pf + sf - delta) - pm            # m entirely above -> down
    return (delta - ov) if pm < pf else -(delta - ov)


def _abut_delta(sol, m, f, delta):
    """(dx, dy) moving block m to abut block f face-to-face on their more-
    separated axis, with a >= delta shared edge on the other axis."""
    xm, ym, wm, hm = sol[m]
    xf, yf, wf, hf = sol[f]
    sx = max(xm, xf) - min(xm + wm, xf + wf)
    sy = max(ym, yf) - min(ym + hm, yf + hf)
    if sx >= sy:                                  # abut on x
        dx = (xf + wf) - xm if xm >= xf else (xf - wm) - xm
        dy = _perp_shift(ym, hm, yf, hf, delta)
    else:                                         # abut on y
        dy = (yf + hf) - ym if ym >= yf else (yf - hm) - ym
        dx = _perp_shift(xm, wm, xf, wf, delta)
    return dx, dy


def _closest_pair(sol, A, B):
    """(sep, a, b) minimal block separation across two component lists."""
    best = (1e18, A[0], B[0])
    for a in A:
        for b in B:
            sx = max(sol[a, 0], sol[b, 0]) - min(sol[a, 0] + sol[a, 2],
                                                 sol[b, 0] + sol[b, 2])
            sy = max(sol[a, 1], sol[b, 1]) - min(sol[a, 1] + sol[a, 3],
                                                 sol[b, 1] + sol[b, 3])
            sep = max(sx, sy)
            if sep < best[0]:
                best = (sep, a, b)
    return best


def _shift_slack(moving, m, preds, succs, pos, size):
    """[lo, hi] for block m's position on one axis, ignoring graph neighbours in
    `moving` (they translate with m, so they cannot constrain it)."""
    lo, hi = -np.inf, np.inf
    for p in preds[m]:
        if int(p) in moving:
            continue
        lo = max(lo, float(pos[p] + size[p]))
    for s in succs[m]:
        if int(s) in moving:
            continue
        hi = min(hi, float(pos[s]) - float(size[m]))
    return lo, hi


def cluster_perp_align(sol, case, adjH, adjV, pre_mask, cfg, S,
                       hpwl_base, area_base, n_soft):
    """Stage DP-1: give split cluster components a real SHARED EDGE.

    Measured dominant residual (70 of 71 violating groups): the two components
    already touch along one axis -- 68 of them at gap exactly 0 -- but their
    intervals on the *other* axis are disjoint, so they meet at a corner and
    shapely's unary_union still reports two pieces. The needed correction is
    often tiny (a few thousandths of S).

    `cluster_repair` computes the right 2D move but applies it as a rigid
    component translation validated against free space, and at ~97% packing
    utilisation `max_penetration` rejects nearly all of them. This stage instead
    moves within **constraint-graph slack**: shifting a block along axis p can
    only collide with its p-graph neighbours (any other pair is kept apart by the
    other axis' edge), so the slack interval is an exact feasibility bound and
    needs no free space at all -- the same argument `polish` relies on.

    Tries the cheapest repair first: one block, then its whole component. Every
    move is gated on the exact official-cost proxy, so wirelength paid can never
    exceed the violation removed.
    """
    cons = case['cons'].numpy()
    groups = [np.nonzero(cons[:, 3] == g)[0] for g in np.unique(cons[:, 3])
              if g > 0]
    if not groups:
        return sol
    delta = cfg['delta_rel'] * S
    adj = (adjH, adjV)
    base = proxy_cost(sol, case, hpwl_base, area_base, n_soft)
    if base >= 10.0:
        return sol                                # infeasible input: leave it
    moves, cap = 0, int(cfg.get('perp_moves', 40))
    improved = True
    while improved and moves < cap:
        improved = False
        for mem in groups:
            comps = _touch_components(sol, mem)
            if len(comps) <= 1:
                continue
            pairs = sorted(((ia, ib) for ia in range(len(comps))
                            for ib in range(ia + 1, len(comps))),
                           key=lambda p: _closest_pair(sol, comps[p[0]],
                                                       comps[p[1]])[0])
            for ia, ib in pairs:
                A, B = comps[ia], comps[ib]
                _sep, a, b = _closest_pair(sol, A, B)
                gx = (max(sol[a, 0], sol[b, 0])
                      - min(sol[a, 0] + sol[a, 2], sol[b, 0] + sol[b, 2]))
                gy = (max(sol[a, 1], sol[b, 1])
                      - min(sol[a, 1] + sol[a, 3], sol[b, 1] + sol[b, 3]))
                # the axis they are DISJOINT on is the one needing overlap; the
                # other one already abuts and must not be disturbed
                sep = 0 if gx >= gy else 1
                preds, succs = adj[sep]

                # a shift along `sep` can drag a block off a boundary wall it is
                # sitting on, trading a grouping fix for a boundary violation --
                # so try movers that are unconstrained on this axis first
                bits_ax = (1 | 2) if sep == 0 else (4 | 8)
                cons_bits = cons[:, 4]
                trials = []
                for m, f, comp in ((b, a, B), (a, b, A)):
                    if pre_mask[m]:
                        continue
                    pin = 1 if int(cons_bits[m]) & bits_ax else 0
                    trials.append((pin, int(m), int(f), [int(m)]))
                    if len(comp) > 1 and not any(pre_mask[i] for i in comp):
                        cpin = 1 if any(int(cons_bits[i]) & bits_ax for i in comp) \
                            else 0
                        trials.append((cpin, int(m), int(f),
                                       [int(i) for i in comp]))
                trials = [t[1:] for t in sorted(trials, key=lambda t: t[0])]

                for m, f, grp in trials:
                    need = _perp_shift(sol[m, sep], sol[m, 2 + sep],
                                       sol[f, sep], sol[f, 2 + sep], delta)
                    if need == 0.0:
                        continue
                    moving = set(grp)
                    lo_d, hi_d = -np.inf, np.inf
                    for i in grp:
                        lo, hi = _shift_slack(moving, i, preds, succs,
                                              sol[:, sep], sol[:, 2 + sep])
                        lo_d = max(lo_d, lo - sol[i, sep])
                        hi_d = min(hi_d, hi - sol[i, sep])
                    d = min(max(need, lo_d), hi_d)
                    if not np.isfinite(d) or abs(d) < 1e-12:
                        continue
                    idx = np.array(grp)
                    old = sol[idx].copy()
                    sol[idx, sep] += d
                    ok = max_penetration(sol) <= _EPS_OVL
                    c = (proxy_cost(sol, case, hpwl_base, area_base, n_soft)
                         if ok else np.inf)
                    if ok and c <= base - 1e-12:
                        base, improved, moves = c, True, moves + 1
                        break
                    sol[idx] = old
                if improved:
                    break                         # components changed: recompute
    return sol


def cluster_repair(sol, case, pre_mask, cfg, S, hpwl_base, area_base, n_soft):
    """Detailed-placement cluster-grouping repair. For each cluster group that
    is split into >1 touching-component, rigidly translate the smaller movable
    component so its nearest block abuts the other component with a real shared
    edge (2D move: close the gap AND align perpendicular). Each move is applied
    tentatively, rejected if it creates any overlap, and kept only if the exact
    official-cost proxy improves -> it can never regress feasibility or score.

    Fixes the dominant residual (diagonally-offset near-miss fragments) that the
    single-axis snapping in snap_soft cannot connect."""
    cons = case['cons'].numpy()
    groups = [np.nonzero(cons[:, 3] == g)[0] for g in np.unique(cons[:, 3])
              if g > 0]
    if not groups:
        return sol
    delta = cfg['delta_rel'] * S
    drag_cap = cfg['cluster_drag_rel'] * S
    base = proxy_cost(sol, case, hpwl_base, area_base, n_soft)
    if base >= 10.0:
        return sol                                # infeasible input: leave it
    moves = 0
    improved = True
    while improved and moves < cfg['cluster_moves']:
        improved = False
        for mem in groups:
            comps = _touch_components(sol, mem)
            if len(comps) <= 1:
                continue
            pairs = [(ia, ib) for ia in range(len(comps))
                     for ib in range(ia + 1, len(comps))]
            pairs.sort(key=lambda p: _closest_pair(sol, comps[p[0]],
                                                   comps[p[1]])[0])
            for ia, ib in pairs:
                A, B = comps[ia], comps[ib]
                a_fixed = any(pre_mask[i] for i in A)
                b_fixed = any(pre_mask[i] for i in B)
                if a_fixed and b_fixed:
                    continue                      # neither can move
                _, a, b = _closest_pair(sol, A, B)
                # try moving the smaller movable component onto the other; if
                # that is blocked/uneconomical, try moving the larger one
                opts = []
                if not b_fixed:
                    opts.append((B, b, a))
                if not a_fixed:
                    opts.append((A, a, b))
                opts.sort(key=lambda o: len(o[0]))
                done = False
                for mvcomp, mvb, anchor in opts:
                    dx, dy = _abut_delta(sol, mvb, anchor, delta)
                    if abs(dx) + abs(dy) > drag_cap:
                        continue
                    idx = np.array(mvcomp)
                    old = sol[idx].copy()
                    sol[idx, 0] += dx
                    sol[idx, 1] += dy
                    if max_penetration(sol) > _EPS_OVL:
                        sol[idx] = old
                        continue
                    c = proxy_cost(sol, case, hpwl_base, area_base, n_soft)
                    if c <= base - 1e-9:
                        base = c
                        improved = True
                        moves += 1
                        done = True
                        break
                    sol[idx] = old
                if done:
                    break                         # recompute comps for group
    return sol


# ------------------------------------------------------------------ stage DP-2: bounded ripple repair

def _topological_order(n, edges):
    """Deterministic topological order, or None if an edge swap made a cycle."""
    succ = [[] for _ in range(n)]
    indeg = np.zeros(n, dtype=np.int64)
    for i, j in edges:
        succ[i].append(j)
        indeg[j] += 1
    ready = [i for i in range(n) if indeg[i] == 0]
    heapq.heapify(ready)
    order = []
    while ready:
        i = heapq.heappop(ready)
        order.append(i)
        for j in succ[i]:
            indeg[j] -= 1
            if indeg[j] == 0:
                heapq.heappush(ready, j)
    return order if len(order) == n else None


def _swap_contact_edge(H, V, a, b, axis, leader, follower):
    """Move pair (a,b) to the contact axis with the requested orientation."""
    pair = (min(a, b), max(a, b))
    hc = [(i, j) for i, j in H if (min(i, j), max(i, j)) != pair]
    vc = [(i, j) for i, j in V if (min(i, j), max(i, j)) != pair]
    (hc if axis == 0 else vc).append((leader, follower))
    return hc, vc


def _graph_is_legal(sol, edges, axis, tol=1e-7):
    pos, size = sol[:, axis], sol[:, 2 + axis]
    return all(pos[j] + tol >= pos[i] + size[i] for i, j in edges)


def _ripple_contact_plans(sol, anchor_comp, moving_comp, delta, pair_cap):
    """Cheap exact-face plans, nearest block pairs first.

    Returns (motion, contact_axis, leader, follower, anchor_block,
    moving_block).  `motion` translates the complete moving component.  The
    perpendicular part guarantees a positive shared edge rather than a corner.
    """
    pairs = []
    for f in anchor_comp:
        for m in moving_comp:
            gx = max(sol[f, 0], sol[m, 0]) - min(
                sol[f, 0] + sol[f, 2], sol[m, 0] + sol[m, 2])
            gy = max(sol[f, 1], sol[m, 1]) - min(
                sol[f, 1] + sol[f, 3], sol[m, 1] + sol[m, 3])
            pairs.append((max(gx, gy), int(f), int(m)))
    pairs.sort(key=lambda z: (z[0], z[1], z[2]))

    plans = []
    for _sep, f, m in pairs[:pair_cap]:
        xf, yf, wf, hf = sol[f]
        xm, ym, wm, hm = sol[m]
        dy = _perp_shift(ym, hm, yf, hf, delta)
        dx = _perp_shift(xm, wm, xf, wf, delta)
        # moving block immediately right / left of the anchor
        plans.append((np.array([xf + wf - xm, dy]), 0, f, m, f, m))
        plans.append((np.array([xf - wm - xm, dy]), 0, m, f, f, m))
        # moving block immediately above / below the anchor
        plans.append((np.array([dx, yf + hf - ym]), 1, f, m, f, m))
        plans.append((np.array([dx, yf - hm - ym]), 1, m, f, f, m))
    plans.sort(key=lambda z: (abs(z[0][0]) + abs(z[0][1]), z[1],
                              z[2], z[3]))
    return plans


def cluster_ripple_repair(sol, case, H, V, pre_mask, cfg, S,
                          hpwl_base, area_base, n_soft):
    """Reconnect close cluster components by bounded DAG ripple movement.

    Existing single/component slides fail when an unrelated block occupies the
    needed space.  For a candidate face contact this routine swaps the chosen
    pair's separation edge onto the contact axis, pins both touching components
    (one translated, one stationary), and re-runs minimal-movement assignment.
    Blocks on the affected predecessor/successor chains ripple out of the way.

    Every trial stays inside the entry bbox and preserves all graph separation
    inequalities.  It is accepted only when official grouping count and the
    exact proxy both strictly improve.  Trial, displacement, moved-block and
    wall-clock caps bound the worst case.
    """
    t0 = time.perf_counter()
    stats = {'trials': 0, 'moves': 0, 'moved_blocks': 0,
             'overlap_proposals': 0, 'projected_trials': 0,
             'group_before': 0, 'group_after': 0, 'runtime_s': 0.0}
    cons = case['cons'].numpy()
    groups = [np.nonzero(cons[:, 3] == g)[0]
              for g in np.unique(cons[:, 3]) if g > 0]
    if not groups:
        return sol, stats

    base_v = _violations_official(sol, cons)
    base_vg = base_v[1]
    stats['group_before'] = base_vg
    stats['group_after'] = base_vg
    if base_vg == 0:
        return sol, stats

    n = len(sol)
    delta = cfg['delta_rel'] * S
    close_cap = cfg['ripple_close_rel'] * S
    drag_cap = cfg['ripple_drag_rel'] * S
    deadline = t0 + cfg['ripple_budget_s']
    trial_cap = int(cfg['ripple_trials'])
    move_cap = int(cfg['ripple_moves'])
    pair_cap = int(cfg['ripple_pair_choices'])
    block_cap = int(cfg['ripple_max_blocks'])
    disp_cap = float(cfg['ripple_total_disp_rel']) * S
    walls = [(float(sol[:, a].min()),
              float((sol[:, a] + sol[:, 2 + a]).max())) for a in (0, 1)]
    base_cost = proxy_cost(sol, case, hpwl_base, area_base, n_soft)
    H, V = list(H), list(V)

    improved = True
    while improved and stats['moves'] < move_cap \
            and stats['trials'] < trial_cap and time.perf_counter() < deadline:
        improved = False
        # Recompute components after every accepted contact.  Component pairs
        # are ordered by their closest rectangle gap, so cheap near misses win.
        open_pairs = []
        for mem in groups:
            comps = _touch_components(sol, mem)
            for ia in range(len(comps)):
                for ib in range(ia + 1, len(comps)):
                    sep, _a, _b = _closest_pair(sol, comps[ia], comps[ib])
                    if max(0.0, sep) <= close_cap:
                        open_pairs.append((max(0.0, sep), comps[ia], comps[ib]))
        open_pairs.sort(key=lambda z: (z[0], len(z[1]) + len(z[2])))

        for _sep, A, B in open_pairs:
            directions = []
            if not pre_mask[np.asarray(B, dtype=np.int64)].any():
                directions.append((A, B))
            if not pre_mask[np.asarray(A, dtype=np.int64)].any():
                directions.append((B, A))
            directions.sort(key=lambda z: len(z[1]))

            plans = []
            for anchor, moving in directions:
                for p in _ripple_contact_plans(sol, anchor, moving,
                                               delta, pair_cap):
                    plans.append((p, anchor, moving))
            plans.sort(key=lambda z: abs(z[0][0][0]) + abs(z[0][0][1]))

            for plan, anchor, moving in plans:
                if stats['trials'] >= trial_cap or time.perf_counter() >= deadline:
                    break
                motion, contact_axis, leader, follower, _f, _m = plan
                if np.max(np.abs(motion)) > drag_cap:
                    continue
                stats['trials'] += 1

                hc, vc = _swap_contact_edge(H, V, leader, follower,
                                            contact_axis, leader, follower)
                ordx = _topological_order(n, hc)
                ordy = _topological_order(n, vc)
                if ordx is None or ordy is None:
                    continue

                cand = sol.copy()
                midx = np.asarray(moving, dtype=np.int64)
                cand[midx, :2] += motion
                stats['overlap_proposals'] += int(
                    max_penetration(cand) > _EPS_OVL)
                fixed = pre_mask.copy()
                fixed[np.asarray(anchor, dtype=np.int64)] = True
                fixed[midx] = True

                # Pin the two existing touching components and minimally
                # reassign everything else.  This is the ripple propagation.
                stats['projected_trials'] += 1
                for axis, edges, order in ((0, hc, ordx), (1, vc, ordy)):
                    target = sol[:, axis].copy()
                    fixed_pos = cand[:, axis].copy()
                    cand[:, axis] = assign_axis(
                        n, edges, target, cand[:, 2 + axis], fixed, fixed_pos,
                        order, wall_hi=walls[axis][1], wall_lo=walls[axis][0])

                if not _graph_is_legal(cand, hc, 0) \
                        or not _graph_is_legal(cand, vc, 1):
                    continue
                if any(cand[:, a].min() < walls[a][0] - 1e-7 or
                       (cand[:, a] + cand[:, 2 + a]).max()
                       > walls[a][1] + 1e-7 for a in (0, 1)):
                    continue
                intended = sol.copy()
                intended[midx, :2] += motion
                pinned = np.nonzero(fixed)[0]
                if np.max(np.abs(cand[pinned, :2] - intended[pinned, :2])) > 1e-7:
                    continue
                if max_penetration(cand) > _EPS_OVL:
                    continue
                moved = np.nonzero(np.max(np.abs(cand[:, :2] - sol[:, :2]),
                                          axis=1) > 1e-8)[0]
                total_disp = float(np.abs(cand[:, :2] - sol[:, :2]).sum())
                if len(moved) > block_cap or total_disp > disp_cap:
                    continue
                _vb, vg, _vm = _violations_official(cand, cons)
                if vg >= base_vg:
                    continue
                cost = proxy_cost(cand, case, hpwl_base, area_base, n_soft)
                if cost >= base_cost - 1e-12:
                    continue

                sol = cand
                H, V = hc, vc
                base_cost, base_vg = cost, vg
                stats['moves'] += 1
                stats['moved_blocks'] += len(moved)
                stats['group_after'] = vg
                improved = True
                break
            if improved:
                break

    stats['runtime_s'] = time.perf_counter() - t0
    return sol, stats


# ------------------------------------------------------------------ stage DP-3: shape-aware repair

def _entry_walls(sol):
    """Return immutable low/high walls for both axes."""
    return [(float(sol[:, a].min()),
             float((sol[:, a] + sol[:, 2 + a]).max())) for a in (0, 1)]


def _assign_inside_walls(cand, target, H, V, fixed, walls):
    """Ripple movable blocks inside ``walls``; return False on graph failure."""
    n = len(cand)
    ordx, ordy = _topological_order(n, H), _topological_order(n, V)
    if ordx is None or ordy is None:
        return False
    fixed_pos = cand[:, :2].copy()
    for axis, edges, order in ((0, H, ordx), (1, V, ordy)):
        cand[:, axis] = assign_axis(
            n, edges, target[:, axis], cand[:, 2 + axis], fixed,
            fixed_pos[:, axis], order, wall_hi=walls[axis][1],
            wall_lo=walls[axis][0])
    if not _graph_is_legal(cand, H, 0) or not _graph_is_legal(cand, V, 1):
        return False
    if np.max(np.abs(cand[fixed, :2] - fixed_pos[fixed])) > 1e-7:
        return False
    for axis in (0, 1):
        if cand[:, axis].min() < walls[axis][0] - 1e-7:
            return False
        if (cand[:, axis] + cand[:, 2 + axis]).max() \
                > walls[axis][1] + 1e-7:
            return False
    return max_penetration(cand) <= _EPS_OVL


def _group_reshape_candidates(sol, cons, shrinkable, close_cap,
                              max_growth, delta):
    """Rank simple one-axis group gaps that one endpoint can elongate across."""
    out = []
    for g in np.unique(cons[:, 3]):
        if g == 0:
            continue
        mem = np.nonzero(cons[:, 3] == g)[0]
        comps = _touch_components(sol, mem)
        for ia in range(len(comps)):
            for ib in range(ia + 1, len(comps)):
                _sep, a, b = _closest_pair(sol, comps[ia], comps[ib])
                for axis in (0, 1):
                    other = 1 - axis
                    alo, ahi = sol[a, axis], sol[a, axis] + sol[a, 2 + axis]
                    blo, bhi = sol[b, axis], sol[b, axis] + sol[b, 2 + axis]
                    gap = max(alo, blo) - min(ahi, bhi)
                    ov = min(sol[a, other] + sol[a, 2 + other],
                             sol[b, other] + sol[b, 2 + other]) \
                        - max(sol[a, other], sol[b, other])
                    if not (0.0 < gap <= close_cap and ov > delta):
                        continue
                    left, right = (a, b) if alo <= blo else (b, a)
                    for grow, direction in ((left, 1), (right, -1)):
                        if not shrinkable[grow]:
                            continue
                        rel = gap / max(sol[grow, 2 + axis], 1e-12)
                        if rel <= max_growth + 1e-12:
                            # Try center-preserving first, then preserve either
                            # perpendicular face.  The latter variants often
                            # retain an older contact in the same group.
                            for perp_anchor in (0, -1, 1):
                                out.append((gap, int(grow), int(left),
                                            int(right), axis, direction,
                                            perp_anchor))
    out.sort(key=lambda z: (z[0], z[1], z[4]))
    return out


def _try_group_reshape(sol, rec, pre_mask, cfg, walls):
    """Area-preserving elongation plus DAG ripple for one contact candidate."""
    gap, grow, left, right, axis, direction, perp_anchor = rec
    cand = sol.copy()
    old_size = float(cand[grow, 2 + axis])
    other = 1 - axis
    area = float(cand[grow, 2] * cand[grow, 3])
    new_size = old_size + gap
    new_other = area / new_size
    if max(new_size / new_other, new_other / new_size) \
            > cfg['reshape_aspect_cap']:
        return None

    # Keep the face away from the gap fixed, and keep the perpendicular center.
    if direction < 0:
        cand[grow, axis] -= gap
    old_other = cand[grow, 2 + other]
    if perp_anchor > 0:                 # preserve high perpendicular face
        cand[grow, other] += old_other - new_other
    elif perp_anchor == 0:              # preserve perpendicular center
        cand[grow, other] += 0.5 * (old_other - new_other)
    cand[grow, 2 + axis] = new_size
    cand[grow, 2 + other] = new_other

    keyx = sol[:, 0] + sol[:, 2] / 2
    keyy = sol[:, 1] + sol[:, 3] / 2
    H, V = build_graph(sol, keyx, keyy)
    leader, follower = left, right
    H, V = _swap_contact_edge(H, V, left, right, axis, leader, follower)
    fixed = pre_mask.copy()
    fixed[[left, right]] = True
    if not _assign_inside_walls(cand, sol, H, V, fixed, walls):
        return None
    return cand


def _contract_high_wall(sol, axis, target_hi, pre_mask, shrinkable, cfg):
    """Contract one high bbox side, reshaping its critical path if necessary."""
    cand = sol.copy()
    walls = _entry_walls(sol)
    if target_hi <= walls[axis][0] + 1e-9:
        return None
    pre_edge = cand[:, axis] + cand[:, 2 + axis]
    if (pre_edge[pre_mask] > target_hi + 1e-7).any():
        return None
    walls[axis] = (walls[axis][0], float(target_hi))

    keyx = sol[:, 0] + sol[:, 2] / 2
    keyy = sol[:, 1] + sol[:, 3] / 2
    H, V = build_graph(sol, keyx, keyy)
    orders = (_topological_order(len(sol), H),
              _topological_order(len(sol), V))
    if orders[0] is None or orders[1] is None:
        return None
    edges = H if axis == 0 else V
    areas = cand[:, 2] * cand[:, 3]
    for _ in range(3):
        extent, _lb, path = min_extent(
            len(cand), edges, cand[:, 2 + axis], pre_mask,
            cand[:, axis], orders[axis], walls[axis][0])
        excess = extent - target_hi
        if excess <= 1e-8:
            break
        eligible = [v for v in path if shrinkable[v]]
        if not eligible:
            return None
        total = sum(cand[v, 2 + axis] for v in eligible)
        ratio = max(1.0 - excess / max(total, 1e-12),
                    1.0 - float(cfg['reshape_max_shrink']))
        accepted = []
        for v in eligible:
            old_axis = cand[v, 2 + axis]
            old_other = cand[v, 3 - axis]
            new_axis = old_axis * ratio
            new_other = areas[v] / new_axis
            old_ar = max(old_axis / old_other, old_other / old_axis)
            new_ar = max(new_axis / new_other, new_other / new_axis)
            # Permit an already-slender block when contraction improves it.
            if new_ar <= max(float(cfg['reshape_aspect_cap']), old_ar) + 1e-12:
                accepted.append(v)
        if not accepted:
            return None
        total = sum(cand[v, 2 + axis] for v in accepted)
        ratio = max(1.0 - excess / max(total, 1e-12),
                    1.0 - float(cfg['reshape_max_shrink']))
        got = 0.0
        for v in accepted:
            old_axis = cand[v, 2 + axis]
            cand[v, 2 + axis] = old_axis * ratio
            cand[v, 3 - axis] = areas[v] / cand[v, 2 + axis]
            got += old_axis - cand[v, 2 + axis]
        if got <= 1e-9:
            return None
    target = sol.copy()
    if not _assign_inside_walls(cand, target, H, V, pre_mask, walls):
        return None
    return cand


def _try_boundary_reshape(sol, rec, pre_mask, shrinkable, cfg):
    """Contract a high side directly, or mirror a low side into a high side."""
    _gap, _block, axis, high, target = rec
    if high:
        return _contract_high_wall(
            sol, axis, target, pre_mask, shrinkable, cfg)
    mirrored = sol.copy()
    mirrored[:, axis] = -(sol[:, axis] + sol[:, 2 + axis])
    cand = _contract_high_wall(
        mirrored, axis, -target, pre_mask, shrinkable, cfg)
    if cand is None:
        return None
    cand[:, axis] = -(cand[:, axis] + cand[:, 2 + axis])
    return cand


def shape_detail_repair(sol, case, pre_mask, shrinkable, cfg, S,
                        hpwl_base, area_base, n_soft):
    """Bounded reshape repair for residual group and boundary violations.

    Trials preserve each changed block's area, never reshape fixed/preplaced/MIB
    blocks, and reassign all movable blocks through an all-pair separation DAG.
    A trial commits only when the exact official violation total and proxy cost
    both improve.  This makes the heuristic safe to leave enabled globally.
    """
    t0 = time.perf_counter()
    stats = {'trials': 0, 'moves': 0, 'group_moves': 0,
             'boundary_moves': 0, 'reshaped_blocks': 0,
             'viol_before': 0, 'viol_after': 0, 'runtime_s': 0.0}
    cons = case['cons'].numpy()
    before = _violations_official(sol, cons)
    base_v = sum(before)
    stats['viol_before'] = base_v
    stats['viol_after'] = base_v
    if base_v == 0:
        return sol, stats

    deadline = t0 + float(cfg['reshape_budget_s'])
    trial_cap = int(cfg['reshape_trials'])
    move_cap = int(cfg['reshape_moves'])
    close_cap = float(cfg['reshape_close_rel']) * S
    disp_cap = float(cfg['reshape_total_disp_rel']) * S
    block_cap = int(cfg['reshape_max_blocks'])
    delta = max(float(cfg['delta_rel']) * S, 1e-9)
    base_cost = proxy_cost(sol, case, hpwl_base, area_base, n_soft)

    while stats['moves'] < move_cap and stats['trials'] < trial_cap \
            and time.perf_counter() < deadline:
        records = []
        if cfg.get('reshape_group', True):
            for rec in _group_reshape_candidates(
                    sol, cons, shrinkable, close_cap,
                    float(cfg['reshape_max_growth']), delta):
                records.append((rec[0], 0, rec))
        if cfg.get('reshape_boundary', True):
            x0, y0 = sol[:, 0], sol[:, 1]
            x1, y1 = x0 + sol[:, 2], y0 + sol[:, 3]
            lo, hi = (float(x0.min()), float(y0.min())), \
                     (float(x1.max()), float(y1.max()))
            for i in np.nonzero(pre_mask & (cons[:, 4] > 0))[0]:
                bits = int(cons[i, 4])
                for axis, low_bit, high_bit in ((0, 1, 2), (1, 8, 4)):
                    if bits & high_bit:
                        target = float(sol[i, axis] + sol[i, 2 + axis])
                        gap = hi[axis] - target
                        if 1e-8 < gap <= close_cap:
                            records.append((gap, 1,
                                            (gap, int(i), axis, True, target)))
                    if bits & low_bit:
                        target = float(sol[i, axis])
                        gap = target - lo[axis]
                        if 1e-8 < gap <= close_cap:
                            records.append((gap, 1,
                                            (gap, int(i), axis, False, target)))
        records.sort(key=lambda z: (z[0], z[1]))
        accepted = False
        walls = _entry_walls(sol)
        for _gap, kind, rec in records:
            if stats['trials'] >= trial_cap or time.perf_counter() >= deadline:
                break
            stats['trials'] += 1
            cand = _try_group_reshape(sol, rec, pre_mask, cfg, walls) \
                if kind == 0 else _try_boundary_reshape(
                    sol, rec, pre_mask, shrinkable, cfg)
            if cand is None:
                continue
            changed_shape = np.nonzero(np.max(
                np.abs(cand[:, 2:4] - sol[:, 2:4]), axis=1) > 1e-8)[0]
            moved = np.nonzero(np.max(
                np.abs(cand[:, :2] - sol[:, :2]), axis=1) > 1e-8)[0]
            if len(np.union1d(changed_shape, moved)) > block_cap:
                continue
            if float(np.abs(cand[:, :2] - sol[:, :2]).sum()) > disp_cap:
                continue
            after = _violations_official(cand, cons)
            total_v = sum(after)
            if total_v >= base_v:
                continue
            cost = proxy_cost(cand, case, hpwl_base, area_base, n_soft)
            if cost >= base_cost - 1e-12:
                continue
            sol = cand
            base_v, base_cost = total_v, cost
            stats['moves'] += 1
            stats['group_moves' if kind == 0 else 'boundary_moves'] += 1
            stats['reshaped_blocks'] += len(changed_shape)
            stats['viol_after'] = total_v
            accepted = True
            break
        if not accepted:
            break

    stats['runtime_s'] = time.perf_counter() - t0
    return sol, stats


# ------------------------------------------------------- stage E: eviction

def _eviction_score(case, weights=None):
    """Per-block cost of evicting it out of the packed core (lower = cheaper).

    Weights are (boundary, cluster, mib, weighted-degree, area). Note the
    boundary weight is deliberately LOW: the shelf sits on the bbox's top edge,
    so an evicted block usually keeps a TOP bit for free -- measured, only 1% of
    boundary violations land on evicted blocks -- while 75% of split cluster
    groups contain one. Protecting group membership matters; protecting boundary
    membership mostly wastes the budget."""
    cons = case['cons'].numpy()
    n = cons.shape[0]
    area = case['area'].numpy().astype(np.float64)
    b2b = case['b2b'].numpy()
    p2b = case['p2b'].numpy()
    wdeg = np.zeros(n)
    if len(b2b):
        np.add.at(wdeg, b2b[:, 0].astype(np.int64), b2b[:, 2])
        np.add.at(wdeg, b2b[:, 1].astype(np.int64), b2b[:, 2])
    if len(p2b):
        np.add.at(wdeg, p2b[:, 1].astype(np.int64), p2b[:, 2])
    wn = wdeg / max(float(wdeg.max()), 1e-12)
    an = area / max(float(area.max()), 1e-12)
    wb, wc, wm, wd, wa = weights if weights else (4.0, 2.0, 1.0, 2.0, 1.0)
    return (wb * (cons[:, 4] > 0) + wc * (cons[:, 3] > 0)
            + wm * (cons[:, 2] > 0) + wd * wn + wa * an)


def _rebuild_evicted(n, H_core, V_core, evicted, keyx, keyy, shelf='above'):
    """Relation set for the evicted set.

    With `shelf='above'`: every evicted block is a V-successor of every core
    block (so it sits above the whole core, and by Lemma 4 can lie on no
    anchor-to-anchor path in either graph), and evicted blocks relate to each
    other in H, ordered by keyx -- they share one shelf ROW instead of stacking.
    `shelf='right'` is the mirror image: H-successors of the core, V-ordered
    among themselves, i.e. one shelf COLUMN beside the core.  Both keep each
    graph acyclic (the added arcs run core -> evicted on one axis and along a
    fixed total order on the other), so the guarantee is identical; the choice
    only decides which bbox side pays for the shelf.
    """
    E = list(evicted)
    Eset = set(E)
    H = [e for e in H_core if e[0] not in Eset and e[1] not in Eset]
    V = [e for e in V_core if e[0] not in Eset and e[1] not in Eset]
    core = [k for k in range(n) if k not in Eset]
    follow, along, key = (V, H, keyx) if shelf == 'above' else (H, V, keyy)
    for m in E:
        for k in core:
            follow.append((k, int(m)))
    ranked = sorted(E, key=lambda m: (key[m], m))
    for a in range(len(ranked)):
        for b in range(a + 1, len(ranked)):
            along.append((int(ranked[a]), int(ranked[b])))
    return H, V


def evict_for_consistency(n, H, V, sol, pre_mask, keyx, keyy, score,
                          max_evict=None, shelf='above'):
    """Make BOTH axis relation sets consistent with the preplaced anchors by
    evicting non-anchor blocks (see FEASIBILITY_ANALYSIS.md 4.1-4.2).

    The x-system {x_j >= x_i + w_i on H, x_i = x_i* on P} is feasible iff no
    anchor pair (a,b) has longest-path d_H(a,b) > x_b* - x_a*; with no fixed
    outline nothing else can bind.  `_find_conflict` (origin=-inf) decides this
    exactly.  Evicting a non-anchor block off a critical path strictly shrinks
    the set of non-anchor blocks lying on any anchor-to-anchor path, so the loop
    terminates after at most n - |P| evictions; the degenerate end state is
    "every movable block in the shelf above", which is trivially consistent.

    Returns (H, V, ordx, ordy, evicted) with both systems conflict-free, or
    None if an anchor pair conflicts directly (the instance itself is broken).
    """
    H_core, V_core = list(H), list(V)
    evicted = []
    cap = max_evict if max_evict is not None else n
    keyx_e = np.asarray(keyx, dtype=np.float64).copy()
    keyy_e = np.asarray(keyy, dtype=np.float64).copy()
    # the shelf axis' key must put the evicted blocks last, so that axis'
    # argsort stays a topological order for the core -> evicted arcs
    shelf_key = keyy_e if shelf == 'above' else keyx_e
    big = float(shelf_key.max()) + 1.0 if n else 1.0
    x, y = sol[:, 0], sol[:, 1]
    w, h = sol[:, 2], sol[:, 3]

    for _ in range(cap + 1):
        ordx = np.lexsort((np.arange(n), keyx_e)).tolist()
        ordy = np.lexsort((np.arange(n), keyy_e)).tolist()
        res = _find_conflict(n, H, x, w, pre_mask, ordx)
        axis_nodes = res[1] if res is not None else None
        if axis_nodes is None:
            res = _find_conflict(n, V, y, h, pre_mask, ordy)
            axis_nodes = res[1] if res is not None else None
        if axis_nodes is None:
            return H, V, ordx, ordy, evicted
        cand = [int(v) for v in axis_nodes
                if not pre_mask[v] and v not in evicted]
        if not cand or len(evicted) >= cap:
            return None
        m = min(cand, key=lambda v: (score[v], v))
        evicted.append(m)
        shelf_key[m] = big + len(evicted)
        H, V = _rebuild_evicted(n, H_core, V_core, evicted, keyx_e, keyy_e,
                                shelf)
    return None


def guaranteed_construction(pred, case, cfg=None):
    """Absolute feasibility floor: overlap-free BY CONSTRUCTION, no search.

    Anchors keep their exact geometry; every movable block is shelf-packed in
    the empty half-plane above the highest anchor, in predicted reading order so
    HPWL locality survives as far as a shelf allows.  Soft dims keep the
    predicted aspect with area = area_scale * target; fixed dims are exact.
    Feasibility does not depend on the input in any way, which is the point:
    this is the candidate that bounds the damage from failure modes we have not
    imagined.
    """
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    sol = np.asarray(pred, dtype=np.float64).copy()
    cons = case['cons'].numpy()
    gt = target_xywh(case).numpy().astype(np.float64)
    pre = cons[:, 1] > 0
    fix = cons[:, 0] > 0
    soft = (~pre) & (~fix)
    area = case['area'].numpy().astype(np.float64)

    asc = float(cfg.get('area_scale', 1.0))
    cap = float(cfg.get('reshape_aspect_cap', 3.6))
    ar = np.clip(sol[soft, 2] / np.maximum(sol[soft, 3], 1e-12), 1.0 / cap, cap)
    tgt = area[soft] * asc
    sol[soft, 2] = np.sqrt(tgt * ar)
    sol[soft, 3] = tgt / sol[soft, 2]
    sol[fix, 2:4] = gt[fix, 2:4]
    sol[pre] = gt[pre]

    mov = np.nonzero(~pre)[0]
    if not len(mov):
        return sol
    if pre.any():
        x_org = float(gt[pre, 0].min())
        y_base = float((gt[pre, 1] + gt[pre, 3]).max())
        shelf_w = float((gt[pre, 0] + gt[pre, 2]).max()) - x_org
    else:
        x_org = float(sol[mov, 0].min())
        y_base = float(sol[mov, 1].min())
        shelf_w = 0.0
    span = float(np.sqrt((sol[mov, 2] * sol[mov, 3]).sum()))
    width = max(shelf_w, span, float(sol[mov, 2].max()))
    order = sorted(mov.tolist(), key=lambda i: (pred[i, 1], pred[i, 0], i))
    cx, cy, row_h = x_org, y_base, 0.0
    for i in order:
        if cx > x_org and cx + sol[i, 2] > x_org + width:
            cx, cy, row_h = x_org, cy + row_h, 0.0
        sol[i, 0], sol[i, 1] = cx, cy
        cx += sol[i, 2]
        row_h = max(row_h, float(sol[i, 3]))
    return sol


def _free_positions(sol, i, xs, ys, others, eps=_EPS_OVL):
    """Boolean grid over (xs, ys) of positions where block i fits with no
    overlap against `others`.  Exact -- the official two-axis test.

    A candidate (x, y) collides with block k iff it penetrates on BOTH axes,
    and each axis' penetration depends on only one of x, y.  So the (A, B) grid
    is a boolean matrix product of the per-axis (A, K) and (B, K) masks, which
    keeps this O(A*K + B*K + A*B) instead of materialising an (A, B, K) cube
    (~110 MB for n=120, times every worker).
    """
    w, h = sol[i, 2], sol[i, 3]
    X = np.asarray(xs)[:, None]
    Y = np.asarray(ys)[:, None]
    ox0 = sol[others, 0][None, :]
    oy0 = sol[others, 1][None, :]
    ox = np.minimum(X + w, ox0 + sol[others, 2][None, :]) - np.maximum(X, ox0)
    oy = np.minimum(Y + h, oy0 + sol[others, 3][None, :]) - np.maximum(Y, oy0)
    OX = (ox > eps).astype(np.float32)          # (A, K)
    OY = (oy > eps).astype(np.float32)          # (B, K)
    return (OX @ OY.T) < 0.5                    # free iff no k penetrates both


def hole_relocate(sol, case, targets, pre_mask, cfg, S, hpwl_base, area_base,
                  n_soft, nbr_i=None, alpha=None):
    """Pull `targets` (typically the blocks stage E evicted into the shelf) back
    into free space.

    Candidate positions are the corner grid of the other blocks -- the classic
    result that an optimal placement of one rectangle among fixed ones has a
    corner-touching optimum, so the grid loses nothing.  Ranking is by a cheap
    surrogate (exact weighted-HPWL delta + bbox growth); only the few best are
    scored with the real proxy.  Every accepted move is verified overlap-free by
    the exact official test AND must strictly improve the proxy, so this stage
    can never turn a feasible layout infeasible.
    """
    t0 = time.perf_counter()
    sol = sol.copy()
    n = sol.shape[0]
    stats = {'tried': 0, 'moved': 0, 'no_free': 0, 'runtime_s': 0.0}
    tg = [int(i) for i in targets if not pre_mask[i]]
    if not tg:
        return sol, stats
    if nbr_i is None:
        nbr_i = _nbr_lists(case, n)
    if alpha is None:
        alpha = 0.5 / max(hpwl_base, 1e-6)
    garea = 0.5 / max(area_base, 1e-6)
    budget = float(cfg.get('reclaim_budget_s', 0.05))
    cap = int(cfg.get('reclaim_trials', 24))
    top = int(cfg.get('reclaim_probe', 8))
    best_cost = proxy_cost(sol, case, hpwl_base, area_base, n_soft)
    if best_cost >= 10.0:
        return sol, stats                     # never gamble on a broken layout

    # biggest first: the shelf blocks that cost the most bbox come back first
    tg.sort(key=lambda i: -(sol[i, 2] * sol[i, 3]))
    for i in tg[:cap]:
        if time.perf_counter() - t0 > budget:
            break
        stats['tried'] += 1
        others = np.array([k for k in range(n) if k != i], dtype=np.int64)
        xs = np.unique(np.concatenate([sol[others, 0],
                                       sol[others, 0] + sol[others, 2]]))
        ys = np.unique(np.concatenate([sol[others, 1],
                                       sol[others, 1] + sol[others, 3]]))
        ok = _free_positions(sol, i, xs, ys, others)
        if not ok.any():
            stats['no_free'] += 1
            continue
        # surrogate: exact HPWL delta on both axes + bbox growth of the union
        bx0 = float(sol[others, 0].min())
        bx1 = float((sol[others, 0] + sol[others, 2]).max())
        by0 = float(sol[others, 1].min())
        by1 = float((sol[others, 1] + sol[others, 3]).max())
        cand = np.argwhere(ok)
        dh = np.empty(len(cand))
        for r, (a, b) in enumerate(cand):
            x, y = float(xs[a]), float(ys[b])
            dh[r] = (_hpwl_delta(i, 0, x, sol, nbr_i)
                     + _hpwl_delta(i, 1, y, sol, nbr_i))
        nx0 = np.minimum(bx0, xs[cand[:, 0]])
        nx1 = np.maximum(bx1, xs[cand[:, 0]] + sol[i, 2])
        ny0 = np.minimum(by0, ys[cand[:, 1]])
        ny1 = np.maximum(by1, ys[cand[:, 1]] + sol[i, 3])
        area_now = (max(bx1, sol[i, 0] + sol[i, 2]) - min(bx0, sol[i, 0])) * \
                   (max(by1, sol[i, 1] + sol[i, 3]) - min(by0, sol[i, 1]))
        da = (nx1 - nx0) * (ny1 - ny0) - area_now
        sur = alpha * dh + garea * da
        probe = np.argsort(sur)[:top]
        keep = (float(sol[i, 0]), float(sol[i, 1]))
        for r in probe:
            a, b = cand[r]
            sol[i, 0], sol[i, 1] = float(xs[a]), float(ys[b])
            c = proxy_cost(sol, case, hpwl_base, area_base, n_soft)
            if c < best_cost - 1e-12:
                best_cost = c
                keep = (float(sol[i, 0]), float(sol[i, 1]))
                stats['moved'] += 1
                break
        sol[i, 0], sol[i, 1] = keep
    stats['runtime_s'] = time.perf_counter() - t0
    return sol, stats


# ------------------------------------------------------------------ driver

def legalize_case(pred_xywh, case, cfg=None, verbose=False, g_iters=None):
    """(n,4) predicted xywh -> feasible xywh (float64). Returns (sol, info)."""
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    t0 = time.time()
    pred = pred_xywh.numpy().astype(np.float64).copy()
    cons = case['cons'].numpy()
    n = pred.shape[0]
    S = float(np.sqrt(case['area'].numpy().astype(np.float64).sum()))
    gt = target_xywh(case).numpy().astype(np.float64)

    pre_mask = cons[:, 1] > 0
    fixed_mask = cons[:, 0] > 0
    pred[pre_mask] = gt[pre_mask]
    pred[fixed_mask, 2:4] = gt[fixed_mask, 2:4]
    # area-shrink: the contest allows soft-block area up to 1% under target, so
    # shrink each soft block's dims by sqrt(area_scale) (area *= area_scale).
    # This frees ~1% packing slack -> a tighter bbox -> smaller area_gap, at zero
    # feasibility cost (strict diff>0.01 check). Applied before MIB tying so an
    # all-soft MIB group scales uniformly (stays identical); a group with a
    # frozen member inherits the frozen (exact) dims and is left unscaled.
    asc = float(cfg.get('area_scale', 1.0))
    if asc < 1.0:
        soft = (~pre_mask) & (~fixed_mask)
        pred[soft, 2:4] *= math.sqrt(asc)
    mib_info = _tie_compatible_mib_dims(pred, case, fixed_mask, pre_mask)
    shrinkable = (~pre_mask) & (~fixed_mask) & (cons[:, 2] == 0)
    areas = pred[:, 2] * pred[:, 3]
    clu_groups = [np.nonzero(cons[:, 3] == g)[0]
                  for g in np.unique(cons[:, 3]) if g > 0]

    hpwl_base, area_base = _baselines(case, pred)
    n_soft = _n_soft_norm(cons)
    alpha = 0.5 / max(hpwl_base, 1e-6)
    nbr_i = _nbr_lists(case, n)
    evict_score = _eviction_score(case, cfg.get('evict_weights'))
    case_stats = {'evict_rounds': 0, 'evicted_total': 0, 'evict_ok': 0,
                  'evicted_set': set()}

    bits = cons[:, 4]
    tgt_ext = (float((pred[:, 0] + pred[:, 2]).max() - pred[:, 0].min()),
               float((pred[:, 1] + pred[:, 3]).max() - pred[:, 1].min()))

    def legal_round(geo, sweeps, span=False):
        """Graph from `geo` -> repair -> critical-chain reshape (dims only)
        -> re-repair -> wall-bounded assignment -> polish.
        span=True drives extents into the preplaced boundary anchors' span
        (aggressive; used as a retry when anchors stay violated).
        Returns (sol, H, V, adjH, adjV)."""
        if cfg.get('graph_key', 'center') == 'edge':
            keyx, keyy = geo[:, 0].copy(), geo[:, 1].copy()
        else:
            keyx = geo[:, 0] + geo[:, 2] / 2
            keyy = geo[:, 1] + geo[:, 3] / 2

        # preplaced boundary blocks pin the walls: the bbox edge must come to
        # them (they cannot move). Repairs treat the low wall as a source.
        wall_lo, pre_hi_req = [None, None], [None, None]
        for axis in (0, 1):
            lo_bit, hi_bit = (1, 8)[axis], (2, 4)[axis]
            lo = [gt[i, axis] for i in np.nonzero(bits)[0]
                  if pre_mask[i] and int(bits[i]) & lo_bit]
            hi = [gt[i, axis] + gt[i, 2 + axis] for i in np.nonzero(bits)[0]
                  if pre_mask[i] and int(bits[i]) & hi_bit]
            # a wall can satisfy the anchors sitting exactly on it; anchors
            # short of it are unsatisfiable no matter what (GT has a few)
            if lo and not (pre_mask & (gt[:, axis] < min(lo) - 1e-9)).any():
                wall_lo[axis] = min(lo)
            if hi and not (pre_mask & (gt[:, axis] + gt[:, 2 + axis]
                                       > max(hi) + 1e-9)).any():
                pre_hi_req[axis] = max(hi)
        org = [w if w is not None else -np.inf for w in wall_lo]

        graph_info = {'repair_failures': 0, 'assign_failures': 0,
                      'final_assignment_ok': False, 'evicted': 0,
                      'evicted_blocks': []}
        H, V = build_graph(geo, keyx, keyy, cfg.get('graph_rule', 'gap'),
                           pre_mask, float(cfg.get('graph_square_k', 0.25)))
        H, V, wd, hd, ok = repair_graph(H, V, geo, keyx, keyy, pre_mask,
                                       shrinkable, areas,
                                       origin_x=org[0], origin_y=org[1])
        graph_info['repair_failures'] += int(not ok)
        ordx = np.lexsort((np.arange(n), keyx)).tolist()
        ordy = np.lexsort((np.arange(n), keyy)).tolist()
        sol = np.empty((n, 4))
        sol[:, 0], sol[:, 1] = geo[:, 0], geo[:, 1]
        sol[:, 2], sol[:, 3] = wd, hd

        # drive critical paths toward the target extents (reshape + flips),
        # then re-check preplaced anchors with the changed dims/edges
        tgt_rep = list(tgt_ext)
        if span:
            for axis in (0, 1):
                if pre_hi_req[axis] is not None:
                    lo_est = wall_lo[axis] if wall_lo[axis] is not None \
                        else float(geo[:, axis].min())
                    tgt_rep[axis] = min(tgt_rep[axis],
                                        pre_hi_req[axis] - lo_est)
        H, V = extent_repair(n, H, V, sol, pre_mask, shrinkable, areas, gt,
                             tgt_rep, keyx, keyy, ordx, ordy)
        geo2 = np.stack([geo[:, 0], geo[:, 1], sol[:, 2], sol[:, 3]], 1)
        H, V, wd, hd, ok = repair_graph(H, V, geo2, keyx, keyy, pre_mask,
                                       shrinkable, areas,
                                       origin_x=org[0], origin_y=org[1])
        graph_info['repair_failures'] += int(not ok)
        sol[:, 2], sol[:, 3] = wd, hd

        def contact_intents():
            """Cluster forest pairs (close in geo) as per-axis abutment
            intents {follower: leader}, keyed by which graph holds the pair."""
            ints = ({}, {})
            edge_ax = {}
            for (i, j) in H:
                edge_ax[(i, j)] = 0
            for (i, j) in V:
                edge_ax[(i, j)] = 1
            geo_now = np.stack([sol[:, 0], sol[:, 1],
                                sol[:, 2], sol[:, 3]], 1)
            tol = 0.05 * math.sqrt(float(areas.sum()))
            for g in np.unique(cons[:, 3]):
                if g == 0:
                    continue
                mem = np.nonzero(cons[:, 3] == g)[0]
                for (i, j) in _cluster_forest(geo_now, mem):
                    gx = max(geo_now[i, 0], geo_now[j, 0]) - min(
                        geo_now[i, 0] + geo_now[i, 2],
                        geo_now[j, 0] + geo_now[j, 2])
                    gy = max(geo_now[i, 1], geo_now[j, 1]) - min(
                        geo_now[i, 1] + geo_now[i, 3],
                        geo_now[j, 1] + geo_now[j, 3])
                    if max(gx, gy) > tol:
                        continue
                    if (i, j) in edge_ax:
                        ld, fol, ax = i, j, edge_ax[(i, j)]
                    elif (j, i) in edge_ax:
                        ld, fol, ax = j, i, edge_ax[(j, i)]
                    else:
                        continue
                    if not pre_mask[fol]:
                        ints[ax].setdefault(int(fol), int(ld))
            return ints

        def cluster_align(axis):
            """Targets pulling a cluster group's members onto a common band on
            `axis`. Members that abut along the *other* axis then land with
            overlapping intervals, i.e. a real shared edge, instead of meeting
            at a corner -- which the measurements show is the dominant grouping
            failure, and which is 98% unrepairable after packing because the
            slack is gone. Applied only perpendicular to the group's spread
            direction, so the group stays free to stretch along its strip."""
            out = {}
            for mem in clu_groups:
                if len(mem) < 2:
                    continue
                cx = sol[mem, 0] + sol[mem, 2] / 2
                cy = sol[mem, 1] + sol[mem, 3] / 2
                spread = 0 if (cx.max() - cx.min()) >= (cy.max() - cy.min()) else 1
                if axis == spread:
                    continue
                band = float(np.median(sol[mem, axis] + sol[mem, 2 + axis] / 2))
                for i in mem:
                    if not pre_mask[i]:
                        out[int(i)] = band - float(sol[i, 2 + axis]) / 2
            return out

        def assign(axis, edges, order, use_wall=True, pin=True,
                   contact=None, align=None):
            pos, size = sol[:, axis], sol[:, 2 + axis]
            lo_bit, hi_bit = (1, 8)[axis], (2, 4)[axis]
            wl = wall_lo[axis] if (use_wall and pin) else None
            x0 = wl if wl is not None else float(pos.min())
            wall = None
            if use_wall:
                wmin, _, _ = min_extent(n, edges, size, pre_mask,
                                        gt[:, axis], order, x0)
                wall = x0 + max(wmin - x0, tgt_ext[axis])
                if pre_mask.any():
                    wall = max(wall,
                               float((gt[:, axis] + size)[pre_mask].max()))
                if pin and pre_hi_req[axis] is not None \
                        and pre_hi_req[axis] >= wmin:
                    wall = pre_hi_req[axis]  # pin bbox max to the R/T anchor
            target = pos.copy()
            bnd_set = set()
            if wall is not None:
                for i in np.nonzero(bits)[0]:   # boundary blocks aim at walls
                    if not pre_mask[i] and int(bits[i]) & lo_bit:
                        target[i] = x0
                        bnd_set.add(int(i))
                    if not pre_mask[i] and int(bits[i]) & hi_bit:
                        target[i] = wall - size[i]
                        bnd_set.add(int(i))
            # cluster band < boundary wall < exact-abutment contact
            for i, t in (align or {}).items():
                if i not in bnd_set:
                    target[i] = t
            con = {f: l for f, l in (contact or {}).items()
                   if f not in bnd_set}         # boundary intent wins
            sol[:, axis] = assign_axis(n, edges, target, size, pre_mask,
                                       gt[:, axis], order,
                                       wall_hi=wall, wall_lo=wl, contact=con)

        def do_assign(use_wall, pin):
            ints = contact_intents()
            al = (cluster_align(0), cluster_align(1)) \
                if cfg.get('cluster_align', True) else (None, None)
            assign(0, H, ordx, use_wall, pin, ints[0], al[0])
            assign(1, V, ordy, use_wall, pin, ints[1], al[1])
            ok = max_penetration(sol) <= _EPS_OVL
            graph_info['assign_failures'] += int(not ok)
            return ok

        # ladder: pinned walls -> unpinned walls -> wall-free repair +
        # unpinned walls -> no walls at all
        assigned = do_assign(True, True)
        if not assigned:
            assigned = do_assign(True, False)
        if not assigned:
            geo3 = np.stack([geo[:, 0], geo[:, 1], sol[:, 2], sol[:, 3]], 1)
            H2, V2, wd2, hd2, ok = repair_graph(
                H, V, geo3, keyx, keyy, pre_mask, shrinkable, areas)
            graph_info['repair_failures'] += int(not ok)
            H[:], V[:] = H2, V2
            sol[:, 2], sol[:, 3] = wd2, hd2
            assigned = do_assign(True, False)
            if not assigned:
                assigned = do_assign(False, False)
        # last rung: the relation set itself is inconsistent with the anchors,
        # so no wall relaxation can help.  Evict non-anchor blocks off the
        # conflicting critical paths until it is consistent -- then lb_i <= U_i
        # holds for every block and the assignment is overlap-free BY
        # CONSTRUCTION (see FEASIBILITY_ANALYSIS.md, Lemma 3 + Theorem 2).
        if not assigned and cfg.get('evict_repair', True):
            mode = cfg.get('evict_shelf', 'auto')
            if mode == 'auto':
                # a shelf laid along the longer bbox side is thinner, so it adds
                # less area for the same set of blocks
                bw = float((sol[:, 0] + sol[:, 2]).max() - sol[:, 0].min())
                bh = float((sol[:, 1] + sol[:, 3]).max() - sol[:, 1].min())
                mode = 'above' if bw >= bh else 'right'
            ev = evict_for_consistency(
                n, H, V, sol, pre_mask, keyx, keyy, evict_score,
                max_evict=int(cfg.get('evict_max', 64)), shelf=mode)
            if ev is not None:
                H2, V2, ordx, ordy, evicted = ev
                H[:], V[:] = H2, V2
                graph_info['evicted'] = len(evicted)
                graph_info['evicted_blocks'] = list(evicted)
                assigned = do_assign(False, False)
                case_stats['evict_rounds'] += 1
                case_stats['evicted_total'] += len(evicted)
                case_stats['evict_ok'] += int(bool(assigned))
                case_stats['evicted_set'].update(int(v) for v in evicted)
        graph_info['final_assignment_ok'] = bool(assigned)
        adjH = _adj_arrays(n, H)
        adjV = _adj_arrays(n, V)
        sol[:] = polish(sol, case, adjH, adjV, pre_mask, sweeps, nbr_i)
        return sol, H, V, adjH, adjV, graph_info

    def finish(rr):
        """Snap a legal_round result; return cost, solution, and its graphs."""
        s, _H, _V, aH, aV, graph_info = rr
        s = snap_soft(s.copy(), case, aH, aV, pre_mask, cfg, S, alpha,
                      n_soft, nbr_i)
        # give still-split cluster groups a real shared edge, using graph slack
        # (free space is already gone by this point -- see cluster_perp_align)
        if cfg.get('perp_align', True):
            s = cluster_perp_align(s, case, aH, aV, pre_mask, cfg, S,
                                   hpwl_base, area_base, n_soft)
        return (proxy_cost(s, case, hpwl_base, area_base, n_soft), s,
                _H, _V, graph_info)

    # G: short local overlap cleanup, then two graph rounds (the second
    # rebuilds the graph from legal geometry -> cleaner axis choices)
    geo = gradient_phase(pred, case, pre_mask, cfg, S, hpwl_base, area_base,
                         iters=g_iters)
    r1 = legal_round(geo, cfg['polish_sweeps'])
    cost, sol, H, V, graph_info = finish(r1)
    # each further round rebuilds the relation set from legal geometry, which
    # gives cleaner axis choices; keep the best result, never a worse one
    prev = r1
    stop_when_flat = bool(cfg.get('legal_round_stop', True))
    for _ in range(max(0, int(cfg.get('legal_rounds', 2)) - 1)):
        if max_penetration(prev[0]) > _EPS_OVL:
            break
        rr = legal_round(prev[0].copy(), cfg['polish_sweeps'])
        c2, s2, h2, v2, g2 = finish(rr)
        gained = c2 < cost - 1e-12
        if gained:
            cost, sol, H, V, graph_info = c2, s2, h2, v2, g2
        prev = rr
        # a rebuild that did not pay predicts the next one will not either, so
        # further rounds are spent only on the cases still improving. Keeps the
        # quality of a fixed 3-round schedule at close to 2-round wall-clock.
        if stop_when_flat and not gained:
            break

    # anchored boundary bits still violated -> retry with the extents driven
    # into the anchors' span (aggressive rung, pays only where needed)
    if cost < 9:
        vb, _vg, _vm = _violations_official(sol, cons)
        anchored = any(pre_mask[i] and bits[i]
                       for i in np.nonzero(bits)[0])
        if vb >= 2 and anchored:
            c3, s3, h3, v3, g3 = finish(
                legal_round(geo, cfg['polish_sweeps'], span=True))
            if c3 < cost:
                cost, sol, H, V, graph_info = c3, s3, h3, v3, g3

    # Stage DP-1: take cheap rigid component contacts first. These moves stay
    # legal, so they remove easy targets without spending projection trials.
    ripple_info = {'trials': 0, 'moves': 0, 'moved_blocks': 0,
                   'overlap_proposals': 0, 'projected_trials': 0,
                   'group_before': 0, 'group_after': 0, 'runtime_s': 0.0,
                   'fresh_graph': False}
    if cost < 9:
        sol = cluster_repair(sol, case, pre_mask, cfg, S, hpwl_base,
                             area_base, n_soft)
        cost = proxy_cost(sol, case, hpwl_base, area_base, n_soft)

    # Stage DP-2: residual rigid contacts are usually blocked. Allow the desired
    # component translation to overlap transiently, then project it through a
    # freshly rebuilt all-pair separation graph. Only the legal, proxy-improving
    # projected result can commit; the entry solution remains the incumbent.
    if cost < 9 and cfg.get('ripple_repair', True):
        keyx = sol[:, 0] + 0.5 * sol[:, 2]
        keyy = sol[:, 1] + 0.5 * sol[:, 3]
        proj_H, proj_V = build_graph(sol, keyx, keyy,
                                     cfg.get('graph_rule', 'gap'), pre_mask,
                                     float(cfg.get('graph_square_k', 0.25)))
        sol, ripple_info = cluster_ripple_repair(
            sol, case, proj_H, proj_V, pre_mask, cfg, S, hpwl_base,
            area_base, n_soft)
        ripple_info['fresh_graph'] = True
        cost = proxy_cost(sol, case, hpwl_base, area_base, n_soft)

    # stage DP-3: target only violations left by every position-only repair.
    # The graph is rebuilt per accepted move, so it reflects ripple and rigid
    # component translations rather than relying on a stale legal-round DAG.
    reshape_info = {'trials': 0, 'moves': 0, 'group_moves': 0,
                    'boundary_moves': 0, 'reshaped_blocks': 0,
                    'viol_before': 0, 'viol_after': 0, 'runtime_s': 0.0}
    if cost < 9 and cfg.get('reshape_repair', True):
        sol, reshape_info = shape_detail_repair(
            sol, case, pre_mask, shrinkable, cfg, S, hpwl_base, area_base,
            n_soft)
        cost = proxy_cost(sol, case, hpwl_base, area_base, n_soft)
    ripple_info['group_final'] = _violations_official(sol, cons)[1]

    # Stage E-2: stage E parks evicted blocks in a shelf above the core, which
    # is feasible but expensive in bbox and HPWL. Pull them back into free
    # holes; exact-overlap + proxy gated, so it only ever improves.
    reclaim_info = {'tried': 0, 'moved': 0, 'no_free': 0, 'runtime_s': 0.0,
                    'round_gain': 0.0}
    if (cost < 9 and cfg.get('reclaim', True)
            and case_stats['evicted_total'] > 0):
        sol, reclaim_info = hole_relocate(
            sol, case, case_stats['evicted_set'], pre_mask, cfg, S,
            hpwl_base, area_base, n_soft, nbr_i, alpha)
        cost = proxy_cost(sol, case, hpwl_base, area_base, n_soft)
        # A reclaimed block sits in a hole that no soft-constraint stage has
        # seen: stage S snapped and DP-1 abutted the layout while that block was
        # still parked in the shelf. Re-running one legal round rebuilds the
        # relation set from the reclaimed geometry -- which also restores the
        # slack invariant those stages rely on -- assigns minimal-movement (so
        # the reclaimed positions survive), then re-polishes and re-snaps.
        if reclaim_info['moved'] and cfg.get('reclaim_round', True):
            rr = legal_round(sol.copy(), cfg['polish_sweeps'])
            c_r, s_r, h_r, v_r, g_r = finish(rr)
            reclaim_info['round_gain'] = float(cost - c_r)
            if c_r < cost:
                cost, sol, H, V, graph_info = c_r, s_r, h_r, v_r, g_r

    # Detailed placement must not perturb immutable fields. Re-stamp them before
    # the authoritative hard gate; any overlap exposed by this restoration makes
    # the candidate infeasible rather than silently moving an immutable block.
    sol[pre_mask] = gt[pre_mask]
    sol[fixed_mask, 2:4] = gt[fixed_mask, 2:4]
    hard = hard_feasibility(sol, case)
    cost = proxy_cost(sol, case, hpwl_base, area_base, n_soft)

    # Absolute feasibility floor: stage E makes the assignment overlap-free by
    # construction, so reaching this branch means an unmodelled failure mode.
    # A shelf construction is worth ~8.5 against the 10.0 penalty -- poor, but
    # it bounds the damage from anything we have not imagined.
    floor_used = False
    if not hard['feasible'] and cfg.get('guaranteed_floor', True):
        alt = guaranteed_construction(pred, case, cfg)
        alt_hard = hard_feasibility(alt, case)
        if alt_hard['feasible']:
            alt_cost = proxy_cost(alt, case, hpwl_base, area_base, n_soft)
            sol, hard, cost, floor_used = alt, alt_hard, alt_cost, True
    pen = hard['max_penetration']
    info = {'proxy_cost': cost, 'penetration': pen, 'hard': hard,
            'floor_used': floor_used, 'reclaim': reclaim_info,
            'evict': {k: (sorted(v) if isinstance(v, set) else v)
                      for k, v in case_stats.items()},
            'move_mean': float(np.abs(sol[:, :2] - pred[:, :2]).mean() / S),
            'runtime_s': time.time() - t0, 'ripple': ripple_info,
            'reshape': reshape_info, 'mib': mib_info, 'graph': graph_info}
    if verbose:
        print(f"  proxy={cost:.4f} hard={hard['feasible']} pen={pen:.2e} "
              f"move={info['move_mean']:.4f} "
              f"ripple={ripple_info['moves']}/{ripple_info['trials']} "
              f"reshape={reshape_info['moves']}/{reshape_info['trials']} "
              f"t={info['runtime_s']:.2f}s")
    return torch.tensor(sol, dtype=torch.float64), info


def legalize_best_of(cand_list, case, cfg=None, verbose=False):
    """Look-ahead legalization over seed candidates (NTUplace LAL): keep the
    best legalized proxy; stop early when good enough or over budget."""
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    t0 = time.time()
    best = None
    for k, pred in enumerate(cand_list):
        sol, info = legalize_case(pred, case, cfg, verbose=verbose)
        info['seed_rank'] = k
        if (best is None
                or _selection_key(info, k)
                < _selection_key(best[1], best[1]['seed_rank'])):
            best = (sol, info)
        if info['hard']['feasible'] and info['proxy_cost'] < cfg['seed_stop']:
            break
        if time.time() - t0 > cfg['budget_s']:
            break
    sol, info = best
    info['runtime_s'] = time.time() - t0
    return sol, info


# ------------------------------------------------------------------ CLI

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred', type=str, required=True)
    ap.add_argument('--out', type=str, default='floordiff/out/legalized.json')
    ap.add_argument('--cases', type=str, default='')
    ap.add_argument('--topk', type=int, default=0)
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()

    preds = json.loads(Path(args.pred).read_text())['cases']
    ns = [int(x) for x in args.cases.split(',')] if args.cases else \
        sorted(int(k) for k in preds)

    out = {'cases': {}, 'meta': {'pred': args.pred, 'legalizer': 'EGL'}}
    t_sum = 0.0
    for nb in ns:
        case = load_validation_case(nb)
        entry = preds[str(nb)]
        cands = [torch.tensor(c, dtype=torch.float64)
                 for c in entry.get('candidates', [entry['positions']])]
        if args.topk > 0:
            cands = cands[:args.topk]
        sol, info = legalize_best_of(cands, case, verbose=args.verbose)
        t_sum += info['runtime_s']
        out['cases'][str(nb)] = {
            'n': nb, 'positions': sol.tolist(),
            'runtime_s': entry.get('runtime_s', 0) + info['runtime_s'],
            'legalize': info,
        }
        print(f"case n={nb:>3}: proxy={info['proxy_cost']:.4f} "
              f"pen={info['penetration']:.1e} t={info['runtime_s']:.2f}s")
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out))
    print(f'wrote {p}  (legalize time total {t_sum:.1f}s, '
          f'avg {t_sum / max(len(ns), 1):.2f}s)')


if __name__ == '__main__':
    main()
