"""Stage-1 evaluation: closeness-to-GT (primary) + quality/feasibility (secondary).

Metrics per design doc C.6:
  1-3  displacement stats, shape error, within-delta fractions   (primary)
  4-6  overlap ratio, weighted-HPWL gap, bbox-area gap, soft violations (secondary)
  7    per-case table lets violations be read against displacement (diagnostic)

Usage:
  python -m floordiff.evaluate --pred out/preds.json
  python -m floordiff.evaluate --gt-check          # harness sanity: GT must score 0-gaps
"""

import argparse
import json
import math
from pathlib import Path

import torch

from .data import VALIDATION_NS, load_validation_case, gt_xywh


# --------------------------------------------------------------------------- kernels

def weighted_hpwl(xywh, case):
    cx = xywh[:, 0] + xywh[:, 2] / 2
    cy = xywh[:, 1] + xywh[:, 3] / 2
    total = 0.0
    b2b = case['b2b']
    if len(b2b):
        i, j, w = b2b[:, 0].long(), b2b[:, 1].long(), b2b[:, 2]
        total += (w * ((cx[i] - cx[j]).abs() + (cy[i] - cy[j]).abs())).sum().item()
    p2b, pins = case['p2b'], case['pins']
    if len(p2b):
        pi, bi, w = p2b[:, 0].long(), p2b[:, 1].long(), p2b[:, 2]
        total += (w * ((cx[bi] - pins[pi, 0]).abs()
                       + (cy[bi] - pins[pi, 1]).abs())).sum().item()
    return total


def bbox_area(xywh):
    W = (xywh[:, 0] + xywh[:, 2]).max() - xywh[:, 0].min()
    H = (xywh[:, 1] + xywh[:, 3]).max() - xywh[:, 1].min()
    return float(W * H)


def overlap_ratio(xywh):
    x0, y0 = xywh[:, 0], xywh[:, 1]
    x1, y1 = x0 + xywh[:, 2], y0 + xywh[:, 3]
    ow = (torch.minimum(x1[:, None], x1) - torch.maximum(x0[:, None], x0)).clamp(min=0)
    oh = (torch.minimum(y1[:, None], y1) - torch.maximum(y0[:, None], y0)).clamp(min=0)
    o = ow * oh
    o.fill_diagonal_(0)
    return float(o.sum() / 2 / (xywh[:, 2] * xywh[:, 3]).sum())


def _touch(a, b, tol):
    """Blocks (x,y,w,h) share an edge segment of positive length (within tol gap)."""
    ax0, ay0, ax1, ay1 = a[0], a[1], a[0] + a[2], a[1] + a[3]
    bx0, by0, bx1, by1 = b[0], b[1], b[0] + b[2], b[1] + b[3]
    yov = min(ay1, by1) - max(ay0, by0)
    xov = min(ax1, bx1) - max(ax0, bx0)
    if (abs(ax1 - bx0) <= tol or abs(bx1 - ax0) <= tol) and yov > tol:
        return True
    if (abs(ay1 - by0) <= tol or abs(by1 - ay0) <= tol) and xov > tol:
        return True
    return False


def soft_violations(xywh, case, tol):
    """(V_boundary, V_grouping, V_mib, N_soft) per contest spec v10 Eq. 4."""
    cons = case['cons']
    n = xywh.shape[0]
    x0s, y0s = xywh[:, 0], xywh[:, 1]
    x1s, y1s = x0s + xywh[:, 2], y0s + xywh[:, 3]
    bx0, bx1 = x0s.min(), x1s.max()
    by0, by1 = y0s.min(), y1s.max()

    v_bnd, n_bnd = 0, 0
    for i in range(n):
        bits = int(cons[i, 4])
        if bits == 0:
            continue
        n_bnd += 1
        ok = True
        if bits & 1:
            ok &= abs(x0s[i] - bx0) <= tol
        if bits & 2:
            ok &= abs(x1s[i] - bx1) <= tol
        if bits & 4:
            ok &= abs(y1s[i] - by1) <= tol
        if bits & 8:
            ok &= abs(y0s[i] - by0) <= tol
        v_bnd += int(not ok)

    v_grp, n_grp = 0, 0
    for g in torch.unique(cons[:, 3]):
        if g == 0:
            continue
        idx = torch.nonzero(cons[:, 3] == g).flatten().tolist()
        n_grp += len(idx) - 1
        # connected components via edge-abutment
        seen, comps = set(), 0
        for start in idx:
            if start in seen:
                continue
            comps += 1
            stack = [start]
            seen.add(start)
            while stack:
                u = stack.pop()
                for v in idx:
                    if v not in seen and _touch(xywh[u], xywh[v], tol):
                        seen.add(v)
                        stack.append(v)
        v_grp += comps - 1

    v_mib, n_mib = 0, 0
    for g in torch.unique(cons[:, 2]):
        if g == 0:
            continue
        idx = torch.nonzero(cons[:, 2] == g).flatten()
        n_mib += len(idx) - 1
        shapes = []
        for i in idx.tolist():
            wh = (float(xywh[i, 2]), float(xywh[i, 3]))
            if not any(abs(wh[0] - a) <= tol and abs(wh[1] - b) <= tol
                       for a, b in shapes):
                shapes.append(wh)
        v_mib += len(shapes) - 1

    return v_bnd, v_grp, v_mib, n_bnd + n_grp + n_mib


# --------------------------------------------------------------------------- per case

def evaluate_case(xywh, case, tol_rel_loose=0.01):
    m = case['metrics']
    gt = gt_xywh(case)
    S = case['area'].sum().sqrt()

    pc = torch.stack([xywh[:, 0] + xywh[:, 2] / 2, xywh[:, 1] + xywh[:, 3] / 2], 1)
    gc = torch.stack([gt[:, 0] + gt[:, 2] / 2, gt[:, 1] + gt[:, 3] / 2], 1)
    disp = (pc - gc).norm(dim=1) / S
    s_pred = 0.5 * torch.log(xywh[:, 2] / xywh[:, 3])
    s_gt = 0.5 * torch.log(gt[:, 2] / gt[:, 3])
    soft = (case['cons'][:, 0] == 0) & (case['cons'][:, 1] == 0)

    hpwl = weighted_hpwl(xywh, case)
    hpwl_base = float(m[6] + m[7])
    area = bbox_area(xywh)
    area_base = float(m[0])

    tol_strict = 1e-4 * float(S)
    tol_loose = tol_rel_loose * float(S)
    vb, vg, vm, ns = soft_violations(xywh, case, tol_strict)
    vb2, vg2, vm2, _ = soft_violations(xywh, case, tol_loose)

    return {
        'n': xywh.shape[0],
        'disp_mean': float(disp.mean()),
        'disp_median': float(disp.median()),
        'within_1pct': float((disp <= 0.01).float().mean()),
        'within_2pct': float((disp <= 0.02).float().mean()),
        'within_5pct': float((disp <= 0.05).float().mean()),
        'shape_err': float((s_pred - s_gt)[soft].abs().mean()) if soft.any() else 0.0,
        'overlap_ratio': overlap_ratio(xywh),
        'hpwl_gap': (hpwl - hpwl_base) / hpwl_base if hpwl_base > 0 else 0.0,
        'area_gap': (area - area_base) / area_base,
        'viol_strict': vb + vg + vm,
        'viol_loose': vb2 + vg2 + vm2,
        'n_soft': ns,
    }


def aggregate(rows):
    """Unweighted and contest-weighted (exp(n/12)) means of every metric."""
    keys = [k for k in rows[0] if k != 'n']
    w = torch.tensor([math.exp(r['n'] / 12) for r in rows], dtype=torch.float64)
    w = w / w.sum()
    out = {}
    for k in keys:
        v = torch.tensor([float(r[k]) for r in rows], dtype=torch.float64)
        out[k] = {'mean': float(v.mean()), 'weighted': float((v * w).sum())}
    return out


def print_report(rows, agg):
    hdr = (f"{'n':>4} {'disp':>7} {'<2%S':>6} {'shape':>7} {'ovl%':>6} "
           f"{'hpwlgap':>8} {'areagap':>8} {'viol':>5} {'viol~':>5} {'Nsoft':>5}")
    print(hdr)
    for r in rows:
        print(f"{r['n']:>4} {r['disp_mean']:>7.4f} {r['within_2pct']:>6.2f} "
              f"{r['shape_err']:>7.4f} {100 * r['overlap_ratio']:>6.2f} "
              f"{r['hpwl_gap']:>8.3f} {r['area_gap']:>8.3f} "
              f"{r['viol_strict']:>5d} {r['viol_loose']:>5d} {r['n_soft']:>5d}")
    print('-' * len(hdr))
    for k in ('disp_mean', 'within_2pct', 'shape_err', 'overlap_ratio',
              'hpwl_gap', 'area_gap', 'viol_strict', 'viol_loose'):
        a = agg[k]
        print(f"{k:>14}: mean {a['mean']:.4f} | contest-weighted {a['weighted']:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred', type=str, help='predictions JSON from floordiff.sample')
    ap.add_argument('--gt-check', action='store_true',
                    help='evaluate ground truth against itself (P0 harness sanity)')
    ap.add_argument('--cases', type=str, default='',
                    help='comma-separated n values (default: all 21..120)')
    args = ap.parse_args()

    ns = [int(x) for x in args.cases.split(',')] if args.cases else VALIDATION_NS
    preds = None
    if args.pred:
        preds = json.loads(Path(args.pred).read_text())['cases']

    rows = []
    for n in ns:
        case = load_validation_case(n)
        if args.gt_check:
            xywh = gt_xywh(case)
        else:
            xywh = torch.tensor(preds[str(n)]['positions'], dtype=torch.float64)
        rows.append(evaluate_case(xywh, case))
    print_report(rows, aggregate(rows))


if __name__ == '__main__':
    main()
