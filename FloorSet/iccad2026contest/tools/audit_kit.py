"""Audit a generated test kit: is it solvable, fair, and still informative?

For every case this recomputes, independently of the generator's own claim:

  * the WITNESS check -- the GT/reference layout must satisfy H1..H4, otherwise
    the case is not provably solvable and an infeasible result is not the
    solver's fault;
  * the witness's soft violations V_rel and hence its cost (the evaluator
    recomputes HPWL/area baselines from the same polygons, so the witness's
    quality gaps are exactly 0 and its cost is exp(2*V_rel_witness));
  * the ACHIEVABLE FLOOR: MIB groups whose members' 1% area bands do not share
    a common point can never all take identical dimensions, so a minimum number
    of distinct shapes is forced.  That minimum is the piercing number of the
    interval family (greedy by right endpoint is exact for intervals), giving
    V_mib_min = sum_q (pierce_q - 1) and a hard lower bound
    cost >= exp(2 * V_mib_min / N_soft) on what ANY solver can score;
  * constraint densities, for comparison against the official validation set.

Usage (from FloorSet/iccad2026contest):
    python tools/audit_kit.py <kit_data_path> [--results run.json] [--out audit.json]
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

CONTEST = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CONTEST))
sys.path.insert(0, str(CONTEST.parent))

import iccad2026_evaluate as ev
from lite_dataset_test import FloorplanDatasetLiteTest
from floordiff import legalizer as lg

AREA_TOL = 0.01


def witness_layout(labels, block_count):
    """Bounding boxes of the reference polygons -- what the evaluator uses as
    `target_positions` and as the baseline source."""
    polygons, metrics = labels
    pos = []
    for i in range(block_count):
        blk = polygons[i]
        valid = blk[blk[:, 0] != -1]
        if len(valid):
            x0, y0 = valid.min(dim=0).values
            x1, y1 = valid.max(dim=0).values
            pos.append((float(x0), float(y0), float(x1 - x0), float(y1 - y0)))
        else:
            pos.append((0.0, 0.0, 1.0, 1.0))
    return pos


def pierce_count(lo, hi):
    """Minimum number of points stabbing every interval [lo_i, hi_i] (exact for
    intervals: repeatedly take the smallest right endpoint)."""
    order = np.argsort(hi)
    pts = 0
    covered = np.zeros(len(lo), dtype=bool)
    for k in order:
        if covered[k]:
            continue
        p = hi[k]
        pts += 1
        covered |= (lo <= p) & (hi >= p)
    return pts


def mib_floor(cons, area):
    """(V_mib_min, n_groups, n_infeasible_groups): violations no solver can avoid."""
    v_min, ngrp, nbad = 0, 0, 0
    for g in np.unique(cons[:, 2]):
        if g == 0:
            continue
        ngrp += 1
        m = np.nonzero(cons[:, 2] == g)[0]
        a = area[m]
        if not (a > 0).all():
            continue
        lo, hi = (1 - AREA_TOL) * a, (1 + AREA_TOL) * a
        k = pierce_count(lo, hi)
        v_min += k - 1
        nbad += int(k > 1)
    return v_min, ngrp, nbad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('data_path')
    ap.add_argument('--results', default=None,
                    help='evaluator JSON to compare against the floor')
    ap.add_argument('--out', default='audit.json')
    a = ap.parse_args()

    ds = FloorplanDatasetLiteTest(a.data_path)
    man_path = Path(a.data_path) / 'manifest.json'
    names, knobs = {}, {}
    if man_path.exists():
        for c in json.load(open(man_path))['cases']:
            names[c['test_id']] = c.get('name', '')
            knobs[c['test_id']] = c
    got = {}
    if a.results:
        for r in json.load(open(a.results))['test_results']:
            got[r['test_id']] = r

    evl = ev.ContestEvaluator(a.data_path, verbose=False)
    rows = []
    for idx in range(len(ds)):
        s = ds[idx]
        area_t, b2b, p2b, pins, cons_t = s['input']
        n = int((area_t != -1).sum().item())
        cons = cons_t[:n].numpy()
        area = area_t[:n].numpy().astype(np.float64)
        base, tgt = evl._extract_baseline(idx, s['label'], b2b, p2b, pins, n)
        wit = witness_layout(s['label'], n)

        frozen = {i for i in range(n) if cons[i, 0] != 0 or cons[i, 1] != 0}
        ovl = ev.check_overlap(wit)
        area_bad = ev.check_area_tolerance(wit, area_t, skip_indices=frozen)
        dim_bad = ev.check_dimension_hard_constraints(wit, tgt, cons_t, n)

        w = np.asarray(wit, dtype=np.float64)
        vb, vg, vm = lg._violations_official(w, cons)
        n_soft = max(lg._n_soft_norm(cons), 1)
        v_rel_w = (vb + vg + vm) / n_soft
        hp = (ev.calculate_hpwl_b2b(wit, b2b) + ev.calculate_hpwl_p2b(wit, p2b, pins))
        gap_h = (hp - base['hpwl_baseline']) / max(base['hpwl_baseline'], 1e-6)
        gap_a = (ev.calculate_bbox_area(wit) - base['area_baseline']) \
            / max(base['area_baseline'], 1e-6)
        cost_w = min((1 + 0.5 * (max(0, gap_h) + max(0, gap_a)))
                     * math.exp(2 * v_rel_w), 10 - 1e-6)

        v_mib_min, n_mib_grp, n_mib_bad = mib_floor(cons, area)
        floor = math.exp(2 * v_mib_min / n_soft)

        row = dict(
            id=idx, name=names.get(idx, ''), n=n,
            witness_feasible=(ovl == 0 and area_bad == 0 and dim_bad == 0),
            wit_overlap=ovl, wit_area_bad=area_bad, wit_dim_bad=dim_bad,
            wit_vb=vb, wit_vg=vg, wit_vm=vm, n_soft=n_soft,
            wit_v_rel=v_rel_w, wit_cost=cost_w,
            v_mib_min=v_mib_min, n_mib_groups=n_mib_grp,
            n_mib_infeasible=n_mib_bad, cost_floor=floor,
            n_pre=int((cons[:, 1] != 0).sum()), n_fix=int((cons[:, 0] != 0).sum()),
            n_bnd=int((cons[:, 4] != 0).sum()),
            n_clu=len([g for g in np.unique(cons[:, 3]) if g > 0]),
            pre_f=float((cons[:, 1] != 0).mean()),
            fix_f=float((cons[:, 0] != 0).mean()),
            bnd_f=float((cons[:, 4] != 0).mean()),
            util=float(area.sum() / max(ev.calculate_bbox_area(wit), 1e-9)),
        )
        if idx in got:
            g = got[idx]
            row.update(got_cost=g['cost'], got_feasible=g['is_feasible'],
                       got_v_rel=g['violations_relative'],
                       got_gap_h=g['hpwl_gap'], got_gap_a=g['area_gap'],
                       got_runtime=g['runtime_seconds'],
                       headroom=g['cost'] - floor,
                       vs_witness=g['cost'] - cost_w)
        rows.append(row)
        print(f"{idx:4d} n={n:3d} pre={row['n_pre']:3d} fix={row['n_fix']:3d} "
              f"bnd={row['n_bnd']:3d} | witness {'OK ' if row['witness_feasible'] else 'BAD'} "
              f"V_rel={v_rel_w:.3f} cost={cost_w:.3f} | floor={floor:.3f}"
              + (f" | got {g['cost']:.3f} ({'feas' if g['is_feasible'] else 'INFEAS'})"
                 if idx in got else ''), flush=True)

    json.dump(rows, open(a.out, 'w'), indent=1)

    def col(k):
        return np.array([r[k] for r in rows if k in r], dtype=float)

    print(f"\n{'='*74}\ncases {len(rows)}")
    bad = [r['id'] for r in rows if not r['witness_feasible']]
    print(f"witness feasible: {len(rows)-len(bad)}/{len(rows)}"
          + (f"  BAD: {bad}" if bad else "  -> every case is provably solvable"))
    for k in ('n', 'pre_f', 'fix_f', 'bnd_f', 'util', 'wit_v_rel', 'wit_cost',
              'cost_floor'):
        v = col(k)
        print(f"  {k:10s} mean {v.mean():7.3f}  p10 {np.percentile(v,10):7.3f}  "
              f"p50 {np.percentile(v,50):7.3f}  p90 {np.percentile(v,90):7.3f}  "
              f"max {v.max():7.3f}")
    nb = sum(r['n_mib_infeasible'] for r in rows)
    ng = sum(r['n_mib_groups'] for r in rows)
    print(f"  MIB groups with NO common legal shape: {nb}/{ng} "
          f"({100*nb/max(ng,1):.0f}%) -> that part of V_rel is unavoidable")
    if got:
        c = col('got_cost')
        f = col('cost_floor')
        w = col('wit_cost')
        print(f"\n  our cost   mean {c.mean():.3f}  p50 {np.median(c):.3f}  max {c.max():.3f}")
        print(f"  floor      mean {f.mean():.3f}   (headroom mean {(c-f).mean():.3f})")
        print(f"  witness    mean {w.mean():.3f}   (we beat it on "
              f"{int((c<w).sum())}/{len(c)} cases)")
        print(f"  feasible   {int(col('got_feasible').sum())}/{len(c)}")


if __name__ == '__main__':
    main()
