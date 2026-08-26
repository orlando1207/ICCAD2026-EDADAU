"""What does stage E sacrifice, and on which blocks?

Runs the legalizer over cached kit candidates and, for the selected solution of
every case where stage E fired, attributes the cost damage:

  * shelf tax      bbox of the whole layout vs bbox of the CORE alone (the
                   evicted blocks' own contribution to the area gap);
  * boundary       which violating blocks are evicted vs core;
  * grouping       which split groups contain an evicted member;
  * HPWL           the weighted wirelength incident on evicted blocks, and how
                   far each evicted block sits from its netlist barycentre.

    python tools/analyze_evict.py /tmp/preds_stress.json [--cfg '{}'] [--out a.json]
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

CONTEST = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CONTEST))
sys.path.insert(0, str(CONTEST.parent))

from floordiff import legalizer as lg
from tools.kitcase import kit_case, official_metrics

EPS = 1e-6


def bnd_violators(sol, cons):
    x0, y0 = sol[:, 0], sol[:, 1]
    x1, y1 = x0 + sol[:, 2], y0 + sol[:, 3]
    bx0, bx1, by0, by1 = x0.min(), x1.max(), y0.min(), y1.max()
    bad = []
    for i in np.nonzero(cons[:, 4])[0]:
        b = int(cons[i, 4])
        ok = True
        if b & 1:
            ok &= abs(x0[i] - bx0) < EPS
        if b & 2:
            ok &= abs(x1[i] - bx1) < EPS
        if b & 4:
            ok &= abs(y1[i] - by1) < EPS
        if b & 8:
            ok &= abs(y0[i] - by0) < EPS
        if not ok:
            bad.append(int(i))
    return bad


def split_groups(sol, cons):
    """Cluster groups that are not one connected component -> member lists."""
    out = []
    for g in np.unique(cons[:, 3]):
        if g == 0:
            continue
        mem = np.nonzero(cons[:, 3] == g)[0]
        comps = lg._touch_components(sol, mem)
        if len(comps) > 1:
            out.append((int(g), [int(m) for m in mem], len(comps)))
    return out


def analyze_case(kit, idx, cands, cfg):
    case, base, n = kit_case(kit, idx)
    cons = case['cons'].numpy()
    best = None
    for k, c in enumerate(cands):
        sol, info = lg.legalize_case(torch.tensor(c, dtype=torch.float64), case, cfg)
        info['seed_rank'] = k
        if best is None or lg._selection_key(info, k) < lg._selection_key(
                best[1], best[1]['seed_rank']):
            best = (sol, info)
    sol_t, info = best
    sol = sol_t.numpy()
    ev = sorted(int(v) for v in info['evict'].get('evicted_set', []))
    m = official_metrics(kit, idx, sol)
    row = dict(id=idx, n=n, cost=m.cost, feasible=bool(m.is_feasible),
               area_gap=m.area_gap, hpwl_gap=m.hpwl_gap,
               v_rel=m.violations_relative, vb=m.boundary_violations,
               vg=m.grouping_violations, vm=m.mib_violations,
               n_evicted=len(ev), evicted=ev,
               n_pre=int((cons[:, 1] != 0).sum()),
               reclaimed=info['reclaim']['moved'],
               floor=info['floor_used'])
    if not ev:
        return row

    # --- shelf tax: bbox with vs without the evicted blocks
    core = np.array([i for i in range(n) if i not in set(ev)], dtype=np.int64)
    full = lg._bbox_area(torch.tensor(sol))
    core_bb = lg._bbox_area(torch.tensor(sol[core]))
    row['bbox_full'] = float(full)
    row['bbox_core'] = float(core_bb)
    row['shelf_tax'] = float((full - core_bb) / max(base['area_baseline'], 1e-9))
    row['core_area_gap'] = float((core_bb - base['area_baseline'])
                                 / max(base['area_baseline'], 1e-9))

    # --- which violations sit on evicted blocks
    evs = set(ev)
    bad_b = bnd_violators(sol, cons)
    row['vb_on_evicted'] = sum(1 for i in bad_b if i in evs)
    row['vb_on_core'] = sum(1 for i in bad_b if i not in evs)
    row['n_bnd_evicted'] = sum(1 for i in ev if cons[i, 4] != 0)
    sg = split_groups(sol, cons)
    row['vg_groups'] = len(sg)
    row['vg_groups_with_evicted'] = sum(1 for _g, mem, _c in sg
                                        if any(m in evs for m in mem))
    row['n_clu_evicted'] = sum(1 for i in ev if cons[i, 3] != 0)

    # --- HPWL incident on evicted blocks, and their displacement from barycentre
    nbr = lg._nbr_lists(case, n)
    cx = sol[:, 0] + sol[:, 2] / 2
    cy = sol[:, 1] + sol[:, 3] / 2
    tot = inc = 0.0
    disp = []
    for i in range(n):
        for (j, w, pxy) in nbr[i]:
            tx = cx[j] if j >= 0 else pxy[0]
            ty = cy[j] if j >= 0 else pxy[1]
            d = w * (abs(cx[i] - tx) + abs(cy[i] - ty))
            tot += d
            if i in evs or (j >= 0 and j in evs):
                inc += d
    for i in ev:
        ws = sum(w for (_j, w, _p) in nbr[i]) or 1.0
        bx = sum(w * (cx[j] if j >= 0 else p[0]) for (j, w, p) in nbr[i]) / ws
        by = sum(w * (cy[j] if j >= 0 else p[1]) for (j, w, p) in nbr[i]) / ws
        S = float(np.sqrt(case['area'].numpy().sum()))
        disp.append((abs(cx[i] - bx) + abs(cy[i] - by)) / S)
    row['hpwl_share_evicted'] = float(inc / max(tot, 1e-9))
    row['evicted_frac'] = len(ev) / n
    row['evicted_disp_mean'] = float(np.mean(disp)) if disp else 0.0
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('preds')
    ap.add_argument('--cfg', default='{}')
    ap.add_argument('--out', default=None)
    ap.add_argument('--workers', type=int, default=24)
    a = ap.parse_args()
    preds = json.loads(Path(a.preds).read_text())
    kit = preds['kit']
    cfg = json.loads(a.cfg)
    from multiprocessing import get_context
    tasks = [(kit, int(k), v['candidates'], cfg)
             for k, v in sorted(preds['cases'].items(), key=lambda kv: int(kv[0]))]
    with get_context('spawn').Pool(a.workers) as pool:
        rows = pool.starmap(analyze_case, tasks)

    fired = [r for r in rows if r['n_evicted']]
    quiet = [r for r in rows if not r['n_evicted']]
    print(f"cases {len(rows)}  feasible {sum(r['feasible'] for r in rows)}  "
          f"stage E fired on {len(fired)}")
    if quiet:
        print(f"  no eviction : mean cost {np.mean([r['cost'] for r in quiet]):.4f}  "
              f"V_rel {np.mean([r['v_rel'] for r in quiet]):.4f}  "
              f"areaΔ {np.mean([max(0,r['area_gap']) for r in quiet]):+.4f}")
    if fired:
        f = fired
        print(f"  eviction    : mean cost {np.mean([r['cost'] for r in f]):.4f}  "
              f"V_rel {np.mean([r['v_rel'] for r in f]):.4f}  "
              f"areaΔ {np.mean([max(0,r['area_gap']) for r in f]):+.4f}")
        print()
        print(f"  evicted blocks: mean {np.mean([r['n_evicted'] for r in f]):.1f} "
              f"({100*np.mean([r['evicted_frac'] for r in f]):.1f}% of blocks)  "
              f"max {max(r['n_evicted'] for r in f)}")
        print(f"  shelf tax (bbox growth from the evicted shelf alone): "
              f"mean {np.mean([r['shelf_tax'] for r in f]):+.4f}  "
              f"of a total areaΔ {np.mean([max(0,r['area_gap']) for r in f]):+.4f}")
        print(f"  core-only area gap:                                    "
              f"mean {np.mean([r['core_area_gap'] for r in f]):+.4f}")
        vbe = sum(r['vb_on_evicted'] for r in f)
        vbc = sum(r['vb_on_core'] for r in f)
        nbe = sum(r['n_bnd_evicted'] for r in f)
        print(f"  boundary violations: {vbe} on evicted blocks, {vbc} on core "
              f"({100*vbe/max(vbe+vbc,1):.0f}% attributable to eviction); "
              f"{nbe} evicted blocks carried a boundary bit")
        gg = sum(r['vg_groups'] for r in f)
        ge = sum(r['vg_groups_with_evicted'] for r in f)
        print(f"  split cluster groups: {gg}, of which {ge} contain an evicted "
              f"member ({100*ge/max(gg,1):.0f}%); "
              f"{sum(r['n_clu_evicted'] for r in f)} evicted blocks were in a group")
        print(f"  HPWL incident on evicted blocks: "
              f"{100*np.mean([r['hpwl_share_evicted'] for r in f]):.1f}% of total; "
              f"mean displacement from netlist barycentre "
              f"{np.mean([r['evicted_disp_mean'] for r in f]):.3f} S")
        print(f"  reclaim moved: {sum(r['reclaimed'] for r in f)} blocks back")
    if a.out:
        Path(a.out).write_text(json.dumps(rows))
        print('wrote', a.out)


if __name__ == '__main__':
    main()
