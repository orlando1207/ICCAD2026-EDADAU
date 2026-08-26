"""Fast, deterministic A/B of legalizer configs over cached predictions.

Sampling is the expensive, GPU-bound half of the pipeline and is unaffected by
legalizer knobs, so dump candidates once

    python -m floordiff.sample --ckpt floordiff/checkpoints/myrun/last.pt \
        --n-seeds 32 --steps 50 --save-topk 24 --out floordiff/out/preds_ab.json

and then sweep configs against that file.  Each variant legalizes every cached
candidate, selects with the production `_selection_key`, and is scored with the
OFFICIAL formula at RuntimeFactor = 1 -- the same runtime-neutral total the
evaluator prints, so numbers here are directly comparable to it.

    python tools/ab_legalize.py floordiff/out/preds_ab.json                 # baseline
    python tools/ab_legalize.py preds.json --variants variants.json         # sweep
    python tools/ab_legalize.py preds.json --cfg '{"g_iters":200}' --tag g200

Reported per variant: exp-weighted TOTAL (the contest metric), mean cost,
feasible count, mean legalize wall-clock, and how often stage E fired.
"""

import argparse
import json
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch

CONTEST = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CONTEST))
sys.path.insert(0, str(CONTEST.parent))

from iccad2026_evaluate import compute_total_score
from floordiff.data import load_validation_case
from floordiff.legalizer import legalize_case, _selection_key
from floordiff.score_official import official_cost
from tools.kitcase import kit_case, official_metrics


def _one_case(args):
    """Legalize every candidate of one case under `cfg`; return the selected."""
    n_blocks, cands, cfg, kit, idx = args
    if kit is None:
        case = load_validation_case(n_blocks)
    else:
        case, _base, n_blocks = kit_case(kit, idx)
    t0 = time.perf_counter()
    best = None
    stats = {'evict_rounds': 0, 'evicted': 0, 'floor': 0, 'reclaimed': 0,
             'repair_fail': 0, 'assign_fail': 0}
    for k, c in enumerate(cands):
        sol, info = legalize_case(torch.tensor(c, dtype=torch.float64), case, cfg)
        info['seed_rank'] = k
        stats['evict_rounds'] += info['evict']['evict_rounds']
        stats['evicted'] += info['evict']['evicted_total']
        stats['floor'] += int(info['floor_used'])
        stats['reclaimed'] += info['reclaim']['moved']
        stats['repair_fail'] += info['graph']['repair_failures']
        stats['assign_fail'] += info['graph']['assign_failures']
        if best is None or _selection_key(info, k) < _selection_key(best[1],
                                                                   best[1]['seed_rank']):
            best = (sol, info)
    sol, info = best
    m = official_cost(sol, case) if kit is None else official_metrics(kit, idx, sol)
    ev_info = info['evict']
    return dict(n=n_blocks, id=idx, cost=m.cost, feasible=bool(m.is_feasible),
                evicted_set=list(ev_info.get('evicted_set', [])),
                hpwl_gap=m.hpwl_gap, area_gap=m.area_gap,
                v_rel=m.violations_relative, vb=m.boundary_violations,
                vg=m.grouping_violations, vm=m.mib_violations,
                seed=info['seed_rank'], t=time.perf_counter() - t0, **stats)


def run(preds, cfg, workers, cases=None, topk=None):
    kit = preds.get('kit')
    tasks = []
    for key, entry in sorted(preds['cases'].items(), key=lambda kv: int(kv[0])):
        k = int(key)
        n = entry.get('n', k)
        if cases and k not in cases:
            continue
        cands = entry.get('candidates') or [entry['positions']]
        tasks.append((n, cands[:topk] if topk else cands, cfg, kit, k))
    ctx = get_context('spawn')
    with ctx.Pool(workers) as pool:
        rows = pool.map(_one_case, tasks)
    return rows


def summarize(tag, rows):
    costs = [r['cost'] for r in rows]
    ns = [r['n'] for r in rows]
    return dict(tag=tag, total=compute_total_score(costs, ns),
                mean=float(np.mean(costs)),
                feasible=int(sum(r['feasible'] for r in rows)), n=len(rows),
                worst=float(max(costs)),
                v_rel=float(np.mean([r['v_rel'] for r in rows])),
                area_gap=float(np.mean([max(0, r['area_gap']) for r in rows])),
                hpwl_gap=float(np.mean([max(0, r['hpwl_gap']) for r in rows])),
                evict_cases=int(sum(1 for r in rows if r['evict_rounds'])),
                evicted=int(sum(r['evicted'] for r in rows)),
                floor=int(sum(r['floor'] for r in rows)),
                rung_fail=int(sum(1 for r in rows if r['assign_fail'])),
                t_mean=float(np.mean([r['t'] for r in rows])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('preds')
    ap.add_argument('--cfg', default='{}')
    ap.add_argument('--tag', default='baseline')
    ap.add_argument('--variants', default=None,
                    help='JSON: {"tag": {cfg overrides}, ...}')
    ap.add_argument('--cases', default=None, help='comma-separated n values')
    ap.add_argument('--topk', type=int, default=None)
    ap.add_argument('--workers', type=int, default=24)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()

    preds = json.loads(Path(a.preds).read_text())
    cases = {int(x) for x in a.cases.split(',')} if a.cases else None
    plan = {a.tag: json.loads(a.cfg)}
    if a.variants:
        plan = json.loads(Path(a.variants).read_text())

    out, base = [], None
    hdr = (f"{'variant':<22} {'TOTAL':>8} {'Δ':>8} {'mean':>7} {'feas':>5} "
           f"{'V_rel':>6} {'areaΔ':>6} {'hpwlΔ':>6} {'evictC':>6} {'rung':>5} {'t/case':>7}")
    print(hdr)
    print('-' * len(hdr))
    for tag, cfg in plan.items():
        t0 = time.time()
        rows = run(preds, cfg, a.workers, cases, a.topk)
        s = summarize(tag, rows)
        s['wall_s'] = time.time() - t0
        s['cfg'] = cfg
        s['rows'] = rows
        if base is None:
            base = s['total']
        d = s['total'] - base
        print(f"{tag:<22} {s['total']:>8.4f} {d:>+8.4f} {s['mean']:>7.4f} "
              f"{s['feasible']:>3d}/{s['n']:<2d} {s['v_rel']:>6.4f} "
              f"{s['area_gap']:>6.4f} {s['hpwl_gap']:>6.4f} {s['evict_cases']:>6d} "
              f"{s['rung_fail']:>5d} {s['t_mean']:>7.3f}", flush=True)
        out.append(s)
    if a.out:
        Path(a.out).write_text(json.dumps(out))
        print(f'\nwrote {a.out}')


if __name__ == '__main__':
    main()
