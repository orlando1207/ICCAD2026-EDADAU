"""
Validation harness for the "ground-truth dimensions known" experiment.

For each validation case it feeds the GT (w,h) of every block (and the GT (x,y)
of preplaced blocks) into the SP floorplanner, then scores with the official
evaluator + provided baselines. Isolates PLACEMENT quality from shaping.

Usage (run from iccad2026contest/):
    python ml-engine/gt_dims_harness.py                 # cases 95-99 (dominant)
    python ml-engine/gt_dims_harness.py all             # all 100
    python ml-engine/gt_dims_harness.py 0 1 2
    python ml-engine/gt_dims_harness.py all --ceiling    # place at GT (upper bound)
    python ml-engine/gt_dims_harness.py all --budget 3   # 3s/case SA budget
"""

import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))                 # ml-engine modules
sys.path.insert(0, str(HERE.parent))          # iccad2026contest
sys.path.insert(0, str(HERE.parent.parent))   # FloorSet root (loaders)

from litetestLoader import FloorplanDatasetLiteTest
import iccad2026_evaluate as ev
from floorplanner import SPFloorplanner


def run(case_ids, ceiling=False, budget=2.0, macros=False, starts=8, save=None):
    ds = FloorplanDatasetLiteTest('../')
    evaluator = ev.ContestEvaluator('../', verbose=False)
    evaluator._load_dataset()
    fp = SPFloorplanner(time_budget=budget, verbose=False, use_macros=macros,
                        n_starts=starts)

    rows = []
    saved = []
    for idx in case_ids:
        sample = ds[idx]
        inputs, labels = sample['input'], sample['label']
        area_target, b2b, p2b, pins_pos, constraints = inputs
        n = int((area_target != -1).sum().item())
        baseline, target_pos = evaluator._extract_baseline(
            idx, labels, b2b, p2b, pins_pos, n)

        dims_wh = np.array([[target_pos[i][2], target_pos[i][3]] for i in range(n)],
                           dtype=float)
        pre_xy = np.array([[target_pos[i][0], target_pos[i][1]] for i in range(n)],
                          dtype=float)

        # size-adaptive budget: spend compute where the e^(n/12) weight is.
        if n >= 90:
            fp.time_budget = budget
        elif n >= 50:
            fp.time_budget = max(1.5, budget * 0.45)
        else:
            fp.time_budget = max(1.0, budget * 0.18)

        t0 = time.time()
        if ceiling:
            positions = [tuple(map(float, target_pos[i])) for i in range(n)]
        else:
            positions = fp.solve_with_dims(
                n, b2b, p2b, pins_pos, constraints, dims_wh, pre_xy,
                baseline['hpwl_baseline'], baseline['area_baseline'])
        rt = time.time() - t0

        saved.append({'test_id': int(idx), 'block_count': int(n),
                      'positions': [list(map(float, p)) for p in positions]})

        m = ev.evaluate_solution(
            {'positions': positions, 'runtime': 1.0}, baseline, constraints,
            b2b, p2b, pins_pos, area_target, target_pos, median_runtime=1.0)
        rows.append((idx, n, m, rt))
        print(f"  [{len(rows):>3}/{len(case_ids)}] case {idx:>3} n={n:>3}  "
              f"cost={m.cost:5.2f}  feas={'Y' if m.is_feasible else 'N'}  "
              f"{rt:5.1f}s", flush=True)

    max_n = max(r[1] for r in rows)
    print(f"{'idx':>3} {'n':>4} {'cost':>6} {'hpwl':>5} {'area':>5} {'vrel':>5} "
          f"{'bnd':>4} {'grp':>4} {'mib':>3} {'feas':>4} {'sec':>6}")
    for idx, n, m, rt in sorted(rows, key=lambda r: math.exp(r[1] - max_n) * r[2].cost,
                                reverse=True):
        print(f"{idx:>3} {n:>4} {m.cost:>6.2f} {m.hpwl_gap:>5.2f} {m.area_gap:>5.2f} "
              f"{m.violations_relative:>5.2f} {m.boundary_violations:>4} "
              f"{m.grouping_violations:>4} {m.mib_violations:>3} "
              f"{'Y' if m.is_feasible else 'N':>4} {rt:>6.2f}")

    feas = sum(1 for _, _, m, _ in rows if m.is_feasible)
    tot_bnd = sum(m.boundary_violations for _, _, m, _ in rows)
    tot_grp = sum(m.grouping_violations for _, _, m, _ in rows)
    tot_mib = sum(m.mib_violations for _, _, m, _ in rows)
    costs = [m.cost for _, _, m, _ in rows]
    ns = [n for _, n, _, _ in rows]
    score = ev.compute_total_score(costs, ns)
    avg_rt = sum(r[3] for r in rows) / len(rows)
    print(f"\nFeasible: {feas}/{len(rows)}   "
          f"viol totals: boundary={tot_bnd} grouping={tot_grp} mib={tot_mib}   "
          f"avg {avg_rt:.2f}s/case")
    print(f"Weighted total score (this subset): {score:.4f}"
          + ("   [GT-position ceiling]" if ceiling else ""))

    if save:
        import json
        with open(save, 'w') as f:
            json.dump({'submission': 'ml-engine-gtdims', 'timestamp': '',
                       'solutions': saved}, f)
        print(f"Saved {len(saved)} solutions to {save}  "
              f"(re-score: python iccad2026_evaluate.py --score {save})")


if __name__ == '__main__':
    args = sys.argv[1:]
    ceiling = '--ceiling' in args
    macros = '--macros' in args
    args = [a for a in args if a not in ('--ceiling', '--macros')]
    budget = 2.0
    starts = 8
    save = None
    if '--budget' in args:
        k = args.index('--budget'); budget = float(args[k + 1]); del args[k:k + 2]
    if '--starts' in args:
        k = args.index('--starts'); starts = int(args[k + 1]); del args[k:k + 2]
    if '--save' in args:
        k = args.index('--save'); save = args[k + 1]; del args[k:k + 2]
    if not args:
        ids = list(range(95, 100))
    elif args == ['all']:
        ids = list(range(100))
    else:
        ids = [int(a) for a in args]
    run(ids, ceiling=ceiling, budget=budget, macros=macros, starts=starts, save=save)
