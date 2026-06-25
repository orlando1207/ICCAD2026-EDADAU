"""
Scoring harness for the SP+SA floorplanner WITHOUT ground-truth dimensions.

Block shapes are derived algorithmically from area_targets alone:
  - free blocks:          w = sqrt(area * aspect),  h = sqrt(area / aspect)
  - fixed-shape blocks:   exact (w,h) from target_positions
  - preplaced blocks:     exact (x,y,w,h) from target_positions
  - MIB groups:           all members identical dims (first-member reference)

GT baselines (hpwl_base, area_base) are pulled from the evaluator and fed to
solve_with_dims() for correct SA cost normalisation — this is purely internal
to the SA and does NOT give the solver any GT positional information.

Usage (from FloorSet/iccad2026contest/):
    python sp-engine/sp_harness.py                   # cases 95-99
    python sp-engine/sp_harness.py all               # all 100
    python sp-engine/sp_harness.py all --budget 20 --starts 12 --save sp_sols.json
    python sp-engine/sp_harness.py 95 99 --no-rot    # ablation: rotation off
    python sp-engine/sp_harness.py 95 99 --aspect 2  # try 2:1 initial shapes
"""

import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent))

from litetestLoader import FloorplanDatasetLiteTest
import iccad2026_evaluate as ev
from compile_problem import dims_from_areas
from floorplanner import SPFloorplanner


def run(case_ids, budget=2.0, starts=8, save=None,
        enable_rotation=True, aspect=1.0, verbose=False):
    ds = FloorplanDatasetLiteTest('../')
    evaluator = ev.ContestEvaluator('../', verbose=False)
    evaluator._load_dataset()
    fp = SPFloorplanner(time_budget=budget, verbose=verbose,
                        use_macros=False, n_starts=starts,
                        enable_rotation=enable_rotation)
    rot_tag = "rot" if enable_rotation else "no-rot"
    asp_tag = f"asp={aspect:.2f}"
    print(f"SP+SA  [{rot_tag}  {asp_tag}  budget={budget}s  starts={starts}]")

    rows = []
    saved = []
    for idx in case_ids:
        sample = ds[idx]
        inputs, labels = sample['input'], sample['label']
        area_target, b2b, p2b, pins_pos, constraints = inputs
        n = int((area_target != -1).sum().item())
        baseline, target_pos = evaluator._extract_baseline(
            idx, labels, b2b, p2b, pins_pos, n)

        # derive dims without any GT dimension knowledge
        dims_wh, pre_xy = dims_from_areas(
            n, area_target, constraints, target_pos, aspect=aspect)

        # size-adaptive budget
        if n >= 90:
            fp.time_budget = budget
        elif n >= 50:
            fp.time_budget = max(1.5, budget * 0.45)
        else:
            fp.time_budget = max(1.0, budget * 0.18)

        t0 = time.time()
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
              f"hpwl={m.hpwl_gap:5.2f}  area={m.area_gap:5.2f}  "
              f"bnd={m.boundary_violations:3d}  {rt:5.1f}s", flush=True)

    max_n = max(r[1] for r in rows)
    print(f"\n{'idx':>3} {'n':>4} {'cost':>6} {'hpwl':>5} {'area':>5} "
          f"{'vrel':>5} {'bnd':>4} {'grp':>4} {'mib':>3} {'feas':>4} {'sec':>6}")
    for idx, n, m, rt in sorted(rows, key=lambda r: math.exp(r[1] - max_n) * r[2].cost,
                                  reverse=True):
        print(f"{idx:>3} {n:>4} {m.cost:>6.2f} {m.hpwl_gap:>5.2f} {m.area_gap:>5.2f} "
              f"{m.violations_relative:>5.2f} {m.boundary_violations:>4} "
              f"{m.grouping_violations:>4} {m.mib_violations:>3} "
              f"{'Y' if m.is_feasible else 'N':>4} {rt:>6.2f}")

    feas = sum(1 for _, _, m, _ in rows if m.is_feasible)
    tot_bnd = sum(m.boundary_violations for _, _, m, _ in rows)
    tot_grp = sum(m.grouping_violations for _, _, m, _ in rows)
    costs = [m.cost for _, _, m, _ in rows]
    ns = [n for _, n, _, _ in rows]
    score = ev.compute_total_score(costs, ns)
    avg_rt = sum(r[3] for r in rows) / len(rows)
    print(f"\nFeasible: {feas}/{len(rows)}   "
          f"viol totals: bnd={tot_bnd} grp={tot_grp}   avg {avg_rt:.2f}s/case")
    print(f"Weighted total score ({rot_tag}  {asp_tag}): {score:.4f}")
    print(f"  Baseline (analytic_legalizer): 1.8537   GT ceiling: ~1.11")

    if save:
        import json
        with open(save, 'w') as f:
            json.dump({'submission': f'sp-sa-{rot_tag}', 'timestamp': '',
                       'solutions': saved}, f)
        print(f"Saved {len(saved)} solutions to {save}  "
              f"(re-score: python iccad2026_evaluate.py --score {save})")
    return score


if __name__ == '__main__':
    args = sys.argv[1:]

    enable_rotation = True
    if '--no-rot' in args:
        enable_rotation = False
        args = [a for a in args if a != '--no-rot']

    budget = 2.0
    starts = 8
    save = None
    aspect = 1.0

    for flag, typ in [('--budget', float), ('--starts', int),
                      ('--aspect', float), ('--save', str)]:
        if flag in args:
            k = args.index(flag)
            val = args[k + 1]
            if flag == '--budget': budget = float(val)
            elif flag == '--starts': starts = int(val)
            elif flag == '--aspect': aspect = float(val)
            elif flag == '--save': save = val
            del args[k:k + 2]

    if not args:
        ids = list(range(95, 100))
    elif args == ['all']:
        ids = list(range(100))
    else:
        ids = [int(a) for a in args]

    run(ids, budget=budget, starts=starts, save=save,
        enable_rotation=enable_rotation, aspect=aspect)
