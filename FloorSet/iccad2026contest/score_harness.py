"""
Fast scoring harness — loads the dataset ONCE and scores a chosen case set with a
full violation/quality breakdown plus the e^n-weighted contribution of each case.

Usage:
    python score_harness.py             # default: cases 95-99 (the score-dominant ones)
    python score_harness.py 95 96 97    # specific cases
    python score_harness.py all         # all 100 (feasibility guard + official-style score)

The weighted total uses the same weight = e^(n - n_max) as compute_total_score, but is
computed over WHATEVER case subset is given, so the "wt%" column shows each case's share
within the subset. For the official number run with `all`.
"""

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from litetestLoader import FloorplanDatasetLiteTest
import iccad2026_evaluate as ev
from analytic_legalizer.optimizer import MyOptimizer


def build_opt_target_pos(constraints, target_pos, n):
    otp = torch.full((n, 4), -1.0)
    if target_pos is None or constraints is None:
        return otp
    nc = constraints.shape[1] if constraints.dim() > 1 else 0
    for i in range(n):
        is_fixed = nc > 0 and constraints[i, 0] != 0
        is_pre = nc > 1 and constraints[i, 1] != 0
        if is_pre:
            tx, ty, tw, th = target_pos[i]
            otp[i] = torch.tensor([tx, ty, tw, th])
        elif is_fixed:
            _, _, tw, th = target_pos[i]
            otp[i, 2] = tw
            otp[i, 3] = th
    return otp


def main(case_ids):
    ds = FloorplanDatasetLiteTest('../')
    evaluator = ev.ContestEvaluator('../', verbose=False)
    evaluator._load_dataset()
    opt = MyOptimizer(verbose=False)

    rows = []
    for idx in case_ids:
        sample = ds[idx]
        inputs, labels = sample['input'], sample['label']
        area_target, b2b, p2b, pins_pos, constraints = inputs
        n = int((area_target != -1).sum().item())
        baseline, target_pos = evaluator._extract_baseline(
            idx, labels, b2b, p2b, pins_pos, n)
        otp = build_opt_target_pos(constraints, target_pos, n)
        positions = opt.solve(n, area_target, b2b, p2b, pins_pos, constraints, otp)
        m = ev.evaluate_solution(
            {'positions': positions, 'runtime': 1.0}, baseline, constraints,
            b2b, p2b, pins_pos, area_target, target_pos, median_runtime=1.0)
        rows.append((idx, n, m))

    max_n = max(r[1] for r in rows)
    total_w = sum(math.exp(r[1] - max_n) for r in rows)

    print(f"{'idx':>3} {'n':>4} {'cost':>6} {'hpwl':>5} {'area':>5} {'vrel':>5} "
          f"{'bnd':>4} {'grp':>4} {'mib':>3} {'feas':>4} {'wt%':>5} {'wcost':>7}")
    weighted_costs = []
    block_counts = []
    for idx, n, m in sorted(rows, key=lambda r: math.exp(r[1] - max_n) * r[2].cost,
                            reverse=True):
        w = math.exp(n - max_n) / total_w
        wcost = math.exp(n - max_n) * m.cost
        print(f"{idx:>3} {n:>4} {m.cost:>6.2f} {m.hpwl_gap:>5.2f} {m.area_gap:>5.2f} "
              f"{m.violations_relative:>5.2f} {m.boundary_violations:>4} "
              f"{m.grouping_violations:>4} {m.mib_violations:>3} "
              f"{'Y' if m.is_feasible else 'N':>4} {w*100:>5.1f} {wcost:>7.3f}")
        weighted_costs.append(m.cost)
        block_counts.append(n)

    feas = sum(1 for _, _, m in rows if m.is_feasible)
    tot_bnd = sum(m.boundary_violations for _, _, m in rows)
    tot_grp = sum(m.grouping_violations for _, _, m in rows)
    tot_mib = sum(m.mib_violations for _, _, m in rows)
    print(f"\nFeasible: {feas}/{len(rows)}   "
          f"viol totals: boundary={tot_bnd} grouping={tot_grp} mib={tot_mib}")
    score = ev.compute_total_score(weighted_costs, block_counts)
    print(f"Weighted total score (this subset): {score:.4f}")


if __name__ == '__main__':
    args = sys.argv[1:]
    if not args:
        ids = list(range(95, 100))
    elif args == ['all']:
        ids = list(range(100))
    else:
        ids = [int(a) for a in args]
    main(ids)
