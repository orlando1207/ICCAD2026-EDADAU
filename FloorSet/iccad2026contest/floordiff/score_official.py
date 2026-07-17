"""Score a predictions/legalized JSON with the OFFICIAL contest evaluator
(`iccad2026_evaluate.evaluate_solution` + `compute_total_score`).

Runtime factor is neutralized (runtime = median = 1.0 -> factor 1.0), so the
reported cost is quality x violations only — the part we control locally.

Run from iccad2026contest/:
  python -m floordiff.score_official --pred floordiff/out/legalized.json
  python -m floordiff.score_official --pred ... --cases 100,110,120
  python -m floordiff.score_official --gt-check          # GT reference costs
"""

import argparse
import json
from pathlib import Path

import torch

from iccad2026_evaluate import compute_total_score, evaluate_solution
from .data import VALIDATION_NS, gt_xywh, load_validation_case


def official_cost(xywh, case):
    positions = [tuple(map(float, r)) for r in xywh]
    tgt = [tuple(map(float, r)) for r in gt_xywh(case)]
    m = evaluate_solution(
        {'positions': positions, 'runtime': 1.0},
        {'hpwl_baseline': float(case['metrics'][6] + case['metrics'][7]),
         'area_baseline': float(case['metrics'][0])},
        case['cons'].float(), case['b2b'], case['p2b'], case['pins'],
        case['area'], target_positions=tgt, median_runtime=1.0)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred', type=str, help='predictions/legalized JSON')
    ap.add_argument('--gt-check', action='store_true')
    ap.add_argument('--cases', type=str, default='')
    args = ap.parse_args()

    preds = json.loads(Path(args.pred).read_text())['cases'] if args.pred else None
    if args.cases:
        ns = [int(x) for x in args.cases.split(',')]
    elif preds:
        ns = sorted(int(k) for k in preds)
    else:
        ns = VALIDATION_NS

    costs, blocks = [], []
    hdr = (f"{'n':>4} {'feas':>5} {'cost':>7} {'hpwl_gap':>9} {'area_gap':>9} "
           f"{'Vb':>3} {'Vg':>3} {'Vm':>3} {'Vrel':>6}")
    print(hdr)
    for nb in ns:
        case = load_validation_case(nb)
        xywh = gt_xywh(case) if args.gt_check else \
            torch.tensor(preds[str(nb)]['positions'], dtype=torch.float64)
        m = official_cost(xywh, case)
        costs.append(m.cost)
        blocks.append(nb)
        print(f"{nb:>4} {str(m.is_feasible):>5} {m.cost:>7.4f} {m.hpwl_gap:>9.4f} "
              f"{m.area_gap:>9.4f} {m.boundary_violations:>3d} "
              f"{m.grouping_violations:>3d} {m.mib_violations:>3d} "
              f"{m.violations_relative:>6.3f}")
    total = compute_total_score(costs, blocks)
    print('-' * len(hdr))
    print(f'cases: {len(ns)} | mean cost {sum(costs) / len(costs):.4f} | '
          f'TOTAL SCORE (exp-weighted over these cases): {total:.4f}')


if __name__ == '__main__':
    main()
