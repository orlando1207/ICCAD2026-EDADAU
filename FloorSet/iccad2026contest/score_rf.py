"""Score a results JSON from the official evaluator against REAL per-case medians.

The official evaluator can only score locally with RuntimeFactor = 1.0 (it has no
cross-submission medians), so `Total Score` in its output is runtime-neutral and
systematically pessimistic for a fast submission. This script re-applies the
runtime term using the alpha-test medians in `C_Median.csv`.

    # 1. produce a results JSON with the official evaluator
    python iccad2026_evaluate.py --evaluate floordiff_optimizer.py -o run.json

    # 2. score it with the real runtime factor
    python score_rf.py run.json                     # default: medians x 0.75
    python score_rf.py run.json --median-scale 1.0  # alpha-test medians as-is
    python score_rf.py a.json b.json --per-case     # compare runs, per-case table

`--median-scale` models the rest of the field getting faster for the final
submission: every median is multiplied by it before RuntimeFactor is computed, so
0.75 assumes every other participant shaved 25% off their alpha-test runtime.
Smaller = more pessimistic for us.

The per-case cost is the official formula verbatim (see `compute_cost` in
`iccad2026_evaluate.py`), including the `max(0, gap)` clamps, the `max(0.01, RF)`
guard, and the `M - 1e-6` feasible cap:

    cost = min((1 + 0.5*(max(0,HPWL_gap) + max(0,Area_gap)))
               * exp(2*V_rel) * max(0.7, max(0.01, RF)**0.3),  10 - 1e-6)
         = 10.0                                                if infeasible

    RF = runtime / (median_runtime * median_scale)

and the total is the weight-exp((n - n_max)/12) average from `compute_total_score`.
"""

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

ALPHA, BETA, GAMMA, M_PENALTY = 0.5, 2.0, 0.3, 10.0
RF_AT_FLOOR = 0.7 ** (1 / GAMMA)      # RF at/below which the 0.7 floor binds
HERE = Path(__file__).resolve().parent


def load_medians(path, scale):
    """{test_id: effective_median_seconds}"""
    out = {}
    with open(path) as fh:
        for row in csv.DictReader(fh):
            out[int(row['test_id'])] = float(row['median_runtime_s']) * scale
    return out


def cost_of(rec, median, neutral=False):
    """Official per-case cost. `neutral=True` reproduces the evaluator's RF=1."""
    if not rec.get('is_feasible', False):
        return M_PENALTY, float('nan'), M_PENALTY
    quality = 1 + ALPHA * (max(0.0, rec['hpwl_gap']) + max(0.0, rec['area_gap']))
    violation = math.exp(BETA * rec['violations_relative'])
    rf = rec['runtime_seconds'] / max(median, 1e-9)
    adj = 1.0 if neutral else max(0.7, math.pow(max(0.01, rf), GAMMA))
    return min(quality * violation * adj, M_PENALTY - 1e-6), rf, quality * violation


def score_run(path, medians):
    recs = json.load(open(path))['test_results']
    n_max = max(r['block_count'] for r in recs)
    rows = []
    for r in recs:
        med = medians.get(r['test_id'])
        if med is None:
            raise SystemExit(f"{path}: no median for test_id {r['test_id']}")
        cost, rf, qv = cost_of(r, med)
        neutral, _, _ = cost_of(r, med, neutral=True)
        rows.append({
            'test_id': r['test_id'], 'n': r['block_count'],
            'w': math.exp((r['block_count'] - n_max) / 12),
            'cost': cost, 'neutral': neutral, 'rf': rf, 'quality_viol': qv,
            'rt': r['runtime_seconds'], 'median': med,
            'feasible': r.get('is_feasible', False),
            'hpwl': r['hpwl_gap'], 'area': r['area_gap'],
            'vrel': r['violations_relative'],
        })
    return rows


def weighted(rows, key):
    tw = sum(r['w'] for r in rows)
    return sum(r['w'] * r[key] for r in rows) / tw if tw else 0.0


def report(name, rows, per_case=False):
    n = len(rows)
    rts = [r['rt'] for r in rows]
    at_floor = sum(1 for r in rows if r['feasible'] and r['rf'] <= RF_AT_FLOOR)
    slower = sum(1 for r in rows if r['feasible'] and r['rf'] > 1.0)
    infeas = [r['test_id'] for r in rows if not r['feasible']]

    print(f"\n{'=' * 78}\n{name}\n{'=' * 78}")
    print(f"  FINAL SCORE (with runtime factor) : {weighted(rows, 'cost'):.4f}")
    print(f"  runtime-neutral score             : {weighted(rows, 'neutral'):.4f}"
          f"   <- what the official evaluator prints")
    print(f"  feasible                          : {n - len(infeas)}/{n}"
          + (f"   INFEASIBLE: {infeas}" if infeas else ""))
    print(f"  runtime  avg {statistics.mean(rts):.2f}s  median "
          f"{statistics.median(rts):.2f}s  max {max(rts):.2f}s")
    rfs = sorted(r['rf'] for r in rows if r['feasible'])
    if rfs:
        print(f"  RuntimeFactor  min {rfs[0]:.3f}  median {statistics.median(rfs):.3f}"
              f"  max {rfs[-1]:.3f}")
        print(f"  at the 0.7 floor (RF <= {RF_AT_FLOOR:.4f}) : {at_floor}/{n}"
              f"      slower than the field : {slower}/{n}")
    print(f"  weighted gaps   V_rel {weighted(rows, 'vrel'):.4f}   "
          f"area {weighted(rows, 'area'):+.4f}   hpwl {weighted(rows, 'hpwl'):+.4f}")

    # headroom: score if every case sat exactly at the floor
    floor_rows = [dict(r, cost=min(r['quality_viol'] * 0.7, M_PENALTY - 1e-6))
                  if r['feasible'] else r for r in rows]
    print(f"  if every case hit the 0.7 floor   : {weighted(floor_rows, 'cost'):.4f}"
          f"   (runtime headroom left: "
          f"{weighted(rows, 'cost') - weighted(floor_rows, 'cost'):+.4f})")

    worst = sorted(rows, key=lambda r: -r['w'] * (r['cost'] - 0.7) / sum(x['w'] for x in rows))
    print(f"\n  top cases by weighted excess over the 0.7 floor:")
    print(f"  {'id':>4}{'n':>5}{'cost':>8}{'RF':>7}{'rt':>7}{'med':>7}"
          f"{'V_rel':>7}{'area':>8}{'hpwl':>8}{'wt%':>7}")
    for r in worst[:10]:
        print(f"  {r['test_id']:>4}{r['n']:>5}{r['cost']:>8.4f}{r['rf']:>7.3f}"
              f"{r['rt']:>7.2f}{r['median']:>7.2f}{r['vrel']:>7.3f}"
              f"{r['area']:>+8.4f}{r['hpwl']:>+8.4f}"
              f"{100 * r['w'] / sum(x['w'] for x in rows):>6.2f}%")

    if per_case:
        print(f"\n  {'id':>4}{'n':>5}{'feas':>6}{'cost':>8}{'neutral':>9}{'RF':>7}"
              f"{'rt':>7}{'med':>7}{'V_rel':>7}{'area':>8}{'hpwl':>8}")
        for r in rows:
            print(f"  {r['test_id']:>4}{r['n']:>5}{str(r['feasible']):>6}"
                  f"{r['cost']:>8.4f}{r['neutral']:>9.4f}{r['rf']:>7.3f}"
                  f"{r['rt']:>7.2f}{r['median']:>7.2f}{r['vrel']:>7.3f}"
                  f"{r['area']:>+8.4f}{r['hpwl']:>+8.4f}")


def main():
    ap = argparse.ArgumentParser(
        description='Score evaluator results JSONs with the real runtime factor.')
    ap.add_argument('results', nargs='+', help='results JSON from --evaluate')
    ap.add_argument('--median-csv', default=str(HERE / 'C_Median.csv'))
    ap.add_argument('--median-scale', type=float, default=0.75,
                    help='multiply every median by this (default 0.75: assume the '
                         'field is 25%% faster than in alpha testing)')
    ap.add_argument('--per-case', action='store_true')
    args = ap.parse_args()

    medians = load_medians(args.median_csv, args.median_scale)
    print(f"medians: {args.median_csv}  x {args.median_scale}  "
          f"(effective avg {statistics.mean(medians.values()):.2f}s, "
          f"range {min(medians.values()):.2f}-{max(medians.values()):.2f}s)")
    print(f"the 0.7 floor binds at runtime <= {100 * RF_AT_FLOOR:.1f}% of the "
          f"effective median")

    runs = []
    for p in args.results:
        rows = score_run(p, medians)
        report(Path(p).name, rows, args.per_case)
        runs.append((Path(p).name, rows))

    if len(runs) > 1:
        print(f"\n{'=' * 78}\nCOMPARISON\n{'=' * 78}")
        print(f"  {'run':<34}{'FINAL':>9}{'neutral':>9}{'rt avg':>9}{'floor':>8}")
        base = None
        for name, rows in runs:
            fin = weighted(rows, 'cost')
            d = '' if base is None else f'  ({fin - base:+.4f})'
            base = fin if base is None else base
            print(f"  {name:<34}{fin:>9.4f}{weighted(rows, 'neutral'):>9.4f}"
                  f"{statistics.mean(r['rt'] for r in rows):>8.2f}s"
                  f"{sum(1 for r in rows if r['rf'] <= RF_AT_FLOOR):>6}/"
                  f"{len(rows)}{d}")


if __name__ == '__main__':
    main()
