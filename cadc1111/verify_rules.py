"""Cross-check rules.py against the official evaluator on saved solutions.

Every number rules.py produces must match iccad2026_evaluate.evaluate_solution
exactly, otherwise a local "feasible" verdict means nothing.
"""
import json
import sys

import torch

import iccad2026_evaluate as off
import rules

DATA = sys.argv[2] if len(sys.argv) > 2 else "/home/lovet/ICCAD2026-EDADAU/FloorSet/"
RESULTS = sys.argv[1] if len(sys.argv) > 1 else "op_wrapper_results.json"

ev = off.ContestEvaluator(data_path=DATA, verbose=False)
ev._load_dataset()
saved = {t["test_id"]: t for t in json.load(open(RESULTS))["test_results"]}

bad = 0
rows = []
for idx in sorted(saved):
    t = saved[idx]
    if t.get("positions") is None:
        continue
    sample = ev.dataset[idx]
    area, b2b, p2b, pins, cons = sample["input"]
    k = int((area != -1).sum().item())
    base, tgt = ev._extract_baseline(idx, sample["label"], b2b, p2b, pins, k)
    pos = [tuple(r) for r in t["positions"]]

    m = off.evaluate_solution({"positions": pos, "runtime": 1.0}, base, cons,
                              b2b, p2b, pins, area, tgt, median_runtime=1.0)

    spec = rules.CaseSpec.from_evaluator(k, area, cons, tgt)
    v = rules.evaluate(pos, spec, b2b, p2b, pins,
                       base["hpwl_baseline"], base["area_baseline"], 1.0)

    checks = {
        "feasible": (m.is_feasible, v.feasible),
        "overlap": (m.overlap_violations, v.hard.n_overlap),
        "area_viol": (m.area_violations, v.hard.n_area),
        "dim_viol": (m.dimension_violations, v.hard.n_dimension),
        "boundary": (m.boundary_violations, v.soft.boundary),
        "grouping": (m.grouping_violations, v.soft.grouping),
        "mib": (m.mib_violations, v.soft.mib),
        "n_soft": (m.max_possible_violations, v.soft.n_soft),
    }
    approx = {
        "hpwl": (m.hpwl_total, v.hpwl),
        "bbox": (m.bbox_area, v.area),
        "v_rel": (m.violations_relative, v.soft.relative),
        "cost": (m.cost, v.cost),
    }
    diffs = [f"{n}: off={a} mine={b}" for n, (a, b) in checks.items() if a != b]
    diffs += [f"{n}: off={a:.9g} mine={b:.9g}" for n, (a, b) in approx.items()
              if abs(a - b) > 1e-6 * max(1.0, abs(a))]
    if diffs:
        bad += 1
        print(f"case {idx:3d} (n={k}) MISMATCH -> " + " | ".join(diffs))
    rows.append((idx, k, v.hard.overlap_margin, v.hard.area_margin,
                 v.hard.dim_margin, v.soft.relative, v.cost))

print(f"\n{len(rows)} cases compared, {bad} mismatches")
print("\nworst hard-constraint slack (smaller = closer to infeasible):")
print(f'{"case":>5}{"n":>5}{"overlap":>12}{"area":>12}{"dim":>12}{"V_rel":>9}{"cost":>8}')
for r in sorted(rows, key=lambda r: min(r[2], r[3], r[4]))[:10]:
    print(f"{r[0]:>5}{r[1]:>5}{r[2]:>12.3e}{r[3]:>12.3e}{r[4]:>12.3e}{r[5]:>9.3f}{r[6]:>8.3f}")
