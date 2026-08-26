"""Guard must (a) never touch a legal layout, (b) always rescue a broken one."""
import argparse
import json
import sys

import numpy as np
import torch

import iccad2026_evaluate as off
import rules
from feasibility import enforce, shelf_pack, guarded_solve

DATA = "/home/lovet/ICCAD2026-EDADAU/FloorSet/"

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--results", default="validation_results.json",
                help="evaluator output for the VALIDATION set; a run against a "
                     "generated testkit writes op_wrapper_results.json in cwd "
                     "and must not be used here")
ap.add_argument("--data-path", default=DATA)
args = ap.parse_args()

ev = off.ContestEvaluator(data_path=args.data_path, verbose=False)
ev._load_dataset()
saved = {t["test_id"]: t for t in json.load(open(args.results))["test_results"]}

# The results file and the dataset must describe the same cases.  They silently
# will not if the file came from a `--data-path <testkit>` run, and the failure
# then shows up as a nonsense 600%-area assertion many lines later.
for _i, _t in saved.items():
    _k = int((ev.dataset[_i]["input"][0] != -1).sum().item())
    if _k != _t["block_count"]:
        sys.exit(f"{args.results} does not match {args.data_path}: test_id {_i} "
                 f"has {_t['block_count']} blocks there, {_k} here.  Re-run the "
                 f"evaluator on the validation set with "
                 f"--output {args.results}.")

rng = np.random.default_rng(0)
n_noop = n_rescued = n_failed = 0
fallback_costs, damage_costs = [], []

for idx in sorted(saved):
    sample = ev.dataset[idx]
    area, b2b, p2b, pins, cons = sample["input"]
    k = int((area != -1).sum().item())
    base, tgt = ev._extract_baseline(idx, sample["label"], b2b, p2b, pins, k)
    spec = rules.CaseSpec.from_evaluator(k, area, cons, tgt)
    good = np.array(saved[idx]["positions"], dtype=float)

    # (a) legal layout must come back byte-identical
    out, info = enforce(good, spec)
    assert info.stage == "clean", (idx, info)
    assert np.array_equal(out, good), idx
    n_noop += 1

    # (b) four independent kinds of damage, each must be rescued
    for kind in ("overlap", "preplaced", "fixed", "area"):
        bad = good.copy()
        if kind == "overlap":                       # collapse 20% of blocks
            hit = rng.choice(k, max(2, k // 5), replace=False)
            bad[hit, 0] = bad[0, 0]
            bad[hit, 1] = bad[0, 1]
        elif kind == "preplaced":
            m = np.nonzero(spec.preplaced_mask)[0]
            if len(m) == 0:
                continue
            bad[m, 0] += 3.0
        elif kind == "fixed":
            m = np.nonzero(spec.fixed_mask & ~spec.preplaced_mask)[0]
            if len(m) == 0:
                continue
            bad[m, 2] *= 1.4
        else:
            m = np.nonzero(spec.soft_mask)[0]
            bad[m, 2] *= 1.5                        # +50% area

        assert not rules.check_hard(bad, spec).feasible, (idx, kind)
        out, info = enforce(bad, spec)
        rep = rules.check_hard(out, spec)
        if rep.feasible:
            n_rescued += 1
        else:
            n_failed += 1
            print(f"case {idx} {kind}: NOT RESCUED -> {rep.summary()}")
        v = rules.evaluate(out, spec, b2b, p2b, pins,
                           base["hpwl_baseline"], base["area_baseline"], 1.0)
        damage_costs.append(v.cost)

    # cost of the pure fallback, for reference
    fb = shelf_pack(spec)
    rep = rules.check_hard(fb, spec)
    assert rep.feasible, (idx, rep.summary())
    v = rules.evaluate(fb, spec, b2b, p2b, pins,
                       base["hpwl_baseline"], base["area_baseline"], 1.0)
    fallback_costs.append(v.cost)

# exception path
class Boom(Exception):
    pass

sol = guarded_solve(lambda: (_ for _ in ()).throw(Boom("simulated crash")), spec)
assert rules.check_hard(sol, spec).feasible

print(f"\nno-op on legal layouts : {n_noop}/100")
print(f"damaged layouts rescued: {n_rescued}  failed: {n_failed}")
print(f"cost after rescue      : min {min(damage_costs):.3f} "
      f"median {np.median(damage_costs):.3f} max {max(damage_costs):.3f}")
print(f"cost of bare fallback  : min {min(fallback_costs):.3f} "
      f"median {np.median(fallback_costs):.3f} max {max(fallback_costs):.3f}  (vs 10.0 infeasible)")
