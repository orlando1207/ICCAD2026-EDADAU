"""
Official Total Score for both pipelines, computed the contest way:

    Total Score = Σ Cost[i]·e^{n_i/12} / Σ e^{n_j/12}

where Cost[i] is the OFFICIAL per-case cost from evaluate_solution (feasibility
checks, exact violation counting, M=10 infeasible penalty) — NOT the
differentiable surrogate. RuntimeFactor is held at 1.0 (local convention).

Compares:
  * grid+greedy + legalize  (current RLSkylineOptimizer pipeline)
  * JOINT placer + the SAME legalize tail (centers_override)

Run:
    cd FloorSet/iccad2026contest
    python3 CNN_RL/gnn_joint_placer/total_score.py --n 100
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_CNN_RL = _HERE.parent
_CONTEST = _CNN_RL.parent
for p in (str(_HERE), str(_CNN_RL), str(_CONTEST)):
    if p not in sys.path:
        sys.path.insert(0, p)

from iccad2026_evaluate import (  # noqa: E402
    FloorplanDatasetLiteTest, evaluate_solution, compute_total_score,
    calculate_hpwl_b2b, calculate_hpwl_p2b,
)
from rl_skyline_optimizer import RLSkylineOptimizer  # noqa: E402
from model import JointPlacer  # noqa: E402


def _free(c, i):
    return (float(c[i][0]) == 0 and float(c[i][1]) == 0 and int(c[i][2]) == 0
            and int(c[i][3]) == 0)


def joint_centers(model, a, c, b2b, bc):
    scale = JointPlacer.canvas_scale(a, bc)
    with torch.no_grad():
        pos = model.positions(model(a, c, b2b, bc), a, bc, scale)
    cx = (pos[:, 0] + pos[:, 2] / 2).numpy()
    cy = (pos[:, 1] + pos[:, 3] / 2).numpy()
    asp = {i: max(float(pos[i, 2]) / max(float(pos[i, 3]), 1e-6), 1e-6)
           for i in range(bc) if _free(c, i)}
    return cx, cy, asp


def opt_target_pos(constraints, gt, bc):
    """Replicate the evaluator's opt_target_pos: fixed blocks get (w,h),
    preplaced get (x,y,w,h); everything else -1. Without this the solver never
    satisfies fixed/preplaced hard constraints -> every case is infeasible."""
    out = torch.full((bc, 4), -1.0)
    nc = constraints.shape[1] if constraints.dim() > 1 else 0
    for i in range(bc):
        is_fixed = nc > 0 and constraints[i, 0] != 0
        is_pre = nc > 1 and constraints[i, 1] != 0
        if is_pre:
            out[i] = torch.tensor(list(gt[i]))
        elif is_fixed:
            out[i, 2], out[i, 3] = gt[i][2], gt[i][3]
    return out


def baseline_of(gt, b2b, p2b, pins, metrics):
    hpwl = calculate_hpwl_b2b(gt, b2b) + calculate_hpwl_p2b(gt, p2b, pins)
    area = None
    if metrics is not None and len(metrics) >= 8:
        if metrics[0] > 0:
            area = float(metrics[0])
        if metrics[-2] > 0 and metrics[-1] >= 0:
            hpwl = float(metrics[-2]) + float(metrics[-1])
    from iccad2026_evaluate import calculate_bbox_area
    return {"hpwl_baseline": hpwl, "area_baseline": area or calculate_bbox_area(gt)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--ckpt", type=str, default=str(_HERE / "joint_placer_poc.pt"))
    ap.add_argument("--data-path", type=str, default="../")
    args = ap.parse_args()

    ds = FloorplanDatasetLiteTest(args.data_path)
    model = JointPlacer()
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu")["model"])
    model.eval()
    opt = RLSkylineOptimizer(verbose=False)

    n = min(args.n, len(ds))
    blk, grid_costs, joint_costs = [], [], []
    grid_infeas = joint_infeas = 0

    for idx in range(n):
        s = ds[idx]
        a, b2b, p2b, pins, c = s["input"]
        bc = int((a != -1).sum())
        poly = s["label"][0]; metrics = s["label"][1]
        gt = []
        for i in range(bc):
            v = poly[i][poly[i][:, 0] != -1]
            x0, y0 = v.min(0).values; x1, y1 = v.max(0).values
            gt.append((float(x0), float(y0), float(x1 - x0), float(y1 - y0)))
        base = baseline_of(gt, b2b, p2b, pins, metrics)
        otp = opt_target_pos(c, gt, bc)

        grid_pos = opt.solve(bc, a, b2b, p2b, pins, c, otp)
        cx, cy, asp = joint_centers(model, a, c, b2b, bc)
        joint_pos = opt.solve(bc, a, b2b, p2b, pins, c, otp,
                              centers_override=(cx, cy, asp))

        gm = evaluate_solution({"positions": grid_pos, "runtime": 1.0}, base, c,
                               b2b, p2b, pins, a, gt, median_runtime=1.0)
        jm = evaluate_solution({"positions": joint_pos, "runtime": 1.0}, base, c,
                               b2b, p2b, pins, a, gt, median_runtime=1.0)
        blk.append(bc); grid_costs.append(gm.cost); joint_costs.append(jm.cost)
        grid_infeas += (not gm.is_feasible); joint_infeas += (not jm.is_feasible)

    gs = compute_total_score(grid_costs, blk)
    js = compute_total_score(joint_costs, blk)
    print(f"cases: {n}  (RuntimeFactor=1.0)\n")
    print(f"{'pipeline':<34} {'TotalScore':>10} {'meanCost':>9} {'infeasible':>11}")
    print("-" * 68)
    print(f"{'grid+greedy + legalize':<34} {gs:>10.4f} "
          f"{sum(grid_costs)/n:>9.4f} {grid_infeas:>8}/{n}")
    print(f"{'JOINT + same legalize tail':<34} {js:>10.4f} "
          f"{sum(joint_costs)/n:>9.4f} {joint_infeas:>8}/{n}")
    print(f"\n(lower TotalScore is better)")


if __name__ == "__main__":
    main()
