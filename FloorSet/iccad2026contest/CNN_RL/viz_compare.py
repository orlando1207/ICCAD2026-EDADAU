"""
Visual comparison for one case:  DRL raw rollout  vs  legalized  vs  ground truth.

Three panels, same axis scale:
  (1) DRL prediction : the raw per-block placement the RL policy rolls out
                       (env.positions BEFORE skyline legalization; may overlap).
  (2) Legalized      : RLSkylineOptimizer.solve() final output (overlap-free).
  (3) Ground truth   : the dataset's reference solution.

Each panel prints bbox area and (for 1,2) the overlap-pair count, so you can see
what legalization buys. Reuses the exact data-loading / optimizer path from
poc_slack_shaping.py so the picture matches a real --evaluate run.

Run:
    cd FloorSet/iccad2026contest
    python3 CNN_RL/viz_compare.py --case 0 --out viz_case0.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

_CONTEST_DIR = Path(__file__).resolve().parent.parent
if str(_CONTEST_DIR) not in sys.path:
    sys.path.insert(0, str(_CONTEST_DIR))
_DL_DIR = Path(__file__).resolve().parent
if str(_DL_DIR) not in sys.path:
    sys.path.insert(0, str(_DL_DIR))

from iccad2026_evaluate import (  # noqa: E402
    FloorplanDatasetLiteTest, calculate_bbox_area, evaluate_solution,
    calculate_hpwl_b2b, calculate_hpwl_p2b,
)
from rl_skyline_optimizer import RLSkylineOptimizer, ASPECT_BUCKETS  # noqa: E402


def _overlap_count(pos, tol=1e-4):
    n = len(pos)
    v = 0
    for i in range(n):
        xi, yi, wi, hi = pos[i]
        for j in range(i + 1, n):
            xj, yj, wj, hj = pos[j]
            ox = min(xi + wi, xj + wj) - max(xi, xj)
            oy = min(yi + hi, yj + hj) - max(yi, yj)
            if ox > tol and oy > tol:
                v += 1
    return v


def _opt_target_pos(constraints, target_pos, block_count):
    out = torch.full((block_count, 4), -1.0)
    nc = constraints.shape[1] if constraints.dim() > 1 else 0
    for i in range(block_count):
        is_fixed = nc > 0 and constraints[i, 0] != 0
        is_pre = nc > 1 and constraints[i, 1] != 0
        if is_pre:
            tx, ty, tw, th = target_pos[i]
            out[i] = torch.tensor([tx, ty, tw, th])
        elif is_fixed:
            _, _, tw, th = target_pos[i]
            out[i, 2] = tw
            out[i, 3] = th
    return out


def _drl_raw_positions(opt, area_t, b2b, p2b, pins, constraints, otp, block_count):
    """Replay the RL rollout and return env.positions BEFORE legalization."""
    from placement_env import PlacementEnv
    from canvas_raster import rasterize_env, CH_FEASIBILITY
    if opt._model is None:
        return None
    gnn, policy, grid = opt._model
    env = PlacementEnv(grid=grid)
    env.reset(area_t, b2b, p2b, pins, constraints, torch.ones(8),
              target_positions=otp)
    node_emb, _ = gnn.encode_problem(area_t, constraints, b2b, block_count)
    done = len(env.order) == 0
    while not done:
        st = env._build_state()
        if st["current_block"] is None:
            break
        cur = int(st["current_block"])
        canvas = rasterize_env(env)
        mask = canvas[CH_FEASIBILITY]
        is_free = (opt.aspect_pass and cur not in env.fixed_dims
                   and cur not in env.mib_dims and int(constraints[cur, 3]) == 0)
        if is_free:
            a, aspect_idx, _lp, _v = policy.act_aspect(canvas, node_emb[cur], mask, greedy=True)
            aspect = ASPECT_BUCKETS[aspect_idx]
        else:
            a, _lp, _v = policy.act(canvas, node_emb[cur], mask, greedy=True)
            aspect = 1.0
        env.step(a, aspect=aspect)
    return [tuple(float(t) for t in env.positions[i]) for i in range(block_count)]


def _draw(ax, pos, title, color, base_area):
    xmin = min(p[0] for p in pos); ymin = min(p[1] for p in pos)
    xmax = max(p[0] + p[2] for p in pos); ymax = max(p[1] + p[3] for p in pos)
    for (x, y, w, h) in pos:
        ax.add_patch(Rectangle((x, y), w, h, facecolor=color, edgecolor="black",
                               linewidth=0.4, alpha=0.55))
    pad = 0.03 * max(xmax - xmin, ymax - ymin, 1e-6)
    ax.set_xlim(xmin - pad, xmax + pad)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_aspect("equal")
    bbox = (xmax - xmin) * (ymax - ymin)
    gap = (bbox - base_area) / max(base_area, 1e-6)
    ax.set_title(f"{title}\nbbox={bbox:.3g}  Area_gap={gap:+.3f}", fontsize=10)
    return bbox


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", type=int, default=0)
    ap.add_argument("--data-path", type=str, default="../")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    ds = FloorplanDatasetLiteTest(args.data_path)
    sample = ds[args.case]
    inputs, labels = sample["input"], sample["label"]
    area_t, b2b, p2b, pins, constraints = inputs
    block_count = int((area_t != -1).sum().item())
    polygons = labels[0]

    gt = []
    for i in range(block_count):
        blk = polygons[i]
        valid = blk[blk[:, 0] != -1]
        if len(valid) > 0:
            x0, y0 = valid.min(dim=0).values
            x1, y1 = valid.max(dim=0).values
            gt.append((float(x0), float(y0), float(x1 - x0), float(y1 - y0)))
        else:
            gt.append((-1.0, -1.0, -1.0, -1.0))
    base_area = calculate_bbox_area([g for g in gt if g[0] >= 0])

    # Baseline metrics for the official cost (RuntimeFactor neutral = 1.0, as in
    # local --evaluate). Prefer the dataset's stored metrics, like _extract_baseline.
    gt_valid = [g for g in gt if g[0] >= 0]
    hpwl_b = calculate_hpwl_b2b(gt_valid, b2b) + calculate_hpwl_p2b(gt_valid, p2b, pins)
    metrics_t = labels[1]
    if metrics_t is not None and len(metrics_t) >= 8:
        if metrics_t[0] > 0:
            base_area = float(metrics_t[0])
        if metrics_t[-2] > 0 and metrics_t[-1] >= 0:
            hpwl_b = float(metrics_t[-2]) + float(metrics_t[-1])
    baseline = {"hpwl_baseline": hpwl_b, "area_baseline": base_area}

    otp = _opt_target_pos(constraints, gt, block_count)
    opt = RLSkylineOptimizer(verbose=True)

    drl = _drl_raw_positions(opt, area_t, b2b, p2b, pins, constraints, otp, block_count)
    legal = opt.solve(block_count, area_t, b2b, p2b, pins, constraints, otp)

    # Official contest cost of the legalized layout (RuntimeFactor = 1.0).
    m = evaluate_solution({"positions": legal, "runtime": 1.0}, baseline,
                          constraints, b2b, p2b, pins, area_t, gt,
                          median_runtime=1.0)
    print(f"\nLegalized cost   = {m.cost:.4f}   (feasible={m.is_feasible})")
    print(f"  HPWL_gap       = {m.hpwl_gap:+.4f}")
    print(f"  Area_gap       = {m.area_gap:+.4f}")
    print(f"  V_relative     = {m.violations_relative:.4f}  "
          f"(soft viol {m.total_soft_violations}/{m.max_possible_violations})")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    if drl is not None:
        _draw(axes[0], drl, f"(1) DRL raw rollout  ovl={_overlap_count(drl)}",
              "#4C9AFF", base_area)
    else:
        axes[0].set_title("(1) DRL raw rollout\n(no model loaded)")
    _draw(axes[1], legal, f"(2) Legalized  ovl={_overlap_count(legal)}",
          "#36B37E", base_area)
    axes[1].set_title(axes[1].get_title() +
                      f"\ncontest cost={m.cost:.3f}  V_rel={m.violations_relative:.3f}"
                      f"  feasible={m.is_feasible}", fontsize=9)
    _draw(axes[2], [g for g in gt if g[0] >= 0], "(3) Ground truth",
          "#FF8B00", base_area)

    fig.suptitle(f"case {args.case}  |  {block_count} blocks", fontsize=12)
    fig.tight_layout()
    out = args.out or f"viz_case{args.case}.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"\nsaved -> {out}")
    print(f"base (GT) area = {base_area:.4g}")


if __name__ == "__main__":
    main()
