"""
End-to-end comparison:  GRID+GREEDY  vs  GNN JOINT PLACER,  both feeding the
SAME skyline legalize tail in RLSkylineOptimizer.solve().

Where "grid+greedy" lives: it is RLSkylineOptimizer._rl_centers() — the CNN+RL
rollout that places blocks one-at-a-time on a 64x64 grid and returns
(cx, cy, aspect). The JointPlacer returns the exact same triple, so we feed it
through solve(centers_override=...) — i.e. the legalize/finish tail is identical
for both; only the center/shape SOURCE differs.

For each case it reports, BEFORE and AFTER legalize, the official-cost
decomposition, so you can see how much of the joint placer's tight Area_gap
survives legalization. Also draws a 2x3 figure (rows = method, cols = pre-legalize
centers / post-legalize / ground truth) for one case.

Run:
    cd FloorSet/iccad2026contest
    python3 CNN_RL/gnn_joint_placer/end2end.py --n 30 --viz-case 0
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

_HERE = Path(__file__).resolve().parent
_CNN_RL = _HERE.parent
_CONTEST = _CNN_RL.parent
for p in (str(_HERE), str(_CNN_RL), str(_CONTEST)):
    if p not in sys.path:
        sys.path.insert(0, p)

from iccad2026_evaluate import FloorplanDatasetLiteTest, calculate_bbox_area  # noqa: E402
from rl_skyline_optimizer import RLSkylineOptimizer  # noqa: E402
from model import JointPlacer  # noqa: E402
from diff_cost import diff_cost  # noqa: E402


def _free(constraints, i):
    c = constraints[i]
    return (float(c[0]) == 0 and float(c[1]) == 0 and int(c[2]) == 0
            and int(c[3]) == 0)


def joint_centers(model, cs):
    """Run JointPlacer -> (cx, cy, aspect_choice) in the form solve() expects."""
    a, c, b2b, bc = cs["a"], cs["c"], cs["b2b"], cs["bc"]
    scale = JointPlacer.canvas_scale(a, bc)
    with torch.no_grad():
        raw = model(a, c, b2b, bc)
        pos = model.positions(raw, a, bc, scale)        # [N,4] x,y,w,h
    cx = (pos[:, 0] + pos[:, 2] / 2).numpy()
    cy = (pos[:, 1] + pos[:, 3] / 2).numpy()
    aspect_choice = {}
    for i in range(bc):
        if _free(c, i):
            w, h = float(pos[i, 2]), float(pos[i, 3])
            aspect_choice[i] = max(w / max(h, 1e-6), 1e-6)
    return cx, cy, aspect_choice, [tuple(map(float, pos[i])) for i in range(bc)]


def parts_of(positions, cs):
    P = torch.tensor(positions, dtype=torch.float32)
    cost, parts = diff_cost(P, cs["b2b"], cs["p2b"], cs["pins"],
                            cs["a"][:cs["bc"]], cs["metrics"], return_parts=True)
    return float(cost), parts


def draw(ax, pos, title, color):
    xmin = min(p[0] for p in pos); ymin = min(p[1] for p in pos)
    xmax = max(p[0] + p[2] for p in pos); ymax = max(p[1] + p[3] for p in pos)
    for (x, y, w, h) in pos:
        ax.add_patch(Rectangle((x, y), w, h, facecolor=color, edgecolor="black",
                               linewidth=0.4, alpha=0.55))
    pad = 0.03 * max(xmax - xmin, ymax - ymin, 1e-6)
    ax.set_xlim(xmin - pad, xmax + pad); ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_aspect("equal"); ax.set_title(title, fontsize=9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--ckpt", type=str, default=str(_HERE / "joint_placer_poc.pt"))
    ap.add_argument("--data-path", type=str, default="../")
    ap.add_argument("--viz-cases", type=str, default="0,99",
                    help="comma-separated case ids to render figures for")
    args = ap.parse_args()

    ds = FloorplanDatasetLiteTest(args.data_path)
    model = JointPlacer()
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu")["model"])
    model.eval()
    opt = RLSkylineOptimizer(verbose=False)

    agg = {m: {"cost": 0.0, "hpwl_gap": 0.0, "area_gap": 0.0, "overlap": 0.0}
           for m in ("grid_pre", "grid_leg", "joint_pre", "joint_leg")}
    n = min(args.n, len(ds))
    joint_wins = 0
    print(f"{'case':>4} {'blk':>4} | {'grid_leg cost':>13} {'gap_A':>7} | "
          f"{'joint_leg cost':>14} {'gap_A':>7} | {'joint_pre gap_A':>15} {'win':>4}")
    print("-" * 86)

    for idx in range(n):
        s = ds[idx]
        a, b2b, p2b, pins, c = s["input"]
        bc = int((a != -1).sum())
        cs = dict(a=a, b2b=b2b, p2b=p2b, pins=pins, c=c, bc=bc, metrics=s["label"][1])
        otp = torch.full((bc, 4), -1.0)

        # GRID+GREEDY: normal solve (rollout -> legalize)
        grid_leg = opt.solve(bc, a, b2b, p2b, pins, c, otp)

        # JOINT: joint centers -> SAME legalize tail
        cx, cy, asp, joint_pre = joint_centers(model, cs)
        joint_leg = opt.solve(bc, a, b2b, p2b, pins, c, otp,
                              centers_override=(cx, cy, asp))

        gl_cost, gl_p = parts_of(grid_leg, cs)
        jl_cost, jl_p = parts_of(joint_leg, cs)
        _, jp_p = parts_of(joint_pre, cs)
        for k, src, p in (("grid_leg", grid_leg, gl_p), ("joint_leg", joint_leg, jl_p),
                          ("joint_pre", joint_pre, jp_p)):
            agg[k]["cost"] += parts_of(src, cs)[0]
            agg[k]["hpwl_gap"] += p["hpwl_gap"]; agg[k]["area_gap"] += p["area_gap"]
            agg[k]["overlap"] += p["overlap_viol"]

        win = "J" if jl_cost < gl_cost - 1e-9 else ""
        if win:
            joint_wins += 1
        print(f"{idx:>4} {bc:>4} | {gl_cost:>13.4f} {gl_p['area_gap']:>+7.3f} | "
              f"{jl_cost:>14.4f} {jl_p['area_gap']:>+7.3f} | {jp_p['area_gap']:>+15.3f} "
              f"{win:>4}")

    print("-" * 86)
    for k in ("grid_leg", "joint_pre", "joint_leg"):
        d = agg[k]
        print(f"{k:>10}: cost={d['cost']/n:.4f}  hpwl_gap={d['hpwl_gap']/n:+.3f}  "
              f"area_gap={d['area_gap']/n:+.3f}  overlap={d['overlap']/n:.4f}")
    print(f"\njoint_leg beats grid_leg on {joint_wins}/{n} cases "
          f"({100*joint_wins/n:.0f}%)")

    # ---- figures, one per requested case --------------------------------
    for vc in [int(x) for x in args.viz_cases.split(",") if x.strip() != ""]:
        s = ds[vc]
        a, b2b, p2b, pins, c = s["input"]; bc = int((a != -1).sum())
        cs = dict(a=a, b2b=b2b, p2b=p2b, pins=pins, c=c, bc=bc, metrics=s["label"][1])
        otp = torch.full((bc, 4), -1.0)
        poly = s["label"][0]
        gt = []
        for i in range(bc):
            v = poly[i][poly[i][:, 0] != -1]
            if len(v):
                x0, y0 = v.min(0).values; x1, y1 = v.max(0).values
                gt.append((float(x0), float(y0), float(x1 - x0), float(y1 - y0)))

        grid_leg = opt.solve(bc, a, b2b, p2b, pins, c, otp)
        cx, cy, asp, joint_pre = joint_centers(model, cs)
        joint_leg = opt.solve(bc, a, b2b, p2b, pins, c, otp, centers_override=(cx, cy, asp))

        def gap(pos):
            return (calculate_bbox_area(pos) - float(cs["metrics"][0])) / float(cs["metrics"][0])

        fig, ax = plt.subplots(2, 2, figsize=(11, 10))
        draw(ax[0, 0], joint_pre, f"JOINT placer output (pre-legalize)\n"
             f"gap_A={gap(joint_pre):+.3f}  (has overlap)", "#4C9AFF")
        draw(ax[0, 1], joint_leg, f"JOINT + legalize tail\n"
             f"gap_A={gap(joint_leg):+.3f}  cost={parts_of(joint_leg, cs)[0]:.3f}", "#36B37E")
        draw(ax[1, 0], grid_leg, f"GRID+GREEDY + legalize (current pipeline)\n"
             f"gap_A={gap(grid_leg):+.3f}  cost={parts_of(grid_leg, cs)[0]:.3f}", "#FFAB00")
        draw(ax[1, 1], gt, "Ground truth", "#FF8B00")
        fig.suptitle(f"case {vc} | {bc} blocks | JOINT vs GRID+GREEDY "
                     f"(same legalize tail)", fontsize=12)
        fig.tight_layout()
        out = _HERE / f"end2end_case{vc}.png"
        fig.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"saved figure -> {out}")


if __name__ == "__main__":
    main()
