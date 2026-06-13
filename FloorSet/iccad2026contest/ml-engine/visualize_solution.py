"""
Visualize our floorplanner's solution next to the ground-truth layout.

Reads a solutions JSON saved by gt_dims_harness.py (--save) and renders, per
case, our layout vs the GT layout side by side. Blocks are colored by constraint
type (preplaced / fixed-shape / cluster / boundary / free) so constraint handling
is visible. Saves one PNG per case.

Usage (from iccad2026contest/):
    # 1) produce solutions
    python ml-engine/gt_dims_harness.py all --budget 20 --starts 12 --save sols.json
    # 2) render some cases (ids, or 'all')
    python ml-engine/visualize_solution.py sols.json 99 95 50
    python ml-engine/visualize_solution.py sols.json all --out viz/
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))        # iccad2026contest
sys.path.insert(0, str(HERE.parent.parent))  # FloorSet root

from litetestLoader import FloorplanDatasetLiteTest
import iccad2026_evaluate as ev


def _gt_positions(labels, n):
    polygons, _ = labels
    pos = []
    for i in range(n):
        blk = polygons[i]
        valid = blk[blk[:, 0] != -1]
        x0, y0 = valid.min(dim=0).values
        x1, y1 = valid.max(dim=0).values
        pos.append((float(x0), float(y0), float(x1 - x0), float(y1 - y0)))
    return pos


def _facecolor(code_row):
    fixed, prep, mib, clu, bnd = [int(x) for x in code_row[:5]]
    if prep:
        return "#d62728"          # red: preplaced (hard pos+dim)
    if fixed:
        return "#ff7f0e"          # orange: fixed-shape (hard dim)
    if clu:
        return plt.cm.tab20(clu % 20)  # cluster: per-group color
    if bnd:
        return "#2ca02c"          # green: boundary
    return "#c7d0d9"              # grey: free


def _draw(ax, positions, constraints, title):
    for i, (x, y, w, h) in enumerate(positions):
        fc = _facecolor(constraints[i])
        ax.add_patch(mpatches.Rectangle((x, y), w, h, facecolor=fc,
                                        edgecolor="black", lw=0.4, alpha=0.85))
    xs = [p[0] for p in positions] + [p[0] + p[2] for p in positions]
    ys = [p[1] for p in positions] + [p[1] + p[3] for p in positions]
    ax.set_xlim(min(xs), max(xs)); ax.set_ylim(min(ys), max(ys))
    ax.set_aspect("equal"); ax.set_title(title, fontsize=10)


def main(sols_path, ids, out_dir):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    with open(sols_path) as f:
        data = json.load(f)
    sols = {s["test_id"]: s["positions"] for s in data["solutions"]}

    ds = FloorplanDatasetLiteTest("../")
    evr = ev.ContestEvaluator("../", verbose=False); evr._load_dataset()

    if ids == ["all"]:
        ids = sorted(sols.keys())
    else:
        ids = [int(i) for i in ids]

    legend = [
        mpatches.Patch(color="#d62728", label="preplaced"),
        mpatches.Patch(color="#ff7f0e", label="fixed-shape"),
        mpatches.Patch(color="#1f77b4", label="cluster"),
        mpatches.Patch(color="#2ca02c", label="boundary"),
        mpatches.Patch(color="#c7d0d9", label="free"),
    ]
    for idx in ids:
        sample = ds[idx]
        inputs, labels = sample["input"], sample["label"]
        area_target, b2b, p2b, pins_pos, constraints = inputs
        n = int((area_target != -1).sum().item())
        baseline, _ = evr._extract_baseline(idx, labels, b2b, p2b, pins_pos, n)
        cons = constraints.numpy()
        ours = sols[idx]
        gt = _gt_positions(labels, n)

        m = ev.evaluate_solution({"positions": [tuple(p) for p in ours], "runtime": 1.0},
                                 baseline, constraints, b2b, p2b, pins_pos,
                                 area_target, gt, median_runtime=1.0)

        fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 6.5))
        _draw(a1, ours, cons,
              f"OURS  n={n}  cost={m.cost:.2f}  hgap={m.hpwl_gap:.2f} "
              f"agap={m.area_gap:.2f} bnd={m.boundary_violations} "
              f"grp={m.grouping_violations} feas={'Y' if m.is_feasible else 'N'}")
        _draw(a2, gt, cons, f"GROUND TRUTH  n={n}")
        fig.legend(handles=legend, loc="lower center", ncol=5, fontsize=8)
        fig.suptitle(f"Case {idx}", fontsize=12)
        fig.tight_layout(rect=[0, 0.04, 1, 1])
        outp = Path(out_dir) / f"case_{idx:03d}_n{n}.png"
        fig.savefig(outp, dpi=130); plt.close(fig)
        print(f"  wrote {outp}  (cost {m.cost:.2f})")


if __name__ == "__main__":
    args = sys.argv[1:]
    out = "ml-engine/viz"
    if "--out" in args:
        k = args.index("--out"); out = args[k + 1]; del args[k:k + 2]
    if not args:
        print("usage: python ml-engine/visualize_solution.py SOLS.json <ids|all> [--out DIR]")
        sys.exit(1)
    main(args[0], args[1:] or ["all"], out)
