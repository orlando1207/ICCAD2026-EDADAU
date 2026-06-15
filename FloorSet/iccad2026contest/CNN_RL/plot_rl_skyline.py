"""Visualize a placement produced by rl_skyline_optimizer.RLSkylineOptimizer,
optionally next to the dataset's ground-truth layout.

Run:
    cd FloorSet/iccad2026contest
    python3 CNN_RL/plot_rl_skyline.py --test-id 0 --out CNN_RL/rl_skyline_case0.png
    python3 CNN_RL/plot_rl_skyline.py --test-id 0 --ground-truth --out CNN_RL/gt_case0.png
"""

import argparse
import sys
from pathlib import Path

_CONTEST_DIR = Path(__file__).resolve().parent.parent
if str(_CONTEST_DIR) not in sys.path:
    sys.path.insert(0, str(_CONTEST_DIR))
_DL_DIR = Path(__file__).resolve().parent
if str(_DL_DIR) not in sys.path:
    sys.path.insert(0, str(_DL_DIR))
_FLOORSET_DIR = _CONTEST_DIR.parent
if str(_FLOORSET_DIR) not in sys.path:
    sys.path.insert(0, str(_FLOORSET_DIR))

import matplotlib
matplotlib.use("Agg")
import torch

from litetestLoader import FloorplanDatasetLiteTest
from rl_skyline_optimizer import RLSkylineOptimizer
from visualize import visualize_lite


def _ground_truth_positions(labels, block_count):
    """Derive (x, y, w, h) per block from the dataset's ground-truth polygons."""
    polygons, _metrics = labels
    positions = []
    for i in range(block_count):
        block = polygons[i]
        valid = block[block[:, 0] != -1]
        x_min, y_min = valid.min(dim=0).values
        x_max, y_max = valid.max(dim=0).values
        positions.append((float(x_min), float(y_min),
                          float(x_max - x_min), float(y_max - y_min)))
    return positions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-id", type=int, default=0)
    ap.add_argument("--out", type=str, default="CNN_RL/rl_skyline_case0.png")
    ap.add_argument("--center-source", choices=["model", "quadratic"], default="model")
    ap.add_argument("--ground-truth", action="store_true",
                     help="Plot the dataset ground-truth layout instead of the optimizer output")
    args = ap.parse_args()

    dataset = FloorplanDatasetLiteTest(str(_CONTEST_DIR) + "/")
    sample = dataset[args.test_id]
    inputs, labels = sample["input"], sample["label"]
    area_target, b2b_conn, p2b_conn, pins_pos, constraints = inputs
    block_count = int((area_target != -1).sum().item())

    if args.ground_truth:
        positions = _ground_truth_positions(labels, block_count)
        title_suffix = " (ground truth)"
    else:
        # Build target_positions like iccad2026_evaluate: fixed blocks keep (w,h),
        # preplaced keep (x,y,w,h) — else the plot square-izes fixed blocks and
        # won't match the actually-scored layout.
        gt = _ground_truth_positions(labels, block_count)
        opt_target_pos = torch.full((block_count, 4), -1.0)
        for i in range(block_count):
            if constraints[i, 1] != 0:                  # preplaced -> x,y,w,h
                opt_target_pos[i] = torch.tensor(gt[i])
            elif constraints[i, 0] != 0:                # fixed -> w,h only
                opt_target_pos[i, 2] = gt[i][2]
                opt_target_pos[i, 3] = gt[i][3]
        optimizer = RLSkylineOptimizer(verbose=True, center_source=args.center_source)
        positions = optimizer.solve(
            block_count, area_target, b2b_conn, p2b_conn, pins_pos, constraints, opt_target_pos
        )
        title_suffix = f" (RLSkyline, {args.center_source})"

    # visualize_lite expects fp_sol elements as (w, h, x, y)
    fp_sol = [(w, h, x, y) for (x, y, w, h) in positions]

    visualize_lite(
        fp_sol,
        b2b_conn,
        p2b_conn,
        pins_pos,
        constraints,
        lind=args.test_id,
    )

    import matplotlib.pyplot as plt
    plt.title(plt.gca().get_title() + title_suffix)
    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
