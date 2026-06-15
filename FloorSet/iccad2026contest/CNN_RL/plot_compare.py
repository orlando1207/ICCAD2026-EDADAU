"""Side-by-side comparison: ground truth (left) vs RLSkylineOptimizer output
(right), for a given test case.

Run:
    cd FloorSet/iccad2026contest
    python3 CNN_RL/plot_compare.py --test-id 0 --out CNN_RL/compare_case0.png
    python3 CNN_RL/plot_compare.py --test-id 0 --center-source quadratic --out CNN_RL/compare_case0_quad.png
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
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle
from shapely.geometry import Polygon
import torch

from visualize import get_hard_color
from litetestLoader import FloorplanDatasetLiteTest
from rl_skyline_optimizer import RLSkylineOptimizer


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


def _draw(ax, positions, b2b_conn, p2b_conn, pins_pos, constraints, title,
          show_pins=True, show_wires=True):
    all_poly = {}
    W = H = 0.0
    for i, (x, y, w, h) in enumerate(positions):
        poly = Polygon([(x, y), (x + w, y), (x + w, y + h), (x, y + h), (x, y)])
        all_poly[i] = poly
        face_color, label_text = get_hard_color(constraints[i])
        patch = mpatches.Polygon(
            list(poly.exterior.coords), closed=True, fill=True,
            edgecolor="black", facecolor=face_color, label=label_text, alpha=0.3,
        )
        ax.add_patch(patch)
        ax.annotate(str(i + 1), (x, y), fontsize=6)
        W = max(W, x + w)
        H = max(H, y + h)

    if show_pins:
        for pname in range(pins_pos.shape[0]):
            x, y = pins_pos[pname]
            ax.add_patch(Circle((x, y), radius=1, color="g"))

    if show_wires:
        for src, dst in b2b_conn[:, :2]:
            src, dst = int(src.item()), int(dst.item())
            if src != -1 and dst != -1:
                llx1, lly1 = all_poly[src].bounds[0], all_poly[src].bounds[1]
                llx2, lly2 = all_poly[dst].bounds[0], all_poly[dst].bounds[1]
                ax.plot((llx1, llx2), (lly1, lly2), color="r", linewidth=0.1)

        for src, dst in p2b_conn[:, :2]:
            src, dst = int(src.item()), int(dst.item())
            if src != -1 and dst != -1:
                llx2, lly2 = all_poly[dst].bounds[0], all_poly[dst].bounds[1]
                ax.plot((pins_pos[src][0], llx2), (pins_pos[src][1], lly2),
                        color="b", linewidth=0.1)

    ax.set_xlim(0, W * 1.25)
    ax.set_ylim(0, H * 1.25)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="upper right",
              title="Placement Constraints", fontsize=6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-id", type=int, default=0)
    ap.add_argument("--out", type=str, default="CNN_RL/compare_case0.png")
    ap.add_argument("--center-source", choices=["model", "quadratic"], default="model")
    ap.add_argument("--no-pins", action="store_true")
    ap.add_argument("--no-wires", action="store_true")
    args = ap.parse_args()

    dataset = FloorplanDatasetLiteTest(str(_CONTEST_DIR) + "/")
    sample = dataset[args.test_id]
    inputs, labels = sample["input"], sample["label"]
    area_target, b2b_conn, p2b_conn, pins_pos, constraints = inputs
    block_count = int((area_target != -1).sum().item())

    gt_positions = _ground_truth_positions(labels, block_count)

    opt_target_pos = torch.full((block_count, 4), -1.0)
    optimizer = RLSkylineOptimizer(verbose=True, center_source=args.center_source)
    rl_positions = optimizer.solve(
        block_count, area_target, b2b_conn, p2b_conn, pins_pos, constraints, opt_target_pos
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    _draw(axes[0], gt_positions, b2b_conn, p2b_conn, pins_pos, constraints,
          title=f"Ground Truth (case {args.test_id}, {block_count} blocks)",
          show_pins=not args.no_pins, show_wires=not args.no_wires)
    _draw(axes[1], rl_positions, b2b_conn, p2b_conn, pins_pos, constraints,
          title=f"RLSkyline ({args.center_source})",
          show_pins=not args.no_pins, show_wires=not args.no_wires)

    plt.tight_layout()
    plt.savefig(args.out, dpi=130)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
