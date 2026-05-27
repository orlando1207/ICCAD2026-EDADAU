"""Show ground-truth cluster member layout + outline for a test."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
from litetestLoader import FloorplanDatasetLiteTest
from analytic_legalizer.constraints import parse_and_init


def gt_from_labels(labels, n):
    polygons, _ = labels
    pos = []
    for i in range(n):
        block = polygons[i]
        valid = block[block[:, 0] != -1]
        if len(valid) > 0:
            x0, y0 = valid.min(dim=0).values
            x1, y1 = valid.max(dim=0).values
            pos.append((float(x0), float(y0), float(x1 - x0), float(y1 - y0)))
        else:
            pos.append((0, 0, 1, 1))
    return pos


def main():
    test_ids = [int(x) for x in sys.argv[1:]] if len(sys.argv) > 1 else [54]
    data_path = str(Path(__file__).parent.parent.parent / "LiteTensorDataTest")
    dataset = FloorplanDatasetLiteTest(data_path)
    for test_id in test_ids:
        sample = dataset[test_id]
        inputs, labels = sample['input'], sample['label']
        area_target, b2b, p2b, pins, constraints = inputs
        n = int((area_target != -1).sum().item())
        gt = gt_from_labels(labels, n)
        # outline = bounding box of all blocks
        x1 = max(g[0] + g[2] for g in gt)
        y1 = max(g[1] + g[3] for g in gt)
        print(f"\n=== test {test_id} n={n} gt_outline≈{x1:.0f}x{y1:.0f} ===")
        opt_target_pos = torch.full((n, 4), -1.0)
        blocks, mib_groups, cluster_groups = parse_and_init(n, area_target, constraints, opt_target_pos)
        for gid, members in cluster_groups.items():
            print(f"  cluster {gid}: members={members}")
            for m in members:
                pp = "PP" if constraints[m, 1] != 0 else "  "
                bc = int(constraints[m, 4].item())
                gx, gy, gw, gh = gt[m]
                print(f"    {pp}[{m}] bc={bc} gt=({gx:.1f},{gy:.1f},{gw:.1f},{gh:.1f}) span x[{gx:.0f},{gx+gw:.0f}] y[{gy:.0f},{gy+gh:.0f}]")


if __name__ == "__main__":
    main()
