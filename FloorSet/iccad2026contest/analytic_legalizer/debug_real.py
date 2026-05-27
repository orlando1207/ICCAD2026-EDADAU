"""
Debug pipeline using REAL preplaced target positions (like the evaluator).
Usage: python -m analytic_legalizer.debug_real [test_ids...]
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch

from litetestLoader import FloorplanDatasetLiteTest
from analytic_legalizer.constraints import (
    parse_and_init, prepack_clusters, slide_boundary, enforce_hard
)
from analytic_legalizer.quadratic_placer import analytic_place
from analytic_legalizer.topology import build_topology, longest_path_pack, augment_topology
from analytic_legalizer.shaping import shape_soft_blocks


def count_overlaps(pos_list):
    n = len(pos_list)
    count = 0
    worst = []
    for i in range(n):
        for j in range(i + 1, n):
            xi, yi, wi, hi = pos_list[i]
            xj, yj, wj, hj = pos_list[j]
            ox = min(xi + wi, xj + wj) - max(xi, xj)
            oy = min(yi + hi, yj + hj) - max(yi, yj)
            if ox > 1e-6 and oy > 1e-6:
                count += 1
                worst.append((i, j, ox, oy))
    return count, worst


def build_target_pos(constraints, block_count, gt_positions):
    """Replicate evaluator's opt_target_pos construction."""
    opt_target_pos = torch.full((block_count, 4), -1.0)
    nc = constraints.shape[1] if constraints.dim() > 1 else 0
    for i in range(block_count):
        is_fixed = nc > 0 and constraints[i, 0] != 0
        is_preplaced = nc > 1 and constraints[i, 1] != 0
        if is_preplaced:
            tx, ty, tw, th = gt_positions[i]
            opt_target_pos[i] = torch.tensor([tx, ty, tw, th])
        elif is_fixed:
            _, _, tw, th = gt_positions[i]
            opt_target_pos[i, 2] = tw
            opt_target_pos[i, 3] = th
    return opt_target_pos


def gt_from_labels(labels, block_count):
    polygons, _ = labels
    positions = []
    for i in range(block_count):
        block = polygons[i]
        valid = block[block[:, 0] != -1]
        if len(valid) > 0:
            x_min, y_min = valid.min(dim=0).values
            x_max, y_max = valid.max(dim=0).values
            positions.append((float(x_min), float(y_min),
                              float(x_max - x_min), float(y_max - y_min)))
        else:
            positions.append((0, 0, 1, 1))
    return positions


def main():
    test_ids = [int(x) for x in sys.argv[1:]] if len(sys.argv) > 1 else [54]

    data_path = str(Path(__file__).parent.parent.parent / "LiteTensorDataTest")
    dataset = FloorplanDatasetLiteTest(data_path)

    for test_id in test_ids:
        sample = dataset[test_id]
        inputs, labels = sample['input'], sample['label']
        area_target, b2b_conn, p2b_conn, pins_pos, constraints = inputs
        n = int((area_target != -1).sum().item())
        gt = gt_from_labels(labels, n)
        opt_target_pos = build_target_pos(constraints, n, gt)

        print(f"\n=== test {test_id} n={n} ===")

        blocks, mib_groups, cluster_groups = parse_and_init(n, area_target, constraints, opt_target_pos)
        super_blocks = prepack_clusters(blocks, cluster_groups)

        # Report preplaced blocks
        for i, b in enumerate(blocks):
            if b.is_preplaced:
                print(f"  preplaced [{i}] cluster={b.cluster_group} fixed=({b.fixed_x:.1f},{b.fixed_y:.1f},{b.w:.1f},{b.h:.1f})")

        for gid, members in cluster_groups.items():
            sb = super_blocks[gid]
            pp = [m for m in members if blocks[m].is_preplaced]
            print(f"  cluster {gid}: members={members} sb=({sb.w:.1f}x{sb.h:.1f}) preplaced_members={pp}")
            if pp:
                print(f"    offsets={[(round(dx,1),round(dy,1)) for dx,dy in sb.offsets]}")

        block_cx, block_cy = analytic_place(blocks, super_blocks, cluster_groups, b2b_conn, p2b_conn, pins_pos)
        hcg_succ, vcg_succ = build_topology(block_cx, block_cy, blocks, super_blocks, cluster_groups)

        pos5 = longest_path_pack(hcg_succ, vcg_succ, blocks, super_blocks, cluster_groups)
        for _ in range(15):
            if not augment_topology(pos5, blocks, super_blocks, cluster_groups, hcg_succ, vcg_succ):
                break
            pos5 = longest_path_pack(hcg_succ, vcg_succ, blocks, super_blocks, cluster_groups)
        n5, ov5 = count_overlaps(pos5)
        print(f"  after step5: {n5} overlaps")
        for (i, j, ox, oy) in ov5[:8]:
            pi = "PP" if blocks[i].is_preplaced else ("C%d" % blocks[i].cluster_group if blocks[i].cluster_group else "  ")
            pj = "PP" if blocks[j].is_preplaced else ("C%d" % blocks[j].cluster_group if blocks[j].cluster_group else "  ")
            print(f"    [{i}{pi}]({pos5[i][0]:.1f},{pos5[i][1]:.1f},{pos5[i][2]:.1f},{pos5[i][3]:.1f}) vs [{j}{pj}]({pos5[j][0]:.1f},{pos5[j][1]:.1f},{pos5[j][2]:.1f},{pos5[j][3]:.1f}) ox={ox:.1f} oy={oy:.1f}")

        pos6 = shape_soft_blocks(pos5, blocks, super_blocks, cluster_groups, mib_groups, area_target, hcg_succ, vcg_succ)
        for _ in range(15):
            pos6 = longest_path_pack(hcg_succ, vcg_succ, blocks, super_blocks, cluster_groups)
            if not augment_topology(pos6, blocks, super_blocks, cluster_groups, hcg_succ, vcg_succ):
                break
        n6, ov6 = count_overlaps(pos6)
        print(f"  after step6: {n6} overlaps")

        pos7 = slide_boundary(pos6, blocks, super_blocks, cluster_groups)
        n7, ov7 = count_overlaps(pos7)
        print(f"  after step7 (slide): {n7} overlaps")

        pos8 = enforce_hard(pos7, blocks, area_target)
        n8, ov8 = count_overlaps(pos8)
        print(f"  after step8 (enforce): {n8} overlaps")

        # Report preplaced dimension/position violations after step8
        dim_viol = 0
        for i, b in enumerate(blocks):
            if b.is_preplaced:
                x, y, w, h = pos8[i]
                if (abs(x - b.fixed_x) > 1e-3 or abs(y - b.fixed_y) > 1e-3 or
                        abs(w - b.w) > 1e-3 or abs(h - b.h) > 1e-3):
                    dim_viol += 1
                    print(f"    PREPLACED VIOL [{i}]: got ({x:.2f},{y:.2f},{w:.2f},{h:.2f}) want ({b.fixed_x:.2f},{b.fixed_y:.2f},{b.w:.2f},{b.h:.2f})")
            elif b.is_fixed_shape:
                x, y, w, h = pos8[i]
                if abs(w - b.w) > 1e-3 or abs(h - b.h) > 1e-3:
                    dim_viol += 1
                    print(f"    FIXED VIOL [{i}]: got wh=({w:.2f},{h:.2f}) want ({b.w:.2f},{b.h:.2f})")
        print(f"  dim_viol={dim_viol}  -> {'FEASIBLE' if (n8==0 and dim_viol==0) else 'INFEASIBLE'}")
        if ov8:
            print(f"  step8 overlaps involve: {sorted(set([x for o in ov8 for x in o[:2]]))}")


if __name__ == "__main__":
    main()
