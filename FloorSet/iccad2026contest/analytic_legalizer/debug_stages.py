"""
Run pipeline step by step on test cases, reporting overlaps at each stage.
Usage: python -m analytic_legalizer.debug_stages [test_ids...]
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
        for j in range(i+1, n):
            xi, yi, wi, hi = pos_list[i]
            xj, yj, wj, hj = pos_list[j]
            ox = min(xi+wi, xj+wj) - max(xi, xj)
            oy = min(yi+hi, yj+hj) - max(yi, yj)
            if ox > 1e-6 and oy > 1e-6:
                count += 1
                worst.append((i, j, xi, yi, wi, hi, xj, yj, wj, hj, ox, oy))
    return count, worst


def main():
    test_ids = [int(x) for x in sys.argv[1:]] if len(sys.argv) > 1 else [3, 7, 9]

    data_path = str(Path(__file__).parent.parent.parent / "LiteTensorDataTest")
    dataset = FloorplanDatasetLiteTest(data_path)

    for test_id in test_ids:
        sample = dataset[test_id]
        inputs, labels = sample['input'], sample['label']
        area_target, b2b_conn, p2b_conn, pins_pos, constraints = inputs
        n = int((area_target != -1).sum().item())
        opt_target_pos = torch.full((n, 4), -1.0)

        print(f"\n=== test {test_id} n={n} ===")

        # Steps 0-2
        blocks, mib_groups, cluster_groups = parse_and_init(n, area_target, constraints, opt_target_pos)
        super_blocks = prepack_clusters(blocks, cluster_groups)

        # Show cluster info
        for gid, members in cluster_groups.items():
            sb = super_blocks[gid]
            bcs = [blocks[m].boundary_code for m in members]
            print(f"  cluster {gid}: members={members} bcs={bcs} sb=({sb.w:.1f}x{sb.h:.1f}) offsets={[(round(dx,1),round(dy,1)) for dx,dy in sb.offsets]}")

        # Step 3
        block_cx, block_cy = analytic_place(blocks, super_blocks, cluster_groups, b2b_conn, p2b_conn, pins_pos)

        # Step 4
        hcg_succ, vcg_succ = build_topology(block_cx, block_cy, blocks, super_blocks, cluster_groups)

        # Step 5
        pos5 = longest_path_pack(hcg_succ, vcg_succ, blocks, super_blocks, cluster_groups)
        for _ in range(3):
            if not augment_topology(pos5, blocks, super_blocks, cluster_groups, hcg_succ, vcg_succ):
                break
            pos5 = longest_path_pack(hcg_succ, vcg_succ, blocks, super_blocks, cluster_groups)
        n5, ov5 = count_overlaps(pos5)
        print(f"  after step5 (longest-path): {n5} overlaps")
        for (i, j, xi, yi, wi, hi, xj, yj, wj, hj, ox, oy) in ov5[:4]:
            print(f"    [{i}]({xi:.1f},{yi:.1f},{wi:.1f},{hi:.1f}) vs [{j}]({xj:.1f},{yj:.1f},{wj:.1f},{hj:.1f})")

        # Show cluster member positions after step 5
        cluster_rep = {gid: members[0] for gid, members in cluster_groups.items()}
        for gid, members in cluster_groups.items():
            rep = cluster_rep[gid]
            rep_pos = pos5[rep]
            print(f"  cluster {gid} rep pos: ({rep_pos[0]:.1f},{rep_pos[1]:.1f})")
            for m in members:
                print(f"    [{m}] bc={blocks[m].boundary_code}: ({pos5[m][0]:.1f},{pos5[m][1]:.1f},{pos5[m][2]:.1f},{pos5[m][3]:.1f})")

        # Step 6
        pos6 = shape_soft_blocks(pos5, blocks, super_blocks, cluster_groups, mib_groups, area_target, hcg_succ, vcg_succ)
        for _ in range(4):
            pos6 = longest_path_pack(hcg_succ, vcg_succ, blocks, super_blocks, cluster_groups)
            if not augment_topology(pos6, blocks, super_blocks, cluster_groups, hcg_succ, vcg_succ):
                break
        n6, ov6 = count_overlaps(pos6)
        print(f"  after step6 (shaping+relegalize): {n6} overlaps")

        # Step 7
        pos7 = slide_boundary(pos6, blocks, super_blocks, cluster_groups)
        n7, ov7 = count_overlaps(pos7)
        print(f"  after step7 (slide): {n7} overlaps")
        for (i, j, xi, yi, wi, hi, xj, yj, wj, hj, ox, oy) in ov7[:4]:
            print(f"    [{i}]({xi:.1f},{yi:.1f},{wi:.1f},{hi:.1f}) vs [{j}]({xj:.1f},{yj:.1f},{wj:.1f},{hj:.1f})")

        # Step 8
        pos8 = enforce_hard(pos7, blocks, area_target)
        n8, ov8 = count_overlaps(pos8)
        print(f"  after step8 (enforce): {n8} overlaps")
        for (i, j, xi, yi, wi, hi, xj, yj, wj, hj, ox, oy) in ov8[:4]:
            print(f"    [{i}]({xi:.1f},{yi:.1f},{wi:.1f},{hi:.1f}) vs [{j}]({xj:.1f},{yj:.1f},{wj:.1f},{hj:.1f})")


if __name__ == "__main__":
    main()
