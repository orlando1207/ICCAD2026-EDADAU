"""
Diagnose remaining overlaps in pipeline output using actual test data.
Usage: python -m analytic_legalizer.debug_overlaps2 [test_ids...]
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import numpy as np

from litetestLoader import FloorplanDatasetLiteTest
from analytic_legalizer.optimizer import MyOptimizer
from analytic_legalizer.constraints import parse_and_init, prepack_clusters, BOUND_LEFT, BOUND_BOTTOM
from analytic_legalizer.quadratic_placer import analytic_place
from analytic_legalizer.topology import build_topology, longest_path_pack


def check_overlaps(pos_list):
    n = len(pos_list)
    overlaps = []
    for i in range(n):
        for j in range(i+1, n):
            xi, yi, wi, hi = pos_list[i]
            xj, yj, wj, hj = pos_list[j]
            ox = min(xi+wi, xj+wj) - max(xi, xj)
            oy = min(yi+hi, yj+hj) - max(yi, yj)
            if ox > 1e-6 and oy > 1e-6:
                overlaps.append((i, j, xi, yi, wi, hi, xj, yj, wj, hj, ox, oy))
    return overlaps


def main():
    test_ids = [int(x) for x in sys.argv[1:]] if len(sys.argv) > 1 else list(range(10))

    data_path = str(Path(__file__).parent.parent.parent / "LiteTensorDataTest")
    dataset = FloorplanDatasetLiteTest(data_path)

    for test_id in test_ids:
        sample = dataset[test_id]
        inputs, labels = sample['input'], sample['label']
        area_target, b2b_conn, p2b_conn, pins_pos, constraints = inputs
        n = int((area_target != -1).sum().item())

        # Build opt_target_pos same as evaluator does
        opt_target_pos = torch.full((n, 4), -1.0)
        # (simplified: evaluator extracts from labels, we just pass -1s for debug)

        # Run full optimizer
        opt = MyOptimizer()
        positions = opt.solve(n, area_target, b2b_conn, p2b_conn, pins_pos, constraints, opt_target_pos)

        if isinstance(positions, list):
            pos_list = [tuple(float(v) for v in p) for p in positions]
        else:
            pos_list = [(float(positions[i,0]), float(positions[i,1]),
                         float(positions[i,2]), float(positions[i,3])) for i in range(n)]

        overlaps = check_overlaps(pos_list)

        if not overlaps:
            continue

        # Get topology for hcg/vcg analysis
        blocks, mib_groups, cluster_groups = parse_and_init(n, area_target, constraints, opt_target_pos)
        super_blocks = prepack_clusters(blocks, cluster_groups)
        block_cx, block_cy = analytic_place(blocks, super_blocks, cluster_groups, b2b_conn, p2b_conn, pins_pos)

        hcg_succ, vcg_succ = build_topology(block_cx, block_cy, blocks, super_blocks, cluster_groups)

        cluster_rep = {gid: members[0] for gid, members in cluster_groups.items()}
        def rep(i): return cluster_rep[blocks[i].cluster_group] if blocks[i].cluster_group > 0 else i

        hcg_set = set()
        vcg_set = set()
        for u in range(n):
            for (v, _) in hcg_succ[u]: hcg_set.add((u, v))
            for (v, _) in vcg_succ[u]: vcg_set.add((u, v))

        def has_hcg(ri, rj):
            return (ri, rj) in hcg_set or (rj, ri) in hcg_set
        def has_vcg(ri, rj):
            return (ri, rj) in vcg_set or (rj, ri) in vcg_set

        print(f"\ntest {test_id} n={n}: {len(overlaps)} overlaps")
        for (i, j, xi, yi, wi, hi, xj, yj, wj, hj, ox, oy) in overlaps[:6]:
            ri, rj = rep(i), rep(j)
            hcg = has_hcg(ri, rj)
            vcg = has_vcg(ri, rj)
            bc_i = blocks[i].boundary_code
            bc_j = blocks[j].boundary_code
            print(f"  [{i}]({xi:.1f},{yi:.1f},{wi:.1f},{hi:.1f}) vs [{j}]({xj:.1f},{yj:.1f},{wj:.1f},{hj:.1f}) "
                  f"ox={ox:.2f} oy={oy:.2f} ri={ri} rj={rj} hcg={hcg} vcg={vcg} bc={bc_i}/{bc_j}")


if __name__ == "__main__":
    main()
