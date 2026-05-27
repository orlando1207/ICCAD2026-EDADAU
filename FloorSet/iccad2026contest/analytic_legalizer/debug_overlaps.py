"""
Run the full pipeline on test cases and report remaining overlaps with hcg/vcg info.
Usage: python debug_overlaps.py [test_ids...]
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np

from analytic_legalizer.optimizer import MyOptimizer
from analytic_legalizer.topology import build_topology, longest_path_pack
from analytic_legalizer.constraints import parse_and_init, prepack_clusters


def load_dataset():
    import pickle, json, os
    base = Path(__file__).parent.parent
    # Try to load from the contest data
    data_path = base / "floorset_lite_val.pt"
    if data_path.exists():
        return torch.load(data_path)
    # Try pickle
    for p in base.glob("*.pt"):
        try:
            return torch.load(p)
        except:
            pass
    return None


def check_overlaps_with_topology(positions, blocks, super_blocks, cluster_groups, block_cx, block_cy):
    """Report overlaps with HCG/VCG relationship info."""
    n = len(positions)

    # Build topology for reference
    hcg_succ, vcg_succ = build_topology(block_cx, block_cy, blocks, super_blocks, cluster_groups)

    # Build adjacency sets for quick lookup (by block index, not rep)
    cluster_rep = {}
    for gid, members in cluster_groups.items():
        cluster_rep[gid] = members[0]

    def rep(i):
        return cluster_rep[blocks[i].cluster_group] if blocks[i].cluster_group > 0 else i

    hcg_set = set()
    vcg_set = set()
    for u in range(n):
        for (v, _) in hcg_succ[u]:
            hcg_set.add((u, v))
            hcg_set.add((v, u))  # symmetric for lookup
    for u in range(n):
        for (v, _) in vcg_succ[u]:
            vcg_set.add((u, v))
            vcg_set.add((v, u))

    overlaps = []
    for i in range(n):
        for j in range(i+1, n):
            xi, yi, wi, hi = positions[i]
            xj, yj, wj, hj = positions[j]
            ox = min(xi+wi, xj+wj) - max(xi, xj)
            oy = min(yi+hi, yj+hj) - max(yi, yj)
            if ox > 1e-6 and oy > 1e-6:
                ri, rj = rep(i), rep(j)
                hcg = (ri, rj) in hcg_set or (rj, ri) in hcg_set
                vcg = (ri, rj) in vcg_set or (rj, ri) in vcg_set
                overlaps.append((i, j, xi, yi, wi, hi, xj, yj, wj, hj, ox, oy, ri, rj, hcg, vcg))
    return overlaps


def main():
    test_ids = [int(x) for x in sys.argv[1:]] if len(sys.argv) > 1 else list(range(10))

    dataset = load_dataset()
    if dataset is None:
        print("Could not load dataset")
        return

    for test_id in test_ids:
        area_targets, constraints, net_info, target_positions = dataset[test_id]
        n = area_targets.shape[0]

        opt = MyOptimizer()
        # Run steps 0-5 manually to get topology + positions before enforce_hard
        blocks, mib_groups, cluster_groups = parse_and_init(n, area_targets, constraints, target_positions)
        super_blocks = prepack_clusters(blocks, cluster_groups)

        # Run full solve to get final positions
        positions = opt.solve(area_targets, constraints, net_info, target_positions)

        # Re-run steps 0-3 to get analytic centers for topology info
        from analytic_legalizer.quadratic_placer import analytic_place
        blocks2, mib_groups2, cluster_groups2 = parse_and_init(n, area_targets, constraints, target_positions)
        super_blocks2 = prepack_clusters(blocks2, cluster_groups2)
        block_cx, block_cy = analytic_place(blocks2, super_blocks2, cluster_groups2, net_info, area_targets)

        # Check overlaps in final positions
        pos_list = [(float(positions[i,0]), float(positions[i,1]), float(positions[i,2]), float(positions[i,3])) for i in range(n)]

        overlaps = check_overlaps_with_topology(pos_list, blocks2, super_blocks2, cluster_groups2, block_cx, block_cy)

        if overlaps:
            print(f"\ntest {test_id} n={n}:")
            for (i, j, xi, yi, wi, hi, xj, yj, wj, hj, ox, oy, ri, rj, hcg, vcg) in overlaps[:8]:
                print(f"  [{i}]({xi:.1f},{yi:.1f},{wi:.1f},{hi:.1f}) vs [{j}]({xj:.1f},{yj:.1f},{wj:.1f},{hj:.1f}) ox={ox:.2f} oy={oy:.2f} ri={ri} rj={rj} hcg={hcg} vcg={vcg}")


if __name__ == "__main__":
    main()
