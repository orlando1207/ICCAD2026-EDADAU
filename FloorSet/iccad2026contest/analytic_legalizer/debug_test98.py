"""
Deep diagnostic for test 98 (or any test ID): shows why area_gap is high.
Reports bbox vs target area at each stage, critical path blocks, and topology stats.

Usage: python -m analytic_legalizer.debug_test98 [test_id]
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import torch

from litetestLoader import FloorplanDatasetLiteTest
from analytic_legalizer.constraints import (
    parse_and_init, prepack_clusters, slide_boundary, enforce_hard,
    BOUND_LEFT, BOUND_RIGHT, BOUND_TOP, BOUND_BOTTOM,
)
from analytic_legalizer.quadratic_placer import analytic_place
from analytic_legalizer.topology import build_topology, longest_path_pack, augment_topology, compact
from analytic_legalizer.shaping import shape_soft_blocks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def rep_fn(blocks, cluster_groups):
    cluster_rep = {gid: members[0] for gid, members in cluster_groups.items()}
    def rep(i):
        return cluster_rep[blocks[i].cluster_group] if blocks[i].cluster_group > 0 else i
    return rep

def bbox(pos):
    x2 = max(p[0] + p[2] for p in pos)
    y2 = max(p[1] + p[3] for p in pos)
    return x2, y2

def area_gap(pos, area_targets):
    x2, y2 = bbox(pos)
    bb_area = x2 * y2
    total = float(area_targets.sum())
    return (bb_area - total) / total if total > 0 else 0.0

def build_target_pos(constraints, block_count, gt_positions):
    opt = torch.full((block_count, 4), -1.0)
    nc = constraints.shape[1] if constraints.dim() > 1 else 0
    for i in range(block_count):
        is_fixed = nc > 0 and constraints[i, 0] != 0
        is_pp = nc > 1 and constraints[i, 1] != 0
        if is_pp:
            opt[i] = torch.tensor(list(gt_positions[i]))
        elif is_fixed:
            opt[i, 2] = gt_positions[i][2]
            opt[i, 3] = gt_positions[i][3]
    return opt

def gt_from_labels(labels, n):
    polygons, _ = labels
    out = []
    for i in range(n):
        block = polygons[i]
        valid = block[block[:, 0] != -1]
        if len(valid) > 0:
            x_min, y_min = valid.min(dim=0).values
            x_max, y_max = valid.max(dim=0).values
            out.append((float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)))
        else:
            out.append((0.0, 0.0, 1.0, 1.0))
    return out

def critical_path(succ, reps, size_fn, label="HCG"):
    """Find critical path length and node sequence."""
    reps_set = set(reps)
    dist = {r: 0.0 for r in reps}
    prev = {r: None for r in reps}
    for u in reps:
        for (v, w) in succ[u]:
            if v not in reps_set:
                continue
            cand = dist[u] + size_fn(u)
            if cand > dist[v]:
                dist[v] = cand
                prev[v] = u
    end = max(dist, key=lambda r: dist[r] + size_fn(r))
    total = dist[end] + size_fn(end)
    path = []
    cur = end
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return total, path

def count_edges(succ, reps_set):
    return sum(len([v for (v, _) in succ[r] if v in reps_set]) for r in reps_set)

def top_n_reps(blocks, cluster_groups, super_blocks, attr, n=10):
    """Return top-n reps by a given attribute (w or h)."""
    cluster_rep = {gid: members[0] for gid, members in cluster_groups.items()}
    def rep(i): return cluster_rep[blocks[i].cluster_group] if blocks[i].cluster_group > 0 else i
    reps = sorted(set(rep(i) for i in range(len(blocks))))
    def get_size(r):
        gid = blocks[r].cluster_group if r < len(blocks) else 0
        if gid > 0 and gid in super_blocks:
            sb = super_blocks[gid]
            return sb.w if attr == 'w' else sb.h
        return blocks[r].w if attr == 'w' else blocks[r].h
    return sorted(reps, key=get_size, reverse=True)[:n]


# ---------------------------------------------------------------------------
# Main diagnostic
# ---------------------------------------------------------------------------

def analyze(test_id):
    data_path = str(Path(__file__).parent.parent.parent / "LiteTensorDataTest")
    dataset = FloorplanDatasetLiteTest(data_path)
    sample = dataset[test_id]
    inputs, labels = sample['input'], sample['label']
    area_target, b2b_conn, p2b_conn, pins_pos, constraints = inputs
    n = int((area_target != -1).sum().item())
    gt = gt_from_labels(labels, n)
    opt_target_pos = build_target_pos(constraints, n, gt)

    total_area = float(area_target[:n].sum())
    print(f"\n{'='*60}")
    print(f"  Test {test_id}  n={n}  total_block_area={total_area:.1f}")
    print(f"{'='*60}")

    # --- Ground-truth bbox ---
    gt_x2 = max(g[0] + g[2] for g in gt)
    gt_y2 = max(g[1] + g[3] for g in gt)
    gt_util = total_area / (gt_x2 * gt_y2)
    print(f"\nGround-truth  bbox=({gt_x2:.1f} x {gt_y2:.1f})  "
          f"area={gt_x2*gt_y2:.1f}  utilization={gt_util:.3f}")

    # ---- Step 0+1: parse ----
    blocks, mib_groups, cluster_groups = parse_and_init(n, area_target, constraints, opt_target_pos)
    super_blocks = prepack_clusters(blocks, cluster_groups)

    n_pp = sum(1 for b in blocks if b.is_preplaced)
    n_fixed = sum(1 for b in blocks if b.is_fixed_shape and not b.is_preplaced)
    n_soft = sum(1 for b in blocks if not b.is_preplaced and not b.is_fixed_shape)
    n_clusters = len(cluster_groups)
    n_mib = len(mib_groups)
    print(f"\nBlocks: {n} total  preplaced={n_pp}  fixed_shape={n_fixed}  "
          f"soft={n_soft}  clusters={n_clusters}  mib_groups={n_mib}")

    # Preplaced summary
    if n_pp:
        print(f"\nPreplaced blocks:")
        for i, b in enumerate(blocks):
            if b.is_preplaced:
                print(f"  [{i:3d}] pos=({b.fixed_x:.1f},{b.fixed_y:.1f})  "
                      f"size=({b.w:.1f}x{b.h:.1f})  area={b.w*b.h:.1f}")

    # Cluster summary
    if n_clusters:
        print(f"\nCluster super-blocks:")
        for gid, members in cluster_groups.items():
            sb = super_blocks[gid]
            print(f"  cluster {gid}: {len(members)} members  "
                  f"sb=({sb.w:.1f}x{sb.h:.1f})  area={sb.w*sb.h:.1f}")

    # ---- Step 3: analytic place ----
    block_cx, block_cy = analytic_place(
        blocks, super_blocks, cluster_groups, b2b_conn, p2b_conn, pins_pos
    )

    # ---- Step 4+5: topology + pack ----
    hcg_succ, vcg_succ = build_topology(
        block_cx, block_cy, blocks, super_blocks, cluster_groups
    )
    pos5 = longest_path_pack(hcg_succ, vcg_succ, blocks, super_blocks, cluster_groups)
    for _ in range(15):
        if not augment_topology(pos5, blocks, super_blocks, cluster_groups, hcg_succ, vcg_succ):
            break
        pos5 = longest_path_pack(hcg_succ, vcg_succ, blocks, super_blocks, cluster_groups)

    def rep(i): return rep_fn(blocks, cluster_groups)(i)
    reps = sorted(set(rep(i) for i in range(n)))
    reps_set = set(reps)

    def bw_rep(r):
        gid = blocks[r].cluster_group if r < n else 0
        if gid > 0 and gid in super_blocks:
            return super_blocks[gid].w
        return blocks[r].w

    def bh_rep(r):
        gid = blocks[r].cluster_group if r < n else 0
        if gid > 0 and gid in super_blocks:
            return super_blocks[gid].h
        return blocks[r].h

    x2_5, y2_5 = bbox(pos5)
    gap5 = area_gap(pos5, area_target[:n])
    print(f"\nAfter Step 5 (pack):  bbox=({x2_5:.1f} x {y2_5:.1f})  "
          f"area={x2_5*y2_5:.1f}  area_gap={gap5:.3f}  "
          f"util={total_area/(x2_5*y2_5):.3f}")

    # Critical paths after step 5
    n_hcg_edges = count_edges(hcg_succ, reps_set)
    n_vcg_edges = count_edges(vcg_succ, reps_set)
    print(f"  HCG edges={n_hcg_edges}  VCG edges={n_vcg_edges}  reps={len(reps)}")

    hcg_len, hcg_path = critical_path(hcg_succ, reps, bw_rep, "HCG")
    vcg_len, vcg_path = critical_path(vcg_succ, reps, bh_rep, "VCG")
    print(f"  HCG critical path length={hcg_len:.1f}  "
          f"path length={len(hcg_path)} blocks")
    print(f"  VCG critical path length={vcg_len:.1f}  "
          f"path length={len(vcg_path)} blocks")
    print(f"  HCG path: {hcg_path[:15]}{'...' if len(hcg_path)>15 else ''}")
    print(f"  VCG path: {vcg_path[:15]}{'...' if len(vcg_path)>15 else ''}")

    # Widest and tallest reps
    widest = sorted(reps, key=bw_rep, reverse=True)[:5]
    tallest = sorted(reps, key=bh_rep, reverse=True)[:5]
    print(f"  Widest reps: {[(r, round(bw_rep(r),1)) for r in widest]}")
    print(f"  Tallest reps: {[(r, round(bh_rep(r),1)) for r in tallest]}")

    # ---- Step 6: shaping ----
    pos6 = shape_soft_blocks(
        pos5, blocks, super_blocks, cluster_groups, mib_groups, area_target[:n],
        hcg_succ, vcg_succ,
    )
    for _ in range(15):
        pos6 = longest_path_pack(hcg_succ, vcg_succ, blocks, super_blocks, cluster_groups)
        if not augment_topology(pos6, blocks, super_blocks, cluster_groups, hcg_succ, vcg_succ):
            break
    x2_6, y2_6 = bbox(pos6)
    gap6 = area_gap(pos6, area_target[:n])
    print(f"\nAfter Step 6 (shape): bbox=({x2_6:.1f} x {y2_6:.1f})  "
          f"area={x2_6*y2_6:.1f}  area_gap={gap6:.3f}  "
          f"util={total_area/(x2_6*y2_6):.3f}")

    hcg_len6, hcg_path6 = critical_path(hcg_succ, reps, bw_rep, "HCG")
    vcg_len6, vcg_path6 = critical_path(vcg_succ, reps, bh_rep, "VCG")
    print(f"  HCG critical path={hcg_len6:.1f}  VCG critical path={vcg_len6:.1f}")

    # ---- Step 6c: compact ----
    pos6c = compact(pos6, blocks, super_blocks, cluster_groups)
    x2_6c, y2_6c = bbox(pos6c)
    gap6c = area_gap(pos6c, area_target[:n])
    print(f"\nAfter Step 6c (compact): bbox=({x2_6c:.1f} x {y2_6c:.1f})  "
          f"area={x2_6c*y2_6c:.1f}  area_gap={gap6c:.3f}  "
          f"util={total_area/(x2_6c*y2_6c):.3f}")

    # How much did each axis compress?
    print(f"  x: {x2_6:.1f} -> {x2_6c:.1f}  ({100*(1-x2_6c/x2_6):.1f}% reduction)")
    print(f"  y: {y2_6:.1f} -> {y2_6c:.1f}  ({100*(1-y2_6c/y2_6):.1f}% reduction)")

    # After compaction: identify which reps are near the bbox boundary (constraining utilization)
    def rep_box_c(r):
        ms = [i for i in range(n) if rep(i) == r]
        lo_x = min(pos6c[m][0] for m in ms)
        lo_y = min(pos6c[m][1] for m in ms)
        hi_x = max(pos6c[m][0] + pos6c[m][2] for m in ms)
        hi_y = max(pos6c[m][1] + pos6c[m][3] for m in ms)
        return lo_x, lo_y, hi_x, hi_y

    # Blocks on x boundary (rightmost)
    reps_sorted_x2 = sorted(reps, key=lambda r: rep_box_c(r)[2], reverse=True)[:5]
    reps_sorted_y2 = sorted(reps, key=lambda r: rep_box_c(r)[3], reverse=True)[:5]
    print(f"\n  Rightmost reps after compact (x+w):")
    for r in reps_sorted_x2:
        lo_x, lo_y, hi_x, hi_y = rep_box_c(r)
        pp = "PP" if blocks[r].is_preplaced else ""
        cg = f"C{blocks[r].cluster_group}" if blocks[r].cluster_group else ""
        print(f"    rep[{r:3d}]{pp}{cg}  x2={hi_x:.1f}  box=({lo_x:.1f},{lo_y:.1f},{hi_x-lo_x:.1f}x{hi_y-lo_y:.1f})")

    print(f"\n  Topmost reps after compact (y+h):")
    for r in reps_sorted_y2:
        lo_x, lo_y, hi_x, hi_y = rep_box_c(r)
        pp = "PP" if blocks[r].is_preplaced else ""
        cg = f"C{blocks[r].cluster_group}" if blocks[r].cluster_group else ""
        print(f"    rep[{r:3d}]{pp}{cg}  y2={hi_y:.1f}  box=({lo_x:.1f},{lo_y:.1f},{hi_x-lo_x:.1f}x{hi_y-lo_y:.1f})")

    # Check if preplaced blocks are constraining the bbox
    print(f"\n  Preplaced block positions vs compact bbox:")
    for i, b in enumerate(blocks):
        if b.is_preplaced:
            hi_x = b.fixed_x + b.w
            hi_y = b.fixed_y + b.h
            x_frac = hi_x / x2_6c
            y_frac = hi_y / y2_6c
            print(f"    PP[{i:3d}] x2={hi_x:.1f} ({x_frac:.2%} of bbox_x)  "
                  f"y2={hi_y:.1f} ({y_frac:.2%} of bbox_y)  "
                  f"size=({b.w:.1f}x{b.h:.1f})")

    # Find pairs still far apart after compaction — blocks that SHOULD be close
    # but are far away, suggesting topology forces separation
    print(f"\n  Topology analysis — reps with many outgoing HCG edges (serialization sources):")
    hcg_out = sorted(reps, key=lambda r: len([v for (v,_) in hcg_succ[r] if v in reps_set]), reverse=True)[:5]
    for r in hcg_out:
        ne = len([v for (v,_) in hcg_succ[r] if v in reps_set])
        lo_x, lo_y, hi_x, hi_y = rep_box_c(r)
        pp = "PP" if blocks[r].is_preplaced else ""
        print(f"    rep[{r:3d}]{pp}  HCG_out={ne}  x=[{lo_x:.1f},{hi_x:.1f}]")

    vcg_out = sorted(reps, key=lambda r: len([v for (v,_) in vcg_succ[r] if v in reps_set]), reverse=True)[:5]
    print(f"\n  Topology analysis — reps with many outgoing VCG edges:")
    for r in vcg_out:
        ne = len([v for (v,_) in vcg_succ[r] if v in reps_set])
        lo_x, lo_y, hi_x, hi_y = rep_box_c(r)
        pp = "PP" if blocks[r].is_preplaced else ""
        print(f"    rep[{r:3d}]{pp}  VCG_out={ne}  y=[{lo_y:.1f},{hi_y:.1f}]")

    # Whitespace analysis: where is the whitespace in the layout?
    print(f"\n  Layout whitespace: ideal bbox area={total_area:.1f}  "
          f"actual={x2_6c*y2_6c:.1f}  excess={x2_6c*y2_6c-total_area:.1f}")
    print(f"  GT bbox area={gt_x2*gt_y2:.1f}  GT utilization={gt_util:.3f}")

    # Count how many reps the HCG critical path goes through vs total reps
    hcg_len_c, hcg_path_c = critical_path(
        {r: [(v,w) for (v,w) in hcg_succ[r] if v in reps_set] for r in reps},
        reps, bw_rep
    )
    vcg_len_c, vcg_path_c = critical_path(
        {r: [(v,w) for (v,w) in vcg_succ[r] if v in reps_set] for r in reps},
        reps, bh_rep
    )
    # What fraction of reps are on critical paths?
    print(f"\n  After compaction critical paths (using original topology):")
    print(f"    HCG: {len(hcg_path_c)}/{len(reps)} reps, length={hcg_len_c:.1f}")
    print(f"    VCG: {len(vcg_path_c)}/{len(reps)} reps, length={vcg_len_c:.1f}")

    # Minimum possible bbox if blocks could be freely packed:
    # width^2 ≈ total_area * (bbox_x/bbox_y)
    gt_ar = gt_x2 / gt_y2
    min_w = (total_area * gt_ar) ** 0.5
    min_h = total_area / min_w
    print(f"\n  If blocks fit at GT aspect ratio ({gt_ar:.2f}):")
    print(f"    Ideal size = {min_w:.1f} x {min_h:.1f}  (area={min_w*min_h:.1f})")
    print(f"    Current = {x2_6c:.1f} x {y2_6c:.1f}")
    print(f"    Constraint overhead: x={x2_6c/min_w:.2f}x  y={y2_6c/min_h:.2f}x")


if __name__ == "__main__":
    test_id = int(sys.argv[1]) if len(sys.argv) > 1 else 98
    analyze(test_id)
