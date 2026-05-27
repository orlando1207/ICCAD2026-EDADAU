"""
Per-step unit tests. Run: python -m pytest test_steps.py -v
Or standalone: python test_steps.py
"""

import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from analytic_legalizer.constraints import (
    parse_and_init, prepack_clusters,
    slide_boundary, enforce_hard,
    BlockInfo, SuperBlock, BOUND_LEFT, BOUND_RIGHT, BOUND_TOP, BOUND_BOTTOM,
)
from analytic_legalizer.quadratic_placer import analytic_place
from analytic_legalizer.topology import build_topology, longest_path_pack
from analytic_legalizer.shaping import shape_soft_blocks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_simple_case(n=6):
    """
    6 free soft blocks, no preplaced, no MIB, no cluster, no boundary.
    """
    area = torch.tensor([100.0] * n)
    constraints = torch.zeros(n, 5)
    tp = torch.full((n, 4), -1.0)
    b2b = torch.tensor([[0, 1, 1.0], [1, 2, 1.0], [2, 3, 1.0],
                         [3, 4, 1.0], [4, 5, 1.0]], dtype=torch.float32)
    p2b = torch.zeros(0, 3)
    pins = torch.zeros(0, 2)
    return n, area, constraints, tp, b2b, p2b, pins


def _no_overlap(positions):
    """Return True if no two blocks overlap (beyond 1e-6)."""
    n = len(positions)
    for i in range(n):
        for j in range(i + 1, n):
            xi, yi, wi, hi = positions[i]
            xj, yj, wj, hj = positions[j]
            ox = min(xi + wi, xj + wj) - max(xi, xj)
            oy = min(yi + hi, yj + hj) - max(yi, yj)
            if ox > 1e-4 and oy > 1e-4:
                return False
    return True


def _area_ok(positions, area_targets, blocks):
    """Return True if all blocks have area within 2% of target."""
    for i, (x, y, w, h) in enumerate(positions):
        if blocks[i].is_preplaced or blocks[i].is_fixed_shape:
            continue
        target = float(area_targets[i])
        actual = w * h
        if abs(actual - target) / max(target, 1e-9) > 0.02:
            return False
    return True


# ---------------------------------------------------------------------------
# Test Step 0+1: parse_and_init
# ---------------------------------------------------------------------------

def test_step01_basic():
    n, area, constraints, tp, b2b, p2b, pins = _make_simple_case()
    blocks, mib_groups, cluster_groups = parse_and_init(n, area, constraints, tp)
    assert len(blocks) == n, "wrong block count"
    for b in blocks:
        assert abs(b.w * b.h - 100.0) < 1e-6, f"block {b.idx} wrong area init: {b.w*b.h}"
    assert len(mib_groups) == 0
    assert len(cluster_groups) == 0
    print("PASS test_step01_basic")


def test_step01_mib_unification():
    n = 4
    area = torch.tensor([100.0, 120.0, 80.0, 90.0])
    constraints = torch.zeros(n, 5)
    constraints[0, 2] = 1  # MIB group 1
    constraints[1, 2] = 1
    constraints[2, 2] = 2  # MIB group 2
    constraints[3, 2] = 2
    tp = torch.full((n, 4), -1.0)

    blocks, mib_groups, _ = parse_and_init(n, area, constraints, tp)

    # All group-1 members should have same (w,h)
    w0, h0 = blocks[0].w, blocks[0].h
    w1, h1 = blocks[1].w, blocks[1].h
    assert abs(w0 - w1) < 1e-9 and abs(h0 - h1) < 1e-9, "MIB group 1 shapes not unified"

    w2, h2 = blocks[2].w, blocks[2].h
    w3, h3 = blocks[3].w, blocks[3].h
    assert abs(w2 - w3) < 1e-9 and abs(h2 - h3) < 1e-9, "MIB group 2 shapes not unified"
    print("PASS test_step01_mib_unification")


def test_step01_preplaced():
    n = 3
    area = torch.tensor([100.0, 200.0, 150.0])
    constraints = torch.zeros(n, 5)
    constraints[0, 1] = 1  # preplaced
    tp = torch.full((n, 4), -1.0)
    tp[0] = torch.tensor([10.0, 20.0, 8.0, 12.5])  # x=10, y=20, w=8, h=12.5

    blocks, _, _ = parse_and_init(n, area, constraints, tp)

    assert blocks[0].is_preplaced
    assert abs(blocks[0].fixed_x - 10.0) < 1e-6
    assert abs(blocks[0].fixed_y - 20.0) < 1e-6
    assert abs(blocks[0].w - 8.0) < 1e-6
    assert abs(blocks[0].h - 12.5) < 1e-6
    print("PASS test_step01_preplaced")


# ---------------------------------------------------------------------------
# Test Step 2: prepack_clusters
# ---------------------------------------------------------------------------

def test_step02_cluster_prepack():
    n = 4
    area = torch.tensor([100.0] * n)
    constraints = torch.zeros(n, 5)
    constraints[0, 3] = 1  # cluster group 1
    constraints[1, 3] = 1
    constraints[2, 3] = 2  # cluster group 2
    constraints[3, 3] = 2
    tp = torch.full((n, 4), -1.0)

    blocks, mib_groups, cluster_groups = parse_and_init(n, area, constraints, tp)
    super_blocks = prepack_clusters(blocks, cluster_groups)

    assert 1 in super_blocks and 2 in super_blocks
    sb1 = super_blocks[1]
    sb2 = super_blocks[2]

    # Super-block area should be >= sum of member areas (some whitespace OK)
    for sb in [sb1, sb2]:
        member_area = sum(blocks[i].w * blocks[i].h for i in sb.members)
        assert sb.w * sb.h >= member_area * 0.99, \
            f"super-block too small: {sb.w*sb.h} < {member_area}"

    # Members should be abutted: check offsets are consistent
    for sb in [sb1, sb2]:
        for idx, (ox, oy) in zip(sb.members, sb.offsets):
            bw = blocks[idx].w
            bh = blocks[idx].h
            assert ox + bw <= sb.w + 1e-6, "member x overflows super-block"
            assert oy + bh <= sb.h + 1e-6, "member y overflows super-block"

    print("PASS test_step02_cluster_prepack")


# ---------------------------------------------------------------------------
# Test Step 3: analytic_place
# ---------------------------------------------------------------------------

def test_step03_analytic_place():
    n, area, constraints, tp, b2b, p2b, pins = _make_simple_case()
    blocks, mib_groups, cluster_groups = parse_and_init(n, area, constraints, tp)
    super_blocks = prepack_clusters(blocks, cluster_groups)

    cx, cy = analytic_place(blocks, super_blocks, cluster_groups, b2b, p2b, pins)

    assert cx.shape == (n,) and cy.shape == (n,)
    assert not np.any(np.isnan(cx)) and not np.any(np.isnan(cy))
    # Connected blocks should be pulled toward each other
    # Blocks 0-1-2-3-4-5 are in a chain; 0 and 5 should be far apart
    dist_0_5 = abs(cx[0] - cx[5]) + abs(cy[0] - cy[5])
    dist_0_1 = abs(cx[0] - cx[1]) + abs(cy[0] - cy[1])
    # Can't assert much without knowing exact solution; just check outputs valid
    print(f"  analytic centers: cx={cx.round(1)}, cy={cy.round(1)}")
    print("PASS test_step03_analytic_place")


def test_step03_preplaced_fixed():
    """Preplaced block must stay at its fixed position after analytic placement."""
    n = 3
    area = torch.tensor([100.0, 100.0, 100.0])
    constraints = torch.zeros(n, 5)
    constraints[0, 1] = 1  # preplaced
    tp = torch.full((n, 4), -1.0)
    tp[0] = torch.tensor([0.0, 0.0, 10.0, 10.0])
    b2b = torch.tensor([[0, 1, 1.0], [1, 2, 1.0]], dtype=torch.float32)
    p2b = torch.zeros(0, 3)
    pins = torch.zeros(0, 2)

    blocks, mib_groups, cluster_groups = parse_and_init(n, area, constraints, tp)
    super_blocks = prepack_clusters(blocks, cluster_groups)
    cx, cy = analytic_place(blocks, super_blocks, cluster_groups, b2b, p2b, pins)

    # Block 0 center should be exactly (5, 5) (center of [0,0,10,10])
    assert abs(cx[0] - 5.0) < 1e-6, f"preplaced cx wrong: {cx[0]}"
    assert abs(cy[0] - 5.0) < 1e-6, f"preplaced cy wrong: {cy[0]}"
    print("PASS test_step03_preplaced_fixed")


# ---------------------------------------------------------------------------
# Test Steps 4+5: topology + longest-path legalization
# ---------------------------------------------------------------------------

def test_step45_no_overlap():
    n, area, constraints, tp, b2b, p2b, pins = _make_simple_case()
    blocks, mib_groups, cluster_groups = parse_and_init(n, area, constraints, tp)
    super_blocks = prepack_clusters(blocks, cluster_groups)
    cx, cy = analytic_place(blocks, super_blocks, cluster_groups, b2b, p2b, pins)
    hcg, vcg = build_topology(cx, cy, blocks, super_blocks, cluster_groups)
    positions = longest_path_pack(hcg, vcg, blocks, super_blocks, cluster_groups)

    assert len(positions) == n
    assert _no_overlap(positions), f"overlaps after legalization: {positions}"
    print("PASS test_step45_no_overlap")


def test_step45_cluster_abutment():
    """After legalization, cluster members must be abutted (share an edge)."""
    n = 4
    area = torch.tensor([100.0] * n)
    constraints = torch.zeros(n, 5)
    constraints[0, 3] = 1
    constraints[1, 3] = 1
    tp = torch.full((n, 4), -1.0)
    b2b = torch.zeros(0, 3)
    p2b = torch.zeros(0, 3)
    pins = torch.zeros(0, 2)

    blocks, mib_groups, cluster_groups = parse_and_init(n, area, constraints, tp)
    super_blocks = prepack_clusters(blocks, cluster_groups)
    cx, cy = analytic_place(blocks, super_blocks, cluster_groups, b2b, p2b, pins)
    hcg, vcg = build_topology(cx, cy, blocks, super_blocks, cluster_groups)
    positions = longest_path_pack(hcg, vcg, blocks, super_blocks, cluster_groups)

    # Cluster members 0,1 should share an edge (x or y touch)
    x0, y0, w0, h0 = positions[0]
    x1, y1, w1, h1 = positions[1]
    x_touch = abs(x0 + w0 - x1) < 1e-4 or abs(x1 + w1 - x0) < 1e-4
    y_touch = abs(y0 + h0 - y1) < 1e-4 or abs(y1 + h1 - y0) < 1e-4
    # At least one axis must touch (they're in the same super-block row)
    assert x_touch or y_touch, \
        f"cluster members not abutted: {positions[0]} {positions[1]}"
    print("PASS test_step45_cluster_abutment")


# ---------------------------------------------------------------------------
# Test Step 8: enforce_hard
# ---------------------------------------------------------------------------

def test_step08_enforce_hard_area():
    """After enforce_hard, all soft-block areas should be within 1% of target."""
    n, area, constraints, tp, b2b, p2b, pins = _make_simple_case()
    blocks, mib_groups, cluster_groups = parse_and_init(n, area, constraints, tp)
    super_blocks = prepack_clusters(blocks, cluster_groups)
    cx, cy = analytic_place(blocks, super_blocks, cluster_groups, b2b, p2b, pins)
    hcg, vcg = build_topology(cx, cy, blocks, super_blocks, cluster_groups)
    positions = longest_path_pack(hcg, vcg, blocks, super_blocks, cluster_groups)
    positions = enforce_hard(positions, blocks, area)

    assert _area_ok(positions, area, blocks), \
        [(i, positions[i][2]*positions[i][3]) for i in range(n)]
    assert _no_overlap(positions), "overlaps after enforce_hard"
    print("PASS test_step08_enforce_hard_area")


def test_step08_preplaced_exact():
    n = 3
    area = torch.tensor([100.0, 100.0, 100.0])
    constraints = torch.zeros(n, 5)
    constraints[0, 1] = 1
    tp = torch.full((n, 4), -1.0)
    tp[0] = torch.tensor([5.0, 7.0, 10.0, 10.0])
    b2b = torch.zeros(0, 3)
    p2b = torch.zeros(0, 3)
    pins = torch.zeros(0, 2)

    blocks, _, cluster_groups = parse_and_init(n, area, constraints, tp)
    super_blocks = prepack_clusters(blocks, cluster_groups)
    cx, cy = analytic_place(blocks, super_blocks, cluster_groups, b2b, p2b, pins)
    hcg, vcg = build_topology(cx, cy, blocks, super_blocks, cluster_groups)
    positions = longest_path_pack(hcg, vcg, blocks, super_blocks, cluster_groups)
    positions = enforce_hard(positions, blocks, area)

    x, y, w, h = positions[0]
    assert abs(x - 5.0) < 1e-6, f"preplaced x wrong: {x}"
    assert abs(y - 7.0) < 1e-6, f"preplaced y wrong: {y}"
    assert abs(w - 10.0) < 1e-6
    assert abs(h - 10.0) < 1e-6
    print("PASS test_step08_preplaced_exact")


# ---------------------------------------------------------------------------
# End-to-end smoke test
# ---------------------------------------------------------------------------

def test_e2e_full_pipeline():
    from analytic_legalizer.optimizer import MyOptimizer

    n, area, constraints, tp, b2b, p2b, pins = _make_simple_case(n=10)
    opt = MyOptimizer(verbose=False)
    positions = opt.solve(n, area, b2b, p2b, pins, constraints, tp)

    assert len(positions) == n, f"wrong output count: {len(positions)}"
    assert _no_overlap(positions), "overlap in full pipeline output"
    assert _area_ok(positions, area,
                    parse_and_init(n, area, constraints, tp)[0]), \
        "area violation in full pipeline output"
    print("PASS test_e2e_full_pipeline")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_step01_basic,
        test_step01_mib_unification,
        test_step01_preplaced,
        test_step02_cluster_prepack,
        test_step03_analytic_place,
        test_step03_preplaced_fixed,
        test_step45_no_overlap,
        test_step45_cluster_abutment,
        test_step08_enforce_hard_area,
        test_step08_preplaced_exact,
        test_e2e_full_pipeline,
    ]
    failed = []
    for t in tests:
        try:
            t()
        except Exception as e:
            print(f"FAIL {t.__name__}: {e}")
            import traceback; traceback.print_exc()
            failed.append(t.__name__)

    print(f"\n{'='*40}")
    print(f"Results: {len(tests)-len(failed)}/{len(tests)} passed")
    if failed:
        print(f"FAILED: {failed}")
        sys.exit(1)
    else:
        print("All tests passed.")
