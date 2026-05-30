"""
Step 3 — Analytic arrangement: quadratic wirelength minimization + spreading.

Nodes = free blocks + cluster super-blocks.
Preplaced blocks = fixed anchors (Dirichlet boundary conditions).
Solves: min Σ_e w_e * [(cx_i - cx_j)² + (cy_i - cy_j)²]  subject to density spreading.
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .constraints import BlockInfo, SuperBlock


def analytic_place(
    blocks: List[BlockInfo],
    super_blocks: Dict[int, SuperBlock],
    cluster_groups: Dict[int, List[int]],
    b2b_connectivity: torch.Tensor,
    p2b_connectivity: torch.Tensor,
    pins_pos: torch.Tensor,
    n_spread_iters: int = 30,
    seed: int = 0,
    noise_std: float = 0.0,
    wl_model: str = "quadratic",
    lse_gamma: Optional[float] = None,
    lse_iters: int = 80,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (cx, cy): center coordinates for ALL original blocks.
    Steps:
      1. Build node list (free blocks + super-blocks replacing cluster members).
      2. Build quadratic system from b2b and p2b connectivity.
      3. Solve with numpy (n≤120, trivially fast).
      4. Run spreading iterations to reduce overlap.
      5. Map super-block centers back to member centers.
    """
    n = len(blocks)

    # --- Build node mapping ---
    # cluster members are represented by their super-block; others are independent
    cluster_to_sb: Dict[int, int] = {}       # cluster_id -> super-block node index
    block_to_node: Dict[int, int] = {}       # original block idx -> node index
    node_widths:  List[float] = []
    node_heights: List[float] = []
    node_fixed_x: Dict[int, float] = {}     # node_idx -> fixed cx
    node_fixed_y: Dict[int, float] = {}

    node_idx = 0
    # First: preplaced blocks (fixed anchors)
    for i, b in enumerate(blocks):
        if b.is_preplaced:
            block_to_node[i] = node_idx
            node_widths.append(b.w)
            node_heights.append(b.h)
            node_fixed_x[node_idx] = b.fixed_x + b.w / 2.0
            node_fixed_y[node_idx] = b.fixed_y + b.h / 2.0
            node_idx += 1

    # Second: cluster super-blocks (one node per cluster)
    for gid, sb in super_blocks.items():
        cluster_to_sb[gid] = node_idx
        for m in sb.members:
            block_to_node[m] = node_idx
        node_widths.append(sb.w)
        node_heights.append(sb.h)
        node_idx += 1

    # Third: remaining free blocks
    for i, b in enumerate(blocks):
        if i not in block_to_node:
            block_to_node[i] = node_idx
            node_widths.append(b.w)
            node_heights.append(b.h)
            node_idx += 1

    num_nodes = node_idx
    node_widths  = np.array(node_widths,  dtype=np.float64)
    node_heights = np.array(node_heights, dtype=np.float64)

    # --- Estimate chip area for initial spread ---
    total_area = sum(node_widths[k] * node_heights[k] for k in range(num_nodes))
    chip_side = math.sqrt(total_area) * 1.2  # 20% utilization margin

    # --- Build quadratic system Ax = bx, Ay = by ---
    # For each edge (i,j,w): add w*[(ci-cj)²] -> W[i,i]+=w, W[j,j]+=w, W[i,j]-=w
    A  = np.zeros((num_nodes, num_nodes), dtype=np.float64)
    bx = np.zeros(num_nodes, dtype=np.float64)
    by = np.zeros(num_nodes, dtype=np.float64)

    # b2b edges
    b2b = b2b_connectivity
    if b2b is not None and b2b.numel() > 0:
        b2b_np = b2b.numpy() if isinstance(b2b, torch.Tensor) else b2b
        for edge in b2b_np:
            bi, bj, w = int(edge[0]), int(edge[1]), float(edge[2])
            ni, nj = block_to_node[bi], block_to_node[bj]
            if ni == nj:
                continue
            A[ni, ni] += w
            A[nj, nj] += w
            A[ni, nj] -= w
            A[nj, ni] -= w

    # p2b edges (pins are fixed)
    p2b = p2b_connectivity
    pins = pins_pos
    if p2b is not None and p2b.numel() > 0:
        p2b_np = p2b.numpy() if isinstance(p2b, torch.Tensor) else p2b
        pins_np = pins.numpy() if isinstance(pins, torch.Tensor) else pins
        for edge in p2b_np:
            pi, bi, w = int(edge[0]), int(edge[1]), float(edge[2])
            ni = block_to_node[bi]
            px, py = float(pins_np[pi, 0]), float(pins_np[pi, 1])
            A[ni, ni] += w
            bx[ni]    += w * px
            by[ni]    += w * py

    # --- Fix preplaced nodes (Dirichlet) ---
    # Replace their rows/cols: A[i,i]=1, b[i]=fixed_c, zero off-diag contributions
    for nf, cx_fixed in node_fixed_x.items():
        cy_fixed = node_fixed_y[nf]
        # Transfer contributions to RHS
        for k in range(num_nodes):
            if k != nf:
                bx[k] -= A[k, nf] * cx_fixed
                by[k] -= A[k, nf] * cy_fixed
                A[k, nf] = 0.0
                A[nf, k] = 0.0
        A[nf, nf] = 1.0
        bx[nf] = cx_fixed
        by[nf] = cy_fixed

    # --- Check if system is well-anchored (has fixed nodes or p2b edges) ---
    has_anchors = len(node_fixed_x) > 0 or (p2b_connectivity is not None and p2b_connectivity.numel() > 0)

    if has_anchors:
        # Add weak spring to chip center only for truly disconnected nodes
        for k in range(num_nodes):
            if abs(A[k, k]) < 1e-12:
                eps = 1e-3
                A[k, k] += eps
                bx[k]   += eps * (chip_side / 2.0)
                by[k]   += eps * (chip_side / 2.0)
        # Solve the anchored system
        try:
            cx = np.linalg.solve(A, bx)
            cy = np.linalg.solve(A, by)
        except np.linalg.LinAlgError:
            cx = _grid_init(num_nodes, node_widths, node_heights, chip_side)
            cy = _grid_init(num_nodes, node_heights, node_widths, chip_side)
        # Keep preplaced fixed after solve (numerical safety)
        for nf, cx_fixed in node_fixed_x.items():
            cx[nf] = cx_fixed
            cy[nf] = node_fixed_y[nf]
    else:
        # No anchors: Laplacian is rank-deficient. Skip linear solve, use grid init.
        cx, cy = _grid_init_xy(num_nodes, node_widths, node_heights, chip_side)

    if wl_model == "lse":
        cx, cy = _lse_refine(
            cx, cy, block_to_node, node_widths, node_heights,
            node_fixed_x, node_fixed_y, b2b_connectivity,
            p2b_connectivity, pins_pos, chip_side, lse_gamma, lse_iters,
        )
    elif wl_model != "quadratic":
        raise ValueError(f"unsupported wl_model: {wl_model}")

    # --- Optional noise injection (for multistart diversity) ---
    if noise_std > 0.0 and seed > 0:
        rng = np.random.default_rng(seed)
        noise = rng.normal(0.0, noise_std * chip_side, size=(num_nodes, 2))
        for k in range(num_nodes):
            if k not in node_fixed_x:
                cx[k] += noise[k, 0]
                cy[k] += noise[k, 1]
                cx[k] = max(node_widths[k] / 2.0,
                            min(chip_side - node_widths[k] / 2.0, cx[k]))
                cy[k] = max(node_heights[k] / 2.0,
                            min(chip_side - node_heights[k] / 2.0, cy[k]))

    # --- Spreading iterations ---
    cx, cy = _spread_nodes(cx, cy, node_widths, node_heights,
                           node_fixed_x, node_fixed_y,
                           chip_side, n_spread_iters)

    # --- Map back to per-original-block centers ---
    block_cx = np.zeros(n, dtype=np.float64)
    block_cy = np.zeros(n, dtype=np.float64)
    for i in range(n):
        ni = block_to_node[i]
        b  = blocks[i]
        if b.cluster_group > 0 and b.cluster_group in super_blocks:
            sb = super_blocks[b.cluster_group]
            mi = sb.members.index(i)
            dx, dy = sb.offsets[mi]
            # super-block center → lower-left → member center
            sb_orig_x = cx[ni] - sb.w / 2.0
            sb_orig_y = cy[ni] - sb.h / 2.0
            block_cx[i] = sb_orig_x + dx + b.w / 2.0
            block_cy[i] = sb_orig_y + dy + b.h / 2.0
        else:
            block_cx[i] = cx[ni]
            block_cy[i] = cy[ni]

    return block_cx, block_cy


def _lse_refine(
    cx: np.ndarray,
    cy: np.ndarray,
    block_to_node: Dict[int, int],
    widths: np.ndarray,
    heights: np.ndarray,
    fixed_x: Dict[int, float],
    fixed_y: Dict[int, float],
    b2b_connectivity: torch.Tensor,
    p2b_connectivity: torch.Tensor,
    pins_pos: torch.Tensor,
    chip_side: float,
    gamma: Optional[float],
    n_iters: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Refine the quadratic solution with smooth L1/LSE wirelength.

    The contest metric is HPWL, not squared wirelength.  For the pairwise nets
    exposed by FloorSet, a differentiable LSE approximation to |dx| + |dy| has
    gradient tanh(dx/gamma), tanh(dy/gamma).  Starting from the quadratic solve
    keeps the global placement stable while this pass reduces the quadratic
    model's tendency to over-penalize long edges.
    """
    cx = cx.copy()
    cy = cy.copy()
    n = len(cx)
    if n == 0 or n_iters <= 0:
        return cx, cy

    gamma = float(gamma) if gamma is not None else max(chip_side * 0.04, 1.0)
    gamma = max(gamma, 1e-3)

    b2b_np = None
    if b2b_connectivity is not None and b2b_connectivity.numel() > 0:
        b2b_np = b2b_connectivity.numpy() if isinstance(b2b_connectivity, torch.Tensor) else b2b_connectivity
    p2b_np = None
    pins_np = None
    if p2b_connectivity is not None and p2b_connectivity.numel() > 0:
        p2b_np = p2b_connectivity.numpy() if isinstance(p2b_connectivity, torch.Tensor) else p2b_connectivity
        pins_np = pins_pos.numpy() if isinstance(pins_pos, torch.Tensor) else pins_pos

    if b2b_np is None and p2b_np is None:
        return cx, cy

    # Normalize step by incident weight so dense hubs do not jump excessively.
    incident = np.ones(n, dtype=np.float64) * 1e-6
    if b2b_np is not None:
        for edge in b2b_np:
            bi, bj, w = int(edge[0]), int(edge[1]), float(edge[2])
            ni, nj = block_to_node[bi], block_to_node[bj]
            if ni != nj:
                incident[ni] += w
                incident[nj] += w
    if p2b_np is not None:
        for edge in p2b_np:
            _, bi, w = int(edge[0]), int(edge[1]), float(edge[2])
            incident[block_to_node[bi]] += w

    base_step = max(chip_side * 0.018, 0.25)
    movable = [k for k in range(n) if k not in fixed_x]
    for it in range(n_iters):
        gx = np.zeros(n, dtype=np.float64)
        gy = np.zeros(n, dtype=np.float64)

        if b2b_np is not None:
            for edge in b2b_np:
                bi, bj, w = int(edge[0]), int(edge[1]), float(edge[2])
                ni, nj = block_to_node[bi], block_to_node[bj]
                if ni == nj:
                    continue
                dx = cx[ni] - cx[nj]
                dy = cy[ni] - cy[nj]
                tx = w * math.tanh(dx / gamma)
                ty = w * math.tanh(dy / gamma)
                gx[ni] += tx
                gy[ni] += ty
                gx[nj] -= tx
                gy[nj] -= ty

        if p2b_np is not None:
            for edge in p2b_np:
                pi, bi, w = int(edge[0]), int(edge[1]), float(edge[2])
                ni = block_to_node[bi]
                dx = cx[ni] - float(pins_np[pi, 0])
                dy = cy[ni] - float(pins_np[pi, 1])
                gx[ni] += w * math.tanh(dx / gamma)
                gy[ni] += w * math.tanh(dy / gamma)

        step = base_step * (0.96 ** it)
        for k in movable:
            cx[k] -= step * gx[k] / incident[k]
            cy[k] -= step * gy[k] / incident[k]
            cx[k] = max(widths[k] / 2.0, min(chip_side - widths[k] / 2.0, cx[k]))
            cy[k] = max(heights[k] / 2.0, min(chip_side - heights[k] / 2.0, cy[k]))

    for nf, x_fixed in fixed_x.items():
        cx[nf] = x_fixed
        cy[nf] = fixed_y[nf]

    return cx, cy


def _grid_init_xy(
    n: int, widths: np.ndarray, heights: np.ndarray, chip_side: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Place nodes on a grid for a well-spread initial layout."""
    cols = max(1, math.ceil(math.sqrt(n)))
    cx = np.zeros(n)
    cy = np.zeros(n)
    col_w = chip_side / cols
    row_h = chip_side / math.ceil(n / cols)
    for k in range(n):
        col = k % cols
        row = k // cols
        cx[k] = (col + 0.5) * col_w
        cy[k] = (row + 0.5) * row_h
    return cx, cy


def _grid_init(n: int, d1: np.ndarray, d2: np.ndarray, chip_side: float) -> np.ndarray:
    cx, _ = _grid_init_xy(n, d1, d2, chip_side)
    return cx


def _spread_nodes(
    cx: np.ndarray, cy: np.ndarray,
    widths: np.ndarray, heights: np.ndarray,
    fixed_x: Dict[int, float], fixed_y: Dict[int, float],
    chip_side: float,
    n_iters: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Iterative pairwise repulsion to reduce overlap density.
    Force = k * overlap_area * (unit vector away from other center).
    """
    n = len(cx)
    step = 0.5

    for it in range(n_iters):
        fx = np.zeros(n)
        fy = np.zeros(n)

        for i in range(n):
            for j in range(i + 1, n):
                # Distance between centers
                dx = cx[i] - cx[j]
                dy = cy[i] - cy[j]

                # Overlap in each axis
                ox = (widths[i] + widths[j]) / 2.0 - abs(dx)
                oy = (heights[i] + heights[j]) / 2.0 - abs(dy)

                if ox <= 0 or oy <= 0:
                    continue  # no overlap

                # Push along the smaller-overlap axis
                overlap = min(ox, oy)
                dist = math.sqrt(dx * dx + dy * dy) + 1e-9
                fx[i] += overlap * dx / dist
                fy[i] += overlap * dy / dist
                fx[j] -= overlap * dx / dist
                fy[j] -= overlap * dy / dist

        # Apply forces, keep fixed nodes still
        for k in range(n):
            if k in fixed_x:
                continue
            cx[k] += step * fx[k]
            cy[k] += step * fy[k]
            # Clamp inside chip
            cx[k] = max(widths[k]  / 2.0, min(chip_side - widths[k]  / 2.0, cx[k]))
            cy[k] = max(heights[k] / 2.0, min(chip_side - heights[k] / 2.0, cy[k]))

        step *= 0.95  # cool the step size

    return cx, cy
