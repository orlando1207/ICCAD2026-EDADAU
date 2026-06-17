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
    n_spread_iters: int = 10,
    seed: int = 0,
    noise_std: float = 0.0,
    n_wl_iters: int = 3,
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

    # --- Collect edges (node indices + base weight) ---
    b2b_edges: List[Tuple[int, int, float]] = []
    b2b = b2b_connectivity
    if b2b is not None and b2b.numel() > 0:
        b2b_np = b2b.numpy() if isinstance(b2b, torch.Tensor) else b2b
        for edge in b2b_np:
            bi, bj, w = int(edge[0]), int(edge[1]), float(edge[2])
            ni, nj = block_to_node[bi], block_to_node[bj]
            if ni != nj:
                b2b_edges.append((ni, nj, w))

    p2b_edges: List[Tuple[int, float, float, float]] = []
    p2b = p2b_connectivity
    pins = pins_pos
    if p2b is not None and p2b.numel() > 0:
        p2b_np = p2b.numpy() if isinstance(p2b, torch.Tensor) else p2b
        pins_np = pins.numpy() if isinstance(pins, torch.Tensor) else pins
        for edge in p2b_np:
            pi, bi, w = int(edge[0]), int(edge[1]), float(edge[2])
            ni = block_to_node[bi]
            p2b_edges.append((ni, float(pins_np[pi, 0]), float(pins_np[pi, 1]), w))

    has_anchors = len(node_fixed_x) > 0 or len(p2b_edges) > 0
    EPS_WL = 1.0  # avoids div-by-zero / over-weighting in the B2B reweighting

    def _build_solve(cxr, cyr, linear):
        """Build & solve the (optionally B2B-reweighted) system. With linear=True
        each edge weight is divided by its current span → the quadratic minimiser
        converges to the linear HPWL optimum. Separate Ax/Ay (per-axis weights)."""
        Ax = np.zeros((num_nodes, num_nodes), dtype=np.float64)
        Ay = np.zeros((num_nodes, num_nodes), dtype=np.float64)
        bx = np.zeros(num_nodes, dtype=np.float64)
        by = np.zeros(num_nodes, dtype=np.float64)
        for (ni, nj, w0) in b2b_edges:
            if linear:
                wx = w0 / max(abs(cxr[ni] - cxr[nj]), EPS_WL)
                wy = w0 / max(abs(cyr[ni] - cyr[nj]), EPS_WL)
            else:
                wx = wy = w0
            Ax[ni, ni] += wx; Ax[nj, nj] += wx; Ax[ni, nj] -= wx; Ax[nj, ni] -= wx
            Ay[ni, ni] += wy; Ay[nj, nj] += wy; Ay[ni, nj] -= wy; Ay[nj, ni] -= wy
        for (ni, px, py, w0) in p2b_edges:
            if linear:
                wx = w0 / max(abs(cxr[ni] - px), EPS_WL)
                wy = w0 / max(abs(cyr[ni] - py), EPS_WL)
            else:
                wx = wy = w0
            Ax[ni, ni] += wx; bx[ni] += wx * px
            Ay[ni, ni] += wy; by[ni] += wy * py
        # Dirichlet: pin preplaced nodes to their fixed center.
        for nf, cxf in node_fixed_x.items():
            cyf = node_fixed_y[nf]
            for k in range(num_nodes):
                if k != nf:
                    bx[k] -= Ax[k, nf] * cxf; Ax[k, nf] = 0.0; Ax[nf, k] = 0.0
                    by[k] -= Ay[k, nf] * cyf; Ay[k, nf] = 0.0; Ay[nf, k] = 0.0
            Ax[nf, nf] = 1.0; bx[nf] = cxf
            Ay[nf, nf] = 1.0; by[nf] = cyf
        # Weak spring to chip centre for otherwise-disconnected nodes.
        for k in range(num_nodes):
            if abs(Ax[k, k]) < 1e-12:
                Ax[k, k] += 1e-3; bx[k] += 1e-3 * (chip_side / 2.0)
            if abs(Ay[k, k]) < 1e-12:
                Ay[k, k] += 1e-3; by[k] += 1e-3 * (chip_side / 2.0)
        try:
            cxs = np.linalg.solve(Ax, bx)
            cys = np.linalg.solve(Ay, by)
        except np.linalg.LinAlgError:
            cxs = _grid_init(num_nodes, node_widths, node_heights, chip_side)
            cys = _grid_init(num_nodes, node_heights, node_widths, chip_side)
        for nf, cxf in node_fixed_x.items():
            cxs[nf] = cxf; cys[nf] = node_fixed_y[nf]
        return cxs, cys

    if has_anchors:
        cx, cy = _build_solve(None, None, linear=False)   # quadratic init
        for _ in range(n_wl_iters):                        # B2B reweight → HPWL
            cx, cy = _build_solve(cx, cy, linear=True)
    else:
        # No anchors: Laplacian is rank-deficient. Use grid init.
        cx, cy = _grid_init_xy(num_nodes, node_widths, node_heights, chip_side)

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
    cx = np.asarray(cx, dtype=np.float64).copy()
    cy = np.asarray(cy, dtype=np.float64).copy()
    W = np.asarray(widths, dtype=np.float64)
    H = np.asarray(heights, dtype=np.float64)
    free = np.array([k not in fixed_x for k in range(n)])
    halfW = (W[:, None] + W[None, :]) / 2.0   # pairwise half-sum of widths
    halfH = (H[:, None] + H[None, :]) / 2.0

    for it in range(n_iters):
        # Vectorized pairwise repulsion (equivalent to the O(n²) loop above).
        DX = cx[:, None] - cx[None, :]
        DY = cy[:, None] - cy[None, :]
        OX = halfW - np.abs(DX)
        OY = halfH - np.abs(DY)
        mask = (OX > 0) & (OY > 0)
        np.fill_diagonal(mask, False)
        overlap = np.minimum(OX, OY)
        dist = np.sqrt(DX * DX + DY * DY) + 1e-9
        coef = np.where(mask, overlap / dist, 0.0)
        fx = (coef * DX).sum(axis=1)
        fy = (coef * DY).sum(axis=1)

        new_cx = np.clip(cx + step * fx, W / 2.0, chip_side - W / 2.0)
        new_cy = np.clip(cy + step * fy, H / 2.0, chip_side - H / 2.0)
        cx = np.where(free, new_cx, cx)   # fixed nodes never move
        cy = np.where(free, new_cy, cy)

        step *= 0.95  # cool the step size

    return cx, cy
