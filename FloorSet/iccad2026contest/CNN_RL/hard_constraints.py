"""
Phase 6 — hard / placement constraints (VERSION_B_RL_PLACER.md §5 Phase 6).

Helpers to (a) turn ground-truth boxes into the contest-style target_positions
spec, and (b) verify a finished placement satisfies the hard constraints.

constraints[N,5] = (fixed, preplaced, mib_group, cluster_group, boundary_code).
boundary_code bits: LEFT=1 RIGHT=2 TOP=4 BOTTOM=8.

target_positions[N,4] = (x,y,w,h), -1 = free:
  * preplaced block -> all four set (location + dims immutable)
  * fixed-shape block -> (w,h) set, (x,y) = -1 (dims immutable, location free)

Boundary is checked against the canvas (the fixed outline) edges, which is the
outline the env packs into; this is the enforceable definition the env masks to.
"""

from __future__ import annotations

from typing import Dict

import torch

LEFT, RIGHT, TOP, BOTTOM = 1, 2, 4, 8


def make_target_positions_from_gt(constraints: torch.Tensor,
                                  gt_boxes: torch.Tensor,
                                  block_count: int) -> torch.Tensor:
    """Build the contest target_positions spec from GT boxes for the flagged
    blocks (used in the training/validation context where GT == the spec)."""
    tp = torch.full((block_count, 4), -1.0)
    for i in range(block_count):
        fixed = constraints[i, 0] > 0
        preplaced = constraints[i, 1] > 0
        if preplaced:
            tp[i] = gt_boxes[i]
        elif fixed:
            tp[i, 2] = gt_boxes[i, 2]
            tp[i, 3] = gt_boxes[i, 3]
    return tp


def touches_boundary(x, y, w, h, canvas_w, canvas_h, code, tol=1e-3) -> bool:
    """Does the block touch every edge required by `code`?"""
    need = []
    if code & LEFT:
        need.append(x <= tol)
    if code & RIGHT:
        need.append(abs((x + w) - canvas_w) <= tol)
    if code & BOTTOM:
        need.append(y <= tol)
    if code & TOP:
        need.append(abs((y + h) - canvas_h) <= tol)
    return all(need) if need else True


def check_hard_constraints(positions: torch.Tensor, constraints: torch.Tensor,
                           target_positions: torch.Tensor, block_count: int,
                           canvas: tuple, tol: float = 1e-2) -> Dict[str, int]:
    """Count hard-constraint violations of a finished placement.
    positions[N,4]=(x,y,w,h). Returns per-type violation counts (0 == satisfied)."""
    cw, ch = canvas
    v = {"fixed": 0, "preplaced": 0, "boundary": 0, "mib": 0}

    for i in range(block_count):
        x, y, w, h = (float(t) for t in positions[i])
        if constraints[i, 0] > 0:  # fixed-shape: dims must match spec
            tw, th = float(target_positions[i, 2]), float(target_positions[i, 3])
            if abs(w - tw) > tol or abs(h - th) > tol:
                v["fixed"] += 1
        if constraints[i, 1] > 0:  # preplaced: all four must match spec
            tx, ty, tw, th = (float(target_positions[i, k]) for k in range(4))
            if (abs(x - tx) > tol or abs(y - ty) > tol
                    or abs(w - tw) > tol or abs(h - th) > tol):
                v["preplaced"] += 1
        code = int(constraints[i, 4])
        # Preplaced blocks are locked to their given location; their boundary is
        # guaranteed by the spec in the true-bbox frame. Our heuristic canvas is
        # a different frame, so we don't re-judge boundary for them here (true
        # bbox-relative boundary is validated post-legalisation, Phase 7/8).
        if code > 0 and constraints[i, 1] <= 0 \
                and not touches_boundary(x, y, w, h, cw, ch, code, tol):
            v["boundary"] += 1

    # MIB: blocks sharing a group id must have identical (w,h)
    groups: Dict[int, list] = {}
    for i in range(block_count):
        g = int(constraints[i, 2])
        if g > 0:
            groups.setdefault(g, []).append(i)
    for g, members in groups.items():
        shapes = {(round(float(positions[i, 2]), 3), round(float(positions[i, 3]), 3))
                  for i in members}
        v["mib"] += max(len(shapes) - 1, 0)

    return v
