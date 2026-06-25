"""
Phase 15 PoC — slack-driven block shaping, measured on top of the current
default pipeline.

Goal (see ALGORITHM.md §"Aspect Ratio: Problem Analysis & Plan"):
quantify the CEILING of moving aspect-ratio decisions from "upfront, blind"
(Phase 13 head) to "after legalization, where whitespace is visible". This
script does NOT touch the model. It:

  1. runs the current default `RLSkylineOptimizer` to get legalized positions,
  2. applies a self-contained slack-shaping pass that reshapes *free* soft
     blocks (area-constant) into adjacent whitespace to shrink the floorplan's
     bounding box,
  3. reports Area_gap (and bbox area, feasibility) BEFORE vs AFTER per case.

The pass is deliberately geometry-only and conservative: it grows a soft block
on its slack axis ONLY into verified-empty space up to the current envelope
edge (so the envelope on that axis never grows), shrinks it area-constant on the
binding axis, then re-packs the binding axis toward the origin. Non-free blocks
(fixed/preplaced/MIB/cluster/boundary) are pinned and never moved or reshaped,
so hard constraints are preserved by construction. Any case where this still
introduces an overlap is reported (the PoC measures where the simple approach is
safe, it does not claim to be the final algorithm).

Run:
    cd FloorSet/iccad2026contest
    python3 CNN_RL/poc_slack_shaping.py --n 30
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch

_CONTEST_DIR = Path(__file__).resolve().parent.parent
if str(_CONTEST_DIR) not in sys.path:
    sys.path.insert(0, str(_CONTEST_DIR))
_DL_DIR = Path(__file__).resolve().parent
if str(_DL_DIR) not in sys.path:
    sys.path.insert(0, str(_DL_DIR))

from iccad2026_evaluate import (  # noqa: E402
    FloorplanDatasetLiteTest, calculate_bbox_area,
)
from rl_skyline_optimizer import RLSkylineOptimizer  # noqa: E402

TOL = 1e-6


# --------------------------------------------------------------------------- #
# Geometry helpers (operate on positions as a list of [x, y, w, h]).
# Axis 0 = x (width), axis 1 = y (height). lo(b,a)=b[a], size(b,a)=b[a+2].
# --------------------------------------------------------------------------- #
def _envelope(pos):
    xmin = min(b[0] for b in pos)
    ymin = min(b[1] for b in pos)
    xmax = max(b[0] + b[2] for b in pos)
    ymax = max(b[1] + b[3] for b in pos)
    return xmin, ymin, xmax, ymax


def _bbox_area(pos):
    xmin, ymin, xmax, ymax = _envelope(pos)
    return (xmax - xmin) * (ymax - ymin)


def _overlap_count(pos, tol=1e-4):
    """Count strictly-overlapping (not merely touching) block pairs."""
    n = len(pos)
    v = 0
    for i in range(n):
        xi, yi, wi, hi = pos[i]
        for j in range(i + 1, n):
            xj, yj, wj, hj = pos[j]
            ox = min(xi + wi, xj + wj) - max(xi, xj)
            oy = min(yi + hi, yj + hj) - max(yi, yj)
            if ox > tol and oy > tol:
                v += 1
    return v


def _reshape_axis(pos, soft, d, ar_max):
    """Reshape free soft blocks: grow on the perpendicular axis p=1-d into the
    verified-empty high-side gap (capped at the current envelope edge so the
    envelope on axis p never grows), shrink area-constant on the binding axis d.
    Only lo[d]/lo[p] are preserved; sizes change. No block moves here."""
    p = 1 - d
    env_p_max = max(b[p] + b[p + 2] for b in pos)
    for i in range(len(pos)):
        if not soft[i]:
            continue
        area = pos[i][2] * pos[i][3]
        lo_p, s_p = pos[i][p], pos[i][p + 2]
        hi_p = lo_p + s_p
        lo_d, hi_d = pos[i][d], pos[i][d] + pos[i][d + 2]

        # Nearest block on the high-p side that overlaps i along axis d (i.e.
        # would collide if i grew along p). Growth is bounded by it / envelope.
        limit = env_p_max
        for j in range(len(pos)):
            if j == i:
                continue
            bj_lo_d, bj_hi_d = pos[j][d], pos[j][d] + pos[j][d + 2]
            if min(hi_d, bj_hi_d) - max(lo_d, bj_lo_d) <= TOL:
                continue                                   # not beside i along d
            bj_lo_p = pos[j][p]
            if bj_lo_p >= hi_p - TOL:                      # sits on the high-p side
                limit = min(limit, bj_lo_p)

        max_s_p = min(limit - lo_p, math.sqrt(area * ar_max))   # gap + AR cap
        if max_s_p > s_p + TOL:
            pos[i][p + 2] = max_s_p
            pos[i][d + 2] = area / max_s_p                 # area-constant shrink on d


def _repack_axis(pos, soft, d):
    """Shadow-compaction along axis d: push free soft blocks toward the origin.
    Non-soft blocks are pinned (act as obstacles only). A block's new lo[d] is
    the highest top of any block that overlaps it on the perpendicular axis and
    currently sits below it."""
    p = 1 - d
    floor0 = min(b[d] for b in pos)
    order = sorted(range(len(pos)), key=lambda i: pos[i][d])
    for i in order:
        if not soft[i]:
            continue
        ai_lo, ai_hi = pos[i][p], pos[i][p] + pos[i][p + 2]
        new_lo = floor0
        cur_lo_d = pos[i][d]
        for j in range(len(pos)):
            if j == i:
                continue
            bj_lo, bj_hi = pos[j][p], pos[j][p] + pos[j][p + 2]
            if min(ai_hi, bj_hi) - max(ai_lo, bj_lo) <= TOL:
                continue                                   # no perpendicular overlap
            j_top = pos[j][d] + pos[j][d + 2]
            if j_top <= cur_lo_d + TOL:                    # j currently below i
                new_lo = max(new_lo, j_top)
        pos[i][d] = new_lo


def slack_shape(positions, soft_mask, ar_max=4.0, iters=8, reshape=True):
    """Iterative slack-driven shaping. Each pass shrinks the larger envelope
    dimension by reshaping free soft blocks into perpendicular slack, then
    re-packs that axis. Monotone on bbox area (stops when no improvement).
    With reshape=False it is a pure compaction control (movement only)."""
    pos = [list(map(float, b)) for b in positions]
    best = _bbox_area(pos)
    for _ in range(iters):
        xmin, ymin, xmax, ymax = _envelope(pos)
        d = 1 if (ymax - ymin) >= (xmax - xmin) else 0     # shrink larger side
        if reshape:
            _reshape_axis(pos, soft_mask, d, ar_max)
        _repack_axis(pos, soft_mask, d)
        cur = _bbox_area(pos)
        if cur < best - 1e-9:
            best = cur
        else:
            break
    return [tuple(b) for b in pos]


# --------------------------------------------------------------------------- #
# Measurement harness.
# --------------------------------------------------------------------------- #
def _free_mask(constraints, block_count, allow_boundary=False):
    """A block is 'free' (reshapeable) iff it has no fixed/preplaced/MIB/cluster
    constraint — columns [fixed, preplaced, mib_id, cluster_id, bound]. With
    allow_boundary, boundary-constrained blocks are also reshapeable (a boundary
    block may change shape as long as it keeps touching its edge — this pass
    keeps the lower corner fixed, so LEFT/BOTTOM touches are preserved)."""
    mask = []
    for i in range(block_count):
        c = constraints[i]
        free = (float(c[0]) == 0 and float(c[1]) == 0 and int(c[2]) == 0
                and int(c[3]) == 0)
        if not allow_boundary:
            free = free and int(c[4]) == 0
        mask.append(free)
    return mask


def _opt_target_pos(constraints, target_pos, block_count):
    """Replicate the evaluator's opt_target_pos construction so the optimizer
    sees the same fixed/preplaced specs as during real --evaluate."""
    out = torch.full((block_count, 4), -1.0)
    if target_pos is None or constraints is None:
        return out
    nc = constraints.shape[1] if constraints.dim() > 1 else 0
    for i in range(block_count):
        is_fixed = nc > 0 and constraints[i, 0] != 0
        is_pre = nc > 1 and constraints[i, 1] != 0
        if is_pre:
            tx, ty, tw, th = target_pos[i]
            out[i] = torch.tensor([tx, ty, tw, th])
        elif is_fixed:
            _, _, tw, th = target_pos[i]
            out[i, 2] = tw
            out[i, 3] = th
    return out


def _area_baseline(labels, block_count):
    polygons, metrics = labels
    pos = []
    for i in range(block_count):
        block = polygons[i]
        valid = block[block[:, 0] != -1]
        if len(valid) > 0:
            x0, y0 = valid.min(dim=0).values
            x1, y1 = valid.max(dim=0).values
            pos.append((float(x0), float(y0), float(x1 - x0), float(y1 - y0)))
        else:
            pos.append((0, 0, 1, 1))
    area = calculate_bbox_area(pos)
    if metrics is not None and len(metrics) >= 8 and metrics[0] > 0:
        area = float(metrics[0])
    return area


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30, help="number of cases to measure")
    ap.add_argument("--ar-max", type=float, default=4.0, help="max aspect ratio cap")
    ap.add_argument("--data-path", type=str, default="../")
    ap.add_argument("--all-free", action="store_true",
                    help="CEILING probe: treat EVERY block as reshapeable "
                         "(ignores hard constraints; measures the upper bound "
                         "of the shaping lever, not a feasible result)")
    ap.add_argument("--allow-boundary", action="store_true",
                    help="also reshape boundary-constrained blocks (keeps lower "
                         "corner fixed, so LEFT/BOTTOM edge touches are preserved)")
    ap.add_argument("--no-reshape", action="store_true",
                    help="control: disable reshaping, run pure compaction "
                         "(movement only) — isolates shaping's contribution")
    args = ap.parse_args()

    ds = FloorplanDatasetLiteTest(args.data_path)
    n_cases = min(args.n, len(ds))
    opt = RLSkylineOptimizer(verbose=False)

    print(f"{'case':>4} {'blk':>4} {'free':>4} {'gap_before':>11} "
          f"{'gap_after':>10} {'Δgap':>9} {'newOvl':>7}")
    print("-" * 56)

    sum_before = sum_after = 0.0
    n_improved = n_unsafe = 0
    tot_blk = tot_free = 0
    deltas = []

    for idx in range(n_cases):
        sample = ds[idx]
        inputs, labels = sample["input"], sample["label"]
        area_target, b2b, p2b, pins, constraints = inputs
        block_count = int((area_target != -1).sum().item())
        polygons = labels[0]
        target_pos = []
        for i in range(block_count):
            blk = polygons[i]
            valid = blk[blk[:, 0] != -1]
            if len(valid) > 0:
                x0, y0 = valid.min(dim=0).values
                x1, y1 = valid.max(dim=0).values
                target_pos.append((float(x0), float(y0),
                                   float(x1 - x0), float(y1 - y0)))
            else:
                target_pos.append((-1, -1, -1, -1))
        otp = _opt_target_pos(constraints, target_pos, block_count)

        positions = opt.solve(block_count, area_target, b2b, p2b, pins,
                              constraints, otp)
        soft = ([True] * block_count if args.all_free
                else _free_mask(constraints, block_count,
                                allow_boundary=args.allow_boundary))
        base = _area_baseline(labels, block_count)

        bbox_b = calculate_bbox_area(positions)
        ovl_b = _overlap_count(positions)
        shaped = slack_shape(positions, soft, ar_max=args.ar_max,
                             reshape=not args.no_reshape)
        bbox_a = calculate_bbox_area(shaped)
        ovl_a = _overlap_count(shaped)

        gap_b = (bbox_b - base) / max(base, 1e-6)
        gap_a = (bbox_a - base) / max(base, 1e-6)
        new_ovl = max(0, ovl_a - ovl_b)

        sum_before += gap_b
        sum_after += gap_a
        deltas.append(gap_b - gap_a)
        tot_blk += block_count
        tot_free += sum(soft)
        if gap_a < gap_b - 1e-9:
            n_improved += 1
        if new_ovl > 0:
            n_unsafe += 1

        print(f"{idx:>4} {block_count:>4} {sum(soft):>4} {gap_b:>11.4f} "
              f"{gap_a:>10.4f} {gap_b-gap_a:>9.4f} {new_ovl:>7}")

    print("-" * 56)
    print(f"cases               : {n_cases}")
    print(f"mean Area_gap before: {sum_before / n_cases:.4f}")
    print(f"mean Area_gap after : {sum_after / n_cases:.4f}")
    print(f"mean Δ Area_gap     : {sum(deltas) / n_cases:.4f}  "
          f"(positive = improvement)")
    print(f"cases improved      : {n_improved}/{n_cases}")
    print(f"cases w/ NEW overlap: {n_unsafe}/{n_cases}  "
          f"(slack pass unsafe here)")
    print(f"avg free/total blk  : {tot_free}/{tot_blk} "
          f"({100*tot_free/max(tot_blk,1):.1f}% reshapeable)")


if __name__ == "__main__":
    main()
