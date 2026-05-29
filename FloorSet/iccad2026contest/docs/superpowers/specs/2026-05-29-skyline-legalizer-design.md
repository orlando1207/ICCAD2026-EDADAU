# Skyline Legalizer — Design Spec

**Date:** 2026-05-29
**Status:** approved (design), pending implementation plan
**Project:** ICCAD 2026 Contest C floorplanner (FloorSet-Lite), `analytic_legalizer/`

## Problem

The current pipeline is 100/100 feasible at weighted score **3.7179**, but its layouts are
only **30–46% utilized** versus the ground truth's ~**96%** (the GT is a dense ~2:1-tall brick
wall). The dominant cost term is `area_gap` (case 99 = 1.12), driven entirely by this whitespace.

Root cause (established by experiment): the longest-path-from-analytic-spread legalizer
(`build_topology` → `longest_path_pack` → `compact`) **preserves the analytic spread** and cannot
densify. Topology edge tweaks plateau ~3.71–3.72; balancing diagonal edges improves area but blows
up case 98's boundary/`V_rel` (cost ∝ `e^(2·V_rel)`), so density and boundary are coupled and the
incremental well is dry. The right edge is also **emergent** in this scheme, so RIGHT/TOP boundary
blocks have no defined edge to reach (boundary stuck ~950 over all 100).

## Goal

Replace the **legalization stage only** (keep analytic placement) with a **deterministic,
one-pass, constructive** legalizer that packs densely like the GT brick wall while protecting every
hard constraint. Success = weighted score below 3.7179 with 100/100 feasible, driven by utilization
rising toward GT's ~96%; HPWL and boundary must not regress.

## Decisions (locked)

- **Scope:** replace legalization only. Keep `parse_and_init`, `prepack_clusters`, `analytic_place`.
  Remove `build_topology`/`longest_path_pack`/`augment_topology`/`shape_soft_blocks`/`compact` from
  the active path. Keep `slide_boundary` (reduced to TOP best-effort) and `enforce_hard` (safety net).
- **Style:** deterministic constructive (reproducible, fast — fits the "deterministic baseline for ML
  augmentation" project goal and avoids hurting the runtime score factor).
- **Algorithm:** skyline strip-packing guided by analytic positions (Approach B).

## Architecture & pipeline placement

```
parse_and_init → prepack_clusters → analytic_place        [KEEP]
  ↓
  skyline_legalize(...)                                    [NEW — replaces middle stages]
  ↓
slide_boundary (TOP best-effort) → enforce_hard            [KEEP as safety net]
```

Single entry point:
```
skyline_legalize(blocks, super_blocks, cluster_groups,
                 analytic_cx, analytic_cy, area_targets) -> List[(x, y, w, h)]
```
Output uses the existing `positions` contract; `enforce_hard` remains the final hard-constraint
guarantee (should become near-no-op but stays so we can never emit an infeasible result).

## Core packing algorithm

A **skyline strip-packer** with fixed container width `W` (height grows up). Skyline = list of
`(x_start, x_end, height)` segments = current top contour.

Place one unit (a block, or a rigid cluster as one `sb.w × sb.h` box):
1. Slide a width-`w` window across the skyline at each segment-boundary candidate x.
2. Landing `y` = max skyline height under the window (rests on highest touched point → no overlap).
3. Score candidate: `landing_y + λ·|x − cx_analytic|` (`landing_y`=density, `λ·|x−cx|`=HPWL keep).
4. Place at min-score position; raise covered segments to `y + h`.

Order = blocks/clusters sorted by analytic `cy` then `cx` (bottom-up, left-to-right) → keeps the
legalized layout close to analytic → bounds HPWL growth.

`λ` is the one primary tunable (density vs wirelength); start small (favor density), tune on harness.

Skyline fills bottom-up uniformly → no empty-quadrant failure mode → utilization set by `W`, not by
an emergent critical path.

## Constraint handling

| Constraint | Type | Handling |
|---|---|---|
| No overlap | hard | by construction (skyline + reserved-rect validity check) |
| Preplaced (x,y,w,h) | hard | stamped as fixed obstacles *before* packing; occupancy = skyline + reserved-rect list; if a landing intersects a preplaced rect, bump y above it and retry (worst case: lands above everything) |
| Fixed-shape (w,h) | hard | placed at exact size, never reshaped |
| Soft area ±1% | hard | each soft block keeps its `√area` square; packer **moves, never resizes** → tolerance never at risk |
| MIB (equal shapes) | soft | already unified upstream; placed as-is |
| Cluster/grouping | soft | rigid `sb.w×sb.h` box, members mapped back by offsets; ULP-snap in `enforce_hard` |
| Boundary | soft | container `[0,W]×[0,H]`: LEFT→x=0; BOTTOM→bottom (place first); RIGHT→x=W−w; TOP→best-effort (pack last, `slide_boundary` nudges) |

Only **TOP is best-effort** (no fixed top until packing ends; boundary is soft so acceptable). Every
hard constraint is protected by construction. L/R/B boundary become genuinely satisfiable thanks to
the fixed container edges — the thing the emergent-bbox pipeline structurally could not do.

### Analytic boundary bias (separately-validated step 3)

Add a **gentle** spring in `analytic_place` pulling each boundary block toward its target edge of the
analytic frame (RIGHT→`x≈chip_side`, TOP→`y≈chip_side`, …), weak relative to b2b/p2b terms, so the
block's connected neighbors drift with it (coherent relocation, not one stretched block). Re-trying
this is justified: it regressed (6.69) under the *old* legalizer where the edge was emergent; under
the fixed-container legalizer the bias and the legalizer finally point the same way. Tunable,
one-flag revert.

### Boundary for clusters

Super-blocks inherit the OR of member boundary codes (`prepack_clusters` already does `bc |= ...`).
A super-block with a LEFT/RIGHT/BOTTOM code is placed against that container edge as a rigid unit; the
analytic bias pulls the cluster's rep node. **Subtlety:** a boundary member only touches the edge if
it sits on the matching *side* of the rigid box, so **`prepack_clusters` becomes boundary-aware** —
RIGHT members along the cluster's right side, BOTTOM on the bottom row, etc. Conflicts (LEFT and RIGHT
member in one cluster) are best-effort (boundary is soft).

## Container width `W` selection

`W` sets aspect and density. Deterministic multi-try:
- `W_min = max(widest unit width, max preplaced right-edge x+w)`; all candidates clamped `≥ W_min`.
- Candidates: `W = sqrt(total_area / aspect)` for aspects `{analytic-bbox aspect, 1.0, 1.3, 1.6, 2.0,
  2.5}`, dedup, clamp. ~6 packer runs (trivial at n≤120). Candidates with `W < ` a preplaced right
  edge are skipped.
- For each: pack → `H` = final skyline top → score `= (W·H) · exp(2·boundary_unmet/n)`.
  `W·H` ∝ area_gap; the `exp` mirrors `e^(2·V_rel)` so we don't pick a compact-but-boundary-stranding
  layout (the mistake the old min-area selection made).
- Keep min-score candidate. Minimizing `W·H` = maximizing utilization = the "balanced, no dead corner"
  outcome.

## Module structure

New file **`analytic_legalizer/skyline_legalizer.py`**:
- `class Skyline` — `place_query(w, cx) -> (x, y, score)`, `add(x, y, w, h)`. Pure geometry,
  unit-testable (place rects → assert no overlap, assert tight fill).
- `_reserve_obstacles(preplaced)` — stamp fixed blocks.
- `_pack_one_width(units, W, λ)` — one deterministic pass (core algo + edge rules).
- `skyline_legalize(...)` — entry: build units, candidate-`W` loop, map clusters back, return positions.

Edits (small, contained):
- `optimizer.py` — swap middle stages for one `skyline_legalize(...)` call; keep slide_boundary +
  enforce_hard.
- `constraints.py` — boundary-aware `prepack_clusters`.
- `quadratic_placer.py` — optional tunable boundary bias (layered in after core is proven).
- `topology.py` — retired from active path (kept in repo).

## Validation

After every increment, via `score_harness.py`:
1. **Hard gate:** full-100 stays 100/100 feasible — revert anything that breaks it.
2. **Primary:** weighted score on 95–99 (and full-100) beats 3.7179; utilization rises toward ~96%.
3. **Watch:** HPWL gap and boundary count don't regress.
4. **Visual:** OUR-vs-GT render on 97/98/99 should look like a brick wall.

Build order (each measured, each revertible):
1. Skyline packer, `λ`-only, no boundary bias → confirm density + feasibility.
2. Boundary-aware `prepack_clusters`.
3. Analytic boundary bias.

## Risks / non-goals

- **Preplaced obstacles** are the inherent hard case for any constructive packer; they can force local
  gaps. Always feasible (grow `H` rather than violate). Not optimizing their neighborhoods beyond
  feasibility + reasonable fill.
- **HPWL** could rise if packing scrambles positions; mitigated by analytic-order packing + `λ`.
- **TOP boundary** not guaranteed (soft).
- Not changing the analytic placer's core solve (only the optional bias). Not pursuing stochastic/SA.
