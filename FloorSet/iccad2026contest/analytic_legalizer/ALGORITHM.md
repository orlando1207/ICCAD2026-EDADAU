# ICCAD 2026 Contest C — Analytic + Skyline-Legalization Floorplanner

## Overview

Solves fixed-outline rectangular floorplanning for FloorSet-Lite (n = 5–120 blocks).
Design is driven by the e^n-weighted score: test 99 (n=120) ≈ 64% of total weight,
test 98 ≈ 23%, test 97 ≈ 9% — the largest cases are essentially the whole score.

**Current results:** 100/100 feasible, weighted total score ≈ **1.827** (`N_STARTS=1`,
~3 s for all 100 cases). Optional `N_STARTS=4` with an area·HPWL selector can lower it
further at ~4× runtime.

Pipeline: `parse → MIB unify → boundary-aware cluster pre-pack → analytic place →
skyline legalize → gap-fill finetune → boundary slide → hard enforce`.

> The legacy longest-path/shaping/compact modules (`topology.py`, `shaping.py`) are
> retained for reference and unit tests but are **no longer on the active path** —
> they were replaced by the skyline legalizer (much denser and ~8–30× faster).

---

## Score Formula

```
Cost = (1 + 0.5·(HPWL_gap + Area_gap)) · e^(2·V_rel) · max(0.7, R^0.3)
```

- **Infeasible** (any overlap / dim violation) = 10.0.
- `V_rel = (V_grouping + V_boundary + V_mib) / N_soft`.
- `HPWL_gap`, `Area_gap` = relative to the per-case baseline (gentle, ×0.5).
- `R` = runtime ratio vs the field median (only `^0.3`, capped at 0.7).

**Priorities, as the design currently stands:**
1. Feasibility is absolute (a feasible-but-bad case can still cost >10 via Area_gap).
2. **HPWL_gap is now the dominant term** (~2× baseline wirelength on big cases) — larger
   than Area_gap or V_rel. Any layout-selection or tuning decision must account for it.
3. Area_gap is second; the skyline packer keeps it small (≈0.3 on the big cases).
4. V_rel is now small (boundary mostly satisfied via the fixed container edges).

---

## Constraint Types

| Column | Constraint | Handling |
|--------|-----------|----------|
| `constraints[:,0]` | Fixed shape — (w,h) locked | `is_fixed_shape`; never reshaped |
| `constraints[:,1]` | Preplaced — (x,y,w,h) locked | `is_preplaced`; fixed obstacle |
| `constraints[:,2]` | MIB group id | members share (w,h) — Step 1 |
| `constraints[:,3]` | Cluster/grouping id | rigid super-block |
| `constraints[:,4]` | Boundary code | LEFT=1, RIGHT=2, TOP=4, BOTTOM=8, corners=OR |

---

## Pipeline

### Step 0+1 — Parse & classify (`constraints.py: parse_and_init`)
- Classify each block (preplaced / fixed-shape / soft); build `mib_groups`, `cluster_groups`.
- **Detach preplaced members from clusters** (set `cluster_group=0`): a rigid super-block
  pinned at a preplaced coordinate collides with other pinned clusters. Concedes ≤1
  grouping violation per affected cluster (the evaluator still expects all original
  members connected), which is unavoidable for a rigid box pinned away from the rest.
- Initial shapes: preplaced/fixed exact; soft blocks square `w=h=√area`.
- **MIB unification** (`_unify_mib_shapes`): all members adopt one (w,h) — the anchor's if
  any preplaced/fixed member exists, else a square from mean area. Gives V_mib=0.

### Step 2 — Boundary-aware cluster pre-pack (`constraints.py: prepack_clusters`)
Each cluster → one rigid `SuperBlock(members, offsets, w, h)`.
- **No boundary members:** shelf-pack (sort by height, fill rows of width ≈ √area).
- **Has boundary members:** `_boundary_aware_pack` frame layout — BOTTOM/TOP members in
  rows on the bottom/top faces, LEFT/RIGHT members in side columns, interior (no-boundary)
  members shelf-packed in the centre. This is what lets a member actually touch the wall
  once the box is placed against it (a member only touches a face it sits on). Conflicting
  codes within one cluster are best-effort (boundary is soft).
- Super-block inherits the OR of member boundary codes.

### Step 3 — Analytic placement (`quadratic_placer.py: analytic_place`)
- **Bound2Bound HPWL model:** start from a quadratic solve `Σ w·(Δx²+Δy²)`, then iterate
  (`n_wl_iters=3`) reweighting each edge `w ← w0 / max(|Δ|, ε)` per axis → the quadratic
  minimiser converges to the *linear* HPWL optimum (the score's metric). Separate Ax/Ay
  systems (per-axis weights); preplaced = Dirichlet anchors; clusters = single node.
- **Light spreading (`n_spread_iters=10`).** Heavy spreading is counterproductive here: it
  scrambles the WL-optimal solve, and the skyline legalizer removes overlap itself. Going
  from 30→10 (and adding B2B) lowered the score ~1.97→1.86.
- Output: per-block centers `cx, cy` (the *guide* for legalization).
- Seed 0 = no noise (deterministic). Seeds 1+ add Gaussian noise (only if `N_STARTS>1`).

### Step 4 — Skyline legalization (`skyline_legalizer.py: skyline_legalize`) — core
Deterministic constructive strip-packing guided by the analytic positions.
- **Container width `W`:** try a ladder of aspects (H/W) `{analytic-bbox, 0.4, 0.6, 0.8,
  1.0, 1.3, 1.6, 2.0, 2.5}` → `W=√(area/aspect)`, clamped to `W_min` (widest unit / max
  preplaced right edge). Includes <1 (wide) so wide-preferring instances are reachable.
  Each candidate is scored by **area · HPWL · e^(2·boundary)** and the min is kept — HPWL
  is the dominant cost term, so an area-only proxy mis-picks (too-narrow boxes with worse
  wirelength; e.g. case 98 it chose aspect 2.5 over the better 2.0).
  Pack each, keep min `area·e^(2·boundary/n)`. A *fixed* W gives real L/R/B edges (the
  emergent-bbox of the old packer could not).
- **`Skyline`:** contour of contiguous `[x_start, x_end, height]` segments over `[0,W]`.
- **Placement** (units = free blocks + rigid cluster boxes), sorted by analytic `(cy, cx)`
  with BOTTOM-forced first, TOP-forced last: for each candidate x, landing
  `y = max skyline under [x,x+w]`, scored `y + λ·|x_center − cx|` (`λ=0.3`; density vs
  HPWL). Place at min score; raise the skyline.
- **Preplaced obstacles** = reserved rectangles: `_land_y` bumps a unit up over any
  obstacle it would intersect, so blocks may sit *below/beside* a floating preplaced block.
- **Boundary edge rules:** LEFT → x=0; RIGHT → x=W−w; BOTTOM → placed first (reaches y=0);
  TOP → best-effort (placed last + `slide_boundary`).
- **Cluster gaps exposed:** a cluster is placed as a box, but the skyline is set to its
  *member contour* (not a flat box top) and members are registered as obstacles — so later
  blocks nest into the cluster's internal pockets (grouping unaffected; the evaluator only
  unions a group's own members).

### Step 5 — Gap-fill finetune (`skyline_legalizer.py: _finetune_fill_gaps`)
Post-pass: relocate bbox-defining (frontier) free blocks into cluster-internal gaps **only
if it strictly shrinks the bbox**. Conservative — preserves the main packing's positions
and HPWL; can't regress.

### Step 5.5 — HPWL detailed placement (`skyline_legalizer.py: _detailed_place`)
Post-legalization wirelength cleanup: slide each **interior** free block (not preplaced,
not in a cluster, **no boundary code**) toward the weighted median of its connected
neighbours' centres — the HPWL-optimal point — clipped to stay overlap-free. Moving toward
the median is monotone non-increasing in HPWL, so it's safe and convergent (a few passes).
Boundary blocks are excluded: dragging one off its edge spikes V_rel (the `e^2·` term) far
more than the HPWL gain is worth. Full-100: ~1.859 → ~1.834.

### Step 6 — Boundary slide (`constraints.py: slide_boundary`)
Slide remaining RIGHT/TOP (and corner) blocks to the bbox edge, clipping to avoid new
overlaps. LEFT/BOTTOM already satisfied by packing. Mostly mops up TOP (the soft, hard case).

### Step 7 — Hard enforcement (`constraints.py: enforce_hard`)
Final guarantee (should be near-no-op after the skyline pack, kept as a safety net):
- **8a** restore preplaced/fixed exact dims & positions.
- **8b** nudge soft-block h to hit exact area (≤1.1%).
- **8c** `_resolve_overlaps` (≤80 passes, push movable/rigid-cluster units) + `_escape_overlaps`.
- **8d** `_snap_cluster_abutment`: align intra-cluster abutting edges to bit-identical values
  (float non-associativity left them ~1 ULP apart → shapely counted spurious components).

### Multistart selection (`optimizer.py`)
For `N_STARTS` seeds, run the full pipeline (Steps 3–7) and keep the layout with the
smallest **area·HPWL** proxy (both computable from the solution + connectivity, no baseline).
HPWL is the dominant term, so selecting on area·HPWL matches the oracle (best actual cost);
the older min-area / area·e^(2·boundary) proxies ignored HPWL and picked *worse* layouts.
Default `N_STARTS=1` (fast, deterministic); set to 4 for the best-quality run.

---

## Key Data Structures

```python
@dataclass
class BlockInfo:
    idx: int; w: float; h: float
    fixed_x: Optional[float]; fixed_y: Optional[float]
    is_preplaced: bool; is_fixed_shape: bool
    mib_group: int        # 0 = none
    cluster_group: int    # 0 = none; members[0] is the rep
    boundary_code: int    # OR of LEFT=1, RIGHT=2, TOP=4, BOTTOM=8

@dataclass
class SuperBlock:
    members: List[int]                  # original block indices
    offsets: List[Tuple[float, float]]  # (dx, dy) per member from box LL
    w: float; h: float
    boundary_code: int                  # OR of member codes
```

---

## Parameter Summary

| Parameter | Location | Value | Purpose |
|-----------|----------|-------|---------|
| `N_STARTS` | `optimizer.py` | 1 | analytic seeds (set 4 for area·HPWL multistart) |
| `NOISE_STD` | `optimizer.py` | 0.12 | per-seed Gaussian noise (only if N_STARTS>1) |
| `n_wl_iters` | `quadratic_placer.py` | 3 | Bound2Bound HPWL reweighting iterations |
| `n_spread_iters` | `quadratic_placer.py` | 10 | analytic spreading iterations (kept low) |
| `lam` (λ) | `skyline_legalizer.py` | 0.3 | skyline density-vs-HPWL weight |
| aspect ladder | `skyline_legalizer.py` | {bbox,0.4,0.6,0.8,1.0,1.3,1.6,2.0,2.5} | candidate widths (H/W; <1 = wide) |
| finetune passes | `skyline_legalizer.py` | 6 | gap-fill refinement passes |
| resolve passes | `constraints.py` | 80 | overlap resolution before escape |
| area nudge limit | `constraints.py` | 1.1% | max relative area correction (8b) |
| ULP snap tol | `constraints.py` | 1e-6 | intra-cluster abutment snap (8d) |

---

## ML Leverage Points

Deterministic baseline; each stage can be replaced/augmented by a learned component.

| Stage | Current method | ML opportunity |
|-------|---------------|----------------|
| Initial positions (Step 3) | Quadratic wirelength + spread | GNN to predict good placement / reduce HPWL |
| Container width (Step 4) | Fixed aspect ladder | Predict the best W/aspect per instance |
| Skyline order & λ (Step 4) | analytic order, fixed λ | Learn placement order / per-block λ for HPWL |
| Cluster pre-pack (Step 2) | Shelf / boundary-frame heuristic | Predict member arrangement from connectivity |
| Best-start selection | area·HPWL proxy over seeds | Learn a cost surrogate / which seed to trust |

---

## Lessons / failure-mode reference (this project)

1. **HPWL is the dominant cost term now** — every selection/tuning decision must include it.
   Min-area and area·e^(2·boundary) selectors actively pick worse layouts.
2. **Grouping ULP bug** — rigid clusters fragmented only because abutting member edges fell
   ~1 ULP apart (float non-associativity at large coords); `_snap_cluster_abutment` fixes it.
3. **Skyline obstacle walling** — stamping a preplaced block as a full-height skyline column
   (instead of a reserved rect with bump-up) forced layout height to ~2×. Use reserved rects.
4. **Cluster boundary = de-group, not placement** — 82% of cluster-member boundary misses had
   the super-block already at the correct wall; the member was just buried. Fix in pre-pack
   (frame layout), not in placement.
5. **Boundary vs density are coupled** — both feed V_rel; a denser layout that strands a
   boundary block can be a net loss. The fixed-container edges resolve most of this.
6. **The e^n weight makes the full-100 average the wrong metric** — optimize the largest cases.
