# ICCAD 2026 Contest C — Analytic + Legalization Floorplanner

## Overview

This pipeline solves fixed-outline rectangular floorplanning for the ICCAD 2026 Contest
(FloorSet-Lite benchmark, n = 5–120 blocks). The design is driven by the e^n-weighted
score: test 99 (n=120) carries ~73% of total weight, test 98 (n=119) carries ~27% —
everything else is negligible.

**Achieved results:** 100/100 feasible, total score ≈ 6.84.

---

## Score Formula

```
Cost = (1 + 0.5·(HPWL_gap + Area_gap)) · e^(2·V_rel) · max(0.7, R^0.3)
```

- **Infeasible** (overlap/dim violation) = 10.0, regardless of other terms.
- `V_rel = (V_grouping + V_boundary + V_mib) / N_soft` — exponential penalty.
- `HPWL_gap`, `Area_gap` — linear, gentle (×0.5).
- `R` = runtime ratio (capped, only hurts badly on very slow solutions).

**What this means for priorities:**
1. Feasibility first — a feasible-but-bad case can still cost >10 if Area_gap is huge.
2. Minimize `V_rel` — one soft violation multiplied by e^2 can double cost.
3. Minimize Area_gap on test 99 and test 98 — they are the entire score.
4. HPWL_gap and small V_rel are secondary.

---

## Constraint Types

| Code | Constraint | How handled |
|------|-----------|-------------|
| `constraints[:,0]` | Fixed shape — (w,h) locked | `is_fixed_shape=True`; never reshaped |
| `constraints[:,1]` | Preplaced — (x,y,w,h) all locked | `is_preplaced=True`; fixed anchor in solver |
| `constraints[:,2]` | MIB group ID | All members get same (w,h) in Step 1 |
| `constraints[:,3]` | Cluster/grouping ID | Rigid super-block in Steps 2–5 |
| `constraints[:,4]` | Boundary code | LEFT=1, RIGHT=2, TOP=4, BOTTOM=8, corners=OR |

---

## Pipeline

### Step 0+1 — Parse & Classify (`constraints.py: parse_and_init`)

**What it does:**
1. Reads `constraints[n,5]` and `target_positions[n,4]`.
2. Classifies each block: preplaced / fixed-shape / soft.
3. Builds `mib_groups: {id → [block indices]}` and `cluster_groups: {id → [block indices]}`.
4. **Detaches preplaced members from their clusters:** if a preplaced block is in a
   cluster, it is removed from the cluster and set as an independent pinned node
   (`cluster_group = 0`). If fewer than 2 non-preplaced members remain, the cluster is
   dissolved entirely. *This is critical: without detachment, two clusters that each
   contain a preplaced block at similar x can form rigid columns at the same strip →
   unavoidable overlap.*
5. Assigns initial shapes: preplaced/fixed exact from `target_positions`; MIB groups
   share `w = h = sqrt(mean_area)`; free soft blocks square `w = h = sqrt(area)`.

**MIB unification (V_mib = 0 by construction):**
- If the group contains a preplaced/fixed anchor, ALL members adopt the anchor's (w,h).
- If multiple anchors with different shapes: pick the one with smaller area (conservative).
- Otherwise: square shape from mean area.

---

### Step 2 — Cluster Pre-pack (`constraints.py: prepack_clusters`)

**What it does:**  
For each cluster group, packs members into a tight rectangle using a shelf/row strategy:
1. Sort members by area (largest first) for a tighter packing.
2. Assign shelf rows: place members left-to-right, start new row when row width exceeds
   `sqrt(total_area) * 1.4` (target aspect ratio close to 1.0).
3. Compute member offsets `(dx, dy)` relative to the super-block origin.
4. Store as a `SuperBlock(members, offsets, w, h)`.

The super-block replaces all members as a single node in Steps 3–5.

---

### Step 3 — Analytic Placement (`quadratic_placer.py: analytic_place`)

**What it does:**  
Quadratic wirelength minimization (WL ∝ Σ w_e [(cx_i - cx_j)²]) with spreading.

**Node mapping:**
- Preplaced blocks → fixed Dirichlet boundary conditions.
- Cluster super-blocks → single node at super-block center.
- Free/fixed-shape blocks → free nodes.

**Solver:**  
Builds a sparse positive-definite linear system `A·x = b` from `b2b_connectivity`
and `p2b_connectivity` (pin-to-block). Solves with `numpy.linalg.solve` (n ≤ 120).

**Spreading** (`n_spread_iters = 30`):  
Iterative pairwise repulsion to reduce overlap. Each iteration adds repulsive forces
proportional to overlap between pairs. Simple Euler integration at each step.

**Output:** `cx[n], cy[n]` — center coordinates for all original blocks.

---

### Step 4 — Topology Extraction (`topology.py: build_topology`)

**What it does:**  
Builds a cycle-free Horizontal Constraint Graph (HCG) and Vertical Constraint Graph (VCG)
from analytic centers.

**Algorithm:**
1. Compute representative (`rep`) for each block: cluster members → `members[0]`.
2. For the HCG: sort reps by `cx`. For each pair `(i, j)` with `cx[i] < cx[j]`:
   - If y-projections overlap → add HCG edge `i → j`.
   - If diagonally separated (no projection overlap) → add one HCG edge to prevent both
     landing at x=0.
3. For the VCG: same, using `cy` and x-projections.
4. Boundary hints: blocks with `BOUND_LEFT` get no HCG in-edges (land at x=0).
   Blocks with `BOUND_BOTTOM` get no VCG in-edges (land at y=0).
5. Preplaced obstacles: insert explicit edges so their neighbors respect the fixed box.

**Acyclicity guarantee:** Edges only go in increasing cx (HCG) or cy (VCG) order, so
Kahn's topological sort always completes.

---

### Step 5 — Longest-Path Packing (`topology.py: longest_path_pack`)

**What it does:**  
Assigns `x = longest_path(HCG)`, `y = longest_path(VCG)` → guaranteed overlap-free,
compacted toward the lower-left.

**Preplaced pinning:**  
Forward + backward propagation ensures preplaced blocks land at their exact (x, y).
Backward pass: if successor is pinned, push predecessor back: `max_u_x = pin_x - w_u`.

**Convergence:** `2*(len(reps)+1)` passes for both forward and backward.

**Clamping:** After propagation, all non-preplaced reps are clamped to x ≥ 0, y ≥ 0,
followed by one final forward pass to re-establish ordering from clamped values.

**Cluster expansion:** Each cluster member placed at `(sb_origin + offset)`.

**Step 5b — Augment + Re-pack loop** (`augment_topology`, repeated ≤ 15 times):  
After packing, scan for residual overlaps. For each overlapping pair, insert the missing
HCG or VCG edge (whichever requires smaller block movement). Stop when no new edges
are added. This handles cases where the initial topology was insufficient.

---

### Step 6 — Soft-Block Shaping (`shaping.py: shape_soft_blocks`)

**What it does:**  
Iteratively adjusts soft-block aspect ratios (w↔h, area conserved) to balance the
HCG and VCG critical paths, shrinking the bounding box.

**Algorithm** (`n_iters = 20`):
1. Compute current `bbox_x2`, `bbox_y2`.
2. For each free soft block:
   - `on_hcg` if `x + w ≥ bbox_x2 - ε`; `on_vcg` if `y + h ≥ bbox_y2 - ε`.
   - If on HCG but NOT VCG: narrow w (→ taller) by ×0.95 while respecting VCG slack.
   - If on VCG but NOT HCG: shrink h (→ wider) by ×0.95 while respecting HCG slack.
   - If on both or neither: skip.
3. Stop when no block improves.

**Parameters:**
- `ar_min = 0.2`, `ar_max = 5.0` — aspect ratio bounds
- `step_factor = 0.95` — conservative step size to avoid overshooting

**MIB lock:** `_apply_shape` locks the entire MIB group if ANY member is preplaced,
fixed-shape, or cluster-locked. A preplaced member's (w,h) is a hard constraint, and
MIB requires all members share the same shape.

**Step 6b — Re-legalize after shaping** (longest_path_pack + augment loop ≤ 15 times):  
Block dimensions changed → re-run topology packing to restore overlap-free layout.

---

### Step 6c — Directional Shadow Compaction (`topology.py: compact`)

**What it does:**  
Removes whitespace left by the longest-path packing without re-serializing blocks.
Alternates x-packing and y-packing, each time building a lean per-axis constraint graph.

**Key insight:** `build_topology` adds diagonal-separation edges for every pair without
projection overlap. This over-serializes: blocks that cannot possibly overlap on the
packed axis are still forced into sequence. `compact` adds edges ONLY between pairs
whose perpendicular projections currently overlap — the minimum set needed for
overlap-freedom.

**Algorithm** (`rounds = 12`):
1. Compute representative extents from current member positions (robust to reshaping).
2. For each round:
   - **x-pack (`_pack_axis(axis=0)`):** Build lean HCG (edges only where y-projections
     overlap). Longest-path with preplaced reps pinned → new x-coordinates.
   - **y-pack (`_pack_axis(axis=1)`):** Same, using x-projections.
   - Early stop if bbox changes < 1e-6 in both dimensions.

**`_pack_axis` detail:**
- Sort reps by current low coordinate on packed axis.
- For each pair `(a, b)` where `lo[a] < lo[b]` AND perpendicular extents overlap:
  add edge `a → b`.
- Forward + backward propagation: `2*(n_reps+1)` passes.
- Backward pass: `new_lo[u] = min(new_lo[v] - size[u])` over pinned/placed successors.
- Clamp non-pinned to ≥ 0; one final forward pass.
- Shift each rep's members rigidly by `(new_lo - old_lo)`.

**Effect:** On large cases (n=119, n=120), reduces area_gap from ~3.9 to ~1.1.
Layout utilization goes from ~13–21% to ~45–85%.

---

### Step 7 — Boundary Slide (`constraints.py: slide_boundary`)

**What it does:**  
Slides RIGHT/TOP (and corner) blocks to the current bounding-box edge.

**Order:** Corners (TL=5, TR=6, BL=9, BR=10) first, then edges (RIGHT=2, TOP=4).
LEFT=1 and BOTTOM=8 are already satisfied by packing to origin.

**Cluster awareness:** Boundary-constrained cluster members → slide entire super-block.
Preserves member offsets (internal abutment maintained).

---

### Step 8 — Hard Constraint Enforcement (`constraints.py: enforce_hard`)

**What it does:**  
Final guarantee: overlap=0, dim_viol=0, area≈0 for all blocks.

**8a — Restore exact dimensions/positions:**
- Preplaced: `pos[i] = (fixed_x, fixed_y, w, h)`.
- Fixed-shape: `pos[i] = (x, y, w, h)` (lock w,h, keep placement position).

**8b — Area nudge:**  
For each soft block: `h_exact = area / w`. Applied only if `|w·h_exact - area| / area ≤ 1.1%`.

**8c — Overlap resolution (`_resolve_overlaps`, ≤ 80 passes):**  
O(n²) pairwise scan. For each overlapping pair:
- Push the movable block (non-preplaced) by minimum separation.
- Direction: min(overlap_x, overlap_y) determines axis; push along that axis.
- Cluster members move as rigid units.
- Preplaced blocks are NEVER moved.
- Overlap tolerance: `1e-9`.

**`_escape_overlaps` fallback (guaranteed escape):**  
After 80 resolve passes, any movable unit still overlapping is relocated to a clean
grid position ABOVE the anchor bounding box. The grid is derived from the maximum
block dimension, guaranteeing the escaped blocks don't overlap each other. Only
a few units escape in practice; quality cost is small since area_gap is already
reduced by Step 6c.

---

## Key Data Structures

```python
@dataclass
class BlockInfo:
    idx: int
    w: float
    h: float
    fixed_x: Optional[float]    # None if free
    fixed_y: Optional[float]
    is_preplaced: bool
    is_fixed_shape: bool
    mib_group: int              # 0 = none
    cluster_group: int          # 0 = none; members[0] is the rep
    boundary_code: int          # OR of LEFT=1, RIGHT=2, TOP=4, BOTTOM=8

@dataclass
class SuperBlock:
    members: List[int]                  # original block indices
    offsets: List[Tuple[float, float]]  # (dx, dy) per member from super-block LL
    w: float
    h: float
    boundary_code: int                  # inherited from any member
```

---

## Parameter Summary

| Parameter | Location | Value | Purpose |
|-----------|----------|-------|---------|
| `n_spread_iters` | `quadratic_placer.py` | 30 | Spreading iterations in analytic placement |
| augment loop cap | `optimizer.py` | 15 | Max augment+repack cycles after Steps 5 and 6b |
| `n_iters` | `shaping.py` | 20 | Shaping iterations per call |
| `ar_min` | `shaping.py` | 0.2 | Min aspect ratio w/h for soft blocks |
| `ar_max` | `shaping.py` | 5.0 | Max aspect ratio w/h for soft blocks |
| `step_factor` | `shaping.py` | 0.95 | AR step size per shaping iteration |
| `rounds` | `topology.py: compact` | 12 | Compaction rounds (x+y = 2 passes each) |
| `compact convergence` | `topology.py` | 1e-6 | Bbox change threshold to stop early |
| resolve passes | `constraints.py` | 80 | Overlap resolution passes before escape |
| overlap tolerance | `constraints.py` | 1e-9 | Min overlap to trigger push |
| area nudge limit | `constraints.py` | 1.1% | Max relative area correction in 8b |
| shelf width factor | `constraints.py` | 1.4 | Target AR for cluster pre-packing |

---

## ML Leverage Points

The pipeline is designed as a deterministic baseline so each stage can be independently
replaced or augmented by a learned component.

| Stage | Current method | ML opportunity |
|-------|---------------|----------------|
| **Initial positions** (Step 3) | Quadratic wirelength CG | GNN/attention to predict good initial placement |
| **Topology extraction** (Step 4) | Heuristic center-gap comparison | Learn which pairs need HCG vs VCG edges from training data |
| **Soft shaping** (Step 6) | Greedy critical-path balancing | RL policy for aspect-ratio selection; supervised from ground-truth shapes |
| **Compaction order** (Step 6c) | Fixed alternating x/y, 12 rounds | Learn number of rounds and axis order per instance |
| **Cluster pre-packing** (Step 2) | Shelf heuristic | Predict optimal member arrangement from connectivity |
| **Augmentation edges** (Step 5b) | Overlap-triggered edge insertion | Predict which edges to add before first pack (avoid repack loops) |

---

## Failure Mode Reference

These bugs cost significant debugging time:

1. **Preplaced blocks in clusters** — two clusters each containing a preplaced block at
   similar x produce rigid columns forced into the same strip → unavoidable overlap.
   Fix: detach preplaced members from their cluster in `parse_and_init`.

2. **MIB group with a preplaced sibling** — `_apply_shape` would reshape the preplaced
   block's (w,h) via a non-preplaced sibling. Since MIB shares one shape, reshaping any
   member overwrites all. Fix: lock the whole MIB group if ANY member is preplaced.

3. **Re-running `build_topology` after packing** — re-adds diagonal-separation edges,
   making area_gap worse (tested: 1.12 → 6.45 for test 99). Use `compact` instead.

4. **`debug_stages.py` hid preplaced bugs** — used `target_positions = -1` so no block
   was treated as preplaced, masking all preplaced-related overlaps. Use `debug_real.py`
   which replicates the evaluator's ground-truth polygon positions.

5. **Backward propagation without clamp** — after backward pass, a BOUND_LEFT rep can
   land at x = -12.5 but is clamped to 0 at position-build time; its successor was not
   updated and sat at x = 12.5, overlapping the cluster spanning [0, 25]. Fix: clamp
   all reps to ≥ 0 then run one final forward pass.
