# B*-Tree SA Floorplanner Workflow

This document describes the intended design for improving
`FloorSet/baseline_optimizer.py` into a hard-feasible B*-tree plus simulated
annealing floorplanner for the ICCAD 2026 FloorSet Lite evaluator.

## 1. Current Baseline Assessment

The existing `baseline_optimizer.py` is a reasonable starting point because it
already has:

- A B*-tree representation with parent, left-child, and right-child arrays.
- A contour-style packing routine.
- A simulated annealing loop with rotate and delete-insert moves.
- Area-preserving dimensions for unconstrained blocks when initialized as
  squares.

However, it is not yet aligned with the requested design or the official v9
evaluator:

- It includes all blocks in the B*-tree, including preplaced blocks. Preplaced
  blocks must be immutable and excluded from perturbation.
- `move_swap()` swaps dimensions instead of swapping block positions in the
  tree. This changes block identities and can violate fixed-shape, MIB, and
  netlist semantics.
- Fixed-shape blocks are initialized from target dimensions but can still be
  perturbed by rotation or dimension swapping unless explicitly guarded.
- Soft-block shape is stored directly as width and height. The safer state is
  aspect ratio, with dimensions derived as `w = sqrt(area * AR)` and
  `h = sqrt(area / AR)`.
- MIB is only indirectly handled, if at all. Blocks in one MIB group should
  share one aspect-ratio or shape state by construction.
- The contour does not initialize static preplaced obstacles.
- Obstacle avoidance is not part of the packing semantics. A final pairwise
  "push up" pass is not enough, because it does not update descendants or the
  contour and can create new geometric inconsistencies.
- The cost function only uses HPWL and a small area term. It does not include
  grouping, boundary, MIB, fixed/preplaced hard penalties, or official-style
  normalization.

The highest priority is hard feasibility. Any overlap, area error, fixed-shape
dimension mismatch, or preplaced mismatch gives official cost `10.0`.

## 2. Optimizer State Model

Separate logical block identity from geometric shape state.

### 2.1 Block Records

Create one internal block record per original block ID:

- `block_id`
- `area`
- `kind`: `soft`, `fixed`, or `preplaced`
- `mib_group`: integer group ID or `0`
- `cluster_group`: integer group ID or `0`
- `boundary_code`: bitmask from the constraint tensor
- `target_x, target_y, target_w, target_h` from `target_positions`
- `allow_rotate`: true for fixed-shape blocks if 90-degree rotation is allowed
  by the chosen interpretation, false for preplaced blocks

Contest v9 says fixed-shape and preplaced dimensions must match the input
specification exactly. If we allow fixed-shape rotation, we must verify whether
the evaluator accepts swapped `(w, h)`. The current evaluator checks exact
`w == target_w` and `h == target_h`, so the implementation should not rotate
fixed-shape blocks unless the evaluator is also changed or the official spec is
confirmed to accept rotation. For local contest scoring, preserve exact target
orientation.

### 2.2 Tree Nodes

Only movable blocks are represented as B*-tree nodes:

- soft blocks
- fixed-shape blocks if their position may move

Preplaced blocks are excluded from the tree and never changed.

Tree nodes store `block_id`, `parent`, `left`, and `right`. They should not own
independent width and height as mutable truth. Width and height are queried from
the shape manager.

### 2.3 Shape Manager

The shape manager owns dimensions by construction:

- Soft non-MIB block: one `AR` variable.
- Soft MIB group: one shared `AR` variable used by every soft member in the
  group.
- Fixed-shape block: immutable `(target_w, target_h)` for evaluator safety.
- Preplaced block: immutable `(target_w, target_h)`.

For every soft block:

```text
w = sqrt(area * AR)
h = sqrt(area / AR)
```

This guarantees exact area in floating point up to numerical tolerance. Clamp
`AR` to a practical interval such as `[0.1, 10.0]` to avoid pathological skinny
blocks.

## 3. Contour And Obstacle-Aware Packing

Packing should be deterministic for a given tree and shape state.

### 3.1 Static Obstacles

Preplaced blocks are static obstacles. The contour should be initialized by
inserting every preplaced rectangle before movable blocks are packed.

Implementation options:

- Simple segment-list skyline: adequate for 21 to 120 blocks.
- Segment tree: faster but more code; not necessary for the first robust
  version.

Use a segment-list representation with half-open x-ranges:

```text
[(x0, x1, y_top), ...]
```

Each update splits overlapping ranges, writes the new `y_top` over
`[x_start, x_end)`, then merges adjacent segments with equal height.

### 3.2 Placement Rule

For each B*-tree node in preorder:

- Root starts at a legal seed x, usually `0.0` unless obstacle seeding suggests
  a better x.
- Left child is placed at the parent right edge.
- Right child is placed at the parent x-coordinate.
- The initial y is `max_contour_y(x, x + w)`.

Then run obstacle clearance before committing the block.

### 3.3 Obstacle Clearance

For a candidate rectangle `(x, y, w, h)`, detect overlap against preplaced
obstacles. If overlap exists, repair before updating the contour.

Use a deterministic "lowest legal position" loop:

1. Compute `y = max_contour_y(x, x + w)`.
2. Find all preplaced obstacles intersecting `[x, x + w)` and `[y, y + h)`.
3. If none overlap, place the block.
4. Otherwise choose a repair:
   - Push upward to `max(obs.y + obs.h)` over overlapping obstacles.
   - Optionally try rightward candidate `x = obs.x + obs.w` and choose the
     candidate with smaller incremental bounding-box area or lower cost proxy.
5. Repeat until no preplaced overlap remains.

The first implementation can use upward-only clearance because the contour
already prevents overlap with previously packed movable blocks. Add rightward
candidate search later if upward-only packing wastes too much area.

### 3.4 Final Feasibility Guard

After packing:

- Combine preplaced and movable positions into original block order.
- Verify no pairwise overlaps.
- Verify every preplaced tuple is unchanged.
- Verify every fixed-shape dimension is unchanged.
- Verify every soft-block area is within 1%.

If a candidate is infeasible, return infinite internal cost and reject it in SA.
Do not rely on a final push pass that mutates positions without updating the
tree/contour state.

## 4. Cost Function

Use an internal cost that tracks the official evaluator closely while remaining
cheap.

```text
Cost =
  alpha * bbox_area
+ beta  * (hpwl_b2b + hpwl_p2b)
+ gamma * grouping_penalty
+ theta * boundary_penalty
+ hard_penalty
```

Recommended behavior:

- `hard_penalty = inf` for overlap, fixed/preplaced mismatch, or soft area
  error.
- Add an MIB penalty only if MIB is not already handled by construction.
- Normalize terms to comparable scales. For example, divide area by total block
  area and HPWL by `sqrt(total_area) * total_net_weight`.
- Start with high weights for soft constraints, then tune after hard feasibility
  is stable.

### 4.1 HPWL

Use the existing evaluator helpers:

- `calculate_hpwl_b2b`
- `calculate_hpwl_p2b`

They use block centers and weighted Manhattan distance.

### 4.2 Grouping Penalty

For each cluster group:

```text
penalty = bounding_box_area(group) - sum(block_area in group)
```

This is a smooth proxy for abutment. It does not exactly match official
connected-component grouping violations, but it is useful during SA because it
gives a gradient-like signal before exact abutment occurs.

For final tie-breaking, also count exact Shapely-style connected components if
available, or a rectangle edge-contact graph if avoiding Shapely in the
optimizer.

### 4.3 Boundary Penalty

Compute the global bounding box after packing. For each constrained block,
calculate Manhattan distance to required edge or corner:

- left: `abs(block.x - bb.x_min)`
- right: `abs(block.x + block.w - bb.x_max)`
- bottom: `abs(block.y - bb.y_min)`
- top: `abs(block.y + block.h - bb.y_max)`

For corners, sum the relevant edge distances.

### 4.4 MIB Handling

Prefer by-construction handling:

- Soft MIB group members share one `AR`.
- Fixed MIB group members already have exact dimensions; if target dimensions
  differ within a MIB group, the hard fixed-shape rule takes priority for local
  evaluator feasibility.

The official evaluator counts distinct `(w, h)` pairs rounded to 4 decimals, so
shared shape state should reduce MIB violations to zero for soft groups.

## 5. Simulated Annealing Moves

Each accepted/rejected move should be reversible by copying the tree and shape
state. With at most 120 blocks, whole-state copying is acceptable.

### 5.1 Swap

Swap two node positions in the B*-tree while preserving block identities and
shape state.

Do not swap dimensions between block IDs. Nets and constraints are attached to
block IDs, so dimension swapping corrupts the problem semantics.

Practical first version: swap `block_id` labels between two tree nodes. This is
simple and changes which block occupies each tree position. It is legal as long
as preplaced blocks are not in the tree and fixed dimensions follow the block
ID through the shape manager.

### 5.2 Move

Delete one node from its current position and insert it as the left or right
child of another random node. Preserve any displaced child by attaching it under
the moved node consistently.

Guard against:

- moving the root into its own subtree
- choosing the moved node as its own target
- creating cycles

### 5.3 Rotate

For fixed-shape blocks, keep disabled for local evaluator compatibility unless
the exact orientation rule is clarified.

For soft blocks, rotation is equivalent to `AR = 1 / AR`. For soft MIB groups,
apply this to the shared group `AR`.

### 5.4 Reshape

Select either:

- one soft non-MIB block, or
- one soft MIB group

Then update:

```text
AR = clamp(AR * (1 + sign * delta), AR_MIN, AR_MAX)
```

Use small deltas early, for example `delta` sampled from `[0.02, 0.15]`. Larger
reshapes can be allowed at high temperature.

## 6. Initial Solution Construction

Build a legal initial solution before SA starts.

1. Parse constraints and target positions.
2. Build block records and shape state.
3. Fix preplaced blocks in the output array.
4. Create movable node list excluding preplaced blocks.
5. Initialize soft/MIB aspect ratios to `1.0`, or use target dimensions if they
   are available and consistent.
6. Order movable blocks with a constructive heuristic:
   - high net degree first,
   - boundary-constrained blocks near the front,
   - cluster members adjacent in insertion order,
   - larger area blocks earlier.
7. Build an initial B*-tree as a balanced tree or area-sorted chain.
8. Pack with preplaced obstacles and verify hard constraints.

A deterministic initial solution makes debugging easier. Random restarts can be
added later.

## 7. SA Schedule

Use a schedule that scales with instance size:

```text
T = initial_temp
while T > final_temp and budget remains:
    for k in range(moves_per_temp):
        propose one move
        pack
        evaluate internal cost
        accept if delta <= 0 or random() < exp(-delta / T)
    T *= cooling_rate
```

Initial parameters:

- `moves_per_temp = max(50, 5 * movable_count)`
- `cooling_rate = 0.95`
- `final_temp = 1e-3 * initial_temp`
- `initial_temp` estimated from sampled random move deltas instead of fixed
  `100.0` once the cost normalization is implemented.

Track:

- best feasible state
- current state
- number of rejected infeasible proposals
- best cost components for debugging

## 8. Implementation Milestones

### Milestone 1: Hard-Feasible Core

- Exclude preplaced blocks from the tree.
- Preserve fixed and preplaced dimensions exactly.
- Replace direct soft dimensions with aspect-ratio-derived dimensions.
- Add contour initialization from preplaced blocks.
- Add hard-feasibility checks in the internal evaluator.

Success criterion:

```bash
cd FloorSet/iccad2026contest
python iccad2026_evaluate.py --validate ../baseline_optimizer.py
python iccad2026_evaluate.py --evaluate ../baseline_optimizer.py --test-id 0 --verbose
```

No hard infeasibility should appear on tested cases.

### Milestone 2: Correct Moves

- Replace dimension-swap with B*-tree node/block-position swap.
- Guard rotate/reshape by block kind.
- Add MIB shared aspect-ratio state.
- Make delete-insert cycle-safe.

### Milestone 3: Soft Constraint Terms

- Add grouping compactness penalty.
- Add boundary distance penalty.
- Add exact MIB-by-construction handling.
- Print/debug cost components when `verbose` is enabled.

### Milestone 4: Quality Tuning

- Normalize HPWL and area terms.
- Estimate initial SA temperature from sampled moves.
- Try multiple initial orders or random restarts under a runtime budget.
- Add rightward obstacle-clearance candidates if preplaced obstacles cause tall
  layouts.

## 9. Risks And Design Choices

- Fixed-shape rotation conflicts with the local evaluator's exact-dimension
  check. Preserve target orientation for now.
- Upward-only obstacle clearance is simpler and safe, but may increase area.
  Rightward candidate search can improve quality later.
- Grouping bounding-box penalty is a proxy, not the exact official grouping
  violation. It should be combined with exact edge-contact checks when tuning.
- Whole-state copying is acceptable for 120 blocks and simplifies correctness.
  Optimize only if runtime becomes the bottleneck.

## 10. Recommended Next Code Changes

The next code edit should be a scoped refactor of `baseline_optimizer.py`:

1. Add block/shape metadata parsing in `solve()`.
2. Introduce `ShapeState` and make B*-tree nodes refer to original `block_id`.
3. Exclude preplaced blocks from the tree and merge them into final positions.
4. Replace `pack()` with obstacle-aware contour packing.
5. Replace `_cost()` with hard-feasible cost components and soft penalties.

After that, run single-case evaluation before doing any broader tuning.
