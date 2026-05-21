# Implemented B*-Tree SA Floorplanner Workflow

This note documents the current implementation in `FloorSet/baseline_optimizer.py`.

## Architecture

The optimizer follows the requested framework:

- B*-tree placement representation.
- 1D contour packing.
- Preplaced blocks as static contour obstacles.
- Hierarchical sub-B*-trees for cluster/group constraints.
- Two-stage simulated annealing.
- Aspect-ratio-based soft-block shaping.
- Shared MIB shape state.

The public contest API is unchanged:

```python
MyOptimizer.solve(
    block_count,
    area_targets,
    b2b_connectivity,
    p2b_connectivity,
    pins_pos,
    constraints,
    target_positions=None,
)
```

It returns one `(x, y, w, h)` tuple per original block ID.

## Block And Shape State

Each original block is parsed into a `BlockInfo` record:

- area target
- fixed/preplaced flags
- MIB group ID
- cluster/group ID
- boundary code
- target geometry from `target_positions`

The B*-tree never treats dimensions as the source of truth. Dimensions come
from `ShapeState`.

For soft blocks:

```text
w = sqrt(area * AR)
h = sqrt(area / AR)
```

Every non-MIB soft block starts with `AR = 1.0`.

For soft MIB groups:

- all members share one `AR`, initialized to `1.0`
- the group uses the average target area of its soft members
- reshaping or rotation of the MIB key updates every member together

For fixed-shape and preplaced blocks, target dimensions are immutable. The local
v9 evaluator checks exact `(w, h)` for fixed blocks and exact `(x, y, w, h)` for
preplaced blocks, so fixed-shape rotation is intentionally disabled for
evaluator compatibility.

Area is not perturbed. Only aspect ratio is perturbed. This preserves the hard
area constraint by construction for ordinary soft blocks.

## Preplaced Blocks And Contour

Preplaced blocks are excluded from every B*-tree.

During packing:

1. The contour is initialized with each preplaced rectangle's top edge.
2. Movable B*-tree nodes are packed in preorder.
3. Candidate blocks query the max contour height over their x-range.
4. If a candidate still intersects a preplaced obstacle, it is pushed upward
   until it clears.
5. The contour is updated with the placed block.

This makes overlap avoidance a construction rule rather than a postprocessing
repair.

## Hierarchical Group Handling

High-temperature stage:

- Each cluster/group with at least two movable members is converted into a
  `GroupMacro`.
- The macro owns a sub-B*-tree over its member block IDs.
- The sub-B*-tree packs members locally.
- The local packed bounding box becomes the macro dimensions.
- The top-level B*-tree places the macro as one item.

Low-temperature stage:

- The best high-temperature layout is expanded back into individual blocks.
- The low-temperature top-level B*-tree contains individual movable block IDs.
- Grouping is no longer enforced by macro construction.
- Grouping compactness and an official-style soft-violation proxy are included
  in the cost.

This matches the intended "cluster first, de-cluster later" flow.

## SA Moves

The implementation samples one move at a time:

- `swap`: swap payloads between two B*-tree nodes.
- `move`: delete a B*-tree node and insert it under another node.
- `subtree`: during the high-temperature stage, perturb a group sub-B*-tree.
- `rotate`: for soft keys, replace `AR` with `1 / AR`.
- `reshape_mul`: update `AR *= 1 +/- delta`.
- `reshape_div`: update `AR /= 1 - delta`, with random reciprocal direction.

MIB moves operate on the shared MIB key, so all members change together.

## Cost Function

The optimizer uses a hard-feasible internal objective:

```text
high stage:
  1.20 * normalized_HPWL
+ 1.00 * normalized_bbox_area
+ 0.60 * normalized_boundary_distance

low stage:
  1.00 * normalized_HPWL
+ 1.15 * normalized_bbox_area
+ 0.50 * normalized_boundary_distance
+ 0.35 * normalized_group_dead_space
+ 0.25 * official_soft_violation_proxy
```

The high-temperature stage emphasizes global HPWL, area, and boundary because
groups are still represented as macros.

The low-temperature stage adds:

- grouping dead-space penalty:
  `bbox_area(group) - sum(member_areas)`
- boundary distance penalty
- MIB distinct-shape proxy
- cluster edge-contact connected-component proxy

The official evaluator remains the source of truth. The internal soft proxy is
only a search signal.

## Feasibility Rules

The inner SA loop checks cheap hard constraints:

- preplaced tuple unchanged
- fixed dimensions unchanged
- soft block area within tolerance

Pairwise overlap checks are not run inside every proposal for runtime reasons.
The contour packer is expected to provide overlap-free placement by
construction, and the official evaluator performs the exact final check.

If the final best candidate is internally infeasible, the optimizer returns a
simple shelf fallback.

## Current Runtime Tuning

The default SA budget is deliberately modest:

- high-temperature steps: `1`
- low-temperature steps: `1`
- moves per temperature: capped at `8`

This keeps the implementation usable as a baseline. Future quality work should
first optimize the packing and incremental cost path, then raise the SA budget.

## Known Tradeoffs

- MIB soft groups use average area by request. If a dataset contains a soft MIB
  group with materially different area targets, exact area feasibility and exact
  MIB shape equality conflict.
- Obstacle repair currently pushes upward. A later variant should compare
  upward and rightward candidates to reduce dead space around preplaced blocks.
- Grouping uses a macro/subtree during high temperature, then a penalty during
  low temperature. Exact abutment is encouraged but not guaranteed.
- Fixed-shape rotation is disabled to satisfy the local evaluator's exact
  orientation check.
