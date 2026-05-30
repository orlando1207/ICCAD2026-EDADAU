# Current Analytic Legalizer Method and Improvement Analysis

This note documents the current `analytic_legalizer` method used by
`iccad2026contest/my_optimizer.py`, and summarizes where the implementation can
be improved for better contest score.

The active entry point is:

```text
my_optimizer.py
  -> analytic_legalizer.MyOptimizer
  -> analytic_legalizer/optimizer.py
```

The current method is a deterministic analytic-placement plus skyline-legalizer
pipeline. It is not a learned method. Its main strength is fast, reproducible
100/100 feasibility on the validation set.

## Current Validation Result

Measured with:

```bash
cd FloorSet/iccad2026contest
python score_harness.py all
```

Current result:

```text
Feasible: 100/100
boundary violations: 445
grouping violations: 125
MIB violations: 0
weighted total score: 1.9675
```

The score is dominated by the largest validation cases because the official
score weights cases by `exp(block_count)`. Cases 99, 98, and 97 contribute most
of the final score.

For case 99:

```text
n=120
cost=1.8984
HPWL gap=0.5887
area gap=0.4016
V_rel=0.1194
feasible=True
```

## Contest Cost Model

The evaluator computes:

```text
Cost = (1 + 0.5 * (HPWL_gap + Area_gap))
       * exp(2 * V_rel)
       * max(0.7, RuntimeFactor^0.3)
```

If a hard constraint is violated, the cost is set to `10.0`.

Hard constraints:

- no overlaps
- soft block area within 1 percent of target
- fixed-shape dimensions exact
- preplaced `(x, y, w, h)` exact

Soft constraints:

- boundary constraints
- grouping constraints
- MIB equal-shape constraints

The current method is already strong on hard constraints and MIB. The remaining
score loss mainly comes from:

1. HPWL gap
2. area gap
3. boundary and grouping soft violations

## Pipeline Overview

The active pipeline is:

```text
parse constraints
  -> initialize block shapes
  -> unify MIB shapes
  -> pre-pack clusters into rigid super-blocks
  -> solve quadratic analytic placement
  -> spread analytic positions
  -> skyline legalize with several strip widths
  -> gap-fill finetune
  -> boundary slide
  -> hard-constraint enforcement
```

In code:

```text
constraints.py       parse_and_init, prepack_clusters, slide_boundary, enforce_hard
quadratic_placer.py  analytic_place
skyline_legalizer.py skyline_legalize
optimizer.py         MyOptimizer.solve orchestration
```

## Step 0 and 1: Constraint Parsing and Shape Initialization

Implemented in `constraints.py::parse_and_init`.

Each block is converted into a `BlockInfo` record:

```text
idx
w, h
fixed_x, fixed_y
is_preplaced
is_fixed_shape
mib_group
cluster_group
boundary_code
```

The constraint tensor columns are:

```text
0 fixed-shape flag
1 preplaced flag
2 MIB group id
3 cluster/grouping id
4 boundary code
```

Boundary codes are bitmasks:

```text
LEFT   = 1
RIGHT  = 2
TOP    = 4
BOTTOM = 8
```

Free soft blocks start as squares:

```text
w = h = sqrt(target_area)
```

Fixed-shape and preplaced blocks use the target dimensions from the validation
label-derived target-position tensor. Preplaced blocks also keep their fixed
lower-left coordinate.

### Preplaced Cluster Detachment

If a cluster contains preplaced members, those preplaced blocks are detached
from the cluster before cluster pre-packing.

Reason: a cluster is modeled as one rigid super-block. If a preplaced member
remained inside the rigid cluster, the whole super-block would be pinned to that
preplaced location, often forcing unavoidable overlap or bad placement.

This may introduce a grouping soft violation, but it protects feasibility.

## Step 1: MIB Shape Unification

Implemented in `constraints.py::_unify_mib_shapes`.

All blocks in each MIB group must share the same `(w, h)`.

Current logic:

- If the group contains a fixed-shape or preplaced member, use that member as an
  anchor shape.
- Otherwise use a square based on the mean area of the group.

This is why the current validation run has:

```text
MIB violations: 0
```

Tradeoff: when MIB member areas differ, forcing one shared shape can create area
error pressure. The current implementation prioritizes satisfying the MIB soft
constraint and then relies on hard-enforcement area correction for free blocks.

## Step 2: Cluster Pre-Packing

Implemented in `constraints.py::prepack_clusters`.

Each non-detached cluster is converted into a `SuperBlock`:

```text
members
offsets from super-block lower-left
super-block width
super-block height
inherited boundary code
```

The legalizer places the super-block as one rigid unit. Later, original member
rectangles are materialized using their stored offsets.

There are three cluster packing modes.

### Normal Shelf Packing

Used when a cluster has no boundary members.

Blocks are sorted by decreasing height and placed into horizontal shelves. The
row width target is approximately:

```text
sqrt(total_cluster_area)
```

This gives a compact near-square super-block.

### Boundary-Aware Frame Packing

Used when a cluster has boundary-constrained members.

The cluster is arranged as a frame:

- BOTTOM members go in a bottom row.
- TOP members go in a top row.
- LEFT-only members go in a left column.
- RIGHT-only members go in a right column.
- unconstrained members are shelf-packed in the middle.

This matters because placing the super-block against a boundary only helps a
member if that member is physically located on the matching face of the
super-block.

### Anchor Packing

There is support for packing around a preplaced anchor, but the active parser
usually detaches preplaced members from clusters before this case matters.

## Step 3: Quadratic Analytic Placement

Implemented in `quadratic_placer.py::analytic_place`.

The analytic placer builds a smaller node graph:

- preplaced blocks are fixed anchor nodes
- each cluster super-block is one node
- remaining free blocks are individual nodes

It solves a quadratic wirelength problem:

```text
minimize sum_edges weight * ((cx_i - cx_j)^2 + (cy_i - cy_j)^2)
```

Block-to-block edges add Laplacian terms. Pin-to-block edges add fixed-pin
springs. Preplaced blocks are enforced as Dirichlet fixed coordinates.

If the system has anchors, it solves:

```text
A * x = bx
A * y = by
```

with NumPy. If there are no anchors, the Laplacian is rank-deficient, so the
method uses a grid initialization instead.

### Analytic Spreading

After the quadratic solve, the placer runs pairwise repulsion for 30 iterations.

For every overlapping pair of analytic nodes, it pushes nodes away from each
other along their center difference. Fixed preplaced nodes do not move.

This reduces analytic overlap before legalization, but it is still only a guide.
The skyline legalizer ultimately determines the final non-overlapping positions.

## Step 4: Skyline Legalization

Implemented in `skyline_legalizer.py::skyline_legalize`.

This is the core active legalizer. It replaced the older topology/longest-path
legalization path.

The skyline legalizer packs movable units into a fixed-width vertical strip:

```text
container width = W
height grows upward
```

Movable units are:

- individual free blocks
- rigid cluster super-blocks

Preplaced blocks are not movable. They are treated as fixed rectangular
obstacles.

### Skyline Data Structure

The skyline is a list of contiguous segments:

```text
[x_start, x_end, current_height]
```

The segments tile `[0, W]`.

When a unit of width `w` is tested at an x-coordinate:

1. The legalizer finds the maximum skyline height under `[x, x + w]`.
2. That height becomes the candidate landing `y`.
3. If the rectangle would intersect a preplaced obstacle, the candidate `y` is
   bumped above the obstacle.
4. The candidate is scored.

Candidate score:

```text
score = landing_y + lambda * abs(unit_center_x - analytic_center_x)
```

The current default is:

```text
lambda = 0.3
```

This balances density against preserving the analytic x-order, which helps
wirelength.

### Placement Order

Units are sorted so that:

- BOTTOM boundary units are placed first
- TOP boundary units are placed last
- otherwise units follow analytic `(cy, cx)` order

This gives BOTTOM units a chance to touch `y=0`, while TOP units are delayed
until the final height is closer to known.

### Boundary Handling

Boundary behavior during skyline placement:

- LEFT units are forced to `x=0`.
- RIGHT units are forced to `x=W-w`.
- BOTTOM units are placed early.
- TOP units are placed late and later helped by `slide_boundary`.

TOP remains the hardest boundary because the final top edge is not known until
after packing finishes.

## Container Width Selection

The skyline packer tries multiple candidate strip widths and keeps the best
proxy score.

Minimum width:

```text
W_min = max(widest_movable_unit, max_preplaced_right_edge)
```

Candidate aspects:

```text
analytic_bbox_aspect
1.0
1.3
1.6
2.0
2.5
```

For each aspect:

```text
W = sqrt(total_area / aspect)
W = max(W, W_min)
```

Each candidate width is packed, materialized, and scored by:

```text
proxy = bbox_area * exp(2 * boundary_unmet / n)
```

This width selection currently ignores HPWL, even though HPWL is a large part
of the final score. This is one of the highest-leverage improvement targets.

## Step 5: Gap-Fill Finetune

Implemented in `skyline_legalizer.py::_finetune_fill_gaps`.

After skyline packing, the method tries to shrink the final bounding box by
moving only frontier free blocks into empty cluster-internal gaps.

It is conservative:

- only non-preplaced, non-cluster free blocks can move
- only blocks defining the current right or top frontier are considered
- a move is accepted only if it strictly reduces bounding-box area
- the pass stops after no further improvement

This protects feasibility and avoids large HPWL damage, but it leaves many
non-frontier local improvements unexplored.

## Step 6: Boundary Slide

Implemented in `constraints.py::slide_boundary`.

After packing, RIGHT and TOP boundary blocks or clusters are translated toward
the final bounding-box edge.

Important behavior:

- preplaced blocks never move
- clusters move as a unit
- corners are processed before edge-only blocks
- proposed slides are clipped to avoid creating new overlaps

This pass improves soft boundary score but is intentionally conservative. If a
block cannot reach the edge without overlap, it stays short of the edge.

## Step 7: Hard Constraint Enforcement

Implemented in `constraints.py::enforce_hard`.

This is the final safety net.

It performs:

1. Restore exact preplaced and fixed-shape dimensions/positions.
2. Nudge soft-block height to recover exact target area.
3. Resolve overlaps by pushing movable blocks or rigid clusters.
4. Escape any still-overlapping movable units to a clean grid above the anchor
   bounding box.
5. Snap tiny intra-cluster floating-point gaps so Shapely sees grouped blocks
   as connected.

This is why the current method reliably reaches 100/100 feasibility.

The downside is that late overlap repair can damage HPWL, area, and boundary
quality. Ideally the earlier legalizer should make `enforce_hard` nearly a
no-op.

## Why the Current Method Works

The method is strong because it separates concerns:

- The analytic stage gives a wirelength-aware guide.
- The skyline stage guarantees dense, non-overlapping constructive placement.
- Cluster pre-packing preserves grouping for most non-preplaced clusters.
- MIB unification removes MIB violations.
- Hard enforcement protects contest feasibility.

This is a pragmatic architecture for the contest because one infeasible case
costs `10.0`, while a feasible but imperfect case is usually much cheaper.

## Current Weaknesses

### 1. HPWL Is Not Used Enough During Legalization

The analytic solve optimizes quadratic wirelength, but the skyline packer then
makes many placement decisions using:

```text
landing_y + lambda * x_distance_to_analytic
```

This only preserves x-position approximately. It does not directly score true
block-to-block or pin-to-block HPWL.

The width selector also ignores HPWL:

```text
bbox_area * exp(2 * boundary_unmet / n)
```

Given the observed case metrics, HPWL gap is now one of the largest score
contributors. For example, case 97 has HPWL gap around `2.00`, making it a
major outlier.

### 2. Width Selection Optimizes Area and Boundary, Not Final Cost

The legalizer tries a small aspect ladder and picks by area plus boundary proxy.
This can select a compact layout that stretches many high-weight nets.

The official quality term is:

```text
1 + 0.5 * (HPWL_gap + Area_gap)
```

The legalizer does not know the hidden baseline gaps directly, but it can still
compute raw HPWL from the candidate solution.

### 3. Placement Order Is Fixed and Greedy

The skyline algorithm is one-pass greedy. Once a block is placed, later blocks
must adapt around it.

Current order is mostly analytic bottom-up. This is fast, but it can be bad for:

- high-degree net hubs
- very large blocks
- right/top boundary blocks
- blocks near preplaced obstacles
- clusters with awkward internal shape

### 4. TOP Boundary Is Structurally Hard

LEFT, RIGHT, and BOTTOM are tied to known strip edges. TOP is different because
the final top edge is only known after packing.

The current approach places TOP blocks late and then slides them, but it does
not reserve top-edge capacity during packing.

### 5. Cluster Pre-Pack Can Increase Area or HPWL

Clusters are packed locally before global placement. A compact cluster is good
for area, but the chosen member arrangement may be poor for external nets.

The current cluster pre-pack does not optimize internal member order against
connectivity. It mostly uses geometry and boundary codes.

### 6. Preplaced Obstacles Are Handled Feasibly, Not Optimally

The current obstacle logic bumps a candidate rectangle above any preplaced
obstacle it would hit.

This is safe, but it can create vertical stacking and local whitespace. There is
no local search around preplaced obstacles after initial placement.

### 7. Shape Choices Are Conservative

Most soft blocks stay square. Since aspect ratio is relaxed by the contest,
there is room to reshape blocks to reduce area or HPWL while maintaining target
area.

The current method only nudges height at the end to correct exact area.

## Highest-Leverage Improvement Directions

### Improvement 1: HPWL-Aware Width Selection

Modify the candidate-width selector to include raw HPWL:

```text
proxy = bbox_area * hpwl_total * exp(2 * boundary_unmet / n)
```

or a normalized weighted sum:

```text
proxy = area_norm + alpha * hpwl_norm + beta * boundary_norm
```

This is low risk because it only changes which already-feasible candidate width
is selected.

Expected benefit:

- lower HPWL gap on score-dominant cases
- better case 97 behavior

Risk:

- if HPWL dominates too strongly, area may regress

Recommended experiment:

```text
try area * hpwl
try area * sqrt(hpwl)
try area * exp(2 * boundary/n) * sqrt(hpwl)
evaluate on cases 95-99 and all
```

### Improvement 2: More Candidate Widths and Aspect Ratios

The current aspect ladder is small:

```text
analytic, 1.0, 1.3, 1.6, 2.0, 2.5
```

Add a denser ladder around the best current region, for example:

```text
0.8, 1.0, 1.15, 1.3, 1.45, 1.6, 1.8, 2.0, 2.25, 2.5, 3.0
```

This is low complexity because `n <= 120` and each skyline pass is cheap.

Expected benefit:

- better area/HPWL tradeoff
- fewer cases stuck with a poor strip shape

Risk:

- runtime increases linearly with candidate count, but runtime score is damped
  by exponent `0.3`, so moderate extra runtime may be worth it.

### Improvement 3: Multi-Start Analytic Placement

`optimizer.py` already contains the structure for `N_STARTS`, but it is set to:

```text
N_STARTS = 1
```

Increasing to 4 uses noisy analytic seeds and selects by area-HPWL proxy.

This is already documented as improving weighted score from about `1.97` to
about `1.88` at roughly 4x runtime.

Expected benefit:

- immediate quality improvement
- low algorithmic risk

Risk:

- runtime increases
- final contest runtime factor may penalize excessive starts

Recommended use:

- keep `N_STARTS=1` for fast iteration
- use `N_STARTS=4` for final quality comparison
- evaluate if `N_STARTS=2` gives most of the gain at half the cost

### Improvement 4: HPWL-Aware Skyline Candidate Scoring

Current per-placement candidate score:

```text
y + lambda * abs(candidate_center_x - analytic_center_x)
```

A stronger version would evaluate incremental HPWL to already-placed neighbors
and pins:

```text
score = density_term
      + lambda_x * analytic_x_distance
      + lambda_net * incremental_HPWL_to_placed_neighbors
      + lambda_pin * pin_distance
```

Because unplaced neighbors do not have final locations yet, use:

- exact final positions for already-placed neighbors
- analytic centers for unplaced neighbors
- fixed coordinates for pins

Expected benefit:

- lower HPWL without fully abandoning greedy skyline packing

Risk:

- slower candidate evaluation
- bad weights can reduce density

This is probably the best medium-effort improvement after width selection.

### Improvement 5: Smarter Placement Ordering

Try alternative deterministic orders and select the best result:

- current analytic `(cy, cx)`
- high-degree or high-net-weight blocks earlier
- large-area blocks earlier
- boundary blocks grouped by edge
- preplaced-neighbor blocks earlier
- clusters earlier than individual blocks

For each order, run the same skyline packer and pick by area-HPWL-boundary proxy.

Expected benefit:

- improves greedy decisions without changing geometry code

Risk:

- runtime increases by number of orders tried

Good initial set:

```text
order 1: current
order 2: large blocks first within analytic rows
order 3: net-critical blocks first
order 4: boundary-priority order
```

### Improvement 6: Better TOP Boundary Handling

TOP violations are hard because final height is unknown.

Potential fixes:

- reserve a top shelf for TOP blocks and pack non-TOP blocks below it
- run a second skyline pass after estimating final height
- place TOP blocks into a temporary upper band, then compact below
- select width using a stronger TOP violation penalty

Expected benefit:

- lower `V_rel`
- especially useful because violation factor is exponential

Risk:

- reserving top space may increase area
- can worsen HPWL if TOP blocks are far from their nets

### Improvement 7: Connectivity-Aware Cluster Pre-Pack

Current cluster layouts are geometric. They do not optimize external or internal
net connectivity.

Improvements:

- order cluster members by internal/external net weights
- place high-external-degree members closer to the side facing their analytic
  neighbors
- try rotated/mirrored versions of a cluster frame and select by HPWL proxy
- try multiple shelf row widths for large clusters

Expected benefit:

- lower HPWL
- fewer grouping/boundary compromises

Risk:

- more code complexity
- local cluster HPWL proxy may conflict with global legalization

### Improvement 8: Shape Optimization for Soft Blocks

The contest allows arbitrary aspect ratio. The current method mostly keeps
blocks square.

Possible approaches:

- elongate blocks in rows to fill skyline gaps
- reshape frontier blocks to reduce bbox area
- choose dimensions based on local whitespace before final hard enforcement
- jointly tune shape for MIB groups

Expected benefit:

- lower area gap
- better utilization around gaps and obstacles

Risk:

- can hurt HPWL by moving centers or changing packing order
- must preserve exact area and fixed/MIB constraints
- extreme aspect ratios may produce ugly or fragile layouts

This is higher risk than HPWL-aware selection.

### Improvement 9: Local Search After Legalization

After a feasible placement is built, run a small bounded local search:

- move a block into nearby skyline gaps
- swap two blocks with similar size
- slide blocks along their row
- try moving frontier blocks inward
- accept only if no hard constraints break and proxy improves

Expected benefit:

- removes greedy artifacts

Risk:

- local geometry checks get expensive
- must avoid breaking clusters and preplaced constraints

This is useful after the lower-risk proxy and ordering improvements are done.

## Recommended Improvement Roadmap

The safest path is to improve candidate selection before changing placement
geometry.

### Phase 1: Low-Risk Selection Improvements

1. Add HPWL to candidate width selection.
2. Add more aspect-ratio candidates.
3. Compare `N_STARTS=1`, `2`, and `4`.

Metrics to watch:

```text
weighted score
cases 95-99 score
case 97 HPWL gap
feasibility 100/100
runtime
```

### Phase 2: Greedy Skyline Quality

1. Add incremental HPWL to candidate x scoring.
2. Try 2-4 deterministic placement orders.
3. Select among order/width candidates using area-HPWL-boundary proxy.

This targets the main remaining score loss: HPWL.

### Phase 3: Constraint-Specific Refinement

1. Improve TOP boundary handling.
2. Add cluster mirror/rotation variants.
3. Add local search around preplaced obstacles.

This targets `V_rel` and stubborn local whitespace.

### Phase 4: Shape Optimization

Only after the above is stable, experiment with soft-block aspect ratios.

This has high upside for area, but it is more likely to introduce hard-constraint
or HPWL regressions.

## Most Promising Immediate Experiments

The best first experiments are:

1. HPWL-aware candidate width selection.
2. Denser aspect ladder.
3. `N_STARTS=2` and `N_STARTS=4`.
4. Special tuning for cases 97-99, because they dominate weighted score.

These require little architectural change and directly target the measured
score bottleneck.

## How to Evaluate Changes

Fast dominant-case check:

```bash
cd FloorSet/iccad2026contest
python score_harness.py 95 96 97 98 99
```

Full validation:

```bash
python score_harness.py all
```

Official evaluator:

```bash
python iccad2026_evaluate.py --evaluate my_optimizer.py
```

Single case:

```bash
python iccad2026_evaluate.py --evaluate my_optimizer.py --test-id 99
```

Before accepting any algorithmic change, require:

```text
Feasible: 100/100
weighted score improves or the target metric improves without major regression
case 97/98/99 do not regress unexpectedly
```

