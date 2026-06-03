# Post-Improvement Bottleneck Analysis and Next Directions

This report analyzes the current improved analytic legalizer after the latest
framework changes:

- optional LSE wirelength refinement for large cases
- denser skyline aspect-ratio candidates
- multiple skyline placement `lambda` values
- final candidate selection using area, HPWL, and soft-violation proxy

The goal of this report is to identify the next bottleneck and propose methods
that can reduce cost further, especially through HPWL and area improvement.

No code changes are proposed as mandatory here. This is an analysis and research
plan.

## Current Measured Status

Measured command:

```bash
cd FloorSet/iccad2026contest
python score_harness.py all
```

Current improved result:

```text
Feasible: 100/100
boundary violations: 419
grouping violations: 117
MIB violations: 0
weighted total score: 1.8205
```

Previous baseline before this improvement pass:

```text
Feasible: 100/100
boundary violations: 445
grouping violations: 125
MIB violations: 0
weighted total score: 1.9675
```

The main gain came from better selection among legalizations:

- case 99 improved from about `1.898` to `1.844`
- case 98 improved from about `1.86` to `1.79`
- case 97 improved strongly from about `2.75` to `1.68`
- dominant-case subset improved from about `1.9682` to `1.8203`

The method remains fully feasible.

## Score-Dominant Cases

Because the official score is exponentially weighted by block count, the largest
cases dominate:

```text
case 99, n=120: about 63 percent of weighted score
case 98, n=119: about 23 percent
case 97, n=118: about 9 percent
case 96, n=117: about 3 percent
case 95, n=116: about 1 percent
```

Together, cases 95-99 essentially determine the validation score. Small cases
still matter for feasibility and robustness, but they are not the main tuning
target.

Current dominant-case metrics:

```text
idx    n   cost  hpwl  area  vrel  bnd  grp mib
99   120   1.84  0.57  0.34  0.12    7    1   0
98   119   1.79  0.63  0.33  0.10    3    2   0
97   118   1.68  0.67  0.37  0.05    2    1   0
96   117   2.00  1.07  0.37  0.08    5    0   0
95   116   1.60  0.45  0.39  0.06    2    2   0
```

These numbers show that soft constraints are no longer the primary issue. The
remaining score is mostly quality loss:

- HPWL gap is still roughly `0.45` to `1.07` on the dominant cases.
- Area gap is still roughly `0.33` to `0.39`.
- `V_rel` is acceptable, with case 99 still the soft-violation outlier.

## Cost Sensitivity

The official feasible cost is:

```text
Cost = (1 + 0.5 * (HPWL_gap + Area_gap))
       * exp(2 * V_rel)
       * runtime_factor
```

Ignoring runtime for analysis, reducing HPWL and area has linear effect through:

```text
1 + 0.5 * (HPWL_gap + Area_gap)
```

Reducing soft violations has exponential effect through:

```text
exp(2 * V_rel)
```

But the current method has already brought soft violations low enough that the
next obvious bottleneck is HPWL plus area. For example:

- Case 97 now has `V_rel=0.05`, so further soft improvement has limited upside.
- Case 96 has `HPWL_gap=1.07`, which is the most obvious remaining quality issue.
- Case 99 has both `HPWL_gap=0.57` and `area_gap=0.34`; reducing either matters
  heavily because case 99 has the largest score weight.

## Current Framework After Improvement

The active framework is now:

```text
parse_and_init
  -> MIB unification
  -> boundary-aware cluster pre-pack
  -> quadratic analytic placement
  -> optional LSE refinement for n >= 116
  -> skyline legalization with multiple lambda values
  -> denser width/aspect sweep
  -> gap-fill finetune
  -> boundary slide
  -> hard enforcement
  -> final candidate selection by area * HPWL * soft proxy
```

The key change from the previous version is that the framework no longer trusts
only the legalizer's internal area-boundary proxy. It creates multiple feasible
candidate layouts and then selects using a stronger final proxy:

```text
proxy = bbox_area * raw_HPWL * exp(soft_proxy_weight * soft_proxy / n)
```

This better matches the contest cost behavior.

## What Is the Bottleneck Now?

### Bottleneck 1: Skyline Greedy Placement Does Not Directly Optimize HPWL

The skyline legalizer still places blocks greedily.

Current local candidate score is essentially:

```text
landing_y + lambda * x_distance_to_analytic_x
```

This means the legalizer indirectly preserves wirelength by staying near the
analytic solution, but it does not directly evaluate:

- high-weight block-to-block nets
- pin-to-block nets
- already placed neighbor centers
- future neighbor positions
- net criticality

The final selector can choose among candidate layouts, but it cannot repair a
bad greedy decision made early in skyline placement.

This is now probably the largest algorithmic bottleneck.

### Bottleneck 2: Area and HPWL Are Coupled by Width Selection

The strip width controls density, aspect ratio, and average net stretch.

Narrower width:

- may reduce or increase area depending on final height
- often compresses x-distance
- can increase y-distance and vertical stacking
- may help boundary

Wider width:

- may reduce height
- may reduce vertical HPWL
- may increase horizontal HPWL
- may increase empty area if packing becomes sparse

The current method tries more widths than before, but the width ladder is still
global and static. It does not know the instance's connectivity structure.

### Bottleneck 3: Placement Order Is Still Mostly Fixed

The legalizer tries multiple `lambda` values, but not multiple placement orders.

Current order is based on:

```text
BOTTOM first
TOP last
then analytic cy, cx
```

This is reasonable for density and boundary, but HPWL-critical blocks may be
placed too late. Once placed late, they must fit around earlier geometry, and
their netlength can become large.

Blocks that should probably receive special order treatment:

- high weighted degree blocks
- blocks connected to many pins
- large-area blocks
- boundary blocks with strong nets to interior
- blocks connected to preplaced anchors
- clusters with many external nets

### Bottleneck 4: Shapes Are Still Mostly Square

The contest relaxes aspect ratio, but the current method mostly keeps soft
blocks as squares.

Area gap around `0.33` to `0.39` means there is still substantial whitespace
relative to the baseline. Some of this may be unavoidable because the baseline
layouts are very dense, but some could likely be reduced by aspect-ratio
adaptation.

Current shape behavior:

- free soft blocks start square
- MIB groups use a shared square or anchor shape
- fixed/preplaced blocks stay fixed
- final hard enforcement only nudges height to exact area

No stage actively reshapes blocks to fill skyline gaps or reduce final bounding
box.

### Bottleneck 5: Cluster Internal Arrangement Is Connectivity-Blind

Clusters are pre-packed before global placement. Current cluster pre-pack is
mostly geometric and boundary-aware, not net-aware.

A cluster member with many external nets may be buried on the wrong side of the
cluster. Once the cluster is rigid, the global placer can only move the whole
cluster, not rearrange members.

This can hurt HPWL while preserving grouping.

### Bottleneck 6: LSE Helps, But Only as a Post-Quadratic Refinement

The added LSE refinement improves some large cases, but it is still not a full
global placement optimizer.

It starts from the quadratic solution and performs smooth-L1 pairwise edge
descent. It does not include:

- density forces tied to the final skyline legalizer
- area/bbox objective
- boundary objective
- cluster-internal net-aware arrangement
- differentiable legalization

So LSE is useful, but not enough by itself.

## Most Promising Non-ML Improvements

### 1. HPWL-Aware Skyline Candidate Scoring

Change each skyline placement decision from:

```text
landing_y + lambda * abs(x_center - analytic_x)
```

to something like:

```text
density_cost
+ analytic_distance_cost
+ incremental_HPWL_cost
+ pin_HPWL_cost
+ boundary_cost
```

For a candidate block position, estimate HPWL using:

- exact final centers for already placed neighbors
- analytic centers for unplaced neighbors
- exact pin positions for pin-to-block nets

This would make the greedy legalizer aware of actual net weights.

Expected upside:

- direct HPWL reduction
- especially useful for cases 96-99

Risk:

- may increase area if HPWL cost dominates density
- requires careful normalization

Recommended initial formula:

```text
score =
    y
  + lambda_x * abs(candidate_cx - analytic_cx)
  + lambda_net * incremental_net_cost / degree_norm
```

where:

```text
incremental_net_cost =
  sum(weight * (abs(candidate_cx - neighbor_cx)
              + abs(candidate_cy - neighbor_cy)))
```

Use placed neighbor centers when available, otherwise analytic centers.

This is likely the next highest-value deterministic improvement.

### 2. Multi-Order Skyline Packing

Keep the same skyline geometry but try multiple placement orders.

Candidate orders:

```text
order A: current boundary + analytic cy/cx
order B: large-area first within boundary groups
order C: high weighted-degree first
order D: high pin-connectivity first
order E: clusters first, then blocks
order F: analytic row order but with high-degree tie-break
```

Then select final layout with the existing area * HPWL * soft proxy.

Expected upside:

- can reduce greedy-order failures
- lower implementation risk than rewriting legalizer

Risk:

- runtime increases linearly with number of orders

Good compromise:

Try only two extra orders for `block_count >= 116`.

### 3. Local HPWL-Preserving Compaction

After skyline legalization, perform local moves that reduce area without
increasing HPWL too much.

Possible moves:

- slide a block left/right/down until it hits an obstacle
- move frontier blocks into empty pockets
- swap two similarly sized free blocks
- move a block to a nearby skyline cavity

Accept move if:

```text
no hard violation
and proxy improves
```

Proxy:

```text
bbox_area * HPWL * exp(soft_weight * soft_proxy / n)
```

Expected upside:

- area reduction
- some HPWL reduction through swaps

Risk:

- geometry checks can become expensive
- local moves can break boundary constraints or grouping if not constrained

### 4. Soft-Block Aspect-Ratio Adaptation

Use flexible aspect ratio to fill whitespace.

Possible low-risk approach:

1. Keep square blocks during analytic placement.
2. During skyline placement, allow a small set of shapes per soft block:

```text
(sqrt(a), sqrt(a))
(1.25*sqrt(a), sqrt(a)/1.25)
(sqrt(a)/1.25, 1.25*sqrt(a))
(1.5*sqrt(a), sqrt(a)/1.5)
(sqrt(a)/1.5, 1.5*sqrt(a))
```

3. For each block candidate, test a few shape variants.
4. Keep exact area.

Expected upside:

- lower area gap
- better gap filling

Risk:

- can increase HPWL by changing centers and skyline order
- MIB groups need shared shape
- fixed/preplaced blocks cannot change

This is a powerful area method, but it should be introduced after HPWL-aware
placement scoring, otherwise it may reduce area while damaging HPWL.

### 5. Connectivity-Aware Cluster Pre-Pack

For each cluster, try several internal arrangements:

- current geometry pack
- mirrored x
- mirrored y
- rotated member order
- net-aware shelf order
- boundary-frame variants

Select cluster variant using external-net proxy:

```text
sum external edge weight * distance from member local center to estimated neighbor direction
```

Expected upside:

- lower HPWL without losing grouping

Risk:

- cluster preprocessing becomes more complex
- local choice may not match final global placement

This should be tested on cases where grouping remains low but HPWL is high.

### 6. Width Prediction or Width Search Expansion

The current aspect ladder is static. A smarter method could choose candidate
widths based on instance features:

- total area
- preplaced obstacle span
- weighted graph aspect from analytic placement
- pin distribution aspect
- boundary count per edge
- cluster area fraction

Simpler deterministic improvement:

- add local search around the best width
- after the best aspect is found, try `W * {0.92, 0.96, 1.04, 1.08}`

Expected upside:

- better area/HPWL tradeoff

Risk:

- additional runtime

## Where ML or DL Could Help Most

ML is most useful where the current pipeline uses handcrafted discrete choices.
The current framework already generates feasible candidates, so ML does not need
to solve the entire problem from scratch. The best role for ML is as a guide or
selector inside the deterministic legalizer.

### ML Aid 1: Candidate Selector / Cost Surrogate

Train a model to predict final contest cost or rank candidate layouts.

Input:

- graph features
- constraints
- candidate layout features
- raw HPWL
- bbox area
- soft proxy
- block count
- boundary counts
- cluster/MIB counts

Output:

- predicted cost
- or pairwise ranking between candidate layouts

How it fits:

```text
generate 4-20 feasible candidates
ML model chooses best candidate
```

Why powerful:

- current selection proxy is handcrafted
- official cost uses hidden baselines, but training labels provide metrics
- ranking candidates is easier than generating layouts

This is probably the safest and highest-ROI ML integration.

Model choices:

- gradient-boosted trees for tabular candidate features
- small MLP
- graph neural network plus candidate-layout pooling

### ML Aid 2: Predict Skyline Parameters

Predict per-instance parameters:

- best aspect/width
- best skyline `lambda`
- whether to use LSE refinement
- best placement order
- soft-proxy weight

Input:

- graph-level features
- analytic placement statistics
- constraint distribution

Output:

- discrete class or continuous parameters

How it fits:

```text
features -> parameter predictor -> deterministic legalizer
```

Why powerful:

- avoids expensive brute-force sweeps
- can specialize behavior for cases like 96 vs 99

This is a good second ML step after candidate ranking.

### ML Aid 3: Placement Order Prediction

Train a model to assign each block a priority score for skyline placement.

Input per block:

- area
- degree
- weighted degree
- pin degree
- boundary code
- cluster id
- MIB id
- analytic x/y
- preplaced/fixed flags

Output:

- placement priority

Then skyline order becomes:

```text
boundary priority + learned priority
```

Why powerful:

- greedy skyline quality depends heavily on order
- order is discrete and hard to tune manually
- learned order can encode graph criticality

Model choices:

- GNN over block graph
- Transformer over blocks
- simple MLP with graph-derived features

### ML Aid 4: Shape Policy

Train a model to predict shape ratio per soft block or MIB group.

Input:

- area
- local graph features
- analytic position
- nearby blocks
- boundary/cluster/MIB flags

Output:

- aspect ratio class, for example:

```text
0.67, 0.8, 1.0, 1.25, 1.5
```

How it fits:

```text
predict shape variants -> skyline legalizer -> hard enforcement
```

Why powerful:

- area gap remains around 0.33-0.39 on dominant cases
- shape flexibility is unused

Risk:

- bad shapes can hurt HPWL
- MIB and fixed constraints complicate training

This is powerful but should be introduced carefully.

### ML Aid 5: Cluster Arrangement Model

Train a model to arrange cluster members locally.

Input:

- subgraph induced by cluster members
- external edges from cluster members
- boundary requirements
- block areas/shapes

Output:

- local order
- side assignment
- mirror/rotation choice

Why powerful:

- grouping requires clusters to be abutted
- cluster internal arrangement affects HPWL but is currently heuristic

This is specialized but likely useful if large clusters contribute significant
HPWL.

### ML Aid 6: Learned Global Placement Initialization

Use a GNN to predict initial block centers before legalization, replacing or
augmenting quadratic/LSE placement.

Input:

- block graph
- pins
- constraints
- areas

Output:

- target centers or relative ordering

How it fits:

```text
GNN centers -> skyline legalizer
```

Why powerful:

- analytic placement is limited by quadratic/LSE objective
- GNN can learn contest-specific placement patterns

Risk:

- harder to train
- must generalize to hidden test data
- feasibility still depends on legalizer

This is higher effort than candidate ranking or parameter prediction.

## Recommended ML Roadmap

### Stage 1: Candidate Ranking Model

Generate multiple candidates per validation/training case using current
deterministic variations:

- quadratic vs LSE
- multiple lambdas
- multiple aspect ladders
- multiple placement orders
- optional shape variants later

For each candidate, compute:

- raw HPWL
- bbox area
- boundary/grouping/MIB proxy
- runtime
- candidate parameters
- official cost on validation/training labels

Train a ranker to choose the best candidate.

This is low risk because every candidate is already feasible.

### Stage 2: Parameter Predictor

Instead of generating all candidates, predict:

- LSE on/off
- lambda
- aspect region
- order type

This reduces runtime and keeps most of the candidate-ranker benefit.

### Stage 3: Learned Order and Shape Policy

Use GNN or Transformer models to predict:

- block placement order
- shape ratio class
- cluster arrangement

This directly improves the legalizer rather than just selecting among variants.

## Recommended Next Deterministic Experiments

The next non-ML experiments should target HPWL directly.

### Experiment 1: Incremental HPWL in Skyline Scoring

Add candidate-level net cost to `_pack_one_width`.

Expected target:

- reduce case 96 HPWL gap from `1.07`
- reduce case 99 HPWL gap from `0.57`

### Experiment 2: Multi-Order Skyline

Try at least:

```text
current order
high weighted degree first
large area first
cluster first
```

Use current final selector.

### Experiment 3: Local Width Refinement

Around the best width, try nearby widths:

```text
W * 0.94
W * 0.97
W * 1.03
W * 1.06
```

Only for `block_count >= 116`.

### Experiment 4: Limited Shape Variants

Try shape variants only for frontier or high-whitespace cases, not all blocks.

Start with:

```text
aspect ratios: 0.8, 1.0, 1.25
```

Avoid extreme ratios first.

## Expected Ceiling

Current dominant-case cost is around:

```text
case 99: 1.84
case 98: 1.79
case 97: 1.68
case 96: 2.00
case 95: 1.60
```

If HPWL gaps could be reduced by about `0.10` to `0.20` and area gaps by about
`0.05` to `0.10` without increasing soft violations, weighted score could
realistically move from about `1.82` toward:

```text
1.65 - 1.75
```

Getting much lower likely requires either:

- learned candidate ranking and learned order/shape policies, or
- a stronger global placement plus legalization co-optimization loop.

## Main Conclusion

The current bottleneck is no longer feasibility or MIB. It is also no longer
mostly soft constraints. The bottleneck is now the quality gap caused by greedy
legalization:

```text
HPWL gap + area gap
```

The strongest next deterministic method is HPWL-aware skyline placement, because
it attacks HPWL at the moment decisions are made instead of only selecting after
the layout is complete.

The strongest ML/DL aid is not a full floorplanner at first. It is a candidate
ranker or parameter predictor wrapped around the deterministic legalizer. That
keeps feasibility guarantees while learning the parts that are hardest to tune
manually.

