# HPWL/Cost Improvement Notes

## Problem Observed

The previous `FloorSet/baseline_optimizer.py` initialized each B*-tree as a
single deterministic left-child chain. In this implementation, a left child is
packed immediately to the right of its parent, so the initial layout was often
nearly one long row.

That was especially bad for validation case 99:

```text
before:
  cost              39.7264
  feasible          yes
  HPWL gap          9.1924
  area gap          13.6111
  HPWL total        13831.07
  bbox area         507222.64
  soft violation    0.5821
```

The optimizer was spending effort on soft-constraint signals after starting
from a very poor geometry for HPWL and bounding-box area.

## Changes Made

### 1. Compact Shelf B*-Tree Initializer

Added `BStarTree.from_rows(rows)`.

The new initializer builds a multi-row B*-tree:

- row heads are connected by right-child links
- blocks inside one row are connected by left-child links
- the packer therefore starts from a dense shelf layout instead of one long
  horizontal chain

This directly reduces bounding-box width and makes the starting point much
closer to a realistic floorplan.

### 2. Shape-Aware Shelf Construction

Added `_build_shelf_tree(...)`, `_estimate_payload_dims(...)`,
`_target_shelf_width(...)`, and `_rebalance_rows(...)`.

The optimizer now estimates dimensions for both normal block payloads and
cluster macro payloads, then splits payloads into rows using a target shelf
width derived from total area and item widths.

This keeps the high-temperature macro stage and the low-temperature individual
block stage both compact.

### 3. Connectivity-Aware Payload Ordering

Added `_connectivity_order(...)`.

Instead of placing blocks purely by area or previous y/x coordinate, the
initializer builds an ordering from:

- block-to-block edge weights
- pin-to-block edge weights
- boundary-constrained payload priority
- payload area as a tie breaker

Strongly connected payloads are therefore more likely to be nearby before SA
starts, improving HPWL without relying only on random moves.

### 4. Low-Stage Order Refinement

Added `_refine_low_order(...)`.

After the high-temperature macro stage is expanded, the low stage keeps the
coarse high-stage spatial order but prioritizes higher-connectivity blocks
within local chunks. This preserves most of the macro placement signal while
still improving local HPWL.

## Validation Result

Command used:

```bash
cd FloorSet
MPLCONFIGDIR=/tmp/matplotlib python iccad2026contest/iccad2026_evaluate.py \
  --evaluate baseline_optimizer.py \
  --test-id 99 \
  --verbose \
  --data-path . \
  --output /tmp/opt99_finalcheck.json
```

Observed result after the change:

```text
after:
  cost              13.0935
  feasible          yes
  HPWL gap          about 2.18
  area gap          about 1.70
  runtime           about 8.16 s
```

An earlier run of the same changed optimizer reached `12.4459` on case 99:

```text
best observed after:
  cost              12.4459
  feasible          yes
  HPWL gap          2.0931
  area gap          1.8468
  HPWL total        4197.36
  bbox area         98828.25
```

The main improvement is HPWL/area quality:

- HPWL total improved from `13831.07` to about `4197-4319`
- bbox area improved from `507222.64` to about `93859-98828`
- cost improved from `39.7264` to about `12.4-13.1`

## Remaining Work

Soft violations are not solved by this change. On case 99, boundary and grouping
violations remain high, and in some runs the relative soft violation is worse
than before. This is an acceptable tradeoff for this iteration because the
request was to stop over-focusing on soft constraints and recover HPWL/wirelength
quality.

The next useful improvement should explicitly place boundary-constrained blocks
on the required global edges and preserve group adjacency during the low stage
without destroying the improved HPWL ordering.
