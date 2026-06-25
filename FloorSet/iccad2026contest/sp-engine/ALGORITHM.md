# SP+SA Floorplanner — Algorithm Overview

This document describes the end-to-end algorithm implemented in `ml-engine/`.
The entry point is `SPFloorplanner.solve()` in `floorplanner.py`.

---

## 1. Problem representation — Sequence Pair (SP)

Every block is a **unit** in the sequence pair.  A pair of orderings `(P, N)`
(also written Γ⁺, Γ⁻) over U units encodes a unique topology:

| Relationship | Condition |
|---|---|
| Unit a is **LEFT of** b | a appears before b in **both** P and N |
| Unit a is **BELOW** b | a appears before b in P but **after** b in N |

This is the Wong (1996) convention.  Every valid (P, N) maps to an
overlap-free, compact floor plan via longest-path packing.

When `use_macros=False` (our default), cluster members are kept as individual
units so boundary constraints can apply per-member.  With `use_macros=True` a
cluster is collapsed into one rigid SuperBlock — this breaks per-member
boundary codes and was found worse in practice.

---

## 2. Block dimensions and baseline calibration (`solve()`)

Before running SA the solver derives block shapes and energy baselines:

- **Dimensions** — free blocks get `w = h = sqrt(area_target)` (unit aspect).
  Fixed-shape and preplaced blocks use `target_positions`.
- **`hpwl_base`** — the SA energy denominator, intended to match GT HPWL.
  A pure analytic placement underestimates badly for dense netlists (case 99:
  7056 edges, analytic = 0.42× GT).  Two estimates are taken and the max wins:
  1. Analytic HPWL from `_simpl_place` (force-directed + area-equalization).
  2. Connectivity-based: `3.0 × Σ edge_weights × mean(√block_area)`.
- **`area_base`** — sum of all block areas (fill fraction ≈ 97 %, close enough
  to GT area).

---

## 3. Warm-start SP construction (`init_place.py — initial_sp`)

Each parallel worker builds its own starting `(P₀, N₀)`:

1. Run `_spread()` — alternating force-directed attraction + area-equalization
   spreading (40 iterations) to obtain analytic centers `(cx, cy)`.  Preplaced
   blocks are fixed anchors throughout.
2. Legalize the centers with longest-path `pack()` or a height-sorted
   shelf-packing, producing a zero-overlap layout.
3. Read off the SP from the legalized layout's geometric relations:
   for every ordered pair `(a, b)` check whether `a` is to the left of `b` or
   below `b`, and build `(P, N)` consistent with those relations.
4. Workers with index `s % 4 == 3` use the shelf method; the rest use the
   simpl (spread+pack) method — providing topological diversity across starts.

---

## 4. Simulated annealing (`anneal.py — anneal()`)

SA perturbs `(P, N)` for `time_budget` seconds, cooling `T *= 0.985` per inner
loop of `max(40, 6U)` moves.

### Move types (enable_rotation=False)

| Probability | Move | Description |
|---|---|---|
| 10 % | **edge_pack** | Push all units of one edge class (L / R / T / B) to their required extreme in P and N (see §5). Only attempted when boundary-constrained blocks exist. |
| 8 % | **cluster_pack** | Move all units of one cluster so they are **consecutive** in both P and N, centered at the group's current median position — encouraging the abutment the contest requires. |
| 82 % | **swap** | Swap two positions in P, in N, or in both simultaneously. 18 % of swaps are boundary-biased: pick a boundary block and swap it toward its required extreme position. |

### Energy function

```
E = hgap + agap + W_RES × residual + outline_penalty + cluster_penalty

hgap = max(0, (HPWL - hpwl_base) / hpwl_base)
agap = max(0, (bbox_area - area_base) / area_base)
```

`HPWL` and `bbox_area` are computed via the Numba-JIT `energy_nb()` (O(U²)
via the packing DAG).  `W_RES = 25` penalises preplaced residual overlap.
`outline_penalty` discourages exceeding a target `(W*, H*)` box.

### Acceptance and reheating

- Accept if `ΔE < 0`; otherwise accept with probability `exp(−ΔE / T)`.
- Big moves (edge_pack, cluster_pack) snapshot the full `(P, N)` before
  applying and restore it on rejection (O(U) cost but infrequent).
- Swap moves use O(1) in-place `_swap` with `undo`.
- When the time budget allows a second pass, the SA **reheats** to `0.6 × T₀`
  from the best feasible solution found so far, then cools again — similar to
  iterated local search.
- Returns the best **feasible** `(P, N)` (residual = 0); falls back to best
  energy `(P, N)` if no feasible was ever found.

---

## 5. SP → geometry: longest-path packing (`packer.py — pack()`)

Given `(P, N)` the packing is a single forward pass in P order:

For unit `j` at position `k` in P:
- **left predecessors** of `j` = units that appear before `j` in P AND have
  smaller N-position (i.e., are **left of** `j`).
  `x_j = max(x_pred + w_pred)` over left predecessors; 0 if none.
- **below predecessors** of `j` = units before `j` in P with **larger**
  N-position (i.e., are **below** `j`).
  `y_j = max(y_pred + h_pred)` over below predecessors; 0 if none.
- Preplaced units are pinned to their target coordinates; any unavoidable
  overlap is added to `residual` and penalised in the SA energy.

This is an O(U²) DAG longest-path and produces the **canonical compact**
packing for the given topology — no block can shift left or down without
violating a constraint-graph edge.

---

## 6. Wirelength slack redistribution (`packer.py — redistribute()`)

After packing, each unit sits at the leftmost/bottommost position allowed by
its topology.  `redistribute()` slides units rightward/upward within the
**slack** their constraint-graph neighbours allow, minimising HPWL:

- Iterates units in descending net-degree order for 12 Gauss-Seidel sweeps.
- For each unit `u`, dynamic bounds `[x_lo, x_hi] × [y_lo, y_hi]` are derived
  from the current positions of its HCG/VCG predecessors and successors.
- The target coordinate is the **weighted median** of net-neighbour centres
  (including fixed pins).  The unit moves to `clamp(target, lo, hi)`.
- **Boundary freeze** (SPFP_BFREEZE=1, the default): units with LEFT/RIGHT
  codes have their x-coordinate frozen; units with TOP/BOTTOM codes have y
  frozen — preserving edge-touch positions that the SA achieved.

---

## 7. Multi-start selection and fallbacks (`floorplanner.py — solve_with_dims()`)

`k` workers run in parallel (multiprocessing fork pool):

- Each worker receives a different seed and a different **target outline
  `(W*, H*)`**, swept from the aspect ratio imposed by preplaced blocks up to
  a 5:1 extreme.  The outline biases the spread seed and penalises SA
  solutions that exceed the target box.
- Additionally, `k/2` **skyline-legalized** candidates are generated from the
  analytic `_simpl_place` positions — these are fast, zero-violation
  alternatives that the SA results compete against.
- The best candidate is selected by `score_layout`:
  `cost ≈ (1 + 0.5(hgap + agap)) × exp(2 × v_rel)` where `v_rel` counts
  soft-constraint violations normalised by `n_soft`.
- If every SA worker returns an infeasible result and all skyline candidates
  fail, a deterministic `_constructive_floor` (shelf-pack) is used as a
  guaranteed feasible fallback.

---

## 8. Post-selection boundary repair (`floorplanner.py — boundary_repair()`)

After the best SA + skyline candidate is chosen, a final geometric pass tries
to satisfy any remaining boundary violations without the SP:

- For each block with a boundary code, try to slide it to its required edge
  (x=0 for LEFT, x=x_max for RIGHT, etc.) by scanning candidate y-slots (or
  x-slots) along the target edge.
- A slot is accepted only if the block fits there overlap-free.
- Corner blocks (two-axis constraints) must satisfy both axes simultaneously.
- The repaired layout replaces the original **only if** it is overlap-free and
  strictly lowers the cost.
- This is a pure coordinate-level move — no SP is involved — so it operates
  on any layout regardless of how it was produced.
