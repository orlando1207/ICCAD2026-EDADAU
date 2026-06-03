# Analytic-Legalizer — Algorithm Report & Upgrade Analysis

*Scope: `FloorSet/iccad2026contest/analytic_legalizer/`. Grounded in a live run of
`score_harness.py` on the score-dominant cases (95–99) on 2026-06-03.*

---

## 1. What we are actually optimizing

The official cost (`iccad2026_evaluate.py:compute_cost`) is:

```
Cost = (1 + α·(HPWL_gap + Area_gap)) · exp(β·V_rel) · max(0.7, R^γ)
       = 10.0  (M_PENALTY)  if infeasible
α = 0.5   β = 2.0   γ = 0.3
```

- **`HPWL_gap`** = `(hpwl − hpwl_baseline)/hpwl_baseline`, baseline = the FloorSet
  *ground-truth* layout (`_extract_baseline`). So a gap of 0.57 means "57% more
  wirelength than the reference optimum."
- **`Area_gap`** = `(bbox_area − area_baseline)/area_baseline`, same baseline source.
- **`V_rel`** = `(V_boundary + V_grouping + V_mib) / N_soft`.
- **`R`** = runtime ratio vs field median, damped (`^0.3`, floored at 0.7) — a weak term.
- The total contest score weights each case by `e^(n − n_max)`, so **n=120 (case 99)
  alone is ~64% of the score** and cases 95–99 are essentially the whole grade.

### Where the cost comes from today (measured)

| idx | n   | cost | HPWL_gap | Area_gap | V_rel | bnd | grp | mib | wt%  |
|-----|-----|------|----------|----------|-------|-----|-----|-----|------|
| 99  | 120 | 1.84 | **0.57** | 0.34     | 0.12  | 7   | 1   | 0   | 63.6 |
| 98  | 119 | 1.67 | 0.53     | 0.34     | 0.08  | 2   | 2   | 0   | 23.4 |
| 97  | 118 | 1.57 | 0.61     | 0.23     | 0.05  | 2   | 1   | 0   | 8.6  |
| 96  | 117 | 1.73 | 0.91     | 0.33     | 0.03  | 2   | 0   | 0   | 3.2  |
| 95  | 116 | 1.60 | 0.45     | 0.39     | 0.06  | 2   | 2   | 0   | 1.2  |

Weighted subset score ≈ **1.77**, 100% feasible.

**Decomposition of the dominant case (99):**
`quality = 1 + 0.5·(0.57 + 0.34) = 1.455`, `violation = exp(2·0.12) = 1.27`,
`runtime ≈ 0.7` floor → `1.455 × 1.27 = 1.84`. ✓

**Reading:** the cost is *quality-bound*, and inside quality, **HPWL is the single
biggest lever** (0.57 vs 0.34 area on case 99). Area is second. Boundary violations
(the 1.27 factor) are a clear third. MIB and grouping are essentially solved.

> Optimization priority, in order of payoff:
> **1. HPWL_gap → 2. Area_gap → 3. Boundary violations → (4. runtime, mostly irrelevant).**

---

## 2. Pipeline — detailed flow

Entry point: `optimizer.py:MyOptimizer.solve()`. Live modules: `constraints.py`,
`quadratic_placer.py`, `skyline_legalizer.py`. (`topology.py` and `shaping.py` are
**dead code** on the active path — only `test_steps.py` imports them; they are the old
longest-path/critical-path-shaping approach that skyline replaced.)

```
parse_and_init ─► prepack_clusters ─► analytic_place ─► skyline_legalize
   (Step 0+1)        (Step 2)            (Step 3)          (Step 4+5)
                                                              │
   enforce_hard ◄── slide_boundary ◄────────────────────────┘
   (Step 7)         (Step 6)                │
        │                                   └─► _local_slide_refine (compaction)
        └─► candidate selection by proxy = area · HPWL · exp(3·soft/n)
```

The whole Steps-3-through-7 chain is run once per *(seed × wirelength-model ×
skyline-config)* combination and the best candidate is kept. For `n ≥ 116` that is
`1 seed × 2 WL models × 8 skyline configs = 16 full legalizations`; otherwise 2.

### Step 0+1 — Parse & classify (`constraints.py:parse_and_init`)
- Builds `BlockInfo` per block: classifies preplaced / fixed-shape / soft, records
  `mib_group`, `cluster_group`, `boundary_code`.
- Initial shape: preplaced & fixed use exact `(w,h)`; soft blocks are squares
  `w=h=√area`.
- **Preplaced members are detached from clusters** (`cluster_group=0`): a rigid
  super-block pinned at a preplaced coordinate would collide with other pinned
  clusters. Concedes ≤1 grouping violation per affected cluster (intentional, since
  grouping is soft).
- **MIB unification** (`_unify_mib_shapes`): all members of a MIB group adopt one
  `(w,h)` — the anchor's if any member is preplaced/fixed, else a square of mean area.
  This drives `V_mib → 0` (confirmed: mib=0 everywhere).

### Step 2 — Cluster pre-pack (`constraints.py:prepack_clusters`)
Each cluster → one rigid `SuperBlock(members, offsets, w, h)`.
- No boundary members → `_shelf_pack` (sort by height, fill rows of width ≈ √area).
- Has boundary members → `_boundary_aware_pack`: frame layout (BOTTOM/TOP members in
  rows on those faces, LEFT/RIGHT in side columns, interior shelf-packed in the
  centre) so a member actually touches the wall once the box is placed against it.
- Anchor (preplaced member present) → `_pack_with_anchor` (anchor at origin, others
  stacked above so offsets stay ≥ 0).

### Step 3 — Analytic placement (`quadratic_placer.py:analytic_place`)
The *guide* positions (centers), not the final layout.
1. Node set = preplaced (Dirichlet-fixed) + one node per cluster super-block + free
   blocks.
2. Build quadratic Laplacian `A` from b2b/p2b weights (clique/star squared-wirelength
   model). Fix preplaced rows (Dirichlet); pins are fixed RHS terms.
3. Solve `A·cx = bx`, `A·cy = by` with dense `numpy.linalg.solve` (n ≤ 120 → trivial).
   If unanchored (rank-deficient Laplacian), fall back to a grid init.
4. **Optional LSE refine** (`wl_model="lse"`): gradient descent on a smooth
   `|Δx|+|Δy|` (HPWL-like) objective via `tanh(Δ/γ)`, step decayed `0.96^it` for
   80 iters — corrects the quadratic model's over-penalty on long nets.
5. **Spreading** (`_spread_nodes`): 30 iterations of O(n²) pairwise overlap repulsion,
   step cooled `0.95^it`.
6. Map super-block centers back to per-member centers.

### Step 4 — Skyline legalization (`skyline_legalizer.py:skyline_legalize`) — *core*
Deterministic constructive strip-packing into a fixed-width container.
- **Container width ladder:** for a set of target aspects `{0.85…2.8, analytic-bbox}`,
  `W = √(area/aspect)`, clamped to `W_min`. Each `W` is packed and scored.
- **`Skyline`** = contour of `[x_start, x_end, height]` segments tiling `[0,W]`.
- **Placement order** (`order_key`): BOTTOM-forced first (reach y=0 while empty),
  TOP-forced last, then a mode-specific key (`analytic` = by `(cy, cx)`, or `net`,
  `cluster`, `area`).
- For each unit, candidate x-positions come from segment starts / flush-right; landing
  `y = max skyline under [x, x+w]`, bumped up over preplaced obstacles (`_land_y`).
  Score = `y + λ·|x_center − cx| + net_weight·(incremental net cost)`. Pick min, raise
  the skyline.
- **Candidate score** (`_candidate_score`) across widths/orders:
  `area · exp(2·boundary_unmet/n) · hpwl^hpwl_weight`.

### Step 5 — Gap-fill finetune (`_finetune_fill_gaps`)
6 passes: relocate frontier (bbox-defining) free blocks into cluster-internal gaps
**only if the bbox area strictly shrinks**. Conservative; can't regress.

### Step 6 — Boundary slide (`constraints.py:slide_boundary`)
Slide RIGHT/TOP (and corner) blocks/clusters to the bbox edge, iteratively clipping
`dx/dy` to avoid creating new overlaps. LEFT/BOTTOM are already satisfied by the
packing origin. Mostly mops up the hard TOP case.

### Step 7 — Hard enforcement (`constraints.py:enforce_hard`)
Feasibility safety net: restore preplaced/fixed exact dims (8a), nudge soft `h` to hit
exact area within 1.1% (8b), `_resolve_overlaps` (≤80 push passes) + `_escape_overlaps`
grid-relocation fallback (8c), and `_snap_cluster_abutment` to close ULP-scale gaps that
shapely would otherwise count as a broken group (8d).

### Post-step — `_local_slide_refine` (in `optimizer.py`)
3 passes of greedy left/down slides on *unconstrained* frontier blocks, accepted only
if the selection proxy improves and no overlap is created. A lightweight compaction.

### Candidate selection
Keeps the layout minimizing `proxy = (x2·y2) · HPWL · exp(3·soft/n)`. The true gaps
can't be computed (no baseline at solve time), so `area·HPWL` is used as a surrogate —
which is well-aligned because HPWL and area are exactly the quality terms.

---

## 3. Upgrade analysis — what to replace, ranked by payoff

The pipeline is a **strong deterministic heuristic** but every quality term is left on
the table by a *greedy/constructive* step. Ranked by expected score impact:

### 🥇 A. Replace constructive packing with seeded local search on the true cost  *(biggest lever — attacks HPWL_gap directly)*

**Problem.** The skyline packer is a one-pass bottom-up strip packer. It achieves good
*density* (Area_gap ≈ 0.34) but only weakly respects wirelength: HPWL enters merely
through the `λ·|x−cx|` term and an optional incremental net cost. The result is a
0.45–0.91 **HPWL gap** — the dominant cost component — because packing order, not
wirelength, dictates the final layout. Strip-packing also **discards the analytic
y-structure** entirely (everything falls to the floor), so good vertical relationships
from Step 3 are lost.

**Replacement.** Add a **fixed-outline simulated-annealing / local-search refiner on a
sequence-pair or B\*-tree representation, *warm-started* from the skyline result**, that
minimizes the *actual* objective `area·(1+HPWL proxy)·exp(violations)`:
- Moves: block swap, rotate, single-block reinsert, and *segment shift* biased toward
  reducing HPWL on the highest-degree nets.
- Because n ≤ 120 and the start is already feasible & dense, even a short budget
  (a few thousand moves, ~1–2 s) recovers a large fraction of the HPWL gap. The contest
  *baseline itself* is B\*-tree SA — the repo's earlier "Bstar bad" result almost
  certainly came from **cold-start** SA; the fix is to seed from the analytic+skyline
  layout, not from scratch.
- Keep the existing pipeline as the initializer and the SA as a refinement stage; select
  by the same `area·HPWL` proxy.

**Alternative (deterministic, exact-ish):** for n ≤ 120 a **sequence-pair + linear-program
compaction** is tractable: fix the relative order from the skyline result, then solve an
LP that minimizes `Σ w·HPWL + bbox` subject to non-overlap (the SP induces a constraint
graph). This is a principled "analytic legalizer" and would directly cut HPWL while
holding area.

> Expected impact: this is where the 0.5–0.9 HPWL gap lives. Even halving it on case 99
> moves quality from 1.455 → ~1.31, i.e. cost 1.84 → ~1.66, on 64% of the grade.

### 🥈 B. Smarter aspect / width search → Area_gap

**Problem.** Container width is chosen from a fixed aspect ladder; Area_gap ≈ 0.33 is
stubborn because the floor-packing wastes the top-right frontier (only patched greedily
by `_finetune_fill_gaps` / `_local_slide_refine`).

**Replacement.**
- Replace the discrete ladder with a **golden-section / 1-D search on W** against the
  realized `area·HPWL` score (cheap — one pack per W).
- Replace the greedy gap-fillers with **constraint-graph compaction** (x-compaction then
  y-compaction along the HCG/VCG — the `topology.py` longest-path machinery already
  exists and is currently dead). This squeezes whitespace systematically instead of the
  current "move one frontier block if area shrinks" hill-climb.

### 🥉 C. Boundary satisfaction → V_rel  *(7 misses on case 99 → ×1.27)*

**Problem.** TOP boundary is "best-effort" (placed last + a clip-limited slide). Boundary
is the only meaningful violation left (15 across the 5 big cases), and it multiplies cost
by `exp(2·V_rel)`.

**Replacement.** Treat boundary as an explicit term in the local-search objective (A), or
add a dedicated **boundary-repair pass**: after legalization, for each unmet TOP/RIGHT
block, pull the *entire frontier set* it belongs to flush to the bbox edge using
compaction (not the current single-block clip-slide, which stalls when blocked). With the
SA refiner this becomes a soft penalty the search naturally minimizes.

### Lower-leverage / keep as-is

| Stage | Verdict |
|-------|---------|
| MIB unification (Step 1) | **Solved** (mib=0). Leave it. |
| Cluster pre-pack (Step 2) | Grouping nearly solved; shelf/frame packing is fine. A learned/optimal internal arrangement is low ROI. |
| Quadratic solve (Step 3) | Fine as a *guide*. The LSE refine already helps; a full ePlace-style electrostatic placer is overkill at n≤120 **and** is wasted if the legalizer ignores it — fix the legalizer (A) first. |
| `_spread_nodes` O(n²) | Cheap at n≤120; not a bottleneck. |
| Hard enforcement (Step 7) | Safety net, near-no-op; keep. |

### Optional D. Learned components (only after A–C)
Once the search-based core exists, ML can *accelerate* rather than replace it:
- A **GNN cost surrogate** to prune skyline configs / SA restarts without full evaluation.
- **Learned packing order or per-block λ** from connectivity (predict which blocks should
  lead the pack to minimize HPWL).
- **Predict the best container aspect** per instance from `(n, total_area, net degree
  distribution)`.

These are research-grade and only pay off after the deterministic objective-driven core
(A) is in place — they have no baseline to beat otherwise.

---

## 4. Recommended sequence of work

1. **Wire a warm-started SA/local-search refiner** (B\*-tree or sequence-pair) after
   `skyline_legalize`, optimizing `area·(1+0.5·HPWL_proxy)·exp(2·boundary)` with a small
   time budget. *(Attacks #1 HPWL and #3 boundary together; biggest single win.)*
2. **Swap the aspect ladder for a 1-D width search** and **replace greedy gap-fill with
   constraint-graph compaction** (reuse `topology.py`). *(Attacks #2 area.)*
3. Fold boundary into the search objective and add a frontier-set boundary-repair pass.
4. Only then consider ML surrogates (D) to cut the runtime of (1).

The current architecture (analytic guide → legalize → select) is sound; the win is to
**stop trusting a single greedy construction and let an objective-driven search refine
it**, exactly the gap between this solver (1.77) and the ground-truth baseline (1.00).
