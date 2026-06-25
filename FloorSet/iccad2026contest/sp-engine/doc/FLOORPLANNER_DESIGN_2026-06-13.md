# A strong fixed-dimension floorplanner for the FloorSet Challenge

*Design + analysis, 2026-06-13. Goal: build the placement engine of `ml-engine/` that,
**given the (w, h) of every block**, places them to beat the analytic_legalizer baseline
(weighted score **1.8537**, 100/100 feasible) by a large margin. This validates the
"ML predicts dimensions → robust floorplanner places them" decomposition before any model
is trained.*

---

## 1. Why this experiment is the right first move (and why it should win big)

### 1.1 With dimensions fixed, the problem collapses to *placement only*

The contest cost (PDF Eq. 2) is

```
Cost = (1 + 0.5·(HPWL_gap + Area_gap)) · exp(2·V_rel) · max(0.7, RuntimeFactor^0.3)   [feasible]
     = 10                                                                              [infeasible]
```

Every term except RuntimeFactor is a pure function of the **positions** once `(w,h)` are
fixed:

| Cost term | Depends on dims? | Depends on placement? | With GT dims… |
|---|---|---|---|
| Area_gap (bbox) | only via block areas (fixed) | **yes** — how tightly we pack | pure packing quality |
| HPWL_gap | via centroids (fixed offset) | **yes** — centroid positions | pure placement quality |
| V_mib (equal shapes) | **yes** | no | **= 0 for free** (GT dims already identical within a MIB group) |
| V_grouping (abutment) | no | **yes** | placement/clustering quality |
| V_boundary (touch edge) | no | **yes** | placement quality |
| Area-tolerance (hard) | **yes** | no | **satisfied for free** (GT areas are within 1%) |
| Fixed/preplaced dims (hard) | **yes** | no | **satisfied for free** (we use the exact dims) |
| Overlap (hard) | no | **yes** | the packer's job |
| Preplaced position (hard) | no | **yes** | the packer's job |

So with GT dimensions, **the entire score is determined by the placement algorithm**, and
the two hard *dimension* constraints vanish. The only hard constraints left are
**no-overlap** and **preplaced position** — both are placement properties.

### 1.2 The ground-truth layout is an existence proof that score ≈ 1.0 is reachable

FloorSet is "optimal-by-construction": the GT layout is feasible *with exactly these
dimensions* and defines the baselines (`HPWL_baseline`, `Area_baseline`). Therefore, for
every case, **there exists an overlap-free placement of these exact rectangles that hits
HPWL_gap = Area_gap = 0 and ~0 soft violations → cost ≈ 1.0.** The placement problem with
GT dims is feasible by construction and bounded below by ~1.0. Our job is to get close.

### 1.3 The headroom is enormous, so even a mediocre placer wins

The baseline's documented gaps (analytic + skyline): **HPWL_gap 0.45–1.94**, Area_gap
0.23–0.85. The baseline pays for this twice: (a) it locks every soft block to a **square**
(`w=h=√area`) when the GT uses aspect ratios spread over **[1/3, 3]** (median 1.07,
only 6.5% of GT blocks are square), and (b) its **skyline strip-packer floor-packs the
layout, discarding the analytic 2D structure** (its own docs' stated ceiling). Our
experiment removes (a) entirely (GT dims) and attacks (b) with a structure-preserving
placer.

Quantitatively: even a placer that reaches **10% gaps** — the level the PDF's own PoC
dismisses as "struggling" — scores `(1 + 0.5·(0.10+0.10))·exp(2·V) ≈ 1.10` before
violations. That already **crushes 1.85**. The PoC's pessimism is about the *full* problem
(it must also size soft blocks); fixing dims collapses the search space and removes the
hardest DOF. **The decomposition is favorable: if this engine lands near ~1.1–1.3 with GT
dims, it proves the placement half is solved and the remaining contest problem is
"predict (w,h) well," which is exactly what the 1M-sample supervised dataset is for.**

### 1.4 What "win" means here

- **Primary validation target:** weighted score (GT dims) **< 1.85** → beats baseline →
  decomposition works.
- **Strong target:** **< 1.4**.
- **Stretch / "tremendous":** **≈ 1.1–1.2**, i.e. HPWL/area gaps in the ~10–20% range with
  near-zero soft violations and 100/100 feasible.

---

## 2. Exact rules the placer must honor (verified against `iccad2026_evaluate.py`)

These tolerances drive the implementation; getting them wrong silently flips feasibility.

- **Overlap** (`check_overlap`): a pair is overlapping iff `overlap_x > 1e-6` **and**
  `overlap_y > 1e-6`. Edge-touching (one of them ≤ 1e-6) is legal. → pack to exact
  coordinates; keep any numerical slop below 1e-6.
- **Preplaced** (`check_dimension_hard_constraints`, tol `1e-4`): final `(x, y, w, h)` must
  match input within `1e-4`. → preplaced must be emitted at their exact input coords.
- **Fixed-shape** (tol `1e-4`): `(w, h)` exact. With GT dims this holds; never reshape them.
- **Area tolerance** (1% relative, soft blocks only): holds automatically with GT dims.
- **Boundary** (`eps = 1e-6`): block must touch the **solution's own bbox** edge. Bitmask
  `LEFT=1, RIGHT=2, TOP=4, BOTTOM=8`; corners are ORs (TL=5, TR=6, BL=9, BR=10). The
  profile shows **only single edges and the 4 corners occur** (no 3/12 multi-edge codes).
  Because the bbox is derived from the placement, "left" blocks must sit at the global
  `x_min`, etc. — these couple all same-edge blocks into a flush row/column.
- **Grouping** (shapely `unary_union` → MultiPolygon): members must form **one** connected
  component via shared edges of **non-zero length, zero gap**. Floating-point abutment must
  be snapped so coincident edges are bit-identical (the baseline hit a real ULP bug here:
  ~1-ULP gaps split a group into spurious components). → snap macro-internal and any
  abutment coordinates.
- **MIB** (rounds `(w,h)` to 4 decimals, counts distinct): GT dims identical → 0.

### Dataset shape (validation, 100 cases, one per size 21–120) — drives priorities

| Constraint | Coverage | Per case (min/mean/max) | Design implication |
|---|---|---|---|
| blocks | all | 21 / 70.5 / 120 | n ≤ 120 → O(n²) packing is cheap |
| **preplaced** (hard) | all 100 | 1 / 2.6 / 9 | few but fatal if wrong → robust obstacle handling + feasibility floor |
| fixed-shape (hard) | all 100 | 1 / 7.1 / 17 | just fixed-dim units in the placer |
| **cluster/grouping** (soft) | all 100 | 3 / 3.6 / 4 groups, sizes 1–10 (mean 5.6) | rigid macros → V_grouping = 0 by construction |
| **MIB** (soft) | all 100 | exactly 1 group, size 3–7 | free with GT dims (no action) |
| **boundary** (soft) | all 100 | 11 / 24 / 37 blocks | the only nonzero soft term; ~70% edges, ~30% corners |
| b2b / p2b edges | all | up to 7056 / 4181 | HPWL eval must be fast & vectorized, **no shapely in inner loop** |

GT packs to **~97% density** (3% whitespace) in a slightly portrait bbox (W/H median 0.81).
That density is the area-gap bar; a structure-preserving packer can approach it, a
floor-packer cannot.

---

## 3. Recommended method: warm-started **Sequence-Pair Simulated Annealing**, exact-cost-driven

This is the classical state of the art for *fixed-outline floorplanning with
hard/soft/preplaced blocks and boundary constraints* — and it is exactly what the contest's
own reference list points to: Murata & Kuh, *"Sequence-pair based placement method for
hard/soft/pre-placed modules"* [29]; Lai et al., *"Module placement with boundary
constraints using the sequence-pair representation"* [17]; plus TCG [21] and CBL [10] as
alternative non-slicing representations. We choose **Sequence Pair (SP)** because:

- It represents **every** non-overlapping packing (it is P-admissible), unlike B*-tree
  which only represents bottom-left-compacted packings — compaction is exactly what hurts
  wirelength and top/right boundary satisfaction. (The repo's prior "Bstar bad" result was
  a *cold-start* B*-tree SA; the lesson is "warm-start + expressive representation," not
  "SA doesn't work.")
- It handles preplaced obstacles and boundary constraints with published techniques.
- Geometry realization is a longest-path on two constraint graphs — `O(n²)` at n ≤ 120 is
  trivially fast, and the **slack** it exposes is the lever for wirelength (§3.4).

> Note on independence from the baseline: SP-SA is a self-contained engine and does **not**
> import `analytic_legalizer`. A quick continuous/constructive layout may *seed* it (§3.5),
> but that seed is generic placement, not the baseline's skyline pipeline.

### 3.1 Constraint compilation (preprocessing)

Turn the raw constraints into placement primitives once per case:

1. **Grouping → rigid macros.** For each grouping group, arrange its members (GT dims
   known) into a single connected, edge-abutting rigid block; store member offsets; snap
   all internal abutment coordinates to bit-identical edges. Arrange members
   connectivity-aware (members heavily wired together adjacent) to cut internal HPWL, but
   freeze the macro afterward. Result: **V_grouping = 0 by construction**, and the group is
   one SP unit. (Edge case: a preplaced member pins the macro — see §3.3; detach it from
   the macro and accept ≤1 grouping violation if it conflicts, as the baseline does.)
2. **MIB:** no action — GT dims are already identical → V_mib = 0.
3. **Fixed-shape:** ordinary SP units with frozen `(w,h)`, no rotation.
4. **Preplaced:** **not** SP units. They are fixed obstacle rectangles at exact coords (§3.3).
5. **Boundary tags:** annotate each unit with its required edge/corner for §3.6.

SP units = {free/soft blocks, fixed-shape blocks, grouping macros}. Preplaced = fixed
obstacles.

### 3.2 Geometry realization (the packer)

From a sequence pair (Γ₊, Γ₋): unit *i* is left-of *j* if *i* precedes *j* in both
sequences; below *j* if *i* precedes in Γ₊ and follows in Γ₋. Build the horizontal and
vertical constraint graphs and compute lower-left coordinates by **longest path** (`O(n²)`).
This yields the minimal-area packing for that topology with **zero overlap by
construction**. Compute bbox, HPWL (vectorized over edges), and boundary/grouping status
analytically — **no shapely in the SA inner loop** (reserve shapely for a final audit only).

### 3.3 Preplaced (the hard, must-not-fail part)

Preplaced are few (mean 2.6) but a single misplacement = cost 10. Two-layer strategy:

- **In-search (soft):** insert each preplaced rectangle into both constraint graphs as a
  node **pinned** to its target coordinate, with edges to SP units per the SP relative
  order. Realize positions by a longest path that clamps pinned nodes to their fixed
  coords. If an SP order is inconsistent with a pinned coord (a unit is forced to overlap
  the obstacle), the realization leaves residual overlap; **penalize it heavily in the SA
  objective** so SA migrates toward SP orders consistent with the obstacles. Because the GT
  exists, at least one consistent SP exists.
- **Guarantee (hard):** the emitted solution always snaps preplaced to exact target and is
  validated overlap-free; we keep and return only the **best feasible** layout ever seen.
  A deterministic constructive fallback (shelf-pack movable units into the free space
  around the fixed obstacles, respecting boundary rows/columns) provides a feasible floor
  so we **never regress below 100/100 feasible**, matching the baseline's robustness.

### 3.4 Wirelength-driven positioning within the topology (the key quality lever)

Longest-path packing compacts everything to the bottom-left — minimal area but **not**
minimal wirelength, and it ignores the slack each unit has before it would collide. This is
precisely the structure the skyline packer throws away. After packing, run a **slack
redistribution** pass: each unit can shift within `[lp_low, lp_high]` (its longest-path
slack window in x and y independently) without creating overlap or growing the bbox; choose
the shift that minimizes weighted HPWL. This is a small separable optimization (per
coordinate, a weighted-median / monotone 1-D problem; or a few coordinate-descent sweeps).
It buys large HPWL reductions *for free on top of any topology* and is the main reason SP
beats skyline on wirelength.

### 3.5 Warm start (defeat the cold-start failure mode)

Cold-start SA on n=120 is the PoC's "struggles >10 min." Seed instead:

- Cheap continuous placement (quadratic/force-directed on the netlist with terminals as
  anchors and preplaced fixed) → sort units by the resulting centers to derive an **initial
  SP** (x-order → Γ relations). This injects the wirelength-optimal 2D relationships into
  the starting topology, then SA only *refines*.
- Start SA at **low temperature** (refinement regime), not the high-temp full-explore
  regime. This is the single most important deviation from the failed B*-tree attempt.

### 3.6 Boundary constraints (soft → penalty + realize)

V_boundary is the only soft term left and it is per-block. Handle in three ways, cheapest
first: (1) after packing, **slide** boundary-tagged units to touch the realized bbox edge
if their slack allows (LEFT/BOTTOM are usually free since packing origins there; RIGHT/TOP
need slack to the frontier); (2) add a **penalty** `exp(2·V_boundary/N_soft)` directly to
the SA objective so the search prefers SP orders that put left-blocks early and
right/top-blocks at the frontier (the [17] technique); (3) optionally bias macro/edge units
to sequence extremes. TOP is structurally hardest (the top edge isn't known until packing
finishes) — same issue the baseline has (it leaves 353 boundary violations total); beating
that is upside, not a requirement.

### 3.7 Objective and SA schedule

Because validation provides `HPWL_baseline` and `Area_baseline`, anneal on the **exact
contest cost** (minus the runtime factor, which is constant during a solve):

```
E = (1 + 0.5·(HPWL_gap + Area_gap)) · exp(2·V_rel) + P_overlap·(residual preplaced overlap)
```

with V_grouping = V_mib = 0 by construction, so `V_rel = V_boundary / N_soft`. At submission
time the test baselines aren't visible, but the gaps are monotone in raw HPWL and raw bbox
area, so substitute `HPWL/Ĥ + Area/Â` with estimates `Ĥ` (from the warm-start layout) and
`Â ≈ Σareas / 0.97`; the *ranking* of candidates is preserved. **Moves:** swap two units in
one sequence; swap in both; relocate a unit; rotate a free/soft unit or a macro (swap w,h —
legal, area-preserving; never rotate fixed-shape). **Schedule:** geometric cooling from a
low warm-start temperature, adaptive acceptance, restart-from-best on stagnation, wall-clock
budget tuned to ≈ baseline (~2.6 s/case; the runtime factor is weak, `^0.3`, capped at −30%,
so a few seconds is safe).

---

## 4. Validation plan (the actual experiment)

Build under `ml-engine/` (independent of `analytic_legalizer/`):

1. **`gt_dims_harness.py`** — mirrors `score_harness.py` but: for each validation case, get
   GT `(w,h,x,y)` via `ContestEvaluator._extract_baseline` (already returns `target_pos`),
   feed **GT (w,h)** (and preplaced GT (x,y)) into the new floorplanner, then score with the
   real `evaluate_solution` + provided baselines. Reports per-case HPWL_gap, Area_gap,
   V_boundary, feasibility, and the `e^(n/12)`-weighted total.
2. **Sanity ceiling first.** Place every block at its GT `(x,y,w,h)` and score. Expect
   **100/100 feasible, cost ≈ 1.0, weighted ≈ 1.0.** If not (e.g. grouping splits from ULP
   gaps), fix the harness/snapping before trusting any optimizer number. This de-risks the
   scorer/feasibility plumbing independently of the algorithm.
3. **Milestones (weighted score, GT dims):**
   - **M0** constructive init only → **100/100 feasible** (safety floor).
   - **M1** + slack wirelength pass + SP-SA refine → **< 1.85** (beats baseline ⇒ thesis holds).
   - **M2** tuned SA → **< 1.4**.
   - **M3** → **≈ 1.1–1.2** (placement effectively solved given dims).
4. **Diagnosis output:** decompose the residual into HPWL_gap vs Area_gap vs V_boundary per
   size bucket, so we know whether remaining loss is packing density, wirelength, or
   boundary — and whether the eventual ML dim-predictor needs to target shape *or* the
   placer needs more search.

---

## 5. Risks and fallbacks

| Risk | Mitigation |
|---|---|
| Preplaced inconsistency → infeasible | pinned-node packing + heavy penalty + always-return-best-feasible + constructive feasibility floor |
| SA too slow at n=120 | O(n²) longest-path, vectorized HPWL, no shapely inner loop, warm-start + low-temp refine, wall-clock budget |
| Grouping ULP splits | snap macro-internal abutment to bit-identical edges; audit with shapely only at the end |
| Slack wirelength pass too complex | fall back to min-area packing + a single global shift; still preserves 2D order ⇒ still beats skyline |
| TOP-boundary hard to satisfy | penalty + post-slide; it's soft, non-fatal; matches/【beats baseline's 353 |
| Submission-time baselines unknown | optimize raw HPWL + raw bbox area with `Â≈Σarea/0.97`; ranking-preserving |

---

## 6. Summary

With GT dimensions, the contest reduces to a **fixed-rectangle placement** problem that is
**feasible by construction** (the GT proves cost ≈ 1.0 is reachable) and has **huge headroom**
over the 1.85 baseline (whose square-shaping and floor-packing are both removed/replaced).
The recommended engine is a **warm-started, exact-cost-driven Sequence-Pair simulated
annealer** with: rigid grouping macros (V_grouping = 0), free MIB (V_mib = 0), obstacle-pinned
preplaced with a feasibility floor, boundary via penalty + slide, and — the key quality
lever the skyline baseline lacks — **slack-based wirelength redistribution within each
topology**. If this lands near ~1.1–1.3 with GT dims, the "ML predicts (w,h) → robust
floorplanner places" decomposition is validated and the next step is the dimension model.
