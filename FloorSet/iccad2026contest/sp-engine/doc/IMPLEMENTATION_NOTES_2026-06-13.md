# SP-SA floorplanner — implementation notes & empirical findings

*Continuation of `FLOORPLANNER_DESIGN_2026-06-13.md`. This logs what was built in
`ml-engine/`, the experiments that drove the design, and the validated results of
the "ground-truth dimensions known" study.*

## Targets (established empirically)

- **Baseline** (`analytic_legalizer`, full 100): weighted **1.8537**, 100/100 feasible
  (boundary=353, grouping=113, mib=0).
- **GT-position ceiling** (place every block at its GT (x,y,w,h)): weighted **1.1079**,
  100/100 feasible. This is the *real* floor with GT dims — not 1.0 — because the GT
  layout itself leaves soft boundary violations (boundary=219, grouping=10 from ULP).
  So the achievable target band with GT dims is ~1.11 (placement-perfect) up to whatever
  the search reaches.

## Architecture (files in `ml-engine/`)

- `spcost.py` — vectorized HPWL / bbox / boundary (numpy), Python reference scorer.
- `compile_problem.py` — compiles constraints into SP **units**: free + fixed-shape
  blocks, preplaced (pinned obstacles), and optionally rigid grouping macros. Retains
  per-block cluster ids (`clu`) for grouping evaluation. Builds redistribution edge lists.
- `init_place.py` — warm start: force-directed attraction + ramped **area-equalization
  spreading**, then extract an SP from the spread layout's geometric left/below relations
  (NOT from center sorts — see finding 1). Boundary units biased to the layout extremes.
- `packer.py` — longest-path SP packing with preplaced pinning; **boundary-aware slack
  redistribution** (per-axis freeze keeps boundary blocks on their edge while wirelength
  tunes the free axis); constraint-graph adjacency for dynamic no-overlap bounds.
- `fastcore.py` — **numba-JIT** energy: pack → expand → HPWL → bbox → boundary →
  exact **grouping** (union-find over edge-adjacency, matches the evaluator) → cost.
- `anneal.py` — SA over the SP with O(1) swap moves + revert, **reheating restarts**,
  and coordinated **edge-pack** (boundary) and **cluster-pack** (grouping) moves.
- `floorplanner.py` — orchestrator: compile → seed → **parallel multistart** SA (each
  worker finalizes and returns its true final cost; best wins) → emit. Constructive
  feasibility floor guarantees a feasible result.
- `gt_dims_harness.py` — feeds GT (w,h) (and preplaced GT (x,y)) into the engine, scores
  with the official evaluator. `--ceiling` places at GT; `--macros` toggles macros;
  size-adaptive budget concentrates compute on the heavily-weighted large cases.

## Key findings (each changed the design)

1. **SP-from-centers is a trap.** A sequence pair extracted by sorting `cx±cy` produces
   staircase topologies → packed area ~2× GT even when fed GT centers. Extracting the SP
   from the *geometric* left/below relations of a spread+legalized layout fixes this; the
   longest-path packer then reproduces GT to ~1.1 (hgap≈agap≈0) on a clean placement.
2. **Rigid macros inflate area.** Forcing cluster members into a rigid shelf-packed
   rectangle floors big-case area gap at ~0.7 (macros can't interleave with non-cluster
   blocks the way GT does). Dropping macros and instead computing **exact grouping in the
   JIT energy** (so SA abuts clusters itself) + a **cluster-pack move** gives both tight
   area *and* near-zero grouping.
3. **Global placement alone is not enough.** Force-directed → legalize (no search) scores
   ~8.7; even with area-equalization spreading ~6.3. Topology *search* (SA) is essential —
   within-slack moves can't cross blocks, so only SA changes the tiling.
4. **Pure-Python SP-SA is too slow at scale** (~1.2k evals/s at U=96). The numba JIT
   energy gives ~18× (21k/s at U=96, 106k/s at U=58), making SA viable on n=120.
5. **SA cools out well before a large budget** and settles in a local min; **reheating
   restarts** + **parallel multistart** (different seeds, keep best-finalized) use the
   full budget and escape basins.
6. **Redistribution must be boundary-aware** — unconstrained slack moves push boundary
   blocks off their edge after SA placed them there; per-axis freezing preserves the
   boundary touch.
7. **Multistart must select on the *finalized* cost**, not the compacted SA proxy, or it
   picks topologies that compact well but finalize poorly.

## Results (GT dims)

Trajectory on the hard size-spanning subset {0,30,50,95,99} (weighted within subset):
5.1 (rigid macros, center-SP) → 3.5 (good seed + JIT) → 3.06 (edge-pack) → 2.70
(grouping-in-energy, macros off) → 2.61 (boundary-aware redistribute) → **2.53**
(multistart + finalized-cost selection).

Small/medium cases reach the ceiling band (idx0 n=21 → 1.20; idx30 n=51 → 1.55;
idx50 n=71 → 1.95). **The largest cases (n≈116–120) remain the bottleneck at ~2.5**
(HPWL/area gaps still high) — n=120 SP-SA does not fully converge even with multistart,
consistent with the contest's premise that classical search struggles at scale.

**Full-100 weighted score (GT dims, size-adaptive budget, 8–12 parallel starts):
~2.0** (1.99 at budget-15, 2.05 at budget-20 — within SA variance), 100/100 feasible,
boundary≈570, grouping≈70, mib=0. Versus **baseline 1.85** and **ceiling 1.11**.

### The decisive comparison (per-case, dominant instances)

| idx | n   | ours (GT dims) | baseline (own dims) |
|-----|-----|----------------|---------------------|
| 99  | 120 | 2.34           | 1.84                |
| 98  | 119 | 2.08           | 1.79                |
| 97  | 118 | 2.10           | 1.68                |
| 96  | 117 | 2.12           | 2.00                |
| 95  | 116 | 2.22           | 1.60                |

On small/medium cases we beat the baseline and reach the ceiling band (n=21→1.29,
n=51→1.55, n=71→1.95). **But on the large, heavily-weighted cases the baseline's
analytic+skyline (even with square dims) beats our from-scratch SP-SA (even with GT
dims).** Our big-case HPWL/area gaps (~0.5–0.7) are search-limited; the baseline's
analytic global placement is simply a stronger starting point at scale than our crude
area-equalization spread.

## Status vs. thesis — honest read

- **The floorplanner/legalizer is sound and not the bottleneck.** Given a GT-quality
  topology it reproduces GT to the ~1.11 ceiling (proven: edge-relation packing from GT
  positions, and GT-seeded SP). Constraints are handled: 100/100 feasible, preplaced
  exact, grouping≈0, MIB=0, boundary well-controlled on small/mid.
- **The bottleneck is the global-placement *seed* on large instances.** From-scratch
  SP-SA does not find a GT-quality topology for n≳110 in seconds; the baseline's analytic
  placer does better there. So with GT dims we land at ~2.0, *not* a "tremendous" win.
- **This sharpens the ML target.** The validated decomposition is now:
  `ML predicts (a) block dimensions AND (b) a placement guide / topology seed (the GT
  layout fp_sol / tree_sol are labels for exactly this) → the robust legalizer realizes
  it feasibly`. The legalizer is done; the hard, learnable part is the big-instance
  topology — which is precisely where a model trained on the 1M GT layouts should win.

## Fixed-outline targeting (preplaced/boundary → optimal bbox)

Insight (user): preplaced blocks are at fixed absolute coords and boundary+preplaced
blocks pin bbox edges, so they reveal the target bounding box. Data check (validation):
76/100 cases have a preplaced+boundary block; a RIGHT/TOP-preplaced pins W*/H* exactly in
~24/25 cases (both dims in only 5); preplaced *extent* reaches ~74%/70% of the GT bbox on
average (a strong lower bound). Combined with **area\* = area_base** (known here;
Σareas/0.97 at deploy), we derive target boxes.

Implemented: `_candidate_outlines()` computes (W\*,H\*) with W\*·H\*=area\*, using preplaced
lower bounds + boundary pins, sweeping the residual aspect freedom across the parallel
multistart workers. A **fixed-outline overflow penalty** in the JIT energy
(`e += w_out·(max(0,W−W*)/W* + max(0,H−H*)/H*)`) drives packing into the box. This made
grouping→0 robustly and improved small/medium cases.

**Key diagnostic (idx99, n=120):** the derived outline is essentially exact — we get
(131,265) vs GT (131,265). With a loose (force-directed) seed, SA packs to width ~195–199
(area_gap ~0.5–0.7) even with a strong penalty. The fix that worked: a **fixed-width
shelf-pack seed at W\*** — packing units into a width-W\* skyline gives a seed with x-chains
≤ W\* *by construction*, which SA cannot discover on its own. This dropped idx99 area_gap to
~0.48. Shelf seeds trade some HPWL (row-major packing ignores nets), so we **mix seed types
across the parallel multistart workers** (half shelf@W\* for compact area, half spread for
low HPWL) and let final-cost selection keep the winner per case.

Effect on the hard subset {0,30,50,95,99}: 2.53 (no outline) → 2.38 (penalty) → **2.21**
(mixed seed). idx99: 2.59 → **2.16** (area 0.48, boundary 9, grouping 0). HPWL on the very
largest cases (~0.80) is now the residual limiter — the joint area+HPWL optimum at n=120 is
still beyond from-scratch SA, but the bbox insight closed a real chunk of the gap.

## Root-cause: the warm-start seed was throwing away wirelength

Per-stage HPWL measurement (idx99, n=120; hpwl_base=1357):
- force-directed (Jacobi + area-equalization) seed, *before* legalization/SA: **+0.70**
- **exact quadratic solve** (direct linear system, preplaced as anchors): **−0.10** (≈ GT!)
- final SP-SA output (old seed): +0.48

So wirelength was lost in the **global-placement seed**, not the legalizer or SA: the
netlist alone reaches GT-quality wirelength (a quadratic solve proves it), but our crude
Jacobi+equalization seed smeared cells across the canvas and destroyed the 2D clustering.
SP-SA cannot recover from a globally-scrambled seed (topology swaps don't un-scramble it).

**Fix (init_place.py `_simpl_place`, method='simpl'):** replace the force-directed seed with
an exact quadratic wirelength solve (preplaced fixed anchors). The quadratic solution is
near-GT wirelength but clumped; the SP longest-path packer + SA + outline penalty then
spread it to feasibility. Effect on the hard subset {0,30,50,95,99}: 2.53 → **2.13**; big-case
HPWL roughly halved (idx99 0.82→0.59, idx95 0.80→0.47). A few 'shelf' seeds are mixed in for
diversity. (Note: a naive uniform-equalization "spreading" of the quadratic solution made
things worse — it re-smears wirelength; feeding the clumped quadratic straight into SP-SA and
letting longest-path spread it is better.)

This is a seed upgrade *inside* the SP-SA engine — analytic placement is the standard
warm-start for SP/B*-tree SA floorplanners; the SP-SA + legalizer core is unchanged.

Residual limiter after the fix: HPWL and area are now co-bound (~0.5 each) on the largest
cases — legalizing the clumped quadratic spreads it (area↑) and SA can't reach GT's joint
density+wirelength at n≈120. Closing that needs a proper density-aware spread (FastPlace/
ePlace-style, preserving locality) or the ML guide.

## Why GT dims don't beat the baseline (decisive analysis)

Per-case averages on the dominant band (n>=100), ours (GT dims) vs the real baseline
(`analytic_legalizer`, square dims, `my_optimizer_results.json`):

| metric | ours (GT dims) | baseline (squares) | GT ceiling |
|---|---|---|---|
| HPWL gap | 0.41 | 0.76 | 0 |
| AREA gap | 0.41 | 0.42 | ~0 |
| boundary vrel | 0.14 | 0.10 | — |
| total boundary viol. | 553 | 353 | — |

We *win* on HPWL (analytic seed) but **tie on AREA despite GT shapes** and **lose on boundary**.
The GT-dimension advantage should show up as low area (GT tiles to 1.03x); instead we pack GT's
rectangles into 1.41x — no tighter than the baseline packs squares. **The shape advantage is
squandered at the arrangement step.** Reason: dense packing of *varied* rectangles (aspect
1/3..3) is the NP-hard combinatorial core, and knowing the dimensions doesn't make it easy —
in fact varied rectangles are harder to tile than uniform squares. The score is dominated by
HPWL + area + boundary, all functions of the *arrangement* (positions), which GT dims do not
address.

## Outline-filling skyline legalizer

Built `skyline.py`: wirelength-aware skyline pack into the known width W*, boundary by
construction, preplaced as obstacles; added as cheap deterministic multistart candidates.
Result: **strong on area+boundary, weak on big-case HPWL** (bottom-up linearization scatters
connected blocks: idx95 HPWL 1.99). Exactly complementary to SP-SA (good HPWL, weak
area/boundary). Mixing them via best-of selection helps small/mid cases marginally but NOT the
weight-dominant big cases (skyline's HPWL loses there). Net full-100: ~1.86 (still ≈ baseline).

A height tolerance to keep blocks near their analytic x (preserve WL) made it *worse* (tall
columns). Minimal-displacement legalization can't help either: longest-path is already the
min-area realization of a topology, so area is bounded by the *topology's* density, and the
analytic-derived topology has ~40% whitespace because clumped quad positions give a noisy
order. **Dense rectangle tiling needs the right combinatorial order, which no constructive
legalization recovers from analytic positions.**

## Conclusion: this is a placement (arrangement) problem, not a shaping problem

Both our method and the baseline cap at ~0.4 HPWL/area gaps because the binding difficulty is
the **combinatorial dense arrangement of rectangles with short wires on the right boundary** --
NP-hard, unaffected by knowing dimensions. We proved the legalizer reaches the ~1.11 ceiling
*given a GT-quality arrangement*; the missing piece is producing that arrangement. So the
project thesis must shift: **ML should predict the arrangement (GT positions `fp_sol` /
structure `tree_sol`), not the dimensions** -- the floorplanner is the legalizer that realizes
it feasibly. Predicting dimensions alone is provably insufficient (this study).

## Boundary repair (the one structural deficit we could fix)

The analysis showed two deficits vs baseline: area (tied despite GT dims) and boundary
(553 vs 353). Area is the NP-hard arrangement wall. Boundary is fixable: we carry ~40%
whitespace, so boundary-violating blocks can be *relocated onto their edge using that
whitespace*. `boundary_repair()` (floorplanner.py): for each violating free block, find the
lowest free slot in the required edge column/row and move it there (overlap-safe; clustered
and preplaced blocks excluded to preserve grouping/feasibility); kept only if the true cost
improves. Effect on subset {0,30,50,95,99}: boundary 34->13, vrel ~0.16->0.06, subset
2.10 -> **1.87**; the dominant cases gained (idx99 2.11->1.90, idx95 2.10->1.84).

**Clean A/B (same solutions, isolating the repair from SA run-to-run variance), full 100:**

| | weighted | boundary | feasible |
|---|---|---|---|
| analytic-seed SP-SA (no repair) | 1.8934 | 553 | 100/100 |
| **+ boundary repair** | **1.7003** | 361 | 100/100 |
| baseline (squares) | 1.8537 | 353 | 100/100 |

**Boundary repair takes 1.89 -> 1.70, finally BEATING the baseline (1.85) with GT dims**, and
brings boundary to baseline level (361 vs 353). The repair is gated per case (kept only if the
true cost improves), so it can never hurt; the +0.19 is clean. (Full *pipeline* runs show SA
run-to-run variance of ~+/-0.1 on the big cases, so a single end-to-end run lands ~1.72-1.80;
more multistarts suppress it. The A/B isolates the repair's true contribution.)

This is the SP-SA path (good HPWL from the analytic seed) finally getting skyline-quality
boundary by construction -- the complementary strengths combined without the skyline's HPWL
penalty.

**Confirmed end-to-end (fresh run, budget 24, 16 starts): weighted 1.7807, 100/100 feasible,
boundary 372.** Verified with the OFFICIAL scorer: `python iccad2026_evaluate.py --score
ml-engine/sols_final.json` -> Total Score 1.7807 (identical), confirming our harness == the
official evaluation. Best (low-variance SA run) reaches 1.70 (A/B). Either way it BEATS the
baseline 1.8537 with GT dims.

Net session arc: 3.24 -> 2.07 -> 1.89 (analytic seed) -> **1.78 end-to-end / 1.70 best**
(boundary repair), vs baseline 1.85 and ceiling 1.11.

## Recommended next steps (in priority order)

1. **Fixed-width constructive seed at W\*** (most promising untried lever): we now know the
   target width W\* accurately. A skyline/shelf pack of the movable units into a strip of
   width W\* (preplaced as obstacles) gives a seed that is compact *by construction*
   (width ≤ W\*), unlike force-directed spread. Extract the SP from that pack and let SP-SA
   refine HPWL/boundary under the outline penalty. This directly attacks the n=120 packing
   wall (the seed already fits the box). A basic shelf packer is standard construction, not
   the baseline's skyline.
2. **Stronger global-placement seed**: density-aware analytic placer (quadratic + bin
   spreading / ePlace electrostatics) → edge-relation SP. Same goal as (1) via a different
   route.
3. **ML placement guide**: regress GT block centers (or `tree_sol`) with a GNN; convert to
   an SP seed; SP-SA + legalizer polish. The thesis payoff; targets the big-instance gap.
4. Cheaper wins: per-size budget/starts tuning; smarter big-move scheduling; a light
   GT-shape rotation DOF (currently disabled).
