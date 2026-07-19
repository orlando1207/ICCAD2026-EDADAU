# EGL: ePlace-Gradient + Graph Legalizer (stage-2 rebuild)

**Goal.** Replace the MPCG LP/MILP legalizer (`floordiff/legalize.py`, ~1.11 score at
7.7 s/case average, 20–48 s on the largest cases) with a legalizer built on the
reference works in `FloorSet/reference/` (DREAMPlace DAC'19, ePlace TECS'15,
NTUplace3 TCAD'08) that runs in **well under 1 s/case on CPU** at comparable quality.
Official RuntimeFactor = `runtime / median_of_all_submissions` per case with exponent
0.3 (floor 0.7), so wall-clock directly buys leaderboard score.

**Input.** A near-legal diffusion prediction `(n,4) xywh` (n = 21–120, overlap ratio a
few %, dims area-exact). **Output.** Feasible `(n,4)` float64: zero overlaps
(official: pair violates only if overlap > 1e-6 on *both* axes), fixed/preplaced dims &
positions exact (tol 1e-4), soft-block areas exact; soft constraints (boundary touch
eps 1e-6, cluster edge-connectivity via `unary_union`, MIB identical dims after
`round(4)`) satisfied when profitable — each violation costs ~`2/N_soft` in
`exp(2 V/N_soft)`.

## Why these papers map onto this problem

- **ePlace**: overlap removal as gradient descent on `W(v) + λ N(v)` with Nesterov +
  Lipschitz step prediction (`α = 1/L̃`, `L̃ = ||Δ∇f||/||Δv||`) and Jacobi
  preconditioning `h_i = deg_i + λ A_i` (mandatory here — every block is a "macro"
  with wildly varying area). At n ≤ 120, direct O(n²) pairwise overlap gradients
  replace the DCT Poisson solve (the spectral machinery pays off at ≥10⁴ cells; a
  14k-pair tensor op is microseconds). λ ramps up until penetration is tiny.
- **DREAMPlace**: the whole pipeline as vectorized tensor ops; warm starts make GP
  converge in far fewer iterations (our diffusion output is the warm start); its LG
  is "Tetris then Abacus" — greedy ordering then per-axis minimal-movement.
- **NTUplace3**: legalize-by-nearest-legal-position in a priority order; **look-ahead
  legalization** (evaluate the *legalized* cost of intermediate states, keep the best
  snapshot); shrink/reshape as a remedy when chains don't fit a fixed span.

## Pipeline (per candidate)

```
stamp hard constraints
  → G: gradient phase (ePlace-lite Nesterov, ~100–300 iters, numpy, O(n²+E)/iter)
  → L: constraint-graph exact legalization (build → repair → assign)
  → P: per-axis L1 coordinate-descent polish (Abacus/median sweeps)
  → S: soft-constraint snapping (boundary, cluster abutment) with slack intervals
  → verify (official semantics) → proxy cost
```

### Stamp
`pre` blocks: xywh = target exactly, immovable. `fixed`: dims = target. MIB groups:
dims copied from the group representative (fixed/preplaced member wins) — MIB is then
satisfied by construction and never re-broken (reshape excludes MIB members).

### G — gradient phase (all vectorized numpy float64, movable mask = ~pre)
**As-built note:** plain Nesterov + Lipschitz stepping (ePlace Alg. 2) *diverged* on
these near-legal warm starts — momentum keeps drifting after pairs separate, and a
force ∝ perpendicular extent doesn't decay with penetration, so blocks flew apart.
The shipped G phase keeps the ePlace force/charge structure but applies it as a
**penetration-proportional impulse solver** (~80 iterations, `omega = 0.8`):
1. overlap impulse: each penetrating pair separates along its cheaper axis by its
   full depth, mass-shared `A_j/(A_i+A_j)` (ePlace charge preconditioning; preplaced
   absorb nothing); deterministic index tiebreak at zero center distance;
2. spring impulses (decoupled from the drift cap): boundary blocks within 5%·S of
   their required side pull toward it; cluster proximity-forest pairs (rebuilt every
   20 iters) close their rectangle gap at rate 0.4, capped 0.01·S per iter;
3. quality drift, capped at 0.002·S per block per iter: exact weighted-HPWL gradient
   (`|d|` smoothed as `sqrt(d²+γ_s²)`, γ_s = 1e-3·S), bbox-area subgradient
   (softmax-shared over near-extreme blocks), weak anchor to the prediction; Jacobi
   preconditioned by `w-degree + A_i/S²` (ePlace Eq. 31-33).

G is deliberately short (local cleanup): at 97% utilization the coordinated global
shifts are the graph stage's job. More iterations measurably *hurt* (A/B tested).

### L — constraint-graph legalization (guarantees zero overlap by construction)
- Pair axis choice from G geometry: H if x-gap ≥ y-gap else V. Every edge is
  oriented by a FIXED per-axis total order (the G centers), so both graphs stay
  acyclic through all repairs and `argsort(key)` is a valid topological order.
- **Walls**: preplaced boundary-anchored blocks pin the bbox walls (the bbox edge
  must come to them — they cannot move). Lower wall = min L/B-anchor position,
  upper = max R/T-anchor extent, each dropped if another preplaced lies outside
  (some GT anchors are mutually inconsistent; the wall satisfies those on it).
- **Repair** with the lower wall as a longest-path source: when a preplaced anchor's
  lower bound exceeds its fixed position — (1) reshape shrinkable path blocks
  (area-exact, aspect ≤ 3.6, shrink ≤ 25%); (2) flip the path edge with the most
  other-axis gap. `extent_repair` additionally drives both axes' critical paths
  toward the predicted extents (reshape first, then flips priced by other-axis
  chain growth via pure-size head/tail).
- **Assign** per axis in topological order:
  `x_i = clip(target_i, max_p(x_p + w_p), U_i)`, `U` backward-propagated from
  preplaced anchors and the wall. Targets: G positions; movable boundary blocks
  target the walls; **contact intents** — cluster forest pairs close in G become
  exact abutment targets (follower targets its leader's edge; the graph edge
  guarantees ordering) so contacts form during assignment, when there is still
  slack, instead of being slid into a packed layout afterwards.
- **Wall ladder** on penetration: pinned walls → unpinned walls (bbox still bounded
  by `max(critical path, predicted extent)`) → wall-free re-repair + unpinned walls
  → no walls. Each rung strictly more feasible; the last cannot penetrate.

### P — polish (Abacus/NTUplace DP analog)
4 sweeps of exact per-block coordinate descent: block i moves within its FRESH slack
interval (recomputed per block — stale intervals can create overlap) to the
**weighted median** of its connected centers/pins (HPWL is L1 ⇒ exact 1-D optimum).
Moves are capped at the entry bbox: polish may only improve HPWL, never area (an
uncapped median chase toward far pins inflated bboxes 25%).

### S — snapping (official semantics, slack-guarded, profit-gated)
A snap happens iff feasible AND its **exact** HPWL delta (recomputed over the
block's incident nets) costs less than the `2/N_soft` violation it clears.
- Cluster first, two passes: single-block slide to exact contact (perpendicular
  shared edge ≥ 1e-3·S), else **rigid component slide** — the whole touching
  component moves together, slack intersected over members vs non-members
  (single blocks are often wedged in a packed layout; components still fit).
- Boundary last (attachments exact at exit), two passes; corner blocks need both
  axes feasible or neither moves. Preplaced never move; bbox never grows.

### Retry rungs (per candidate)
Round 2 rebuilds the graph from the round-1 legal geometry (cleaner axis choices);
best post-snap proxy wins. If ≥ 2 boundary violations remain and preplaced anchors
exist, one `span=True` retry drives the extents into the anchors' span (aggressive
reshaping pays only where anchors are otherwise unsatisfiable).

## Seeds (look-ahead legalization, NTUplace §2.2)
`legalize_best_of(top-k)`: run the pipeline per candidate, keep best proxy; stop
early below 1.05; per-case exploration budget 4 s. (A longer-G escalation pass was
tried and measured to contribute exactly nothing — removed.)

## Measured (100 validation cases, offline from stored top-4 predictions)
Contest-weighted proxy 1.137 (old MPCG official: 1.113) at **0.9 s/case average
legalization, max 4.6 s** — vs MPCG's 7.7 s average / 48 s max. All 100 feasible.
No LP/MILP anywhere.

## Files
- `floordiff/legalizer.py` — new module (this design). CLI mirrors old one:
  `python -m floordiff.legalizer --pred ... --out ...`.
- `floordiff_optimizer.py` — import switched to `floordiff.legalizer`.
- `floordiff/legalize.py` — **deleted** (MPCG retired; history in git).
