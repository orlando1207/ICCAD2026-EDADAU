# FloorDiff Stage-2: Minimal-Perturbation Constraint-Graph Legalizer (MPCG)

Date: 2026-07-16. Design only — no code in this step. Companion to
`2026-07-14-diffusion-repo-anatomy-and-contest-model-design.md` (the model) and
`2026-07-15-floordiff-implementation.md` (stage-1 implementation).

## 0. The problem, quantified from the actual predictions

Stage-1 measurements (`myrun`, 200k steps, best-of-16, 50 DDIM steps, all 100 validation
cases; contest-weighted `e^{n/12}` aggregates):

| Metric | Raw prediction | Ground truth (noise floor) |
|---|---|---|
| Mean center displacement / S | **0.015** | 0 |
| Weighted HPWL gap | **−0.001** | 0 |
| Bbox area gap | **+0.013** | 0 |
| Overlap ratio (Σoverlap / Σarea) | **0.024** | 0 |
| Soft violations, strict (eps ≈ 1e-4·S) | 50.4 | 3.8 |
| Soft violations, loose (1%·S tolerance) | **2.7** | 1.2 |

Three decisive observations:

1. **Overlaps are slivers.** 2.4% total overlap spread over ~7k pairs means typical
   penetration depths ≪ 1%·S. Blocks are essentially in the right place — they need
   *separating*, not *re-placing*.
2. **Soft constraints are almost satisfied.** 50 strict violations collapse to 2.7 at 1%·S
   tolerance: boundary blocks sit ~0.1–1 unit from the bbox edge, cluster members ~0.1–1 unit
   from abutment. The soft-constraint job is **snapping to exact contact**, not search.
   (MIB is already identically 0 — dims are tied at decode.)
3. **Quality must not degrade.** HPWL gap is already at/below baseline; the legalizer's moves
   must stay at sliver scale and be wirelength-aware, or it will destroy the model's main
   achievement.

This is precisely the regime that **minimal-perturbation, topology-preserving legalization**
was invented for. It is the opposite regime from what packing-style legalizers (B*-tree /
sequence-pair re-pack, skyline, Tetris) assume — those rebuild geometry from scratch and would
discard the near-optimal structure we already have.

## 1. What the literature offers

**Constraint-graph floorplan repair — FLOORIST** (Moffitt, Ng, Markov, Pollack, DAC'06 /
TODAES'08). The closest ancestor to our problem: given a *rough floorplan with overlaps*,
encode pairwise relative order as horizontal/vertical constraint graphs, then repair only
what is violated ("conflict-directed"), explicitly to *preserve the qualities of the original
layout*. Removed 100% of overlaps from Capo/FengShui/APlace layouts on IBM-HB at negligible
runtime with only a few % wirelength increase. Validates the core idea: derive the pair
order from the prediction, fix violations minimally within that order.

**Constraint-graph + LP macro legalization** (Chen et al., constraint-graph macro placement
for mixed-size designs; the XDP family). Standard industrial-strength recipe: overlap ⇒
negative slack on a constraint-graph edge; solve min-displacement as an LP over the graph.
Our variant differs in two lucky ways: *no fixed outline* (the graph is always satisfiable —
feasibility is unconditional) and *2-pin weighted nets* (exact HPWL is LP-representable, so
we can optimize true wirelength, not a proxy, during legalization).

**Constraint-graph MILP for dense heterogeneous components** (DAC'25 PCB legalization).
Modern upgrade where the H-vs-V separation *choice* per pair is a binary variable inside a
MILP, enlarging the solution space in high-density cases. Relevant as an upgrade path; at
our densities (sliver overlaps, n ≤ 120) heuristic direction choice + LP should suffice and
is orders of magnitude faster.

**Diffusion-native legality — ChipDiffusion (ICML'25), FlowPlace (2026).** ChipDiffusion
pushes legality *into sampling* via a potential `φ = Σ min(0, d_ij)²` with universal
guidance (legality 0.82 → 0.997 — but still not 1.0); FlowPlace adds a projection operator
during sampling for hard non-overlap. Two lessons: (a) guidance alone leaves residual
illegality — unacceptable here, where any overlap ⇒ cost 10, so a *deterministic geometric
post-pass is mandatory regardless*; (b) projection/guidance can be added to our sampler
later (§C.9 of the model doc) to *shrink* the legalizer's work, but never replaces it.

**SA-based repair — PARSAC / CA-SA** (Mostafa et al., Intel — the contest organizers'
solver). Constraints-aware SA that repairs constraint violations during annealing. Powerful
but stochastic and runtime-hungry; the contest scores runtime (uncapped slowness penalty).
Appropriate as a last-resort fallback, not the main path.

**Row-based legalizers (Tetris, Abacus).** Minimal-displacement principle is right, but they
assume rows/uniform heights; not applicable structurally. Not used.

## 2. Why MPCG fits this framework specifically

- The prediction supplies a **trustworthy relative order** (displacement 1.5%·S ⇒ the
  left-of/below relations between almost all pairs match GT). Freezing that order turns
  legalization into a *convex* problem — two small LPs — with no combinatorial search.
- **All contest soft constraints are linear** in block coordinates once the order is fixed:
  boundary = coordinate equals bbox min/max (bbox extremes are LP variables); grouping
  abutment = zero-gap equality on one axis + interval-overlap inequalities on the other;
  MIB = already handled upstream (dims never change during legalization moves).
- **Exact HPWL is linear** (weighted 2-pin Manhattan) — the LP can trade sliver moves
  against true wirelength, protecting the −0.1% HPWL gap.
- **Hard constraints are structural, not penalized**: preplaced = equality; fixed dims =
  untouched variables; soft-block area = dims not variables at all in the base pass
  (moves can't violate area); overlap-free = the constraint graph itself. The output is
  feasible **by construction**, matching the contest's infeasible = 10 cliff.
- n ≤ 120 ⇒ per-axis LP has ~360 structural variables and ≤ ~10k constraints. HiGHS
  (scipy ≥ 1.9, we have 1.18) solves this in milliseconds; total well under 1 s/case on CPU.

## 3. Algorithm design

Input: decoded prediction `(x, y, w, h)` per block (area-exact, fixed/preplaced exact, MIB
tied — all guaranteed by `floordiff.data.decode`). Optionally the top-k of the N sampled
seeds. Output: feasible `(x, y, w, h)`.

### Stage A — pair classification and constraint-graph extraction

For every unordered pair (i, j):
1. Compute predicted gaps on both axes: `gx = max(xi, xj) − min(xi+wi, xj+wj)` (negative ⇒
   x-overlap), same for `gy`. Overlap ⇔ both negative.
2. Choose the separation axis: the one with the **smaller violation depth** (classic
   FLOORIST/XDP rule) — i.e., separate along the axis where blocks already nearly clear each
   other; ties broken toward the axis that hurts weighted HPWL less (both centers' nets
   considered). Non-overlapping pairs get their (already satisfied) dominant-axis edge.
3. Edge direction follows predicted center order (`cx_i < cx_j` ⇒ H-edge i→j). Because
   direction always follows the coordinate order of the *prediction*, each axis subgraph is
   acyclic by construction.
4. **Preplaced pairs**: orientation derived from their known (input) geometry, not the
   prediction — they are already mutually consistent.
5. Prune with transitive reduction per axis (optional — only for LP size; correctness holds
   without it).

Robustness note (why a *full* pair cover, not just overlapping pairs): separating only the
currently-overlapping pairs can create new overlaps when blocks move. Covering every pair
with an H- or V-edge makes any LP-feasible point overlap-free, unconditionally.

### Stage B — two anchored LPs (x-axis, then y-axis)

Variables (x-pass): `x_i` for non-preplaced blocks, bbox extremes `x_min, x_max`, aux vars
for absolute values. Objective:

```
min   α · Σ_nets  w_net · |cx_i − cx_j|            (exact weighted HPWL, x-part; p2b too)
    + β · Σ_i     |x_i − x_i_pred|                 (anchor: minimal perturbation)
    + γ · (x_max − x_min)                          (bbox width → area gap)
```

Subject to:
```
x_i + w_i + ε ≤ x_j            for every H-edge (ε = tiny separation margin, e.g. 1e-6·S)
x_i = x_i_input                preplaced (hard equality)
x_min ≤ x_i,  x_i + w_i ≤ x_max                    for all i (defines the bbox)
x_i = x_min   /   x_i + w_i = x_max                boundary L / R attachments (see below)
x_i + w_i = x_j                cluster H-abutment equalities (spanning forest, see below)
```

Then the y-pass, identical in form (V-edges, T/B attachments, V-abutments), *plus* the
cross-axis interval-overlap inequalities required by H-contacts chosen in Stage C
(an H-abutting pair must overlap in y by ≥ δ for the shared edge to have positive length).

Weight scheme: β dominates on scale (anchor keeps moves sliver-sized), α preserves HPWL
directionality among equally-cheap moves, γ small (area gap is already only +1.3%; pushing
too hard on bbox fights the anchor). Initial values α=1, β=2/S, γ=0.5 — to be tuned on the
validation set; the contest-weighted official cost is the tuning metric.

### Stage C — soft-constraint attachments (the "snapping" layer)

- **Boundary**: for each boundary-constrained block, attach the required side(s) to the
  corresponding bbox extreme as an equality. Attachments are attempted **hard first**; if
  the LP reports infeasibility (rare — e.g., two full-width TOP blocks whose H-order forbids
  both touching), retry with attachments as high-weight penalties (`λ·|x_i − x_min|`,
  λ ≫ β), which keeps the LP always feasible and converts an impossible snap into a
  best-effort one. Corners are just two attachments.
- **Grouping**: per cluster group, build the *predicted contact graph* (pairs whose gap
  < 2%·S). Take a spanning forest; if the group is fragmented into components, connect
  nearest component pairs (closing the loose fragmentation, measured at only ~1–2 per case).
  Each chosen contact pair gets: contact axis = the axis of smallest predicted gap;
  zero-gap equality on that axis; overlap ≥ δ inequalities on the other axis. By
  construction the contact equality agrees with (tightens) that pair's Stage-A edge.
- **MIB**: nothing to do — dims are tied at decode and Stage B never modifies dims.

### Stage D — verify, fallback ladder, and selection

1. Exact-area re-snap (float hygiene: `w·h = a` to machine precision), preplaced/fixed
   re-stamped from inputs.
2. Verify with the **official checker semantics** (already mirrored exactly in
   `floordiff/evaluate.py`).
3. Fallback ladder, escalating only on failure:
   a. residual numeric overlap → increase ε ×10 and re-solve (LP, ms);
   b. infeasible with hard attachments → soften attachments (Stage C);
   c. pathological case (never observed so far) → greedy directional push ("Tetris on the
      constraint graph": longest-path pass per axis), which always terminates legally, then
      one anchored-LP polish.
4. **Best-of-k selection**: run the whole pipeline on the top-k prediction seeds (k = 2–4,
   selected by pre-legalization proxy) and keep the lowest *official* cost. Legalization is
   ms-fast, so this is nearly free and hedges against an unlucky frozen order.

### Stage E (optional, P2) — sequential-LP reshape refinement

The base pass treats dims as constants. Soft blocks, however, may reshape (that's the one
lever generic legalizers never have). A cheap refinement: alternate x/y LPs where soft-block
`w_i` becomes a variable in `[w_min, w_max]` (aspect ∈ [1/3, 3] band observed in data) and
`h_i = a_i/w_i` is linearized around the current point each round (2–3 rounds, trust-region
±5%, exact-area projection + verification after each round). Expected gain: shave the
remaining bbox-area gap (+1.3% → ~0) by flattening blocks on the critical bbox path — the
mechanism the analytic literature uses for fixed-outline satisfaction. Strictly optional:
ship the constant-dims legalizer first, add reshape only if the area gap is worth it.

## 4. Feasibility argument

- **Acyclicity**: each axis subgraph orders blocks by predicted center coordinate, so every
  edge points "rightward"/"upward" in that order ⇒ no cycles.
- **LP feasibility (base pass)**: with no fixed outline, any DAG of separations is
  satisfiable (place blocks along a topological order with enough spacing); preplaced
  equalities remain feasible because free blocks are unbounded on both sides; hard boundary
  attachments are the only possible source of infeasibility and they auto-demote to
  penalties. Hence the pipeline **cannot fail to produce a feasible solution** short of a
  software bug — and Stage D verifies with official semantics before emitting.
- **Quality preservation**: moves are bounded in practice by sliver depth + snapping
  distance (≪ 1%·S each, measured); the anchor term forbids drift; the HPWL term makes the
  LP choose, among minimal separations, those that shorten nets. On this input distribution
  the expected post-legalization deltas are: overlap 2.4% → 0; loose violations 2.7 → ≈ GT's
  1.2 floor (strict); HPWL gap −0.001 → −0.001 ± 0.005; area gap +0.013 → ≤ +0.013 (γ term
  pushes down; Stage E can push toward 0).
- **Runtime**: 2 LPs (+ rare re-solves) × ~10k sparse constraints via `scipy.optimize.linprog`
  (HiGHS, already installed) — milliseconds each; whole pipeline including verification
  < 1 s/case CPU-only, comfortably inside the runtime-factor sweet spot.

## 5. Rejected alternatives (and why)

| Alternative | Why not primary |
|---|---|
| Gradient polish / guided sampling only (§C.9, ChipDiffusion-style) | Leaves residual overlap (ChipDiffusion: 0.997 ≠ 1.0); any overlap ⇒ cost 10. Useful *upstream* to shrink legalizer work, never sufficient alone. |
| Sequence-pair / B*-tree re-pack of the prediction | Legal by construction but re-packs to a corner: destroys pin alignment, boundary geometry, and the −0.1% HPWL gap; ignores preplaced anchoring. Packing is for building layouts, not preserving them. |
| PARSAC-style CA-SA repair | Stochastic, runtime-heavy under an uncapped slowness penalty; overkill when violations are slivers. Keep as conceptual fallback only. |
| Full MILP with binary H/V choices (DAC'25 PCB style) | Optimal direction choice, but heavy; heuristic choice errs only on near-diagonal pairs, which the best-of-k seed hedge already covers. Upgrade path if Stage-A choices prove limiting. |
| Force-directed spreading (WL-driven, e.g. DREAMPlace-style fields) | No feasibility guarantee, needs tuning per case; the LP gives the guarantee for free. |

## 6. Validation plan & exit criteria

Implemented next as `floordiff/legalize.py` (library + CLI: `preds.json → legalized.json`,
same schema, so `floordiff.evaluate` and `floordiff.visualize` work unchanged).

1. **Correctness**: official-semantics feasibility = 100/100 cases (hard requirement).
2. **Quality**: contest-weighted official cost of legalized predictions; targets —
   HPWL gap within ±0.5% of raw, area gap ≤ raw, V_rel ≤ 0.02 (GT floor ≈ 0.05/Nsoft·case),
   runtime < 1 s/case.
3. **Diagnostics**: per-case Δdisplacement introduced by legalization (should be ≤ overlap
   sliver scale); which stage each case exits at (hard attach vs. soft fallback counts).
4. **Ablations**: γ sweep (bbox vs. anchor), hard-vs-soft boundary attachment rates,
   best-of-k (k = 1/2/4), ± Stage E reshape.
5. End state: a single `MyOptimizer.solve()` = featurize → sample (best-of-N) → decode →
   MPCG legalize → verified output; scored end-to-end with `iccad2026_evaluate.py`.

## Sources

- Moffitt, Ng, Markov, Pollack, *Constraint-Driven Floorplan Repair* — [DAC'06 paper](https://web.eecs.umich.edu/~imarkov/pubs/conf/dac06-fp.pdf), [TODAES'08 extended](https://dl.acm.org/doi/10.1145/1391962.1391975)
- *Constraint graph-based macro placement for modern mixed-size circuit designs* — [ICCAD](https://www.researchgate.net/publication/221627468_Constraint_graph-based_macro_placement_for_modern_mixed-size_circuit_designs)
- *Constraint Graph-Based PCB Legalization Considering Dense, Heterogeneous, Irregular-Shaped, and Any-Oriented Components* — [DAC 2025](https://dl.acm.org/doi/10.1109/DAC63849.2025.11133304)
- Lee et al., *Chip Placement with Diffusion Models* (ChipDiffusion, ICML 2025) — [arXiv:2407.12282](https://arxiv.org/html/2407.12282v2), [code](https://github.com/vint-1/chipdiffusion)
- *FlowPlace: Flow Matching for Chip Placement* — [arXiv:2604.23658](https://arxiv.org/html/2604.23658v1)
- Mostafa et al., *PARSAC: Fast, Human-quality Floorplanning for Modern SoCs with Complex Design Constraints* — [arXiv:2405.05495](https://arxiv.org/abs/2405.05495), [IntelLabs/parsac](https://github.com/IntelLabs/parsac)
