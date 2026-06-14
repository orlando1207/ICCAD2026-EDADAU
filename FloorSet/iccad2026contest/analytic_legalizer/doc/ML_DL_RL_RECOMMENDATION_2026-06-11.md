# Beating the analytical method with ML/DL/RL — strategy & recommendation

*Context: the analytic-place → skyline-legalize pipeline is now at weighted score
**1.8537** (100/100 feasible) and the per-instance heuristic is hitting a ceiling
(see `IRLS_L1_WIRELENGTH_2026-06-11.md`). This doc recommends how to use the contest's
1M training set to push past it. Grounded in the actual data format and cost function.*

---

## 0. The single most important fact

**The contest gives you the answers.** Each of the 1M training samples
(`lite_dataset.py`, `iccad2026_evaluate.py:1467`) is:

| Tensor | Shape | What it is |
|--------|-------|-----------|
| `area_target` | `[n]` | per-block target area |
| `placement_constraints` | `[n,4..5]` | fixed / preplaced / mib / cluster / boundary |
| `b2b_connectivity` | `[E,3]` | block–block net edges `(i, j, w)` |
| `p2b_connectivity` | `[E,3]` | pin–block edges `(pin, block, w)` |
| `pins_pos` | `[P,2]` | fixed pin coordinates |
| **`tree_sol`** | `[n−1, 3]` | **GT structural tree** (B\*-tree-style relative-placement encoding) |
| **`fp_sol`** | `[n, 4]` | **GT layout** `(w, h, x, y)` per block |
| `metrics_sol` | `[8]` | baseline metrics (incl. `area_baseline`, `b2b_wl`, `p2b_wl`) |

Plus a ready-made **differentiable cost** `compute_training_loss_differentiable`
(`iccad2026_evaluate.py:1263`) — the exact scoring formula with a smooth overlap/area
proxy, `loss.backward()`-able.

Two consequences that determine the whole strategy:

1. **The cost gaps are clamped at zero** (`max(0, hpwl_gap)`, `max(0, area_gap)`). The
   baseline *is* the GT layout. So a solution that reproduces the GT scores
   `gap ≈ 0` → cost `≈ 1·exp(2·V)·R`. **The GT is the ceiling, and it is given to you.**
   The entire game is "get closer to the GT than the heuristic does" (our HPWL gap is
   0.45–1.94 — enormous headroom). That is a **supervised / imitation** problem, not a
   from-scratch exploration problem.

2. **You get the GT *structure* (`tree_sol`), not just coordinates.** That matters
   because the analytical bottleneck is precisely that the legalizer *throws away 2D
   structure* (skyline floor-packs, so analytic `y` survives only as pack order).
   `tree_sol` is a label for exactly the structure we are failing to produce.

> **Headline recommendation:** do **imitation learning of the GT structure + a learned
> guide**, keep the existing legalizer for feasibility, and use RL only as a
> *fine-tuning* step on top — not as the primary method. Pure RL-from-scratch is the
> wrong first move here (§2).

---

## 1. Where ML plugs into the current architecture

Do **not** replace the pipeline wholesale. The legalizer's job — guaranteeing hard
feasibility (no overlap, exact areas, fixed/preplaced) — is something ML should never
be trusted with directly (a 1e-6 overlap → cost 10.0). Keep it. Insert ML where the
heuristic is weak: **the guide and the structure feeding legalization, and the
candidate selection.**

```
            ┌──────────── ML here (learned) ────────────┐
  netlist → GNN guide (x,y + soft shapes)  ┐
  +constr → GNN structure (predict tree/SP) ┼─► legalize/compact ─► candidates
            └ GNN cost surrogate (rank/prune)┘        (classical, feasible)   │
                                                                              ▼
                                              learned selector picks best ─► output
```

- **Feasibility stays classical and exact.** ML only proposes; the legalizer disposes.
- **Everything is hybrid.** The win is a far better *initialization/structure*, refined
  by the deterministic machinery you already trust.

---

## 2. Why not pure RL first (read before committing compute)

RL for macro/floorplacement is famous (Google, *Nature* 2021) — and famously
contested: follow-up work (Cheng et al., and the open *Stronger Baselines* analyses)
showed well-tuned **simulated annealing / analytical placers matched or beat** the RL
agent, which also needed enormous compute and generalized poorly. For **n ≤ 120 with a
seconds-scale budget**, a strong classical optimizer is already near the frontier — the
regime where RL-from-scratch has the *least* edge.

RL's structural problems here:
- **Sparse, delayed reward** (cost only at full placement) → sample-inefficient.
- **Hard-constraint feasibility** is brittle to learn (one overlap = catastrophic 10.0).
- **Generalization across instances/sizes** is the exact thing RL struggles with.

But you have the antidote the *Nature* work lacked at this scale: **1M labeled optima
(`fp_sol`, `tree_sol`).** That makes the **AlphaGo recipe** viable — *behavioral
cloning first, RL fine-tuning second* — which is the only RL path I'd recommend (§5,
Phase 3). RL's unique value is **escaping the GT** on the cases where the GT itself is
suboptimal (it is not provably optimal), but capture that *last*, after imitation has
done the heavy lifting.

---

## 3. Recommended approach, by payoff (the core of this doc)

### 🥇 Phase 1 — Learned guide + learned selection (highest ROI, lowest risk)

Two small models that **augment** the current pipeline as extra candidate sources /
better ranking. Both train in hours, both are pure supervised regression, neither can
hurt feasibility.

**1a. GNN placement guide (replaces/augments the analytic solve).**
A graph neural net that regresses GT block centers — an *amortized* placer that has
seen 1M optima, used as another `wl_model`-style guide into the skyline sweep.
- *Representation:* blocks = nodes; b2b = weighted edges; pins = fixed-coordinate nodes
  with p2b edges. Node features: `log(area)`, boundary one-hot, fixed/preplaced flags
  (+ their coords/dims if any), cluster/mib id embeddings, degree, Σ incident weight.
  **Normalize all coordinates and areas by `√(Σ area)`** → scale-invariant; predict
  normalized centers.
- *Model:* 4–8 layers of message passing (GraphSAGE/GAT) or a graph transformer
  (n ≤ 120 → global attention is cheap). Heterogeneous edges (b2b vs p2b).
- *Output:* per-block normalized `(cx, cy)` **and** a soft-block **aspect ratio**
  `r = h/w` → `w = √(a/r), h = √(a·r)`. This unlocks the DOF the heuristic never uses
  (soft blocks are locked to squares today) — directly attacks both HPWL and area.

**1b. GNN cost / baseline surrogate (fixes candidate selection).**
The current selector ranks candidates by `area·HPWL·exp(3·soft/n)` — a *multiplicative*
proxy that mis-ranks (it caused the 4 small regressions) and, critically, **cannot
compute the true gaps because it doesn't know the baseline.** Train a GNN to predict
`metrics_sol` (`hpwl_baseline`, `area_baseline`) from the netlist. Then rank candidates
by the **true** estimated cost `1 + 0.5·(ĤPWL_gap + Ârea_gap)` — additive, matching the
scorer. Same model prunes the 24-config skyline sweep (predict the winning config →
runtime win → better `R` factor). *Cheapest, broadest win; do this first.*

> Expected: 1a tightens the guide on the heavy mid-large cases (where score lives);
> 1b removes selection mis-ranks and runtime. Both are safe supersets of today's flow.

### 🥈 Phase 2 — Learned structural prior (attacks the actual bottleneck)

This is the conceptually right fix and the reason `tree_sol` is gold. Instead of
floor-packing away 2D structure, **learn to predict a relative-placement structure and
realize it with feasible-by-construction compaction.**

- **Target = `tree_sol`** (the GT B\*-tree-style encoding) — you already have the label
  for every one of 1M instances. Train a model to map *netlist → tree*.
- *Model:* autoregressive **pointer-network / graph-transformer decoder** that emits the
  `(n−1)×3` tree (each step: pick a block, attach to a parent with a left/above
  relation) conditioned on the GNN-encoded netlist. This is sequence generation over a
  graph — well-trodden (Ptr-Nets, "Learning to Branch", neural combinatorial opt).
- *Realize the tree:* a B\*-tree/slicing decode + **constraint-graph compaction** (HCG
  then VCG longest-path) gives an **overlap-free, 2D-structure-preserving** layout —
  exactly what the deleted `topology.py` attempted, but now *warm-started by a learned
  structure* instead of noisy order extraction (the likely reason "Bstar bad" earlier).
- *Why it beats skyline:* the structure carries the analytic/GT vertical relationships
  all the way to the output, instead of collapsing them to pack order.

Feasibility note: tree-decode + compaction is overlap-free by construction; still run
`enforce_hard` as the exact safety net for areas/fixed/preplaced.

### 🥉 Phase 3 — RL fine-tuning (highest ceiling, do last)

Only after Phases 1–2 give a strong learned policy. **Behavioral-clone** the tree
decoder on `tree_sol` (Phase 2 *is* the BC pretrain), then improve it with policy
gradient (PPO / REINFORCE-with-baseline) where:
- *State:* netlist embedding + partial tree.
- *Action:* next block + attachment.
- *Reward:* `−(differentiable or real) cost` of the realized layout, using the
  **provided differentiable cost** as a dense shaped signal and the real legalized cost
  as the terminal reward. Weight episodes by `e^(n/12)` to spend RL where score lives.
- *Payoff:* pushes past GT-imitation on instances where the GT is suboptimal — the only
  way to drive the (clamped-at-0) gaps to true 0 across the board.

Keep this scoped: RL here is *refinement of a pretrained policy with a known reward*,
not exploration from scratch — the regime where RL actually delivers.

---

## 4. Training mechanics (the parts that decide success)

**Loss — avoid naïve coordinate MSE.** Absolute GT coords are not canonical (global
translation/rotation/reflection + permutation of interchangeable blocks). Regressing
raw `fp_sol` coordinates will fight these symmetries. Use, in order of robustness:
1. **The provided differentiable cost** as the primary loss (`...differentiable`,
   `:1263`) — self-supervised, symmetry-free, *exactly the metric*. Add the smooth
   overlap/area penalty it already includes so the guide is legalization-friendly.
2. **+ a structural/relative loss** to break symmetry and steer into the GT basin:
   pairwise *relative-order* cross-entropy (sign of `Δx`,`Δy`) for connected pairs, or
   tree cross-entropy on `tree_sol`. (Relative, hence transform-invariant.)
3. Optionally a small **Procrustes-aligned** coordinate term (align prediction to
   `fp_sol` by best rigid transform before MSE) if you want a direct positional anchor.

**Train *through* the metric, validate on the real one.** The differentiable cost is a
proxy (soft overlap ≠ hard feasibility). Always close the loop: model → real legalizer →
real `evaluate_solution`. Select checkpoints on the **real weighted score**, not the
training loss.

**Weight by `e^(n/12)` and focus capacity on n ≳ 90.** The hard lesson from the
multistart experiment: small-n wins don't move the aggregate. Sample/curriculum the 1M
set so mid-large instances dominate the gradient.

**Data hygiene:** the public eval is FloorSet-Lite test cases (21–120 blocks); train on
the matching distribution (Lite worker shards) and hold out a validation split that
mirrors the 100 scored sizes. Cache featurized graphs; 1M samples → use streaming.

---

## 5. Concrete first experiment (1–2 days, decides everything)

Before any RL or big training run, validate the central premise cheaply:

1. **Train Phase-1b (baseline-metric surrogate)** — small GNN, regress
   `area_baseline`,`hpwl_baseline`. Plug into candidate selection. *Measure the real
   weighted score.* This alone should recover the proxy mis-ranks and is ~free.
2. **Train Phase-1a (GNN guide)** with the differentiable cost as loss; feed its output
   as an extra guide into the existing skyline sweep. *Measure.*
3. **Gate decision:** if the GNN guide's *legalized* layouts beat the IRLS guide's on
   the mid-large cases, commit to Phase 2 (structural). If not, the guide isn't the
   bottleneck — go straight to Phase 2, because the structure is.

A clean go/no-go in two days, no wasted RL compute.

---

## 6. Honest risk assessment

| Approach | Upside | Risk / cost | Verdict |
|----------|--------|-------------|---------|
| 1b surrogate selection | removes mis-ranks, runtime | tiny | **do now** |
| 1a GNN guide + learned shapes | better guide, unlocks aspect DOF | moderate; may be capped by legalizer | **do, with §5 gate** |
| 2 structural (tree) + compaction | attacks the real bottleneck; feasible-by-construction; has labels | new decode+compaction code; tree semantics | **highest-value build** |
| 3 RL fine-tune | beats GT on suboptimal cases | compute, tuning, instability | **last, optional** |
| pure RL from scratch | — | high compute, brittle, weak at n≤120 | **avoid** |

**The realistic ceiling:** imitation toward the GT can plausibly cut the dominant HPWL
gap substantially (we are at 0.45–1.94; GT is 0). Even halving it on the heavy cases
moves the weighted score toward the ~1.0–1.3 region — a far bigger jump than any
remaining heuristic tweak. The risk is not "will ML help" but "will the *legalizer*
let the learned structure through" — which is exactly why **Phase 2 (learn the
structure, compact it) is the bet**, not just a learned coordinate guide.

---

## 7. TL;DR for the team

1. You have GT layouts **and** GT trees **and** a differentiable cost for 1M cases —
   this is an **imitation** problem; gaps clamp at 0 so the GT is the ceiling.
2. Keep the classical legalizer for hard feasibility; ML proposes, legalizer disposes.
3. **Order of work:** (1b) learned baseline-surrogate selection → (1a) GNN guide with
   learned soft-block shapes → (2) **learn `tree_sol` → constraint-graph compaction**
   (the real fix for the floor-packing bottleneck) → (3) RL fine-tune to pass the GT.
4. Train on the differentiable cost + a relative/tree structural loss; **validate on
   the real weighted score**; weight everything by `e^(n/12)` and spend capacity on
   n ≳ 90, because that is where the score actually lives.
