# ICCAD 2026 Contest C — GNN + CNN + RL Sequential Placer (DL_RL)

## Overview

Explores a learned, sequential-decision alternative to the deterministic
`analytic_legalizer/` pipeline (score ≈ 1.785, on `vision`/`basic-baseline`):
**"Google chip placement" style** (Mirhoseini et al., *A graph placement
methodology for fast chip design*, Nature 2021), adapted to FloorSet-Lite
(rectangular-only blocks, n = 5–120).

```
GNN encodes the netlist (who connects to whom)        <- the "graph" half
  + the partial placement is rasterized to a grid      <- the "vision" half
  + a CNN reads that grid
  + PPO (RL) picks where to drop each block, one at a time
  + reward = official differentiable contest cost
  + legalize the result -> FloorplanOptimizer.solve()
```

**Status: exploratory, closing the gap.** Phase 9 (`rl_skyline_optimizer.py`)
replaced the model's own row-packer with `analytic_legalizer`'s
`skyline_legalize()`: same Phase 8 checkpoint, quality-only cost
10.05 -> **2.09** (analytic baseline: 2.02). End-to-end pipeline is 100/100
feasible. Kept independent under `CNN_RL/` — **does not modify
`analytic_legalizer/`**, only imports its pure functions; `my_optimizer.py`
remains the scored baseline. See §"Results" and "Lessons" below before
extending this.

---

## Score Formula (same contest, different proxy during training)

Official (`ProblemC.md`):
```
Cost = (1 + 0.5*(HPWL_gap + Area_gap)) * e^(2*V_soft) * max(0.7, RuntimeFactor^0.3)
```
Feasible range `[0.7, ~11]`; infeasible (any overlap / >1% area error) = fixed `10`.

Training reward uses `compute_training_loss_differentiable` (same formula,
two deliberate differences for differentiability/density):
1. no runtime term -> floor is **1.0**, not 0.7;
2. overlap is a **soft** `e^(2*overlap_ratio)` penalty, not a hard 10.

So env-internal costs during RL/BC (e.g. "2.4–4.6 for random rollouts") are
*not* directly comparable to the 1.77 baseline — only the **post-legalization,
`--evaluate`-scored** number is.

---

## MDP Formulation

| Element | Definition |
|---|---|
| **State** | placed blocks so far (each `(x,y,w,h)`) + multi-channel canvas raster + GNN node/graph embeddings + id of the next block |
| **Action** | flat grid-cell index `[0, G*G)` = lower-left corner for the next block; optional aspect-ratio bucket (Phase 5) |
| **Placement order** | fixed at reset: preplaced/fixed first (placed immediately, excluded from RL decisions), remainder by area descending |
| **Transition** | place block at chosen cell (snapped to boundary/hard constraints if applicable) -> update canvas -> advance to next block |
| **Reward** | terminal only: `-(compute_training_loss_differentiable(...))` |
| **Episode end** | every block in `env.order` has been placed |

Canvas size is estimated from `sqrt(sum(area_target)) * margin` and divided
into a `G x G` grid (grid size is a hyperparameter, used 16–84 across phases).

---

## Architecture data flow (GNN + CNN → RL)

Two encoders in series feed the RL decision; **both are always used**.

```
   netlist graph ──GNN──▶ per-block embeddings ┐
   (blocks=nodes, b2b=edges)   [N, D]           │
                                                ├─▶ CNN (policy_net) ─▶ [G,G] logits
   partial placement ──rasterize──▶ [C,G,G] ────┘        │  (+ value, + aspect heads)
   (occupancy/density/wl_pull/…)   canvas                │
                                                         ▼
                                          mask illegal cells, softmax
                                                         │
                                              sample / argmax a cell
                                                         │
                                          env.step → place block, update canvas
                                                         │
                                       (repeat for every block, in order)
                                                         ▼
                                   per-block centers (cx,cy) ─▶ skyline legalize
                                                         ▼
                            reward = −(legalized contest cost)  → PPO / BC update
```

| | **GNN** (graph half) | **CNN** (vision half) |
|---|---|---|
| Module | `gnn_encoder.py` | `policy_net.py` |
| Input | netlist: node features `[N,10]` + b2b edges | canvas raster `[C,G,G]` + current block's GNN embedding (broadcast as channels) |
| Sees | **connectivity / topology** (who connects to whom, area, constraints) — no coordinates | **spatial geometry** (what's placed where, density, low-HPWL spots, legal cells) |
| Output | one embedding per block `[N,D]` | per-cell placement logits `[G,G]` (+ value, + aspect) |
| Runs | **once per problem** (netlist is static) | **once per block** (canvas changes each step) |

The current block's GNN embedding conditions the CNN, so the spatial "where to
place" decision knows *which* block (and its graph context) it is placing.
GNN = "what is this block and what's it wired to"; CNN = "what does the layout
look like and where's a good spot". The **same GNN+CNN produce the centers**
that the (Phase 9) skyline legalizer consumes — switching the legalizer did not
replace either encoder.

---

## Feature Embeddings

### What is an embedding?

Raw problem features are sparse, heterogeneous, and hard to do arithmetic on
(e.g. "block i has area 0.04, is fixed, connects to blocks 3 and 17, touches
the LEFT boundary"). A neural network needs dense, continuous vectors instead.
**Embedding** = a learned function that maps each raw input to a fixed-size
real-valued vector (`[D]`) so that semantically similar inputs land near each
other in vector space.

In our architecture there are two distinct embeddings, one per encoder half:

---

### GNN Node Embedding  (`gnn_encoder.py`)

**Input — raw node features `[N, 10]` per block:**

| Index | Feature | Meaning |
|---|---|---|
| 0 | `sqrt(area) / max_sqrt_area` | normalized size of the block |
| 1 | `area / total_area` | fraction of total chip area this block takes |
| 2 | `fixed` | 1 if block is pre-placed (location fixed) |
| 3 | `preplaced` | 1 if block has a given position (soft preplacement hint) |
| 4 | `has_mib` | 1 if block belongs to a Must-be-In-Boundary group |
| 5 | `has_cluster` | 1 if block belongs to a cluster (packed together) |
| 6–9 | `boundary_{L,R,T,B}` | one-hot: which canvas edge this block must touch |

These 10 numbers describe **one block in isolation** — they say nothing about
how it connects to other blocks.

**Message passing — GraphSAGE-style:**

```
for each layer l:
    for each node i:
        agg_i  = weighted_mean({ h_j^(l)  for j in neighbours(i) })
        h_i^(l+1) = ReLU( W * concat(h_i^(l), agg_i) )
```

Each block collects information from its netlist neighbours (b2b edges), then
updates its own hidden state. After 3 layers, a block that connects to many
large preplaced blocks will have a very different embedding from an isolated
small soft block — even if their raw `[N,10]` features look similar.

**Output — node embeddings `[N, 128]`:**

Each block now has a 128-dimensional vector that encodes **both its own
properties AND its graph neighbourhood** (connectivity, topology, who it shares
nets with). This vector is static for the whole episode (the netlist does not
change during placement) and is computed once per problem instance.

A single block's embedding is extracted as `node_emb[cur]` (`[128]`) and fed
into the CNN at each placement step to condition the spatial decision on
*which* block is being placed right now.

---

### Canvas Raster as Spatial Embedding  (`canvas_raster.py`)

The second embedding is **not a vector** but a multi-channel image `[5, G, G]`
— a spatial encoding of the *current partial placement state*:

| Ch | What it encodes |
|---|---|
| `CH_OCCUPANCY` | binary mask: where is space already taken |
| `CH_DENSITY` | how many blocks overlap each cell (overlap hot-spots) |
| `CH_WL_PULL` | for the current block, how much HPWL each cell would save if b2b neighbours are already placed |
| `CH_FEASIBILITY` | binary: where can this block's lower-left corner legally go (fits inside canvas) |
| `CH_PIN_PULL` | same as WL_PULL but for I/O pins (fixed coordinates), so the model sees the pin ring |

This changes **every step** as blocks are placed. It is the policy's "look at
the board" signal — pure spatial geometry, no graph topology.

---

### How they combine in `policy_net.py`

```
canvas [5, G, G]                          ← "board state"
  + node_emb[cur] broadcast → [node_channels, G, G]   ← "which block am I placing"
  ──────────────────────────────────────────────
  = [5 + node_channels, G, G]   → CNN trunk (dilated convs + global branch)
                                           ↓
                              policy logits [G, G]     → argmax / sample
                              value scalar  [1]
                              aspect logits [5]         → aspect bucket
```

The GNN embedding is **broadcast** (repeated at every spatial location) so
that each pixel of the CNN's feature map "knows" what block is being placed.
The CNN then merges graph context (embedding) with spatial context (canvas) to
output a distribution over `G×G` grid cells.

**Intuition**: the GNN embedding says "I am a large block wired to the two
biggest preplaced macros on the right edge"; the canvas raster says "the right
half of the board is mostly empty and pin-pull is high there"; the CNN puts
these together to output high logits on the right half.

---

## Pipeline / Components

### Step 1 — Environment (`placement_env.py`)
`PlacementEnv(grid, area_margin, device)` wraps one problem:
- `reset(area_target, b2b, p2b, pins_pos, constraints, metrics, target_positions=None)`
  - builds `env.order` (placement sequence), `env.canvas_w/h`
  - if `target_positions` is given (Phase 6): preplaced blocks are placed
    immediately at spec and removed from `order`; fixed-shape blocks get
    `fixed_dims`; MIB-group members get shared `mib_dims`
- `_block_dims(block_idx, aspect=1.0)`: returns `(w,h)` — checks
  `fixed_dims`/`mib_dims` first, else `w=sqrt(area*aspect), h=sqrt(area/aspect)`
  (area is exact for any aspect)
- `_snap_boundary(...)`: snaps a placed block to the canvas edge implied by
  its `boundary_code` bits (LEFT=1, RIGHT=2, TOP=4, BOTTOM=8)
- `step(action, aspect=1.0)` -> `(next_state, reward, done, info)`; reward is
  0 until the last block, then `-cost`
- done when `len(env.order)` blocks have been placed (not `block_count`,
  since preplaced blocks are excluded from `order`)

### Step 2 — Canvas rasterizer (`canvas_raster.py`)
`rasterize(positions, canvas, grid, current_block, current_dims, b2b, p2b, pins, device)`
-> `Tensor[N_CHANNELS, G, G]`, channels:

| Ch | Name | Meaning |
|---|---|---|
| 0 | `CH_OCCUPANCY` | 1 where any placed block covers the cell |
| 1 | `CH_DENSITY` | count of placed blocks covering the cell (overlap signal) |
| 2 | `CH_WL_PULL` | for the current block, a [0,1] field high where placing it gives LOW weighted HPWL to its placed **b2b neighbours** |
| 3 | `CH_FEASIBILITY` | 1 where the current block's lower-left corner can legally go (fits in canvas) |
| 4 | `CH_PIN_PULL` | for the current block, a [0,1] field high where placing it gives LOW weighted HPWL to its connected **I/O pins** (p2b) — added so the model finally "sees" pins |

`rasterize_env(env)` is the adapter used by the policy/training code.

> **Pins/p2b feature (this phase).** Previously the model was blind to pin
> locations (GNN used b2b only; wl_pull used b2b only) — a likely cause of the
> "blocks spill past the pin ring" artefact. `CH_PIN_PULL` adds a pin-aware
> spatial signal with the same construction as `CH_WL_PULL`, but using the fixed
> pin coordinates of the current block's `p2b` edges (pins are always "placed").
> `N_CHANNELS` is now 5; the saved checkpoint records `in_channels` so the
> optimizer/loaders pick the right width (old 4-channel checkpoints still load).

### Step 3 — GNN encoder (`gnn_encoder.py`)
Pure-PyTorch hand-rolled message passing (GraphSAGE-style weighted-mean
aggregation) — **no PyTorch Geometric dependency**, runs for variable N
(5..120) and isolated-edge graphs.

- `build_node_features(area_target, constraints, block_count)` -> `[N, 10]`:
  `[sqrt(area) normalized, area/total, fixed, preplaced, has_mib, has_cluster,
  boundary_left(6), right(7), top(8), bottom(9)]`
- `build_edges(b2b, block_count, device)`: b2b made undirected, padding
  (`<0`) rows dropped
- `SAGELayer`: weighted-mean message passing
- `GNNEncoder(node_in_dim=10, hidden=128, n_layers=3, out_dim=128)`:
  `forward()` -> `node_emb[N,D]`, `graph_emb[D]`;
  `encode_problem(area_target, constraints, b2b, block_count, device)` is the
  convenience entry point

### Step 4 — Policy / value network (`policy_net.py`)
`PolicyValueNet(in_channels=4, node_dim=128, node_channels=32, hidden=64,
n_conv=4, n_aspect=5)`:
- CNN trunk over the `[4,G,G]` raster; the current block's GNN embedding is
  broadcast as extra channels (conditions the spatial decision on *which*
  block and its graph context)
- Policy head -> `[G,G]` logits, masked by `CH_FEASIBILITY` (illegal cells ->
  `-inf`), softmax -> position distribution
- Value head -> scalar `V(s)` (PPO baseline)
- `forward()` is the 3-tuple (position-only) path; `forward_aspect()` adds an
  `aspect_probs[5]` head (Phase 5); `act()` / `act_aspect()` sample or
  greedy-argmax both heads

### Step 5 — PPO training loop (`train_rl.py`)
Standard clipped-PPO with GAE (`PPOAgent`, `Episode`/`Step`, `compute_gae`,
`update()`). `train(samples, iters, grid, gnn_out, hidden, lr, gamma, lam,
rollouts, warmstart_ckpt, seed)`:
- **`rollouts` matters**: with 2 episodes/iter PPO is too noisy to learn
  (cost went 3.87->4.58, i.e. *worse*). `rollouts=8` (16 episodes/iter) gives
  a real downward trend (4.14->3.70, best 3.54 on a 2-sample overfit set).
- `warmstart_ckpt`: load a BC checkpoint's GNN/policy weights before RL
  fine-tuning (see Step 6).

### Step 6 — Behaviour-cloning warm-start (`pretrain_bc.py`)
Pure RL from random rarely learns on a 50-150 cell-wide action space in
reasonable time. We have GT placements (`fp_sol`, validation set polygons —
bbox is exact for rectangular Lite blocks, no 1M download needed for this
step):
- `gt_boxes_from_fp(fp_sol, block_count)` -> `[N,4] = (x,y,w,h)` GT boxes
- `_target_cell(env, x, y, w, h)`: GT lower-left corner -> grid action, clamped
  to the feasible region
- `build_bc_episode(env, sample)`: teacher-forces the env along GT, recording
  `(canvas, block, mask, target_cell)` per step
- `pretrain(samples, epochs, grid, gnn_out, hidden, lr, seed)`: cross-entropy
  on the target cell; tracks exact `cell_acc` and `near_acc` (±1 cell
  Chebyshev). On a 2-sample overfit set: loss 7.73->1.92, cell_acc 0->0.54,
  near_acc 0->0.65. Saved checkpoint: `checkpoints/bc_warmstart.pt`.

### Step 7 — Aspect-ratio action (`ar_utils.py`)
Soft blocks don't have to be square. `ASPECT_BUCKETS = [0.25, 0.5, 1.0, 2.0,
4.0]` (log-spaced, `N_ASPECT=5`); `w=sqrt(area*aspect), h=sqrt(area/aspect)`
keeps area exact for any chosen aspect. Oracle check: using GT shapes instead
of forced squares lowers cost from ~1.16-1.43 down to 1.00 (area/HPWL gap
mostly closes) — the shape lever is worth learning.

**UPDATE (Phase 13): the aspect head IS trained end-to-end now.**
`policy_net.aspect_head` (`Linear→5`) is BC-trained in `train_fast.py` against
`ar_utils.gt_aspect_bucket(gw, gh)` labels with `aspect_weight=0.5`
(`aspect_acc ~0.47` vs 0.20 chance); shipped in `phase13_aspect.pt`.
`rl_skyline_optimizer._rl_centers` calls `policy.act_aspect()` per free block and
`solve()` overrides `blocks[i].w/h` with the predicted bucket *before*
`skyline_legalize` (2.0172→1.9705).

**BUT the prediction is mostly clobbered downstream** — `_row_assign_reshape`
(inside `skyline_legalize`) and now B1's contour shaping (`shape_fit=True`,
default) re-decide the shape per block, so the learned aspect rarely survives to
the final geometry. The doc's original "not yet trained / TODO" is obsolete; the
real open problem is the **prediction↔legalize conflict** (see §"Aspect Ratio"
below) and that B1's legalize-reshape currently *beats* prediction-led shaping
(1.8517 < 1.8650), so the learned head's value is capped by its quality (20k
samples, free-blocks-only, clobbered).

**Phase 16 (pin-bbox canvas) — tried, refuted.** Hypothesis: the model places
into a canvas sized `sqrt(area·1.6)` ≈ **1.80× GT bbox**, while the pin bbox is
only **1.23×**; sizing the canvas to the pin extent should force tighter centers
→ smaller output bbox → lower Area_gap. Retrained `phase16_pinbox.pt` on the
tight canvas. **Result: no change** — Area_gap 0.477→0.484, output bbox
1.47×→1.48×, Total Score 1.8517→**1.8688** (worse). Reason: `skyline_legalize`
chooses its own strip width and re-packs the *relative* arrangement
(scale-invariant) independently of the canvas, so canvas scale washes out of the
final geometry. **Area_gap is downstream of how skyline packs, not of where the
model predicts** — confirming (again) that the lever is block *reshaping*, not
the prediction frame. Reverted `estimate_canvas`; default stays phase13.

### Step 8 — Hard constraints (`hard_constraints.py`)
- `make_target_positions_from_gt(constraints, gt_boxes, block_count)`: builds
  the `target_positions` tensor `env.reset()` consumes
- `touches_boundary(x,y,w,h,canvas_w,canvas_h,code,tol)`,
  `check_hard_constraints(...)`: verifies preplaced (location+dims),
  fixed-shape (dims), MIB (shared shape), boundary (touches the right edge)
  are *enforced*, not learned — random actions still pass with 0 violations.

### Step 9 — Legalization + `FloorplanOptimizer` wrapper (`rl_optimizer.py`)
The RL/BC rollout can still produce some overlap or non-rectangular packing
gaps, so the raw model output is *not* the final answer — it's legalized:

- **`shelf_legalize(...)`**: self-contained, no model needed. Computes
  area-exact `(w,h)` per block (fixed/preplaced honour given dims, soft ->
  square), places preplaced blocks at spec, then **tallest-first row-packs**
  the rest above them. Overlap-free + area-exact -> always feasible. This is
  the **Phase 7 fallback / floor**: Total Score **10.05** (avg cost 8.07),
  100/100 feasible.
- **`model_guided_legalize(block_count, dims, centers, preplaced_pos, W,
  y_start)`** (Phase 8): position-preserving. Runs the trained model greedily
  to get a predicted center `(cx,cy)` per block, sorts movable blocks into
  rows by `(cy, cx)`, row-packs left-to-right. Intent: reconstruct the
  model's learned 2D arrangement as a non-overlapping layout.
- **`RLPlacerOptimizer(FloorplanOptimizer)`**: `__init__` tries to load
  `checkpoints/phase8_bc.pt` (GNN + policy state dicts + `grid/gnn_out/hidden`
  config); if present, `solve()` calls `_model_solve()` (greedy rollout +
  `model_guided_legalize`), else (or on any exception) falls back to
  `shelf_legalize()`. `Optimizer = RLPlacerOptimizer` alias so the
  `--evaluate` loader's name-matching finds it across importlib module copies.

---

## Results

| Variant | Total Score (incl. runtime) | Quality-only cost (no runtime term) | Feasible |
|---|---|---|---|
| Phase 7 — `shelf_legalize` (no model) | **10.05** | 8.07 | 100/100 |
| Phase 8 — `model_guided_legalize` (1500-sample BC checkpoint, grid=48) | 15.60 | 8.72 | 100/100 |
| **Phase 9 — `rl_skyline_optimizer.py`: RL centers -> `skyline_legalize()`** | 3.32 | **2.0931** | 100/100 |
| Ablation — quadratic-placer centers through the *same* skyline pipeline | 4.85 | 2.0249 | 100/100 |
| `analytic_legalizer` (`my_optimizer.py`, baseline) | 3.09 | 2.0249 | 100/100 |

Phase 8 training (`train_phase8.py`, streaming BC over real training data,
1500 samples / 1 epoch / grid=48, ~26 min CPU): loss 5.62->4.38, exact
`cell_acc` 0.05->0.11. Real learning signal (cell_acc far above the ~1/2300
random baseline for a 48x48 grid), but **not accurate enough** to drive
`model_guided_legalize` — see Lessons.

### Phase 9 — RL centers reuse `skyline_legalize()` (this session)

`rl_skyline_optimizer.py` replaces `model_guided_legalize`'s own row-packer
with `analytic_legalizer`'s validated chain: RL greedy rollout -> per-block
`(cx,cy)` -> `parse_and_init`/`prepack_clusters` -> `skyline_legalize` ->
`slide_boundary` -> `enforce_hard` -> `_detailed_place`. Does not modify
`analytic_legalizer/` — only imports its pure functions.

**Headline result: 10.05 -> 2.09 (quality-only)**, i.e. the *same untrained-ish*
Phase 8 checkpoint (`cell_acc`≈0.11) that produced 15.60 via
`model_guided_legalize` produces **2.09** once its centers are fed to
`skyline_legalize` instead. This confirms the ML Leverage Points hypothesis
from the previous session: `model_guided_legalize`'s row-packer (sort by
`(cy,cx)`, pack left-to-right) was the bottleneck, not the centers themselves.

**Transplant correctness check** (ablation row above): feeding
`quadratic_placer.analytic_place()`'s own centers through the *identical*
`rl_skyline_optimizer.py` code path reproduces `my_optimizer.py`'s quality
metrics bit-for-bit (`hpwl_gap`/`area_gap`/`violations_relative` identical per
test case; quality-only cost 2.0249 == 2.0249). The "Total Score (incl.
runtime)" column differs (4.85 vs 3.09) purely because the standalone-script
harness has higher per-call wall-clock overhead than `my_optimizer.py`'s — and
`median_runtime=1.0` is hardcoded in `iccad2026_evaluate.py`, so
`runtime_factor = runtime_seconds` directly. **Compare quality-only costs, not
raw Total Score, across different harness invocations** — runtime is not
comparable unless both optimizers run inside the same script/process.

**Remaining gap to analytic (2.0931 vs 2.0249, ~3.4%)**: the RL-predicted
centers (from a checkpoint trained on only 1500 samples, `cell_acc`≈0.11) are
still slightly worse than the quadratic placer's HPWL-optimal analytic
solve — expected given the training data volume. But the gap is now small
enough that further BC/PPO training on this *same* `skyline_legalize` backend
is the natural next lever (previously, any model improvement was masked by
`model_guided_legalize`'s packing losses).

**RL rollout runtime**: `rl_skyline_optimizer` averages 2.25s/case vs
`my_optimizer`'s 0.55s/case (both run standalone) — the `PlacementEnv` rollout
does one `env.step()` (rasterize + GNN lookup + CNN forward) per block,
sequentially, on CPU (see Lesson 2 below). For n=120 this is the dominant
runtime cost and feeds into `R^0.3` in the official cost formula.

---

## Lessons / Failure-mode reference

1. **`model_guided_legalize` made things worse (10.05 -> 15.60); reusing
   `skyline_legalize()` instead fixes it (-> 2.09, Phase 9).** The model's
   predicted `(cx,cy)` centers are directionally plausible (visual inspection
   shows blocks roughly clustering into the same relative regions as GT — see
   `phase8_case0_compare.png`) but too noisy as a *sort key* for a naive
   row-packer: small ordering errors compound into uneven row widths, gaps,
   and isolated "floating" blocks. `skyline_legalize`'s contour-based
   constructive packing absorbs this noise far better — same centers, 10.05
   -> 2.09. **The legalizer choice mattered far more than the model's
   accuracy.**
2. **Training is CPU-bound, ~1 sample/sec, and GPU would not obviously
   help.** The env does `env.step()` once per block per sample (teacher
   forcing): each step rasterizes (Python/numpy loops over cells), updates
   occupancy, and does ONE forward+backward on a single `[4,G,G]` canvas. The
   CNN forward itself is microseconds; GPU kernel-launch overhead on
   batch-size-1 tiny tensors would likely be *slower*. To use a GPU
   meaningfully, the environment needs to be vectorized (batch many problems'
   raster + GNN + CNN together) — a real rewrite, not a flag flip.
   Consequence: 1M-sample training (Phase 8's original target) is infeasible
   at this rate (~12 days); 1500 samples (~26 min) was the practical subset.
3. **Connectivity-aware legalizer ordering also regressed** (10.05 -> 14.13
   on a Prim-like b2b-frontier order) — reverted. Heuristic-only tweaks to
   `shelf_legalize` are low-leverage; the packing order needs either (a) a
   much more accurate model, or (b) reuse of `analytic_legalizer`'s
   `skyline_legalize()` (a larger integration decision — it's coupled to
   `analytic_legalizer`'s internal `BlockInfo`/`SuperBlock` structures).
4. **BC label/feature index bugs are easy to make.** The 10-dim GNN node
   feature vector's boundary bits start at index 6, not 5
   (`[sqrt, rel_area, fixed, preplaced, has_mib, has_cluster, L(6), R(7),
   T(8), B(9)]`) — a test asserted index 5 and silently checked the wrong bit.
5. **PPO needs batch size >= ~8 episodes/iter** to show a learning trend on
   this action space; 2 episodes/iter is pure noise (cost went *up*).
6. **`Path(__file__)`-derived checkpoint dirs break under mid-run renames.**
   If the containing directory is renamed while a background training process
   is running, `CKPT_DIR.mkdir()` at the end fails with `FileNotFoundError`
   (the string captured at import time points to a path that no longer
   exists) — the entire training run's compute is lost since nothing was
   checkpointed along the way. Re-run after any directory restructuring.

---

## Aspect Ratio: Problem Analysis & Plan (Phase 15)

### The problem with the current approach

Aspect ratio (a soft block's `w/h` given a fixed area) is currently decided in
**two conflicting places**, both *before the final packing geometry is known*:

1. **Phase 13 upfront head** — `rl_skyline_optimizer.solve()` overrides
   `blocks[i].w/h` with the policy's predicted aspect bucket *before* calling
   `skyline_legalize`.
2. **`_row_assign_reshape`** (inside `skyline_legalize`,
   [`skyline_legalizer.py:152`](../../analytic_legalizer/skyline_legalizer.py))
   — already reshapes *interior soft blocks* per-row so each shelf fills width
   `W` exactly (area-constant). **This overwrites the Phase 13 head's choice for
   any interior soft block.**

So the learned aspect head is partially dead on arrival: for interior soft
blocks its output is clobbered by row-fill reshaping; only boundary / non-row
blocks keep the predicted shape. This is a likely contributor to the observed
"almost everything ends up square" behaviour — it is not (only) that the head is
weak, it is that its output is discarded downstream.

### Why upfront is the wrong place — theory

Aspect ratio is a **continuous sizing problem that is only well-posed once the
relative placement (topology) is fixed**. Deciding it upfront forces the network
to predict the final packing it cannot yet see, so BC can only imitate GT aspect
rather than respond to the actual local whitespace.

The classical CAD division of labour (and the right one here):

- **Discrete topology / relative order** (who is left/below whom) is the NP-hard
  combinatorial part → learn it (GNN+CNN already does this well).
- **Continuous shaping** (pick `w_i,h_i` given fixed topology) is **convex /
  polynomial-time** in most representations → solve it analytically, do not
  BC it.

Foundational results behind this split:

| Paper | Relevance |
|---|---|
| Stockmeyer 1983, *Optimizing aspect ratios in slicing floorplans* | shape curves; topology-given optimal shaping is polynomial |
| Yan & Chu, *Optimal Slack-Driven Block Shaping in Fixed-Outline Floorplanning*, ISPD 2008 | **the** legalize-position + reshape combo: fix relative placement, compute per-block slack from H/V constraint graphs, reshape soft blocks into whitespace |
| Murata et al. 1996, *Sequence Pair* | fixing relative placement makes shaping an LP/convex program on a DAG |
| Adya & Markov (Parquet), *Fixed-outline floorplanning* | sequence-pair + soft-block slack shaping in practice |
| Boyd et al., geometric programming for floorplanning | shaping given topology as a convex GP — natural fit if we later want a differentiable shaping layer |
| Lin & Chu, *DeFer*, DAC 2008 | deferred decision-making — argues *against* committing shape too early |

Note: Google chip placement (Mirhoseini, Nature 2021) has **no** aspect/shaping
step at all — it places *hard* macros whose `w/h` is fixed pre-tapeout. So the
ML literature is thin here; classical slack/GP shaping is the state of the art.

### The plan: move shaping to where whitespace is visible

Decide aspect **during or after legalization**, where the packing geometry (and
thus the whitespace each block could expand into) is actually known. The learned
head is **demoted to a prior** that seeds the geometric search, not the final
answer.

Three candidate placements (we will validate, not assume):

| Option | When | Pros | Cons |
|---|---|---|---|
| A — current upfront head | before legalize | one forward pass | blind; clobbered by `_row_assign_reshape` |
| **B — contour-aware shaping** | *during* skyline packing | sees current skyline contour; picks shape minimising dead-space | greedy, not global |
| **C — slack-driven post-shaping** | *after* legalize (topology fixed) | near-optimal; directly attacks `Area_gap`; reuses existing `topology.py` constraint-graph (`build_topology`/`compact`/`longest_path_pack`) | needs constraint-graph slack pass |

**Decision: pursue C first.** `Area_gap` (whole-floorplan bbox vs Σarea) is the
big lever in the cost formula and lives in the jagged skyline whitespace; the
constraint-graph machinery already exists in `topology.py`. The learned head
stays as a prior for later.

### Validation roadmap (step-by-step, measure before committing)

0. **PoC measurement (this step).** A standalone, no-ML script that runs the
   *current* default pipeline, then a slack-driven shaping pass on the legalized
   layout, and reports `Area_gap` (and `HPWL_gap`, feasibility) **before vs
   after** across the 100 cases. Goal: quantify the ceiling — if slack shaping
   can't move `Area_gap`, the whole direction is dead and we stop here.
   → `CNN_RL/poc_slack_shaping.py`.
1. If the PoC shows a real `Area_gap` reduction with no new overlaps, fold the
   pass into `rl_skyline_optimizer.solve()` behind a flag (like `compact_pass`)
   and run the full `--evaluate` for a real Total Score delta.
2. Resolve the `_row_assign_reshape` conflict: either disable the upfront head
   when the slack pass is on, or have the head emit a *prior* the slack pass is
   regularised toward.
3. Only if C plateaus, try B (contour-aware shaping inside the skyline loop).

### PoC success criteria

- `Area_gap` decreases on a clear majority of cases, average reduction
  measurable (target: a few % of the cost's `0.5*(HPWL_gap+Area_gap)` term).
- **No** new overlaps and **no** hard-constraint violations introduced
  (feasibility stays 100/100).
- `HPWL_gap` does not regress more than `Area_gap` improves (net win on the
  combined term).

### PoC RESULTS (`poc_slack_shaping.py`, 30 cases) — decisive

The slack-shaping pass was implemented as a self-contained, geometry-only,
overlap-safe operation (grow a soft block on its slack axis into verified-empty
space up to the current envelope edge, shrink area-constant on the binding axis,
re-pack the binding axis). Measured `Area_gap` before/after under four masks:

| Configuration | Reshapeable blocks | Δ Area_gap | Cases improved | New overlaps |
|---|---|---|---|---|
| free-only (no fixed/preplaced/MIB/cluster/boundary) | ~25% | **0.0000** | 0/30 | 0 |
| free + boundary blocks | ~37% | **0.0000** | 0/30 | 0 |
| all-free, **compaction only** (movement, no reshape) | 100% | 0.0410 | — | 0 |
| all-free, **with reshape** (ceiling probe) | 100% | **0.2465** | 30/30 | 0 |

(Ceiling: mean `Area_gap` 0.524 → 0.278, ≈47% reduction.)

**Three conclusions, two of them surprising:**

1. **The shaping lever is real and large.** At the ceiling, reshaping accounts
   for ~0.21 of the 0.25 `Area_gap` reduction (compaction-only gives just 0.04).
   Aspect ratio — not block movement — is the dominant lever for `Area_gap`. The
   pass is geometrically sound: zero new overlaps even when reshaping all blocks.

2. **A post-legalize pass on free blocks only CANNOT capture it (= 0.00).** This
   kills Option C as I built it. Pinning the constrained majority (boundary
   46.6%, cluster 37.3%, MIB 14.1%, fixed 9.0%, preplaced 3.8% — only ~25% fully
   free) leaves a *rigid skeleton* that fixes the envelope. The free blocks are
   interspersed inside it with no room to move the envelope, so reshaping them
   does nothing. The ceiling only appears when **every** block can co-reshape and
   re-flow together.

3. **Therefore shaping must be integrated INTO legalization, not bolted on
   after.** The shape of a block must be a decision variable *while* it (and its
   clustered/boundary neighbours) is being placed, so the whole layout co-adapts
   — i.e. **Option B (contour-aware shaping during skyline packing)**, not the
   frozen-skeleton post-pass. This reverses the "pursue C first" decision above,
   on data: C-on-free-blocks is a dead end; the lever lives inside the packer.

**Revised plan:** move shaping into the legalizer. Two concrete directions,
in increasing order of effort:

- **B1 — contour-aware shape choice in `skyline_legalize`.** When the skyline
  packer places each block onto the current contour, choose the aspect that
  minimises the notch/dead-space it creates against the contour. This
  generalises the existing `_row_assign_reshape` (which reshapes interior soft
  blocks per row) to a per-block contour-fit objective.
- **B2 — joint shape+position solve given the net's topology.** Fix the relative
  order from the model, then solve `(w_i, h_i, x_i, y_i)` as an LP/geometric
  program (Yan-Chu slack distribution / Murata sequence-pair shaping) over **all**
  blocks subject to area + aspect + hard constraints. This is the principled
  version that the PoC ceiling (0.25) suggests is worth ~0.12 on the cost's
  `0.5·(HPWL_gap+Area_gap)` term if fully captured.

The learned aspect head stays demoted to a *prior* feeding B1/B2's search.

### B1 RESULT (`skyline_shape.py`, full 100-case `--evaluate`) — a clean win

B1 was implemented as `skyline_legalize_shaped` (`CNN_RL/skyline_shape.py`): it
imports `analytic_legalizer`'s pure packer helpers (does **not** modify
`analytic_legalizer/`) and adds a contour-aware shape choice in the per-width
pack — a soft, non-boundary, non-cluster block tries a ladder of area-constant
aspects (`SHAPE_ASPECTS`) and keeps the `(shape, x)` minimising
`landing_y + lam·|center − analytic_cx|`. It is added as a **third candidate
mode** alongside the stock square + row-assign modes and scored by the *same*
proxy, so it is never worse than stock by construction. Wired into
`rl_skyline_optimizer.solve()` behind `shape_fit` (now default `True`).

| Variant | Total Score | Avg Cost | Feasible | Runtime |
|---|---|---|---|---|
| stock (`shape_fit=False`) | 1.8650 | 1.9705 | 100/100 | 0.81s |
| **B1 (`shape_fit=True`, default)** | **1.8517** | **1.9593** | 100/100 | 0.87s |

Total Score 1.8650 → **1.8517** (−0.7%), Avg Cost 1.9705 → 1.9593, feasibility
unchanged, runtime ≈ flat. The realised gain is well below the 0.25 PoC ceiling
because (a) shaped mode only wins where it beats the proxy (conservative,
never-worse design), (b) boundary/cluster/MIB blocks still don't reshape, and
(c) the HPWL term in the proxy caps how aggressively a block can deform. Closing
more of the ceiling is what **B2** (joint shape+position over *all* blocks) is
for — B1 confirms the lever is real and feasibly capturable; B2 is the next step
if the remaining headroom is worth the LP/GP machinery.

### B1+ RESULT (`SHAPE_HEIGHT_BETA`) — height-aware landing objective, the real win

The original B1 landing objective `landing_y + lam·|center − cx|` was copied
verbatim from the stock packer, where it is correct: with a *fixed* shape,
minimising the landing-y (the block's bottom) also minimises its top. But B1
chooses among shapes of **different heights**, and the objective never penalised a
shape for being tall — so the greedy systematically preferred tall narrow shapes
that drop into contour notches but **spike the skyline**, exactly opposite to the
Area_gap goal of keeping the packed envelope low at fixed width. The shaped pass
therefore lost the proxy gate to square/row on most cases (modest realised gain).

The fix internalises the block's own height contribution:

```python
score = landing_y + β·h + lam·|center − cx|      # β = SHAPE_HEIGHT_BETA
```

A 100-case sweep (β reproduces original B1 at 0; the curve is **non-monotonic**
because the proxy gate accepts/rejects the shaped layout discretely per case):

| β | Total Score | case0 | case99 |
|---|---|---|---|
| 0.0 (original B1) | 1.8517 | 1.8566 | 1.7179 |
| 0.2 | 1.8278 | 1.8612 | 1.7706 |
| **0.3 (default)** | **1.8088** | 1.8605 | 1.7160 |
| 0.4 | 1.8595 | 1.8605 | 1.7590 |
| 0.5 | 1.8302 | 1.8605 | 1.7590 |
| 0.7 | 1.8442 | 1.8605 | 1.7100 |
| 1.0 | 1.8677 | 1.7943 | 1.7607 |

**Total Score 1.8517 → 1.8088 (−0.0429, ~2.3%)** with `β=0.3` — a larger gain
than B1 itself delivered over stock, and the single biggest legalize-side win in
the project. The improvement concentrates on large cases (which dominate the
`e^{n/12}`-weighted Total Score), so case0/case99 barely move while the aggregate
drops sharply. `β` is now the default in `skyline_shape.py`; re-sweep via the
`SHAPE_HEIGHT_BETA` env var.

**Overfitting check (split-half).** Because `β` was tuned on the same 100 cases
that define the score, we re-derived the optimum on disjoint halves: select `β` on
even-indexed cases, score it on odd (and vice versa), plus a hash-based split.
`β=0.3` is the **independent minimum on every holdout** (even→odd 1.8044,
odd→even 1.8134, hash-A→B 1.7959, hash-B→A 1.8223), each equal to that holdout's
own best — so 0.3 is not a full-set fluke. The non-monotonic 0.4 bump reproduces
on *both* halves, confirming it is a real proxy-gate artifact, not noise. Caveat:
all 100 cases share the FloorSet validation distribution; the entire `β∈[0.2,0.3]`
band beats baseline on every subset, so the choice is safe even if the hidden
test set shifts the exact optimum slightly.

### Real-cost gate (`real_cost_gate`, default ON) — fixes the proxy mismatch

Three reshape levers — **B-MIB, B2, and B-cluster** — all regressed Total Score
the *same* way, despite each being feasible and reducing the bbox. The common
cause is the acceptance gate: the shaped pass kept a candidate only when it beat
the **`bbox·HPWL` proxy**, which (a) has **no violation term** and (b) *multiplies*
area·HPWL instead of the normalised *addition* the contest uses. So a candidate
that shrinks `bbox·HPWL` but worsens violations or the real area/HPWL trade-off
wins the proxy yet loses the actual cost `(1+0.5(area_gap+hpwl_gap))·e^{2V}`.

The fix scores candidates with the **exact contest-cost form**, computed *after*
the full `_finish` (slide+enforce+detailed), relative to a shared reference
candidate (`RLSkylineOptimizer._gate_real_cost`): the reference's bbox/HPWL are
used as the `area_baseline`/`hpwl_baseline`, so the 0.5-weighted area/HPWL
trade-off and the `e^{2V}` multiplier are applied in exactly the contest's form.
This reuses the evaluator's own `evaluate_solution` (no metrics tensor needed),
so the gate's violation counting matches the scorer's exactly.

| Gate | Reshape variant | Total Score | Runtime |
|---|---|---|---|
| `bbox·HPWL` proxy | + B-cluster | 1.8151 (loss) | 0.93s |
| **real-cost** | **+ B-cluster (default)** | **1.8032 (win)** | **0.92s** |
| real-cost | + B-cluster + B-MIB | 1.8033 (MIB adds nothing) | — |

**Total Score 1.8088 → 1.8032** with B-cluster gated on real cost. The gate runs
the legalize+evaluate per candidate (≈2× work) but is **short-circuited to the
single default path whenever the problem has no reshapeable cluster**, so runtime
is unchanged (0.92s). MIB is null even under the correct gate (1.8033, tied),
confirming the earlier B-MIB result was real and not just a proxy artifact. The
gate is the reusable mechanism: future reshape levers (incl. B2) can be added to
its variant list and can only help, since selection is now on the true metric.

### Obstacle-aware x-candidates (`OBS_XCAND`, default ON) — the biggest area win

The skyline packer represents the placed area as a **1-D top contour** (`Skyline`):
`candidate_xs` only proposes left-x positions at skyline-segment boundaries, and
`_land_y` handles a preplaced obstacle solely by *bumping the block up over it*.
Consequence: a rigid cluster super-block that is wider than the free gap beside a
**preplaced obstacle** can never be offered an x that slots into that gap — the
skyline doesn't know the gap exists — so it is bumped **on top of** the obstacle,
pushed outward, growing the bbox. All 100 validation cases have both preplaced
blocks (1–3) and clusters (3–4), so this waste is universal.

The fix (`_pack_one_width_shaped`, `OBS_XCAND`): for every interior unit, augment
the candidate-x list with the **flush-left (`ox − w`) and flush-right (`ox + ow`)
positions of each preplaced obstacle**. The unit can then land *beside* the
obstacle (no x-overlap → low landing-y) instead of over it. `_land_y` still does
the exact collision/bump check, so nothing infeasible is introduced; the extra
candidates can only expose a lower-scoring placement.

| | Total Score | Avg Cost | Feasible | Runtime |
|---|---|---|---|---|
| B-cluster + real-cost gate | 1.8032 | 1.9186 | 100/100 | 0.92s |
| **+ OBS_XCAND (default)** | **1.7608** | **1.8828** | 100/100 | 0.93s |

**Total Score 1.8032 → 1.7608 (−0.0424, ~2.4%)** — the largest area win after the
height-aware β, and **runtime-free** (just more candidate x's per landing). Unlike
β it is not a tuned scalar but a structural fix (more placement options, gated by
the real-cost / proxy selection), so overfitting risk is low. case0 1.8605→1.7960,
case99 1.7160→1.6373. Combines with B-cluster: a reshaped (narrower) super-block
plus a beside-obstacle x is what actually fills the gap.

### Gate baseline fix + deformable cohesion clusters — the largest win

**The rigid super-block is the dominant remaining waste.** A cluster's only hard
constraint is *connectivity* (grouping: members form one edge-connected component);
the rigid rectangle from `prepack_clusters` is an implementation choice, not a
requirement. Diagnostic on case 99: dropping the rigid super-block and placing the
members as individual reshapeable blocks cut **bbox −16% and HPWL −13%** — but they
scattered into 5 components (grouping 1→5), and the `e^{2V}` penalty made it lose.
So the lever is *deformable but connected* clusters.

**Two pieces were needed.**

1. **Gate baseline fix (Total 1.7608 → 1.7471).** `compute_cost` clamps each gap
   with `max(0,·)` (you can't beat the GT optimum). The real-cost gate baselined
   against candidate 0, so any candidate *better* on area got a negative gap →
   clamped to 0 → its improvement was invisible, and only the violation term
   decided (the rigid, lowest-violation candidate always won). Fix
   (`_gate_real_cost`): baseline against the **`min` bbox and `min` HPWL across
   candidates**, so every gap ≥ 0 and the smallest layout is the zero-gap reference.
   Inference-valid (no ground truth). This alone improved B-cluster selection.

2. **Cohesion packing (Total 1.7471 → 1.6039, the biggest single win).** The
   `FREE_CLUSTER` candidate packs cluster members as individual reshapeable blocks
   (empty super-blocks), but `_pack_one_width_shaped` adds a **cohesion term** to the
   landing objective: `score += COHESION_W · gap_to_nearest_placed_group_member`
   (L1 rectangle gap, 0 when abutting). Same-group members are placed consecutively
   (anchored at the group's lowest `cy`), so each lands abutting the blob already
   down → the cluster grows as a *connected, contour-conforming* shape instead of a
   bottom-left rectangle. Only the cohesion pass runs for this candidate (passes 1/2
   place members disconnected and their area·HPWL proxy ignores grouping). The
   member's `cluster_group` attribute stays set during packing (cohesion reads it)
   and is zeroed only for `slide_boundary`/`_finish`.

`COHESION_W` sweep (100 cases): 2→1.6339, 4→1.6158, **8→1.6039**, 16→1.6065,
32→1.6089 — a smooth U with a broad basin. Default 8.0. **Split-half validated**
(like β): `W=8` is the **independent minimum on every holdout** (even→odd 1.5923,
odd→even 1.6164, hash-A→B 1.5917, hash-B→A 1.6167), each equal to that holdout's
own best, and the U is monotonic on *both* halves (no spurious bumps) — so 8.0 is
not a full-set fluke. Cleaner than the β curve, which had a reproducible 0.4 bump.

| | Total Score | Avg Cost | case99 | Feasible | Runtime |
|---|---|---|---|---|---|
| OBS_XCAND (prev default) | 1.7608 | 1.8828 | 1.6373 | 100/100 | 0.93s |
| + gate baseline fix | 1.7471 | — | 1.6373 | 100/100 | — |
| **+ deformable cohesion clusters (default)** | **1.6039** | **1.6847** | **1.4555** | 100/100 | 0.83s |

**Total Score 1.7608 → 1.6039 (−0.157, ~9%)** — by far the largest legalize win,
and runtime-free (the gate short-circuits; cohesion replaces the rigid pack, not
adds to it). The real-cost gate is essential: it keeps the deformable layout only
when the area/HPWL win beats whatever grouping violation remains.

### B2 RESULT (`topo_shape.py`, `b2_pass`) — works, but flat on Total Score

B2 = Yan-Chu critical-path slack shaping on `analytic_legalizer`'s constraint
graph. Unlike B1's greedy per-block contour fit, it fixes the relative order from
B1's layout, computes each block's **global** horizontal/vertical slack from the
HCG/VCG longest paths, and reshapes blocks on the *binding critical path* into
their *perpendicular slack* (shorter+wider to cut height, or narrower+taller to
cut width) — including boundary blocks, which B1 skips (only clusters/MIB/
fixed/preplaced stay rigid). Kept only if overlap-free and the `bbox·HPWL` proxy
beats B1 (never-worse).

**Key implementation finding — the re-pack backend is everything.** The first
cut re-packed via `longest_path_pack` and was **~2× worse** (bbox 10089 → 22066):
the constraint-graph packer over-serialises (diagonal-separation edges), which is
exactly why the codebase replaced it with skyline. Switching the re-pack to lean
directional `compact` (no diagonal edges) brought it to 0.91–1.18× of skyline and
made B2 win 7/15 small cases.

| | Total Score | Avg Cost | Feasible | Runtime |
|---|---|---|---|---|
| B1 (default) | 1.8517 | 1.9593 | 100/100 | 0.87s |
| B1 + B2 (`b2_pass=True`) | **1.8517** | **1.9550** | 100/100 | 0.88s |

**Avg Cost improves (−0.0043) but Total Score is flat.** B2's wins concentrate on
*small* cases (21–35 blocks); Total Score weights by `e^{n/12}`, so the large
cases that dominate the headline barely move — and on large cases reshapeable
singletons are scarce (clusters ≈37% are rigid, and `compact` re-pack doesn't
reliably help dense HPWL). So B2 is **kept off by default** (`b2_pass=False`,
`topo_shape.py`); promoting it adds pipeline cost for no Total-Score gain.

**To actually move Total Score, B2 must reshape the rigid majority on large
cases** — i.e. a true joint **GP/LP solve over all blocks including clusters**
(reshape super-blocks, not just singletons), with a tight-packing guarantee the
constraint-graph packer doesn't provide. That is the remaining, heavier B2 (scipy
is available; cvxpy is not). B1 remains the default best.

**B2 under the real-cost gate (`RCG_B2=1`, default OFF).** Once the proxy gate was
replaced by the real-cost gate, B2 was added as an extra candidate (apply
`legalize_b2` to each base layout, `_finish`, then let `_gate_real_cost` pick).
Result: Avg Cost 1.9186 → **1.9144** (B2 again wins small cases) but Total Score
only 1.8032 → **1.8028** (−0.0004) — the same small-case-concentration story,
now confirmed under the *correct* gate, so the flat Total is structural, not a
gate artifact. It also costs +0.13s/case (B2 has no cheap short-circuit, unlike
cluster, so it runs every case). **Kept off by default**: the marginal Total gain
doesn't justify the runtime, especially given leaderboard runtime uncertainty.

---

## ML Leverage Points (for future work)

| Stage | Current | Opportunity |
|---|---|---|
| Block ordering (legalize) | ✅ done — `skyline_legalize()` reused (Phase 9) | further: learn an ordering/λ that correlates with low HPWL |
| Centers quality (Phase 9 remaining gap) | RL centers ~3.4% worse than quadratic placer's (2.0931 vs 2.0249) | more BC training data (>1500 samples) on the *same* skyline backend — gains are no longer masked by a bad packer |
| Shape selection (aspect) | wired but untrained end-to-end | train BC/PPO aspect head with `gt_aspect_bucket` labels |
| Training throughput | ~1 sample/sec, CPU, batch=1 | vectorize `PlacementEnv` + rasterizer for batched GPU training |
| RL rollout runtime | 2.25s/case avg (sequential `env.step()`) vs analytic 0.55s | feeds `R^0.3` in cost; batch/vectorize rollout if runtime factor becomes binding |

---

## File Map

| File | Role |
|---|---|
| `placement_env.py` | RL environment (reset/step/reward, hard-constraint snapping) |
| `canvas_raster.py` | partial placement -> `[4,G,G]` multi-channel raster |
| `gnn_encoder.py` | netlist -> node/graph embeddings (pure-torch SAGE-style) |
| `policy_net.py` | CNN + policy head (position + aspect) + value head |
| `train_rl.py` | PPO training loop |
| `pretrain_bc.py` | behaviour-cloning warm-start from GT |
| `ar_utils.py` | aspect-ratio action buckets |
| `hard_constraints.py` | `target_positions` builder + violation checker |
| `train_phase8.py` | streaming BC training over the 1M training set |
| `rl_optimizer.py` | Phase 7/8 `FloorplanOptimizer` subclass + `shelf_legalize`/`model_guided_legalize` |
| `rl_skyline_optimizer.py` | **Phase 9** — RL centers -> `analytic_legalizer.skyline_legalize()` chain (current best, quality-only 2.09) |
| `checkpoints/` | saved `bc_warmstart.pt` / `phase8_bc.pt` |
| `test_*.py` | acceptance tests, one per module (see below) |

### Acceptance tests

| Test | Checks |
|---|---|
| `test_placement_env.py` | random rollout reset->done runs cleanly, finite terminal cost |
| `test_canvas_raster.py` | 4-channel raster exact values on a hand-crafted placement + visual PNG |
| `test_gnn_encoder.py` | variable N (5/60/120), isolated graphs, gradients, real sample |
| `test_policy_net.py` | masked distribution sums to 1, value scalar, gradients, end-to-end smoke |
| `test_train_rl.py` | PPO overfit on a tiny sample set -> mean cost decreases |
| `test_pretrain_bc.py` | BC loss drops, cell_acc/near_acc rise, checkpoint save+reload |
| `test_ar_utils.py` | aspect action correctness, aspect head distribution/gradients, oracle shape benefit |
| `test_hard_constraints.py` | random actions -> zero hard-constraint violations |

Run any test with `cd FloorSet/iccad2026contest && python3 DL_RL/<test>.py`.

---

## Running it

```bash
cd FloorSet/iccad2026contest

# Full 100-case evaluation (uses checkpoints/phase8_bc.pt if present, else shelf fallback)
python3 iccad2026_evaluate.py --evaluate DL_RL/rl_optimizer.py

# Single test case
python3 iccad2026_evaluate.py --evaluate DL_RL/rl_optimizer.py --test-id 0

# Re-run BC training (overwrites checkpoints/phase8_bc.pt)
python3 DL_RL/train_phase8.py --num-samples 1500 --epochs 1 --grid 48

# Force the shelf fallback (bypass the trained model)
mv DL_RL/checkpoints/phase8_bc.pt DL_RL/checkpoints/phase8_bc.pt.bak
python3 iccad2026_evaluate.py --evaluate DL_RL/rl_optimizer.py
```
