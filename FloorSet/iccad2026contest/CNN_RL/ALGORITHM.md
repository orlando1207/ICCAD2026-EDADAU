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
mostly closes) — the shape lever is worth learning, **but it is not yet
trained end-to-end** (BC/PPO labels for aspect are TODO).

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
