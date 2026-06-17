# CNN_RL — GNN + CNN + RL Sequential Placer (ICCAD 2026 Contest C)

Independent, learned alternative to `analytic_legalizer/`'s deterministic
pipeline: a GNN+CNN policy picks placement centers, which are then run through
`analytic_legalizer`'s own legalization chain. **Does not modify
`analytic_legalizer/`** — only imports its pure functions (`skyline_legalize`,
`parse_and_init`, `prepack_clusters`, `slide_boundary`, `enforce_hard`,
`_detailed_place`).

## TL;DR — current status

Evaluated with the corrected scoring formula (`Total Score = Σ Cost[i]·e^{n_i/12}
/ Σ e^{n_j/12}`, `RuntimeFactor=1.0`):

| Optimizer | Avg Cost | Total Score | Feasible |
|---|---|---|---|
| `analytic_legalizer/my_optimizer.py` (baseline) | 2.0632 | 1.9788 | 100/100 |
| `rl_skyline_optimizer.py` (Phase 11, `phase11_pin_soft.pt`, square soft blocks) | 2.0172 | 1.8977 | 100/100 |
| `rl_skyline_optimizer.py` (Phase 13, `phase13_aspect.pt`, predicted aspect ratios) | 1.9705 | 1.8650 | 100/100 |
| `rl_skyline_optimizer.py` (Phase 15/B1 contour-aware shaping, `shape_fit=True`, β=0) | 1.9593 | 1.8517 | 100/100 |
| `rl_skyline_optimizer.py` (Phase 15/B1+ height-aware shaping, `SHAPE_HEIGHT_BETA=0.3`) | 1.9343 | 1.8088 | 100/100 |
| `rl_skyline_optimizer.py` (+ B-cluster reshape via real-cost gate) | 1.9186 | 1.8032 | 100/100 |
| `rl_skyline_optimizer.py` (+ obstacle-aware x-candidates `OBS_XCAND`) | 1.8828 | 1.7608 | 100/100 |
| **`rl_skyline_optimizer.py`** (default, + deformable cohesion clusters `FREE_CLUSTER`) | **1.6847** | **1.6039** | 100/100 |
| `rl_skyline_optimizer.py` (`phase14_aspect_100k.pt`, 100k-sample continuation) | 2.0281 | 1.9159 | 100/100 |
| `rl_skyline_optimizer_compact.py` (+ whitespace compaction, on Phase 11 ckpt) | 2.0351 | 1.9242 | 100/100 |

`rl_skyline_optimizer.py` is the file to submit/use — it currently **beats the
analytic baseline** on both metrics. Phase 13 adds an aspect-ratio head: for
"free" soft blocks (no fixed/MIB/cluster shape constraint), the policy picks a
non-square aspect bucket (`ar_utils.ASPECT_BUCKETS`) instead of the default
forced square, area-preserving (`aspect_pass=True`). **Phase 15/B1**
(`shape_fit=True`, now default) adds *contour-aware in-packer shaping*
(`skyline_shape.py`): when the skyline packer lands a soft block it picks the
area-constant aspect that best fills the current contour notch, as a third
candidate mode that is never worse than stock — 1.8650 → 1.8517. **Phase 15/B1+**
adds a height penalty `β·h` to the landing objective (`SHAPE_HEIGHT_BETA=0.3`):
the original objective only minimised landing-y, so the greedy preferred tall
narrow shapes that spike the skyline; penalising the block's own height keeps the
packed envelope low and drops Total Score 1.8517 → **1.8088** (−2.3%, the biggest
legalize-side win). See `ALGORITHM.md` §"Aspect Ratio" for the PoC that motivated
it (a free-blocks-only post-pass measured 0.00; shaping must live *inside* the
packer) and the full β sweep. **Phase 15/B-cluster** (`real_cost_gate`, now
default) re-packs cluster super-blocks into a contour-fitting aspect (area +
connectivity preserved) and selects via a gate scored on the *exact* contest cost
`(1+0.5(area_gap+hpwl_gap))·e^{2V}` instead of the old `bbox·HPWL` proxy — which
omits the violation term and had made cluster/MIB/B2 reshaping look like losses.
This unlocks cluster reshaping: 1.8088 → **1.8032**, runtime unchanged (the gate
short-circuits when no cluster is reshapeable). **Obstacle-aware x-candidates**
(`OBS_XCAND`, now default) then fix the largest remaining structural waste: the
skyline only tracks the top contour, so a rigid cluster super-block wider than the
gap beside a *preplaced* obstacle was bumped **over** it (pushed outward, growing
the bbox). Offering flush-left/right-of-obstacle x positions lets it slot *beside*
the obstacle instead — all 100 cases have preplaced+cluster, so this is a big,
runtime-free win: 1.8032 → **1.7608**.
`phase14_aspect_100k.pt` continued training from Phase 13 on 80k fresh samples
(indices 20k–100k) but **regressed** (1.9705→2.0281 avg cost) despite better
training-curve metrics — `phase13_aspect.pt` remains the default checkpoint.
`rl_skyline_optimizer_compact.py` adds a whitespace-compaction pass that was a
net loss on the Phase 11 checkpoint; not yet retried.

## Quick start

All commands run from `FloorSet/iccad2026contest/`.

### Evaluate (this is the submission entry point)

```bash
python3 iccad2026_evaluate.py --evaluate CNN_RL/rl_skyline_optimizer.py
```

Add `--test-id 0 --verbose` to debug a single case.

### Visualize a placement

```bash
python3 CNN_RL/plot_rl_skyline.py --test-id 0 --out CNN_RL/rl_skyline_case0.png
python3 CNN_RL/plot_rl_skyline.py --test-id 0 --ground-truth --out CNN_RL/gt_case0.png
python3 CNN_RL/plot_compare.py --test-id 0 --out CNN_RL/compare_case0.png
```

### Train from scratch

One script, `train_fast.py` — vectorised rasterizer + batched GNN/CNN forward
+ parallel `DataLoader` workers (~10+ samples/s on a 24-core box):

```bash
python3 CNN_RL/train_fast.py \
    --num-samples 20000 --epochs 1 --grid 64 --workers 16 \
    --soft-sigma 1.5 --aspect-weight 0.5 --ckpt-name phase13_aspect.pt
```

- `--soft-sigma 1.5`: Gaussian-blurred target-cell label (soft cross-entropy)
  instead of one-hot — improved avg cost 2.0944 -> 2.0570.
- The 5th raster channel (`CH_PIN_PULL`, pin-aware HPWL field, always on in
  `canvas_raster.py`) improved avg cost 2.0570 -> 2.0408.
- `--aspect-weight 0.5`: weight of the aspect-bucket BC loss (Phase 13) —
  predicts a non-square shape for "free" soft blocks, avg cost 2.0172 -> 1.9705.
- Checkpoint is written to `checkpoints/<ckpt-name>`; `rl_skyline_optimizer.py`
  loads `checkpoints/phase13_aspect.pt` by default (`_DEFAULT_CKPT`).

### Continue training (more data, same checkpoint)

The full dataset has 1,008,000 samples, so a 20k run only ever sees indices
`[0, 20000)`. To keep training the same model on fresh data:

```bash
python3 CNN_RL/train_fast.py \
    --num-samples 20000 --start-idx 20000 \
    --epochs 1 --grid 64 --workers 16 \
    --soft-sigma 1.5 --aspect-weight 0.5 \
    --init-ckpt CNN_RL/checkpoints/phase13_aspect.pt \
    --ckpt-name phase13_aspect.pt
```

- `--init-ckpt <path>`: load gnn/policy weights from an existing checkpoint
  before training (optimizer state is fresh Adam, not resumed).
- `--start-idx N`: skip the first `N` dataset samples, so the next run trains
  on data the model hasn't seen yet instead of repeating the same slice.
- `--ckpt-name` overwrites in place if it matches `--init-ckpt`, or use a new
  name (e.g. `phase14_aspect.pt`) to A/B against the previous checkpoint.

### Run the acceptance tests

Each `test_*.py` is a standalone acceptance test for the module of the same
name:

```bash
python3 CNN_RL/test_placement_env.py
python3 CNN_RL/test_canvas_raster.py
python3 CNN_RL/test_gnn_encoder.py
python3 CNN_RL/test_policy_net.py
python3 CNN_RL/test_pretrain_bc.py
python3 CNN_RL/test_hard_constraints.py
```

## Repo layout

```
CNN_RL/
├── rl_skyline_optimizer.py           # main FloorplanOptimizer (current best, submit this)
├── rl_skyline_optimizer_compact.py   # + whitespace-compaction pass (experiment, currently a net loss)
├── rl_skyline_optimizer_quad_ablation.py  # sanity check: quadratic-placer centers
│                                           #   through the same legalizer chain
├── train_fast.py                     # training entrypoint (BC, vectorised + parallel)
├── placement_env.py                  # RL environment (reset/step/reward)
├── canvas_raster.py                  # partial placement -> [5,G,G] raster
├── gnn_encoder.py                    # netlist -> node embeddings (SAGE-style)
├── policy_net.py                     # CNN trunk + policy/value heads
├── ar_utils.py                       # aspect-ratio bucket utilities (Phase 5/13)
├── skyline_shape.py                  # Phase 15/B1 contour-aware in-packer shaping (skyline_legalize_shaped)
├── topo_shape.py                     # Phase 15/B2 critical-path slack shaping (legalize_b2, off by default)
├── poc_slack_shaping.py              # Phase 15 PoC: measures the Area_gap ceiling of block reshaping
├── pretrain_bc.py / train_phase8.py  # utility modules imported by train_fast.py
├── hard_constraints.py               # hard-constraint snapping used by placement_env.py
├── plot_rl_skyline.py / plot_compare.py  # visualization helpers
├── test_*.py                         # one acceptance test per active module
├── checkpoints/phase13_aspect.pt     # current default checkpoint (Phase 13, 20k samples, aspect-ratio head)
├── checkpoints/phase14_aspect_100k.pt  # Phase 14 continuation (100k samples total) — net loss, not default
├── ALGORITHM.md                      # full phase-by-phase history & lessons
└── archive/                          # deprecated/abandoned, kept for reference
    ├── rl_optimizer.py               # Phase 7/8 (Total Score 10.05 / 15.60)
    ├── train_ppo.py                  # Phase 12 PPO+KL-anchor (abandoned)
    ├── test_ar_utils.py              # Phase 5 acceptance test for the aspect-ratio head
    ├── train_rl.py / test_train_rl.py     # Phase 4 PPO loop, superseded by train_fast.py
    ├── checkpoints/                  # old/intermediate checkpoints incl. phase11_pin_soft.pt
    └── HANDOFF.md, VERSION_B_RL_PLACER.md  # superseded planning docs
```

---

## Architecture (condensed from `ALGORITHM.md`)

```
GNN encodes the netlist (who connects to whom)        <- "graph" half
  + the partial placement is rasterized to a grid      <- "vision" half
  + a CNN reads that grid
  + greedy policy picks where to drop each block, one at a time
  + per-block centers (cx,cy) -> analytic_legalizer.skyline_legalize()
  + slide_boundary -> enforce_hard -> _detailed_place
```

| | GNN (graph half) | CNN (vision half) |
|---|---|---|
| Module | `gnn_encoder.py` | `policy_net.py` |
| Input | netlist node features `[N,10]` + b2b edges | canvas raster `[5,G,G]` + current block's GNN embedding |
| Sees | connectivity/topology, area, constraints | spatial geometry: density, low-HPWL/pin spots, legal cells |
| Output | per-block embedding `[N,D]` | per-cell placement logits `[G,G]` (+ value head) |
| Runs | once per problem | once per block |

Canvas channels (`canvas_raster.py`, `N_CHANNELS=5`): occupancy, density,
b2b wirelength-pull, feasibility mask, **pin-pull** (p2b HPWL field, Phase 11).

### MDP

- **State**: placed blocks so far + canvas raster + GNN embeddings + next block id
- **Action**: grid cell `[0, G*G)` = lower-left corner for the next block
- **Order**: preplaced/fixed blocks first (placed immediately), then remaining
  blocks by area descending
- **Reward**: terminal only, `-(compute_training_loss_differentiable(...))`

### Pipeline stages

1. `placement_env.py` — env reset/step, hard-constraint snapping
2. `canvas_raster.py` — partial placement -> `[5,G,G]` raster
3. `gnn_encoder.py` — pure-PyTorch SAGE-style message passing, `[N,10] -> [N,D]`
4. `policy_net.py` — CNN trunk + policy/value heads, masked softmax over `[G,G]`
5. `train_fast.py` — batched behaviour-cloning vs. GT placements (soft-label CE)
6. `rl_skyline_optimizer.py` — greedy rollout -> centers -> `analytic_legalizer`
   skyline chain -> `FloorplanOptimizer.solve()`

### Results progression (quality-only avg cost, lower is better)

| Variant | Avg Cost | Notes |
|---|---|---|
| Phase 7 — `shelf_legalize` (no model) | 8.07 | feasibility floor |
| Phase 8 — `model_guided_legalize` (1500-sample BC) | 8.72 | row-packer was the bottleneck |
| Phase 9 — RL centers -> `skyline_legalize` | 2.0931 | same checkpoint, -75% by reusing analytic's legalizer |
| Phase 10 — hard-label BC, 20k samples | 2.0944 | more data alone ~flat |
| Phase 10 — + soft-label CE (`soft-sigma=1.5`) | 2.0570 | |
| Phase 11 — + pin-pull channel (`CH_PIN_PULL`) | 2.0408 | superseded by Phase 13 |
| Phase 11 + compaction pass | 2.0351 (avg) / Total Score 1.9242 | net loss vs 1.8977 — kept as `_compact` variant |
| Phase 12 — PPO + KL-anchor fine-tune | no improvement | abandoned, see `archive/train_ppo.py` |
| Phase 13 — + aspect-ratio head for "free" soft blocks | 1.9705 (avg) / Total Score 1.8650 | `phase13_aspect.pt`, 20k samples |
| Phase 14 — continued training on 80k fresh samples (`--init-ckpt phase13 --start-idx 20000`) | 2.0281 (avg) / Total Score 1.9159 | **net loss** — despite better training metrics; fresh Adam on resume likely cause |
| Phase 15/B1 — contour-aware in-packer shaping (`skyline_shape.py`, `shape_fit=True`, β=0) | 1.9593 (avg) / Total Score 1.8517 | never-worse third pack mode |
| Phase 15/B1+ — height-aware landing objective (`SHAPE_HEIGHT_BETA=0.3`) | 1.9343 (avg) / Total Score 1.8088 | `score += β·h` stops tall shapes spiking the skyline (−2.3%) |
| Phase 15/B-cluster — re-pack cluster super-blocks, gated on real cost (`real_cost_gate`) | 1.9186 (avg) / Total Score 1.8032 | real-cost gate (exact `(1+0.5(gaps))·e^{2V}` form) unlocks cluster reshaping the `bbox·HPWL` proxy rejected; short-circuited so runtime unchanged |
| Phase 15/OBS_XCAND — obstacle-aware x-candidates in the shaped packer | 1.8828 (avg) / Total Score 1.7608 | lets a super-block sit *beside* a preplaced obstacle instead of bumped over it; runtime-free (−2.4%) |
| Phase 15/gate-baseline fix — `min`-over-candidates baseline in the real-cost gate | Total Score 1.7471 | the contest clamps gaps with `max(0,·)`; baselining against candidate 0 hid any *better* candidate (negative gap → 0). `min` baseline makes area wins visible |
| Phase 15/FREE_CLUSTER — deformable cohesion clusters | **1.6847 (avg) / Total Score 1.6039** | **current default**; place cluster members as individual reshapeable blocks with a cohesion pull so they form a connected blob that conforms to the contour instead of a rigid bottom-left rectangle (−8% Total) |
| Phase 15/B2 — critical-path slack shaping (`topo_shape.py`, `b2_pass=True`) | 1.9550 (avg) / Total Score 1.8517 | off by default — improves Avg Cost but **Total Score flat** (wins are on small cases) |
| `analytic_legalizer` baseline | 2.0632 (avg) / Total Score 1.9788 | CNN_RL now beats this |

### Key lessons

1. **The legalizer matters more than model accuracy.** Same Phase-8
   checkpoint: `model_guided_legalize` -> 8.72, `skyline_legalize` -> 2.09
   (Phase 9).
2. **Total Score weighting**: `e^{n_i/12} / Σ e^{n_j/12}` (`n` = block count,
   21-120) — larger cases dominate but every size has non-zero weight.
   `RuntimeFactor` is fixed at 1.0 locally (no cross-submission median is
   available); only relative quality (`hpwl_gap`/`area_gap`/`violations`) is
   comparable across optimizers run in the same harness.
3. **Soft labels + the pin-aware raster channel** were the two real wins
   after Phase 9 (2.0931 -> 2.0570 -> 2.0408); compaction and PPO fine-tuning
   were tried on top of this checkpoint and did not help.
4. Training is CPU-bound on the rasterizer, not the CNN forward — parallel
   `DataLoader` workers (`--workers`) are the lever that scales
   (~1.4 -> 10+ samples/s).
5. **The "shape lever"**: forcing every soft block to a square (`w=h=sqrt(area)`)
   was a real cost. Phase 13's `aspect_head` picks one of
   `ar_utils.ASPECT_BUCKETS` per "free" block (BC-trained on GT aspect ratios,
   `aspect_acc` ~0.46-0.48 vs 0.20 chance) and overrides that block's (w,h)
   area-preservingly before legalization — 2.0172 -> 1.9705.
6. **Continuing BC training on fresh data can hurt** (Phase 14). Resuming from
   a checkpoint with `--init-ckpt` resets the Adam optimizer state (momentum
   and variance are cold), which can cause a destabilizing transient in the
   early steps of the continuation run. Despite better training-curve metrics
   (`cell_acc` 0.09→0.15, `aspect_acc` 0.47→0.52), eval cost regressed
   1.9705→2.0281. Possible mitigations: lower `--lr` for continuation runs
   (e.g. `1e-4` vs default `1e-3`), or save/restore optimizer state alongside
   model weights.

For full phase-by-phase detail, failure modes, and file-level notes, see
[`ALGORITHM.md`](ALGORITHM.md). For deprecated/abandoned code, see
[`archive/README.md`](archive/README.md).
