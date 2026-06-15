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
| **`rl_skyline_optimizer.py`** (default, `phase11_pin_soft.pt`) | **2.0172** | **1.8977** | 100/100 |
| `rl_skyline_optimizer_compact.py` (+ whitespace compaction) | 2.0351 | 1.9242 | 100/100 |

`rl_skyline_optimizer.py` is the file to submit/use — it currently **beats the
analytic baseline** on both metrics. `rl_skyline_optimizer_compact.py` adds a
whitespace-compaction pass that is a net loss on this checkpoint; kept around
because the compaction logic hasn't been tried with the newer pin-aware
checkpoint yet.

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
    --soft-sigma 1.5 --ckpt-name phase11_pin_soft.pt
```

- `--soft-sigma 1.5`: Gaussian-blurred target-cell label (soft cross-entropy)
  instead of one-hot — improved avg cost 2.0944 -> 2.0570.
- The 5th raster channel (`CH_PIN_PULL`, pin-aware HPWL field, always on in
  `canvas_raster.py`) improved avg cost 2.0570 -> 2.0408.
- Checkpoint is written to `checkpoints/<ckpt-name>`; `rl_skyline_optimizer.py`
  loads `checkpoints/phase11_pin_soft.pt` by default (`_DEFAULT_CKPT`).

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
├── pretrain_bc.py / train_phase8.py  # utility modules imported by train_fast.py
├── hard_constraints.py               # hard-constraint snapping used by placement_env.py
├── plot_rl_skyline.py / plot_compare.py  # visualization helpers
├── test_*.py                         # one acceptance test per active module
├── checkpoints/phase11_pin_soft.pt   # current default checkpoint
├── ALGORITHM.md                      # full phase-by-phase history & lessons
└── archive/                          # deprecated/abandoned, kept for reference
    ├── rl_optimizer.py               # Phase 7/8 (Total Score 10.05 / 15.60)
    ├── train_ppo.py                  # Phase 12 PPO+KL-anchor (abandoned)
    ├── ar_utils.py / test_ar_utils.py     # Phase 5 aspect-ratio action head (unused)
    ├── train_rl.py / test_train_rl.py     # Phase 4 PPO loop, superseded by train_fast.py
    ├── checkpoints/                  # old/intermediate checkpoints
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
| Phase 11 — + pin-pull channel (`CH_PIN_PULL`) | **2.0408** | current default |
| Phase 11 + compaction pass | 2.0351 (avg) / Total Score 1.9242 | net loss vs 1.8977 — kept as `_compact` variant |
| Phase 12 — PPO + KL-anchor fine-tune | no improvement | abandoned, see `archive/train_ppo.py` |
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

For full phase-by-phase detail, failure modes, and file-level notes, see
[`ALGORITHM.md`](ALGORITHM.md). For deprecated/abandoned code, see
[`archive/README.md`](archive/README.md).
