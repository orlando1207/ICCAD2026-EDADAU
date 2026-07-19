# Diffusion Model for FloorSet: Repo Anatomy & Contest Variation Design

Date: 2026-07-14. Supersedes `2026-07-13-diffusion-floorplan-model-design.md` (this version adds
the full anatomy of `diff-model/` and drops everything legalizer-related — **this stage's only
goal is making the model's raw prediction as close as possible to the ground truth**). No code is
modified in this stage.

---

# Part A — Anatomy of the current repo (`iccad2026contest/diff-model/`)

## A.1 What it is

Official code of **MacroDiff** (Yoon, Jeon, Kang — "A Geometric Diffusion Model for Macro
Placement Generation", DAC'25 Late-Breaking Result). Targets ISPD2005 macro placement:
f macros + zero-size IO pads on a **fixed canvas**.

⚠ The PDF inside the folder (`2605.16451v2.pdf`) is **MacroDiff+** (arXiv, May 2026) — the
authors' nixed-dimensionewer system with a different architecture (dual-branch HeteroGNN + Transformer U-Net,
coordinate diffusion, physics-guided sampling). **The code does not implement that paper.** The
code implements the older net-HPWL-diffusion LBR. Treat the PDF as the roadmap, the code as a
parts bin.

## A.2 File map

| File | Role |
|---|---|
| `train.py` | Dataset class, training loop, periodic validation-by-sampling |
| `test.py` | Sampling + position recovery (10 seeds × 500-step Adam), plotting |
| `models/model.py` | `GraphPlacer` denoiser (`GraphConv` hetero-GNN + time embedding + output heads) |
| `models/diffusion.py` | `MacroDiff` DDPM wrapper: schedules, `q_sample`, posterior, `compute_loss`, `sample_ddpm` |
| `utils/score.py` | HPWL (exact + weighted-average smooth) and overlap (exact + smooth) kernels, torch_scatter-based |
| `utils/normalize.py` | `normalize(x, max) = 2x/max − 1`, inverse, `log_normalize` |
| `utils/plot.py` | Layout PNG rendering |
| `dataset/test.pt` | 8 ISPD2005 designs as PyG `HeteroData` |
| `checkpoint/checkpoint.ckpt` | Pretrained `GraphPlacer` weights (ISPD; useless for us — input semantics differ) |

## A.3 Data representation (PyG `HeteroData` per design)

- Two node types: **`node`** (cells: first `num_io` are IO pads with zero size and fixed
  positions, then `num_macro` movable macros) and **`net`** (one node per hypernet).
- Edges: `('node','out','net')` and `('net','in','node')` — a bipartite cell↔net graph.
  `edge_attr` = **pin offsets** (2-D, pin position relative to macro origin); the raw copy is
  kept in `['offset']`, the normalized copy in `.edge_attr`.
- Node features: `node.pos` (2), `node.size` (2); net features: `net.degree` (log-normalized).
- Everything spatial is normalized by `max_size` (the fixed canvas W, H) into [−1, 1].

## A.4 The diffusion design: what is actually diffused

**The diffusion variable is `d` — the vector of per-net HPWL values**, not positions:

1. From ground-truth positions, compute per-net smooth HPWL `d_0` (`get_hpwl_diff*`,
   weighted-average approximation with γ=0.01), normalize by (W+H).
2. Forward-noise it: `d_t = q_sample(d_0, t, ε)`, cosine β-schedule, T=300.
3. The denoiser sees: `node.x = size` (2ch), `net.x = [d_t, log_degree]` (2ch), pin-offset edge
   attrs, timestep `t` — and predicts the **noise on `d`** (ε-prediction, MSE loss, masked over
   padding).
4. At inference (`sample_ddpm`): 300 full DDPM steps produce a sampled `d` — a *prescription of
   how long each net should be* — with **no positions**.
5. Position recovery (`test.py`): for each of 10 seeds, initialize macro positions randomly and
   run **500 Adam steps** (lr 0.01, tanh reparameterization into the canvas, LR decayed to 0)
   minimizing `MSE(d_pred, d_sampled) + α·HPWL + β(step)·overlap` with α hand-tuned per design
   (5e-4 or 1e-3) and β ramping 0→0.1. Final positions are clamped into the canvas.

## A.5 Denoiser architecture (`GraphPlacer`, exact instantiation)

```
t ─► SinusoidalTimeEmb(512) ─► Linear 512→512 ─► SiLU ─► Linear 512→128 ──┐ (added to every
                                                                          │  layer input)
node.x (2) ─► Linear 2→128 ─► BN ─► ELU ─┐                                ▼
net.x  (2) ─► Linear 2→128 ─► BN ─► ELU ─┤   3 × HeteroConv layer:
                                         │     ('node','out','net'): GATv2Conv(128→128,
                                         └──►                        edge_dim=2, heads=4,
                                               ('net','in','node'):  concat=False)
                                               aggr='mean'
                                               + residual x ← 0.5·conv(x) + 0.5·x_prev
                                               + BatchNorm + ELU + dropout
        heads:  net_fc: 128→1  (the trained ε-prediction on d)
                cell_fc: 128→2 (DEAD — never enters the loss)
```

~0.6M parameters. Optimizer Adam lr=1e-2, batch 4 designs (padded to max length with −1 and
masked). Training loss = masked MSE on the **net head only**.

## A.6 Known quirks (confirmed by reading, relevant when reusing code)

1. `CircuitDataset` loads `dataset/test.pt` for *both* train and test — train set == test set.
2. The coordinate head (`cell_fc`) is dead weight; positions are never a training target.
3. `sample_ddpm(seed=None)` starts from `q_sample(d_0, random t)` — i.e. **leaks ground truth**
   into sampling; `test.py` passes explicit seeds to get pure noise, but the net mask
   (`d_0 != 0`) is still computed from ground-truth positions.
4. `design_list` is hardcoded; α in recovery is hand-tuned per design name.
5. All HPWL/overlap kernels are **unweighted** (ISPD nets have no weights) — contest HPWL is
   weighted (weights span ~60× within a case).
6. Normalization assumes a fixed canvas `max_size` — which does not exist in the contest.

**What is worth reusing**: the schedule/posterior math in `models/diffusion.py` (standard and
correct), the torch_scatter smooth-HPWL and smooth-overlap kernels in `utils/score.py` (after
adding net weights), the GATv2 hetero-conv pattern as an optional branch. **What is not**: the
net-HPWL diffusion variable, the recovery loop, the canvas normalization, the dataset plumbing.

---

# Part B — Why this doesn't fit the contest as-is

## B.1 The representation is wrong for our problem

- **All contest nets are 2-pin** (`b2b: (i, j, w)`, `p2b: (pin, block, w)` — verified on the
  tensors). Per-net HPWL degenerates to pairwise Manhattan distances; recovering positions from
  distances is a lossy distance-geometry problem (reflection/rotation ambiguities), *strictly
  harder* than predicting coordinates — which we have **1M ground-truth labels** for.
- The 500-step × 10-seed recovery is the runtime bottleneck and re-introduces the local-optima
  problem diffusion was supposed to avoid.
- **No shape output.** Contest blocks are soft (`w·h ∈ a·(1±1%)` is a *hard* constraint);
  net-HPWL carries zero shape information. MacroDiff simply has no channel for our most
  important degree of freedom.
- No mechanism for any of the contest constraints (fixed-shape, preplaced, MIB, grouping,
  boundary).

## B.2 Problem-setting gaps

| Aspect | MacroDiff (ISPD) | Contest (FloorSet-Lite, spec v10) |
|---|---|---|
| Canvas | fixed die; coordinates normalized to it | none; bbox area is an objective; terminals anchor an absolute frame |
| Block dims | fixed | soft ±1% (hard), fixed-shape/preplaced exact (hard) |
| Nets | multi-pin hypernets, pin offsets, unweighted | weighted 2-pin, center-to-center, b2b + p2b |
| Constraints | none | hard: overlap, area, fixed dims, preplaced pos+dims; soft: grouping, MIB, boundary |
| Data | 8 designs + rewiring augmentation, DREAMPlace pseudo-labels | 1M optimal-by-construction labels |
| Scale | 100s–1000s macros | 21–120 blocks; n=120 ≈ complete graph (~7k b2b edges) |

## B.3 Measured facts about our data (validation configs 21/60/100/120)

- GT layouts anchored at origin; **terminals ring the layout bbox** (0 interior pins in all
  checked cases) → pins and solution share an absolute frame the model can learn.
- GT utilization **96–97%** (near-abutting); GT area error exactly 0.
- GT aspect ratios ∈ **[0.34, 3.0]** → `s = ½log(w/h) ∈ [−0.55, 0.55]` — bounded, well-scaled.
- MIB groups have identical area targets and identical GT dims.
- Typical constraint mix (n=100): 10 fixed, 7 preplaced, 1 MIB group (~5 blocks), 3 cluster
  groups (~24 blocks), 30 boundary blocks.
- Contest score weight `e^{n/12}` ⇒ n=120 counts ~3800× n=21 → **quality at n≈100–120 is what
  matters**, exactly where the b2b graph is densest.

---

# Part C — The contest variation: **FloorDiff** (prediction-only stage)

Goal of this stage: **generate `(x, y, w, h)` for all blocks as close as possible to the ground
truth**. GT is effectively unique per case (the terminal frame pins down translation/rotation),
so GT-closeness is a well-posed target. No legalizer; no downstream repair assumed.

## C.1 Diffusion variable: `z_i = (cx_i, cy_i, s_i)` per block

- `(cx, cy)` = block **center**; `s = ½·log(w/h)`; decode `w = √a·e^s`, `h = √a·e^{−s}`.
- **The area hard constraint is satisfied exactly, by construction** — the model cannot emit an
  illegal area. One entire infeasibility mode (score = 10) is eliminated before training starts.
- 3 channels, all near-unit scale (scale `s` by ~2). No rotation variable — spec v10 forbids
  w/h swap for fixed/preplaced blocks, and for soft blocks rotation is absorbed by `s`.

### Constraints handled by construction, not by learning

| Constraint | Mechanism | Effect |
|---|---|---|
| Preplaced (hard) | freeze all 3 channels at input values; **inpaint** during sampling (re-impose `q_sample(z_known, t)` after every step); mask out of the loss | exact by construction; acts as fixed spatial context, like MacroDiff's IO pads |
| Fixed-shape (hard) | freeze `s` only; diffuse `(cx, cy)` | dims exact by construction |
| MIB (soft) | **tie `s` across the group** (one shared latent / average predictions each step); groups have equal areas (verified) so tied `s` ⇒ identical `(w,h)` | `V_mib ≡ 0` by construction; if the group contains a fixed block, inherit its frozen `s` |
| Boundary / grouping (soft) | features + attention bias + auxiliary losses (below) — learned, not constructed | minimized, not guaranteed |

After this, the model's entire learning burden is: **where blocks go, what aspect they take,
staying overlap-free, honoring boundary/grouping tendencies** — i.e., exactly reproducing GT.

### Normalization (no canvas exists)

- Scale `S = √(Σ aᵢ)` (deterministic from inputs; ≈ bbox side, since GT util ≈ 97%).
- Keep the **terminal-anchored absolute frame**: shift by the terminal-bbox center (computable at
  inference), divide by `S`. Never center on the GT solution (train-only information).
- GT centers land in ≈[−0.6, 0.6]² — comfortable diffusion range.

## C.2 Denoiser architecture

For ≤120 blocks (+ ≤~400 terminals) everything is tiny; a dense Transformer is the right tool —
especially since at n=120 the b2b graph is near-complete, which defeats sparse message passing
anyway (the structure is in the *weights*, not the topology).

```
                      ┌────────────────────────────────────────────────────┐
 static feats (§C.3) ─┤  per-block token:  [ z_t (3) | x̂0_selfcond (3) |   │
 noisy z_t ───────────┤                      static features ]  → Linear   │
                      │                                                    │
 t ─► sinusoidal ─► MLP ─► AdaLN-Zero (scale/shift/gate every sublayer)    │
                      │                                                    │
 global feats ─► register token(s)                                         │
                      │                                                    │
                      │  N × Transformer block (8–12):                     │
                      │    self-attention with additive per-head bias      │
                      │      b_ij = MLP(log ŵ_ij, same_cluster, same_mib)  │
                      │    (optional) cross-attention → terminal tokens    │
                      │    FFN                                             │
                      │                                                    │
                      │  head: Linear → x̂0 prediction (3 per block)        │
                      └────────────────────────────────────────────────────┘
```

Design choices and why:

- **x̂0- (or v-) prediction**, not ε: for low-dimensional structured outputs it converges faster,
  and it lets auxiliary geometric losses and inference guidance act on decoded geometry directly.
  Use **min-SNR-γ** loss weighting.
- **AdaLN-Zero** time conditioning (DiT recipe) instead of MacroDiff's "add t to every layer
  input" — better-conditioned, standard.
- **Connectivity-biased attention** (Graphormer-style) carries the netlist: bias from
  quantile-normalized `log w_ij` + `same_cluster`/`same_mib` pair flags. This is where grouping
  and MIB relations enter — **group IDs are arbitrary per sample, so identity must be relational
  (pairwise), never a categorical embedding.**
- **Self-conditioning** on the previous x̂0 estimate (dropped 50% during training).
- **Terminals**: primary mechanism is per-block *pin-pull summary features* (§C.3) — zero
  sequence cost; add terminal tokens via cross-attention only if ablation shows the summary
  loses signal (e.g. blocks pulled by two opposite terminal clusters).
- **Optional GNN branch** (GATv2 over the weighted block graph, reusing the repo's hetero-conv
  pattern, fused by residual addition every k layers) — an *ablation*, not a commitment.
  MacroDiff+'s dual branch exists because hypernets are awkward for Transformers; our 2-pin nets
  fit attention natively, and its gradient-projector fusion `(∇x HPWL)ᵀ·ε_net` reduces, for
  2-pin nets, to weighted pairwise attraction — which the attention bias + aux HPWL loss already
  express.
- Size: 8–12 layers, d_model 256–384, 8 heads ⇒ ~10–25M params. Small on purpose: 1M samples,
  and inference must stay fast (runtime is scored).

## C.3 Feature engineering

### Per-block static features

| Group | Features |
|---|---|
| Size | `log(aᵢ/ā)`, `√aᵢ/S` |
| Fixed-shape | flag; frozen `s`; `(w,h)/S` |
| Preplaced | flag; `(cx,cy)/S`; `(w,h)/S` |
| Boundary | 4-bit one-hot from bitmask (L=1,R=2,T=4,B=8; corners = 2 bits) |
| MIB / Cluster | in-group flags; group sizes (identity is pairwise-only, in attention bias) |
| Connectivity | weighted b2b degree; neighbor count; total p2b weight; **pin-pull centroid** `(Σw·p)/(Σw)/S`; pin-pull dispersion |
| Self-cond | previous `x̂0` estimate (3) |

### Global features (register token + AdaLN input)

`n`, `log S`, terminal-bbox `(w,h)/S`, terminal count, total b2b weight, per-type constraint
counts. `n` matters doubly: score weight concentrates on large `n`, and packing statistics shift
with `n`.

### Pairwise (attention-bias) features

`log ŵ_ij` (quantile-normalized b2b weight), `same_cluster`, `same_mib`. (Pin offsets — the
repo's edge feature — don't exist in Lite; this is the slot they vacate.)

## C.4 Training design

- **Data**: 1M training samples (`get_training_dataloader` shards → preprocessed graph samples).
  Bucket-batch by `n` (pad + mask). The 100 validation cases are held out strictly for eval.
- **Case sampling**: oversample large `n` softly (∝ `e^{n/24}`) — mirrors the contest weighting
  without starving small cases.
- **Primary loss**: masked MSE on `x0` (excluding frozen channels), min-SNR-γ weighted. This *is*
  the "close to ground truth" objective.
- **Auxiliary geometry losses** on decoded x̂0, weighted toward low-noise timesteps (meaningless
  at high `t`): weighted-HPWL match to GT; smooth pairwise-overlap penalty; boundary-distance
  penalty (distance of required block side to the soft bbox side, smooth min/max over blocks);
  cluster-cohesion (pairwise center gaps beyond abutment distance). All four kernels can be
  adapted from `utils/score.py` + net weights. These teach the *physics* behind GT rather than
  just its coordinates, and empirically (MacroDiff+ ablations) beat guidance-only physics.
- **Augmentation**: dihedral symmetries applied consistently to blocks + terminals + metadata —
  boundary bitmasks remap under flips (x-flip: L↔R), `s → −s` under 90° rotations. Because
  fixed/preplaced dims may not swap, use only the 4 orientation-preserving symmetries (id,
  x-flip, y-flip, 180°) on cases containing them, or remap their frozen dims and skip rotations.
  Block-order permutation is free (set model). **No coordinate jitter** — it destroys abutment.
- **Diffusion setup**: cosine schedule, T=1000 (continuous-time optional), EMA 0.9999, grad clip.
- Compute: ~20M params × 1M samples ⇒ single-GPU-days (env: `~/miniconda3/envs/iccad`).

## C.5 Inference pipeline

```
inputs ─► featurize (frame = terminal bbox center, scale = S)
       ─► freeze preplaced (cx,cy,s) & fixed (s); tie MIB s
       ─► sample N seeds AS ONE BATCH (tiny graphs → one forward pass)
            DDIM / DPM-Solver++, 30–50 steps
            each step:
              x̂0 ← model prediction
              physics-guided correction (MacroDiff+ Alg. 1):
                 K ≈ 5–10 grad steps on
                 L = w₁·weightedHPWL + w₂·overlap + w₃·boundary + w₄·clusterGap + w₅·bboxArea
                 (two-phase schedule: quality first → feasibility late)
              re-derive ε_guided, take the DDIM step
              re-impose inpainted channels (preplaced) via q_sample
       ─► short clean-space polish (≤100–300 Adam steps, overlap-weight ramp)
       ─► snap: rescale (w,h) by √(a/(w·h)) → area exact; MIB already tied; preplaced exact
       ─► rank the N samples by evaluator-cost proxy; keep the best
       ─► emit (x, y, w, h)
```

- **All guidance terms are differentiable through `s`** (`(w,h) = √a·(e^s, e^{−s})`), so guidance
  *reshapes* blocks, not just moves them — that is how soft-block flexibility is actually
  exploited (flatten a block to slip beside a fixed one).
- The polish step is the vestige of MacroDiff's recovery loop, but ~30–50× cheaper because the
  sampler already outputs a placement rather than a distance prescription.
- Runtime: n≤120, 50 steps × K grad steps × N∈[8,32] batched ⇒ seconds per case on GPU, still
  CPU-viable at reduced N/steps if the judging environment demands it.

## C.6 Evaluation protocol (GT-closeness, no legalizer)

On the 100 validation cases, reported per-n bucket (21–60 / 61–100 / 101–120) and
contest-weighted (`e^{n/12}`):

**Closeness to ground truth (primary at this stage)**
1. Mean/median normalized center displacement `|ĉ − c_GT| / S` per block.
2. Shape error `|ŝ − s_GT|` (equivalently relative dim error) on soft blocks.
3. Fraction of blocks within δ of their GT position (δ = 1%, 2%, 5% of S).

**Quality/feasibility of the raw prediction (secondary, sanity)**
4. Overlap ratio: Σ pairwise overlap / Σ block area.
5. Weighted-HPWL gap and bbox-area gap vs. baseline (`metrics[6]+metrics[7]`, `metrics[0]`),
   computed with the official `calculate_hpwl_*` / bbox code from `iccad2026_evaluate.py`.
6. Soft-violation counts via the official evaluator (MIB must read 0 — anything else is a bug in
   the tying logic).

Ablation ladder: backbone alone → + constraint freezing/tying → + aux physics losses → +
physics-guided sampling → + polish + best-of-N → ± GNN branch → ± terminal tokens.

## C.7 Risks / open questions

- **Grouping abutment** is measure-zero for a continuous model — expect near-contact, not exact
  contact. At this stage that's acceptable (it barely affects GT-closeness metrics); revisit when
  feasibility becomes the target.
- **Whitespace mismatch**: GT is 96–97% packed; diffusion output will be a few % looser. Track
  bbox-area gap; if systematic, strengthen the bbox aux term.
- **Frame assumption** (terminal ring ⇒ absolute anchor) verified on 4 validation cases; verify
  on ~1k training-shard samples before locking normalization. Fallback: pin-bbox-centered frame.
- **Multimodality**: if training shards contain near-ties (multiple optimal layouts), pure
  MSE-to-GT metrics will plateau even for a good model — watch the gap between closeness metrics
  (1–3) and quality metrics (4–6); quality metrics are the tie-breaker.
- Contest anti-reverse-engineering clause: supervised learning on the provided data is the
  intended use; stay clear of generator-inversion tricks.

## C.8 Phased plan

| Phase | Deliverable | Exit criterion |
|---|---|---|
| P0 | Data pipeline (shards → normalized samples) + eval harness (§C.6) | harness reproduces official evaluator numbers on GT exactly |
| P1 | Backbone diffusion on `(cx,cy,s)`, no constraints/guidance | displacement and HPWL gap clearly beat a "pin-pull + greedy pack" trivial baseline |
| P2 | Constraint conditioning: freezing, inpainting, MIB tying, boundary/cluster features + bias | preplaced/fixed exact; MIB = 0; closeness improves on constrained blocks' neighbors |
| P3 | Aux physics losses | overlap ratio and HPWL gap drop without closeness regressing |
| P4 | Physics-guided sampling + polish + best-of-N | best contest-weighted proxy; overlap < ~3% |
| P5 | Ablations (GNN branch, terminal tokens, sampler steps ↓ for runtime) | final architecture locked for the next stage |
