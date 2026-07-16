# Diffusion Model for FloorSet: Repo Anatomy & Contest Variation Design

Date: 2026-07-14, design part (Part C) revised 2026-07-15. Supersedes
`2026-07-13-diffusion-floorplan-model-design.md`. **This stage's only goal is making the model's
raw prediction as close as possible to the ground truth** — which, because every GT layout is
feasible, simultaneously drives the prediction toward constraint satisfaction for free
(imitation-first principle, §C.0). All constraint *enforcement* machinery is deferred to the
future legalization stage; the model keeps only *conditioning*. No code is modified in this
stage.

---

# Part A — Anatomy of the current repo (`iccad2026contest/diff-model/`)

## A.1 What it is

Official code of **MacroDiff** (Yoon, Jeon, Kang — "A Geometric Diffusion Model for Macro
Placement Generation", DAC'25 Late-Breaking Result). Targets ISPD2005 macro placement:
fixed-dimension macros + zero-size IO pads on a **fixed canvas**.

⚠ The PDF inside the folder (`2605.16451v2.pdf`) is **MacroDiff+** (arXiv, May 2026) — the
authors' newer system with a different architecture (dual-branch HeteroGNN + Transformer U-Net,
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

## C.0 Design philosophy: imitation-first, enforcement deferred

The single organizing principle: **every ground-truth layout already satisfies every constraint**
(zero overlap, exact areas, boundary/grouping/MIB all met — verified on the data). Therefore a
model that gets close to GT is *automatically* close to violation-free. We do not need any
machinery that actively pushes samples toward constraint satisfaction — that is the future
legalization stage's job, and exact satisfaction (abutment, zero-overlap) is measure-zero for a
continuous model anyway. What the model needs instead is **input features that tell it which
constraint role each block plays** (§C.3), so it can imitate the corresponding GT patterns.

The one distinction that matters — and where "defer everything" would over-reach — is
**enforcement vs. conditioning**:

- *Enforcement* = pushing the sample toward feasibility (physics-guided sampling, auxiliary
  overlap/boundary/grouping losses, clean-space polish). GT imitation makes these redundant for
  closeness; **all deferred** (kept only as optional add-ons, §C.9).
- *Conditioning* = using values that are **given in the input**. Preplaced `(x, y, w, h)` and
  fixed-shape `(w, h)` are inputs, not predictions to be made; making the model regress them
  wastes capacity and injects error that corrupts the neighbors GT placed relative to their
  *true* values. Freezing/inpainting them is the same move as image inpainting not regenerating
  known pixels — it strictly *improves* GT-closeness. **Kept.**
- Likewise the `s = ½log(w/h)` output parameterization is *representation*, not enforcement: GT
  always has `w·h = a` exactly, so width/height carry one real degree of freedom; predicting
  `(w, h)` independently adds a spurious dimension the data doesn't have and makes the
  regression harder. **Kept.**

| Mechanism | Verdict |
|---|---|
| Physics-guided sampling, aux physics losses, overlap polish | **Deferred** (§C.9) — enforcement |
| Boundary / grouping handling beyond input features | **Deferred** — imitation covers it |
| MIB `s`-tying | Optional ablation — cheap either way; feature-only + post-snap is the default |
| Preplaced/fixed freezing + inpainting | **Kept** — conditioning on known inputs |
| Area-exact `s` parameterization | **Kept** — output representation matching the data manifold |

## C.1 Diffusion variable: `z_i = (cx_i, cy_i, s_i)` per block

- `(cx, cy)` = block **center**; `s = ½·log(w/h)`; decode `w = √a·e^s`, `h = √a·e^{−s}`.
- **The area hard constraint is satisfied exactly, by construction** — the model cannot emit an
  illegal area. One entire infeasibility mode (score = 10) is eliminated before training starts.
- 3 channels, all near-unit scale (scale `s` by ~2). No rotation variable — spec v10 forbids
  w/h swap for fixed/preplaced blocks, and for soft blocks rotation is absorbed by `s`.

### Constraint handling: conditioning where values are known, features everywhere else

| Constraint | Mechanism | Rationale (per §C.0) |
|---|---|---|
| Preplaced (hard) | freeze all 3 channels at input values; **inpaint** during sampling (re-impose `q_sample(z_known, t)` after every step); mask out of the loss | conditioning on given inputs — exact for free, and neighbors denoise against the *true* positions GT placed them relative to |
| Fixed-shape (hard) | freeze `s` at `½log(w_in/h_in)`; diffuse `(cx, cy)` | same — dims are inputs, not predictions |
| MIB (soft) | **default: input features only** (in-group flag + pairwise `same_mib` attention bias); imitation yields near-identical dims, a trivial post-snap (average `(w,h)` within group) equalizes them exactly. `s`-tying kept as an ablation arm | low stakes either way; feature-only keeps the model simplest |
| Boundary / grouping (soft) | **input features only**: boundary one-hot, cluster flags + pairwise `same_cluster` bias — learned by imitating GT | GT satisfies them, so closeness ⇒ near-satisfaction; exact satisfaction is the legalization stage's job |

Note that inpainting is a **sampling-time-only** mechanism (RePaint works on models trained
without it), so the conditioning-vs-prediction question for preplaced/fixed blocks can be settled
empirically with one trained model, evaluated both ways (§C.8, P2).

After this, the model's entire learning burden is: **where blocks go and what aspect they take**
— i.e., exactly reproducing GT, with constraint roles visible in the input.

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
  and it keeps the door open for the deferred geometry-aware add-ons (§C.9) without retraining.
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
  2-pin nets, to weighted pairwise attraction — which the attention bias already expresses.
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
- **Loss = masked MSE on `x0` (excluding frozen channels), min-SNR-γ weighted — and nothing
  else.** This *is* the "close to ground truth" objective; per §C.0, no physics/constraint terms
  in the critical path. (Auxiliary geometry losses — weighted-HPWL match, smooth overlap,
  boundary distance, cluster cohesion — are catalogued in §C.9 as a deferred ablation, to be
  tried only if the closeness metrics plateau with systematic geometric error.)
- **Augmentation**: dihedral symmetries applied consistently to blocks + terminals + metadata —
  boundary bitmasks remap under flips (x-flip: L↔R), `s → −s` under 90° rotations. Because
  fixed/preplaced dims may not swap, use only the 4 orientation-preserving symmetries (id,
  x-flip, y-flip, 180°) on cases containing them, or remap their frozen dims and skip rotations.
  Block-order permutation is free (set model). **No coordinate jitter** — it destroys abutment.
- **Diffusion setup**: cosine schedule, T=1000 (continuous-time optional), EMA 0.9999, grad clip.
- Compute: ~20M params × 1M samples ⇒ single-GPU-days (env: `~/miniconda3/envs/iccad`).

## C.5 Inference pipeline

Deliberately minimal — a plain conditional sampler plus conditioning and bookkeeping; no
guidance, no polish (those live in §C.9 if ever needed):

```
inputs ─► featurize (frame = terminal bbox center, scale = S)
       ─► freeze preplaced (cx,cy,s) & fixed-shape (s)
       ─► sample N seeds AS ONE BATCH (tiny graphs → one forward pass)
            DDIM / DPM-Solver++, 30–50 steps
            each step:
              x̂0 ← model prediction  →  standard sampler update
              re-impose inpainted channels (preplaced / fixed s) via q_sample
       ─► decode: (w,h) = √a·(e^s, e^{−s})  → soft-block areas exact by construction
       ─► snap: preplaced/fixed exact from input; MIB post-snap (average (w,h) in group)
       ─► rank the N samples (GT unavailable at test time → rank by evaluator-cost proxy:
          weighted HPWL + bbox area + overlap); keep the best
       ─► emit (x, y, w, h)
```

- Best-of-N is *selection*, not enforcement — it needs no gradients and no extra model passes
  beyond the batched sampling, and it directly buys closeness (diffusion is stochastic; some
  seeds land nearer GT).
- Runtime: n≤120, 30–50 steps × N∈[8,32] batched ⇒ well under a second per case on GPU, and
  CPU-viable at reduced N/steps if the judging environment demands it.

## C.6 Evaluation protocol (GT-closeness, no legalizer)

On the 100 validation cases, reported per-n bucket (21–60 / 61–100 / 101–120) and
contest-weighted (`e^{n/12}`):

**Closeness to ground truth (primary at this stage)**
1. Mean/median normalized center displacement `|ĉ − c_GT| / S` per block.
2. Shape error `|ŝ − s_GT|` (equivalently relative dim error) on soft blocks.
3. Fraction of blocks within δ of their GT position (δ = 1%, 2%, 5% of S).

**Quality/feasibility of the raw prediction (secondary — the "closeness ⇒ near-feasibility"
check)**
4. Overlap ratio: Σ pairwise overlap / Σ block area.
5. Weighted-HPWL gap and bbox-area gap vs. baseline (`metrics[6]+metrics[7]`, `metrics[0]`),
   computed with the official `calculate_hpwl_*` / bbox code from `iccad2026_evaluate.py`.
6. Soft-violation counts via the official evaluator (MIB after post-snap must read 0; boundary
   and grouping measured raw and with a small tolerance band, since exact contact is
   measure-zero).
7. **Diagnostic**: violations plotted *against* displacement per case. If the imitation-first
   premise holds, they fall together; a case with low displacement but high violations flags a
   systematic geometric failure worth a §C.9 add-on.

Ablation ladder: backbone alone → + constraint-type features → + inpainting on/off
(sampling-time, same trained model) → + `s` vs. `(w,h)` output head → + MIB `s`-tying on/off →
best-of-N sweep → ± GNN branch → ± terminal tokens.

## C.7 Risks / open questions

- **Imitation ceiling**: a model can be close to GT on average yet leave residual violations
  (overlap slivers, near-miss boundary contact) — that residual is *by design* the legalization
  stage's input. What must be monitored now is that violations shrink proportionally with
  displacement (§C.6 diagnostic 7); if they don't, imitation alone has a systematic geometric
  blind spot and a §C.9 add-on gets promoted.
- **Grouping abutment** is measure-zero for a continuous model — expect near-contact, not exact
  contact. Acceptable at this stage; it barely affects GT-closeness metrics.
- **Whitespace mismatch**: GT is 96–97% packed; diffusion output will be a few % looser. Track
  bbox-area gap; if systematic, this is the first candidate for a §C.9 add-on.
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
| P1 | Backbone diffusion on `(cx,cy,s)` with full constraint-type features (§C.3), plain sampler | displacement and HPWL gap clearly beat a "pin-pull + greedy pack" trivial baseline |
| P2 | Conditioning ablations on the same trained model: inpainting on/off (sampling-time), `s` vs. `(w,h)` head (needs retrain), MIB tying on/off | inpainting + `s` confirmed (or refuted) on displacement metrics; preplaced/fixed exact |
| P3 | Best-of-N + sampler-efficiency sweep (steps ↓, solver choice) | best contest-weighted closeness at acceptable runtime |
| P4 | Architecture ablations (GNN branch, terminal tokens, size) | final architecture locked |
| P5 | *Contingent*: promote a §C.9 add-on only if the §C.6 diagnostic exposes a systematic gap | closeness/feasibility gap closed |

## C.9 Deferred enforcement machinery (out of the critical path)

Catalogued so nothing is lost; none of it is needed to pursue GT-closeness, and all of it can be
added **without retraining** (items 1–2) or with a cheap fine-tune (item 3):

1. **Physics-guided sampling** (MacroDiff+ Alg. 1): at each denoise step, K gradient steps on
   `L = w₁·weightedHPWL + w₂·overlap + w₃·boundary + w₄·clusterGap + w₅·bboxArea` applied to x̂0,
   then re-derive the guided noise. All terms differentiable through `s` (`(w,h) = √a·(e^s,
   e^{−s})`), so guidance can *reshape* blocks, not just move them.
2. **Clean-space polish**: ≤100–300 Adam steps on the same `L` with an overlap-weight ramp — the
   cheap descendant of MacroDiff's 500×10-seed recovery loop.
3. **Auxiliary geometry losses** at train time (weighted-HPWL match, smooth overlap, boundary
   distance, cluster cohesion; kernels adaptable from `utils/score.py` + net weights), weighted
   toward low-noise timesteps.

Promotion rule: an item enters the pipeline only when §C.6 diagnostic 7 shows violations *not*
tracking displacement — i.e., when imitation demonstrably stops being enough.
