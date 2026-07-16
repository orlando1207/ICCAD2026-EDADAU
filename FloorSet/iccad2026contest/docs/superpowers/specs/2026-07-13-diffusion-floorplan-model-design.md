# Diffusion Floorplan Model — Stage-1 Design (Prediction Quality)

Date: 2026-07-13. Scope: analysis of `diff-model/` (MacroDiff) + contest spec v10, and the
design of a diffusion-based predictor for FloorSet-Lite. **This stage optimizes raw model
prediction quality only** — the legalizer consumes the prediction later. No code is written yet.

---

## 1. What `diff-model/` actually is (and isn't)

The repo is the official code for **MacroDiff (DAC'25 LBR)**. The PDF bundled in the folder
(`2605.16451v2.pdf`) is **MacroDiff+** (arXiv 2605.16451, May 2026) — a *substantially different,
newer system*. The code does **not** implement the bundled paper:

| | MacroDiff (the code) | MacroDiff+ (the PDF) |
|---|---|---|
| What is diffused | **Per-net HPWL scalars** `d` (net-level geometric quantity) | **Macro coordinates** `x` directly |
| Denoiser | Hetero GNN only (`GraphPlacer`: bipartite node↔net GATv2, 3 layers, hidden 128; net head is the only trained output — `cell_fc` is dead weight, see `models/diffusion.py:174-179`) | Dual-branch: Hetero GNN (topology) + Transformer U-Net (geometry), fused via an HPWL-gradient projector `ε = λ_cell·ε_cell + λ_net·(∇x HPWL)ᵀ·ε_net` |
| Position recovery | **Post-hoc**: 500 Adam steps per sample matching sampled `d` + overlap penalty (`test.py:132-158`), 10 seeds per design | Positions come out of the sampler directly |
| Constraint handling | None during sampling | **Physics-Guided Sampling**: at every denoise step, take `x̂0`, run K gradient steps on `w_hpwl·L_hpwl + w_ovl·L_ovl` (two-phase weight schedule), re-derive the guided noise (Alg. 1) |

Code quirks to be aware of (all confirmed by reading):
- `train.py` loads `test.pt` for *both* train and test splits; `design_list` (8 ISPD2005 designs) is hardcoded.
- Loss trains only the net-noise head; the per-cell (coordinate) head is never supervised.
- All normalization is by `max_size` — a **fixed canvas**, which does not exist in our contest.
- Edge features are **pin offsets** (macro pins); FloorSet-Lite nets are center-to-center, so this feature slot is vacant for us (we will repurpose it for net weight).
- Net weights don't exist in ISPD, so MacroDiff ignores weights everywhere (HPWL utils in `utils/score.py` are unweighted) — but contest HPWL is **weighted** (weights span ~60×, verified `5.7e-4 … 3.4e-2` on config_60).
- `sample_ddpm` with `seed=None` leaks ground truth (`d_t = q_sample(d_0, t)` from the true placement); `test.py` avoids this by passing seeds, but the net mask (`d_0 != 0`) still comes from ground truth positions.

### 1.1 Why net-HPWL diffusion is the wrong representation for this contest

1. **All contest nets are 2-pin** (weighted edges `(i, j, w)` for b2b, `(pin, block, w)` for p2b —
   verified against the tensors). Per-net HPWL then degenerates to *pairwise Manhattan distances*.
   Recovering positions from a distance set is a Manhattan distance-geometry problem with
   reflection/rotation/translation ambiguities — strictly harder and lossier than just predicting
   coordinates, which we have 1M ground-truth labels for.
2. **The 500-step Adam recovery is the runtime bottleneck** and its result is only as good as its
   random init; the contest scores runtime explicitly.
3. **No shape output.** Our blocks are soft — the model must predict `(w, h)` too. Net-HPWL
   carries no shape information at all.
4. The bundled MacroDiff+ paper itself abandoned this representation for coordinate diffusion.

**Decision: keep the repo as a reference for its diffusion utilities (schedules, buffers,
q_sample/posterior math) and its differentiable HPWL/overlap kernels (`utils/score.py`, which are
good and reusable after adding weights), but the model we build diffuses coordinates + shapes,
following the MacroDiff+ blueprint adapted to soft blocks and constraints.**

---

## 2. Problem gap: ISPD macro placement → FloorSet-Lite contest

| Aspect | MacroDiff's problem (ISPD/MMS) | Our problem (contest v10) |
|---|---|---|
| Canvas | Fixed die, positions normalized to it | **No fixed outline**; bbox area is an objective; but terminals anchor an absolute frame |
| Block dims | Fixed `(w,h)` per macro | **Soft**: any `(w,h)` with `w·h ∈ a·(1±0.01)` — *hard* constraint; fixed-shape/preplaced dims exact — *hard* |
| Nets | Multi-pin hypernets, pin offsets | Weighted **2-pin** edges, center-to-center; b2b + p2b (terminals) |
| Constraints | None | Hard: no overlap, area tol, fixed dims, preplaced pos+dims. Soft: grouping (abutment-connected), MIB (identical dims), boundary (touch bbox edge/corner) |
| Training data | 8 designs + degree-preserving rewiring augmentation, DREAMPlace pseudo-labels | **1M optimal-by-construction labeled samples** (huge advantage — fully supervised) |
| Scale | 100s–1000s of macros | **21–120 blocks** (tiny graphs; but n=120 has ~7,000 b2b edges ≈ complete graph) |
| Objective | HPWL | `(1+0.5(HPWL_gap+Area_gap))·e^{2V_rel}·max(0.7, RTF^0.3)`, capped at 10⁻ᵉ; infeasible = 10; case weight `e^{n/12}` → **big cases dominate** |

### 2.1 Measured data facts (validation set, configs 21/60/100/120)

- Ground-truth layouts are anchored at origin `(0,0)` and terminals sit on a ring just outside the
  layout bbox (0 interior pins in all 4 checked cases) → **the absolute coordinate frame is
  shared between pins and solution**; the model can and should learn absolute anchoring.
- GT utilization is **96–97%** (near-abutting, ~3% whitespace) — the target distribution is
  extremely compact. GT area error is exactly 0.
- GT aspect ratios span **[0.34, 3.0]** → `s = ½·log(w/h) ∈ [−0.55, 0.55]`. Bounded and
  well-conditioned as a diffusion variable.
- MIB groups have **identical area targets and identical GT dims** (verified) — dimension-tying
  is sound.
- Typical constraint mix (n=100): 10 fixed, 7 preplaced, 1 MIB group (~5 blocks), 3 cluster
  groups (~24 blocks), 30 boundary blocks. Roughly half the blocks carry some constraint.
- p2b weights are constant within a case; b2b weights vary ~60×.
- The scoring weight `e^{n/12}` means n=120 has ~3800× the weight of n=21 → **prediction quality
  at n≈100–120 is what matters**; that's also where b2b graphs are dense.

---

## 3. Core representation (the most consequential decision)

### 3.1 Diffuse `z_i = (cx_i, cy_i, s_i)` per block — 3 channels, not 4

- `(cx, cy)`: block **center** (centers make HPWL and symmetry handling clean).
- `s = ½·log(w/h)`: log-aspect. Decode: `w = √a·e^s`, `h = √a·e^{−s}`.

**Why this beats diffusing `(x, y, w, h)`:**
1. **The area hard constraint is satisfied *by construction*, exactly** — the model cannot emit
   an infeasible area no matter what it predicts. This removes one entire hard-constraint failure
   mode from the learning problem (the one that makes a case score 10.0).
2. One fewer channel, and `s` has near-unit natural scale (multiply by ~2 for unit variance)
   while `w, h` would need per-block scale handling.
3. Downstream guidance/legalization can move `s` freely without ever re-checking area.

### 3.2 Hard constraints handled by conditioning/freezing, not learning

- **Preplaced blocks**: `(cx, cy, s)` fully known from input → **inpainting conditioning**: hold
  their channels at the clean values throughout denoising (replace after every step with
  `q_sample` of the true value, RePaint-style), and mask them out of the training loss. They act
  as fixed spatial context — exactly the role IO pads play in MacroDiff.
- **Fixed-shape blocks**: freeze `s` at `½·log(w_in/h_in)`; diffuse only `(cx, cy)`. (No rotation
  variable: spec v10 requires `w = w_input ∧ h = h_input` exactly, so w/h swap is illegal.)
- **MIB groups**: **tie `s` across the group** (one shared latent per group, or average the
  predictions/noise within the group at every step). Since MIB areas are identical, tied `s` ⇒
  identical `(w, h)` ⇒ `V_mib = 0` by construction. If a group contains a fixed-shape block, the
  whole group inherits its frozen `s`.

With this, the model's only remaining failure modes are: overlap (hard), boundary + grouping
(soft), HPWL/area quality. That's the right place to spend model capacity.

### 3.3 Normalization without a canvas

- Scale: `S = √(Σ a_i)` — deterministic from inputs, ≈ bbox side scale (util ≈ 97%).
  All coordinates (blocks, terminals) divided by `S`; `s` multiplied by ~2.
- Origin: **the terminal frame is real** — keep the input frame's origin (or center on the
  terminal bbox center, applied identically to pins and GT at train time). Do *not* center on the
  per-sample solution (that leaks label information at train time that inference can't reproduce).
- GT centers then live in roughly `[0, 1.1]²` after scaling — well within diffusion range after
  an affine shift to `[−1, 1]`-ish.

---

## 4. Denoiser architecture

For n ≤ 120 blocks + ≤ ~400 terminals, everything is small. Full attention is O((n+r)²) ≈ 530²
— trivially cheap. **We do not need a U-Net.** Recommended architecture, in order of expected
value:

### 4.1 Backbone: constraint- and connectivity-aware Transformer (DiT-style)

- **Tokens** = blocks (+ optionally terminal tokens, see 4.3). Per-token input: noisy
  `(cx, cy, s)_t` concatenated with the static feature vector (§5), projected to `d_model`.
- **Time conditioning**: sinusoidal `t` embedding → MLP → **AdaLN-Zero** modulation per block
  (the modern DiT recipe; strictly better-behaved than MacroDiff's "add t to every layer input").
- **Connectivity-biased attention** (Graphormer-style): add a learned per-head scalar bias
  `b_ij = f(log ŵ_ij, same_cluster_ij, same_mib_ij)` to attention logits, where `ŵ_ij` is the
  quantile-normalized b2b weight. At n=120 the b2b graph is near-complete, so *biased dense
  attention is a better fit than sparse message passing* — the graph structure lives in the
  weights, not the topology.
- **Prediction target: `x̂0` (or v-prediction), not ε**, with min-SNR-γ loss weighting. For
  low-dimensional structured outputs, x0/v-prediction converges faster and enables the physics
  guidance and auxiliary losses to operate on decoded geometry directly.
- **Self-conditioning**: feed the previous `x̂0` estimate as extra input channels (50% dropout at
  train time). Cheap, consistently helps sample quality.
- Size: 8–12 layers, `d_model` 256–384, 8 heads → ~10–25M params. Deliberately small; we have 1M
  samples and want sub-second per-case inference.

### 4.2 Optional second branch: GATv2 over the weighted graph (MacroDiff+ dual-branch)

MacroDiff+'s dual branch exists because *hypernet* topology is awkward for a Transformer. Our
nets are 2-pin, so the attention bias already carries the topology. Keep the GNN branch as an
**ablation** (fuse by residual addition into the Transformer stream every k layers), not a
day-one commitment. MacroDiff+'s gradient-projector fusion `(∇x HPWL)ᵀ·ε_net` degenerates for
2-pin nets into weighted pairwise attraction along net directions — we get the same physics more
directly via the guidance terms (§7) and an HPWL auxiliary loss (§6).

### 4.3 Terminals: summary features first, tokens second

p2b pull is what anchors blocks in the absolute frame. Two mechanisms, in priority order:
1. **Per-block pin-pull summary features** (§5): weighted centroid of connected terminals,
   total pull weight, pull dispersion. Covers most of the signal at zero sequence-length cost.
2. **Terminal tokens** with frozen positions attended via cross-attention (or as ordinary tokens
   that never get denoised) — add if ablation shows the summary is insufficient (e.g., a block
   connected to two opposite-side terminal clusters, where the centroid is misleading).

---

## 5. Feature engineering

### Per-block static features (concatenated to each token)

| Group | Features |
|---|---|
| Size | `log(a_i / ā)`, `√a_i / S` (block scale relative to design and absolute) |
| Fixed-shape | flag; frozen `s` value; `(w, h)/S` |
| Preplaced | flag; `(cx, cy)/S`; `(w, h)/S` (also frozen in the latent) |
| Boundary | 4-bit one-hot decoded from bitmask (L=1, R=2, T=4, B=8; corners = 2 bits set) |
| MIB | in-group flag; group size; (identity is *relational*, see below) |
| Cluster | in-group flag; group size |
| Connectivity | weighted b2b degree; b2b neighbor count; total p2b weight; **pin-pull centroid** `(Σw·p)/(Σw·S)`; pin-pull dispersion (weighted std of connected pin positions) |
| Self-cond | previous `x̂0` estimate `(ĉx, ĉy, ŝ)` |

**Group identity is relational, not categorical.** Group IDs are arbitrary per-sample indices; a
learned embedding of "group 2" is meaningless. Encode groups only via pairwise channels:
`same_mib(i,j)` and `same_cluster(i,j)` flags feeding the attention bias (§4.1). This is
permutation-safe and generalizes.

### Global conditioning (a register/CLS token or AdaLN input)

`n` (block count), `log S`, terminal-bbox `(w, h)/S`, terminal count, total b2b weight, counts of
each constraint type. Block count matters doubly: the scoring weight concentrates on large n, and
density/whitespace statistics shift with n.

### Per-edge (attention-bias) features

`log ŵ_ij` (quantile-normalized b2b weight), `same_cluster`, `same_mib`. Nothing else — pin
offsets don't exist in Lite.

---

## 6. Training design

- **Data**: 1M training samples via `get_training_dataloader` shards. Bucket-batch by `n` to
  minimize padding; mask padded tokens everywhere. Hold out the 100 validation cases strictly for
  evaluation.
- **Sampling of cases**: oversample large `n` to mirror the contest weight `e^{n/12}` — but
  softly (e.g., weight ∝ `e^{n/24}`), since small cases still regularize.
- **Loss** = masked MSE on `x0` (excluding frozen channels: preplaced `(cx,cy,s)`, fixed `s`)
  with min-SNR-γ weighting, plus **auxiliary geometry losses on the decoded prediction**,
  weighted toward low-noise timesteps (they are meaningless at high `t`):
  - weighted HPWL of `x̂0` vs. GT HPWL (use the repo's WA kernels from `utils/score.py`, extended
    with net weights);
  - smooth pairwise overlap penalty (repo's `get_overlap_diff`);
  - boundary penalty: distance from the required side of the block to the corresponding side of
    the *soft bbox* (smooth min/max over block extents);
  - cluster cohesion: mean pairwise center gap within each cluster group beyond abutment
    distance.
  These inject the physics into training, not just inference — the MacroDiff+ ablation shows
  guidance alone is weaker than guidance + topology-aware training.
- **Augmentation**: the 8 dihedral symmetries of the plane, applied consistently to blocks,
  terminals, *and constraint metadata* — boundary bitmasks must be remapped under flips
  (e.g., x-flip swaps LEFT↔RIGHT), `s → −s` under 90° rotations (w/h swap)… **except fixed-shape
  and preplaced blocks, whose dims may not swap** → restrict to the 4 symmetries that preserve
  axis orientation (identity, x-flip, y-flip, 180°) when a case has fixed/preplaced blocks, or
  remap their frozen dims accordingly and skip the rotations. Block-order permutation is free
  (set model).
- **Diffusion setup**: cosine schedule, T=1000 train steps (continuous-time optional), EMA of
  weights (decay 0.9999), gradient clipping. Standard.
- Compute: ~20M params × 1M samples — single-GPU-days scale, fits the lab environment.

---

## 7. Inference pipeline (this stage's deliverable)

```
inputs ─► featurize ─► [freeze preplaced/fixed channels, tie MIB s]
       ─► DDIM/DPM-Solver++ sampler, 30–50 steps
            └─ every step: physics-guided correction (MacroDiff+ Alg. 1)
                 x̂0 ← predict; for k in 1..K (K≈5–10):
                     L = w_hpwl·HPWL_w(x̂0) + w_ovl·Overlap(x̂0)
                       + w_bnd·Boundary(x̂0) + w_grp·ClusterGap(x̂0) + w_bb·BBoxArea(x̂0)
                     x̂0 ← x̂0 − η·∇L        (two-phase schedule: quality → feasibility)
                 ε_guided ← reproject; take DDIM step with ε_guided
       ─► clean-space polish: 100–300 Adam steps on the same L with overlap-weight ramp
       ─► snap: exact w·h = a (rescale w,h by √(a/wh)); MIB dims already tied; preplaced exact
       ─► emit (x, y, w, h) per block  →  [stage 2: legalizer]
```

Key points:
- **All guidance terms are differentiable through `s`** (since `(w,h) = √a·(e^s, e^{−s})`), so
  guidance reshapes blocks, not just moves them — this is the mechanism by which "soft blocks"
  actually get exploited (e.g., flattening a block to slide under a fixed one).
- The polish step replaces MacroDiff's 500×10-seed recovery: our init is already a placement, so
  a short polish suffices; budget matters (runtime factor).
- **Best-of-N**: sample N = 8–32 seeds *as one batch* (tiny graphs — one forward pass), rank by
  the exact evaluator cost proxy (feasibility-weighted), keep the top 1–3 for the legalizer.
- Preplaced inpainting during sampling: after every sampler step, overwrite preplaced channels
  with `q_sample(z_known, t)`.

---

## 8. Evaluation protocol for "prediction power" (pre-legalizer)

Run on the 100 validation cases; report per-n-bucket (21–60 / 61–100 / 101–120) and
contest-weighted aggregates:

1. **Overlap ratio**: Σ pairwise overlap area / Σ block area (target: < 2–3% before legalization
   — analytic-placement-quality inits legalize with small displacement).
2. **Raw HPWL gap** vs. baseline (`metrics[6] + metrics[7]`), computed with the official
   `calculate_hpwl_*` functions.
3. **Raw bbox-area gap** vs. `metrics[0]`.
4. **Soft violations** via the official evaluator's counters (boundary / grouping / MIB — MIB
   should be identically 0 by construction; treat any nonzero as a bug).
5. **End-to-end proxy** (the ultimate metric even in this stage): feed the prediction as the
   initial placement into the existing `analytic_legalizer_v2` skyline pipeline and score with
   `score_harness.py` — measures "how much easier did we make the legalizer's life", which is
   the stated goal. Track legalization displacement (mean |Δcenter|/S) as the coupling metric.

Ablation ladder: (a) backbone alone → (b) + constraint freezing/tying → (c) + aux physics losses
→ (d) + physics-guided sampling → (e) + polish + best-of-N → (f) ± GNN branch ± terminal tokens.

---

## 9. Risks and open questions

- **Grouping (abutment) is the hardest soft constraint for a continuous model** — exact
  edge-sharing is measure-zero in continuous space. Strategy: model learns proximity, guidance's
  ClusterGap pulls to near-contact, legalizer creates exact abutment. If validation shows
  persistent fragmentation, consider cluster *prepacking* (place the group as a macro-block, as
  `analytic_legalizer_v2.prepack_clusters` does) as a conditioning alternative.
- **Whitespace mismatch**: GT is 96–97% utilization; diffusion output will be looser. The
  legalizer compacts, but bbox-area gap in the raw prediction should be monitored — if the model
  systematically over-spreads, add a bbox-area auxiliary loss term (it's in §6 via guidance; may
  need a training-loss counterpart).
- **Frame assumption**: verified on 4 validation cases (pins ring the solution bbox, origin
  shared). Verify on a training shard sample (~1k cases) before committing to absolute-frame
  normalization; fall back to pin-bbox-centered frame if training data differs.
- **Dense graphs at n=120** (~7k edges ≈ complete): fine for biased attention; would be slow for
  edge-list message passing — another reason the Transformer is the primary backbone.
- **Runtime environment for final evaluation** (GPU availability, median-runtime pool) is
  unknown; keep a CPU-viable path (30-step sampler, K=5, N=8 → a few seconds per case even on
  CPU for these sizes).
- **Contest anti-reverse-engineering clause**: learning from the dataset is the intended use;
  we're safely within it, but avoid any generator-inversion shortcuts.

## 10. Phased plan

| Phase | Deliverable | Exit criterion |
|---|---|---|
| P0 | Data pipeline (shard reader → normalized graph samples), eval harness (§8 metrics vs. validation) | metrics reproduce official evaluator on GT (0 gaps) |
| P1 | Backbone diffusion on `(cx, cy, s)`, no constraints, no guidance | raw HPWL/area gap < 30%, overlap < 15% |
| P2 | Constraint conditioning: freezing, inpainting, MIB tying, boundary/cluster features | MIB = 0; preplaced/fixed exact; gaps improve |
| P3 | Aux physics losses + physics-guided sampling + polish | overlap < 3%, boundary/grouping violations < 25% of naive |
| P4 | Best-of-N + ablations (GNN branch, terminal tokens) | best contest-weighted proxy score |
| P5 | Legalizer integration (stage 2, separate design) | beats current best engine on `score_harness.py all` |
