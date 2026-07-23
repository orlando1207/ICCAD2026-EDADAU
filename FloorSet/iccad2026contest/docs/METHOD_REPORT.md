# FloorDiff + EGL: Method Report

**Scope.** This report documents the current submission pipeline for the ICCAD 2026
FloorSet Challenge (Lite track): a diffusion model that predicts a near-legal floorplan
from the netlist and constraints, followed by a legalizer that turns that prediction into
a feasible, high-quality solution. It covers design rationale, exact math, and measured
results for both stages. No code changes accompany this report.

**Result at a glance.** Official evaluator, 100 validation cases: **total score 1.1398**
at **1.43 s/case average** runtime (up from a prior LP/MILP-based legalizer's 1.1134 at
7.69 s/case average, 47.7 s worst case). All 100 cases feasible.

**Code map.**

| Stage | Module | Role |
|---|---|---|
| Data / featurization | `floordiff/data.py` | shard loading, latent encode/decode, per-block/global/pairwise features |
| Denoiser | `floordiff/model.py` | `FloorDiffNet`: DiT-style Transformer |
| Diffusion process | `floordiff/diffusion.py` | cosine schedule, x0-prediction loss, DDIM sampler, EMA |
| Training | `floordiff/train.py` | training loop, bucketed batching, checkpointing |
| Legalization | `floordiff/legalizer.py` | EGL: gradient cleanup + constraint-graph assignment + polish + snapping |
| Contest entry point | `floordiff_optimizer.py` | wires sampling + legalization into the `FloorplanOptimizer` API |

Design docs referenced throughout: `docs/superpowers/specs/2026-07-14-diffusion-repo-anatomy-and-contest-model-design.md`
(diffusion design, "Part C"), `docs/superpowers/specs/2026-07-15-floordiff-implementation.md`
(diffusion implementation notes), `docs/superpowers/specs/2026-07-20-egl-legalizer-design.md`
(legalizer design, referencing DREAMPlace/ePlace/NTUplace3).

---

## Part 1 — Problem framing

Each case is a netlist of `n` rectangular blocks (21–120) with:
- a target area per block (soft blocks may deviate ±1%; fixed/preplaced blocks must not deviate at all),
- weighted block-to-block (b2b) and pin-to-block (p2b) connectivity edges,
- placement constraints per block: `fixed` (dimensions locked), `preplaced` (position and
  dimensions locked), `mib_group` (multi-instantiation — same group ⇒ identical `(w,h)`),
  `cluster_group` (same group ⇒ must form one edge-connected component), `boundary_bitmask`
  (must touch the specified bbox side(s): LEFT=1, RIGHT=2, TOP=4, BOTTOM=8, corners are ORs).

The official cost for one case is

```
cost = min( (1 + 0.5·(HPWL_gap + Area_gap)) · exp(2·V_rel) · max(0.7, RuntimeFactor^0.3),  10 - 1e-6 )
     = 10.0                                                                    if infeasible
```

where `HPWL_gap`/`Area_gap` are relative gaps to reference baselines, `V_rel` is the
fraction of soft-constraint violations (boundary/grouping/MIB) out of the maximum
possible, and `RuntimeFactor = your_runtime / median_runtime_across_submissions` for
that case (floor 0.7, exponent 0.3 — a strong incentive to be fast, since a slower but
otherwise-tied submission is strictly worse). Infeasibility (any block overlap, a
soft-block area outside 1%, or a fixed/preplaced deviation) caps the cost at 10. The
per-case costs are averaged with weight `exp((n - n_max)/12)`, so the 120-block cases
dominate the score but the 21-block cases still matter.

This shapes the whole pipeline: **feasibility is non-negotiable** (any overlap or hard-
constraint miss is catastrophic), **speed is scored directly**, and everything else
(wirelength, area, soft violations) trades off continuously. The two-stage design follows
directly: a model trained purely to imitate ground-truth layouts (which are already good
on wirelength/area/soft-constraints) produces a *prediction* that is close but not exactly
legal (small overlaps, soft-constraint misses); a fast, constraint-aware legalizer then
repairs it into something provably feasible while disturbing the prediction as little as
possible.

---

## Part 2 — Diffusion model: predicting a near-legal layout

### 2.1 Design philosophy: imitation-first

The model's *only* training objective is to reproduce the ground-truth layout as closely
as possible — no physics losses, no differentiable constraint penalties, no legalization
machinery baked into training. The rationale (design doc §C.0/§C.4, confirmed by the
implementation notes): the training-set ground truths are themselves synthetic optimized
layouts with **very few soft violations already** (measured: 1–4 boundary misses and at
most one grouping split per case, official cost of GT itself ≈ 1.09–1.13, not 1.0). If the
model learns to imitate GT well, it inherits GT's near-feasibility "for free," and a
downstream legalizer only needs to do local cleanup rather than global constraint
satisfaction from scratch. This was validated online: `--gt-check` (scoring GT against
itself through the same metric pipeline) gives exactly zero displacement/shape-error/
overlap/HPWL-gap/area-gap, confirming the metric and latent-decode pipeline round-trips
losslessly (`decode(featurize(case).z0) == GT` to ≤2e-5).

The promotion rule stated in the design docs — add physics guidance or constraint
enforcement to the diffusion model *only if* violations stop tracking displacement — was
never triggered; violations track displacement closely enough that the constraint
enforcement work was pushed entirely to legalization (Part 3), which is also where it
belongs given the cost formula's sharp feasibility cliff.

### 2.2 Latent representation

Each block `i` is represented by a 3-dimensional latent `z_i = (cx̃, cỹ, s̃)`:

```
s      = 0.5 · log(w / h)                          (log aspect ratio, area-independent)
cx̃     = (cx - ox) / S · COORD_SCALE                (normalized center x)
cỹ     = (cy - oy) / S · COORD_SCALE                (normalized center y)
s̃      = s · S_SCALE
```

where `(cx, cy)` is the block's center in raw coordinates, `(ox, oy)` is the center of the
terminal (pin) bounding box for the case (the frame origin — chosen so the frame is
insensitive to the arbitrary global translation of the layout), `S = sqrt(Σ areas)` is a
per-case length scale, and `COORD_SCALE = 2.9`, `S_SCALE = 3.8` are constants chosen so
each latent channel has roughly unit variance over the training set (measured empirical
std: 0.341 for centers, 0.263 for `s`) — a standard diffusion-model normalization concern,
since the forward noising process assumes roughly-standardized data.

**Decoding is exact-area by construction.** Given a block's known target area `a_i` and
the predicted `s`, the shape is decoded as

```
w = sqrt(a) · exp(s),   h = sqrt(a) · exp(-s)     ⟹   w·h = a  exactly, for any s
```

so the model can never predict a shape with the wrong area — it only predicts an aspect
ratio and a position. This eliminates an entire failure mode (area violations) from the
generative model's responsibility; the legalizer never needs to fix soft-block area either
(it stays exactly on target throughout, since dimensions are only ever reshaped
area-preservingly during legalization — see Part 3).

**Hard-constrained blocks are inpainted, not predicted.** For `fixed` blocks, `s` is
known and frozen to the target aspect ratio. For `preplaced` blocks, all three latent
channels are known and frozen (position and shape both fixed). The diffusion process
never needs to learn these values — they are supplied and held constant throughout
sampling (see 2.4). For blocks in an MIB group, the model still predicts each member's `s`
independently, but decoding post-snaps every group's `s` to a shared value (the frozen
member's value if the group contains one, else the group's mean) — this keeps area exact
per block while forcing shared dimensions across the group by construction, rather than
asking the network to learn exact equality (which point predictions from a stochastic
sampler would never hit exactly anyway).

### 2.3 Model architecture

`FloorDiffNet` (`floordiff/model.py`) is a DiT-style (Diffusion Transformer) encoder,
operating on the `n` blocks as a token sequence with **no positional encoding** — block
identity/order carries no geometric meaning, so the network relies entirely on per-block
features and pairwise attention bias to know which token is which.

- **Token input** (per block): the noisy latent `z_t` (3 dims) concatenated with a
  self-conditioning estimate of `x0` (3 dims, zeroed on the first pass) and 24 static
  features (`N_FEAT`, see below) — projected to `d_model = 256` by a linear layer.
- **Timestep conditioning**: sinusoidal embedding of the diffusion step `t`, passed
  through an MLP.
- **Global conditioning**: 12 case-level features (`N_GLOBAL`, see below), passed through
  a small MLP.
- Both conditioning vectors are summed into a single vector `c` and injected into every
  block via **AdaLN-Zero** (adaptive layer-norm with zero-initialized modulation — the
  DiT paper's conditioning mechanism): each `DiTBlock` predicts six modulation vectors
  (shift/scale/gate for attention and for the MLP) from `c` via a linear layer initialized
  to zero, so at initialization every block is the identity function and training starts
  from a stable, near-linear regime.
- **Connectivity-biased attention** (Graphormer-style): a 3-dimensional pairwise feature
  tensor (`N_PAIR`, see below) is passed through a small MLP to produce a per-head additive
  bias added to the attention logits (`bias_mlp`, output size `n_heads`) — so a strongly
  connected pair of blocks, or two blocks in the same MIB/cluster group, gets a learned
  attention prior toward each other, without needing a hand-designed graph attention
  mechanism.
- **Depth**: 8 `DiTBlock` layers, 8 heads, `d_model = 256`, MLP ratio 4 → **9.8M parameters**
  (checkpoint `floordiff/checkpoints/myrun/last.pt`, trained 200,000 steps).
- **Output head**: predicts `x0-hat` directly (the denoised latent, not the noise
  `epsilon`) — x0-parameterization, which pairs naturally with the min-SNR loss weighting
  below and with self-conditioning.

**Self-conditioning**: during training, with probability 0.5 the model is first run once
(no gradient) to get an estimate of `x0` from the current noisy input, and that estimate is
fed back in as an extra input channel for the real (gradient) forward pass — a
well-established diffusion trick (Chen et al.) that lets the model refine its own guess
rather than predicting from scratch at every step, improving sample quality without
architecture changes.

#### Per-block features (`N_FEAT = 24`)

Static, non-latent information the model conditions on for every block: log-normalized
area, normalized `sqrt(area)`, fixed/preplaced flags, the known `s`/`(w,h)`/center for
frozen blocks (0 if not applicable), the block's 4 boundary-bit indicators, MIB/cluster
group membership flags and (normalized) group sizes, weighted+unweighted b2b degree, p2b
weight sum, the weighted-mean pin-pull position and its dispersion (`disp`), and a
has-pin flag. In short: everything the network needs to know about a block's role and
connectivity that doesn't require attending to other blocks.

#### Global features (`N_GLOBAL = 12`)

Case-level context: normalized block count and pin count, log-scale of `S`, terminal
bbox aspect (`tw/S`, `th/S`), fractions of blocks that are fixed/preplaced/in an MIB
group/in a cluster group/boundary-constrained, edge density (`2E/(n(n-1))`, capped at 1),
and utilization (`mean_area · n / S²`).

#### Pairwise features (`N_PAIR = 3`)

For every ordered pair `(i,j)`: log-weighted b2b connectivity strength (`log1p(w/w̄)`), a
same-MIB-group indicator, a same-cluster-group indicator. These become the attention bias
described above — the mechanism by which the network learns "blocks in the same cluster
group tend to sit adjacent" or "heavily connected blocks tend to sit close" without any
explicit geometric loss term enforcing it.

### 2.4 Diffusion process

**Forward (noising) process**: a cosine noise schedule (Nichol & Dhariwal),
`alphas_bar[t] = cos²((t/T + s)/(1+s) · π/2)` normalized to 1 at `t=0`, `s=0.008`, over
`T=1000` steps. `q_sample(x0, t, noise) = sqrt(ᾱ_t)·x0 + sqrt(1-ᾱ_t)·noise`.

**Training loss**: masked MSE between the model's `x0-hat` and the true `z0`, weighted by
**min-SNR-γ** (Hang et al., γ=5): `weight = min(SNR_t, γ)` where `SNR_t = ᾱ_t/(1-ᾱ_t)`.
This down-weights the loss at very low noise levels (where the raw x0-parameterized MSE
would otherwise dominate training and starve the model of signal from harder, higher-noise
timesteps) — a standard fix for a known pathology of x0-prediction training. Frozen
channels (preplaced position, fixed/preplaced shape) are excluded from the loss via a
per-channel mask, since the model is never asked to predict something it will be told
exactly at sampling time anyway.

**Sampling**: standard DDIM (deterministic, `eta=0`) over 50 steps (subsampled from the
1000-step schedule), starting from Gaussian noise, with self-conditioning carried across
steps. **Inpainting**: at every single sampling step, the frozen channels are overwritten
with `q_sample(z_known, t, fresh_noise)` — the *correctly noised* version of the known
value at the current timestep — before the denoising network sees them, and set to the
exact known value at the final step. This is the standard diffusion inpainting recipe
(used e.g. in RePaint): rather than training a separate conditional model, the known
region is forced to look like a genuine noisy sample from the forward process at every
step, so the network's learned denoising dynamics stay in-distribution for those tokens
while everything else is generated conditioned on them through attention. The predicted
`x0-hat` is clamped to `±Z_CLAMP=3.5` at every step (the empirical latent range, guards
against runaway extrapolation early in sampling when the network's guess is unreliable).

**EMA**: an exponential moving average (decay 0.9999) of the model weights is maintained
throughout training and used for sampling/evaluation — standard variance reduction for
diffusion model inference.

### 2.5 Training data and regime

Training data: ~9,000 shard files (`floorset_lite/worker_*/layouts_*.th`), each holding
112 same-block-count layouts (~1M samples total), loaded via a `BucketBatchSampler` that
draws each batch from a single shard file — since all layouts in a shard share the same
`n`, this gives **zero padding** without needing a variable-length attention mask, and the
sampler applies soft oversampling of large-`n` shards (`exp(n/24)` weight) to match the
contest's own `n`-dependent score weighting. **Augmentation**: 4 orientation-preserving
symmetries (identity, flip-x, flip-y, flip-both) applied per sample, with boundary bitmask
remapped consistently (L↔R on x-flip, T↔B on y-flip) — no 90° rotations, since those would
require also swapping `(w,h)` for fixed/preplaced blocks, which are hard-frozen quantities
that must not change.

Training: AdamW (`lr=1e-4`, weight decay 0.01, linear warmup 1000 steps then cosine decay
to 10% of peak), gradient-norm clipping at 1.0, bf16 mixed precision, 200,000 steps, batch
size 64. Checkpointed every 5,000 steps with a running EMA shadow model. A held-out
quick-validation (mean normalized center displacement on n=60/100/120, 1 seed, 20 DDIM
steps) is logged every 2,000 steps as the primary training-health signal.

### 2.6 Inference: best-of-N seeds

At solve time (`floordiff_optimizer.py`), the model samples **32 independent seeds** in a
single batched DDIM pass (same feature tensors, different noise), each decoded to
`(x,y,w,h)`. A cheap ranking proxy — normalized weighted-HPWL + normalized bbox area + 5×
overlap ratio — picks the **top 6** candidates to hand to the legalizer, which legalizes
each and keeps the best by the (near-exact) official-cost proxy (see 3.6). Sampling all 32
seeds together as one batch is essentially free on GPU (it's just a larger batch
dimension); the cost that matters is downstream legalization, which is why only the
top-6 by the cheap proxy are legalized rather than all 32.

---

## Part 3 — EGL legalizer: from prediction to feasible solution

### 3.1 Design goals and why a rebuild was needed

The prior legalizer (MPCG — an LP/MILP-based minimal-perturbation constraint-graph
approach) achieved good quality (official score 1.1134) but was far too slow: 7.69 s/case
average, up to 47.7 s on the largest cases, because it repeatedly called `scipy`'s
`linprog`/`milp` solvers per axis per candidate per retry rung. Since the contest's
`RuntimeFactor` penalizes slow submissions directly (and a per-case budget realistically
needs to support many candidate seeds), a full rebuild was undertaken based on techniques
from three placement-legalization papers (`FloorSet/reference/`): **DREAMPlace** (GPU-
vectorized placement, "Tetris then Abacus" legalization), **ePlace** (electrostatics-based
density/overlap forces, Nesterov optimization with Lipschitz step prediction), and
**NTUplace3** (log-sum-exp wirelength, look-ahead legalization, macro-shifting via nearest-
legal-position search, critical-chain reshaping). None of these papers solve exactly this
problem (macro-scale floorplanning with MIB/cluster/boundary constraints), so the result,
**EGL** (ePlace-Gradient + Graph legalizer, `floordiff/legalizer.py`), adapts their
*techniques* — force-based overlap removal, minimal-movement assignment, look-ahead
best-of-k — to this problem's specific hard/soft constraint set. **There is no LP or MILP
solver anywhere in EGL** — every stage is either a closed-form vectorized numpy
computation or a small graph/greedy algorithm.

The pipeline is four stages — **G**radient cleanup, **L**egalization (constraint graph),
**P**olish, **S**napping — run per candidate, with the whole pipeline re-run per seed and
the best result kept (Section 3.7).

### 3.2 Preprocessing: stamping hard constraints

Before any of the four stages: preplaced blocks are stamped to their exact target position
and dimensions (immovable throughout); fixed blocks get their exact target dimensions
(position stays free); every MIB group's dimensions are tied to a representative member
(a fixed/preplaced member if one exists, else the first member) — so MIB is satisfied by
construction from this point forward and is never re-broken by any later stage (the set of
"shrinkable" blocks eligible for area-preserving reshaping explicitly excludes MIB members,
for the same reason).

### 3.3 Stage G — gradient-based overlap cleanup

**What the papers suggested vs. what actually worked.** ePlace's textbook approach is
Nesterov-accelerated gradient descent with a Lipschitz-constant step-size estimate (their
Eq. 29) on a penalized objective `wirelength + λ · density_penalty`, with `λ` ramped up
over hundreds of iterations from a randomly-initialized, heavily-overlapping layout. That
recipe was implemented first and **diverged**: on a near-legal diffusion-model warm start
(only small overlaps to begin with), Nesterov's momentum term kept accumulating and
overshooting after pairs separated, since the force model (proportional to the
*perpendicular extent* of an overlap, not its *depth*) doesn't naturally decay to zero as
two blocks approach non-overlap — it only becomes zero the instant they're fully separated,
so momentum could carry blocks well past that point and back into overlap on the other
side, or into large excursions elsewhere. This is a fundamentally different regime from
ePlace's target scenario (thousands of tiny standard cells starting from a randomized dense
initial placement), where a fixed-fraction Nesterov step is a small perturbation and
momentum helps convergence.

**The shipped solution**: keep ePlace's force/charge *structure* (perpendicular-extent
force, area-based mass weighting, Jacobi preconditioning) but replace the integrator with
a **penetration-proportional impulse solver** — a physically-motivated relaxation scheme
closer to constraint-based rigid-body separation than to gradient descent:

- **Overlap impulse**: for every penetrating pair `(i,j)`, resolve the *full* penetration
  depth along the cheaper axis (the axis with less overlap — `min(ox, oy)`), splitting the
  motion between the two blocks by inverse-mass sharing `A_j/(A_i+A_j)` (lighter blocks move
  more — the discrete analog of ePlace's charge-proportional force, and directly evocative
  of DREAMPlace/ePlace's practice of treating block area as electrostatic charge). Preplaced
  blocks have infinite mass (absorb none of the motion). A relaxation factor `omega=0.8` is
  applied (under-relaxation for stability) rather than resolving 100% per iteration, over
  roughly 80 iterations.
- **Spring impulses** (decoupled from the quality-drift cap below, since these need to
  actually close a gap, not just nudge toward one): boundary-constrained blocks within 5%
  of `S` from their required bbox side are pulled toward it at a bounded rate; cluster-group
  proximity pairs (a per-group minimum spanning forest by gap distance, rebuilt every 20
  iterations as the geometry changes) are pulled together along whichever axis has the
  smaller gap.
- **Quality drift**, capped at `0.002·S` per block per iteration (small compared to the
  overlap impulses, so it never destabilizes the separation dynamics): the exact weighted-
  HPWL gradient (with an `|d| ≈ sqrt(d²+γ²)` smoothing, `γ=1e-3·S`, to avoid a
  non-differentiable kink at zero separation), a bbox-area subgradient (softmax-shared
  across near-extreme blocks rather than a hard argmax, so the gradient doesn't jump
  discontinuously between blocks as the extremum changes), and a weak spring back toward
  the original prediction (minimal-perturbation prior). All three are divided by a Jacobi
  preconditioner `(weighted b2b/p2b degree) + area/S²` — ePlace's diagonal-Hessian
  approximation (Eq. 31–33 in their paper), which matters here specifically because every
  block is "macro-sized" with widely varying area and connectivity, exactly the regime
  ePlace's ablation shows preconditioning is necessary for (without it, large/high-degree
  blocks get outsized gradient steps and oscillate).

G is intentionally short — only local cleanup. An ablation (more G iterations, up to 600)
measurably **hurt** quality on a weighted test subset: at ~97% packing utilization, the
large-scale rearrangement needed to fully legalize is the graph stage's job (Section 3.4),
not something a smooth force field should attempt; more G iterations just add drift without
doing useful separation work once penetration is already small.

### 3.4 Stage L — constraint-graph legalization (exact, zero-overlap by construction)

This stage guarantees the *hard* no-overlap constraint by construction — it never needs to
verify overlap-freedom because the algorithm's structure makes overlap impossible.

**Graph construction.** For every pair of blocks, choose a separation axis from the G-phase
geometry: horizontal if the x-gap is at least the y-gap, else vertical. Direction (which
block is the "leader") is fixed by each block's center position along a **globally
consistent per-axis ordering key** (the G-phase centers) — critically, this key is fixed
once and reused through every subsequent repair, so both the H and V graphs stay acyclic
throughout the whole process and a simple sort by key is always a valid topological order.
(A naive re-derivation of direction after each repair step risks introducing cycles; fixing
the key up front avoids that class of bug entirely.)

**Preplaced walls.** Preplaced blocks with a boundary constraint pin the bbox wall on that
side — the bbox edge is *required* to coincide with them, since they cannot move. The lower
wall is the minimum such lower-anchor position; the upper wall the maximum such
upper-anchor extent; either is dropped if it would leave some *other* preplaced block
outside the resulting span (some ground-truth layouts have boundary anchors that are
mutually inconsistent by construction — e.g. two preplaced right-boundary blocks at
slightly different x-extents — in which case at least one boundary violation is
unavoidable even for the ground truth itself, and the wall is set to whichever anchor is
satisfiable).

**Conflict repair.** A greedy per-pair axis/direction choice can produce a chain of
separation constraints between two preplaced anchors that is *longer* than the fixed span
between them — infeasible by construction. This is detected via an anchored longest-path
computation (walk the topological order, propagate `lower_bound[j] = max(lower_bound[j],
lower_bound[i] + size[i])` along graph edges from a preplaced/wall source; if any preplaced
node's bound exceeds its fixed position, that's a conflict). Two remedies, tried in order
(directly following NTUplace3/FLOORIST-style conflict-directed repair):
1. **Reshape** — shrink the chain-axis dimension of the shrinkable (non-MIB, non-fixed,
   non-preplaced) blocks on the conflicting path, growing the perpendicular dimension to
   preserve area exactly, capped at a 25% shrink and a 3.6 aspect-ratio limit per block.
   This is often sufficient on its own: the diffusion model's predicted shapes carry some
   error relative to ground truth, while ground-truth layouts are packed at ~97%
   utilization with near-zero slack, so a chain that "doesn't fit" is frequently a shape
   error rather than a structural conflict.
2. **Flip** — if reshaping can't fully absorb the excess (aspect caps reached), move the
   most promising edge on the conflicting path to the *other* axis instead, choosing the
   edge whose resulting other-axis chain (computed via a size-based head/tail longest-path,
   no repositioning needed to evaluate this) stays shortest. Each edge is flipped at most
   once to guarantee termination.

A second use of the same machinery (`extent_repair`) additionally drives *both* axes'
critical paths toward the diffusion-predicted bbox extents (not just toward preplaced
feasibility) — reshaping first, then flips priced by exactly the same other-axis chain-growth
metric — so the legalized bbox doesn't end up needlessly larger than the model's own
prediction.

**Assignment.** With the (now-consistent) graph fixed, positions are assigned in
topological order by a Tetris-style minimal-movement rule:
`pos_i = clip(target_i, max_{p in predecessors}(pos_p + size_p), U_i)`, where `U_i` is a
backward-propagated upper bound (from preplaced anchors and the bbox wall) ensuring every
successor also has room. This is exactly DREAMPlace's "Tetris-like" legalization principle
(place each block at the position closest to its target that doesn't violate any
already-placed constraint) generalized from 1-D row-packing to a general precedence graph.
Two additional target overrides are applied before this clip: movable boundary-constrained
blocks target the wall directly (so they arrive already touching it, rather than being
snapped afterward against possibly-zero slack), and **contact intents** — cluster
proximity-forest pairs that are already close in the G-phase geometry — get the follower's
target set to *exact abutment* with its leader's position (`leader_pos + leader_size`).
This last point mattered a great deal in practice: sliding blocks into contact *after* the
layout is already packed frequently has zero slack to work with (Section 3.6), whereas
forming the contact *during* assignment, while there is still room to negotiate, succeeds
far more often.

**Wall-feasibility ladder.** If the wall-constrained assignment above still produces
overlap (which can happen when a wall itself turns out to be over-constraining given the
repaired graph), the process retries with progressively weaker guarantees: pinned walls →
unpinned walls (bbox still bounded by `max(critical-path minimum, predicted extent)`, but
not forced to align with preplaced anchors) → a full wall-free re-repair of the graph
followed by another unpinned-wall attempt → no walls at all. Each rung is strictly more
permissive than the last, and the final rung (no walls) is provably always feasible given a
correctly-repaired graph, so this ladder never fails to produce *some* overlap-free layout.

### 3.5 Stage P — polish (Abacus-flavored L1 median sweeps)

Once positions are overlap-free, four sweeps of exact per-block coordinate descent refine
wirelength: each movable block moves, independently per axis, to the **weighted median** of
its connected neighbors' centers/pin positions. Since weighted-HPWL is an L1 (Manhattan)
objective, the weighted median is the *exact* 1-D optimizer for a single block with all
others held fixed — this is the same insight behind Abacus's row-based cluster-collapse
legalization (DREAMPlace's cited technique), specialized here to a general graph rather
than a row of standard cells. Each candidate move is clipped to a **freshly recomputed**
slack interval (the block's feasible range given its current graph predecessors/successors
— recomputed every single block, not cached across the sweep) so a move can never
reintroduce overlap, and additionally clipped to the *entry* bounding box of the polish
stage, so a sweep can only improve wirelength — it can never grow the bbox. (This bbox cap
was added after an early version let polish chase far-away pins and inflate the bbox area
by up to 25%; capping fixed that regression outright.)

### 3.6 Stage S — snapping (profit-gated, official-semantics-exact)

The final stage tries to convert remaining soft-constraint violations (boundary, cluster
grouping) into exact satisfaction — but **only when it's profitable**: a snap is applied
if and only if it is geometrically feasible (checked against a freshly recomputed slack
interval, so it cannot create overlap) *and* its exact cost — computed as the true
weighted-HPWL delta over the moved block's incident nets, not an approximation — is less
than the `2/N_soft` cost reduction the snap would earn in the official formula. This
profit gate matters because a "free" snap that drags a well-connected block far from its
optimal position can cost more in wirelength than the violation it fixes is worth.

- **Cluster abutment** runs first (two passes, since resolving one gap can enable the
  next): for each cluster group, a proximity spanning forest identifies which pairs should
  touch; for each still-open gap, first try sliding a single follower block to exact
  contact. If that's infeasible (the follower is wedged, with zero slack — common in a
  packed layout), fall back to a **rigid component slide**: identify the whole
  touching-connected-component on each side of the gap (via the official union semantics)
  and slide the *entire component* together, with slack computed only against blocks
  *outside* the component (internal relative positions are unaffected, so internal net
  costs don't change at all — only nets crossing the component boundary are priced).
- **Boundary snapping** runs last (so bbox extremes are settled first, and attachments end
  up exact rather than approximately close), also in two passes since one snap can shift
  the bbox extreme and enable another. Movable, boundary-constrained blocks are moved
  exactly onto whichever wall(s) their bitmask requires, gated by the same feasibility +
  profit check per axis (a corner block needs both axes to check out, or neither moves).

Preplaced blocks are never touched by this stage, and no snap is ever permitted to grow the
bounding box.

### 3.7 Verification, retries, and seed selection

**Verification** uses the exact official overlap rule (`both` axes' overlap must exceed
`1e-6` for a pair to count as a violation — touching is legal). If the wall ladder somehow
still leaves a residual (essentially never, given the ladder's last rung is provably safe,
but checked defensively), an epsilon-margin re-assignment is attempted as a final
fallback.

**Anchor-span retry.** After the primary two-round pipeline (a second graph round rebuilt
from the first round's *legal* geometry, since legal geometry tends to produce cleaner axis
choices than the raw G-phase output — kept only if it scores better), if at least two
boundary violations remain *and* the case has preplaced boundary anchors, one additional
retry drives the critical-path extents into the anchors' exact span (a more aggressive,
otherwise-too-costly reshape rung) rather than the diffusion-predicted extent. This rung is
gated behind that specific condition because it was measured to hurt average quality when
applied unconditionally (over-aggressive reshaping on cases that didn't need it) but help
specifically on cases where anchors are otherwise unreachable.

**Proxy cost.** Throughout, candidates are ranked by a proxy that recomputes the *exact*
official cost formula (weighted HPWL gap, bbox area gap, boundary/grouping/MIB violation
counts via the same shapely `unary_union` connectivity check the official evaluator uses)
against the case's official baselines, with `RuntimeFactor` held neutral (since it isn't
knowable locally) — this proxy and the official evaluator's score agree essentially exactly
on any given layout.

**Look-ahead legalization across seeds** (NTUplace3 §2.2's core idea, adapted from
"legalize-inside-the-optimization-loop" to "legalize-each-candidate-independently"):
`legalize_best_of` runs the full G→L→P→S pipeline on each of the diffusion model's top-6
candidate seeds (Section 2.6), in order, stopping early once a candidate's proxy cost drops
below 1.05 (already excellent — no need to burn time on remaining seeds) or a 4-second
per-case wall-clock budget is exhausted, and returns the best-scoring result seen. An
earlier design additionally escalated to a much longer gradient phase on poor-scoring
cases; this was measured, on the cases it was meant to help, to change the result by
exactly zero (confirming the G-phase-length ablation in 3.3 at a different granularity) and
was removed as dead weight.

### 3.8 Measured performance

Offline (legalizing the 100 validation cases' stored top-4 diffusion candidates, so numbers
below reflect the legalizer only): contest-weighted proxy score **1.140**, average
legalization time **0.65 s/case**, maximum **4.1 s**, **100/100 feasible**. Through the
full official evaluator (diffusion sampling + legalization together, 32 seeds/6 top-k):
**total score 1.1398**, average runtime **1.43 s/case**, 100/100 feasible — versus the
prior MPCG legalizer's 1.1134 at 7.69 s/case average (worst case 47.7 s). The small
remaining quality gap versus MPCG is concentrated in a handful of cases with pathological
constraint geometry — most notably one case whose preplaced boundary anchors are mutually
inconsistent (at least one boundary violation is unavoidable, even for ground truth) and a
few cases with unusually scattered cluster-group members. Given the `RuntimeFactor^0.3`
term in the official cost, the ~5.4× runtime reduction is expected to translate into
additional leaderboard score beyond what the local (`RuntimeFactor`-neutral) numbers above
show.

---

## Part 4 — Key design lessons

A few decisions in this pipeline are non-obvious enough, and were expensive enough to
discover, to call out explicitly for future modification:

1. **Textbook Nesterov + gradient-based overlap forces diverge on near-legal warm starts.**
   The ePlace/DREAMPlace recipe assumes a heavily-overlapping random initial placement,
   where momentum accelerates convergence; on an *already close to legal* input, the same
   momentum overshoots. A penetration-proportional impulse scheme (mass-shared, no
   momentum) converges reliably instead. This is the single largest deviation from the
   reference papers' prescribed algorithms.
2. **More optimization iterations can hurt, not just plateau.** Both the gradient phase's
   iteration count and an escalation-on-failure retry were measured, via direct A/B
   comparison on held-out cases, to contribute nothing or negative value beyond a small
   iteration budget — the graph-based legalization stage, not iterative refinement, is
   what's actually doing the constraint-satisfaction work.
3. **Unconstrained local optimization needs an explicit "no regression" cap.** The polish
   stage (exact 1-D median optimization — provably optimal for wirelength in isolation) still
   needed an explicit bbox cap, because "optimal for one objective" is not the same as
   "harmless with respect to another" (it inflated area chasing wirelength until capped).
4. **Contacts should form while there's still slack, not be forced afterward.** Any
   constraint intended to hold in the final layout (here: cluster abutment) is far more
   reliably satisfied by encoding it as a target *during* the constructive assignment phase
   than by trying to achieve it as a post-hoc correction on an already-packed layout, where
   the necessary slack has often already been consumed by other blocks.
5. **Ground truth is not the same as feasible-and-violation-free.** The training labels
   themselves carry a handful of soft violations and an official cost around 1.09–1.13 —
   this sets a natural floor below which chasing further improvement in either the
   diffusion model or the legalizer is chasing label noise rather than real headroom.
