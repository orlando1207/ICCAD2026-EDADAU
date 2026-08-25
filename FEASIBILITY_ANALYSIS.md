# FloorDiff + EGL: the full pipeline, and why its output is now provably feasible

**What this document is.** The end-to-end algorithm of the ICCAD-2026 Problem C (FloorSet Lite)
submission — conditional diffusion sampling followed by constraint-graph legalization — with the
mathematical formulation of every stage, and the proof that the pipeline can no longer emit an
overlapping placement. The new part is **stage E** (§6.3) and the barriers under it (§6.8, §7.3);
everything else is the existing pipeline, documented here because stage E's guarantee is a
statement about the object stage L builds and only makes sense in that context.

Scope: `FloorSet/iccad2026contest/`, branch `feature/diffusion-improvement`.
Written 2026-08-25; every number was measured on this tree (commands in §11).

---

## 0. Summary

Per case the pipeline samples 32 candidate layouts from a conditional diffusion model, decodes
them so that block areas and immutable geometry are exact by construction, ranks them, and
legalizes the top-k concurrently through an eight-stage geometric pipeline, keeping the best by
the official cost formula.

Stage L, the core of legalization, chooses for every pair of blocks *which* of the four
non-overlap half-plane constraints to enforce. That choice — a **relation set** — turns placement
into two independent systems of difference constraints. With preplaced blocks pinned, a relation
set can be **infeasible**, and the old repair (bounded reshape, edge flips, wall relaxation) was
a heuristic that could terminate with the system still infeasible; the pipeline then returned the
overlapping layout anyway. That was the entire residual infeasibility: 88 of 88 overlapping pairs
on the stress kit involved an immutable block.

**Stage E** closes it. It decides feasibility of the relation set exactly (Theorem 1), and when
that fails it applies an **eviction** operator that provably reduces the conflict and terminates
in a feasible relation set (Theorem 2); the existing assignment is then overlap-free by
construction (Lemma 3). Below it sit two unconditional barriers: a shelf construction whose
feasibility does not depend on the input at all, and an exception net in `solve()`.

Measured: stress kit **40/56 → 56/56 feasible**, the 16 previously infeasible cases at mean cost
**1.896** (against **8.467** for a build carrying only the construction floor, and 10.0 before).
On the official 100 cases stage E never fires and per-case costs are **bit-identical** to the
pre-change code.

---

## 1. Problem and scoring

Input per case: `n ∈ [21,120]` blocks with target areas `a_i`; block-to-block edges
`(i,j,w_ij)`; pin-to-block edges `(p,i,w_pi)` with terminal positions `t_p`; a constraint matrix
`C ∈ Z^{n×5}` with columns `[fixed, preplaced, mib_gid, cluster_gid, boundary_bits]`
(`L=1, R=2, T=4, B=8`, corners are ORs). Output: `(x_i, y_i, w_i, h_i)` lower-left, floating
point, **no fixed outline** and **no aspect-ratio limit**.

```
HPWL(x) = Σ_{(i,j)} w_ij(|cx_i − cx_j| + |cy_i − cy_j|)
        + Σ_{(p,i)} w_pi(|t_px − cx_i| + |t_py − cy_i|)          cx_i = x_i + w_i/2
BBox(x) = (max_i(x_i+w_i) − min_i x_i)·(max_i(y_i+h_i) − min_i y_i)
gap_H   = (HPWL − HPWL*)/HPWL*,   gap_A = (BBox − BBox*)/BBox*   (GT baselines, clamped ≥ 0)

Cost = min( (1 + α(gap_H + gap_A))·exp(β·V_rel)·max(0.7, RF^γ),  M − ε )
     = M = 10.0                            if any hard constraint is violated
α = 0.5,  β = 2,  γ = 0.3,  M = 10,  RF = runtime / field median runtime for that case
Total = Σ_i Cost_i·e^{(n_i−n_max)/12} / Σ_i e^{(n_i−n_max)/12}
```

**Hard constraints** (any violation ⇒ `Cost = M`):
```
H1 overlap    ∀i<j: min(ox_ij, oy_ij) ≤ 1e−6,  ox_ij = min(x_i+w_i, x_j+w_j) − max(x_i, x_j)
H2 area       ∀i soft:      |w_i h_i − a_i|/a_i ≤ 0.01       (official test is a strict >)
H3 fixed dims ∀i fixed:     |w_i − w_i*| ≤ 1e−4 ∧ |h_i − h_i*| ≤ 1e−4
H4 preplaced  ∀i preplaced: additionally |x_i − x_i*| ≤ 1e−4 ∧ |y_i − y_i*| ≤ 1e−4
```
H1 requires **both** axes to penetrate by more than `1e−6`: edge contact is legal, which every
abutment and snapping stage below exploits.

**Soft constraints** feed `V_rel`:
```
V_rel  = (V_bnd + V_grp + V_mib)/N_soft ∈ [0,1]
N_soft = |{i : bits_i ≠ 0}| + Σ_q (|M_q| − 1) + Σ_p (|G_p| − 1)
V_bnd  = #{i : bits_i set but block i does not touch the required bbox edge(s)}
V_grp  = Σ_p (#connected components of ⋃_{i∈G_p} box_i − 1)   (a shared edge of positive length
         connects; a bare corner touch does NOT — shapely unary_union semantics)
V_mib  = Σ_q (#distinct (round(w,4), round(h,4)) in M_q − 1)
```

### The two inequalities that dictate the design

**(a) A feasible-but-careless solution is worth nothing.** Feasible cost is capped at `M − ε`,
so the value of feasibility is the *gap* between the rescued cost and 10. A packing that ignores
boundary and grouping constraints produces `V_rel ≈ 0.7` (we measured 0.468–0.779 for exactly
that), so the violation factor alone is `exp(1.4) = 4.06`; with `gap_H ≈ 1.9, gap_A ≈ 1.0` the
quality factor is `2.45`, and

```
2.45 × 4.06 = 9.9 ≈ M
```

"Just make it feasible" is therefore a mathematically empty goal. **Feasibility must come from
perturbing the layout, not replacing it** — §6.3.4 bounds the perturbation damage, and §10
confirms it: 8.467 for reconstruction versus 1.896 for repair.

**(b) Runtime spent on feasibility is nearly free.** `max(0.7, RF^0.3)` means doubling runtime
costs 23% and 5× costs 62%, while rescuing one case gains ~8 absolute cost units — a ~25:1
trade. The project's standing rule ("never buy quality with wall-clock") is about *quality*; it
does not apply to feasibility.

---

## 2. The pipeline at a glance

```
solve(case)                                                   floordiff_optimizer.py
 │
 ├─ 1. featurize        case → (feat, pair, gfeat, z_known, freeze)     §3   data.py
 ├─ 2. diffusion.sample 32 seeds, DDIM 50 steps, hard constraints inpainted  §4  diffusion.py
 ├─ 3. decode × 32      latent → (x,y,w,h); rank by proxy; keep top-k    §5   data.py
 ├─ 4. legalize top-k CONCURRENTLY over the worker pool                  §6   legalizer.py
 │       stamp → G → [ L(+E) → P ] ×2 → S → DP1 → DP2 → DP3 → E-2 → gate
 └─ 5. select best by official-cost proxy, feasible-first                §7   parallel.py
```

| Stage | Role | Hard-constraint status |
|---|---|---|
| 1 featurize | scale-free frame, per-block/pair/global conditioning, inpainting masks | — |
| 2 diffusion | sample block centres + log-aspects | H3/H4 exact by inpainting |
| 3 decode | latent → geometry | **H2 exact by construction**; H3/H4 stamped |
| 4.0 stamp | immutables, area slack, MIB tying | H2/H3/H4 preserved; MIB tied only when compatible |
| 4.1 G | ePlace-lite overlap impulse + quality drift | reduces H1 penetration, does not guarantee |
| 4.2 L | relation set + anchored assignment | **H1 guaranteed iff the relation set is anchor-consistent** |
| 4.3 **E** | eviction until anchor-consistent | **makes that condition always hold** |
| 4.4 P | Abacus L1 median polish inside graph slack | H1 preserved (slack is exact) |
| 4.5 S | profit-gated exact snapping | H1 preserved (slack + bbox guards) |
| 4.6 DP1-3 | grouping/boundary repair: rigid slides, ripple, reshape | H1–H4 preserved (proxy-gated; proxy = 10 if infeasible) |
| 4.7 E-2 | reclaim evicted blocks into free holes | H1 preserved (exact overlap test + proxy gate) |
| 4.8 gate | re-stamp, validate, floor | **absolute**: infeasible ⇒ replaced by the construction |
| 5 select | feasible-first lexicographic key | cannot prefer an infeasible candidate |

`S = sqrt(Σ_i a_i)` throughout; every `*_rel` knob is a fraction of `S`.

---

## 3. Stage 1 — featurization (`data.py`)

**Frame.** `S = sqrt(Σ a_i)` and origin `(o_x, o_y)` = centre of the terminal bounding box. Every
case is mapped into this scale-free, pin-centred frame, so the model never sees absolute units.

**Latent.** Per block the model works with 3 channels, *not* `(x,y,w,h)`:
```
z_i = ( (cx_i − o_x)/S·κ ,  (cy_i − o_y)/S·κ ,  s_i·σ ) ,     s_i = ½ log(w_i/h_i)
κ = COORD_SCALE = 2.9 ,  σ = S_SCALE = 3.8 ,  |x̂_0| clamped to Z_CLAMP = 3.5
```
`κ, σ` were chosen so each channel has ~unit variance over the training data (measured centre
std 0.341, `s` std 0.263).

**Conditioning.** `N_FEAT = 24` per block (log area, `sqrt(a_i)/S`, fixed/preplaced flags, known
`s` and `w,h`/S, known centre, 4 boundary bits, MIB/cluster membership and group size, weighted
degree, degree, pin-weight sum, weighted pin centroid, pin dispersion, has-pin); `N_GLOBAL = 12`
per case; `N_PAIR = 10` per pair used as an additive attention bias — log b2b weight, same-MIB,
same-cluster, 6 one-hot buckets of netlist shortest-path hop distance (Graphormer spatial
encoding: hop = 1, 2, 3, 4, 5 ≤ hop < ∞, disconnected), shared-boundary-side. Changing any of
these widths invalidates every checkpoint.

**Inpainting masks.** Hard constraints enter as *known values*, not loss terms:
```
freeze[i,:] = 1     if i preplaced      (all three channels known)
freeze[i,2] = 1     if i fixed-shape    (the log-aspect channel known)
z_known[i]  = latent encoding of the given (x*,y*,w*,h*) / (w*,h*)
```

---

## 4. Stage 2 — conditional diffusion (`model.py`, `diffusion.py`)

**Denoiser.** `FloorDiffNet`: DiT-style transformer over blocks as tokens, `d_model = 384`,
12 layers, 8 heads, AdaLN-Zero timestep conditioning, QK-norm, SwiGLU FFN, additive pairwise
attention bias from `pair`, x0-parameterisation with self-conditioning (~32.7 M parameters).

**Forward process** (cosine schedule, `s = 0.008`, `T = 1000`):
```
q(z_t | z_0):  z_t = sqrt(ᾱ_t)·z_0 + sqrt(1 − ᾱ_t)·ε ,   ε ~ N(0, I)
```

**Training objective** — min-SNR-γ weighting (`γ = 5`), masked to the *unknown* channels, plus a
connectivity-weighted relative-position term:
```
L = E_{t,ε}[ w_t ‖ M ⊙ (x̂_0(z_t, t, c) − z_0) ‖² ] / E[ w_t ‖M‖² ]
  + λ_edge · Σ_ij W_ij ‖ (ĉ_i − ĉ_j) − (c_i − c_j) ‖² / Σ_ij W_ij
M = ¬freeze ,   w_t = min(SNR_t, γ) ,   SNR_t = ᾱ_t/(1 − ᾱ_t) ,   W = pair[...,0]
```
The mask is why the model never has to *learn* H3/H4: those channels are inputs, not predictions.

**Sampling** — DDIM (`η = 0`, 50 steps) with inpainting re-imposed at every step:
```
for t in t_1 > … > t_K:
    z_t     ← where(freeze, q_sample(z_known, t, ε), z_t)        # re-impose knowns
    x̂_0     ← clamp(model(z_t, t, c), ±Z_CLAMP)
    ε̂       ← (z_t − sqrt(ᾱ_t)·x̂_0)/sqrt(1 − ᾱ_t)
    z_{t−1} ← sqrt(ᾱ_{t−1})·x̂_0 + sqrt(1 − ᾱ_{t−1})·ε̂
return where(freeze, z_known, z_0)
```
All 32 seeds are one batch with identical conditioning and different noise, so sampling costs one
forward pass per step regardless of seed count.

What the model is *not* trained for: routing blocks around a **dense** anchor field. The
preplaced fraction never exceeds 8.9% in the 1M training layouts, while the stress cases reach
62% (§12).

---

## 5. Stage 3 — decode and candidate ranking

**Decode is where H2 becomes structural:**
```
w_i = sqrt(a_i)·exp(s_i) ,   h_i = a_i / w_i        ⇒   w_i·h_i = a_i   exactly
cx_i = z_{i,0}/κ·S + o_x ,   x_i = cx_i − w_i/2     (dually for y)
```
`decode()` then overwrites fixed/preplaced dims with the exact given values, sets preplaced
`(x,y)` exactly, and equalises `s` within each MIB group (the frozen member's `s`, else the group
mean). So a raw model sample already satisfies H2, H3, H4 — only later stages that touch `w, h`
can break them, and §8 tracks each one.

**Ranking.** Legalizing costs far more than sampling, so candidates are pre-ranked by a cheap
triple and only the top-k are legalized:
```
rank_cost(x) = ( HPWL(x) , BBox(x) , ovl(x) ) ,  ovl(x) = Σ_{i<j} area(box_i ∩ box_j) / Σ_i w_i h_i
score_k      = HPWL_k/min_l HPWL_l + BBox_k/min_l BBox_l + 5·ovl_k
```
`topk` defaults to the pool width, because with the pool extra candidates are ~free in wall-clock
(bounded by the slowest) while extra *seeds* cost GPU time — measured: 48 seeds scored worse than
32 on the real score.

---

## 6. Stage 4 — legalization (`legalizer.py`)

### 6.0 Stamp, area slack, MIB tying

```
sol[pre]     ← (x*, y*, w*, h*)                      H4 exact
sol[fix,2:4] ← (w*, h*)                              H3 exact
soft dims    ← predicted dims · sqrt(area_scale)     area_scale = 0.991
shrinkable   ← ¬pre ∧ ¬fix ∧ mib_gid = 0             who may be reshaped later
areas_i      ← w_i·h_i after scaling                 every later reshape preserves exactly this
```
`area_scale = 0.991` exploits the *asymmetry* of H2: the contest permits 1% **under** target and
the official test is a strict `> 0.01`, so 0.9% under is safe and buys ~0.9% of packing slack,
which shows up directly as a smaller `gap_A`.

**MIB tying is applied only when compatible with every hard rule**
(`_tie_compatible_mib_dims`). MIB equality is *soft*, so it must never cost feasibility:
```
group with a frozen member:  all frozen members must agree on (w,h) to 1e−4;
                             the common area must lie in every soft member's 1% band;
                             otherwise leave the group UNTIED.
all-soft group:              common-area interval [ max_i 0.99·a_i , min_i 1.01·a_i ];
                             empty ⇒ UNTIED, else clamp the representative's area into it
                             and keep its aspect ratio.
```
An untied group pays `V_mib`, which is unavoidable: equal dimensions imply equal areas, and areas
differing by more than 2% cannot both sit inside their 1% bands. (This is the fix for the beta
failure — §9.)

### 6.1 Stage G — ePlace-lite overlap cleanup on centres

Dimensions are held fixed; centres `c_i` move by
```
move_i = ω·disp_i + spring_i − P⁻¹∇Q_i ,     ω = 0.8 ,   move[pre] = 0
```

*Overlap as an impulse, not a force.* For each penetrating pair, resolve along the **cheaper**
axis by its full depth, shared by area-mass with `mass[pre] = ∞`:
```
disp_i = Σ_{j : pen(i,j)} sign(c_i − c_j)·depth_ij·m_j/(m_i + m_j)   on axis argmin(ox_ij, oy_ij)
```
For a near-legal warm start this converges geometrically and cannot fly apart, unlike a raw
penalty force. It stops when `max penetration < 1e−4·S`.

*Quality drift.* `∇Q = ∇(α·HPWL) + ∇(γ_A·BBox) + k_a(c − c_0)` with `α = 0.5/HPWL*` and
`γ_A = 0.5/BBox*`, so the gradient is in **cost units**; HPWL smoothed as `d/sqrt(d² + γ²)` with
`γ = 1e−3·S`; the bbox subgradient softmax-shared over the extreme blocks (`τ = 0.02·S`); anchor
spring `k_a = 0.02/S` back to the prediction. Jacobi preconditioner
`P = diag(α·weighted_degree + a_i/S²)` (ePlace Eq. 31), and the whole quality step is clipped to
`0.002·S` per block per iteration — quality may only *drift*, never fight the impulse.

*Springs* (decoupled from the drift cap, closing `0.4·gap` per iteration up to `0.01·S`):
boundary attachment within `0.05·S` of the wall, and cluster-forest contact pulls, where the
forest is the minimum-gap spanning forest of each cluster group, rebuilt every 20 iterations.

### 6.2 Stage L — relation set and anchored assignment

#### 6.2.1 Placement as a disjunctive program

For a pair `{i,j}` non-overlap is
```
x_i + w_i ≤ x_j  ∨  x_j + w_j ≤ x_i  ∨  y_i + h_i ≤ y_j  ∨  y_j + h_j ≤ y_i        (1)
```
A **relation set** `R` selects one disjunct per pair, written as two digraphs: `(i,j) ∈ H` means
`x_j ≥ x_i + w_i`, `(i,j) ∈ V` means `y_j ≥ y_i + h_i`. `build_graph` enumerates all `n(n−1)/2`
pairs, so `|H| + |V| = n(n−1)/2`:
```
pair {i,j} → H iff gap_x(i,j) ≥ gap_y(i,j) ,   leader = argmin over the pair of (key, index)
keyx = G-phase centre x ,   keyy = G-phase centre y
```
Orienting by a **fixed per-axis total order** makes `H` and `V` DAGs and makes `argsort(key)` a
valid topological order for both — an invariant that survives every later edge flip, which is
what lets every stage below use `argsort` instead of re-running a topological sort.

Given `R`, (1) splits into two **independent** systems of difference constraints:
```
(Sx)   x_j − x_i ≥ w_i   ∀(i,j) ∈ H ,      x_i = p_i   ∀i ∈ P   (preplaced anchors)
```
and dually `(Sy)`. Non-anchor variables are otherwise **unbounded** — this contest has no fixed
outline, the structural fact the whole method rests on (§6.2.2).

#### 6.2.2 Theorem 1 — exact feasibility of an anchored relation set

For a directed path `π = (v_0 → … → v_k)` in `H` write `ℓ(π) = Σ_{t<k} w_{v_t}` (every block on
the path except the last) and `d_H(a,b) = max{ℓ(π) : π from a to b}` (`−∞` if none).

> **Theorem 1.** Let `H` be a DAG, `P` the anchor set with pinned values `p_i`, and all
> non-anchor variables free. Then `(Sx)` is feasible **iff**
> ```
> d_H(a,b) ≤ p_b − p_a      for every ordered pair a, b ∈ P.
> ```

*Proof.* Put `(Sx)` in standard difference-constraint form with a source `s`: each equality
`x_i = p_i` becomes arcs `s → i` of weight `p_i` and `i → s` of weight `−p_i`. Such a system is
feasible iff its constraint graph has no positive-weight cycle. Every cycle must use `s`
(deleting `s` leaves `H`, a DAG) and enters it exactly once, so it has the form
`s → a → … → b → s` with weight `p_a + ℓ(π) − p_b`. "No positive cycle" is therefore exactly
`ℓ(π) ≤ p_b − p_a` over all anchor-to-anchor paths. ∎

Two consequences shape the whole design:

* **No fixed outline is what keeps the criterion this small.** An outline would add `0 ≤ x_i` and
  `x_i + w_i ≤ W` for every block — two more arcs per block to and from `s` — admitting positive
  cycles `s → i → … → j → s` that involve no anchor at all (the familiar "this chain does not fit
  in the row"). Without the outline only anchor-to-anchor paths can bind, the anchors are given
  data, and infeasibility is therefore always *local to the anchor field* and always repairable
  by moving non-anchor blocks off those paths. A guarantee is achievable here that would not be
  achievable in a fixed-outline formulation.
* **Walls are extra sources.** The assignment optionally pins the bounding box to a preplaced
  boundary anchor (`wall_lo`), i.e. an arc `s → i` of weight `wall_lo` for every block — which
  creates additional positive cycles. This is why `_find_conflict` takes an `origin` argument, why
  the existing rung ladder (which relaxes walls) is the right response to *wall*-induced
  conflicts, and why it is useless against *anchor-to-anchor* ones.

`_find_conflict(n, edges, pos, size, pre_mask, order, origin)` decides Theorem 1 in one pass: it
relaxes `lb` along the topological order with `lb_i = p_i` for anchors and `lb = origin`
elsewhere, returning the first anchor with `lb_i > p_i` plus its critical path. Cost
`O(n + |E|) = O(n²)`.

#### 6.2.3 Repair: reshape, then flip

Per detected conflict, `repair_graph` (≤200 rounds) tries, in order:

1. **Reshape** shrinkable critical-path blocks along the conflict axis, area exact:
   `size ← size·r`, `other ← a_i/size`, with `r ≥ 1 − 0.25` and aspect `≤ 3.6`.
2. **Flip** the path edge with the largest other-axis gap onto the other graph (each unordered
   pair at most once; **anchor–anchor edges are never flipped** — load-bearing for Lemma 6).

`extent_repair` (≤60 rounds) additionally drives each axis' critical path toward the target
extent: reshape the critical chain first, else flip the path edge whose other-axis chain
`head_o[a] + size_o[a] + tail_o[b]` stays shortest (NTUplace-style rebalancing), accepting only
if the other axis stays within `1.05·max(wmin_o, tgt_o)`.

Both are **incomplete** — bounded rounds, bounded shrink, one flip per pair — which is exactly the
gap stage E fills.

#### 6.2.4 Assignment, and Lemma 3

`assign_axis` computes, in reverse topological order,
```
U_i = min( wall_hi − w_i ,  min_{j ∈ succ(i)} (U_j − w_i − ε) ) ,       U_i = p_i if i ∈ P
```
then forward,
```
i ∈ P :  pos_i = p_i
i ∉ P :  lb_i  = max( wall_lo , max_{p ∈ pred(i)} (pos_p + w_p + ε) )
         pos_i = max( lb_i , min( max(target_i, lb_i), U_i ) )
```
`target_i` is the block's current position, overridden in priority order by
**cluster band < boundary wall < exact abutment with a contact leader** (§6.2.5).

> **Lemma 3.** Suppose the axis system is feasible per Theorem 1, with `wall_lo` counted as an
> extra source (`origin = wall_lo`, or `−∞` on a wall-free rung). Then
> (a) the `U` computed above is the exact componentwise upper bound `U*_i = max{x_i : x feasible}`;
> (b) `lb_i ≤ U_i` and `pos_i ∈ [lb_i, U_i]` for every block;
> (c) every arc is satisfied — the assignment is overlap-free — and anchors keep `pos_i = p_i`.
>
> Conversely, if the system is infeasible the forward pass writes `pos_j = p_j < lb_j` at some
> anchor `j`, violating an arc into `j`. Hence **overlap occurs exactly at arcs into anchors**,
> never between two non-anchors.

*Proof.* A difference-constraint system is max-closed: the componentwise maximum of two feasible
points is feasible, so a non-empty feasible set has a unique componentwise-maximum solution
`x^max` (`+∞` where unbounded), and `U* = x^max`.

(a) `x^max_i = p_i` for anchors, and for a non-anchor `i` it is the largest value consistent with
its outgoing constraints and the wall,
`x^max_i = min(wall_hi − w_i, min_{j ∈ succ(i)}(x^max_j − w_i))` — exactly the backward recursion
above, with the same anchor override. So the computed `U` equals `U*`.

(b) Forward induction along the topological order. For every arc `(p,i)`, `U*_p + w_p ≤ U*_i`:
pick a feasible `x` with `x_p = U*_p` (it exists by definition of the maximum solution), then
`x_i ≥ x_p + w_p` and `x_i ≤ U*_i`. Anchors satisfy `pos_i = p_i = U*_i`. For a non-anchor `i`,
the hypothesis gives `pos_p ≤ U*_p` for every predecessor, so `max_p(pos_p + w_p + ε) ≤ U*_i`;
feasibility with sources at `origin` gives `wall_lo ≤ U*_i`; hence `lb_i ≤ U_i`, and
`max(lb_i, min(·, U_i)) ∈ [lb_i, U_i]`.

(c) Arcs into a non-anchor hold because `pos_j ≥ lb_j ≥ pos_i + w_i + ε` by construction; arcs
into an anchor `j` hold because `lb_j ≤ U*_j = p_j` by (b). For the converse, `pos_j = p_j` is
written unconditionally for anchors, so `max_{i ∈ pred(j)}(pos_i + w_i) > p_j` violates that arc
by exactly that amount; between two non-anchors this cannot happen, by (c). ∎

Lemma 3 is the precise version of "the last rung sacrifices the upper bound": the line
`t = max(t, lb)` in the code keeps every non-anchor relation intact and pays by pushing a block
through the **anchor** downstream — the theory-side prediction of the measured 88/88. It also
gives the converse we need: a relation set satisfying Theorem 1 with `origin = −∞`, assigned with
no walls, **cannot** overlap.

#### 6.2.5 Walls, cluster bands, contact intents

* **Walls.** `wall_lo` is a preplaced L/B boundary anchor; `wall_hi` is pinned to a preplaced R/T
  anchor when reachable, else `x_0 + max(wmin − x_0, tgt_ext)`. Unsatisfiable anchors (one sitting
  short of the wall it must touch — the GT has a few) are detected and skipped.
* **Cluster bands** (`cluster_align`). For each cluster group, on the axis *perpendicular* to the
  group's spread, all members target the median band centre `band − size_i/2`. This is the fix
  that matters for grouping: 70 of 71 violating groups were components already touching on one
  axis — 68 at gap exactly 0 — but with disjoint intervals on the other axis, so they met at a
  corner and shapely still saw two pieces. Doing it *at assignment time*, while slack still
  exists, is what works; 98% of the residue is slack-limited afterwards.
* **Contact intents.** Cluster-forest pairs that are close in the entry geometry become
  `{follower: leader}` maps per axis; the follower targets `pos_leader + size_leader`, i.e. exact
  abutment forms *during* assignment rather than being slid into place afterwards.

#### 6.2.6 The rung ladder

```
rung 1   walls pinned to the boundary anchors
rung 2   walls unpinned                             } fix WALL-induced conflicts
rung 3   wall-free re-repair, then unpinned walls    } (Theorem 1 with a finite origin)
rung 4   no walls
rung 5   STAGE E  (§6.3)                            } fixes ANCHOR-to-ANCHOR conflicts
```
The ordering is deliberate: rungs 1–4 preserve quality and address the wall class (measured: 5 of
the 100 official cases, all recovered); stage E is the only thing that can address the anchor
class, and it costs quality, so it runs last. Two full L-rounds are run (the second rebuilds the
relation set from legal geometry, giving cleaner axis choices), plus an aggressive `span=True`
retry when ≥2 boundary bits on anchored blocks remain violated.

### 6.3 Stage E — eviction to anchor-consistency (the new stage)

#### 6.3.1 The operator

Let `E` be the evicted set (initially empty) and `m ∉ P ∪ E`. `evict(R, m)` rebuilds the relation
set from the **core** relations (between blocks not in `E ∪ {m}`, kept verbatim) plus, for
`E' = E ∪ {m}`:
```
∀ core k, ∀ e ∈ E' :   (k, e) ∈ V        every evicted block follows every core block
∀ e, e' ∈ E' :         {e, e'} ∈ H, oriented by (keyx, index)
keyy_e ← max(keyy) + rank(e)             so argsort(keyy) stays a topological order for V
```
Geometrically the evicted blocks form one shelf above the whole core, side by side in predicted
`x` order. Structurally, that is the only property needed:

> **Lemma 4 (path invisibility).** After `evict`, no anchor-to-anchor path in `H` or `V` passes
> through any evicted block.

*Proof.* Anchors are never evicted, so `P ∩ E = ∅`. In `V` an evicted block has only in-arcs
(`core → evicted`), so it cannot be interior to a path, and it cannot be an endpoint because
endpoints are anchors. In `H` its only arcs join other evicted blocks; a path reaching it must
have started inside `E` (not anchors), and leaving it towards an anchor would need an arc to a
core block, of which there are none. ∎

> **Lemma 5 (monotonicity).** `d_{H'}(a,b) ≤ d_H(a,b)` and `d_{V'}(a,b) ≤ d_V(a,b)` for all
> `a,b ∈ P`: eviction never creates a new conflict.

*Proof.* Core-to-core arcs are preserved verbatim; arcs incident to `m` are removed from their old
axis, and deleting arcs cannot lengthen a longest path. Every added arc is incident to an evicted
block and by Lemma 4 lies on no anchor-to-anchor path, so it contributes to no `d(a,b)`. ∎

> **Lemma 6 (anchor–anchor arcs are always satisfied).** If the anchors are pairwise
> non-overlapping, every arc of `H` (resp. `V`) between two anchors satisfies `p_a + w_a ≤ p_b`.

*Proof.* Non-overlap of `a,b` gives `gap_x ≥ 0` or `gap_y ≥ 0`, so `max(gap_x, gap_y) ≥ 0`.
`build_graph` assigns the pair to the axis attaining that maximum, on which the two intervals are
disjoint; ordering by interval centre then coincides with ordering by interval, giving
`p_a + w_a ≤ p_b`. `repair_graph` and `extent_repair` both refuse to flip an edge whose endpoints
are both anchors, so the property is preserved. ∎

#### 6.3.2 Theorem 2 — termination and completeness

> **Theorem 2.** Assume the anchors are pairwise non-overlapping. The loop
> ```
> while ∃ conflict (Theorem 1 violated on H with origin −∞, or on V):
>     pick a non-anchor block m on the reported critical path, m ∉ E
>     R ← evict(R, m)
> ```
> terminates after at most `n − |P|` iterations in a relation set satisfying Theorem 1 on both
> axes. Assigned with no walls, it yields an overlap-free placement in which every anchor keeps
> its exact position.

*Proof.* Each iteration adds one block to `E`, bounding the iteration count. The loop cannot get
stuck: a critical path realising `d_H(a,b) > p_b − p_a` cannot consist of anchors only, since by
Lemma 6 every anchor–anchor arc satisfies `p_a + w_a ≤ p_b` and a chain of these telescopes to
`ℓ(π) ≤ p_b − p_a`, contradicting the violation; so its interior contains a non-anchor block, and
by Lemma 4 no already-evicted block lies on an anchor-to-anchor path, so that block is not in `E`.
Progress: by Lemma 5 no new conflict appears, and `m` is removed from every anchor-to-anchor path.
In the extreme case every non-anchor block is evicted, `H` retains only arcs inside `E` and `V`
only `core → E` arcs, so `d_H(a,b) = −∞` for distinct anchors except direct anchor–anchor arcs,
which are satisfied by Lemma 6 — the terminal state is feasible. Feasibility plus Lemma 3 (with
`wall_lo = −∞`, so its proviso is vacuous) gives an overlap-free assignment; anchors are written
as `pos_i = p_i`, so H4 stays exact. ∎

The degenerate terminal state is exactly the shelf construction of §6.8 — stage E and the floor
are the same algorithm at two extremes, with quality degrading continuously in `|E|` between them.

#### 6.3.3 Which block to evict

Any choice terminates; the choice only affects quality. `_eviction_score` prefers blocks whose
removal from the core damages the objective least:
```
score_i = 4·[bits_i ≠ 0] + 2·[clu_i ≠ 0] + 1·[mib_i ≠ 0] + 2·wdeg_i/max wdeg + area_i/max area
```
— boundary membership first (an evicted block almost surely loses its wall), then cluster and MIB
membership, then connectivity (HPWL damage) and area (shelf height). The victim is the `argmin`
over the critical path, ties broken by index, so the stage is deterministic.

#### 6.3.4 Why eviction beats reconstruction, quantitatively

Let `k = |E|`. Every non-evicted block keeps its relations, hence its geometry up to the
compaction the assignment performs, so the soft-violation damage is bounded by what the evicted
blocks themselves carry:
```
ΔV_bnd ≤ k                                         each evicted block may lose its wall
ΔV_grp ≤ Σ_{p : G_p ∩ E ≠ ∅} (|G_p| − 1)
ΔV_mib = 0                                         MIB is about dimensions; eviction moves
⇒ ΔV_rel ≤ ( k + Σ_{p : G_p ∩ E ≠ ∅} (|G_p| − 1) ) / N_soft
```
**Linear in `k`**, with `N_soft` large (`n_bnd` alone is 30–40% of the blocks). A reconstruction
instead pays `V_rel ≈ 0.6–0.8` *independently of the instance*, because it abandons every boundary
attachment and every group adjacency at once. Linear-in-`k` damage versus a constant catastrophe,
through `exp(2·V_rel)`, is the whole argument — and the measurement agrees: `k = 8` of 40 blocks
gives cost 1.9, reconstruction gives 8.5.

#### 6.3.5 Complexity

Per iteration: one `_find_conflict` per axis (`O(n²)` each) plus an `O(n²)` relation-set rebuild.
With `evict_max = 64` and `n ≤ 120` the worst case is ~`10⁶` elementary operations; measured
0.14 s per candidate on the `n = 40` stress cases against 0.06 s when stage E does not fire.
Candidates are legalized concurrently, so per-case wall-clock is bounded by the slowest candidate,
not the sum.

### 6.4 Stage P — Abacus-style polish

Per axis sweep (4 sweeps), each block moves to the **weighted L1 median** of its connected
coordinates, clipped to its *freshly recomputed* graph slack and to the entry bbox:
```
lo_i, hi_i = max_{p ∈ pred(i)}(pos_p + size_p) ,  min_{s ∈ succ(i)} pos_s − size_i
targets    = { pos_j + size_j/2 − size_i/2 : (j, w_j) ∈ nbr(i) } ∪ { pin coordinate }
pos_i      ← clip( weighted_median(targets, weights),
                   max(lo_i, bbox_lo), min(hi_i, bbox_hi − size_i) )
```
The slack interval is an **exact** feasibility bound: every pair lives in exactly one of the two
graphs, so a move inside `[lo_i, hi_i]` cannot create overlap with anything. Clipping to the entry
bbox means polish can only improve HPWL, never `gap_A`.

### 6.5 Stage S — profit-gated exact snapping

A snap is applied iff it is inside the slack interval **and** its exact HPWL delta costs less than
the violation it removes:
```
apply iff  lo_i ≤ target ≤ hi_i   ∧   α·ΔHPWL(i → target)  <  β/N_soft = 2/N_soft
```
`2/N_soft` is the first-order marginal value of one soft violation in `exp(β·V_rel)`. Cluster
abutment runs first (two passes; single-block slides, else a rigid slide of the whole touching
component under the intersection of its members' slacks), boundary sides last so attachments are
exact at exit.

### 6.6 Stages DP-1…DP-3 — grouping and boundary repair

All three are bounded and all are gated on `proxy_cost`, which returns 10.0 for an infeasible
layout — so no accepted move can break H1–H4.

* **DP-1 `cluster_perp_align`** — gives corner-touching cluster components a real shared edge by
  shifting **within graph slack** rather than free space. A block shifted along axis `p` can only
  collide with its `p`-graph neighbours (every other pair is held apart by the other axis' arc),
  so the slack interval is an exact feasibility bound. This is why it succeeds where free-space
  translation fails: at ~97% utilisation, `max_penetration` rejects nearly every free-space move.
  Movers prefer blocks with no boundary bit on the shift axis. `cluster_repair` additionally
  merges rigid touching components (≤48 moves, drag ≤ `0.30·S`).
* **DP-2 `cluster_ripple_repair`** — allows a component translation to overlap *transiently*, then
  projects it through a freshly rebuilt all-pair separation graph; only a legal, proxy-improving
  projection commits. Caps: 32 trials, 4 moves, ≤24 blocks, `0.04 s`.
* **DP-3 `shape_detail_repair`** — closes residual group gaps by **elongating** a soft block
  toward contact (area exact) and contracts a bbox side beyond a preplaced boundary anchor by
  shrinking a critical path. Caps: 16 trials, growth/shrink ≤10%, aspect ≤3.6, ≤48 blocks,
  `0.025 s`. Never reshapes `¬shrinkable` blocks, so H2/H3 hold.

### 6.7 Stage E-2 — reclaiming evicted blocks

Eviction parks blocks in a shelf, which is feasible but pays `gap_A` and `gap_H`.
`hole_relocate` tries to bring each one back.

**Candidates**: the corner grid `{x_k^0, x_k^1}_k × {y_k^0, y_k^1}_k` of the other blocks. For the
area term this loses nothing (the classical argument that a single rectangle among fixed ones can
be slid until it touches two others); for the HPWL term it is a restriction, since the L1 optimum
along an axis can also sit at a neighbour's centre. That is acceptable because acceptance is
decided by the true proxy, not by the grid.

**Exact free-position test, cheaply.** A candidate `(x,y)` collides with block `k` iff it
penetrates on *both* axes, and each axis' penetration depends on one coordinate only:
```
free(x,y) ⟺ ¬∃k : ox_k(x) > ε ∧ oy_k(y) > ε
          ⟺ (OX·OYᵀ)[a,b] = 0 ,   OX[a,k] = [ox_k(x_a) > ε] ,  OY[b,k] = [oy_k(y_b) > ε]
```
so the whole `A×B` grid is one boolean matrix product — `O(AK + BK + AB)` instead of materialising
an `A×B×K` cube. At `n = 120` that is 0.5 ms per block instead of ~110 MB of temporaries per
worker. Verified against a brute-force triple loop on 200 random layouts.

**Acceptance.** Candidates are ranked by a cheap surrogate (exact weighted-HPWL delta plus bbox
growth of the union); only the best few are scored with the real `proxy_cost`. A move commits only
if it is exactly overlap-free **and** strictly improves the proxy, so the stage can never turn a
feasible layout infeasible, and it refuses to touch an already-infeasible one. Measured on the two
smallest failing cases: 1.928 → 1.801 and 2.169 → 1.907.

### 6.8 Exit gate and the construction floor

```
sol[pre] ← target ;  sol[fix,2:4] ← target      detailed placement may not perturb immutables
hard = hard_feasibility(sol, case)              official-equivalent H1-H4 validator
if not hard.feasible and guaranteed_floor:
    alt = guaranteed_construction(pred, case)
    if hard_feasibility(alt, case).feasible:  sol ← alt
```
`hard_feasibility` reproduces H1–H4 including the `1e−6` two-axis rule, the strict `> 0.01` area
rule and the `1e−4` immutable tolerance, plus non-finite/non-positive guards, and reports the
offending pairs and blocks.

**`guaranteed_construction`** — anchors keep their exact geometry; every movable block is
shelf-packed in the half-plane above `y_top = max_{i∈P}(p_i^y + h_i)` in predicted reading order,
soft dims taking the predicted aspect at area `area_scale·a_i`, fixed dims exact. Overlap-freedom
is structural: shelf rows are disjoint in `y`, blocks within a row disjoint in `x`, and the whole
shelf sits above every anchor. `O(n log n)`, no search, no dependence on prediction quality. It is
adopted only if it validates, so it can never make things worse. On its own it is worth ~8.5
against the 10.0 penalty (§1a) — its job is not quality but bounding the damage from failure modes
we have not imagined.

---

## 7. Stage 5 — selection, parallelism, and the outer net

### 7.1 Feasible-first selection

```
key(info) = (0, proxy_cost, seed_rank)                                   if feasible
          = (1, total_viol, ovl, area, dim, numeric, proxy, seed_rank)    otherwise
```
`proxy_cost` reimplements the official formula (runtime-neutral, 10.0 if hard-invalid) so
candidate choice optimises the real objective. Lexicographic ordering means quality can never
outrank feasibility.

### 7.2 Parallelism

Runtime is wall-clock around `solve()` and the top-k candidates are independent, so legalizing
them concurrently turns per-case cost from `Σ_k t_k` into `~max_k t_k` at identical quality
(measured 3.6–7.3× on the heavy cases). Two constraints are load-bearing: **spawn, never fork**
(forking after the CUDA context exists deadlocks the workers, and a module-level fork pool dies
under the evaluator's import machinery), and **the worker function must live in a real importable
module** (the evaluator imports the submission as synthetic module `optimizer_module`, which a
child cannot import). The pool and the checkpoint are built in `__init__`, which the evaluator
does not bill to any case.

### 7.3 The outer net

`MyOptimizer.solve` wraps `_solve` and answers any exception — in featurization, sampling, the
pool, or legalization — with `_rescue`, which builds the same shelf construction from square soft
blocks and needs no model at all. It also validates the returned candidate with `hard_feasibility`
and substitutes the construction if it somehow fails.

> **Invariant.** `MyOptimizer.solve` returns a hard-feasible placement whenever the instance
> admits one (anchors pairwise non-overlapping, each soft area realizable), and never raises.

---

## 8. Where each hard constraint is guaranteed

| | Mechanism | Can any stage break it? |
|---|---|---|
| **H2** area | `decode`: `w = sqrt(a)·exp(s), h = a/w` ⇒ `wh = a` exactly; `area_scale = 0.991`; every reshape sets `other ← a_i/size` | reshapes exclude `¬shrinkable`; MIB tying checks the 1% band; DP stages proxy-gated. Measured 0 violations in 56 stress cases |
| **H3** fixed dims | stamped at 6.0, re-stamped at 6.8; `shrinkable` excludes fixed | no. 0 violations measured |
| **H4** preplaced | stamped, pinned by `assign_axis`, `move[pre] = 0` in G, every DP mover skips `pre_mask`, re-stamped at 6.8 | no. 0 violations measured |
| **H1** overlap | stage L is overlap-free **iff** the relation set is anchor-consistent (Lemma 3); stage E makes it so (Theorem 2); P/S/DP-1 move only inside exact graph slack; DP-2/DP-3/E-2 proxy-gated; 6.8 validates and can substitute the construction; 7.3 catches exceptions | **this was the gap** — §9 |

Two independent barriers now stand behind H1: stage E (quality-preserving, terminating) and the
construction floor (unconditional). No code path returns overlap.

---

## 9. Why it used to fail (compressed)

On the stress kit — 56 cases derived from real validation ground truth with extra
preplaced/fixed/boundary/MIB constraints that the GT layout provably satisfies, so an infeasible
result is always the solver's fault — the pre-change code was 40/56 feasible with:

* **0 area, 0 dimension violations**: H2/H3/H4 were already sound;
* **88 of 88 overlapping pairs involving an immutable block** — exactly Lemma 3's converse;
* feasibility 24/24 below 25% preplaced, 16/32 above;
* on failing cases all 3 `repair_graph` calls and all 4 rungs reporting failure, for all 6
  candidate seeds — systematic, so best-of-k could not rescue it;
* penetration up to `0.14·S` — whole blocks, not numerical noise.

`repair_failures = 3/3` is the diagnosis in one number: the repair loop exhausted its bounded
remedies (§6.2.3) with an anchor's longest-path lower bound still above its pinned position, and
`legal_round` returned the layout anyway.

The earlier and *separate* beta failure (29/100 feasible on the hidden proxy set) was blind MIB
dimension tying, fixed in `5fd36e6`: 93% of training-distribution layouts (390/420) contain an MIB
group whose members' target areas differ by more than 1% — median worst intra-group relative error
1.22 — so tying every member to a representative broke H2 almost everywhere, while the official
validation set contains **zero** such groups, which is why local runs were 100/100 feasible. §6.0
is the fix.

---

## 10. Measured behaviour

### Stress kit (56 cases) — the feasibility reference

End-to-end through the official evaluator, identical conditions, only the flags differ:

| build | feasible | avg cost | total | the 16 previously-failing cases |
|---|---|---|---|---|
| before (`3eaaf8c`) | 40/56 | — | — | 16 × 10.0 |
| construction floor only (stage E off) | 56/56 | 3.459 | 3.761 | mean **8.467**, median 9.006 |
| **stage E + reclaim** | **56/56** | **1.431** | **1.462** | mean **1.896**, median 1.860, max 2.522 |

Zero overlap, zero area, zero dimension violations. Per case on the failing set:
`10.0 → 1.24 … 2.52`. Six already-feasible cases improved slightly (e.g. 1.779 → 1.734) from the
reclaim pass. The middle row is the empirical form of §1a: a rescue that ignores the soft
constraints recovers almost none of the loss.

Caveat, since the kit is self-generated: its absolute costs are not comparable with the official
set (baselines are recomputed from the GT polygons). **Feasibility is the signal here**; the
quality column is ordinal.

### Official 100 — the quality reference

* Stage E **never fires**: 0 evictions, 0 floor uses, 0 reclaim moves across all 100 cases. Five
  cases hit intermediate rung failures, which the pre-existing wall ladder recovers from — the
  wall class of §6.2.2, not the anchor class.
* Deterministic A/B (1 candidate per case, flags on vs off): **0 cases with any cost difference**,
  identical mean cost to 16 significant digits, identical mean legalize time (0.241 s). The change
  is provably inert where the pipeline already worked.
* Pool run at the default 24 workers: **100/100 feasible, runtime-neutral total 1.0569**, avg cost
  1.0475, reproduced exactly across two independent runs (1.0732 at 12 workers, since fewer
  workers means fewer legalized candidates).

Two caveats on that headline:

* **Wall-clock from this session is unusable.** The box carried another user's job throughout
  (load average 15–24, GPU 0 pinned at 100%), inflating runtime to ~1.2 s/case against the
  0.53 s/case reference. The runtime-neutral total and the deterministic A/B are the
  load-independent facts; re-measure runtime on an idle machine before any runtime claim.
* **1.0569 vs the 1.0521 in `tune_g100_results.json` is not this change.** 99 of 100 cases differ
  from that older run, roughly half better and half worse — the spread from the legalizer commits
  landed in between (`reshape`, `hard repair 2.0`). The A/B above shows this change contributes
  exactly zero to it.

---

## 11. Implementation map and verification

| Piece | Symbol |
|---|---|
| featurization, latent, decode | `data.py`: `featurize`, `decode`, `COORD_SCALE`, `S_SCALE`, `Z_CLAMP` |
| model / diffusion | `model.py`: `FloorDiffNet`, `ModelConfig`; `diffusion.py`: `FloorDiffusion.loss`, `.sample`, `EMA` |
| ranking | `sample.py`: `rank_cost`; `evaluate.py`: `weighted_hpwl`, `bbox_area`, `overlap_ratio` |
| stamp / MIB | `_tie_compatible_mib_dims`, `DEFAULT_CFG['area_scale']` |
| stage G | `gradient_phase` |
| stage L | `build_graph`, `repair_graph`, `extent_repair`, `reshape_chain`, `min_extent`, `assign_axis`, `legal_round` |
| Theorem 1 test | `_find_conflict(..., origin=-inf)` |
| **stage E** | `evict_for_consistency`, `_rebuild_evicted`, `_eviction_score` (rung 5 in `legal_round`) |
| stage P / S | `polish`, `snap_soft` |
| DP-1/2/3 | `cluster_perp_align`, `cluster_repair`, `cluster_ripple_repair`, `shape_detail_repair` |
| stage E-2 | `hole_relocate`, `_free_positions` |
| exit gate / floor | `hard_feasibility`, `proxy_cost`, `guaranteed_construction` |
| selection / pool | `_selection_key`; `parallel.py`: `make_pool`, `legalize_parallel` |
| outer net | `floordiff_optimizer.py`: `solve` → `_solve` / `_rescue` |
| knobs | `evict_repair`, `evict_max=64`, `reclaim`, `reclaim_trials=24`, `reclaim_probe=8`, `reclaim_budget_s=0.05`, `guaranteed_floor` |

`floordiff/test_evict_completeness.py` — 9 tests asserting the claims of §6.2–§6.8 rather than
outputs. The fixture is a **trapped corridor**: two anchors 2 units apart with three 3×3 soft
blocks between them, where no reshape (aspect cap) and no edge flip can help, so only eviction can
restore consistency.

* the conflict is detected on the initial relation set (Theorem 1 fires);
* after eviction: no conflict on either axis, both graphs are DAGs, and the returned orders are
  valid topological orders (Lemmas 4–6, Theorem 2);
* the assignment is then overlap-free per the *official* `check_overlap`, and anchors have not
  moved (Lemma 3 + H4);
* `legalize_case` is feasible on the fixture, and still feasible with `evict_repair=False` — i.e.
  the floor catches what stage E is not allowed to (§6.8);
* `guaranteed_construction` is feasible on 25 random anchor fields and preserves immutable
  geometry exactly;
* `hole_relocate` never worsens the proxy, never breaks feasibility, and refuses an infeasible
  input.

```bash
cd FloorSet/iccad2026contest                    # conda activate iccad
python -m unittest floordiff.test_evict_completeness floordiff.test_hard_feasibility \
                   floordiff.test_legalizer_reshape floordiff.test_legalizer_ripple   # 22 tests
python iccad2026_evaluate.py --evaluate floordiff_optimizer.py -o run.json
python score_rf.py run.json                     # the number that matters (real runtime factor)
python iccad2026_evaluate.py --data-path ../../cadc1111/floorset_testkit \
                            --evaluate floordiff_optimizer.py           # feasibility reference
python tools/val_evict.py /tmp/on.json cuda:0   # does stage E fire on the official 100?
python tools/diag_testkit.py ../../cadc1111/floorset_testkit            # per-case H1..H4
```
`tools/README.md` documents the diagnostic scripts behind every number in §10.

---

## 12. Limits and what to do next

* **Rescued cases cost ~1.9 against ~1.25 for cases that never needed a rescue.** A guarantee is
  not the same as good quality. The way to close this is to make stage E fire less often and evict
  fewer blocks when it does:
  1. *Anchor-aware axis choice* — for a soft–anchor pair, choose the disjunct consistent with which
     side of the anchor the block actually sits on, instead of `gap_x ≥ gap_y`. Removes
     single-anchor conflicts by construction, leaving only two-anchor corridors. Cheap, but it
     changes axis choices globally, so it must be A/B'd on the official 100 with `score_rf.py`.
  2. *Corridor-capacity pre-pass* — for each aligned anchor pair, capacity `p_b − (p_a + w_a)`;
     reassign the excess blocks to another corridor **before** the assignment. This is row-capacity
     legalization (Tetris/Abacus row assignment; the min-cost-flow legalizers of
     Doll–Johannes–Antreich and Brenner) applied to anchor corridors, and it repairs by
     *reassignment* rather than eviction, so it should cost far less quality.
  3. *Obstacle-aware stage G* — at 25–60% preplaced the free space is fragmented and the local
     overlap impulse deadlocks; hole-seeking long jumps would hand stage L a better-conditioned
     relation set.
  4. *Training-distribution augmentation* — the model has never seen more than 8.9% preplaced in
     1M training layouts while the stress cases reach 62%. Promoting random GT blocks to
     preplaced/fixed during training (what `cadc1111/gencase.py` already does for testing) attacks
     the problem at its source.
* **The hidden-set hypothesis is unverified.** Every failure measured here needs ≥25% preplaced,
  which neither the validation set nor the training shards produce; the beta failure is better
  explained by the MIB bug. Both are addressed, but a third, unmeasured mode is possible — which is
  exactly why the floor and the outer net exist.
* **`area_scale = 0.991` leaves 0.1pp of margin** against a strict `> 0.01` check. Every reshape
  recomputes the other dimension in float64 so drift is ~1e−16 relative, but a future stage
  composing two reshapes multiplicatively could erode it. Cheap insurance: `0.990`, and assert
  `max_area_error < 0.01` at the exit gate (already reported there).
* **The soft-violation semantics live in three places** — the official evaluator,
  `_violations_official`, and `snap_soft.comp_of`. They must stay in sync; the corner-touch rule in
  particular is easy to get wrong.
* **`cadc1111/` has drifted.** `floordiff/legalizer.py` and `parallel.py` in the submission copy
  predate this work; re-copy and re-run the stress kit against `op_wrapper.py` before packaging.
