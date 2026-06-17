# gnn_joint_placer — continuous, joint, cost-trained placer (PoC)

A proof-of-concept alternative to the `CNN_RL` grid placer. It removes the three
limits of the CNN+RL pipeline **at once**:

| limit of CNN_RL | here |
|---|---|
| 64×64 **grid** quantization | **continuous** (x,y) — real numbers |
| **sequential greedy** (one block at a time, no look-back) | **joint** — self-attention over all blocks, all placed in one shot → shape & position **co-adapt** |
| **behaviour cloning** (imitates ground truth) | trained **directly on the differentiable contest cost** by SGD |

Area is **exact by construction**: the head predicts only a log-aspect; `w,h` are
derived from the known target area (`w=√(a·ar), h=√(a/ar)`), so the area-tolerance
violation term is 0 by design.

## Testing protocol (READ FIRST — every method change must report all three)

A single case is misleading. When you measure ANY change here, always report:

1. **case 0** (smallest, 21 blocks) — surfaces *subtle* quality changes; barely
   weighted in the score, so it's the magnifying glass, not the verdict.
2. **case 99** (largest, 120 blocks) — dominates the official score (the
   `e^{n/12}` weighting makes the big cases carry most of the weight), so this is
   where wins/losses actually count.
3. **official Total Score over all 100 cases** — `python3 total_score.py`. This is
   the verdict: `Total = Σ Cost·e^{n/12} / Σ e^{n/12}`, with the OFFICIAL per-case
   cost (feasibility, real violation counts, M=10 infeasible penalty), **not** the
   differentiable surrogate. The surrogate (`diff_cost`) ignores hard constraints,
   so it can look fine while the real score is infeasible — see the gotcha below.

Per-case figures: `python3 end2end.py --viz-cases 0,99`
Verdict number:   `python3 total_score.py`

> Gotcha that bit us: `solve()` must be given a real `opt_target_pos` (fixed
> blocks → (w,h), preplaced → (x,y,w,h)); passing all-`-1` starves the solver of
> hard-constraint specs and makes EVERY case infeasible (Total Score = 10.0). The
> surrogate cost hides this; the official Total Score catches it.

## Files
- `model.py` — `JointPlacer`: reuses `../gnn_encoder.py`, adds a Transformer
  encoder over blocks + a 3-output head `(x_raw, y_raw, log_aspect)`.
- `diff_cost.py` — vectorised differentiable contest cost. Validated to match
  `iccad2026_evaluate.compute_training_loss_differentiable` to ~2e-7 (`python3 diff_cost.py`).
- `train_poc.py` — train on mean cost over a train split, report cost +
  decomposition on a held-out val split, vs the grid+greedy baseline.
- `end2end.py` — joint placer → existing legalize tail; per-case decomposition
  over all 100 cases + figures for chosen cases (`--viz-cases 0,99`).
- `total_score.py` — OFFICIAL Total Score over all 100 cases for both pipelines.

## Run
```bash
cd FloorSet/iccad2026contest
python3 CNN_RL/gnn_joint_placer/train_poc.py --n 120 --steps 1200 --batch 12
```
Train **in log space** (`log(cost) = log(quality) + β·V_soft`): the raw cost has an
`exp(β·overlap)` term (β=2) that explodes the moment blocks pile up and diverges
training. Log space has the same minimizer with a bounded gradient. (The first
run before this fix diverged to 1e31 — see git history.)

## Result (val = 30 held-out cases, 1200 steps, ~2 min CPU)

| metric (val mean) | grid+greedy baseline | **GNN joint placer** |
|---|---|---|
| surrogate cost | **1.596** | 1.814 |
| hpwl_gap | +0.695 | **+0.516** |
| area_gap | +0.497 | **+0.155** |
| overlap_viol | 0.000 (legalized) | 0.153 (no legalize yet) |

### Verdict
The joint placer **wins on BOTH quality terms** — wirelength (0.516 vs 0.695) and
especially **area (0.155 vs 0.497, ~3× tighter)**. Its *quality factor* is
`1+α(0.516+0.155)=1.34` vs the baseline's `1.60`. This directly confirms the
**"Option B" co-adaptation lever**: deciding block shape and position together,
continuously, is what closes Area_gap — not grid+greedy.

The **only** reason its total surrogate cost (1.81) is above the baseline (1.60)
is the leftover `overlap_viol=0.15` (the `exp(β·0.15)=1.36` multiplier), because
this PoC has **no legalize tail** — soft overlap penalty alone never reaches
exactly 0. That is precisely the job the existing skyline legalizer already does.

### Implied path
`JointPlacer` (continuous co-adapted shapes+positions) **→ existing skyline
legalize** (drive overlap→0). If legalize removes the 0.15 overlap without
wrecking quality, total cost lands near **~1.34**, beating the grid+greedy
baseline (~1.60) by ~16%. Next step is to wire the legalize tail in and measure
the feasible end-to-end score against the current 1.897.

## End-to-end test: joint placer → existing skyline legalize (`end2end.py`)

"grid+greedy" is `RLSkylineOptimizer._rl_centers()` — the only place it acts: the
CNN+RL rollout that emits `(cx, cy, aspect)`. `solve()` now takes
`centers_override=(cx,cy,aspect)` so the JointPlacer can feed the **identical**
legalize tail. Mean over **all 100 cases** (verified on case 0 AND case 99 — a
single case is not enough):

| stage | cost | hpwl_gap | area_gap | overlap |
|---|---|---|---|---|
| grid+greedy + legalize (current pipeline) | **1.587** | +0.690 | +0.483 | 0 |
| JOINT placer (pre-legalize) | — | +0.561 | **+0.114** | 0.107 |
| JOINT + **existing** legalize tail | 1.748 | +0.946 | **+0.551** | 0 |

`joint_leg` beats `grid_leg` on only **19/100 cases**. The pre-legalize tightness
is even more striking at scale — case 99 (120 blocks) has `area_gap=+0.077` before
legalize — which makes the loss to legalization all the more visible.

### Official Total Score (the verdict — `total_score.py`, all 100 cases, RF=1.0)

| pipeline | **Total Score** | mean cost | infeasible |
|---|---|---|---|
| grid+greedy + legalize (current) | **1.852** | 1.959 | 0/100 |
| JOINT + same legalize tail | 2.312 | 2.308 | 0/100 |

Both are fully feasible, but JOINT+existing-legalize scores **2.31 vs 1.85** — the
legalize tail erases the joint placer's area advantage, and the loss is heaviest
on the large, heavily-weighted cases. So this combination is **not** a win yet;
the model is sound, the legalize tail is the blocker.

### Verdict — bolting on the existing legalizer does NOT work
The joint placer's tight `area_gap=0.114` is **destroyed by legalization** (→0.551,
even above the grid baseline's 0.483), and hpwl_gap gets worse too. So
`JOINT + existing-legalize` (cost 1.748) is **worse** than the current pipeline
(1.587), winning only 19% of cases. The figures (`end2end_case0.png`,
`end2end_case99.png`) show why: the pre-legalize layout is a tight interlocking
blob; the skyline legalizer is a *shelf re-packer*, not a minimal-perturbation
overlap remover — it throws the supplied layout away, re-derives positions from
the centers, and re-introduces whitespace.

This sharpens (and re-confirms) the "Option B" finding: the continuous-joint
co-adaptation lever is real **only while placement and legalization are the same
process**. You cannot harvest it by handing a frozen tight layout to a re-packing
legalizer, just as you couldn't harvest it with a frozen post-pass.

### Two ways forward (the lever is real pre-legalize; the tail is the problem)
1. **Minimal-perturbation legalizer** — remove overlap while keeping each block
   near its joint position (anchored spreading / flow-based), instead of skyline
   shelf re-packing. Then the 0.150 area_gap can survive.
2. **Near-feasible joint training** — push `overlap_viol`→~0 in the model
   (stronger overlap weight, longer training, a differentiable spreading term) so
   only a *light* touch-up is needed, not a full re-pack.

## Out of scope for this PoC (same philosophy as `poc_slack_shaping.py`)
Hard constraints (boundary / MIB / cluster / preplaced) and the legalize tail are
not enforced here — this measures the *lever*, not a feasible submission. Numbers
are the differentiable surrogate, not a final contest score.
