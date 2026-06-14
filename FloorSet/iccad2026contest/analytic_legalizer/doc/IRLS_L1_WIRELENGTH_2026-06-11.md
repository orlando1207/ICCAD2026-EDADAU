# Analytic core upgrade — IRLS / true-L1 wirelength (2026-06-11)

*Implementation pass following the method analysis. Builds on the gating fix +
cleanup recorded in `CHANGES_2026-06-11.md`. All scores from
`python score_harness.py all` (official `compute_total_score`, `e^(n/12)` weights).*

---

## TL;DR

| Step | Weighted score (all 100) | Boundary viol. | Feasible |
|------|--------------------------|----------------|----------|
| Original baseline | 1.9157 | 415 | 100/100 |
| + rich-search gating fix (prior pass) | 1.8668 | 363 | 100/100 |
| **+ IRLS L1 wirelength (this pass)** | **1.8537** | 353 | 100/100 |

Cumulative **−3.24%** vs. the original. **86/100 cases improved**, 4 negligible
regressions (+0.01…+0.04, proxy tie-break noise), feasibility never lost.

---

## Why this change

The analysis established that the contest's wirelength term is **edge-based L1**
(Manhattan), not net-bounding-box HPWL — `iccad2026_evaluate.py:153-194` computes
exactly

```
HPWL = Σ_e w_e·(|Δx_center| + |Δy_center|)   over b2b edges + pin→center terms
```

and it is the **single largest cost gap** (0.45–1.94 across cases). The analytic
placer, however, minimized this only approximately:

1. a **one-shot squared-wirelength** (L2) solve (`quadratic_placer.py:90-161`), which
   over-penalizes long nets and over-clusters; then
2. an optional **tanh-LSE gradient descent** patch (`_lse_refine`) — a smooth-L1
   approximation with hand-tuned `gamma`/`step`/iteration count, decayed `0.96^it`.

Neither minimizes the metric the scorer actually measures. Since the L1 objective is
**separable per axis** (`|Δx| + |Δy|`), each axis is a clean L1 minimization with a
well-known exact solver: **IRLS (Iteratively-Reweighted Least Squares)**.

## What was implemented

### `quadratic_placer.py` — new `_irls_refine()` + dispatch

IRLS rewrites `|d|` as `d² / |d_prev|`, turning each iteration into a **weighted
least-squares solve** of the same Laplacian system already used by the quadratic
build, with edge weight `w_e / max(|d_prev|, ε)`. Iterating ~8 times converges to the
true L1 (Manhattan) minimizer. Properties:

- **Exactly the scorer's objective** — minimizes edge-based L1, not a surrogate.
- **Parameter-free** — no `gamma`/step/decay tuning; just an iteration count.
- **Reuses the dense solver** (`np.linalg.solve`); trivial at n ≤ 120 (8 solves of an
  ≤120×120 system).
- **Robust**: a tiny **proximal term** (`mu`, ~`1e-3·avg_w/ε`) is added to every node's
  diagonal toward its current position. This keeps both per-axis systems
  non-singular — handling the anchor-free, translation-invariant case — and damps
  oscillation between iterations. Preplaced blocks keep their **Dirichlet** rows
  (fixed (x,y)); pins enter as fixed RHS terms; results are clamped into the chip and
  re-fixed for numerical safety, mirroring the existing quadratic path.

Wired in via `wl_model == "irls"` in `analytic_place` (alongside `quadratic`, `lse`).

### `optimizer.py` — IRLS added as a candidate generator

```python
WL_MODELS = ("quadratic", "lse", "irls") if RICH_SEARCH else ("quadratic",)
```

IRLS is **added**, not substituted. Each WL model produces a guide; the full skyline
config sweep runs on each; the `area·HPWL·exp(3·soft/n)` proxy selects the best
legalized layout per case. Because the candidate set is a strict **superset**,
proxy-selection can only pick an equal-or-better result — confirmed by the 86–4
improve/regress split (the 4 regressions are ≤0.04 and are the known
multiplicative-proxy mis-rank, not an IRLS defect).

## Result detail (original → this pass)

Biggest gains land on the previously under-served mid/small cases — and, crucially for
the `e^(n/12)` weighting, on **higher-weight mid-large** cases too (which is what moved
the aggregate):

```
idx 8  n= 29  2.86 -> 1.91   idx 66 n= 87  2.64 -> 1.79
idx 14 n= 35  3.03 -> 2.16   idx 41 n= 62  2.58 -> 1.81
idx 71 n= 92  2.04 -> 1.69   idx 55 n= 76  2.29 -> 1.93
```

## An experiment that was tested and rejected: multistart

I also implemented and measured **multistart** (3 noise-seeded analytic restarts) for
small cases — the other Tier-1 lever from the analysis. On the small cases it worked
(e.g. idx12 2.36→1.99, idx2 1.80→1.67), **but the full weighted score did not move
(1.8537 → 1.8536).**

Reason, and a lesson for future tuning: under `e^(n/12)` the smallest cases carry
negligible weight (n=33 ≈ `e^(-7.25)` ≈ 0.0007), so even large per-case wins there
don't register in the aggregate. **Score-moving improvements must target mid-large
cases (n ≳ 90).** Multistart's extra restarts there are expensive, so it was
**reverted** (`N_STARTS = 1`). IRLS earned its place precisely because it improved
mid-large cases, not just small ones.

## Cost / tradeoff

Adding a third WL model raises wall-clock ~50% (≈4m18s → ≈6m for all 100;
sub-10s/case). The analytic IRLS solve itself is sub-millisecond; the added cost is the
extra skyline legalization sweep per guide. The official runtime factor is
`max(0.7, R^0.3)` (weak, floored) and computed **per case** vs. the cross-submission
median, so this is a low-risk trade for the quality gain. *Option if runtime becomes a
concern on the leaderboard:* gate `irls` to `block_count < 116` — it does not help the
5 biggest cases (they were essentially unchanged) yet they are the slowest.

## Files changed

- `analytic_legalizer/quadratic_placer.py` — added `_irls_refine()`; added the
  `wl_model == "irls"` dispatch branch in `analytic_place()`.
- `analytic_legalizer/optimizer.py` — added `"irls"` to `WL_MODELS`.

## Still on the table (unchanged, biggest first)

The ceiling identified in the analysis still holds: **the skyline legalizer
floor-packs away the analytic `y`**, so the analytic layout reaches the result only as
an x-hint + pack order. The highest-remaining levers:

1. **Stop discarding the analytic 2D structure** — a warm-started constraint-graph
   (sequence-pair) compaction or legalization-in-the-loop feedback. This is the
   analytical-method-native version of the report's "warm-started SA" and the biggest
   remaining HPWL win, concentrated on the heavy mid-large cases.
2. **Per-candidate-width affine consistency** of the guide before the `λ·|x−cx|` term
   (cheap bias removal; the width sweep compares against a fixed-aspect guide today).
3. **Selection proxy**: it is multiplicative (`area·HPWL`) while true quality is
   additive (`1 + 0.5·(HPWL_gap+Area_gap)`) — the source of the 4 small regressions;
   worth revisiting for candidate ranking.
