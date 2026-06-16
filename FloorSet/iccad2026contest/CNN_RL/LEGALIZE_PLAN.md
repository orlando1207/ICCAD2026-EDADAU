# Legalize Optimization Plan — bbox / Area_gap / boundary

Focused implementation plan for the legalizer-side wins, all centred on the
**bounding box**: shrinking it (Area_gap) and making forced blocks reach its
edges (boundary violations). Derived from this session's measurements on the
current default (B1, Total Score **1.8517**).

## Cost anatomy (Total-Score weighted, 100 cases)

```
Cost = (1 + 0.5·(Area_gap + HPWL_gap)) · e^(2·V_soft) · Runtime
         └──────────── 1.537 ───────────┘   └─ 1.201 ─┘
```

| Term | Value | Notes |
|---|---|---|
| Area_gap | 0.477 | our bbox is 48% larger than the GT baseline bbox |
| HPWL_gap | 0.597 | (out of scope here — placement/centers problem) |
| V_soft | 0.092 → `e^{2V}=1.20` | **multiplicative 20% tax** |
| ↳ boundary | 0.064 (70% of V) | 334 viols, 99/100 cases |
| ↳ grouping (cluster disconnect) | 0.028 (30%) | 121 viols, 72/100 cases |
| ↳ MIB | 0.000 | already perfect (`_unify_mib_shapes`) |

Two bbox levers worth ~`−12%` (boundary) and `−15%` (full Area_gap) if fully
captured; realistically a fraction of each.

## What we RULED OUT

- **Compaction (`compact_pass`) does NOT help.** Measured: Area_gap only
  0.477→0.464 (−0.013), boundary violations got **worse** (351→395; RIGHT +34,
  TOP +10 — pulling blocks toward the origin pulls forced blocks *away* from the
  max edges), Total Score regressed 1.8517→1.8827. **Area_gap is structural
  (block shapes don't tessellate to baseline's envelope), not removable
  whitespace.** The area lever is *reshaping*, not repositioning.

## Reshapeability ground truth

Only **fixed** + **preplaced** are HARD-locked (can't reshape). Everything else's
shape is a free variable (area must stay within ±1% — HARD):

| Type | Reshape? | Mechanism | Constraint |
|---|---|---|---|
| fixed / preplaced | ❌ | locked dims (+pos) | HARD |
| free soft | ✅ independent | area-preserving, AR≤3 | area HARD |
| boundary | ✅ independent | reshape + keep edge contact | edge SOFT |
| MIB | ✅ joint | one shared (w,h) per group | shape-match SOFT |
| cluster | ✅ joint | re-pack super-block, keep connected | grouping SOFT |

## Boundary violation anatomy (351 missed edge-bits)

NOT a snap bug: median miss 20 units (**9.6% of chip width**), 80% miss >10 units.

| Edge | Count | Reshape-fixable? |
|---|---|---|
| BOTTOM | 160 | ❌ position problem (block bumped off y=0) |
| TOP | 118 | ✅ grow taller to y_max (soft 47, cluster 18, mib 9) / ❌ pre+fix 44 |
| RIGHT | 62 | ✅ grow wider to x_max (soft 3, cluster 30, mib 4) / ❌ pre+fix 25 |
| LEFT | 11 | ~ok |

TOP/RIGHT reshape-addressable = **111 of 351** (32%); 69 are preplaced/fixed
(only fixable by shrinking *other* blocks so the bbox edge reaches them).

---

## Work items (priority order)

### A — Targeted TOP/RIGHT boundary reshape  ⟵ START HERE
**Problem:** a TOP/RIGHT-forced soft block lands short of y_max/x_max.
**Fix:** post-legalize, grow the block (area-preserving → it gets narrower as it
grows taller) up into the empty space above it until its top hits y_max (RIGHT:
grow right to x_max). Overlap-safe by construction (narrower ⇒ no new side
overlap; only grow into verified-empty cells up to the existing bbox edge ⇒ bbox
never grows). Apply only when it reaches the edge (fixes the boolean violation)
and stays overlap-free.
**Scope:** free singleton soft blocks first; then MIB (grow the shared shape) and
cluster (grow the super-block member touching the edge).
**Expected:** ~−2% (soft only) … ~−4% (incl cluster/MIB). Never-worse.
**Status:** ✗ **DONE — soft-only is a dud (−1/53 violations).** Built
(`boundary_shape.py`, `boundary_pass`, default OFF) and measured: of 16 soft
TOP/RIGHT candidates in 40 cases, **94% are geometrically impossible** — 8 need
AR>3 to reach the edge (sliver), 7 are blocked by occupied space above/right.
Only 1 fired, 0 new overlaps. The soft-singleton opportunity is just too small
(50/351 bits) and mostly infeasible. **The reshape-addressable boundary mass is
in clusters (T/R: 48 bits), which need joint reshaping → folds into Item B.**

### B — Shape lever for the rigid majority (MIB + cluster)  → THE master lever
**Why this is now the priority:** the cheap soft-singleton levers are exhausted
(compaction OUT, B2 flat, Item A dud). The remaining mass is in the *rigid*
blocks, and reshaping them is **triple-hit**:
  1. **Area_gap** — the PoC ceiling (0.52→0.28) needs *all* blocks reshaped;
     clusters are ≈37% of blocks and dominate large cases (→ Total Score).
  2. **boundary** — clusters carry 48 of the reshape-addressable TOP/RIGHT bits
     (Item A's missing mass), AND shrinking the bbox via reshape pulls its edges
     onto forced blocks (the boundary fix compaction couldn't deliver).
  3. **grouping** — re-packing clusters can also keep them connected.

**B-MIB (tractable, do first):** pick one shared (w,h) per MIB group that best
fits the skyline contour; fold into `skyline_shape.py` (B1 currently skips MIB).
Small, low-risk.

**B-cluster (hard, the real prize):** reshape a cluster super-block to a target
aspect by re-packing its members (`prepack_clusters` with an aspect target),
then let the packer place the reshaped super-block. Keeps area + connectivity.
This is where the Total-Score-moving Area_gap lives.

**Expected:** the only remaining lever that moves Total Score on large cases.
**Status:** ☐ next — start with B-MIB

### C — BOTTOM/LEFT placement fix  → biggest boundary bucket (deferred)
**Problem:** 171 BOTTOM/LEFT viols — forced blocks not at y=0/x=0 (position, not
shape). BOTTOM (160) is the single biggest bucket.
**Fix:** packer-side — guarantee forced blocks reach 0; resolve multi-block
same-edge conflicts; or let these blocks define the bbox min.
**Expected:** large but needs packer ordering changes (riskier). Not reshape.
**Status:** ☐ deferred (do after B)

---

## Principles (carry from B1/B2)

- **Never-worse:** every pass is gated — keep the new layout only if feasible
  (no overlap, hard constraints intact) AND the `bbox·HPWL` proxy / violation
  count improves. So a pass can only help or no-op.
- **Don't modify `analytic_legalizer/`** — import its pure helpers only.
- **Measure on the full 100-case `--evaluate`** (Total Score is the metric;
  watch that Avg-Cost wins aren't just on small cases).
