# FloorDiff Stage-1 Implementation Notes

Date: 2026-07-15. Implements the design in
`2026-07-14-diffusion-repo-anatomy-and-contest-model-design.md` (Part C, imitation-first
revision): a conditional diffusion model whose only objective is **raw prediction close to
ground truth** — no legalizer, no constraint enforcement (deferred per §C.0/§C.9).

Code: `iccad2026contest/floordiff/` (new package; nothing outside it was modified).

---

## 1. Package layout

| File | What it does |
|---|---|
| `data.py` | Shard/validation loading, featurization, augmentation, latent encode/decode, bucketed batching |
| `model.py` | `FloorDiffNet`: DiT-style Transformer, AdaLN-Zero, connectivity-biased attention |
| `diffusion.py` | Cosine schedule, x0-prediction loss (min-SNR-5, frozen-channel masked), DDIM sampler with inpainting, EMA |
| `train.py` | Training CLI (bucketed loader, EMA, bf16, quick-val, checkpointing, resume) |
| `sample.py` | Inference CLI: batched best-of-N sampling on the validation cases → predictions JSON |
| `evaluate.py` | Stage-1 metrics CLI/library (closeness + feasibility), `--gt-check` harness sanity |
| `visualize.py` | GT-vs-prediction side-by-side renderer (the "current stage" visualizer) |

## 2. How to run (from `iccad2026contest/`, env `~/miniconda3/envs/iccad`)

```bash
PY=~/miniconda3/envs/iccad/bin/python

# train (first run scans+caches the ~9k shard index, ~10 min)
$PY -m floordiff.train --run base --steps 200000 --batch-size 64 --device cuda:3
$PY -m floordiff.train --resume floordiff/checkpoints/base/last.pt ...   # resume

# smoke variant (small model, 20 shards, minutes)
$PY -m floordiff.train --run smoke --max-files 20 --steps 300 --batch-size 32 \
    --d-model 128 --n-layers 4 --n-heads 4

# sample raw predictions (best-of-16, 50 DDIM steps, all 100 validation cases)
$PY -m floordiff.sample --ckpt floordiff/checkpoints/base/last.pt \
    --n-seeds 16 --steps 50 --out floordiff/out/preds.json

# evaluate (closeness + feasibility, per-case table + contest-weighted aggregate)
$PY -m floordiff.evaluate --pred floordiff/out/preds.json
$PY -m floordiff.evaluate --gt-check            # harness sanity (P0)

# visualize (GT | prediction side-by-side PNGs, constraint-colored)
$PY -m floordiff.visualize --pred floordiff/out/preds.json --cases 60,100,120 --nets 30
$PY -m floordiff.visualize --gt-only --cases 21 # inspect GT alone
```

## 3. Design → code mapping

| Design decision (doc §) | Where implemented |
|---|---|
| Latent `z = (cx, cy, s=½log(w/h))`, area exact by construction (C.1) | `data.featurize` / `data.decode`; scales `COORD_SCALE=2.9`, `S_SCALE=3.8` make each channel ~unit variance (measured std 0.341 / 0.263) |
| Terminal-anchored frame, `S = √Σaᵢ` (C.1) | `data.featurize` (origin = terminal-bbox center) |
| Preplaced frozen (all ch) / fixed-shape frozen (`s`), inpainted (C.0/C.1) | `freeze`/`z_known` in `data.featurize`; loss mask in `diffusion.loss`; per-step `q_sample` re-imposition in `diffusion.sample` |
| MIB feature-only + post-snap (C.1) | pairwise `same_mib` bias + `data.decode` (group-mean `s` keeps area exact; a fixed member dictates the group) |
| Constraint-type features, 24 per block + 12 global + 3 pairwise (C.3) | `data.featurize` (`N_FEAT/N_GLOBAL/N_PAIR`) |
| DiT backbone: AdaLN-Zero, biased attention, x0-pred, self-cond (C.2) | `model.py` (`DiTBlock`, `BiasedAttention`, zero-init heads); default 256×8 = 9.8M params |
| MSE-to-GT only, min-SNR-γ=5, no physics losses (C.4) | `diffusion.loss` |
| Bucketed batches, soft large-n oversampling `exp(n/24)` (C.4) | `data.BucketBatchSampler` — each batch drawn from one shard file (uniform n ⇒ zero padding) |
| 4 orientation-preserving symmetries with boundary-bit remap (C.4) | `data.augment_case` (flips only; fixed/preplaced dims never swap) |
| Plain DDIM + inpainting, batched best-of-N, cost-proxy ranking (C.5) | `diffusion.sample`, `sample.predict_case` |
| Stage-1 metrics incl. violations-vs-displacement diagnostic (C.6) | `evaluate.py` |

## 4. Verified facts and sanity results

- **Encode/decode roundtrip**: `decode(featurize(case).z0) == GT` to ≤2e-5 on all checked
  cases; augmentation is an exact involution and augmented cases decode to their own GT.
- **`--gt-check` (P0 exit criterion)**: displacement, shape error, overlap, HPWL gap, area gap
  all exactly 0 on GT.
- **Violation checker agrees with the official evaluator exactly** (verified per-case against
  `iccad2026_evaluate.evaluate_solution` on n=21/40/80/100/120: identical V_boundary,
  V_grouping, V_mib).
- ⚠ **The ground truth itself has soft violations**: 1–4 boundary violations per validation
  case (blocks sitting 1–4 units from the bbox edge; official eps is 1e-6) and one grouping
  violation at n=100. Official cost of GT ≈ **1.09–1.13**, not 1.0. Consequences:
  (a) V_rel ≈ 0.05 is the *label noise floor* — don't chase violations below it;
  (b) matching GT exactly can still leave ~5% V_rel, consistent with the imitation-first plan
  (legalization stage cleans up).
- **Training shards**: `floorset_lite/worker_*/layouts_*.th`, 9,000 files × 112 same-size
  layouts (~1M samples); `fp_sol` is `(w, h, x, y)` with lower-left corner (verified: zero
  overlap, util 0.973, bbox == metrics area); no `-1` padding inside a file; p2b columns are
  `(pin, block, weight)`.

## 5. Smoke test (pipeline verification, not a quality result)

1.28M-param model, 20 shards, 300 steps: loss 1.10 → 0.57, ~800 samples/s on one A6000.
Sampling 4 seeds × 30 steps: 0.1–1 s/case. Evaluation and visualization ran end-to-end
(`floordiff/out/preds_smoke.json`, `floordiff/out/viz_smoke/`). Metrics are garbage at this
scale (disp ≈ 0.28, overlap ≈ 250%) — expected; the smoke run only proves plumbing.

## 6. Training run in progress

A real run was launched (2026-07-15):

```
run:    base   (d_model 256, 8 layers, 8 heads, 9.8M params, T=1000)
data:   all ~9k shards, batch 64, augment on, exp(n/24) file weighting
steps:  200,000   (bf16, EMA 0.9999, AdamW lr 1e-4 warmup+cosine)
device: cuda:3    log: floordiff/checkpoints/base/train.log
```

Monitor with `tail -f floordiff/checkpoints/base/train.log` — quick-val (mean normalized
displacement on n=60/100/120, 1 seed, 20 steps) prints every 2k steps; checkpoints every 5k
(`last.pt` always newest). Resume with `--resume floordiff/checkpoints/base/last.pt`.

## 7. Deliberately not implemented (per design §C.0/§C.9)

Physics-guided sampling, clean-space polish, auxiliary geometry losses, grouping/boundary
enforcement of any kind, terminal tokens, the GNN branch (ablation candidates), multi-GPU
DDP. The promotion rule stands: only add §C.9 machinery if the violations-vs-displacement
diagnostic shows violations *not* tracking displacement.

## 8. Next steps (P1→P4 of the plan)

1. Let `base` converge; track quick-val displacement.
2. Full sample+evaluate at checkpoints (16 seeds, 50 steps); record the §C.6 table.
3. P2 ablations: inpainting on/off (sampling-time toggle, same checkpoint), MIB snap on/off,
   `(w,h)` head vs `s` (needs retrain), augmentation on/off.
4. Best-of-N sweep (N ∈ {1, 4, 16, 32}) and DDIM step sweep (20/50/100) for the
  closeness-vs-runtime frontier.
