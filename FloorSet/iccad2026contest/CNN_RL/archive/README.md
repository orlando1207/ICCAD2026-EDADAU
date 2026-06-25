# Archive — deprecated / abandoned

Kept for reference only; not part of the current pipeline. See
[`../README.md`](../README.md) for the current pipeline.

- **`rl_optimizer.py`** — Phase 7/8 `FloorplanOptimizer`
  (`shelf_legalize` / `model_guided_legalize`). Total Score 10.05 / 15.60 —
  superseded by `rl_skyline_optimizer.py` (Phase 9+), which reuses
  `analytic_legalizer.skyline_legalize()` instead of a custom row-packer.
- **`train_ppo.py`** — Phase 12 PPO + KL-anchor fine-tuning of
  `phase11_pin_soft.pt`. Two attempts (with/without the KL-anchor) did not
  improve on the BC checkpoint within a ~2h compute budget.
- **`checkpoints/`** — intermediate/superseded checkpoints (`bc_warmstart.pt`,
  `phase8_bc.pt`, `phase10_*`, `phase11_pin_smoke.pt`, `phase12_ppo_kl.pt`,
  `phase11_pin_soft.pt`). The current default is
  `../checkpoints/phase13_aspect.pt` (Phase 13, predicts soft-block aspect
  ratios — beats `phase11_pin_soft.pt` on both Total Score and Avg Cost).
- **`HANDOFF.md`, `VERSION_B_RL_PLACER.md`** — early planning docs, superseded
  by `../README.md` and `../ALGORITHM.md`.
- **`train_rl.py`, `test_train_rl.py`** — Phase 4 PPO training loop (overfit
  acceptance test). Superseded by `../train_fast.py`'s behaviour-cloning
  approach; not imported by anything in the active pipeline.
- **`test_ar_utils.py`** — Phase 5 acceptance test for the aspect-ratio head.
  `../ar_utils.py` was restored from this archive when aspect-ratio
  prediction (Phase 13) was picked back up — see `../README.md`.
