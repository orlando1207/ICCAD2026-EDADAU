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
  `phase8_bc.pt`, `phase10_*`, `phase11_pin_smoke.pt`, `phase12_ppo_kl.pt`).
  The current default is `../checkpoints/phase11_pin_soft.pt`.
- **`HANDOFF.md`, `VERSION_B_RL_PLACER.md`** — early planning docs, superseded
  by `../README.md` and `../ALGORITHM.md`.
- **`train_rl.py`, `test_train_rl.py`** — Phase 4 PPO training loop (overfit
  acceptance test). Superseded by `../train_fast.py`'s behaviour-cloning
  approach; not imported by anything in the active pipeline.
- **`ar_utils.py`, `test_ar_utils.py`** — Phase 5 aspect-ratio action head
  (`forward_aspect`/`act_aspect`). `../policy_net.py` still defines these
  methods/`aspect_head` itself, but the active pipeline (`train_fast.py`,
  `rl_skyline_optimizer.py`) only uses position actions (`forward`/`act`) —
  the aspect head is unused and untrained.
