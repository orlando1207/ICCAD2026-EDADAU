"""
Phase 4 acceptance test (VERSION_B_RL_PLACER.md §5 Phase 4).

Overfit a tiny fixed sample set with PPO and confirm the mean contest cost
goes DOWN (reward up) — i.e. the policy is actually learning to place better
than the random Phase-0 baseline.

Run:
    cd FloorSet/iccad2026contest
    python3 DL_RL/test_train_rl.py
"""

import sys
from pathlib import Path

import torch

_FLOORSET_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_FLOORSET_ROOT) not in sys.path:
    sys.path.insert(0, str(_FLOORSET_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lite_dataset_test import FloorplanDatasetLiteTest  # noqa: E402
from train_rl import train  # noqa: E402


def main():
    ds = FloorplanDatasetLiteTest(str(_FLOORSET_ROOT))
    samples = [ds[0], ds[1]]  # 21 & 22 blocks, fixed overfit set

    # rollouts=8 -> 16 episodes/iter: PPO is too noisy with a 2-episode batch.
    history = train(samples, iters=35, grid=24, gnn_out=64, hidden=32,
                    lr=5e-4, rollouts=8, seed=0, verbose=False)

    for f in history:
        assert f == f and f > 0, "non-finite / non-positive cost"  # NaN guard

    init = sum(history[:8]) / 8
    final = sum(history[-8:]) / 8
    print(f"initial mean_cost (first 8 iters) = {init:.4f}")
    print(f"final   mean_cost (last 8 iters)  = {final:.4f}")
    print(f"best    mean_cost seen            = {min(history):.4f}")
    print(f"improvement (init-final)          = {init - final:+.4f}")

    assert final < init, "PPO did not reduce mean cost — policy not learning"
    assert min(history) < init, "no improvement over the initial window"
    print("\nPhase 4 PASS: PPO reduces mean contest cost on the overfit set "
          "(convergence quality is improved further by Phase 4.5 BC warm-start).")


if __name__ == "__main__":
    main()
